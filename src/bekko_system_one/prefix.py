"""ModernBERT with independent prefix and bidirectional document branches.

Per-layer rotated K/V retain their autograd graph during training. They live only
within one forward; optimizer updates never reuse stale representations.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb


def attention_packing(qmask, kmask):
    """Prepare once per branch, avoiding device synchronization at every layer."""
    qlens, klens = qmask.sum(1), kmask.sum(1)
    return (
        qmask.flatten().nonzero().flatten(),
        kmask.flatten().nonzero().flatten(),
        F.pad(qlens.cumsum(0).to(torch.int32), (1, 0)),
        F.pad(klens.cumsum(0).to(torch.int32), (1, 0)),
        int(qlens.max()),
        int(klens.max()),
    )


def attention(q, k, v, qmask, kmask, window, backend, packing=None):
    """Suffix-aligned attention, matching FA2's unequal Q/K sequence semantics."""
    if backend == "flash_attention_2":
        # Installed by the optional fa2 extra.
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        qi, ki, cuq, cuk, maxq, maxk = (
            packing if packing is not None else attention_packing(qmask, kmask)
        )
        values = flash_attn_varlen_func(
            q.flatten(0, 1).index_select(0, qi),
            k.flatten(0, 1).index_select(0, ki),
            v.flatten(0, 1).index_select(0, ki),
            cuq,
            cuk,
            maxq,
            maxk,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1) if window is None else (window, window),
        )
        output = torch.zeros_like(q).flatten(0, 1).index_copy(0, qi, values)
        return output.view_as(q)
    if backend != "sdpa":
        raise ValueError(backend)
    # Masks can contain gaps between a padded prefix and a document. Positions
    # are ranks among valid tokens, not padded array indices.
    qp = qmask.long().cumsum(1) - 1 + (kmask.sum(1) - qmask.sum(1))[:, None]
    kp = kmask.long().cumsum(1) - 1
    allowed = kmask[:, None, :].expand(-1, q.shape[1], -1)
    if window is not None:
        allowed = allowed & ((qp[:, :, None] - kp[:, None, :]).abs() <= window)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed[:, None],
        dropout_p=0.0,
    ).transpose(1, 2)
    return output * qmask[:, :, None, None]


class PrefixEncoder(nn.Module):
    def __init__(self, backbone, backend="flash_attention_2", gradient_checkpointing=False):
        super().__init__()
        self.backbone = backbone
        self.backend = backend
        self.gradient_checkpointing = gradient_checkpointing

    def qkv(self, layer, hidden, positions):
        attn = layer.attn
        values = attn.Wqkv(layer.attn_norm(hidden))
        q, k, v = values.view(*hidden.shape[:2], 3, -1, attn.head_dim).unbind(2)
        cos, sin = self.backbone.rotary_emb(hidden, positions, layer.attention_type)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return q.transpose(1, 2), k.transpose(1, 2), v

    def update(self, layer, hidden, attended):
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(attended.flatten(2)))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def _prefix_layer(self, layer, hidden, positions, mask, packing):
        q, k, v = self.qkv(layer, hidden, positions)
        window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
        hidden = self.update(
            layer, hidden, attention(q, k, v, mask, mask, window, self.backend, packing)
        )
        return hidden, k, v

    def _doc_layer(self, layer, hidden, positions, mask, combined_mask, pk, pv, owners, packing):
        q, k, v = self.qkv(layer, hidden, positions)
        k, v = torch.cat((pk[owners], k), 1), torch.cat((pv[owners], v), 1)
        window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
        return self.update(
            layer, hidden, attention(q, k, v, mask, combined_mask, window, self.backend, packing)
        )

    def _run_layer(self, fn, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def encode_prefix(self, ids, mask):
        hidden = self.backbone.embeddings(ids)
        positions = (mask.long().cumsum(1) - 1).clamp_min(0)
        packing = attention_packing(mask, mask) if self.backend == "flash_attention_2" else None
        cache = []
        for layer in self.backbone.layers:
            hidden, k, v = self._run_layer(
                partial(self._prefix_layer, layer), hidden, positions, mask, packing
            )
            cache.append((k, v))
        return cache

    def score_documents(self, cache, pmask, ids, mask, owners):
        hidden = self.backbone.embeddings(ids)
        prefix_mask = pmask[owners]
        positions = (mask.long().cumsum(1) - 1).clamp_min(0) + prefix_mask.sum(1)[:, None]
        combined_mask = torch.cat((prefix_mask, mask), dim=1)
        packing = (
            attention_packing(mask, combined_mask) if self.backend == "flash_attention_2" else None
        )
        for layer, (pk, pv) in zip(self.backbone.layers, cache, strict=True):
            hidden = self._run_layer(
                partial(self._doc_layer, layer),
                hidden,
                positions,
                mask,
                combined_mask,
                pk,
                pv,
                owners,
                packing,
            )
        return self.backbone.final_norm(hidden)

    def forward(self, prefix_ids, prefix_mask, doc_ids, doc_mask, owners):
        cache = self.encode_prefix(prefix_ids, prefix_mask)
        return self.score_documents(cache, prefix_mask, doc_ids, doc_mask, owners)
