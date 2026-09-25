"""Keep valid tokens packed through projection, attention and feed-forward layers.

The padded implementation in prefix.py remains the correctness reference.
Only branch boundaries and a gather map are built once per forward; each layer
then operates on real tokens, with prefix K/V shared across document branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from .prefix import PrefixEncoder


@dataclass
class TokenLayout:
    indices: Tensor
    lengths: Tensor
    cumulative: Tensor
    positions: Tensor
    maximum: int

    @classmethod
    def from_mask(cls, mask):
        indices = mask.flatten().nonzero().flatten()
        lengths = mask.sum(1)
        cumulative = F.pad(lengths.cumsum(0).to(torch.int32), (1, 0))
        positions = (mask.long().cumsum(1) - 1).flatten().index_select(0, indices)
        return cls(indices, lengths, cumulative, positions, int(lengths.max()))


@dataclass
class DocumentPlan:
    layout: TokenLayout
    gather: Tensor
    cumulative_k: Tensor
    maximum_k: int
    positions: Tensor

    @classmethod
    def from_inputs(cls, prefix_layout, mask, owners):
        layout = TokenLayout.from_mask(mask)
        prefix_lengths = prefix_layout.lengths.index_select(0, owners)
        kv_lengths = prefix_lengths + layout.lengths
        cuk = F.pad(kv_lengths.cumsum(0).to(torch.int32), (1, 0))
        total_kv, maxkv = int(cuk[-1]), int(kv_lengths.max())
        branches = torch.repeat_interleave(
            torch.arange(len(owners), device=mask.device), kv_lengths, output_size=total_kv
        )
        offsets = torch.arange(total_kv, device=mask.device) - cuk[branches]
        prefix_count = prefix_layout.indices.numel()
        # Concatenating all prefix tokens and all document tokens once avoids a
        # padded [candidate, max_prefix, heads, dim] copy at every layer.
        gather = torch.where(
            offsets < prefix_lengths[branches],
            prefix_layout.cumulative[owners[branches]] + offsets,
            prefix_count + layout.cumulative[branches] + offsets - prefix_lengths[branches],
        ).long()
        token_branches = torch.repeat_interleave(
            torch.arange(len(owners), device=mask.device),
            layout.lengths,
            output_size=layout.indices.numel(),
        )
        positions = layout.positions + prefix_lengths[token_branches]
        return cls(layout, gather, cuk, maxkv, positions)


@dataclass
class PackedPrefixCache:
    layers: list[tuple[Tensor, Tensor]]
    layout: TokenLayout


def packed_attention(q, k, v, cuq, cuk, maxq, maxk, window, backend):
    if backend == "flash_attention_2":
        # Installed by the optional fa2 extra.
        from flash_attn import flash_attn_varlen_func  # ty: ignore[unresolved-import]

        return flash_attn_varlen_func(
            q,
            k,
            v,
            cuq,
            cuk,
            maxq,
            maxk,
            dropout_p=0.0,
            causal=False,
            window_size=(-1, -1) if window is None else (window, window),
        )
    if backend != "sdpa":
        raise ValueError(backend)
    # Explicit correctness path for tiny CPU tests, not a CUDA fallback.
    outputs = []
    qs, ks = cuq.tolist(), cuk.tolist()
    for i in range(len(qs) - 1):
        a, b = qs[i : i + 2]
        c, d = ks[i : i + 2]
        allowed = None
        if window is not None:
            qp = torch.arange(b - a, device=q.device) + (d - c) - (b - a)
            kp = torch.arange(d - c, device=q.device)
            allowed = (qp[:, None] - kp[None, :]).abs() <= window
        result = F.scaled_dot_product_attention(
            q[a:b].transpose(0, 1),
            k[c:d].transpose(0, 1),
            v[c:d].transpose(0, 1),
            attn_mask=allowed,
        )
        outputs.append(result.transpose(0, 1))
    return torch.cat(outputs)


class PackedPrefixEncoder(PrefixEncoder):
    fused_rotary = False

    def _rotary(self, hidden, positions):
        return {
            kind: tuple(
                x.squeeze(0) for x in self.backbone.rotary_emb(hidden, positions.unsqueeze(0), kind)
            )
            for kind in set(self.backbone.config.layer_types)
        }

    def _packed_qkv(self, layer, hidden, rotary):
        values = layer.attn.Wqkv(layer.attn_norm(hidden))
        if self.fused_rotary and values.is_cuda and values.dtype == torch.bfloat16:
            from .cuda_training import training_rotary_qkv

            return training_rotary_qkv(values, rotary[0], rotary[1], layer.attn.head_dim)
        q, k, v = values.view(hidden.shape[0], 3, -1, layer.attn.head_dim).unbind(1)
        q, k = apply_rotary_pos_emb(q, k, *rotary, unsqueeze_dim=1)
        return q, k, v

    def _packed_update(self, layer, hidden, attended):
        hidden = hidden + layer.attn.out_drop(layer.attn.Wo(attended.flatten(1)))
        return hidden + layer.mlp(layer.mlp_norm(hidden))

    def _packed_prefix_layer(self, layer, hidden, rotary, layout):
        q, k, v = self._packed_qkv(layer, hidden, rotary)
        window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
        attended = packed_attention(
            q,
            k,
            v,
            layout.cumulative,
            layout.cumulative,
            layout.maximum,
            layout.maximum,
            window,
            self.backend,
        )
        return self._packed_update(layer, hidden, attended), k, v

    def encode_prefix(self, ids, mask, *, layout=None):
        if layout is None:
            layout = TokenLayout.from_mask(mask)
        hidden = self.backbone.embeddings(ids.flatten().index_select(0, layout.indices))
        rotary = self._rotary(hidden, layout.positions)
        layers = []
        for layer in self.backbone.layers:
            hidden, k, v = self._run_layer(
                partial(self._packed_prefix_layer, layer),
                hidden,
                rotary[layer.attention_type],
                layout,
            )
            layers.append((k, v))
        return PackedPrefixCache(layers, layout)

    def _packed_document_layer(self, layer, hidden, rotary, pk, pv, gather, cuq, cuk, maxq, maxk):
        q, k, v = self._packed_qkv(layer, hidden, rotary)
        k = torch.cat((pk, k)).index_select(0, gather)
        v = torch.cat((pv, v)).index_select(0, gather)
        window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
        attended = packed_attention(q, k, v, cuq, cuk, maxq, maxk, window, self.backend)
        return self._packed_update(layer, hidden, attended)

    def score_documents(self, cache, pmask, ids, mask, owners, *, plan=None):
        if plan is None:
            plan = DocumentPlan.from_inputs(cache.layout, mask, owners)
        layout, gather = plan.layout, plan.gather
        cuk, maxkv, positions = plan.cumulative_k, plan.maximum_k, plan.positions
        hidden = self.backbone.embeddings(ids.flatten().index_select(0, layout.indices))
        rotary = self._rotary(hidden, positions)
        for layer, (pk, pv) in zip(self.backbone.layers, cache.layers, strict=True):
            hidden = self._run_layer(
                partial(self._packed_document_layer, layer),
                hidden,
                rotary[layer.attention_type],
                pk,
                pv,
                gather,
                layout.cumulative,
                cuk,
                layout.maximum,
                maxkv,
            )
        hidden = self.backbone.final_norm(hidden)
        # Restore the public padded token-output interface only once, at the end.
        output = hidden.new_zeros((ids.numel(), hidden.shape[-1]))
        return output.index_copy(0, layout.indices, hidden).view(*ids.shape, hidden.shape[-1])
