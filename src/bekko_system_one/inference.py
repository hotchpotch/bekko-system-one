"""Fixed-layout shared-prefix inference with a private snapshot and CUDA Graph.

Create a session for each exact mask/owner layout. Token values may change,
but shapes, masks and prefix ownership must remain fixed. A cached-prefix
session additionally fixes the prefix token values. Model preparation and
graph capture are one-time costs; this class does not change training models.
"""

from __future__ import annotations

import copy
import threading
import weakref
from collections import OrderedDict

import torch
from torch import nn

from .modules import SharedPrefix
from .packed_prefix import DocumentPlan, PackedPrefixEncoder, TokenLayout, packed_attention


class FusedPrefixEncoder(PackedPrefixEncoder):
    def _packed_qkv(self, layer, hidden, rotary):
        from .cuda_kernels import rotary_qkv

        values = layer.attn.Wqkv(layer.attn_norm(hidden))
        return rotary_qkv(values, rotary[0], rotary[1], layer.attn.head_dim)

    def _packed_document_layer(self, layer, hidden, rotary, pk, pv, gather, cuq, cuk, maxq, maxk):
        from .cuda_kernels import gather_kv

        q, k, v = self._packed_qkv(layer, hidden, rotary)
        k, v = gather_kv(pk, pv, k, v, gather)
        window = None if layer.attn.sliding_window is None else layer.attn.sliding_window - 1
        attended = packed_attention(q, k, v, cuq, cuk, maxq, maxk, window, self.backend)
        return self._packed_update(layer, hidden, attended)


def prepare_shared_inference(model, *, precast_linear=True, fused_ops=True):
    """Return an inference-only snapshot for variable-length ST predict calls.

    Run it under BF16 autocast/inference_mode. This runtime snapshot is not a
    training or save_pretrained artifact; save the source checkpoint instead.
    """
    encoder = model[0]
    if not isinstance(encoder, SharedPrefix) or not isinstance(
        encoder.encoder, PackedPrefixEncoder
    ):
        raise ValueError("A shared-prefix model with token packing is required")
    if (
        next(model.parameters()).device.type != "cuda"
        or encoder.encoder.backend != "flash_attention_2"
    ):
        raise ValueError("CUDA and flash_attention_2 are required")
    if model.training:
        raise ValueError("Call model.eval() before preparing inference")
    result = copy.deepcopy(model).eval().requires_grad_(False)
    result[0].encoder.fused_rotary = False
    if fused_ops:
        original = result[0].encoder
        result[0].encoder = FusedPrefixEncoder(original.backbone, backend=original.backend).eval()
    if precast_linear:
        # Keep embeddings, LayerNorm and residuals FP32; precompute only the
        # BF16 Linear weights that autocast would otherwise cast per call.
        for module in result[0].modules():
            if isinstance(module, nn.Linear):
                module.to(dtype=torch.bfloat16)
    return result


class SharedPrefixInference:
    def __init__(
        self,
        model,
        example_features,
        *,
        cache_prefix=False,
        use_graph=True,
        precast_linear=True,
        fused_ops=True,
        _prepared=False,
    ):
        self.device = next(model.parameters()).device
        self._model = (
            model
            if _prepared
            else prepare_shared_inference(model, precast_linear=precast_linear, fused_ops=fused_ops)
        )
        self._encoder = self._model[0].encoder
        keys = ("prefix_ids", "prefix_mask", "doc_ids", "doc_mask", "owners")
        self._head_indices = copy.deepcopy(example_features.get("head_indices"))
        self._device_head_indices = (
            {
                task: None
                if index is None
                else torch.as_tensor(index, device=self.device, dtype=torch.long)
                for task, index in self._head_indices.items()
            }
            if self._head_indices is not None
            else None
        )
        keys = (*keys, *(k for k in ("choice_indices", "choice_mask") if k in example_features))
        self._reference = {k: example_features[k].detach().cpu().clone() for k in keys}
        self._static = {k: v.to(self.device).clone() for k, v in self._reference.items()}
        self._layout = TokenLayout.from_mask(self._static["prefix_mask"])
        self._plan = DocumentPlan.from_inputs(
            self._layout, self._static["doc_mask"], self._static["owners"]
        )
        self.cache_prefix = cache_prefix
        self._cache = None
        self._graph = None
        self._lock = threading.Lock()
        self._stream = torch.cuda.current_stream(self.device)
        with (
            torch.cuda.device(self.device),
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False),
        ):
            if cache_prefix:
                self._cache = self._encode_prefix()
            if use_graph:
                warmup = torch.cuda.Stream(device=self.device)
                warmup.wait_stream(self._stream)
                with torch.cuda.stream(warmup):
                    for _ in range(3):
                        self._forward()
                self._stream.wait_stream(warmup)
                self._graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._graph):
                    self._output = self._forward()

    def _encode_prefix(self):
        return self._encoder.encode_prefix(
            self._static["prefix_ids"], self._static["prefix_mask"], layout=self._layout
        )

    def _forward(self):
        f = self._static
        cache = self._cache if self.cache_prefix else self._encode_prefix()
        hidden = self._encoder.score_documents(
            cache, f["prefix_mask"], f["doc_ids"], f["doc_mask"], f["owners"], plan=self._plan
        )
        mask = f["doc_mask"]
        result = {"sentence_embedding": (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1)[:, None]}
        if self._head_indices is not None:
            result["head_indices"] = self._device_head_indices
        for key in ("choice_indices", "choice_mask"):
            if key in f:
                result[key] = f[key]
        for index in range(1, len(self._model)):
            result = self._model[index](result)
        return result["scores"]

    def _validate(self, features):
        if features.get("head_indices") != self._head_indices:
            raise ValueError("Task routing changed; prepare another session")
        for key in ("choice_indices", "choice_mask"):
            if (key in features) != (key in self._reference):
                raise ValueError("Choice grouping changed; prepare another session")
        for key, expected in self._reference.items():
            value = features[key]
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"Input layout changed: {key}; prepare another session")
            fixed = key in {"prefix_mask", "doc_mask", "owners", "choice_indices", "choice_mask"}
            if fixed and not torch.equal(value.detach().cpu(), expected):
                raise ValueError(f"Input layout changed: {key}; prepare another session")
        if self.cache_prefix and not torch.equal(
            features["prefix_ids"].detach().cpu(), self._reference["prefix_ids"]
        ):
            raise ValueError("Cached prefix changed; prepare another session")

    @torch.inference_mode()
    def score(self, features):
        """Return an independent CUDA scores tensor; validation and input copies included.

        Supply preprocess output on CPU to avoid device-to-host validation.
        Use the CUDA stream active at construction. Calls on that stream are
        serialized, so static input/output storage cannot race between threads.
        """
        with self._lock, torch.cuda.device(self.device):
            if torch.cuda.current_stream(self.device) != self._stream:
                raise ValueError("Use the CUDA stream on which this session was created")
            self._validate(features)
            for key in ("doc_ids",) if self.cache_prefix else ("prefix_ids", "doc_ids"):
                self._static[key].copy_(features[key])
            if self._graph is not None:
                self._graph.replay()
                return self._output.clone()
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                return self._forward().clone()


# Keep runtime state outside nn.Module: saving a model never saves snapshots/graphs.
_RUNTIMES = weakref.WeakKeyDictionary()
_RUNTIME_LOCK = threading.RLock()


def clear_inference_cache(model):
    """Release cached snapshots/graphs, e.g. after changing adapters or configuration."""
    with _RUNTIME_LOCK:
        _RUNTIMES.pop(model, None)


def _revision(model):
    return (
        tuple(
            (n, id(t), t._version, t.device, t.dtype)
            for n, t in (*model.named_parameters(), *model.named_buffers())
        ),
        repr(model[0].settings),
        model[0].encoder.backend,
        tuple((n, id(m)) for n, m in model.named_modules()),
    )


class _Runtime:
    def __init__(self, model, *, fused_ops):
        cuda_fa2 = model.device.type == "cuda" and model[0].encoder.backend == "flash_attention_2"
        self.fused_ops = fused_ops and cuda_fa2
        if cuda_fa2:
            self.model = prepare_shared_inference(model, fused_ops=fused_ops)
        else:
            self.model = copy.deepcopy(model).eval().requires_grad_(False)
            self.model[0].encoder.fused_rotary = False
        self.layouts = OrderedDict()
        self.lock = threading.RLock()

    def score(self, features):
        from .data import to_device

        if not self.fused_ops:
            return self.model(to_device(features, self.model.device))["scores"]

        # Exact masks/ownership/routing are required, not just padded shapes.
        key = (
            tuple(
                (k, tuple(features[k].shape), tuple(features[k].flatten().tolist()))
                for k in ("prefix_mask", "doc_mask", "owners")
            ),
            repr(features.get("head_indices")),
            tuple(
                (k, tuple(features[k].shape), tuple(features[k].flatten().tolist()))
                for k in ("choice_indices", "choice_mask") if k in features
            ),
            torch.cuda.current_stream(self.model.device).cuda_stream,
        )
        with self.lock:
            if key in self.layouts:
                session = self.layouts.pop(key)
                if session is None:
                    session = SharedPrefixInference(self.model, features, _prepared=True)
                self.layouts[key] = session
                return session.score(features)
            # Bound graph memory. Capture only when a layout is seen again.
            if len(self.layouts) >= 2:
                self.layouts.popitem(last=False)
            self.layouts[key] = None
            return self.model(to_device(features, self.model.device))["scores"]


def inference_runtime(model, mode):
    """Resolve optimized CUDA/FA2 inference, with the portable legacy fallback."""
    if mode not in {"optimized", "fast", "legacy"}:
        raise ValueError("inference must be 'optimized', 'fast' or 'legacy'")
    if mode == "legacy":
        return None
    with _RUNTIME_LOCK:
        revision = (_revision(model), mode)
        cached = _RUNTIMES.get(model)
        if cached is None or cached[0] != revision:
            cached = (revision, _Runtime(model, fused_ops=mode == "fast"))
            _RUNTIMES[model] = cached
        return cached[1]
