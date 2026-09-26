"""Public model construction and candidate-group inference."""

import torch
from sentence_transformers import SentenceTransformer

from .data import Group, collate_groups, prepare_batch, prepare_groups, to_device, work_tokens
from .inference import inference_runtime
from .modules import DecisionHeads, SharedPrefix


def build_model(model_name_or_path, *, tasks=("reranker",), device="cuda", **kwargs):
    kwargs.setdefault("lora", dict(rank=16, alpha=32, dropout=0.0))
    encoder = SharedPrefix(model_name_or_path, **kwargs)
    return SentenceTransformer(
        modules=[encoder, DecisionHeads(encoder.hidden_size, tasks)],
        device=device,
    )


@torch.inference_mode()
def predict(model, groups, *, inference="optimized", token_budget=16000):
    """Return candidate probabilities with optimized inference by default.

    Tokenize once and batch complete groups within token_budget (an oversized
    group runs alone). CUDA/FA2 uses a cached BF16 snapshot. Explicit fast
    inference adds Triton fusion and captures repeated layouts. inference="legacy" retains the original single-batch model forward.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if inference not in {"optimized", "fast", "legacy"}:
        raise ValueError("inference must be 'optimized', 'fast' or 'legacy'")
    if not groups:
        return []
    was_training = model.training
    try:
        model.eval()
        runtime = inference_runtime(model, inference)
        if inference == "legacy":
            scores = model(prepare_batch(model, groups))["scores"].flatten()
            return [
                s.float().softmax(0).cpu()
                for s in scores.split([len(g.candidates) for g in groups])
            ]
        prepared = prepare_groups(groups, model[0])
        results, batch = [], []

        def flush():
            features = collate_groups(batch, model[0])
            scores = (
                runtime.score(features)
                if runtime is not None
                else model(to_device(features, model.device))["scores"]
            )
            results.extend(
                s.float().softmax(0).cpu()
                for s in scores.flatten().split([len(g.documents) for g in batch])
            )

        for group in prepared:
            # Account for dense document storage as well as attention work.
            if batch and work_tokens([*batch, group]) > token_budget:
                flush()
                batch = []
            batch.append(group)
        if batch:
            flush()
        return results
    finally:
        model.train(was_training)


@torch.inference_mode()
def rank(model, query, documents, *, chunk_size=32, inference="optimized"):
    """Score documents with one prefix computation across all candidate chunks.

    Returns raw logits in the original document order. Sort descending to rank.
    The cache lives only for this call, so model updates cannot leave stale K/V.
    """
    if inference not in {"optimized", "fast", "legacy"}:
        raise ValueError("inference must be 'optimized', 'fast' or 'legacy'")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not documents:
        return torch.empty(0)
    Group(query, documents)
    if "reranker" not in model[1].tasks:
        raise ValueError("This model has no reranker head")
    was_training = model.training
    try:
        model.eval()
        runtime = inference_runtime(model, inference)
        active = runtime.model if runtime is not None else model
        encoder = active[0]
        with torch.autocast(
            model.device.type, dtype=torch.bfloat16, enabled=model.device.type == "cuda"
        ):
            qids, dids = encoder.tokenize_branches([query], documents)
            first = to_device(encoder.collate_tokens(qids, dids[:1], [0]), model.device)
            cache = encoder.encoder.encode_prefix(first["prefix_ids"], first["prefix_mask"])
            scores = []
            for start in range(0, len(dids), chunk_size):
                chunk = dids[start : start + chunk_size]
                f = to_device(encoder.collate_tokens(qids, chunk, [0] * len(chunk)), model.device)
                hidden = encoder.encoder.score_documents(
                    cache,
                    first["prefix_mask"],
                    f["doc_ids"],
                    f["doc_mask"],
                    f["owners"],
                )
                features = dict(
                    sentence_embedding=encoder.pool(hidden, f["doc_mask"]),
                    head_indices={"reranker": None},
                )
                scores.append(active[1](features)["scores"].flatten())
            return torch.cat(scores).float().cpu()
    finally:
        model.train(was_training)


class InferenceEngine:
    """Candidate-group inference with optional, lazily compiled Triton kernels.

    The default uses BF16 snapshots and batching without custom Triton kernels.
    prepare_fast_inference() opts into fusion and automatic graph preparation;
    representative inputs are optional, and unseen layouts always remain valid.
    """

    def __init__(self, model, *, inference="optimized", token_budget=16000):
        if inference not in {"optimized", "fast", "legacy"}:
            raise ValueError("inference must be 'optimized', 'fast' or 'legacy'")
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self.model = model
        self.inference = inference
        self.token_budget = token_budget

    @torch.inference_mode()
    def prepare_fast_inference(self, example_groups=None):
        """Enable fusion; optionally warm up prediction with representative groups.

        Without examples, no tokenization or kernel execution is needed here.
        Compilation occurs on first use when no matching Triton cache exists.
        Repeated layouts automatically acquire CUDA Graphs. CPU/non-FA2 models
        continue using their portable forward. Returns self for chaining.
        """
        was_training = self.model.training
        try:
            self.model.eval()
            inference_runtime(self.model, "fast")
        finally:
            self.model.train(was_training)
        self.inference = "fast"
        if example_groups is not None:
            groups = list(example_groups)
            # First pass initializes kernels; second prepares repeated layouts.
            self.predict(groups)
            self.predict(groups)
            if self.model.device.type == "cuda":
                torch.cuda.synchronize(self.model.device)
        return self

    def predict(self, groups):
        """Return probabilities in input order, using the selected inference mode."""
        return predict(self.model, groups, inference=self.inference, token_budget=self.token_budget)

    def rank(self, query, documents, *, chunk_size=32):
        """Return raw reranker scores, reusing the prefix across candidate chunks."""
        return rank(self.model, query, documents, chunk_size=chunk_size, inference=self.inference)
