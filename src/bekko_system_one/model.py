"""Public model construction and candidate-group inference."""

import torch
from sentence_transformers import SentenceTransformer

from .data import Group, prepare_batch, to_device
from .modules import DecisionHeads, SharedPrefix


def build_model(model_name_or_path, *, tasks=("reranker",), device="cuda", **kwargs):
    kwargs.setdefault("lora", dict(rank=16, alpha=32, dropout=0.0))
    encoder = SharedPrefix(model_name_or_path, **kwargs)
    return SentenceTransformer(
        modules=[encoder, DecisionHeads(encoder.hidden_size, tasks)],
        device=device,
    )


@torch.inference_mode()
def predict(model, groups):
    """Return one probability distribution per group, preserving candidate order."""
    if not groups:
        return []
    was_training = model.training
    try:
        model.eval()
        scores = model(prepare_batch(model, groups))["scores"].flatten()
        return [
            s.float().softmax(0).cpu() for s in scores.split([len(g.candidates) for g in groups])
        ]
    finally:
        model.train(was_training)


@torch.inference_mode()
def rank(model, query, documents, *, chunk_size=32):
    """Score documents with one prefix computation across all candidate chunks.

    Returns raw logits in the original document order. Sort descending to rank.
    The cache lives only for this call, so model updates cannot leave stale K/V.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not documents:
        return torch.empty(0)
    Group(query, documents)
    if "reranker" not in model[1].tasks:
        raise ValueError("This model has no reranker head")
    encoder = model[0]
    was_training = model.training
    try:
        model.eval()
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
                scores.append(model[1](features)["scores"].flatten())
            return torch.cat(scores).float().cpu()
    finally:
        model.train(was_training)
