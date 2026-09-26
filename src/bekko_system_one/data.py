"""Validated candidate groups, shared tokenization, and whole-group batching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Group:
    query: str
    candidates: list[str]
    task: str = "reranker"
    target: list[float] | None = None

    def __post_init__(self):
        if self.task not in {"reranker", "choice", "noul", "score"}:
            raise ValueError(f"Unknown task: {self.task}")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query must be nonempty text")
        if not self.candidates or any(not isinstance(c, str) for c in self.candidates):
            raise ValueError("candidates must contain text")
        if self.target is not None and (
            len(self.target) != len(self.candidates)
            or any(not math.isfinite(t) or t < 0 for t in self.target)
            or abs(sum(self.target) - 1) > 1e-5
        ):
            raise ValueError("target must be a normalized distribution aligned with candidates")

    @classmethod
    def from_dict(cls, row):
        # Metadata and targets are never concatenated into inference inputs.
        return cls(
            query=row["query"],
            candidates=row["candidates"],
            task=row.get("task", "reranker"),
            target=row.get("target"),
        )


@dataclass
class PreparedGroup:
    key: str
    task: str
    query: list[int]
    documents: list[list[int]]
    target: list[float] | None

    @property
    def cost(self):
        return len(self.query) * len(self.documents) + sum(map(len, self.documents))


def prepare_groups(groups, encoder):
    queries = list(dict.fromkeys(g.query for g in groups))
    documents = list(dict.fromkeys((g.task, d) for g in groups for d in g.candidates))
    qids, dids = encoder.tokenize_branches(
        queries, [d for _, d in documents], [t for t, _ in documents]
    )
    qmap, dmap = dict(zip(queries, qids, strict=True)), dict(zip(documents, dids, strict=True))
    return [
        PreparedGroup(
            g.query, g.task, qmap[g.query], [dmap[g.task, d] for d in g.candidates], g.target
        )
        for g in groups
    ]


def collate_groups(groups, encoder):
    queries, docs, owners, seen, indices = [], [], [], {}, {}
    for group in groups:
        if group.key not in seen:
            seen[group.key] = len(queries)
            queries.append(group.query)
        indices.setdefault(group.task, []).extend(
            range(len(docs), len(docs) + len(group.documents))
        )
        docs.extend(group.documents)
        owners.extend([seen[group.key]] * len(group.documents))
    features = encoder.collate_tokens(queries, docs, owners)
    features["head_indices"] = {next(iter(indices)): None} if len(indices) == 1 else indices
    return features


def to_device(features, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}


def prepare_batch(model, groups):
    """Build a feature dictionary accepted by an ordinary SentenceTransformer."""
    return to_device(collate_groups(prepare_groups(groups, model[0]), model[0]), model.device)


def work_tokens(items, padded_documents=True):
    cost = sum(x.cost for x in items)
    if padded_documents and items:
        lengths = [len(d) for x in items for d in x.documents]
        cost += len(lengths) * max(lengths) - sum(lengths)
    return cost


def pack_groups(items, budget, padded_documents=True):
    """First-fit decreasing; never split the candidates of a training group."""
    if budget < 1:
        raise ValueError("Token budget must be positive")
    groups, statistics, ordered = [], [], []
    for item in items:
        lengths = [len(d) for d in item.documents]
        ordered.append((item, item.cost, len(lengths), sum(lengths), max(lengths)))
    for item, cost, count, total, maximum in sorted(
        ordered, key=lambda x: (x[4], x[1]), reverse=True
    ):
        for i, (old_cost, old_count, old_total, old_max) in enumerate(statistics):
            combined = old_cost + cost, old_count + count, old_total + total, max(old_max, maximum)
            estimate = combined[0]
            if padded_documents:
                estimate += combined[1] * combined[3] - combined[2]
            if estimate <= budget:
                groups[i].append(item)
                statistics[i] = combined
                break
        else:
            groups.append([item])
            statistics.append((cost, count, total, maximum))
    return groups
