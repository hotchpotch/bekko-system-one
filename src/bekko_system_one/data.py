"""Validated candidate groups, shared tokenization, and whole-group batching."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .query_budget import QueryParts


@dataclass(frozen=True)
class DecisionMetadata:
    """Non-tokenized candidate alignment and source identity; never model features."""

    candidate_ids: tuple[str, ...] | None = None
    candidate_values: tuple[float, ...] | None = None
    kind: str | None = None
    case_id: str | None = None
    group_id: str | None = None
    decision_id: str | None = None

    def __post_init__(self):
        if self.kind not in {None, "judgment", "ranking"}:
            raise ValueError("metadata kind must be judgment or ranking")
        if self.candidate_ids is not None:
            ids = tuple(self.candidate_ids)
            if not ids or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError("candidate_ids must be nonempty strings")
            if len(set(ids)) != len(ids):
                raise ValueError("candidate_ids must be unique")
            object.__setattr__(self, "candidate_ids", ids)
        if self.candidate_values is not None:
            values = tuple(self.candidate_values)
            if not values or any(not math.isfinite(v) for v in values):
                raise ValueError("candidate_values must be finite")
            if len(set(values)) != len(values):
                raise ValueError("candidate_values must be unique")
            object.__setattr__(self, "candidate_values", values)
        for identity in (self.case_id, self.group_id, self.decision_id):
            if identity is not None and not isinstance(identity, str):
                raise ValueError("metadata identities must be strings")

    @property
    def yes_index(self):
        ids = self.candidate_ids
        if ids is not None and set(ids) in ({"true", "false"}, {"yes", "no"}):
            return ids.index("true" if "true" in ids else "yes")
        return None

    @property
    def score_values(self):
        if self.kind == "judgment" and self.candidate_values is not None:
            if len(self.candidate_values) >= 2:
                return self.candidate_values
        return None


@dataclass(frozen=True)
class Group:
    query: str
    candidates: list[str]
    task: str = "reranker"
    target: list[float] | None = None
    query_parts: QueryParts | None = None
    metadata: DecisionMetadata | None = None

    def __post_init__(self):
        if self.metadata is not None:
            for values in (self.metadata.candidate_ids, self.metadata.candidate_values):
                if values is not None and len(values) != len(self.candidates):
                    raise ValueError("candidate metadata must align with candidates")
        if self.query_parts is not None and self.query_parts.render() != self.query:
            raise ValueError("query must match query_parts.render()")
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
            query_parts=QueryParts(**row["query_parts"]) if row.get("query_parts") else None,
            metadata=DecisionMetadata(**row["metadata"]) if row.get("metadata") else None,
        )


@dataclass
class PreparedGroup:
    key: str | tuple[int, ...]
    task: str
    query: list[int]
    documents: list[list[int]]
    target: list[float] | None
    metadata: DecisionMetadata | None = None

    @property
    def cost(self):
        return len(self.query) * len(self.documents) + sum(map(len, self.documents))


def prepare_groups(groups, encoder):
    queries = list(dict.fromkeys((g.query, g.query_parts) for g in groups))
    documents = list(dict.fromkeys((g.task, d) for g in groups for d in g.candidates))
    qids, dids = encoder.tokenize_branches(
        [q for q, _ in queries],
        [d for _, d in documents],
        [t for t, _ in documents],
        query_parts=[p for _, p in queries],
    )
    qmap, dmap = dict(zip(queries, qids, strict=True)), dict(zip(documents, dids, strict=True))
    return [
        PreparedGroup(
            tuple(qmap[g.query, g.query_parts]),
            g.task,
            qmap[g.query, g.query_parts],
            [dmap[g.task, d] for d in g.candidates],
            g.target,
            g.metadata,
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
    choice_rows, offset = [], 0
    for group in groups:
        count = len(group.documents)
        if group.task == "choice":
            choice_rows.append(list(range(offset, offset + count)))
        offset += count
    if choice_rows:
        width = max(map(len, choice_rows))
        features["choice_indices"] = torch.tensor(
            [row + [0] * (width - len(row)) for row in choice_rows], dtype=torch.long
        )
        features["choice_mask"] = torch.tensor(
            [[True] * len(row) + [False] * (width - len(row)) for row in choice_rows]
        )
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
