"""Typed interpretations of candidate distributions, independent of model execution."""

from dataclasses import dataclass

import torch

from .data import Group


@dataclass(frozen=True)
class ChoicePrediction:
    selected_id: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class NoulPrediction:
    probability_yes: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScorePrediction:
    score: float
    normalized_score: float
    probabilities: dict[str, float]
    values: dict[str, float]


def validate_typed_group(group: Group):
    metadata = group.metadata
    if metadata is None or metadata.candidate_ids is None or metadata.kind == "ranking":
        raise ValueError(
            "Typed prediction requires judgment candidate IDs; use predict for ranking"
        )
    if group.task == "noul" and metadata.yes_index is None:
        raise ValueError("Noul requires true/false or yes/no candidate IDs")
    if group.task == "score" and metadata.score_values is None:
        raise ValueError(
            "Score requires judgment metadata and at least two distinct numeric values"
        )
    if group.task not in {"choice", "noul", "score"}:
        raise ValueError("Typed prediction supports choice, noul and ordinal score")


def interpret_prediction(
    group: Group, probabilities
) -> ChoicePrediction | NoulPrediction | ScorePrediction:
    """Map probabilities to IDs and values; never infer numbers from candidate text."""
    validate_typed_group(group)
    p = torch.as_tensor(probabilities, dtype=torch.float64).detach().cpu()
    if (
        p.ndim != 1
        or len(p) != len(group.candidates)
        or not torch.isfinite(p).all()
        or (p < 0).any()
        or abs(p.sum().item() - 1) > 1e-5
    ):
        raise ValueError("Expected normalized probabilities aligned with candidates")
    metadata = group.metadata
    assert metadata is not None and metadata.candidate_ids is not None
    mapping = dict(zip(metadata.candidate_ids, p.tolist(), strict=True))
    if group.task == "choice":
        return ChoicePrediction(metadata.candidate_ids[int(p.argmax())], mapping)
    if group.task == "noul":
        assert metadata.yes_index is not None
        return NoulPrediction(p[metadata.yes_index].item(), mapping)
    values = metadata.score_values
    assert values is not None
    score = sum(prob * value for prob, value in zip(p.tolist(), values, strict=True))
    return ScorePrediction(
        score,
        (score - min(values)) / (max(values) - min(values)),
        mapping,
        dict(zip(metadata.candidate_ids, values, strict=True)),
    )
