from dataclasses import asdict, replace

import pytest
import torch
from test_model import model_for

from bekko_system_one import (
    ChoicePrediction,
    DecisionMetadata,
    Group,
    InferenceEngine,
    NoulPrediction,
    ScorePrediction,
    interpret_prediction,
    predict,
    predict_typed,
)
from bekko_system_one.data import collate_groups, pack_groups, prepare_groups
from bekko_system_one.metrics import DecisionMetrics
from bekko_system_one.release import macro_metrics
from bekko_system_one.usage import TrainingUsage, batch_usage


def score_group():
    return Group(
        "query",
        ["good", "bad", "other", "long", "text"],
        "score",
        [0, 0, 0, 1, 0],
        metadata=DecisionMetadata(
            candidate_ids=("a", "b", "c", "d", "e"),
            candidate_values=(0, 1, 2, 3, 4),
            kind="judgment",
            case_id="case-a",
            decision_id="rating",
        ),
    )


def test_score_expectation_and_permutation():
    g = score_group()
    p = [0, 0, 0.12, 0.60, 0.28]
    answer = interpret_prediction(g, p)
    assert isinstance(answer, ScorePrediction)
    assert answer.score == pytest.approx(3.16)
    assert answer.normalized_score == pytest.approx(0.79)
    assert g.metadata is not None and g.target is not None
    assert g.metadata.candidate_ids is not None and g.metadata.candidate_values is not None
    order = [3, 0, 4, 1, 2]
    shuffled = replace(
        g,
        candidates=[g.candidates[i] for i in order],
        target=[g.target[i] for i in order],
        metadata=replace(
            g.metadata,
            candidate_ids=tuple(g.metadata.candidate_ids[i] for i in order),
            candidate_values=tuple(g.metadata.candidate_values[i] for i in order),
        ),
    )
    other = interpret_prediction(shuffled, [p[i] for i in order])
    assert other == answer
    assert Group.from_dict(asdict(g)) == g


def test_nonuniform_scale_and_yes_id_not_position():
    g = Group(
        "query",
        ["good", "bad"],
        "score",
        metadata=DecisionMetadata(
            candidate_ids=("high", "low"), candidate_values=(0.7, 0.3), kind="judgment"
        ),
    )
    answer = interpret_prediction(g, [0.6, 0.4])
    assert isinstance(answer, ScorePrediction)
    assert answer.score == pytest.approx(0.54)
    for ids, p in [(("true", "false"), [0.3, 0.7]), (("no", "yes"), [0.7, 0.3])]:
        g = Group("query", ["good", "bad"], "noul", metadata=DecisionMetadata(candidate_ids=ids))
        answer = interpret_prediction(g, p)
        assert isinstance(answer, NoulPrediction)
        assert answer.probability_yes == pytest.approx(0.3)


def test_metrics_eligibility_and_binary_brier():
    g = score_group()
    m = DecisionMetrics()
    m.update(g, torch.tensor([1e-30, 1e-30, 0.12, 0.60, 0.28]).log())
    # Add a ranking group with the same score head; it must not dilute ordinal MAE.
    rank = Group(
        "query", ["good", "bad"], "score", [0.5, 0.5], metadata=DecisionMetadata(kind="ranking")
    )
    m.update(rank, torch.zeros(2))
    m.update(Group("query", ["good", "bad"], "score", [0.5, 0.5]), torch.zeros(2))
    result = m.result()
    assert result["groups"] == 3
    assert result["score_mae"] == pytest.approx(0.16, abs=1e-6)
    assert result["score_normalized_mae"] == pytest.approx(0.04, abs=1e-6)
    assert result["metric_counts"]["score_mae"] == 1
    assert result["binary_brier"] is None
    n = DecisionMetrics()
    noul = Group(
        "query",
        ["good", "bad"],
        "noul",
        [0.7, 0.3],
        metadata=DecisionMetadata(candidate_ids=("true", "false")),
    )
    n.update(noul, torch.tensor([0.3, 0.7]).log())
    assert n.result()["binary_brier"] == pytest.approx(0.16)
    assert n.result()["brier"] == pytest.approx(0.32)
    r = DecisionMetrics()
    r.update(rank, torch.zeros(2))
    summary = macro_metrics({"mixed": {"score": result}, "ranking": {"score": r.result()}})
    assert summary["tasks"]["score"]["score_mae"] == pytest.approx(0.16, abs=1e-6)
    assert summary["tasks"]["score"]["metric_counts"]["score_mae"] == 1
    assert summary["tasks"]["score"]["metric_datasets"]["score_mae"] == 1
    assert r.result()["score_mae"] is None


def test_equal_expectation_different_distribution():
    g = Group(
        "query",
        ["good", "bad", "other"],
        "score",
        [0, 1, 0],
        metadata=DecisionMetadata(candidate_values=(0, 1, 2), kind="judgment"),
    )
    results = []
    for p in ([0.1, 0.8, 0.1], [0.4, 0.2, 0.4]):
        m = DecisionMetrics()
        m.update(g, torch.tensor(p).log())
        results.append(m.result())
    assert all(r["score_mae"] == pytest.approx(0, abs=1e-7) for r in results)
    assert results[0]["cross_entropy"] < results[1]["cross_entropy"]
    assert results[0]["brier"] < results[1]["brier"]


def test_typed_api_metadata_survives_packing_but_not_features(tiny_encoder):
    model = model_for(tiny_encoder)
    groups = [
        score_group(),
        Group(
            "query",
            ["good", "bad"],
            "choice",
            metadata=DecisionMetadata(candidate_ids=("a", "b"), case_id="case-b"),
        ),
    ]
    expected = [
        interpret_prediction(g, p) for g, p in zip(groups, predict(model, groups), strict=True)
    ]
    assert predict_typed(model, groups) == expected
    assert InferenceEngine(model).predict_typed(groups) == expected
    assert isinstance(expected[1], ChoicePrediction)
    prepared = prepare_groups(groups, tiny_encoder)
    packed = pack_groups(prepared, 20)
    assert {g.metadata.case_id for b in packed for g in b} == {"case-a", "case-b"}
    assert "metadata" not in collate_groups(prepared, tiny_encoder)
    assert prepared[0].metadata == groups[0].metadata
    with pytest.raises(ValueError, match="Typed prediction"):
        predict_typed(model, [Group("query", ["good"])])


def test_usage_counts_unique_cases_and_repeated_decisions(tiny_encoder):
    g = score_group()
    n = Group(
        "query",
        ["good", "bad"],
        "noul",
        [0.7, 0.3],
        metadata=DecisionMetadata(
            candidate_ids=("false", "true"), case_id="case-a", kind="judgment"
        ),
    )
    groups = [g, n]
    usage = batch_usage(prepare_groups(groups, tiny_encoder))
    tracker = TrainingUsage()
    tracker.update("source", groups, usage)
    tracker.update("source", groups, usage)
    result = tracker.result()["sources"]["source"]
    assert result["unique_cases"] == 1
    assert result["tasks"]["score"]["unique_cases"] == 1
    assert result["tasks"]["noul"]["decisions"] == 2
    assert result["tasks"]["score"]["candidates"] == 10
    assert usage["score"]["attention_work_tokens"] == (
        5 * usage["score"]["query_tokens"] + usage["score"]["candidate_tokens"]
    )
    anonymous = Group("query", ["good"], "choice")
    tracker.update("legacy", [anonymous], batch_usage(prepare_groups([anonymous], tiny_encoder)))
    assert tracker.result()["sources"]["legacy"]["unique_cases"] is None
    assert not tracker.result()["sources"]["legacy"]["case_identity_complete"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"candidate_ids": ["a", "a"]},
        {"candidate_values": [0, float("nan")]},
        {"candidate_values": [1, 1]},
        {"kind": "unknown"},
    ],
)
def test_invalid_metadata(metadata):
    with pytest.raises(ValueError):
        DecisionMetadata(**metadata)


def test_native_renderer_and_packed_evaluation(tiny_encoder):
    from datasets import Dataset
    from test_dataset_training import native_case

    from bekko_system_one.dataset_schema import dataset_features
    from bekko_system_one.release import DecisionSource
    from bekko_system_one.training import evaluate

    source = DecisionSource(Dataset.from_list([native_case()], features=dataset_features()))
    groups = source.groups([0, 1, 2, 3])
    assert groups[2].metadata.candidate_values == (0.0, 1.0)
    assert groups[3].metadata.kind == "ranking"
    metrics = evaluate(model_for(tiny_encoder), {"native": source}, 4, 40)["native"]
    assert metrics["score"]["groups"] == 2
    assert metrics["score"]["metric_counts"]["score_mae"] == 1
    assert metrics["score"]["metric_counts"]["cross_entropy"] == 2
    assert metrics["noul"]["metric_counts"]["binary_brier"] == 1
    assert metrics["noul"]["brier"] == pytest.approx(2 * metrics["noul"]["binary_brier"])
    assert metrics["choice"]["score_mae"] is None
