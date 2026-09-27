import json

import pytest
from datasets import Dataset

from bekko_system_one.data import Group
from bekko_system_one.dataset_schema import VERSION, dataset_features
from bekko_system_one.dataset_training import training_view
from bekko_system_one.release import DecisionSource, render_group
from bekko_system_one.stratification import select_balanced_cases


def native_case(case_id="c1", *, label=True, include_target=True):
    decisions = [
        {
            "id": "choice",
            "kind": "judgment",
            "type": "choice",
            "instructions_json": json.dumps({"ask": "pick"}),
            "system_prompt": "choose",
            "criteria": [
                {"id": "a", "description_json": json.dumps({"text": "A"}), "value": None},
                {"id": "b", "description_json": json.dumps("B"), "value": None},
            ],
            "documents": [],
            "scoring": None,
        },
        {
            "id": "binary",
            "kind": "judgment",
            "type": "noul",
            "instructions_json": json.dumps("Is it true?"),
            "system_prompt": "",
            "criteria": [
                {"id": "false", "description_json": json.dumps("No"), "value": None},
                {"id": "true", "description_json": json.dumps("Yes"), "value": None},
            ],
            "documents": [],
            "scoring": None,
        },
        {
            "id": "rating",
            "kind": "judgment",
            "type": "score",
            "instructions_json": json.dumps("Rate it"),
            "system_prompt": "",
            "criteria": [
                {"id": "low", "description_json": json.dumps("Low"), "value": 0.0},
                {"id": "high", "description_json": json.dumps("High"), "value": 1.0},
            ],
            "documents": [],
            "scoring": None,
        },
        {
            "id": "rank",
            "kind": "ranking",
            "type": None,
            "instructions_json": json.dumps({"query": "find answers"}),
            "system_prompt": "rank",
            "criteria": [],
            "documents": [
                {"id": "d1", "content_json": json.dumps({"text": "Document A"})},
                {"id": "d2", "content_json": json.dumps("Document B")},
            ],
            "scoring": "relative",
        },
    ]
    targets = []
    probs = {
        "choice": [1.0, 0.0] if label else [0.0, 1.0],
        "binary": [0.0, 1.0] if label else [1.0, 0.0],
        "rating": [1.0, 0.0],
        "rank": [1.0, 0.0],
    }
    ids = {
        "choice": ["a", "b"],
        "binary": ["false", "true"],
        "rating": ["low", "high"],
        "rank": ["d1", "d2"],
    }
    if include_target:
        for did in ("choice", "binary", "rating", "rank"):
            targets.append(
                {
                    "decision_id": did,
                    "kind": "ranking_distribution" if did == "rank" else "judgment_distribution",
                    "annotation_kind": "native",
                    "ids": ids[did],
                    "probabilities": probs[did],
                    "metadata_json": "{}",
                }
            )
    return {
        "schema_version": VERSION,
        "case_id": case_id,
        "group_id": case_id,
        "input_hash": "hash",
        "split": "train",
        "language": "en",
        "input": {
            "state_json": json.dumps({"query": "Q", "context": [1, 2]}),
            "decisions": decisions,
        },
        "targets": targets,
        "provenance_json": '{"private":"ignored"}',
        "legacy_aux_json": None,
    }


def test_native_rows_render_all_decision_kinds_without_metadata_leaks():
    row = native_case()
    view = training_view(row)
    assert "source_json" not in view and "private" not in json.dumps(view)
    groups = [render_group(row, i) for i in range(4)]
    assert all(isinstance(group, Group) for group in groups)
    assert groups[0].candidates[0].endswith('{"text":"A"}')
    assert groups[0].query.startswith('choose\n\nInstruction: {"ask":"pick"}')
    assert "Candidate: false: No" in groups[1].candidates[0]
    assert "Candidate: low: Low" in groups[2].candidates[0]
    assert groups[3].candidates == ['Document: {"text":"Document A"}', "Document: Document B"]


def test_native_missing_target_is_rejected_by_training_renderer():
    row = native_case(include_target=False)
    with pytest.raises(ValueError, match="Decision/target alignment"):
        DecisionSource(Dataset.from_list([row], features=dataset_features()))


def test_native_balanced_sampling_is_deterministic():
    dataset = Dataset.from_list(
        [native_case(f"c{i}", label=bool(i % 2)) for i in range(8)], features=dataset_features()
    )
    first, report1 = select_balanced_cases(dataset, cap=4, seed=17, name="native")
    second, report2 = select_balanced_cases(dataset, cap=4, seed=17, name="native")
    assert first == second
    assert report1 == report2
