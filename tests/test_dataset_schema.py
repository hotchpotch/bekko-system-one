import json

import pytest
from datasets import Dataset

from bekko_system_one.dataset_schema import (
    VERSION,
    dataset_features,
    from_legacy,
    inference_input,
    to_jev_request,
    to_legacy,
    validate_row,
)


def legacy(*, ranking=False, score=False):
    options = [
        {
            "id": "b",
            "description": "Second",
            "value": 1 if score else None,
            "metadata_json": '{"x":2}',
        },
        {"id": "a", "description": "First", "value": 0 if score else None, "metadata_json": "{}"},
    ]
    did = "rank" if ranking else "d"
    return {
        "schema_version": "typed_decision.v1",
        "case_id": "case",
        "group_id": "group",
        "input_hash": "hash",
        "split": "train",
        "language": "en",
        "state_json": '{"query":"q","ctx":3}',
        "source_json": '{"source":"demo"}',
        "metadata_json": '{"extra":true}',
        "decisions": [
            {
                "decision_id": did,
                "type": "score" if ranking or score else "choice",
                "instructions": "legacy instruction",
                "options": options,
                "target": {
                    "kind": "positive_distribution" if ranking else "annotator_distribution",
                    "option_ids": ["b", "a"],
                    "probabilities": [0.75, 0.25],
                    "metadata_json": '{"annotation_counts":[3,1]}',
                },
                "metadata_json": '{"decision_extra":"kept"}',
            }
        ],
        "decision_prompts": [
            {
                "decision_id": did,
                "system_prompt": "sys",
                "instruction": "effective instruction",
                "input_format": "reranking" if ranking else "choice",
                "prompt_extra": "kept",
            }
        ],
        "evaluation_suite": "suite-a",
    }


@pytest.mark.parametrize("row", [legacy(), legacy(score=True), legacy(ranking=True)])
def test_legacy_roundtrip_is_exact_and_structured_valid(row):
    converted = from_legacy(row)
    assert converted["schema_version"] == VERSION
    assert to_legacy(converted) == row
    validate_row(converted)


def test_features_and_score_criteria_canonical_order_preserve_target_ids():
    converted = from_legacy(legacy(score=True))
    assert [c["id"] for c in converted["input"]["decisions"][0]["criteria"]] == ["a", "b"]
    assert converted["targets"][0]["ids"] == ["b", "a"]
    assert dataset_features()["input"]["decisions"].feature["kind"].dtype == "string"
    arrow = Dataset.from_list([converted], features=dataset_features())
    restored = arrow[0]
    assert to_legacy(restored) == legacy(score=True)
    assert json.dumps(to_legacy(restored), sort_keys=True) == json.dumps(
        legacy(score=True), sort_keys=True
    )


def test_ranking_is_relative_and_inference_projection_has_no_label_fields():
    converted = from_legacy(legacy(ranking=True))
    decision = converted["input"]["decisions"][0]
    assert (decision["kind"], decision["type"], decision["scoring"]) == (
        "ranking",
        None,
        "relative",
    )
    parsed = inference_input(converted)
    request = to_jev_request(
        converted,
        ranking_question={
            "type": "noul",
            "instructions": "Assess document relevance.",
            "criteria": {"false": "No", "true": "Yes"},
        },
    )
    assert parsed["state"] == json.loads(converted["input"]["state_json"])
    assert request["state"] == parsed["state"]
    assert len(request["questions"]) == 2
    assert all(q["type"] == "noul" for q in request["questions"].values())
    assert "targets" not in parsed and "provenance" not in json.dumps(parsed)


def test_rejects_bad_distribution():
    converted = from_legacy(legacy())
    converted["targets"][0]["probabilities"] = [0.7, 0.2]
    with pytest.raises(ValueError, match="sum to one"):
        validate_row(converted)


@pytest.mark.parametrize("missing", [True, False])
def test_absent_and_null_legacy_fields_survive_arrow(missing):
    row = legacy(score=True)
    decision = row["decisions"][0]
    if missing:
        del decision["instructions"]
        del decision["options"][0]["metadata_json"]
    else:
        decision["instructions"] = None
        decision["options"][0]["metadata_json"] = None
    row["decision_prompts"][0]["system_prompt"] = ""
    restored = to_legacy(Dataset.from_list([from_legacy(row)], features=dataset_features())[0])
    assert json.dumps(restored, sort_keys=True) == json.dumps(row, sort_keys=True)
