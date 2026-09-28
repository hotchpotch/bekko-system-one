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


@pytest.mark.parametrize("aux", [None, "", "{}", " { } "])
def test_native_empty_legacy_aux_uses_structured_inputs(aux):
    row = native_case()
    expected = training_view(row)
    row["legacy_aux_json"] = aux
    assert training_view(row) == expected
    source = DecisionSource(Dataset.from_list([row], features=dataset_features()))
    assert len(source.groups(list(range(len(source))))) == 4


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


@pytest.mark.parametrize("layout", ["instruction_state", "state_instruction"])
@pytest.mark.parametrize("state", [{"noul": "original", "context": "evidence"}, [1, 2], "text", None])
def test_noul_definitions_share_training_and_inference_state(layout, state):
    from copy import deepcopy

    from bekko_system_one import render_input_group

    row = native_case()
    row["input"]["state_json"] = json.dumps(state)
    decision = row["input"]["decisions"][1]
    decision["criteria"][0]["description_json"] = json.dumps("Not supported, not necessarily false")
    decision["criteria"][1]["description_json"] = json.dumps("Supported by the evidence")
    # Both candidate and target order are independent of yes/no mapping.
    decision["criteria"].reverse()
    before = deepcopy(row)
    trained = render_group(row, 1, prefix_layout=layout)
    inferred = render_input_group(row["input"], 1, prefix_layout=layout)
    assert inferred.target is None
    assert trained.target == [1.0, 0.0]
    assert (trained.query, trained.candidates, trained.query_parts) == (
        inferred.query, inferred.candidates, inferred.query_parts
    )
    assert json.loads(inferred.query_parts.context) == {
        "noul": {"yes": "Supported by the evidence", "no": "Not supported, not necessarily false"},
        "state": state,
    }
    assert row == before
    # Each decision sees only its own definitions; other tasks retain their state.
    for position in (0, 2, 3):
        group = render_input_group(row["input"], position, prefix_layout=layout)
        assert group.query_parts.context == row["input"]["state_json"]


def test_noul_requires_explicit_binary_criteria_without_fallback():
    from bekko_system_one import render_input_group

    row = native_case()
    criteria = row["input"]["decisions"][1]["criteria"]
    criteria[0]["id"], criteria[1]["id"] = "no", "yes"
    group = render_input_group(row["input"], 1)
    assert json.loads(group.query_parts.context)["noul"] == {"yes": "Yes", "no": "No"}
    criteria.pop()
    with pytest.raises(ValueError, match="Noul requires explicit"):
        render_input_group(row["input"], 1)


def test_renew_multiple_noul_decisions_do_not_share_definitions():
    from copy import deepcopy

    from bekko_system_one import render_input_group

    row = native_case()
    second = deepcopy(row["input"]["decisions"][1])
    second["id"] = "another"
    second["criteria"][0]["description_json"] = json.dumps("Different negative meaning")
    row["input"]["decisions"].append(second)
    first = render_input_group(row["input"], 1)
    other = render_input_group(row["input"], 4)
    assert "Different negative meaning" not in first.query
    assert "Different negative meaning" in other.query
