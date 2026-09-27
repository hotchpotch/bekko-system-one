import json

import pytest
from datasets import Dataset, DatasetDict

from bekko_system_one.dataset_schema import dataset_features, from_legacy, to_legacy
from bekko_system_one.release import load_release, render_group


def legacy_case(index, positive, *, ranking=False):
    if ranking:
        decisions = [
            {
                "decision_id": "rank",
                "type": "score",
                "instructions": "Rank documents by relevance.",
                "options": [
                    {
                        "id": "d2",
                        "description": {"text": "second"},
                        "value": None,
                        "metadata_json": "{}",
                    },
                    {
                        "id": "d1",
                        "description": {"text": "first"},
                        "value": None,
                        "metadata_json": "{}",
                    },
                ],
                "target": {
                    "kind": "positive_distribution",
                    "option_ids": ["d1", "d2"],
                    "probabilities": [1.0, 0.0],
                    "metadata_json": '{"source":"qrels"}',
                },
                "metadata_json": "{}",
            }
        ]
        prompts = [
            {
                "decision_id": "rank",
                "system_prompt": "Ranking system.",
                "instruction": "Rank documents by relevance.",
                "input_format": "reranking",
            }
        ]
    else:
        decisions = [
            {
                "decision_id": "relevant",
                "type": "noul",
                "instructions": "Is the document relevant?",
                "options": [
                    {
                        "id": "false",
                        "description": "No.",
                        "value": None,
                        "metadata_json": "{}",
                    },
                    {
                        "id": "true",
                        "description": "Yes.",
                        "value": None,
                        "metadata_json": "{}",
                    },
                ],
                "target": {
                    "kind": "hard_label",
                    "option_ids": ["false", "true"],
                    "probabilities": [0.0 if positive else 1.0, 1.0 if positive else 0.0],
                    "metadata_json": "{}",
                },
                "metadata_json": "{}",
            },
            {
                "decision_id": "quality",
                "type": "score",
                "instructions": "Rate the answer quality.",
                "options": [
                    {
                        "id": "0",
                        "description": "Poor.",
                        "value": 0,
                        "metadata_json": "{}",
                    },
                    {
                        "id": "1",
                        "description": "Good.",
                        "value": 1,
                        "metadata_json": "{}",
                    },
                ],
                "target": {
                    "kind": "annotator_distribution",
                    "option_ids": ["0", "1"],
                    "probabilities": [0.25, 0.75],
                    "metadata_json": '{"n_annotations":4}',
                },
                "metadata_json": "{}",
            },
        ]
        prompts = [
            {
                "decision_id": d["decision_id"],
                "system_prompt": "",
                "instruction": d["instructions"],
                "input_format": "candidate",
            }
            for d in decisions
        ]
    return {
        "schema_version": "typed_decision.v1",
        "case_id": f"case-{index}",
        "group_id": f"group-{index}",
        "input_hash": f"hash-{index}",
        "split": "train",
        "language": "en",
        "state_json": json.dumps({"query": f"question {index}", "context": ["a", "b"]}),
        "source_json": '{"dataset":"fixture"}',
        "metadata_json": "{}",
        "decisions": decisions,
        "decision_prompts": prompts,
    }


def save_structured_release(root):
    root.mkdir()
    rows = [from_legacy(legacy_case(i, i % 2 == 1)) for i in range(12)]
    data = Dataset.from_list(rows, features=dataset_features())
    DatasetDict({"train": data, "validation": data.select(range(8))}).save_to_disk(
        str(root / "fixture")
    )
    (root / "training-manifest.json").write_text(
        json.dumps(
            {
                "train": [{"dataset": "fixture", "split": "train"}],
                "evaluation": [{"dataset": "fixture", "split": "validation"}],
            }
        )
    )


def test_structured_roundtrip_and_group_rendering_preserve_reranking_inputs():
    legacy = legacy_case(3, True, ranking=True)
    structured = from_legacy(legacy)
    assert to_legacy(structured) == legacy
    group = render_group(structured, 0)
    expected = render_group(legacy, 0)
    assert group == expected
    assert group.candidates == ["Document: {'text': 'second'}", "Document: {'text': 'first'}"]
    assert group.target == [0.0, 1.0]


@pytest.mark.parametrize("sampling", ["uniform", "balanced"])
def test_structured_release_sampling_and_source_offsets(tmp_path, sampling):
    root = tmp_path / "release"
    save_structured_release(root)
    sources, audit = load_release(root, "train", sample_cases=5, seed=13, sampling=sampling)
    source = sources["fixture"]
    assert len(source.dataset) == 5
    assert len(source) == 10
    assert len(set(audit["datasets"]["fixture"]["case_ids"])) == 5
    groups = source.groups(list(range(len(source))))
    assert [group.task for group in groups] == ["noul", "score"] * 5
    assert all(group.query.startswith("Instruction:") for group in groups)
    for case_position, case_id in enumerate(source.dataset["case_id"]):
        positive = int(case_id.removeprefix("case-")) % 2 == 1
        assert groups[case_position * 2].target == ([0.0, 1.0] if positive else [1.0, 0.0])
    assert all(group.target == [0.25, 0.75] for group in groups[1::2])


def test_structured_groups_restore_each_multidecision_case_once(tmp_path, monkeypatch):
    root = tmp_path / "release"
    save_structured_release(root)
    sources, _ = load_release(root, "train")
    source = sources["fixture"]
    import bekko_system_one.release as release

    original = release.training_view
    restored_ids = []

    def counted(row):
        restored_ids.append(row["case_id"])
        return original(row)

    monkeypatch.setattr(release, "training_view", counted)
    source.groups(list(range(len(source))))
    assert restored_ids == source.dataset["case_id"]
