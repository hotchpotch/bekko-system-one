"""Supplemental releases retain explicit splits and native typed inputs."""

import json

import pytest
from datasets import Dataset

from bekko_system_one.dataset_schema import from_legacy
from bekko_system_one.release_mixing import load_additional_releases


def save_native_release(root, count=5):
    rows = []
    for index in range(count):
        rows.append(
            from_legacy(
                {
                    "case_id": str(index),
                    "group_id": str(index),
                    "input_hash": str(index),
                    "split": "train",
                    "language": "en",
                    "state_json": '{"evidence":"visible"}',
                    "source_json": '{"secret":"hidden"}',
                    "decisions": [
                        {
                            "decision_id": "d",
                            "type": "choice",
                            "instructions": "Choose.",
                            "options": [
                                {"id": "a", "description": "first"},
                                {"id": "b", "description": "second"},
                            ],
                            "target": {
                                "kind": "distribution",
                                "option_ids": ["b", "a"],
                                "probabilities": [0.2, 0.8],
                            },
                        }
                    ],
                    "decision_prompts": [
                        {"decision_id": "d", "instruction": "Choose.", "system_prompt": ""}
                    ],
                }
            )
        )
    root.mkdir()
    for split, selected in [("train", rows), ("validation", rows[:2]), ("test", rows[-2:])]:
        Dataset.from_list([{**r, "split": split} for r in selected]).save_to_disk(
            str(root / "task" / split)
        )
    (root / "training-manifest.json").write_text(
        json.dumps(
            {
                "train": [{"dataset": "task", "split": "train"}],
                "evaluation": [
                    {"dataset": "task", "split": "validation"},
                    {"dataset": "task", "split": "test"},
                ],
            }
        )
    )


def load(roots, existing_names=()):
    return load_additional_releases(
        roots,
        existing_names=set(existing_names),
        prefix_layout="instruction_state",
        validation_samples=None,
        validation_sampling="uniform",
        validation_seed=42,
        hub_pins={},
    )


def test_namespaced_sources_preserve_soft_targets_and_never_use_test(tmp_path):
    root = tmp_path / "release"
    save_native_release(root)
    sources, validation, audits, resolved = load({"supplement": str(root)}, ["task"])
    assert len(sources["supplement/task"]) == 5
    assert len(validation["supplement/task"]) == 2
    group = sources["supplement/task"].groups([0])[0]
    assert group.target == [0.8, 0.2]
    assert "hidden" not in group.query
    assert set(audits["validation"]["supplement"]["datasets"]["task"]["splits"]) == {"validation"}
    assert resolved == {"supplement": str(root)}
    with pytest.raises(ValueError, match="collision"):
        load({"supplement": str(root)}, ["supplement/task"])


@pytest.mark.parametrize("roots", [[], None, {"bad/name": "unused"}, {"": "unused"}, {1: "unused"}])
def test_invalid_release_namespaces(roots):
    with pytest.raises(ValueError, match="additional_roots"):
        load(roots)


def test_native_round_training_counts_metadata_and_checkpoint_reload(tmp_path, tiny_encoder):
    from sentence_transformers import SentenceTransformer

    from bekko_system_one.modules import DecisionHeads
    from bekko_system_one.training import run

    base, supplement = tmp_path / "base", tmp_path / "supplement"
    save_native_release(base, count=5)
    save_native_release(supplement, count=4)
    checkpoint = tmp_path / "checkpoint"
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    )
    model.save_pretrained(str(checkpoint), create_model_card=False)
    output = tmp_path / "run"
    result = run(
        {
            "seed": 42,
            "model": {"checkpoint": str(checkpoint)},
            "data": {
                "root": str(base),
                "additional_roots": {"supplement": str(supplement)},
                "sampling": "round_passes",
                "rounds": 2,
                "dataset_passes": {"supplement/task": 2},
                "dataset_samples": {"task": 3},
                "prefix_layout_weights": {"instruction_state": 1, "state_instruction": 1},
            },
            "evaluation": {"validation_samples": None},
            "training": {
                "output_dir": str(output),
                "device": "cpu",
                "batch_size": 2,
                "initial_tokens": 128,
                "learning_rate": 1e-4,
                "head_learning_rate": 1e-4,
                "eval_steps": 6,
            },
        }
    )
    assert result["steps"] == 6
    assert result["trained_decisions"] == 11
    plan = json.loads((output / "sampling_plan.json").read_text())
    assert plan["planned_decisions"] == {"task": 3, "supplement/task": 8}
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert len(history) == 6
    assert {entry["round"] for entry in history} == {1, 2}
    assert {entry["source_pass"] for entry in history if entry["source"] == "supplement/task"} == {
        1,
        2,
    }
    audit = json.loads((output / "data_manifest.json").read_text())
    assert "supplement" in audit["train"]["additional_releases"]
    metrics = json.loads((output / "validation-instruction_state.json").read_text())
    assert set(metrics) == {"task", "supplement/task"}
    assert (output / "model" / "release_rendering.json").exists()
