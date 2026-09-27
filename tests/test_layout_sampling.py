import json

import pytest

from bekko_system_one.layout_sampling import LayoutSampler, validate_layout_weights


def test_layout_weights_reproducible_ratio_and_zero_weight():
    weights = {"instruction_state": 3, "state_instruction": 1}
    a = LayoutSampler(weights, 42)
    whole = a.sample(10000)
    b = LayoutSampler(dict(reversed(list(weights.items()))), 42)
    assert whole == b.sample(13) + b.sample(9987)
    assert 7300 < a.counts["instruction_state"] < 7700
    assert LayoutSampler({"state_instruction": 1}, 42).sample(20) == ["state_instruction"] * 20
    assert whole != LayoutSampler(weights, 43).sample(10000)


@pytest.mark.parametrize(
    "weights",
    [
        {},
        None,
        [],
        {"state": 1},
        {"instruction_state": 0},
        {"instruction_state": -1},
        {"instruction_state": True},
        {"instruction_state": "1"},
        {"instruction_state": float("nan")},
        {"instruction_state": float("inf")},
    ],
)
def test_invalid_layout_weights(weights):
    with pytest.raises(ValueError, match="prefix_layout_weights"):
        validate_layout_weights(weights)


def test_conflicting_layout_settings_rejected_before_loading(tmp_path):
    from bekko_system_one.training import run

    with pytest.raises(ValueError, match="not both"):
        run(
            dict(
                training=dict(device="cpu"),
                data=dict(
                    root=str(tmp_path),
                    prefix_layout="instruction_state",
                    prefix_layout_weights={"state_instruction": 1},
                ),
            )
        )
    with pytest.raises(ValueError, match="requires data.root"):
        run(
            dict(
                training=dict(device="cpu"),
                data=dict(sources={}, prefix_layout_weights={"state_instruction": 1}),
            )
        )


@pytest.mark.parametrize("mixed", [True, False])
def test_mixed_training_and_both_fixed_evaluations(tmp_path, tiny_encoder, monkeypatch, mixed):
    import torch
    from sentence_transformers import SentenceTransformer
    from test_release import save_release

    from bekko_system_one import training
    from bekko_system_one.modules import DecisionHeads

    save_release(tmp_path / "release")
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    )
    model.save_pretrained(str(tmp_path / "initial"), create_model_card=False)
    original_step, original_eval = training.train_step, training.evaluate
    observed, evaluated = [], []

    def step(model, groups, *args):
        observed.extend(
            "state_instruction" if g.query.startswith("State:") else "instruction_state"
            for g in groups
        )
        assert all(
            g.candidates[0] == "Candidate: b: bad" and g.target == [0.2, 0.8] for g in groups
        )
        return original_step(model, groups, *args)

    def evaluate(model, sources, *args):
        layouts = {s.prefix_layout for s in sources.values()}
        assert len(layouts) == 1
        evaluated.append(layouts.pop())
        return original_eval(model, sources, *args)

    monkeypatch.setattr(training, "train_step", step)
    monkeypatch.setattr(training, "evaluate", evaluate)
    out = tmp_path / "run"
    weights = dict(instruction_state=1, state_instruction=1)
    result = training.run(
        dict(
            seed=42,
            model=dict(checkpoint=str(tmp_path / "initial")),
            data=dict(
                root=str(tmp_path / "release"),
                sampling="uniform",
                epoch_fraction=1,
                dataset_samples={"a": 12},
                **(
                    dict(prefix_layout_weights=weights)
                    if mixed
                    else dict(prefix_layout="instruction_state")
                ),
            ),
            evaluation=dict(
                test_root=str(tmp_path / "release"),
                before_training=True,
                token_budget=128,
                prefix_layouts=list(weights),
            ),
            training=dict(
                device="cpu",
                output_dir=str(out),
                batch_size=4,
                eval_steps=2,
                initial_tokens=128,
                learning_rate=0.001,
                head_learning_rate=0.001,
            ),
        )
    )
    assert observed == (
        LayoutSampler(weights, 42).sample(12) if mixed else ["instruction_state"] * 12
    )
    assert result["trained_decisions"] == 12
    assert evaluated == ["instruction_state", "state_instruction"] * (len(evaluated) // 2)
    for label in ["test-initial", "validation-initial", "validation-2", "validation", "test"]:
        for layout in weights:
            assert (out / f"{label}-{layout}-summary.json").is_file()
        assert not (out / f"{label}-summary.json").exists()
    manifest = json.loads((out / "data_manifest.json").read_text())
    if mixed:
        assert manifest["train"]["prefix_layout_weights"] == weights
    assert manifest["test"]["prefix_layouts"] == list(weights)
    rendering = json.loads((out / "model/release_rendering.json").read_text())
    if mixed:
        assert rendering["prefix_layout_weights"] == weights
    restored = SentenceTransformer(
        str(out / "model"), device="cpu", local_files_only=True, trust_remote_code=True
    )
    assert any(not torch.equal(p, restored.state_dict()[n]) for n, p in model.state_dict().items())
    history = [json.loads(line) for line in (out / "history.jsonl").read_text().splitlines()]
    if mixed:
        assert sum(result["prefix_layout_counts"].values()) == 12
        assert all(sum(row["prefix_layout_counts"].values()) == row["groups"] for row in history)
