import json

import pytest
from datasets import Dataset, DatasetDict

from bekko_system_one.release import load_release, macro_metrics, render_group


def case(i, *, ranking=False):
    decision = dict(
        decision_id="d",
        type="score" if ranking else "choice",
        options=[dict(id="b", description="bad"), dict(id="a", description="good")],
        target=dict(option_ids=["a", "b"], probabilities=[0.8, 0.2]),
    )
    return dict(
        case_id=str(i),
        state_json='{"query":"question"}',
        decisions=[decision, decision],
        decision_prompts=[
            dict(
                decision_id="d",
                instruction="Choose.",
                system_prompt="",
                input_format="reranking" if ranking else "candidate",
            )
        ]
        * 2,
        metadata_json='{"secret_label":"answer"}',
    )


def save_release(root):
    root.mkdir()
    data = Dataset.from_list([case(i) for i in range(12)])
    DatasetDict(
        {
            "train": data,
            "validation": data.select(range(6)),
            "validation_extra": data.select(range(6, 12)),
            "test": data,
        }
    ).save_to_disk(str(root / "a"))
    DatasetDict({"test": data}).save_to_disk(str(root / "test_only"))
    manifest = dict(
        train=[dict(dataset="a", split="train")],
        evaluation=[dict(dataset="a", split=s) for s in ["validation", "validation_extra", "test"]]
        + [dict(dataset="test_only", split="test")],
    )
    (root / "training-manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("sampling", ["uniform", "balanced"])
def test_sample_cases_across_splits_and_test_isolation(tmp_path, sampling):
    save_release(tmp_path / "release")
    sources, audit = load_release(
        tmp_path / "release", "validation", sample_cases=5, sampling=sampling
    )
    again, repeat = load_release(
        tmp_path / "release", "validation", sample_cases=5, sampling=sampling
    )
    assert list(sources) == ["a"]
    assert len(sources["a"].dataset) == 5
    assert len(sources["a"]) == 10
    assert repeat == audit
    assert sources["a"].groups([0, 1, 9]) == again["a"].groups([0, 1, 9])
    test, _ = load_release(tmp_path / "release", "test")
    assert len(test["test_only"]) == 24
    train, _ = load_release(tmp_path / "release", "train")
    assert list(train) == ["a"]
    assert len(train["a"]) == 24


def test_render_alignment_soft_targets_and_no_label_leakage():
    row = case(0)
    group = render_group(row, 0)
    assert group.target == [0.2, 0.8]
    assert group.query == 'Instruction: Choose.\nState: {"query":"question"}'
    assert group.candidates == ["Candidate: b: bad", "Candidate: a: good"]
    assert render_group(case(0, ranking=True), 0).candidates == ["Document: bad", "Document: good"]
    row["decision_prompts"][0]["decision_id"] = "wrong"
    with pytest.raises(ValueError, match="alignment"):
        render_group(row, 0)


def test_invalid_manifest_and_sample(tmp_path):
    save_release(tmp_path / "release")
    with pytest.raises(ValueError, match="sample_cases"):
        load_release(tmp_path / "release", "validation", sample_cases=0)
    (tmp_path / "release/training-manifest.json").write_text(
        json.dumps({"train": [{"dataset": "../outside", "split": "train"}]})
    )
    with pytest.raises(ValueError, match="Invalid manifest"):
        load_release(tmp_path / "release", "train")


def test_macro_is_not_weighted_by_dataset_size():
    metrics = macro_metrics(
        {
            "a": {"choice": dict(groups=10, accuracy=1.0, cross_entropy=0.0)},
            "b": {"choice": dict(groups=100, accuracy=0.0, cross_entropy=2.0)},
        }
    )
    assert metrics["tasks"]["choice"]["accuracy"] == 0.5
    assert metrics["mean_cross_entropy"] == 1.0


def test_exclusion_skips_loading_and_records_selection(tmp_path):
    root = tmp_path / "release"
    save_release(root)
    path = root / "training-manifest.json"
    manifest = json.loads(path.read_text())
    # An excluded source must never be opened, rendered or counted in the budget.
    manifest["train"].append(dict(dataset="excluded", split="train"))
    path.write_text(json.dumps(manifest))
    sources, audit = load_release(root, "train", exclude_datasets=["excluded"])
    assert list(sources) == ["a"]
    assert sum(map(len, sources.values())) == 24
    assert audit["excluded_splits"] == {"excluded": ["train"]}
    assert audit["exclude_datasets"] == ["excluded"]
    validation, _ = load_release(root, "validation", exclude_datasets=["a"])
    assert not validation
    test, _ = load_release(root, "test", exclude_datasets=["test_only"])
    assert list(test) == ["a"]
    unfiltered, _ = load_release(root, "test")
    assert set(unfiltered) == {"a", "test_only"}


@pytest.mark.parametrize("excluded", ["a", [None], [""], ["a", "a"], ["typo"]])
def test_exclusion_rejects_invalid_config(tmp_path, excluded):
    save_release(tmp_path / "release")
    with pytest.raises(ValueError, match="exclude_datasets"):
        load_release(tmp_path / "release", "train", exclude_datasets=excluded)


@pytest.mark.parametrize("prefix_layout", ["instruction_state", "state"])
def test_checkpoint_release_training_and_full_test(
    tmp_path, tiny_encoder, monkeypatch, prefix_layout
):
    import torch
    from sentence_transformers import SentenceTransformer

    from bekko_system_one import training
    from bekko_system_one.modules import DecisionHeads

    save_release(tmp_path / "release")
    manifest_path = tmp_path / "release/training-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["train"].append(dict(dataset="excluded", split="train"))
    manifest_path.write_text(json.dumps(manifest))
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint), create_model_card=False)
    original = {k: v.clone() for k, v in model.state_dict().items()}
    original_step = training.train_step
    checked = []

    def step(loaded, *args):
        assert args[0][0].query.startswith("State:" if prefix_layout == "state" else "Instruction:")
        if not checked:
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            assert loaded[0].query_length == 48
            checked.append(True)
        return original_step(loaded, *args)

    monkeypatch.setattr(training, "train_step", step)
    out = tmp_path / "run"
    training.run(
        dict(
            model=dict(checkpoint=str(checkpoint), query_length=48),
            data=dict(
                root=str(tmp_path / "release"),
                epoch_fraction=1,
                sampling_alpha=0.5,
                exclude_datasets=["excluded"],
                prefix_layout=prefix_layout,
            ),
            evaluation=dict(
                test_root=str(tmp_path / "release"),
                validation_samples=5,
                validation_sampling="balanced",
                before_training=True,
                token_budget=128,
                validation_exclude_datasets=["test_only"],
                test_exclude_datasets=["a"],
            ),
            training=dict(
                output_dir=str(out),
                device="cpu",
                max_steps=2,
                batch_size=4,
                learning_rate=1e-3,
                head_learning_rate=1e-3,
                eval_steps=1,
                initial_tokens=128,
            ),
        )
    )
    assert checked
    audit = json.loads((out / "data_manifest.json").read_text())
    assert all(audit[role]["prefix_layout"] == prefix_layout for role in audit)
    assert json.loads((out / "model/release_rendering.json").read_text()) == dict(
        prefix_layout=prefix_layout
    )
    assert audit["train"]["excluded_splits"] == {"excluded": ["train"]}
    assert audit["validation"]["datasets"]["a"]["cases"] == 5
    assert audit["validation"]["sampling"] == "balanced"
    assert "test_only" not in audit["validation"]["datasets"]
    results = json.loads((out / "test.json").read_text())
    assert set(results) == {"test_only"}
    assert results["test_only"]["choice"]["groups"] == 24
    restored = SentenceTransformer(
        str(out / "model"), device="cpu", local_files_only=True, trust_remote_code=True
    )
    assert restored[0].query_length == 48
    assert any(not torch.equal(v, original[k]) for k, v in restored.state_dict().items())
    assert (out / "test-initial-summary.json").is_file()


@pytest.mark.parametrize("ranking", [False, True])
def test_state_prefix_moves_all_instructions_and_preserves_targets(ranking):
    row = case(0, ranking=ranking)
    row["decision_prompts"] = [
        dict(row["decision_prompts"][0], instruction="good", system_prompt="first system"),
        dict(row["decision_prompts"][1], instruction="bad", system_prompt="second system"),
    ]
    a, b = [render_group(row, i, prefix_layout="state") for i in range(2)]
    assert a.query == b.query == 'State: {"query":"question"}'
    assert a.target == b.target == [0.2, 0.8]
    option = "Document: bad" if ranking else "Candidate: b: bad"
    assert a.candidates[0] == f"first system\n\nInstruction: good\n{option}"
    assert b.candidates[0] == f"second system\n\nInstruction: bad\n{option}"
    assert "secret_label" not in a.query + "".join(a.candidates)
    # Labels cannot affect inference text in either layout.
    row["decisions"][0]["target"]["probabilities"] = [1.0, 0.0]
    changed = render_group(row, 0, prefix_layout="state")
    assert (changed.query, changed.candidates) == (a.query, a.candidates)


def test_state_sharing_matches_separate_decisions_and_gradients(tiny_encoder):
    import torch
    from sentence_transformers import SentenceTransformer

    from bekko_system_one.data import prepare_batch
    from bekko_system_one.loss import distribution_loss_sum
    from bekko_system_one.modules import DecisionHeads

    row = case(0)
    row["decision_prompts"] = [
        dict(row["decision_prompts"][0], instruction="good"),
        dict(row["decision_prompts"][1], instruction="bad"),
    ]
    groups = [render_group(row, i, prefix_layout="state") for i in range(2)]
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    )
    model.eval()
    features = prepare_batch(model, groups)
    assert features["prefix_ids"].shape[0] == 1
    assert features["owners"].tolist() == [0, 0, 0, 0]
    together = model(features)["scores"]
    targets = [g.target for g in groups]
    joint_loss = distribution_loss_sum(together, targets) / len(groups)
    joint_loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    separate = []
    for group in groups:
        scores = model(prepare_batch(model, [group]))["scores"]
        separate.append(scores.detach())
        (distribution_loss_sum(scores, [group.target]) / len(groups)).backward()
    torch.testing.assert_close(together, torch.cat(separate), atol=1e-6, rtol=1e-5)
    for name, parameter in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(parameter.grad, grads[name], atol=1e-6, rtol=1e-4)
    assert any(g.abs().sum() > 0 for n, g in grads.items() if "lora_B" in n)


@pytest.mark.parametrize("layout", ["typo", None, 1, []])
def test_invalid_prefix_layout_is_rejected_before_loading(tmp_path, layout):
    with pytest.raises(ValueError, match="prefix_layout"):
        load_release(tmp_path / "missing", "train", prefix_layout=layout)
    with pytest.raises(ValueError, match="prefix_layout"):
        render_group(case(0), 0, prefix_layout=layout)


def test_prefix_layout_requires_structured_release(tmp_path):
    from bekko_system_one.training import run

    with pytest.raises(ValueError, match="prefix_layout requires data.root"):
        run(dict(training=dict(device="cpu"), data=dict(sources={}, prefix_layout="state")))
