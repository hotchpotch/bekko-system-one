import json
from types import SimpleNamespace

import pytest
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from test_release import case
from torch import nn

from bekko_system_one import hub_release, training
from bekko_system_one.modules import DecisionHeads
from bekko_system_one.release import load_release


@pytest.fixture
def hub(monkeypatch):
    calls, pins = [], []
    inventory = {
        "a": ["train", "validation", "validation_extra", "test", "calibration", "ood"],
        "test_only": ["test"],
    }

    class Api:
        def __init__(self, token=None):
            assert token is None or isinstance(token, bool)

        def dataset_info(self, repo, revision=None):
            pins.append((repo, revision))
            return SimpleNamespace(sha="a" * 40)

    def load(repo, name, *, split, revision, **kwargs):
        assert revision == "a" * 40
        calls.append((name, split))
        offset = 100 if split == "validation_extra" else 0
        return Dataset.from_list([case(offset + i) for i in range(8)])

    monkeypatch.setattr(hub_release, "HfApi", Api)
    monkeypatch.setattr(hub_release, "get_dataset_config_names", lambda *a, **k: list(inventory))
    monkeypatch.setattr(
        hub_release, "get_dataset_split_names", lambda repo, name, **k: inventory[name]
    )
    monkeypatch.setattr(hub_release, "load_dataset", load)
    return calls, pins


def test_hub_split_selection_sampling_and_exclusion(hub):
    calls, _ = hub
    root = dict(repo_id="owner/typed-data", token=True)
    sources, audit = load_release(root, "validation", sample_cases=5)
    assert list(sources) == ["a"]
    assert len(sources["a"]) == 10
    assert calls == [("a", "validation"), ("a", "validation_extra")]
    assert audit["source"]["revision"] == "a" * 40
    assert "token" not in audit["source"]
    calls.clear()
    test, _ = load_release(root, "test", exclude_datasets=["a"])
    assert list(test) == ["test_only"]
    assert calls == [("test_only", "test")]
    calls.clear()
    missing, _ = load_release({**root, "configs": ["test_only"]}, "validation")
    assert not missing and not calls


def test_unknown_configs_and_invalid_auth_are_rejected(hub):
    with pytest.raises(ValueError, match="Unknown Hub configurations"):
        load_release(dict(repo_id="owner/data", configs=["missing"]), "train")
    for extra in [
        dict(token="secret"),
        dict(configs=[]),
        dict(configs=["a", "a"]),
        dict(revision=""),
    ]:
        with pytest.raises(ValueError):
            load_release(dict(repo_id="owner/data", **extra), "train")
    assert not hub[0]


def test_pin_shared_by_distinct_config_selections(hub):
    cache = {}
    for token, configs in ((True, ["a"]), (False, ["test_only"])):
        result = hub_release.pin_hub_reference(
            dict(repo_id="owner/data", configs=configs, token=token), cache
        )
        assert result["revision"] == "a" * 40
    assert hub[1] == [("owner/data", None)]


def test_hub_errors_propagate_without_local_fallback(hub, monkeypatch):
    def fail(*args, **kwargs):
        raise PermissionError("private dataset authentication required")

    monkeypatch.setattr(hub_release, "load_dataset", fail)
    with pytest.raises(PermissionError, match="authentication"):
        load_release(dict(repo_id="owner/data"), "train")


def test_training_with_hub_sources_and_reload(tmp_path, tiny_encoder, hub):
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    )
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint), create_model_card=False)
    root = dict(repo_id="owner/typed-data", configs=["a"], token=True)
    result = training.run(
        dict(
            model=dict(checkpoint=str(checkpoint)),
            data=dict(
                root=root,
                sampling="uniform",
                dataset_samples={"a": 4},
                prefix_layout_weights={"instruction_state": 1, "state_instruction": 1},
            ),
            evaluation=dict(
                test_root=root, validation_samples=2, before_training=True, token_budget=128
            ),
            training=dict(
                output_dir=str(tmp_path / "run"),
                device="cpu",
                batch_size=2,
                learning_rate=1e-3,
                head_learning_rate=1e-3,
                eval_steps=1,
                initial_tokens=128,
                lr_scheduler="cosine",
            ),
        )
    )
    assert result["model"]
    records = json.loads((tmp_path / "run/data_manifest.json").read_text())
    for role in ("train", "validation", "test"):
        assert records[role]["source"]["revision"] == "a" * 40
    assert len((tmp_path / "run/history.jsonl").read_text().splitlines()) == 2
    restored = SentenceTransformer(
        result["model"], device="cpu", local_files_only=True, trust_remote_code=True
    )
    heads = restored[1]
    assert isinstance(heads, DecisionHeads)
    choice = heads.heads["choice"]
    assert isinstance(choice, nn.Linear)
    assert choice.weight.isfinite().all()


def test_latest_backbone_revision_resolved_once(monkeypatch, tmp_path):
    calls = []

    class Api:
        def model_info(self, name, revision=None):
            calls.append((name, revision))
            return SimpleNamespace(sha="b" * 40)

    monkeypatch.setattr(hub_release, "HfApi", Api)
    config = {"model_name_or_path": "owner/backbone"}
    resolved = hub_release.pin_model_reference(config)
    assert resolved["revision"] == "b" * 40
    assert "revision" not in config
    assert calls == [("owner/backbone", None)]
    local = {"model_name_or_path": str(tmp_path)}
    assert hub_release.pin_model_reference(local) == local
    assert len(calls) == 1
