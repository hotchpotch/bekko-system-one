import argparse

import pytest

from bekko_system_one.cli import resolve_config, train_percent
from bekko_system_one.sampling import uniform_counts


def test_fraction_applies_after_caps_and_isolates_outputs():
    config = dict(
        data=dict(sampling="uniform", epoch_fraction=0.3, dataset_samples={"large": 300000}),
        training=dict(output_dir="runs/base"),
    )
    smoke = resolve_config(config, smoke=True)
    one = resolve_config(config, percent=1)
    assert smoke["training"]["output_dir"] == "runs/base-smoke"
    assert one["training"]["output_dir"] == "runs/base-1pct"
    assert one["data"]["epoch_fraction"] == 0.01
    assert config["data"]["epoch_fraction"] == 0.3
    _, selected = uniform_counts(
        {"large": 1000000, "small": 999},
        smoke["data"]["dataset_samples"],
        smoke["data"]["epoch_fraction"],
    )
    assert selected == {"large": 300, "small": 0}
    explicit = resolve_config(config, percent=1, output_dir="runs/retry", max_steps=2)
    assert explicit["training"] == dict(output_dir="runs/retry", max_steps=2)


@pytest.mark.parametrize("value", ["0", "-1", "101", "nan", "inf", "bad"])
def test_invalid_percent(value):
    with pytest.raises(argparse.ArgumentTypeError):
        train_percent(value)


def test_source_passes_rejects_percentage_override():
    with pytest.raises(ValueError, match="source_passes|uniform or weighted"):
        resolve_config(dict(data=dict(sampling="source_passes")), smoke=True)


def test_main_passes_effective_config(tmp_path, monkeypatch):
    from bekko_system_one import training

    path = tmp_path / "config.yaml"
    path.write_text("data:\n  sampling: uniform\ntraining:\n  output_dir: runs/base\n")
    monkeypatch.setattr(
        "sys.argv",
        ["bekko-system-one", "--config", str(path), "--train-percent", "1", "--max-steps", "3"],
    )
    calls = []
    monkeypatch.setattr(training, "run", lambda config: calls.append(config))
    training.main()
    assert calls[0]["data"]["epoch_fraction"] == 0.01
    assert calls[0]["training"] == dict(output_dir="runs/base-1pct", max_steps=3)
