"""Tracking uses numeric reports and closes failed runs without hiding failures."""

import json
import sys
from types import SimpleNamespace

import pytest

from bekko_system_one import training
from bekko_system_one.tracking import RunTracker


@pytest.fixture
def backend(monkeypatch):
    class FakeRun:
        id, url, project, entity = "id", "https://example.test/run", "project", "owner"

        def __init__(self):
            self.summary, self.logs, self.metrics, self.artifacts, self.exits = {}, [], [], [], []

        def define_metric(self, *args, **kwargs):
            self.metrics.append((args, kwargs))

        def log(self, values):
            self.logs.append(values)

        def log_artifact(self, artifact):
            self.artifacts.append(artifact)

        def finish(self, exit_code):
            self.exits.append(exit_code)

    class Artifact:
        def __init__(self, *args, **kwargs):
            self.files = []

        def add_file(self, path, name):
            self.files.append(name)

    run = FakeRun()
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=init,
            Settings=lambda **kw: kw,
            Table=lambda **kw: kw,
            Artifact=Artifact,
        ),
    )
    return run, calls


def test_metrics_revisions_and_reports(tmp_path, backend):
    run, calls = backend
    tracker = RunTracker({"project": "project", "mode": "offline", "log_every": 10})
    config = {
        "model": {"revision": "model-sha"},
        "data": {"root": {"revision": "data-sha", "token": True}},
    }
    tracker.start(tmp_path, config, {"dataset_cap": 300000}, 2)
    assert calls[0]["config"]["training_config"] == config
    assert calls[0]["save_code"] is False
    entry = dict(
        step=2, source="source", groups=32, loss=0.7, learning_rates=[1e-4, 5e-4], step_seconds=2
    )
    tracker.training(entry, 64)
    assert not run.logs
    tracker.training(entry, 64, final=True)
    assert run.logs[0]["train/loss"] == 0.7
    assert run.logs[0]["train/decisions"] == 64
    assert run.logs[0]["train/head_learning_rate"] == 5e-4
    tracker.evaluation(
        "test",
        "state_instruction",
        2,
        {"source": {"score": {"score_mae": 0.2}}},
        {"mean_cross_entropy": 0.3},
    )
    assert run.summary["test/final/state_instruction/mean_cross_entropy"] == 0.3
    assert run.logs[-1]["test/state_instruction/datasets"]["data"] == [
        ["source", "score", "score_mae", 0.2]
    ]
    for name in ["result.json", "resolved_config.json", "data_manifest.json", "raw_rows.json"]:
        (tmp_path / name).write_text("{}")
    tracker.result({"steps": 2, "model": "output/model"}, {})
    assert run.artifacts[0].files == ["resolved_config.json", "result.json"]
    assert json.loads((tmp_path / "wandb_run.json").read_text())["id"] == "id"
    tracker.finish(exit_code=0)
    assert run.exits == [0]


def test_training_failure_closes_run(tmp_path, backend, monkeypatch):
    def fail(config, steps, tracker):
        tracker.start(tmp_path, config, None, 1)
        raise RuntimeError("original failure")

    monkeypatch.setattr(training, "_run", fail)
    with pytest.raises(RuntimeError, match="original failure"):
        training.run({"wandb": {"project": "project"}})
    assert backend[0].exits == [1]


def test_tracking_disabled_by_default(tmp_path, backend):
    tracker = RunTracker()
    tracker.start(tmp_path, {}, None, 1)
    tracker.finish(exit_code=0)
    assert not backend[1]


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "invalid"},
        {"project": "p", "api_key": "secret"},
        {"mode": "online"},
        {"project": "p", "log_every": 0},
    ],
)
def test_invalid_tracking_config(options):
    with pytest.raises(ValueError):
        RunTracker(options)
