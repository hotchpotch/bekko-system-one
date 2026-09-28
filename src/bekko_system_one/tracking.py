"""Optional W&B metrics and reproducibility reports, without training examples."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def scalar_metrics(values, prefix=""):
    """Flatten numeric metrics while leaving text, tables and missing values out."""
    result = {}
    for key, value in values.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            result.update(scalar_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
    return result


class RunTracker:
    """One explicit run; absent configuration leaves W&B completely unused."""

    def __init__(self, options=None):
        self.options = {} if options is None else dict(options)
        allowed = {"mode", "project", "entity", "name", "group", "tags", "job_type", "log_every"}
        if set(self.options) - allowed:
            raise ValueError(f"Unknown wandb options: {sorted(set(self.options) - allowed)}")
        self.mode = self.options.get("mode", "disabled" if options is None else "online")
        if self.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline or disabled")
        self.log_every = self.options.get("log_every", 10)
        if (
            isinstance(self.log_every, bool)
            or not isinstance(self.log_every, int)
            or self.log_every < 1
        ):
            raise ValueError("wandb.log_every must be a positive integer")
        if self.mode != "disabled" and not self.options.get("project"):
            raise ValueError("wandb.project is required when tracking is enabled")
        for key in ("project", "entity", "name", "group", "job_type"):
            if key in self.options and (
                not isinstance(self.options[key], str) or not self.options[key].strip()
            ):
                raise ValueError(f"wandb.{key} must be a nonempty string")
        if "tags" in self.options and (
            not isinstance(self.options["tags"], list)
            or any(not isinstance(t, str) or not t.strip() for t in self.options["tags"])
        ):
            raise ValueError("wandb.tags must be a list of nonempty strings")
        self.run = None
        self.directory = None
        self.wandb = None

    def start(self, directory, config, sample_plan, steps):
        if self.mode == "disabled":
            return
        import wandb

        self.wandb = wandb
        self.directory = Path(directory)
        kwargs = {k: v for k, v in self.options.items() if k not in {"mode", "log_every"}}
        kwargs.setdefault("name", self.directory.name)
        # Authentication belongs in the environment or the user's saved login.
        # The config accepts no API-key field and Hub token values are booleans.
        self.run = wandb.init(
            **kwargs,
            mode=self.mode,
            dir=str(self.directory),
            config={
                "training_config": config,
                "sampling_plan": sample_plan,
                "planned_steps": steps,
            },
            save_code=False,
            settings=wandb.Settings(disable_git=True),
        )
        self.run.define_metric("train/step")
        self.run.define_metric("*", step_metric="train/step")
        (self.directory / "wandb_run.json").write_text(
            json.dumps(
                {
                    "id": self.run.id,
                    "url": self.run.url,
                    "project": self.run.project,
                    "entity": self.run.entity,
                    "mode": self.mode,
                },
                indent=2,
            )
        )

    def parameters(self, model):
        if self.run is not None:
            self.run.summary.update(
                {
                    "model/parameters": sum(p.numel() for p in model.parameters()),
                    "model/trainable_parameters": sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                }
            )

    def training(self, entry, decisions, *, final=False):
        if self.run is None or (
            entry["step"] % self.log_every and entry["step"] != 1 and not final
        ):
            return
        metrics = scalar_metrics({k: v for k, v in entry.items() if k != "usage_by_task"}, "train")
        metrics.update(
            {
                "train/decisions": decisions,
                "train/source": entry["source"],
                "train/encoder_learning_rate": entry["learning_rates"][0],
                "train/head_learning_rate": entry["learning_rates"][1],
                "train/decisions_per_second": entry["groups"] / max(entry["step_seconds"], 1e-9),
            }
        )
        metrics.update(scalar_metrics(entry.get("usage_by_task", {}), "train/batch_usage"))
        self.run.log(metrics)

    def evaluation(self, label, layout, step, results, summary):
        if self.run is None:
            return
        role = label.split("-", 1)[0]
        prefix = f"{role}/{layout}"
        metrics = scalar_metrics(summary, prefix)
        metrics["train/step"] = step
        self.run.log(metrics)
        assert self.wandb is not None
        rows: list[Iterable[Any]] = [
            [dataset, task, metric, value]
            for dataset, tasks in sorted(results.items())
            for task, values in sorted(tasks.items())
            for metric, value in values.items()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        ]
        table = self.wandb.Table(columns=["dataset", "task", "metric", "value"], data=rows)
        self.run.log({"train/step": step, f"{prefix}/datasets": table})
        phase = "initial" if label.endswith("-initial") else "final" if label == role else None
        if phase:
            self.run.summary.update(scalar_metrics(summary, f"{role}/{phase}/{layout}"))

    def result(self, result, usage):
        if self.run is None:
            return
        self.run.summary.update(scalar_metrics(result, "result"))
        self.run.summary["model/local_checkpoint"] = result["model"]
        self.run.summary["usage"] = usage
        # Small numeric reports/configs only; no raw rows, predictions or model weights.
        assert self.wandb is not None and self.directory is not None
        artifact = self.wandb.Artifact(f"training-reports-{self.run.id}", type="training-report")
        for path in sorted(self.directory.glob("*.json")):
            if path.name in {
                "resolved_config.json",
                "sampling_plan.json",
                "result.json",
                "training_usage.json",
            } or path.name.endswith("-summary.json"):
                artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact)

    def finish(self, *, exit_code):
        if self.run is not None:
            try:
                self.run.finish(exit_code=exit_code)
            except Exception:
                if not exit_code:
                    raise
                logging.getLogger(__name__).exception("W&B cleanup failed after training failure")
            finally:
                self.run = None
