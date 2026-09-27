"""Resolve reusable training configurations with explicit CLI budget overrides."""

import argparse
import copy
import math
from pathlib import Path


def train_percent(value):
    try:
        percent = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("train percent must be a number in (0, 100]") from exc
    if not math.isfinite(percent) or not 0 < percent <= 100:
        raise argparse.ArgumentTypeError("train percent must be in (0, 100]")
    return percent


def resolve_config(config, *, smoke=False, percent=None, output_dir=None, max_steps=None):
    """Replace epoch_fraction; keep caps and evaluation unchanged; isolate trial outputs."""
    if smoke and percent is not None:
        raise ValueError("Choose smoke or train percent")
    effective = copy.deepcopy(config)
    if smoke or percent is not None:
        percent = 0.1 if smoke else train_percent(str(percent))
        if effective["data"].get("sampling", "weighted") == "source_passes":
            raise ValueError("Percentage overrides require uniform or weighted sampling")
        effective["data"]["epoch_fraction"] = percent / 100
        if output_dir is None:
            base = Path(effective["training"]["output_dir"])
            suffix = "smoke" if smoke else f"{percent:g}pct"
            output_dir = str(base.with_name(f"{base.name}-{suffix}"))
    if output_dir is not None:
        effective["training"]["output_dir"] = str(output_dir)
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        effective["training"]["max_steps"] = max_steps
    return effective
