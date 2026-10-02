"""Per-source, without-replacement training budgets in decision units."""

import math
from decimal import Decimal, InvalidOperation
from typing import Any


def uniform_counts(counts, dataset_samples, epoch_fraction, *, dataset_cap=None):
    """Apply per-source percentage/count caps, then floor the global fraction."""
    if not math.isfinite(epoch_fraction) or not 0 < epoch_fraction <= 1:
        raise ValueError("uniform epoch_fraction must be in (0, 1]")
    if not isinstance(dataset_samples, dict):
        raise ValueError("dataset_samples must be a mapping")
    unknown = set(dataset_samples) - set(counts)
    if unknown:
        raise ValueError(f"Unknown or excluded dataset_samples names: {sorted(unknown)}")
    if dataset_cap is not None and (
        isinstance(dataset_cap, bool) or not isinstance(dataset_cap, int) or dataset_cap < 1
    ):
        raise ValueError("dataset_cap must be a positive integer")
    bases, selected = {}, {}
    for name, count in counts.items():
        value = dataset_samples.get(name, "100%")
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 0:
                raise ValueError("dataset_samples counts must be nonnegative")
            base = min(value, count)
        elif isinstance(value, str) and value.endswith("%"):
            try:
                ratio = Decimal(value[:-1]) / 100
            except InvalidOperation as exc:
                raise ValueError(
                    "dataset_samples requires an integer or percentage string"
                ) from exc
            if not ratio.is_finite() or not 0 <= ratio <= 1:
                raise ValueError("dataset_samples percentages must be in [0%, 100%]")
            base = int(count * ratio)
        else:
            raise ValueError("dataset_samples requires an integer or percentage string")
        if dataset_cap is not None:
            base = min(base, dataset_cap)
        bases[name] = base
        selected[name] = int(base * Decimal(str(epoch_fraction)))
    return bases, selected


def round_plan(
    counts,
    dataset_samples,
    epoch_fraction,
    *,
    batch_size,
    rounds,
    dataset_passes=None,
    dataset_cap=None,
) -> dict[str, Any]:
    """Plan exact source passes, with capped pools distributed across fixed rounds."""
    for name, value in [("batch_size", batch_size), ("rounds", rounds)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    dataset_passes = {} if dataset_passes is None else dataset_passes
    if not isinstance(dataset_passes, dict):
        raise ValueError("dataset_passes must be a mapping")
    unknown = set(dataset_passes) - set(counts)
    if unknown:
        raise ValueError(f"Unknown or excluded dataset_passes names: {sorted(unknown)}")
    for value in dataset_passes.values():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("dataset_passes must contain positive integers")
    bases, selected = uniform_counts(
        counts, dataset_samples, epoch_fraction, dataset_cap=dataset_cap
    )
    passes = {name: dataset_passes.get(name, 1) for name in counts}
    batches = {
        name: math.ceil(count / batch_size) * passes[name] for name, count in selected.items()
    }
    round_counts = [
        {
            name: total * (index + 1) // rounds - total * index // rounds
            for name, total in batches.items()
        }
        for index in range(rounds)
    ]
    return {
        "mode": "round_passes",
        "unit": "decision",
        "batch_size": batch_size,
        "rounds": rounds,
        "epoch_fraction": epoch_fraction,
        "dataset_cap": dataset_cap,
        "source_counts": dict(counts),
        "base_counts": bases,
        "selected_counts": selected,
        "pass_counts": passes,
        "planned_decisions": {name: count * passes[name] for name, count in selected.items()},
        "source_batch_counts": batches,
        "round_batch_counts": round_counts,
        "steps": sum(batches.values()),
    }


def round_batches(plan, *, seed):
    """Yield complete source batches with exact coverage, tails, round and pass identity.

    A fixed cap subset is reused, but each pass has a separate permutation. Within
    each round a source gets one batch in each of its equally spaced time slots.
    Seeded jitter randomizes relative source order without unbounded clustering.
    """
    import random

    def batches(name):
        count = plan["source_counts"][name]
        pool_rng = random.Random(f"{seed}:{name}:uniform-v1")
        pool = pool_rng.sample(range(count), plan["base_counts"][name])[
            : plan["selected_counts"][name]
        ]
        for pass_index in range(1, plan["pass_counts"][name] + 1):
            rows = pool.copy()
            random.Random(f"{seed}:{name}:{pass_index}:round-pass-v1").shuffle(rows)
            for start in range(0, len(rows), plan["batch_size"]):
                yield rows[start : start + plan["batch_size"]], pass_index

    streams = {name: batches(name) for name, count in plan["selected_counts"].items() if count}
    for round_index, quotas in enumerate(plan["round_batch_counts"], 1):
        schedule = []
        for name, quota in sorted(quotas.items()):
            rng = random.Random(f"{seed}:{name}:{round_index}:round-schedule-v1")
            phase = rng.random()
            for index in range(quota):
                schedule.append(((index + phase) / quota, name))
        for _, name in sorted(schedule):
            rows, pass_index = next(streams[name])
            yield name, rows, {"round": round_index, "source_pass": pass_index}
