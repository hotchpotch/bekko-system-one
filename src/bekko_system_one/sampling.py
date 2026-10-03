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


def task_grouped_batches(plan, source_tasks, stream, *, seed):
    """Shuffle decisions across sources within each round, batching only by task.

    Preserve the source sampler's selected references and round allocation.
    Each reference retains its source, row and pass for reproducible usage audits.
    Only one round of references is buffered; text is rendered by the consumer.
    """
    import itertools
    import random

    if set(source_tasks) != set(plan["source_counts"]):
        raise ValueError("Expected task indices for every source")
    for name, tasks in source_tasks.items():
        if len(tasks) != plan["source_counts"][name]:
            raise ValueError(f"Task index length mismatch: {name}")
        if set(tasks) - {"choice", "noul", "score"}:
            raise ValueError(f"Unsupported decision task in source: {name}")
    for round_index, batches in itertools.groupby(stream, key=lambda b: b[2]["round"]):
        buckets = {}
        for name, rows, metadata in batches:
            for row in rows:
                task = source_tasks[name][row]
                buckets.setdefault(task, []).append(
                    (name, row, metadata.get("source_pass", metadata.get("draw_batch")))
                )
        schedule = []
        for task, refs in sorted(buckets.items()):
            random.Random(f"{seed}:{task}:{round_index}:task-round-v1").shuffle(refs)
            size = plan["batch_size"]
            count = math.ceil(len(refs) / size)
            phase = random.Random(f"{seed}:{task}:{round_index}:task-schedule-v1").random()
            for index in range(count):
                schedule.append(((index + phase) / count, task, index))
        for _, task, index in sorted(schedule):
            refs = buckets[task][index * size : (index + 1) * size]
            yield task, refs, {"round": round_index, "batch_task": task}


def task_round_batches(plan, source_tasks, *, seed):
    """Mix exact source-pass decisions into homogeneous task batches per round."""
    yield from task_grouped_batches(plan, source_tasks, round_batches(plan, seed=seed), seed=seed)


def virtual_weighted_plan(
    counts,
    dataset_samples,
    epoch_fraction,
    *,
    batch_size,
    alpha=0.5,
    dataset_cap=None,
    dataset_multipliers=None,
    rounds=10,
):
    """Budget replacement sampling by virtual source volume, preserving capped pools."""
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha < 0
    ):
        raise ValueError("sampling_alpha must be finite and nonnegative")
    for name, value in [("batch_size", batch_size), ("rounds", rounds)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    multipliers = {} if dataset_multipliers is None else dataset_multipliers
    if not isinstance(multipliers, dict):
        raise ValueError("dataset_multipliers must be a mapping")
    if set(multipliers) - set(counts):
        raise ValueError("Unknown or excluded dataset_multipliers names")
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in multipliers.values()):
        raise ValueError("dataset_multipliers values must be positive integers")
    bases, _ = uniform_counts(counts, dataset_samples, epoch_fraction, dataset_cap=dataset_cap)
    virtual = {n: c * multipliers.get(n, 1) for n, c in bases.items()}
    weights = {n: c**alpha for n, c in virtual.items() if c}
    if not all(math.isfinite(w) for w in weights.values()):
        raise ValueError("sampling weights must be finite")
    budget = int(sum(virtual.values()) * Decimal(str(epoch_fraction)))
    return dict(
        mode="virtual_weighted",
        unit="decision",
        source_counts=dict(counts),
        base_counts=bases,
        virtual_counts=virtual,
        dataset_multipliers=dict(multipliers),
        sampling_weights=weights,
        sampling_alpha=alpha,
        epoch_fraction=epoch_fraction,
        decision_budget=budget,
        sampled_decisions=budget // batch_size * batch_size,
        steps=budget // batch_size,
        source_grouped_steps=budget // batch_size,
        batch_size=batch_size,
        rounds=rounds,
        dataset_cap=dataset_cap,
        replacement=True,
    )


def virtual_weighted_batches(plan, *, seed):
    """Yield source batches from identical deterministic replacement draws for either grouping."""
    import random

    names = sorted(plan["sampling_weights"])
    if not names:
        return
    weights = [plan["sampling_weights"][n] for n in names]
    source_rng = random.Random(f"{seed}:virtual-source-v1")
    pools, row_rngs = {}, {}
    steps = plan["source_grouped_steps"]
    for index in range(steps):
        name = source_rng.choices(names, weights=weights, k=1)[0]
        if name not in pools:
            pools[name] = random.Random(f"{seed}:{name}:uniform-v1").sample(
                range(plan["source_counts"][name]), plan["base_counts"][name]
            )
            row_rngs[name] = random.Random(f"{seed}:{name}:virtual-row-v1")
        rows = row_rngs[name].choices(pools[name], k=plan["batch_size"])
        yield name, rows, {"round": index * plan["rounds"] // steps + 1, "draw_batch": index}


def task_virtual_weighted_batches(plan, source_tasks, *, seed):
    """Rebatch the same weighted references by task, with cross-source round shuffling."""
    yield from task_grouped_batches(
        plan, source_tasks, virtual_weighted_batches(plan, seed=seed), seed=seed
    )
