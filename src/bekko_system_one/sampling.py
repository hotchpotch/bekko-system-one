"""Per-source, without-replacement training budgets in decision units."""

import math
from decimal import Decimal, InvalidOperation


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
