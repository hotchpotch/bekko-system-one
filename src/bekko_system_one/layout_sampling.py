"""Reproducible per-decision input-order augmentation."""

import math
import random
from collections import Counter

LAYOUTS = ("instruction_state", "state_instruction")


def validate_layout_weights(weights):
    if not isinstance(weights, dict) or not weights or set(weights) - set(LAYOUTS):
        raise ValueError("prefix_layout_weights must map supported layouts to nonnegative weights")
    if any(
        isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
        for w in weights.values()
    ):
        raise ValueError("prefix_layout_weights must be finite nonnegative numbers")
    scale = max(weights.values())
    if scale == 0:
        raise ValueError("prefix_layout_weights must contain a positive weight")
    return {name: weights.get(name, 0) / scale for name in LAYOUTS}


class LayoutSampler:
    def __init__(self, weights, seed):
        self.weights = validate_layout_weights(weights)
        self.rng = random.Random(f"{seed}:prefix-layout-v1")
        self.counts = Counter({name: 0 for name in LAYOUTS})

    def sample(self, count):
        layouts = self.rng.choices(LAYOUTS, weights=list(self.weights.values()), k=count)
        self.counts.update(layouts)
        return layouts
