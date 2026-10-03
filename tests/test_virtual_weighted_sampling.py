"""Virtual source multiplicity, weighted draws and equivalent task rebatching."""

import math
from collections import Counter

import pytest

from bekko_system_one.sampling import (
    task_virtual_weighted_batches,
    virtual_weighted_batches,
    virtual_weighted_plan,
)


def test_budget_weights_caps_and_small_sources():
    plan = virtual_weighted_plan(
        {"base": 1000, "gen": 16, "tiny": 1},
        {},
        0.1,
        batch_size=8,
        dataset_cap=300,
        dataset_multipliers={"gen": 10},
    )
    assert plan["base_counts"] == {"base": 300, "gen": 16, "tiny": 1}
    assert plan["virtual_counts"] == {"base": 300, "gen": 160, "tiny": 1}
    assert plan["decision_budget"] == 46
    assert plan["sampled_decisions"] == 40
    assert plan["sampling_weights"]["gen"] == math.sqrt(160)
    assert "tiny" in plan["sampling_weights"]
    draws = list(virtual_weighted_batches(plan, seed=42))
    assert draws == list(virtual_weighted_batches(plan, seed=42))
    assert len(draws) == 5
    assert all(len(rows) == 8 for _, rows, _ in draws)


def test_task_mixing_preserves_every_source_row_and_draw_identity():
    counts = {"a": 103, "b": 16, "tiny": 1}
    plan = virtual_weighted_plan(
        counts, {}, 1, batch_size=8, rounds=4, dataset_multipliers={"b": 10}
    )
    source = list(virtual_weighted_batches(plan, seed=42))
    tasks = {
        "a": ["choice", "noul", "score"] * 34 + ["choice"],
        "b": ["choice", "score"] * 8,
        "tiny": ["noul"],
    }
    mixed = list(task_virtual_weighted_batches(plan, tasks, seed=42))
    assert mixed == list(task_virtual_weighted_batches(plan, tasks, seed=42))
    assert Counter(
        (name, row, meta["draw_batch"]) for name, rows, meta in source for row in rows
    ) == Counter(ref for _, refs, _ in mixed for ref in refs)
    assert any(len({n for n, _, _ in refs}) > 1 for _, refs, _ in mixed)
    for task, refs, meta in mixed:
        assert {tasks[n][row] for n, row, _ in refs} == {task}
        assert 0 < len(refs) <= 8
    visits = Counter(row for n, rows, _ in source if n == "b" for row in rows)
    assert max(visits.values()) > 1


@pytest.mark.parametrize(
    "options",
    [
        {"dataset_multipliers": {"missing": 10}},
        {"dataset_multipliers": {"a": True}},
        {"dataset_multipliers": {"a": 0}},
        {"dataset_multipliers": []},
        {"alpha": -1},
        {"alpha": float("inf")},
        {"alpha": True},
        {"batch_size": 0},
        {"rounds": 0},
    ],
)
def test_invalid_virtual_settings(options):
    with pytest.raises(ValueError):
        virtual_weighted_plan({"a": 20}, {}, 0.1, **{"batch_size": 8, **options})
