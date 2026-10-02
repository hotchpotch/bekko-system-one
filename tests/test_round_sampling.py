"""Coverage and scheduling contracts for repeated, capped release sources."""

from collections import Counter, defaultdict

import pytest

from bekko_system_one.sampling import round_batches, round_plan


def test_repeated_source_gets_one_complete_pass_per_round():
    plan = round_plan(
        {"base": 103, "supplement": 16},
        {},
        1,
        batch_size=4,
        rounds=10,
        dataset_passes={"supplement": 10},
    )
    batches = list(round_batches(plan, seed=42))
    visits = defaultdict(list)
    for name, rows, metadata in batches:
        assert len(rows) == len(set(rows)) <= 4
        visits[name].extend(rows)
    assert Counter(visits["base"]) == Counter(range(103))
    assert Counter(visits["supplement"]) == Counter({i: 10 for i in range(16)})
    for round_index in range(1, 11):
        selected = [
            (rows, meta)
            for name, rows, meta in batches
            if name == "supplement" and meta["round"] == round_index
        ]
        assert len(selected) == 4
        assert sorted(i for rows, _ in selected for i in rows) == list(range(16))
        assert {meta["source_pass"] for _, meta in selected} == {round_index}
    assert plan["steps"] == len(batches) == 66
    assert plan["planned_decisions"] == {"base": 103, "supplement": 160}


def test_caps_remain_the_same_subset_across_passes_and_keep_tails():
    plan = round_plan(
        {"a": 29, "b": 7, "off": 20},
        {"a": 11, "off": 0},
        1,
        batch_size=4,
        rounds=4,
        dataset_passes={"a": 3},
        dataset_cap=12,
    )
    batches = list(round_batches(plan, seed=42))
    assert batches == list(round_batches(plan, seed=42))
    passes = defaultdict(list)
    for name, rows, metadata in batches:
        passes[name, metadata["source_pass"]].extend(rows)
    assert len(passes) == 4
    for pass_index in [1, 2, 3]:
        assert len(passes["a", pass_index]) == len(set(passes["a", pass_index])) == 11
        assert set(passes["a", pass_index]) == set(passes["a", 1])
    assert passes["a", 1] != passes["a", 2]
    assert set(passes["a", 1]) != set(range(11))
    assert sorted(passes["b", 1]) == list(range(7))
    assert sum(len(rows) for _, rows, _ in batches) == 40
    assert plan["steps"] == 11


def test_source_schedule_is_spread_across_each_round():
    plan = round_plan(
        {"large": 400, "small": 16}, {}, 1, batch_size=4, rounds=2, dataset_passes={"small": 2}
    )
    batches = list(round_batches(plan, seed=42))
    for index in [1, 2]:
        current = [name for name, _, metadata in batches if metadata["round"] == index]
        positions = [i for i, name in enumerate(current) if name == "small"]
        assert len(positions) == 4
        assert all(b - a >= 8 for a, b in zip(positions, positions[1:]))


def test_rounds_more_than_batches_and_passes_more_than_rounds():
    plan = round_plan({"tiny": 3}, {}, 1, batch_size=8, rounds=5, dataset_passes={"tiny": 7})
    batches = list(round_batches(plan, seed=9))
    assert len(batches) == 7
    assert {meta["source_pass"] for _, _, meta in batches} == set(range(1, 8))
    assert all(sorted(rows) == [0, 1, 2] for _, rows, _ in batches)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_passes": {"missing": 2}},
        {"dataset_passes": {"a": 0}},
        {"dataset_passes": {"a": True}},
        {"dataset_passes": {"a": 1.5}},
        {"dataset_passes": []},
        {"rounds": 0},
        {"rounds": True},
        {"rounds": 1.5},
        {"batch_size": 0},
        {"batch_size": True},
    ],
)
def test_invalid_round_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        round_plan({"a": 10}, {}, 1, **{"batch_size": 4, "rounds": 2, **kwargs})


def test_round_fraction_keeps_nested_cap_subsets_and_source_order_independence():
    seen = set()
    for fraction in [0.25, 0.5, 1]:
        plan = round_plan(
            {"a": 100, "b": 7}, {"a": 40}, fraction, batch_size=8, rounds=4, dataset_passes={"a": 4}
        )
        batches = list(round_batches(plan, seed=42))
        current = {i for name, rows, _ in batches if name == "a" for i in rows}
        assert len(current) == int(40 * fraction)
        assert seen <= current
        seen = current
        reverse = round_plan(
            {"b": 7, "a": 100}, {"a": 40}, fraction, batch_size=8, rounds=4, dataset_passes={"a": 4}
        )
        assert batches == list(round_batches(reverse, seed=42))


def test_source_batches_round_mode_keeps_the_pair_interface():
    from bekko_system_one.training import source_batches

    batches = list(
        source_batches(
            {"a": 4},
            {},
            mode="round_passes",
            batch_size=2,
            seed=42,
            rounds=2,
            dataset_passes={"a": 2},
        )
    )
    assert len(batches) == 4
    assert Counter(i for _, rows in batches for i in rows) == Counter({i: 2 for i in range(4)})
    with pytest.raises(ValueError, match="dataset_samples"):
        list(
            source_batches(
                {"a": 4}, {}, mode="round_passes", batch_size=2, seed=42, dataset_samples=[]
            )
        )


def test_task_rounds_mix_sources_preserve_passes_and_round_coverage():
    from bekko_system_one.sampling import task_round_batches

    plan = round_plan(
        {"base": 103, "supplement": 16},
        {},
        1,
        batch_size=8,
        rounds=10,
        dataset_passes={"supplement": 10},
    )
    tasks = {
        "base": ["choice", "noul", "score"] * 34 + ["choice"],
        "supplement": ["choice", "score"] * 8,
    }
    batches = list(task_round_batches(plan, tasks, seed=42))
    assert batches == list(task_round_batches(plan, tasks, seed=42))
    visits = Counter()
    mixed = False
    for task, refs, metadata in batches:
        assert 0 < len(refs) <= 8
        assert {tasks[name][row] for name, row, _ in refs} == {task}
        visits.update((name, row, pass_index) for name, row, pass_index in refs)
        mixed |= len({name for name, _, _ in refs}) > 1
    assert mixed
    assert visits == Counter(
        [("base", row, 1) for row in range(103)]
        + [("supplement", row, p) for p in range(1, 11) for row in range(16)]
    )
    for r in range(1, 11):
        assert sorted(
            row
            for _, refs, meta in batches
            if meta["round"] == r
            for name, row, p in refs
            if name == "supplement"
        ) == list(range(16))


def test_task_rounds_keep_cap_subsets_and_validate_indices():
    from bekko_system_one.sampling import task_round_batches

    plan = round_plan({"a": 29}, {"a": 11}, 1, batch_size=4, rounds=4, dataset_passes={"a": 3})
    batches = list(task_round_batches(plan, {"a": ["noul"] * 29}, seed=3))
    visits = Counter(row for _, refs, _ in batches for _, row, _ in refs)
    assert len(visits) == 11
    assert set(visits.values()) == {3}
    with pytest.raises(ValueError, match="length"):
        list(task_round_batches(plan, {"a": ["noul"]}, seed=3))
    with pytest.raises(ValueError, match="Unsupported"):
        list(task_round_batches(plan, {"a": ["unknown"] * 29}, seed=3))
