from collections import defaultdict

import torch

from bekko_system_one.training import scheduler_for, source_batches


def test_source_passes_finish_permutations_before_recycling():
    counts = {"a": 7, "b": 4}
    specs = {n: {"passes": 3} for n in counts}
    streams = defaultdict(list)
    for name, rows in source_batches(counts, specs, mode="source_passes", batch_size=3, seed=42):
        streams[name].extend(rows)
    for name, count in counts.items():
        for start in range(0, 3 * count, count):
            assert sorted(streams[name][start : start + count]) == list(range(count))


def test_weighted_batches_are_repeatable_and_never_repeat_within_batch():
    counts = {"a": 13, "b": 7, "small": 2}
    kwargs = dict(mode="weighted", batch_size=5, seed=42, epoch_fraction=10.0)
    first = list(source_batches(counts, {}, **kwargs))
    assert first == list(source_batches(counts, {}, **kwargs))
    assert len(first) == 44
    assert all(name != "small" and len(rows) == len(set(rows)) == 5 for name, rows in first)


def test_cosine_scheduler_warmup_and_group_lr_ratio():
    params = [torch.nn.Parameter(torch.ones(1)) for _ in range(2)]
    optimizer = torch.optim.AdamW(
        [dict(params=[params[0]], lr=2e-4), dict(params=[params[1]], lr=2e-3)]
    )
    scheduler = scheduler_for(optimizer, steps=10, warmup_ratio=0.2, kind="cosine")
    rates = []
    for _ in range(11):
        rates.append([g["lr"] for g in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    assert rates[0] == [0.0, 0.0]
    assert rates[2] == [2e-4, 2e-3]
    assert rates[10] == [0.0, 0.0]
    for low, high in rates:
        assert abs(high - low * 10) < 1e-12


def test_source_passes_defaults_for_manifest_sources():
    batches = list(source_batches({"a": 5}, {}, mode="source_passes", batch_size=3, seed=42))
    assert sorted(i for _, rows in batches for i in rows) == list(range(5))


def test_uniform_quotas_nested_subsets_and_small_tails():
    from bekko_system_one.sampling import uniform_counts

    counts = {"large": 1000, "absolute": 99, "small": 9}
    caps = {"large": "20%", "absolute": 30}
    assert uniform_counts(counts, caps, 0.1)[1] == {"large": 20, "absolute": 3, "small": 0}
    seen = {}
    for fraction in [0.1, 0.5, 1.0]:
        kwargs = dict(
            mode="uniform", batch_size=8, seed=42, epoch_fraction=fraction, dataset_samples=caps
        )
        batches = list(source_batches(counts, {}, **kwargs))
        assert batches == list(source_batches(counts, {}, **kwargs))
        rows = defaultdict(list)
        for name, indices in batches:
            assert 0 < len(indices) <= 8
            rows[name].extend(indices)
        selected = uniform_counts(counts, caps, fraction)[1]
        for name, expected in selected.items():
            assert len(rows[name]) == len(set(rows[name])) == expected
            assert set(seen.get(name, [])) <= set(rows[name])
        seen = rows
    assert len(seen["small"]) == 9


def test_uniform_rejects_invalid_limits():
    import pytest

    from bekko_system_one.sampling import uniform_counts

    for caps in [
        {"typo": 1},
        {"a": -1},
        {"a": True},
        {"a": 0.2},
        {"a": "NaN%"},
        {"a": "101%"},
        {"a": "bad%"},
        [],
    ]:
        with pytest.raises(ValueError):
            uniform_counts({"a": 10}, caps, 0.1)
    for fraction in [0, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            uniform_counts({"a": 10}, {}, fraction)
    assert uniform_counts({"a": 10}, {"a": 100}, 1)[1] == {"a": 10}
    assert uniform_counts({"a": 10}, {"a": 0}, 1)[1] == {"a": 0}


def test_weighted_caps_bound_pool_and_control_weights_and_budget():
    counts = {"large": 10000, "small": 16, "tiny": 3, "disabled": 100}
    kwargs = dict(
        mode="weighted",
        batch_size=4,
        seed=42,
        epoch_fraction=100,
        alpha=0.5,
        dataset_samples={"large": 16, "disabled": 0},
    )
    batches = list(source_batches(counts, {}, **kwargs))
    assert batches == list(source_batches(counts, {}, **kwargs))
    assert len(batches) == (16 + 16 + 3) * 100 // 4
    seen = defaultdict(set)
    selected = defaultdict(int)
    for name, rows in batches:
        assert name in {"large", "small"}
        assert len(set(rows)) == 4
        assert all(0 <= i < counts[name] for i in rows)
        seen[name].update(rows)
        selected[name] += 1
    assert len(seen["large"]) == len(seen["small"]) == 16
    assert max(seen["large"]) > 16  # randomized subset, not the first cap rows
    assert 0.4 < selected["large"] / len(batches) < 0.6
    short_kwargs = dict(
        kwargs, epoch_fraction=0.5, dataset_samples={"large": "0.16%", "disabled": 0}
    )
    short = list(source_batches(counts, {}, **short_kwargs))
    assert len(short) == int(35 * 0.5) // 4
    assert short == batches[: len(short)]


def test_global_cap_applied_before_fraction_and_weighted_selection():
    from bekko_system_one.sampling import uniform_counts

    counts = {"large": 1000, "medium": 200, "small": 10}
    base, selected = uniform_counts(counts, {"medium": 40}, 0.1, dataset_cap=100)
    assert base == {"large": 100, "medium": 40, "small": 10}
    assert selected == {"large": 10, "medium": 4, "small": 1}
    batches = list(
        source_batches(
            counts, {}, mode="uniform", batch_size=8, seed=42, epoch_fraction=0.1, dataset_cap=100
        )
    )
    assert sum(len(rows) for _, rows in batches) == 21
    weighted = list(
        source_batches(
            counts, {}, mode="weighted", batch_size=10, seed=42, epoch_fraction=10, dataset_cap=100
        )
    )
    assert len(weighted) == 210
    seen = defaultdict(set)
    for name, rows in weighted:
        seen[name].update(rows)
    assert len(seen["large"]) <= 100
    assert max(seen["large"]) >= 100


def test_global_cap_rejects_invalid_values():
    import pytest

    from bekko_system_one.sampling import uniform_counts

    for cap in [True, 0, -1, 0.5, "300000"]:
        with pytest.raises(ValueError, match="dataset_cap"):
            uniform_counts({"a": 100}, {}, 1.0, dataset_cap=cap)
