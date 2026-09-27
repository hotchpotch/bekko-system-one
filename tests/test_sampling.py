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
