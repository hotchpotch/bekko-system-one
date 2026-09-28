"""Deterministic, case-preserving evaluation sampling with a strict case cap."""

import hashlib
import json
import random
from collections import Counter, defaultdict

import numpy as np

from .dataset_schema import VERSION as DATASET_SCHEMA_VERSION
from .dataset_training import training_view


def _signature(decision):
    return hashlib.sha256(
        json.dumps(sorted((o["id"], o["description"]) for o in decision["options"])).encode()
    ).hexdigest()


def _decisions(dataset):
    # Keep decoding bounded; structured conversion also needs its reversible legacy metadata.
    if "input" not in dataset.column_names:
        for batch in dataset.select_columns(["decisions"]).iter(batch_size=256):
            yield from batch["decisions"]
        return
    for batch in dataset.iter(batch_size=256):
        for row_index in range(len(batch["input"])):
            if (
                "schema_version" in batch
                and batch["schema_version"][row_index] != DATASET_SCHEMA_VERSION
            ):
                raise ValueError(f"Expected {DATASET_SCHEMA_VERSION} rows in structured release")
            row = {name: values[row_index] for name, values in batch.items()}
            yield training_view(row)["decisions"]


def select_balanced_cases(dataset, *, cap, seed, name):
    """Balance Noul / repeated Choice classes; keep soft targets and whole cases.

    The cap is never expanded for class coverage. Multi-axis coverage is greedy,
    so unmet quotas and missing classes are reported rather than guaranteed.
    """
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("cap must be a positive integer")
    rng = random.Random(f"{seed}:{name}:balanced-v1")
    signatures = Counter(
        _signature(d) for ds in _decisions(dataset) for d in ds if d["type"] == "choice"
    )
    features, bands = [], []
    axis_labels = defaultdict(set)
    for decisions in _decisions(dataset):
        fs, confidence = set(), {}
        for d in decisions:
            if d["type"] == "noul":
                axis = "noul:" + d["decision_id"]
            elif d["type"] == "choice" and signatures[_signature(d)] >= 2:
                axis = "choice:" + d["decision_id"] + ":" + _signature(d)
            else:
                continue
            target = d.get("target")
            if not target:
                raise ValueError("Balanced sampling requires labeled decisions")
            peak = max(target["probabilities"])
            winners = [
                label
                for label, p in zip(target["option_ids"], target["probabilities"], strict=True)
                if abs(p - peak) < 1e-8
            ]
            label = winners[0] if len(winners) == 1 else "__tie__"
            axis_labels[axis].update(o["id"] for o in d["options"])
            axis_labels[axis].add(label)
            key = (axis, label)
            fs.add(key)
            confidence[key] = (
                "hard" if peak >= 1 - 1e-6 else "soft_high" if peak >= 0.8 else "soft_low"
            )
        features.append(fs)
        bands.append(confidence)
    available = Counter(f for fs in features for f in fs)
    target_count = min(cap, len(features))
    single = bool(features) and len(axis_labels) == 1 and all(len(fs) == 1 for fs in features)
    quotas = Counter()
    if single:
        mode = "single_axis_uniform_classes"
        buckets = defaultdict(list)
        for i, fs in enumerate(features):
            buckets[next(iter(fs))].append(i)
        keys = sorted(buckets)
        rng.shuffle(keys)
        # Allocate equal quotas, redistributing shortages; the odd extra is seeded.
        remaining = target_count
        while remaining:
            for key in keys:
                if quotas[key] < len(buckets[key]) and remaining:
                    quotas[key] += 1
                    remaining -= 1
        chosen = []
        for key in keys:
            quota = quotas[key]
            strata = defaultdict(list)
            for i in buckets[key]:
                strata[bands[i][key]].append(i)
            weights = {b: quota * len(ids) / len(buckets[key]) for b, ids in strata.items()}
            counts = {b: int(v) for b, v in weights.items()}
            order = sorted(weights)
            rng.shuffle(order)
            order.sort(key=lambda b: weights[b] - counts[b], reverse=True)
            for b in order[: quota - sum(counts.values())]:
                counts[b] += 1
            for b in sorted(strata):
                chosen.extend(rng.sample(strata[b], counts[b]))
    elif available:
        mode = "multilabel_case_preserving"
        keys = sorted(available)
        keyindex = {key: i for i, key in enumerate(keys)}
        rr, cc = [], []
        for i, fs in enumerate(features):
            for f in sorted(fs):
                rr.append(i)
                cc.append(keyindex[f])
        row_indices, columns = np.asarray(rr), np.asarray(cc)
        for axis, labels in axis_labels.items():
            eligible = sum(any(f[0] == axis for f in fs) for fs in features)
            desired = max(1, round(target_count * eligible / len(features) / len(labels)))
            for label in labels:
                quotas[(axis, label)] = min(available[(axis, label)], desired)
        desired = np.array([quotas[k] for k in keys], dtype=float)
        supply = np.array([available[k] for k in keys], dtype=float)
        selected = np.zeros(len(keys))
        used = np.zeros(len(features), dtype=bool)
        jitter = np.array([rng.random() * 1e-9 for _ in features])
        chosen = []
        for _ in range(target_count):
            weights = 1000 * (selected == 0) + np.maximum(0, desired - selected) / np.maximum(
                1, desired
            ) / np.sqrt(supply)
            scores = (
                np.bincount(row_indices, weights=weights[columns], minlength=len(features)) + jitter
            )
            scores[used] = -np.inf
            index = int(scores.argmax())
            chosen.append(index)
            used[index] = True
            for key in features[index]:
                selected[keyindex[key]] += 1
    else:
        mode = "uniform"
        chosen = rng.sample(range(len(features)), target_count)
    observed = Counter(f for i in chosen for f in features[i])
    confidence_available = Counter((key, band) for bs in bands for key, band in bs.items())
    confidence_selected = Counter((key, band) for i in chosen for key, band in bands[i].items())
    labels = []
    for axis, values in sorted(axis_labels.items()):
        for label in sorted(values):
            key = (axis, label)
            desired = quotas[key]
            labels.append(
                dict(
                    axis=axis,
                    label=label,
                    available=available[key],
                    desired=desired,
                    selected=observed[key],
                    unavailable=available[key] == 0,
                    missing=available[key] > 0 and observed[key] == 0,
                    shortfall=max(0, desired - observed[key]),
                    confidence_bands={
                        b: dict(
                            available=confidence_available[(key, b)],
                            selected=confidence_selected[(key, b)],
                        )
                        for b in ["hard", "soft_high", "soft_low"]
                    },
                )
            )
    return sorted(chosen), dict(
        version="balanced-v1-strict-cap",
        policy=mode,
        seed=seed,
        source_cases=len(features),
        requested_cap=cap,
        selected_cases=len(chosen),
        expanded_for_class_coverage=False,
        labels=labels,
    )
