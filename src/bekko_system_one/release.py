"""Read manifest-based typed-decision releases without materializing rendered text."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
from datasets import Dataset, concatenate_datasets, load_from_disk

from .data import Group
from .stratification import select_balanced_cases


def validate_prefix_layout(prefix_layout):
    if prefix_layout not in ("instruction_state", "state_instruction"):
        raise ValueError("prefix_layout must be instruction_state or state_instruction")


def render_group(row, position, *, prefix_layout="instruction_state"):
    """Render only state, aligned instructions and options; preserve soft targets."""
    validate_prefix_layout(prefix_layout)
    decision = row["decisions"][position]
    prompts = row["decision_prompts"]
    if prompts is None or len(prompts) != len(row["decisions"]):
        raise ValueError("decision/prompt alignment mismatch")
    prompt = prompts[position]
    if prompt is None or prompt.get("decision_id") != decision["decision_id"]:
        raise ValueError("decision/prompt alignment mismatch")
    instruction = prompt.get("instruction")
    system = prompt.get("system_prompt") or ""
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(system, str):
        raise ValueError("Expected nonempty instruction and string system_prompt")
    prefix = f"{system}\n\n" if system.strip() else ""
    if prefix_layout == "state_instruction":
        query = f"{prefix}State: {row['state_json']}\nInstruction: {instruction}"
    else:
        query = f"{prefix}Instruction: {instruction}\nState: {row['state_json']}"
    options = decision["options"]
    ids = [o["id"] for o in options]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate option IDs")
    if prompt.get("input_format") == "reranking":
        if decision["type"] != "score":
            raise ValueError("Document ranking requires the score head")
        candidates = [f"Document: {o['description']}" for o in options]
    else:
        candidates = [f"Candidate: {o['id']}: {o['description']}" for o in options]
    target = decision.get("target")
    if target is None:
        raise ValueError("Release training/evaluation requires labels")
    target_ids = target["option_ids"]
    if len(set(target_ids)) != len(target_ids) or set(target_ids) != set(ids):
        raise ValueError("Target IDs must match option IDs")
    mapping = dict(zip(target_ids, target["probabilities"], strict=True))
    return Group(query, candidates, decision["type"], [mapping[i] for i in ids])


class DecisionSource:
    """Index individual decisions with Arrow offsets, retaining complete case samples."""

    def __init__(self, dataset, *, prefix_layout="instruction_state"):
        validate_prefix_layout(prefix_layout)
        self.prefix_layout = prefix_layout
        if dataset._indices is not None:
            dataset = dataset.flatten_indices(keep_in_memory=True)
        self.dataset = dataset
        if not {"decisions", "decision_prompts", "state_json"} <= set(dataset.column_names):
            raise ValueError("Release requires decisions, decision_prompts and state_json")
        lengths = []
        for chunk in dataset.data.column("decisions").chunks:
            if chunk.null_count or chunk.values.field("target").null_count:
                raise ValueError("Unlabeled decisions cannot enter training/evaluation")
            lengths.append(np.diff(chunk.offsets.to_numpy()))
        self.ends = np.cumsum(np.concatenate(lengths) if lengths else [], dtype=np.int64)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def groups(self, indices):
        cases = np.searchsorted(self.ends, indices, side="right")
        # Decode a multi-decision case only once within this batch.
        rows = {int(i): self.dataset[int(i)] for i in np.unique(cases)}
        return [
            render_group(
                rows[int(case)],
                int(index - (self.ends[case - 1] if case else 0)),
                prefix_layout=self.prefix_layout,
            )
            for index, case in zip(indices, cases, strict=True)
        ]


def load_release(
    root,
    role,
    *,
    sample_cases=None,
    seed=42,
    exclude_datasets=None,
    sampling="uniform",
    prefix_layout="instruction_state",
):
    """Select only explicit manifest entries; validation never falls back to test.

    Sampling is without replacement, capped across all matching splits
    of a dataset. All decisions in a selected case remain together.
    """
    validate_prefix_layout(prefix_layout)
    if role not in {"train", "validation", "test"}:
        raise ValueError("role must be train, validation or test")
    if sampling not in {"uniform", "balanced"}:
        raise ValueError("sampling must be uniform or balanced")
    if sample_cases is not None and (
        not isinstance(sample_cases, int) or isinstance(sample_cases, bool) or sample_cases < 1
    ):
        raise ValueError("sample_cases must be a positive integer or null")
    root = Path(root)
    manifest_path = root / "training-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if exclude_datasets is None:
        exclude_datasets = []
    if not isinstance(exclude_datasets, list) or any(
        not isinstance(name, str) or not name.strip() for name in exclude_datasets
    ):
        raise ValueError("exclude_datasets must be a list of nonempty dataset names")
    if len(set(exclude_datasets)) != len(exclude_datasets):
        raise ValueError("Duplicate exclude_datasets name")
    known = {
        entry["dataset"]
        for section in ("train", "evaluation")
        for entry in manifest.get(section, [])
    }
    unknown = set(exclude_datasets) - known
    if unknown:
        raise ValueError(f"Unknown exclude_datasets: {sorted(unknown)}")
    selected = {}
    seen = set()
    for entry in manifest["train" if role == "train" else "evaluation"]:
        name, split = entry["dataset"], entry["split"]
        for part in [name, split]:
            if (
                not isinstance(part, str)
                or not part
                or Path(part).name != part
                or part in {".", ".."}
            ):
                raise ValueError("Invalid manifest dataset/split")
        if (name, split) in seen:
            raise ValueError("Duplicate manifest entry")
        seen.add((name, split))
        if "unlabeled" in split or not (split == role or split.startswith(role + "_")):
            continue
        selected.setdefault(name, []).append(split)
    excluded = {name: selected.pop(name) for name in exclude_datasets if name in selected}
    sources, audit = {}, {}
    for name, splits in sorted(selected.items()):
        datasets, records = [], {}
        for split in sorted(splits):
            dataset = load_from_disk(str(root / name / split))
            if not isinstance(dataset, Dataset):
                raise ValueError("Manifest must reference an explicit Dataset split")
            datasets.append(dataset)
            records[split] = dict(cases=len(dataset), fingerprint=dataset._fingerprint)
        dataset = concatenate_datasets(datasets)
        count = len(dataset)
        indices = None
        sampling_audit = None
        if sample_cases is not None:
            if sampling == "balanced":
                indices, sampling_audit = select_balanced_cases(
                    dataset, cap=sample_cases, seed=seed, name=f"{name}:{role}"
                )
            else:
                indices = sorted(
                    random.Random(f"{seed}:{name}:{role}").sample(
                        range(count), min(sample_cases, count)
                    )
                )
            dataset = dataset.select(indices)
        source = DecisionSource(dataset, prefix_layout=prefix_layout)
        if not len(source):
            raise ValueError(f"Empty labeled source: {name}")
        sources[name] = source
        audit[name] = dict(
            splits=records,
            cases=len(dataset),
            decisions=len(source),
            indices=indices,
            case_ids=list(dataset["case_id"]) if indices is not None else None,
            sampling=sampling_audit,
        )
    return sources, dict(
        role=role,
        prefix_layout=prefix_layout,
        sample_cases=sample_cases,
        sampling=sampling,
        seed=seed,
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        exclude_datasets=sorted(exclude_datasets),
        excluded_splits=excluded,
        datasets=audit,
    )


def source_groups(source, indices):
    if isinstance(source, DecisionSource):
        return source.groups(indices)
    return [Group.from_dict(row) for row in source.select(indices)]


def macro_metrics(results):
    """Equal dataset weight within each task, then equal task weight for CE."""
    tasks = sorted({task for values in results.values() for task in values})
    macro = {}
    for task in tasks:
        values = [v[task] for v in results.values() if task in v]
        macro[task] = dict(
            datasets=len(values),
            groups=sum(v["groups"] for v in values),
            **{k: sum(v[k] for v in values) / len(values) for k in values[0] if k != "groups"},
        )
    return dict(
        tasks=macro,
        mean_cross_entropy=(
            sum(v["cross_entropy"] for v in macro.values()) / len(macro) if macro else None
        ),
    )
