"""Single-device training of Sentence Transformers candidate-scoring modules."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import get_cosine_schedule_with_warmup, set_seed

from .batching import TokenBudget, adaptive_backward
from .data import collate_groups, prepare_groups, to_device
from .loss import distribution_loss_sum
from .metrics import DecisionMetrics
from .model import build_model
from .release import load_release, macro_metrics, source_groups, validate_prefix_layout
from .usage import TrainingUsage, batch_usage


def load_sources(specs):
    sources = {}
    for name, spec in specs.items():
        path = Path(spec["path"])
        dataset = (
            load_from_disk(str(path))
            if path.is_dir()
            else load_dataset("json", data_files=str(path), split="train")
        )
        if not isinstance(dataset, Dataset):
            raise ValueError("Each source path must select one explicit split (Dataset or JSONL)")
        if not len(dataset):
            raise ValueError(f"Empty source: {name}")
        sources[name] = dataset
    if not sources:
        raise ValueError("At least one source is required")
    return sources


def source_batches(
    counts,
    specs,
    *,
    mode,
    batch_size,
    seed,
    epoch_fraction=1.0,
    alpha=0.5,
    dataset_samples=None,
    dataset_cap=None,
    dataset_passes=None,
    rounds=10,
):
    """Yield source-homogeneous logical batches with reproducible row indices."""
    if batch_size < 1 or not math.isfinite(epoch_fraction) or epoch_fraction <= 0:
        raise ValueError("batch_size and epoch_fraction must be positive")
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("sampling_alpha must be finite and nonnegative")
    rng = random.Random(seed)
    if mode == "round_passes":
        from .sampling import round_batches, round_plan

        plan = round_plan(
            counts,
            {} if dataset_samples is None else dataset_samples,
            epoch_fraction,
            batch_size=batch_size,
            rounds=rounds,
            dataset_passes=dataset_passes,
            dataset_cap=dataset_cap,
        )
        for name, rows, _ in round_batches(plan, seed=seed):
            yield name, rows
    elif mode == "uniform":
        from .sampling import uniform_counts

        bases, selected = uniform_counts(
            counts,
            {} if dataset_samples is None else dataset_samples,
            epoch_fraction,
            dataset_cap=dataset_cap,
        )
        schedule = [n for n in sorted(counts) for _ in range(math.ceil(selected[n] / batch_size))]
        rng.shuffle(schedule)
        pools, offsets = {}, {}
        for name in schedule:
            if name not in pools:
                source_rng = random.Random(f"{seed}:{name}:uniform-v1")
                pools[name] = source_rng.sample(range(counts[name]), bases[name])[: selected[name]]
                offsets[name] = 0
            start = offsets[name]
            rows = pools[name][start : start + batch_size]
            offsets[name] += len(rows)
            yield name, rows
    elif mode == "source_passes":
        batch_counts = {}
        for name, count in counts.items():
            passes = specs.get(name, {}).get("passes", 1)
            if not isinstance(passes, int) or passes < 1:
                raise ValueError("passes must be a positive integer")
            batch_counts[name] = math.ceil(count / batch_size) * passes
        schedule = [name for name, count in batch_counts.items() for _ in range(count)]
        rng.shuffle(schedule)

        def batches(name, source_rng):
            for _ in range(specs.get(name, {}).get("passes", 1)):
                rows = list(range(counts[name]))
                source_rng.shuffle(rows)
                for start in range(0, len(rows), batch_size):
                    yield rows[start : start + batch_size]

        streams = {name: batches(name, random.Random(rng.getrandbits(64))) for name in batch_counts}
        for name in schedule:
            yield name, next(streams[name])
    elif mode == "weighted":
        # Source selection and row shuffling use independent RNG streams.
        # Drop a pool's tail instead of repeating rows within a logical batch.
        from .sampling import uniform_counts

        bases, _ = uniform_counts(counts, dataset_samples or {}, 1.0, dataset_cap=dataset_cap)
        names = sorted(n for n in bases if bases[n] >= batch_size)
        if not names:
            raise ValueError("No source can produce a complete logical batch")
        array_rng = np.random.default_rng(seed)
        weights = [bases[n] ** alpha for n in names]
        pools, offsets, capped_rows = {}, {}, {}
        for _ in range(int(sum(bases.values()) * epoch_fraction) // batch_size):
            name = rng.choices(names, weights)[0]
            if name not in pools or offsets[name] + batch_size > len(pools[name]):
                if name not in capped_rows:
                    cap_rng = random.Random(f"{seed}:{name}:weighted-cap-v1")
                    capped_rows[name] = np.asarray(
                        cap_rng.sample(range(counts[name]), bases[name])
                        if bases[name] < counts[name]
                        else range(counts[name]),
                        dtype=np.int64,
                    )
                pools[name] = array_rng.permutation(capped_rows[name])
                offsets[name] = 0
            start = offsets[name]
            offsets[name] += batch_size
            yield name, pools[name][start : start + batch_size].tolist()
    else:
        raise ValueError("sampling must be uniform, source_passes, round_passes or weighted")


def optimizer_for(model, learning_rate, head_learning_rate, weight_decay=0.01):
    return torch.optim.AdamW(
        [
            dict(params=[p for p in model[0].parameters() if p.requires_grad], lr=learning_rate),
            dict(
                params=[p for p in model[1].parameters() if p.requires_grad], lr=head_learning_rate
            ),
        ],
        weight_decay=weight_decay,
    )


def train_step(model, groups, optimizer, controller, *, group_sources=None):
    """Accumulate microbatch loss sums divided by the actual logical batch size."""
    if not groups or any(g.target is None for g in groups):
        raise ValueError("Training requires target distributions for every group")
    if any(g.task not in model[1].tasks for g in groups):
        raise ValueError("Training group task has no model head")
    prepared = prepare_groups(groups, model[0])
    device = model.device
    model.train()
    base_bytes = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        base_bytes = torch.cuda.memory_allocated(device)

    def attempt(microbatches):
        total = torch.zeros((), device=device)
        for batch in microbatches:
            features = to_device(collate_groups(batch, model[0]), device)
            scores = model(features)["scores"]
            loss = distribution_loss_sum(scores, [g.target for g in batch]) / len(groups)
            loss.backward()
            total += loss.detach()
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite training loss")
        return total.item()

    stats = adaptive_backward(prepared, controller, optimizer, attempt, padded_documents=True)
    if device.type == "cuda" and not stats["oom_retries"]:
        controller.observe(
            tokens=stats["max_work_tokens"],
            peak_bytes=torch.cuda.max_memory_allocated(device),
            base_bytes=base_bytes,
        )
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    stats["grad_norm"] = norm.item()
    if device.type == "cuda":
        stats["peak_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    stats["usage_by_task"] = batch_usage(prepared)
    if group_sources is not None:
        per_source = {}
        for source, group in zip(group_sources, prepared, strict=True):
            per_source.setdefault(source, []).append(group)
        stats["usage_by_source"] = {
            source: batch_usage(items) for source, items in per_source.items()
        }
    return stats


def scheduler_for(optimizer, steps, warmup_ratio=0.1, kind="linear"):
    if kind == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.ceil(steps * warmup_ratio),
            num_training_steps=steps,
        )
    if kind != "linear":
        raise ValueError("lr_scheduler must be linear or cosine")
    warmup = max(1, int(steps * warmup_ratio))
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (
            (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup))
        ),
    )


@torch.inference_mode()
def evaluate(model, sources, batch_size, token_budget):
    from .data import pack_groups

    results = {}
    was_training = model.training
    model.eval()
    try:
        for name, dataset in sources.items():
            totals = {}
            for start in range(0, len(dataset), batch_size):
                groups = source_groups(
                    dataset, list(range(start, min(start + batch_size, len(dataset))))
                )
                if any(g.target is None for g in groups):
                    raise ValueError("Evaluation requires targets")
                for batch in pack_groups(prepare_groups(groups, model[0]), token_budget):
                    features = to_device(collate_groups(batch, model[0]), model.device)
                    logits = (
                        model(features)["scores"].flatten().split([len(g.documents) for g in batch])
                    )
                    for group, scores in zip(batch, logits, strict=True):
                        totals.setdefault(group.task, DecisionMetrics()).update(group, scores)
            results[name] = {task: metrics.result() for task, metrics in totals.items()}
    finally:
        model.train(was_training)
    return results


def run(config, max_steps=None):
    from .tracking import RunTracker

    tracker = RunTracker(config.get("wandb"))
    try:
        result = _run(config, max_steps, tracker)
    except BaseException:
        tracker.finish(exit_code=1)
        raise
    tracker.finish(exit_code=0)
    return result


def _run(config, max_steps, tracker):
    set_seed(config.get("seed", 42))
    train, data = config["training"], config["data"]
    device = train.get("device", "cuda")
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Expose one available GPU with CUDA_VISIBLE_DEVICES")
    batch_size = train.get("batch_size", 512)
    if batch_size < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("batch_size and max_steps must be positive")
    if "additional_roots" in data and "root" not in data:
        raise ValueError("additional_roots requires data.root")
    if "root" in data and "sources" in data:
        raise ValueError("Choose data.root or data.sources, not both")
    from .layout_sampling import LAYOUTS, LayoutSampler

    layout_sampler = None
    if "prefix_layout_weights" in data:
        if "prefix_layout" in data:
            raise ValueError("Choose prefix_layout or prefix_layout_weights, not both")
        if "root" not in data:
            raise ValueError("prefix_layout_weights requires data.root")
        layout_sampler = LayoutSampler(data["prefix_layout_weights"], config.get("seed", 42))
    prefix_layout = data.get("prefix_layout", "instruction_state")
    validate_prefix_layout(prefix_layout)
    if "prefix_layout" in data and "root" not in data:
        raise ValueError("data.prefix_layout requires data.root; text sources are already rendered")
    max_steps = max_steps if max_steps is not None else train.get("max_steps")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("max_steps must be a positive integer")
    if train.get("eval_steps", 500) < 1:
        raise ValueError("eval_steps must be positive")
    if not 0 <= train.get("warmup_ratio", 0.1) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    evaluation = config.get("evaluation", {})
    evaluation_layouts = evaluation.get("prefix_layouts")
    if evaluation_layouts is not None:
        if "root" not in data or not isinstance(evaluation_layouts, list) or not evaluation_layouts:
            raise ValueError("evaluation.prefix_layouts requires data.root and a nonempty list")
        for layout in evaluation_layouts:
            validate_prefix_layout(layout)
        if len(set(evaluation_layouts)) != len(evaluation_layouts):
            raise ValueError("evaluation.prefix_layouts must not contain duplicates")
        if layout_sampler is not None and set(evaluation_layouts) != set(LAYOUTS):
            raise ValueError("Mixed training requires evaluation of both prefix layouts")
    elif layout_sampler is not None:
        evaluation_layouts = list(LAYOUTS)
    eval_budget = evaluation.get("token_budget", 16000)
    if eval_budget < 1:
        raise ValueError("evaluation.token_budget must be positive")
    audits = {}
    from .hub_release import pin_hub_reference, pin_model_reference

    resolved_config = copy.deepcopy(config)
    resolved_config["model"] = pin_model_reference(config["model"])
    hub_pins = {}
    release_root = pin_hub_reference(data["root"], hub_pins) if "root" in data else None
    test_root = (
        pin_hub_reference(evaluation["test_root"], hub_pins)
        if evaluation.get("test_root")
        else None
    )
    if release_root is not None:
        resolved_config["data"]["root"] = release_root
    if test_root is not None:
        resolved_config["evaluation"]["test_root"] = test_root
    if "root" in data:
        sources, audits["train"] = load_release(
            release_root,
            "train",
            exclude_datasets=data.get("exclude_datasets"),
            prefix_layout=prefix_layout,
        )
        validation, audits["validation"] = load_release(
            release_root,
            "validation",
            sample_cases=evaluation.get("validation_samples", 5),
            sampling=evaluation.get("validation_sampling", "uniform"),
            seed=evaluation.get("validation_seed", config.get("seed", 42)),
            exclude_datasets=evaluation.get("validation_exclude_datasets"),
            prefix_layout=prefix_layout,
        )
    else:
        if "exclude_datasets" in data or "validation_exclude_datasets" in evaluation:
            raise ValueError("Dataset exclusion lists require data.root")
        sources = load_sources(data["sources"])
        validation = (
            load_sources(data["validation_sources"]) if data.get("validation_sources") else {}
        )
    if "additional_roots" in data:
        from .release_mixing import load_additional_releases

        extra_sources, extra_validation, extra_audits, extra_roots = load_additional_releases(
            data["additional_roots"],
            existing_names=set(sources) | set(validation),
            prefix_layout=prefix_layout,
            validation_samples=evaluation.get("validation_samples", 5),
            validation_sampling=evaluation.get("validation_sampling", "uniform"),
            validation_seed=evaluation.get("validation_seed", config.get("seed", 42)),
            hub_pins=hub_pins,
        )
        sources.update(extra_sources)
        validation.update(extra_validation)
        resolved_config["data"]["additional_roots"] = extra_roots
        for role in ("train", "validation"):
            audits[role]["additional_releases"] = extra_audits[role]
    test = {}
    if evaluation.get("test_root"):
        test, audits["test"] = load_release(
            test_root,
            "test",
            exclude_datasets=evaluation.get("test_exclude_datasets"),
            prefix_layout=prefix_layout,
        )
        if not test:
            raise ValueError("test_root contains no labeled test splits")
    elif "test_exclude_datasets" in evaluation:
        raise ValueError("test_exclude_datasets requires evaluation.test_root")
    if not sources:
        raise ValueError("No training datasets remain after selection")
    counts = {n: len(d) for n, d in sources.items()}
    mode = data.get("sampling", "weighted")
    fraction = data.get("epoch_fraction", 1.0)
    if not math.isfinite(fraction) or fraction <= 0:
        raise ValueError("epoch_fraction must be finite and positive")
    if mode not in {"weighted", "source_passes", "round_passes", "uniform"}:
        raise ValueError("sampling must be uniform, source_passes, round_passes or weighted")
    alpha = data.get("sampling_alpha", 0.5)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("sampling_alpha must be finite and nonnegative")
    if mode != "round_passes" and ("dataset_passes" in data or "rounds" in data):
        raise ValueError("dataset_passes and rounds require sampling: round_passes")
    batch_grouping = data.get("batch_grouping", "source")
    if batch_grouping not in {"source", "task"}:
        raise ValueError("batch_grouping must be source or task")
    if batch_grouping == "task" and mode != "round_passes":
        raise ValueError("batch_grouping: task requires sampling: round_passes")
    source_tasks = None
    sample_plan: dict[str, Any] | None = None
    selected = {}
    if mode == "round_passes":
        from .sampling import round_plan

        if "sampling_alpha" in data:
            raise ValueError("round_passes does not use sampling_alpha; remove it")
        if any("passes" in spec for spec in data.get("sources", {}).values()):
            raise ValueError("round_passes uses dataset_passes, not sources.*.passes")
        sample_plan = round_plan(
            counts,
            data.get("dataset_samples", {}),
            fraction,
            batch_size=batch_size,
            rounds=data.get("rounds", 10),
            dataset_passes=data.get("dataset_passes"),
            dataset_cap=data.get("dataset_cap"),
        )
        sample_plan["seed"] = config.get("seed", 42)
        sample_plan["batch_grouping"] = batch_grouping
        if batch_grouping == "task":
            from .sampling import task_round_batches

            if not all(hasattr(source, "decision_tasks") for source in sources.values()):
                raise ValueError("Task grouping requires manifest-based decision releases")
            source_tasks = {name: source.decision_tasks() for name, source in sources.items()}
            sample_plan["source_grouped_steps"] = sample_plan["steps"]
            sample_plan["steps"] = sum(
                1
                for _ in task_round_batches(sample_plan, source_tasks, seed=config.get("seed", 42))
            )
    elif mode == "uniform":
        from .sampling import uniform_counts

        if "sampling_alpha" in data:
            raise ValueError("uniform sampling does not use sampling_alpha; remove it")
        bases, selected = uniform_counts(
            counts, data.get("dataset_samples", {}), fraction, dataset_cap=data.get("dataset_cap")
        )
        sample_plan = dict(
            mode=mode,
            seed=config.get("seed", 42),
            unit="decision",
            source_counts=counts,
            base_counts=bases,
            selected_counts=selected,
            epoch_fraction=fraction,
        )
    elif mode == "weighted":
        from .sampling import uniform_counts

        bases, _ = uniform_counts(
            counts, data.get("dataset_samples", {}), 1.0, dataset_cap=data.get("dataset_cap")
        )
        sample_plan = dict(
            mode=mode,
            seed=config.get("seed", 42),
            unit="decision",
            source_counts=counts,
            base_counts=bases,
            epoch_fraction=fraction,
            sampling_alpha=alpha,
            sampling_weights={n: c**alpha for n, c in bases.items() if c >= batch_size},
            excluded_small_sources=[n for n, c in bases.items() if c < batch_size],
        )
    elif "dataset_samples" in data or "dataset_cap" in data:
        raise ValueError("dataset_samples and dataset_cap require sampling: uniform or weighted")
    steps = (
        sum(
            math.ceil(n / batch_size) * data.get("sources", {}).get(name, {}).get("passes", 1)
            for name, n in counts.items()
        )
        if mode == "source_passes"
        else int(sum(counts.values()) * fraction) // batch_size
    )
    if mode == "uniform":
        steps = sum(math.ceil(n / batch_size) for n in selected.values())
    elif mode == "weighted":
        steps = int(sum(bases.values()) * fraction) // batch_size
        assert sample_plan is not None
        sample_plan["steps"] = steps
        sample_plan["sampled_decisions"] = steps * batch_size
    if mode == "round_passes":
        assert sample_plan is not None
        steps = sample_plan["steps"]
    steps = min(steps, max_steps) if max_steps is not None else steps
    if mode == "round_passes":
        assert sample_plan is not None
        sample_plan["run_steps"] = steps
        sample_plan["budget_truncated"] = steps < sample_plan["steps"]
    if steps < 1:
        raise ValueError("Training budget does not contain a complete batch")
    out = Path(train["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    if sample_plan is not None:
        sample_plan["dataset_cap"] = data.get("dataset_cap")
        (out / "sampling_plan.json").write_text(json.dumps(sample_plan, indent=2))
    (out / "training_config.json").write_text(json.dumps(config, indent=2))
    if layout_sampler is not None:
        audits["train"].pop("prefix_layout", None)
        audits["train"]["prefix_layout_weights"] = data["prefix_layout_weights"]
    if evaluation_layouts is not None:
        for role in ["validation", "test"]:
            if role in audits:
                audits[role].pop("prefix_layout", None)
                audits[role]["prefix_layouts"] = evaluation_layouts
    (out / "data_manifest.json").write_text(json.dumps(audits, indent=2))
    (out / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2))
    tracker.start(out, resolved_config, sample_plan, steps)
    model_config = dict(resolved_config["model"])
    checkpoint = model_config.pop("checkpoint", None)
    choice_interaction = model_config.pop("choice_interaction", None)
    if checkpoint is not None:
        allowed = {
            "query_length",
            "query_truncation",
            "document_length",
            "attention_backend",
            "gradient_checkpointing",
            "frozen_linear_bf16",
            "fused_rotary",
        }
        if set(model_config) - allowed:
            raise ValueError("Checkpoint accepts only length and runtime overrides")
        model = SentenceTransformer(
            checkpoint,
            device=device,
            local_files_only=True,
            trust_remote_code=True,
            model_kwargs=model_config,
        )
        files = {
            str(p.relative_to(checkpoint)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(checkpoint).rglob("*"))
            if p.is_file()
        }
        (out / "initial_model_sha256.json").write_text(json.dumps(files, indent=2))
    else:
        model = build_model(**model_config, device=device)

    if choice_interaction is not None:
        from .modules import DecisionHeads

        heads = model[1]
        if not isinstance(heads, DecisionHeads):
            raise ValueError("Choice interaction requires DecisionHeads")
        heads.enable_choice_interaction(choice_interaction)

    tracker.parameters(model)
    evaluation_step = 0

    def save_evaluation(label, selected):
        layouts = evaluation_layouts if evaluation_layouts is not None else (None,)
        for layout in layouts:
            rendered = selected
            suffix = label
            if layout is not None:
                rendered = {name: copy.copy(source) for name, source in selected.items()}
                for source in rendered.values():
                    source.prefix_layout = layout
                suffix = f"{label}-{layout}"
            result = evaluate(model, rendered, batch_size, eval_budget)
            (out / f"{suffix}.json").write_text(json.dumps(result, indent=2))
            summary = macro_metrics(result)
            (out / f"{suffix}-summary.json").write_text(json.dumps(summary, indent=2))
            tracker.evaluation(label, layout or prefix_layout, evaluation_step, result, summary)

    if evaluation.get("before_training", False):
        if validation:
            save_evaluation("validation-initial", validation)
        if test:
            save_evaluation("test-initial", test)
    optimizer = optimizer_for(
        model,
        float(train["learning_rate"]),
        float(train["head_learning_rate"]),
        train.get("weight_decay", 0.01),
    )
    scheduler = scheduler_for(
        optimizer, steps, train.get("warmup_ratio", 0.1), train.get("lr_scheduler", "linear")
    )
    initial = train.get("initial_tokens", 16384)
    maximum = train.get("max_tokens", 524288)
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        fraction_gpu = train.get("memory_fraction", 0.8)
        if not 0 < fraction_gpu < 1:
            raise ValueError("memory_fraction must be between zero and one")
        target_bytes = int(min(total, free + torch.cuda.memory_reserved()) * fraction_gpu)
    else:
        target_bytes, maximum = 2**60, initial
    controller = TokenBudget(initial, maximum=maximum, target_bytes=target_bytes)
    started = time.monotonic()
    trained_decisions = 0
    usage = TrainingUsage()
    if mode == "round_passes":
        from .sampling import round_batches

        if batch_grouping == "task":
            from .sampling import task_round_batches

            training_batches = task_round_batches(
                sample_plan, source_tasks, seed=config.get("seed", 42)
            )
        else:
            training_batches = round_batches(sample_plan, seed=config.get("seed", 42))
    else:
        training_batches = (
            (name, rows, {})
            for name, rows in source_batches(
                counts,
                data.get("sources", {}),
                mode=mode,
                batch_size=batch_size,
                seed=config.get("seed", 42),
                epoch_fraction=fraction,
                alpha=data.get("sampling_alpha", 0.5),
                dataset_samples=data.get("dataset_samples"),
                dataset_cap=data.get("dataset_cap"),
            )
        )
    with (out / "history.jsonl").open("w") as history:
        for step, (name, rows, schedule_stats) in enumerate(training_batches, 1):
            step_started = time.monotonic()
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            layout_stats = {}
            if batch_grouping == "task":
                layouts = (
                    layout_sampler.sample(len(rows))
                    if layout_sampler is not None
                    else [prefix_layout] * len(rows)
                )
                grouped_refs = {}
                for position, (source, row, pass_index) in enumerate(rows):
                    grouped_refs.setdefault(source, []).append((position, row, layouts[position]))
                groups = [None] * len(rows)
                for source, refs in grouped_refs.items():
                    rendered = sources[source].groups(
                        [r[1] for r in refs], prefix_layouts=[r[2] for r in refs]
                    )
                    for ref, group in zip(refs, rendered, strict=True):
                        groups[ref[0]] = group
                schedule_stats["source_counts"] = dict(Counter(ref[0] for ref in rows))
                if layout_sampler is not None:
                    layout_stats = dict(
                        prefix_layout_counts=dict(Counter(layouts)),
                        prefix_layout_sha256=hashlib.sha256(
                            json.dumps(layouts).encode()
                        ).hexdigest(),
                    )
            elif layout_sampler is not None:
                layouts = layout_sampler.sample(len(rows))
                groups = sources[name].groups(rows, prefix_layouts=layouts)
                layout_stats = dict(
                    prefix_layout_counts=dict(Counter(layouts)),
                    prefix_layout_sha256=hashlib.sha256(json.dumps(layouts).encode()).hexdigest(),
                )
            else:
                groups = source_groups(sources[name], rows)
            trained_decisions += len(groups)
            if batch_grouping == "task":
                stats = train_step(
                    model, groups, optimizer, controller, group_sources=[ref[0] for ref in rows]
                )
            else:
                stats = train_step(model, groups, optimizer, controller)
            if batch_grouping == "task":
                for source, refs in grouped_refs.items():
                    source_batch = [groups[ref[0]] for ref in refs]
                    source_usage = stats["usage_by_source"][source]
                    usage.update(source, source_batch, source_usage)
            else:
                usage.update(name, groups, stats["usage_by_task"])
            scheduler.step()
            entry = dict(
                step=step,
                source=name,
                groups=len(groups),
                row_indices_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                seconds=time.monotonic() - started,
                step_seconds=time.monotonic() - step_started,
                learning_rates=learning_rates,
                **stats,
                **layout_stats,
                **schedule_stats,
            )
            evaluation_step = step
            tracker.training(entry, trained_decisions, final=step == steps)
            history.write(json.dumps(entry) + "\n")
            history.flush()
            print(json.dumps(entry), flush=True)
            if validation and step % train.get("eval_steps", 500) == 0:
                save_evaluation(f"validation-{step}", validation)
            if step >= steps:
                break
    training_seconds = time.monotonic() - started
    (out / "training_usage.json").write_text(json.dumps(usage.result(), indent=2))
    optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(str(out / "model"), create_model_card=False)
    if "root" in data:
        (out / "model" / "release_rendering.json").write_text(
            json.dumps(
                dict(
                    prefix_layout_weights=data["prefix_layout_weights"],
                    evaluation_prefix_layouts=list(LAYOUTS),
                )
                if layout_sampler is not None
                else dict(prefix_layout=prefix_layout),
                indent=2,
            )
        )
    del scheduler, optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    model = SentenceTransformer(
        str(out / "model"),
        device=device,
        local_files_only=True,
        trust_remote_code=True,
    )
    if validation:
        save_evaluation("validation", validation)
    if test:
        save_evaluation("test", test)
    result = dict(
        steps=steps,
        trained_decisions=trained_decisions,
        source_decisions=counts,
        excluded_small_sources=[n for n, c in bases.items() if c < batch_size]
        if mode == "weighted"
        else [],
        training_seconds=training_seconds,
        seconds=time.monotonic() - started,
        model=str(out / "model"),
    )
    if layout_sampler is not None:
        result["prefix_layout_counts"] = dict(layout_sampler.counts)
    (out / "result.json").write_text(json.dumps(result, indent=2))
    tracker.result(result, usage.result())
    return result


def main():
    from .cli import resolve_config, train_percent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument("--smoke", action="store_true", help="Train 0.1%% of the configured pool")
    budget.add_argument(
        "--train-percent",
        type=train_percent,
        help="Replace epoch_fraction with this percentage (after caps for uniform sampling)",
    )
    parser.add_argument(
        "--output-dir", help="Override output directory; existing paths are rejected"
    )
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    config = resolve_config(
        yaml.safe_load(Path(args.config).read_text()),
        smoke=args.smoke,
        percent=args.train_percent,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
    run(config)


if __name__ == "__main__":
    main()
