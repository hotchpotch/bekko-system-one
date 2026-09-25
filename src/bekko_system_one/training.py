"""Single-device training of Sentence Transformers candidate-scoring modules."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from transformers import get_cosine_schedule_with_warmup, set_seed

from .batching import TokenBudget, adaptive_backward
from .data import Group, collate_groups, prepare_groups, to_device
from .loss import distribution_loss_sum
from .model import build_model


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


def source_batches(counts, specs, *, mode, batch_size, seed, epoch_fraction=1.0, alpha=0.5):
    """Yield source-homogeneous logical batches with reproducible row indices."""
    rng = random.Random(seed)
    if mode == "source_passes":
        batch_counts = {}
        for name, count in counts.items():
            passes = specs[name].get("passes", 1)
            if not isinstance(passes, int) or passes < 1:
                raise ValueError("passes must be a positive integer")
            batch_counts[name] = math.ceil(count / batch_size) * passes
        schedule = [name for name, count in batch_counts.items() for _ in range(count)]
        rng.shuffle(schedule)

        def batches(name, source_rng):
            for _ in range(specs[name].get("passes", 1)):
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
        names = sorted(n for n in counts if counts[n] >= batch_size)
        if not names:
            raise ValueError("No source can produce a complete logical batch")
        array_rng = np.random.default_rng(seed)
        weights = [counts[n] ** alpha for n in names]
        pools, offsets = {}, {}
        for _ in range(int(sum(counts.values()) * epoch_fraction) // batch_size):
            name = rng.choices(names, weights)[0]
            if name not in pools or offsets[name] + batch_size > len(pools[name]):
                pools[name] = array_rng.permutation(counts[name])
                offsets[name] = 0
            start = offsets[name]
            offsets[name] += batch_size
            yield name, pools[name][start : start + batch_size].tolist()
    else:
        raise ValueError("sampling must be source_passes or weighted")


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


def train_step(model, groups, optimizer, controller):
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
                groups = [
                    Group.from_dict(dataset[i])
                    for i in range(start, min(start + batch_size, len(dataset)))
                ]
                if any(g.target is None for g in groups):
                    raise ValueError("Evaluation requires targets")
                for batch in pack_groups(prepare_groups(groups, model[0]), token_budget):
                    features = to_device(collate_groups(batch, model[0]), model.device)
                    logits = (
                        model(features)["scores"].flatten().split([len(g.documents) for g in batch])
                    )
                    for group, scores in zip(batch, logits, strict=True):
                        target = scores.new_tensor(group.target)
                        if not torch.isfinite(scores).all():
                            raise FloatingPointError("Nonfinite evaluation scores")
                        item = totals.setdefault(
                            group.task, dict(groups=0, cross_entropy=0.0, accuracy=0.0)
                        )
                        item["groups"] += 1
                        item["cross_entropy"] += -(target * scores.log_softmax(0)).sum().item()
                        item["accuracy"] += float(target[scores.argmax()] == target.max())
            results[name] = {
                task: {k: v if k == "groups" else v / m["groups"] for k, v in m.items()}
                for task, m in totals.items()
            }
    finally:
        model.train(was_training)
    return results


def run(config, max_steps=None):
    set_seed(config.get("seed", 42))
    train, data = config["training"], config["data"]
    device = train.get("device", "cuda")
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise ValueError("Expose one available GPU with CUDA_VISIBLE_DEVICES")
    batch_size = train.get("batch_size", 512)
    if batch_size < 1 or (max_steps is not None and max_steps < 1):
        raise ValueError("batch_size and max_steps must be positive")
    sources = load_sources(data["sources"])
    counts = {n: len(d) for n, d in sources.items()}
    mode = data.get("sampling", "weighted")
    fraction = data.get("epoch_fraction", 1.0)
    if fraction <= 0:
        raise ValueError("epoch_fraction must be positive")
    steps = (
        sum(
            math.ceil(n / batch_size) * data["sources"][name].get("passes", 1)
            for name, n in counts.items()
        )
        if mode == "source_passes"
        else int(sum(counts.values()) * fraction) // batch_size
    )
    steps = min(steps, max_steps) if max_steps is not None else steps
    if steps < 1:
        raise ValueError("Training budget does not contain a complete batch")
    out = Path(train["output_dir"])
    out.mkdir(parents=True, exist_ok=False)
    (out / "training_config.json").write_text(json.dumps(config, indent=2))
    model = build_model(**config["model"], device=device)
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
    validation = load_sources(data["validation_sources"]) if data.get("validation_sources") else {}
    started = time.monotonic()
    with (out / "history.jsonl").open("w") as history:
        for step, (name, rows) in enumerate(
            source_batches(
                counts,
                data["sources"],
                mode=mode,
                batch_size=batch_size,
                seed=config.get("seed", 42),
                epoch_fraction=fraction,
                alpha=data.get("sampling_alpha", 0.5),
            ),
            1,
        ):
            step_started = time.monotonic()
            learning_rates = [group["lr"] for group in optimizer.param_groups]
            groups = [Group.from_dict(row) for row in sources[name].select(rows)]
            stats = train_step(model, groups, optimizer, controller)
            scheduler.step()
            entry = dict(
                step=step,
                source=name,
                groups=len(groups),
                seconds=time.monotonic() - started,
                step_seconds=time.monotonic() - step_started,
                learning_rates=learning_rates,
                **stats,
            )
            history.write(json.dumps(entry) + "\n")
            history.flush()
            print(json.dumps(entry), flush=True)
            if validation and step % train.get("eval_steps", 500) == 0:
                result = evaluate(model, validation, batch_size, controller.limit)
                (out / f"validation-{step}.json").write_text(json.dumps(result, indent=2))
            if step >= steps:
                break
    training_seconds = time.monotonic() - started
    optimizer.zero_grad(set_to_none=True)
    model.save_pretrained(str(out / "model"), create_model_card=False)
    if validation:
        (out / "validation.json").write_text(
            json.dumps(
                evaluate(model, validation, batch_size, controller.limit),
                indent=2,
            )
        )
    result = dict(
        steps=steps,
        training_seconds=training_seconds,
        seconds=time.monotonic() - started,
        model=str(out / "model"),
    )
    (out / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text()), args.max_steps)


if __name__ == "__main__":
    main()
