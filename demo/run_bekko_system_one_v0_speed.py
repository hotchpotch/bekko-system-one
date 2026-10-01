# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "torch==2.10.0", "transformers==5.17.0", "sentence-transformers==6.1.0",
#   "datasets>=4,<5", "huggingface-hub>=1,<2", "rich>=14,<15",
# ]
# [tool.uv.sources]
# torch = { index = "torch-cu130" }
# [[tool.uv.index]]
# name = "torch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///
"""Standalone Bekko v0 S1MB Choice/Noul/Score inference speed demo for the terminal."""

import argparse
import importlib
import json
import math
import os
import random
import select
import statistics
import sys
import termios
import time
import tty
import unicodedata
import warnings
from collections import Counter, defaultdict, deque
from contextlib import nullcontext
from pathlib import Path

REVISIONS = {
    "17m": "2147c3d9d00559bf972616c2589eee967e266f3b",
    "68m": "6eb1bae2d35066b0d634fabaf8c79beafc6fd9f1",
    "400m": "4aeb85b9d4042d75d8b8adf6ff7ba9e4629510ba",
}


class TraceLog:
    def __init__(self, path):
        self.path = path
        self.started = time.perf_counter()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x"):
                pass

    def event(self, name, **fields):
        if self.path is not None:
            with self.path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {"event": name, "elapsed_s": time.perf_counter() - self.started, **fields},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


class RollingThroughput:
    def __init__(self, started):
        self.samples = deque([(started, 0, 0, 0.0)])
        self.updated = started
        self.rates = {"rolling_tokens": 0.0, "rolling_decisions": 0.0}

    def update(self, now, stats):
        if now - self.updated < 1.0:
            return self.rates
        self.updated = now
        self.samples.append((now, stats["tokens"], stats["completed"], stats["inference_seconds"]))
        cutoff = now - 5.0
        while len(self.samples) > 1 and self.samples[1][0] <= cutoff:
            self.samples.popleft()
        baseline = self.samples[0]
        if baseline[0] < cutoff and len(self.samples) > 1:
            following = self.samples[1]
            fraction = (cutoff - baseline[0]) / (following[0] - baseline[0])
            baseline = tuple(
                previous + fraction * (current - previous)
                for previous, current in zip(baseline, following, strict=True)
            )
        duration = max(stats["inference_seconds"] - baseline[3], 1e-9)
        self.rates = {
            "rolling_tokens": max(0, stats["tokens"] - baseline[1]) / duration,
            "rolling_decisions": max(0, stats["completed"] - baseline[2]) / duration,
        }
        return self.rates


def live_frame(args, device_name, counts, total, state, now=None):
    state = dict(state)
    if state.get("final") is not None:
        return state["final"]
    now = time.perf_counter() if now is None else now
    snapshot = dict(state["stats"])
    if not state.get("clock_stopped", False):
        snapshot["wall_seconds"] = now - state["started"]
    status = state.get("phase", "LIVE INFERENCE")
    pending = state.get("pending")
    if pending is not None:
        age = max(0, now - pending["started"])
        snapshot["inference_seconds"] += age
        snapshot["activity"] = (
            f"Batch {pending['number']}/{pending['total']} · {age:.1f}s in progress"
        )
        if age >= 0.5:
            snapshot["activity"] += " · compile / inference"
    if state.get("throughput") is not None:
        snapshot.update(state["throughput"].update(now, snapshot))
    return display(args, device_name, counts, total, snapshot, state.get("example"), status)


class TerminalKeys:
    def __init__(self, enabled=True, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.enabled = enabled and self.stream.isatty()
        self.fd = None
        self.saved = None

    def __enter__(self):
        if self.enabled:
            self.fd = self.stream.fileno()
            self.saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd, termios.TCSANOW)
        return self

    def __exit__(self, *_):
        if self.saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)

    def poll(self, timeout=0):
        if self.fd is None or not select.select([self.fd], [], [], timeout)[0]:
            return None
        key = os.read(self.fd, 1)
        if key == b"\x1b":
            if select.select([self.fd], [], [], 0.03)[0]:
                prefix = os.read(self.fd, 1)
                if prefix in (b"[", b"O"):
                    for _ in range(32):
                        if not select.select([self.fd], [], [], 0.03)[0]:
                            break
                        suffix = os.read(self.fd, 1)
                        if b"@" <= suffix <= b"~":
                            break
                    return None
            return "escape"
        if not key:
            return "escape"
        if key in (b"\n", b"\r"):
            return "enter"
        return None

    def wait(self):
        if self.fd is None:
            raise RuntimeError("Interactive key input requires a terminal")
        while True:
            key = self.poll(0.1)
            if key in {"enter", "escape"}:
                return key


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def warmup_batches(value):
    if value == "all":
        return value
    try:
        count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a nonnegative batch count or 'all'") from None
    if count < 0:
        raise argparse.ArgumentTypeError("Use a nonnegative batch count or 'all'")
    return count


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=REVISIONS, default="17m")
    parser.add_argument("--device", choices=["cuda:0", "cuda:1", "cpu"], default="cuda:0")
    parser.add_argument(
        "--data-dir", type=Path, help="Explicit offline S1MB root with Parquet files"
    )
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--batch-size", "--bs", type=positive, default=32)
    parser.add_argument(
        "--input-max", type=positive, help="Default: checkpoint context limit (S1MB native)"
    )
    parser.add_argument("--query-max", type=positive, help="Default: checkpoint query limit")
    parser.add_argument(
        "--candidate-max", type=positive, help="Default: checkpoint candidate limit"
    )
    parser.add_argument("--token-budget", type=positive, default=64000)
    parser.add_argument("--limit", type=positive, help="Maximum distinct decisions, before replay")
    parser.add_argument("--passes", type=positive, default=3)
    parser.add_argument(
        "--warmup-batches",
        type=warmup_batches,
        default=16,
        help="Distinct batches to prepare: 16 (default), all, or 0 to skip",
    )
    parser.add_argument("--fps", type=positive, default=12)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument(
        "--allow-recompile",
        action="store_true",
        help="Allow new compilation after Ready (can pause the demo)",
    )
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument(
        "--wait", action="store_true", help="Press Enter after warmup before starting"
    )
    parser.add_argument("--output", type=Path, help="Write measured summary to a NEW JSON file")
    parser.add_argument(
        "--verbose-log",
        nargs="?",
        type=Path,
        const=Path(f"output/bekko-speed-demo/trace-{time.time_ns()}.jsonl"),
        help="Write batch/shape/compiler timing JSONL to a new file; no input text",
    )
    args = parser.parse_args()
    if args.input_max is not None and args.input_max < 4:
        parser.error("--input-max must be at least 4")
    if args.output and args.output.exists():
        parser.error("--output already exists; choose a new file")
    return args


def resolve_input_limits(args, runtime):
    args.input_max = runtime.context_length if args.input_max is None else args.input_max
    args.query_max = min(
        runtime.query_length if args.query_max is None else args.query_max, args.input_max - 2
    )
    args.candidate_max = min(
        runtime.document_length if args.candidate_max is None else args.candidate_max,
        args.input_max - 3,
    )


def compile_label(args):
    if args.no_compile:
        return "EAGER"
    if getattr(args, "allow_recompile", False) or getattr(args, "warmup_batches", 1) == 0:
        return "COMPILED / RECOMPILE ENABLED"
    return "COMPILED + EAGER FALLBACK"


def select_rows(root, limit, seed):
    import pyarrow.parquet as parquet

    manifest = json.loads((root / "training-manifest.json").read_text())
    entries = []
    for subset in manifest["evaluation"]:
        name = subset["dataset"]
        if subset["split"] != "test":
            continue
        dataset = []
        for filename in subset["data_files"]:
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("Dataset path escapes root")
            path = root / filename
            dataset.extend(parquet.read_table(path).to_pylist())
        if len(dataset) != subset["cases"]:
            raise ValueError(f"Incomplete active subset: {name}")
        found = set()
        for row in dataset:
            if row["split"] != "test":
                raise ValueError("Unexpected row split")
            if row["case_id"] in found:
                raise ValueError(f"Duplicate case: {row['case_id']}")
            found.add(row["case_id"])
            targets = {target["decision_id"]: target for target in row["targets"]}
            for decision in row["input"]["decisions"]:
                if decision["kind"] == "judgment" and decision["type"] in {
                    "choice",
                    "noul",
                    "score",
                }:
                    entries.append(
                        {
                            "source": name,
                            "case_id": row["case_id"],
                            "target": aligned_target(decision, targets[decision["id"]]),
                            "input": {
                                "state_json": row["input"]["state_json"],
                                "decisions": [decision],
                            },
                        }
                    )
    references = score_references(entries)
    for entry in entries:
        entry["reference_scope"] = references
    random.Random(seed).shuffle(entries)
    entries = entries[:limit]
    if not entries:
        raise ValueError("No active Choice/Noul/Score test decisions found")
    return entries


def aligned_target(decision, target):
    ids = [candidate["id"] for candidate in decision["criteria"]]
    if (
        target["kind"] != "judgment_distribution"
        or len(set(ids)) != len(ids)
        or len(set(target["ids"])) != len(target["ids"])
        or set(ids) != set(target["ids"])
    ):
        raise ValueError("Invalid target/candidate alignment")
    mapping = dict(zip(target["ids"], target["probabilities"], strict=True))
    values = [mapping[key] for key in ids]
    if (
        any(not math.isfinite(value) or not 0 <= value <= 1 for value in values)
        or abs(sum(values) - 1) > 1e-5
    ):
        raise ValueError("Invalid target probabilities")
    return values


def benchmark_key(entry):
    return entry["source"] + "/" + entry["input"]["decisions"][0]["type"]


def score_references(entries):
    buckets = defaultdict(list)
    for entry in entries:
        buckets[benchmark_key(entry)].append(entry)
    references = {}
    for key, rows in buckets.items():
        task = rows[0]["input"]["decisions"][0]["type"]
        reference = {"task": task, "count": len(rows), "baseline": None}
        if task == "choice":
            uniform = sum(1 / len(row["target"]) for row in rows) / len(rows)
            id_gains, position_gains = defaultdict(float), defaultdict(float)
            for row in rows:
                criteria = row["input"]["decisions"][0]["criteria"]
                for position, (candidate, target) in enumerate(
                    zip(criteria, row["target"], strict=True)
                ):
                    gain = (target - 1 / len(criteria)) / len(rows)
                    id_gains[candidate["id"]] += gain
                    position_gains[position] += gain
            reference["baseline"] = uniform + max(0, *id_gains.values(), *position_gains.values())
        elif task == "score":
            values = [normalized_score(row, row["target"]) for row in rows]
            median = statistics.median(values)
            reference["baseline"] = sum(abs(value - median) for value in values) / len(values)
        references[key] = reference
    return references


def normalized_score(entry, probabilities):
    values = [candidate["value"] for candidate in entry["input"]["decisions"][0]["criteria"]]
    if any(value is None or not math.isfinite(value) for value in values) or max(values) <= min(
        values
    ):
        raise ValueError("Score requires a nondegenerate numeric scale")
    low, high = min(values), max(values)
    return sum(
        (value - low) / (high - low) * probability
        for value, probability in zip(values, probabilities, strict=True)
    )


def score_summary(entries, predictions):
    references = entries[0]["reference_scope"]
    buckets = defaultdict(list)
    for entry, probabilities in zip(entries, predictions, strict=True):
        if probabilities is None:
            continue
        if (
            len(probabilities) != len(entry["target"])
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities)
            or abs(sum(probabilities) - 1) > 1e-5
        ):
            raise ValueError("Invalid prediction probabilities")
        buckets[benchmark_key(entry)].append((entry, probabilities))
    results = {}
    for key, reference in references.items():
        rows = buckets[key]
        complete = len(rows) == reference["count"]
        skill = None
        task = reference["task"]
        if complete and task == "choice" and reference["baseline"] < 1 - 1e-12:
            accuracy = sum(
                entry["target"][max(range(len(probabilities)), key=probabilities.__getitem__)]
                for entry, probabilities in rows
            ) / len(rows)
            skill = (accuracy - reference["baseline"]) / (1 - reference["baseline"])
        elif complete and task == "noul":
            true_mass = false_mass = true_correct = false_correct = 0.0
            for entry, probabilities in rows:
                ids = [candidate["id"] for candidate in entry["input"]["decisions"][0]["criteria"]]
                positive = ids.index("true" if "true" in ids else "yes")
                negative = ids.index("false" if "false" in ids else "no")
                target = entry["target"][positive]
                false_target = entry["target"][negative]
                true_mass += target
                false_mass += false_target
                true_correct += target if probabilities[positive] >= 0.5 else 0
                false_correct += false_target if probabilities[positive] < 0.5 else 0
            if true_mass > 1e-12 and false_mass > 1e-12:
                skill = true_correct / true_mass + false_correct / false_mass - 1
        elif complete and task == "score" and reference["baseline"] > 1e-12:
            errors = []
            for entry, probabilities in rows:
                values = [
                    candidate["value"] for candidate in entry["input"]["decisions"][0]["criteria"]
                ]
                errors.append(
                    abs(
                        sum(
                            value * (actual - target)
                            for value, actual, target in zip(
                                values, probabilities, entry["target"], strict=True
                            )
                        )
                    )
                    / (max(values) - min(values))
                )
            skill = 1 - statistics.mean(errors) / reference["baseline"]
        results[key] = {
            "task": task,
            "complete": complete,
            "count": len(rows),
            "expected": reference["count"],
            "score": max(0, min(1, skill)) * 100 if skill is not None else None,
        }
    tasks = {}
    for task in ["noul", "choice", "score"]:
        benchmarks = [row for row in results.values() if row["task"] == task]
        eligible = [row["score"] for row in benchmarks if row["score"] is not None]
        tasks[task] = (
            statistics.mean(eligible)
            if eligible and all(row["complete"] for row in benchmarks)
            else None
        )
    measured_entries = [
        entry
        for entry, probabilities in zip(entries, predictions, strict=True)
        if probabilities is not None
    ]
    return {
        "tasks": tasks,
        "avg": statistics.mean(tasks.values())
        if all(value is not None for value in tasks.values())
        else None,
        "benchmarks_total": len(references),
        "benchmarks_complete": sum(row["complete"] for row in results.values()),
        "benchmarks_processed": sum(row["count"] > 0 for row in results.values()),
        "unique_questions": len(measured_entries),
        "unique_cases": len({(entry["source"], entry["case_id"]) for entry in measured_entries}),
        "datasets": len({entry["source"] for entry in measured_entries}),
        "benchmarks": results,
        "protocol": "baseline-adjusted-v1 / 0–100 / equal benchmark weights within task, "
        "equal task weights / first measured pass only / incomplete tasks N/A",
    }


def batches_for(prepared, batch_size, budget):
    batch = []
    for index in sorted(
        range(len(prepared)),
        key=lambda index: (
            prepared[index].task,
            len(prepared[index].documents),
            len(prepared[index].query),
            max(map(len, prepared[index].documents)),
        ),
    ):
        proposed = batch + [index]
        groups = [prepared[position] for position in proposed]
        cost = sum(len(group.documents) for group in groups) * (
            max(len(group.query) for group in groups)
            + max(len(document) for group in groups for document in group.documents)
        )
        if batch and (len(proposed) > batch_size or cost > budget):
            yield batch
            batch = []
        batch.append(index)
    if batch:
        yield batch


def token_counts(groups):
    logical = sum(len(group.query) + sum(map(len, group.documents)) for group in groups)
    prefixes = {group.key: group.query for group in groups}
    encoded = sum(map(len, prefixes.values())) + sum(
        sum(map(len, group.documents)) for group in groups
    )
    return logical, encoded


def execution_batches(prepared, batch_size, budget, seed):
    batches = list(batches_for(prepared, batch_size, budget))
    for batch in batches:
        batch.sort(
            key=lambda index: (
                len(prepared[index].query),
                max(map(len, prepared[index].documents)),
            ),
            reverse=True,
        )
    random.Random(seed).shuffle(batches)
    return batches


def prepare_execution(count, repeats, infer, synchronize, update):
    first_output = None
    latencies = []
    for position in range(count):
        for repeat in range(repeats):
            update(position, repeat)
            synchronize()
            started = time.perf_counter()
            output = infer(position)
            synchronize()
            duration = (time.perf_counter() - started) * 1000
            if position == 0:
                first_output = output
        latencies.append(duration)
    return first_output, latencies


def screen_text(value, style="", *, single_line=False):
    from rich.text import Text

    cleaned = "".join(
        character
        for character in str(value)
        if character == "\n" or not unicodedata.category(character).startswith("C")
    )
    if single_line:
        cleaned = " ".join(cleaned.split())
    return Text(cleaned, style=style, no_wrap=single_line, overflow="ellipsis")


def state_text(value):
    from rich.text import Text

    rendered = Text()
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))

    def append_value(item, depth=0):
        indent = "  " * depth
        if isinstance(item, dict) and item:
            for position, (key, child) in enumerate(item.items()):
                if position:
                    rendered.append("\n")
                rendered.append(screen_text(f"{indent}{key}", "bold cyan", single_line=True))
                rendered.append("\n")
                append_value(child, depth + 1)
        elif isinstance(item, list) and item:
            for child in item:
                rendered.append(f"{indent}•\n", style="cyan")
                append_value(child, depth + 1)
        else:
            text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for line in text.split("\n"):
                rendered.append(screen_text(indent + line))
                rendered.append("\n")

    append_value(value)
    rendered.rstrip()
    return rendered


def large_number(value, precision=0):
    from rich.text import Text

    glyphs = {
        "0": ("┏━┓", "┃ ┃", "┗━┛"),
        "1": (" ╻ ", " ┃ ", " ╹ "),
        "2": ("╺━┓", "┏━┛", "┗━╸"),
        "3": ("╺━┓", " ━┫", "╺━┛"),
        "4": ("╻ ╻", "┗━┫", "  ╹"),
        "5": ("┏━╸", "┗━┓", "╺━┛"),
        "6": ("┏━╸", "┣━┓", "┗━┛"),
        "7": ("╺━┓", "  ┃", "  ╹"),
        "8": ("┏━┓", "┣━┫", "┗━┛"),
        "9": ("┏━┓", "┗━┫", "╺━┛"),
        ",": (" ", " ", "╹"),
        ".": (" ", " ", "●"),
    }
    digits = f"{value:,.{precision}f}"
    return Text(
        "\n".join(" ".join(glyphs[digit][row] for digit in digits) for row in range(3)),
        style="bold bright_green",
        no_wrap=True,
        overflow="ellipsis",
    )


def ready_screen(args, device_name, counts, total, tokens, warmup_count=1):
    from rich.console import Group
    from rich.panel import Panel

    return Panel(
        Group(
            screen_text("bekko-system-one Decision Models — Speed Demo", "bold bright_cyan"),
            screen_text(f"bekko-system-one-v0-{args.model}", "bold bright_cyan", single_line=True),
            screen_text(device_name, "bold", single_line=True),
            screen_text(
                f"SDPA / {compile_label(args)}"
                f" / batch ≤ {args.batch_size} / context ≤ {args.input_max}",
                single_line=True,
            ),
            screen_text(
                f"Input limits: query {getattr(args, 'query_max', 'native')} / "
                f"candidate {getattr(args, 'candidate_max', 'native')} tokens",
                single_line=True,
            ),
            screen_text(f"TOTAL QUESTIONS (unique): {sum(counts.values()):,}", "bold"),
            screen_text(
                f"Noul: {counts['noul']:,} questions  |  Choice: {counts['choice']:,} questions"
                f"  |  Score: {counts['score']:,} questions"
            ),
            screen_text(f"TOTAL TO PROCESS: {total:,} questions / {args.passes} replay passes"),
            screen_text(
                f"Noul: {counts['noul'] * args.passes:,}  |  "
                f"Choice: {counts['choice'] * args.passes:,}  |  "
                f"Score: {counts['score'] * args.passes:,}  |  Input tokens: {tokens:,}"
            ),
            screen_text(
                f"Inputs prepared · Warmup: {warmup_count} batches · No cached answers", "dim"
            ),
            screen_text(
                "New shapes use eager fallback unless --allow-recompile or zero warmup is selected.",
                "dim",
            ),
            screen_text(
                f"Batch order: shuffled / seed {getattr(args, 'seed', 42)} / same order each pass"
            ),
        ),
        title="BEKKO / SYSTEM ONE / DECISION — READY",
        border_style="cyan",
    )


def display(args, device_name, counts, total, stats, example=None, status="LIVE INFERENCE"):
    from rich.console import Group
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table

    elapsed = max(stats["wall_seconds"], 1e-9)
    inference = max(stats["inference_seconds"], 1e-9)
    done = stats["completed"]
    rate = stats.get("rolling_tokens", stats["tokens"] / inference)
    decision_rate = stats.get("rolling_decisions", done / inference)
    header = Panel(
        screen_text(
            f"BEKKO / SYSTEM ONE / DECISION   |   bekko-system-one-v0-{args.model}   |   {device_name}"
            f"   |   {status}",
            "bold cyan",
            single_line=True,
        ),
        border_style="cyan",
    )
    speed = Panel(
        Group(
            screen_text("INPUT TOKENS / SECOND", "bold bright_green", single_line=True),
            large_number(rate),
            screen_text(
                f"{rate:,.0f} input tok/s  •  {decision_rate:,.0f} decisions/s",
                "bold bright_green",
                single_line=True,
            ),
            screen_text(f"Wall incl. UI: {stats['tokens'] / elapsed:,.0f} tok/s", single_line=True),
            screen_text(
                "Last 5s inference average · updated every 1s"
                if "rolling_tokens" in stats
                else stats.get("activity", "Cumulative inference rate"),
                "dim",
                single_line=True,
            ),
        ),
        title="THROUGHPUT",
        border_style="green",
    )
    completed = stats.get("completed_tasks", {})
    progress = Panel(
        Group(
            screen_text(
                f"{done / total:8.3%}   {done:,} / {total:,} decisions",
                "bold bright_cyan",
                single_line=True,
            ),
            screen_text(
                "━" * int(done / total * 40) + "░" * (40 - int(done / total * 40)),
                "bright_cyan",
                single_line=True,
            ),
            screen_text(
                f"INPUT TOKENS  {stats['tokens']:,} / {stats.get('total_tokens', 0):,}",
                "bold",
                single_line=True,
            ),
            screen_text(
                f"Choice {completed.get('choice', 0):,} / {counts['choice'] * args.passes:,}",
                "bold cyan",
                single_line=True,
            ),
            screen_text(
                f"Noul   {completed.get('noul', 0):,} / {counts['noul'] * args.passes:,}",
                "bold cyan",
                single_line=True,
            ),
            screen_text(
                f"Score  {completed.get('score', 0):,} / {counts['score'] * args.passes:,}",
                "bold cyan",
                single_line=True,
            ),
            screen_text(
                f"Elapsed {elapsed:8.2f}s   Inference {stats['inference_seconds']:8.2f}s",
                single_line=True,
            ),
            screen_text(
                f"Batch {stats['last_ms']:.2f} ms / {stats['last_size']} decisions | {args.passes} passes",
                single_line=True,
            ),
        ),
        title="OVERALL PROGRESS",
        border_style="cyan",
    )
    instruction = screen_text("Waiting for the first completed batch…")
    state = screen_text("")
    winner_text = screen_text("Waiting for model probabilities…")
    probabilities_table = Table.grid(expand=True, padding=(0, 1))
    probabilities_table.add_column(width=20, no_wrap=True, overflow="crop")
    probabilities_table.add_column(width=8, no_wrap=True)
    probabilities_table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    details = screen_text("Real model inference • no cached answers")
    source = "INPUT / CURRENT BATCH"
    if example:
        entry, probabilities, group = example
        decision = entry["input"]["decisions"][0]
        source = f"INPUT / LAST COMPLETED BATCH · {decision['type'].upper()}"
        instruction = screen_text(json.loads(decision["instructions_json"]))
        state = state_text(json.loads(entry["input"]["state_json"]))
        ranked = sorted(
            zip(decision["criteria"], probabilities, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        winner, probability = ranked[0]
        winner_text = Group(
            screen_text(winner["id"], "bold bright_green", single_line=True),
            screen_text(f"P(selected) = {probability:.2%}", "bold bright_green", single_line=True),
            screen_text(json.loads(winner["description_json"])),
        )
        if decision["type"] == "score":
            expected_score = sum(candidate["value"] * value for candidate, value in ranked)
            winner_text = Group(
                screen_text(
                    f"EXPECTED SCORE = {expected_score:.4f}", "bold bright_green", single_line=True
                ),
                screen_text(
                    f"Top level: {winner['value']} / {winner['id']}  P = {probability:.2%}",
                    single_line=True,
                ),
                screen_text(json.loads(winner["description_json"])),
            )
        for candidate, value in ranked[:5]:
            probabilities_table.add_row(
                screen_text("█" * round(value * 20) + "░" * (20 - round(value * 20)), "green"),
                screen_text(f"{value:7.2%}", "bold", single_line=True),
                screen_text(
                    f"{candidate['value']} / {candidate['id']}"
                    if decision["type"] == "score"
                    else candidate["id"],
                    single_line=True,
                ),
            )
        details = Group(
            screen_text(f"Source: {entry['source']}", single_line=True),
            screen_text(
                f"Query: {len(group.query):,} tokens  |  Candidates: "
                f"{sum(map(len, group.documents)):,} tokens",
                single_line=True,
            ),
            screen_text(
                f"{len(group.documents)} alternatives / softmax Σp = {sum(probabilities):.6f}",
                single_line=True,
            ),
            screen_text("Shared prefix reused / CPU → model → CPU probabilities", single_line=True),
            screen_text(
                "One sample from latest batch • probability is NOT accuracy",
                "dim",
                single_line=True,
            ),
        )
    inputs = Layout(name="inputs")
    inputs.split_column(
        Layout(Panel(instruction, title="INSTRUCTION", border_style="cyan"), size=8),
        Layout(
            Panel(state, title="STATE / original text · token limits apply", border_style="cyan")
        ),
    )
    outputs = Layout(name="outputs")
    outputs.split_column(
        Layout(Panel(winner_text, title="PREDICTED ANSWER", border_style="green"), size=8),
        Layout(
            Panel(probabilities_table, title="PROBABILITIES / TOP 5", border_style="green"), size=7
        ),
        Layout(Panel(details, title="INFERENCE DETAILS", border_style="dim")),
    )
    body = Layout()
    body.split_row(
        Layout(Panel(inputs, title=source, border_style="cyan")),
        Layout(Panel(outputs, title="OUTPUT / TYPED DECISION", border_style="green")),
    )
    if status == "COMPLETE" and "evaluation" in stats:
        body = results_screen(stats["evaluation"], counts, stats, args.passes)
    telemetry = Layout()
    telemetry.split_row(Layout(progress), Layout(speed))
    footer = Group(
        screen_text(
            f"SDPA / {compile_label(args)} / BS ≤ {args.batch_size}"
            f" / context ≤ {args.input_max} / native adaptive truncation",
            single_line=True,
        ),
        screen_text(
            "Pretokenized input tokens. Query once/decision + candidates;"
            " reuse included, padding excluded.",
            single_line=True,
        ),
        screen_text(
            "Timing: transfer + model + probabilities. Load/tokenize/warmup excluded."
            + (
                "  Complete. Esc / Enter to exit."
                if status == "COMPLETE"
                else "  Stopped before all passes completed."
                if status == "STOPPED"
                else "  Esc to stop (after current batch)."
            ),
            single_line=True,
        ),
    )
    layout = Layout()
    layout.split_column(
        Layout(header, size=3),
        Layout(telemetry, size=10),
        body,
        Layout(Panel(footer, border_style="dim"), size=5),
    )
    return layout


def results_screen(evaluation, counts, stats, passes):
    from rich.console import Group
    from rich.layout import Layout
    from rich.panel import Panel

    wall = max(stats["wall_seconds"], 1e-9)
    inference = max(stats["inference_seconds"], 1e-9)
    speed = Layout()
    speed.split_row(
        Layout(
            Panel(
                Group(
                    large_number(stats["completed"] / wall),
                    screen_text(
                        "decisions / second · wall time including UI", "bold", single_line=True
                    ),
                    screen_text(
                        f"Inference only: {stats['completed'] / inference:,.0f} decisions/s",
                        "dim",
                        single_line=True,
                    ),
                ),
                title="DECISION THROUGHPUT",
                border_style="green",
            )
        ),
        Layout(
            Panel(
                Group(
                    large_number(stats["completed"]),
                    screen_text(
                        f"decisions executed in {wall:.2f} seconds", "bold", single_line=True
                    ),
                    screen_text("Total work includes replay passes", "dim", single_line=True),
                ),
                title="TOTAL WORK COMPLETED",
                border_style="cyan",
            )
        ),
    )
    questions_per_pass = sum(counts.values())
    completed_passes, partial_questions = divmod(stats["completed"], questions_per_pass)
    loop_summary = f"{completed_passes:,} / {passes:,} LOOPS COMPLETED"
    if partial_questions:
        loop_summary += (
            f"  •  Loop {completed_passes + 1:,}: "
            f"{partial_questions:,} / {questions_per_pass:,} questions"
        )
    scope = Group(
        screen_text(loop_summary, "bold bright_cyan"),
        screen_text(
            f"{questions_per_pass:,} questions per loop • "
            "Each loop reruns the same selected inputs",
            "dim",
        ),
        screen_text(
            f"{evaluation['benchmarks_complete']:,} / {evaluation['benchmarks_total']:,}"
            " BENCHMARKS COMPLETED",
            "bold bright_cyan",
        ),
        screen_text(
            f"{evaluation['unique_questions']:,} UNIQUE QUESTIONS  •  "
            f"{evaluation['unique_cases']:,} CASES  •  {evaluation['datasets']:,} DATASETS",
            "bold",
        ),
        screen_text(
            f"Noul {counts['noul']:,}  |  Choice {counts['choice']:,}  |  Score {counts['score']:,}"
        ),
        screen_text(f"{stats['completed']:,} TOTAL DECISIONS EXECUTED (including replay passes)"),
        screen_text(
            f"{stats['tokens']:,} input tokens processed • "
            f"{stats['tokens'] / wall:,.0f} input tok/s including UI"
        ),
        screen_text(
            "Measured model inference, not cached answers. Initialization excluded.", "dim"
        ),
    )
    values = [("AVG", evaluation["avg"])] + [
        (task.upper(), evaluation["tasks"][task]) for task in ["noul", "choice", "score"]
    ]
    score_line = "   |   ".join(
        f"{label} {value:.2f}" if value is not None else f"{label} N/A" for label, value in values
    )
    quality = Panel(
        Group(
            screen_text(score_line, single_line=True),
            screen_text(
                "Baseline-adjusted / 100; not raw accuracy. First pass only; incomplete tasks N/A.",
                "dim",
                single_line=True,
            ),
        ),
        title="BENCHMARK QUALITY / REFERENCE",
        border_style="dim",
    )
    layout = Layout()
    layout.split_column(
        Layout(speed, size=7),
        Layout(Panel(scope, title="S1MB / RUN RESULTS", border_style="cyan")),
        Layout(quality, size=4),
    )
    return layout


def main():
    args = arguments()
    trace = TraceLog(args.verbose_log)
    trace.event("initialization_begin", model=args.model, seed=args.seed)
    from rich.console import Console
    from rich.live import Live

    console = Console()
    if args.verbose_log:
        console.print(f"Diagnostic log: {args.verbose_log}")
    console.print(
        "[bold cyan]BEKKO / SYSTEM ONE / DECISION — Initializing · Loading dependencies…[/]"
    )
    import torch
    import torch._inductor.config as inductor_config
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    inductor_config.triton.cudagraph_dynamic_shape_warn_limit = None
    warnings.filterwarnings("ignore", message="TensorFloat32 tensor cores for float32.*")
    disable_progress_bars()
    torch.set_num_threads(args.threads)

    def compiler_counts():
        if not args.verbose_log:
            return {}
        from torch._dynamo.utils import counters

        return {name: dict(counters[name]) for name in ("frames", "stats", "inductor")}

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Explicitly use --device cpu for CPU inference.")
        torch.cuda.set_device(device)
        free, _ = torch.cuda.mem_get_info(device)
        console.print(f"GPU {device}: {free / 2**30:.1f} GiB free")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU / FP32"
    with console.status("Initializing / Loading S1MB active test inputs…"):
        if args.data_dir:
            root = args.data_dir.resolve()
            provenance = json.loads((root / "hub-source.json").read_text())
        else:
            revision = (
                HfApi().dataset_info("hotchpotch/s1mb-dataset", revision=args.dataset_revision).sha
            )
            manifest_path = hf_hub_download(
                "hotchpotch/s1mb-dataset",
                "training-manifest.json",
                repo_type="dataset",
                revision=revision,
            )
            active = json.loads(Path(manifest_path).read_text())["evaluation"]
            files = [
                filename
                for subset in active
                if subset["split"] == "test"
                for filename in subset["data_files"]
            ]
            root = Path(
                snapshot_download(
                    "hotchpotch/s1mb-dataset",
                    repo_type="dataset",
                    revision=revision,
                    allow_patterns=["training-manifest.json", *files],
                )
            )
            provenance = {"repo_id": "hotchpotch/s1mb-dataset", "revision": revision}
        entries = select_rows(root, args.limit, args.seed)
    trace.event("dataset_loaded", decisions=len(entries))
    repo = f"hotchpotch/bekko-system-one-v0-{args.model}"
    revision = REVISIONS[args.model]
    with console.status("Initializing / Loading pinned Bekko model…"):
        model_class = get_class_from_dynamic_module(
            "inference_v0.BekkoSentenceTransformer",
            repo,
            revision=revision,
            code_revision=revision,
        )
        model = model_class(
            repo,
            revision=revision,
            device=args.device,
            trust_remote_code=True,
            attn_implementation="sdpa",
        ).eval()
        runtime = model[0]
        resolve_input_limits(args, runtime)
        native = importlib.import_module(model_class.__module__)
    trace.event("model_loaded", device=device_name)
    with console.status("Initializing / Rendering, tokenizing and packing inputs…"):
        groups = [
            native.input_groups(entry["input"], prefix_layout=runtime.prefix_layout)[0]
            for entry in entries
        ]
        prepared = []
        for start in range(0, len(groups), 128):
            prepared.extend(
                runtime.prepare_groups(
                    groups[start : start + 128],
                    context_length=args.input_max,
                    query_length=min(args.query_max, args.input_max - 2),
                    document_length=min(args.candidate_max, args.input_max - 3),
                )
            )
        batches = execution_batches(prepared, args.batch_size, args.token_budget, args.seed)
        payloads = [
            native.collate_groups([prepared[index] for index in batch], runtime)
            for batch in batches
        ]
        lengths = [[len(prepared[index].documents) for index in batch] for batch in batches]
        tokens = [token_counts([prepared[index] for index in batch]) for batch in batches]
    trace.event("inputs_prepared", batches=len(batches), decisions=len(entries))

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    @torch.inference_mode()
    def infer(position):
        scores = runtime(native.to_device(dict(payloads[position]), device))["scores"].flatten()
        values = torch.cat(
            [part.float().softmax(0) for part in scores.split(lengths[position])]
        ).cpu()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite model probabilities")
        return [part.tolist() for part in values.split(lengths[position])]

    warmup_count = (
        len(batches) if args.warmup_batches == "all" else min(args.warmup_batches, len(batches))
    )
    delta = None
    native_delta = None
    if warmup_count:
        with console.status("Initializing / Eager smoke check…"):
            reference = infer(0)
            native_outputs = model.predict(
                [entries[index]["input"] for index in batches[0]],
                batch_size=args.batch_size,
                token_budget=args.token_budget,
                context_length=args.input_max,
                query_length=min(args.query_max, args.input_max - 2),
                document_length=min(args.candidate_max, args.input_max - 3),
                show_progress_bar=False,
            )
            native_delta = 0.0
            for index, probabilities, result in zip(
                batches[0], reference, native_outputs, strict=True
            ):
                metadata = prepared[index].metadata
                expected = result[metadata.decision_id]["probabilities"]
                difference = max(
                    abs(value - expected[key])
                    for key, value in zip(metadata.candidate_ids, probabilities, strict=True)
                )
                native_delta = max(native_delta, difference)
            if native_delta > (0.03 if device.type == "cuda" else 1e-5):
                raise ValueError(f"Prepared inference differs from native API: {native_delta}")
    if not args.no_compile:
        model.compile_inference(mode="reduce-overhead", dynamic=True)
    trace.event("warmup_begin", batches=warmup_count, compiler=compiler_counts())
    stage = (
        "Warmup / Eager inference"
        if args.no_compile
        else "Warmup / Kernel compilation + CUDA Graph capture + replay check (CUDA only)"
    )
    warmup_repeats = 3 if not args.no_compile and device.type == "cuda" else 1
    warmup_replay_ms = []
    if warmup_count:
        with console.status(f"{stage} · 0/{warmup_count} batches") as spinner:
            output, warmup_replay_ms = prepare_execution(
                warmup_count,
                warmup_repeats,
                infer,
                synchronize,
                lambda position, repeat: spinner.update(
                    f"{stage} · batch {position + 1}/{warmup_count} · preparation {repeat + 1}/{warmup_repeats}"
                ),
            )
            delta = max(
                abs(actual - expected)
                for actual_row, expected_row in zip(output, reference, strict=True)
                for actual, expected in zip(actual_row, expected_row, strict=True)
            )
            if delta > 0.03:
                raise ValueError(f"Smoke parity failed: maximum probability delta {delta}")
    else:
        console.print(
            "Warmup skipped / No pre-run inference. "
            "Kernel compilation and CUDA Graph capture, if enabled, are timed."
        )
    counts = Counter(group.task for group in prepared)
    trace.event(
        "ready",
        warmup_batches=warmup_count,
        warmup_replay_ms=warmup_replay_ms,
        compiler=compiler_counts(),
    )
    total = len(entries) * args.passes
    use_ui = not args.no_ui and console.is_terminal and sys.stdin.isatty()
    if use_ui or args.wait:
        console.print(
            ready_screen(
                args,
                device_name,
                counts,
                total,
                sum(pair[0] for pair in tokens) * args.passes,
                warmup_count,
            )
        )
        prompt = (
            "Ready. Press Enter to start. "
            if warmup_count
            else "Inputs ready. Cold start includes compilation. Press Enter to start. "
        )
        if use_ui:
            with TerminalKeys() as keys:
                console.print(screen_text(prompt + "Esc to cancel.", "bold green"))
                if keys.wait() == "escape":
                    console.print("Cancelled before execution.")
                    return
        else:
            console.input(screen_text(prompt, "bold green"))
    trace.event("start_requested")
    stats = {
        "total_tokens": sum(pair[0] for pair in tokens) * args.passes,
        "completed": 0,
        "tokens": 0,
        "encoded_tokens": 0,
        "inference_seconds": 0.0,
        "wall_seconds": 0.0,
        "last_ms": 0.0,
        "last_size": 0,
    }
    samples = []
    first_predictions = [None] * len(entries)
    task_done = Counter()
    example = None
    last_example = None
    started = time.perf_counter()
    refreshed = started
    final_screen = None
    stopped = False
    state = {
        "stats": dict(stats),
        "started": started,
        "example": None,
        "pending": None,
        "throughput": RollingThroughput(started),
    }
    execution_stance = (
        "default"
        if args.no_compile or args.allow_recompile or warmup_count == 0
        else "eager_on_recompile"
    )
    compiler_context = (
        torch.compiler.set_stance(execution_stance) if not args.no_compile else nullcontext()
    )
    context = (
        Live(
            get_renderable=lambda: live_frame(args, device_name, counts, total, state),
            console=console,
            screen=True,
            auto_refresh=True,
            refresh_per_second=args.fps,
        )
        if use_ui
        else nullcontext()
    )
    with compiler_context, TerminalKeys(use_ui) as keys, context as live:
        trace.event("execution_begin", stance=execution_stance, compiler=compiler_counts())
        for pass_index in range(args.passes):
            for position, batch in enumerate(batches):
                if keys.poll() == "escape":
                    stopped = True
                    break
                synchronize()
                trace.event(
                    "batch_begin",
                    pass_index=pass_index,
                    batch_index=position,
                    decisions=len(batch),
                    shapes={
                        name: list(payloads[position][name].shape)
                        for name in ("prefix_ids", "doc_ids")
                    },
                    compiler=compiler_counts(),
                )
                tick = time.perf_counter()
                state["pending"] = {"started": tick, "number": position + 1, "total": len(batches)}
                results = infer(position)
                synchronize()
                duration = time.perf_counter() - tick
                trace.event(
                    "batch_end",
                    pass_index=pass_index,
                    batch_index=position,
                    duration_ms=duration * 1000,
                    compiler=compiler_counts(),
                )
                last_example = entries[batch[0]], results[0], prepared[batch[0]]
                if pass_index == 0:
                    for index, probabilities in zip(batch, results, strict=True):
                        first_predictions[index] = probabilities
                samples.append(duration * 1000)
                stats["inference_seconds"] += duration
                stats["completed"] += len(batch)
                stats["tokens"] += tokens[position][0]
                stats["encoded_tokens"] += tokens[position][1]
                stats["last_ms"] = duration * 1000
                stats["last_size"] = len(batch)
                task_done.update(prepared[index].task for index in batch)
                stats["completed_tasks"] = dict(task_done)
                now = time.perf_counter()
                stats["wall_seconds"] = now - started
                state.update(stats=dict(stats), example=last_example, pending=None)
                if use_ui and (len(samples) == 1 or now - refreshed >= 1 / args.fps):
                    index = batch[0]
                    example = entries[index], results[0], prepared[index]
                    live.refresh()
                    refreshed = time.perf_counter()
            if stopped:
                break
        stats["wall_seconds"] = time.perf_counter() - started
        state.update(stats=dict(stats), phase="COMPUTING SCORES", pending=None, clock_stopped=True)
        trace.event("execution_end", completed=stats["completed"], stopped=stopped)
        stats["evaluation"] = score_summary(entries, first_predictions)
        stats["status"] = "stopped" if stopped else "complete"
        example = last_example
        if use_ui:
            final_screen = display(
                args,
                device_name,
                counts,
                total,
                stats,
                example,
                "STOPPED" if stopped else "COMPLETE",
            )
            state["final"] = final_screen
            live.refresh()
            if not stopped:
                try:
                    keys.wait()
                except KeyboardInterrupt:
                    pass
    if final_screen is not None:
        console.print(final_screen)
    report = {
        "model": repo,
        "model_revision": revision,
        "dataset": provenance,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "device": device_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "unique_decisions": len(entries),
        "query_token_range": [
            min(len(group.query) for group in prepared),
            max(len(group.query) for group in prepared),
        ],
        "candidate_token_range": [
            min(len(document) for group in prepared for document in group.documents),
            max(len(document) for group in prepared for document in group.documents),
        ],
        "unique_tasks": dict(counts),
        "completed_tasks": dict(task_done),
        "microbatches_per_pass": len(batches),
        **stats,
        "input_tokens_per_second": stats["tokens"] / max(stats["inference_seconds"], 1e-9),
        "decisions_per_second": stats["completed"] / max(stats["inference_seconds"], 1e-9),
        "wall_input_tokens_per_second": stats["tokens"] / max(stats["wall_seconds"], 1e-9),
        "batch_p50_ms": statistics.median(samples) if samples else None,
        "batch_p95_ms": sorted(samples)[math.ceil(len(samples) * 0.95) - 1] if samples else None,
        "smoke_max_probability_delta": delta,
        "native_smoke_max_probability_delta": native_delta,
        "warmup_batches_actual": warmup_count,
        "warmup_executions_per_batch": warmup_repeats if warmup_count else 0,
        "warmup_final_execution_ms": warmup_replay_ms,
        "first_measured_batches_ms": samples[:8],
        "execution_compiler_stance": execution_stance,
        "order": "Seeded shuffle of length-packed batches; identical order for replay passes",
        "timing": "CPU prepared tensors -> device -> forward -> CPU probabilities; synchronized. "
        "Excludes loading, rendering, tokenization, collation, warmup and UI. "
        "Wall includes UI. Later shape recompilation, if any, remains timed.",
        "tokens_definition": "Post-truncation query once per decision plus candidate tokens; "
        "includes special tokens, excludes padding. encoded_tokens additionally "
        "deduplicates identical query prefixes within each microbatch. "
        "Not generated tokens. Replays are counted explicitly.",
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    if use_ui:
        return
    console.print(
        f"[bold green]COMPLETE: {total:,} decisions | "
        f"{report['decisions_per_second']:,.0f} decisions/s | "
        f"{report['input_tokens_per_second']:,.0f} input tokens/s[/]"
    )
    console.print(
        f"Inference {stats['inference_seconds']:.3f}s | Wall {stats['wall_seconds']:.3f}s | "
        f"batch p50 {report['batch_p50_ms']:.2f}ms / p95 {report['batch_p95_ms']:.2f}ms"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit("\nStopped by user.") from None
