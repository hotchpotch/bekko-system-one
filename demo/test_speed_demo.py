import importlib.util
import io
import json
import os
import pty
import sys
import termios
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow as arrow
import pyarrow.parquet as parquet
import pytest
from rich.console import Console

spec = importlib.util.spec_from_file_location(
    "bekko_speed_demo", Path(__file__).with_name("run_bekko_system_one_v0_speed.py")
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def test_rolling_throughput_updates_once_per_second_and_expires_old_work():
    throughput = demo.RollingThroughput(0)
    stats = {"tokens": 100, "completed": 10, "inference_seconds": 0.5}
    assert throughput.update(0.9, stats)["rolling_tokens"] == 0
    assert throughput.update(1, stats)["rolling_tokens"] == 200
    stats = {"tokens": 200, "completed": 20, "inference_seconds": 1}
    assert throughput.update(1.9, stats)["rolling_decisions"] == 20
    for second in range(2, 8):
        stats["inference_seconds"] = second - 1
        rates = throughput.update(second, stats)
    assert rates == {"rolling_tokens": 0, "rolling_decisions": 0}
    assert len(throughput.samples) <= 6


def test_live_header_does_not_toggle_between_batches():
    args = SimpleNamespace(model="17m", no_compile=False, passes=1, batch_size=32, input_max=7999)
    stats = dict(
        completed=0,
        tokens=0,
        total_tokens=100,
        wall_seconds=0,
        inference_seconds=0,
        last_ms=0,
        last_size=0,
    )
    state = dict(stats=stats, started=0, pending=None, throughput=demo.RollingThroughput(0))
    for pending in (None, {"started": 0, "number": 1, "total": 10}):
        state["pending"] = pending
        console = Console(width=180, height=48, file=io.StringIO(), record=True)
        console.print(demo.live_frame(args, "GPU", Counter(score=10), 10, state, now=0.1))
        output = console.export_text()
        assert "LIVE INFERENCE" in output
        assert "BATCH 1/10 RUNNING" not in output
        assert "Last 5s inference average" in output


def test_state_single_key_shows_only_text():
    assert demo.state_text({"message": "Hello\nworld"}).plain == "Hello\nworld"


def test_state_multiple_keys_have_styled_labels_and_plain_values():
    rendered = demo.state_text({"query": "[bold]literal[/bold]", "document": "Some text"})
    assert rendered.plain == "query\n  [bold]literal[/bold]\n\ndocument\n  Some text"
    assert any(span.style == "bold cyan" for span in rendered.spans)


@pytest.mark.parametrize(
    "value", [None, True, 42, [], {}, {"nested": {"score": 3}}, ["first", "second"]]
)
def test_state_handles_structured_and_scalar_values(value):
    assert demo.state_text(value).plain


def test_state_sanitizes_control_characters():
    assert "\x1b" not in demo.state_text({"key\x1b": "text\x1b", "other": 1}).plain


def test_trace_log_is_file_only_and_does_not_overwrite(tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = demo.TraceLog(path)
    trace.event("batch_begin", batch_index=1, shapes={"prefix_ids": [32, 256]})
    trace.event("batch_end", batch_index=1, duration_ms=5000)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["batch_begin", "batch_end"]
    assert rows[1]["elapsed_s"] >= rows[0]["elapsed_s"]
    with pytest.raises(FileExistsError):
        demo.TraceLog(path)


def test_inflight_frame_updates_clock_without_inventing_progress():
    args = SimpleNamespace(model="17m", no_compile=False, passes=1, batch_size=32, input_max=7999)
    stats = {
        "completed": 31,
        "tokens": 15645,
        "total_tokens": 100000,
        "wall_seconds": 0.02,
        "inference_seconds": 0.01,
        "last_ms": 12.43,
        "last_size": 31,
    }
    state = {
        "stats": stats,
        "started": 10,
        "pending": {"started": 10.02, "number": 2, "total": 100},
    }
    console = Console(width=180, height=48, file=io.StringIO(), record=True)
    console.print(demo.live_frame(args, "RTX 5090", Counter(score=100), 100, state, now=15))
    output = console.export_text()
    assert "31 / 100 decisions" in output
    assert "Elapsed 5.00s" in output
    assert "Batch 2/100" in output and "compile / inference" in output
    assert stats["wall_seconds"] == 0.02
    assert stats["completed"] == 31


def test_compile_policy_labels_and_initial_blank_frame():
    args = SimpleNamespace(model="17m", no_compile=False, passes=1, batch_size=32, input_max=7999)
    assert demo.compile_label(args) == "COMPILED + EAGER FALLBACK"
    args.allow_recompile = True
    assert "RECOMPILE ENABLED" in demo.compile_label(args)
    args.allow_recompile = False
    args.warmup_batches = 0
    assert "RECOMPILE ENABLED" in demo.compile_label(args)
    stats = {
        "completed": 0,
        "tokens": 0,
        "total_tokens": 100000,
        "wall_seconds": 0,
        "inference_seconds": 0,
        "last_ms": 0,
        "last_size": 0,
    }
    state = {"stats": stats, "started": 10, "pending": None}
    console = Console(width=180, height=48, file=io.StringIO(), record=True)
    console.print(demo.live_frame(args, "RTX 5090", Counter(score=100), 100, state, now=10))
    output = console.export_text()
    assert "0 / 100 decisions" in output
    assert "Waiting for the first completed batch" in output
    assert "EXPECTED SCORE" not in output


def test_escape_enter_arrow_keys_and_terminal_restoration():
    master, slave = pty.openpty()
    with os.fdopen(slave) as stream:
        saved = termios.tcgetattr(stream.fileno())
        try:
            os.write(master, b"\x1b")
            with demo.TerminalKeys(stream=stream) as keys:
                assert keys.poll(0.2) == "escape"
                assert not termios.tcgetattr(stream.fileno())[3] & termios.ICANON
                os.write(master, b"\x1b[A")
                assert keys.poll(0.2) is None
                os.write(master, b"\x1b")
                assert keys.poll(0.2) == "escape"
                os.write(master, b"\n")
                assert keys.wait() == "enter"
            assert termios.tcgetattr(stream.fileno()) == saved
            with pytest.raises(ValueError):
                with demo.TerminalKeys(stream=stream):
                    raise ValueError("test cleanup")
            assert termios.tcgetattr(stream.fileno()) == saved
        finally:
            os.close(master)


def test_no_predictions_are_not_counted_as_completed():
    entries, _ = scoring_examples()
    result = demo.score_summary(entries, [None] * len(entries))
    assert result["unique_questions"] == result["benchmarks_complete"] == 0
    assert result["avg"] is None


@pytest.mark.parametrize("value,expected", [("0", 0), ("1", 1), ("12", 12), ("all", "all")])
def test_warmup_batch_options(monkeypatch, value, expected):
    monkeypatch.setattr(sys, "argv", ["demo", "--warmup-batches", value])
    assert demo.arguments().warmup_batches == expected


def test_warmup_defaults_to_sixteen(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo"])
    assert demo.arguments().warmup_batches == 16


def test_default_input_limits_match_native_s1mb(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["demo"])
    args = demo.arguments()
    assert (args.input_max, args.query_max, args.candidate_max) == (None, None, None)
    runtime = SimpleNamespace(context_length=7999, query_length=7997, document_length=3800)
    demo.resolve_input_limits(args, runtime)
    assert (args.input_max, args.query_max, args.candidate_max) == (7999, 7997, 3800)
    args.input_max, args.query_max, args.candidate_max = 512, None, None
    demo.resolve_input_limits(args, runtime)
    assert (args.input_max, args.query_max, args.candidate_max) == (512, 510, 509)


@pytest.mark.parametrize("value", ["-1", "invalid", "1.5"])
def test_invalid_warmup_count(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["demo", "--warmup-batches", value])
    with pytest.raises(SystemExit):
        demo.arguments()


def group(query, documents, task="choice"):
    return SimpleNamespace(query=query, documents=documents, key=tuple(query), task=task)


def scoring_examples():
    entries, predictions = [], []
    for task, targets, outputs in [
        ("choice", [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ("noul", [[0.2, 0.8], [0.8, 0.2]], [[0, 1], [1, 0]]),
        ("score", [[1, 0], [0, 1]], [[1, 0], [0.5, 0.5]]),
    ]:
        for index, (target, output) in enumerate(zip(targets, outputs, strict=True)):
            entries.append(
                {
                    "source": "fixture",
                    "case_id": f"{task}-{index}",
                    "target": target,
                    "input": {
                        "decisions": [
                            {
                                "type": task,
                                "criteria": [
                                    {"id": "false", "value": 2},
                                    {"id": "true", "value": 8},
                                ],
                            }
                        ]
                    },
                }
            )
            predictions.append(output)
    references = demo.score_references(entries)
    for entry in entries:
        entry["reference_scope"] = references
    return entries, predictions


def test_official_task_aggregation_and_incomplete_coverage():
    entries, predictions = scoring_examples()
    summary = demo.score_summary(entries, predictions)
    assert summary["tasks"] == pytest.approx({"choice": 100, "noul": 60, "score": 50})
    assert summary["avg"] == pytest.approx(70)
    assert summary["benchmarks_complete"] == 3
    partial = demo.score_summary(entries[:-1], predictions[:-1])
    assert partial["avg"] is None
    assert partial["tasks"]["score"] is None
    assert partial["benchmarks_complete"] == 2
    assert entries[-1]["reference_scope"]["fixture/score"]["baseline"] == 0.5


def test_benchmark_weights_are_equal_not_question_weighted():
    entries, predictions = scoring_examples()
    for index in range(20):
        entries.append(
            {
                "source": "larger_choice",
                "case_id": str(index),
                "target": [1, 0] if index % 2 else [0, 1],
                "input": entries[0]["input"],
            }
        )
        predictions.append([1, 0])
    references = demo.score_references(entries)
    for entry in entries:
        entry["reference_scope"] = references
    result = demo.score_summary(entries, predictions)
    assert result["tasks"]["choice"] == pytest.approx(50)
    assert result["avg"] == pytest.approx((50 + 60 + 50) / 3)


def test_completed_screen_replaces_inputs_and_outputs():
    entries, predictions = scoring_examples()
    evaluation = demo.score_summary(entries, predictions)
    args = SimpleNamespace(model="17m", no_compile=False, passes=2, batch_size=32, input_max=512)
    stats = {
        "completed": 12,
        "tokens": 1000,
        "total_tokens": 1000,
        "wall_seconds": 1,
        "inference_seconds": 0.5,
        "last_ms": 3,
        "last_size": 6,
        "evaluation": evaluation,
    }
    console = Console(width=180, height=48, record=True, file=io.StringIO())
    console.print(
        demo.display(
            args, "RTX 5090", Counter(choice=2, noul=2, score=2), 12, stats, status="COMPLETE"
        )
    )
    output = console.export_text()
    assert "S1MB / RUN RESULTS" in output
    assert "DECISION THROUGHPUT" in output
    assert "TOTAL WORK COMPLETED" in output
    assert "BENCHMARK QUALITY / REFERENCE" in output
    assert "3 / 3 BENCHMARKS COMPLETED" in output
    assert "6 UNIQUE QUESTIONS" in output
    assert "12 TOTAL DECISIONS EXECUTED" in output
    assert "2 / 2 LOOPS COMPLETED" in output
    assert "6 questions per loop" in output
    assert "INPUT / CURRENT BATCH" not in output
    assert "OUTPUT / TYPED DECISION" not in output
    assert "generated tokens" not in output
    for label in ["AVG", "NOUL", "CHOICE", "SCORE"]:
        assert label in output


@pytest.mark.parametrize(
    "completed,expected",
    [
        (0, "0 / 3 LOOPS COMPLETED"),
        (6, "1 / 3 LOOPS COMPLETED"),
        (8, "1 / 3 LOOPS COMPLETED  •  Loop 2: 2 / 6 questions"),
    ],
)
def test_results_show_completed_and_partial_loops(completed, expected):
    entries, predictions = scoring_examples()
    stats = dict(completed=completed, tokens=100, wall_seconds=1, inference_seconds=0.5)
    console = Console(width=180, height=40, record=True, file=io.StringIO())
    console.print(
        demo.results_screen(
            demo.score_summary(entries, predictions), Counter(choice=2, noul=2, score=2), stats, 3
        )
    )
    assert expected in console.export_text()


def test_token_counts_include_prefix_once_and_exclude_padding():
    groups = [group([1, 2, 3], [[4, 5], [6]]), group([1, 2, 3], [[7]])]
    assert demo.token_counts(groups) == (10, 7)


def test_batch_limits_and_oversized_single_decision():
    prepared = [group([1] * 10, [[2] * 10] * 2) for _ in range(5)]
    batches = list(demo.batches_for(prepared, 2, 80))
    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert list(demo.batches_for(prepared, 32, 1)) == [[0], [1], [2], [3], [4]]
    assert sorted(index for batch in batches for index in batch) == list(range(5))


def test_seed_controls_batch_order_without_changing_membership():
    prepared = [group([1] * (index + 1), [[2], [3]]) for index in range(20)]
    first = demo.execution_batches(prepared, 2, 64000, 42)
    assert first == demo.execution_batches(prepared, 2, 64000, 42)
    second = demo.execution_batches(prepared, 2, 64000, 43)
    assert first != second
    assert sorted(index for batch in first for index in batch) == list(range(20))
    assert sorted(map(tuple, first)) == sorted(map(tuple, second))
    for batch in first:
        lengths = [len(prepared[index].query) for index in batch]
        assert lengths == sorted(lengths, reverse=True)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_graph_preparation_repeats_only_selected_batches(count):
    calls, synchronized, progress = [], [], []

    def infer(position):
        calls.append(position)
        return [[position]]

    output, latencies = demo.prepare_execution(
        count,
        3,
        infer,
        lambda: synchronized.append(True),
        lambda position, repeat: progress.append((position, repeat)),
    )
    assert calls == [position for position in range(count) for _ in range(3)]
    assert len(synchronized) == count * 6
    assert len(progress) == count * 3
    assert len(latencies) == count
    assert output == ([[0]] if count else None)


def test_active_manifest_only_and_no_targets(tmp_path):
    row = {
        "case_id": "example",
        "split": "test",
        "targets": [],
        "input": {
            "state_json": '{"message":"hello"}',
            "decisions": [
                {"id": "one", "kind": "judgment", "type": "choice"},
                {"id": "two", "kind": "judgment", "type": "noul"},
                {"id": "three", "kind": "judgment", "type": "score"},
            ],
        },
    }
    for decision in row["input"]["decisions"]:
        decision["criteria"] = [{"id": "false", "value": 0}, {"id": "true", "value": 10}]
        row["targets"].append(
            {
                "decision_id": decision["id"],
                "kind": "judgment_distribution",
                "ids": ["true", "false"],
                "probabilities": [0.6, 0.4],
            }
        )
    parquet.write_table(arrow.Table.from_pylist([row]), tmp_path / "active.parquet")
    parquet.write_table(arrow.Table.from_pylist([row]), tmp_path / "inactive.parquet")
    manifest = {
        "evaluation": [
            {"dataset": "fixture", "split": "test", "cases": 1, "data_files": ["active.parquet"]}
        ]
    }
    (tmp_path / "training-manifest.json").write_text(json.dumps(manifest))
    entries = demo.select_rows(tmp_path, None, 42)
    assert len(entries) == 3
    assert {entry["input"]["decisions"][0]["type"] for entry in entries} == {
        "choice",
        "noul",
        "score",
    }
    assert all(set(entry["input"]) == {"state_json", "decisions"} for entry in entries)
    assert all("targets" not in entry for entry in entries)
    assert all(entry["target"] == [0.4, 0.6] for entry in entries)
    assert len(demo.select_rows(tmp_path, 1, 42)) == 1
    manifest["evaluation"][0]["cases"] = 2
    (tmp_path / "training-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Incomplete"):
        demo.select_rows(tmp_path, None, 42)


def test_wide_terminal_render():
    args = SimpleNamespace(model="17m", no_compile=False, passes=3, batch_size=32, input_max=512)
    stats = {
        "completed": 32,
        "tokens": 1234,
        "wall_seconds": 0.1,
        "inference_seconds": 0.01,
        "last_ms": 10,
        "last_size": 32,
    }
    console = Console(width=180, height=48, record=True)
    console.print(demo.display(args, "RTX 5090", Counter(choice=32, noul=32), 192, stats))
    text = console.export_text()
    assert "123,400" in text
    assert "INPUT / CURRENT BATCH" in text
    assert "OUTPUT / TYPED DECISION" in text
    assert "NOT generated tokens" not in text
    lines = text.splitlines()
    task_lines = [
        next(index for index, line in enumerate(lines) if label in line)
        for label in ["Choice 0 /", "Noul 0 /", "Score 0 /"]
    ]
    assert task_lines == list(range(task_lines[0], task_lines[0] + 3))


def test_score_displays_expected_value_on_authored_scale():
    args = SimpleNamespace(model="17m", no_compile=False, passes=1, batch_size=32, input_max=512)
    stats = {
        "completed": 1,
        "tokens": 4,
        "total_tokens": 4,
        "wall_seconds": 1,
        "inference_seconds": 0.5,
        "last_ms": 3,
        "last_size": 1,
    }
    entry = {
        "source": "fixture",
        "input": {
            "state_json": "{}",
            "decisions": [
                {
                    "type": "score",
                    "instructions_json": '"Rate this"',
                    "criteria": [
                        {"id": "low", "value": 2, "description_json": '"Low"'},
                        {"id": "high", "value": 8, "description_json": '"High"'},
                    ],
                }
            ],
        },
    }
    console = Console(width=180, height=48, file=io.StringIO(), record=True)
    console.print(
        demo.display(
            args,
            "RTX 5090",
            Counter(score=1),
            1,
            stats,
            (entry, [0.75, 0.25], group([1, 2], [[3], [4]], "score")),
        )
    )
    output = console.export_text()
    assert "EXPECTED SCORE = 3.5000" in output
    assert "Top level: 2 / low" in output
    assert "8 / high" in output


@pytest.mark.parametrize("width,height", [(160, 44), (180, 48), (120, 40)])
def test_sections_do_not_move_with_content(width, height):
    args = SimpleNamespace(
        model="17m", no_compile=False, passes=12, batch_size=32, input_max=512, wait=True
    )
    stats = {
        "completed": 123456,
        "tokens": 12345678,
        "total_tokens": 90000000,
        "wall_seconds": 5.2,
        "inference_seconds": 4.0,
        "last_ms": 2.8,
        "last_size": 32,
    }
    anchors = [
        "THROUGHPUT",
        "OVERALL PROGRESS",
        "INSTRUCTION",
        "STATE /",
        "PREDICTED ANSWER",
        "PROBABILITIES / TOP 5",
        "INFERENCE DETAILS",
    ]
    positions = []
    for content in ["Short input", "Very long line " * 100 + "\nMore lines" * 100]:
        entry = {
            "source": "fixture",
            "input": {
                "state_json": json.dumps({"text": content}),
                "decisions": [
                    {
                        "type": "choice",
                        "instructions_json": json.dumps(content),
                        "criteria": [
                            {"id": content, "description_json": json.dumps(content)},
                            {"id": "second", "description_json": '"Alternative"'},
                        ],
                    }
                ],
            },
        }
        console = Console(width=width, height=height, file=io.StringIO(), record=True)
        console.print(
            demo.display(
                args,
                "NVIDIA GeForce RTX 5090",
                Counter(choice=32, noul=32),
                200000,
                stats,
                (entry, [0.75, 0.25], group([1, 2], [[3], [4]])),
            )
        )
        lines = console.export_text().splitlines()
        positions.append(
            [next(index for index, line in enumerate(lines) if label in line) for label in anchors]
        )
        assert len(lines) == height
        assert any("75.00%" in line for line in lines)
        assert any("Σp = 1.000000" in line for line in lines)
    assert positions[0] == positions[1]


def test_ready_screen_and_control_characters():
    args = SimpleNamespace(model="17m", no_compile=False, passes=12, batch_size=32, input_max=512)
    console = Console(width=160, file=io.StringIO(), record=True)
    console.print(demo.ready_screen(args, "RTX 5090", Counter(choice=32, noul=32), 768, 123456))
    output = console.export_text()
    assert "bekko-system-one-v0-17m" in output
    assert "RTX 5090" in output
    assert "123,456" in output
    assert "bekko-system-one Decision Models — Speed Demo" in output
    assert "TOTAL QUESTIONS (unique): 64" in output
    assert "Noul: 32 questions" in output
    assert "Choice: 32 questions" in output
    assert "TOTAL TO PROCESS: 768 questions" in output
    assert "Noul: 384" in output
    assert "Choice: 384" in output
    assert "Score: 0 questions" in output
    assert demo.screen_text("hello\r\nworld\x1b\u202e", single_line=True).plain == "hello world"


def test_progress_rows_stay_fixed_from_zero_to_complete_without_color():
    args = SimpleNamespace(model="17m", no_compile=False, passes=1, batch_size=32, input_max=512)
    positions = []
    for completed in [0, 1, 64]:
        stats = {
            "completed": completed,
            "tokens": completed * 100,
            "total_tokens": 6400,
            "wall_seconds": 1,
            "inference_seconds": 0.5,
            "last_ms": 3,
            "last_size": 32,
        }
        console = Console(width=160, height=44, file=io.StringIO(), record=True, no_color=True)
        console.print(demo.display(args, "RTX 5090", Counter(choice=32, noul=32), 64, stats))
        lines = console.export_text().splitlines()
        positions.append(
            [
                next(index for index, line in enumerate(lines) if label in line)
                for label in ["INPUT TOKENS 0", "Elapsed", "Batch"]
            ]
            if not completed
            else [
                next(index for index, line in enumerate(lines) if label in line)
                for label in [f"INPUT TOKENS {completed * 100:,}", "Elapsed", "Batch"]
            ]
        )
    assert positions[0] == positions[1] == positions[2]
