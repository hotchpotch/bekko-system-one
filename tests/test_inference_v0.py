"""Portable runtime parity, typed rendering and remote-code checkpoint loading."""

import copy
import importlib.util
import sys

import pytest
import torch
from sentence_transformers import SentenceTransformer
from test_dataset_training import native_case

from bekko_system_one import BekkoSentenceTransformer
from bekko_system_one.data import Group
from bekko_system_one.export_v0 import export_model, portable_module, runtime_source
from bekko_system_one.inference_v0 import input_groups
from bekko_system_one.model import predict
from bekko_system_one.modules import DecisionHeads
from bekko_system_one.query_budget import QueryParts
from bekko_system_one.release import render_input_group


def test_native_renderer_matches_training_for_both_layouts():
    row = native_case()
    for layout in ("instruction_state", "state_instruction"):
        groups = input_groups(row["input"], prefix_layout=layout)
        for i, actual in enumerate(groups):
            expected = render_input_group(row["input"], i, prefix_layout=layout)
            assert actual.query == expected.query
            assert actual.query_parts == expected.query_parts
            assert actual.candidates == expected.candidates
            assert actual.task == expected.task
            assert actual.metadata.candidate_ids == expected.metadata.candidate_ids
            assert actual.metadata.candidate_values == expected.metadata.candidate_values
    with pytest.raises(ValueError, match="input object only"):
        input_groups(row)


def test_ranking_renderer_matches_training():
    case_input = {
        "state_json": '{"query":"query"}',
        "decisions": [
            {
                "id": "rank",
                "kind": "ranking",
                "type": None,
                "scoring": "relative",
                "instructions_json": '"Find relevant documents"',
                "system_prompt": "",
                "criteria": [],
                "documents": [
                    {"id": "a", "content_json": '"good"'},
                    {"id": "b", "content_json": '{"body":"bad"}'},
                ],
            }
        ],
    }
    actual = input_groups(case_input)[0]
    expected = render_input_group(case_input, 0)
    assert actual.query == expected.query
    assert actual.candidates == expected.candidates
    assert actual.task == "score"
    assert actual.metadata.kind == "ranking"


@pytest.mark.parametrize("interaction", [None, {"width": 8, "heads": 2}])
def test_portable_encoder_matches_reference_and_compiles(tiny_encoder, interaction):
    model = SentenceTransformer(
        modules=[
            tiny_encoder,
            DecisionHeads(32, ["choice", "noul", "score"], choice_interaction=interaction),
        ],
        device="cpu",
    ).eval()
    # The exporter merges adapters; make their effect nonzero before comparing.
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                parameter.fill_(0.015)
    runtime = portable_module(model)
    parts = QueryParts("query", "good bad other")
    groups = [
        Group(parts.render(), ["good", "bad other"], task, query_parts=parts)
        for task in ["choice", "noul", "score"]
    ]
    expected = predict(model, groups, inference="legacy")
    actual = runtime.predict_groups(groups)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    assert all("lora" not in k for k in runtime.state_dict())
    before_keys = set(runtime.state_dict())
    runtime.compile_inference(backend="eager")
    for a, b in zip(runtime.predict_groups(groups), actual, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    assert set(runtime.state_dict()) == before_keys
    runtime.disable_compile()


def test_exported_st_remote_module_roundtrip(tmp_path, tiny_encoder, monkeypatch):
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul", "score"])], device="cpu"
    ).eval()
    checkpoint = tmp_path / "source"
    model.save_pretrained(str(checkpoint), create_model_card=False)
    out = export_model(checkpoint, tmp_path / "portable")
    restored = BekkoSentenceTransformer(
        str(out), device="cpu", trust_remote_code=True, local_files_only=True
    )
    assert type(restored[0]).__module__.startswith("transformers_modules.")
    assert "bekko_system_one." not in (out / "modules.json").read_text()
    source = (out / "inference_v0.py").read_text()
    assert "from ." not in source
    assert "import peft" not in source and "import datasets" not in source
    request = native_case()["input"]
    portable = restored
    actual = portable.predict(request)
    groups = input_groups(request)
    expected = portable_module(model).predict(request)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key]["probabilities"] == pytest.approx(
            expected[key]["probabilities"], abs=1e-6
        )
    batched = portable.predict(
        [request, request], show_progress_bar=False, query_length=32, document_length=32
    )
    assert len(batched) == 2
    for item in batched:
        for key in actual:
            assert item[key]["probabilities"] == pytest.approx(
                actual[key]["probabilities"], abs=2e-6
            )
    assert len(actual) == len(groups)
    assert portable.predict({"state_json": "{}", "decisions": []}) == {}
    spec = importlib.util.spec_from_file_location("standalone_v0", out / "inference_v0.py")
    assert spec is not None and spec.loader is not None
    standalone = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "standalone_v0", standalone)
    spec.loader.exec_module(standalone)
    independent = standalone.BekkoSentenceTransformer(
        str(out), device="cpu", trust_remote_code=True, local_files_only=True
    )
    independent_result = independent.predict(request, show_progress_bar=False)
    for key in actual:
        assert independent_result[key]["probabilities"] == pytest.approx(
            actual[key]["probabilities"], abs=2e-6
        )
    assert runtime_source() == source
    with pytest.raises(ValueError, match="new output"):
        export_model(checkpoint, out)


def test_score_requires_explicit_numeric_scale():
    request = copy.deepcopy(native_case()["input"])
    decision = next(d for d in request["decisions"] if d["type"] == "score")
    decision["criteria"][0]["value"] = None
    with pytest.raises(ValueError, match="numeric values"):
        input_groups(request)


@pytest.fixture
def portable_runtime(tiny_encoder):
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul", "score"])], device="cpu"
    ).eval()
    return portable_module(model)


def test_batch_predict_matches_single_and_preserves_order(portable_runtime):
    requests = [copy.deepcopy(native_case()["input"]) for _ in range(5)]
    for i, request in enumerate(requests):
        request["state_json"] = '"' + "good " * (i + 1) + '"'
    requests.insert(2, {"state_json": "{}", "decisions": []})
    expected = [portable_runtime.predict(r, show_progress_bar=False) for r in requests]
    actual = portable_runtime.predict(requests, batch_size=4, show_progress_bar=False)
    for observed, reference in zip(actual, expected, strict=True):
        assert observed.keys() == reference.keys()
        for key in reference:
            assert observed[key]["probabilities"] == pytest.approx(
                reference[key]["probabilities"], abs=2e-6
            )
    assert actual[2] == {}
    assert portable_runtime.predict([], show_progress_bar=False) == []


def test_batch_budget_and_length_overrides(portable_runtime, monkeypatch):
    parts = QueryParts("good " * 30, "bad " * 30)
    groups = [Group(parts.render(), ["good " * 30, "bad " * 30], "choice", query_parts=parts)] * 4
    shapes = []
    original = portable_runtime.forward

    def record(features, **kwargs):
        shapes.append((features["prefix_ids"].shape, features["doc_ids"].shape))
        return original(features, **kwargs)

    monkeypatch.setattr(portable_runtime, "forward", record)
    settings = dict(portable_runtime.settings)
    merged = portable_runtime.predict_groups(
        groups,
        query_length=12,
        document_length=8,
        show_progress_bar=False,
    )
    assert len(shapes) == 1
    assert shapes[0][0][1] <= 12 and shapes[0][1] == (8, 8)
    shapes.clear()
    separate = portable_runtime.predict_groups(
        groups,
        query_length=12,
        document_length=8,
        token_budget=1,
        show_progress_bar=False,
    )
    assert len(shapes) == 4  # Oversized decisions run intact, alone.
    for a, b in zip(merged, separate, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
    assert portable_runtime.settings == settings
    assert portable_runtime.query_length == settings["query_length"]
    assert portable_runtime.document_length == settings["document_length"]


@pytest.mark.parametrize(
    "options",
    [
        {"batch_size": 0},
        {"token_budget": 0},
        {"query_length": 2},
        {"document_length": 1},
        {"query_length": 1000000},
        {"context_length": 129},
        {"context_length": 0},
    ],
)
def test_invalid_batch_options_even_for_empty_input(portable_runtime, options):
    with pytest.raises(ValueError):
        portable_runtime.predict([], show_progress_bar=False, **options)


def test_progress_default_and_disabled(portable_runtime, capsys):
    portable_runtime.predict([native_case()["input"]])
    assert "Predict" in capsys.readouterr().err
    portable_runtime.predict([native_case()["input"]], show_progress_bar=False)
    assert capsys.readouterr().err == ""


def test_nonfinite_batch_rejected(portable_runtime, monkeypatch):
    def nonfinite(features, **kwargs):
        return {"scores": torch.full((features["doc_ids"].shape[0], 1), float("nan"))}

    monkeypatch.setattr(portable_runtime, "forward", nonfinite)
    with pytest.raises(FloatingPointError):
        portable_runtime.predict(native_case()["input"], show_progress_bar=False)


def test_public_model_prediction_and_compile(portable_runtime):
    model = BekkoSentenceTransformer(modules=[portable_runtime], device="cpu")
    request = native_case()["input"]
    options = dict(
        batch_size=2,
        token_budget=64,
        query_length=24,
        document_length=16,
        prefix_layout="state_instruction",
        show_progress_bar=False,
    )
    expected = portable_runtime.predict([request, request], **options)
    assert model.predict([request, request], **options) == expected
    assert model.compile_inference(backend="eager") is model
    actual = model.predict([request, request], **options)
    for row, reference in zip(actual, expected, strict=True):
        for key in reference:
            assert row[key]["probabilities"] == pytest.approx(
                reference[key]["probabilities"], abs=2e-6
            )
    assert model.disable_compile() is model
    groups = input_groups(request)
    for a, b in zip(
        model.predict_groups(groups, show_progress_bar=False),
        portable_runtime.predict_groups(groups, show_progress_bar=False),
        strict=True,
    ):
        torch.testing.assert_close(a, b)


def test_public_model_rejects_training_modules(tiny_encoder):
    with pytest.raises(ValueError, match="exported v0 checkpoint"):
        BekkoSentenceTransformer(
            modules=[tiny_encoder, DecisionHeads(32, ["choice"])],
            device="cpu",
        )


@pytest.mark.parametrize("policy", ["right", "balanced"])
def test_adaptive_budget_uses_unused_candidate_capacity(portable_runtime, policy):
    runtime = portable_runtime
    runtime.query_truncation = policy
    runtime.context_length = 7999
    runtime.encoder.backbone.config.max_position_embeddings = 7999
    runtime.query_length, runtime.document_length = 7997, 3800
    parts = QueryParts("good", "long " * 9000)
    groups = [
        Group(parts.render(), ["good " * 798, "bad"], "choice", query_parts=parts),
        Group(parts.render(), ["good " * 5000, "bad"], "choice", query_parts=parts),
    ]
    prepared = runtime.prepare_groups(groups)
    assert [(len(g.query), max(map(len, g.documents))) for g in prepared] == [
        (7200, 799),
        (4199, 3800),
    ]
    for g, original in zip(prepared, groups, strict=True):
        alone = runtime.prepare_groups([original])[0]
        assert g.query == alone.query and g.documents == alone.documents
        assert g.query[-1] == runtime.tokenizer.sep_token_id
        assert all(d[-1] == runtime.tokenizer.sep_token_id for d in g.documents)


def test_adaptive_budget_counts_task_marker(portable_runtime):
    runtime = portable_runtime
    runtime.task_token_ids = {"choice": runtime.tokenizer.cls_token_id}
    parts = QueryParts("good", "long " * 150)
    group = Group(parts.render(), ["good " * 60, "bad"], "choice", query_parts=parts)
    prepared = runtime.prepare_groups([group], document_length=32)[0]
    assert max(map(len, prepared.documents)) == 32
    assert len(prepared.query) == 96
    assert prepared.documents[0][0] == runtime.tokenizer.cls_token_id
    assert prepared.documents[0][-1] == runtime.tokenizer.sep_token_id


def test_adaptive_batch_splits_incompatible_padding(portable_runtime, monkeypatch):
    runtime = portable_runtime
    runtime.query_truncation = "right"
    groups = [
        Group("good " * 100, ["bad", "good"], "choice"),
        Group("good " * 10, ["bad " * 79, "good"], "choice"),
    ]
    shapes = []
    forward = runtime.forward

    def record(features, **kwargs):
        shapes.append((features["prefix_ids"].shape[1], features["doc_ids"].shape[1]))
        return forward(features, **kwargs)

    monkeypatch.setattr(runtime, "forward", record)
    actual = runtime.predict_groups(groups, document_length=80, show_progress_bar=False)
    assert len(shapes) == 2
    assert all(q + d <= 128 for q, d in shapes)
    expected = [
        runtime.predict_groups([g], document_length=80, show_progress_bar=False)[0] for g in groups
    ]
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("policy", ["right", "balanced"])
def test_adaptive_caps_and_half_shares_can_be_overridden(portable_runtime, policy):
    runtime = portable_runtime
    runtime.query_truncation = policy
    parts = QueryParts("good", "bad " * 200)
    group = Group(parts.render(), ["good " * 200, "bad"], "choice", query_parts=parts)
    prepared = runtime.prepare_groups([group], context_length=101, document_length=90)[0]
    assert len(prepared.query) == 50
    assert max(map(len, prepared.documents)) == 51  # Candidates receive the odd token.
    shorter = QueryParts("good", "bad")
    group = Group(shorter.render(), group.candidates, "choice", query_parts=shorter)
    prepared = runtime.prepare_groups([group], context_length=101, document_length=90)[0]
    assert max(map(len, prepared.documents)) == 90
    assert len(prepared.query) + max(map(len, prepared.documents)) <= 101
    model = BekkoSentenceTransformer(modules=[runtime], device="cpu")
    actual = model.predict(
        native_case()["input"], context_length=100, document_length=80, show_progress_bar=False
    )
    expected = runtime.predict(
        native_case()["input"], context_length=100, document_length=80, show_progress_bar=False
    )
    assert actual == expected
    assert runtime.context_length == 128
