"""Portable runtime parity, typed rendering and remote-code checkpoint loading."""

import copy
from typing import Any, cast

import pytest
import torch
from sentence_transformers import SentenceTransformer
from test_dataset_training import native_case

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


def test_exported_st_remote_module_roundtrip(tmp_path, tiny_encoder):
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul", "score"])], device="cpu"
    ).eval()
    checkpoint = tmp_path / "source"
    model.save_pretrained(str(checkpoint), create_model_card=False)
    out = export_model(checkpoint, tmp_path / "portable")
    restored = SentenceTransformer(
        str(out), device="cpu", trust_remote_code=True, local_files_only=True
    )
    assert type(restored[0]).__module__.startswith("transformers_modules.")
    assert "bekko_system_one." not in (out / "modules.json").read_text()
    source = (out / "inference_v0.py").read_text()
    assert "from ." not in source
    assert "import peft" not in source and "import datasets" not in source
    request = native_case()["input"]
    portable = cast(Any, restored[0])
    actual = portable.predict(request)
    groups = input_groups(request)
    expected = portable_module(model).predict(request)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key]["probabilities"] == pytest.approx(
            expected[key]["probabilities"], abs=1e-6
        )
    assert len(actual) == len(groups)
    assert portable.predict({"state_json": "{}", "decisions": []}) == {}
    assert runtime_source() == source
    with pytest.raises(ValueError, match="new output"):
        export_model(checkpoint, out)


def test_score_requires_explicit_numeric_scale():
    request = copy.deepcopy(native_case()["input"])
    decision = next(d for d in request["decisions"] if d["type"] == "score")
    decision["criteria"][0]["value"] = None
    with pytest.raises(ValueError, match="numeric values"):
        input_groups(request)
