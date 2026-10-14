"""AutoClass remote loading, standalone prediction and save/reload parity."""

import json
import subprocess
import sys

import pytest
from sentence_transformers import SentenceTransformer
from test_dataset_training import native_case
from transformers import AutoConfig, AutoModel, PreTrainedModel

from bekko_system_one.export_v0 import export_model, portable_module
from bekko_system_one.modules import DecisionHeads


def test_automodel_standalone_roundtrip(tmp_path, tiny_encoder):
    source = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul", "score"])], device="cpu"
    ).eval()
    source.save_pretrained(tmp_path / "source")
    export = export_model(tmp_path / "source", tmp_path / "export")
    config = AutoConfig.from_pretrained(export, trust_remote_code=True)
    assert config.model_type == "bekko_system_one"
    model = AutoModel.from_pretrained(
        export, trust_remote_code=True, device_map={"": "cpu"}, attn_implementation="sdpa"
    )
    assert isinstance(model, PreTrainedModel)
    request = native_case()["input"]
    expected = portable_module(source).predict(request, show_progress_bar=False)
    actual = model.predict(request, show_progress_bar=False)
    for key in expected:
        assert actual[key]["probabilities"] == pytest.approx(expected[key]["probabilities"], abs=2e-6)
    with pytest.raises(ValueError, match="single-device"):
        AutoModel.from_pretrained(export, trust_remote_code=True, device_map="auto")
    model.save_pretrained(tmp_path / "resaved")
    (tmp_path / "request.json").write_text(json.dumps(request))
    script = '''
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname == "bekko_system_one" or fullname.startswith("bekko_system_one."):
   raise ImportError("No training package")
sys.meta_path.insert(0, Block())
from transformers import AutoModel
model=AutoModel.from_pretrained(sys.argv[1], trust_remote_code=True, local_files_only=True)
print(json.dumps(model.predict(json.load(open(sys.argv[2])), show_progress_bar=False)))
'''
    process = subprocess.run(
        [sys.executable, "-I", "-c", script, str(tmp_path / "resaved"), str(tmp_path / "request.json")],
        check=True, capture_output=True, text=True, cwd=tmp_path,
    )
    reloaded = json.loads(process.stdout)
    for key in expected:
        assert reloaded[key]["probabilities"] == pytest.approx(actual[key]["probabilities"], abs=2e-6)


def test_systemone_matches_native_and_preserves_values(tmp_path, tiny_encoder):
    source = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul", "score"])], device="cpu"
    ).eval()
    source.save_pretrained(tmp_path / "source")
    export = export_model(tmp_path / "source", tmp_path / "export")
    model = AutoModel.from_pretrained(export, trust_remote_code=True)
    request = {"state": "Refund requested", "questions": {
        "refund": {"type": "noul", "instructions": "Refund?"},
        "department": {"type": "choice", "criteria": {"refunds": "Refunds", "other": "Other"}},
        "urgency": {"type": "score", "criteria": [
            {"description": "Normal", "value": 0},
            {"description": "Priority", "value": 2},
            {"description": "Immediate", "value": 4},
        ]},
    }}
    native = model.predict(model._systemone_input(request), show_progress_bar=False)
    answers = model.systemone(request, show_progress_bar=False)["answers"]
    assert answers["refund"]["noul"] == native["refund"]["probability_yes"]
    assert answers["department"]["choice"] == native["department"]["selected_id"]
    assert answers["urgency"]["numeric_values"] == {"0": 0, "1": 2, "2": 4}
    assert answers["urgency"]["expected_value"] == native["urgency"]["score"]
    assert answers["urgency"]["expected_value"] == pytest.approx(2 * answers["urgency"]["score"])
    batch = model.systemone([request, request], show_progress_bar=False)
    assert len(batch) == 2
    for response in batch:
        for qid in answers:
            assert response["answers"][qid]["probabilities"] == pytest.approx(answers[qid]["probabilities"], abs=2e-6)
    model.save_pretrained(tmp_path / "resaved")
    restored = AutoModel.from_pretrained(tmp_path / "resaved", trust_remote_code=True)
    assert restored.systemone(request, show_progress_bar=False) == {"answers": answers}
    for bad in [
        {"state": "x", "questions": {}},
        {"state": "x", "questions": {"x": {"type": "choice", "criteria": {}}}},
        {"state": "x", "questions": {"x": {"type": "score", "criteria": ["one"]}}},
        {"state": "x", "questions": {"x": {"type": "score", "criteria": [
            {"description": "A", "value": True}, {"description": "B", "value": 2}]}}},
        {"state": "x", "questions": {"x": {"type": "noul", "criteria": {"yes": "yes"}}}},
        dict(request, targets={}),
    ]:
        with pytest.raises(ValueError):
            model.systemone(bad, show_progress_bar=False)
