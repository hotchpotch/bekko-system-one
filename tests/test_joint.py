import copy

import pytest
import torch
from sentence_transformers import SentenceTransformer

from bekko_system_one.data import Group, prepare_batch
from bekko_system_one.export_v0 import portable_module
from bekko_system_one.joint import JointEncoder, fair_lengths
from bekko_system_one.model import predict, rank
from bekko_system_one.modules import DecisionHeads

VARIANTS = [
    ("cross_encoder", "cls"),
    ("cross_encoder", "mean"),
    ("listwise", "mean"),
    ("listwise", "token_before"),
    ("listwise", "token_after"),
]


def joint_model(tiny_encoder, architecture, pooling, **kwargs):
    backbone = copy.deepcopy(tiny_encoder.encoder.backbone).merge_and_unload().requires_grad_(True)
    encoder = JointEncoder(
        backbone=backbone,
        tokenizer=copy.deepcopy(tiny_encoder.tokenizer),
        architecture=architecture,
        pooling=pooling,
        candidate_token="[CAND]" if pooling.startswith("token_") else None,
        query_length=32,
        document_length=32,
        attention_backend="sdpa",
        frozen_linear_bf16=False,
        fused_rotary=False,
        lora=None,
        **kwargs,
    )
    return SentenceTransformer(modules=[encoder, DecisionHeads(32, ("choice",))], device="cpu")


@pytest.mark.parametrize("architecture,pooling", VARIANTS)
def test_joint_roundtrip_and_gradient(tiny_encoder, tmp_path, architecture, pooling):
    model = joint_model(tiny_encoder, architecture, pooling)
    groups = [Group("query", ["good text", "bad"], "choice")]
    features = prepare_batch(model, groups)
    scores = model(features)["scores"]
    (-scores.flatten().log_softmax(0)[0]).backward()
    assert model[0].encoder.backbone.embeddings.tok_embeddings.weight.grad.abs().sum() > 0
    if pooling.startswith("token_"):
        marker = model[0].candidate_token_id
        assert (
            model[0].encoder.backbone.embeddings.tok_embeddings.weight.grad[marker].abs().sum() > 0
        )
    expected = predict(model, groups)[0]
    model.save_pretrained(str(tmp_path), create_model_card=False)
    loaded = SentenceTransformer(
        str(tmp_path), device="cpu", local_files_only=True, trust_remote_code=True
    )
    torch.testing.assert_close(predict(loaded, groups)[0], expected)
    torch.testing.assert_close(
        rank(loaded, "query", groups[0].candidates, task="choice").softmax(0), expected
    )
    with pytest.raises(ValueError, match="SharedPrefix"):
        portable_module(model)
    with pytest.raises(ValueError, match="inference"):
        rank(loaded, "query", groups[0].candidates, task="choice", inference="invalid")
    shortened = SentenceTransformer(
        str(tmp_path), device="cpu", local_files_only=True, trust_remote_code=True,
        model_kwargs={"max_length": 16},
    )
    assert shortened[0].max_length == 16
    assert shortened[0].settings["max_length"] == 16
    with pytest.raises(ValueError, match="max_length"):
        SentenceTransformer(
            str(tmp_path), device="cpu", local_files_only=True, trust_remote_code=True,
            model_kwargs={"max_length": 2},
        )


@pytest.mark.parametrize("architecture,pooling", VARIANTS)
def test_decision_isolation(tiny_encoder, architecture, pooling):
    model = joint_model(tiny_encoder, architecture, pooling).eval()
    a = Group("query", ["good", "bad"], "choice")
    b = Group("query", ["other long text", ""], "choice")
    separate = [predict(model, [g])[0] for g in [a, b]]
    together = predict(model, [a, b])
    for actual, expected in zip(together, separate, strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("pooling", ["mean", "token_before", "token_after"])
def test_listwise_truncation_and_marker_positions(tiny_encoder, pooling):
    encoder = joint_model(tiny_encoder, "listwise", pooling, max_length=16)[0]
    query = [2] + [4] * 30 + [3]
    docs = [[5] * 30 + [3], [6] * 30 + [3], [3]]
    ids, spans = encoder._sequence(query, docs)
    assert len(ids) == 16
    assert ids[0] == 2
    assert len(spans) == 3 and all(spans)
    if pooling.startswith("token_"):
        assert all(ids[s[0]] == encoder.candidate_token_id for s in spans)
        for s in spans[:2]:
            assert ids[s[0] + (1 if pooling == "token_before" else -1)] in {5, 6}
    else:
        assert all(ids[i] == 5 for i in spans[0])
        assert all(ids[i] == 6 for i in spans[1])
    with pytest.raises(ValueError, match="Candidate count"):
        encoder._sequence(query, docs * 20)


def test_fair_budget():
    assert fair_lengths([2, 100, 100], 12) == [2, 5, 5]
    assert fair_lengths([0, 2, 10], 20) == [0, 2, 10]


@pytest.mark.parametrize("architecture,pooling", [("cross_encoder", "mean"), ("listwise", "mean")])
def test_joint_attention_matches_backbone(tiny_encoder, architecture, pooling):
    model = joint_model(tiny_encoder, architecture, pooling).eval()
    f = prepare_batch(model, [Group("query", ["good text", "bad"], "choice")])
    encoder = model[0]
    with torch.no_grad():
        actual = encoder.encoder.encode_bidirectional(f["input_ids"], f["attention_mask"])
        expected = encoder.encoder.backbone(
            input_ids=f["input_ids"], attention_mask=f["attention_mask"]
        ).last_hidden_state
    torch.testing.assert_close(actual, expected[f["attention_mask"]], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("architecture,pooling", VARIANTS)
def test_candidate_attention_scope(tiny_encoder, architecture, pooling):
    model = joint_model(tiny_encoder, architecture, pooling).eval()
    first = rank(model, "query", ["good", "bad"], task="choice")[0]
    changed = rank(model, "query", ["good", "other long text"], task="choice")[0]
    if architecture == "cross_encoder":
        torch.testing.assert_close(first, changed, atol=1e-6, rtol=1e-5)
    else:
        assert abs(float(first - changed)) > 1e-5


@pytest.mark.cuda
@pytest.mark.parametrize("architecture,pooling", VARIANTS)
def test_joint_cuda_matches_sdpa(tiny_encoder, architecture, pooling):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    pytest.importorskip("flash_attn")
    reference = joint_model(tiny_encoder, architecture, pooling).to("cuda").eval()
    actual = copy.deepcopy(reference)
    actual[0].encoder.backend = "flash_attention_2"
    groups = [Group("query", ["good text", "bad"], "choice")]
    a = actual(prepare_batch(actual, groups))["scores"]
    b = reference(prepare_batch(reference, groups))["scores"]
    torch.testing.assert_close(a, b, atol=3e-3, rtol=3e-2)
    a[0].backward()
    b[0].backward()
    for p, q in zip(actual.parameters(), reference.parameters(), strict=True):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-3, rtol=5e-2)
