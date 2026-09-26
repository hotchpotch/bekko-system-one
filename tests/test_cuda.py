import pytest
import torch
from sentence_transformers import SentenceTransformer
from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

from bekko_system_one import DecisionHeads, Group, prepare_batch
from bekko_system_one.data import collate_groups, prepare_groups

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def test_fused_rotary_exact_gradient():
    from bekko_system_one.cuda_training import training_rotary_qkv

    torch.manual_seed(0)
    tokens, heads, dim = 257, 12, 64
    a = torch.randn(
        tokens, 3 * heads * dim, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    b = a.detach().clone().requires_grad_()
    cos, sin = [torch.randn(tokens, dim, device="cuda") for _ in range(2)]
    q, k, v = a.view(tokens, 3, heads, dim).unbind(1)
    q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
    actual = training_rotary_qkv(b, cos, sin, dim)
    expected = (q, k, v)
    for x, y in zip(actual, expected, strict=True):
        torch.testing.assert_close(x, y, atol=0, rtol=0)
    weights = [torch.randn_like(x) for x in actual]
    torch.autograd.backward(actual, weights)
    torch.autograd.backward(expected, weights)
    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)


@pytest.mark.parametrize("cache_prefix", [False, True])
def test_cuda_graph_and_routing_validation(tiny_encoder, cache_prefix):
    pytest.importorskip("flash_attn")
    from bekko_system_one.inference import SharedPrefixInference

    tiny_encoder.encoder.backend = "flash_attention_2"
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul"])], device="cuda"
    ).eval()
    groups = [Group("query", ["good", "bad"], "choice"), Group("query", ["other", "good"], "noul")]
    features = collate_groups(prepare_groups(groups, tiny_encoder), tiny_encoder)
    with torch.inference_mode():
        expected = model(prepare_batch(model, groups))["scores"]
        session = SharedPrefixInference(model, features, cache_prefix=cache_prefix)
        torch.testing.assert_close(session.score(features), expected, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(session.score(features), expected, atol=2e-4, rtol=2e-3)
    changed = {**features, "head_indices": {"choice": None}}
    with pytest.raises(ValueError, match="routing"):
        session.score(changed)


def test_fast_inference_snapshot_graph_and_weight_update(tiny_encoder):
    pytest.importorskip("flash_attn")
    from bekko_system_one import InferenceEngine, clear_inference_cache, predict
    from bekko_system_one.inference import _RUNTIMES

    tiny_encoder.encoder.backend = "flash_attention_2"
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice", "noul", "reranker"])], device="cuda"
    )
    groups = [Group("query", ["good", "bad"], "choice"), Group("other", ["bad", "good"], "noul")]
    engine = InferenceEngine(model).prepare_fast_inference()
    before = {n: p.clone() for n, p in model.named_parameters()}
    expected = predict(model, groups, inference="legacy")
    for _ in range(3):
        actual = engine.predict(groups)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)
    # Same layout, different tokens must update both prefix and candidate inputs.
    changed = [Group("other", ["bad", "good"], "choice"), Group("query", ["good", "bad"], "noul")]
    for a, b in zip(
        engine.predict(changed), predict(model, changed, inference="legacy"), strict=True
    ):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)
    assert model.training
    runtime = _RUNTIMES[model][1]
    assert any(session is not None for session in runtime.layouts.values())
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n], atol=0, rtol=0)
    torch.testing.assert_close(
        engine.rank("query", ["good", "bad"], chunk_size=1),
        InferenceEngine(model, inference="legacy").rank("query", ["good", "bad"], chunk_size=1),
        atol=2e-4,
        rtol=2e-3,
    )
    with torch.no_grad():
        next(model[1].parameters()).add_(0.5)
    actual = engine.predict(groups)
    assert _RUNTIMES[model][1] is not runtime
    for a, b in zip(actual, predict(model, groups, inference="legacy"), strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)
    clear_inference_cache(model)
    assert model not in _RUNTIMES

    # Layout churn is bounded; a different routing cannot replay an old graph.
    for size in range(2, 6):
        different = [Group("query", ["good"] * size, "reranker")]
        engine.predict(different)
        engine.predict(different)
        assert len(_RUNTIMES[model][1].layouts) <= 2
    clear_inference_cache(model)


def test_unprepared_engine_does_not_run_triton(tiny_encoder, monkeypatch):
    pytest.importorskip("flash_attn")
    from bekko_system_one import InferenceEngine
    from bekko_system_one.inference import _RUNTIMES

    tiny_encoder.encoder.backend = "flash_attention_2"
    tiny_encoder.encoder.fused_rotary = True
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(32, ["choice"])], device="cuda"
    )
    groups = [Group("query", ["good", "bad"], "choice")]

    def forbidden(*args, **kwargs):
        raise AssertionError("Triton invoked before explicit preparation")

    with monkeypatch.context() as patch:
        patch.setattr("bekko_system_one.cuda_kernels.rotary_qkv", forbidden)
        patch.setattr("bekko_system_one.cuda_kernels.gather_kv", forbidden)
        patch.setattr("bekko_system_one.cuda_training.training_rotary_qkv", forbidden)
        engine = InferenceEngine(model)
        expected = engine.predict(groups)
        engine.predict(groups)
        assert not _RUNTIMES[model][1].layouts
        # No representative inputs means enabling the path executes no kernels.
        assert engine.prepare_fast_inference() is engine
    actual = engine.predict(groups)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)
    engine.prepare_fast_inference(example_groups=groups)
    assert any(session is not None for session in _RUNTIMES[model][1].layouts.values())
    assert tiny_encoder.encoder.fused_rotary is True
