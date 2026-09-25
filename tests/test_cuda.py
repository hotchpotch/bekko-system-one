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
