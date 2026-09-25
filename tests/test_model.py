import copy
import json

import pytest
import torch
from sentence_transformers import SentenceTransformer

from bekko_system_one import DecisionHeads, Group, predict, prepare_batch, rank
from bekko_system_one.batching import TokenBudget, adaptive_backward
from bekko_system_one.data import collate_groups, prepare_groups
from bekko_system_one.loss import distribution_loss_sum
from bekko_system_one.training import optimizer_for, source_batches, train_step


def model_for(encoder, tasks=("choice", "noul", "score")):
    return SentenceTransformer(
        modules=[encoder, DecisionHeads(encoder.hidden_size, tasks)], device="cpu"
    )


def test_st_roundtrip_and_frozen_adapter(tmp_path, tiny_encoder, monkeypatch):
    model = model_for(tiny_encoder)
    groups = [Group("query", ["good", "bad"], task) for task in model[1].tasks]
    expected = predict(model, groups)
    model.requires_grad_(False)
    path = tmp_path / "model"
    model.save_pretrained(str(path), create_model_card=False)
    modules = json.loads((path / "modules.json").read_text())
    assert len(modules) == 2
    assert all(m["type"].startswith("bekko_system_one.") for m in modules)

    # No base-model loader is allowed during a checkpoint roundtrip.
    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint attempted to load the original base model")

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", forbidden)
    loaded = SentenceTransformer(
        str(path), device="cpu", local_files_only=True, trust_remote_code=True
    )
    for actual, wanted in zip(predict(loaded, groups), expected, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, loaded.state_dict()[name], rtol=0, atol=0)
    assert any(p.requires_grad for n, p in loaded.named_parameters() if "lora_" in n)


def test_group_accumulation_gradient_equivalence(tiny_encoder):
    model = model_for(tiny_encoder)
    groups = [
        Group("query", ["good", "bad"], "choice", [0.7, 0.3]),
        Group("other query", ["bad", "long text", "good"], "score", [0.0, 0.1, 0.9]),
        Group("query", ["good", "other"], "noul", [1.0, 0.0]),
    ]
    loss = distribution_loss_sum(
        model(prepare_batch(model, groups))["scores"], [g.target for g in groups]
    ) / len(groups)
    loss.backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for group in groups:
        (
            distribution_loss_sum(model(prepare_batch(model, [group]))["scores"], [group.target])
            / len(groups)
        ).backward()
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], atol=2e-6, rtol=2e-5)


def test_rank_chunk_equivalence_and_prefix_count(tiny_encoder, monkeypatch):
    model = model_for(tiny_encoder, ("reranker",)).eval()
    docs = ["good", "long text", "other", "bad"]
    expected = model(prepare_batch(model, [Group("query", docs)]))["scores"].flatten()
    calls = []
    original = tiny_encoder.encoder.encode_prefix

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(tiny_encoder.encoder, "encode_prefix", counted)
    torch.testing.assert_close(rank(model, "query", docs, chunk_size=1), expected)
    assert len(calls) == 1
    # Standard ST encode() also works for the single-head pair interface.
    actual = model.encode(
        [("query", d) for d in docs], convert_to_tensor=True, show_progress_bar=False
    )
    torch.testing.assert_close(actual.flatten(), expected)


def test_real_training_updates_adapters_and_heads_only(tiny_encoder):
    model = model_for(tiny_encoder)
    old = {n: p.detach().clone() for n, p in model.named_parameters()}
    groups = [Group("query", ["good", "bad"], t, [1.0, 0.0]) for t in model[1].tasks]
    optimizer = optimizer_for(model, 2e-3, 2e-3)
    stats = train_step(model, groups, optimizer, TokenBudget(20, maximum=20, target_bytes=2**40))
    assert stats["microbatches"] >= 2
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, old[n])]
    assert any("lora_" in n for n in changed)
    assert any("heads" in n for n in changed)
    assert all("lora_" in n or "heads" in n for n in changed)


def test_oom_replay_discards_partial_gradients_and_restores_rng(tiny_encoder):
    model = model_for(tiny_encoder, ("reranker",))
    optimizer = optimizer_for(model, 2e-4, 2e-3)
    groups = prepare_groups([Group("query", ["good", "bad"], target=[1.0, 0.0])], tiny_encoder)
    draws = []
    parameter = next(model[1].parameters())

    def attempt(batches):
        draws.append(torch.rand(3))
        assert parameter.grad is None
        parameter.sum().backward()
        if len(draws) == 1:
            raise torch.cuda.OutOfMemoryError("simulated")
        return 1.0

    stats = adaptive_backward(
        groups,
        TokenBudget(128, maximum=128, target_bytes=2**40),
        optimizer,
        attempt,
        cleanup=lambda: None,
    )
    assert stats["oom_retries"] == 1
    torch.testing.assert_close(draws[0], draws[1])
    torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))


def test_source_pass_counts_and_tails():
    specs = {"a": {"passes": 1}, "b": {"passes": 3}}
    batches = list(
        source_batches({"a": 5, "b": 3}, specs, mode="source_passes", batch_size=2, seed=42)
    )
    from collections import Counter

    counts = Counter((name, row) for name, rows in batches for row in rows)
    assert counts == Counter({**{("a", i): 1 for i in range(5)}, **{("b", i): 3 for i in range(3)}})


def test_targets_do_not_enter_features(tiny_encoder):
    groups = [Group("query", ["good", "bad"], target=[1.0, 0.0])]
    other = [Group("query", ["good", "bad"], target=[0.0, 1.0])]
    a = collate_groups(prepare_groups(groups, tiny_encoder), tiny_encoder)
    b = collate_groups(prepare_groups(other, tiny_encoder), tiny_encoder)
    for k in a:
        if isinstance(a[k], torch.Tensor):
            assert torch.equal(a[k], b[k])
        else:
            assert a[k] == b[k]
    with pytest.raises(ValueError, match="normalized"):
        Group("query", ["good", "bad"], target=[1.0, 1.0])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fa2_fused_forward_backward(tiny_encoder):
    pytest.importorskip("flash_attn")
    from bekko_system_one.cuda_training import precast_frozen_linears

    reference = model_for(tiny_encoder, ("reranker",)).cuda()
    reference[0].encoder.backend = "flash_attention_2"
    precast_frozen_linears(reference[0])
    reference[0].settings["frozen_linear_bf16"] = True
    fused = copy.deepcopy(reference)
    fused[0].encoder.fused_rotary = True
    groups = [
        Group("query", ["good", "bad"], target=[0.8, 0.2]),
        Group("long query", ["long text", "other", "bad"], target=[1.0, 0.0, 0.0]),
    ]
    results = []
    for model in (reference, fused):
        scores = model(prepare_batch(model, groups))["scores"]
        distribution_loss_sum(scores, [g.target for g in groups]).backward()
        results.append(scores)
    torch.testing.assert_close(*results, rtol=0, atol=0)
    for (name, p), (_, q) in zip(
        reference.named_parameters(), fused.named_parameters(), strict=True
    ):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=2e-2)
