"""Task-conditioned branches preserve prefix sharing and checkpoint semantics."""

import copy

import pytest
import torch
from sentence_transformers import SentenceTransformer
from transformers import ModernBertModel

from bekko_system_one import DecisionHeads, Group, predict, prepare_batch, rank
from bekko_system_one.batching import TokenBudget
from bekko_system_one.data import collate_groups, prepare_groups
from bekko_system_one.modules import SharedPrefix
from bekko_system_one.training import optimizer_for, train_step

TOKENS = {"choice": "[CHOICE]", "noul": "[NOUL]", "score": "[SCORE]"}


def make_model(tiny, *, lora=None, document_length=8):
    encoder = SharedPrefix(
        backbone=ModernBertModel(tiny.encoder.backbone.config),
        tokenizer=copy.deepcopy(tiny.tokenizer),
        query_length=32,
        document_length=document_length,
        attention_backend="sdpa",
        frozen_linear_bf16=False,
        fused_rotary=False,
        lora=lora,
        task_tokens=TOKENS,
    )
    return SentenceTransformer(
        modules=[encoder, DecisionHeads(encoder.hidden_size, list(TOKENS))], device="cpu"
    )


def test_task_identity_shares_prefix_and_reserves_marker_budget(tiny_encoder):
    model = make_model(tiny_encoder)
    groups = [Group("query", ["good " * 20, "bad"], task, [1.0, 0.0]) for task in TOKENS]
    prepared = prepare_groups(groups, model[0])
    features = collate_groups(prepared, model[0])
    assert len(features["prefix_ids"]) == 1
    assert features["owners"].tolist() == [0] * 6
    for group in prepared:
        for doc in group.documents:
            assert doc[0] == model[0].task_token_ids[group.task]
            assert doc[-1] == model[0].tokenizer.sep_token_id
            assert len(doc) <= 8
    # Targets cannot affect the model input.
    changed = [Group(g.query, g.candidates, g.task, [0.0, 1.0]) for g in groups]
    other = prepare_batch(model, changed)
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(v, other[k], rtol=0, atol=0)
    with torch.no_grad():
        expected = model(prepare_batch(model, groups))["scores"].flatten()
        # Changing another task's marker cannot contaminate this group's prefix/candidates.
        isolated = model(prepare_batch(model, groups[:1]))["scores"].flatten()
    torch.testing.assert_close(isolated, expected[:2], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("lora", [None, {"rank": 2, "alpha": 4, "dropout": 0.0}])
def test_marker_training_and_offline_roundtrip(tmp_path, tiny_encoder, lora):
    model = make_model(tiny_encoder, lora=lora)
    groups = [Group("query", ["good", "bad"], task, [1.0, 0.0]) for task in TOKENS]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = optimizer_for(model, 1e-3, 1e-3)
    train_step(model, groups, optimizer, TokenBudget(64, maximum=64, target_bytes=2**40))
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    if lora:
        assert any("trainable_tokens" in n for n in changed)
        assert all("lora_" in n or "trainable_tokens" in n or "heads" in n for n in changed)
    else:
        assert any("tok_embeddings" in n for n in changed)
    expected = predict(model, groups, inference="legacy")
    path = tmp_path / "model"
    model.save_pretrained(str(path), create_model_card=False)
    loaded = SentenceTransformer(
        str(path), device="cpu", local_files_only=True, trust_remote_code=True
    )
    assert loaded[0].task_token_ids == model[0].task_token_ids
    for a, b in zip(predict(loaded, groups), expected, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for n, p in model.state_dict().items():
        torch.testing.assert_close(p, loaded.state_dict()[n], rtol=0, atol=0)


def test_score_rank_uses_marker_and_matches_group_forward(tiny_encoder):
    model = make_model(tiny_encoder).eval()
    docs = ["good", "long text", "bad"]
    with torch.no_grad():
        expected = model(prepare_batch(model, [Group("query", docs, "score")]))["scores"].flatten()
    for chunk in [1, 2, 3]:
        torch.testing.assert_close(
            rank(model, "query", docs, task="score", chunk_size=chunk), expected
        )
    with pytest.raises(ValueError, match="no reranker head"):
        rank(model, "query", docs)


def test_invalid_task_marker_configuration(tiny_encoder):
    for tokens in [
        {"unknown": "[X]"},
        {"choice": "[SEP]"},
        {"choice": "good"},
        {"choice": "[X]", "score": "[X]"},
    ]:
        with pytest.raises(ValueError):
            SharedPrefix(
                backbone=ModernBertModel(tiny_encoder.encoder.backbone.config),
                tokenizer=copy.deepcopy(tiny_encoder.tokenizer),
                frozen_linear_bf16=False,
                task_tokens=tokens,
            )


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_task_tokens_cuda_training_and_fast_inference(tiny_encoder):
    model = make_model(tiny_encoder).cuda()
    model[0].encoder.backend = "flash_attention_2"
    model[0].settings["attention_backend"] = "flash_attention_2"
    model[0].encoder.fused_rotary = True
    groups = [Group("query", ["good", "bad"], task, [0.8, 0.2]) for task in TOKENS]
    train_step(
        model,
        groups,
        optimizer_for(model, 1e-3, 1e-3),
        TokenBudget(64, maximum=64, target_bytes=2**40),
    )
    expected = predict(model, groups, inference="legacy")
    for mode in ["optimized", "fast"]:
        for _ in range(2):
            actual = predict(model, groups, inference=mode)
            for a, b in zip(actual, expected, strict=True):
                torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)
