import copy

import pytest
import torch
from sentence_transformers import SentenceTransformer

from bekko_system_one import DecisionHeads, Group, predict, prepare_batch
from bekko_system_one.loss import distribution_loss_sum


def contextual_model(encoder):
    return SentenceTransformer(
        modules=[
            encoder,
            DecisionHeads(
                encoder.hidden_size,
                ["choice", "noul"],
                choice_interaction={"width": 16, "heads": 2},
            ),
        ],
        device="cpu",
    )


def activate(model):
    with torch.no_grad():
        model[1].choice_interaction.output.weight.normal_(std=0.3)


def test_zero_initialization_and_trainable_interaction(tiny_encoder):
    model = contextual_model(tiny_encoder)
    groups = [Group("query", ["good", "bad", "other"], "choice", [0.0, 1.0, 0.0])]
    reference = copy.deepcopy(model)
    reference[1].choice_interaction = None
    torch.testing.assert_close(
        model(prepare_batch(model, groups))["scores"],
        reference(prepare_batch(reference, groups))["scores"],
        rtol=0,
        atol=0,
    )
    optimizer = torch.optim.Adam(model[1].parameters(), lr=0.01)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = distribution_loss_sum(
            model(prepare_batch(model, groups))["scores"], [groups[0].target]
        )
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
    assert losses[-1] < losses[0]
    assert model[1].choice_interaction.qkv.weight.grad.abs().sum() > 0


def test_permutation_padding_group_isolation_and_cross_candidate_effect(tiny_encoder):
    model = contextual_model(tiny_encoder).eval()
    activate(model)
    a = Group("query", ["good", "bad", "other"], "choice")
    b = Group("query", ["long text", "good"], "choice")
    c = Group("query", ["good", "bad"], "noul")
    batch = predict(model, [a, b, c], inference="legacy")
    for group, expected in zip([a, b, c], batch, strict=True):
        torch.testing.assert_close(predict(model, [group], inference="legacy")[0], expected)
    permutation = [2, 0, 1]
    permuted = Group(a.query, [a.candidates[i] for i in permutation], "choice")
    torch.testing.assert_close(predict(model, [permuted])[0], batch[0][permutation])
    changed = Group(a.query, ["good", "bad", "long long text"], "choice")
    original_scores = model(prepare_batch(model, [a]))["scores"].flatten()
    changed_scores = model(prepare_batch(model, [changed]))["scores"].flatten()
    assert (
        abs(
            (original_scores[0] - original_scores[1]) - (changed_scores[0] - changed_scores[1])
        ).item()
        > 1e-6
    )
    # A different decision may share the same prefix but cannot change this one.
    torch.testing.assert_close(predict(model, [b, changed, c])[0], batch[1])
    independent = copy.deepcopy(model)
    independent[1].choice_interaction = None
    torch.testing.assert_close(predict(independent, [c])[0], batch[2])


def test_group_gradients_and_roundtrip(tiny_encoder, tmp_path):
    model = contextual_model(tiny_encoder)
    activate(model)
    groups = [
        Group("query", ["good", "bad"], "choice", [0.8, 0.2]),
        Group("query", ["bad", "other", "good"], "choice", [0.0, 0.3, 0.7]),
    ]
    distribution_loss_sum(
        model(prepare_batch(model, groups))["scores"], [g.target for g in groups]
    ).backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for g in groups:
        distribution_loss_sum(model(prepare_batch(model, [g]))["scores"], [g.target]).backward()
    for n, p in model.named_parameters():
        if n in expected:
            torch.testing.assert_close(p.grad, expected[n], atol=2e-6, rtol=2e-5)
    path = tmp_path / "model"
    model.save_pretrained(str(path), create_model_card=False)
    loaded = SentenceTransformer(
        str(path), device="cpu", trust_remote_code=True, local_files_only=True
    )
    for a, b in zip(predict(model, groups), predict(loaded, groups), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize("config", [{"width": 0}, {"heads": 3}, {"width": True}, {"typo": 1}])
def test_invalid_config(config):
    with pytest.raises(ValueError):
        DecisionHeads(32, ["choice"], choice_interaction=config)


def test_missing_groups_fail_closed(tiny_encoder):
    model = contextual_model(tiny_encoder)
    with pytest.raises(ValueError, match="complete groups"):
        model[1]({"sentence_embedding": torch.randn(2, 32), "head_indices": {"choice": None}})


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_contextual_cuda_graph(tiny_encoder):
    pytest.importorskip("flash_attn")
    from bekko_system_one.data import collate_groups, prepare_groups
    from bekko_system_one.inference import SharedPrefixInference

    tiny_encoder.encoder.backend = "flash_attention_2"
    model = contextual_model(tiny_encoder).to("cuda").eval()
    activate(model)
    groups = [
        Group("query", ["good", "bad"], "choice"),
        Group("query", ["other", "good", "bad"], "choice"),
    ]
    features = collate_groups(prepare_groups(groups, model[0]), model[0])
    with torch.inference_mode():
        expected = model(prepare_batch(model, groups))["scores"]
        session = SharedPrefixInference(model, features, cache_prefix=True)
        torch.testing.assert_close(session.score(features), expected, atol=2e-4, rtol=2e-3)
        regrouped = [
            Group("query", ["good", "bad", "other"], "choice"),
            Group("query", ["good", "bad"], "choice"),
        ]
        changed = collate_groups(prepare_groups(regrouped, model[0]), model[0])
        with pytest.raises(ValueError, match="layout changed"):
            session.score(changed)
        for _ in range(3):
            for a, b in zip(
                predict(model, regrouped, inference="fast"),
                predict(model, regrouped, inference="legacy"),
                strict=True,
            ):
                torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)


def test_contextual_rank_collects_complete_group(tiny_encoder):
    from bekko_system_one import rank

    model = contextual_model(tiny_encoder).eval()
    activate(model)
    group = Group("query", ["good", "bad", "other"], "choice")
    expected = model(prepare_batch(model, [group]))["scores"].flatten()
    for chunk in [1, 2, 3]:
        torch.testing.assert_close(
            rank(model, group.query, group.candidates, task="choice", chunk_size=chunk), expected
        )
