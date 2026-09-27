"""Balanced truncation preserves both fields and survives model serialization."""

from dataclasses import replace

import pytest
from sentence_transformers import SentenceTransformer

from bekko_system_one import DecisionHeads, Group, QueryParts, prepare_batch
from bekko_system_one.data import prepare_groups
from bekko_system_one.query_budget import allocate_query_budget, balanced_query_ids


@pytest.mark.parametrize(
    "instruction,context,budget,expected",
    [
        (8000, 1000, 4000, (3000, 1000)),
        (1000, 7000, 4000, (1000, 3000)),
        (8000, 7000, 4000, (2000, 2000)),
        (500, 7000, 4000, (500, 3500)),
        (1, 1, 10, (1, 1)),
        (10, 10, 5, (3, 2)),
        (0, 10, 5, (0, 5)),
    ],
)
def test_allocation(instruction, context, budget, expected):
    assert allocate_query_budget(instruction, context, budget) == expected


def test_fields_retained_identically_across_layouts(tiny_encoder):
    tok = tiny_encoder.tokenizer
    a = QueryParts("good " * 80, "bad " * 7, "query")
    b = replace(a, layout="state_instruction")
    rows = balanced_query_ids(tok, [a, b], 32)
    good, bad = tok.convert_tokens_to_ids(["good", "bad"])
    for ids in rows:
        assert len(ids) == 32
        assert ids.count(bad) == 7
        assert ids.count(good) > 7  # Unused context quota flows to instruction.
        assert ids[0] == tok.cls_token_id and ids[-1] == tok.sep_token_id
    assert sorted(rows[0]) == sorted(rows[1])
    assert rows[0].index(good) < rows[0].index(bad)
    assert rows[1].index(bad) < rows[1].index(good)
    # System and headers consume the budget, instead of being appended over the limit.
    with pytest.raises(ValueError, match="System prompt"):
        balanced_query_ids(tok, [replace(a, system="query " * 40)], 32)


def test_balanced_training_and_roundtrip(tmp_path, tiny_encoder):
    tiny_encoder.query_truncation = "balanced"
    tiny_encoder.settings["query_truncation"] = "balanced"
    parts = QueryParts("good " * 40, "bad " * 40)
    group = Group(parts.render(), ["good", "bad"], "choice", [1.0, 0.0], parts)
    model = SentenceTransformer(
        modules=[tiny_encoder, DecisionHeads(tiny_encoder.hidden_size, ["choice"])], device="cpu"
    )
    features = prepare_batch(model, [group, group])
    assert features["prefix_ids"].shape[0] == 1
    loss = model(features)["scores"].sum()
    loss.backward()
    before = prepare_groups([group], tiny_encoder)[0]
    model.save_pretrained(str(tmp_path / "model"), create_model_card=False)
    restored = SentenceTransformer(
        str(tmp_path / "model"), device="cpu", local_files_only=True, trust_remote_code=True
    )
    assert restored[0].query_truncation == "balanced"
    assert prepare_groups([group], restored[0])[0].query == before.query
    # Explicit override also works when loading an existing checkpoint.
    legacy = SentenceTransformer(
        str(tmp_path / "model"),
        device="cpu",
        local_files_only=True,
        trust_remote_code=True,
        model_kwargs={"query_truncation": "right"},
    )
    assert legacy[0].query_truncation == "right"
    with pytest.raises(ValueError, match="QueryParts"):
        prepare_groups([Group(parts.render(), ["good"])], tiny_encoder)


def test_structured_group_roundtrip_and_legacy(tiny_encoder):
    from dataclasses import asdict

    parts = QueryParts("good", "Instruction: bad\nState: other")
    group = Group.from_dict(
        dict(query=parts.render(), candidates=["good"], query_parts=asdict(parts))
    )
    assert group.query_parts == parts
    assert (
        prepare_groups([group], tiny_encoder)[0].query
        == tiny_encoder.tokenize_branches([parts.render()], ["good"])[0][0]
    )
    with pytest.raises(ValueError, match="match"):
        Group("different", ["good"], query_parts=parts)
