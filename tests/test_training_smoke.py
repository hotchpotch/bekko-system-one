"""Check the ML dependency stack without downloading model weights."""

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import modules
from transformers import BertConfig, BertModel


def test_transformer_backward() -> None:
    model = BertModel(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    output = model(input_ids=torch.tensor([[1, 2, 3]]))
    loss = output.last_hidden_state.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.embeddings.word_embeddings.weight.grad is not None


def test_sentence_transformer_modules() -> None:
    model = SentenceTransformer(modules=[modules.Pooling(16), modules.Normalize()], device="cpu")
    output = model(
        {
            "token_embeddings": torch.ones(2, 3, 16),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
    )["sentence_embedding"]
    assert output.shape == (2, 16)
    torch.testing.assert_close(output.norm(dim=-1), torch.ones(2))
