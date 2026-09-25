import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import ModernBertConfig, ModernBertModel, PreTrainedTokenizerFast

from bekko_system_one.modules import SharedPrefix


@pytest.fixture
def tiny_encoder():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    vocab = {
        token: i
        for i, token in enumerate(
            ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "query", "good", "bad", "other", "long", "text"]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )
    backbone = ModernBertModel(
        ModernBertConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=128,
            local_attention=16,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            embedding_dropout=0.0,
            pad_token_id=0,
            cls_token_id=2,
            sep_token_id=3,
            bos_token_id=2,
            eos_token_id=3,
        )
    )
    return SharedPrefix(
        backbone=backbone,
        tokenizer=tokenizer,
        query_length=32,
        document_length=32,
        attention_backend="sdpa",
        frozen_linear_bf16=False,
        fused_rotary=False,
        lora=dict(rank=2, alpha=4, dropout=0.0),
    )
