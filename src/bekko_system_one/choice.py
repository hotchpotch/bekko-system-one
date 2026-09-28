"""Permutation-equivariant residual comparison of independent candidate vectors."""

from torch import nn
from torch.nn import functional as F


class ChoiceInteraction(nn.Module):
    """One small attention block, isolated by decision rather than shared prefix.

    Zero-initializing only the final projection preserves the existing scorer.
    No candidate position embeddings or dropout are used.
    """

    def __init__(self, hidden_size, width=128, heads=4):
        super().__init__()
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in (width, heads)):
            raise ValueError("Choice width and heads must be positive integers")
        if width % heads:
            raise ValueError("Choice width must be divisible by heads")
        self.width, self.heads = width, heads
        self.project = nn.Linear(hidden_size, width)
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.attention_out = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )
        self.output = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, hidden, indices, mask):
        x = self.project(hidden)[indices]
        batch, count, _ = x.shape
        qkv = self.qkv(self.norm(x)).reshape(batch, count, 3, self.heads, -1)
        q, k, v = qkv.unbind(2)
        attended = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=mask[:, None, None, :],
            )
            .transpose(1, 2)
            .reshape(batch, count, self.width)
        )
        x = x + self.attention_out(attended)
        x = x + self.ffn(x)
        delta = self.output(x).masked_fill(~mask.unsqueeze(-1), 0)
        return hidden.new_zeros((hidden.shape[0], 1)).index_add(
            0, indices.flatten(), delta.flatten(0, 1)
        )
