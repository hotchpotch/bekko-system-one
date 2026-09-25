"""Optional BF16 LoRA training optimizations; no persistent activation cache."""

from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import nn
from torch.autograd.function import once_differentiable

from .cuda_kernels import rotary_qkv


def precast_frozen_linears(module):
    """Precast only frozen Linear weights; adapters, norms and embeddings stay FP32.

    Call after PEFT has frozen the base, before constructing the optimizer.
    BF16 autocast remains required. Frozen values would receive the same cast
    at every forward (and checkpoint recomputation) without this preparation.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            for parameter in layer.parameters(recurse=False):
                if not parameter.requires_grad:
                    parameter.data = parameter.data.to(torch.bfloat16)


@triton.jit
def _bf16_rn(x):
    bits = x.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    rounded = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, rounded)
    return rounded.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=["T"])
def _rotary_backward(
    GQ,
    GK,
    GV,
    COS,
    SIN,
    OUT,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < T * 3 * H
    row = index // (3 * H)
    component = (index // H) % 3
    col = index % H
    d = col % D
    partner_d = (d + D // 2) % D
    partner_col = col // D * D + partner_d
    ptr = tl.where(component == 0, GQ, GK)
    grad = tl.load(ptr + row * H + col, valid & (component < 2), 0).to(tl.float32)
    partner = tl.load(ptr + row * H + partner_col, valid & (component < 2), 0).to(tl.float32)
    cosine = tl.load(COS + row * D + d, valid, 0)
    sine = tl.load(SIN + row * D + partner_d, valid, 0)
    a = grad * cosine
    b = partner * sine
    b = tl.where(d < D // 2, b, -b)
    # ModernBERT calls q.float() twice, so autograd casts the two paths
    # separately to BF16 *before* summing. Match that rounding, not merely
    # the algebraically equivalent FP32 inverse rotation.
    result = _bf16_rn(_bf16_rn(a).to(tl.float32) + _bf16_rn(b).to(tl.float32))
    gv = tl.load(GV + row * H + col, valid & (component == 2), 0)
    tl.store(OUT + index, tl.where(component == 2, gv, result), valid)


class _TrainingRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, cos, sin, head_dim):
        if cos.requires_grad or sin.requires_grad:
            raise ValueError("Trainable rotary frequencies are not supported")
        ctx.save_for_backward(cos, sin)
        ctx.head_dim = head_dim
        return rotary_qkv(values, cos, sin, head_dim)

    @staticmethod
    @once_differentiable
    def backward(ctx, gq, gk, gv):
        cos, sin = ctx.saved_tensors
        # Materialized zero gradients also handle unused Q/K/V outputs.
        gq, gk, gv = gq.contiguous(), gk.contiguous(), gv.contiguous()
        tokens, heads, dim = gq.shape
        hidden = heads * dim
        output = gq.new_empty((tokens, 3 * hidden))
        cast(Any, _rotary_backward)[(triton.cdiv(tokens * 3 * hidden, 256),)](
            gq,
            gk,
            gv,
            cos,
            sin,
            output,
            tokens,
            hidden,
            ctx.head_dim,
            256,
            enable_fp_fusion=False,
        )
        return output, None, None, None


def training_rotary_qkv(values, cos, sin, head_dim):
    """First-order autograd for the existing exact BF16 fused forward kernel."""
    return _TrainingRotary.apply(values, cos, sin, head_dim)
