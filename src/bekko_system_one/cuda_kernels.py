"""Inference-only Triton kernels; FP32 RoPE arithmetic before BF16 rounding."""

from typing import Any, cast

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def _rotary(QKV, COS, SIN, OUT, T, H: tl.constexpr, D: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < T * H
    col = index % H
    row = index // H
    component = tl.program_id(1)
    d = col % D
    partner = col // D * D + (d + D // 2) % D
    x = tl.load(QKV + row * 3 * H + component * H + col, valid, 0).to(tl.float32)
    y = tl.load(QKV + row * 3 * H + component * H + partner, valid, 0).to(tl.float32)
    y = tl.where(d < D // 2, -y, y)
    cosine = tl.load(COS + row * D + d, valid, 0)
    sine = tl.load(SIN + row * D + d, valid, 0)
    # Explicit scalar FP32 operations prevent contraction/vector conversion
    # from changing tie cases at the final BF16 rounding boundary.
    rotated = tl.inline_asm_elementwise(
        "{ .reg .f32 a, b; mul.rn.f32 a, $1, $2; mul.rn.f32 b, $3, $4; add.rn.f32 $0, a, b; }",
        constraints="=f,f,f,f,f",
        args=[x, cosine, y, sine],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    # Explicit round-to-nearest-even avoids a packed BF16 conversion tie
    # discrepancy observed on sm_120. Preserve the eager torch cast exactly.
    bits = rotated.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    rounded = tl.where((bits & 0x7FFFFFFF) > 0x7F800000, 0x7FFF, rounded)
    result = rounded.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    tl.store(OUT + component * T * H + index, result, valid)


@triton.jit
def _gather(
    PK,
    PV,
    K,
    V,
    INDEX,
    OUT,
    N: tl.constexpr,
    H: tl.constexpr,
    P: tl.constexpr,
    PKS: tl.constexpr,
    PVS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < 2 * N * H
    row = (index // H) % N
    col = index % H
    is_k = index < N * H
    source = tl.load(INDEX + row, valid, 0)
    prefix_ptr = tl.where(is_k, PK + source * PKS + col, PV + source * PVS + col)
    doc_ptr = tl.where(is_k, K + (source - P) * KS + col, V + (source - P) * VS + col)
    prefix = tl.load(prefix_ptr, valid & (source < P), 0)
    document = tl.load(doc_ptr, valid & (source >= P), 0)
    tl.store(OUT + index, tl.where(source < P, prefix, document), valid)


def rotary_qkv(values, cos, sin, head_dim):
    if torch.is_grad_enabled():
        raise RuntimeError("Fused CUDA kernels are inference-only")
    tokens, three_hidden = values.shape
    hidden = three_hidden // 3
    assert values.is_cuda and values.is_contiguous() and values.dtype == torch.bfloat16
    assert cos.is_contiguous() and sin.is_contiguous() and head_dim % 2 == 0
    result = values.new_empty((2, tokens, hidden // head_dim, head_dim))
    cast(Any, _rotary)[(triton.cdiv(tokens * hidden, 256), 2)](
        values, cos, sin, result, tokens, hidden, head_dim, 256, enable_fp_fusion=False
    )
    v = values.view(tokens, 3, hidden // head_dim, head_dim)[:, 2]
    return result[0], result[1], v


def gather_kv(pk, pv, k, v, indices):
    if torch.is_grad_enabled():
        raise RuntimeError("Fused CUDA kernels are inference-only")
    rows, heads, dim = indices.numel(), k.shape[1], k.shape[2]
    hidden = heads * dim
    for tensor in (pk, pv, k, v):
        assert tensor.is_cuda and tensor.stride(1) == dim and tensor.stride(2) == 1
    result = k.new_empty((2, rows, heads, dim))
    cast(Any, _gather)[(triton.cdiv(2 * rows * hidden, 256),)](
        pk,
        pv,
        k,
        v,
        indices,
        result,
        rows,
        hidden,
        pk.shape[0],
        pk.stride(0),
        pv.stride(0),
        k.stride(0),
        v.stride(0),
        256,
    )
    return result[0], result[1]
