#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Triton-Ascend implementation of the CANN Bench FusedMulAddLayernormMulAdd interface.

residual = x + gate * attn_out
y        = LayerNorm(residual) * (1 + scale) + shift

Three Triton kernels cover the channel/row-count design space; the host
entry routes by shape metadata only. Per-case scalars (channel dim, row
count, eps, program split) travel in one small float32 meta tensor so
every kernel signature stays within the 10-argument coding-rule budget;
the shared rounding chain is factored into `_mul_add_store`,
`_layernorm_core`, and `_modulate_store` helpers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_add_store(x32, a32, g32, dt, row_base, cols, mask,
                   attn_out_ptr, gate_ptr, residual_ptr):
    """residual = x + gate * attn_out with native-dtype rounding; stores it."""
    ga = (g32 * a32).to(dt)
    r = (x32 + ga.to(tl.float32)).to(dt)
    tl.store(residual_ptr + row_base + cols, r, mask=mask)
    return r


@triton.jit
def _layernorm_core(rf, mask, c_dim, eps):
    """fp32 two-pass LayerNorm statistics; returns the normalized fp32 row."""
    mean = tl.sum(rf, axis=0) / c_dim
    d = tl.where(mask, rf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / c_dim
    rstd = 1.0 / tl.sqrt(var + eps)
    return (rf - mean) * rstd


@triton.jit
def _modulate_store(nn32, s32, h32, dt, row_base, cols, mask, y_ptr):
    """y = norm * (1 + scale) + shift with native-dtype rounding; stores it."""
    t = (1.0 + s32).to(dt)
    y1 = (nn32 * t.to(tl.float32)).to(dt)
    y = (y1.to(tl.float32) + h32).to(dt)
    tl.store(y_ptr + row_base + cols, y, mask=mask)
    return y


@triton.jit
def _fused_mul_add_layernorm_mul_add_kernel(
    x_ptr, attn_out_ptr, gate_ptr, scale_ptr, shift_ptr,
    residual_ptr, y_ptr, meta_ptr,
    block_c: tl.constexpr,
):
    """One program per token row; covers C <= 2048.

    meta = [c_dim, n_dim, eps]: channel dim, token dim per batch, and the
    LayerNorm epsilon. The host passes block_c = next_pow2(C) for small
    channels or the exact channel count for 1024 < C <= 2048, so a single
    kernel serves both layouts (every lane live, one pass per row).

    The layer-norm statistics are accumulated in fp32 with a two-pass
    (mean, then centered second moment) scheme so that near-constant rows
    stay numerically stable; the elementwise chain keeps the golden
    implementation's native-dtype rounding points.
    """
    row = tl.program_id(0)
    c_dim = tl.load(meta_ptr).to(tl.int32)
    n_dim = tl.load(meta_ptr + 1).to(tl.int32)
    eps = tl.load(meta_ptr + 2)
    b = row // n_dim
    cols = tl.arange(0, block_c)
    mask = cols < c_dim
    base = row * c_dim
    pbase = b * c_dim
    dt = x_ptr.dtype.element_ty

    x32 = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    a32 = tl.load(attn_out_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    g32 = tl.load(gate_ptr + pbase + cols, mask=mask, other=0.0).to(tl.float32)

    r = _mul_add_store(x32, a32, g32, dt, base, cols, mask,
                       attn_out_ptr, gate_ptr, residual_ptr)

    rf = tl.where(mask, r.to(tl.float32), 0.0)
    norm = _layernorm_core(rf, mask, c_dim, eps)
    nn32 = norm.to(dt).to(tl.float32)

    s32 = tl.load(scale_ptr + pbase + cols, mask=mask, other=0.0).to(tl.float32)
    h32 = tl.load(shift_ptr + pbase + cols, mask=mask, other=0.0).to(tl.float32)

    _modulate_store(nn32, s32, h32, dt, base, cols, mask, y_ptr)


@triton.jit
def _fused_mul_add_layernorm_mul_add_wide_rows_kernel(
    x_ptr, attn_out_ptr, gate_ptr, scale_ptr, shift_ptr,
    residual_ptr, y_ptr, meta_ptr,
    block_c: tl.constexpr,
):
    """Multi-row variant for large row counts (B*N > 512, 1024 < C <= 2048).

    meta = [c_dim, n_dim, eps, programs_per_batch]. Each program serves
    several token rows of one batch element via a strided row loop;
    gate/scale/shift are loop-invariant for the whole program
    (b = pid // programs_per_batch) and are loaded once before the loop
    instead of once per row.
    """
    pid = tl.program_id(0)
    c_dim = tl.load(meta_ptr).to(tl.int32)
    n_dim = tl.load(meta_ptr + 1).to(tl.int32)
    eps = tl.load(meta_ptr + 2)
    programs_per_batch = tl.load(meta_ptr + 3).to(tl.int32)
    b = pid // programs_per_batch
    p_local = pid % programs_per_batch
    cols = tl.arange(0, block_c)
    mask = cols < c_dim
    pbase = b * c_dim
    dt = x_ptr.dtype.element_ty

    g32 = tl.load(gate_ptr + pbase + cols, mask=mask, other=0.0).to(tl.float32)
    s32 = tl.load(scale_ptr + pbase + cols, mask=mask, other=0.0).to(tl.float32)
    h32 = tl.load(shift_ptr + pbase + cols, mask=mask, other=0.0).to(tl.float32)
    t = (1.0 + s32).to(dt)

    for k in tl.range(p_local, n_dim, programs_per_batch):
        base = (b * n_dim + k) * c_dim
        x32 = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
        a32 = tl.load(attn_out_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)

        r = _mul_add_store(x32, a32, g32, dt, base, cols, mask,
                           attn_out_ptr, gate_ptr, residual_ptr)

        rf = tl.where(mask, r.to(tl.float32), 0.0)
        norm = _layernorm_core(rf, mask, c_dim, eps)
        nn32 = norm.to(dt).to(tl.float32)

        y1 = (nn32 * t.to(tl.float32)).to(dt)
        y = (y1.to(tl.float32) + h32).to(dt)
        tl.store(y_ptr + base + cols, y, mask=mask)


@triton.jit
def _fused_mul_add_layernorm_mul_add_blocked_kernel(
    x_ptr, attn_out_ptr, gate_ptr, scale_ptr, shift_ptr,
    residual_ptr, y_ptr, meta_ptr,
    block_c: tl.constexpr,
):
    """Blocked persistent variant for large channel counts (C > 2048).

    meta = [c_dim, n_dim, eps, num_rows]. Each program owns several token
    rows via a grid-stride loop (grid = min(B*N, num vector cores));
    within a row the channel dimension is traversed in block_c-sized
    sub-blocks so that the vector lanes stay fully utilized for
    non-power-of-two C.

    The precision chain (six native-dtype rounding points) is identical
    to the one-shot kernel; statistics are fp32 with a two-pass
    (sum, then centered second moment) scheme accumulated per sub-block.
    Passes 2 and 3 re-read the just-written residual, which is served
    from L2 cache.
    """
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    c_dim = tl.load(meta_ptr).to(tl.int32)
    n_dim = tl.load(meta_ptr + 1).to(tl.int32)
    eps = tl.load(meta_ptr + 2)
    num_rows = tl.load(meta_ptr + 3).to(tl.int32)
    dt = x_ptr.dtype.element_ty
    for row in range(pid, num_rows, num_progs):
        b = row // n_dim
        base = row * c_dim
        pbase = b * c_dim

        _sum = tl.zeros([block_c], dtype=tl.float32)
        for off in range(0, c_dim, block_c):
            cols = off + tl.arange(0, block_c)
            m = cols < c_dim
            x32 = tl.load(x_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            a32 = tl.load(attn_out_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            g32 = tl.load(gate_ptr + pbase + cols, mask=m, other=0.0).to(tl.float32)
            ga = (g32 * a32).to(dt)
            r = (x32 + ga.to(tl.float32)).to(dt)
            tl.store(residual_ptr + base + cols, r, mask=m)
            _sum += tl.where(m, r.to(tl.float32), 0.0)
        mean = tl.sum(_sum, axis=0) / c_dim

        _var = tl.zeros([block_c], dtype=tl.float32)
        for off in range(0, c_dim, block_c):
            cols = off + tl.arange(0, block_c)
            m = cols < c_dim
            rf = tl.load(residual_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            d = tl.where(m, rf - mean, 0.0)
            _var += d * d
        var = tl.sum(_var, axis=0) / c_dim
        rstd = 1.0 / tl.sqrt(var + eps)

        for off in range(0, c_dim, block_c):
            cols = off + tl.arange(0, block_c)
            m = cols < c_dim
            rf = tl.load(residual_ptr + base + cols, mask=m, other=0.0).to(tl.float32)
            s32 = tl.load(scale_ptr + pbase + cols, mask=m, other=0.0).to(tl.float32)
            h32 = tl.load(shift_ptr + pbase + cols, mask=m, other=0.0).to(tl.float32)
            norm = (rf - mean) * rstd
            nn32 = norm.to(dt).to(tl.float32)
            t = (1.0 + s32).to(dt)
            y1 = (nn32 * t.to(tl.float32)).to(dt)
            y = (y1.to(tl.float32) + h32).to(dt)
            tl.store(y_ptr + base + cols, y, mask=m)


def _make_contiguous(x, attn_out, gate, scale, shift):
    """Ensure all five inputs are contiguous."""
    return (
        x if x.is_contiguous() else x.contiguous(),
        attn_out if attn_out.is_contiguous() else attn_out.contiguous(),
        gate if gate.is_contiguous() else gate.contiguous(),
        scale if scale.is_contiguous() else scale.contiguous(),
        shift if shift.is_contiguous() else shift.contiguous(),
    )


def _make_meta(values, ref):
    """Pack per-case scalars into one small float32 tensor on the input device."""
    return torch.tensor(values, dtype=torch.float32, device=ref.device)


def fused_mul_add_layernorm_mul_add(
    x: torch.Tensor,
    attn_out: torch.Tensor,
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DiT attention residual update fused with the MLP's adaptive LayerNorm.

    residual = x + gate * attn_out (gate/scale/shift [B, C] broadcast over N);
    y = LayerNorm(residual) * (1 + scale) + shift, no affine parameters.
    eps: variance regularizer (DiT uses 1e-6). Returns (residual, y).
    """
    x, attn_out, gate, scale, shift = _make_contiguous(x, attn_out, gate, scale, shift)

    batch, seq, channels = x.shape
    residual = torch.empty_like(x)
    y = torch.empty_like(x)

    if channels <= 1024:
        meta = _make_meta([channels, seq, eps], x)
        grid = (batch * seq,)
        _fused_mul_add_layernorm_mul_add_kernel[grid](
            x, attn_out, gate, scale, shift, residual, y,
            meta, block_c=max(triton.next_power_of_2(channels), 16),
        )
    elif channels <= 2048:
        total_rows = batch * seq
        if total_rows > 512:
            programs_per_batch = min(seq, max(1, 40 // batch))
            meta = _make_meta([channels, seq, eps, programs_per_batch], x)
            grid = (batch * programs_per_batch,)
            _fused_mul_add_layernorm_mul_add_wide_rows_kernel[grid](
                x, attn_out, gate, scale, shift, residual, y,
                meta, block_c=channels,
            )
        else:
            meta = _make_meta([channels, seq, eps], x)
            grid = (total_rows,)
            _fused_mul_add_layernorm_mul_add_kernel[grid](
                x, attn_out, gate, scale, shift, residual, y,
                meta, block_c=channels,
            )
    else:
        meta = _make_meta([channels, seq, eps, batch * seq], x)
        grid = (min(batch * seq, 40),)
        _fused_mul_add_layernorm_mul_add_blocked_kernel[grid](
            x, attn_out, gate, scale, shift, residual, y,
            meta, block_c=128,
        )
    return residual, y
