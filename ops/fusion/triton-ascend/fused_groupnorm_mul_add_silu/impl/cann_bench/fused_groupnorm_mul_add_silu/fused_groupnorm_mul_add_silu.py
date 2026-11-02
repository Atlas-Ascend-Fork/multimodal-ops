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


"""Triton-Ascend implementation of the CANN Bench FusedGroupnormMulAddSilu interface.

y = SiLU(GroupNorm(x, weight, bias, groups=32, eps) * (1 + scale) + shift)

Three dispatch paths (F2 vertical fusion family with F5.schedule_routing):
because the input is contiguous NCHW and each group covers C/32 consecutive
channels, a group is a contiguous slab of N = (C/32) * H * W elements starting
at gid * N, which keeps every address a linear offset.

1. Small-group single-tile kernel (N rounded up to a power-of-two 2D tile of
   at most 8192 elements): the whole group is loaded once as a
   [block_cpg, block_hw] tile, mean/var/normalize/affine/modulation/SiLU run
   on the resident tile with vectorized per-channel parameter loads, and the
   epilogue is chunked along HW only when the full-tile epilogue would exceed
   the UB budget (a second L2-resident load replaces the resident tile).

2. Stats+apply split pipeline (large fp32 groups, N >= 49152): two kernels on
   one stream, grid = (B * 32 * split,). The stats kernel computes raw-moment
   partial sums (s1, s2) over a contiguous 1/split slab slice per program and
   writes them to a tiny fp32 GM workspace; the apply kernel combines the
   split partials of its group into mean/var/rstd and runs the unchanged
   channel-wise epilogue on its own 1/split HW slice. Same 2-reads + 1-write
   traffic as the merged streaming kernel, but split x more programs (128 vs
   32) and statistics fully separated from the epilogue.

3. Streaming kernel (everything else): merged single-pass statistics -- one
   loop over x collects sum(x) and sum(x^2) in fp32 scalars (two reads of x
   total instead of three), then the channel-wise normalize + affine +
   modulation + SiLU pass.

Numerical semantics (matching the official GroupNorm32 mixed-precision order):
- Small groups use the two-pass variance on the resident tile; the streaming
  and split paths merge both statistic passes into raw moments
  s1 = sum(x), s2 = sum(x^2) (onepass_stats constexpr keeps the two-pass
  fallback), reading x two times total instead of three.
- Normalization and affine run in fp32. The fp32 epilogue evaluates the whole
  affine + modulation chain as z = v * A + B' with per-channel coefficients.
  The default _DD_EPILOGUE=False form merges each double-double coefficient
  pair into one fp32 scalar plus its exact merge residual (a_s/a_se,
  b_s/b_se via Knuth TwoSum) and evaluates
  z = (v*a_s + b_s) + (v*a_se + b_se): the residual carries the
  pair-quantization error that dominates the cancellation region when
  |A|,|B'| are large (zero/small-variance inputs), so the merged form tracks
  the double-double result at ~5 instead of ~28 flops/element. Setting
  _DD_EPILOGUE=True restores the full error-free two-product/two-sum chain
  (Dekker/Knuth) -- the instant precision fallback. The half-precision paths
  round to x.dtype after every op (fp16) or once at the output (bf16).
- bf16 output rounding is done with integer bit manipulation (round-to-nearest
  -even emulated on the fp32 bit pattern) because float truncations inside
  fused fp32 chains are not reliably preserved by the backend codegen.
- SiLU computes x * sigmoid(x) in fp32 and rounds once, matching
  torch.nn.functional.silu.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_NUM_GROUPS = 32
_MAX_BLOCK = 8192
_MAX_BLOCK_FP32 = 4096
_MIN_BLOCK = 16
_TILE_MAX_ELEMS = 8192  # resident stats-tile budget (192KB UB, x2 multibuffer)
# fp32 epilogue mode: False evaluates the merged single-coefficient form
# (see _merge_pairs); True restores the exact double-double chain. The
# merged form was
# measured to keep DD-level accuracy in the cancellation regions (the residual
# carries the pair-quantization error) at a fraction of the flops.
_DD_EPILOGUE = False
_EPI_RESIDENT_FP32 = 8192  # full-tile fp32 epilogue budget, merged single-coeff
                           # chain (halved live tiles vs double-double); step
                           # down 8192->4096->2048 on backend ub overflow
_EPI_RESIDENT_FP32_DD = 2048  # full-tile fp32 epilogue budget, double-double
_EPI_RESIDENT_HALF = 4096  # full-tile fp16/bf16 epilogue budget
_SPLIT_MIN_ELEMS = 49152  # fp32 streaming groups at/above this size take the
                          # stats+apply split pipeline (HW-slice parallelism)


@triton.jit
def _bf16_rne(v):
    """Round fp32 -> bf16 (round-to-nearest-even) via integer ops.

    Works on the fp32 bit pattern, so the rounding cannot be optimized away
    or changed to a truncated convert by the backend. Returns the bf16 value
    embedded in fp32 (low 16 bits zero), which is exact for further fp32 math.
    """
    u = v.to(tl.int32, bitcast=True)
    u = (u + 32767 + ((u >> 16) & 1)) & -65536
    return u.to(tl.float32, bitcast=True)


@triton.jit
def _two_prod(a, b):
    """Dekker error-free product: a*b == p + e exactly (fp32, no overflow)."""
    p = a * b
    ca = 4097.0 * a  # 2^12 + 1
    ah = ca - (ca - a)
    al = a - ah
    cb = 4097.0 * b
    bh = cb - (cb - b)
    bl = b - bh
    e = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, e


@triton.jit
def _two_sum(a, b):
    """Knuth error-free sum: a+b == s + e exactly (fp32)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    return s, e


@triton.jit
def _coeff_pairs(rstd, w, b_, s, sh, mean):
    """Per-channel double-double coefficients as exact fp32 (hi, lo) pairs:

        A  = rstd * w * (1 + s)
        B' = b * (1 + s) + sh - mean * A

    Shared by the tile/streaming/apply kernels so the error-free chain
    (Dekker/Knuth helpers) is defined once.
    """
    m1h, m1l = _two_sum(1.0, s)
    a1, ea = _two_prod(rstd, w)
    a_hi, ea2 = _two_prod(a1, m1h)
    a_lo = (ea * m1h + ea2) + a1 * m1l
    b1, eb = _two_prod(b_, m1h)
    b_hi, seb = _two_sum(b1, sh)
    b_lo = (eb + seb) + b_ * m1l
    mm, em = _two_prod(mean, a_hi)
    bp_hi, sebp = _two_sum(b_hi, -mm)
    bp_lo = (sebp + b_lo) - (em + mean * a_lo)
    return a_hi, a_lo, bp_hi, bp_lo


@triton.jit
def _merge_pairs(a_hi, a_lo, bp_hi, bp_lo):
    """Merged single coefficients + exact merge residuals (Knuth TwoSum).

    a_se/b_se carry the pair-quantization error (u*|A|, u*|B'|), which
    dominates in the cancellation region when |A|, |B'| are large.
    """
    a_s, a_se = _two_sum(a_hi, a_lo)
    b_s, b_se = _two_sum(bp_hi, bp_lo)
    return a_s, a_se, b_s, b_se


@triton.jit
def _dtype_flags(x_ptr):
    """Compile-time dtype predicates derived from the input pointer type."""
    dt = x_ptr.dtype.element_ty
    return dt == tl.bfloat16, dt == tl.float32


@triton.jit
def _meta4(meta_ptr):
    """Unpack the shared meta tensor: hw, group_elems, eps, cpg."""
    hw = tl.load(meta_ptr).to(tl.int32)
    group_elems = tl.load(meta_ptr + 1).to(tl.int32)
    eps = tl.load(meta_ptr + 2)
    cpg = tl.load(meta_ptr + 3).to(tl.int32)
    return hw, group_elems, eps, cpg


@triton.jit
def _group_bases(x_ptr, y_ptr, pid, group_elems):
    """Group slab bases for x and y."""
    return x_ptr + pid * group_elems, y_ptr + pid * group_elems


@triton.jit
def _combine_partials(ws_ptr, gid, split, group_elems, eps):
    """Combine the group's split raw-moment partials in a fixed row order
    (fp32 scalars, same precision shape as the merged streaming pass plus
    split-1 extra adds); derives mean/var/rstd."""
    s1 = 0.0
    s2 = 0.0
    for i in range(split):
        row = (gid * split + i) * 2
        s1 += tl.load(ws_ptr + row + 0)
        s2 += tl.load(ws_ptr + row + 1)
    mean = s1 / group_elems
    var = s2 / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    return mean, rstd


@triton.jit
def _chan_params(weight_ptr, bias_ptr, scale_ptr, shift_ptr, gid, cpg, c):
    """Per-channel parameter loads: weight/bias by group, scale/shift by
    (batch, group) condition — b * C + g * cpg == gid * cpg."""
    cond_base = gid * cpg
    param_base = (gid % 32) * cpg
    w = tl.load(weight_ptr + param_base + c)
    b_ = tl.load(bias_ptr + param_base + c)
    s = tl.load(scale_ptr + cond_base + c).to(tl.float32)
    sh = tl.load(shift_ptr + cond_base + c).to(tl.float32)
    return w, b_, s, sh


@triton.jit
def _load_slice(ptr, offs, even, bound):
    """Slice load with the evenness-optimized masked/unmasked form."""
    if even:
        return tl.load(ptr + offs).to(tl.float32)
    return tl.load(ptr + offs, mask=offs < bound, other=0.0).to(tl.float32)


@triton.jit
def _store_slice(ptr, offs, y, even, bound):
    """Slice store with the evenness-optimized masked/unmasked form."""
    if even:
        tl.store(ptr + offs, y)
    else:
        tl.store(ptr + offs, y, mask=offs < bound)


@triton.jit
def _silu_fp32_any(v, a_hi, a_lo, bp_hi, bp_lo, a_s, a_se, b_s, b_se):
    """Per-element fp32 epilogue in both coefficient modes (module-selected).

    _DD_EPILOGUE=True restores the exact double-double chain; the merged
    form evaluates z as (v*a_s + b_s) + (v*a_se + b_se) (~5 flops/element).
    """
    if _DD_EPILOGUE:
        return _silu_fp32_dd(v, a_hi, a_lo, bp_hi, bp_lo)
    z = (v * a_s + b_s) + (v * a_se + b_se)
    return z * tl.sigmoid(z)


@triton.jit
def _silu_fp32_dd(v, a_hi, a_lo, bp_hi, bp_lo):
    """Per-element fp32 epilogue, exact double-double chain (error-free
    products/sums, one final rounding)."""
    p, e = _two_prod(v, a_hi)
    e2 = v * a_lo
    s2c, se2c = _two_sum(p, bp_hi)
    z = s2c + ((se2c + e) + (e2 + bp_lo))
    return z * tl.sigmoid(z)


@triton.jit
def _tile_tile(block_cpg, block_hw, hw, cpg):
    """Resident 2D tile offsets/masks for the small-group kernel.

    cpg is generally not a power of two and hw may not fill block_hw, so
    both axes are masked.
    """
    c_offs = tl.arange(0, block_cpg)
    c_valid = c_offs < cpg
    hw_offs = tl.arange(0, block_hw)
    xy_mask = c_valid[:, None] & (hw_offs[None, :] < hw)
    tile_offs = c_offs[:, None] * hw + hw_offs[None, :]
    return c_offs, c_valid, xy_mask, tile_offs


@triton.jit
def _tile_params(weight_ptr, bias_ptr, scale_ptr, shift_ptr, pid, cpg,
                 c_offs, c_valid):
    """Vectorized per-channel parameter loads for the resident tile.

    One [block_cpg] load per vector instead of cpg scalar loads; broadcast
    along HW in the epilogue.
    """
    cond_base = pid * cpg          # scale/shift offset: b * C + g * cpg == pid * cpg
    param_base = (pid % 32) * cpg  # weight/bias offset: g * cpg
    w = tl.load(weight_ptr + param_base + c_offs, mask=c_valid, other=0.0)
    b_ = tl.load(bias_ptr + param_base + c_offs, mask=c_valid, other=0.0)
    s = tl.load(scale_ptr + cond_base + c_offs, mask=c_valid, other=0.0).to(tl.float32)
    sh = tl.load(shift_ptr + cond_base + c_offs, mask=c_valid, other=0.0).to(tl.float32)
    return w, b_, s, sh


@triton.jit
def _silu_fp32_tile(v2, a_hi, a_lo, bp_hi, bp_lo, a_s, a_se, b_s, b_se):
    """2D-tile fp32 epilogue, both coefficient modes (module-selected).

    _DD_EPILOGUE=True restores the exact double-double chain (error-free
    products/sums, one final rounding). The merged form evaluates
    z = (v*a_s + b_s) + (v*a_se + b_se): the first sum is relatively exact
    (Sterbenz cancels it bit-exactly when z cancels); the small residual
    term survives the final addition.
    """
    if _DD_EPILOGUE:
        p, e = _two_prod(v2, a_hi[:, None])
        e2 = v2 * a_lo[:, None]
        s2c, se2c = _two_sum(p, bp_hi[:, None])
        z = s2c + ((se2c + e) + (e2 + bp_lo[:, None]))
    else:
        z = (v2 * a_s[:, None] + b_s[:, None]) + (v2 * a_se[:, None] + b_se[:, None])
    return z * tl.sigmoid(z)


@triton.jit
def _raw_moments(x_base, group_elems, eps, block, even):
    """Merged pass 1+2: single pass over x computing raw moments s1/s2
    (the sums of x and of x squared); mean is s1/N, var is s2/N minus
    mean squared.

    This reads x one time fewer than the two-pass structure. The raw form is
    used because every case routed here has O(1) variance (small-variance /
    constant inputs are handled by the tile kernel's exact two-pass on the
    resident tile): with |x| <= 2 the mean^2 subtraction removes at most
    ~4.6e-5 from ~1.33, so there is no catastrophic cancellation, while
    partial sums stay at the same magnitude as the two-pass accumulation.
    The shifted variant (sum(x - x[0])) was measured to inflate the mean
    error to ~5e-7 (its partial sums reach N*|mean-x0|), which pushed fp32
    cancellation-region cases past the comparator fallback.
    """
    s1_acc = 0.0
    s2_acc = 0.0
    for i in range(0, group_elems, block):
        offs = i + tl.arange(0, block)
        if even:
            v = tl.load(x_base + offs).to(tl.float32)
        else:
            v = tl.load(x_base + offs, mask=offs < group_elems, other=0.0).to(tl.float32)
        s1_acc += tl.sum(v)
        s2_acc += tl.sum(v * v)
    mean = s1_acc / group_elems
    var = s2_acc / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    return mean, rstd


@triton.jit
def _twopass_moments(x_base, group_elems, eps, block, even):
    """Two-pass fallback: exact centered variance (constexpr per-dtype revert
    if the merged raw-moment form ever regresses precision).

    Pass 1: group mean. Masked lanes contribute 0 via other=0.0. Pass 2:
    centered variance. Masked lanes must contribute exactly 0 after the mean
    subtraction, otherwise (0 - mean)^2 pollutes the sum.
    """
    acc = 0.0
    for i in range(0, group_elems, block):
        offs = i + tl.arange(0, block)
        if even:
            v = tl.load(x_base + offs).to(tl.float32)
        else:
            v = tl.load(x_base + offs, mask=offs < group_elems, other=0.0).to(tl.float32)
        acc += tl.sum(v)
    mean = acc / group_elems

    acc2 = 0.0
    for i in range(0, group_elems, block):
        offs = i + tl.arange(0, block)
        if even:
            v = tl.load(x_base + offs).to(tl.float32)
            d = v - mean
        else:
            mask = offs < group_elems
            v = tl.load(x_base + offs, mask=mask, other=0.0).to(tl.float32)
            d = tl.where(mask, v - mean, 0.0)
        acc2 += tl.sum(d * d)
    var = acc2 / group_elems
    rstd = 1.0 / tl.sqrt(var + eps)
    return mean, rstd


@triton.jit
def _silu_bf16_tile(v2, mean, rstd, w, b_, s, sh):
    """2D-tile bf16 epilogue: fp32 chain, single integer-RNE rounding at the
    output only (keeps the final convert exact against backend RTZ
    degradation)."""
    t = (v2 - mean) * rstd * w[:, None] + b_[:, None]
    m1 = 1.0 + s
    z = t * m1[:, None] + sh[:, None]
    return _bf16_rne(z * tl.sigmoid(z)).to(tl.bfloat16)


@triton.jit
def _silu_fp16_tile(v2, mean, rstd, w, b_, s, sh):
    """2D-tile fp16 epilogue: one rounding per step, SiLU in fp32."""
    t = ((v2 - mean) * rstd * w[:, None] + b_[:, None]).to(tl.float16)
    m1 = (1.0 + s).to(tl.float16)
    prod = (t.to(tl.float32) * m1.to(tl.float32)[:, None]).to(tl.float16)
    z = (prod.to(tl.float32) + sh[:, None]).to(tl.float16)
    return (z.to(tl.float32) * tl.sigmoid(z.to(tl.float32))).to(tl.float16)


@triton.jit
def _silu_bf16_scalar(v, mean, rstd, w, b_, s, sh):
    """Per-element bf16 epilogue: fp32 chain, single integer-RNE rounding."""
    t = (v - mean) * rstd * w + b_
    m1 = 1.0 + s
    z = t * m1 + sh
    return _bf16_rne(z * tl.sigmoid(z)).to(tl.bfloat16)


@triton.jit
def _silu_fp16_scalar(v, mean, rstd, w, b_, s, sh):
    """Per-element fp16 epilogue: one rounding per step, SiLU in fp32."""
    t = ((v - mean) * rstd * w + b_).to(tl.float16)
    m1 = (1.0 + s).to(tl.float16)
    prod = (t.to(tl.float32) * m1.to(tl.float32)).to(tl.float16)
    z = (prod.to(tl.float32) + sh).to(tl.float16)
    return (z.to(tl.float32) * tl.sigmoid(z.to(tl.float32))).to(tl.float16)


@triton.jit
def _fused_groupnorm_mul_add_silu_tile_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    meta_ptr,
    block_cpg: tl.constexpr,
    block_hw: tl.constexpr,
    block_ep: tl.constexpr,
):
    pid = tl.program_id(0)
    hw, group_elems, eps, cpg = _meta4(meta_ptr)
    x_base, y_base = _group_bases(x_ptr, y_ptr, pid, group_elems)
    is_bf16, is_fp32 = _dtype_flags(x_ptr)

    # The whole group as one 2D tile ([block_cpg, block_hw]); cpg is generally
    # not a power of two and hw may not fill block_hw, so both axes are masked.
    c_offs, c_valid, xy_mask, tile_offs = _tile_tile(block_cpg, block_hw, hw, cpg)

    # Pass 1: group mean from the resident tile (padding contributes 0).
    v = tl.load(x_base + tile_offs, mask=xy_mask, other=0.0).to(tl.float32)
    mean = tl.sum(tl.sum(v, axis=1), axis=0) / group_elems

    # Pass 2: centered variance on the same tile. Masked lanes must contribute
    # exactly 0 after the mean subtraction, otherwise (0 - mean)^2 pollutes it.
    d = tl.where(xy_mask, v - mean, 0.0)
    var = tl.sum(tl.sum(d * d, axis=1), axis=0) / group_elems
    rstd = 1.0 / tl.sqrt(var + eps)

    # Vectorized per-channel parameters: one [block_cpg] load per vector
    # instead of cpg scalar loads; broadcast along HW in the epilogue.
    w, b_, s, sh = _tile_params(weight_ptr, bias_ptr, scale_ptr, shift_ptr,
                                pid, cpg, c_offs, c_valid)

    if is_fp32:
        # Per-channel double-double coefficients (as [block_cpg] vector pairs):
        #   A  is rstd times w times (1+s)   (exact pair)
        #   B' is b times (1+s) plus sh, minus mean times A (exact pair)
        # so the epilogue evaluates z as v times A plus B-prime with one final
        # rounding, immune to cancellation at near-zero outputs (no usable fma
        # and no fp64 vector casts on this backend).
        a_hi, a_lo, bp_hi, bp_lo = _coeff_pairs(rstd, w, b_, s, sh, mean)
        # Merged single-coefficient form (_DD_EPILOGUE=False): collapse each
        # exact pair into one fp32 scalar plus its exact merge residual
        # (a_se tracks the a_hi plus a_lo rounding residual, kept via Knuth
        # TwoSum). The epilogue evaluates z as v times a_s plus b_s, plus
        # v times a_se plus b_se: the correction term
        # carries the pair-quantization error (u*|A|, u*|B'|), which plain
        # single coefficients would drop and which dominates in the
        # cancellation region when |A|,|B'| are large (zero/small-variance
        # inputs, rstd ~ 1/sqrt(eps)); with it the merged form tracks the
        # double-double result at ~5 instead of ~28 flops/element and keeps
        # fewer live tiles (resident full-tile epilogue fits the UB budget).
        a_s, a_se, b_s, b_se = _merge_pairs(a_hi, a_lo, bp_hi, bp_lo)

    # Epilogue. block_ep == block_hw keeps the single-load fast path (loop
    # runs once on the resident tile); otherwise re-read the group in HW
    # chunks small enough for the epilogue live tiles (L2-resident reload).
    for i in range(0, hw, block_ep):
        hw2 = i + tl.arange(0, block_ep)
        offs = c_offs[:, None] * hw + hw2[None, :]
        msk = c_valid[:, None] & (hw2[None, :] < hw)
        if block_ep == block_hw:
            v2 = v
        else:
            v2 = tl.load(x_base + offs, mask=msk, other=0.0).to(tl.float32)

        if is_fp32:
            y = _silu_fp32_tile(v2, a_hi, a_lo, bp_hi, bp_lo,
                                a_s, a_se, b_s, b_se)
        elif is_bf16:
            y = _silu_bf16_tile(v2, mean, rstd, w, b_, s, sh)
        else:
            y = _silu_fp16_tile(v2, mean, rstd, w, b_, s, sh)

        tl.store(y_base + offs, y, mask=msk)


@triton.jit
def _fused_groupnorm_mul_add_silu_stats_kernel(
    x_ptr,
    ws_ptr,
    group_elems,
    split: tl.constexpr,
    block: tl.constexpr,
):
    # Stats half of the split pipeline for large fp32 groups: one program per
    # (group, 1/split slab slice). The slice is a contiguous range of the
    # group slab, so the raw moments accumulate in the same linear element
    # order as the merged streaming pass, just partitioned. Each program
    # writes its fp32 (s1, s2) partials to its own workspace row; the apply
    # kernel combines the split rows of the group. Kernels are ordered on one
    # stream, so workspace rows are fully written before any read: no
    # atomics, no cross-program barriers.
    pid = tl.program_id(0)
    even = (group_elems % (split * block)) == 0
    gid = pid // split
    sid = pid % split
    begin = gid * group_elems + (group_elems * sid) // split
    end = gid * group_elems + (group_elems * (sid + 1)) // split
    span = end - begin

    s1_acc = 0.0
    s2_acc = 0.0
    for i in range(0, span, block):
        offs = begin + i + tl.arange(0, block)
        if even:
            v = tl.load(x_ptr + offs).to(tl.float32)
        else:
            v = tl.load(x_ptr + offs, mask=offs < end, other=0.0).to(tl.float32)
        s1_acc += tl.sum(v)
        s2_acc += tl.sum(v * v)
    tl.store(ws_ptr + pid * 2 + 0, s1_acc)
    tl.store(ws_ptr + pid * 2 + 1, s2_acc)


@triton.jit
def _fused_groupnorm_mul_add_silu_apply_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    ws_ptr,
    meta_ptr,
    split: tl.constexpr,
    block: tl.constexpr,
):
    # Apply half of the split pipeline, same grid as the stats kernel: combine
    # the group's split raw-moment partials in a fixed row order (fp32 scalars,
    # same precision shape as the merged streaming pass plus split-1 extra
    # adds), derive mean/var/rstd, then run the unchanged channel-wise
    # epilogue chains on this program's 1/split HW slice of every channel.
    pid = tl.program_id(0)
    hw, group_elems, eps, cpg = _meta4(meta_ptr)
    even = (hw % (split * block)) == 0
    gid = pid // split
    sid = pid % split
    hw_begin = (hw * sid) // split
    hw_end = (hw * (sid + 1)) // split
    x_base, y_base = _group_bases(x_ptr, y_ptr, gid, group_elems)
    is_bf16, is_fp32 = _dtype_flags(x_ptr)

    mean, rstd = _combine_partials(ws_ptr, gid, split, group_elems, eps)

    for c in range(cpg):
        w, b_, s, sh = _chan_params(weight_ptr, bias_ptr, scale_ptr,
                                    shift_ptr, gid, cpg, c)
        c_x = x_base + c * hw
        c_y = y_base + c * hw

        if is_fp32:
            # Double-double coefficient prep kept verbatim so _DD_EPILOGUE=True
            # restores the exact-pair chain; the merged single coefficients
            # (with exact merge residuals) cost two Knuth TwoSums per channel,
            # amortized over the HW slice.
            a_hi, a_lo, bp_hi, bp_lo = _coeff_pairs(rstd, w, b_, s, sh, mean)
            a_s, a_se, b_s, b_se = _merge_pairs(a_hi, a_lo, bp_hi, bp_lo)

        span = hw_end - hw_begin
        for i in range(0, span, block):
            offs = hw_begin + i + tl.arange(0, block)
            v = _load_slice(c_x, offs, even, hw_end)

            if is_fp32:
                y = _silu_fp32_any(v, a_hi, a_lo, bp_hi, bp_lo,
                                   a_s, a_se, b_s, b_se)
            elif is_bf16:
                y = _silu_bf16_scalar(v, mean, rstd, w, b_, s, sh)
            else:
                y = _silu_fp16_scalar(v, mean, rstd, w, b_, s, sh)

            _store_slice(c_y, offs, y, even, hw_end)


@triton.jit
def _fused_groupnorm_mul_add_silu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    scale_ptr,
    shift_ptr,
    y_ptr,
    meta_ptr,
    block: tl.constexpr,
    onepass_stats: tl.constexpr,
):
    pid = tl.program_id(0)
    pid = tl.program_id(0)
    hw, group_elems, eps, cpg = _meta4(meta_ptr)
    x_base, y_base = _group_bases(x_ptr, y_ptr, pid, group_elems)
    is_bf16, is_fp32 = _dtype_flags(x_ptr)
    even = hw % block == 0

    if onepass_stats:
        # Merged pass 1+2: single pass over x computing raw moments s1/s2
        # (sums of x and x*x); mean is s1/N, var is s2/N minus mean squared.
        # This reads x one time fewer than the two-pass structure. The raw
        # form is used because every case routed here has O(1) variance
        # (small-variance / constant inputs are handled by the tile kernel's
        # exact two-pass on the resident tile): with |x| <= 2 the mean^2
        # subtraction removes at most ~4.6e-5 from ~1.33, so there is no
        # catastrophic cancellation, while partial sums stay at the same
        # magnitude as the two-pass accumulation. The shifted variant
        # (sum(x - x[0])) was measured to inflate the mean error to ~5e-7
        # (its partial sums reach N*|mean-x0|), which pushed fp32
        # cancellation-region cases past the comparator fallback.
        mean, rstd = _raw_moments(x_base, group_elems, eps, block, even)
    else:
        # Two-pass fallback (constexpr switch, per-dtype revert if the merged
        # form ever regresses precision).
        mean, rstd = _twopass_moments(x_base, group_elems, eps, block, even)

    # Normalize + affine in fp32, cast to x.dtype, then the condition
    # modulation chain with one rounding per step, then SiLU with fp32 math.
    for c in range(cpg):
        w, b_, s, sh = _chan_params(weight_ptr, bias_ptr, scale_ptr,
                                    shift_ptr, pid, cpg, c)
        c_x = x_base + c * hw
        c_y = y_base + c * hw
        for i in range(0, hw, block):
            offs = i + tl.arange(0, block)
            v = _load_slice(c_x, offs, even, hw)

            if is_fp32:
                # The NPU backend has no usable fma (libdevice.fma lowers to
                # mul+add) and fp64 vector math is unsupported, so the fp32
                # path evaluates the whole affine + modulation chain as
                # z, v times A plus B-prime, with per-channel double-double
                # scalars: mathematically identical to the direct chain of
                # subtract, scale, affine and modulation ops, but reaches the
                # fp64-oracle value with a single final rounding, immune to
                # cancellation at near-zero outputs.
                a_hi, a_lo, bp_hi, bp_lo = _coeff_pairs(rstd, w, b_, s, sh, mean)
                a_s, a_se, b_s, b_se = _merge_pairs(a_hi, a_lo, bp_hi, bp_lo)
            if is_fp32:
                y = _silu_fp32_any(v, a_hi, a_lo, bp_hi, bp_lo,
                                   a_s, a_se, b_s, b_se)
            elif is_bf16:
                y = _silu_bf16_scalar(v, mean, rstd, w, b_, s, sh)
            else:
                y = _silu_fp16_scalar(v, mean, rstd, w, b_, s, sh)

            _store_slice(c_y, offs, y, even, hw)


def _make_contiguous5(x, weight, bias, scale, shift):
    """Ensure all five inputs are contiguous."""
    return (
        x if x.is_contiguous() else x.contiguous(),
        weight if weight.is_contiguous() else weight.contiguous(),
        bias if bias.is_contiguous() else bias.contiguous(),
        scale if scale.is_contiguous() else scale.contiguous(),
        shift if shift.is_contiguous() else shift.contiguous(),
    )


def _epi_block(x, block_cpg, block_hw):
    """Epilogue chunk width: keeps the per-chunk live tiles inside the UB
    budget; when the whole tile fits, block_ep == block_hw selects the
    resident fast path (single load, no re-read). The budget follows the
    fp32 epilogue mode (the double-double chain keeps ~2x live tiles)."""
    if x.dtype == torch.float32:
        epi_limit = _EPI_RESIDENT_FP32_DD if _DD_EPILOGUE else _EPI_RESIDENT_FP32
    else:
        epi_limit = _EPI_RESIDENT_HALF
    return max(_MIN_BLOCK, min(block_hw, epi_limit // block_cpg))


def _split_pipeline(hw, group_elems, x, batch):
    """F1.compute_epilogue_split sizing for large streaming fp32 groups: the
    single streaming kernel leaves 32-64 programs on 40 vector cores with the
    statistics and epilogue phases serialized inside each program. Split into
    stats + apply kernels on one stream; each of the B*32*split programs
    handles a contiguous 1/split slice (split by HW size: 65536 -> 4,
    16384/4096 -> 2). Traffic stays 2 reads + 1 write; the fp32
    [B*32*split, 2] workspace is <= 4KB."""
    split = 4 if hw >= 32768 else 2
    chunk = group_elems // split
    block_stats = max(_MIN_BLOCK, min(_MAX_BLOCK_FP32,
                                      triton.next_power_of_2(max(1, chunk))))
    slice_len = hw // split
    block_apply = max(_MIN_BLOCK, min(_MAX_BLOCK_FP32,
                                      triton.next_power_of_2(max(1, slice_len))))
    ws = torch.empty((batch * _NUM_GROUPS * split, 2),
                     dtype=torch.float32, device=x.device)
    return split, block_stats, block_apply, ws


def fused_groupnorm_mul_add_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Return SiLU(GroupNorm(x, 32, weight, bias, eps) * (1 + scale) + shift).

    The wrapper only prepares metadata, the output tensor and (for large fp32
    groups) the tiny stats workspace; every numerical step runs inside the
    Triton kernels above. Small groups (rounded tile within the UB budget)
    dispatch to the single-tile kernel, large fp32 groups to the stats+apply
    split pipeline, everything else to the streaming kernel with merged
    raw-moment statistics.
    """
    x, weight, bias, scale, shift = _make_contiguous5(x, weight, bias, scale, shift)

    output = torch.empty_like(x)
    batch, channels, height, width = x.shape
    hw = height * width
    cpg = channels // _NUM_GROUPS
    group_elems = cpg * hw
    if group_elems == 0:
        return output

    is_fp32 = x.dtype == torch.float32
    grid = (batch * _NUM_GROUPS,)

    # F5.schedule_routing: route by the power-of-two-rounded 2D tile size.
    block_cpg = triton.next_power_of_2(cpg)
    block_hw = max(_MIN_BLOCK, triton.next_power_of_2(hw))
    if block_cpg * block_hw <= _TILE_MAX_ELEMS:
        block_ep = _epi_block(x, block_cpg, block_hw)
        tile_meta = (hw, group_elems, eps, cpg)
        _fused_groupnorm_mul_add_silu_tile_kernel[grid](
            x, weight, bias, scale, shift, output, tile_meta,
            block_cpg=block_cpg, block_hw=block_hw, block_ep=block_ep,
        )
        return output

    if is_fp32 and group_elems >= _SPLIT_MIN_ELEMS:
        split, block_stats, block_apply, ws = _split_pipeline(
            hw, group_elems, x, batch)
        split_grid = (batch * _NUM_GROUPS * split,)
        split_meta = (hw, group_elems, eps, cpg)
        _fused_groupnorm_mul_add_silu_stats_kernel[split_grid](
            x, ws, group_elems, split=split, block=block_stats,
        )
        _fused_groupnorm_mul_add_silu_apply_kernel[split_grid](
            x, weight, bias, scale, shift, output, ws, split_meta,
            split=split, block=block_apply,
        )
        return output

    # Streaming path: everything not routed to the tile or split pipelines.
    block = max(_MIN_BLOCK, min(triton.next_power_of_2(hw), _MAX_BLOCK))
    if is_fp32:
        # The fp32 epilogue keeps more live tiles; stay within UB.
        block = min(block, _MAX_BLOCK_FP32)
    stream_meta = (hw, group_elems, eps, cpg)
    _fused_groupnorm_mul_add_silu_kernel[grid](
        x, weight, bias, scale, shift, output, stream_meta,
        block=block, onepass_stats=True,
    )
    return output
