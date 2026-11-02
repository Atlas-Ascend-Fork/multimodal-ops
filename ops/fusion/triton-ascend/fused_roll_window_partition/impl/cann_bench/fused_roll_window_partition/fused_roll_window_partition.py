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

"""Triton-Ascend implementation of the CANN Bench FusedRollWindowPartition interface.

The operator fuses Swin Transformer's cyclic shift (roll) with window
partition.  Because both stages are pure data movement, the whole operator is
a single pass of bit-exact copy:

    windows[n, u, v, c] = x[b, (i*M + u + s) % H, (j*M + v + s) % W, c]
    n = (b * (H // M) + i) * (W // M) + j

Key structural facts used by the kernel (one output row per grid-stride
iteration):

1. One output row (fixed ``u``, spanning ``v`` and ``c``) covers ``M * C``
   elements that are contiguous in the output tensor, so the store side needs
   no per-element address computation.
2. The source row index ``h = (i*M + u + s) % H`` is constant along the row,
   so the H-wrap is resolved once with a scalar modulo.
3. Along the row the source columns advance as ``w0 + v`` and wrap at most
   once back to the start of the same source row.  Instead of correcting the
   address with a per-element select (which fragments MTE2 burst transfers),
   the row is copied as two strictly contiguous segments:

       segment A: x[row_base + k]              for k in [0, la)
       segment B: x[row_base - W*C + la + k]   for k in [0, lb)

   with ``la = min(break_elem, ROW_ELEMS)`` and ``lb = ROW_ELEMS - la``.
   Every load/store address within a segment is strictly sequential, so the
   hardware can merge requests into wide bursts.  When ``s == 0`` (and
   whenever the wrap point lies past the row end) ``la >= ROW_ELEMS`` holds
   and segment B is skipped naturally.
4. The grid is fixed to the physical Vector core count and each program walks
   the rows with a grid-stride loop, so per-program launch/scheduling overhead
   is paid once per core instead of once per row.

Row path vs window-tile path (F5.fusion_boundary_routing):

5. The row path above issues one MTE2 load+store transaction pair per output
   row (per segment).  For shape groups with many rows per window (M large)
   and wide rows, that per-transaction fixed cost dominates over the raw
   bandwidth time.  A second kernel therefore processes one whole window per
   grid-stride iteration: the (up to) four wrap-quadrant source rectangles of
   the window are copied as contiguous 2D tiles of TU rows × TC columns, so
    TU rows share one transaction pair instead of paying one each.  A
      host-side scalar-metadata predicate (window count, UB budget)
      selects the tile path only where it measurably wins and where the
      window fits the UB budget (whole-window tile, or the per-chunk TU*TC
      working set for row widths up to a measured cap); every other shape
      takes the unchanged row path.  Zero-shift groups additionally run a
      SHIFT_ZERO-specialized tile variant: with ``s == 0`` every window
      satisfies ``la == ROW_ELEMS`` and ``u1 == M``, so the compiled kernel
      keeps only quadrant A and drops the three post-wrap quadrant loops
      and their scalar wrap-point computations at compile time.
     All routing metadata are Python scalars passed as kernel arguments — no
     host-side tensors and no H2D transfers.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Physical Vector core count of Ascend910B3 (device_info.json: 20 AI Cores,
# Cube 20 + Vector 40).
_NUM_VECTOR_CORES = 40


def _select_block(row_elems: int) -> int:
    """Host-side BLOCK routing by row width.

    Keeps one transfer per row whenever possible while staying well inside the
    UB budget: fp32 4096 * 4B * 4 buffers = 64KB < 96KB per-buffer-pipeline
    budget (192KB UB, compiler multibuffering), fp16/bf16 half of that.
    """
    if row_elems <= 1024:
        return 1024
    if row_elems <= 2048:
        return 2048
    return 4096


# F5.fusion_boundary_routing: eligibility of the 2D per-window tile path.
# The predicates are ordinary shape/dtype metadata (no benchmark identity):
#   - at least _TILE_MIN_WINDOWS windows.  Lowered from 40 to 32 (iter5:
#     nwin 32 groups win with 80% utilization) and now to 12 after
#     forced-tile cold-L2 measurements on the remaining groups: the tile
#     kernel groups TU rows per transaction pair, so the transaction-count
#     reduction outweighs the reduced core utilization even at
#     min(nwin, 40) reaching 12-18 programs (nwin 16 zero-shift [2,14,28,96]
#     gains 48%, nwin 16 [4,14,14,384] gains 11-16%, nwin 12 [2,14,21,65]
#     gains 29%);
#   - row width at least _TILE_MIN_ROW_ELEMS elements.  Lowered from 1536
#     to 1 after forced-tile cold-L2 measurements refuted the earlier
#     "narrow rows gain nothing" assumption (nwin 64 with row_elems 672
#     gains 37%, row_elems 896 gains 27%);
#   - UB budget, two admission tiers.  Primary (unchanged): the whole
#     window tile (M rows x row_elems elements) fits the 96KB per-kernel
#     budget.  Secondary (this round): when the whole-window tile exceeds
#     the budget, the group is still admitted if the kernel's actual
#     per-chunk in-flight working set TU*TC*elem_bytes fits the budget AND
#     the row width is at most _TILE_MAX_ROW_ELEMS_CHUNKED.  The kernel
#     never holds more than TU*TC elements in flight, so per-chunk is the
#     true occupancy bound (the whole-window check was only a conservative
#     proxy).  Evidence (develop/iter5 T3 forced-tile, cold-L2 medians,
#     bitwise exact): [2,48,48,256] fp32 (M 12, row_elems 3072, three
#     TC 1024 column iterations per quadrant row) runs 24.6% faster than
#     its row path, while [4,24,24,512] fp32 (row_elems 6144, six column
#     iterations) regresses by 21.9% because the extra iterations negate
#     the transaction-count reduction; the _TILE_MAX_ROW_ELEMS_CHUNKED cap
#     admits the former and keeps the latter on the row path.
_TILE_MIN_WINDOWS = 12
_TILE_MIN_ROW_ELEMS = 1
_TILE_UB_BUDGET_BYTES = 96 * 1024
_TILE_MAX_ROW_ELEMS_CHUNKED = 3072

# Tile geometry: TU rows per 2D tile chunk, TC columns per chunk (both
# constexpr, power of two).  TU 8 / TC 1024 is the measured-best
# configuration on the routed SwinB-384 group (cold-L2 medians across
# fp16/fp32/bf16 all improve 10-14% over TU 4; TC 2048 loses to masking
# lane waste); per-chunk working set of 8*1024 elements stays at 16KB
# (2B dtypes) / 32KB (fp32), well inside the 192KB UB.
_TILE_TU = 8


def _select_tile_cols(row_elems: int) -> int:
    """Column width TC of one 2D tile chunk (power of two)."""
    return 512 if row_elems <= 512 else 1024


def _use_tile_kernel(num_windows: int, m: int, row_elems: int,
                     elem_bytes: int) -> bool:
    """Route many-window shape groups to the 2D tile kernel.

    Primary admission (unchanged): whole-window tile inside the UB budget.
    Secondary admission: whole-window tile over budget but the kernel's
    actual per-chunk working set TU*TC*elem_bytes inside the budget for
    rows up to _TILE_MAX_ROW_ELEMS_CHUNKED (measured-forced-tile winner,
    e.g. [2,48,48,256] fp32), while row_elems=6144 groups (measured
    forced-tile regression) remain excluded.
    """
    if num_windows < _TILE_MIN_WINDOWS or row_elems < _TILE_MIN_ROW_ELEMS:
        return False
    if m * row_elems * elem_bytes <= _TILE_UB_BUDGET_BYTES:
        return True
    return (
        row_elems <= _TILE_MAX_ROW_ELEMS_CHUNKED
        and _TILE_TU * _select_tile_cols(row_elems) * elem_bytes
        <= _TILE_UB_BUDGET_BYTES
    )


@triton.jit
def _roll_window_partition_kernel(x_ptr, out_ptr, meta_ptr,
                                  block: tl.constexpr):
    """Row-stride kernel: one output row per grid-stride iteration.

    meta = [m, s, height, width, channels, win_cnt, rows, row_elems]. Each
    row is copied as two strictly contiguous segments around the W-wrap
    point so MTE2 burst merging is preserved; segment B is skipped naturally
    when the wrap point lies past the row end (e.g. s == 0).
    """
    m = tl.load(meta_ptr).to(tl.int32)
    s = tl.load(meta_ptr + 1).to(tl.int32)
    height = tl.load(meta_ptr + 2).to(tl.int32)
    width = tl.load(meta_ptr + 3).to(tl.int32)
    channels = tl.load(meta_ptr + 4).to(tl.int32)
    win_w = tl.load(meta_ptr + 5).to(tl.int32)
    win_cnt = tl.load(meta_ptr + 6).to(tl.int32)
    rows_total = tl.load(meta_ptr + 7).to(tl.int32)
    row_elems = tl.load(meta_ptr + 8).to(tl.int32)
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    wc = width * channels

    for row in range(pid, rows_total, num_core):
        # Split the flat row id into (batch, window row, window col,
        # in-window row).
        n = row // m
        u = row - n * m
        b = n // win_cnt
        rem = n - b * win_cnt
        i = rem // win_w
        j = rem - i * win_w

        # Source row: H-wrap resolved once with a scalar modulo (0 <= j*M + S
        # < W always holds because 0 <= S < M <= W, so the column start never
        # wraps).
        h = (i * m + u + s) % height
        w0 = j * m + s
        row_base = (b * height + h) * wc + w0 * channels
        break_elem = (width - w0) * channels

        la = tl.minimum(break_elem, row_elems)
        lb = row_elems - la

        out_base = row * row_elems

        # Segment A: strictly contiguous copy of the pre-wrap part
        # x[row_base, row_base + la).
        for chunk_a in range(0, (la + block - 1) // block):
            idx = chunk_a * block + tl.arange(0, block)
            mask_a = idx < la
            val = tl.load(x_ptr + row_base + idx, mask=mask_a)
            tl.store(out_ptr + out_base + idx, val, mask=mask_a)

        # Segment B: strictly contiguous copy of the post-wrap part
        # x[row_base - W*C + la, +lb) (column 0 onwards of the same source
        # row).  Skipped entirely when la >= ROW_ELEMS (e.g. s == 0).
        base_b = row_base - wc
        for chunk_b in range(0, (lb + block - 1) // block):
            idx = chunk_b * block + tl.arange(0, block)
            mask_b = idx < lb
            val = tl.load(x_ptr + base_b + la + idx, mask=mask_b)
            tl.store(out_ptr + out_base + la + idx, val, mask=mask_b)


@triton.jit
def _roll_window_partition_tile_kernel(x_ptr, out_ptr, meta_ptr,
                                       tu: tl.constexpr, tc: tl.constexpr,
                                       shift_zero: tl.constexpr):
    """2D per-window tile copy (one window per grid-stride iteration).

    meta = [m, s, height, width, channels, win_w, win_cnt, nwin, row_elems].

    Window ``n`` maps to source rectangle top-left ``(h0, w0)`` with
    ``h0 = i*M + S``, ``w0 = j*M + S``.  The rolled source of the window is
    the plain rectangle ``x[b, h0:h0+M, w0:w0+M, :]`` with at most one wrap
    in H (last window row) and one in W (last window column), so it splits
    into four quadrants around the break points

        la = min((W - w0) * C, ROW_ELEMS)   # column break (W-wrap)
        u1 = min(H - h0, M)                 # row break (H-wrap)

    The four quadrants are walked by a static unrolled loop; each quadrant is
    a strided 2D block (row stride W*C on the load side, M*C on the store
    side; columns contiguous) copied in TU x TC tiles.  Empty quadrants
    (no W-wrap when la == ROW_ELEMS, no H-wrap when u1 == M, e.g. s == 0)
    degenerate to zero-trip loops.  Every in-tile address is strictly
    sequential along columns, so MTE2 burst merging is preserved while TU
    rows share one transaction pair.

    SHIFT_ZERO (compile-time, host passes ``shift_size == 0``) walks
    quadrant A only: the three post-wrap quadrant iterations are eliminated
    from the compiled kernel entirely (constexpr loop bound); the general
    variant walks all four quadrants with the unchanged body.
    """
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)
    m = tl.load(meta_ptr).to(tl.int32)
    s = tl.load(meta_ptr + 1).to(tl.int32)
    height = tl.load(meta_ptr + 2).to(tl.int32)
    width = tl.load(meta_ptr + 3).to(tl.int32)
    channels = tl.load(meta_ptr + 4).to(tl.int32)
    win_w = tl.load(meta_ptr + 5).to(tl.int32)
    win_cnt = tl.load(meta_ptr + 6).to(tl.int32)
    nwin = tl.load(meta_ptr + 7).to(tl.int32)
    row_elems = tl.load(meta_ptr + 8).to(tl.int32)
    wc = width * channels
    rr = tl.arange(0, tu)
    cc = tl.arange(0, tc)
    for n in range(pid, nwin, num_core):
        b = n // win_cnt
        rem = n - b * win_cnt
        i = rem // win_w
        j = rem - i * win_w
        out_base = n * m * row_elems
        # General source rectangle: with s == 0 the wrap breaks degenerate
        # (la reaches row_elems, u1 reaches m), so the general formula covers
        # the zero-shift case bit-exactly; only the quadrant count shrinks.
        w0 = j * m + s
        h0 = i * m + s
        src = (b * height + h0) * wc + w0 * channels
        la = tl.minimum((width - w0) * channels, row_elems)
        u1 = tl.minimum(height - h0, m)
        # Quadrants A-D walked by a static unrolled loop: (r_lo, r_hi,
        # c_lo, c_hi) per quadrant with the matching source-base adjustment;
        # empty quadrants degenerate to zero-trip loops.  SHIFT_ZERO
        # (compile-time, host passes shift_size == 0) walks quadrant A only:
        # the three post-wrap quadrant iterations are eliminated from the
        # compiled kernel entirely (constexpr loop bound; the arithmetic
        # below folds the shift to zero at compile time).
        n_quadrants = 4 - 3 * shift_zero
        for q in tl.static_range(n_quadrants):
            qh = q // 2
            qo = q % 2
            r_lo = u1 * qh
            r_hi = u1 + (m - u1) * qh
            c_lo = la * qo
            c_hi = la + (row_elems - la) * qo
            src_q = src - height * wc * qh - wc * qo
            # Tile iterations flattened to a single loop (row-major tile
            # order, identical to the original c0-outer/t0-inner traversal).
            rows_tiles = (r_hi - r_lo + tu - 1) // tu
            n_tiles = ((c_hi - c_lo + tc - 1) // tc) * rows_tiles
            for t in range(0, n_tiles):
                t0 = t % rows_tiles
                cpos = c_lo + (t // rows_tiles) * tc
                r0 = r_lo + t0 * tu
                rnum = tl.minimum(r_hi - r0, tu)
                cnum = tl.minimum(c_hi - cpos, tc)
                m2 = (rr[:, None] < rnum) & (cc[None, :] < cnum)
                val = tl.load(x_ptr + src_q + (r0 + rr)[:, None] * wc
                              + cpos + cc[None, :], mask=m2)
                tl.store(out_ptr + out_base + (r0 + rr)[:, None] * row_elems
                         + cpos + cc[None, :], val, mask=m2)


def fused_roll_window_partition(
    x: torch.Tensor,
    window_size: int,
    shift_size: int = 0,
) -> torch.Tensor:
    """Fused cyclic shift and window partition for NHWC features.

    Args:
        x: Contiguous NHWC features ``[B, H, W, C]``.
        window_size: Square window side ``M``; ``H`` and ``W`` must be
            divisible by ``M``.
        shift_size: Up/left cyclic displacement ``s``, with ``0 <= s < M``.

    Returns:
        Contiguous windows ``[B * (H // M) * (W // M), M, M, C]`` on the same
        device and with the same dtype as ``x``.  Windows are ordered by
        batch, window row, then window column.

    Dispatch (F5.fusion_boundary_routing): shape groups with many windows,
    wide rows and a whole-window tile inside the UB budget take the 2D
    per-window tile kernel; everything else takes the row-stride kernel.
    Both paths are bit-exact copies of the same address mapping.  The tile
    launch additionally compiles a SHIFT_ZERO-specialized variant when
    ``shift_size == 0`` (host scalar predicate, no runtime cost on the
    general variant).
    """
    if not x.is_contiguous():
        x = x.contiguous()

    batch, height, width, channels = x.shape
    m = window_size
    win_h = height // m
    win_w = width // m
    row_elems = m * channels
    rows = batch * win_h * win_w * m
    num_windows = batch * win_h * win_w

    output = torch.empty(
        (num_windows, m, m, channels),
        dtype=x.dtype,
        device=x.device,
    )
    if rows == 0 or row_elems == 0:
        return output

    meta = (m, shift_size, height, width, channels, win_w, win_h * win_w)

    if _use_tile_kernel(num_windows, m, row_elems, x.element_size()):
        tile_meta = meta + (num_windows, row_elems)
        _roll_window_partition_tile_kernel[(
            min(num_windows, _NUM_VECTOR_CORES),)](
            x,
            output,
            tile_meta,
            tu=_TILE_TU,
            tc=_select_tile_cols(row_elems),
            shift_zero=(shift_size == 0),
        )
        return output

    block = _select_block(row_elems)
    grid = (_NUM_VECTOR_CORES,)
    _roll_window_partition_kernel[grid](
        x,
        output,
        meta + (rows, row_elems),
        block=block,
    )
    return output
