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

"""Shared constants and helpers for fused_conv_tanh kernels."""

import torch.nn.functional as F
import pypto_pro.language as pl

# Tile dimensions for tanh Cube engine layout (Ascend 950).
TP = 64
TC = 64
KB = 64
VEC = 32
_CACHE_LIMIT = 16  # tanh: max cached prepare_tensors entries


def _decompose_ki(ki, kw_s, cb_n):
    """Split K-loop index into kernel-height, kernel-width, channel-block indices."""
    kh_val = ki // (kw_s * cb_n)
    remainder = ki % (kw_s * cb_n)
    kw_val = remainder // cb_n
    cs_val = remainder % cb_n
    return kh_val, kw_val, cs_val


def _matmul_step(acc_buf, left_op, right_op, ki, total_k):
    """Execute one matmul step with correct accumulation phase for tanh conv."""
    if ki == 0:
        pl.matmul(acc_buf, left_op, right_op, phase=pl.AccPhase.Partial)
    elif ki == total_k - 1:
        pl.matmul_acc(acc_buf, acc_buf, left_op, right_op,
                      phase=pl.AccPhase.Final)
    else:
        pl.matmul_acc(acc_buf, acc_buf, left_op, right_op,
                      phase=pl.AccPhase.Partial)


def _compute_tile_indices(t, m_tiles, n_tiles, oh_dim):
    """Map linear tile index to (oh, bidx, co0, ow0) for tanh 2-kernel layout.
    Decomposition order: m -> n -> oh -> batch (innermost first)."""
    m_blk = t % m_tiles
    div1 = t // m_tiles
    n_blk = div1 % n_tiles
    div2 = div1 // n_tiles
    oh = div2 % oh_dim
    bidx = div2 // oh_dim
    ow0 = n_blk * TP  # tanh: spatial offset
    co0 = m_blk * TC  # tanh: channel offset
    return oh, bidx, co0, ow0


def _pad_and_split(xp, shape_info, split_cfg):
    """Pad input columns and perform even/odd split for stride-2 tanh convolutions.

    shape_info = (batch_size, in_channels, in_ch, in_ch_padded, n_tiles)
    split_cfg  = (kernel_w, dw, sw, use_split)
    """
    batch_size, in_channels, in_ch, in_ch_padded, n_tiles = shape_info
    kernel_w, dw, sw, use_split = split_cfg

    # Compute required spatial width based on tiling and stride.
    if use_split:
        required_w = (n_tiles * TP - 1) + ((kernel_w - 1) * dw) // 2 + 1
    else:
        required_w = (n_tiles * TP - 1) * sw + (kernel_w - 1) * dw + 1
    spatial_w = xp.shape[3]
    col_pad = required_w - spatial_w
    if col_pad > 0:
        xp = F.pad(xp, (0, col_pad, 0, 0))
    spatial_w = xp.shape[3]

    # Even/odd column split for stride-2, or identity for stride-1.
    if use_split:
        xe = xp[:, :, :, 0::2]
        xo = xp[:, :, :, 1::2]
        even_w = xe.shape[3]
        even_need = (n_tiles * TP - 1) + ((kernel_w - 1) * dw) // 2 + 1
        even_pad = even_need - even_w
        odd_pad = even_need - xo.shape[3]
        if even_pad > 0 or odd_pad > 0:
            xe = F.pad(xe, (0, max(even_pad, 0), 0, 0))
            xo = F.pad(xo, (0, max(odd_pad, 0), 0, 0))
            even_w = xe.shape[3]
        xe = xe.contiguous()
        xo = xo.contiguous()
    else:
        xe = xp.contiguous()
        xo = xe
        even_w = 1

    # Channel-pad to in_ch_padded if the raw channel count is not aligned.
    # Channel-pad to in_ch_padded if the raw channel count is not aligned.
    ch_needs_pad = in_ch_padded > in_ch
    if ch_needs_pad:
        xe2 = xe.reshape(batch_size, in_channels, -1)
        xe2 = F.pad(xe2, (0, 0, 0, in_ch_padded - in_ch))
        xe2 = xe2.reshape(batch_size * in_ch_padded, -1).contiguous()
        if use_split:
            xo2 = xo.reshape(batch_size, in_channels, -1)
            xo2 = F.pad(xo2, (0, 0, 0, in_ch_padded - in_ch))
            xo2 = xo2.reshape(
                batch_size * in_ch_padded, -1).contiguous()
        else:
            xo2 = xe2
    else:
        xe2 = xe.reshape(
            batch_size * in_channels, -1).contiguous()
        xo2 = (xo.reshape(batch_size * in_channels, -1).contiguous()
                if use_split else xe2)

    return xp, xe2, xo2, spatial_w, even_w


class TanhK1Tiling:
    __slots__ = ('num_tiles', 'm_tiles', 'n_tiles', 'oh_dim', 'ci_pad',
                 'wp_dim', 'wpe_dim', 'sh_s', 'sw_s', 'dh_s', 'dw_s',
                 'kh_s', 'kw_s', 'cb_n', 'use_split', 'ws_transposed')


def _make_tanh_k1_tiling(d):
    """Create TanhK1Tiling from dict. Keeps __init__ param count at 2."""
    t = TanhK1Tiling()
    for k, v in d.items():
        setattr(t, k, v)
    return t


class TanhK2Tiling:
    __slots__ = ('wow_dim', 'co_dim', 'oh_dim', 'n_tiles', 'm_tiles',
                 'batch')


def _make_tanh_k2_tiling(d):
    """Create TanhK2Tiling from dict. Keeps __init__ param count at 2."""
    t = TanhK2Tiling()
    for k, v in d.items():
        setattr(t, k, v)
    return t


class TanhK1FP32Tiling:
    __slots__ = ('num_tiles', 'm_tiles', 'n_tiles', 'oh_dim', 'wow_dim',
                 'co_dim', 'ci_pad', 'taps', 'wp_dim', 'wpe_dim', 'sh_s',
                 'sw_s', 'dh_s', 'dw_s', 'kh_s', 'kw_s', 'cb_n',
                 'use_split')


def _make_tanh_k1_fp32_tiling(d):
    """Create TanhK1FP32Tiling from dict. Keeps __init__ param count at 2."""
    t = TanhK1FP32Tiling()
    for k, v in d.items():
        setattr(t, k, v)
    return t


class TanhK2FP32Tiling:
    __slots__ = ('scale', 'wow_dim', 'co_dim', 'oh_dim', 'n_tiles',
                 'm_tiles', 'batch')


def _make_tanh_k2_fp32_tiling(d):
    """Create TanhK2FP32Tiling from dict. Keeps __init__ param count at 2."""
    t = TanhK2FP32Tiling()
    for k, v in d.items():
        setattr(t, k, v)
    return t
