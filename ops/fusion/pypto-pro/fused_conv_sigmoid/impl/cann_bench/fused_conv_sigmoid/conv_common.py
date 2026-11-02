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

"""Shared constants and helpers for fused_conv_sigmoid v2 kernels."""

import pypto_pro.language as pl

# v2 fused kernel tile dimensions (Ascend 950 dav-3510).
TP = 64
TC = 64
KB = 64
VEC = 32

_CACHE_LIMIT = 16  # v2: max cached prepare_tensors entries


def _compute_tile_indices(t, m_tiles, n_tiles, oh_dim):
    """Decompose linear tile index into (mb, nb, oh, bidx)."""
    mb = t % m_tiles
    q = t // m_tiles
    nb = q % n_tiles
    q2 = q // n_tiles
    oh = q2 % oh_dim
    bidx = q2 // oh_dim
    return mb, nb, oh, bidx


def _decompose_ki(ki, kw_s, cb_n):
    """Decompose K-loop index into (kh, kw, cs)."""
    kh = ki // (kw_s * cb_n)
    rem = ki % (kw_s * cb_n)
    kw = rem // cb_n
    cs = rem % cb_n
    return kh, kw, cs


def _load_k_operands(step, tensors, pos, conv_cfg, ki_info):
    """Load A/B operands for one K-loop iteration.

    tensors = (f2, ones_b, xe2, xo2)
    conv_cfg = (ci_pad, kw_s, cb_n, dw_s, sw_s, wp_dim, wpe_dim, use_split)
    ki_info = (h_row, kh, kw, cs, b_off, kb_size)
    """
    ki, n_conv, cur_a, cur_b = step
    co0, ow0 = pos
    f2, ones_b, xe2, xo2 = tensors
    ci_pad, kw_s, cb_n, dw_s, sw_s, wp_dim, wpe_dim, use_split = conv_cfg
    h_row, kh, kw, cs, b_off, kb_size = ki_info
    if ki == n_conv:
        pl.load(cur_a, f2, [co0, kh * kw_s * ci_pad])
        pl.load(cur_b, ones_b, [0, 0])
    elif use_split == 1:
        cbase = ow0 + (kw * dw_s) // 2
        if (kw * dw_s) % 2 == 1:
            pl.load(cur_b, xo2, [b_off + cs * kb_size, h_row * wpe_dim + cbase])
        else:
            pl.load(cur_b, xe2, [b_off + cs * kb_size, h_row * wpe_dim + cbase])
        pl.load(cur_a, f2, [co0, (kh * kw_s + kw) * ci_pad + cs * kb_size])
    else:
        cbase = ow0 * sw_s + kw * dw_s
        pl.load(cur_b, xe2, [b_off + cs * kb_size, h_row * wp_dim + cbase])
        pl.load(cur_a, f2, [co0, (kh * kw_s + kw) * ci_pad + cs * kb_size])


def _matmul_step_partial(ac, al, br, ki):
    """Execute one matmul step (Partial only, for v2 fused kernels)."""
    if ki == 0:
        pl.matmul(ac, al, br, phase=pl.AccPhase.Partial)
    else:
        pl.matmul_acc(ac, ac, al, br, phase=pl.AccPhase.Partial)


def _cube_slot_wait(t):
    """Wait for own acc slot drain (FIX->M sync)."""
    if t % 3 == 0:
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=4)
    elif t % 3 == 1:
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=5)
    else:
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=6)


def _cube_drain_and_signal(t, lnd, ac):
    """Wait FREE, move acc->UB, set drain event, signal READY."""
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=7)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=7)
    if t % 3 == 0:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=2)
        pl.move(lnd, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=4)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=0)
    elif t % 3 == 1:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=3)
        pl.move(lnd, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=5)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)
    else:
        pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=9)
        pl.move(lnd, ac, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=6)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=8)


def _vec_slot_wait(t):
    """Wait for READY signal from cube."""
    if t % 3 == 0:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=0)
    elif t % 3 == 1:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=1)
    else:
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=8)


def _vec_slot_free(t):
    """Set FREE signal for cube."""
    if t % 3 == 0:
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)
    elif t % 3 == 1:
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
    else:
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=9)


def _cube_preset_fix_events():
    """Pre-set all three per-slot FIX->M events (acc slots start drained)."""
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=4)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=5)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=6)


def _vec_preset_free_events():
    """Pre-set the FREE flag of all three landing slots."""
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=2)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=3)
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=9)


def _cube_k_iteration(ki, bufs, ac, load_ctx):
    """One K-loop iteration: next buffers, decompose, load, move, matmul."""
    tensors, co0, ow0, conv_cfg, tile_info, kb_size = load_ctx
    a_l1, b_l1, a_l0, b_l0 = bufs
    f2, ones_b, xe2, xo2 = tensors
    ci_pad, kw_s, cb_n, dw_s, sw_s, wp_dim, wpe_dim, use_split = conv_cfg
    oh, sh_s, dh_s, n_conv, bidx = tile_info
    cur_a = a_l1.next()
    cur_b = b_l1.next()
    al = a_l0.next()
    br = b_l0.next()
    kh, kw, cs = _decompose_ki(ki, kw_s, cb_n)
    h_row = oh * sh_s + kh * dh_s
    _load_k_operands(
        ki, n_conv, cur_a, cur_b, tensors, co0, ow0,
        conv_cfg, (h_row, kh, kw, cs, bidx * ci_pad, kb_size))
    pl.move(al, cur_a)
    pl.move(br, cur_b)
    _matmul_step_partial(ac, al, br, ki)


class FusedTiling:
    __slots__ = ('num_tiles', 'm_tiles', 'n_tiles', 'oh_dim', 'wow_dim',
                 'co_dim', 'ci_pad', 'wp_dim', 'wpe_dim', 'sh_s', 'sw_s',
                 'dh_s', 'dw_s', 'kh_s', 'kw_s', 'cb_n', 'use_split',
                 'core_stride', 'out')


def _make_fused_tiling(d):
    """Create FusedTiling from dict. Keeps __init__ param count at 2."""
    t = FusedTiling()
    for k, v in d.items():
        setattr(t, k, v)
    return t


class GconvTiling:
    __slots__ = ('num_tiles', 'm_tiles', 'n_tiles', 'oh_dim', 'wow_dim',
                 'co_dim', 'cb_n', 'core_stride', 'out')


def _make_gconv_tiling(d):
    """Create GconvTiling from dict. Keeps __init__ param count at 2."""
    t = GconvTiling()
    for k, v in d.items():
        setattr(t, k, v)
    return t
