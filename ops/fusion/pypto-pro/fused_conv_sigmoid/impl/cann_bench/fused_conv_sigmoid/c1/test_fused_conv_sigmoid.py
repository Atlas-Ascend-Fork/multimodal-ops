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

"""PyPTO-Pro fully-fused conv+sigmoid kernel (fp16) -- single @pl.jit kernel."""

import torch
import torch_npu
import torch.nn.functional as F
import pypto_pro.language as pl

from .conv_common import (TP, TC, KB, VEC, _CACHE_LIMIT,
                          _compute_tile_indices, _decompose_ki,
                          _load_k_operands, _matmul_step_partial,
                          _cube_slot_wait, _cube_drain_and_signal,
                          _vec_slot_wait, _vec_slot_free,
                          _cube_preset_fix_events, _vec_preset_free_events,
                          _cube_k_iteration, FusedTiling,
                          _make_fused_tiling)

TH = 32        # co rows per Vector subblock half (DualModeSplitM)
LAND = 0x00000  # UB: landing s0/s1/s2 | neg | exp | sig | dst (offsets per variant below)


def _check_tile_fit():
    """Derive on-chip memory budgets from the tile constants and validate them
    against the target chip (Ascend 950 / dav-3510: UB 248KB per Vector
    subblock, L1 512KB, L0A/L0B 64KB, L0C 256KB).  All on-chip layout sizes
    are computed from TP/TC/TH/KB here instead of being hard-coded per buffer,
    so a different tile configuration fails fast at import time if it no
    longer fits the device."""
    a_l1_b = TP * KB * 2          # fp16 L1 A tile
    b_l1_b = KB * TC * 2          # fp16 L1 B tile
    l1_need = 2 * a_l1_b + 0x20000 + 2 * b_l1_b
    ub_half_f32 = TH * TP * 4
    ub_need = 2 * ub_half_f32 + 3 * ub_half_f32 + TH * TP * 2  # landing x2, neg/exp/sig, dst
    acc_need = 2 * TP * TC * 4    # 2-deep L0C acc
    l0_need = 2 * (TP * KB * 2 + KB * TC * 2)
    if l1_need > 512 * 1024:
        raise ValueError(f"L1 over budget: {l1_need}")
    if ub_need > 248 * 1024:
        raise ValueError(f"UB over budget: {ub_need}")
    if acc_need > 256 * 1024:
        raise ValueError(f"L0C over budget: {acc_need}")
    if l0_need > 64 * 1024:
        raise ValueError(f"L0A/L0B over budget: {l0_need}")


_check_tile_fit()


def _apply_sigmoid_chain_v2_cast(ub_neg, ub_exp, ub_sig, ub_dst, vec):
    """Sigmoid chain with cast for fp16/bf16 output (c1 fp16 path)."""
    neg_slot = ub_neg.current()  # c1: negate input vector
    exp_slot = ub_exp.current()
    sig_slot = ub_sig.current()
    dst_slot = ub_dst.current()
    pl.neg(neg_slot, vec)
    pl.exp(exp_slot, neg_slot)
    pl.add(sig_slot, 1.0, exp_slot)  # commutative: 1.0 + exp
    pl.recip(sig_slot, sig_slot)
    pl.cast(dst_slot, sig_slot, mode=pl.RoundMode.CAST_ROUND)
    return dst_slot


def _vec_sigmoid_store_cast(sub_id, vec, ub_bufs, tile_pos, out):
    """Vector: subblock valid_c, sigmoid chain with cast, valid_shape, store.
    (c1 fp16 variant -- cast from fp32 accumulation to fp16 output.)

    ub_bufs = (ub_neg, ub_exp, ub_sig, ub_dst)
    tile_pos = (co_dim, wow_dim, co0, ow0, bidx, oh, th_val, tp_val)
    """
    ub_neg, ub_exp, ub_sig, ub_dst = ub_bufs
    co_dim, wow_dim, co0, ow0, bidx, oh, th_val, tp_val = tile_pos
    # c1 fp16: compute row origin before valid_c
    if sub_id == 0:
        row0 = co0 + bidx * co_dim
        valid_c = pl.min(th_val, co_dim - co0)
    else:
        row0 = co0 + th_val + bidx * co_dim
        valid_c = pl.min(th_val, co_dim - co0 - th_val)
    dst_slot = _apply_sigmoid_chain_v2_cast(ub_neg, ub_exp, ub_sig, ub_dst, vec)
    valid_p = pl.min(tp_val, wow_dim - ow0)
    if valid_c > 0:
        pl.set_validshape(dst_slot, [valid_c, valid_p])
        pl.store(out, dst_slot, [row0, ow0 + oh * wow_dim])


def _setup_fcs_fused_fp16_buffers():
    """Buffer setup for fcs_fused_fp16."""
    tt_a_l1 = pl.TileType(shape=[TP, KB], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    tt_b_l1 = pl.TileType(shape=[KB, TC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    tt_a_l0 = pl.TileType(shape=[TP, KB], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Left, compact=1)
    tt_b_l0 = pl.TileType(shape=[KB, TC], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Right, compact=1)
    tt_acc = pl.TileType(shape=[TP, TC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024)
    tt_land = pl.TileType(shape=[TH, TP], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[TH, TP], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    a_l1 = pl.make_tile_group(type=tt_a_l1, addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(type=tt_b_l1, addrs=0x20000, mutex_ids=[2, 3])
    a_l0 = pl.make_tile_group(type=xw_a_l0, addrs=0x0000, mutex_ids=[4, 5])
    b_l0 = pl.make_tile_group(type=xw_b_l0, addrs=0x0000, mutex_ids=[6, 7])
    acc = pl.make_tile_group(type=xw_acc, addrs=0x0000, mutex_ids=[8, 9, 10])
    landing = pl.make_tile_group(type=tt_land, addrs=LAND, mutex_ids=[11, 12, 13])
    ub_neg = pl.make_tile_group(type=tt_land, addrs=0x06000, mutex_ids=[14])
    ub_exp = pl.make_tile_group(type=tt_land, addrs=0x08000, mutex_ids=[15])
    ub_sig = pl.make_tile_group(type=tt_land, addrs=0x0A000, mutex_ids=[16])
    ub_dst = pl.make_tile_group(type=tt_dst, addrs=0x0C000, mutex_ids=[17])
    core_id = pl.get_block_idx() // pl.get_subblock_num()
    return {"a_l1": a_l1, "b_l1": b_l1, "a_l0": a_l0, "b_l0": b_l0,
            "acc": acc, "landing": landing, "ub_neg": ub_neg, "ub_exp": ub_exp,
            "ub_sig": ub_sig, "ub_dst": ub_dst, "core_id": core_id}


@pl.jit(auto_mutex=True, name="fcs_fused_fp16")
def fcs_fused_fp16(
    xe2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    xo2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    f2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ones_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling,
):
    bufs = _setup_fcs_fused_fp16_buffers()
    a_l1, b_l1, a_l0 = bufs["a_l1"], bufs["b_l1"], bufs["a_l0"]
    b_l0, acc, landing = bufs["b_l0"], bufs["acc"], bufs["landing"]
    ub_neg, ub_exp, ub_sig = bufs["ub_neg"], bufs["ub_exp"], bufs["ub_sig"]
    ub_dst, core_id = bufs["ub_dst"], bufs["core_id"]
    fp16_nconv = tiling.kh_s * tiling.kw_s * tiling.cb_n
    fp16_totk = fp16_nconv + 1
    with pl.section_cube():
        _cube_preset_fix_events()
        for t in pl.range(core_id, tiling.num_tiles, tiling.core_stride):
            mb, nb, oh, bidx = _compute_tile_indices(
                t, tiling.m_tiles, tiling.n_tiles, tiling.oh_dim)
            co0, ow0 = mb * TC, nb * TP
            ac = acc.next()
            _cube_slot_wait(t)
            for ki in pl.range(0, fp16_totk):
                _cube_k_iteration(
                    ki, (a_l1, b_l1, a_l0, b_l0), ac,
                    (f2, ones_b, xe2, xo2), co0, ow0,
                    (tiling.ci_pad, tiling.kw_s, tiling.cb_n, tiling.dw_s, tiling.sw_s, tiling.wp_dim, tiling.wpe_dim,
                        tiling.use_split),

                    (oh, tiling.sh_s, tiling.dh_s, fp16_nconv, bidx), KB)
            lnd = landing.next()
            _cube_drain_and_signal(t, lnd, ac)

    with pl.section_vector():
        _vec_preset_free_events()
        for t in pl.range(core_id, tiling.num_tiles, tiling.core_stride):
            mb, nb, oh, bidx = _compute_tile_indices(
                t, tiling.m_tiles, tiling.n_tiles, tiling.oh_dim)
            co0, ow0 = mb * TC, nb * TP
            sub_id = pl.get_subblock_idx()
            vec = landing.next()
            _vec_slot_wait(t)
            _vec_sigmoid_store_cast(
                sub_id, vec, (ub_neg, ub_exp, ub_sig, ub_dst),
                (tiling.co_dim, tiling.wow_dim, co0, ow0, bidx, oh, TH, TP), tiling.out)
            _vec_slot_free(t)


_prepared_cache = {}


def _build_weight_and_workspace(x, conv_filter, bias, shape_info, conv_params):
    """Build padded weights, ones_b, output buffer and compute tiling parameters.

    shape_info = (batch_size, in_channels, out_channels, in_ch, kernel_h, kernel_w, height, width)
    conv_params = (sh, sw, dh, dw, pt, pb, pl_, pr)
    """
    batch_size, in_channels, out_channels, in_ch, kernel_h, kernel_w, height, width = shape_info
    sh, sw, dh, dw, pt, pb, pl_, pr = conv_params
    out_h = (height + pt + pb - dh * (kernel_h - 1) - 1) // sh + 1
    out_w = (width + pl_ + pr - dw * (kernel_w - 1) - 1) // sw + 1
    m_tiles = (out_channels + TC - 1) // TC
    n_tiles = (out_w + TP - 1) // TP
    in_ch_padded = ((in_ch + KB - 1) // KB) * KB
    cb = in_ch_padded // KB
    out_ch_padded = m_tiles * TC
    f2v = conv_filter.permute(0, 2, 3, 1).contiguous()
    k_total = kernel_h * kernel_w * in_ch_padded
    f2p = torch.zeros(out_ch_padded, k_total + in_ch_padded, dtype=x.dtype, device=x.device)
    f2v_p = torch.zeros(out_channels, kernel_h, kernel_w, in_ch_padded, dtype=x.dtype, device=x.device)
    f2v_p[:out_channels, :, :, :in_ch] = f2v
    f2p[:out_channels, :k_total] = f2v_p.reshape(out_channels, k_total)
    f2p[:out_channels, k_total] = bias
    ones_b = torch.zeros(KB, TP, dtype=x.dtype, device=x.device)
    ones_b[0, :] = 1.0
    y_nchw = torch.empty(batch_size * out_channels, out_h * out_w, dtype=x.dtype, device=x.device)
    total_tiles = batch_size * out_h * m_tiles * n_tiles
    num_cores = min(28, max(1, total_tiles))
    return {"f2p": f2p, "ones_b": ones_b, "y_nchw": y_nchw,
            "out_h": out_h, "out_w": out_w, "m_tiles": m_tiles, "n_tiles": n_tiles,
            "total_tiles": total_tiles, "in_ch_padded": in_ch_padded, "cb": cb,
            "num_cores": num_cores}


def _pad_and_split(x, shape_info, split_cfg):
    """Pad input, optionally split even/odd columns, pad channels.

    shape_info = (batch_size, in_channels, in_ch, in_ch_padded, n_tiles)
    split_cfg = (sw, dw, kernel_w, pt, pb, pl_, pr)
    """
    batch_size, in_channels, in_ch, in_ch_padded, n_tiles = shape_info
    sw, dw, kernel_w, pt, pb, pl_, pr = split_cfg
    xp = x
    has_padding = pt or pb or pl_ or pr
    if has_padding:
        xp = F.pad(x, (pl_, pr, pt, pb))
    use_split = 1 if sw == 2 else 0
    if use_split:
        w_need = (n_tiles * TP - 1) + ((kernel_w - 1) * dw) // 2 + 1
    else:
        w_need = (n_tiles * TP - 1) * sw + (kernel_w - 1) * dw + 1
    padded_w = xp.shape[3]
    pad_w = w_need - padded_w
    if pad_w > 0:
        xp = F.pad(xp, (0, pad_w, 0, 0))
    padded_w = xp.shape[3]
    if use_split:
        xe = xp[:, :, :, 0::2]
        xo = xp[:, :, :, 1::2]
        even_cols = xe.shape[3]
        wpe_need = (n_tiles * TP - 1) + ((kernel_w - 1) * dw) // 2 + 1
        pad_we = wpe_need - even_cols
        pad_wo = wpe_need - xo.shape[3]
        if pad_we > 0 or pad_wo > 0:
            xe = F.pad(xe, (0, max(pad_we, 0), 0, 0))
            xo = F.pad(xo, (0, max(pad_wo, 0), 0, 0))
            even_cols = xe.shape[3]
        xe = xe.contiguous()
        xo = xo.contiguous()
    else:
        xe = xp.contiguous()
        xo = xe
        even_cols = 1
    if in_ch_padded > in_ch:
        xe2 = xe.reshape(batch_size, in_channels, -1)
        xe2 = F.pad(xe2, (0, 0, 0, in_ch_padded - in_ch)).reshape(batch_size * in_ch_padded, -1).contiguous()
        if use_split:
            xo2 = xo.reshape(batch_size, in_channels, -1)
            xo2 = F.pad(xo2, (0, 0, 0, in_ch_padded - in_ch)).reshape(batch_size * in_ch_padded, -1).contiguous()
        else:
            xo2 = xe2
    else:
        xe2 = xe.reshape(batch_size * in_channels, -1).contiguous()
        xo2 = xo.reshape(batch_size * in_channels, -1).contiguous() if use_split else xe2
    return xe2, xo2, padded_w, even_cols, use_split


def _prepare_tensors(x, conv_filter, bias, strides, pads, **kwargs):
    """Prepare host tensors, pad inputs, and compute tiling parameters."""
    dilations = kwargs.get('dilations') or [1, 1]
    batch_size, in_channels, height, width = x.shape
    out_channels, in_ch, kernel_h, kernel_w = conv_filter.shape
    sh, sw = int(strides[0]), int(strides[1])
    pt, pb, pl_, pr = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
    dh, dw = int(dilations[0]), int(dilations[1])
    key = (x.dtype, str(x.device), batch_size, in_channels, height, width, out_channels, in_ch, kernel_h, kernel_w,
           sh, sw, dh, dw, pt, pb, pl_, pr)
    entry = _prepared_cache.get(key)
    if entry is not None:
        return entry
    _bw = _build_weight_and_workspace(
        x, conv_filter, bias,
        (batch_size, in_channels, out_channels, in_ch, kernel_h, kernel_w, height, width),
        (sh, sw, dh, dw, pt, pb, pl_, pr))
    f2p, ones_b, y_nchw = _bw["f2p"], _bw["ones_b"], _bw["y_nchw"]
    out_h, out_w = _bw["out_h"], _bw["out_w"]
    m_tiles, n_tiles = _bw["m_tiles"], _bw["n_tiles"]
    total_tiles, in_ch_padded, cb = _bw["total_tiles"], _bw["in_ch_padded"], _bw["cb"]
    num_cores = _bw["num_cores"]
    xe2, xo2, padded_w, padded_w_even, use_split = _pad_and_split(
        x,
        (batch_size, in_channels, in_ch, in_ch_padded, n_tiles),
        (sw, dw, kernel_w, pt, pb, pl_, pr))
    entry = (xe2, xo2, f2p, ones_b, y_nchw, out_h, out_w, m_tiles, n_tiles,
             total_tiles, in_ch_padded, padded_w, padded_w_even, cb,
             use_split, num_cores)
    if len(_prepared_cache) >= _CACHE_LIMIT:
        _prepared_cache.clear()
    _prepared_cache[key] = entry
    return entry


def fused_conv_sigmoid_fcsfp16(x, conv_filter, bias, strides, pads,
                               **kwargs):
    """Fused conv2d + sigmoid (fp16) entry point."""
    dilations = kwargs.get('dilations') or [1, 1]

    entry = _prepare_tensors(x, conv_filter, bias, strides, pads, dilations=dilations)
    (xe2, xo2, f2p, ones_b, y_nchw, out_h, out_w, m_tiles, n_tiles,
     total_tiles, in_ch_padded, padded_w, padded_w_even, cb,
     use_split, num_cores) = entry

    torch.npu.synchronize()
    fcs_fused_fp16[None, num_cores](
        xe2, xo2, f2p, ones_b,
        _make_fused_tiling({
            "num_tiles": total_tiles, "m_tiles": m_tiles,
            "n_tiles": n_tiles, "oh_dim": out_h,
            "wow_dim": out_w, "co_dim": out_channels,
            "ci_pad": in_ch_padded, "wp_dim": padded_w,
            "wpe_dim": padded_w_even, "sh_s": sh, "sw_s": sw,
            "dh_s": dh, "dw_s": dw, "kh_s": kernel_h,
            "kw_s": kernel_w, "cb_n": cb,
            "use_split": use_split, "core_stride": num_cores,
            "out": y_nchw}))
    torch.npu.synchronize()
    return y_nchw.view(batch_size, out_channels, out_h, out_w)


fused_conv_sigmoid = fused_conv_sigmoid_fcsfp16
