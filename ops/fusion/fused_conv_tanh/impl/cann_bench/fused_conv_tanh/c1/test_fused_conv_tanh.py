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

"""PyPTO-Pro fused_conv_tanh kernel (fp16)."""

import torch
import torch_npu
import torch.nn.functional as F
import pypto_pro.language as pl

from .conv_common import (TP, TC, KB, VEC, _CACHE_LIMIT,
                          _compute_tile_indices, _decompose_ki,
                          _matmul_step, _pad_and_split,
                          TanhK1Tiling, TanhK2Tiling,
                          _make_tanh_k1_tiling, _make_tanh_k2_tiling)


def _load_k_operands(iter_state, gm_bufs, pos, conv_params, ki_info):
    """Load A/B operands for one K-loop iteration (tanh fp16 variant).

    gm_bufs     = (f2, ones_b, xe2, xo2)
    conv_params = (ci_pad, kw_s, cb_n, dw_s, sw_s, wp_dim, wpe_dim, use_split)
    ki_info     = (h_row, kh, kw, cs, b_off)
    """
    ki, n_conv, cur_a, cur_b = iter_state
    co0, ow0 = pos
    f2, ones_b, xe2, xo2 = gm_bufs
    ci_pad, kw_s, cb_n, dw_s, sw_s, wp_dim, wpe_dim, use_split = conv_params
    h_row, kh, kw, cs, b_off = ki_info
    if ki == n_conv:
        pl.load(cur_a, f2, [co0, kh * kw_s * ci_pad])
        pl.load(cur_b, ones_b, [0, 0])
    elif use_split == 1:
        col_start = ow0 + (kw * dw_s) // 2
        is_odd = (kw * dw_s) % 2 == 1
        if is_odd:
            pl.load(cur_b, xo2,
                    [b_off + cs * KB, h_row * wpe_dim + col_start])
        else:
            pl.load(cur_b, xe2,
                    [b_off + cs * KB, h_row * wpe_dim + col_start])
        pl.load(cur_a, f2,
                [co0, (kh * kw_s + kw) * ci_pad + cs * KB])
    else:
        col_start = ow0 * sw_s + kw * dw_s
        pl.load(cur_b, xe2,
                [b_off + cs * KB, h_row * wp_dim + col_start])
        pl.load(cur_a, f2,
                [co0, (kh * kw_s + kw) * ci_pad + cs * KB])


def _setup_cube_buffers(op_dtype):
    """Allocate Cube engine tile types and groups for tanh conv (c1 fp16)."""
    a_l1_type = pl.TileType(
        shape=[TP, KB], dtype=op_dtype,
        target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    b_l1_type = pl.TileType(
        shape=[KB, TC], dtype=op_dtype,
        target_memory=pl.MemorySpace.Mat, layout=pl.NZ, compact=1)
    a_l0_type = pl.TileType(
        shape=[TP, KB], dtype=op_dtype,
        target_memory=pl.MemorySpace.Left, compact=1)
    b_l0_type = pl.TileType(
        shape=[KB, TC], dtype=op_dtype,
        target_memory=pl.MemorySpace.Right, compact=1)
    acc_type = pl.TileType(
        shape=[TP, TC], dtype=pl.DT_FP32,
        target_memory=pl.MemorySpace.Acc, layout=pl.NZ, fractal=1024)
    # tanh c1 fp16: tile groups with standard address layout
    a_l1 = pl.make_tile_group(type=a_l1_type, addrs=0x00000, mutex_ids=[0, 1])
    b_l1 = pl.make_tile_group(type=b_l1_type, addrs=0x20000, mutex_ids=[2, 3])
    a_l0 = pl.make_tile_group(type=a_l0_type, addrs=0x0000, mutex_ids=[5, 6])
    acc = pl.make_tile_group(type=acc_type, addrs=0x0000, mutex_ids=[10])
    b_l0 = pl.make_tile_group(type=b_l0_type, addrs=0x0000, mutex_ids=[7, 8])
    return a_l1, b_l1, a_l0, b_l0, acc


def _apply_tanh_chain(ub_bufs, ub_dst, scol):
    """Apply tanh: exp(z), neg(z), exp(-z), sub, add, div -> cast."""
    ub_e1, ub_neg, ub_e2, ub_num, ub_den, ub_res = ub_bufs
    e1cur = ub_e1.current()
    pl.exp(e1cur, scol)
    ncur = ub_neg.current()
    pl.neg(ncur, scol)
    e2cur = ub_e2.current()
    pl.exp(e2cur, ncur)
    mcur = ub_num.current()
    pl.sub(mcur, e1cur, e2cur)
    dcur = ub_den.current()
    pl.add(dcur, e1cur, e2cur)
    rcur = ub_res.current()
    pl.div(rcur, mcur, dcur)
    dcol = ub_dst.next()
    pl.cast(dcol, rcur, mode=pl.RoundMode.CAST_ROUND)
    return dcol


@pl.jit(auto_mutex=True, name="fct_kernel_fp16")
def fct_kernel_fp16(
    xe2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    xo2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    f2: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    ones_b: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling,
):
    a_l1, b_l1, a_l0, b_l0, acc = _setup_cube_buffers(pl.DT_FP16)
    th_nconv = tiling.kh_s * tiling.kw_s * tiling.cb_n
    th_totk = th_nconv + 1
    th_copad = tiling.m_tiles * TC
    th_owpad = tiling.n_tiles * TP
    th_cid = pl.get_block_idx()
    th_ncores = pl.get_block_num()

    with pl.section_cube():
        for t in pl.range(th_cid, tiling.num_tiles, th_ncores):
            oh, bidx, th_co, th_ow = _compute_tile_indices(
                t, tiling.m_tiles, tiling.n_tiles, tiling.oh_dim)
            ac = acc.current()
            for ki in pl.range(0, th_totk):
                cur_a = a_l1.next()
                cur_b = b_l1.next()
                al = a_l0.next()
                br = b_l0.next()
                kh, kw, cs = _decompose_ki(ki, tiling.kw_s, tiling.cb_n)
                th_hrow = oh * tiling.sh_s + kh * tiling.dh_s
                th_boff = bidx * tiling.ci_pad
                _load_k_operands(
                    (ki, th_nconv, cur_a, cur_b),
                    (f2, ones_b, xe2, xo2), (th_co, th_ow),
                    (tiling.ci_pad, tiling.kw_s, tiling.cb_n, tiling.dw_s, tiling.sw_s,
                     tiling.wp_dim, tiling.wpe_dim, tiling.use_split),
                    (th_hrow, kh, kw, cs, th_boff))
                pl.move(al, cur_a)
                pl.move(br, cur_b)
                _matmul_step(ac, al, br, ki, th_totk)
            pl.store(tiling.ws_transposed, ac, [bidx * th_copad + th_co, oh * th_owpad + th_ow],
                phase=pl.STPhase.Final)


@pl.jit(auto_mutex=True, name="fct_tanh2nchw_fp16")
def fct_tanh2nchw_fp16(
    ws_transposed: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP32],
    out: pl.Tensor[[pl.DYNAMIC, pl.DYNAMIC], pl.DT_FP16],
    tiling,
):
    tt_src = pl.TileType(shape=[TP, TC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_mid = pl.TileType(shape=[TP, TC], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec)
    tt_dst = pl.TileType(shape=[TC, TP], dtype=pl.DT_FP16, target_memory=pl.MemorySpace.Vec)
    ub_src = pl.make_tile_group(type=tt_src, addrs=0x00000, mutex_ids=[0, 1])
    ub_e1 = pl.make_tile_group(type=tt_mid, addrs=0x08000, mutex_ids=[4])
    ub_neg = pl.make_tile_group(type=tt_mid, addrs=0x0C000, mutex_ids=[5])
    ub_e2 = pl.make_tile_group(type=tt_mid, addrs=0x10000, mutex_ids=[6])
    ub_num = pl.make_tile_group(type=tt_mid, addrs=0x14000, mutex_ids=[7])
    ub_den = pl.make_tile_group(type=tt_mid, addrs=0x18000, mutex_ids=[8])
    ub_res = pl.make_tile_group(type=tt_mid, addrs=0x1C000, mutex_ids=[9])
    ub_dst = pl.make_tile_group(type=tt_dst, addrs=0x20000, mutex_ids=[2, 3])
    th_copad = tiling.m_tiles * TC
    th_owpad = tiling.n_tiles * TP
    th_work = tiling.batch * tiling.oh_dim * tiling.n_tiles * tiling.m_tiles
    th_cid = pl.get_block_idx()
    th_ncores = pl.get_block_num()

    with pl.section_vector():
        for t in pl.range(th_cid, th_work, th_ncores):
            oh, bidx, th_co, th_ow = _compute_tile_indices(
                t, tiling.m_tiles, tiling.n_tiles, tiling.oh_dim)
            scol = ub_src.next()
            pl.load(scol, ws_transposed, [bidx * th_copad + th_co, oh * th_owpad + th_ow])
            dcol = _apply_tanh_chain(
                (ub_e1, ub_neg, ub_e2, ub_num, ub_den, ub_res),
                ub_dst, scol)
            th_vc = pl.min(TC, tiling.co_dim - th_co)
            th_vp = pl.min(TP, tiling.wow_dim - th_ow)
            pl.set_validshape(dcol, [th_vc, th_vp])
            pl.store(out, dcol, [bidx * tiling.co_dim + th_co, oh * tiling.wow_dim + th_ow])


_prepared_cache = {}


def _build_weight_and_workspace(x, conv_filter, bias, weight_cfg, tiling_cfg):
    """Build padded weight/bias/workspace for the tanh fp16 two-kernel design.

    weight_cfg = (out_channels, in_ch, kernel_h, kernel_w, in_ch_padded, k_total)
    tiling_cfg = (out_ch_padded, batch_size, out_h, out_w, n_tiles)
    """
    oc, ic, kh, kw, ic_pad, k_tot = weight_cfg
    oc_pad, bs, oh, ow, nt = tiling_cfg
    f2v = conv_filter.permute(0, 2, 3, 1).contiguous()
    f2p = torch.zeros(oc_pad, k_tot + ic_pad,
                      dtype=x.dtype, device=x.device)
    f2v_p = torch.zeros(oc, kh, kw, ic_pad,
                        dtype=x.dtype, device=x.device)
    f2v_p[:oc, :, :, :ic] = f2v
    f2p[:oc, :k_tot] = f2v_p.reshape(oc, k_tot)
    f2p[:oc, k_tot] = bias
    ones_b = torch.zeros(KB, TP, dtype=x.dtype, device=x.device)
    ones_b[0, :] = 1.0
    ws_t = torch.empty(bs * oc_pad, oh * nt * TP,
                       dtype=torch.float32, device=x.device)
    y_out = torch.empty(bs * oc, oh * ow,
                        dtype=x.dtype, device=x.device)
    return f2p, ones_b, ws_t, y_out


def _prepare_tensors(x, conv_filter, bias, strides, pads, **kwargs):
    """Prepare host tensors, pad inputs, and compute tiling parameters."""
    dilations = kwargs.get('dilations') or [1, 1]
    batch_size, in_channels, height, width = x.shape
    out_channels, in_ch, kernel_h, kernel_w = conv_filter.shape
    sh, sw = int(strides[0]), int(strides[1])
    pt, pb, pl_, pr = int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])
    dh, dw = int(dilations[0]), int(dilations[1])

    key = (x.dtype, str(x.device),
           batch_size, in_channels, height, width,
           out_channels, in_ch, kernel_h, kernel_w,
           sh, sw, dh, dw, pt, pb, pl_, pr)
    entry = _prepared_cache.get(key)
    if entry is not None:
        return entry

    out_h = (height + pt + pb - dh * (kernel_h - 1) - 1) // sh + 1
    out_w = (width + pl_ + pr - dw * (kernel_w - 1) - 1) // sw + 1
    m_tiles = (out_channels + TC - 1) // TC
    n_tiles = (out_w + TP - 1) // TP
    out_ch_padded = m_tiles * TC
    in_ch_padded = ((in_ch + KB - 1) // KB) * KB
    cb = in_ch_padded // KB
    k_total = kernel_h * kernel_w * in_ch_padded

    f2p, ones_b, ws_t, y_out = _build_weight_and_workspace(
        x, conv_filter, bias,
        (out_channels, in_ch, kernel_h, kernel_w, in_ch_padded, k_total),
        (out_ch_padded, batch_size, out_h, out_w, n_tiles))

    xp = x
    has_padding = pt or pb or pl_ or pr
    if has_padding:
        xp = F.pad(x, (pl_, pr, pt, pb))
    use_split = 1 if sw == 2 else 0

    xp, xe2, xo2, padded_w, padded_w_even = _pad_and_split(
        xp,
        (batch_size, in_channels, in_ch, in_ch_padded, n_tiles),
        (kernel_w, dw, sw, use_split))

    total_tiles = batch_size * out_h * m_tiles * n_tiles
    num_cores = min(28, max(1, total_tiles))
    num_cores_t = min(56, max(1, total_tiles))
    entry = (xe2, xo2, f2p, ones_b, ws_t, y_out,
             out_h, out_w, m_tiles, n_tiles, total_tiles,
             in_ch_padded, padded_w, padded_w_even,
             cb, use_split, num_cores, num_cores_t,
             batch_size, out_channels, kernel_h, kernel_w, sh, sw, dh, dw)
    if len(_prepared_cache) >= _CACHE_LIMIT:
        _prepared_cache.clear()
    _prepared_cache[key] = entry
    return entry


def fused_conv_tanh_fctfp16(x, conv_filter, bias, strides, pads,
                               **kwargs):
    """Fused conv2d + tanh (fp16) entry point."""
    dilations = kwargs.get('dilations') or [1, 1]

    entry = _prepare_tensors(x, conv_filter, bias, strides, pads, dilations=dilations)
    (xe2, xo2, f2p, ones_b, ws_t, y_out,
     out_h, out_w, m_tiles, n_tiles, total_tiles,
     in_ch_padded, padded_w, padded_w_even,
     cb, use_split, num_cores, num_cores_t,
     batch_size, out_channels, kernel_h, kernel_w,
     sh, sw, dh, dw) = entry

    torch.npu.synchronize()
    fct_kernel_fp16[None, num_cores](
        xe2, xo2, f2p, ones_b,
        _make_tanh_k1_tiling({
            "num_tiles": total_tiles, "m_tiles": m_tiles,
            "n_tiles": n_tiles, "oh_dim": out_h,
            "ci_pad": in_ch_padded, "wp_dim": padded_w,
            "wpe_dim": padded_w_even, "sh_s": sh,
            "sw_s": sw, "dh_s": dh, "dw_s": dw,
            "kh_s": kernel_h, "kw_s": kernel_w,
            "cb_n": cb, "use_split": use_split,
            "ws_transposed": ws_t}))
    torch.npu.synchronize()
    fct_tanh2nchw_fp16[None, num_cores_t](
        ws_t, y_out,
        _make_tanh_k2_tiling({
            "wow_dim": out_w, "co_dim": out_channels,
            "oh_dim": out_h, "n_tiles": n_tiles,
            "m_tiles": m_tiles, "batch": batch_size}))
    torch.npu.synchronize()
    return y_out.view(batch_size, out_channels, out_h, out_w)


fused_conv_tanh = fused_conv_tanh_fctfp16
