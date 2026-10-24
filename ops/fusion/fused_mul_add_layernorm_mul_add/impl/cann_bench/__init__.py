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

from cann_bench.fused_mul_add_layernorm_mul_add import fused_mul_add_layernorm_mul_add as _impl


def fused_mul_add_layernorm_mul_add(x, attn_out, gate, scale, shift, eps=1e-6, **kwargs):
    """residual = x + gate * attn_out; y = LayerNorm(residual) * (1 + scale) + shift — Triton-Ascend fused kernel."""
    return _impl(x, attn_out, gate, scale, shift, eps=eps, **kwargs)
