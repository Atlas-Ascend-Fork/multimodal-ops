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

from cann_bench.fused_patch_concat_layernorm import (
    fused_patch_concat_layernorm as _impl,
)


def fused_patch_concat_layernorm(x, weight, bias, height, width, eps=1e-5):
    """Swin v1 patch concat + LayerNorm (pre-linear) — Triton-Ascend fused."""
    return _impl(x, weight, bias, height, width, eps)
