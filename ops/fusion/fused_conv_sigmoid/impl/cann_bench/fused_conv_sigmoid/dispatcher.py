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

"""True-fusion (v2) dim/dtype dispatcher for fused_conv_sigmoid."""

import importlib.util
from pathlib import Path

_OP_NAME = 'fused_conv_sigmoid'
_CLASSES = [
    {"subdir": "c1", "signature": [[4, "float16"], [4, "float16"], [1, "float16"]]},
]
_BASE = Path(__file__).resolve().parent
_impl_cache = {}


def _dtype_name(val):
    """Return short dtype string for v2 fused-kernel dispatch (e.g. 'float32')."""
    return str(val).rsplit(".", 1)[-1]


def _signature(positional_args):
    """Extract a dim/dtype signature from positional arguments (v2 pipeline).
    Each tensor yields [ndim, dtype_str]; tensor lists use the first element."""
    result = []
    for item in positional_args:
        is_tlist = (isinstance(item, (list, tuple)) and item
                    and hasattr(item[0], "dim") and hasattr(item[0], "dtype"))
        if is_tlist:
            result.append([int(item[0].dim()), _dtype_name(item[0].dtype)])
        elif hasattr(item, "dim") and hasattr(item, "dtype"):
            result.append([int(item.dim()), _dtype_name(item.dtype)])
    return result


def _load(subdir):
    """Dynamically load and cache the v2 fused kernel for *subdir*."""
    if subdir not in _impl_cache:
        fpath = _BASE / subdir / f"test_{_OP_NAME}.py"
        spec = importlib.util.spec_from_file_location(
            f"{_OP_NAME}_{subdir}_test", fpath)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        impl = next(
            (getattr(loaded, nm)
             for nm in (_OP_NAME, f"{_OP_NAME}_wrapper")
             if callable(getattr(loaded, nm, None))),
            None,
        )
        if impl is None:
            raise AttributeError(
                f"{fpath} has no '{_OP_NAME}' or '{_OP_NAME}_wrapper'")
        _impl_cache[subdir] = impl
    return _impl_cache[subdir]


def fused_conv_sigmoid(*args, **kwargs):
    """v2 true-fusion dispatch: route to the single-kernel fused implementation."""
    result = _signature(args)
    for cls in _CLASSES:
        cnt = len(cls["signature"])
        if result[:cnt] == cls["signature"]:
            return _load(cls["subdir"])(*args, **kwargs)
    raise ValueError(
        f"no {_OP_NAME} v2 class for signature {result}; classes={_CLASSES}")
