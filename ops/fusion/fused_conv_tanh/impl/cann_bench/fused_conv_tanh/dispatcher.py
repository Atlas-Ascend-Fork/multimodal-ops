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

"""Auto-generated dim/dtype dispatcher for `fused_conv_tanh`. Do not edit by hand."""

import importlib.util
from pathlib import Path

_OP_NAME = 'fused_conv_tanh'
_CLASSES = [
    {"subdir": "c1", "signature": [[4, "float16"], [4, "float16"], [1, "float16"]]},
]
_BASE = Path(__file__).resolve().parent
_impl_cache = {}


def _dtype_name(dtype_val):
    """Extract the short dtype name from a torch dtype string."""
    return str(dtype_val).rsplit(".", 1)[-1]


def _signature(args):
    """Build a signature list from positional arguments for tanh dispatch."""
    type_sig = []
    for arg in args:
        is_tensor_list = (
            isinstance(arg, (list, tuple))
            and arg
            and hasattr(arg[0], "dim")
            and hasattr(arg[0], "dtype")
        )
        if is_tensor_list:
            type_sig.append([int(arg[0].dim()), _dtype_name(arg[0].dtype)])
        elif hasattr(arg, "dim") and hasattr(arg, "dtype"):
            type_sig.append([int(arg.dim()), _dtype_name(arg.dtype)])
    return type_sig


def _load(subdir):
    """Dynamically load the tanh kernel implementation for the given subdir."""
    if subdir not in _impl_cache:
        mod_path = _BASE / subdir / f"test_{_OP_NAME}.py"
        spec = importlib.util.spec_from_file_location(
            f"{_OP_NAME}_{subdir}_test", mod_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        entry_fn = next(
            (
                getattr(mod, attr_name)
                for attr_name in (_OP_NAME, f"{_OP_NAME}_wrapper")
                if callable(getattr(mod, attr_name, None))
            ),
            None,
        )
        if entry_fn is None:
            raise AttributeError(
                f"{mod_path} exposes no '{_OP_NAME}' or"
                f" '{_OP_NAME}_wrapper' entry"
            )
        _impl_cache[subdir] = entry_fn
    return _impl_cache[subdir]


def fused_conv_tanh(*args, **kwargs):
    """Dispatch to the correct tanh kernel based on argument signatures."""
    type_sig = _signature(args)
    for entry in _CLASSES:
        sig_len = len(entry["signature"])
        if type_sig[:sig_len] == entry["signature"]:
            return _load(entry["subdir"])(*args, **kwargs)
    raise ValueError(f"no {_OP_NAME} class for signature {type_sig}; classes={_CLASSES}")
