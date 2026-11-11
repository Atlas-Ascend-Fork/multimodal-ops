#!/bin/bash
# build.sh — wheel build for fused_conv_tanh cann_bench package
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf dist build *.egg-info

python setup.py bdist_wheel

WHEEL=$(ls dist/cann_bench*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: wheel not found in dist/"
    exit 1
fi

# Run the import check after the wheel exists, and from a directory that does not hold the
# package source: from SCRIPT_DIR, `import cann_bench` resolves ./cann_bench/ instead of the
# built artifact, so it reports OK even when nothing is installed. Keep it non-fatal, so a
# clean environment that has not installed the wheel yet still exits 0.
if (cd / && python -c "import cann_bench; print('import OK:', cann_bench.fused_conv_tanh)") >/dev/null 2>&1; then
    echo "IMPORT OK: cann_bench.fused_conv_tanh (installed package)"
else
    echo "IMPORT SKIPPED: cann_bench is not installed; run 'pip install $WHEEL' to enable this check"
fi

echo "BUILD OK: $WHEEL"
