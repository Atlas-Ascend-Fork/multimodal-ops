# build.sh — wheel build for fused_mul_add_layernorm_mul_add cann_bench package
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf dist build *.egg-info

python -c "import cann_bench; print('import OK:', cann_bench.fused_mul_add_layernorm_mul_add)"

python setup.py bdist_wheel

WHEEL=$(ls dist/cann_bench*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: wheel not found in dist/"
    exit 1
fi

echo "BUILD OK: $WHEEL"
