#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

rm -rf dist build *.egg-info

python -c "import cann_bench; print('import OK:', cann_bench.fused_conv_sigmoid)"

python setup.py bdist_wheel

WHEEL=$(ls dist/cann_bench*.whl 2>/dev/null | head -1)
if [ -z "$WHEEL" ]; then
    echo "ERROR: wheel not found in dist/"
    exit 1
fi

echo "BUILD OK: $WHEEL"
