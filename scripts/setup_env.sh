#!/usr/bin/env bash
set -euo pipefail

# Local developer setup. On H800, prefer the operatorsforge:h800-v1.0 Docker flow.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla

python -m pip install -e .

echo "Dry-run harness installed. For real L2 persistent experiments, rebuild with:"
echo "GPU_L2_BUILD_EXT=1 python -m pip install -e . --no-build-isolation"
