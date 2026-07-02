#!/usr/bin/env bash
set -euo pipefail

# Local developer dry-run. On H800, prefer scripts/docker_dry_run.sh.
RUN_ID="${1:-dryrun}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla

mkdir -p "logs/${RUN_ID}" "results/${RUN_ID}"
{
  echo "timestamp=$(date -Is)"
  echo "command=scripts/dry_run.sh ${RUN_ID}"
  echo "config=configs/gemm_persistent.yaml"
} >> "logs/${RUN_ID}/command.log"

python -m gpu_l2_harness.cli collect-env --out "results/${RUN_ID}/env.json"
python -m gpu_l2_harness.cli plan --config configs/gemm_persistent.yaml --run-id "${RUN_ID}" --dry-run
python -m gpu_l2_harness.cli report --run-id "${RUN_ID}"
