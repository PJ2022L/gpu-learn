#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${1:-docker_dryrun}"
CONTAINER="${GPU_LEARN_DOCKER_CONTAINER:-l2_mla_study}"
REPO_CONTAINER="${GPU_LEARN_REPO_CONTAINER:-/data1/user/peijun/work/gpu-learn}"

mkdir -p "logs/${RUN_ID}"
{
  echo "timestamp=$(date -Is)"
  echo "container=${CONTAINER}"
  echo "repo_container=${REPO_CONTAINER}"
  echo "command=scripts/docker_dry_run.sh ${RUN_ID}"
} >> "logs/${RUN_ID}/command.log"

docker exec "${CONTAINER}" \
  bash -lc "cd '${REPO_CONTAINER}' && python -m pip install -e . --no-build-isolation && python -m unittest discover -s tests -v && gpu-l2 plan --config configs/gemm_persistent.yaml --run-id ${RUN_ID} --dry-run && gpu-l2 report --run-id ${RUN_ID}"
