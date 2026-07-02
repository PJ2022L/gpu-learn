#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${GPU_LEARN_DOCKER_CONTAINER:-l2_mla_study}"
REPO_CONTAINER="${GPU_LEARN_REPO_CONTAINER:-/data1/user/peijun/work/gpu-learn}"

mkdir -p "logs/docker"
{
  echo "timestamp=$(date -Is)"
  echo "container=${CONTAINER}"
  echo "repo_container=${REPO_CONTAINER}"
  echo "command=docker exec -it ${CONTAINER} bash"
} >> "logs/docker/command.log"

docker exec -it "${CONTAINER}" bash -lc "cd '${REPO_CONTAINER}' && exec bash"
