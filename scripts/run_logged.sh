#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: scripts/run_logged.sh <run_id> <command...>" >&2
  exit 2
fi

RUN_ID="$1"
shift

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla

mkdir -p "logs/${RUN_ID}"
{
  echo "timestamp=$(date -Is)"
  printf 'command='
  printf '%q ' "$@"
  echo
} >> "logs/${RUN_ID}/command.log"

"$@" 2>&1 | tee -a "logs/${RUN_ID}/run.log"

