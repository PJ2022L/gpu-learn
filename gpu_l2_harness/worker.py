from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_l2_harness.adapters import get_adapter
from gpu_l2_harness.adapters.mock_flashmla import make_mock_result
from gpu_l2_harness.env import collect_env
from gpu_l2_harness.runner import ensure_run_dirs, write_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    task = json.loads(args.task_json)
    if args.mock:
        task["mock"] = True
    env = collect_env()
    result = make_mock_result(task, env) if args.mock else get_adapter(task["operator"]).run(task, env)
    dirs = ensure_run_dirs(args.run_id)
    write_result(Path(dirs["raw"]), result)


if __name__ == "__main__":
    main()
