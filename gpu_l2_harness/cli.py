from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gpu_l2_harness.analyzer import analyze_results
from gpu_l2_harness.env import write_env
from gpu_l2_harness.planner import plan_from_config, write_manifest
from gpu_l2_harness.reporter import generate_report
from gpu_l2_harness.runner import append_command_log, ensure_run_dirs, run_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-l2")
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("collect-env")
    p_env.add_argument("--out", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--config", required=True)
    p_plan.add_argument("--run-id", default="dryrun")
    p_plan.add_argument("--device", type=int, default=0)
    p_plan.add_argument("--dry-run", action="store_true")

    p_run = sub.add_parser("run")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--run-id", required=True)
    p_run.add_argument("--device", type=int, default=0)
    p_run.add_argument("--mock", action="store_true")

    p_profile = sub.add_parser("profile")
    p_profile.add_argument("--task-manifest", required=True)
    p_profile.add_argument("--run-id", default=None)

    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--run-id", required=True)

    p_report = sub.add_parser("report")
    p_report.add_argument("--run-id", required=True)

    args = parser.parse_args(argv)

    if args.command == "collect-env":
        write_env(args.out)
        return 0

    if args.command == "plan":
        ensure_run_dirs(args.run_id)
        append_command_log(args.run_id, sys.argv if argv is None else ["gpu-l2", *argv], {"dry_run": args.dry_run})
        _, _, tasks = plan_from_config(args.config, device=args.device)
        manifest = Path("results") / args.run_id / "task_manifest.json"
        write_manifest(tasks, manifest, args.config, dry_run=args.dry_run)
        print(json.dumps({"manifest": str(manifest), "task_count": len(tasks)}, sort_keys=True))
        return 0

    if args.command == "run":
        ensure_run_dirs(args.run_id)
        append_command_log(args.run_id, sys.argv if argv is None else ["gpu-l2", *argv], {"mock": args.mock})
        _, _, tasks = plan_from_config(args.config, device=args.device)
        manifest = Path("results") / args.run_id / "task_manifest.json"
        write_manifest(tasks, manifest, args.config, dry_run=False)
        run_manifest(manifest, args.run_id, mock=args.mock)
        return 0

    if args.command == "profile":
        run_id = args.run_id or Path(args.task_manifest).parent.name
        ensure_run_dirs(run_id)
        append_command_log(run_id, sys.argv if argv is None else ["gpu-l2", *argv])
        print("Profile command is scaffolded. Select tasks with analyzer before running NCU.")
        return 0

    if args.command == "analyze":
        ensure_run_dirs(args.run_id)
        append_command_log(args.run_id, sys.argv if argv is None else ["gpu-l2", *argv])
        analyze_results(args.run_id)
        return 0

    if args.command == "report":
        ensure_run_dirs(args.run_id)
        append_command_log(args.run_id, sys.argv if argv is None else ["gpu-l2", *argv])
        path = generate_report(args.run_id)
        print(str(path))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

