from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from gpu_l2_harness.env import collect_env


def ensure_run_dirs(run_id: str) -> dict[str, Path]:
    base = Path("results") / run_id
    logs = Path("logs") / run_id
    dirs = {
        "base": base,
        "raw": base / "raw",
        "parsed": base / "parsed",
        "ncu": base / "ncu",
        "figures": base / "figures",
        "reports": base / "reports",
        "logs": logs,
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def append_command_log(run_id: str, argv: list[str], extra: dict[str, Any] | None = None) -> None:
    dirs = ensure_run_dirs(run_id)
    payload = {"argv": argv, "extra": extra or {}}
    with (dirs["logs"] / "command.log").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_manifest(manifest_path: str | Path, run_id: str, mock: bool = False) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    dirs = ensure_run_dirs(run_id)
    results: list[dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        if task.get("status") == "unsupported":
            result = unsupported_result(task)
            write_result(dirs["raw"], result)
            results.append(result)
            continue
        cmd = [
            sys.executable,
            "-m",
            "gpu_l2_harness.worker",
            "--task-json",
            json.dumps(task, sort_keys=True),
            "--run-id",
            run_id,
        ]
        if mock:
            cmd.append("--mock")
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        stdout_path = dirs["logs"] / f"{task['task_id']}.stdout"
        stderr_path = dirs["logs"] / f"{task['task_id']}.stderr"
        stdout_path.write_text(proc.stdout, encoding="utf-8")
        stderr_path.write_text(proc.stderr, encoding="utf-8")
        if proc.returncode != 0:
            result = failed_result(task, proc.returncode, str(stdout_path), str(stderr_path))
            write_result(dirs["raw"], result)
            results.append(result)
            continue
        result_path = dirs["raw"] / f"{task['task_id']}.json"
        results.append(json.loads(result_path.read_text(encoding="utf-8")))
    return results


def write_result(raw_dir: Path, result: dict[str, Any]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{result['task_id']}.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def unsupported_result(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "operator": task["operator"],
        "shape": task["shape"],
        "dtype": task["dtype"],
        "repeat": task["repeat"],
        "warmup": task["warmup"],
        "l2": task["l2"],
        "timing": {"mean_ms": None, "std_ms": None, "cv": None},
        "tflops": None,
        "effective_bandwidth_gbps": None,
        "env": collect_env(),
        "artifacts": {"stdout": "", "stderr": "", "ncu_csv": ""},
        "status": "unsupported",
        "unsupported_reason": task.get("unsupported_reason"),
    }


def failed_result(task: dict[str, Any], returncode: int, stdout: str, stderr: str) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "operator": task["operator"],
        "shape": task["shape"],
        "dtype": task["dtype"],
        "repeat": task["repeat"],
        "warmup": task["warmup"],
        "l2": task["l2"],
        "timing": {"mean_ms": None, "std_ms": None, "cv": None},
        "tflops": None,
        "effective_bandwidth_gbps": None,
        "env": collect_env(),
        "artifacts": {"stdout": stdout, "stderr": stderr, "ncu_csv": ""},
        "status": "failed",
        "returncode": returncode,
    }

