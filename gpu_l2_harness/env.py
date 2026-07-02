from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=20)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive environment probe.
        return {"ok": False, "error": str(exc), "stdout": "", "stderr": ""}


def get_git_commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], check=False, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def collect_env() -> dict[str, Any]:
    env: dict[str, Any] = {
        "git_commit": get_git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvcc": _run(["nvcc", "--version"]),
        "ncu": _run(["ncu", "--version"]),
        "torch": {"ok": False},
        "gpu": {"ok": False},
    }
    try:
        import torch

        env["torch"] = {
            "ok": True,
            "version": torch.__version__,
            "cuda": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu"] = {
                "ok": True,
                "name": props.name,
                "count": torch.cuda.device_count(),
                "l2_cache_size": int(getattr(props, "L2_cache_size", 0)),
                "total_memory": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
            }
        else:
            env["gpu"] = {"ok": False, "reason": "torch.cuda.is_available() is False"}
    except Exception as exc:
        env["torch"] = {"ok": False, "error": str(exc)}
        env["gpu"] = {"ok": False, "reason": "torch import/probe failed"}
    return env


def write_env(path: str | Path) -> dict[str, Any]:
    data = collect_env()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return data

