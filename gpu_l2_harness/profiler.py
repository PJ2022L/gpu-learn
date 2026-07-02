from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Any


PREFERRED_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.sum",
    "dram__bytes.sum",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "launch__shared_mem_per_block_static",
]


def query_metrics() -> set[str]:
    proc = subprocess.run(["ncu", "--query-metrics"], check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        return set()
    metrics: set[str] = set()
    for line in proc.stdout.splitlines():
        token = line.strip().split(" ", 1)[0]
        if token:
            metrics.add(token)
    return metrics


def available_preferred_metrics() -> list[str]:
    available = query_metrics()
    if not available:
        return []
    return [metric for metric in PREFERRED_METRICS if metric in available]


def parse_ncu_csv(path: str | Path, required_metrics: list[str] | None = None) -> dict[str, Any]:
    required = required_metrics or PREFERRED_METRICS
    parsed: dict[str, Any] = {"metrics": {}, "warnings": []}
    p = Path(path)
    if not p.exists():
        parsed["warnings"].append(f"NCU CSV not found: {p}")
        return parsed

    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        for row in reader:
            name = row.get("Metric Name") or row.get("Metric") or row.get("Name")
            value = row.get("Metric Value") or row.get("Value")
            if name:
                parsed["metrics"][name] = value

    for metric in required:
        if metric not in parsed["metrics"]:
            parsed["warnings"].append(f"Missing NCU metric: {metric}")
    return parsed

