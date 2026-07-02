from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def classify_latency_ratio(ratio: float | None) -> str:
    if ratio is None:
        return "unknown"
    if ratio < 1.05:
        return "not_l2_sensitive"
    if ratio < 1.15:
        return "mild_l2_sensitive"
    if ratio < 1.30:
        return "moderate_l2_sensitive"
    return "strong_l2_sensitive"


def load_results(run_id: str) -> list[dict[str, Any]]:
    raw_dir = Path("results") / run_id / "raw"
    if not raw_dir.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(raw_dir.glob("*.json"))]


def analyze_results(run_id: str) -> dict[str, Any]:
    results = load_results(run_id)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        shape_key = json.dumps(row.get("shape", {}), sort_keys=True)
        groups[(row.get("operator", ""), row.get("dtype", ""), shape_key)].append(row)

    summaries: list[dict[str, Any]] = []
    for (operator, dtype, shape_key), rows in groups.items():
        full = next((r for r in rows if r.get("l2", {}).get("effective_l2_label") == "full"), None)
        full_latency = _latency(full)
        for row in rows:
            latency = _latency(row)
            ratio = (latency / full_latency) if latency is not None and full_latency and full_latency > 0 else None
            effective = row.get("l2", {}).get("effective_available_bytes")
            full_effective = full.get("l2", {}).get("effective_available_bytes") if full else None
            removed_mb = ((full_effective - effective) / (1024 * 1024)) if full_effective and effective is not None else None
            slope = ((latency - full_latency) / removed_mb) if latency is not None and full_latency and removed_mb else None
            summaries.append(
                {
                    "task_id": row.get("task_id"),
                    "operator": operator,
                    "dtype": dtype,
                    "shape": json.loads(shape_key),
                    "status": row.get("status"),
                    "effective_l2_label": row.get("l2", {}).get("effective_l2_label"),
                    "latency_ms": latency,
                    "latency_ratio": ratio,
                    "capacity_slope_ms_per_mb_removed": slope,
                    "classification": classify_latency_ratio(ratio),
                }
            )

    out = {"run_id": run_id, "summary": summaries}
    parsed_dir = Path("results") / run_id / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / "analysis.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _latency(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    value = row.get("timing", {}).get("mean_ms")
    return float(value) if value is not None else None

