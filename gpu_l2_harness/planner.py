from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gpu_l2_harness.l2_persistent import L2Props, query_l2_props
from gpu_l2_harness.schemas import ExperimentConfig, dtype_size, parse_config


MB = 1024 * 1024


@dataclass(frozen=True)
class PlannedTask:
    task_id: str
    operator: str
    shape: dict[str, Any]
    dtype: str
    repeat: int
    warmup: int
    l2: dict[str, Any]
    status: str
    unsupported_reason: str | None = None
    flashmla_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_effective_l2(value: str | float | int, total_bytes: int) -> int:
    if isinstance(value, str) and value.lower() == "full":
        return int(total_bytes)
    return int(round(float(value) * MB))


def shape_working_set(operator: str, shape: dict[str, Any], dtype: str) -> int | None:
    if operator != "gemm":
        return None
    required = {"M", "N", "K"}
    if not required.issubset(shape):
        return None
    elem = dtype_size(dtype)
    return elem * (int(shape["M"]) * int(shape["K"]) + int(shape["K"]) * int(shape["N"]) + int(shape["M"]) * int(shape["N"]))


def make_task_id(operator: str, shape: dict[str, Any], dtype: str, effective_label: str) -> str:
    shape_part = "_".join(f"{k}{shape[k]}" for k in sorted(shape))
    raw = json.dumps({"op": operator, "shape": shape, "dtype": dtype, "l2": effective_label}, sort_keys=True)
    suffix = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{operator}_{shape_part}_{dtype}_l2{effective_label}_{suffix}".replace(".", "p")


def plan_tasks(config: ExperimentConfig, props: L2Props) -> list[PlannedTask]:
    tasks: list[PlannedTask] = []
    total = int(props.total_bytes)
    for shape in config.shapes:
        working_set = shape_working_set(config.operator, shape, config.dtype)
        for effective_value in config.l2.effective_l2_mb:
            label = str(effective_value).lower().replace(" ", "")
            effective_bytes = parse_effective_l2(effective_value, total) if total > 0 else 0
            setaside = max(total - effective_bytes, 0) if total > 0 else 0
            status = "planned"
            reason = None

            if total <= 0:
                status = "unsupported"
                reason = "Unable to query device L2 size"
            elif effective_bytes > total:
                status = "unsupported"
                reason = f"Requested effective_l2_bytes={effective_bytes} exceeds total_l2_bytes={total}"
            elif setaside > 0 and props.persisting_max_bytes <= 0:
                status = "unsupported"
                reason = "Persisting L2 max size unavailable; CUDA extension/device support is required"
            elif setaside > props.persisting_max_bytes:
                status = "unsupported"
                reason = (
                    f"Requested setaside_bytes={setaside} exceeds "
                    f"persistingL2CacheMaxSize={props.persisting_max_bytes}"
                )
            elif setaside > props.access_policy_max_window_bytes:
                status = "unsupported"
                reason = (
                    f"Requested access window bytes={setaside} exceeds "
                    f"accessPolicyMaxWindowSize={props.access_policy_max_window_bytes}"
                )

            l2 = {
                "total_bytes": total,
                "persisting_setaside_bytes": setaside,
                "effective_available_bytes": effective_bytes,
                "effective_l2_label": label,
                "hit_ratio": config.l2.hit_ratio,
                "prime_policy": config.l2.prime_policy,
                "keeper_repeats": config.l2.keeper_repeats,
                "props_source": props.source,
            }
            if working_set is not None:
                l2["working_set_bytes"] = working_set
                l2["working_set_over_effective_l2"] = (working_set / effective_bytes) if effective_bytes else None

            tasks.append(
                PlannedTask(
                    task_id=make_task_id(config.operator, shape, config.dtype, label),
                    operator=config.operator,
                    shape=dict(shape),
                    dtype=config.dtype,
                    repeat=config.repeat,
                    warmup=config.warmup,
                    l2=l2,
                    status=status,
                    unsupported_reason=reason,
                    flashmla_path=config.flashmla_path,
                )
            )
    return tasks


def plan_from_config(path: str | Path, device: int = 0) -> tuple[ExperimentConfig, L2Props, list[PlannedTask]]:
    config = parse_config(path)
    props = query_l2_props(device=device, allow_torch_fallback=True)
    return config, props, plan_tasks(config, props)


def write_manifest(tasks: list[PlannedTask], path: str | Path, config_path: str | Path, dry_run: bool) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": str(config_path),
        "dry_run": dry_run,
        "task_count": len(tasks),
        "tasks": [task.to_dict() for task in tasks],
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

