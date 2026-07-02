from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


FORBIDDEN_CONFIG_KEYS = {"pollute_size_mb", "cache_state"}
VALID_PRIME_POLICIES = {"per_repeat", "case_once", "none"}
DTYPE_BYTES = {"fp16": 2, "float16": 2, "bf16": 2, "bfloat16": 2, "fp32": 4, "float32": 4}


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


@dataclass(frozen=True)
class L2Config:
    effective_l2_mb: list[str | float] = field(default_factory=lambda: ["full", 40, 32, 25, 16, 12.5])
    hit_ratio: float = 1.0
    prime_policy: str = "per_repeat"
    keeper_repeats: int = 8


@dataclass(frozen=True)
class ExperimentConfig:
    operator: str
    dtype: str
    shapes: list[dict[str, Any]]
    repeat: int
    warmup: int
    l2: L2Config
    flashmla_path: str | None = None


def _walk_forbidden_keys(value: Any, path: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in FORBIDDEN_CONFIG_KEYS:
                hits.append(child_path)
            hits.extend(_walk_forbidden_keys(child, child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            hits.extend(_walk_forbidden_keys(child, f"{path}[{idx}]"))
    return hits


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")
    return data


def parse_config(path: str | Path) -> ExperimentConfig:
    data = load_yaml(path)
    forbidden = _walk_forbidden_keys(data)
    if forbidden:
        joined = ", ".join(forbidden)
        raise ConfigError(f"Forbidden legacy cache-pollution keys in config: {joined}")

    try:
        operator = str(data["operator"])
        dtype = str(data.get("dtype", "bf16"))
        shapes = data["shapes"]
        repeat = int(data.get("repeat", 100))
        warmup = int(data.get("warmup", 20))
    except KeyError as exc:
        raise ConfigError(f"Missing required config key: {exc.args[0]}") from exc

    if not isinstance(shapes, list) or not shapes or not all(isinstance(x, dict) for x in shapes):
        raise ConfigError("Config key 'shapes' must be a non-empty list of mappings")
    if repeat <= 0 or warmup < 0:
        raise ConfigError("repeat must be positive and warmup must be non-negative")
    if dtype not in DTYPE_BYTES:
        raise ConfigError(f"Unsupported dtype '{dtype}'. Supported: {sorted(DTYPE_BYTES)}")

    l2_data = data.get("l2", {})
    if not isinstance(l2_data, dict):
        raise ConfigError("Config key 'l2' must be a mapping")
    l2 = L2Config(
        effective_l2_mb=list(l2_data.get("effective_l2_mb", ["full", 40, 32, 25, 16, 12.5])),
        hit_ratio=float(l2_data.get("hit_ratio", 1.0)),
        prime_policy=str(l2_data.get("prime_policy", "per_repeat")),
        keeper_repeats=int(l2_data.get("keeper_repeats", 8)),
    )
    if not (0.0 < l2.hit_ratio <= 1.0):
        raise ConfigError("l2.hit_ratio must be in (0, 1]")
    if l2.prime_policy not in VALID_PRIME_POLICIES:
        raise ConfigError(f"l2.prime_policy must be one of {sorted(VALID_PRIME_POLICIES)}")
    if l2.keeper_repeats <= 0:
        raise ConfigError("l2.keeper_repeats must be positive")

    flashmla = data.get("flashmla", {})
    flashmla_path = None
    if isinstance(flashmla, dict):
        flashmla_path = flashmla.get("path")

    return ExperimentConfig(
        operator=operator,
        dtype=dtype,
        shapes=shapes,
        repeat=repeat,
        warmup=warmup,
        l2=l2,
        flashmla_path=flashmla_path,
    )


def dtype_size(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ConfigError(f"Unsupported dtype '{dtype}'") from exc

