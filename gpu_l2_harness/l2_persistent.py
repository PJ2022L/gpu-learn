from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class L2Props:
    total_bytes: int
    persisting_max_bytes: int
    access_policy_max_window_bytes: int
    source: str = "unknown"


def _load_ext() -> Any:
    try:
        from gpu_l2_harness import _l2p_ext  # type: ignore

        return _l2p_ext
    except Exception as exc:
        raise RuntimeError(
            "gpu_l2_harness._l2p_ext is not built. Install/build the CUDA extension "
            "before running real persistent-cache experiments."
        ) from exc


def query_l2_props(device: int = 0, allow_torch_fallback: bool = True) -> L2Props:
    try:
        ext = _load_ext()
        raw = ext.query_l2_props(device)
        return L2Props(
            total_bytes=int(raw["l2CacheSize"]),
            persisting_max_bytes=int(raw["persistingL2CacheMaxSize"]),
            access_policy_max_window_bytes=int(raw["accessPolicyMaxWindowSize"]),
            source="cuda_extension",
        )
    except Exception:
        if not allow_torch_fallback:
            raise

    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            return L2Props(
                total_bytes=int(getattr(props, "L2_cache_size", 0)),
                persisting_max_bytes=0,
                access_policy_max_window_bytes=0,
                source="torch_fallback_no_persisting_limits",
            )
    except Exception:
        pass

    return L2Props(0, 0, 0, source="unavailable")


class L2PersistentGuard(AbstractContextManager["L2PersistentGuard"]):
    """Set and prime CUDA persisting L2 for a single benchmark process."""

    def __init__(
        self,
        setaside_bytes: int,
        hit_ratio: float,
        keeper_repeats: int,
        prime_policy: str,
        device: int = 0,
    ) -> None:
        self.setaside_bytes = int(setaside_bytes)
        self.hit_ratio = float(hit_ratio)
        self.keeper_repeats = int(keeper_repeats)
        self.prime_policy = prime_policy
        self.device = int(device)
        self._keeper = None
        self._ext = None
        self.actual_limit_bytes = 0

    def __enter__(self) -> "L2PersistentGuard":
        if self.setaside_bytes <= 0 or self.prime_policy == "none":
            return self
        self._ext = _load_ext()
        import torch

        torch.cuda.set_device(self.device)
        self.actual_limit_bytes = int(self._ext.set_persisting_limit(self.setaside_bytes))
        self._keeper = torch.empty((self.setaside_bytes,), dtype=torch.uint8, device=f"cuda:{self.device}")
        if self.prime_policy == "case_once":
            self.prime()
        return self

    def prime(self) -> None:
        if self.setaside_bytes <= 0 or self.prime_policy == "none":
            return
        if self._ext is None or self._keeper is None:
            raise RuntimeError("L2PersistentGuard must be entered before prime()")
        self._ext.prime_persisting(self._keeper, self.setaside_bytes, self.hit_ratio, self.keeper_repeats)

    def prime_before_repeat(self) -> None:
        if self.prime_policy == "per_repeat":
            self.prime()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._ext is not None:
            try:
                self._ext.reset_persisting_cache()
                self._ext.set_persisting_limit(0)
            finally:
                self._keeper = None

