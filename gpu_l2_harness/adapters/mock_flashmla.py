from __future__ import annotations

from typing import Any

from gpu_l2_harness.adapters.base import AdapterError, BenchmarkAdapter


class MockFlashMLAAdapter(BenchmarkAdapter):
    """Placeholder adapter until FlashMLA is present.

    This adapter intentionally does not rewrite FlashMLA kernels. It provides a
    mock path for unit tests and a clear runtime error for real runs.
    """

    def __init__(self, operator: str) -> None:
        self.operator = operator

    def run(self, task: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
        if task.get("mock", False):
            return make_mock_result(task, env)
        raise AdapterError(
            f"{self.operator} requires official FlashMLA Python APIs in-process. "
            "Set flashmla.path in config and implement the thin API binding once "
            "the local FlashMLA checkout is available."
        )


def make_mock_result(task: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "operator": task["operator"],
        "shape": task["shape"],
        "dtype": task["dtype"],
        "repeat": task["repeat"],
        "warmup": task["warmup"],
        "l2": task["l2"],
        "timing": {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0},
        "tflops": 0.0,
        "effective_bandwidth_gbps": 0.0,
        "env": env,
        "artifacts": {"stdout": "", "stderr": "", "ncu_csv": ""},
        "status": "mock",
    }
