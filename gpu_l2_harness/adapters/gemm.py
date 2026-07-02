from __future__ import annotations

import math
import statistics
from typing import Any

from gpu_l2_harness.adapters.base import BenchmarkAdapter
from gpu_l2_harness.l2_persistent import L2PersistentGuard
from gpu_l2_harness.schemas import dtype_size


TORCH_DTYPES = {
    "fp16": "float16",
    "float16": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}


class GemmAdapter(BenchmarkAdapter):
    def run(self, task: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
        import torch

        shape = task["shape"]
        m, n, k = int(shape["M"]), int(shape["N"]), int(shape["K"])
        dtype_name = TORCH_DTYPES[task["dtype"]]
        dtype = getattr(torch, dtype_name)
        repeat = int(task["repeat"])
        warmup = int(task["warmup"])
        l2 = task["l2"]

        a = torch.randn((m, k), device="cuda", dtype=dtype)
        b = torch.randn((k, n), device="cuda", dtype=dtype)
        guard = L2PersistentGuard(
            setaside_bytes=int(l2["persisting_setaside_bytes"]),
            hit_ratio=float(l2["hit_ratio"]),
            keeper_repeats=int(l2["keeper_repeats"]),
            prime_policy=str(l2["prime_policy"]),
        )

        latencies_ms: list[float] = []
        with guard:
            for _ in range(warmup):
                guard.prime_before_repeat()
                torch.matmul(a, b)
            torch.cuda.synchronize()

            for _ in range(repeat):
                guard.prime_before_repeat()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                c = torch.matmul(a, b)
                end.record()
                end.synchronize()
                latencies_ms.append(float(start.elapsed_time(end)))
                del c

        mean_ms = statistics.fmean(latencies_ms)
        std_ms = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
        flops = 2.0 * m * n * k
        tflops = flops / (mean_ms / 1000.0) / 1.0e12 if mean_ms > 0 else 0.0
        working_set = dtype_size(task["dtype"]) * (m * k + k * n + m * n)
        bandwidth = working_set / (mean_ms / 1000.0) / 1.0e9 if mean_ms > 0 else 0.0

        return {
            "task_id": task["task_id"],
            "operator": task["operator"],
            "shape": shape,
            "dtype": task["dtype"],
            "repeat": repeat,
            "warmup": warmup,
            "l2": l2,
            "timing": {
                "mean_ms": mean_ms,
                "std_ms": std_ms,
                "cv": std_ms / mean_ms if mean_ms > 0 and math.isfinite(mean_ms) else 0.0,
            },
            "tflops": tflops,
            "effective_bandwidth_gbps": bandwidth,
            "env": env,
            "artifacts": {"stdout": "", "stderr": "", "ncu_csv": ""},
            "status": "success",
        }

