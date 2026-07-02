from __future__ import annotations

from gpu_l2_harness.adapters.base import AdapterError, BenchmarkAdapter
from gpu_l2_harness.adapters.gemm import GemmAdapter
from gpu_l2_harness.adapters.mock_flashmla import MockFlashMLAAdapter

__all__ = ["AdapterError", "BenchmarkAdapter", "GemmAdapter", "MockFlashMLAAdapter", "get_adapter"]


def get_adapter(operator: str) -> BenchmarkAdapter:
    if operator == "gemm":
        return GemmAdapter()
    if operator in {"flashmla_dense_decode", "flashmla_sparse_decode", "flashmla_sparse_prefill"}:
        return MockFlashMLAAdapter(operator)
    raise AdapterError(f"Unknown operator: {operator}")

