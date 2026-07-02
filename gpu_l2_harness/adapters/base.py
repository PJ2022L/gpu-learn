from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AdapterError(RuntimeError):
    """Raised when a benchmark adapter cannot run."""


class BenchmarkAdapter(ABC):
    @abstractmethod
    def run(self, task: dict[str, Any], env: dict[str, Any]) -> dict[str, Any]:
        """Run a benchmark task and return a standard result dictionary."""

