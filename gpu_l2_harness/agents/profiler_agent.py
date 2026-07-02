from __future__ import annotations

from gpu_l2_harness.agents.base import AgentRole


class ProfilerAgent:
    role = AgentRole(
        name="Profiler Agent",
        doc_path="docs/agents/profiler.md",
        responsibility="运行 Nsight Compute，保存 CSV、profile metadata 和 metric warning。",
        reads=("results/<run_id>/task_manifest.json", "results/<run_id>/raw/*.json"),
        writes=("results/<run_id>/ncu/*.csv", "results/<run_id>/ncu/*.json"),
    )

    def handoff_summary(self) -> dict[str, object]:
        return self.role.handoff_summary()

