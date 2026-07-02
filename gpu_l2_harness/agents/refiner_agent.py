from __future__ import annotations

from gpu_l2_harness.agents.base import AgentRole


class RefinerAgent:
    role = AgentRole(
        name="Refiner Agent",
        doc_path="docs/agents/refiner.md",
        responsibility="根据 Analyzer 输出生成补实验和 NCU 候选点。",
        reads=("results/<run_id>/parsed/analysis.json", "results/<run_id>/task_manifest.json"),
        writes=("results/<run_id>/refine_plan.json",),
    )

    def handoff_summary(self) -> dict[str, object]:
        return self.role.handoff_summary()

