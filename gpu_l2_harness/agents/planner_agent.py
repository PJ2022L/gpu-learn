from __future__ import annotations

from gpu_l2_harness.agents.base import AgentRole


class PlannerAgent:
    """Main controller role. The implementation lives in future agent work."""

    role = AgentRole(
        name="Planner Agent",
        doc_path="docs/agents/planner.md",
        responsibility="主 agent：生成计划、调度五个子 agent、决定补实验。",
        reads=(
            "AGENTS.md",
            "PLAN.md",
            "docs/phases/*.md",
            "configs/*.yaml",
            "results/<run_id>/parsed/analysis.json",
        ),
        writes=("results/<run_id>/task_manifest.json", "logs/<run_id>/decisions.log"),
    )

    def handoff_summary(self) -> dict[str, object]:
        return self.role.handoff_summary()

