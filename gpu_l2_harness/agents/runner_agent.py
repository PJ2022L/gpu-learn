from __future__ import annotations

from gpu_l2_harness.agents.base import AgentRole


class RunnerAgent:
    role = AgentRole(
        name="Runner Agent",
        doc_path="docs/agents/runner.md",
        responsibility="执行 benchmark，保存 raw JSON、stdout、stderr 和环境信息。",
        reads=("results/<run_id>/task_manifest.json", "configs/*.yaml"),
        writes=("results/<run_id>/raw/*.json", "logs/<run_id>/*.stdout", "logs/<run_id>/*.stderr"),
    )

    def handoff_summary(self) -> dict[str, object]:
        return self.role.handoff_summary()

