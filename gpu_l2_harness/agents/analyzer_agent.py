from __future__ import annotations

from gpu_l2_harness.agents.base import AgentRole


class AnalyzerAgent:
    role = AgentRole(
        name="Analyzer Agent",
        doc_path="docs/agents/analyzer.md",
        responsibility="解析 raw/ncu 结果，计算 sensitivity 指标和证据状态。",
        reads=("results/<run_id>/raw/*.json", "results/<run_id>/ncu/*.csv"),
        writes=("results/<run_id>/parsed/analysis.json", "results/<run_id>/parsed/summary.csv"),
    )

    def handoff_summary(self) -> dict[str, object]:
        return self.role.handoff_summary()

