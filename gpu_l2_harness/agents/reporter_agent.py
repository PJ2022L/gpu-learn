from __future__ import annotations

from gpu_l2_harness.agents.base import AgentRole


class ReporterAgent:
    role = AgentRole(
        name="Reporter Agent",
        doc_path="docs/agents/reporter.md",
        responsibility="生成 Markdown 报告、CSV 表格、图和绘图脚本。",
        reads=("results/<run_id>/parsed/analysis.json", "results/<run_id>/ncu/*.csv"),
        writes=("results/<run_id>/reports/report.md", "results/<run_id>/figures/<fig_name>/plot.py"),
    )

    def handoff_summary(self) -> dict[str, object]:
        return self.role.handoff_summary()

