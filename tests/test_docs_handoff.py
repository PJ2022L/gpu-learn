from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocsHandoffTests(unittest.TestCase):
    def test_agents_entrypoint_exists_and_routes_to_planner(self) -> None:
        text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("Planner Agent 是主 agent", text)
        self.assertIn("docs/agents/planner.md", text)
        self.assertIn("operatorsforge:h800-v1.0", text)
        self.assertIn("l2_mla_study", text)

    def test_all_phase_docs_have_required_sections(self) -> None:
        for idx in range(10):
            path = ROOT / "docs" / "phases" / f"phase_{idx:02d}_{self._suffix(idx)}.md"
            self.assertTrue(path.exists(), str(path))
            text = path.read_text(encoding="utf-8")
            for section in ["## 目标", "## 执行 Agent", "## 输入", "## 输出", "## 验收标准"]:
                self.assertIn(section, text, f"{path} missing {section}")

    def test_all_agent_docs_exist(self) -> None:
        for name in ["planner", "runner", "profiler", "analyzer", "refiner", "reporter"]:
            path = ROOT / "docs" / "agents" / f"{name}.md"
            self.assertTrue(path.exists(), str(path))
            self.assertIn("## 职责", path.read_text(encoding="utf-8"))

    def test_agent_role_imports(self) -> None:
        from gpu_l2_harness.agents import PlannerAgent, RunnerAgent

        self.assertIn("Planner Agent", PlannerAgent().handoff_summary()["name"])
        self.assertIn("Runner Agent", RunnerAgent().handoff_summary()["name"])

    @staticmethod
    def _suffix(idx: int) -> str:
        return [
            "env",
            "gemm",
            "flashmla_official",
            "flashmla_end2end",
            "dense_decode",
            "sparse_decode",
            "sparse_prefill",
            "ncu",
            "analyze_refine",
            "report",
        ][idx]


if __name__ == "__main__":
    unittest.main()
