from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from gpu_l2_harness.adapters.mock_flashmla import make_mock_result
from gpu_l2_harness.l2_persistent import L2Props
from gpu_l2_harness.planner import plan_tasks
from gpu_l2_harness.profiler import parse_ncu_csv
from gpu_l2_harness.schemas import ConfigError, L2Config, ExperimentConfig, parse_config


class HarnessTests(unittest.TestCase):
    def test_schema_rejects_legacy_pollution_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "operator: gemm\n"
                "dtype: bf16\n"
                "shapes: [{M: 1, N: 1, K: 1}]\n"
                "pollute_size_mb: [64]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                parse_config(path)

    def test_schema_rejects_cache_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text(
                "operator: gemm\n"
                "dtype: bf16\n"
                "shapes: [{M: 1, N: 1, K: 1}]\n"
                "cache_state: [warm, cold]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                parse_config(path)

    def test_planner_converts_effective_l2_to_setaside(self) -> None:
        cfg = ExperimentConfig(
            operator="gemm",
            dtype="bf16",
            shapes=[{"M": 1024, "N": 1024, "K": 1024}],
            repeat=10,
            warmup=2,
            l2=L2Config(effective_l2_mb=["full", 40]),
        )
        props = L2Props(
            total_bytes=50 * 1024 * 1024,
            persisting_max_bytes=40 * 1024 * 1024,
            access_policy_max_window_bytes=40 * 1024 * 1024,
            source="test",
        )
        tasks = plan_tasks(cfg, props)
        self.assertEqual(tasks[0].l2["persisting_setaside_bytes"], 0)
        self.assertEqual(tasks[1].l2["persisting_setaside_bytes"], 10 * 1024 * 1024)
        self.assertEqual(tasks[1].status, "planned")

    def test_planner_marks_unsupported_when_setaside_too_large(self) -> None:
        cfg = ExperimentConfig(
            operator="gemm",
            dtype="bf16",
            shapes=[{"M": 1, "N": 1, "K": 1}],
            repeat=1,
            warmup=0,
            l2=L2Config(effective_l2_mb=[12.5]),
        )
        props = L2Props(
            total_bytes=50 * 1024 * 1024,
            persisting_max_bytes=16 * 1024 * 1024,
            access_policy_max_window_bytes=16 * 1024 * 1024,
            source="test",
        )
        task = plan_tasks(cfg, props)[0]
        self.assertEqual(task.status, "unsupported")
        self.assertIn("exceeds", task.unsupported_reason or "")

    def test_mock_flashmla_result_has_standard_keys(self) -> None:
        task = {
            "task_id": "t0",
            "operator": "flashmla_dense_decode",
            "shape": {"batch_size": 1, "seq_len": 1024},
            "dtype": "bf16",
            "repeat": 1,
            "warmup": 0,
            "l2": {
                "total_bytes": 0,
                "persisting_setaside_bytes": 0,
                "effective_available_bytes": 0,
                "hit_ratio": 1.0,
                "prime_policy": "per_repeat",
            },
        }
        result = make_mock_result(task, {"git_commit": "abc"})
        for key in ["task_id", "operator", "shape", "dtype", "repeat", "warmup", "l2", "timing", "env", "artifacts"]:
            self.assertIn(key, result)

    def test_mock_gemm_result_has_standard_keys(self) -> None:
        task = {
            "task_id": "g0",
            "operator": "gemm",
            "shape": {"M": 16, "N": 16, "K": 16},
            "dtype": "bf16",
            "repeat": 1,
            "warmup": 0,
            "l2": {
                "total_bytes": 0,
                "persisting_setaside_bytes": 0,
                "effective_available_bytes": 0,
                "hit_ratio": 1.0,
                "prime_policy": "per_repeat",
            },
        }
        result = make_mock_result(task, {"git_commit": "abc"})
        self.assertEqual(result["operator"], "gemm")
        self.assertEqual(result["status"], "mock")

    def test_ncu_parser_warns_on_missing_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ncu.csv"
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["Metric Name", "Metric Value"])
                writer.writeheader()
                writer.writerow({"Metric Name": "dram__bytes.sum", "Metric Value": "123"})
            parsed = parse_ncu_csv(path, required_metrics=["dram__bytes.sum", "lts__t_sector_hit_rate.pct"])
            self.assertEqual(parsed["metrics"]["dram__bytes.sum"], "123")
            self.assertTrue(any("lts__t_sector_hit_rate" in w for w in parsed["warnings"]))


if __name__ == "__main__":
    unittest.main()
