# 数据契约

本文件定义 agent 之间传递的数据格式。后续 agent 修改字段前必须同步更新这里。

## Task Manifest

路径：

```text
results/<run_id>/task_manifest.json
```

核心字段：

```json
{
  "config_path": "configs/gemm_persistent.yaml",
  "dry_run": true,
  "task_count": 1,
  "tasks": [
    {
      "task_id": "gemm_K1024_M1024_N1024_bf16_l2full_xxxxxxxx",
      "operator": "gemm",
      "shape": {"M": 1024, "N": 1024, "K": 1024},
      "dtype": "bf16",
      "repeat": 100,
      "warmup": 20,
      "l2": {
        "total_bytes": 52428800,
        "persisting_setaside_bytes": 0,
        "effective_available_bytes": 52428800,
        "effective_l2_label": "full",
        "hit_ratio": 1.0,
        "prime_policy": "per_repeat",
        "keeper_repeats": 8,
        "props_source": "cuda_extension"
      },
      "status": "planned",
      "unsupported_reason": null
    }
  ]
}
```

`status` 可选值：

- `planned`
- `unsupported`
- `running`
- `success`
- `failed`

## Benchmark Result JSON

路径：

```text
results/<run_id>/raw/<task_id>.json
```

必须包含：

```json
{
  "task_id": "",
  "operator": "",
  "shape": {},
  "dtype": "bf16",
  "repeat": 100,
  "warmup": 20,
  "l2": {
    "total_bytes": 0,
    "persisting_setaside_bytes": 0,
    "effective_available_bytes": 0,
    "hit_ratio": 1.0,
    "prime_policy": "per_repeat"
  },
  "timing": {"mean_ms": 0.0, "std_ms": 0.0, "cv": 0.0},
  "tflops": 0.0,
  "effective_bandwidth_gbps": 0.0,
  "env": {},
  "artifacts": {"stdout": "", "stderr": "", "ncu_csv": ""},
  "status": "success"
}
```

失败时 `timing/tflops/effective_bandwidth_gbps` 可为 null，但必须有 `status` 和失败 artifact。

## NCU CSV

路径：

```text
results/<run_id>/ncu/<task_id>.csv
```

Profiler 必须保存：

- 实际 ncu 命令。
- ncu CSV。
- 当前 ncu 版本。
- 可用 metric 列表和缺失 metric warning。

优先指标见 [agents/profiler.md](agents/profiler.md)。

## Analysis JSON

路径：

```text
results/<run_id>/parsed/analysis.json
```

核心字段：

```json
{
  "run_id": "",
  "summary": [
    {
      "task_id": "",
      "operator": "",
      "shape": {},
      "latency_ratio": 1.0,
      "capacity_slope_ms_per_mb_removed": 0.0,
      "classification": "not_l2_sensitive",
      "evidence_status": "latency_only"
    }
  ]
}
```

最终科学结论不能只用 `latency_ratio`，必须结合 NCU 证据。

## Report Artifacts

路径：

```text
results/<run_id>/reports/report.md
results/<run_id>/parsed/summary.csv
results/<run_id>/figures/<fig_name>/plot.py
```

每张图必须保留生成脚本。

