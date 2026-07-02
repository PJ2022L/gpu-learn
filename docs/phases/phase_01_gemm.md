# Phase 01: GEMM Microbenchmark

## 目标

用 GEMM 建立 L2 persistent set-aside 实验方法的对照组。

## 执行 Agent

Planner 调度 Runner；必要时调度 Analyzer。

## 输入

- `configs/gemm_persistent.yaml`
- [../l2_persistent.md](../l2_persistent.md)

## 操作

```bash
gpu-l2 plan --config configs/gemm_persistent.yaml --run-id <run_id> --dry-run
gpu-l2 run --config configs/gemm_persistent.yaml --run-id <run_id>
gpu-l2 analyze --run-id <run_id>
```

## 输出

- `results/<run_id>/task_manifest.json`
- `results/<run_id>/raw/*.json`
- `results/<run_id>/parsed/analysis.json`

## 验收标准

- 每个 GEMM task 输出 latency、TFLOPS、working set bytes。
- full L2 baseline 存在。
- 不支持的 set-aside 点标记 `unsupported`，不静默截断。

