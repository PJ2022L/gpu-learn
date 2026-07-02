# Phase 07: Nsight Compute Profile

## 目标

对关键 shape 采集 NCU 指标，验证 latency 变化是否由 L2/DRAM 行为解释。

## 执行 Agent

Planner 调度 Profiler。

## 输入

- Analyzer 生成的 profile candidates。
- `results/<run_id>/task_manifest.json`
- `results/<run_id>/raw/*.json`

## 操作

1. `ncu --query-metrics`。
2. 过滤可用指标。
3. 对候选 task 运行 ncu。
4. 保存 CSV 和 profile metadata。

## 输出

- `results/<run_id>/ncu/<task_id>.csv`
- `results/<run_id>/ncu/<task_id>.json`

## 验收标准

- 每个 profile 点记录实际 ncu 命令。
- 缺失 metric 写 warning。
- 至少覆盖每类 kernel 的最敏感点。

