# Phase 08: 分析与自动补实验

## 目标

结合 benchmark 和 NCU 结果，判断 L2 sensitivity，并生成必要补实验。

## 执行 Agent

Planner 调度 Analyzer 和 Refiner。

## 输入

- `results/<run_id>/raw/*.json`
- `results/<run_id>/ncu/*.csv`
- `results/<run_id>/task_manifest.json`

## 操作

Analyzer：

```text
计算 latency_ratio、capacity_slope、l2_hit_drop、dram_bytes_ratio。
标记 evidence_status。
输出 classification。
```

Refiner：

```text
对突变区间加密 effective L2 点。
对高噪声点增加 repeat。
对证据不足点请求 NCU。
```

## 输出

- `results/<run_id>/parsed/analysis.json`
- `results/<run_id>/parsed/summary.csv`
- `results/<run_id>/refine_plan.json`

## 验收标准

- 结论区分 latency-only 和 NCU-supported。
- 补实验数量受控。
- 不把 unsupported 点转换成 pollute 实验。

