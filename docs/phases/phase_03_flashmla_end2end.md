# Phase 03: FlashMLA 整体性能

## 目标

观察整体 MLA 性能随 effective L2 容量变化的趋势。

## 执行 Agent

Planner 调度 Runner，Analyzer 做初步汇总。

## 输入

- FlashMLA 官方或项目封装的整体 benchmark 入口。
- L2 persistent 配置。

## 操作

1. Planner 选择少量 batch/seq_len 组合。
2. Runner 对每个组合执行 full L2 和 reduced effective L2。
3. Analyzer 计算 latency ratio。

## 输出

- `results/<run_id>/raw/flashmla_end2end_*.json`
- `results/<run_id>/parsed/analysis.json`

## 验收标准

- 能比较 full L2 与至少两个 reduced effective L2 点。
- 能指出整体性能最敏感的 seq_len 区间。
- 无 NCU 证据时只给 latency 现象，不给最终因果结论。

