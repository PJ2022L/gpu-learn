# Phase 06: Sparse Prefill 拆解

## 目标

判断 Sparse Prefill 是 compute-bound、L2-sensitive 还是 HBM-bound。

## 执行 Agent

Planner 调度 Runner；Analyzer 初步判断。

## 输入

- `configs/flashmla_sparse_prefill.yaml`
- FlashMLA Sparse Prefill adapter

## 操作

1. 选择 prefill_len sweep。
2. 运行 effective L2 sweep。
3. 记录 latency、TFLOPS、effective bandwidth。

## 输出

- `results/<run_id>/raw/flashmla_sparse_prefill_*.json`
- `results/<run_id>/parsed/analysis.json`

## 验收标准

- latency 随 prefill_len 的变化可比较。
- Analyzer 能标记 compute-bound / memory-bound / uncertain 候选。
- 对关键 shape 生成 NCU profile 请求。

