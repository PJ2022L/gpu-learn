# Phase 04: Dense Decoding 拆解

## 目标

研究 Dense Decoding 的连续 KV 访问是否受有效 L2 容量影响。

## 执行 Agent

Planner 调度 Runner；Analyzer 初步判断。

## 输入

- `configs/flashmla_dense_decode.yaml`
- FlashMLA Dense Decode adapter

## 操作

1. 选择 seq_len sweep。
2. 对每个 seq_len 运行 effective L2 sweep。
3. 记录 latency、tokens/s、effective bandwidth。

## 输出

- `results/<run_id>/raw/flashmla_dense_decode_*.json`
- `results/<run_id>/parsed/analysis.json`

## 验收标准

- 至少覆盖短、中、长 seq_len。
- 每个 shape 有 full L2 baseline。
- Analyzer 能给出 Dense Decode 是否需要 NCU profile 的建议。

