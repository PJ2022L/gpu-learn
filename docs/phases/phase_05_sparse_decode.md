# Phase 05: Sparse Decoding 拆解

## 目标

研究 Sparse Decoding 中 sparse index、top-k metadata 和非连续 KV 访问对 L2 的依赖。

## 执行 Agent

Planner 调度 Runner；Analyzer 初步判断。

## 输入

- `configs/flashmla_sparse_decode.yaml`
- FlashMLA Sparse Decode adapter

## 操作

1. 选择 seq_len、top_k_blocks、block_size 的小规模组合。
2. 运行 effective L2 sweep。
3. 对 random/local/clustered 等 sparse pattern 保持配置可追踪。

## 输出

- `results/<run_id>/raw/flashmla_sparse_decode_*.json`
- `results/<run_id>/parsed/analysis.json`

## 验收标准

- 能比较 top_k 增大时的 latency ratio。
- 能区分访存量减少和局部性变差两个可能原因。
- 需要 NCU 时给 Profiler candidate。

