# Phase 02: FlashMLA 官方 Benchmark 跑通

## 目标

先不做容量实验，确认 FlashMLA 官方 Dense/Sparse Decode 和 Sparse Prefill benchmark 能在 Docker/H800 环境运行。

## 执行 Agent

Planner 调度 Runner。

## 输入

- `configs/env.yaml`
- FlashMLA checkout，默认 `third_party/FlashMLA`
- 官方 FlashMLA README 或 benchmark 入口

## 操作

Runner 需要：

1. 定位 FlashMLA 源码。
2. 安装/编译 FlashMLA。
3. 运行官方最小 benchmark。
4. 保存原始 stdout/stderr。

## 输出

- `results/<run_id>/raw/flashmla_official_dense_decode.json`
- `results/<run_id>/raw/flashmla_official_sparse_decode.json`
- `results/<run_id>/raw/flashmla_official_sparse_prefill.json`

## 验收标准

- 三类官方 benchmark 至少能输出 latency 或吞吐指标。
- Runner 记录 FlashMLA commit。
- 不重写 FlashMLA kernel。

