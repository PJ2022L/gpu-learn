# Analyzer Agent

Analyzer Agent 负责把 raw benchmark 和 NCU 数据转成结论候选。

## 职责

- 解析 `results/<run_id>/raw/*.json`。
- 解析 `results/<run_id>/ncu/*.csv`。
- 计算 latency ratio、capacity slope、L2 hit drop、DRAM bytes ratio。
- 初步分类 compute-bound、L2-sensitive、HBM-bound 或 uncertain。
- 输出需要 profile 或补实验的点。

## 输入

- `results/<run_id>/raw/*.json`
- `results/<run_id>/ncu/*.csv`
- `docs/data_contracts.md`

## 输出

- `results/<run_id>/parsed/analysis.json`
- `results/<run_id>/parsed/summary.csv`
- 给 Refiner 的 followup candidates
- 给 Reporter 的 evidence summary

## 分类规则

基础 latency 分类：

```text
latency_ratio < 1.05        not_l2_sensitive
1.05 <= ratio < 1.15        mild_l2_sensitive
1.15 <= ratio < 1.30        moderate_l2_sensitive
ratio >= 1.30               strong_l2_sensitive
```

强证据需要同时满足：

```text
latency 上升
L2 hit rate 下降
DRAM bytes 或 DRAM throughput 上升
```

如果只有 latency，没有 NCU 证据，结论必须标记 `latency_only` 或 `uncertain`。

## 失败处理

- raw JSON 缺失：标记 task incomplete。
- NCU CSV 缺失：不要强行给因果结论。
- full L2 baseline 缺失：该 shape 不计算 ratio，交给 Refiner 补 baseline。

