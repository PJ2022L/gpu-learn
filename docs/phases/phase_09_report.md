# Phase 09: 报告生成

## 目标

生成最终 Markdown 报告、CSV 表格和可复现图。

## 执行 Agent

Planner 调度 Reporter。

## 输入

- `results/<run_id>/env.json`
- `results/<run_id>/parsed/analysis.json`
- `results/<run_id>/parsed/summary.csv`
- `results/<run_id>/ncu/*.csv`

## 操作

Reporter 生成：

```text
实验环境
实验方法
GEMM 对照结果
FlashMLA 整体结果
Dense Decoding
Sparse Decoding
Sparse Prefill
四类 kernel 对比
结论与后续工作
```

## 输出

- `results/<run_id>/reports/report.md`
- `results/<run_id>/parsed/summary.csv`
- `results/<run_id>/figures/<fig_name>/plot.py`
- 图像文件

## 验收标准

- 每张图有对应 `plot.py`。
- 每个强结论都指向 NCU 证据。
- 数据不足处明确标记 incomplete 或 uncertain。

