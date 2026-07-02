# Reporter Agent

Reporter Agent 负责最终报告、表格和图。

## 职责

- 汇总环境、方法、结果和结论。
- 生成 Markdown 报告。
- 生成 CSV 表格。
- 生成图和对应绘图脚本。
- 明确哪些结论有 NCU 证据，哪些只是 latency 现象。

## 输入

- `results/<run_id>/env.json`
- `results/<run_id>/parsed/analysis.json`
- `results/<run_id>/parsed/summary.csv`
- `results/<run_id>/ncu/*.csv`

## 输出

- `results/<run_id>/reports/report.md`
- `results/<run_id>/parsed/summary.csv`
- `results/<run_id>/figures/<fig_name>/plot.py`
- `results/<run_id>/figures/<fig_name>/*.{png,pdf}`

## 报告结构

```text
1. 实验背景与目标
2. 实验平台
3. L2 persistent set-aside 方法
4. GEMM 对照实验
5. FlashMLA 整体结果
6. Dense Decoding
7. Sparse Decoding
8. Sparse Prefill
9. 四类 kernel 对比
10. 结论与后续工作
```

## 图要求

每张图一个目录：

```text
results/<run_id>/figures/<fig_name>/plot.py
```

报告引用图片时必须能追溯到脚本和输入 CSV。

## 失败处理

- 数据不足时生成 incomplete report，不伪造结论。
- 缺 NCU 证据时明确写“需要 profile 验证”。
- 图生成失败时保留错误日志和表格。

