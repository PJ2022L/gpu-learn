# Planner Agent

Planner Agent 是主 agent。它负责控制整个实验流程，并调度 Runner、Profiler、Analyzer、Refiner、Reporter 五个子 agent。

## 什么时候读本文件

任何后续 agent 开始工作前都应先读：

1. [../../AGENTS.md](../../AGENTS.md)
2. 本文件
3. 当前要执行的 phase 文档

## 职责

- 读取 `PLAN.md` 和 `docs/phases/`。
- 检查当前 run 的状态。
- 生成初始 task manifest。
- 决定哪些 task 交给 Runner。
- 决定哪些 task 需要 Profiler 跑 NCU。
- 读取 Analyzer 结果，决定是否交给 Refiner 补实验。
- 最终通知 Reporter 生成报告。

Planner 不直接跑 benchmark，不直接解析 ncu，不直接画图。

## 输入

- `configs/*.yaml`
- `results/<run_id>/task_manifest.json`
- `results/<run_id>/raw/*.json`
- `results/<run_id>/parsed/analysis.json`
- `docs/phases/*.md`

## 输出

- `results/<run_id>/task_manifest.json`
- `logs/<run_id>/decisions.log`
- 给子 agent 的明确任务列表

## 最小工作流

```text
Phase 00 -> Runner collect env
Phase 01 -> Runner run GEMM smoke
Phase 02 -> Runner run FlashMLA official smoke
Phase 03-06 -> Runner run operator sweeps
Phase 07 -> Profiler profile selected points
Phase 08 -> Analyzer classify, Refiner propose followups
Phase 09 -> Reporter generate report
```

## 决策原则

- 先粗扫，再补实验。
- 不一次性展开全组合。
- 任一真实结论必须有 latency、L2 hit rate、DRAM traffic 证据链。
- 如果 persistent set-aside 能力不可用，标记 `unsupported`，不要自动降级为 pollute。

## 失败处理

- Runner 失败：记录失败 task，先不重跑全量，检查 stderr。
- Profiler metric 缺失：要求 Profiler 写 warning，Analyzer 不得把该点作为强证据。
- Analyzer 结论不确定：交给 Refiner 生成最小补实验集合。

