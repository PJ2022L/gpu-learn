# H800 L2 Cache Sensitivity Multi-Agent Plan

本文件是总路线索引，不承载全部细节。后续 agent 必须从 [AGENTS.md](AGENTS.md) 开始，再按任务读取 `docs/` 下的局部文档。

## 0. 当前目标

构建一个多 agent 交接框架，用于研究 H800 上各个算子对 L2 cache 容量的敏感性。

当前阶段只完成：

- 文档入口与路由。
- 原始 10 phase 执行路线。
- CUDA L2 persistent set-aside 的工具代码引子。
- Docker 运行方式说明。

真实 H800 实验、FlashMLA 全量适配、NCU 批量 profile 和最终报告由后续 agent 按 phase 完成。

## 1. 核心方法

不使用 cache pollution kernel。

使用 CUDA L2 persistent set-aside：

```text
total_l2_bytes = runtime query
effective_l2_bytes = target available L2 for benchmark
persisting_setaside_bytes = total_l2_bytes - effective_l2_bytes
```

Runner 在同一进程中设置 persisting limit、prime keeper buffer，再运行目标算子。最终结论必须用 latency、L2 hit rate、DRAM traffic 共同支持。

方法细节见 [docs/l2_persistent.md](docs/l2_persistent.md)。

## 2. Agent 架构

主 agent：

- [Planner Agent](docs/agents/planner.md)

子 agent：

- [Runner Agent](docs/agents/runner.md)
- [Profiler Agent](docs/agents/profiler.md)
- [Analyzer Agent](docs/agents/analyzer.md)
- [Refiner Agent](docs/agents/refiner.md)
- [Reporter Agent](docs/agents/reporter.md)

`gpu_l2_harness/` 是工具库，不是 agent 系统本身。

## 3. Docker 环境

H800 默认使用已经构建好的容器：

```text
image: operatorsforge:h800-v1.0
container name: l2_mla_study
```

推荐入口见 [docs/docker.md](docs/docker.md)。

当前开发会话没有 Docker daemon 权限，因此不能在本次会话验证容器执行。

## 4. 10 Phase 路线

按顺序推进：

1. [Phase 00: 环境检查](docs/phases/phase_00_env.md)
2. [Phase 01: GEMM Microbenchmark](docs/phases/phase_01_gemm.md)
3. [Phase 02: FlashMLA 官方 Benchmark 跑通](docs/phases/phase_02_flashmla_official.md)
4. [Phase 03: FlashMLA 整体性能](docs/phases/phase_03_flashmla_end2end.md)
5. [Phase 04: Dense Decoding 拆解](docs/phases/phase_04_dense_decode.md)
6. [Phase 05: Sparse Decoding 拆解](docs/phases/phase_05_sparse_decode.md)
7. [Phase 06: Sparse Prefill 拆解](docs/phases/phase_06_sparse_prefill.md)
8. [Phase 07: Nsight Compute Profile](docs/phases/phase_07_ncu.md)
9. [Phase 08: 分析与自动补实验](docs/phases/phase_08_analyze_refine.md)
10. [Phase 09: 报告生成](docs/phases/phase_09_report.md)

## 5. 数据契约

所有 agent 通过文件交接。字段定义见 [docs/data_contracts.md](docs/data_contracts.md)。

关键目录：

```text
configs/                  实验配置
gpu_l2_harness/           工具层代码
gpu_l2_harness/agents/    agent 轻量入口
csrc/                     L2 persistent CUDA extension
docs/                     渐进披露文档
results/<run_id>/         实验输出
logs/<run_id>/            命令、stdout/stderr、决策日志
```

## 6. 禁止事项

- 不恢复 `pollute_size_mb`、`cache_state`、`l2_pollute.*`。
- 不硬编码 H800 L2 容量。
- 不在没有 NCU 证据时写强因果结论。
- 不把 FlashMLA kernel 重写到本仓库。
- 不覆盖历史结果。
