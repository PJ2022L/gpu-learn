# 项目概览

## 研究问题

目标是在 H800 上研究各类算子对有效 L2 容量的敏感性：

- GEMM microbenchmark
- FlashMLA 整体性能
- FlashMLA Dense Decoding
- FlashMLA Sparse Decoding
- FlashMLA Sparse Prefill

最终报告需要回答：

- 哪类 kernel 最 L2-sensitive。
- 哪些 shape 区间出现容量边界效应。
- latency 变化是否能被 L2 hit rate 和 DRAM traffic 解释。
- 整体 MLA 性能变化主要来自哪个拆解 kernel。

## 方法变更

本项目不使用 cache pollution kernel 作为主方法。原因：

- pollute kernel 容易引入额外调度、带宽和同步干扰。
- pollute size 不等价于目标 kernel 可用 L2 容量。
- 对 FlashMLA 这类复杂 kernel，pollute 后观察 latency 不容易建立因果链。

采用 CUDA L2 persistent set-aside：

```text
total_l2_bytes = cudaDeviceProp.l2CacheSize
effective_l2_bytes = 目标可用 L2 容量
persisting_setaside_bytes = total_l2_bytes - effective_l2_bytes
```

实验流程：

1. `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persisting_setaside_bytes)`。
2. 创建 keeper buffer。
3. 用 stream access policy window 将 keeper buffer 标为 `cudaAccessPropertyPersisting`。
4. prime keeper buffer，使 set-aside 部分尽可能被 persistent lines 占用。
5. 在同一进程中运行目标算子，但目标算子不标记 persisting。
6. 用 Nsight Compute 检查 L2 hit rate 和 DRAM traffic。

注意：这不是硬件物理切分。normal access 在 set-aside 未被 persistent 数据占用时仍可能使用该区域。因此最终结论必须结合 NCU 证据。

## 文档路由

- 环境和 Docker：读 [docker.md](docker.md) 和 [phases/phase_00_env.md](phases/phase_00_env.md)。
- 要做任务规划：读 [agents/planner.md](agents/planner.md)。
- 要执行 benchmark：读 [agents/runner.md](agents/runner.md)。
- 要 profile：读 [agents/profiler.md](agents/profiler.md)。
- 要分析结果：读 [agents/analyzer.md](agents/analyzer.md)。
- 要补实验：读 [agents/refiner.md](agents/refiner.md)。
- 要生成报告：读 [agents/reporter.md](agents/reporter.md)。

