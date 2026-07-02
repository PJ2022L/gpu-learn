# Profiler Agent

Profiler Agent 负责 Nsight Compute 采集。

## 职责

- 只 profile Planner/Analyzer 选出的关键点。
- 先运行 `ncu --query-metrics`，过滤当前版本不存在的 metric。
- 保存 ncu 命令、CSV、缺失 metric warning。
- 不修改 benchmark 代码路径。

## 输入

- `results/<run_id>/task_manifest.json`
- `results/<run_id>/raw/*.json`
- Analyzer 给出的 profile candidate list

## 输出

- `results/<run_id>/ncu/<task_id>.csv`
- `results/<run_id>/ncu/<task_id>.json`
- `logs/<run_id>/command.log`

## 优先指标

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct
lts__t_sectors.sum
dram__bytes.sum
smsp__sass_thread_inst_executed_op_hmma_pred_on.sum
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
sm__warps_active.avg.pct_of_peak_sustained_active
launch__registers_per_thread
launch__shared_mem_per_block_static
```

## 选择 profile 点

优先级：

1. latency ratio 最大的点。
2. effective L2 从 full 降到较小时变化最明显的 shape。
3. 工作集接近 H800 L2 的 GEMM shape。
4. dense/sparse 差异最大的 FlashMLA 点。
5. Analyzer 标记为不确定但重要的点。

## 失败处理

- ncu 无权限或不可用：写入 warning，不中断整个 run。
- metric 名称不存在：记录缺失列表，继续采集可用指标。
- profile 开销导致超时：报告给 Planner，让 Refiner 缩小 profile 集合。

