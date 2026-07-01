# H800 上 L2 Cache 对 GEMM 与 FlashMLA 性能影响的自动化实验 Agent Plan

## 0. 项目目标

构建一个自动化 agent 系统，用于研究 **H800 上 L2 cache 压力对 GEMM 和 FlashMLA 性能的影响**。

研究对象分为两层：

1. **整体 MLA 性能层面**

   * 观察 L2 cache pressure 对 FlashMLA end-to-end 推理性能的影响。
   * 指标包括 latency、tokens/s、TFLOPS、effective bandwidth、L2 hit rate、DRAM traffic。

2. **kernel 拆解层面**

   * GEMM
   * FlashMLA Dense Decoding
   * FlashMLA Sparse Decoding
   * FlashMLA Sparse Prefill

注意：H800 属于 Hopper 架构实验平台，Hopper/H100 级架构 L2 容量为 50MB，并支持 L2 persistence 控制；本项目不是物理修改 L2 容量，而是通过 working set、cache warm/cold、pollution kernel、L2 persistence 等方式构造不同 L2 cache pressure。

FlashMLA 官方仓库已包含 Dense Decoding、Sparse Decoding、Sparse Prefill 相关 benchmark，并给出了 H800 SXM5 上的性能数据，因此本项目优先基于官方 FlashMLA 代码进行自动化封装。

---

## 1. 项目总路线

整体自动化流程如下：

```text
Step 0：环境检测
↓
Step 1：GEMM microbenchmark 建立 L2 实验方法
↓
Step 2：FlashMLA 官方 benchmark 跑通
↓
Step 3：整体 MLA 性能测试
↓
Step 4：拆解 Dense Decoding / Sparse Decoding / Sparse Prefill
↓
Step 5：加入 L2 warm/cold/pollution/persistence 实验变量
↓
Step 6：用 Nsight Compute 采集关键指标
↓
Step 7：自动判断 L2 sensitivity
↓
Step 8：自动补实验
↓
Step 9：生成最终报告
```

---

## 2. Agent 系统角色划分

不要让一个 agent 什么都做。建议拆成 6 个角色。

### 2.1 Planner Agent

职责：

```text
生成实验矩阵
决定哪些 shape 先粗扫
决定哪些 shape 需要补实验
避免全组合爆炸
```

输入：

```text
GPU 类型
L2 容量
FlashMLA kernel 类型
用户设定的 shape 范围
历史实验结果
```

输出：

```text
YAML 实验配置
下一轮实验任务列表
```

---

### 2.2 Runner Agent

职责：

```text
执行 benchmark
远程连接 H800 服务器
运行 GEMM / FlashMLA 脚本
保存 stdout、stderr、CSV、JSON
```

要求：

```text
不能改 evaluator
不能改 hidden test
不能覆盖历史结果
每次运行必须记录 git commit、CUDA 版本、PyTorch 版本、GPU 信息
```

---

### 2.3 Profiler Agent

职责：

```text
调用 Nsight Compute
采集 kernel 级性能指标
保存 ncu CSV
```

重点指标：

```text
kernel latency
SM utilization
Tensor Core utilization
L2 hit rate
L2 traffic
DRAM bytes
DRAM throughput
achieved occupancy
register usage
shared memory usage
```

Nsight Compute 的 profiling guide 明确说明，当 L2 缓存没有命中时，L2 hit rate 会下降，DRAM throughput 会升高，SM Tensor Pipe Throughput 可能因为等待数据而下降；这正好对应本项目判断 L2 瓶颈的核心证据链。

---

### 2.4 Analyzer Agent

职责：

```text
读取 benchmark CSV
读取 ncu CSV
计算 L2 sensitivity score
判断性能下降是否来自 L2
区分 compute-bound、L2-bound、HBM-bound
```

输出：

```text
每个 kernel 的 L2 敏感性结论
需要补实验的 shape
异常点说明
```

---

### 2.5 Refiner Agent

职责：

```text
根据 Analyzer 结果自动加密实验
如果发现 50MB 附近有性能拐点，就在附近增加 shape
如果噪声太大，就增加 repeat
如果 warm/cold 差异明显，就增加 ncu profile
```

---

### 2.6 Reporter Agent

职责：

```text
生成最终报告
生成图表
生成表格
总结整体 MLA 和三个 kernel 的因果关系
```

报告必须包含：

```text
实验环境
实验方法
GEMM 对照结果
FlashMLA 整体结果
Dense Decoding 拆解
Sparse Decoding 拆解
Sparse Prefill 拆解
L2 sensitivity 总结
结论与后续工作
```

---

## 3. 推荐目录结构

```text
l2_mla_agent/
├── README.md
├── requirements.txt
├── setup_env.sh
│
├── configs/
│   ├── env.yaml
│   ├── gemm_sweep.yaml
│   ├── flashmla_end2end.yaml
│   ├── flashmla_dense_decode.yaml
│   ├── flashmla_sparse_decode.yaml
│   └── flashmla_sparse_prefill.yaml
│
├── benchmarks/
│   ├── bench_gemm.py
│   ├── bench_flashmla_end2end.py
│   ├── bench_flashmla_dense_decode.py
│   ├── bench_flashmla_sparse_decode.py
│   ├── bench_flashmla_sparse_prefill.py
│   ├── l2_pollute.cu
│   └── l2_pollute.py
│
├── runners/
│   ├── run_task.py
│   ├── run_sweep.py
│   ├── run_remote_ssh.py
│   ├── run_ncu.py
│   └── collect_env.py
│
├── analyzers/
│   ├── parse_benchmark.py
│   ├── parse_ncu.py
│   ├── compute_working_set.py
│   ├── l2_sensitivity.py
│   ├── classify_bottleneck.py
│   └── plot_results.py
│
├── agents/
│   ├── planner_agent.py
│   ├── runner_agent.py
│   ├── profiler_agent.py
│   ├── analyzer_agent.py
│   ├── refiner_agent.py
│   └── reporter_agent.py
│
├── results/
│   ├── raw/
│   ├── parsed/
│   ├── ncu/
│   ├── figures/
│   └── reports/
│
├── logs/
│   ├── runs/
│   ├── errors/
│   └── decisions/
│
└── controller.py
```

---

## 4. Phase 0：环境搭建与自检

### 4.1 目标

确认 H800 服务器上具备完整实验环境。

### 4.2 Agent 需要执行

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"
ncu --version
nsys --version
```

### 4.3 需要记录

```text
GPU 型号
GPU 数量
显存容量
CUDA driver version
CUDA runtime version
PyTorch version
FlashMLA commit hash
Nsight Compute version
服务器 CPU
系统内核
```

### 4.4 验收标准

```text
可以识别 H800
可以运行 CUDA kernel
可以运行 PyTorch CUDA
可以运行 ncu
可以编译 FlashMLA
```

---

## 5. Phase 1：跑通 FlashMLA 官方 benchmark

### 5.1 目标

先不做 L2 实验，先确认 FlashMLA 官方 benchmark 能跑通。

### 5.2 Agent 需要做

```text
clone FlashMLA
安装依赖
编译扩展
运行官方 test / benchmark
保存原始输出
```

FlashMLA 官方 README 已经包含 Dense MLA Decoding、Sparse MLA Decoding、Sparse MLA Prefill 等测试入口，因此第一版不要自己重写 kernel，而是先封装官方入口。

### 5.3 输出文件

```text
results/raw/flashmla_official_dense_decode.json
results/raw/flashmla_official_sparse_decode.json
results/raw/flashmla_official_sparse_prefill.json
```

### 5.4 验收标准

```text
Dense Decoding 能跑通
Sparse Decoding 能跑通
Sparse Prefill 能跑通
每个 benchmark 至少得到 latency 或 TFLOPS
```

---

## 6. Phase 2：构建 GEMM microbenchmark

### 6.1 目标

用 GEMM 验证 L2 pressure 实验方法是否有效。

GEMM 是对照组，原因是：

```text
GEMM 计算结构清晰
working set 容易计算
L2 footprint 容易控制
可以对比 compute-bound 和 memory-sensitive 场景
```

### 6.2 实验变量

```yaml
task: gemm
dtype:
  - bf16
  - fp16
shape_family:
  - square
  - tall_skinny
  - wide
M:
  - 128
  - 256
  - 512
  - 1024
  - 2048
  - 4096
N:
  - 128
  - 256
  - 512
  - 1024
  - 2048
  - 4096
K:
  - 128
  - 256
  - 512
  - 1024
  - 2048
  - 4096
cache_state:
  - warm
  - cold
pollute_size_mb:
  - 0
  - 16
  - 32
  - 64
  - 128
  - 256
repeat: 100
warmup: 20
```

### 6.3 Working set 估计

对 GEMM：

```text
working_set_bytes ≈ sizeof(dtype) × (M×K + K×N + M×N)
```

Agent 需要自动计算：

```text
working_set / 50MB
```

并把实验点分成：

```text
小于 L2
接近 L2
大于 L2
远大于 L2
```

### 6.4 验收标准

```text
能输出 latency
能输出 TFLOPS
能输出 working set size
能输出 warm/cold ratio
能输出 pollute sensitivity
```

---

## 7. Phase 3：构造 L2 pressure 方法

本项目使用四种方法，不直接声称“修改 L2 大小”。

### 7.1 Working set sweep

让活跃数据规模扫过 L2 容量：

```text
10MB
25MB
40MB
50MB
64MB
80MB
128MB
256MB
```

目标：

```text
观察性能是否在 L2 附近出现拐点
```

---

### 7.2 Warm / cold cache 对比

```text
warm：同一个 shape 连续运行，丢弃前若干次 warmup
cold：每次目标 kernel 前先访问大 buffer 污染 L2
```

判断规则：

```text
cold_latency / warm_latency > 1.10
说明该 kernel 可能存在 L2 reuse
```

---

### 7.3 Pollution sweep

在目标 kernel 前插入 L2 pollution kernel：

```text
pollute_size_mb = 0, 16, 32, 64, 128, 256
```

判断规则：

```text
如果 pollute size 越大，目标 kernel latency 越高
说明性能对 L2 保留状态敏感
```

---

### 7.4 L2 persistence 实验

Hopper 支持控制 L2 cache 中数据 persistence 行为，可以作为扩展实验，用来观察 KV cache 或 metadata 是否可以通过 L2 persistence 受益。

第一版 MVP 可以先不做 persistence，等 warm/cold 和 pollution 结果稳定后再加入。

---

## 8. Phase 4：FlashMLA 整体性能实验

### 8.1 目标

先从整体 MLA 性能看 L2 pressure 的影响。

### 8.2 实验变量

```yaml
task: flashmla_end2end
gpu: H800
mode:
  - dense_decode
  - sparse_decode
  - sparse_prefill
batch_size:
  - 1
  - 2
  - 4
  - 8
seq_len:
  - 1024
  - 2048
  - 4096
  - 8192
  - 16384
  - 32768
  - 65536
cache_state:
  - warm
  - cold
pollute_size_mb:
  - 0
  - 32
  - 64
  - 128
  - 256
repeat: 100
warmup: 20
```

### 8.3 输出指标

```text
latency
tokens/s
TFLOPS
effective bandwidth
warm/cold ratio
pollute sensitivity
```

### 8.4 验收标准

```text
能够判断整体 MLA 在哪些 seq_len 下对 L2 更敏感
能够判断 dense decode / sparse decode / sparse prefill 谁贡献主要性能变化
```

---

## 9. Phase 5：Dense Decoding 拆解

### 9.1 研究问题

```text
Dense Decoding 的连续 KV 访问是否能从 L2 reuse 中受益？
当 seq_len 变大后，KV working set 超过 L2 是否导致 latency 上升？
```

### 9.2 实验变量

```yaml
task: flashmla_dense_decode
dtype:
  - bf16
batch_size:
  - 1
  - 2
  - 4
  - 8
seq_len:
  - 1024
  - 2048
  - 4096
  - 8192
  - 16384
  - 32768
  - 65536
cache_state:
  - warm
  - cold
pollute_size_mb:
  - 0
  - 32
  - 64
  - 128
  - 256
repeat: 100
warmup: 20
```

### 9.3 重点分析

```text
latency vs seq_len
L2 hit rate vs seq_len
DRAM bytes vs seq_len
warm/cold ratio vs seq_len
pollute_size vs latency
```

---

## 10. Phase 6：Sparse Decoding 拆解

### 10.1 研究问题

```text
Sparse Decoding 是否因为 sparse index / top-k metadata / 非连续 KV block 访问而更依赖 L2？
稀疏模式是降低访存量，还是破坏访存局部性？
```

### 10.2 实验变量

```yaml
task: flashmla_sparse_decode
kv_dtype:
  - fp8
compute_dtype:
  - bf16
batch_size:
  - 1
  - 2
  - 4
  - 8
seq_len:
  - 4096
  - 8192
  - 16384
  - 32768
  - 65536
top_k_blocks:
  - 16
  - 32
  - 64
  - 128
block_size:
  - 64
  - 128
sparsity_pattern:
  - random
  - local
  - clustered
cache_state:
  - warm
  - cold
pollute_size_mb:
  - 0
  - 32
  - 64
  - 128
  - 256
repeat: 100
warmup: 20
```

### 10.3 重点分析

```text
top_k 越大，L2 pressure 是否越强
random sparsity 是否比 local sparsity 更差
index metadata 是否有明显 L2 reuse
FP8 KV cache 是否降低 DRAM 压力
```

---

## 11. Phase 7：Sparse Prefill 拆解

### 11.1 研究问题

```text
Sparse Prefill 是 compute-bound、L2-bound 还是 HBM-bound？
长 prefill_len 下，稀疏访问模式是否造成 L2 miss 上升？
```

### 11.2 实验变量

```yaml
task: flashmla_sparse_prefill
dtype:
  - bf16
batch_size:
  - 1
  - 2
  - 4
prefill_len:
  - 1024
  - 2048
  - 4096
  - 8192
  - 16384
  - 32768
top_k_blocks:
  - 16
  - 32
  - 64
  - 128
block_size:
  - 64
  - 128
cache_state:
  - warm
  - cold
pollute_size_mb:
  - 0
  - 32
  - 64
  - 128
  - 256
repeat: 100
warmup: 20
```

### 11.3 重点分析

```text
prefill_len 增大时 latency 是否线性增长
L2 hit rate 是否下降
DRAM throughput 是否接近峰值
Tensor Core utilization 是否仍然较高
```

---

## 12. Phase 8：Nsight Compute 指标采集

### 12.1 不要所有点都跑 ncu

Nsight Compute 开销较大。自动化策略是：

```text
所有点跑快速 benchmark
只对关键点跑 ncu
```

关键点包括：

```text
warm/cold 差异最大的 shape
pollution sensitivity 最大的 shape
working set 接近 50MB 的 shape
性能出现突变的 shape
dense/sparse 差异最大的 shape
```

### 12.2 推荐采集指标

```text
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__t_sector_hit_rate.pct
lts__t_sectors.sum
dram__bytes.sum
smsp__sass_thread_inst_executed_op_hmma_pred_on.sum
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
```

注意：Nsight Compute 不同版本 metric 名称可能不同，Profiler Agent 必须先运行：

```bash
ncu --query-metrics
```

然后选择当前版本存在的指标。

### 12.3 判断逻辑

```text
如果 latency 上升，同时 L2 hit rate 下降，DRAM bytes 上升：
    高概率是 L2 miss / HBM traffic 导致

如果 latency 上升，但 L2 hit rate 不变：
    可能不是 L2，而是 occupancy、register、shared memory、调度或同步问题

如果 Tensor Core utilization 高，DRAM throughput 低：
    更可能是 compute-bound

如果 DRAM throughput 高，L2 hit rate 低：
    更可能是 memory-bound 或 HBM-bound

如果 warm/cold 差异大：
    说明存在 L2 temporal reuse

如果 pollution size 增大导致 latency 单调上升：
    说明对 L2 保留状态敏感
```

---

## 13. Phase 9：L2 Sensitivity Score

Analyzer Agent 为每个 kernel 计算一个 L2 敏感性分数。

### 13.1 基础分数

```text
warm_cold_ratio = cold_latency / warm_latency

pollution_ratio = polluted_latency / clean_latency

dram_ratio = polluted_dram_bytes / clean_dram_bytes

l2_hit_drop = clean_l2_hit_rate - polluted_l2_hit_rate
```

### 13.2 综合分数

```text
L2_sensitivity_score =
    0.35 × warm_cold_ratio
  + 0.35 × pollution_ratio
  + 0.20 × dram_ratio
  + 0.10 × l2_hit_drop_normalized
```

### 13.3 分类标准

```text
score < 1.05：
    L2 不敏感

1.05 <= score < 1.15：
    轻度 L2 敏感

1.15 <= score < 1.30：
    中度 L2 敏感

score >= 1.30：
    强 L2 敏感
```

这个 score 不是最终科学结论，只是自动补实验的触发器。最终报告必须回到原始数据和 Nsight 指标解释。

---

## 14. Phase 10：自动补实验规则

Refiner Agent 根据结果自动补实验。

### 14.1 如果 working set 接近 50MB 时性能突变

补充：

```text
40MB
45MB
50MB
55MB
60MB
64MB
70MB
```

目标：

```text
确认是否存在 L2 容量边界效应
```

---

### 14.2 如果 warm/cold ratio > 1.10

补充：

```text
增加 repeat
增加 ncu profile
测试更多 pollute_size
```

目标：

```text
确认是否是 L2 reuse 导致
```

---

### 14.3 如果结果噪声大

判断标准：

```text
std / mean > 5%
```

补充：

```text
repeat 从 100 增加到 300
固定 GPU clocks
增加 warmup
隔离其他进程
```

---

### 14.4 如果 L2 hit rate 与 latency 不一致

补充：

```text
采集 occupancy
采集 register usage
采集 shared memory usage
采集 achieved occupancy
采集 Tensor Core utilization
```

目标：

```text
排除非 L2 因素
```

---

## 15. Controller 主循环

```python
def main():
    env = collect_env()
    assert env["gpu_name"].contains("H800")

    tasks = planner.generate_initial_tasks()

    for task in tasks:
        result = runner.run_benchmark(task)
        database.save(result)

        quick_analysis = analyzer.analyze_latency(result)

        if quick_analysis.need_ncu:
            ncu_result = profiler.run_ncu(task)
            database.save(ncu_result)

        decision = analyzer.classify_bottleneck(task)

        if decision.need_refine:
            new_tasks = refiner.generate_followup_tasks(task, decision)
            tasks.extend(new_tasks)

    reporter.generate_final_report()
```

---

## 16. 每轮实验日志格式

每个实验点必须保存 JSON。

```json
{
  "task_id": "dense_decode_s8192_b4_cold_p64",
  "kernel": "flashmla_dense_decode",
  "gpu": "H800",
  "cuda_version": "12.x",
  "flashmla_commit": "...",
  "shape": {
    "batch_size": 4,
    "seq_len": 8192,
    "head_dim": 128
  },
  "cache_state": "cold",
  "pollute_size_mb": 64,
  "repeat": 100,
  "warmup": 20,
  "latency_ms_mean": 0.0,
  "latency_ms_std": 0.0,
  "tflops": 0.0,
  "effective_bandwidth_gbps": 0.0,
  "ncu_profile": "results/ncu/xxx.csv",
  "status": "success"
}
```

---

## 17. 最终报告结构

```text
# H800 上 L2 Cache 对 GEMM 与 FlashMLA 性能影响分析

## 1. 实验背景与目标

## 2. 实验平台
- GPU
- CUDA
- PyTorch
- FlashMLA commit
- Nsight Compute version

## 3. 实验方法
- working set sweep
- warm/cold cache
- pollution kernel
- Nsight Compute profiling

## 4. GEMM 对照实验
- latency vs working set
- TFLOPS vs working set
- L2 hit rate vs working set
- 结论

## 5. FlashMLA 整体 MLA 性能
- dense decode
- sparse decode
- sparse prefill
- end-to-end latency
- L2 pressure 下的性能变化

## 6. Dense Decoding 拆解
- latency
- L2 hit rate
- DRAM bytes
- bottleneck classification

## 7. Sparse Decoding 拆解
- top-k 影响
- sparsity pattern 影响
- L2 sensitivity

## 8. Sparse Prefill 拆解
- prefill_len 影响
- compute-bound / memory-bound 判断

## 9. 四类 kernel 对比
- GEMM
- Dense Decoding
- Sparse Decoding
- Sparse Prefill

## 10. 结论
- 哪个 kernel 最受 L2 影响
- 哪个 shape 区间最敏感
- 整体 MLA 性能下降主要由哪个 kernel 贡献
- 后续优化方向
```

---

## 18. MVP 版本

第一版不要做太大。建议 MVP 只做：

```text
GEMM：10 个 shape
Dense Decoding：5 个 seq_len
Sparse Decoding：5 个 seq_len × 3 个 top_k
Sparse Prefill：5 个 prefill_len
cache_state：warm / cold
pollute_size：0 / 64MB / 128MB
每个点 repeat=100
每类 kernel 选 2 个最敏感点跑 ncu
```

MVP 验收标准：

```text
能够跑完整流程
能够生成 CSV
能够生成至少 5 张图
能够判断哪个 kernel 最 L2-sensitive
能够生成一版 Markdown 报告
```

---

## 19. Agent 启动 Prompt

把下面这段给主 agent：

```text
你是一个 GPU 性能实验自动化 agent。

项目目标：
研究 H800 上 L2 cache pressure 对 GEMM 和 FlashMLA 性能的影响。FlashMLA 需要分析整体 MLA 性能，并拆解 Dense Decoding、Sparse Decoding、Sparse Prefill 三个 kernel。

你必须按以下顺序工作：

1. 检查 H800、CUDA、PyTorch、Nsight Compute、FlashMLA 环境。
2. 跑通 FlashMLA 官方 benchmark。
3. 构建 GEMM microbenchmark，用 working set sweep、warm/cold、pollution kernel 验证 L2 实验方法。
4. 对 FlashMLA 整体 MLA 性能做相同 L2 pressure 实验。
5. 分别拆解 Dense Decoding、Sparse Decoding、Sparse Prefill。
6. 对关键 shape 调用 Nsight Compute，采集 L2 hit rate、DRAM traffic、SM/Tensor Core utilization。
7. 根据 latency、L2 hit rate、DRAM bytes、warm/cold ratio 判断是否 L2-sensitive。
8. 如果证据不足，自动补实验。
9. 最终生成 Markdown 报告、CSV 表格和图。

约束：
- 不要一次性跑全组合实验。
- 先粗扫，再加密。
- 不允许修改 evaluator。
- 每次实验必须记录环境、commit hash、shape、repeat、warmup、cache_state、pollute_size。
- 所有结论必须由数据支持。
- 如果 latency 变化不能被 L2 hit rate 和 DRAM traffic 解释，必须标记为不确定，并补充 occupancy、register、shared memory、Tensor Core utilization 指标。
```

---

## 20. 最关键的执行原则

```text
先用 GEMM 验证 L2 pressure 方法是否可靠；
再跑 FlashMLA 整体性能；
最后拆 Dense Decoding、Sparse Decoding、Sparse Prefill；
所有结论必须同时满足：
latency 变化 + L2 hit rate 变化 + DRAM traffic 变化。
```

不要只根据 latency 下结论。latency 只是现象，L2 hit rate 和 DRAM traffic 才是解释 L2 影响的核心证据。
