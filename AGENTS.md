# gpu-learn Agent 入口

本项目研究 H800 上不同算子对 L2 cache 容量的敏感性。当前阶段的目标不是一次性跑完实验，而是给后续 agent 留下清晰入口、执行路线和少量可复用工具代码。

## 先读什么

后续 agent 从这里开始：

1. 读本文件，确认项目边界和禁止事项。
2. 读 [docs/overview.md](docs/overview.md)，理解研究问题和 L2 persistent 方法。
3. 读 [docs/agents/planner.md](docs/agents/planner.md)。Planner Agent 是主 agent。
4. 按当前任务读对应 phase，例如环境检查读 [docs/phases/phase_00_env.md](docs/phases/phase_00_env.md)。
5. 需要字段格式时读 [docs/data_contracts.md](docs/data_contracts.md)。
6. 需要 L2 persistent 细节时读 [docs/l2_persistent.md](docs/l2_persistent.md)。

不要一开始通读所有文档。按任务路由逐步打开。

## Agent 架构

本项目有 6 个 agent：

- Planner Agent：主 agent。负责读取计划、生成任务、分派子 agent、决定是否补实验。
- Runner Agent：执行 benchmark，保存 stdout/stderr/raw JSON。
- Profiler Agent：执行 Nsight Compute，保存 ncu CSV 和 profile 元数据。
- Analyzer Agent：解析 benchmark 和 ncu，判断 L2 sensitivity 和瓶颈类型。
- Refiner Agent：根据 Analyzer 结论生成补实验任务。
- Reporter Agent：生成 Markdown 报告、CSV 表格和图。

`gpu-l2` CLI 和 `gpu_l2_harness/` 只是工具层，不是主 agent。不要把 Python CLI 当成完整 agent 系统。

## 必须遵守

- L2 容量实验使用 CUDA L2 persistent set-aside：`cudaDeviceSetLimit` 加 stream access policy window 加 keeper buffer prime。
- 不硬编码 H800 L2 容量，必须运行时查询。
- 每次真实实验必须记录命令、超参数、git commit、CUDA/PyTorch/Nsight 版本、GPU 信息。
- 绘图脚本必须放到对应 figure 目录，例如 `results/<run_id>/figures/<fig_name>/plot.py`。
- FlashMLA 不 vendor 到本仓库，默认通过 `third_party/FlashMLA` 或配置指定路径。

## Docker 入口

H800 上使用已经构建好的容器：

```text
image: operatorsforge:h800-v1.0
container name: l2_mla_study
```

推荐入口：

```bash
docker exec -it l2_mla_study bash
cd /data1/user/peijun/work/gpu-learn
```

容器内从 [docs/docker.md](docs/docker.md) 继续。

当前会话无法验证 Docker：访问 `/var/run/docker.sock` 返回 permission denied。后续在 H800 机器上执行时需要有 Docker 权限。

## 当前代码状态

已有少量工具层代码：

- `gpu_l2_harness/l2_persistent.py`：Python guard，封装 L2 persistent set-aside。
- `csrc/l2_persistent_ext.cpp` 和 `csrc/l2_persistent_kernel.cu`：CUDA/PyTorch 扩展源码。
- `gpu_l2_harness/planner.py`：把配置展开成 task manifest。
- `gpu_l2_harness/runner.py`、`profiler.py`、`analyzer.py`、`reporter.py`：工具层雏形。
- `gpu_l2_harness/agents/`：六个 agent 的轻量接口说明和占位入口。

这些代码只是引子。后续 agent 应按 `docs/phases/` 逐步扩展，不要假设当前已经完成 FlashMLA 全量适配。
