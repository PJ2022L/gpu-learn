# Phase 00: 环境检查

## 目标

确认 H800 Docker 环境具备实验所需基础能力。

## 执行 Agent

Planner 调度 Runner。

## 输入

- [../docker.md](../docker.md)
- `configs/env.yaml`
- 已有 Docker 容器 `l2_mla_study`，镜像来源 `operatorsforge:h800-v1.0`

## 操作

容器内执行：

```bash
python --version
nvidia-smi
nvcc --version
ncu --version
python -m pip install -e . --no-build-isolation
GPU_L2_BUILD_EXT=1 python -m pip install -e . --no-build-isolation
gpu-l2 collect-env --out results/<run_id>/env.json
```

## 输出

- `results/<run_id>/env.json`
- `logs/<run_id>/command.log`

## 验收标准

- 能识别 H800。
- 能运行 PyTorch CUDA。
- 能运行 `ncu --version`。
- 能构建 `_l2p_ext`。
- `query_l2_props()` 返回非零 `l2CacheSize`、`persistingL2CacheMaxSize`、`accessPolicyMaxWindowSize`。
