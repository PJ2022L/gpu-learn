# Runner Agent

Runner Agent 负责执行 benchmark 和保存原始结果。

## 职责

- 在 Docker/H800 环境中运行 benchmark。
- 每个 task 独立子进程执行，避免 CUDA context 状态串扰。
- 在同一子进程内完成 L2 persistent setup、keeper prime、目标 kernel 计时。
- 保存 stdout、stderr、result JSON。
- 记录命令、超参数和环境信息。

## 输入

- `results/<run_id>/task_manifest.json`
- `configs/*.yaml`
- `gpu_l2_harness/l2_persistent.py`

## 输出

- `results/<run_id>/raw/<task_id>.json`
- `logs/<run_id>/<task_id>.stdout`
- `logs/<run_id>/<task_id>.stderr`
- `logs/<run_id>/command.log`

## 执行规则

- 只执行 `status=planned` 的 task。
- `status=unsupported` 的 task 直接写 unsupported result，不尝试改配置。
- 真实实验前必须构建 `_l2p_ext`。
- 目标 kernel 不标记为 persisting。
- 默认 `prime_policy=per_repeat`。

## Docker 中常用命令

```bash
python -m pip install -e . --no-build-isolation
GPU_L2_BUILD_EXT=1 python -m pip install -e . --no-build-isolation
gpu-l2 run --config configs/gemm_persistent.yaml --run-id <run_id>
```

## 失败处理

- CUDA extension 未构建：失败信息必须指出需要 `GPU_L2_BUILD_EXT=1`。
- FlashMLA API 缺失：不要重写 FlashMLA kernel，记录 adapter error。
- GPU 非 H800：记录环境，是否继续由 Planner 决定。

