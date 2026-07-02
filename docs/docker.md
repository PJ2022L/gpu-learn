# Docker 环境

H800 上使用已经构建好的容器，不要默认新建容器：

```text
image: operatorsforge:h800-v1.0
container name: l2_mla_study
```

## 推荐进入方式

在 H800 主机上执行：

```bash
docker exec -it l2_mla_study bash
cd /data1/user/peijun/work/gpu-learn
```

如果容器没有运行，先检查容器状态：

```bash
docker ps -a --filter name=l2_mla_study
```

是否启动已有容器由 H800 机器管理员或后续 agent 根据现场状态决定。本项目文档默认进入已有容器，不自动 `docker run` 新容器。

## 容器内最小验证

进入容器后执行：

```bash
python --version
nvidia-smi
nvcc --version
ncu --version
python -m pip install -e . --no-build-isolation
python -m unittest discover -s tests -v
gpu-l2 plan --config configs/gemm_persistent.yaml --run-id docker_smoke --dry-run
```

真实 L2 persistent 实验前构建 CUDA 扩展：

```bash
GPU_L2_BUILD_EXT=1 python -m pip install -e . --no-build-isolation
python - <<'PY'
from gpu_l2_harness.l2_persistent import query_l2_props
print(query_l2_props())
PY
```

## 日志要求

所有实验命令必须写入：

```text
logs/<run_id>/command.log
```

可以使用：

```bash
scripts/run_logged.sh <run_id> <command...>
```

