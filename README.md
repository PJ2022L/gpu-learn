# gpu-learn L2 Persistent Harness

本仓库是 H800 L2 cache sensitivity 实验的多 agent 交接工程。后续 agent
请先读 [AGENTS.md](AGENTS.md)，再按 `docs/` 中的路由进入具体 phase。

The Python package is only a tool layer for planning, running, profiling,
analyzing, and reporting. The main controller is the Planner Agent described in
`docs/agents/planner.md`.

The harness uses CUDA L2 persisting cache controls instead of cache pollution:

1. Set a device-level persisting L2 set-aside with
   `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`.
2. Prime a keeper buffer through a stream access policy window with
   `cudaAccessPropertyPersisting`.
3. Run the target operator in the same process while it is not marked
   persisting, so its normal L2 capacity is approximated by
   `total_l2 - setaside`.

Important caveat: L2 set-aside is not a hard physical partition. Normal memory
accesses may use the set-aside when persisting accesses are not actively using
it. The keeper buffer must be primed, and Nsight Compute metrics should be used
to validate that a result is explained by L2 hit-rate and DRAM-traffic changes.

## Local Quick Start Outside Docker

This is for local development only. On H800, use Docker below.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla
pip install -e .

gpu-l2 collect-env --out results/manual/env.json
gpu-l2 plan --config configs/gemm_persistent.yaml --run-id dryrun --dry-run
gpu-l2 report --run-id dryrun
```

The CUDA extension is optional for dry-run and unit tests. Build it before real
GPU experiments:

```bash
GPU_L2_BUILD_EXT=1 pip install -e . --no-build-isolation
```

The convenience script below performs a dry run and writes the command line to
`logs/<run_id>/command.log`:

```bash
scripts/dry_run.sh
```

Dry run creates manifests and report skeletons only. It does not run benchmark
experiments.

## H800 Docker

On H800, use:

```bash
scripts/docker_shell.sh
```

This enters the existing container `l2_mla_study`, built from
`operatorsforge:h800-v1.0`. See [docs/docker.md](docs/docker.md).
