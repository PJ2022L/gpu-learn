# L2 Persistent Set-Aside 方法

## CUDA 机制

核心 API：

- `cudaGetDeviceProperties`
- `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)`
- `cudaDeviceGetLimit(cudaLimitPersistingL2CacheSize, &actual)`
- `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)`
- `cudaCtxResetPersistingL2Cache`

stream access policy window 字段：

- `base_ptr`：keeper buffer 起始地址。
- `num_bytes`：窗口大小，不能超过 `accessPolicyMaxWindowSize`。
- `hitRatio`：被标记为 persisting 的比例 hint。
- `hitProp = cudaAccessPropertyPersisting`。
- `missProp = cudaAccessPropertyStreaming`。

## 实验语义

配置中使用目标可用 L2：

```yaml
l2:
  effective_l2_mb: [full, 40, 32, 25, 16, 12.5]
  hit_ratio: 1.0
  prime_policy: per_repeat
  keeper_repeats: 8
```

Planner 运行时转换：

```text
persisting_setaside_bytes = total_l2_bytes - effective_l2_bytes
```

若 set-aside 超过 `persistingL2CacheMaxSize` 或 `accessPolicyMaxWindowSize`，该 task 必须标记为 `unsupported`，不能静默截断。

## Runner 使用方式

Runner 在同一子进程中完成：

1. 查询 L2 属性。
2. 设置 persisting limit。
3. 创建 keeper buffer。
4. prime keeper buffer。
5. 执行目标 benchmark。
6. reset persisting L2 cache。
7. 恢复 limit 为 0。

目标 kernel 不应被标记为 persisting。目标 kernel 只是处在 persistent set-aside 已被 keeper 占用后的环境中。

## Caveat

L2 set-aside 不是硬隔离。normal/global access 在 persistent access 未使用 set-aside 时仍可能使用该部分 L2。因此：

- keeper 必须 prime。
- `prime_policy=per_repeat` 是默认策略。
- 最终结论必须用 NCU 证明 latency 变化伴随 L2 hit rate 下降和 DRAM traffic 上升。

## 已有代码

- Python wrapper：[../gpu_l2_harness/l2_persistent.py](../gpu_l2_harness/l2_persistent.py)
- C++ extension：[../csrc/l2_persistent_ext.cpp](../csrc/l2_persistent_ext.cpp)
- CUDA touch kernel：[../csrc/l2_persistent_kernel.cu](../csrc/l2_persistent_kernel.cu)

