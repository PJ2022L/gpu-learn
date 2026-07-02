# Refiner Agent

Refiner Agent 根据 Analyzer 输出生成补实验任务。

## 职责

- 对性能突变区间加密 effective L2 sweep。
- 对噪声大的点增加 repeat。
- 对证据不足的点请求 Profiler 采集 NCU。
- 控制补实验数量，避免组合爆炸。

## 输入

- `results/<run_id>/parsed/analysis.json`
- `results/<run_id>/task_manifest.json`
- Analyzer 的 uncertainty 列表

## 输出

- `results/<run_id>/refine_plan.json`
- 新增 task manifest 或 patch 建议
- 给 Planner 的下一轮任务列表

## 补实验规则

- 如果 25MB 到 40MB 之间 latency ratio 变化明显，在附近增加点：`45, 40, 36, 32, 28, 25, 20` MB。
- 如果 `timing.cv > 0.05`，把 repeat 提高到原来的 2-3 倍。
- 如果 latency ratio 高但无 NCU，加入 profile candidate。
- 如果 L2 hit rate 与 latency 不一致，补 occupancy、register、shared memory、Tensor Core 指标。

## 失败处理

- 不生成全组合。
- 不改变已经完成 task 的原始结果。
- 不把 unsupported task 改成 pollute 实验。

