# model_performance

`model_performance/` 负责运行资源、性能档、checkpoint 与内存/变换缓存；不拥有模型算法、配置语义或生命周期。

- `resource_planner.py`：构建 workload、探测预算（psutil 为硬依赖）、规划单模型及融合资源，并生成 estimator 运行参数；由编排层调用，不反向依赖编排包。ensemble 顶层 `validation.performance` 只接受 `ensemble_parallel_workers`/`total_thread_limit`/`memory_limit_bytes`，单模型轴键（window/quantile/output/model_thread_count/profile_ref）在顶层声明即 RAISE，须写入成员配置。
- `performance_profiles.py`：性能签名与已采纳性能档解析；签名绑定模型实现及适用 workload，不把不同环境的测量冒充可复用证据。
- `checkpoints.py`：`FileFitCheckpoint`、实现指纹与恢复错误类型；实现 `forecasting_core.execution.checkpoints.FitCheckpoint`，训练层只依赖合同。实现指纹对全部实现包（含 `model_building/`）的 .py 文件与库版本取哈希，进程内缓存（输入在进程生命周期内不变）；批跑不再逐任务重复全仓哈希。`prune_fit_checkpoints()` 提供有界保留（默认 30 天 / 10 GiB + 孤儿 .tmp 清理），批入口在 `_batch_state` 上调用；删除安全——miss 只是重新拟合，从不触碰非 .fit/.tmp/.lock 文件。
- `transform_cache.py`：`FoldTransformCache` 与 fold 变换指纹；按显式训练窗和变换语义复用状态。
- `batch_memory.py`：`BoundedPayloadCache`、载荷保留字节估计、`SampledRSS`；采样 RSS 不等于连续峰值测量。

## 身份与边界

checkpoint 的实现指纹递归包含受保护源码目录，`model_building/wrappers/` 随 `models/` 自动纳入。实现变化会使缓存/恢复身份变化，不改变 YAML 的配置语义，也不自动删除已有结果。

raw-design 编译缓存属于 `feature_engineering/cache.py`，批调度和产物验收属于 `pipeline/`。性能档只能在其签名和资源条件满足时使用，不能借运行参数静默改算法、训练窗或输出合同。
