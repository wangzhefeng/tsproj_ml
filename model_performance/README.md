# model_performance

资源门禁按估计器需要的逻辑设计量规划，不能用mmap/紧凑块体积冒充训练内存。pipeline在全设计构造前做准入；设计按需准备后复核资源，保留父调度器授予的线程份额。运行证据另记实际保留数组字节。磁盘编译缓存与容量准入已移除；fit checkpoint、内存复用及正式结果保留。

`model_performance/` 负责运行资源、性能档、checkpoint 与内存/变换缓存；不拥有模型算法、配置语义或生命周期。

- `resource_planner.py`：构建 workload、探测预算、规划单模型及融合资源，并生成 estimator 运行参数；由编排层调用，不反向依赖编排包。
- `performance_profiles.py`：性能签名与已采纳性能档解析；签名绑定模型实现及适用 workload，不把不同环境的测量冒充可复用证据。
- `checkpoints.py`：`FileFitCheckpoint`、实现指纹与恢复错误类型；实现 `forecasting_core.checkpoints.FitCheckpoint`，训练层只依赖合同。
- `transform_cache.py`：`FoldTransformCache` 与 fold 变换指纹；按显式训练窗和变换语义复用状态。
- `batch_memory.py`：`BoundedPayloadCache`、载荷保留字节估计、`SampledRSS`；采样 RSS 不等于连续峰值测量。

## 身份与边界

checkpoint 的实现指纹递归包含受保护源码目录，`models/wrappers/` 随 `models/` 自动纳入。实现变化会使缓存/恢复身份变化，不改变 YAML 的配置语义，也不自动删除已有结果。

原始设计身份由`feature_engineering/design_identity.py`计算，无磁盘编译缓存，批调度和产物验收属于 `model_pipeline/`。性能档只能在其签名和资源条件满足时使用，不能借运行参数静默改算法、训练窗或输出合同。
