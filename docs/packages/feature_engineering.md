# feature_engineering

Canonical 特征编译与训练窗内拟合变换；数据物化归 data_loading，监督标签、窗口与生命周期归 pipeline。

## 结构

- `compilation/compiler.py`：公共 single/batch 入口、single 执行、信息可见性与 provider 接线。
- `compilation/batch.py`：独立 batch 执行器；仅同一快照和同一下界共享历史，静态不支持项先判定后回退。
- `compilation/indexed.py`：规则 source-time 历史的因子化特征，不构造 Y；监督标签由 `pipeline/labels.py` 构造。
- `compilation/planning.py`：输出名称、依赖引用、操作参数和规则身份；碰撞在读取数据前拒绝。
- `compilation/requirements.py`：有限历史长度与全前缀统计预热需求的唯一实现，训练与在线会话共同消费。
- `compilation/contracts.py`：FeatureSchema、CompiledFeatures、VisibilityProof、BatchEligibility 与作用域状态。
- `kernels/`：history（窗口/前缀/事件）、seasonal（同槽/近期状态/残差基线）、spectral（FFT/DWT/熵）、windows（分位窗/偏移窗）。
- `statistics/`：只读 provider 协议与精确 EWM/事件/expanding 状态；更新生命周期由 pipeline.online 管理。
- `selection.py`：逐训练窗监督选择，各 call 对应 horizon/target 独立评分后聚合，不平均不同物理目标值。
- `transform_specs.py`：特征/目标变换归一化；通用高级特征字段语法在 `forecasting_core/specs/feature_transformations.py`。
- `transforms/`：scaling 为特征缩放；pipeline 为 canonical 目标接口；stack/calendar/target_scaling/per_series 分别负责栈、日历、目标缩放和序列隔离；windows 确定唯一标签拟合窗口。

## 执行合同

- 特征仅使用预测原点可见信息；observed-past 越界访问需要显式 provider。
- Direct lag 锚点由 `direct.align_to_target` 决定；rolling/expanding/difference 基于原点历史，偏移窗由显式 offsets 决定。
- schema 与 planner 名称一致，禁止字典覆盖旧列；DataFrame scaler 按名称校验和重排。
- fixed rolling 必须完整；未定义样本统计 RAISE。向量内核可保留不可消费的预热前缀，进入训练/预测的行必须满足需求。
- single/batch/indexed 共用规格与数值内核，但浮点后端不保证逐位一致；模型级旧新对照不能被特征容差替代。
- ContextVar 隔离每次编译的帧和派生缓存，异常退出恢复外层作用域。计时是最近调用诊断，不承诺同实例并发计时。
- `generator_defined` 保留逐行 available_at；block_weather 使用完整调用块及最晚可得时间，稀疏 horizon 不缩短取数块。
- 派生来源证据由规划规则补充 inputs、operation、parameters、identity；`available_at_upper_bound` 是保守可用上界，不冒充真实发布时间。
- 目标变换固定 calendar → decomposition → scaling，point/quantile 逆序恢复原单位，状态按 `(series_id,target)` 隔离。

## 身份与兼容性

磁盘编译缓存已退役；`pipeline/design_identity.py` 组合数据、依赖与全链实现身份，用于内存共享及证据，不等于配置 fingerprint。旧结果不自动删除或重跑。

本次子包迁移不提供旧模块兼容壳。含迁移类的旧 bundle/在线快照可能需要旧版代码读取；新格式按真实持久化回读验证，不承诺跨版本 pickle 兼容。

## 详细文档

- [严格边界、数值语义与新增窗口配置](../config/feature-contracts.md)
- [在线增量、快照恢复与资源边界](../config/online-prediction.md)
- [同槽、近期状态、块天气与残差基线](../config/seasonal-features.md)

编译基准：`env -u PYTHONPATH .venv/bin/python tests/benchmark_feature_compiler.py --output <new-json-path>`。只测物化后的特征编译，不等于端到端训练收益。
