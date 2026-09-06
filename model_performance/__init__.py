"""model_performance

`model_performance/` 是运行性能优化包：资源规划、性能档签名、拟合 checkpoint、
折变换缓存与有界内存缓存。2026-09-06 行为不变重构 R2 自 model_forecasting/ 迁入
（方案 v3，.hermes/plans/2026-09-06_162605）。

- `resource_planner.py`：从真实 workload 和父预算生成统一执行计划；约束外层并行轴
  与模型线程乘积；`validation.performance` 只由 planner 读取，不进 semantic fingerprint。
- `performance_profiles.py`：已采用性能 profile 的签名、来源与适用性校验。
- `checkpoints.py`：`FileFitCheckpoint` 与 `implementation_fingerprint()`；仅恢复完成的
  拟合单元，身份变化 miss，损坏报错。
- `transform_cache.py`：批内 fold transform 复用及指纹。
- `batch_memory.py`：有界内存缓存和采样 RSS。

本包依赖 forecasting_core/model_training/models/feature_engineering；批量调度
（batch_runtime/batch_artifacts）属流程编排，位于 model_forecasting。
"""
