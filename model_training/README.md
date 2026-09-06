# model_training

`model_training/` 只负责模型训练行为，不拥有结果 IO 或 bundle 类型。

- `trainer.py`：`CanonicalTrainer`，训练 strategy artifact。
- `quantile.py`：`CanonicalMarginalQuantileTrainer`/`CanonicalMarginalQuantileArtifact`，按 quantile grid 逐 level 训练编排。
- `objectives.py`：quantile 模型支持性检查。
- `strategies/`：七种标准多步 executor。
- `estimators/`：能力探测和 independent/chain/native adapter。

本包依赖 `forecasting_core` 与 `models`，不依赖 `probabilistic` 或 `model_forecasting`。Quantile estimator factory 由上层注入。

## 输入输出与策略组织

`CanonicalTrainer` 接收 `ForecastConfigSpec`、显式 `estimator_factory`、`EstimatorCapabilities`、固定 `feature_schema` 和可选 `FitCheckpoint`。`train()` 消费按策略调用组织的二维设计序列 `X_by_call` 与标签 `Y`，返回 `CanonicalStrategyArtifact`，而不是最终部署 bundle。

`strategies/base.py` 维护 target plan、坐标和模型组 artifact，七个策略模块复用这组合同。Direct 的 horizon-feature 属于 layout，不是新策略；Local/Global 属于 training scope。MO 分块合法性由 core spec 校验。

`estimators/capabilities.py` 从模型 catalog 与原生能力探测建立支持矩阵；`estimators/multi_target.py` 提供 independent、regressor-chain、native adapter。能力不足直接报错，不以自动降级掩盖不支持的目标维度或线程策略。

## 恢复与边界

checkpoint 通过 `forecasting_core.checkpoints.FitCheckpoint` 注入，训练层不导入运行时文件实现。特征准备、目标变换和 final bundle 落盘由上层负责；quantile 层注入不同分位点的 estimator factory，不在训练包复制目标函数参数规则。
