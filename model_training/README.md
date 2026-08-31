# model_training

`model_training/` 只负责模型训练行为，不拥有结果 IO 或 bundle 类型。

- `trainer.py`：`CanonicalTrainer`，训练 strategy artifact。
- `strategies/`：七种标准多步 executor。
- `estimators/`：能力探测和 independent/chain/native adapter。

本包依赖 `forecasting_core` 与 `models`，不依赖 `probabilistic` 或 `model_forecasting`。Quantile estimator factory 由上层注入。
