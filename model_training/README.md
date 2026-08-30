# training 模块说明

`model_training/` 是**训练层**（流水线第 3 阶段）。2026-08-30 自 `model_forecasting/`（trainer.py + strategies/ + estimators/）迁为顶层包。

## 文件职责

| 文件 | 职责 |
|---|---|
| `trainer.py` | `CanonicalTrainer`：模型拟合 + 目标变换编排 + 样本裁剪 + bundle 构建；`DirectMultiOutputRegressor` / `HorizonAlignedDirectRegressor` |
| `strategies/` | 七种标准多步策略执行器（recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo） |
| `estimators/` | estimator 能力探测（`capabilities.py`，不支持即 RAISE 不静默降级）与多目标适配（`multi_target.py`：independent/chain/native） |

## 边界

- 依赖 `model_forecasting.specs/tensors`（合同）、`models/`（工厂与序列化）、`probabilistic.spec/types`（quantile 编排合同）；
- 能力声明类型 `EstimatorCapabilities` 是 spec 层合同，定义在 `model_forecasting/specs/estimator.py`，此处只做行为探测。
