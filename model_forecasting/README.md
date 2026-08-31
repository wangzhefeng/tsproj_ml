# model_forecasting

`model_forecasting/` 是单模型生命周期编排与推理层。

- `design.py`：information set、特征设计、监督样本和 fixed-step 回测几何。
- `fit_service.py`：fold/final transform、训练与预测服务。
- `backtest_runtime.py`：calendar-month 回测生命周期。
- `persistence.py`：schema-2 bundle 构造与持久化。
- `runtime.py`：`CanonicalBaseModelRunner` 与顶层单模型生命周期编排。
- `forecaster.py`：point 与 marginal quantile 推理，recursive quantile 使用 median path。
- `transforms.py`：目标/特征变换状态与严格逆变换。
- `results.py`：canonical long result 读写和绘图。

稳定类型全部来自 `forecasting_core/`。本包可以向下编排各流水线阶段，但不得 import `model_ensemble`。
