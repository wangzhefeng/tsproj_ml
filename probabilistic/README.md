# probabilistic

`probabilistic/` 提供 quantile 训练和独立概率后处理能力。

- `training.py`：每个 quantile level 训练 canonical strategy artifact。
- `objectives.py`：估计器 quantile objective 注入。
- `postprocessing.py`：DataFrame quantile crossing 修复。
- `pipeline.py`：DataFrame forecast 后处理与可选 CQR 工具入口。
- `calibration.py`：CQR 数学内核。

稳定概率合同已下沉 `forecasting_core/{probabilistic_spec,artifacts}.py`；point/quantile forecaster 位于 `model_forecasting/forecaster.py`。Canonical runtime 当前不执行 CQR，不输出 `predict_pi*`。
