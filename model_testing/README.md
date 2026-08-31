# model_testing

`model_testing/` 提供纯时间几何和回测原语。

- `validation.py`：fixed-step rolling-origin、完整 calendar-month folds、标签非重叠校验。
- `backtest.py`：actual tensor、seasonal-naive、forecast origin 和正整数校验。

本包只依赖 `forecasting_core.tensors`。指标计算属于 `model_evaluation/`。
