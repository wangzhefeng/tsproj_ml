# model_evaluation

`model_evaluation/` 集中全部生产评估公式，不做训练、预测或文件 IO。

- `point.py`：MAE/RMSE/MAPE/Accuracy、seasonal-naive、aggregate weighting。
- `marginal.py`：pinball、central interval coverage/width/Winkler/gap。
- `mask.py`：percentile/absolute/combined eval mask。
- `metrics.py`：概率指标纯函数。

点预测 aggregate 使用配置权重；概率 aggregate 按有效点池化。输入合同来自 `forecasting_core`。
