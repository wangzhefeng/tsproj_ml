# model_evaluation

`model_evaluation/` 集中全部生产评估公式，不做训练、预测或文件 IO。

- `point.py`：MAE/RMSE/MAPE/Accuracy、seasonal-naive、aggregate weighting。
- `marginal.py`：pinball、central interval coverage/width/Winkler/gap。
- `point_intervals.py`：独立点评估区间的 coverage/width/Winkler/gap，按目标与 horizon 报告，只计算可用且通过点评估 mask 的样本；无可用区间时 n_points=0、指标 NaN，不制造 pinball。
- `mask.py`：percentile/absolute/combined eval mask。
- `metrics.py`：概率指标纯函数。

点预测 aggregate 使用配置权重；概率 aggregate 按有效点池化。输入合同来自 `forecasting_core`。

## 公共接口与输出

- `evaluate_point_forecasts()` 接受点预测、真值及可选 naive 对照，返回 per-target 与 aggregate 评分。
- `evaluate_marginal_distribution()` 计算边际分位数与中心区间指标；不把边际覆盖率解释为联合轨迹覆盖率。
- `evaluate_point_intervals()` 评分点模型的残差校准区间（scope 仅 target/horizon，无 aggregate 行）。
- `build_eval_mask_payload()` 为点和概率评估生成共用口径，`build_eval_mask()` 执行阈值筛选。评估掩码只改变评分样本，不改训练数据和原始预测曲线；`validation.eval_mask.mode` 在配置解析期做白名单校验（`forecasting_core/specs/validation.py`），非法值不进入运行期。
- `metrics.py` 的 `pinball_loss/interval_metrics/crossing_metrics/wilson_interval` 是指标内核，不负责修复预测；`crossing_metrics` 由 `model_predicting/loops/predictor.py` 组装段在 `crossing.report_raw=true` 时消费，诊断写入分布 metadata 的 `crossing_report` 并随回测逐窗 execution_evidence 落盘。

结果 CSV 的 long schema、写盘和绘图由 `model_predicting/artifacts/results.py` 承载；crossing 修复由 forecaster 承载，CQR 校准由 `probabilistic/calibration.py` 承载。Ensemble 的融合 OOF 评分复用本包，不另维护一份指标公式。
