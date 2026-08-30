# evaluation 模块说明

`model_evaluation/` 是**评估层**（独立评估模块，2026-08-30 流水线阶段重排新建）：全部评估指标计算集中于此，不写结果文件、不做预测后处理。

## 文件职责

| 文件 | 职责 |
|---|---|
| `point.py` | 点预测评估：`evaluate_point_forecasts`（MAE/RMSE/MAPE/Accuracy + seasonal-naive 对照 + per-target/aggregate）、聚合加权 `resolve_aggregate_weighting`、掩码构造 `build_eval_mask_payload` |
| `marginal.py` | 边际 quantile 评估：`evaluate_marginal_distribution`（逐 level pinball + central 区间覆盖率/宽度/winkler/coverage gap，eval_mask 口径与点评估一致） |
| `mask.py` | 评估掩码 `build_eval_mask`（percentile/absolute/combined 三模式 + 上限正交）——点/概率评估共用同一业务口径 |
| `metrics.py` | 指标内核纯函数：`pinball_loss`、`interval_metrics`、`wilson_interval`、`crossing_metrics` 等 |

## 消费方

- canonical 回测（`model_forecasting/runtime.py`）：逐窗点评估 + quantile 模式概率评估 → `results_test/test_scores_df.csv` / `test_scores_probabilistic_df.csv`；
- ensemble 融合 OOF 评分（`model_ensemble/model_evaluation.py`）；
- 场景融合脚本（如 `fuse_aidc_results.py`）。

## 边界

- 只读张量合同（`model_forecasting.tensors`）与概率类型合同（`probabilistic.types`）；
- 结果文件读写/schema 在 `model_forecasting/results.py`；crossing 修复在 `probabilistic/postprocessing.py`；legacy 宽表概率报告工具在 `probabilistic/model_evaluation.py`（仅测试消费）。
