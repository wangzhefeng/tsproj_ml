# probabilistic

`probabilistic/` 是概率校准能力包（CQR）。2026-09-06 R4 行为不变重构后仅含校准内核：
quantile 逐 level 训练编排在 `model_training/quantile.py`，objective 支持性检查在
`model_training/objectives.py`（方案 v3，.hermes/plans/2026-09-06_162605）。

- `calibration.py`：CQR 数学内核与 apply-before-collect 追踪器；score 统一调用 `compute_nonconformity_scores`。有限且有效的倒置区间直接报错，整批 score 校验通过后才追加记录；原有非有限值及评估掩码筛选保留。
- 张量 crossing 修复由 `model_forecasting.forecaster.repair_marginal_quantile_crossing` 承载，不在本包维护重复的 DataFrame 通路。
- 原生 quantile 参数注入唯一实现为 `models.catalog.quantile_parameters`。

稳定概率合同位于 `forecasting_core/{probabilistic_spec,artifacts}.py`；point/quantile forecaster 位于 `model_forecasting/forecaster.py`。Canonical runtime 显式启用 CQR 时输出 `predict_pi*`，final 修正量随 bundle 保存并用于部署，不在部署期重校准。

旧 DataFrame `pipeline.py`/`postprocessing.py`、重复的 `inject_quantile_objective` 已退出，不提供兼容 shim。删除旧 API 不改变合法 canonical 配置的原生 objective 参数；直接传给 `quantile_parameters` 的非法分位点、未知模型会明确报错。

## 训练与校准接口

`CanonicalMarginalQuantileTrainer` 接收分位点对应的 estimator factory、训练能力和特征 schema，复用 `CanonicalTrainer` 生成 `CanonicalMarginalQuantileArtifact`；其中 `artifacts_by_level` 的顺序必须与 quantile grid 一致，不能携带虚构的依赖模型。

`CalibrationRecord` 保存可得时刻与 nonconformity score，`select_calibration_records()` 负责 as-of 选择，`calibrate_with_records()` / `calibrate_quantile_band()` 计算修正。`ConformalCalibrationTracker` 在每折先应用已有历史修正，再收集该折可用 score，不能先把当前验证误差放入历史池。

final 校准状态写入 bundle，部署只应用已保存修正。CQR 提供边际区间，不提供联合样本、联合概率或任意分布漂移下的覆盖保证。crossing 修复、区间校准与指标评估是不同责任，不因都处理 quantile 而合并为重复流水线。
