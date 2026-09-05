# probabilistic

`probabilistic/` 提供 canonical quantile 训练与 CQR 校准。

- `training.py`：每个 quantile level 训练 canonical strategy artifact。
- `objectives.py`：复用训练层能力注册表做支持性检查；原生参数注入唯一实现为 `models.catalog.quantile_parameters`，保留现役 metric/eval_metric 参数语义。
- 张量 crossing 修复由 `model_forecasting.forecaster.repair_marginal_quantile_crossing` 承载，不在本包维护重复的 DataFrame 通路。
- `calibration.py`：CQR 数学内核与 apply-before-collect 追踪器；score 统一调用 `compute_nonconformity_scores`。有限且有效的倒置区间直接报错，整批 score 校验通过后才追加记录；原有非有限值及评估掩码筛选保留。

稳定概率合同位于 `forecasting_core/{probabilistic_spec,artifacts}.py`；point/quantile forecaster 位于 `model_forecasting/forecaster.py`。Canonical runtime 显式启用 CQR 时输出 `predict_pi*`，final 修正量随 bundle 保存并用于部署，不在部署期重校准。

旧 DataFrame `pipeline.py`/`postprocessing.py`、重复的 `inject_quantile_objective` 已退出，不提供兼容 shim。删除旧 API 不改变合法 canonical 配置的原生 objective 参数；直接传给 `quantile_parameters` 的非法分位点、未知模型会明确报错。
