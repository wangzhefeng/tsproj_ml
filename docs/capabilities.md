# 当前能力

- 七种多步策略：`recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo`。
- 统一预测张量：point `(N,H,K)`，边际 quantile `(N,H,K,Q)`。
- 显式数据角色：`target/observed_past/known_future/static/key/ignored`。
- 严格 information set：动态 source 执行 `available_at <= forecast_origin`，缺失、重复和越界直接 RAISE；监督训练仅通过内部 `target_access=supervised_labels` 读取预测期 target 标签。
- 模型：LightGBM、XGBoost、CatBoost、RandomForest、HistGradientBoosting、Ridge、ElasticNet、Lasso、QuantileRegressor、SeasonalTemplate。
- 多目标 adapter：independent、regressor-chain、native capability probe；不支持时 RAISE。
- 目标变换：calendar normalization → decomposition → scaling，按 `(series_id,target)` 隔离并严格逆序恢复。
- 引用式 Ensemble：averaging、weighted、linear blending、stacking；成员 OOF、内容寻址缓存、自包含 bundle。
- 点评估：MAE/RMSE/MAPE/Accuracy、seasonal-naive、eval mask。
- 概率评估：pinball、coverage、width、Winkler、coverage gap。

不支持：联合轨迹样本生成、ensemble-of-ensemble。CQR runtime 已支持严格 as-of 校准，并可在回测与部署结果中输出 `predict_pi*`；未声明 `probabilistic.calibration` 的配置不启用。

> 2026-09-25 场景收敛后的活跃面：活动配置仅 `aidc_load_15min_short` 的 LightGBM 单模型；quantile、calendar_month、ensemble 能力保留代码与合成测试兜底，无活动配置。
