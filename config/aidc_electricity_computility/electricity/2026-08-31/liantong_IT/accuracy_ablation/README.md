# 精度候选：Direct pointwise 单因素对照

基座为 `../baseline/lgbm_direct-pointwise.yaml`。本目录只存完整物理 YAML 与本说明；不改变原有六组，不组合算力、测点、目标分解或 seasonal baseline。

全部保留 5min、288 点预测、4032 点原始历史、1729 个训练原点、17 折及原有目标/节假日源。仅支持 `run.py --config-yaml <path> --backtest-only`，不支持 final fit/bundle。

| 文件 | 相对基座唯一实验因素 |
|---|---|
| lgbm_direct-pointwise_control.yaml | 原样对照；仅输出目录改变，fingerprint 应与基座一致 |
| lgbm_direct-pointwise_recent-state.yaml | 只加 value 的 recent_state：6/12/36 点，level/mean/std/diff/slope |
| lgbm_direct-pointwise_regularized.yaml | 复杂度控制组合：num_leaves=15、max_depth=5、min_child_samples=40、reg_alpha=0.1、reg_lambda=1.0；不改学习率、树数或损失。不能归因到单个参数 |
| lgbm_direct-pointwise_l2.yaml | 只将 objective 改为 regression_l2；报告仍比较 MAE/RMSE |
| lgbm_direct-pointwise_huber.yaml | 只将 objective 改为 huber，alpha=0.9 为显式损失参数 |
| lgbm_direct-pointwise_selection.yaml | 每训练窗 f_regression，最多20列、至少10列，强制保留原七个 target lag；不全月预选 |
| ridge_direct-pointwise.yaml | Ridge alpha=1.0 与必要的训练窗内 standard 特征缩放；不缩放目标 |
| xgb_direct-pointwise.yaml | XGBoost，显式 L1 objective、300树、学习率0.05、深度6、行列采样0.8、seed42；不是参数等价的 LightGBM |

输出为 `results/results_test/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/accuracy_ablation/<variant>/<result_identity>/backtest_only/`。不覆盖已有 baseline 结果。

所有参数是预先指定的候选，不是调优结果。主指标 MAE，辅指标 RMSE/MAPE、逐日表现；保留8月19日既定评分，另报排除该补值日的敏感性分析。未完成正式回测前不宣称改善，组合候选须等单因素验证后另行构建；同月选型不是独立泛化验证。
