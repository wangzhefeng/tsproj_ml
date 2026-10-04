# 精度候选：Direct pointwise 单因素对照

基座为 `../baseline/lgbm_direct-pointwise.yaml`。本目录只存完整物理 YAML 与本说明；不改变原有六组，不组合算力、测点、目标分解或 seasonal baseline。

全部保留5min、288点预测、4032点原始历史、17折及原有目标/节假日源。现已整组迁移到新rolling合同、每日23:55锚点采样和逐折重训，不再使用全部1729个候选原点；新路径支持final fit/bundle。父baseline仍是旧密集训练合同，当前组内对照使用control文件，不能要求它与父baseline的fingerprint相同，也不能混用历史结果。

| 文件 | 相对基座唯一实验因素 |
|---|---|
| lgbm_direct-pointwise_control.yaml | 组内对照；模型/特征同基座，但新窗口与采样使身份不同于未迁移的父baseline |
| lgbm_direct-pointwise_recent-state.yaml | 只加 value 的 recent_state：6/12/36 点，level/mean/std/diff/slope |
| lgbm_direct-pointwise_regularized.yaml | 复杂度控制组合：num_leaves=15、max_depth=5、min_child_samples=40、reg_alpha=0.1、reg_lambda=1.0；不改学习率、树数或损失。不能归因到单个参数 |
| lgbm_direct-pointwise_l2.yaml | 只将 objective 改为 regression_l2；报告仍比较 MAE/RMSE |
| lgbm_direct-pointwise_huber.yaml | 只将 objective 改为 huber，alpha=0.9 为显式损失参数 |
| lgbm_direct-pointwise_selection.yaml | 每训练窗 f_regression，最多20列、至少10列，强制保留原七个 target lag；不全月预选 |
| ridge_direct-pointwise.yaml | Ridge alpha=1.0 与必要的训练窗内 standard 特征缩放；不缩放目标 |
| xgb_direct-pointwise.yaml | XGBoost，显式 L1 objective、300树、学习率0.05、深度6、行列采样0.8、seed42；不是参数等价的 LightGBM |

输出为 `results/results_test/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/accuracy_ablation/<variant>/<result_identity>/backtest_only/`。不覆盖已有 baseline 结果。

所有参数是预先指定的候选，不是调优结果。主指标 MAE，辅指标 RMSE/MAPE、逐日表现；保留8月19日既定评分，另报排除该补值日的敏感性分析。未完成正式回测前不宣称改善，组合候选须等单因素验证后另行构建；同月选型不是独立泛化验证。
