# ESS 储能站用电预测：模型测试设计总览

> 覆盖 `config/aidc_ess_selfuse_load` 下 A/B 两路共82个模型配置（每路41个）。A/B结构完全对称，本文按场景合并说明。本文描述当前测试设计，不代表已有有效结果；配置或数据变更后必须重跑，指标以各setting下的`test_scores_df.csv`为准。

## 场景矩阵

| 场景 | 配置/路 | 方法 | 测试目标 |
|---|---:|---|---|
| baseline | 5 | USBR/USMD/USMDP/USMDR/USMR | 概率预测方法基线 |
| add_decomposition | 12 | USMD/USMDP/USMDR/USMR × 3 | linear、STL(288)、MSTL(288,2016) 分解 |
| add_endogenous_actual_strategy | 7 | MSBR/MSMD/MSMDR/MSMR | 实际PCS内生变量、horizon与auxiliary；仅point |
| add_exogenous_weather_date | 4 | USMD/USMDP/USMDR/USMR | 严格天气 + date_type + datetime |
| add_exogenous_plan_strategy | 4 | USMD/USMDP/USMDR/USMR | 显式未来 PCS 计划 |
| add_exogenous_weather_date_plan_strategy | 4 | USMD/USMDP/USMDR/USMR | 严格天气 + 日期 + PCS计划；无分解 |
| add_strategy_features | 5 | USMD/USMD-horizon/USMDP/USMDR/USMR | baseline + C5完整策略特征；quantile+CQR |

## 共用口径

- 数据：A/B 站用电清洗数据，5min 粒度，每日 288 点；多变量组使用 ESS+PCS 合并数据。
- 所有配置：34天评估历史、30天总窗口、预测288点、5窗；每窗实际训练29天、测试1天。
- quantile配置共68个，point配置共14个；point只存在于`add_endogenous_actual_strategy`。
- 评估：窗口 MAPE 中位数为主，同时报告 MAE/RMSE、有效点比例和昨日同时刻 Naive。
- Gate：A/B 同向且 median ΔMAPE ≥ 0.005 才判定有效。
- USMDP统一使用self-lag safe-lag：`enable_lags_features=true`、`align_direct_features_to_target=true`、`min(lags)>=horizon`。
- USMDR 使用 `block_size=96`，288步预测由3个block组成；每个block只训练/输出96步并把预测回填到下一block的lag状态。
- 所有配置`is_testing=true`、`is_forecasting=false`，只生成滑窗测试结果，不训练/保存final forecast。

## 1. Baseline

五种概率预测结构：

| 配置 | 方法 | 目标处理 |
|---|---|---|
| `lgbm_usbr_prob_mean.yaml` | USBR | none（blend 不支持分解） |
| `lgbm_usmd_prob_mean.yaml` | USMD | none |
| `lgbm_usmdp_prob_mean.yaml` | USMDP | none |
| `lgbm_usmdr_prob_mean.yaml` | USMDR | none |
| `lgbm_usmr_prob_mean.yaml` | USMR | none |

baseline关闭datetime/date/weather/custom和`ModelEnsemble`，只使用目标self-lag及方法允许的目标rolling/diff。USBR是Direct+Recursive多步策略，不是模型融合。

## 2. Add decomposition

四种非blend方法各测试三种分解，且全部关闭datetime，避免分解消融混入时间外生特征：

| 文件模式 | 分解 | 周期 |
|---|---|---:|
| `lgbm_<method>_prob_mean_decomp_linear.yaml` | linear trend | — |
| `lgbm_<method>_prob_mean_decomp_stl288.yaml` | robust STL | 288（1 天） |
| `lgbm_<method>_prob_mean_decomp_mstl288-2016.yaml` | robust MSTL | 288、2016（1 天、1 周） |

方法为USMD/USMDP/USMDR/USMR；每路12个，setting suffix分别为`-decomp-linear`、`-decomp-stl288`、`-decomp-mstl288-2016`。USBR因blend与分解硬性不兼容，不进入本组。

## 3. Actual PCS endogenous

目标`ess_power`，附加内生变量`pcs_power`；每路保留MSBR、MSMD、MSMD-horizon、MSMDR persistence/auxiliary、MSMR persistence/auxiliary共7个point配置，全部`decomposition_method=none`。auxiliary只适用于需要逐步回填的MSMR/MSMDR；MSMD/MSBR不支持用单一开关接入辅助PCS轨迹。结果路径与目录同名：`add_endogenous_actual_strategy`。

## 4. Weather + date

- 日期：开启 `date_type`，按 categorical 处理；同时保留 datetime。
- 天气：7 列派生量 `rt_ssr/rt_tt2/cal_rh/rt_ws10/tt2_mean_3h/tt2_diff_1h/ssr_mean_3h`。
- 严格三角色：actual history 截止 07-28；backtest 使用历史raw中06-28～07-28的 `pred_*`；final future 从07-29起使用future raw的 `pred_*`。
- 来源契约：history=`actual`、backtest=`forecast`、future=`forecast`；backtest/future 带 `source_ts/available_at`。每个CV fold要求 `available_at <= fold_origin` 且精确覆盖288个目标点。
- 方法对齐：USMD/USMDR启用horizon-aware外生，但第h个输出模型只消费第h个目标时刻外生；USMR逐步使用future weather；USMDP按未来时间戳逐点使用。

## 5. PCS plan

PCS 计划走通用 `custom_features` 注册表，而非 weather loader：历史/未来列同名，无 `pred_*→rt_*` 映射。契约为：

```yaml
future_strategy: explicit
availability: forecast_origin
strict_information_set: true
available_at_col: available_at
```

当前计划数据契约声明目标日完整计划在前一日23:55可用；缺目标点或发布时间晚于fold origin直接失败。USMD/USMDR按输出horizon使用计划；USMR逐步传入custom future；USMDP逐时间戳使用。

本组关闭datetime，只增加显式PCS计划；当前仍使用linear分解。

## 6. Weather + date + plan

组合严格天气、categorical`date_type`、datetime和显式`pcs_plan`，全部`decomposition_method=none`。结果路径与目录同名：`add_exogenous_weather_date_plan_strategy`。

USBR 不进入外生组：Direct/Recursive 子模型共享 X，当前不能同时表达 Direct 的 horizon-aware 外生轨迹和 Recursive 的逐步外生量；baseline USBR 作为 no-exogenous control。

## 7. Strategy features C5 model matrix

策略特征数据仍按C1→C2→C3→C5累积构建，但模型配置只消费C5完整50列，不再保留C0/C1/C2/C3阶段性YAML。每路配置为：

- `lgbm_usmd_prob_mean_conformal.yaml`
- `lgbm_usmd_mean_prob_horizon_conformal.yaml`
- `lgbm_usmdp_prob_mean_conformal.yaml`
- `lgbm_usmdr_prob_mean_conformal.yaml`
- `lgbm_usmr_prob_mean_conformal.yaml`

五个配置均以baseline为基座，追加同一C5 custom source，使用q10/q50/q90+CQR、无分解、关闭datetime/date/weather/ensemble并保留目标self-lag。目标日D只允许D日完整计划和D−1及以前的ESS/实际PCS；C5固定reference到2026-06-27。

## 结果产物

当前配置只生成`test_scores_df.csv`、`cv_plot_df.csv`、`test_prediction.png`、`window_plots/`及概率/残差诊断，不生成final`prediction.csv`与final预训练模型。下表是2026-08-23旧31窗+forecast口径的历史结果；当前窗口数、分解、datetime、结果路径和forecast开关均已改变，因此这些数值只作历史证据，不得用于当前配置判定。

| 组 | 方法 | A median MAPE | B median MAPE |
|---|---|---:|---:|
| weather+date | USMD | 0.279681 | 0.279222 |
| weather+date | USMDP | 0.277351 | 0.227365 |
| weather+date | USMDR | 0.286474 | 0.230928 |
| weather+date | USMR | 0.278513 | 0.214317 |
| plan | USMD | 0.264958 | 0.244060 |
| plan | USMDP | 0.240176 | 0.201512 |
| plan | USMDR | 0.270433 | 0.223051 |
| plan | USMR | 0.266761 | 0.214405 |
| weather+date+plan | USMD | 0.276896 | 0.249917 |
| weather+date+plan | USMDP | 0.259609 | 0.213409 |
| weather+date+plan | USMDR | 0.275527 | 0.218231 |
| weather+date+plan | USMR | 0.262723 | 0.213367 |
