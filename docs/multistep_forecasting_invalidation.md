# 多步预测重构结果失效清单

> 状态：已审计，待按业务优先级重跑
> 审计日期：2026-08-25
> 权威设计：`docs/multistep_forecasting_redesign.md` §11.4、§17

## 1. 结论

- 当前共有 **863 个模型 YAML**；独立聚合/检测 YAML 不计入。
- 语义修复涉及 **94 个唯一配置**。分类之间允许重叠，不能把各分类数量直接相加。
- 当前路径下命中旧产物：测试结果 **43** 组、模型 **0** 组、正式预测 **0** 组。
- 命中的旧产物均标记为 **stale**：不得与重构后结果混用，也不得作为当前语义的验收证据。
- 本轮遵守“不进行全量重跑”的约束：**未删除、未覆盖、未重训**这些结果；只完成清单、契约回归和兼容读取验收。

## 2. 重跑原则

1. 单个配置重跑时，必须同时重建滑窗测试、最终模型和正式预测；仅回测通过不得称完成。
2. 重跑前保留或导出旧指标，避免同一 `setting` 覆盖后失去前后对照。
3. 分类 A1/A2 属数值语义变化，优先级高于纯产物格式迁移。
4. A4/A5 若仅分析历史 CSV，可继续保留旧表；若需要独立加载模型推理，必须生成新 typed artifact。
5. Global panel 当前没有现役 YAML，不存在存量结果失效；新配置必须从零完成双序列验收。

## 3. 逐配置清单

### A1. MSMD/MSMDR 目标时点修复

配置数：**34**。

| 配置 | 当前结果路径（`<scenario>/<setting>`） | model | test | forecast |
|---|---|---:|---:|---:|
| `config/ETT-small/ETTm1/cab_msmd.yaml` | `ETT-small/ETTm1/catboost-ETTm1-msmd-15` | 无 | 无 | 无 |
| `config/ETT-small/ETTm1/cab_msmdr.yaml` | `ETT-small/ETTm1/catboost-ETTm1-msmdr-15` | 无 | 无 | 无 |
| `config/ETT-small/ETTm1/lgbm_msmd.yaml` | `ETT-small/ETTm1/lightgbm-ETTm1-msmd-15` | 无 | 无 | 无 |
| `config/ETT-small/ETTm1/lgbm_msmdr.yaml` | `ETT-small/ETTm1/lightgbm-ETTm1-msmdr-15` | 无 | 无 | 无 |
| `config/ETT-small/ETTm1/xgb_msmd.yaml` | `ETT-small/ETTm1/xgboost-ETTm1-msmd-15` | 无 | 无 | 无 |
| `config/ETT-small/ETTm1/xgb_msmdr.yaml` | `ETT-small/ETTm1/xgboost-ETTm1-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A1_01a/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_01a/lightgbm-df_selected-msmd-15` | 无 | 有（stale） | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A1_01a/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_01a/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A1_201/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_201/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A1_201/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_201/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A1_IT/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A1_IT/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A3_01e/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A3_01e/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-06-11/A3_01e/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A3_01e/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/A2_IT/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/A2_IT/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/A3_IT/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/A3_IT/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/liantong_IT/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/liantong_IT/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/yancheng_IT/lgbm_msmd.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmd-15` | 无 | 无 | 无 |
| `config/aidc_electricity_computility/electricity/2026-08-10/yancheng_IT/lgbm_msmdr.yaml` | `aidc_electricity_computility/electricity/2026-06-11/A1_IT/lightgbm-df_selected-msmdr-15` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmd_mean_pcs.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmd-30` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmd_mean_pcs_horizon.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmd-30-horizon` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmd_prob_mean_pcs_conformal.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmd-30-quantile-conformal` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmdr_mean_pcs.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmdr-30` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmdr_mean_pcs_aux.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmdr-30-aux` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmdr_prob_mean_pcs_conformal.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmdr-30-quantile-conformal` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmd_mean_pcs.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmd-30` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmd_mean_pcs_horizon.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmd-30-horizon` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmd_prob_mean_pcs_conformal.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmd-30-quantile-conformal` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmdr_mean_pcs.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmdr-30` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmdr_mean_pcs_aux.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmdr-30-aux` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmdr_prob_mean_pcs_conformal.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmdr-30-quantile-conformal` | 无 | 无 | 无 |

### A2. 默认 DirRec 训练宽度 H→B

配置数：**28**。

| 配置 | 当前结果路径（`<scenario>/<setting>`） | model | test | forecast |
|---|---|---:|---:|---:|
| `config/aidc_load_15min_short/route_A/add_decomposition/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_load_15min_short/route_A/add_decomposition/lightgbm-A_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-linear` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_A/add_decomposition/lgbm_usmdr_prob_mean_decomp_stl96.yaml` | `aidc_load_15min_short/route_A/add_decomposition/lightgbm-A_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-stl96` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_A/add_exogenous_weather/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_A/add_exogenous_weather/lightgbm-A_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_A/add_load_state/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_load_15min_short/route_A/add_load_state/lightgbm-A_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-linear-load-state` | 无 | 无 | 无 |
| `config/aidc_load_15min_short/route_A/add_load_state/lgbm_usmdr_prob_mean_decomp_stl96.yaml` | `aidc_load_15min_short/route_A/add_load_state/lightgbm-A_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-stl96-load-state` | 无 | 无 | 无 |
| `config/aidc_load_15min_short/route_A/baseline/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_A/baseline/lightgbm-A_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_B/add_decomposition/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_load_15min_short/route_B/add_decomposition/lightgbm-B_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-linear` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_B/add_decomposition/lgbm_usmdr_prob_mean_decomp_stl96.yaml` | `aidc_load_15min_short/route_B/add_decomposition/lightgbm-B_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-stl96` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_B/add_exogenous_weather/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_B/add_exogenous_weather/lightgbm-B_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_B/add_load_state/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_load_15min_short/route_B/add_load_state/lightgbm-B_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-linear-load-state` | 无 | 无 | 无 |
| `config/aidc_load_15min_short/route_B/add_load_state/lgbm_usmdr_prob_mean_decomp_stl96.yaml` | `aidc_load_15min_short/route_B/add_load_state/lightgbm-B_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-decomp-stl96-load-state` | 无 | 无 | 无 |
| `config/aidc_load_15min_short/route_B/baseline/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_B/baseline/lightgbm-B_Loads_15min_mean_20251001_20260731-usmdr-14-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_month/route_A/lgbm_usmdr_mean.yaml` | `aidc_load_month/route_A/lightgbm-A_Loads_1day_mean_20251001_20260731-usmdr-150` | 无 | 无 | 无 |
| `config/aidc_load_month/route_A/lgbm_usmdr_prob_mean.yaml` | `aidc_load_month/route_A/lightgbm-A_Loads_1day_mean_20251001_20260731-usmdr-150-quantile` | 无 | 无 | 无 |
| `config/aidc_load_month/route_B/lgbm_usmdr_mean.yaml` | `aidc_load_month/route_B/lightgbm-B_Loads_1day_mean_20251001_20260731-usmdr-150` | 无 | 无 | 无 |
| `config/aidc_load_month/route_B/lgbm_usmdr_prob_mean.yaml` | `aidc_load_month/route_B/lightgbm-B_Loads_1day_mean_20251001_20260731-usmdr-150-quantile` | 无 | 无 | 无 |
| `config/aidc_power_month/route_A/freq_1day/add_decomposition/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_power_month/route_A/freq_1day/add_decomposition/lightgbm-A_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-linear-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/add_decomposition/lgbm_usmdr_prob_mean_decomp_stl7.yaml` | `aidc_power_month/route_A/freq_1day/add_decomposition/lightgbm-A_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-stl7-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/add_exogenous_weather_date/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_power_month/route_A/freq_1day/add_exogenous_weather/lightgbm-A_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/add_load_state/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_power_month/route_A/freq_1day/add_load_state/lightgbm-A_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-linear-load-state-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/add_load_state/lgbm_usmdr_prob_mean_decomp_stl7.yaml` | `aidc_power_month/route_A/freq_1day/add_load_state/lightgbm-A_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-stl7-load-state-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/baseline/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_power_month/route_A/freq_1day/baseline/lightgbm-A_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/add_decomposition/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_power_month/route_B/freq_1day/add_decomposition/lightgbm-B_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-linear-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/add_decomposition/lgbm_usmdr_prob_mean_decomp_stl7.yaml` | `aidc_power_month/route_B/freq_1day/add_decomposition/lightgbm-B_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-stl7-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/add_exogenous_weather_date/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_power_month/route_B/freq_1day/add_exogenous_weather/lightgbm-B_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/add_load_state/lgbm_usmdr_prob_mean_decomp_linear.yaml` | `aidc_power_month/route_B/freq_1day/add_load_state/lightgbm-B_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-linear-load-state-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/add_load_state/lgbm_usmdr_prob_mean_decomp_stl7.yaml` | `aidc_power_month/route_B/freq_1day/add_load_state/lightgbm-B_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-decomp-stl7-load-state-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/baseline/lgbm_usmdr_prob_mean_conformal.yaml` | `aidc_power_month/route_B/freq_1day/baseline/lightgbm-B_Loads_1day_sum_20251001_20260731-usmdr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |

### A3. USMDP safe-lag 正式启用

配置数：**10**。

| 配置 | 当前结果路径（`<scenario>/<setting>`） | model | test | forecast |
|---|---|---:|---:|---:|
| `config/aidc_ess_selfuse_load/route_A/tuning/lgbm_usmdp_prob_mean_rolllag.yaml` | `aidc_ess_selfuse_load/route_A/tuning/lightgbm-A_GateEnergys_5min_20251001_20260728_remove_outlier-usmdp-30-quantile-rolllag` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/tuning/lgbm_usmdp_prob_mean_rolllag.yaml` | `aidc_ess_selfuse_load/route_B/tuning/lightgbm-B_GateEnergys_5min_20251001_20260728_remove_outlier-usmdp-30-quantile-rolllag` | 无 | 无 | 无 |
| `config/aidc_power_month/route_A/freq_1month/window_length_10/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_A/freq_1month/window_length_10/lightgbm-A_Loads_1month_sum_20251001_20260731-usmdp-10-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1month/window_length_7/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_A/freq_1month/window_length_7/lightgbm-A_Loads_1month_sum_20251001_20260731-usmdp-7-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1month/window_length_8/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_A/freq_1month/window_length_8/lightgbm-A_Loads_1month_sum_20251001_20260731-usmdp-8-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1month/window_length_9/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_A/freq_1month/window_length_9/lightgbm-A_Loads_1month_sum_20251001_20260731-usmdp-9-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1month/window_length_10/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_B/freq_1month/window_length_10/lightgbm-B_Loads_1month_sum_20251001_20260731-usmdp-10-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1month/window_length_7/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_B/freq_1month/window_length_7/lightgbm-B_Loads_1month_sum_20251001_20260731-usmdp-7-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1month/window_length_8/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_B/freq_1month/window_length_8/lightgbm-B_Loads_1month_sum_20251001_20260731-usmdp-8-quantile` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1month/window_length_9/lgbm_usmdp_prob_mean.yaml` | `aidc_power_month/route_B/freq_1month/window_length_9/lightgbm-B_Loads_1month_sum_20251001_20260731-usmdp-9-quantile` | 无 | 有（stale） | 无 |

### A4. auxiliary 回填自描述化

配置数：**4**。

| 配置 | 当前结果路径（`<scenario>/<setting>`） | model | test | forecast |
|---|---|---:|---:|---:|
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmdr_mean_pcs_aux.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmdr-30-aux` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msmr_mean_pcs_aux.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msmr-30-aux` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmdr_mean_pcs_aux.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmdr-30-aux` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msmr_mean_pcs_aux.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msmr-30-aux` | 无 | 无 | 无 |

### A5. ridge-stacking 权重随产物保存

配置数：**20**。

| 配置 | 当前结果路径（`<scenario>/<setting>`） | model | test | forecast |
|---|---|---:|---:|---:|
| `config/aidc_ess_selfuse_load/route_A/add_endogenous_actual_strategy/lgbm_msbr_mean_pcs.yaml` | `aidc_ess_selfuse_load/route_A/add_endogenous/lightgbm-A_ESS_PCS_merged_5min_20251001_20260728-msbr-30` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_A/baseline/lgbm_usbr_prob_mean.yaml` | `aidc_ess_selfuse_load/route_A/baseline/lightgbm-A_GateEnergys_5min_20251001_20260728_remove_outlier-usbr-30-quantile` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/add_endogenous_actual_strategy/lgbm_msbr_mean_pcs.yaml` | `aidc_ess_selfuse_load/route_B/add_endogenous/lightgbm-B_ESS_PCS_merged_5min_20251001_20260728-msbr-30` | 无 | 无 | 无 |
| `config/aidc_ess_selfuse_load/route_B/baseline/lgbm_usbr_prob_mean.yaml` | `aidc_ess_selfuse_load/route_B/baseline/lightgbm-B_GateEnergys_5min_20251001_20260728_remove_outlier-usbr-30-quantile` | 无 | 无 | 无 |
| `config/aidc_load_15min_daily/route_A/add_exogenous_weather/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_daily/route_A/add_exogenous_weather/lightgbm-A_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal-control-no-weather` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_daily/route_A/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_daily/route_A/baseline/lightgbm-A_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_daily/route_B/add_exogenous_weather/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_daily/route_B/add_exogenous_weather/lightgbm-B_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal-control-no-weather` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_daily/route_B/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_daily/route_B/baseline/lightgbm-B_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_rolling/route_A/add_exogenous_weather/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_rolling/route_A/add_exogenous_weather/lightgbm-A_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal-control-no-weather` | 无 | 无 | 无 |
| `config/aidc_load_15min_rolling/route_A/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_rolling/route_A/baseline/lightgbm-A_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_rolling/route_B/add_exogenous_weather/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_rolling/route_B/add_exogenous_weather/lightgbm-B_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal-control-no-weather` | 无 | 无 | 无 |
| `config/aidc_load_15min_rolling/route_B/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_rolling/route_B/baseline/lightgbm-B_Loads_15min_mean_20251001_20260731-usbr-30-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_A/add_exogenous_weather/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_A/add_exogenous_weather/lightgbm-A_Loads_15min_mean_20251001_20260731-usbr-14-quantile-conformal-control-no-weather` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_A/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_A/baseline/lightgbm-A_Loads_15min_mean_20251001_20260731-usbr-14-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_B/add_exogenous_weather/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_B/add_exogenous_weather/lightgbm-B_Loads_15min_mean_20251001_20260731-usbr-14-quantile-conformal-control-no-weather` | 无 | 有（stale） | 无 |
| `config/aidc_load_15min_short/route_B/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_load_15min_short/route_B/baseline/lightgbm-B_Loads_15min_mean_20251001_20260731-usbr-14-quantile-conformal` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/add_exogenous_weather_date/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_power_month/route_A/freq_1day/add_exogenous_weather/lightgbm-A_Loads_1day_sum_20251001_20260731-usbr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_A/freq_1day/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_power_month/route_A/freq_1day/baseline/lightgbm-A_Loads_1day_sum_20251001_20260731-usbr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/add_exogenous_weather_date/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_power_month/route_B/freq_1day/add_exogenous_weather/lightgbm-B_Loads_1day_sum_20251001_20260731-usbr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |
| `config/aidc_power_month/route_B/freq_1day/baseline/lgbm_usbr_prob_mean_conformal.yaml` | `aidc_power_month/route_B/freq_1day/baseline/lightgbm-B_Loads_1day_sum_20251001_20260731-usbr-120-quantile-conformal-calendar-month` | 无 | 有（stale） | 无 |

## 4. 不要求数值重跑的兼容项

- 8 个 `aidc_power_month/**/freq_1month/window_length_{7,8,9,10}/lgbm_usmr_prob_mean.yaml` 删除了 Direct 专用的 `align_direct_features_to_target` 非适用字段；USMR one-step 仍由 Recursive 目标行天然读取目标月天气，定向契约测试覆盖，不改变预期数值语义。
- 20 个 USMDP 配置（`aidc_ess_selfuse_load/route_*/tuning/lgbm_usmdp_prob_mean_{decay14,decay30,hist21w14,hist30w14}.yaml`、`aidc_load_month/route_*/{,decomposition/}lgbm_usmdp_*.yaml`）清除了 `align_direct_features_to_target: false` 下不生效的 `lags` 残留（改为 `enable_lags_features: false` + `lags: []`）；这些配置均未启用 advanced_features，运行行为零变化，既有结果不受影响。
- `LegacyArtifactAdapter` 只读适配裸估计器、legacy quantile dict、blend dict 与 auxiliary dict；不原地改写旧 pkl。
- Global panel 新增端到端能力但现役配置数为 0，因此无历史结果需要失效。

## 4a. DirRec 块长退化修复（2026-08-25 追加）

审计发现 66 个 `block_size: 0` 的 DirRec 配置按 `B = min(lags)` 推导出退化块长：36 个 B=1（等价 Recursive），30 个 B=H（等价单块 Direct），方法名不副实。处置：

| 组 | 处置 | 数量 | 新块长 | 结果影响 |
|---|---|---:|---|---|
| `aidc_load_15min_daily`/`rolling` 全部 usmdr | 显式 `block_size: 24` | 24 | 24（4 块，lag 不消费预测值） | 语义变更，旧结果作废 |
| `aidc_ess_selfuse_load` msmdr（pcs 系） | 显式 `block_size: 96` | 6 | 96（与同场景 usmdr 显式值一致） | 语义变更，旧结果作废 |
| `aidc_load_15min_short` usmdr | 显式 `block_size: 4` | 12 | 4（4 块，短 lag 消费块内预测） | 本就在 A2 失效清单，重跑即新语义 |
| `aidc_load_month` usmdr | 显式 `block_size: 7` | 4 | 7（~5 块） | 本就在 A2 失效清单 |
| `aidc_power_month` freq_1day（calendar_month）usmdr | 显式 `block_size: 7` | 12 | 7（~5 块） | 本就在 A2 失效清单 |
| `aidc_power_month` freq_1month usmdr（H=1） | **删除配置** | 8 | — | H=1 下四推进族数学等价，保留无对比价值；无历史结果 |

修复后全部 70 个 DirRec 配置（66-8+12 个原有显式块长）经 `resolve_strategy` 复核无退化。模型 YAML 总数 863 → 855。15min daily/rolling（24 个）为语义变更新增失效，其余均已在 A2 覆盖范围内。

## 5. 已完成的安全验收

- `python -m unittest discover -s tests -p "test_*.py"`：371 tests，OK。
- 增强路径定向回归：42 tests，OK。
- `scripts/check_model_configs.py`：863 个模型 YAML 全部通过。
- 九方法双序列真实 Forecaster 冒烟：全部方法每个 series 严格返回 H 行；MSMR/MSMDR/MSBR 走 persistence backfill。
- 双序列完整配置冒烟：YAML → DataLoader → Ridge 训练 → `ForecastModelBundle` → pkl round-trip → A/B 正式预测。
- `compileall` 与 `git diff --check`：exit 0。

## 6. 后续重跑完成条件

- [ ] A1/A2 中仍在使用的配置按场景优先级重跑。
- [ ] A3 两个 safe-lag 配置完成 A/B 同口径消融。
- [ ] A4/A5 如继续保留为现役能力，生成新模型产物并验证离线独立推理。
- [ ] 每个重跑配置同时核对 `test_scores_df.csv`、`model.pkl`、`prediction.csv`。
- [ ] 更新本清单状态；未完成前不得把旧 CSV 宣称为重构后结果。
