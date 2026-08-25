# ESS 储能站用电预测：模型测试设计总览

> 覆盖 `config/aidc_ess_selfuse_load` 下 A/B 两路共 94 个模型配置（忽略 `ensemble`、`params_tuning`）。A/B 结构一致，本文按场景合并说明。本文描述测试设计，不代表已有有效结果；配置或数据变更后必须重跑，指标以各 setting 下的 `test_scores_df.csv` 为准。

## 场景矩阵

| 场景 | 配置/路 | 方法 | 测试目标 |
|---|---:|---|---|
| baseline | 5 | USBR/USMD/USMDP/USMDR/USMR | 概率预测方法基线 |
| add_decomposition | 8 | USMD/USMDP/USMDR/USMR × 2 | linear 与 STL(288) 分解 |
| tuning | 7 | USMD/USMDP | horizon、CQR、衰减、窗口等增强 |
| add_endogenous | 10 | MSBR/MSMD/MSMDR/MSMR | 实际 PCS 内生变量及回填策略 |
| add_exogenous_weather_date | 4 | USMD/USMDP/USMDR/USMR | 严格天气 + date_type + datetime |
| add_exogenous_plan_strategy | 4 | USMD/USMDP/USMDR/USMR | 显式未来 PCS 计划 |
| add_exogenous_all | 4 | USMD/USMDP/USMDR/USMR | 严格天气 + 日期 + PCS 计划 |
| add_strategy_features | 5 | USMDP | C0–C5 因果策略特征消融 |

## 共用口径

- 数据：A/B 站用电清洗数据，5min 粒度，每日 288 点；多变量组使用 ESS+PCS 合并数据。
- 默认滑窗：34 天评估历史、30 天训练窗、预测 288 点、5 窗；`window_length=14` 的 tuning 配置使用 18 天评估历史，同样保持 5 窗。
- 本轮 baseline/add_decomposition/三类外生组只维护 quantile q10/q50/q90，`quantile_monotone=true`，不新增 point；既有 tuning、add_endogenous、add_strategy_features 中的 point 实验保持不变。
- 评估：窗口 MAPE 中位数为主，同时报告 MAE/RMSE、有效点比例和昨日同时刻 Naive。
- Gate：A/B 同向且 median ΔMAPE ≥ 0.005 才判定有效。
- USMDP 不生成框架目标 lag，配置显式 `enable_lags_features=false`、`lags=[]`。
- USMDR 使用 `block_size=96`，288步预测由3个block组成；每个block只训练/输出96步并把预测回填到下一block的lag状态。
- 多输出 quantile 不透传时间衰减 sample weight，运行时 WARNING 后跳过。

## 1. Baseline

五种概率预测结构：

| 配置 | 方法 | 目标处理 |
|---|---|---|
| `lgbm_usbr_prob_mean.yaml` | USBR | none（blend 不支持分解） |
| `lgbm_usmd_prob_mean.yaml` | USMD | linear |
| `lgbm_usmdp_prob_mean.yaml` | USMDP | linear |
| `lgbm_usmdr_prob_mean.yaml` | USMDR | linear |
| `lgbm_usmr_prob_mean.yaml` | USMR | linear |

USBR 与其他方法的横向差异同时包含预测结构和目标分解差异，因此严格分解对照由下一组承担。

## 2. Add decomposition

四种非 blend 方法各测试两种分解：

| 文件模式 | 分解 | 周期 |
|---|---|---:|
| `lgbm_<method>_prob_mean_decomp_linear.yaml` | linear trend | — |
| `lgbm_<method>_prob_mean_decomp_stl288.yaml` | robust STL | 288（1 天） |

方法为 USMD/USMDP/USMDR/USMR；每路 8 个，setting suffix 分别为 `-decomp-linear`、`-decomp-stl288`。USBR 因 blend 与分解硬性不兼容，不进入本组。

## 3. Tuning

保留 7 个单因素增强：USMD horizon feature、USMD CQR、USMDP decay14/30、历史命名为 21/14 与 30/14 的窗口配置、USMDP rolllag。为统一加速测试，两个 `window_length=14` 配置当前均使用 `history_length=18`，得到 5 个滑窗；文件名和 setting suffix 只保留实验谱系标识，不再表示当前 history_length。rolllag 仍作为独立诊断项保留。

## 4. Actual PCS endogenous

目标 `ess_power`，附加内生变量 `pcs_power`；比较 MSBR/MSMD/MSMDR/MSMR、horizon feature、persistence 与 auxiliary 回填、quantile+CQR。目标日真实 PCS 不作为未来已知输入。

## 5. Weather + date

- 日期：开启 `date_type`，按 categorical 处理；同时保留 datetime。
- 天气：7 列派生量 `rt_ssr/rt_tt2/cal_rh/rt_ws10/tt2_mean_3h/tt2_diff_1h/ssr_mean_3h`。
- 严格三角色：actual history 截止 07-28；backtest 使用历史raw中06-28～07-28的 `pred_*`；final future 从07-29起使用future raw的 `pred_*`。
- 来源契约：history=`actual`、backtest=`forecast`、future=`forecast`；backtest/future 带 `source_ts/available_at`。每个CV fold要求 `available_at <= fold_origin` 且精确覆盖288个目标点。
- 方法对齐：USMD/USMDR启用horizon-aware外生，但第h个输出模型只消费第h个目标时刻外生；USMR逐步使用future weather；USMDP按未来时间戳逐点使用。

## 6. PCS plan

PCS 计划走通用 `custom_features` 注册表，而非 weather loader：历史/未来列同名，无 `pred_*→rt_*` 映射。契约为：

```yaml
future_strategy: explicit
availability: forecast_origin
strict_information_set: true
available_at_col: available_at
```

当前计划数据契约声明目标日完整计划在前一日23:55可用；缺目标点或发布时间晚于fold origin直接失败。USMD/USMDR按输出horizon使用计划；USMR逐步传入custom future；USMDP逐时间戳使用。

## 7. Weather + date + plan

组合严格天气、categorical `date_type`、datetime 和显式 `pcs_plan`。解读时分别与 baseline、weather+date、plan 做同路同方法比较，判断联合增益与冗余。

USBR 不进入外生组：Direct/Recursive 子模型共享 X，当前不能同时表达 Direct 的 horizon-aware 外生轨迹和 Recursive 的逐步外生量；baseline USBR 作为 no-exogenous control。

## 8. Strategy features C0–C5

| 层级 | 新增信息 |
|---|---|
| C0 | datetime 基线，无框架 lag |
| C1 | ESS/PCS 日滞后、上一完整调度周期摘要 |
| C2 | 当前计划与计划周期画像 |
| C3 | 相似日、稳健近期模板、novelty 与 gate |
| C5 | D−1 联合聚类 one-hot/距离/rare/ready |

目标日 D 只允许 D 日完整计划和 D−1 及以前的 ESS/实际 PCS；C5 固定 reference 到 2026-06-27，仅 testing，不做 final forecasting。

## 结果产物

每个 setting 目录均包含 `test_scores_df.csv`、`cv_plot_df.csv`、`test_prediction.png`、`window_plots/`、final `prediction.csv` 与预训练模型。2026-08-23 的历史批次已完成三类外生组 A/B 共24个正式配置：每配置31个测试窗口、288步 final forecast，失败0，所有分位数满足 q10≤q50≤q90。当前配置已统一改为5窗，下次运行会覆盖同 setting 下的历史31窗测试产物。

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
