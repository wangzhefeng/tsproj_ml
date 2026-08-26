# ESS 储能站用电预测

本目录维护储能站用电预测的模型配置、严格外生信息集、策略特征工程和消融实验。

## 入口与目录

| 路径 | 作用 |
|---|---|
| `build_strategy_features.py` | 薄 CLI 入口，只负责参数解析和调用流水线 |
| `strategy_features.yaml` | v2 数据边界、运行阈值、相似日参数和 A/B 路数据源 |
| `strategy_features/` | v2 可测试实现：时间窗口、状态编码、周期画像、相似日与主流水线 |
| `derive_weather.py` | 构建 5min actual history、historical forecast backtest、final forecast 三段派生天气 |
| `route_{A,B}/baseline/` | 纯目标 self-lag quantile 基线：USBR/USMD/USMDP/USMDR/USMR |
| `route_{A,B}/add_decomposition/` | 关闭 datetime；USMD/USMDP/USMDR/USMR × linear/STL(288)/MSTL(288,2016) |
| `route_{A,B}/add_endogenous_actual_strategy/` | `ess_power + pcs_power` 多变量 point 测试，含 horizon 与 auxiliary 对照 |
| `route_{A,B}/add_exogenous_weather_date/` | 无目标分解；baseline + datetime + date_type + 严格天气 |
| `route_{A,B}/add_exogenous_plan_strategy/` | 关闭 datetime；显式未来 PCS 计划 quantile 测试 |
| `route_{A,B}/add_exogenous_weather_date_plan_strategy/` | 无目标分解；datetime + 日期 + 天气 + PCS 计划联合测试 |
| `route_{A,B}/add_strategy_features/` | 无目标分解；baseline + C5 全量策略特征的5种 LightGBM+CQR 配置 |

生成数据落在 `dataset/aidc_ess_selfuse_load/forecasting_data/strategy_features/`，由仓库级 `dataset/` 忽略规则排除，不提交 Git。

## 模型测试矩阵

当前 A/B 两路结构完全对称，每路41个、合计82个模型配置：34个 quantile+CQR/quantile 配置与7个 point配置。

| 组 | 每路配置数 | 方法/变量 |
|---|---:|---|
| baseline | 5 | USBR、USMD、USMDP、USMDR、USMR；quantile；无分解、无外生、无 datetime |
| add_decomposition | 12 | 四种非 blend 方法 × linear/STL(288)/MSTL(288,2016)；quantile；关闭 datetime |
| add_endogenous_actual_strategy | 7 | MSBR、MSMD、MSMD-horizon、MSMDR/MSMR persistence 与 auxiliary；point；无分解 |
| add_exogenous_weather_date | 4 | USMD、USMDP、USMDR、USMR |
| add_exogenous_plan_strategy | 4 | USMD、USMDP、USMDR、USMR；关闭 datetime |
| add_exogenous_weather_date_plan_strategy | 4 | USMD、USMDP、USMDR、USMR；无分解 |
| add_strategy_features | 5 | USMD、USMD-horizon、USMDP、USMDR、USMR；baseline + C5；quantile+CQR；无分解 |

USBR/MSBR 是 Direct+Recursive 多步策略，不是 `ModelEnsemble`；本场景所有配置均关闭模型融合。USBR 不进入外生组：其 Direct/Recursive 子模型共享一套 X，不能同时表达 Direct 的 horizon-aware 外生轨迹与 Recursive 的逐步外生量。USMD/USMDR 外生配置启用 `use_horizon_exogenous_for_direct`，每个输出模型只消费对应目标 horizon 的外生列；USMDR 使用 `block_size: 96`，288步预测真实执行3个block。USMR 逐步传入 future weather/custom；USMDP 统一使用 self-lag safe-lag：`enable_lags_features=true`、`align_direct_features_to_target=true`、`min(lags)>=horizon`。

所有配置统一为仅测试模式：`is_testing: true`、`is_forecasting: false`。当前 `history_length=34`、`window_length=30`、`predict_steps=288`，形成5个滑窗；每窗总长30天，其中训练29天、测试1天。相同 testing-only 契约同时适用于 `aidc_load_15min_daily`、`aidc_load_15min_rolling`、`aidc_load_15min_short` 和 `aidc_power_month` 的全部模型YAML。

`add_endogenous_actual_strategy` 的输出路径与配置目录同名。auxiliary未来PCS轨迹只适用于需要逐步回填的 MSMR/MSMDR；MSMD一次性Direct输出、MSBR双分支共享输入，当前不支持只靠 `endogenous_backfill_strategy=auxiliary` 接入，配置中不伪造该能力。

## 严格天气信息集

执行：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python \
  config/aidc_ess_selfuse_load/derive_weather.py
```

生成：

| 文件 | 角色 |
|---|---|
| `weather_derived_in_20250101_20260728.csv` | 仅含截至 07-28 的 `rt_*` actual history |
| `weather_derived_backtest_forecast_20260628_20260728.csv` | 历史raw中的 `pred_*` 归档，覆盖31日；当前5窗测试消费07-24～07-28，带 `source_ts/available_at` |
| `weather_derived_future_20260729_20260814.csv` | 仅由 `pred_*` 派生的 forecast future，带 `available_at` |

天气配置使用原生 strict weather 通路：history=`actual`、backtest=`forecast`、future=`forecast`。每个CV fold按 `available_at <= fold_origin` 选择目标日历史预报；测试/预测时间戳缺失时直接失败。7 个派生列由 `weather_features` 显式声明；预计算 `cal_rh` 保持原值，不重复覆盖。

PCS 计划继续使用通用 `custom_features` 注册表，而不是增加 PCS 专用 loader：计划历史/未来列同名、无 weather 的 `pred_*→rt_*` 映射需求。当前业务契约把目标日完整计划的 `available_at` 设为前一日23:55；`future_strategy: explicit`、`availability: forecast_origin` 和 strict coverage 保证每个fold只读取预测原点前已发布的完整288点计划。

## 构建与验证

在仓库根目录执行：

```bash
# 完整计算和契约校验，不写文件
UV_CACHE_DIR=.uv_cache uv run --no-sync python \
  config/aidc_ess_selfuse_load/build_strategy_features.py \
  --config config/aidc_ess_selfuse_load/strategy_features.yaml \
  --validate-only

# 写入输出；已有文件时必须显式 --force
UV_CACHE_DIR=.uv_cache uv run --no-sync python \
  config/aidc_ess_selfuse_load/build_strategy_features.py \
  --config config/aidc_ess_selfuse_load/strategy_features.yaml \
  --force
```

可用 `--data-root` 覆盖默认数据根目录，便于测试临时数据或其他数据版本。

## v2 特征结构

v2 同时维护两个时间轴：

- **自然日** `[00:00, 23:55]`：用于相似日检索和全天计划曲线。
- **调度周期** `[22:00, 次日 21:55]`：用于充电、待机、放电周期画像。

模型特征分为四层：

| 层 | 主要特征 | 未来可用性 |
|---|---|---|
| 日滞后与已完成周期 | `ess_lag_288`、`pcs_actual_lag_288`、前一已完成周期时长及一致率 | 来源时间必须不晚于 `as_of_time` |
| 已知计划与周期画像 | `pcs_plan`、计划方向 one-hot、充放电时长/能量/切换次数/首时刻周期编码 | 由未来计划直接构造 |
| 相似日模板 | 计划距离、新颖度、有效样本数、相似日/稳健近期模板及 gate 融合模板 | 只检索历史完整自然日 |
| 联合聚类 lag1 | D−1 的 ESS/实际 PCS/计划 PCS 三视图 one-hot、中心距离、rare、ready | artifact 只在固定 reference 期拟合，D 日只读取 D−1 |

联合聚类使用 `2025-11-01～2026-06-27` 固定 reference：每个 view 独立拟合 StandardScaler+PCA（解释率 ≥90%），block 归一后在 K=2～5 中按 silhouette 选 KMeans；小类不合并，只标记 rare。回测 artifact 与未来最终 forecast artifact 不得混用。

## C5 策略特征模型矩阵

策略特征数据流水线仍按 C1→C2→C3→C5 累积构建，但模型配置不再保留 C0/C1/C2/C3 的阶段性YAML。`add_strategy_features` 统一以 baseline 为模型基座，追加 C5 完整50列：

| 层 | 列数 | 主要特征 |
|---|---:|---|
| C1 | 10 | ESS/实际PCS日滞后、上一完成周期状态与 readiness |
| C2增量 | 21 | 已知计划、方向、周期时长/段数/能量/切换与首时刻编码 |
| C3增量 | 11 | 计划距离/新颖度、相似日与稳健模板、Gate及 readiness |
| C5增量 | 8 | D−1 联合聚类 one-hot、中心距离、rare、readiness |
| C5合计 | 50 | 上述全部因果可用策略特征 |

每路模型配置为：

- `lgbm_usmd_prob_mean_conformal.yaml`
- `lgbm_usmd_mean_prob_horizon_conformal.yaml`
- `lgbm_usmdp_prob_mean_conformal.yaml`
- `lgbm_usmdr_prob_mean_conformal.yaml`
- `lgbm_usmr_prob_mean_conformal.yaml`

它们全部使用 quantile q10/q50/q90 + CQR、`decomposition_method: none`、关闭 datetime/date/weather/ensemble，并继承 baseline 的目标 self-lag。普通配置使用 `-conformal-strategy-c5`，horizon配置使用 `-horizon-conformal-strategy-c5`，结果互不碰撞。

## 因果与数据契约

1. `as_of_time` 是历史事实边界；任何未来特征不得读取该时刻之后的 ESS 或实际 PCS。
2. 未来必须是完整、唯一、递增且严格对齐 5 分钟网格的自然日；历史缺口只记录审计，不直接阻断。
3. `contracts.py` 维护唯一的模型列清单、未来关键列和禁止列/命名模式。
4. 未来关键列必须有限且无缺失；禁止出现目标、实际 PCS、同日事后聚类等泄漏列。
5. `lag_feature_ready` 与 `template_feature_ready` 是显式 readiness 标志，不用隐式 NaN 表示是否可用。
6. 默认不覆盖已有输出；写入必须显式 `--force`，避免配置与历史产物不一致。

## 输出

每路生成：

- `model_features_history_{A,B}.csv`
- `model_features_future_{A,B}.csv`
- `audit/calendar_day_quality_{A,B}.csv`
- `audit/dispatch_cycle_summary_{A,B}.csv`
- `audit/similar_day_matches_{A,B}.csv`
- `audit/feature_build_audit_{A,B}.json`
- `audit/joint_cluster_assignments_{A,B}_fit-20260627.csv`
- `artifacts/joint_cluster_{A,B}_fit-20260627.joblib`

## 测试

```bash
UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest discover \
  -s tests -p "test_*.py"
```

策略特征专项测试为 `tests/test_ess_strategy_*.py`，覆盖状态编码、双时间轴、周期画像、相似日、联合聚类以及流水线泄漏/落盘契约。
