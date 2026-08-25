# AIDC 月总用电量预测 · freq_1day 模型测试设计说明

> **场景**：`aidc_power_month`，A/B 两路数据中心。业务问题 = **月初预测下一完整自然月的总用电量（kWh）**。
> **路线**：日频建模——以 5min 功率聚合的日电量序列为目标，逐日预测 28~31 天，求和得到月总量。这是本场景的生产主链（月频 `freq_1month` 仅作低频对照）。
> **本文范围**：`config/aidc_power_month/route_{A,B}/freq_1day/` 下 baseline、add_exogenous_weather_date、add_decomposition、add_load_state 四组共 124 个配置（tuning 调参组不在本文范围）。A/B 两路配置结构完全相同，仅数据不同，下文不再分开赘述。

---

## 1. 测试总体思路

```text
历史日电量 304 天（2025-10-01 ~ 2026-07-31）
        │
        ├─ 滑窗回测：6 个完整自然月 fold（2026-02 ~ 2026-07）
        │   每个 fold：固定最近 120 天训练 → 预测当月 28~31 天 → 与真实值评分
        │
        └─ 最终预测：全部 304 天训练 → 预测 2026-08-01 ~ 08-31（31 天）
                     逐日预测求和 = 2026-08 月总用电量
```

| 设计要素 | 为什么需要 | 简单示例（数值仅示意） |
|---|---|---|
| 自然月 fold（`horizon_mode: calendar_month`） | 月总量预测的评测单位必须是完整自然月；用固定 30 天滚动窗会把跨月片段混进"月度"指标 | 2 月 fold 测 28 天、7 月 fold 测 31 天，各自独立计 MAPE |
| 固定训练天数（`train_window_length: 120`） | 各 fold 测试天数天然不等（28~31），若训练长度也随之漂移，模型间比较不公平 | 每个 fold 都用 120 天：2 月 fold 训 2025-10-04~02-01 前，7 月 fold 训 2026-04-02~06-30 |
| `predict_steps: 30` 为遗留兼容值 | calendar_month 模式下真实 horizon 由目标月天数决定，30 只是让旧校验通过的占位 | 校验时用 30 估 lag 可用行，实际输出 31 点 |
| setting 编码 `train_window_length` + `-calendar-month` | 避免与旧固定步长结果目录混淆 | `lightgbm-...-usmr-120-quantile-conformal-calendar-month` |

> **核心认识**：回测和正式预测共用同一条链路（同一特征工程、同一模型类、同一天气契约），回测 6 个自然月 fold 的中位数就是正式预测可信度的直接证据。

---

## 2. 数据处理

### 2.1 目标序列

| 项 | 约定 | 为什么 |
|---|---|---|
| 来源 | 5min 功率（kW）`sum × 1/12` 聚合为日电量（kWh） | 业务计量单位是电量而非功率 |
| 范围 | 2025-10-01 ~ 2026-07-31，A/B 各 304 行，无缺失 | 覆盖 10 个自然月，保证 6 个回测 fold + 120 天训练窗 |
| `now_time` | 2026-07-31（最后一个已知数据点） | 预测原点；正式预测目标 = 2026-08 全月 |

### 2.2 天气三段式严格信息集

月度/日度预测最易踩的坑是"用测试月的真实天气预测测试月"。本场景把天气数据按可得时间切成三段，并 fail-fast 校验：

| 段 | 文件 | 内容 | 为什么需要 |
|---|---|---|---|
| 历史（训练） | `weather_daily_stats_20251001_20260731.csv` | 1h 实测聚合的日统计（actual） | 训练期天气已知，可用实测 |
| 回测（测试期"未来"） | `weather_daily_stats_backtest_proxy_20260101_20260731.csv`（212 天） | 目标日取**上一年同日**实测，`available_at` 早于目标月 | 模拟回测时点真实可得的信息；禁止读测试月实测 |
| 正式未来 | `weather_daily_stats_future_20260801_20260831.csv`（31 天） | 2025-08 同日纯代理，`available_at = 2026-07-31` | 站在 7-31 原点，8 月任何实测都不存在 |

```yaml
# 所有启用天气的配置（100 个）统一契约
strict_weather_information_set: true
weather_history_source: actual
weather_backtest_source: proxy
weather_future_source: proxy
```

> **核心认识**：校验是硬性的——backtest 行 `available_at` 必须早于目标月、future 行不得晚于 `now_time`、目标日期缺失直接 RAISE。泄露差分实证：扰动 7 月实测天气回测预测完全不变，扰动 proxy 预测才变（11,105,254 → 11,096,473 kWh）。

### 2.3 负荷状态特征（add_load_state 组）

`load_state_features/` 下 15 个"预测原点状态"列（30 日稳健 z、斜率、日内离散度、A/B 路差、volatile 计数等），来自负荷事件分析产物。它们的问题是**当天结束后才能算出**，直接同日 merge 会形成同行监督泄露。契约：

| 环节 | 处理 | 为什么 |
|---|---|---|
| 声明 | `future_strategy: freeze_last_observation` + `availability: end_of_period` | 告诉框架这是"当期结束后可得"的状态 |
| Direct/DirRec 训练 | 原点行保留 `state(t)`，**不生成 `state_h*`**（所有 horizon 共用原点状态） | 避免训练用真实 `state(t+h)` 而预测只能冻结 `state(t)` 的漂移 |
| Recursive/Pointwise 训练 | source 时间戳后移 1 天，行 t 用 `state(t-1)` | 预测 t 时只能看到 t-1 日末状态 |
| CV/final future | 取 cutoff 前最后状态冻结到全 horizon | 与训练语义一致 |

### 2.4 目标分解（add_decomposition / add_load_state 组）

`decomposition_method: linear | stl`（STL 周期 7）。因果边界：

- 每个回测 fold **只用该 fold 的训练段**拟合分解器，测试段 y 保持原始电平；
- 最终预测才在全部 304 天上拟合；
- 与 `scale_target`、Conformal、USBR blend 互斥（配置层面已遵守）。

---

## 3. 特征工程

### 3.1 特征清单（baseline 为例，36 个 predictor）

| 类别 | 内容 | 为什么需要 | 示例（仅示意） |
|---|---|---|---|
| 滞后 lag | `[1,2,3,4,5,6,7,14,28]` | 近一周逐日 + 上周/上上周同日 + 月周期 | `y_lag_7` = 上周同日电量 |
| 滚动统计 | 窗口 `[7,14,28]` × mean/std/min/max | 平滑日内噪声，表达近期水平与波动 | `y_rolling_mean_7` = 近 7 天日均 |
| 差分 | 周期 `[1,7,28]` | 趋势与周/月环比 | `y_diff_7` = 今日 − 上周同日 |
| 日历 | 12 个 datetime 派生（day/day_of_week/week_of_year/month/days_in_month/quarter/day_of_year/year/四个边界标记），已剔除 minute/hour | 月初月末、周末、月长度都是电量驱动因子 | `dt_days_in_month` 区分 28/31 天的月 |
| 天气（weather 组） | 4 列：rt_tt2（日均气温）、cal_rh（湿度）、rt_ssr（日总辐射）、rt_ws10（风速） | 空调负荷与气温/辐射强相关 | `rt_tt2` 单位 K |
| 负荷状态（load_state 组） | 15 列 origin-frozen 状态 | 原点处"近期负荷形态"画像 | `state_z30_robust` = 近 30 日稳健 z 分数 |

### 3.2 Direct 系方法的目标日外生对齐

多步 Direct 的训练行 t 要预测 `y(t+1..t+H)`，外生特征必须对齐**目标日**而非原点日：

| 机制 | 做法 | 为什么 |
|---|---|---|
| horizon 展开（USMD/USMDR，`use_horizon_exogenous_for_direct: true`） | 生成 `col_h1..col_hH`，行 t 的 `col_h(h) = col(t+h)` | 第 h 个输出模型看到目标日天气/日历 |
| horizon-feature melt | 宽表行 t 按 h 从 `*_h{h}` 折叠回基础列名，再加 horizon 索引训练 N×H 长表 | 单模型共享样本，成本 H×→1×；lag/rolling 保持原点值 |
| origin-frozen 排除 | `state_*` 不参与展开，所有 h 共用 `state(t)` | 状态在原点已知，目标期真实状态不可见 |

### 3.3 各方法的特征面差异

| 方法 | 特征特点 |
|---|---|
| USMD/USMDR | lag + rolling + diff + 日历 + 天气（horizon 展开）+ state |
| USMDP | **无 lag**（`enable_lags_features: false`，日历/天气逐点模板）；rolling/diff 关闭 |
| USMR | lag + 日历 + 天气 + state（无 rolling/diff，递归逐步构造） |
| USBR | Direct（shift_1..H）+ Recursive（shift_0）双子模型共享 X，**天气组降级为无天气 control**（共享 X 无法同时满足两子模型的外生时点） |
| ST（USMR） | wrapper 只消费 lag 与 dt_day_of_week，天气/state 列注入但不生效——在 weather/load_state 组中是 no-op 对照 |

---

## 4. 模型与配置分组

### 4.1 四组配置矩阵（每路）

| 分组 | 数量/路 | 构成 | 目的 |
|---|---:|---|---|
| `baseline` | 10 | 6 个 quantile+conformal（lgbm usmd/usmdr/usmr/usmdp/usbr、horizon-feature usmd）+ 4 个 point（ridge/enet/lasso usmd、st usmr） | 无天气基础对照，含概率区间主链（qr 已移除） |
| `add_exogenous_weather_date` | 10 | baseline 同构 + 严格天气信息集 | 天气消融实验组（qr 已移除） |
| `add_decomposition` | 16 | 8 个模型/方法 × {linear, STL7, quadratic, damped}，天气开启、conformal 关闭 | 目标分解消融实验组（qr 已移除） |
| `add_load_state` | 32 | decomposition 同构 + 15 列 origin-frozen 状态 | 负荷状态消融实验组（qr 已移除） |

### 4.2 模型与方法选型理由

| 模型 | 方法 | 为什么这样搭配 |
|---|---|---|
| LightGBM | USMD/USMDR/USMR/USMDP/USBR + horizon-feature | 主力非线性模型；五种方法覆盖 direct/recursive/blend 全谱系对比 |
| Ridge/ElasticNet/Lasso | USMD point + `scale_features: true` | 线性基线；特征量纲差异大必须标准化；Lasso 兼做特征选择 |
| QuantileRegressor | USMD quantile + 标准化 + 关闭时间衰减 | 线性分位数对照；多输出 quantile 不透传 sample_weight |
| SeasonalTemplate | USMR point | NNLS 学 lag 权重的季节模板，是 Naive 的可学习推广；单输出递归避免多输出 wrapper 不兼容 |

### 4.3 训练增强与约束

| 配置 | 取值 | 为什么 |
|---|---|---|
| `quantile_monotone: true` | 全部 84 个 quantile 配置 | 消除 q50 > q90 的 quantile crossing（月频实测曾大量出现） |
| 时间衰减样本权重 | halflife 60 天；多输出 quantile / USBR 显式关闭 | 近期样本更重要；不支持的路径不得"假启用" |
| Conformal（CQR） | baseline/weather 概率组开启，取最近 5 窗 score | 6 fold × ~30 天 ≈ 150 个校准分，超过 min_scores=30 可真实生效 |
| 分解 × conformal | 硬互斥，decomposition 组全部关闭 | 框架约束 |
| 并行 | `window_parallel_workers: 8`，窗口内强制单线程 | 6 fold 并行不超额订阅（本机 8 核） |

---

## 5. 评估协议

| 项 | 约定 | 为什么 |
|---|---|---|
| 主指标 | 每 fold 的 MAPE（eval_mask percentile=5 过滤近零值）+ **Monthly Total MAPE**（月总量相对误差） | 日级精度与业务月总量精度都要看 |
| 汇总 | 6 个 fold 取 **median**，不用 mean | 单 fold MAPE 爆炸不拖垮结论 |
| 对照 | 每 fold 附带 seasonal naive（昨日同时刻）同 mask 评分 | 没有胜过 naive 的模型不具备上线价值 |
| 判定标准 | A/B 各 31 窗、ΔMAPE ≥ 0.5pp 且两路同向才推广 | 防单路过拟合与噪声决策 |
| 产物 | `results_test/<scenario>/<setting>/`：test_scores_df.csv、cv_plot_df.csv、test_prediction.png、window_plots/ | 每个 setting 一个目录，结果与配置一一对应 |

---

## 6. 当前状态与注意

- 自然月版本（124 配置）已完成 P0/P1 修复（严格天气、目标日外生对齐、origin-frozen 状态、quantile 单调化、setting 命名），代表链路冒烟通过，但**全量回测尚未运行**，`results_test` 下暂无正式结果。
- horizon=1 的月频实验已证明 LightGBM 类在小样本上劣于 naive；日频 120 天训练窗样本量充足，结论可能不同，以回测为准。
- ST 在 weather/load_state 组是 no-op 对照，汇总时不要计入"天气/状态模型"。
- 概率区间是**逐日**的；逐日 P10/P90 直接求和不等于月总量区间（日误差相关），月总量区间需另行评估。
