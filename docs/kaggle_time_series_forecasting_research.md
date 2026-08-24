# Kaggle 时间序列预测调研：面向 tsproj_ml 的能力提升建议

> 文档性质：外部技术调研与项目能力差距分析  
> 调研日期：2026-08-22；更新：2026-08-24——多步预测四项语义修复、时间序列分解重构均已落地；概率预测重构 Phase 0–4 已全部完成：quantile/PI 分离、概率指标闭环、prequential/as-of CQR、`ProbabilisticSpec`/typed bundle/`TargetTransformPipeline`、9 方法族迁移与唯一 `ForecastModelBundle` 持久化（证据见 `docs/time_series_probabilistic_forecasting_redesign.md` §15）。
> 调研范围：Kaggle 竞赛、获奖方案、Notebook、数据集和讨论；重点关注电力负荷、能源时序、多步预测、时间序列分解、概率预测、天气外生变量
> 本文不构成实施授权；概率重构代码侧已收口，当前欠账按优先级为：baseline suite/skill score（本文 P0）、受 MSMD/MSMDR 修复影响的存量结果重跑、safe-lag/global panel 正式 A/B 消融，再按单配置消融规则逐项实施。

## 1. 结论先行

**没有一个 Kaggle 方案适合直接移植到 `tsproj_ml`。** 项目现有主链已经覆盖多步 Direct/Recursive/DirRec/Blend、分位数预测、CQR、严格天气信息集、滑窗回测和多种树模型；Kaggle 的主要增量不是“再接一个模型”，而是以下三项：

1. **概率预测评估与因果校准闭环已于 Phase 0–1 落地**：现已输出 pinball loss、区间覆盖率、平均/归一化区间宽度、Winkler、quantile crossing、coverage gap 和 horizon-wise 指标；CQR 按 `label_available_at` 只选完整历史窗口，并以 prequential 方式评价当前窗口。M5 Uncertainty 直接以加权缩放 pinball loss 评估概率分布；其高排名方案也按多个相邻时间折验证方向一致性，而不是只看中位数点预测误差。[4][16][23]
2. **把回测从“窗口总分”扩展为“预测步 × 场景 × 基线”的 skill 分析**：当前 `test_scores_df.csv` 只给整窗 R2/MSE/RMSE/MAE/MAPE 和昨日同时刻 naive，容易掩盖远 horizon 退化、节假日/突变日失效和区间失准。
3. **新增少数有明确信息增量的特征族**：严格 trailing 的 EWM、天气 forecast vintage/`hours_ahead`、天气相对近期基准的 anomaly、冷/热度时与交互项；每项都必须遵循现有 A/B 两路、相同 31 窗、median MAPE 的消融规则。Enefit 获奖方案反复使用天气预报生成时刻、`hours_ahead`、历史目标变换和天气相对近期均值，而不是简单把“实测天气”并入未来。[18][19][26]

专项代码复核发现的四项多步问题已在当前实现中修复：MSMD/MSMDR 目标从 `shift_1` 开始；Direct point/quantile 输出长度必须精确等于 horizon；USMDP 可通过 `align_direct_features_to_target=true` 启用 `min(lags) >= horizon` 的多步 safe-lag；global panel 的目标 shift、lag 和顺序型高级特征按 `series_id` 分组。现有 MSMD/MSMDR 旧结果仍是修复前口径，必须重跑；多步 safe-lag 与 global panel 也只是工程能力就位，尚未完成 A/B 消融。时间序列分解重构已全部收口（2026-08-24）：linear+damped lookback、MSTL robust、离线周期指标三处语义缺陷修复并验收；架构迁移到 `decomposition/` 包（spec/pipeline/registry）；部署持久化升级为 v2 内嵌式 bundle（model+scaler+pipeline 单文件）；残差频谱诊断工具就位。多步预测详见 §6，分解详见 §7，能力矩阵见 §11。

深度模型、层级协调和真正的跨序列全局模型有研究价值，但应排在评估闭环之后。Enefit 第一名的 XGBoost+GRU 比单模型更好，但 1D-CNN、Transformer 和更长多日 GRU 输入没有奏效；M5 的 DeepAR 主要价值是补充分布形状和层级不确定性，并非证明深度模型普遍优于 GBDT。[16][18]

## 2. 调研样本与对本项目的参考价值

| 来源 | 核心问题 | 可参考价值 | 不能直接照搬的原因 |
|---|---|---:|---|
| Enefit Prosumer | 小时级消费/生产、48h 天气预报、在线揭示目标 | ★★★★ | 目标含光伏生产，测试采用 time-series API，实体和信息延迟与 AIDC 不同 |
| ASHRAE Great Energy Predictor III | 千余建筑、多站点、多计量类型、历史天气 | ★★★ | 更接近反事实建筑能耗建模，不是严格的未来多步调度预测；实体泄漏和随机切分风险高 |
| M5 Uncertainty | 大规模层级时序、多个分位数、加权缩放 pinball | ★★★★ | 零膨胀零售计数与连续负荷不同，但概率评估和层级方法高度可复用 |
| Panama Load Forecasting | 小时级负荷、天气、节假日、学校日历、官方预调度预测 | ★★★★ | 周预测前有 72h 不可见 gap，需要按本项目业务时延重新配置 |
| PJM Hourly Load | 十年以上小时负荷、季节/天气/naive 对照 | ★★★ | 数据只有区域级负荷，天气和实体维度较少，适合 EDA/基线而非完整生产对照 |
| Spain Energy + Weather | 负荷、发电、价格、天气和 TSO 预测 | ★★★ | 可用于外部 benchmark，但不应把市场价格特征无条件迁入 AIDC 负荷场景 |
| Store Item / Store Sales | 大规模多序列、lag/rolling/EWM、时间验证 | ★★ | 零售促销和库存机制与电力负荷不同，主要复用表格时序实现方式 |

Enefit 明确区分 forecast weather 与 historical weather，并用 `data_block_id` 表达“同一预测时刻实际可见的数据”，`hours_ahead` 表达预报生成时刻到目标时刻的 lead；这与项目现有 `available_at` 和 strict weather contract 高度同构。[1][26]

ASHRAE 数据包含三年小时级计量、建筑元数据和邻近气象站天气，说明跨实体全局建模、实体类别和天气交互是常见能源预测范式。[2]

Panama 数据还给出了 14 组训练/测试对，并要求周预测前保留 72 小时不可见区间，是验证“预测原点信息延迟”的直接样本。[9]

## 3. EDA：从画折线升级为可建模审计

### 3.1 数据完整性与采样规则

高价值 EDA 首先不是相关系数，而是验证时间轴和信息质量：

- 每条序列的起止时间、预期点数、缺失点、重复点、非整槽时间戳和最长连续缺口；
- 合法零、通信零、停表/卡表、异常后补记等不同零值机制；
- 多实体 coverage heatmap：行是 meter/route/room，列是时间，分别画 missing 和 zero；
- 频率变化、夏令时/时区、聚合前后能量守恒；
- 外生数据在每个 forecast origin 的可见性与覆盖率。

伦敦智能电表 Notebook 用 meter×time heatmap 检查完整性、插值前后缺口和零值分布；PJM 数据卡也提醒区域组成随年份变化，不能默认所有区域覆盖相同时间段。[10][20] 这部分可沉淀为独立 `data_process/time_series_eda.py`，输出 CSV 审计和自包含 HTML，而不是只留 Notebook 图。

### 3.2 周期、趋势和多尺度 profile

建议标准化以下图表与统计：

1. 原始、日/周/月聚合曲线；
2. `hour × weekday`、`hour × month` 的均值/中位数/P10/P90 heatmap；
3. 工作日/周末/节假日、季节、route/room 的典型日 profile；
4. ACF/PACF 与候选 lag 对照；
5. 一阶差分、日差分、周差分的分布和自相关；
6. rolling level、rolling volatility、峰谷时刻和 ramp 分布；
7. 目标与天气的线性相关、分箱曲线和交互热图。

PJM Notebook 同时做 naive、ACF/PACF、天气对照和误差残差分析；Spain 能源 Notebook 将 decomposition、stationarity、ACF/PACF、cross-correlation、naive 和多类模型放在同一流程中。[8][21] 需要注意：PJM 示例中温度与负荷的整体线性相关不高，但极端温度对应峰值，这说明天气关系常是 U 形、阈值型或分季节变化，单一 Pearson 相关不足以判定天气无效。[8]

### 3.3 预测误差 EDA

当前项目已有 per-window 图，但还应新增结构化误差切片：

- `horizon_step`：第 1 步到第 H 步的 MAE/MAPE/pinball/coverage；
- `hour_of_day`、`day_of_week`、节假日、月份；
- 负荷水平分位数、ramp 强度、事件类型、天气区间；
- forecast weather `hours_ahead`；
- route/room/series；
- 区间是否覆盖、区间宽度和 quantile crossing。

这种切片能回答“模型总体 median MAPE 没变，但是否改善远 horizon、峰值或高风险窗口”，比继续堆特征更有决策价值。

## 4. 特征工程：可复用项与项目映射

### 4.1 已覆盖，不应重复造轮子

项目当前已经覆盖：

- datetime/cyclical、lag、rolling、expanding、diff；
- Direct 目标日外生对齐和 horizon feature；
- 天气历史/backtest/future 三路与 `available_at` 契约；
- 自定义外生特征注册表；
- 目标分解、时间衰减、特征选择和 categorical 支持。

因此 Kaggle 常见的“增加 weekday/month/lag/rolling + LightGBM”本身不是新增能力。Store Item Notebook 的核心是 lag、rolling、EWM 和时间留出集；其中 lag/rolling 项目已有，真正尚未接入主链的是 EWM。[7]

### 4.2 建议试验：严格 trailing EWM

EWM 对缓慢漂移和近期状态变化的响应比固定 rolling mean 更平滑，适合负荷水平和波动率特征。关键约束是**先 shift，再 EWM**，确保行 `t` 不消费 `y_t`：

```python
def add_trailing_ewm(
    df: pd.DataFrame,
    group_cols: list[str],
    target_col: str,
    lags: list[int],
    alphas: list[float],
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, sort=False)[target_col]
    for lag in lags:
        for alpha in alphas:
            name = f"{target_col}_lag_{lag}_ewm_a{alpha:g}"
            df[name] = grouped.transform(
                lambda series: series.shift(lag).ewm(alpha=alpha, adjust=False).mean()
            )
    return df
```

Kaggle 示例对每个 store/item 先 shift，再计算多个 alpha 的 EWM；其代码还会给 lag/rolling 加随机噪声，但随机噪声只是一种竞赛正则化技巧，不建议进入确定性的生产特征链。[7]

建议首轮只测 2–3 个与业务周期对应的 alpha，不生成大矩阵：

- 高频 daily：`lag=1` 或最短安全 lag，alpha 对应约 1 日、3 日有效记忆；
- 日频/月度：alpha 对应约 7 日、30 日有效记忆；
- 先做 route_A 单配置 31 窗；只有达到门槛，再补 route_B 正式判定。

### 4.3 建议试验：forecast-vintage 天气特征

Enefit 第一名用 `origin_datetime + hours_ahead` 重建天气目标时刻，并把天气按县聚合；第五名保留 `hours_ahead`，同时使用 forecast local weather 相对过去一周均值的差值、太阳辐射、温度、露点和天气预报 lead。[18][19] 对本项目更合理的派生项是：

- `weather_lead_steps` / `hours_ahead`；
- forecast temperature − 最近可得 actual temperature；
- forecast temperature − 同一 forecast vintage 的近期均值；
- heating/cooling degree hour；
- apparent temperature；
- 温度×湿度、温度×hour、天气×工作日/节假日；
- 24h/7d forecast-weather rolling summary；
- 对光伏/发电目标才使用 capacity×radiation 一类物理交互；纯负荷场景不照搬。

项目的严格天气契约已经要求 backtest 使用 forecast/proxy、future 禁止 actual/mixed，并验证 `available_at <= forecast_origin`，这比多数 Kaggle Notebook 更严谨（`utils/weather_contract.py:12-113`）。因此改动重点应是**派生 forecast-vintage 特征**，不是放松信息集。

### 4.4 实体尺度归一化与相对目标

Enefit 第一名尝试 `(target-target_shift2)/installed_capacity` 和 `target/installed_capacity`，第五名分别按 installed capacity 或近期同槽位均值归一化生产/消费目标；两者的共同思想是把“规模”与“形状/偏差”拆开。[18][19] 对 AIDC 可对应为：

- 若有稳定容量或活跃设备数：预测 `load / active_capacity`，再还原；
- 若规模变量不可在预测原点可靠获得，不采用；
- 若使用近期基线归一化，基线必须只用 cutoff 前数据，并在 train/CV/final 三条路径完全一致。

这比无条件 log/scale 更有物理意义，但会改变目标变换链，属于高风险实验；应晚于概率指标和 EWM。

## 5. 模型方案：真实判断

### 5.1 GBDT 仍应保持主链

Enefit 第一名以 4 个 XGBoost + 2 个 GRU 融合获胜；第五名使用消费/生产分开的 LightGBM VotingRegressor，并通过三折时间 CV、独立末段验证和在线重训筛选特征。[18][19] 这说明：

- GBDT 对 lag、天气、实体类别和大量稀疏派生特征仍然强；
- 模型之间的收益主要来自**误差结构差异**，不是同模型换 seed；
- 按目标机制分模型可能有效，但前提是预测时能因果识别机制。

项目现有 LightGBM/XGBoost/CatBoost 主链不落后。短期不建议引入新的常规树模型；应先改善评估和特征。

### 5.2 深度时序模型只作为独立研究支线

M5 高排名方案用负二项 DeepAR 处理间歇性低层序列，用 Gaussian DeepAR 给聚合层 LightGBM 中位数提供 quantile/median 比率，并用最近三个 28 天窗口验证；同 seed ensemble 没有明显收益。[16] Enefit 第一名的 GRU 提供了异构增益，但 Transformer、1D-CNN 和多日输入没有成功。[18]

因此建议：

- 不把 GRU/DeepAR 塞进 `ModelFactory` 的 sklearn 回归器接口；
- 若启动，建立独立 panel forecasting 支线，输入/输出、保存和评估与主链解耦；
- 先让同一概率指标套件可以公平比较 GBDT quantile、CQR、DeepAR/GRU；
- 只有在 A/B 两路和完整窗口上稳定优于 GBDT，才考虑生产接入。

### 5.3 层级预测与 reconciliation

M5 的核心价值是同时评估多个聚合层；高排名方案按层级的数据分布分别建模，并指出 bottom-up 只能自然得到聚合中位数，完整分位数还需要分布模型或协调方法。[16][23] 对本项目可映射为：

- room → route → site total；
- A/B 分路 → 合计；
- 设备级 → 房间级 → 园区级。

只有在这些层级目标都真实存在、且业务要求加总一致时才值得实现。当前不应为了“先进”预建 reconciliation 空壳。

## 6. 多步预测专项研究：Kaggle 策略、项目映射与缺陷审计

### 6.1 研究问题与结论

本主题回答三个问题：Kaggle 常见的多步预测如何组织训练样本和未来信息；这些方法与项目 9 种策略的实质差别是什么；项目当前属于“方法局限”的部分和属于“代码缺陷”的部分分别有哪些。

结论分三层：

1. **策略覆盖面**：项目比多数 Kaggle Notebook 更完整，已经同时具备 pointwise direct、MIMO Direct、Recursive、DirRec、Direct+Recursive Blend，以及单变量/多变量对应路径；不存在“多步策略种类太少”的问题。
2. **缺陷处置**：MSMD/MSMDR 的 Direct 目标 off-by-one 已修复为 `shift_1..shift_H`；Direct 输出长度已改为精确匹配、异常即 RAISE。受影响的旧 MSMD/MSMDR 结果仍须重跑，不能与新口径混用。
3. **能力补齐**：USMDP 已开放多步 safe-lag，契约为 `align_direct_features_to_target=true`、lag 开启且 `min(lags) >= horizon`；global panel 的目标 shift、单/多变量 lag、horizon 外生 shift 与顺序型高级特征均按 `series_id` 分组。两项尚未经过现役场景 A/B 消融，不能把“代码可用”写成“效果已验证”。

### 6.2 Kaggle 多步预测的三类典型路径

| 路径 | Kaggle 代表 | 样本组织 | 未来目标历史的处理 | 与本项目最接近的方法 |
|---|---|---|---|---|
| Pooled pointwise direct | Store Item LightGBM | 每个实体×未来时间点一行，共享一个模型 | 只用保证可得的长 lag，不回填预测 | USMDP |
| Global recursive | M5 LightGBM | 按商品 `id` 组织长表，共享一个模型 | 逐日预测并写回，下一日重新计算 lag/rolling | USMR/MSMR |
| Vector/MIMO direct | Enefit GRU | 一个输入窗口直接输出 24 小时向量 | 各 horizon 共享序列表征，不逐步回填 | USMD/MSMD、horizon-feature |

Store Item 把未来日期展开为逐行样本，按 store/item 分组生成 lag、rolling 和 EWM；其长 lag 大于预测跨度，使整个未来区间都只引用 cutoff 前真实历史。[7]

M5 LightGBM 按 `id` 分组生成 lag7/lag28 与 rolling 特征，并在 28 天推理循环中逐日写回 `sales` 后重新计算下一日特征，是标准 global recursive。[29]

Enefit XGBoost 使用预测时可见的 revealed-target lag 和 forecast weather 做逐小时 pointwise 预测；第一名同时训练 24 输入→24 输出的 GRU，与 XGBoost 形成异构融合。[18][26]

### 6.3 本项目多步策略的真实语义

| 方法 | 训练目标 | 推理方式 | 优点 | 当前判断 |
|---|---|---|---|---|
| USMDP | 行 `t` 预测 `y_t` (`shift_0`) | 所有未来行一次 pointwise predict | 单模型、成本低、天然消费逐时外生 | 默认不生成目标 lag；开启 align 后支持 H>1 safe-lag |
| USMD | 原点 `t` 预测 `y_{t+1..t+H}` | 一次输出 H 步 | 无递归误差累积 | 语义正确；默认 H 个独立模型 |
| USMR | 未来行 `t+h` 预测本行目标 | 逐步预测并写回 | 可使用短 lag、模型仅 1 个 | 有固有误差累积 |
| USMDR | 原点预测下一 block | block 内 Direct、block 间回填 | 直接/递归折中 | 单变量目标起点正确 |
| MSMD | 多变量原点预测 H 步 | 一次输出 H 步 | 可使用多个内生变量历史 | 已修复为 `shift_1..shift_H` |
| MSMR | 多变量未来行预测本行目标 | 逐步预测目标，其他内生量回填 | 可用多内生历史 | `shift_0` 在此语义下正确 |
| MSMDR | 多变量原点预测下一 block | block Direct、块间回填 | 多变量 DirRec | 已修复为 `shift_1..shift_H` |
| USBR/MSBR | Direct `shift_1..H` + Recursive `shift_0` | 两路预测后融合 | 组合两种误差结构 | 目标起点正确；收益需消融 |

判断 `shift_0` 是否正确不能只看列名，必须看“特征行代表预测原点还是目标时点”：Recursive/Pointwise 的特征行就是目标时点，`shift_0` 正确；Direct/DirRec 的特征行是预测原点，第一个目标必须是 `shift_1`。

### 6.4 Kaggle 与项目的关键差异

#### 6.4.1 Pointwise direct 的安全 lag

Kaggle 的 pointwise direct 并不等于“不要 lag”。它通过保证最短 lag 不小于 horizon，使每个未来行都只引用 cutoff 前历史：

$$
\hat y_{t+h}=f(y_{t+h-L_1},\ldots,x_{t+h},h,\text{entity}),\qquad \min(L_i)\ge H
$$

例如预测 96 个 15min 点并使用 lag96：第 96 个未来点的 lag96 恰好指向最后一个历史点；整个 horizon 无需递归，也没有未来真值泄漏。项目当前已开放这条路径：USMDP 设置 `align_direct_features_to_target=true` 和 `enable_lags_features=true`，并配置非空正整数 lag；运行时、预测时和配置 dry-run 共同校验 `min(lags) >= horizon`。未开启 align 的 USMDP 保持原行为，不生成目标 lag。

#### 6.4.2 Recursive 的实体分组

Kaggle global recursive 的 lag、rolling 和回填均按实体 ID 分组。项目现已为 global panel 建立实体边界：target shift、单/多变量 lag、horizon 外生 shift、rolling、expanding、diff、pct_change 和 time-since 都在 `series_id_feature` 内计算；缺少或含空值的 series ID 直接 RAISE，lag 可用性按最短实体长度判断。当前现役 YAML 仍未启用 global，因此该修复消除了代码层串值风险，但尚无真实 panel 精度结论。

#### 6.4.3 Direct multi-output 的参数共享

Enefit GRU 的 24 个输出共享序列表征；项目默认 `DirectMultiOutputRegressor` 为每个 horizon 训练独立回归器。项目的 `horizon_feature` 可改成一个共享模型，`regressor_chain` 可让后续 horizon 消费前序输出，但三者是不同偏差—方差—成本取舍，不存在一个对所有频率都更优的默认答案。

#### 6.4.4 概率轨迹的一致性

项目按 quantile、horizon 独立训练，再做逐时点 quantile 单调化和边际 CQR；它能表达每个未来时点的边际 P10/P50/P90，但不能直接表达整条轨迹的联合概率、月总量分布或连续超限概率。M5 的 DeepAR 通过条件分布采样得到路径级 quantile，能力层级不同。[16]

### 6.5 已修复缺陷与影响范围

#### 6.5.1 【已修复 2026-08-24】MSMD/MSMDR 目标 off-by-one

原问题是 MSMD/MSMDR 调用 `extend_direct_multi_step_targets` 时沿用默认 `start_step=0`，生成 `y_shift_0..H-1`；forecast 却把输出第 1 项写到未来第 1 点。当前两条多变量 Direct/DirRec 分支均显式传入 `start_step=1`，与单变量 USMD/USMDR 对称。

以 `y=[10,11,12,13,14,15]`、`H=3` 为例，修复后的目标契约为：

| 方法 | 当前目标列 | 第 1 行目标值 | 正确性 |
|---|---|---:|---|
| USMD | `shift_1,shift_2,shift_3` | 11,12,13 | 正确 |
| MSMD | `shift_1,shift_2,shift_3` | 11,12,13 | 正确 |
| USMDR | `shift_1,shift_2,shift_3` | 11,12,13 | 正确 |
| MSMDR | `shift_1,shift_2,shift_3` | 11,12,13 | 正确 |

历史影响范围不变：修复前训练的 MSMD/MSMDR point、quantile、conformal 底层预测和 MSMD `horizon_feature` 都是旧目标口径。当前代码修复不会自动修正已有 CSV、模型和图，相关结果必须重跑；USMD、USMR、USMDR、USMDP、MSMR 和 USBR/MSBR 不因该缺陷作废。

#### 6.5.2 【已修复 2026-08-24】global panel 跨序列串值

原实现只保留 `series_id` 作为特征，lag 与 target 仍使用整表 `shift`；连续存放 `A=[1,2,3]`、`B=[100,200,300]` 时会出现 A 尾行读取 B 目标、B 首行读取 A 尾值。当前统一通过 `series_id_feature` 建立分组边界，覆盖 target shift、单变量 lag、多变量 lag、horizon 外生 shift、rolling、expanding、diff、pct_change 和 time-since；global 模式缺少或包含空 `series_id` 时直接 RAISE，lag 支撑长度按最短实体计算。

当前 YAML 仍未启用 global，因此旧现役结果没有受到串值污染；修复只解除工程阻塞，不代表 global panel 已经优于单序列模型，仍须基于真实 panel 数据做独立消融。

#### 6.5.3 【已修复 2026-08-24】Direct 输出长度异常被静默掩盖

原实现会在模型输出少于请求 horizon 时使用 `np.pad(..., mode="edge")` 复制最后一个预测，过长时则截断；point 与 quantile 都可能把模型版本、target schema 或动态 horizon 错误伪装成完整预测。当前统一要求每个 point 输出和每个 quantile 输出的长度精确等于 `len(df_future)`：过短或过长都立即 RAISE，并在错误中报告 expected/got，不再补齐或截断。

### 6.6 设计限制：不能与 bug 混为一谈

| 限制 | 原因 | 项目已有缓解 | 判断 |
|---|---|---|---|
| Recursive 误差累积 | 训练消费真值，推理消费自身预测 | DirRec、Blend、auxiliary | 固有限制 |
| Direct horizon 独立 | H 个模型不共享参数和输出协方差 | horizon-feature、regressor_chain | 设计取舍 |
| Quantile/trajectory 非联合 | 每个 q、h 独立训练 | monotone、CQR 仅改善边际 | 能力边界 |
| horizon-feature `N×H` 成本 | 高频大 H 重复展开样本 | 仅在低频/小 H 使用 | 适用范围限制 |
| 多变量递归外生内生量未知 | 非目标内生变量未来不可得 | persistence、auxiliary | 信息集约束 |

### 6.7 测试证据与剩余验证边界

当前新增三组回归测试：

- `tests/test_multistep_target_alignment.py`：断言 USMD↔MSMD、USMDR↔MSMDR 的 Direct 目标均从下一时点开始，同时覆盖 point/quantile 输出过短与 point 输出过长的 fail-fast；
- `tests/test_usmdp_safe_lag.py`：断言 H=3、lag=[3,4] 的未来三行只读取 cutoff 前历史，并拒绝 `min(lags) < horizon`；
- `tests/test_global_panel_feature_boundaries.py`：用 A/B 两条量级差异序列验证 target、lag、horizon 外生和顺序型高级特征在实体边界重置，并验证缺少 series ID 时失败。

这些测试证明时间语义与边界契约已接线；它们不替代真实模型重跑，也不证明 safe-lag/global panel 有精度收益。MSMD/MSMDR 仍须重跑受影响配置，USMDP safe-lag 和 global panel 仍须按 A/B 同窗标准消融。

### 6.8 处置状态与实验影响

1. ✅ MSMD/MSMDR Direct 目标已改为 `shift_1..shift_H`，MSMR 的 `shift_0` 保持不变；
2. ✅ Direct point/quantile 输出已改为精确长度契约，禁止 edge padding 和静默截断；
3. ✅ USMDP 多步 safe-lag 已开放，配置契约为 `align=true`、lag 开启且 `min(lags) >= horizon`；
4. ✅ global panel 的 target、lag、horizon 外生和顺序型高级特征已按 `series_id` 分组；
5. ⏳ 修复前的 MSMD/MSMDR 结果仍须重跑，不能与新口径混合比较；
6. ⏳ safe-lag/global panel 只完成工程接线，尚未做正式 A/B 消融；
7. ⏳ horizon-wise MAPE/MAE/pinball/coverage 仍待补齐，用于判断远 horizon 和概率区间表现。

## 7. 时间序列分解专项研究：Kaggle 实践、因果边界与项目审计

### 7.1 研究范围与结论

时间序列分解在 Kaggle 中至少有五种不同用途：观察趋势/季节性；显式构造趋势/季节特征；先预测结构分量再预测残差；把 OOF 分解预测作为二阶段特征；让模型内部学习可解释分解。若进一步按输入到输出的建模关系划分，除 Trend、Seasonality、Hybrid Models 之外还存在分量级多模型、乘性/变换、非平稳模态分解和端到端 basis 分解等模式。它们的泄漏边界和工程成本完全不同，不能笼统归为“使用 STL”。

本项目当前实现的核心路线是：在每个可见训练窗内拟合确定性的趋势/季节分量，现有 GBDT/线性模型只预测剩余项，最后把确定分量加回 point 和全部 quantile。该路线与 Kaggle Learn 的 boosted hybrid 高度同构，而且当前 CV 接线比多数公开 Notebook 更严格。[30][33]

结论：

1. **主架构正确**：CV 先切原始电平、再仅在训练窗拟合 decomposer；final forecast 才用全部已知历史；测试标签保持原始电平。
2. **不是精度结论**：重构、持久化和无泄漏测试通过，只证明接线正确，不证明 linear/STL 一定优于 `none`。
3. **已修复（原现役缺陷）**：linear+damped 忽略 `decomposition_trend_lookback` 已于 2026-08-23 修复——damped 近期斜率改为从观测 y 的 lookback 段估计（对全局多项式拟合线取 lookback 数学上无效，属实施偏差，见 §7.7.1 与设计文档 §15.7）；当前仅 2 个 damped 配置且均未运行，无既有结果作废。
4. **已修复（原潜伏缺陷）**：MSTL 已通过 `stl_kwargs={'robust': ...}` 透传 `decomposition_robust` 并补 spy 接线测试；当前仍无 MSTL YAML。
5. **已修复（原诊断缺陷）**：离线周期工具已允许恰好两个完整周期（`2*period > n`，与主链一致），新增标准 `stl_seasonal_strength`/`stl_trend_strength` 输出，旧 `stl_seasonal_ratio` 保留并标注 legacy。

### 7.2 Kaggle 分解实践的五类层级

| 层级 | Kaggle 做法 | 是否直接改变预测目标 | 对本项目价值 | 主要风险 |
|---|---|---:|---:|---|
| EDA 分解 | classical decomposition/STL 展示 trend/seasonal/resid | 否 | ★★★ | 全序列可看，但不能把结果直接喂模型 |
| 显式季节特征 | 时间指标、Fourier、趋势基函数 | 否 | ★★★ | 周期选错、特征冗余 |
| Residual hybrid | 线性趋势模型 + XGBoost 残差模型 | 是 | ★★★★ | 必须只用训练期拟合第一阶段 |
| OOF 分解预测特征 | MSTL/FFT/ETS 的窗口外预测作为 CatBoost 特征 | 间接 | ★★★★ | 实现复杂、必须生成训练期 OOF 特征 |
| Learned decomposition | N-BEATS trend/seasonality basis | 是 | ★★ | 独立深度管线、成本高、解释未必稳定 |

Kaggle 官方课程把 Trend、Seasonality、Hybrid Models 分成独立课程：趋势用时间基函数或移动平均刻画，季节性用指标/Fourier 表达。[30][31][32]

Hybrid Models 再用第二模型学习第一模型未解释的残差。[33]

通用 STL Notebook 主要解释如何把序列分成 trend、seasonal 和 residual，适合 EDA 与候选周期理解；它本身不提供生产级 forecast-origin 契约。[35]

N-BEATS Notebook 展示模型内部的趋势/季节 basis 分解和概率预测，但它需要独立神经网络训练、窗口化和模型保存体系，不适合直接塞进当前 sklearn 风格 `ModelFactory`。[37]

### 7.3 EDA 分解与预测分解的因果边界

#### 7.3.1 仅用于 EDA：完整序列分解可以接受

电价预测 Notebook 在 EDA 阶段对完整 `price actual` 和其对数做 `seasonal_decompose`，用途是观察趋势、季节和残差，后续建模并未把这些完整序列分量作为未来特征。[21] 这种用法可以查看完整历史或离线分析区间，但结论只能是“候选周期/结构存在”，不能直接等同于可部署预测收益。

#### 7.3.2 反面案例：完整目标先分解、再切 train/validation/test

伦敦电力负荷 Notebook 先在完整 `AggregateLoad` 上计算年度、日度和周度 decomposition，把 `dailyTrend/dailySeasonal/dailyResid/weekly*` 加入特征，之后才执行 70%/20%/10% 时间切分。[20] 由于 validation/test 的真实目标已经参与分解，这些特征包含未来目标信息；其结果不能作为项目接入“分解特征”的证据。

#### 7.3.3 可复用做法：训练期趋势 + 训练残差

Kaggle Hybrid Models 先按时间切 train/test，只在 train 上拟合线性/二次趋势；然后用 `y_train - trend_fit` 训练 XGBoost，最终预测为 `trend_pred + residual_pred`。[33] 这与项目的 `TargetDecomposer + 原模型残差预测 + restore` 是同一类 boosted hybrid。

#### 7.3.4 更强但更复杂：窗口 OOF 分解预测作为特征

MSTL+CatBoost 4th-place Notebook 对训练期做多窗口时序交叉验证：每折只分解该折训练段，用 FFT 外推季节分量、ETS 外推趋势，并把 out-of-sample `TS_Pred` 及分量作为最终 CatBoost 的输入。[34] 关键价值不在“用了 MSTL”，而在训练期分解预测是 OOF；若用全训练拟合值代替 OOF 特征，CatBoost 会看到过于理想的第一阶段结果。

### 7.4 Trend/Seasonality/Hybrid 之外的其他实现模式

#### 7.4.1 分量级多模型分解

通用分解 Notebook 把 SMA/EMA、经典移动平均、STL、FFT、Wavelet、Prophet、动态谐波回归、ICA 和 BSTS 放在同一方法谱系中。[36] 更激进的实现会先把序列分成若干频带或模态，再为每个分量训练模型并合并；Kaggle 上存在 VMD-CNN-BiLSTM-Attention 一类样例。[42] 这条路线训练成本约随分量数线性增加，还要处理端点效应、分量数稳定性和未来外推合法性，不适合作为当前主链。

#### 7.4.2 确定性分量预测，而不是把确定分量从目标中移除

MSTL+CatBoost 方案不是让 CatBoost 直接预测 `y - deterministic_component`，而是把趋势/季节的窗口外预测作为普通特征，再预测原始目标。[34] 项目如果未来选择这条路，训练期必须使用 OOF 分解预测；当前 residual hybrid 更简单，先不引入。

#### 7.4.3 变换域与乘性分解

电价 Notebook 在 EDA 中对原始价格和对数价格分别做 classical decomposition，说明 `log(y)` 后再加性分解常用于把乘法关系改写为加法关系。[21] Prophet 也可直接配置 `seasonality_mode='multiplicative'`，使季节项按趋势电平成比例缩放。[39] 项目当前只支持加性目标分解；`log1p/yeo-johnson` 属于目标 scaler 而不是当前 `TargetDecomposer` 的一部分，因此不能把“已有目标 scaler”误报为已实现乘性 STL/MSTL。

#### 7.4.4 数据质量用途：STL 填补与残差异常检测

Kaggle 上也把 STL 用作缺失填补和异常检测工具，而不是直接预测器：例如 STL imputation 与 STL residual 异常检测。[40][41] 这类用途与项目的 `fill_method_backtest.py`、`outlier_process.py` 属于相邻能力；若引入，必须保持因果边界，不能把含未来信息的填补值用于训练标签。

#### 7.4.5 端到端模型内部分解

Kaggle 的 DLinear 实现在模型内部先用 moving-average decomposition layer 得到 residual/seasonal 与 trend，再由两个线性层分别外推并相加。[38] 它不是训练前的 STL/MSTL 预处理，而是神经网络结构的一部分；属于独立深度学习管线，不应混入当前 sklearn 风格 `ModelFactory`。

### 7.5 当前项目 TargetDecomposer 架构

项目使用加性确定分量：

$$
y_t = d_t + r_t,\qquad d_t = trend_t + \sum_k seasonal_{k,t}
$$

模型训练目标为 $r_t$，预测还原为：

$$
\hat y_{t+h}^{(q)} = \hat r_{t+h}^{(q)} + \hat d_{t+h}
$$

| `decomposition_method` | 历史分解 | 未来确定分量 | 模型实际预测 |
|---|---|---|---|
| `none` | 不分解 | 0 | 原始电平 |
| `linear` | 一次/二次多项式趋势 | polynomial 或 damped 趋势 | `y - trend` |
| `stl` | 单周期 STL | 多项式/阻尼趋势 + 最近若干周期的相位均值模板 | `y - trend - seasonal` |
| `mstl` | 多周期 MSTL | 趋势 + 各周期相位模板之和 | `y - trend - Σseasonal` |

实现位置为 `features/TargetDecomposition.py:11-246`。它要求时间戳严格递增、等间隔、目标有限；STL/MSTL 每个周期至少覆盖两个完整周期。

#### 7.5.1 滑窗测试

`models/ModelTesting.py:149-209` 先切原始 train/test，再只对 `df_history_train` 执行 `TargetDecomposer.fit_transform`；测试真实值保持原始电平。Forecaster 在评分前加回该窗口确定分量。因此改变测试段值不应改变训练窗分解参数。

#### 7.5.2 最终预测

`main.py:830-857` 在全部已知历史上拟合 final decomposer，模型训练残差；`models/ModelForecasting.py:1258-1270` 给 point 和所有 quantile 加回同一确定分量。确定量平移保持条件分位数顺序和间距。

#### 7.5.3 持久化与组合约束

`models/ModelTraining.py` 现只写一个 `model.pkl`，其内容为 schema v1 `ForecastModelBundle`：模型（point 或 `ProbabilisticModelBundle`）、feature scaler、`TargetTransformPipeline`、selected features 与输入 schema 同构持久化；`probabilistic_schema.json` 仅保存可读 metadata。分解、target scaling、quantile 与 CQR 已通过共享变换栈统一到 target space；USBR/MSBR blend 与分解仍因两条分量路径空间未统一而 fail-fast。

### 7.6 当前实现的优势

1. **因果 CV**：不复制 Kaggle 常见的“完整目标先分解再切窗”泄漏。
2. **复用主模型矩阵**：USMDP/USMD/USMR/USMDR 等仍走原训练与推理，只替换目标空间。
3. **point/quantile 同构还原**：全部分位数加回相同确定分量，属于合法的分布平移。
4. **未来分量显式可解释**：趋势和季节模板可以单独审计，不依赖未来真实目标。
5. **fail-fast 边界**：周期不足、时间不规则、目标含 NaN/inf、未知方法和不支持组合会直接失败。
6. **持久化已覆盖**：测试验证 decomposer 保存/加载后未来 restore 一致。

### 7.7 缺陷处置状态与集成缺口

> 处置状态：7.7.1–7.7.4 全部关闭。7.7.1–7.7.3 于 2026-08-23 修复；7.7.4 的局部 `DecompositionBundle` 于 2026-08-24 完成，随后已由概率重构中的唯一 `ForecastModelBundle` 取代。最新验收事实见 `docs/time_series_probabilistic_forecasting_redesign.md` §15；以下原文保留为审计记录。

#### 7.7.1 【已修复 2026-08-23】linear+damped 忽略 `decomposition_trend_lookback`

修复说明：实际修法与本节原建议不同——对全局多项式拟合线取任何 lookback 窗口重拟合，斜率都相同，lookback 数学上不可能生效。最终实现从**观测 y** 的最近 lookback 段估计 damped 近期斜率；合成序列探针验证 lookback=7 局部斜率 10.0、lookback=40 全窗斜率 1.79，未来确定分量明显分化。测试：`tests/test_target_decomposition.py::test_linear_damped_forecast_consumes_trend_lookback`。该偏差已登记于设计文档 §15.7（偏差 #1），Phase 2 架构迁移时须保持此语义。

STL/MSTL 路径在 `_fit_trend_forecast` 中按 lookback 计算近期斜率；linear 路径却在 `features/TargetDecomposition.py:134-144` 直接对全部趋势做 `np.polyfit`，没有读取 lookback。可执行探针把近期趋势改成陡升，并比较 lookback=3 与 28，未来 10 步确定分量最大差仍为 0。当前 A/B 各有一份 `lgbm_usmdp_mean_damped.yaml`，因此这是现役实验语义缺陷。

#### 7.7.2 【已修复 2026-08-23】MSTL 忽略 `decomposition_robust`

修复说明：MSTL 构造改为 `MSTL(y, periods=..., stl_kwargs={"robust": decomposition_robust})`，与本节原建议一致；spy 测试 `test_mstl_passes_robust_to_statsmodels` 对 robust True/False 两值断言 `stl_kwargs` 接线。当前仍无 MSTL YAML，创建 MSTL 配置的前置条件已满足。

代码读取了 `decomposition_robust`，STL 调用传入 `robust=...`，MSTL 却只调用 `MSTL(y, periods=...)`。当前 statsmodels 的 MSTL 通过 `stl_kwargs={'robust': True}` 控制底层 STL；默认 `_stl_kwargs={}`。当前 YAML 没有启用 MSTL，所以这是启用前必须修复的潜伏缺陷，不影响现役 linear/STL 结果；`tests/test_target_decomposition.py` 当前也没有 MSTL 接线测试。

#### 7.7.3 【已修复 2026-08-23】离线周期诊断与主链边界不一致

修复说明：边界改为 `2*period > len(y)`（与主链一致，恰好两个完整周期可用）；新增标准 `stl_seasonal_strength` / `stl_trend_strength`（FPP 公式 `max(0, 1-Var(R)/Var(S+R))`）；`stl_seasonal_ratio` 保留并标注 legacy。测试：新增 `tests/test_periodicity_analysis.py`（强季节正弦强度>0.5、白噪声∈[0,0.5)、两周期可分解三个用例）。已有报告 CSV 不含新列，需重跑工具才会出现。

`data_process/periodicity_analysis.py:126-135` 用 `period >= len(y)//2` 拒绝 STL，导致恰好两个完整周期时返回 None；主链 `TargetDecomposer` 和配置检查器允许 `2*period == n`。探针已确认同一 24 点、period12 序列在离线工具被拒，而主链可拟合。

该工具还计算：

$$
stl\_seasonal\_ratio = \frac{std(seasonal)}{std(y-seasonal)}
$$

分母 `y-seasonal` 仍包含 trend，不是 STL remainder。标准季节强度应使用：

$$
F_S = \max\left(0,1-\frac{Var(R)}{Var(S+R)}\right)
$$

因此当前 ratio 只能视为自定义描述量，不能据此决定是否启用 STL/MSTL。

#### 7.7.4 【已完成 2026-08-24】保存了 decomposer，但缺少统一预训练加载编排

完成说明（后续状态更新）：局部 `DecompositionBundle` 曾关闭该缺口；当前实现进一步收敛为唯一 `ForecastModelBundle`，selected features、input schema、feature scaler、目标变换栈和 point/quantile model 已全部进入 `model.pkl`。loader 兼容旧 quantile dict，并对未知 schema version fail-fast。

代码能够保存和手动加载 `target_decomposer.pkl`，但全仓没有发现与 `model.pkl`、scaler、selected feature schema 一起自动恢复 decomposer 的统一预测入口。当前 `main.py` 每次 final forecast 都现场训练，因此不受影响；若未来走独立加载预训练模型部署，必须把这些对象作为一个 bundle 同构恢复。

### 7.8 设计限制：不是代码 bug

| 限制 | 影响 | 判断 |
|---|---|---|
| 只支持加性分解 | 振幅随水平增长的乘法季节性表达不足 | 先用现有消融，不立即扩展 |
| 趋势/季节视为确定量 | 分位数只描述 residual 不确定性，忽略结构分量估计误差 | 当前语义清晰，但区间可能偏窄 |
| polynomial 二次外推 | 长 horizon 可能快速发散 | 二次趋势仅作受控实验，damped 更安全 |
| 周期手工配置 | 周期选错会把信号误分到 residual | 由 EDA 提候选、CV 裁决，不自动全局选择 |
| 不做分量级独立建模 | 无法复制 EMD/VMD/N-BEATS component ensemble | 成本高，暂不优先 |
| 只支持加性 `TargetDecomposer` | 不能直接表达乘性季节或 log 后加性 STL/MSTL | 先由 A/B 消融判断业务必要性 |
| 不做 VMD/CEEMDAN/Wavelet 分量级预测 | 对非平稳模态的表达能力不足 | 分量数、端点效应与成本不划算 |
| 不把 OOF 分解预测作为特征 | 与 MSTL+CatBoost 二阶段路线相比表达面较窄 | 需要独立 stacking/OOF 管线 |
| 与 blend 互斥 | Blend 的 Direct/Recursive 分量空间尚未统一 | 保留 fail-fast；CQR/scale 已解除互斥 |

### 7.9 与 Kaggle 方法的差距

| 能力 | Kaggle 可见实践 | 当前项目 | 建议 |
|---|---|---|---|
| EDA 分解 | classical/STL 可视化 | periodicity_analysis + STL | ✅ 已完成（2026-08-23）：标准强度输出 + 两周期边界修正 |
| Residual hybrid | 线性趋势 + XGBoost residual | 已实现，且 CV 更严格 | 保留为主路线 |
| 多季节分解 | MSTL + FFT/ETS | MSTL + phase template | robust 已修复，可进入 96/672 消融 |
| OOF 分解预测特征 | rolling MSTL forecast → CatBoost | 未实现 | 仅在 residual hybrid 不足时研究 |
| Learned decomposition | N-BEATS basis | 未接入 | 独立研究支线，不进 ModelFactory |
| 分量不确定性 | 概率 component/path forecast | deterministic shift | 先补边际概率评估，再考虑 |

### 7.10 推荐修复与消融顺序

1. ✅ 已完成（2026-08-23）：linear+damped 近期斜率改为从观测 y 的 lookback 段估计（对全局多项式拟合线取 lookback 无效，属实施偏差，见设计文档 §15.7）；补齐不同 lookback 产生不同未来分量的测试。
2. ✅ 已完成（2026-08-23）：MSTL 增加 `stl_kwargs={'robust': decomposition_robust}`，robust true/false 接线测试就位；创建 MSTL 配置的前置条件已满足。
3. ✅ 已完成（2026-08-23）：允许恰好两个周期（`2*period <= n`），输出标准 `stl_seasonal_strength/stl_trend_strength`，旧 ratio 保留并标注 legacy。
4. ✅ 已完成并升级（2026-08-24）：`model.pkl` 保存唯一 schema v1 `ForecastModelBundle`，包含 model、feature scaler、`TargetTransformPipeline`、selected features、input schema 与 `ProbabilisticSpec`；另写不含模型对象的 `probabilistic_schema.json`，兼容旧 quantile dict loader。
5. 多步预测 P0 已修复；判定分解精度前须重跑受影响的 MSMD/MSMDR 结果，避免沿用错位旧口径。
6. 高频场景按单配置比较 `none / linear / STL(日周期) / MSTL(日+周)`；MSTL 只在修复 robust 后进入。
7. 日频按 `none / linear-polynomial / linear-damped / quadratic / STL7` 比较；周季节性弱时不因“可分解”而默认启用。
8. 月频样本不足两个年度周期时保持 `none`，不要用不足周期的 STL12 制造分量。
9. 正式推广仍要求 A/B 相同窗口、median MAPE 改善绝对值 ≥0.005 且方向一致；同时报告 MAE、naive skill、horizon-wise error 和未来确定分量范围。
10. 乘性/变换域、OOF 分解特征、分量级模型和 DLinear/N-BEATS 都属于后续研究，不在当前主链扩展。

## 8. 交叉验证设计

### 8.1 保留当前 rolling-origin，多补三项

M5 方案明确指出单个 28 天验证区间波动太大，采用最近三个 28 天区间，并观察各实验在不同 fold 上是否方向一致。[16] Enefit 第五名采用三折、每折两个月的时间 CV，再用最后三个月做第二阶段验证。[19] 这与项目当前“多窗口 + A/B 同向 + median”原则一致，应继续保留。

需要新增：

1. **gap/embargo**：验证集开始前减去真实数据延迟。Panama 周负荷预测要求在目标周前保留 72h 不可见 gap；如果本项目某数据源存在 T+1/T+2 才可得，应在 splitter 中显式表达，而不是只靠特征工程约定。[9]
2. **horizon-wise score**：每个预测步的点/概率指标；固定 horizon 和 `calendar_month` 都要支持。
3. **regime stratification**：负荷分位数、ramp、节假日、天气 lead、事件前后；标签必须是预测原点可见或只用于离线诊断。

可复用的 gap 切分骨架：

```python
def rolling_origin_slices(
    n_rows: int,
    train_rows: int,
    horizon: int,
    gap_steps: int,
    stride: int,
):
    forecast_start = train_rows + gap_steps
    while forecast_start + horizon <= n_rows:
        train_end = forecast_start - gap_steps
        train_start = max(0, train_end - train_rows)
        yield (
            slice(train_start, train_end),
            slice(forecast_start, forecast_start + horizon),
        )
        forecast_start += stride
```

### 8.2 不采用随机 KFold

ASHRAE 讨论提醒随机/k-fold 可能让某些 fold 缺失特定 building/month，或者利用相同 building 在 train/test 的重叠；其建议本质上仍是按月份构造能代表测试分布的验证。[3] 对项目而言，任何随机 KFold 只适合训练窗内部无时序语义的局部调参，而且不能替代外层 rolling-origin 测试。

### 8.3 指标要与业务代价同构

Kaggle Notebook 常按竞赛指标优化：Store Item 用 SMAPE，M5 用加权缩放 pinball，ASHRAE 用 RMSLE。[4][6][7] 项目不能照搬某个 Kaggle 指标，而应：

- 点预测继续保留 median MAPE + MAE/RMSE；
- 补充相对 seasonal baseline 的 skill score；
- 概率预测使用 pinball、coverage、width/Winkler；
- 如上下偏差业务成本不同，再增加 asymmetric cost；
- 指标必须按窗口、horizon、route 和场景共同输出。

## 9. 概率预测：基础闭环已完成，高级校准待实证

### 9.1 当前项目事实

- 点指标继续保留 R2/MSE/RMSE/MAE/MAPE 和 naive MAPE；
- `probabilistic/metrics.py` 已实现 pinball、coverage、width、normalized width、Winkler、coverage gap、Wilson CI 和 crossing 指标；
- `ModelTesting` 写出 `probabilistic_predictions_df.csv`、`probabilistic_scores_df.csv`、`horizon_scores_df.csv`；
- forecast 中 `predict_q*` 保留模型分位数，CQR 改用独立 `predict_pi<coverage>_lower/upper`；
- CQR score 在 q50 锚定 repair 后计算，可由保存的 processed boundaries 复算；
- `CalibrationRecord` 携带 `forecast_origin/target_time/label_available_at/horizon_step/window`，prequential 与 final 共用 as-of selector/calibrator；`calibration_report.csv` 审计实际选择范围、完整窗口数、score 数、correction 和状态。

因此项目现在可以系统回答 raw/processed quantile 与因果 CQR 的基础概率质量；剩余问题是：

- CQR 在各 route/scenario 的严格历史未知窗口上是否稳定优于原始 quantile；
- 是否需要 horizon-binned、asymmetric、block 或 adaptive conformal。

### 9.2 Phase 0 已实现指标集合

当前实现位于 `probabilistic/metrics.py`：

| 指标 | 回答的问题 | 输出粒度 |
|---|---|---|
| Pinball loss(q) | 单个分位数预测是否准确 | q × window × horizon |
| Mean/weighted pinball | 整体概率精度 | window、route、scenario |
| Empirical coverage | 区间覆盖率是否达标 | interval × window × horizon |
| Average interval width | 覆盖率是否靠过宽换来 | interval × window × horizon |
| Winkler score | 同时惩罚过宽和漏覆盖 | interval × window × horizon |
| Quantile crossing rate | 分位数顺序是否失真 | window × horizon |
| Calibration error | 名义覆盖与实际覆盖差 | interval × regime |

M5 的快速实现先按层级计算 scale/weight，再对每个 q 计算 pinball 并聚合；项目不需要照搬零售权重和 12 层 hierarchy，但 pinball 核心可以直接等价重写。[23]

```python
def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> np.ndarray:
    error = np.asarray(y_true) - np.asarray(y_pred)
    return np.maximum(quantile * error, (quantile - 1.0) * error)


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    covered = (np.asarray(y_true) >= lower) & (np.asarray(y_true) <= upper)
    return float(np.mean(covered))
```

### 9.3 CQR 的下一步

先评估，再改校准算法。只有确认以下模式后才扩展：

- coverage 随 horizon 明显下降 → 做 horizon-binned conformal；
- 高负荷/突变窗口漏覆盖 → 做 regime-aware diagnostics，不能直接使用未来标签校准；
- 上下尾误差明显不对称 → 考虑 asymmetric conformal score；
- 校准样本过少 → 保持现有 skip/WARNING，不用少量窗口硬校准。

## 10. 基线和 skill score

当前项目测试中只有“昨日同时刻”naive（`models/ModelTesting.py:590-616`）。Kaggle 能源 Notebook 普遍先建立 naive/persistence，再判断复杂模型是否真正增益；PJM 使用一年同期 persistence，伦敦智能电表使用一周 persistence。[8][20]

建议统一输出：

1. `naive_last_day`；
2. `naive_last_week`；
3. `naive_recent_same_slot_median`；
4. `SeasonalTemplate`；
5. 可选业务官方/调度基线（若有）；
6. `skill = 1 - model_loss / baseline_loss`。

不同频率自动选择合法基线：

- 5min/15min/1h：昨日、上周同槽位；
- 1D：上周、去年同日或近期同 weekday；
- 1ME/1MS：去年同月、趋势 naive；
- `calendar_month`：同时比较逐日误差和月总量误差。

## 11. 面向 tsproj_ml 的能力差距矩阵

| 能力 | 当前状态 | Kaggle 增量 | 优先级 | 真实代价 |
|---|---|---|---:|---|
| MSMD/MSMDR 目标对齐 | ✅ 已修复为 `shift_1..shift_H`，并有单/多变量对称契约测试 | Kaggle Direct/recursive 均保持目标时点一致 | 已关闭 | 修复前结果作废，待按新口径重跑 |
| USMDP pointwise direct | ✅ 已开放 H>1 safe-lag；align 开启时强制 lag 非空、为正且 `min(lags) >= horizon` | Store Item/Enefit 使用可得 lag 的 pooled pointwise | 已关闭 | 工程接线完成，待单配置 A/B 消融 |
| Direct 输出长度契约 | ✅ point 与每个 quantile 均要求精确等于 horizon，过短/过长直接 RAISE | Kaggle 提交通常要求完整逐步生成 | 已关闭 | 已补 shape 回归测试 |
| 分解因果接线 | 每窗口仅用 train 拟合，point/quantile 同构 restore | Kaggle residual hybrid / OOF decomposition | 保持 | 接线正确，继续消融 |
| linear+damped lookback | ✅ 已于 2026-08-23 修复（近期斜率取观测 y 的 lookback 段） | 近期趋势阻尼外推 | 已关闭 | damped 消融尚未运行，无既有结果作废 |
| MSTL robust | ✅ 已于 2026-08-23 修复并有接线测试；当前仍无 MSTL YAML | robust multi-seasonal decomposition | 已关闭 | 可启用 MSTL 配置进入消融 |
| 周期诊断 | ✅ 已于 2026-08-23 修复边界并输出标准强度（旧 ratio 标注 legacy） | STL/MSTL 周期候选与强度分析 | 已关闭 | 已完成 |
| decomposer 部署加载 | ✅ 已完成（2026-08-24）：唯一 `ForecastModelBundle`（model+feature scaler+目标变换栈+selected features+input schema），ModelDeployPkl 自动识别并拒绝未知 schema | 可复现 hybrid pipeline | 已关闭 | 单文件部署产物就绪 |
| 概率预测训练 | quantile + monotone + CQR | 分布模型/层级 quantile | P2 | 新管线、高训练成本 |
| 概率评估 | ✅ Phase 0 已实现 pinball/coverage/width/Winkler/crossing/coverage gap | M5 完整概率评分 | 已关闭 | 三张概率 artifact 已接入 |
| horizon 诊断 | ✅ Phase 0 已按 `horizon_step` 输出概率指标、n_points/n_windows | 按步/场景分解 | 已关闭 | fixed/calendar_month 共用 |
| baseline suite | 仅昨日同时刻 naive | 日/周/模板/业务基线 skill | **P0** | 低到中等 |
| 天气信息集 | strict contract 已较强 | forecast vintage、lead、anomaly | P1 | 中等，需保持 train/CV/final 对称 |
| EWM | 主链未接入 | lag 后 EWM | P1 | 低到中等，需消融防冗余 |
| gap/embargo | 未作为通用 splitter 参数 | Panama 72h gap 范式 | P1 | 中等，影响窗口索引和配置 |
| 全局 panel | ✅ target/lag/horizon 外生与顺序型高级特征已按 `series_id` 分组；当前场景仍关闭 | Kaggle 全局模型严格按实体分组构造 lag | P2 | 代码边界已修，仍需建立 panel 数据与消融 |
| 层级协调 | 未见主链能力 | M5 hierarchy/reconciliation | P2 | 高，需先有真实加总层级 |
| 深度模型 | 未接入主链 | GRU/DeepAR 可提供异构分布 | P2 | 高，独立训练与保存管线 |
| 在线更新 | 每次运行可用当前历史重训 | scoring 期按新数据周期更新 | P1/P2 | 取决于生产调度，不一定需要新模型代码 |

## 12. 推荐实施顺序与验收标准

### 阶段 0：多步语义修复（代码已完成，结果重跑待执行）

1. ✅ MSMD/MSMDR 的 Direct 目标已改为 `shift_1..shift_H`；MSMR 的 `shift_0` 保持不变；
2. ✅ 已新增 USMD/MSMD、USMDR/MSMDR 目标起点对称测试；
3. ✅ Direct point/quantile 输出长度不等于请求 horizon 时直接 RAISE；
4. ✅ USMDP 已开放多步 safe-lag，并在 main、forecast 和配置检查三处复用同一契约；
5. ✅ global panel 顺序型特征已按 `series_id` 分组，缺失实体 ID 时 fail-fast；
6. ⏳ 标记并重跑所有 MSMD/MSMDR 既有结果，旧指标不可继续用于比较；
7. ⏳ safe-lag/global panel 分别做单配置冒烟和 A/B 同窗消融，未验证前不全矩阵推广。

### 阶段 0.5：修正分解配置与诊断语义（全部完成）

1. ✅ 修复 linear+damped 的 `decomposition_trend_lookback`（斜率取自观测 y，见 §7.7.1 修复说明）；
2. ✅ 修复 MSTL 的 `decomposition_robust` 透传；
3. ✅ 修正周期工具的两周期边界与标准季节/趋势强度；
4. ✅ 建立 decomposer 与 model/scaler 的统一加载契约（2026-08-24，v2 内嵌式 bundle；feature scaler/selected features schema 未纳入，当前部署链路不消费）；
5. ✅ 分解架构已迁移到 `decomposition/` 包并补齐残差频谱诊断（`residual_diagnostics.csv`）；damped/MSTL 消融前置条件全部解除。

### 阶段 A：只增强评估，不改变预测值

1. ✅ 概率指标函数及单测（2026-08-24，`probabilistic/metrics.py`）；
2. ✅ `ModelTesting` 输出 `probabilistic_scores_df.csv`（2026-08-24）；
3. ✅ 新增 `horizon_scores_df.csv`（2026-08-24）；
4. ⏳ 新增 baseline suite 与 skill score（**本文 P0 欠账**；昨日同时刻 naive 的 `naive_skill` 已随 horizon 聚合落地 2026-08-24，多基线套件仍待做）；
5. ✅ 在已有 quantile 配置上实跑，确认预测 CSV 数值不变、只新增评估产物（/tmp 隔离验收）；
6. ✅ 跑完整 unittest（327 tests OK）。

验收：

- q10/q50/q90 pinball 可手算复核；
- coverage/width/Winkler 在合成样本上有确定结果；
- crossing rate 在单调化前后符合预期；
- 点预测配置不生成无意义概率指标；
- 固定 horizon 与 `calendar_month` 均能按步输出。

### 阶段 B：低风险单配置消融

候选 1：EWM。  
候选 2：weather lead/vintage/anomaly。  
候选 3：多 baseline 只作为评估，不参与模型。

实验纪律：

1. 从当前最优配置复制变体，用 `setting_suffix` 隔离；
2. route_A 先跑 31 窗；
3. route_A 达到候选门槛后补 route_B 31 窗；
4. 正式判定仍为两路 median MAPE 同向且绝对差 ≥ 0.005；
5. 概率能力额外要求 pinball 改善，coverage 接近名义值且 width 不出现不可接受膨胀；
6. 失败即回退配置与结果，不全矩阵铺开。

### 阶段 C：结构性研究

按价值排序：

1. 真正的跨 route/room global panel GBDT；
2. A/B/room/site 的层级一致性与 reconciliation；
3. 独立 GRU/DeepAR 概率支线；
4. 只在证明存在稳定异构收益后接入 ModelEnsemble。

## 13. 不建议直接采用的 Kaggle 做法

| 做法 | 判断 | 原因 |
|---|---|---|
| 随机 KFold | 不采用 | 时序泄漏、实体/月覆盖失真 |
| 给 lag/rolling 加随机噪声 | 暂不采用 | 生产确定性和可复现性差，已有数据增强通路 |
| 仅凭 leaderboard 调特征 | 不采用 | 公开榜过拟合，不符合 A/B 多窗判定 |
| 用测试期实测天气 | 禁止 | 违反项目 strict information set |
| 一次生成数百个 lag/天气特征 | 不采用 | 高冗余、假模式和训练成本；先单族消融 |
| 完整目标先分解再切 train/test | 禁止 | trend/seasonal/resid 已消费验证与测试目标，形成直接泄漏。[20] |
| 把 in-sample 分解拟合值直接当二阶段特征 | 不采用 | 二阶段训练条件过于理想；必须使用 OOF 分解预测。[34] |
| 同模型多 seed 简单平均 | 低优先级 | M5 方案报告无明显收益；项目融合消融也未显示稳定增益。[16] |
| 直接接 Transformer | 不采用 | Enefit 第一名报告 Transformer/1D-CNN 未奏效，且与当前主链接口不匹配。[18] |
| 把 DeepAR/GRU 塞进 ModelFactory | 不采用 | sklearn 单输出回归器契约不适配概率序列模型 |
| 复制 Kaggle 数据进仓库 | 不采用 | 许可证和业务分布不同，只复用方法与公开代码模式 |

## 14. 数据与代码复用边界

- Enefit 竞赛数据页标注 CC BY-NC-SA 4.0，不能默认作为商业项目自由数据源。[1]
- PJM Hourly Energy Consumption 与 Spain Energy + Weather 标注 CC0，可作为外部 benchmark 数据候选。[10][11]
- Panama 数据标注 CC BY 4.0，使用时需保留署名。[9]
- Demand LightGBM、PJM 和 London smart meter Notebook 页面标注 Apache 2.0。[7][8][20]
- M5 WSPL 与 Enefit starter Notebook 页面同样标注 Apache 2.0；可以等价重写小型算法模式，若复制实质性代码需保留许可证和 attribution。[23][26]
- Kaggle Learn Hybrid、MSTL+CatBoost 和 STL Notebook 仅作为方法参考；复用实质性代码时需分别核对并保留许可证与 attribution。[33][34][35]
- N-BEATS Notebook 同样只作为独立深度分解路线的参考。[37]
- 本次调研未把任何 Kaggle 数据下载或写入项目，只读取公开/登录可见页面和 Notebook 源码。

## 15. 最终建议

**多步目标对齐、Direct 长度契约、USMDP safe-lag、global panel 实体边界、分解配置语义和概率重构 Phase 0–4 已完成代码修复与验收；当前正确方向是补齐 baseline suite/skill score（本文唯一 P0），重跑受影响结果，再做特征和模型消融。**

推荐立即进入的工程目标为：

1. 结果重建：重跑修复前受影响的 MSMD/MSMDR 配置，确保旧目标口径的模型、CSV 和图不再参与比较；
2. 能力消融：USMDP safe-lag 先做单配置冒烟，再按 A/B 各 31 窗判定；global panel 等真实 panel 数据契约就位后单独评估；
3. 分解集成：linear+damped lookback、MSTL robust、周期强度/边界、统一加载 bundle（v2 内嵌式）、`decomposition/` 包架构迁移已于 2026-08-23~24 全部完成并验收；残差频谱诊断工具就位，VMD/Wavelet 是否启动取决于实跑 `residual_diagnostics.csv` 的 stable_band 结果；
4. 评估闭环：`probabilistic_metrics`、`probabilistic_scores_df.csv`、`horizon_scores_df.csv` 已于概率重构 Phase 0–4 落地（2026-08-24，327 tests）；**仍未完成**：baseline suite 与 skill score（本文唯一 P0 欠账）。

EWM、weather vintage 和 gap/embargo 作为下一批独立消融；hierarchical reconciliation、GRU/DeepAR 作为后续结构性研究。global panel 不再有已知跨序列特征串值阻塞，但在真实 panel 消融前仍不作为现役默认。这样既吸收 Kaggle 的高价值经验，也不破坏项目当前已经建立的严格信息集、A/B 多窗判定和最小改动原则。

## Sources

[1] https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/data
[2] https://www.kaggle.com/competitions/ashrae-energy-prediction/data
[3] https://www.kaggle.com/competitions/ashrae-energy-prediction/discussion/112872
[4] https://www.kaggle.com/competitions/m5-forecasting-uncertainty
[6] https://www.kaggle.com/competitions/store-sales-time-series-forecasting
[7] https://www.kaggle.com/code/nurlan91/demand-forecasting-by-using-lightgbm
[8] https://www.kaggle.com/code/alcheng10/forecasting-electricity-pjm-us-east-coast-grid
[9] https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting/data
[10] https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
[11] https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather
[16] https://www.kaggle.com/competitions/m5-forecasting-uncertainty/discussion/163219
[18] https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/writeups/hyd-1st-place-solution
[19] https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/writeups/loh-maa-5th-place-solution
[20] https://www.kaggle.com/code/jaksanders/electric-load-forecasting-with-deep-learning
[21] https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda
[23] https://www.kaggle.com/code/kneroma/fast-wsp-loss-implementation-5s
[26] https://www.kaggle.com/code/rafiko1/enefit-xgboost-starter
[29] https://www.kaggle.com/code/kneroma/m5-first-public-notebook-under-0-50
[30] https://www.kaggle.com/learn/time-series
[31] https://www.kaggle.com/code/ryanholbrook/trend
[32] https://www.kaggle.com/code/ryanholbrook/seasonality
[33] https://www.kaggle.com/code/ryanholbrook/hybrid-models
[34] https://www.kaggle.com/code/antonysamuelb/mstl-catboost-4th-place
[35] https://www.kaggle.com/code/eugeniyosetrov/seasonal-trend-decomposition-using-loess-stl
[36] https://www.kaggle.com/code/alisadeghiaghili/time-series-decomposition-methods
[37] https://www.kaggle.com/code/antonellomartiello/n-beats-for-time-series-forecasting
[38] https://www.kaggle.com/code/rakeshpanigrahy/trasformers-for-timeseries-autoformer-dlinear
[39] https://www.kaggle.com/code/pythonafroz/time-series-forecast-with-fb-prophet
[40] https://www.kaggle.com/code/ahmedabdulhamid/data-imputation-demystified-time-series-data
[41] https://www.kaggle.com/code/artemig/time-series-anomaly-detection-with-iforest-and-stl
[42] https://www.kaggle.com/code/liuxuhui/notebookdb70dc3973
