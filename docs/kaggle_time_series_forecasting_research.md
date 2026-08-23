# Kaggle 时间序列预测调研：面向 tsproj_ml 的能力提升建议

> 文档性质：外部技术调研与项目能力差距分析  
> 调研日期：2026-08-22  
> 调研范围：Kaggle 竞赛、获奖方案、Notebook、数据集和讨论；重点关注电力负荷、能源时序、多步预测、概率预测、天气外生变量  
> 本文不构成实施授权；建议先修复多步预测语义，再补评估闭环，最后按单配置消融规则逐项实施。

## 1. 结论先行

**没有一个 Kaggle 方案适合直接移植到 `tsproj_ml`。** 项目现有主链已经覆盖多步 Direct/Recursive/DirRec/Blend、分位数预测、CQR、严格天气信息集、滑窗回测和多种树模型；Kaggle 的主要增量不是“再接一个模型”，而是以下三项：

1. **补齐概率预测评估闭环**：pinball loss、区间覆盖率、平均区间宽度、Winkler score、quantile crossing，以及按 horizon/场景分组的校准诊断。M5 Uncertainty 直接以加权缩放 pinball loss 评估概率分布；其高排名方案也按多个相邻时间折验证方向一致性，而不是只看中位数点预测误差。[4][16][23]
2. **把回测从“窗口总分”扩展为“预测步 × 场景 × 基线”的 skill 分析**：当前 `test_scores_df.csv` 只给整窗 R2/MSE/RMSE/MAE/MAPE 和昨日同时刻 naive，容易掩盖远 horizon 退化、节假日/突变日失效和区间失准。
3. **新增少数有明确信息增量的特征族**：严格 trailing 的 EWM、天气 forecast vintage/`hours_ahead`、天气相对近期基准的 anomaly、冷/热度时与交互项；每项都必须遵循现有 A/B 两路、相同 31 窗、median MAPE 的消融规则。Enefit 获奖方案反复使用天气预报生成时刻、`hours_ahead`、历史目标变换和天气相对近期均值，而不是简单把“实测天气”并入未来。[18][19][26]

专项代码复核还发现：**MSMD/MSMDR 当前存在 `shift_0`→未来第 1 步的 off-by-one 错配，优先级高于新增指标和特征。** USMDP 在多步 horizon 下不能使用 Kaggle 常见的安全 lag，`enable_global_training` 也尚未对 lag/target shift 按 `series_id` 分组。这三项详见 §6 和 §10。

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
2. **确定缺陷**：MSMD/MSMDR 的 Direct 目标从 `shift_0` 开始，却把第一个输出写到第一个未来时点，形成确定的 off-by-one；受影响结果必须在修复后重跑。
3. **结构缺口**：USMDP 缺少 Kaggle 常用的多步安全 lag；global panel 的 lag/target shift 未按 `series_id` 分组；Direct 输出长度异常时会静默 edge padding。这些问题不能靠模型调参解决。

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
| USMDP | 行 `t` 预测 `y_t` (`shift_0`) | 所有未来行一次 pointwise predict | 单模型、成本低、天然消费逐时外生 | H>1 默认无目标 lag，能力不完整 |
| USMD | 原点 `t` 预测 `y_{t+1..t+H}` | 一次输出 H 步 | 无递归误差累积 | 语义正确；默认 H 个独立模型 |
| USMR | 未来行 `t+h` 预测本行目标 | 逐步预测并写回 | 可使用短 lag、模型仅 1 个 | 有固有误差累积 |
| USMDR | 原点预测下一 block | block 内 Direct、block 间回填 | 直接/递归折中 | 单变量目标起点正确 |
| MSMD | 多变量原点预测 H 步 | 一次输出 H 步 | 可使用多个内生变量历史 | **当前 off-by-one** |
| MSMR | 多变量未来行预测本行目标 | 逐步预测目标，其他内生量回填 | 可用多内生历史 | `shift_0` 在此语义下正确 |
| MSMDR | 多变量原点预测下一 block | block Direct、块间回填 | 多变量 DirRec | **当前 off-by-one** |
| USBR/MSBR | Direct `shift_1..H` + Recursive `shift_0` | 两路预测后融合 | 组合两种误差结构 | 目标起点正确；收益需消融 |

判断 `shift_0` 是否正确不能只看列名，必须看“特征行代表预测原点还是目标时点”：Recursive/Pointwise 的特征行就是目标时点，`shift_0` 正确；Direct/DirRec 的特征行是预测原点，第一个目标必须是 `shift_1`。

### 6.4 Kaggle 与项目的关键差异

#### 6.4.1 Pointwise direct 的安全 lag

Kaggle 的 pointwise direct 并不等于“不要 lag”。它通过保证最短 lag 不小于 horizon，使每个未来行都只引用 cutoff 前历史：

$$
\hat y_{t+h}=f(y_{t+h-L_1},\ldots,x_{t+h},h,\text{entity}),\qquad \min(L_i)\ge H
$$

例如预测 96 个 15min 点并使用 lag96：第 96 个未来点的 lag96 恰好指向最后一个历史点；整个 horizon 无需递归，也没有未来真值泄漏。项目 USMDP 默认不生成 lag，`align_direct_features_to_target=true` 又被 `main.py:223-225` 限制为 `horizon=1`，因此当前无法表达这条 Kaggle 常见路径。

#### 6.4.2 Recursive 的实体分组

Kaggle global recursive 的 lag、rolling 和回填均按实体 ID 分组。项目 USMR/MSMR 在“每个配置只含一条目标序列”时语义正确；一旦开启 panel/global，所有 shift/rolling 都必须先按 `series_id` 分组，否则序列边界会相互污染。

#### 6.4.3 Direct multi-output 的参数共享

Enefit GRU 的 24 个输出共享序列表征；项目默认 `DirectMultiOutputRegressor` 为每个 horizon 训练独立回归器。项目的 `horizon_feature` 可改成一个共享模型，`regressor_chain` 可让后续 horizon 消费前序输出，但三者是不同偏差—方差—成本取舍，不存在一个对所有频率都更优的默认答案。

#### 6.4.4 概率轨迹的一致性

项目按 quantile、horizon 独立训练，再做逐时点 quantile 单调化和边际 CQR；它能表达每个未来时点的边际 P10/P50/P90，但不能直接表达整条轨迹的联合概率、月总量分布或连续超限概率。M5 的 DeepAR 通过条件分布采样得到路径级 quantile，能力层级不同。[16]

### 6.5 确定缺陷与影响范围

#### 6.5.1 P0：MSMD/MSMDR 目标 off-by-one

`features/FeatureEngineering.py:1253-1257,1287-1291` 调用 `extend_direct_multi_step_targets` 时沿用默认 `start_step=0`，生成 `y_shift_0..H-1`；forecast 却从最后历史行取原点输入，并把输出第 1 项写到未来第 1 点（`models/ModelForecasting.py:925-936,1090-1108`）。

以 `y=[10,11,12,13,14,15]`、`H=3` 的可执行探针为例：

| 方法 | 实际目标列 | 第 1 行目标值 | 正确性 |
|---|---|---:|---|
| USMD | `shift_1,shift_2,shift_3` | 11,12,13 | 正确 |
| MSMD | `shift_0,shift_1,shift_2` | 10,11,12 | **错一格** |
| USMDR | `shift_1,shift_2,shift_3` | 11,12,13 | 正确 |
| MSMDR | `shift_0,shift_1,shift_2` | 10,11,12 | **错一格** |

该缺陷影响 MSMD/MSMDR 的 point、quantile、conformal 底层预测，以及 MSMD `horizon_feature`；不影响 USMD、USMR、USMDR、USMDP、MSMR 和使用 `shift_1..H` 的 USBR/MSBR Direct 子模型。既有 MSMD/MSMDR 回测、forecast 和模型产物在修复后应视为旧口径结果。

#### 6.5.2 P1：global panel 跨序列串值

`enable_global_training` 当前只保留 `series_id` 作为特征，而 lag 与 target 仍使用整表 `shift`。两条连续存放的序列 `A=[1,2,3]`、`B=[100,200,300]` 在探针中得到：A 最后一行 `y_shift_1=100`，B 第一行 `y_lag_1=3`。当前 YAML 未启用 global，因此未污染现役结果；但该开关在 grouped lag/target/rolling/diff 修复前不能直接启用。

#### 6.5.3 P1：Direct 输出长度异常被静默掩盖

USMD/MSMD 在模型输出少于请求 horizon 时使用 `np.pad(..., mode="edge")` 复制最后一个预测（`models/ModelForecasting.py:767-772,930-935`），quantile 也使用同样逻辑（`models/ModelForecasting.py:558-567`）。这会把模型版本、target schema 或动态 horizon 错误伪装成尾部常数预测，应改为长度不等即 RAISE。

### 6.6 设计限制：不能与 bug 混为一谈

| 限制 | 原因 | 项目已有缓解 | 判断 |
|---|---|---|---|
| Recursive 误差累积 | 训练消费真值，推理消费自身预测 | DirRec、Blend、auxiliary | 固有限制 |
| Direct horizon 独立 | H 个模型不共享参数和输出协方差 | horizon-feature、regressor_chain | 设计取舍 |
| Quantile/trajectory 非联合 | 每个 q、h 独立训练 | monotone、CQR 仅改善边际 | 能力边界 |
| horizon-feature `N×H` 成本 | 高频大 H 重复展开样本 | 仅在低频/小 H 使用 | 适用范围限制 |
| 多变量递归外生内生量未知 | 非目标内生变量未来不可得 | persistence、auxiliary | 信息集约束 |

### 6.7 测试证据与测试缺口

当前工作树新增的 `HorizonAlignedDirectRegressor` 能让每个 Direct 输出只消费对应 `*_hN` 外生列，定向测试 3/3 通过；它不修改 MSMD/MSMDR 的 target 起点，因此不能消除 off-by-one。

专项复核时完整 unittest 为 194/194 通过，但现有测试没有断言：MSMD/MSMDR 第一个输出必须对应下一时点；global panel 的序列边界不得跨组 shift；Direct 输出长度必须精确等于请求 horizon。测试通过证明现有断言满足，不证明这些未覆盖的时间语义正确。

### 6.8 修复顺序与实验影响

1. MSMD/MSMDR 的 Direct 目标改为 `shift_1..shift_H`，MSMR 的 `shift_0` 保持不变；
2. 新增 USMD↔MSMD、USMDR↔MSMDR 对称契约测试，并覆盖 point、quantile、horizon-feature 和 DirRec block；
3. Direct 输出长度改为 fail-fast；
4. 修复后使既有 MSMD/MSMDR 结果失效并重跑，不能与新口径混合比较；
5. 第二阶段扩展 USMDP safe-lag，配置契约为 `min(lags) >= horizon`；
6. 第三阶段把 global 模式所有 lag/target/rolling/diff 改为按 `series_id` 分组，再开展 panel 消融；
7. 最后补 horizon-wise MAPE/MAE/pinball/coverage，验证修复是否改善远 horizon，而不是只看整窗 median。

## 7. 交叉验证设计

### 7.1 保留当前 rolling-origin，多补三项

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

### 7.2 不采用随机 KFold

ASHRAE 讨论提醒随机/k-fold 可能让某些 fold 缺失特定 building/month，或者利用相同 building 在 train/test 的重叠；其建议本质上仍是按月份构造能代表测试分布的验证。[3] 对项目而言，任何随机 KFold 只适合训练窗内部无时序语义的局部调参，而且不能替代外层 rolling-origin 测试。

### 7.3 指标要与业务代价同构

Kaggle Notebook 常按竞赛指标优化：Store Item 用 SMAPE，M5 用加权缩放 pinball，ASHRAE 用 RMSLE。[4][6][7] 项目不能照搬某个 Kaggle 指标，而应：

- 点预测继续保留 median MAPE + MAE/RMSE；
- 补充相对 seasonal baseline 的 skill score；
- 概率预测使用 pinball、coverage、width/Winkler；
- 如上下偏差业务成本不同，再增加 asymmetric cost；
- 指标必须按窗口、horizon、route 和场景共同输出。

## 8. 概率预测：当前最大能力缺口

### 8.1 当前项目事实

- `models/ModelTesting.py:619-671` 只计算 R2/MSE/RMSE/MAE/MAPE 和 naive MAPE；
- `models/ModelTesting.py:384-389` 在启用 conformal 时记录 nonconformity score；
- `main.py:680-727` 在 final forecast 阶段校准首尾 quantile，并做 quantile 单调化；
- 全仓源码没有 pinball、PICP、Winkler、CRPS 或 coverage-rate 实现。

因此项目“能产出概率预测”，但还不能系统回答：

- 每个 quantile 是否准确；
- 80% 区间实际覆盖是否接近 80%；
- 覆盖率提升是否只是区间过宽；
- 哪些 horizon/场景失准；
- CQR 是否优于原始 quantile，而不是只看 q50 MAPE。

### 8.2 P0 指标集合

建议新增 `utils/probabilistic_metrics.py`：

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

### 8.3 CQR 的下一步

先评估，再改校准算法。只有确认以下模式后才扩展：

- coverage 随 horizon 明显下降 → 做 horizon-binned conformal；
- 高负荷/突变窗口漏覆盖 → 做 regime-aware diagnostics，不能直接使用未来标签校准；
- 上下尾误差明显不对称 → 考虑 asymmetric conformal score；
- 校准样本过少 → 保持现有 skip/WARNING，不用少量窗口硬校准。

## 9. 基线和 skill score

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

## 10. 面向 tsproj_ml 的能力差距矩阵

| 能力 | 当前状态 | Kaggle 增量 | 优先级 | 真实代价 |
|---|---|---|---:|---|
| MSMD/MSMDR 目标对齐 | 当前从 `shift_0` 开始，forecast 从未来第 1 步接收输出 | Kaggle Direct/recursive 均保持目标时点一致 | **P0** | 修两处分支、补契约测试、旧结果作废重跑 |
| USMDP pointwise direct | 多步默认不生成 lag，安全 lag 仅支持 H=1 | Store Item/Enefit 使用可得 lag 的 pooled pointwise | P1 | 增加 `min(lag) >= horizon` 契约与推理上下文 |
| Direct 输出长度契约 | 输出不足时 edge padding | Kaggle 提交通常要求完整逐步生成 | P1 | 改 fail-fast 并补 shape 测试 |
| 概率预测训练 | quantile + monotone + CQR | 分布模型/层级 quantile | P2 | 新管线、高训练成本 |
| 概率评估 | **缺 pinball/coverage/width/Winkler** | M5 完整概率评分 | **P0** | 中等，主要是指标与输出测试 |
| horizon 诊断 | 仅整窗指标和 per-window 图 | 按步/场景分解 | **P0** | 中等，新增结果表和报告 |
| baseline suite | 仅昨日同时刻 naive | 日/周/模板/业务基线 skill | **P0** | 低到中等 |
| 天气信息集 | strict contract 已较强 | forecast vintage、lead、anomaly | P1 | 中等，需保持 train/CV/final 对称 |
| EWM | 主链未接入 | lag 后 EWM | P1 | 低到中等，需消融防冗余 |
| gap/embargo | 未作为通用 splitter 参数 | Panama 72h gap 范式 | P1 | 中等，影响窗口索引和配置 |
| 全局 panel | 有配置字段但 lag/target shift 未按 `series_id` 分组；当前场景均关闭 | Kaggle 全局模型严格按实体分组构造 lag | P1/P2 | 先修分组契约，再建立 panel 数据与消融 |
| 层级协调 | 未见主链能力 | M5 hierarchy/reconciliation | P2 | 高，需先有真实加总层级 |
| 深度模型 | 未接入主链 | GRU/DeepAR 可提供异构分布 | P2 | 高，独立训练与保存管线 |
| 在线更新 | 每次运行可用当前历史重训 | scoring 期按新数据周期更新 | P1/P2 | 取决于生产调度，不一定需要新模型代码 |

## 11. 推荐实施顺序与验收标准

### 阶段 0：先修多步语义，不运行旧口径实验

1. MSMD/MSMDR 的 Direct 目标改为 `shift_1..shift_H`；MSMR 的 `shift_0` 保持不变；
2. 新增四方法对称测试：USMD/MSMD、USMDR/MSMDR 的第 1 个输出都必须对应下一时点；
3. 覆盖普通 multioutput、horizon_feature、quantile 和 DirRec block 四条路径；
4. Direct 输出长度不等于请求 horizon 时直接 RAISE；
5. 标记并重跑所有 MSMD/MSMDR 既有结果，旧指标不可继续用于比较；
6. global 模式在按 `series_id` 分组修复前保持关闭。

### 阶段 A：只增强评估，不改变预测值

1. 新增概率指标函数及单测；
2. `ModelTesting` 输出 `probabilistic_scores_df.csv`；
3. 新增 `horizon_scores_df.csv`；
4. 新增 baseline suite 与 skill score；
5. 在一个已有 quantile 配置上实跑，确认预测 CSV 数值不变、只新增评估产物；
6. 跑完整 unittest。

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

## 12. 不建议直接采用的 Kaggle 做法

| 做法 | 判断 | 原因 |
|---|---|---|
| 随机 KFold | 不采用 | 时序泄漏、实体/月覆盖失真 |
| 给 lag/rolling 加随机噪声 | 暂不采用 | 生产确定性和可复现性差，已有数据增强通路 |
| 仅凭 leaderboard 调特征 | 不采用 | 公开榜过拟合，不符合 A/B 多窗判定 |
| 用测试期实测天气 | 禁止 | 违反项目 strict information set |
| 一次生成数百个 lag/天气特征 | 不采用 | 高冗余、假模式和训练成本；先单族消融 |
| 同模型多 seed 简单平均 | 低优先级 | M5 方案报告无明显收益；项目融合消融也未显示稳定增益。[16] |
| 直接接 Transformer | 不采用 | Enefit 第一名报告 Transformer/1D-CNN 未奏效，且与当前主链接口不匹配。[18] |
| 把 DeepAR/GRU 塞进 ModelFactory | 不采用 | sklearn 单输出回归器契约不适配概率序列模型 |
| 复制 Kaggle 数据进仓库 | 不采用 | 许可证和业务分布不同，只复用方法与公开代码模式 |

## 13. 数据与代码复用边界

- Enefit 竞赛数据页标注 CC BY-NC-SA 4.0，不能默认作为商业项目自由数据源。[1]
- PJM Hourly Energy Consumption 与 Spain Energy + Weather 标注 CC0，可作为外部 benchmark 数据候选。[10][11]
- Panama 数据标注 CC BY 4.0，使用时需保留署名。[9]
- Demand LightGBM、PJM 和 London smart meter Notebook 页面标注 Apache 2.0。[7][8][20]
- M5 WSPL 与 Enefit starter Notebook 页面同样标注 Apache 2.0；可以等价重写小型算法模式，若复制实质性代码需保留许可证和 attribution。[23][26]
- 本次调研未把任何 Kaggle 数据下载或写入项目，只读取公开/登录可见页面和 Notebook 源码。

## 14. 最终建议

**一次到位的正确方向是：先修复 MSMD/MSMDR 多步目标对齐和 Direct 长度契约，再把 `tsproj_ml` 从“能输出 quantile”升级为“能验证概率预测”，最后做特征和模型消融。**

推荐立即进入的工程目标分两批：

1. 多步语义修复：MSMD/MSMDR `shift_1..H`、Direct shape fail-fast、对应契约测试；
2. 评估闭环：`probabilistic_metrics`、`probabilistic_scores_df.csv`、`horizon_scores_df.csv`、baseline suite 和 skill score。

EWM、weather vintage 和 gap/embargo作为下一批独立消融；global panel、hierarchical reconciliation、GRU/DeepAR 作为后续结构性研究。这样既吸收 Kaggle 的高价值经验，也不破坏项目当前已经建立的严格信息集、A/B 多窗判定和最小改动原则。

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
