# aidc_load_15min_daily 模型测试配置重设计与实施记录（2026-09-01/03）

> 状态：**V3 Latin-square 融合配置已构建（2026-09-03）**。§一至§八保留历次设计与实施过程，不能当作当前目录清单；当前状态见 §九。

## 一、现状盘点（核实于 2026-09-01 仓库现场）

### 1.1 现有结构（route_A/route_B 对称，共 122 模型 YAML + 4 个 ensemble member）

| 组 | 数量/路 | 特征内容 | 策略覆盖 |
|---|---|---|---|
| baseline | 10 | 时间戳 + 目标（lag/rolling/diff），无 datetime | direct/horizon/pointwise/recmo/recursive/blend + 3 线性 + st |
| add_exogenous_weather_date | 8 | + 严格天气(4列) + datetime(10项) | 同上减 st/blend（8 个） |
| add_load_state | 8 | + 11 列负荷状态（observed_past + persistence provider） | 同 exogenous 组 |
| add_endogenous_cross_route_weather_date | 4+4 | MS 系多变量（对路负荷 observed_past）+ 天气 + datetime；US 系单变量对照 | direct/pointwise?/recmo/recursive/blend |
| add_decomposition | 27 | exogenous 组基础上 + 目标变换分解（linear/stl96/mstl96-672） | 9 模型 × 3 分解 |

发现的问题（现状 vs 用户目标）：

1. **概率预测**：全部 lgbm 配置为 quantile(0.1/0.5/0.9) + CQR calibration；线性/st 组为 point。与用户"全部点预测"的目标不一致。
2. **特征不对称**：
   - baseline 组**没有** datetime 特征，但用户定义 baseline = 时间戳+目标特征——"时间戳"信息在 canonical 下由 datetime_features 承载，现状 baseline 与 exogenous 组差异不止外生特征，还夹带 datetime 开关，消融不纯。
   - 节假日特征（date_type）**完全未接入**本场景（exogenous_date_raw/ 数据齐备但零配置引用）；ess 场景有 file-source 先例，chinese_holiday generated source（2026-09-01 落地）零配置使用。
   - 天气现役只用 4 列，源表还有 rt_dt（露点温度）未用。
3. **命名混乱**：
   - 文件名 US*/MS* 缩写是 legacy 迁移标签（usmdp≠pointwise，实为 safe-lag pointwise；usmd=direct；usmr=recursive；usmdr=recmo B=24），canonical 真实身份在 strategy 字段，文件名具有误导性。
   - `-conformal`/`-decomp-*` 后缀与 `probabilistic`/`transformations.target` 段耦合，改点预测后后缀全部失效。
   - `add_load_state` 与用户目标 `add_endogenous`（对路功率/负荷状态）语义不同：load_state 是**本路**状态特征，cross_route 是**对路**负荷。两组特征在用户定义里被合并为"内生协变量"。
4. **重复/冗余**：
   - `add_load_state` 组 8 配置 = exogenous 组同名单 + load_state source，是 exogenous 的纯增量消融，符合设计；但用户新分类里它该并入 add_endogenous（本路状态）还是保留独立组，需裁决（见 §3.2 决策点）。
   - baseline 组 lgbm 6 个策略变体（direct/horizon/pointwise/recmo/recursive/blend）覆盖了策略消融，与特征消融正交，但 horizon 变体（single_model_horizon+align_to_target:false）与 pointwise 变体（single_model_horizon+align_to_target:true+safe-lag）仅在 additive/base 情形一致，其余各组全目录均已带齐 5~6 个策略拷贝，配置量 = 组数 × 策略数（122 个 YAML 中策略轴重复度极高）。
   - add_endogenous 组有 4 个 `-control-no-cross-route` 单变量对照，其特征集与 exogenous 组同名单配置**完全相同**（除 data stem），是跨组冗余。

### 1.2 数据条件（dataset/aidc_load_15min_daily/）

| 数据 | 文件 | 覆盖期 | 状态 |
|---|---|---|---|
| A/B 路负荷 | `A/B_Loads_15min_mean_20251001_20260731.csv` | 2025-10-01~2026-07-31 | ✅ 主数据 |
| 双路合并 | `forecasting_data/AB_Loads_15min_mean_20251001_20260731.csv` | 同上 | ✅ 跨路协变量 |
| 天气实测 | `weather_15min_20250101_20260731.csv`（5 特征列 + available_at） | 2025-01-01~ | ✅ 历史训练 |
| 天气回测代理 | `weather_15min_backtest_proxy_20260101_20260731.csv` | 2026-01-01~07-31 | ✅ 回测 |
| 天气未来代理 | `weather_15min_future_proxy_20260801_20260831.csv` | 2026-08 | ✅ future |
| 日历/节假日 | `exogenous_date_raw/date_in_*.csv`（date_type 1/2/4~9 编码）+ future 08-01~08-14 | 与主数据同窗 | ✅ 未接线 |
| 负荷状态 | `load_state_features/A/B_load_state_history.csv`（15 列 state_*） | 同主数据 | ✅ add_load_state 在用 |
| 事件标签 | `event_label_features/`（lbl_* 有未来信息，禁入特征） | 同主数据 | 仅分析用 |

### 1.3 可复用的 canonical 能力

- `data.sources` 多源声明（target/observed_past/known_future/generated 角色投影 + as-of 可得性）——天气/日期/状态/对路负荷全部有现役通路。
- `chinese_holiday` builtin generator（`data_loading/holiday_generator.py`，BUILTIN_GENERATORS 自动注册，runtime 自动合并）——节假日一键接入，逐点精确匹配 15min 网格，带 VisibilityProof。**现役零配置使用，本场景可首发**。
- `transformations.target.decomposition`（linear/stl96/mstl96-672 三种现役参数）。
- `model_ensemble` 引用式融合（averaging/linear_blending 现役 24 配置有先例；members 走 config_ref + ensemble_members/ 子目录）。
- canonical 七策略：direct / recursive / recmo(B=24) / pointwise(safe-lag direct) / horizon 变体。
- 严格天气信息集三段路径（history actual / backtest proxy / future proxy）已验证。

## 二、新测试矩阵设计

### 2.1 特征组定义（用户裁决的五个特征组 + 融合轴）

| 组 | 特征 = 前组 + 增量 | 增量内容 | 数据载体 |
|---|---|---|---|
| `baseline` | 时间戳 + 目标 | datetime_features(10项) + 目标 lag/rolling/diff | 单路 CSV |
| `add_exogenous` | + 气象 + 节假日 | weather(rt_tt2/cal_rh/rt_ssr/rt_ws10/rt_dt 5列，严格三段) + chinese_holiday(is_holiday/next_holiday_days) | + 天气三段 + generated |
| `add_endogenous_cross_route` | + 内生协变量（对路功率） | 对路负荷(observed_past + persistence provider)，多变量建模 | + AB 合并宽表 |
| `add_endogenous_state` | + 内生协变量（负荷状态） | 本路 11 列 state_*(observed_past + persistence) | + state 表 |
| `add_decomposition` | + 时间序列分解 | transformations.target.decomposition 三变体 | 复用 add_exogenous 基础 |

设计取舍说明：

- **节假日进 add_exogenous**（用户定义"节假日外生特征"）：首选 `chinese_holiday` generated source（零数据文件、逐点精确、自带可得性证明）；date_type file source 作为备选（需先验证日粒度行在 15min 网格的精确匹配行为，风险见 §4）。
- **add_endogenous 拆为两个独立组**（用户裁决）：`add_endogenous_cross_route`（仅对路负荷，多变量）与 `add_endogenous_state`（仅本路状态）。两个组都基于 add_exogenous 特征集，增量各自独立、目录各自隔离——增量归因天然可分，代价是组数 +1。
- **add_decomposition 基座 = add_exogenous**（不带内生）：与现状一致（分解组在 exogenous 基础上），保持"分解 vs 端到端"问题纯净；若要检验"分解+内生"交互，属扩展项。

### 2.2 模型/策略轴（全部点预测）

每个特征组的固定策略集（每路）：

| 配置名（新命名） | strategy | estimator | 定位 |
|---|---|---|---|
| `lgbm_direct` | direct(independent_models, align_to_target:false) | lightgbm | 主力 |
| `lgbm_direct_horizon` | direct + single_model_horizon + horizon_feature(align_to_target:false) | lightgbm | 共享模型变体 |
| `lgbm_direct_pointwise` | direct + single_model_horizon + safe-lag(align_to_target:true, min_lag=96=H) | lightgbm | 逐点长 lag 变体 |
| `lgbm_recmo` | recmo(B=24) | lightgbm | 混合策略 |
| `lgbm_recursive` | recursive | lightgbm | 递归对照 |
| `enet_direct` / `ridge_direct` / `lasso_direct` | direct | enet/ridge/lasso | 线性对照（feature_scaling: minmax） |
| `st_recursive` | recursive | seasonaltemplate | 季节模板基线（仅 baseline + decomposition 组） |
| `lgbm_ensemble_averaging`（融合轴，见 2.4） | ensemble(members: direct+recursive) | — | 融合对照 |

命名规则（丢弃 US*/MS*）：

```
{model}_{strategy}[_{variant}].yaml
策略短码：direct / direct-horizon / direct-pointwise / recmo / recursive
变体短码：-decomp-{linear|stl96|mstl96-672}（仅分解组）
组合名：  lgbm_ensemble_averaging.yaml（引用式，members 用 config_ref 指向组内 direct/recursive）
```

- `setting_suffix` 一律空串（组目录已表达特征差异；分解变体后缀保留 `-decomp-*`）；
- `scenario_subpath` 同步为新组目录名；
- 旧缩写 → 新名映射：usmd→direct、usmd+horizon→direct-horizon、usmdp→direct-pointwise、usmdr→recmo、usmr→recursive、msmd→(多变量)direct、msmr→(多变量)recursive、msmdr→(多变量)recmo、usbr/msbr→ensemble。

### 2.3 概率 → 点预测

所有配置 `probabilistic: {mode: point}`；删除 quantiles/point_quantile/crossing/intervals/calibration 全部字段。
预计连带影响：`-conformal` 后缀消失；`test_scores_probabilistic_df.csv` 不再产出；点评估 test_scores_df.csv 口径不变。

### 2.4 模型融合（用户裁决：四种方法全部设计）

每路每特征组 4 个融合配置（引用式，复用组内成员），成员结构统一：

```
lgbm_ensemble_{averaging|weighted|linear-blending|stacking}.yaml
  ensemble.members: [direct(config_ref: lgbm_direct.yaml),
                     recursive(config_ref: ensemble_members/lgbm_recursive_member.yaml)]
  oof: {train_window_steps: 30, fold_count: 1, stride_steps: 1}  # 沿用现役
```

| 方法 | method.name | 权重学习 | 本场景适配性 |
|---|---|---|---|
| 等权平均 | `averaging` | 无（1/K） | 零学习成本，少窗口稳健基准 |
| 逆误差加权 | `weighted` | OOF 逐 target 反误差权重（RMSE 默认，可 mae/mape） | 5 窗 OOF 样本少，权重噪声风险中等 |
| 线性混合 | `linear_blending` | OOF 上 NNLS 单纯形系数（point 模式） | 同上，非负约束天然防负权重 |
| Ridge 堆叠 | `stacking` | OOF 逐 target Ridge(alpha=1.0) 元模型 | point-only（本场景全点预测正好满足）；参数固定不可调 |

设计约定：

- 融合配置放 4 个特征组（baseline / add_exogenous / add_endogenous_cross_route / add_endogenous_state；分解组不融——分解是目标变换，融分解成员的问题定义混叠）；
- 4 方法 × 4 组 × 2 路 = **32 个融合配置**；
- 成员 = 组内同特征版 direct + recursive（`ensemble_members/` 子目录，canonical 单模型 YAML）；
- 融合产物 identity 自动 `ensemble-{method}-...`，四方法天然目录隔离；
- 已知风险：weighted/linear_blending/stacking 都依赖 OOF 权重/系数学习，本场景仅 5 个回测窗（OOF 样本量小），历史记录（ess、2026-08 消融）显示学习型融合在少窗口下曾全面劣化——四方法并列设计正是为了在同一数据条件下实测这一结论是否复现。

### 2.5 矩阵规模（按裁决后设计）

| 组 | 每路配置数 | 两路合计 |
|---|---|---|
| baseline | 9 单模型（5 lgbm + 3 线性 + st） + 4 ensemble = 13 | 26 |
| add_exogenous | 8 单模型（5 lgbm + 3 线性） + 4 ensemble = 12 | 24 |
| add_endogenous_cross_route | 4 单模型（多变量 direct/direct-pointwise/recmo/recursive） + 4 ensemble = 8 | 16 |
| add_endogenous_state | 8 单模型（5 lgbm + 3 线性） + 4 ensemble = 12 | 24 |
| add_decomposition | 9（lgbm 三主力 × 3 分解，线性/st 不做变体） | 18 |
| **合计** | **54/路** | **108** |

较现状 122：conformal/quantile 轴消失、cross-route 4 个 control 冗余删除、decomposition 变体 27→9；增加：四方法融合 16 组配置（现状 USBR/MSBR 仅 2 个）、load_state 独立成组。

## 三、决策记录（2026-09-01 裁决）

| # | 决策点 | 裁决 | 来源 |
|---|---|---|---|
| 1 | load_state 归属 | **拆为两个独立组**：`add_endogenous_cross_route`（仅对路负荷）+ `add_endogenous_state`（仅本路状态） | **用户显式选择**（2026-09-01 复议） |
| 2 | 节假日载体 | `chinese_holiday` generated source | **用户显式选择**（2026-09-01 复议） |
| 3 | decomposition 基座 | add_exogenous（不带内生） | **用户显式选择**（2026-09-01 复议） |
| 4 | 分解变体范围 | 收敛为 9 个：lgbm 三主力（direct/recmo/recursive）× 3 分解；线性与 st 不做变体 | **用户显式选择** |
| 5 | 融合方法 | **四种方法全部设计**（averaging/weighted/linear_blending/stacking） | **用户显式选择** |
| 6 | rt_dt 露点温度 | 纳入 weather source（known_future 5 列） | **用户显式选择** |

## 四、合理性分析（需求 2）

### 4.1 合理的部分

1. 五个特征组递进与用户定义一致（endogenous 家族拆 cross_route/state 两支），消融路径清晰：外生信息 → 内生信息 → 结构先验（分解），每组回答一个独立问题。
2. baseline 纳入 datetime 特征后与"时间戳+目标特征"定义对齐，消融比现状更纯（现状 baseline 组 datetime 开关与 exogenous 组混淆）。
3. 全点预测大幅降低训练成本（quantile ×3 消失，direct 组拟合时间约降 2/3），且与"模型选型/特征消融"的问题匹配。
4. 融合作为独立轴（每组成员 = 组内单模型）是干净的加法：融合对比的是"组合 vs 最优单成员"，不与特征消融混叠。

### 4.2 遗漏与需裁决的决策点（已裁决，见 §三）

1. **add_load_state 的归属** ✅ 裁决=拆为两个独立组 `add_endogenous_cross_route` + `add_endogenous_state`（用户显式选择，2026-09-01 复议）：对路负荷与本路状态是两个独立增量，各自成组、目录隔离，增量归因天然可分；代价是特征组 4→5、融合配置相应增加。
2. **节假日特征载体** ✅ 裁决=chinese_holiday generated source（用户显式选择，2026-09-01 复议）：`chinese_holiday` generated source（推荐，零文件、逐点精确）vs `date_type` file source（ess 先例，但日粒度行在 15min 网格精确匹配需验证 registry 行为）。已裁决取前者（chinese_holiday），date_type 不接入。
3. **add_decomposition 的基座** ✅ 裁决=add_exogenous（不带内生）（用户显式选择，2026-09-01 复议）：保持「分解 vs 端到端」问题纯净，不带内生协变量。
4. **分解变体数量** ✅ 裁决=收敛为 9 个（用户显式选择）：现役 3 种（linear/stl96/mstl96-672）×8 模型 = 24 配置偏重。可收敛为 lgbm_direct/recmo/recursive 3 个主力 × 3 分解 = 9（线性/st 不做分解变体——线性模型对趋势项处理有独立参数化，收益低）。已按收敛落地（27→9）。
5. **融合方法** ✅ 裁决=四种方法全部设计（用户显式选择）：averaging 现役先例最多；linear_blending 在 ess 场景有劣化记录（少窗口权重不稳）。四种方法并列进矩阵（见 §2.4），用于实测少窗口下学习型融合的历史劣化结论是否复现。
6. **rt_dt（露点温度）** ✅ 裁决=纳入（用户显式选择）：天气源表已有但现役未用。湿度受温度调制，露点是更直接的水汽指标，已纳入 add_exogenous 的 weather source（known_future 第 5 列），零成本。
7. **事件标签（lbl_\*）不进任何特征**：含居中窗口=未来信息，仅作离线样本分析——与现状一致，维持。

### 4.3 冗余（相对现状削减）

- cross-route 组 4 个 `-control-no-cross-route` 配置删除：它们 = add_exogenous 组同名配置换 data stem，控制变量角色由 exogenous 组承担；
- conformal 变体轴整体删除（quantile → point 后自动消失）；
- 线性模型不做 decomposition 变体（决策 4 已裁决：收敛为 lgbm 三主力）；
- st 只在 baseline（模板基线定位）与 decomposition（分解后模板是否受益）出现。

## 五、实施风险与前置验证

1. **chinese_holiday × 15min 网格**：generator 逐点精确匹配，holiday 边界日（调休补班）值来自日历，无风险；但需先跑 1 个配置验证 generated source 在回测 5 窗 + future 的覆盖断言。
2. **多变量 + 状态特征叠加**：add_endogenous 组 data.sources 达 4 个（target+AB 合并、weather、state），observed_past_lags 键数量 12（B_load + 11 state），特征膨胀需盯 direct 组拟合时间（历史实测 single 组 ~3 min/配置量级，可接受）。
3. **结果目录清理**：新 naming 落地后 `scenario_subpath` 全变，现有 `results/aidc_load_15min_daily/route_A/baseline/`（conformal 实跑产物）按"配置组重跑后清理旧 setting"惯例需归档或删除（fingerprint 已变，旧结果不可比）。
4. **文档同步**：`模型测试说明.md` 需整体重写（组定义/命名/概率口径全变）。

## 六、实施步骤（裁决完成，待启动）

1. ~~用户裁决决策点~~（已完成，见 §三）；
2. 生成 route_A 全部新配置（脚本化：从现役配置映射改造 + 批量注入 directories/point 概率段）；
3. `check_model_configs.py` dry-run 全量 + `load_yaml_config` 断言；
4. 抽 1 个配置实跑验证（含 chinese_holiday generated source 的覆盖断言）；
5. route_B 镜像生成 + 全量校验；
6. 重写 `模型测试说明.md`；
7. 清理旧结果目录（用户确认范围后）。

## 七、实施结果（completed 2026-09-02）

### 7.1 最终矩阵

- 三个场景 `aidc_load_15min_{daily,rolling,short}` × A/B 两路采用同一六组结构：`baseline`、`add_exogenous`、`add_endogenous_cross_route`、`add_endogenous_state`、`add_decomposition`、`add_ensemble`。
- 每路 41 份现役 YAML：9 + 8 + 3 + 8 + 9 + 4；每场景 82 份，三个场景合计 **246 份**。
- 全部使用 `probabilistic.mode: point`；天气为 5 列严格三段 source；节假日使用 `chinese_holiday` generated source；旧 US*/MS* 文件名已退出三个场景。
- 模型融合仅位于 `add_ensemble/`，四个方法直接引用 `../add_exogenous/lgbm_direct.yaml` 与 `../add_exogenous/lgbm_recursive.yaml`；不维护 `ensemble_members/` 重复副本。

### 7.2 相对方案正文的执行偏差

| 偏差 | 最终处理 | 原因 |
|---|---|---|
| 融合原计划按四个特征组铺开 32 份 | 收敛为独立 `add_ensemble`，每路 4 份，基座固定 add_exogenous | 用户 2026-09-02 追加裁决：融合配置集中存放，避免融合轴与特征组乘积膨胀 |
| cross-route 中间产物名为 `lgbm_recmo.yaml` | 更名为 `lgbm_mimo.yaml` | YAML 的真实 `strategy.name` 是 `mimo`；按 canonical 真实语义修复文件名 |
| 融合初期保留专用 recursive member | 删除副本，直接引用 add_exogenous 现役 recursive | 用户纠正：不得恢复 `ensemble_members`；相同语义只保留一个事实源 |
| 设计稿预计 54 份/路 | 最终 41 份/路 | 融合不再乘四组、cross-route 仅保留 direct/mimo/recursive 三个多变量模型 |

### 7.3 可重复维护与验证

- `scripts/generate_load_15min_matrix.py` 默认只读校验完整 246 份矩阵；`--write-derived` 只重写 24 个 `add_ensemble` 顶层 YAML，不覆盖人工审定的基础单模型。
- 配置合约验证覆盖：严格文件清单、全部 point、`scenario_subpath`、文件名/strategy 对齐、ensemble member resolve 与 source subset。
- 2026-09-02 验证基线：矩阵脚本 `single=222 ensemble=24 total=246`；项目配置 checker 与全量 unittest 的最终结果以本次工作结束汇报为准。

## 八、V2 全面收敛设计（2026-09-02，已完成）

### 8.1 触发原因

对 222 份单模型执行真实 `FeatureCompiler` 训练设计编译后，204 份通过、18 份
`add_endogenous_cross_route` 全部因同一 source 混合 target/observed-past 的行缓存
碰撞而失败。进一步审计发现：现有 246 份矩阵配置轴较宽，但外层回测仅覆盖
daily/rolling 最后 5 天、short 连续约 28 小时；24 份 ensemble 的 OOF 均为
`train_window_steps=30/fold_count=1/stride_steps=1`，不足以支撑学习型融合结论。

另有两项消融混叠：state-only 组使用了由对路负荷计算的
`state_route_diff_pct`；cross-route Direct 未继承 add_exogenous Direct 的
rolling/difference。V2 先修运行和归因，再用较少配置覆盖更关键的能力。

### 8.2 已批准目标矩阵

- `route_A/route_B` 保留六组结构；daily/rolling 各 37 份/路，short 36 份/路，
  两路共 220 份。
- 每场景新增 `route_AB/add_endogenous_joint/`，包含 K=2 MIMO 与 Recursive 各一份，
  共 6 份。
- 三场景最终 **226 份**：202 单模型 + 24 ensemble；全仓预计 **687 份**：
  657 单模型 + 30 ensemble。
- 删除 74 份冗余单模型，新增 54 份关键配置；删除范围已由用户在
  “全面收敛”选项中显式批准。

### 8.3 配置轴调整

1. baseline 删除 Lasso，保留 ElasticNet 作为唯一稀疏线性基准。
2. add_exogenous 删除 ElasticNet/Lasso，保留 Ridge；新增 weather-only、
   holiday-only、XGB Direct、MIMO、DirRec、DirMO、DirRecMO，使七种 canonical
   策略在统一完整外生基座上可比较。
3. state 删除三线性模型、Direct horizon、Direct pointwise；保留 LGBM
   Direct/RecMO/Recursive，并新增泄漏安全的 `lgbm_direct_selected`。
   state-only 不再声明 `state_route_diff_pct`。
4. decomposition：daily/rolling 收敛为 Direct×linear/STL/MSTL +
   RecMO×MSTL + Recursive×MSTL（5 份/路）；short 不使用周周期 MSTL，保留
   Direct×linear/STL + RecMO×STL + Recursive×STL（4 份/路）。
5. ensemble 四方法全部保留，成员扩为 LGBM Direct、LGBM Recursive、
   XGB Direct，不创建 `ensemble_members`。

### 8.4 验证几何

| 场景 | history_steps | train_window_steps | fold_count | stride_steps | OOF |
|---|---:|---:|---:|---:|---|
| daily | 6336 | 2784 | 31 | 96 | 2784 / 5 / 96 |
| rolling | 6336 | 2784 | 31 | 96 | 2784 / 5 / 96 |
| short | 5072 | 1424 | 31 | 96 | 1424 / 7 / 96 |

short 改为每天同一时刻抽取 4 小时窗口，避免连续 28 小时样本把时段差异误当成
模型稳定性。增大的 history 同时为外层最早窗口之前的嵌套 OOF 保留完整训练窗。
实施期按真实折几何校正 daily/rolling history：原设计 6240 会使最早 nested OOF
第一折只有 2689 个训练 origin；增加一个 H=96 的标签隔离段后取 6336，全部恢复
为 2784。

### 8.5 验收边界

本批次必须通过配置加载、资产审计、全部 202 份单模型真实训练设计编译、六份
K=2 训练/部署设计和全量 unittest。226 份配置的完整 31 折训练属于后续实验批次，
本次不以“全矩阵已完成模型评估”作为完成声明。详细步骤见
`.hermes/plans/2026-09-02-aidc-load-15min-matrix-v2.md`。

2026-09-02 fresh 验收证据：

- 矩阵门禁：`single=202 ensemble=24 total=226`；
- 真实训练设计门禁：`compiled_count=202 failure_count=0`；
- 全量回归：`Ran 669 tests in 303.370s`，`OK`；
- `compileall` 与 `git diff --check` 均为 exit 0。

## 九、V3 固定三模型 × 三策略 Latin-square 融合（2026-09-03）

### 9.1 用户裁决

- 每个 Ensemble 固定三个成员模型：ST、LightGBM、Ridge；
- 预测策略固定为 Direct、Recursive、MIMO；
- 特征固定为 baseline 的 target lag + rolling + expanding + datetime；
- 不使用天气、节假日、状态、跨路协变量或时间序列分解；
- 每个成员组分别使用 averaging、weighted、linear_blending、stacking；
- 本批次只构建和静态校验配置，不运行训练、回测或预测。

### 9.2 实施矩阵

每路 baseline 提供 9 个普通成员（3 模型 × 3 策略）。三组 Latin-square 为：

| 组 | ST | LightGBM | Ridge |
|---|---|---|---|
| Latin A | Recursive | MIMO | Direct |
| Latin B | Direct | Recursive | MIMO |
| Latin C | MIMO | Direct | Recursive |

每组 × 四种融合方法 = 12 个 Ensemble/路。daily/rolling 各 50 份/路并另有
2 份 route_AB，场景各 102 份；short 为 49 份/路并另有 2 份 route_AB，场景
100 份。三个场景合计 304 份（232 单模型 + 72 Ensemble）。

### 9.3 固定特征合同

- daily/rolling：lags `[96,192,288,384,480,576,672]`，rolling windows
  `[96,192,384,672]`；
- short：lags `[1,2,3,4,8,12,16,96,192,672]`，rolling windows
  `[16,96,672]`；
- rolling stats 固定 `mean/std/min/max`，expanding stats 固定 `mean/std`，
  datetime 固定 10 项；不保留 difference；
- Ridge 单独使用 MinMax 缩放；ST 虽接收同一编译输入表，但模型实现只消费 lag
  与可选 day-of-week。

### 9.4 验证边界

本批次只允许执行 YAML 解析、矩阵数量/字段合同、成员引用、Source 资产和 manifest
静态审计。按用户要求未运行任何 estimator fit、rolling backtest、forecast 或真实设计
编译门禁；因此“配置已构建”不等于“模型效果已验证”。

2026-09-03 静态验收证据：

- 矩阵：`single=232 ensemble=72 total=304`；
- 配置 checker：`checked=765 passed=765 hard_failures=0 warnings=0`；
- 资产审计：765 个配置，缺失路径/声明列均为 0；
- Ensemble 审计：681 ordinary single + 6 member + 78 ensemble，228 个引用，
  `failures=[]`；
- manifest：765 entries，481 个子日单模型；
- 12 个静态定向测试通过。

## 十、V4 六组全因子 + 独立 Latin Ensemble（2026-09-03）

### 10.1 当前矩阵

用户要求六个单模型实验组对特征变体、9 模型、9 策略和分解变体（如有）做
完整物理 YAML 笛卡尔积：

- 每路：baseline 81 + add_exogenous 243 + cross-route 81 + state 81 +
  decomposition 243 = 729 个单模型；
- 每场景：A/B 两路 1,458 + route_AB joint 81 = 1,539 个单模型；
- 三场景：4,617 个单模型。

V3 的 72 个 `add_ensemble` 不属于上述六组乘积，但仍是独立有效实验。实施过程中
曾按“只保留六组”的过度收口设计删除，用户随即要求重建；当前恢复为每路三组
Latin-square × 四方法共 12 个，仍直接引用 baseline 九个成员且无
`ensemble_members/`。因此三个 AIDC 场景当前共 **4,689** 份配置：4,617 单模型 +
72 Ensemble；每场景 1,563 份。

### 10.2 关键合同

- 九模型：ST/Ridge/Lasso/ElasticNet/LightGBM/XGBoost/CatBoost/RandomForest/HistGradientBoosting；
- 九策略/布局变体：Direct pointwise、Direct pointwise+horizon、Direct、Recursive、
  DirRec、DirMO、RecMO、DirRecMO、MIMO；用户输入 `recom` 映射为 canonical `recmo`；
- 两种 pointwise 使用 `single_model_horizon + align_to_target=true`，以 horizon 是否增加
  sin/cos 周期编码区分；
- short pointwise 的 target lag 仅保留 `[16,96,192,672]`，避免 Direct 访问原点后的
  target；rolling/expanding 始终截断在 forecast origin；
- rolling 调度原点为 `2026-07-31T14:00:00`，与天气 13:45 历史截止和 14:00 future
  起点一致；daily 原点为 23:45，short 原点为 14:00；
- 全部为 point mode；不执行训练、回测、预测、OOF 或 bundle smoke。

### 10.3 静态验收

状态：已实施并完成静态验收。

- 生成器：`expected=4689`，`create/rewrite/delete=0`，`unchanged=4689`；其中
  `forecast_configs=4617`、`ensemble_configs=72`；
- 全仓 checker：`checked=5150 passed=5150 hard_failures=0 warnings=0`；
- 资产审计：`model_config_count=5150`，缺失路径、声明列和成员引用均为 0；
- Ensemble 审计：5,066 ordinary single + 6 member + 78 Ensemble，228 个成员引用，
  `failures=[]`；AIDC 有 72 份 Ensemble、0 个 `ensemble_members`；
- manifest：5,150 entries，5,072 single，78 Ensemble，4,866 个子日单模型；
- 文件名/字段审计：4,689/4,689，`failure_count=0`；
- ModelFactory：九种模型默认实例全部构造成功，未 fit；
- 静态定向 unittest：17 + 4 + 1 + 1 = 23 tests，全部 `OK`；
- `compileall -q scripts tests` 与 `git diff --check` 均 exit 0。

按用户要求未运行 `audit_aidc_load_15min_designs.py`、estimator fit、rolling backtest、
forecast、融合 OOF 或 bundle smoke。上述证据只证明配置、引用、资产与静态合同成立，
不证明 4,617 个机械组合可实际训练或有预测收益。
