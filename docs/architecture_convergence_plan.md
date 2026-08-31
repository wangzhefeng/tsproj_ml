# 项目架构收敛设计、Coder Review 与整改方案

> 文档状态：**v4 整改已 `closed`（2026-08-31）**。§1–§10 和 §11 的旧基线保留为历史证据，不得当作当前验收结果；目标架构与实施要求见 §12–§16，二轮 fresh 证据见 §17.5。C1–C6 实现与当前态文档、H16 short + H31 calendar-month 两个 quantile linear-blending 代表配置以及 C7 代码/配置/资产门禁均已通过。**任何整改片均不自动 commit**（review-then-commit）。
> 创建日期：2026-08-29（v1）／ v2：纳入 ensemble v4 现状 ／ v3：新增流水线第一性原理审计并完成原收敛切片 ／ v4：重构后 coder review 与整改方案
> 作者：Machine-C（MC）
> 定位：本文记录架构收敛的历史决策、重构后真实状态审查，以及下一轮修正方案。`docs/` 下实验性 Python/ipynb 按 review 范围明确排除；生产代码、配置、Markdown 文档、测试和运行产物合同均纳入核查。

> **版本变更链**：
> **v1 → v2**：工作区相继完成 canonical legacy 收口与 ensemble v4 两次大重构，撤销 D5（pickle 垫片，存量 pkl 已清空）、改 D2 为整包删除（契约字段词汇在现役 848 YAML 零出现）、缩小 D3 范围（E1 已抽 `forecasting/validation.py`）、新增 §2.5 ensemble 专项分析。
> **v2 → v3**：应用户要求，从「时序预测 训练/测试/预测 第一性原理」对全部 13 个主题逐项审计（§2.7），新增发现 F1–F8 与决策 D12–D15；按 2026-08-29 裁决执行，记录见 §9–§10。
> **v3 → v4**：不采信历史 pass 计数，重新运行全量 unittest、配置审计、compileall、包间 AST 门禁，并对现役 YAML→FeatureCompiler、Quantile Linear Blending、月频 seasonal-naive、Ensemble 顶层合同、CLI override 和 source 文件逐项做可执行探测。结果表明 v3 完成的是原定结构切片，不等于现役生产合同闭环；新增 RF1–RF12 与 C0–C7 整改切片。

## 1. 目的与范围

### 1.1 要解决的问题

ensemble v4 重构再次验证了正确的包间姿势（单向依赖 + Protocol 边界），但它是**绕开**旧包问题建成的孤岛——`models/`、`features/`、`data_provider/` 与 forecasting 之间的结构性问题原样保留。当前五类问题：

1. 数据构造/处理逻辑分散，`data_provider/` 整包是被绕过的孤儿；
2. 特征工程残留半死包 `features/`（feature_engineering 已移入 `docs/` 归档，包本体了断未完成）；
3. 回测/评估职责仍以私有函数形态溶解在 `runtime.py`，与 Trainer/Forecaster 的类形态不对称；
4. 包间存在两个循环依赖（`models ↔ forecasting`、`probabilistic ↔ models ↔ forecasting`），ensemble 包用自己的延迟 import 技巧独善其身，但没有解决存量；
5. **流水线语义缺口**（v3 新增）：canonical 化把「按业务掩码过滤的 MAPE/Accuracy」这一评估语义静默丢掉（F6），且「缺失/异常 → RAISE 的预测性维护默认」从未被声明（F1/F2）——这两项不是代码放置问题，是第一性原理层面的语义完整性问题。

### 1.2 范围内

- 包间分层规则的定义与执行（含 `ensemble/` 的层定位）；
- `models/`、`features/`、`data_provider/`、`utils/` 四个旧包的收敛处置；
- 回测/评估职责的公开化与归一；循环依赖消除（以 ensemble 为模板）；
- ensemble 对 forecasting **私有符号**依赖的公开化（E-B）；
- **流水线审计发现 F1–F8 的处置**（D12–D15，v3 新增）。

### 1.3 范围外（本轮不做，明确排除）

- **canonical 内部设计**与 **ensemble v4 已验收行为**：七策略、specs、张量合同、OOF/缓存/parity golden 机制均刚验收（848 配置全过、555 tests OK），不动语义。
- **`config/<场景>/` 下场景数据脚本**：只立约定（D8），不做存量搬迁。
- **`docs/` 学习资料区**（含 `docs/feature_engineering/` 归档）：非生产代码。
- **CQR 接入**：维持 redesign 文档 D10 结论。

## 2. 现状诊断（证据核查结论，全部基于当前工作区）

### 2.1 数据通路：主链唯一通路已确立，`data_provider/` 是被绕过的孤儿

| 位置 | 内容 | 与主链关系 |
|---|---|---|
| `forecasting/data/`（registry 671 行 + information_set + providers） | 读文件、schema 校验、as-of vintage、known_future/static | **主链唯一通路**（AGENTS.md 已声明） |
| `data_provider/exogenous_contract.py` | 角色切分 + `select_asof_rows`（:107） | 仅 `tests/test_exogenous_information_contract.py` 消费，主链不经过 |
| `data_provider/weather_contract.py` | `strict_weather_information_set`/`weather_*_source`/`weather_backtest_path` 等 legacy 字段校验（:14-35） | **死代码**：生产零调用、测试零引用；字段词汇在现役 848 YAML 中**零处出现** |
| `data_provider/outlier_handling.py` | `enable_train_outlier_handling` 训练期清洗 | 主链不调用；`run.py:173` 将同名 CLI 项列为 legacy 不支持项 |
| `data_process/` | 6 个独立分析工具（聚合/填补回测/事件检测/异常值/峰谷/周期） | 离线场景工具；与主链唯一交叉是 decomposition 反向 import 其私有函数（§2.4） |

关键证据：

- **天气严格信息集的现役载体是 canonical 通用机制**：现役天气 YAML（如 `config/aidc_power_month/route_A/freq_1day/baseline/enet_usmd_mean.yaml:22`）用 `DataSpec` 的 `availability: source_time` + registry as-of 表达。**AGENTS.md:49「月度/日度气象严格信息集」条目仍在描述 legacy 字段词汇，与现役配置脱节**——文档漂移，P0 修正。

### 2.2 特征工程：`features/` 半死包，了断未完成

| 位置 | 内容 | 状态 |
|---|---|---|
| `forecasting/features/compiler.py`（992 行） | FeatureCompiler：可见性证明 + lineage | **主链核心** |
| `features/TargetTransformation.py` | calendar normalization + target scaling 编排 | 活代码但挂错包：被 `forecasting/transforms.py:15` 依赖 |
| `features/FeatureSelection.py` + `DataAugment.py` | — | **死导入**：`models/ModelTraining.py:52-53` import 后正文零使用 |
| `features/FeatureScalering.py::FeatureScaler`（:261 起） | 特征标准化器 | **孤儿**：全仓无消费者（同文件 `TargetScaler` 是活代码）；canonical 特征缩放实际由 `forecasting/transforms.py::CanonicalFeatureScaler`（:263）承担 |
| `docs/feature_engineering/` | fft/wavelet/holiday 等 | 用户已归档出 `features/`，范围外 |

### 2.3 回测/评估：E1 已抽折生成，剩余私有面缩小

E1 已把共享时间合同抽到 `forecasting/validation.py`（`TimeGeometry` + `rolling_origin_folds`，165 行），outer backtest（`runtime.py:13,687`）与成员 runner（`runtime.py:1114`）同源消费。仍存在的问题：

| 原 ModelTesting 职责 | 现居地 | 形态 |
|---|---|---|
| 折生成（滚动滑窗） | `forecasting/validation.py` | **已公开共享** ✓ |
| seasonal-naive 基线 / actual 张量 | `runtime.py::_seasonal_naive_tensor(:724)`、`_actual_tensor(:909)` | 私有 |
| 预测原点解析 | `runtime.py::_resolve_origin(:1558)` | 私有，**且被 ensemble 跨包私有导入**（§2.5 E-B） |
| 点预测评估 | `forecasting/results.py::evaluate_point_forecasts` | 公开 ✓，但掩码语义缺失（F6） |
| 概率评估 | `probabilistic/evaluation.py`(822行) + `metrics.py` | 公开，仅 tests 消费 |
| 回测绘图 | `forecasting/results.py::_plot_backtest` | 私有；`utils/cv_plot.py`/`utils/metrics.py` 为零消费孤儿 |

### 2.4 包间循环（v4 之后原样保留）

```text
forecasting/runtime.py    ──→ models/ (Trainer/Forecaster/Factory/SaveLoad)
models/ModelTraining.py   ──→ forecasting/ (specs, strategies, estimators)      ← 环 1
probabilistic/pipeline.py ──→ models/ModelForecasting.py                        ← 环 2
probabilistic/training.py:102 ──(函数内延迟 import)──→ models.ModelTraining     ← 环 2
forecasting/{runtime,transforms,results}.py ──→ probabilistic/*（顶层）         ← 与环 2 构成三包环
forecasting/transforms.py:15 ──→ features/TargetTransformation.py               ← 目标变换链倒挂
decomposition/residual_diagnostics.py:20 ──→ data_process/_acf_periods,_fft_dominant_period ← 私有倒挂
```

- **环 1**：`models/ModelTraining.py:38,44,45` 与 `models/ModelForecasting.py:9-11` 顶层导入 forecasting；`forecasting/runtime.py:36-44` 反向导入 models。
- **环 2**：`probabilistic/pipeline.py:12` 顶层导入 models；`probabilistic/training.py:102` 函数内延迟导入 models。
- **三包环全景**：forecasting → probabilistic（runtime.py:39-41 等）→ models → forecasting。

### 2.5 `ensemble/` 模块专项分析

包概况：15 个文件、2003 行。已验收：848 配置审计全过、`tests/test_ensemble_parity.py` 锁 golden（单模型 `a8d1ae0a5c44`、weighted 端到端 `e4c3eb2b7dbf`/`bf0677d2a169`）。

**做对了的（全仓包间规则的模板）**：

1. **单向依赖 + Protocol 边界**：`ensemble/` 只向下依赖 forecasting（runtime/specs）与 probabilistic.types，对 `models/` **零依赖**；边界靠 `ensemble/contracts.py` 的 `BaseModelRunner`/`FusionMethod` Protocol（:17,:59）维持，参考实现 `forecasting.runtime.CanonicalBaseModelRunner`（:1083，已入 `__all__`）。
2. **时间合同同源**：OOF 折与外层回测共享 `TimeGeometry`（`runtime.py:1114`），未复制时间数学。
3. **内容寻址 OOF 缓存**（`cache.py`，SHA-256），原子写。
4. **bundle 类型判别**：`model_type="ensemble"`；入口分派清晰（`config_loader.py:118` 互斥字段分派，`main.py:78-83` 明确拒绝、`run.py:260` 承接）。

**存在的问题**：

- **E-B 跨包私有符号依赖**：`ensemble/runtime.py:130` `from forecasting.runtime import _resolve_origin`；:40 经 `forecasting.runtime` 中转取 `SourceRegistry`（应直连 `forecasting.data`）。runtime 的 `__all__`（:1582，仅 3 项）与实际公开面不一致。
- **E-C 折生成双实现并存**：`validation.rolling_origin_folds`（外层回测）与 `ensemble/oof.py::oof_fold_origins`（OOF 特化）两套代码，共享 TimeGeometry 但策略不同：

  | 维度 | validation.rolling_origin_folds | ensemble/oof.oof_fold_origins |
  |---|---|---|
  | 候选池 | 最近 `history_length` 个 origins | 全部 origins 最新优先，无历史上限 |
  | 折数控制 | `max_windows` + stride | `fold_count × stride` 截断 |
  | 独有参数 | — | `gap_steps`（隔离带）、`outer_cutoff_origin`（排除触及外层 holdout 的折） |
  | 无训练样本 | RAISE | 静默跳过（`continue`，oof.py:65） |

  `gap/cutoff/静默跳过` 差异是否有意无文档记录；静默跳过与全仓「不静默降级」纪律存在张力。
- **E-D loader 耦合 canonical YAML 内部结构**：`ensemble/loader.py:115,:135` 直接操作 canonical 配置 payload 键。成员身份已锁定为现役单模型 YAML，当前可接受。
- **E-H 包内私有交叉导入**：`methods/linear_blending.py:13` 导入 `methods/weighted.py` 的 `_target_key/_weight_matrix`。

**层定位结论**：`ensemble/` 与 `probabilistic/` 同属 L3 能力扩展层；L3 内部允许单向 `ensemble → probabilistic.types`。

### 2.6 各包现状一览（含 ensemble 后的全景）

```text
L4 入口      main.py / run.py / config/config_loader.py
L3 扩展      probabilistic/（含死导入环）、ensemble/（干净，依赖 L2 私有符号）
L2 预测      forecasting/（内部正交 ✓；向上被 models 环、向下挂 features 倒挂）
L1 基础设施  models/（Trainer/Forecaster 位置错误 + 环源；Factory/SaveLoad 合法）
             decomposition/（私有倒挂一处 + 死文件 F7）、data_process/（离线数据算法工具）
L0 基础      utils/（log/frequency/runtime_env/eval_mask 活跃；metrics/cv_plot 孤儿待清）
孤儿包       data_provider/（三个模块全部无主）、features/（半死）
```

### 2.7 流水线第一性原理逐项审计（v3 新增）

以「训练 → 测试（回测）→ 预测」的时序预测本质流程为轴，逐项核对当前方案的覆盖度。判定含义：**✓ 合理**（方案已覆盖且设计与第一性原理一致）／**△ 有缺口**（发现 Fx，处置见决策表）。

| # | 用户审计项 | 现状载体（证据） | 第一性原理评估 | 判定 |
|---|---|---|---|---|
| 1 | 数据获取/读取 | `SourceRegistry`（registry.py）：文件读取+schema 校验+as-of vintage；`providers.py`：observed_past 显式 provider / known_future trajectory / static；配置层 `custom_features` 多文件注册表；「获取」由 config/ 场景脚本承担（D8 约定） | 读取必须单入口、严格 schema、预测期严格 as-of——已满足；获取与读取分离（脚本产 CSV → registry 读）是合理边界 | ✓ |
| 2a | 缺失值检测/处理/填充 | 信息集层**显式 RAISE**：列级 NaN RAISE（registry.py:291）、时间戳解析失败 RAISE（:317）、要求序列缺失 RAISE（:397,:604）；特征阶 NaN 由 lag 起点裁剪（`_supervised_arrays`/`_label_start`，runtime.py:536+）天然排除；填补只存在于离线：`data_process/fill_method_backtest.py`（掩码回测裁决） | **预测性维护默认**：进入模型的信息集必须完整，缺数据即失败（不静默填补造成幻觉值）——RAISE 是正确默认；但该默认**从未写成文字**，且离线填补工具与主链的关系（先离线填、后进模型）无声明 | △ F1 |
| 2b | 异常值检测/处理 | 离线：`data_process/outlier_process.py`（标记/清洗/可视化）；`data_process/load_event_detection.py`（事件检测，有单测）；legacy 在线：`data_provider/outlier_handling.py`（主链不调用，CLI 项判 legacy 不支持） | 训练窗口清洗是改数据操作，必须与评估掩码（不改数据）严格分离——legacy 曾有该区分（`EvalMaskConfig ≠ TrainOutlierConfig`）；canonical 后**异常值处理在线通路整体缺位**：`validation.outlier_strategy` 之类配置点不存在 | △ F2 |
| 2c | 数据变换 | target 三段变换 calendar normalization → decomposition → scaling（`transforms.py`，按 `(series_id,target)` 隔离、严格逆序）；特征缩放 `CanonicalFeatureScaler`；孤儿 `FeatureScaler`（§2.2） | 变换必须训练态可保存、推理态可逆——已满足；孤儿双 scaler 是分层违例残留 | ✓（F3 随 D4） |
| 2d | 时序检测（趋势/周期/事件） | 趋势/周期：decomposition（STL 等）+ `data_process/periodicity_analysis.py`、`peak_valley_detection.py`（离线）；事件：`load_event_detection.py`（shift/stress/burst/spike 四类，15min↔日频一致） | 检测结果用途分两类：① 进模型当特征（须无泄漏 trailing）② 仅离线分析/样本筛选（可含居中窗口）——项目已有 `lbl_` 前缀约定管住这个边界 | ✓ |
| 3 | 特征工程（目标/内生/外生/静态/其他/选择） | `FeatureCompiler` 统一通路：列角色 target/observed_past/known_future/static/key/ignored 显式声明，lag/datetime/交互特征可见性证明 + lineage（compiler.py 992 行）；horizon-feature 是 Direct layout | 第一性原理：特征合法性 = 可得性（每个特征在预测原点必须可得），角色化 + as-of 证明是该原理的正确实现；五类特征与五种角色的映射完备；**特征选择在 canonical 是刻意空缺**（GBDT 内生重要性 + 配置期消融），死导入应删 | ✓（F4 随 D15 注） |
| 4 | 模型构造 | `ModelFactory`（989 行，10 种模型类型，L1 工厂）；quantile 目标 `probabilistic/objectives.py`（GBDT 分位损失注入） | 工厂模式正确；estimator 能力显式探测（capabilities.py，不静默降级）符合第一性原理 | ✓ |
| 5 | 模型训练/损失/指标 | `CanonicalTrainer`（models/ModelTraining.py:214）+ 三种 multi-target adapter（independent/chain/native）+ 并行能力探测；损失：point=GBDT 原生、quantile=pinball（objectives.py）；`models/losses.py` scorer 被 ModelTraining:58 消费；GridSearchCV/RandomizedSearchCV import（:34-35）**死代码**（`perform_tuning` 已判 legacy 不支持，run.py:171） | 训练器承担「模型拟合 + 目标变换编排 + 样本裁剪」，位置应在 L2（环 1 源头）；死导入是 legacy 残留 | ✓（F5 随 D15） |
| 6 | 模型测试（回测） | `validation.rolling_origin_folds`（滚动滑窗，`window_length` + stride + 非重叠合同）共享给外层回测与 ensemble OOF；扩展窗口 = `window_length` 取全长（约定级支持）；基线 = seasonal-naive（私有）；折/基线/actual/origin 原语部分私有（D3） | 回测 = 用「过去模拟未来」：折叠生成器必须与训练样本裁剪共享同一时间合同——E1 已做到；滚动/扩展窗口两种形态中扩展窗口未显式参数化 | ✓（私有面随 D3） |
| 7 | 模型评估 | 点：`evaluate_point_forecasts`（MAE/RMSE/MAPE/Accuracy + 聚合加权 `resolve_aggregate_weighting`）；概率：`probabilistic/evaluation.py`（pinball/interval/crossing/prequential）；**评估掩码**：`utils/eval_mask.py::build_eval_mask` 实现完整但**零消费者**；`validation.eval_mask` 在 14 个 YAML 声明、被 `check_model_configs.py:159` 白名单接受、进入 validation payload（即进入 fingerprint），**但 canonical 运行时从不读取**（MAPE 有效性仅 `actual != 0`，results.py:149） | 评估语义 = 业务口径：掩码过滤（低负荷时段不计入 MAPE）是电力负荷预测的核心业务口径，「已完成历史评估全部是掩码后中位数」曾是全仓统一约定——canonical 迁移把它静默丢了。配置活着、审计活着、运行时死了，是**语义回归**而非代码放置问题 | △ **F6（v3 最大发现）** |
| 8 | 模型预测（推理/多步） | `run_canonical_config` final 训练 + `write_forecast_results`；多步 = 七策略（recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo）；预测原点 as-of 校验；CLI 切片清晰（run.py 删 `--is-testing/--is-forecasting`） | 推理与训练必须同一条特征/变换链（compile → transform → predict → 逆变换）——canonical 单链已保证；future 数据经 `future_path` + 严格 as-of | ✓（F1 声明后） |
| 9a | 时间序列分解 | `decomposition/`（spec/pipeline/extractors/registry/bundle 接入主链）；`forecasters.py`（PolyTrend/PhaseTemplate）+ `composers.py`（AdditiveComposer）为**独立分量预测链，全仓零消费者**（grep 复核）；`bundle.py` 部分类型零消费 | 分解作为 target 变换段（减/还原）已闭环；「分量单独建模再合成」的独立链是未完成分支的死代码，留着会误导（仿佛支持该能力） | △ F7 |
| 9b | 多模型融合 | `ensemble/` v4（§2.5）：引用式成员 + OOF + 四方法 + parity golden | 已按第一性原理重构完毕（OOF 折与外层回测同源、防泄漏） | ✓（D9/D10） |
| 9c | 点预测/概率预测 | point `(N,H,K)` / marginal quantile `(N,H,K,Q)`；crossing repair（probabilistic/pipeline.py:62）；CQR 显式不支持（文档声明） | 边际分布 vs 联合分布的边界已被类型合同管住（samples 显式 unsupported）——诚实 | ✓ |
| 10 | 结果保存 | schema-2 `ForecastModelBundle`（estimator+transform 状态+lineage+fingerprint）；long schema 结果（`(series_id,time,target[,window])` 唯一键）；OOF 内容寻址缓存；`ModelSaveLoad` pickle protocol=2；PMML 类零消费者 | 产物必须自足可复载（bundle 含逆变换状态）——已满足；PMML 是死路径 | ✓（D7） |
| 11 | 模型配置 | schema_version:2 严格解析（未知字段 RAISE）+ 语义 fingerprint + 双审计脚本（848 全过）；ensemble 文档式 loader | 「配置即实验的完整描述」已达成；**唯一例外：`validation` 段是无类型 Mapping**（specs/config.py:185），与六 spec 正交类型化的设计不一致——本轮回不展开，记录在案 | ✓（注） |
| 12 | 单元测试 | 64 文件 / 555 tests + parity golden + 门禁测试（断言已删目录不存在）+ 双审计脚本；**缺**：包间循环依赖门禁、公开面（`__all__`）门禁 | 测试金字塔完整；按 AGENTS.md 纪律测试必须随模块同步维护——P1 加循环依赖门禁测试固化收敛成果 | ✓（P1 补门禁） |
| 13 | 补充项 | 训练/验证切分（holdout，stacking 权重嵌套验证段消费）；时间对齐（TimeGeometry 统一 offset/label 边界）；Global panel `(series_id,time)` 主键；日志/运行时环境（utils/log_util、runtime_env）；可视化（回测/预测图，results.py 私有绘图） | 训练/验证/回测三层数据切分边界由同一时间合同保证——已闭环；绘图私有化不影响语义，随 D3 顺带 | ✓ |

### 2.8 审计发现明细（F1–F8）

- **F1 缺失处理「预测性维护默认」未声明**：信息集层 RAISE（registry.py:291,317,397）+ 特征 lag 起点裁剪，行为正确但无文字契约；离线填补（`fill_method_backtest.py` 掩码回测）与主链的先后关系（离线填好 → 进模型）未声明。线上 as-of 语义下的任何填补都必须走 config 驱动的 compiler 前置步骤，不允许 ad-hoc。→ **D12**。
- **F2 异常值在线通路缺位**：canonical 配置无 `outlier_strategy` 类字段，训练窗口清洗不可配置（legacy `outlier_handling` 无主）。低风险： RAISE 默认下这是「数据质量前置」哲学的自洽结果；但 `outlier_handling` 文件必须明确归宿与身份。→ **D12**。
- **F3 孤儿 `FeatureScaler`**：与 `CanonicalFeatureScaler` 构成双特征缩放实现（一个无消费者、一个在主链），分层上 L0/L1 包承载 L2 语义。→ 随 **D4** 删除。
- **F4 特征选择空缺**：canonical 无特征选择步骤（设计使然：GBDT + 配置期消融），`FeatureSelection.py` 是死导入。保留文件只会暗示能力存在。→ 随 **D15** 删导入、**D4** 删文件。
- **F5 GridSearchCV 死导入**：`models/ModelTraining.py:34-35` import 后零使用（`perform_tuning` 已被 run.py:171 判 legacy 不支持）。→ **D15**。
- **F6 评估掩码语义静默丢失（v3 最大发现）**：`validation.eval_mask` 三重活着（14 个 YAML、审计白名单 check_model_configs.py:159、validation payload 进 fingerprint）+ 一重死了（运行时零读取；`utils/eval_mask.py::build_eval_mask` 零消费者；MAPE 有效性仅 `actual!=0`，results.py:149）。业务影响：这 14 个配置的历史评估分数是**未按业务口径过滤**的，且配置间 fingerprint 差异来自一个不生效的字段。→ **D13**（rewire 或 delete，语义方向相反，需 wangzf 裁决）。
- **F7 decomposition 双链死代码**：`forecasters.py`（PolyTrendForecastForecaster/PhaseTemplateForecaster）+ `composers.py`（AdditiveComposer）为「分量独立预测再合成」独立链，全仓零消费者；主链只用 spec/pipeline/extractors/registry/bundle。死链暗示不存在的第二模式。→ **D14**。
- **F8 分层图方向勘误（v2 文字缺陷）**：v2 §3.1 写「依赖只允许从上往下（L4→L3→L2→L1→L0）」，图中 forecasting(L2)→models(L1) 是合法箭头；真正的违例是 models 内部的**点状逆向**类型检查导入（`ModelSaveLoad.py:74` probabilistic.types、`:103` decomposition.bundle——L1→L3/L2），D1 落地时应改为 duck-typing（isinstance 检查降级为属性探测或移除），否则环 2 消除不彻底。

## 3. 目标架构

### 3.1 分层规则（唯一权威，P0 写入 AGENTS.md）

```text
L4  入口层      main.py / run.py / config/config_loader.py（按 schema 分派 Forecast|Ensemble）
L3  能力扩展层  probabilistic/（概率 executor + 独立评估工具）
                ensemble/（融合：specs/loader/contracts/oof/cache/methods/trainer/predictor/runtime）
L2  预测层      forecasting/（specs/tensors/validation/data/features/strategies/estimators
                /trainer/forecaster/backtest/runtime/results/transforms）
L1  基础设施层  models/（收缩为 estimator 工厂 + 序列化）、decomposition/、data_process/
L0  基础层      utils/（log/frequency/runtime_env；无项目内依赖）
```

规则：

1. 依赖只允许**从上往下**（高层 import 低层；即 L4→L3→L2→L1→L0 方向），同层禁止互依；唯一豁免：L3 内 `ensemble → probabilistic.types` 单向。L2 的 forecasting 调用 L1 的 models 工厂是合法箭头；**低层不得反向 import 高层**——包括函数内延迟 import（F8：`ModelSaveLoad` 的类型检查导入随 D1 改 duck-typing）；
2. `forecasting/` 是 L2 唯一包；目标变换（现 `features/TargetTransformation.py`）并入其中；
3. 跨包复用的算法放 L1 并以**公开函数**暴露；禁止下划线私有跨包导入——ensemble 现有两处（E-B）随本规则一并修复；
4. 任何循环依赖视为架构缺陷，出现即修，不允许函数内延迟 import 长期绕开；
5. 新增能力包一律按 ensemble 模式建：独立包 + Protocol 边界 + 单向依赖 + parity golden；
6. **流水线数据契约**（v3）：进入模型的信息集默认「缺失/异常 = RAISE」（预测性维护）；填补与清洗只允许发生在离线数据准备阶段（`data_process/`，场景维度），或以 config 驱动的 compiler 前置步骤引入（须满足 as-of 可得性）；训练窗口内清洗与评估掩码严格分离（前者改数据、后者只改评估口径）。

### 3.2 各包目标职责

| 包 | 收敛后职责 | 处置摘要 |
|---|---|---|
| `forecasting/` | 现有内容 + **trainer.py / forecaster.py**（自 models 迁入）+ **backtest.py**（基线/actual/origin 公开化）+ eval_mask 评估口径（按 D13） | 环的 L2 侧收敛；公开面补齐（E-B） |
| `ensemble/` | 不动结构；仅随 P3 修复两处私有导入 | 层定位 L3 写入文档 |
| `probabilistic/` | 保留双身份：主链 quantile executor + 独立评估工具库 | 环的 L3 侧收敛 |
| `models/` | 收缩为 estimator 工厂与序列化：ModelFactory、ModelSaveLoad（删 PMML 类）、learning_rate、losses | Trainer/Forecaster 迁出后环消；类型检查导入改 duck-typing（F8） |
| `features/` | **整包删除**：TargetTransformation 并入 forecasting/transforms，删死导入与 FeatureScaler | D4 |
| `data_provider/` | **整包删除**：outlier_handling 并入 data_process（声明 legacy 身份）；两个契约模块为 legacy 死代码直接删 | D2 + D12 |
| `data_process/` | **离线数据准备工具库**（场景维度：聚合/填补/清洗/事件/峰谷/周期），README 声明「进模型前」定位与 F1 默认的关系；吸收 outlier_handling；`_acf_periods/_fft_dominant_period` 提公开名 | D12 + 消除私有倒挂 |
| `decomposition/` | 保留主链通路（spec/pipeline/extractors/registry/bundle）；删 forecasters/composers 死链 | D14 + 导入修正 |
| `utils/` | 删 metrics.py、cv_plot.py；eval_mask 按 D13 裁决（rewire 则留并接线，delete 则删） | D6 + D13 |

### 3.3 目标依赖图（收敛后，无任何反向箭头）

```text
main/run ──→ config_loader ──→ { forecasting, ensemble }
ensemble ──→ forecasting ──→ models(factory/pkl)          （L3→L2→L1）
ensemble ──→ probabilistic.types                           （L3 内单向豁免）
probabilistic ──→ forecasting                              （L3→L2）
forecasting ──→ { decomposition, data_process, utils }     （L2→L1/L0）
features/ data_provider/ 不存在
```

## 4. 决策表

**裸「确认」= 全部采纳推荐项。** v2 已撤销项：pickle 垫片（存量 pkl 已清空）、契约合并（改为直接删除）、weather 专项核实（已核实，见 §2.1）。

> **2026-08-29 裁决（wangzf，全部采纳推荐项）**：D13=rewire｜D12=只立契约｜D7/D14/D15=全部删除｜D9=文档声明差异为有意｜切片顺序=P1 消环先行。裁决明细见 §9。

| # | 决策 | 推荐 | 彻底选项（备选） | 代价对比 |
|---|---|---|---|---|
| D1 | `CanonicalTrainer/Forecaster` 去留 | **迁入 `forecasting/trainer.py`、`forecasting/forecaster.py`**，models 收缩为工厂+pkl；`ModelSaveLoad` 的类型检查导入（:74,:103）改 duck-typing 或移除（F8 收口） | 整个 `models/`（含 Factory 989 行）并入 `forecasting/estimators/` | 推荐：改 4 个生产 import 文件 + 7 个测试文件；彻底：pkl/工厂消费面一并翻动，收益边际。v4 之后此项是环的唯一成因，必须做 |
| D2 | `data_provider/` 处置 | **整包删除**：`outlier_handling` → `data_process/`（保留测试）；`exogenous_contract`、`weather_contract` 为 legacy 死代码删除（exogenous 契约测试同步删，as-of 唯一实现在 registry） | 保留 exogenous_contract 迁入 forecasting/data 供场景脚本用 | 两契约字段词汇在现役 848 YAML 零出现、主链已由 DataSpec+registry 覆盖其语义；保留只会延续两套 as-of 的错觉 |
| D3 | 回测公开化形态 | **新建 `forecasting/backtest.py`**：`seasonal_naive_baseline / actual_tensor / resolve_origin` 自 runtime 公开化迁入，runtime 改为调用方并修复 ensemble 私有导入（E-B）；`SourceRegistry` 消费方改从 `forecasting.data` 直连 | 只把 `_resolve_origin` 提公开名 | 推荐项约 150 行纯搬运 + ensemble 两行 import 修正，行为零变化；备选项留下划线跨包导入持续违反规则 3 |
| D4 | `features/` 处置 | **整包删除**：TargetTransformation 并入 `forecasting/transforms.py`，删 `FeatureScaler`（F3）、`ModelTraining.py:52-53` 两个死导入（F4） | 保留包名仅挪文件 | 包内活代码只剩 TargetTransformation；备选项保留单文件包名，无意义 |
| D5 | ~~pickle 兼容垫片~~ | **撤销（v1→v2）**：results/ 存量 pkl 为 0，类迁移无兼容对象 | — | 如迁移与重跑窗口重叠再议 |
| D6 | `utils/metrics.py` + `cv_plot.py` | **删除**（零消费复查通过）；指标终态 = `forecasting/results.py`（点）+ `probabilistic/metrics.py`（概率）两处 | 接线复用 | 接线方向反了：L2 依赖 L0 领域实现会让 utils 变成第二指标库 |
| D7 | `ModelSaveLoad` 卫生 | **删除 `ModelDeploy`/`ModelDeployPmml`**（全仓零消费者），连带移除 sklearn2pmml/pypmml import 与依赖声明；保留 `ModelDeployPkl` | 仅 PMML import 下沉 | PMML 整条路径零消费，留类即留死代码与重依赖 |
| D8 | 场景数据脚本约定 | **只立规矩不改代码**：新场景脚本一律从 `data_process`/`forecasting.data` 导入公共 IO 与 as-of，禁止复制实现；存量不动 | 收敛 30+ 存量脚本 | 存量为一次性数据工具，churn 大收益低 |
| D9 | 折生成双实现（E-C） | **文档声明差异为有意**：两函数分工（外层 RAISE 语义 vs OOF gap/cutoff/静默跳过）写进 redesign 文档与 docstring；`TimeGeometry` 单一时间合同已满足 | 收敛为单函数参数化 | 收敛需重立 parity 证据；当前双实现刚过 555 tests + golden，先声明后观察 |
| D10 | ensemble 层定位与规则化 | **L3 定位写入 AGENTS.md**；`ensemble → probabilistic.types` 同层单向豁免；E-D 记为已知边界 | probabilistic.types 下沉 forecasting.tensors | 备选翻动全部消费方，收益小 |
| D11 | E-H 包内卫生 | `methods/weighted.py` 的 `_target_key/_weight_matrix` 提公开名，linear_blending 改公开导入 | 不动 | 两行改动，随 P4 |
| D12 | **预处理定位声明（F1/F2）** | **行为零改动，立契约**：① AGENTS.md/redesign 文档声明 canonical 信息集「缺失/异常 = RAISE」为预测性维护默认，填补/清洗只发生在离线 `data_process/`（进模型前）或未来 config 驱动的 compiler 前置步骤（须 as-of 合规）；② `outlier_handling` 并入 `data_process/` 时在模块 docstring 声明 legacy 身份（不接 canonical runtime）；③ `data_process/README.md` 声明离线工具库定位及其与主链信息集契约的关系 | 在 canonical 配置新增 `validation.outlier_strategy`（如 `interpolate|raise`）接回在线清洗 | 备选是**语义新增**（训练窗口改数据），影响全部配置的信息集行为，需要独立设计 + 消融验证证明不伤精度——不该捆进架构收敛。推荐项零行为风险；若 wangzf 需要在线清洗能力，批准后另立专项 |
| D13 | **评估掩码 eval_mask（F6）** | **重接线（rewire）**：canonical 评估消费 `validation.eval_mask`——把 `utils/eval_mask.py::build_eval_mask` 接入 `forecasting/results.py` 的指标计算（MAPE/Accuracy 按掩码过滤、`Valid Points` 用掩码计数），掩码未配置 = 行为零变化；`build_eval_mask` 移入 `forecasting/`（或 results 内联），补单测（percentile/absolute/combined 三模式）| **删除**：14 个 YAML 去掉 eval_mask 字段 + 审计白名单移除 + utils/eval_mask.py 删除；业务口径改由结果后处理脚本承担 | 两案语义方向相反，wangzf 裁决：rewire 恢复「业务口径过滤的 MAPE」历史约定（电力负荷低出力时段不应计入 MAPE），代价 ~60 行 + 定向测试；**注意 rewire 对那 14 个配置是真实行为变化**（历史 test_scores 未过滤），需重跑对比并在偏差表登记；delete 则承认 canonical 口径为全点 MAPE，业务口径后置 |
| D14 | **decomposition 死链（F7）** | **【执行期撤销——F7 为误诊】**P4 删除前复查发现 `pipeline.py:15-17,:42,:69-76,:116-132` 在包内消费 `AdditiveComposer/PolyTrendForecastForecaster/PhaseTemplateForecaster`（§2.7 审计时 grep 排除了 `./decomposition/` 自身导致漏判），三者为 `DecompositionPipeline` 主链活代码，**文件保留未删**；「零消费 bundle 类型」未动 | ~~删除~~ | 误诊源于消费者审计不完整——教训已记入 redesign 文档偏差表 |
| D15 | **训练残留死导入（F4/F5）** | 删 `ModelTraining.py:34-35` GridSearchCV/RandomizedSearchCV 死导入（随 P1 迁移顺带）；`losses.py` 保留（scorer 活跃消费） | 同时删 losses.py | losses 有消费（ModelTraining:58）；GridSearch 两行纯死重 |

## 5. 实施切片

依赖链：

```text
P0 文档先行 ──→ P1 包间消环（D1/D15） ──→ P2 data_provider 了断（D2/D12） ──→ P3 回测公开化 + 评估口径（D3/D6/D7/D13） ──→ P4 features/decomposition 了断 + 收尾（D4/D9/D10/D11/D14）
```

顺序理由：P1 先行使后续各片 import 只改一次；P2/P3/P4 相互独立可单独回退；D13（rewire）是唯一行为变化片，放在结构收敛之后单独验证。**每片完成后跑 §6 全量验证并汇报「改动清单 + 验证证据」，全部等待 wangzf 明确指示才 commit。**

### P0 — 文档先行（评审通过即执行）

| 项 | 内容 |
|---|---|
| 范围 | 本文档按评审结论定稿；AGENTS.md 新增「包间分层规则」节（§3.1 含规则 6 数据契约）+「流水线默认」（D12 声明）；**修正 AGENTS.md:49 气象条目**为 DataSpec `availability`/as-of 现役口径；ensemble L3 定位（D10） |
| 验证 | 文档内 `path:line` 引用抽查复核 |

### P1 — 包间消环（D1 + D15，最大风险片）

| 项 | 内容 |
|---|---|
| 范围 | `CanonicalTrainer` → `forecasting/trainer.py`、`CanonicalForecaster` → `forecasting/forecaster.py`（类代码零修改，纯迁移）；改导入方：`forecasting/runtime.py`、`probabilistic/pipeline.py`、`probabilistic/training.py`（顺带消除 :102 延迟 import）、7 个测试文件；`models/` 收缩为 Factory/SaveLoad/learning_rate/losses；`ModelSaveLoad` 类型检查导入改 duck-typing（F8）；删 GridSearch 死导入（D15） |
| 不变 | `ModelFactory/ModelSaveLoad` 留守；`ensemble/` 不动 |
| 规模 | 生产 4 文件 + 测试 7 文件 import 调整 |
| 验证 | 全量 unittest + `test_canonical_*`、`test_config_entrypoints`、`test_multi_target_adapters` 定向 + 单配置 canonical 冒烟（fingerprint 不变）+ **新增循环依赖门禁测试**（断言 models 不 import forecasting/probabilistic） |

### P2 — data_provider 了断（D2 + D12）

| 项 | 内容 |
|---|---|
| 范围 | `outlier_handling.py` → `data_process/`（docstring 声明 legacy 身份，测试同步改导入）；删 `exogenous_contract.py` + `weather_contract.py` + `tests/test_exogenous_information_contract.py`；删 `data_provider/` 包；新增 `data_process/README.md`（离线工具库定位 + F1 数据契约声明）；门禁测试断言 `data_provider/` 不存在 |
| 前提复核 | 删除前 grep 全仓对三模块的 import（已知消费者：1 个场景脚本用 `empty_train_outlier_report`——同步改导入） |
| 验证 | 全量 unittest + `test_outlier_handling` 定向 + compileall |

### P3 — 回测公开化 + 评估口径（D3 + D6 + D7 + D13）

| 项 | 内容 |
|---|---|
| 范围 | 新建 `forecasting/backtest.py`：`_seasonal_naive_tensor/_actual_tensor/_resolve_origin` 公开化迁入（纯搬运），runtime 改调用方；`ensemble/runtime.py:40,130` 改公开导入（E-B）；删 `utils/metrics.py`、`utils/cv_plot.py`；删 `ModelDeploy/ModelDeployPmml` + PMML 依赖（D7）；**D13 按裁决执行**：rewire = eval_mask 接入 `results.py` 指标计算 + `build_eval_mask` 迁入 forecasting + 三模式单测；delete = 14 YAML 清字段 + 审计白名单移除 + utils/eval_mask.py 删除 |
| 验证 | 全量 unittest + 单配置回测输出 diff 为空（无掩码配置 `test_scores_df.csv` 逐值相等）+ `test_ensemble_runtime` 定向；rewire 时：三模式掩码单测 + 任一 eval_mask 配置重跑前后分数对比登记偏差表 |

### P4 — features/decomposition 了断 + 收尾（D4 + D9 + D10 + D11 + D14）

| 项 | 内容 |
|---|---|
| 范围 | TargetTransformation 并入 `forecasting/transforms.py`；删 `features/` 整包；删 `decomposition/forecasters.py` + `composers.py` 及零消费 bundle 类型（D14，删前 grep tests）；门禁测试断言 `features/`、`data_provider/` 不存在；`data_process` 提公开名 + decomposition 改公开导入；`methods/weighted` 提公开名（D11）；D9/D10 文档声明落 redesign 文档 |
| 验证 | 全量 unittest + 门禁测试 + 单配置冒烟 + `test_canonical_transforms`、`test_decomposition_*` 定向 |

## 6. 验证命令

```bash
# 全量测试（AGENTS.md 标准命令；v4 验收基线 555 tests OK，以实测为准）
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s tests -p "test_*.py"

# 语法/导入完整性
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m compileall -q forecasting ensemble models probabilistic data_process decomposition utils

# 单配置 canonical 冒烟（CONFIG_YAML 指向任一现役 YAML）
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py

# 配置审计（P1-P4 全程不变性证据）
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python scripts/audit_ensemble_configs.py
```

验收基线：评审通过后先实测记录全量测试数与 check_model_configs=848 passed 为基线；每片完成后必须维持全绿 + 该片定向测试通过 + audit 零失败；**除 D13（rewire 选项）与 D12（备选）外，所有片行为零变化**，以 golden parity / 输出 diff 为证。

## 7. 风险与控制

| 风险 | 影响 | 控制 |
|---|---|---|
| P1 类迁移改动面大（4 生产 + 7 测试文件） | 修错导入运行时才发现 | 纯搬运零逻辑改动；compileall + 全量测试 + 冒烟三道闸；parity golden 提供行为不变证据 |
| D13-rewire 改变 14 个 eval_mask 配置的历史评估分数 | 与旧结果不可比 | 掩码未配置 = 零行为变化（其余 834 配置不受影响）；受影响配置重跑前后分数对比登记偏差表；此项本质是**恢复既有业务口径**而非引入新语义 |
| D13 若选 delete | 业务口径（掩码 MAPE）永久缺失 | 后处理脚本承担口径；wangzf 明确裁决后执行 |
| D2/D14 删除死代码，下游有未发现消费者 | 场景脚本/笔记本断裂 | 每次删除前强制全仓 grep（tests/、config/scripts/）；`temp/`、`docs/` 引用不修，删除 commit message 记录 |
| D7 删 PMML 类影响仓外部署 | 外部脚本断裂 | 全仓零消费者已核实；wangzf 评审时确认，否则 D7 降级为 import 下沉 |
| D12 备选（在线清洗）若被要求 | 全配置信息集行为变化 | 不捆进本轮；单独设计 + 消融验证 |
| 折生成双实现语义漂移（E-C 不收敛） | 改一处漏一处 | D9 文档声明 + docstring 互引；OOF 语义演化时触发收敛 |
| **【执行期发现 2026-08-30】v4 折合同与存量大部配置几何错位**：824 单模型配置中 618 个（aidc_load_15min_{short,daily,rolling}、aidc_ess_selfuse_load 全场景 + computility 部分）满足不了 E1 合同 `history_length > horizon + (max_windows-1)×stride`——这些配置的 `history_length`（16/34 等）≤ `horizon`（16/96），非重叠合同下候选池内不存在任何合格训练样本，外层回测一律 RAISE；可运行的 206 个几乎全是 aidc_power_month（v4 验收实际跑过的场景） | 五个 load/ess 场景的滑窗回测在现配置下全部不可运行 | **本会话未改动折生成逻辑**（git diff 证明：validation.py 零改动、runtime 折区域零改动；562 tests 含 E1 契约测试全绿）——问题属于 v4 收口语义与存量配置既有值的错位，属 canonical 语义范畴（方案 §1.3 范围外），不在架构收敛轮内修；已按用户纪律「YAML 语义变更只报告不擅改」。修复方向两选：① 提高存量配置 `history_length`（≥ horizon+1）；② E1 合同增加显式「仅 final forecast」模式（`max_test_windows: 0` 类语义）。需 wangzf 裁决后另立配置修复批次 |
| D13-rewire 待真实重跑验证 | 14 配置分数变化未实测 | 结构性证据已齐（fingerprint 不变 + 掩码逻辑对账测试）；等上述折合同问题裁决后随重跑批次一并执行 |
| 迁移触发 fingerprint/identity 变化 | 结果目录漂移、848 审计失败 | 结构收敛零配置变更；双审计脚本每片必跑（D13-rewire 不改 payload，fingerprint 不变） |
| 大片改动 review 负担 | 合入质量风险 | review-then-commit 逐片走；每片独立可验证、可单独回退；**未经明确指示不 commit** |

## 8. 文档同步义务

| 文档 | 时机 | 内容 |
|---|---|---|
| `AGENTS.md` | P0 | 新增「包间分层规则」节（含规则 6 流水线数据契约）；修正气象严格信息集条目为现役口径；ensemble L3 定位与单向豁免 |
| `AGENTS.md` | P1 完成 | 「训练与产物」条目中 Trainer/Forecaster 位置更新 |
| `AGENTS.md` | P3 完成 | eval_mask 最终口径（rewire = 评估掩码条目恢复；delete = 显式声明全点 MAPE 口径） |
| `AGENTS.md` | P4 完成 | 删除「docs/feature_engineering 归档」过渡条目；门禁断言清单更新 |
| `docs/multistep_forecasting_redesign.md` | 各片完成后 | §7 模块架构图补包间分层；D9 折语义声明；D12 预处理默认声明；偏差追加「实施进度记录」偏差表（completed 附命令与证据） |
| `docs/production_sync.md` | 全部完成后 | 同步生产口径 |

## 9. 裁决记录（2026-08-29，wangzf）

| 裁决项 | 结论 |
|---|---|
| D13 评估掩码 eval_mask | **rewire**——接回掩码口径（`build_eval_mask` 接入 results.py 指标计算），14 个配置历史评估分数真实变化，重跑并在偏差表登记 |
| D12 预处理定位 | **只立契约**——「缺失/异常 = RAISE」声明为预测性维护默认，填补/清洗只发生在离线 `data_process/`（进模型前）；不内建在线清洗 |
| D7/D14/D15 死代码 | **全部删除**——PMML 类 + sklearn2pmml/pypmml 依赖、decomposition 分量预测死链（forecasters.py+composers.py）、GridSearchCV 死导入 |
| D9 折生成双实现 | **文档声明差异为有意**——外层严格 RAISE vs OOF 泄漏防护特化，写进 redesign 文档与 docstring，不收敛实现 |
| 切片顺序 | **P1 消环先行**——先迁 Trainer/Forecaster 消除两包环，后续各片 import 只改一次 |

其余决策（D1–D6、D8、D10、D11）采纳推荐项，未提出异议。§1.3 范围边界与 commit 纪律（每片汇报证据，明确指示才 commit）生效。

**下一步**：待 wangzf 开工指令后按 P0→P1→P2→P3→P4 执行。

## 10. 执行记录与完成状态（2026-08-30 收口）

### 10.1 决策逐项核销

| 决策 | 状态 | 证据/说明 |
|---|---|---|
| D1 Trainer/Forecaster 迁移 | ✅ 完成 | `forecasting/trainer.py`/`forecaster.py` 落地，models 留双 shim；F8 duck-typing 一并完成；环 1/环 2 消除，`tests/test_package_layering.py` 门禁固化 |
| D2 data_provider 整包删除 | ✅ 完成 | outlier_handling → data_process（legacy docstring）；两契约 + 其测试删除；门禁断言包不存在 |
| D3 backtest.py 公开化 | ✅ 完成 | 4 原语纯搬运（AST 级比对）；runtime 私有别名转发；E-B 两处修复 |
| D4 features 整包删除 | ✅ 完成 | TargetTransformation/TargetNormalization/TargetScaler 并入 `forecasting/transforms.py`（1202 行）；死导入、FeatureScaler、包本体删除；门禁固化 |
| D5 pickle 垫片 | ✅ 撤销（v2） | 无兼容对象 |
| D6 utils 孤儿删除 | ✅ 完成 | metrics.py、cv_plot.py 已删 |
| D7 PMML 死链删除 | ✅ 完成 | ModelDeploy/ModelDeployPmml + sklearn2pmml/pypmml import 与 pyproject 依赖声明删除 |
| D8 场景脚本约定 | ✅ 完成（立规矩） | AGENTS.md 数据契约 + data_process/README |
| D9 折双实现声明 | ✅ 完成 | redesign 文档 §7 声明分工；docstring 互引已存在（oof.py:3-4 引 rolling_origin_folds） |
| D10 ensemble L3 定位 | ✅ 完成 | AGENTS.md 分层规则节 + redesign 文档 |
| D11 weighted 提公开名 | ✅ 完成 | `target_key`/`weight_matrix`；linear_blending/stacking/predictor 改公开导入；修复中引入的两处变量遮蔽 bug 已验证修复 |
| D12 预处理契约 | ✅ 完成 | AGENTS.md 流水线数据契约 + data_process/README + outlier_handling docstring legacy 身份 |
| D13 eval_mask rewire | ✅ 代码完成 / ⚠️ 真实重跑待做 | results.py 接线（掩码未配置=零变化）+ 9 项单测 + runtime 接线；14 配置 fingerprint 不变已验证；**分数前后对比待重跑**（见 §10.3） |
| D14 decomposition 死链 | ⚠️ **撤销（误诊）** | F7 诊断错误：forecasters.py/composers.py 是 `DecompositionPipeline` 包内活代码（pipeline.py 多处消费），P4 删除前复查发现，文件保留；教训记入偏差表 |
| D15 GridSearch 死导入 | ✅ 完成 | 随 P1 迁移消失；实际清理的死导入共 27 个符号（GridSearch 仅是其中之一，其余为 sklearn 8 项、stdlib 7 项、models/probabilistic/utils 12 项零使用符号） |

### 10.2 切片完成状态与验证基线

| 切片 | 状态 | 全量测试 | 审计 |
|---|---|---|---|
| P0 文档先行 | ✅ | — | — |
| P1 包间消环 | ✅ | 558 OK | 848 passed |
| P2 data_provider 了断 | ✅ | 552 OK（删 9 契约用例 +1 门禁） | 848 passed |
| P3 回测公开化+评估口径 | ✅ | 561 OK（+9 掩码单测） | 848 passed |
| P4 features 了断+收尾 | ✅ | 562 OK（+1 门禁） | 848 passed |
| **终态** | **全部完成** | **562 OK**（81s，较基线 101s 加速） | **848 passed / audit 零失败** |

真实配置冒烟：`st_usmr_mean`（recursive×st×滑窗）**535.9s 完成**，全产物落地（见 §10.3 性能修复）。

### 10.3 执行期新增工作（超出原方案范围，经追加裁决完成）

| # | 工作 | 触发 | 结果 |
|---|---|---|---|
| 1 | **折合同配置修复**：618 个 YAML `history_length` 按 `2H+1` 规则提升（short 33 / daily·rolling 193 / ess·computility 577 等） | 真实冒烟暴露 v4 折合同与存量配置几何错位（618/824 不可运行）；wangzf 裁决「批量修配置，纯部署时修改配置即可」 | 824/824 几何可行；check_model_configs 848 passed；断言旧值的 2 个测试同步更新为非重叠合同断言 |
| 2 | **信息集性能修复（方案 A）**：`MaterializedInformationSet` 增加行位置缓存 API（`row_position_lookup`/`register_row_position_lookup`，不可变对象级共享）；`_exact_temporal_row`/`_labels` 接入 O(1) 定位；registry 增加 `_validated_cache`（验证帧按 path+source+version 复用） | 修复后首跑 st_usmr 2.7h+ 无产出，cProfile 定位三层浪费（_labels 全列扫描、特征编译全列扫描、每 origin 重复验证+deep copy） | st_usmr 冒烟 **2.7h+ → 535.9s 完成**；全量测试 101s→81s；41 项信息集/编译/冒烟定向测试 + 562 全量通过 |

### 10.4 遗留清单（本方案关闭后仍开放）

| # | 事项 | 状态 |
|---|---|---|
| 1 | D13-rewire 真实重跑：14 个 eval_mask 配置的分数前后对比（现折合同已修，可执行） | 待重跑批次 |
| 2 | 618 配置修复后的全量重跑与消融（`history_length` 语义变化使回测更严格、分数与 legacy 不可比） | 待 wangzf 排期 |
| 3 | 全部改动 review 后 commit（review-then-commit 纪律） | 待指示 |
| 4 | `validation` 段无类型 Mapping（§2.7 审计项 11 注记） | 记录在案，未来轮次 |
| 5 | recursive 类策略的 16-call teacher-forced 编译形状（性能修复只做了缓存，编译量本身未减） | 记录在案：若后续仍嫌慢可评估「单模型策略只编译消费 call」 |

**v3 原方案关闭**：§1–§10 记录的原定结构切片已完成或显式撤销；其“终态完成”结论已被 v4 fresh review 取代。后续实施以 §11–§16 为准，未经明确指示不 commit、不 push。

---

## 11. v4 Coder Review 结果（2026-08-30）

### 11.1 Provenance 与审查边界

| 项 | 内容 |
|---|---|
| Agent | Machine-C（MC） |
| 审查对象 | 当前 `dev` 工作区生产代码、848 个 schema-2 模型 YAML、Markdown 文档、测试、配置审计器和运行产物合同 |
| 明确排除 | `docs/` 下未接入生产的 Python/ipynb 实验脚本 |
| 方法 | 逐文件阅读 + AST import/消费者审计 + 全量命令实跑 + 配置字段/数据路径程序化统计 + 关键路径最小复现 |
| 纪律 | 只做 review 与计划记录；未修改生产代码、配置和测试，未运行真实批量实验 |

### 11.2 Fresh 基线

| 验证 | 命令/方法 | 结果 |
|---|---|---|
| 工作区 | `git status --short --branch` | `dev...origin/dev`，review 开始时 clean |
| 全量测试 | `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest discover -s tests -p "test_*.py"` | **600 tests / 84.626s / OK** |
| 分层门禁 | `python -m unittest tests.test_package_layering -v` | **12 tests / OK** |
| 语法与 diff | `python -m compileall -q ... && git diff --check` | exit 0 |
| 配置 checker | `python scripts/check_model_configs.py` | `checked=848 passed=848 hard_failures=0 warnings=0` |
| Ensemble 审计 | `python scripts/audit_ensemble_configs.py` | 800 单模型 + 24 Ensemble + 24 成员；零失败 |
| YAML→运行时合同探测 | 对 824 个带 `features` 的基模型/成员 YAML 比对 `FeatureCompiler` 实际允许字段 | **624 不兼容，200 兼容** |
| 数据资产探测 | 对所有 source 的 history/backtest/future path 做存在性核对 | 14 个唯一缺失文件，影响 **396/848** 配置 |

> **判定原则**：`600 tests OK` 和 `848 passed` 是现有门禁的真实结果，但不是“848 个配置均可运行”的证据。配置 checker 与生产 runtime 使用不同合同，是本轮最大的系统性问题。

### 11.3 总体成熟度结论

| 维度 | 结论 | 说明 |
|---|---|---|
| 架构方向 | **B** | 显式列角色、as-of、七策略、统一张量、能力门禁和独立评估方向正确 |
| 单模型运行正确性 | **D** | 624 个基模型 YAML 在 FeatureCompiler 合同处不兼容；时间几何语义未闭环 |
| Ensemble 生产能力 | **D** | 10 个 quantile linear_blending 配置存在维度错误；运行时不持久化 bundle/result |
| 测试可信度 | **C** | 覆盖面广，但大量测试只验证 parser/checker/合成 fixture，未覆盖现役配置→生产 runtime |
| 文档一致性 | **D** | 多份文档继续描述已删除模块、旧 schema、历史测试计数和未接线能力 |
| 当前可交付性 | **不通过** | 不得以现有全绿门禁宣称重构完成或全部配置可运行 |

### 11.4 Review findings（RF1–RF12）

#### RF1 — P0：624/824 个基模型 YAML 与 FeatureCompiler 的 canonical schema 不一致

- 运行时只接受 `direct/advanced/feature_scaling/target/datetime_categorical/interactions`（`feature_engineering/compiler.py:95-104,569-572`）。
- checker 同时放行旧 `direct_layout/target_transform/rolling_windows/diff_periods/pct_change_periods`（`scripts/check_model_configs.py:44-66`）。
- 现役例：`config/aidc_power_month/route_A/freq_1month/window_length_10/lgbm_usmd_prob_mean.yaml:67-72`。
- 最小复现：`ValueError: unsupported feature transformations: ['direct_layout', 'target_transform']`。

**影响**：配置可加载、checker 全绿，但进入生产编译即失败；`848 passed` 为假绿。

**结论**：只保留 compiler 当前新嵌套 schema，一次性迁移 624 YAML；不得新增双 schema 兼容层。

#### RF2 — P0：10 个 quantile + linear_blending Ensemble 维度合同错误

- 当前分布：point averaging=12、point linear_blending=2、quantile linear_blending=10。
- 现役例：`config/aidc_power_month/route_A/freq_1day/baseline/lgbm_usbr_prob_mean_conformal.yaml:23-48`。
- `fit_linear_blending()` 把 `(fold,H,K,Q)` 展开为 `fold×H×Q` 行，却用 `(fold,H,K)` actual 的 `fold×H` 行拟合 NNLS（`model_ensemble/methods/linear_blending.py:28-42`）。
- 最小复现：`ValueError: Incompatible dimensions`。

**影响**：10 个现役 Ensemble 配置无法完成融合器训练。

**结论**：quantile 融合必须使用明确的 proper objective。推荐按 target 在 simplex 约束下最小化跨 level 的 pooled pinball loss；同一 target 的全部 quantile level 共享一组权重。不得通过简单重复 actual 后继续 MSE 伪装修复。

#### RF3 — P0：Ensemble lifecycle 在内存结果处终止，无生产产物

- `model_ensemble/runtime.py:101-124` 只返回 artifact/oof/member values/combined values/audit。
- 没有写 `prediction.csv`、`model.pkl`、`resolved_model.json`、`resolved_config.json`，也没有自包含成员 bundle。
- `tests/test_canonical_ensemble_runtime.py:127-131` 反而把 bundleless 结果固化为“complete”。
- `run.py:267-270` 正常调用传 `output_root=None`；`model_ensemble/runtime.py:107-110` 又仅在 output_root 非空时保存 OOF cache。

**影响**：24 个 Ensemble 不能作为可部署模型运行；CLI 的“run complete”与外部状态不符。

**结论**：补齐 outer/backtest、final member bundles、Ensemble ForecastModelBundle、long prediction、resolved metadata 和 OOF reference；正常默认 results_root 必须真实保存缓存。

#### RF4 — P1：固定步长训练窗口单位错位，618 个子日配置受影响

- runtime 把 `history_length/window_length` 直接当监督 origin 数（`model_forecasting/runtime.py:695-723`）。
- 文档与旧 checker 口径把 `window_length` 当天数并乘 `samples_per_day`。
- 典型形状：15min/H=96/window=30，运行时每窗只取 30 个监督样本；按“30 天”原意应约为 `30×96−96=2784`。
- final fit 又使用全部 `X_all/Y_all`（`model_forecasting/runtime.py:1287-1302`），与回测训练窗不一致。

**影响**：回测训练规模远小于配置名义值，且不能代表最终模型；618 个配置虽通过几何门禁，指标语义仍不可信。

**结论**：废止无单位的 `history_length/window_length` 运行解释；固定步长明确使用 step-based typed geometry，calendar-month 使用独立 month/day geometry。迁移值必须依据原业务意图重算，不能原数照搬。

#### RF5 — P1：126 个 calendar_month 配置未进入生产 runtime

- 单模型 runtime 不读取 `horizon_mode/schedule_mode/train_window_length`。
- 自然月折只存在 checker 私有 helper（`scripts/check_model_configs.py:509-592`），生产消费者为 0。
- `tests/test_calendar_month_horizon.py:9,35-55` 测的是 checker helper，不是 `run_canonical_config()`。

**影响**：最终 2026-08 的 31 步预测只是 YAML 写死 H=31 的偶然匹配；历史回测不会按 28/29/30/31 天完整自然月切分。

**结论**：自然月 geometry 下沉 `model_testing.validation`，生产 runtime、checker 和 Ensemble OOF 共享同一 typed contract。

#### RF6 — P1：月频 seasonal-naive 不支持 MonthEnd/MonthBegin

- `model_testing/backtest.py:42-45` 对 `builder.offset` 调用 `pd.Timedelta()`。
- `1ME` 实测报 `ValueError: ... not MonthEnd`。
- 约 56 个月频基模型配置受影响。

**结论**：按频率类型显式定义 naive lag；月频默认/配置必须是月步数（如 1 或 12），不得通过一天除 offset 推导。

#### RF7 — P1：Ensemble 顶层共享合同和 cache 内容寻址不完整

- `validate_member_sources()` 定义于 `model_ensemble/loader.py:201-289`，生产调用为 0。
- `resolve_members()` 只比较成员之间，不比较顶层 `ensemble.problem`；实测顶层 horizon=99、成员 horizon=2 仍被接受。
- 顶层 `validation.forecast_origin` 未成为权威，成员自己的 origin 继续生效。
- source hash 只按 `source.name` 保存且只 hash `history_path`；同名成员 source 会覆盖，OOF 实际读取的 `backtest_path` 未进入 fingerprint（实际读取见 `data_loading/registry.py:467-479`）。

**结论**：顶层 problem/data/probabilistic/validation 为唯一权威；runner factory 必须用顶层 origin；cache manifest 按 `(member, source, path_role)` hash 所有实际读取文件。

#### RF8 — P1：396/848 配置引用不存在的数据资产

- 138 个唯一 source path 中 14 个不存在，共 792 次引用，影响 396 个配置。
- 主要为各场景 `ETTm1_exogenous/df_date.csv` 与 `df_date_future.csv`。
- `dataset/README.md:113-121` 自身也要求启用外生时文件必须存在。

**结论**：把配置审计拆为 schema audit 与 runtime asset audit；活动配置必须满足文件、列、时间范围、future horizon、available_at 合同。可再生数据需有已实跑的唯一生成命令，否则配置降为非现役或删除。

#### RF9 — P1：所谓 strict loader 仍接受未知 validation/output 字段

- `ForecastConfigSpec.probabilistic/validation/output` 仍是无类型 Mapping（`model_forecasting/specs/config.py:178-188`）。
- parser 只检查“是 mapping”（`:458-461`）。
- 实测 `validation.totally_unknown_field` 和 `output.totally_unknown_field` 均被接受。

**结论**：新增 `RuntimeValidationSpec`、`OutputSpec`，直接复用/收敛 `ProbabilisticSpec`；删除 checker 的第二套 nested whitelist。未知字段必须在 `load_yaml_config()` 返回前 RAISE。

#### RF10 — P2：包级依赖仍非 DAG，延迟 import 被门禁白名单合法化

- `model_training/trainer.py:28-29` import probabilistic；`probabilistic/training.py:102` 函数内反向 import `CanonicalTrainer`。
- `model_forecasting` 同时承载低层 specs/tensors 与高层 runtime/forecaster，形成 `model_forecasting ↔ data_loading/feature_engineering/model_training/model_evaluation/probabilistic` 包级双向边。
- `tests/test_package_layering.py:48-53,64-71` 显式允许这些方向，门禁证明的是“符合白名单”，不是“无循环”。
- `model_forecasting/runtime.py` 1650 行，其中 `_RegistryDesignBuilder` 405 行、`CanonicalBaseModelRunner` 566 行；`transforms.py` 1203 行。

**结论**：把稳定合同下沉独立 `forecasting_core/`；高层 runtime 只向下编排；probabilistic 通过 Protocol/工厂消费训练和点预测能力，不反向 import 具体实现。

#### RF11 — P2：测试套件存在假绿、重复和测试专用 legacy 链

- 全量 600 tests 真实通过，但未覆盖 RF1/RF2/RF3/RF5/RF6。
- `test_calendar_month_horizon.py` 测死 helper；`test_canonical_ensemble_runtime.py` 固化 bundleless；quantile Ensemble 仅测 averaging。
- `test_canonical_ensemble.py` 的 averaging/strategy/NNLS 断言已被其他测试重复覆盖。
- `probabilistic/evaluation.py` 约 774 行，生产零消费者，仅由 `test_probabilistic_evaluation.py`、`test_probabilistic_prequential.py` 消费。
- `data_process/outlier_handling.py` 的清洗算法只被测试消费，生产场景脚本只使用空报告 schema。

**结论**：新增现役配置→运行时门禁后，再合并/删除重复与 tests-only legacy 链；不得用删测试提高通过率。

#### RF12 — P2/P3：文档、CLI、依赖与死代码仍残留 legacy

- 根 README、config/scripts/probabilistic/utils/tests 等 README 继续引用已删除包、旧 schema、旧测试计数和未接线能力。
- `run.py:216-242` 暴露大量 canonical 不支持参数，部分如 `augmentation_ratio` 被静默忽略；Ensemble 的 `--data-dir` 等 override 同样静默忽略。
- `requirements.txt` 缺 `openpyxl/statsmodels`，却仍含已从 pyproject 删除的 `pypmml/sklearn2pmml` 及其传递依赖。
- `models/learning_rate.py` 当前零消费者；`models/losses.py` 需在删除前再做一次全仓符号级消费者审计。

**结论**：CLI 收缩为 config/seed/output-root；依赖统一 `pyproject.toml + uv.lock`；当前态文档只保留当前实现，历史交给 git。

## 12. v4 修正后的目标架构

### 12.1 分层目标

```text
L5  入口与配置分派
    main.py / run.py / config/config_loader.py
                  │
                  ├───────────────┐
                  ▼               ▼
L4  单模型生命周期编排        Ensemble 生命周期编排
    model_forecasting/        model_ensemble/
                  │               │（仅依赖 Protocol；runner factory 由入口注入）
                  └───────┬───────┘
                          ▼
L3  能力层
    probabilistic/（quantile 训练/后处理）
                          ▼
L2  流水线阶段
    data_loading → feature_engineering → model_training
                         ├→ model_testing
                         └→ model_evaluation
                          ▼
L1  稳定合同与基础设施
    forecasting_core/{specs,tensors,artifacts,protocols}
    models/  decomposition/  data_process/
                          ▼
L0  utils/
```

### 12.2 关键边界

1. `forecasting_core/` 不 import 任何流水线/能力/编排包；承载 problem/data/feature/strategy/estimator/runtime/output specs、point/quantile tensor、bundle artifact 基础类型与 runner Protocol。
2. `model_training/` 不 import `probabilistic/`；概率模式作为 core spec/枚举输入，quantile estimator factory 由上层注入。
3. `probabilistic/` 可依赖 core 与 model_training，但不得 import `model_forecasting.runtime/forecaster/results`；点预测执行能力通过 Protocol 注入。
4. `model_ensemble/` 不 import `CanonicalBaseModelRunner` 实现；入口注入 `BaseModelRunnerFactory`，从而消除 L4 同层反向依赖。
5. `model_forecasting/` 只做单模型生命周期、推理执行、结果写出；`runtime.py` 拆为 design、fit、backtest lifecycle、persistence 四个窄模块。
6. config parser、checker、runtime 不得各自维护字段白名单；typed spec 是唯一 schema 事实源。

### 12.3 修正后的数据与产物流

```text
YAML
 → strict typed specs
 → runtime asset validation
 → SourceRegistry / InformationSet
 → FeatureCompiler（唯一 transformation schema）
 → supervised design（显式 step/month geometry）
 → per-fold transform + train
 → backtest + point/quantile evaluation
 → final fit（与回测训练窗策略同合同）
 → forecast
 → atomic result directory
    ├── pretrained_models/model.pkl
    ├── pretrained_models/resolved_model.json
    ├── results_test/{cv_plot_df,test_scores_df[,probabilistic_scores],metadata}
    └── results_forecast/{prediction.csv,resolved_config.json}

Ensemble：成员 OOF → proper fusion objective → 成员 final bundles → Ensemble bundle → 同一结果布局
```

## 13. 整改决策（RD1–RD10）

| # | 决策 | v4 推荐方案 | 不采用方案及原因 |
|---|---|---|---|
| RD1 | transformation schema | 以 compiler 当前 `direct/target/advanced.*` 嵌套结构为唯一 canonical，迁移 624 YAML | 运行时长期兼容旧键：恢复双轨，继续制造 fingerprint/行为歧义 |
| RD2 | validation/output | 新建强类型 spec，loader 直接严格解析 | checker 后置白名单：生产入口可绕过 checker，无法保证严格 |
| RD3 | 固定步长单位 | 所有时间几何字段显式 `*_steps`；迁移 manifest 记录旧值、目标值、公式 | 继续用无单位 length；或在 runtime 隐式乘 n_per_day：两者都不可审计 |
| RD4 | 自然月 | 独立 calendar-month geometry，按目标月动态 H；生产/checker/OOF 同源 | YAML 写死 31：跨月份立即失真 |
| RD5 | quantile blending | simplex 约束下最小 pooled pinball，每 target 一组权重共享 Q | 重复 actual 后 NNLS/MSE：不是 proper quantile objective |
| RD6 | Ensemble 产物 | 自包含成员 bundle + method artifact + OOF reference，写 canonical long 结果 | bundleless 内存结果：不可部署、不可复载、不可审计 |
| RD7 | 包间 DAG | 建 `forecasting_core` + Protocol 注入，消除所有函数内反向 import | 扩大白名单：掩盖循环而非修复 |
| RD8 | 活动配置资产 | 文件不存在即 runtime-asset audit hard fail；可再生文件必须有已验证生成命令 | checker 只看 schema：继续产生假绿 |
| RD9 | legacy 测试/模块 | 先迁移唯一活消费者，再整链删除 tests-only legacy 实现 | 保留“以后也许用”：与 canonical-only/YAGNI 冲突 |
| RD10 | 文档 | 当前态文档只写当前实现；历史设计/计数由 git 溯源 | 在同一文档累加全部历史：事实源继续漂移 |

## 14. 实施切片 C0–C7

依赖链：

```text
C0 锁失败证据
  → C1 配置合同统一
    → C2 时间几何修复
      → C3 Ensemble 闭环
        → C4 包级 DAG 收敛
          → C5 测试/依赖/死代码清理
            → C6 当前态文档收口
              → C7 全量与真实运行验收
```

> 已知 bug 修复前不录制“全链逐值 golden”。C0 只锁定应失败的合同与正确预期；C1–C3 修正后再建立新的正确 golden，避免把错误行为固化为规范。

### C0 — 建立 RED 门禁与审计清单

**Files**：

- Create: `tests/test_active_config_runtime_contract.py`
- Create: `tests/test_calendar_month_runtime.py`
- Create: `tests/test_ensemble_quantile_blending.py`
- Create: `tests/test_ensemble_persistence.py`
- Create: `tests/test_cli_override_contract.py`
- Create: `scripts/audit_runtime_assets.py`

**工作**：

1. 用现役 YAML 建参数化测试，证明旧 transformation 键在 runtime 不可执行；预期先 RED。
2. 用 `1D + calendar_month` 合成数据直接跑生产 runtime，锁 28/29/30/31 动态 H；预期先 RED。
3. 用 `(N=1,H=2,K=1,Q=3)` 和 `(N=2,H=2,K=2,Q=3)` 锁 quantile blending；预期先 RED。
4. 要求 Ensemble 生成完整 result/bundle；预期先 RED。
5. 所有 CLI 非空 override 必须“生效或 RAISE”，禁止 silent ignore。
6. 生成缺失 source manifest（配置、source、path role、文件、生成入口、owner），不自动造数据。

**HARD 验收**：失败原因必须精确命中 RF1/RF2/RF3/RF5/RF6，不得出现无关 collection/import 错误；manifest count 与 fresh 统计一致。

### C1 — Canonical 配置单一事实源

**Files**：

- Modify: `model_forecasting/specs/config.py`
- Modify: `model_forecasting/specs/feature.py`
- Modify: `probabilistic/spec.py`
- Modify: `config/config_loader.py`
- Modify: `feature_engineering/compiler.py`
- Modify: `scripts/check_model_configs.py`
- Modify: 624 个受影响 YAML
- Test: `tests/test_active_config_runtime_contract.py`
- Test: `tests/test_check_model_configs.py`
- Test: `tests/test_config_entrypoints.py`

**工作**：

1. 新增 typed `RuntimeValidationSpec/OutputSpec`；把概率配置收敛到唯一 `ProbabilisticSpec`。
2. transformation 只接受 `direct/target/advanced.*`；字段和值在 spec/Compiler 构造期完成校验。
3. 程序化迁移旧键：`direct_layout → direct.layout`、`target_transform → target`、旧 advanced 列表转为带 columns/windows/periods/stats 的明确 mapping；无法无歧义迁移的配置 hard fail 并列清单，不猜列集合。
4. checker 只调用生产 parser/constructors，不再维护 `_CANONICAL_NESTED_FIELDS` 第二事实源。
5. 校验迁移前后 intended semantic manifest；行为变化项单列，不用 fingerprint 相等掩盖 schema 变更。

**HARD 验收**：824/824 基模型/成员通过完整 spec + compiler + transform constructibility；旧键全仓活动 YAML 为 0；任意未知 validation/output/probabilistic 字段在 loader 阶段 RAISE；848 配置审计通过。

### C2 — 时间几何与回测/最终训练同合同

**Files**：

- Modify: `model_testing/validation.py`
- Modify: `model_testing/backtest.py`
- Modify: `model_forecasting/runtime.py`
- Modify: `model_ensemble/oof.py`
- Modify: `scripts/check_model_configs.py`
- Modify: 全部 validation geometry YAML 字段
- Test: `tests/test_calendar_month_runtime.py`
- Test: `tests/test_calendar_month_horizon.py`（替换 checker-helper 测试，不保留旧形态）
- Test: `tests/test_canonical_base_model_runner.py`
- Test: `tests/test_ensemble_oof.py`

**工作**：

1. 固定步长建立 `FixedStepBacktestSpec(history_steps, train_window_steps, fold_count, stride_steps)`。
2. 自然月建立 `CalendarMonthBacktestSpec(train_window_days, fold_count, stride_months)`；每 fold 的 H 由目标月决定。
3. 从 618 配置原始业务意图重算训练 steps，输出迁移 manifest；不得把当前 15/30 原样解释为 step。
4. 明确 final fit 策略：默认与回测训练窗同样使用最近 `train_window_steps`；如要 full-history，必须是显式枚举并在结果 metadata 中记录，不能隐式不同。
5. seasonal-naive 按 fixed/day/month 类型解析 lag；月频增加真实回测。

**HARD 验收**：618 个子日配置的实际 training_sample_count 与迁移 manifest 一致；126 个 calendar_month 配置按真实月份动态 H；56 个 1ME 配置不再在 naive 基线失败；训练标签严格不跨 holdout；final fit policy 可从 resolved_config/model 复算。

### C3 — Ensemble 正确性、缓存与生产产物闭环

**Files**：

- Modify: `model_ensemble/loader.py`
- Modify: `model_ensemble/specs.py`
- Modify: `model_ensemble/oof.py`
- Modify: `model_ensemble/cache.py`
- Modify: `model_ensemble/methods/linear_blending.py`
- Modify: `model_ensemble/methods/weighted.py`
- Modify: `model_ensemble/artifacts.py`
- Modify: `model_ensemble/trainer.py`
- Modify: `model_ensemble/predictor.py`
- Modify: `model_ensemble/runtime.py`
- Modify: `run.py`
- Test: `tests/test_ensemble_loader.py`
- Test: `tests/test_ensemble_oof.py`
- Test: `tests/test_ensemble_methods.py`
- Test: `tests/test_ensemble_quantile_blending.py`
- Test: `tests/test_ensemble_persistence.py`

**工作**：

1. runtime 强制调用共享 problem/data/probabilistic/origin/source 合同验证。
2. quantile linear blending 实现 per-target simplex pinball 优化；保存目标、level、样本数、优化状态和 fallback reason。
3. cache fingerprint 包含 `(member,source,path_role,file_sha256)`；损坏 RAISE，内容变更必失效。
4. final 每个成员保存完整 schema-2 base bundle；Ensemble bundle 保存 method artifact、成员顺序、OOF fingerprint/folds、source lineage、config fingerprint。
5. 写与单模型同布局的 long result；默认 `results_root` 同时用于 OOF cache 与正式产物。
6. outer holdout/OOF/final 三层 origin 和标签边界写入审计 metadata。

**HARD 验收**：24/24 Ensemble 可构造；10 个 quantile linear_blending 端到端通过；point/quantile、K1/K2 均保持 axes；默认 CLI 真实写全套产物；pickle reload 后在不读取 YAML/OOF cache 的条件下预测一致；任一 top/member 合同不一致立即 RAISE。

### C4 — 包级 DAG 与大模块拆分

**Files**：

- Create: `forecasting_core/`
- Move/Adapt: `model_forecasting/specs/` → `forecasting_core/specs/`
- Move/Adapt: `model_forecasting/tensors.py` → `forecasting_core/tensors.py`
- Create: `forecasting_core/artifacts.py`
- Create: `forecasting_core/protocols.py`
- Split: `model_forecasting/runtime.py` → `design.py`、`fit_service.py`、`backtest_runtime.py`、`persistence.py`、窄 `runtime.py`
- Split: `model_forecasting/transforms.py` → feature/target transform 窄模块（公开入口保持一个）
- Modify: `model_training/trainer.py`
- Modify: `probabilistic/training.py`
- Modify: `probabilistic/pipeline.py`
- Modify: `model_ensemble/contracts.py`
- Modify: `main.py`、`run.py`、`config/config_loader.py`
- Modify: `tests/test_package_layering.py`

**工作**：

1. core 只放不可变合同/类型/Protocol，无 pandas 文件 IO、训练或编排副作用。
2. `model_training` 去除 probabilistic import；quantile factory/模式由上层注入。
3. `probabilistic` 去除对具体 forecaster/trainer 的反向 import。
4. Ensemble runner factory 由入口注入，`model_ensemble` 不 import 单模型 runner 实现。
5. 分层门禁改为包级 DAG + 子模块白名单；删除 orchestrator 全放行和互相允许的循环白名单。

**HARD 验收**：AST 全仓（含函数内 import）无反向边、无 SCC>1；`forecasting_core` 无项目高层 import；删除所有为绕环而存在的函数内 project import；全量测试和 C1–C3 golden 不变。

### C5 — 测试、CLI、依赖与死代码清理

**Files**：

- Merge/Delete: `tests/test_canonical_ensemble.py`、`tests/test_canonical_ensemble_runtime.py` 中重复断言并入 `test_ensemble_*`
- Replace: `tests/test_calendar_month_horizon.py`
- Review/Delete chain: `probabilistic/evaluation.py`、`tests/test_probabilistic_evaluation.py`、`tests/test_probabilistic_prequential.py`
- Adapt/Delete chain: `data_process/outlier_handling.py`、`tests/test_outlier_handling.py`
- Review/Delete: `models/learning_rate.py`、`models/losses.py`
- Modify: `run.py`
- Regenerate or remove: `requirements.txt`

**工作**：

1. 先做全仓符号级消费者审计；测试消费、包 re-export、场景脚本消费分别列清。
2. CQR 保留纯数学 `calibration.py`；删除未接 canonical 的旧宽表评估编排，除非先把它升级成生产通路。
3. 把算力融合脚本需要的空报告 schema 移到实际消费者/通用结果 schema，再删除 legacy 窗口清洗算法。
4. 合并重复 Ensemble fixture，保留 specs/methods/cache/runtime/persistence 各层最少但完整的测试。
5. CLI 只保留 `--config-yaml`、`--seed` 和明确的 `--output-root`；其他 override 删除，不留 silent no-op。
6. 依赖以 `pyproject.toml + uv.lock` 为唯一事实源；若生产必须用 requirements，则从当前 pyproject 重生并加一致性门禁。

**HARD 验收**：删除项零生产/脚本消费者；无 tests-only production module（明确保留的离线 CLI 除外）；CLI 非声明参数 argparse 直接拒绝；依赖集合一致；全量测试不通过删断言来降数。

### C6 — 当前态文档收口

**Files**：

- Rewrite: `README.md`
- Rewrite: `config/README.md`
- Rewrite: `dataset/README.md`
- Rewrite: `tests/README.md`
- Rewrite: `scripts/README.md`
- Rewrite: 各活动包 README
- Rewrite: `docs/multistep_forecasting_redesign.md` 为当前设计说明
- Rewrite: `docs/production_sync.md` 为当前同步矩阵
- Retire after explicit approval: `docs/model_ensemble_redesign.md`、`docs/time_series_probabilistic_forecasting_redesign.md`
- Condense after implementation: 本文 §1–§10 历史正文（Git 保留）

**工作**：

1. 当前态文档只写当前路径、能力、限制和已实跑命令。
2. 所有历史测试数、迁移方法表、已删除模块和旧输出布局从当前态文档移除；历史交给 git。
3. 测试/配置数字只在一个权威验证快照维护，其余文档用链接引用。
4. `docs/` 内 Python/ipynb 实验资料不处理，仅不再被当前架构文档当生产能力引用。

**HARD 验收**：活动 Markdown 无已删除路径、旧 schema、旧计数和未接线能力声明；所有链接存在；README 快速开始命令实跑 exit 0。

### C7 — 最终全量与真实运行验收

**HARD 命令**：

```bash
# 全量单测
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"

# 语法与分层
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m compileall -q forecasting_core data_loading feature_engineering \
  model_training model_testing model_evaluation model_forecasting \
  probabilistic model_ensemble models decomposition data_process utils \
  config scripts tests main.py run.py

# 现役配置与数据资产
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/audit_runtime_assets.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/audit_ensemble_configs.py

git diff --check
```

**真实运行矩阵（不得用 mock 代替）**：

| # | 配置类型 | 必验内容 |
|---|---|---|
| 1 | fixed-step point 单模型 | 回测训练样本数、final policy、prediction/model/resolved metadata |
| 2 | fixed-step quantile 单模型 | `(N,H,K,Q)`、crossing、概率评分文件 |
| 3 | calendar-month quantile 单模型 | 历史月动态 H + final 目标月完整边界 |
| 4 | `1ME` 月频单模型 | seasonal-naive、月步时间轴、产物复载 |
| 5 | point Ensemble | OOF 防泄漏、cache 复用、bundle/long result |
| 6 | quantile linear_blending Ensemble | pinball 融合、同权重跨 Q、bundle reload |
| 7 | Global N2K2 | series/target 隔离、aggregate 口径、axes 一致 |

**完成定义**：

1. 848 个模型 YAML 全部满足唯一 schema；824 个基模型/成员均通过 runtime constructibility。
2. 活动配置 source 资产零缺失，或配置被明确移出活动集合。
3. fixed-step、calendar-month、monthly 三套时间几何由生产代码同源执行。
4. 24 个 Ensemble 均生成可复载自包含 bundle 与 canonical long result。
5. 包级依赖是 DAG，无延迟 import 绕环。
6. 全量测试、三项配置/资产审计、compileall、diff check 全绿。
7. 七组真实运行产物逐项核对通过；未实跑的能力不得写“完成”。

## 15. 风险、行为变化与控制

| 风险 | 真实影响 | 控制 |
|---|---|---|
| 624 YAML schema 迁移 | fingerprint/identity 变化，旧结果失效 | 生成迁移 manifest；旧键不兼容层；受影响结果统一重跑 |
| 618 配置训练窗重算 | 回测分数和训练成本显著变化 | 先单配置 smoke，再 A/B 各 31 同窗；记录实际 sample count 与耗时 |
| calendar-month 接入 | 旧固定 31 步历史结果不可比 | 按目标月完整折重跑；metadata 写实际 H/month |
| quantile blending 改 proper objective | 10 个配置权重/结果改变 | averaging 作为硬基线；learned 方法必须优于 best single 和 averaging 才推广 |
| Ensemble bundle 落地 | 新产物 schema，部署读取方需同步 | schema version + reload parity + 不依赖 YAML/OOF 的部署测试 |
| forecasting_core 拆包 | import 面大，易漏场景脚本 | AST 全仓门禁 + compileall + 每片全量测试；不留 shim 双轨 |
| 数据文件补齐 | 错误生成可能造成信息泄漏 | 每个生成入口审计 available_at 与目标期覆盖；禁止用真实未来补代理 |
| 删除 tests-only legacy | 仓外消费者未知 | 删除前全仓/场景脚本消费者清单；仓外需求由 wangzf 明确裁决 |
| 文档历史清理 | 丢失迁移背景 | git 保留；当前态文档仅保留必要“为什么”而非执行流水账 |

## 16. v4 状态机与执行纪律

| 状态 | 含义 | 进入条件 |
|---|---|---|
| `reviewed` | 问题和方案已写入本文 | 本节完成并通过文档检查 |
| `approved` | wangzf 批准 C0–C7 或指定切片 | 明确开工指令 |
| `in_progress` | 当前只有一个切片执行 | RED 证据已建立，前置切片 HARD gate 全过 |
| `blocked` | 外部数据/裁决阻断 | 写清阻断、影响配置和替代方案，不伪造结果 |
| `completed` | 单切片完成 | 该片全部 HARD gate 有 fresh 命令证据 |
| `closed` | v4 整改关闭 | C7 七组真实运行 + 全量门禁 + 当前态文档全部通过 |

执行规则：

1. C0–C7 未获明确开工指令前只停留在 `reviewed`；本文更新不等于批准生产改动。
2. 每片先写失败测试，再做最小实现；已知 bug 修复前不录错误 golden。
3. 每片只动当前目标必要文件；不捆绑真实模型效果消融。
4. 配置语义变化使旧结果失效时，先报告受影响 identity，再按用户指示清理/重跑；删除 results 仍需单独确认。
5. 每片完成汇报“改动、fresh 验证、行为变化、剩余风险”；未经明确指示不 commit、不 push。
6. 任何历史文档中的“completed/全绿”都不能覆盖 v4 fresh 失败证据；当前状态以本节状态机和 C7 验收为准。

## 17. v4 整改执行记录（2026-08-31）

| Slice | 状态 | 已落地事实 | Fresh 证据 |
|---|---|---|---|
| C0 | completed | RED/fresh 基线、生产 parser/constructor、活动资产投影视图门禁已建立 | 既有 600-test 与 845-config 证据保留；后续切片新增 RED 按垂直 TDD 执行 |
| C1 | completed | 单模型/Ensemble 共用 strict typed `probabilistic/validation/output`；未知嵌套字段在 loader 前 RAISE | parser/spec/fingerprint 定向测试与 845-config checker 通过 |
| C2 | completed | 方案 A 已落地：`FixedStepBacktestSpec` / `CalendarMonthBacktestSpec`、845 YAML 迁移和语义 manifest | 615 个子日单模型真实 training origin count mismatch=0；TASK27 46/46；geometry 定向总集 97 tests OK |
| C3 | completed | bundle-only point/quantile 部署、优化状态/样本量/fallback 审计和可读 `resolved_model` 已落地 | 76 Ensemble tests + 18 分层/依赖 tests OK；用户批准的 H16 short + H31 calendar-month 两个活动配置均 exit 0 且完整产物核验通过 |
| C4 | completed | 稳定概率合同唯一位于 `forecasting_core/{probabilistic_spec,artifacts}.py`；重复 `probabilistic/spec.py` / `types.py` 已删除 | 概率合同/管线 49 tests OK；删除后生产/测试 import 残留 0 |
| C5 | completed | 三项活动审计改为 typed 双分支/成员类型校验；CLI/checker 不再访问不可达 `.ensemble`；零消费者 grammar 常量删除 | catalog=845（821 single/member + 24 ensemble）；checker 845/845；资产缺失=0；Ensemble 48 refs、failures=[] |
| C6 | completed | README、config/package README、权威设计、AGENTS 和本节已同步 typed geometry 与 deployment 边界 | AGENTS 当前规则旧 geometry 探针命中 0；活动 Markdown 27 份、broken links=0 |
| C7 | completed | fresh unittest、配置/资产/Ensemble/catalog、compileall、diff、Markdown 和两个代表运行均已通过 | 二轮完整证据见 §17.5；v4 关闭 |

### 17.1 已修根因

1. `SourceRegistry._validate_frame()` 增加 path-role 合同后，Global series discovery 漏传 `path_version="history"`；已补齐并由 Global N2K2 执行矩阵覆盖。
2. information-set visibility 与 advanced rolling/difference 最小历史长度统一由生产 runtime 执行，避免 checker 可过而 runtime 不可构造。
3. `model_ensemble` 通过 `EnsembleRuntimeServices` 获取 runner factory 和 bundle persistence，消除对单模型 runtime 实现的反向 import。
4. 单模型 runtime 的 design、fit、calendar-month backtest、persistence 已拆为独立模块；`runtime.py` 只保留 runner 与生命周期编排。
5. `align_to_target=false` 现在真实控制 Direct 历史 lag 原点冻结；默认/target-aligned Direct 仍严格拒绝未来 target lag，known-future 继续按目标时点读取。
6. 原点冻结最大 lag 的最小历史行数修正为 `max_lag+1`；`_supervised_arrays()` 在特征编译前按 typed `validation.history_steps` 截取最近监督 origins，避免先编译全历史再丢弃。
7. Ensemble 现在把融合 OOF actual/prediction 写为 canonical `cv_plot_df.csv`，与点评分和概率评分共同构成完整 test 产物。
8. `DataSourceSpec.columns` 已裁决为入模投影视图：声明列缺失继续 RAISE，未声明物理列在 registry 边界丢弃且绝不隐式入模；新增回归测试锁定投影行为。
9. legacy runtime 的 `cal_rh` 现场派生已迁到离线数据入口：23 个 CSV/27,180 行按同一 Magnus–Tetens 公式计算并沿用线性/双向填充；逐文件验证去除新列后的原始单元格 SHA 不变。
10. 3 个 `add_training_inference_pod` 配置因声明列与资产列族错配，经批准按字节原样迁至 `docs/archived_configs/`；活动模型基线调整为 845，不放松 SourceRegistry 缺列校验。
11. C2 方案 A 将 fixed-step 与 calendar-month 分成两种 typed geometry，并迁移全部 845 YAML；迁移 manifest 记录新旧字段/fingerprint，615 个子日单模型的实际 final training origin count 与 manifest 全部一致。
12. Ensemble bundle 新增无需成员 YAML/OOF cache 的 point/quantile 部署入口；调用方只提供按 bundle input schema 编译的特征和必要的 feature provider，scaler、selection、策略 artifact、target 逆变换与融合权重均来自 bundle。
13. Quantile linear blending 产物保存每 target 的 quantile grid、有效样本数、optimizer success/status/message 与 fallback reason；同一审计同时写入 pickle artifact 和可读 `resolved_model.json`。
14. `audit_forecast_configs.py` 改为生产 parser 的 `ForecastConfigSpec | EnsembleConfigSpec` 双分支 catalog；`main.py`/checker 的不可达 `.ensemble` 读取和单模型 parser 内两个零消费者 ensemble grammar 常量已删除。
15. 重复概率合同 `probabilistic/spec.py` / `types.py` 经全仓消费者审计后删除；稳定合同唯一位于 `forecasting_core/probabilistic_spec.py` 与 `forecasting_core/artifacts.py`，负空间测试禁止恢复双轨。

### 17.2 首轮 C7 历史证据（已撤回关闭效力）

| Gate | 命令摘要 | Fresh 结果 |
|---|---|---|
| 全量 unittest | `python -m unittest discover -s tests -p "test_*.py"` | **600 tests / 197.432s / OK** |
| 语法与 diff | `python -m compileall -q ... && git diff --check` | exit 0 |
| 配置 | `python scripts/check_model_configs.py` | `checked=845 passed=845 hard_failures=0 warnings=0` |
| 路径资产 | `python scripts/audit_runtime_assets.py` | `model_configs=845 unique_paths=124 missing_unique=0 missing_refs=0 affected_configs=0 missing_column_refs=0 column_affected_configs=0` |
| Ensemble | `python scripts/audit_ensemble_configs.py` | 797 single + 24 ensemble + 24 member；48 refs；failures=[] |
| Source header | 活动 YAML 非 ignored 声明列 vs CSV header | missing refs/configs/files=`0/0/0` |
| 分层/依赖 | `python -m unittest tests.test_package_layering tests.test_dependency_contract -v` | 18 tests / OK |
| 活动 Markdown | README/AGENTS/权威文档链接与旧路径探针 | 25 份；broken links=0；AGENTS.md 当前合同已同步 |

### 17.3 七类最小运行矩阵

统一临时根：`/tmp/tsproj_ml_c7.oy7vws`。每类均核对 model.pkl、resolved_model、test score、cv long、prediction、resolved_config、bundle reload、fingerprint 和唯一键。

| 类别 | 配置/来源 | Fresh 产物证据 |
|---|---|---|
| fixed-step point | `config/aidc_load_month/route_A/lgbm_usmr_mean.yaml` | `(1,34,1)`；prediction=34；cv=170/5 windows |
| fixed-step quantile | `config/aidc_load_month/route_A/lgbm_usmr_prob_mean.yaml` | `(1,34,1)`；q10/q50/q90；probabilistic scores；cv=170/5 windows |
| calendar-month quantile | `config/aidc_power_month/route_B/freq_1day/baseline/lgbm_usmr_prob_mean_conformal.yaml` | 2026-08-01..31；prediction=31；cv=181/6 windows |
| `1ME` point | `config/aidc_power_month/route_B/freq_1month/window_length_8/st_usmd_mean.yaml` | `(1,1,1)`；forecast=2026-08-31；cv=3/3 windows |
| point Ensemble | `config/aidc_load_15min_short/route_B/add_endogenous_cross_route_weather_date/lgbm_usbr_mean.yaml` | averaging；2 self-contained members；prediction/cv=16；OOF `2eed336d...` |
| quantile linear blending | `config/aidc_power_month/route_B/freq_1day/baseline/lgbm_usbr_prob_mean_conformal.yaml` | pooled-pinball；2 members；prediction/cv=31；OOF `94173aec...` |
| Global N2K2 | 活动 YAML 数量为 0；经批准使用独立 target-history fixture 作能力验收 | recursive ridge；`(2,4,2)`；prediction=16；cv=32/2 windows；series A/B、targets load/power |

### 17.4 保留边界与审计备注

1. 活动 YAML 中仍没有生产 Global N2K2 配置；用户已明确接受完整 fixture 作为能力验收，文档不得把该 fixture 表述为现役生产配置。
2. 23 个迁移后的天气 CSV 位于 gitignored `dataset/`，真实数据已落盘且 SHA/formula 校验通过；可复现脚本和测试进入工作树，代码 review 不会展示数据文件本体。
3. 旧 quantile Ensemble 临时根的 OOF cache 在复用时正确 RAISE hash mismatch；最终证据改用独立新根生成，未删除旧临时 cache。

以上证据是 2026-08-31 首轮整改的历史快照；经同日复核，因 C1/C2/C3/C5/C6 HARD gate 尚有缺口，不能据此关闭 v4。当前状态以 §17 表格为准，待新增切片全部 fresh 验收后再记录最终关闭结论。

### 17.5 二轮整改 fresh 证据与剩余阻塞

| Gate | Fresh 结果 |
|---|---|
| 全量 unittest | `615 tests / 305.814s / OK`；TASK27 `46/46` |
| 配置 checker | `checked=845 passed=845 hard_failures=0 warnings=0`；Ensemble semantic problem 现在进入 hard failure |
| Typed catalog | 845 行：821 single/member + 24 Ensemble；845 unique paths；64 位 fingerprint 异常 0 |
| 运行资产 | `model_configs=845 unique_paths=124 missing_unique=0 missing_refs=0 missing_column_refs=0` |
| Ensemble refs | 797 single + 24 Ensemble + 24 member；48 refs 逐项解析为 `ForecastConfigSpec`；`failures=[]` |
| 分层/依赖 | 18 tests / OK；包 DAG 无环 |
| 语法/diff/文档 | compileall exit 0；`git diff --check` exit 0；活动 Markdown 27 份、broken links=0 |

用户裁决原计划的 10 个 quantile linear-blending 全跑不是本轮代码正确性的必要门槛，改为两个代表配置。统一临时根：`/var/folders/h1/zznyfyrx5qjc4x410n2xpjc00000gn/T/tsproj_ml_quantile_blending_representative_q91i4y9v`。

| 代表 | 正式运行与产物核验 |
|---|---|
| H16 short | `run.py` exit 0；OOF `189dad161c9a...`；prediction/cv=16/16；point/probabilistic score rows=2/18；q10/q50/q90；2 个内嵌成员；optimizer success=true；fallback=null |
| H31 calendar-month | `run.py` exit 0；OOF `d239c3ae1182...`；prediction/cv=31/31；point/probabilistic score rows=2/18；q10/q50/q90；2 个内嵌成员；optimizer success=true；fallback=null |

首次自动 verifier 在 `/tmp` 反序列化 bundle 时未把仓库根加入 `sys.path`，把两个 exit-0 运行误记为 `ModuleNotFoundError`；随后修正 `sys.path`，并按 canonical long schema 使用 `predict_value` 而非错误假定的 `predict` 列，直接对原产物重验为 2/2 passed。最终 manifest 保留 `initial_verifier_error` 与纠正标记，不隐藏验收工具偏差。

项目根 `AGENTS.md` 已经用户审批同步为 typed geometry，旧几何字段探针命中 0；C6/C7 的最后阻塞解除，v4 二轮整改关闭。
