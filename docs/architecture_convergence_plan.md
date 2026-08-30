# 项目架构收敛设计与重构实施方案

> 文档状态：**v3 执行完成（2026-08-30）**——P0–P4 五片全部落地；执行期新增两项超范围工作（折合同配置修复 618 YAML、信息集性能修复）经 wangzf 追加裁决后完成，见 §10 执行记录与偏差；测试基线 562 tests OK / check_model_configs 848 passed / audit_ensemble 零失败。**任何片均不自动 commit**（review-then-commit），全部改动待 review。
> 创建日期：2026-08-29（v1）／ v2：纳入 ensemble v4 现状 ／ v3：新增时序预测流水线第一性原理逐项审计（§2.7–2.8，决策 D12–D15）
> 作者：Machine-C（MC）
> 定位：本文是 `docs/multistep_forecasting_redesign.md`（预测架构权威文档）的**包间边界补充 + 流水线完整性审计**，不改动 canonical 内部设计与 v4 引用式融合的已验收行为。诊断基于 2026-08-29 对**当前工作区**的逐文件核查，全部结论附 `path:line` 证据。

> **版本变更链**：
> **v1 → v2**：工作区相继完成 canonical legacy 收口与 ensemble v4 两次大重构，撤销 D5（pickle 垫片，存量 pkl 已清空）、改 D2 为整包删除（契约字段词汇在现役 848 YAML 零出现）、缩小 D3 范围（E1 已抽 `forecasting/validation.py`）、新增 §2.5 ensemble 专项分析。
> **v2 → v3**：应用户要求，从「时序预测 训练/测试/预测 第一性原理」对全部 13 个主题逐项审计（§2.7），新增发现 F1–F8 与决策 D12–D15：评估掩码 `eval_mask` 语义在 canonical 迁移中被静默丢弃（F6，14 个 YAML + 审计白名单活着、运行时死了）、在线缺失/异常处理默认无声明（F1/F2）、decomposition 双链死代码（F7）、GridSearchCV 死导入（F5）、§3.1 分层图与依赖方向勘误（F8）。
> **v3 裁决（2026-08-29，wangzf）**：D13=rewire、D12=只立契约、D7/D14/D15 死代码全部删除、D9=文档声明差异为有意、P1 消环先行——执行按裁决推进，详见 §9。

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

**方案关闭**：v3 范围内工作全部完成或显式撤销（D14 误诊）；新增工作（折合同、性能）已并入本文档记录。commit 待 wangzf 指示。
