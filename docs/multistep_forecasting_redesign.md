# 多步预测正交架构设计与实施记录

> 文档状态：Implemented / 全部实施切片完成，待人工 review 与 commit
> **模型融合重设计（ensemble redesign, v4）：核心实施完成（E0–E7，2026-08-29）**——计划见 `.hermes/plans/2026-08-29_172929-model-ensemble-redesign.md`（副本 `docs/model_ensemble_redesign.md`）；E8 效果消融待独立授权。引用式融合：顶层 `ensemble/` 包（specs/loader/contracts/oof/cache/artifacts/trainer/predictor/runtime/methods），四方法 averaging/weighted/linear_blending/stacking，OOF SHA-256 内容寻址缓存，`CanonicalBaseModelRunner`（forecasting.runtime）为成员唯一通路。单模型主链行为逐值不变（tests/test_ensemble_parity.py 分级 parity：单模型 golden `a8d1ae0a5c44`、weighted 端到端 golden `e4c3eb2b7dbf`/`bf0677d2a169`、NNLS 函数级 golden；stacking 迁移回测指标允许变化、旧结果作废）。24 配置迁移：Direct 引用现有单模型、新增 24 个 `ensemble_members/*_recursive.yaml`，方法分布 averaging=12 / linear_blending=12，calendar_month 仅 `fold_count=1`。验收（2026-08-29 fresh）：`check_model_configs.py` **checked=848 passed=848 hard_failures=0 warnings=0**；`audit_ensemble_configs.py` 零失败（800+24+24=848、48 引用有效、孤儿缓存报告）；全量 unittest **555 tests OK**；compileall / git diff --check 干净。`run.py` 已删 `--is-testing/--is-forecasting`；`ForecastModelBundle` 校验分单模型/ensemble 双分支（ensemble bundle `model_type="ensemble"`、top-level strategy_spec/estimator_spec 均 None）；`models/ModelEnsemble.py` 与 `forecasting/ensemble.py` 兼容 shim 已删除（零生产消费，`fit_nonnegative_stacking_weights` 迁至 `ensemble/methods/linear_blending.py`）；`ForecastConfigSpec` base-only（`strategy=None` 构造期 RAISE）。基线对照：原 824 YAML/24 ensemble/485 tests（2026-08-29 锁定）
> 创建日期：2026-08-27；重构完成：2026-08-29
> 批准计划：`.hermes/plans/2026-08-27_113009-forecast-problem-orthogonal-redesign.md`
> legacy 状态：迁移已完成，legacy 只读层与迁移工具已于 2026-08-29 删除（见 §8）；旧产物在本仓库不可读

## 1. 文档定位

本文是 tsproj_ml 预测架构的唯一权威文档，描述**当前实现**并记录实施结果。全项目不再按版本世代区分架构：

- **canonical（当前架构）**：824 个现役模型 YAML 均以 `schema_version: 2` 声明；`main.py::CanonicalModel → forecasting.runtime.run_canonical_config()` 是唯一运行时，使用七种标准多步策略、显式数据角色、统一预测张量 `(N,H,K[,Q])`、source registry/FeatureCompiler、schema-2 `ForecastModelBundle` 和 long result。
- **legacy（已删除）**：2026-08 之前的九种 US/MS 方法、扁平 `base_config + overrides` 配置、旧 pkl/dict 产物与宽表结果。只读层与迁移工具已于 2026-08-29 删除（§8），旧产物在本仓库不可读。
- **术语约定**：`schema_version: 2` 是配置与产物的格式标记字段（类似文件格式版本号），不代表「某一代架构」；文档与代码叙述只用 canonical / legacy 二分。

历史设计文档已被本文取代并合并，历史细节通过 git 溯源。

## 2. 历史动机

2026-08 前的实现把预测问题编码进方法名，导致四类结构性缺陷：

1. **方法名承担过多职责**：输入范围（单/多变量）、rollout 语义、训练布局、Global 开关和融合权重全被拼进九个 US/MS 方法身份，无法独立组合。
2. **训练目标表被当成策略协议**：用 `shift_1..H` 列名和输出宽度反推推进语义，策略与数据耦合。
3. **推理上下文是大型可变对象**：跨阶段共享可变状态，边界不清。
4. **兼容策略偏向「继续跑」**：`USMDR/MSMDR` 实为单模型分块递归（标准术语 RecMO）却命名为 DirRec；`USBR/MSBR` 的 Direct+Recursive 加权融合被当作第九种多步策略；空 `target_series_numeric_features` 隐式推断内生列并默认 persistence 回填。

正交化结论：strategy、数据角色、target coupling、TrainingScope、概率模式拆为独立维度；Blend 归入 Ensemble，Pointwise/horizon-feature 归入 Direct 的训练布局，Global 归入 TrainingScope。

## 3. 预测张量合同

| 符号 | 含义 |
|---|---|
| `N` | series/entity 数；Local 时 `N=1` |
| `H` | forecast horizon，即预测步数 |
| `K` | target 数；单目标时 `K=1` |
| `Q` | quantile 水平数 |
| `S` | 联合样本数；仅保留类型与 shape 合同 |

输出形状：

- point：`(N,H,K)`；
- marginal quantile：`(N,H,K,Q)`，只表示各 target 的边际分布；
- samples：只保留 `(N,S,H,K)` 类型合同，生成器明确 `NotImplementedError`，metadata 记录 `dependence_model=None`，不得据此宣称联合概率能力。

Estimator adapter 如需二维标签，固定 time-major 顺序：`(h1,target1), …, (h1,targetK), (h2,target1), …`，该顺序随 artifact 保存，不依赖字典序或列名猜测。

## 4. 数据角色与严格可得性

### 4.1 唯一允许的数据角色

| 角色 | 时间语义 | 预测期要求 |
|---|---|---|
| `target` | 需要预测的变量；历史可见 | 未来值只能由 strategy 产生 |
| `observed_past` | 仅在观测发生后可见 | 预测原点之后不得直接读取 |
| `known_future` | 预测原点已知未来计划/日历/场景 | 必须覆盖所需 horizon，且通过 as-of 校验 |
| `static` | 无时间维，按 entity key 关联 | 预测原点可用，实体键必须完整 |
| `key` | time、series/entity 等主键 | 仅用于索引、关联和隔离 |
| `ignored` | 明确不参与建模 | loader/compiler 必须排除 |

每个输入列必须且只能拥有一个角色；禁止通过「目标以外的数值列」自动推断 observed-past，禁止通过文件类别（date/weather/custom）代替语义角色。

### 4.2 可得性规则

所有动态 source 必须声明 `source_time`、`available_at` 语义，并在每个 forecast origin 满足 `available_at <= forecast_origin`。对目标时间 `target_time`：

- `target`：`target_time > forecast_origin` 时真实值不可见；
- `observed_past`：只能使用 `source_time <= forecast_origin` 的记录；
- `known_future`：值可对应未来 `target_time`，但 `available_at` 必须不晚于 `forecast_origin`；
- `static`：按 entity key 唯一匹配，不允许跨 series 回填；
- 缺失、重复、越界、as-of 违规均 RAISE，不静默裁剪或替换。

### 4.3 禁止隐式 persistence

`observed_past` 变量在预测推进中需要未来值时，只允许三种显式建模：提升为 `target`；绑定显式 forecast provider；改为有合规来源的 `known_future` scenario。禁止隐式 persistence / 默认冻结最后值。

## 5. 七种标准多步策略

设块长 `B`，多目标输出宽度 `K`：

| Strategy | 标准定义 | 模型数量 | 单次输出 | 推进依赖 |
|---|---|---:|---|---|
| `recursive` | 一个一步模型，逐步预测并回填 | 1 | `K` | 每步消费此前预测 |
| `direct` | 每个 horizon 一个独立模型 | `H` | `K` | 各 horizon 不消费此前预测 |
| `mimo` | 一个模型一次输出全部 horizon 与 target | 1 | `H×K` | 单次完成 |
| `recmo` | 一个模型每次输出一个块并按块递归 | 1 | `B×K` | 后续块消费此前块预测 |
| `dirrec` | 每个 horizon 一个模型，第 h 个消费此前预测 | `H` | `K` | 逐 horizon 递归依赖 |
| `dirmo` | 每个块一个独立模型 | `H/B` | `B×K` | 各块不消费此前预测 |
| `dirrecmo` | 每个块一个模型，后续块消费此前块预测 | `H/B` | `B×K` | 按块递归依赖 |

严格块约束：`recmo`/`dirmo`/`dirrecmo` 必须满足 `1 < B < H` 且 `H % B == 0`；`B=1` 必须使用 recursive/dirrec，`B=H` 必须使用 mimo；非整除尾块在配置解析期直接 RAISE，不得由运行时截断掩盖。

策略只在 `forecasting/specs/strategy.py` 定义，由 `training/strategies/` 的七个 executor 执行（2026-08-30 流水线阶段重排后位置）。

## 6. 正交规格与能力边界

canonical 配置由以下正交 specs 组成：

| Spec | 职责 | 不得包含 |
|---|---|---|
| `ForecastProblemSpec` | targets、horizon、频率、keys、预测模式 | estimator 类型 |
| `DataSpec` | source、列角色、时间列、entity key、availability | rollout 分支 |
| `FeatureSpec` | lag、datetime、交互、可见性驱动的生成规则 | 通过 US/MS 名称猜语义 |
| `ForecastStrategySpec` | 七种标准多步推进与合法 chunk | 单/多变量、Global、Blend |
| `EstimatorSpec` | model、params、target adapter、能力要求 | horizon 推进语义 |
| `TrainingScopeSpec` | Local/Global、series key、完整性策略 | strategy 别名 |
| `ProbabilisticSpec` | point/quantile/samples 模式与 quantile grid | 把边际 quantile 说成联合分布 |
| `EnsembleSpec` | Direct+Recursive 等成员组合与权重 | 伪装成第八种 strategy |

多目标 estimator adapter：

| Adapter | 语义 | 能力门禁 |
|---|---|---|
| `independent` | 每个输出维度独立拟合 | 必须记录无 target dependence 建模 |
| `regressor_chain` | 固定顺序逐输出建模 | 顺序固定并写入 artifact；不支持的 quantile 组合 RAISE |
| `native` | 使用估计器原生 multi-output | 仅在能力检测明确验证后可选；不得因 fit 恰好接受矩阵就假定完整支持 |

能力检测由 `training/estimators/capabilities.py` 显式探测，结果进入 resolved config、运行日志和 artifact；禁止静默从 `native` 降为 `independent`。

## 7. 模块架构

```text
forecasting/
  specs/        # problem/data/feature/strategy/estimator/config 不可变 specs
  tensors.py    # Point/Quantile/Sample ForecastTensor
  data/         # SourceRegistry + InformationSetRequest + providers
  features/     # FeatureCompiler（schema/lineage/visibility proof）
  strategies/   # recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo 七个 executor
  estimators/   # capabilities + multi_target adapter
  validation.py # 共享 rolling-origin 时间合同（outer backtest 与 ensemble OOF 同源）
  backtest.py   # 回测公开原语：seasonal-naive/actual/origin/validation-int + eval_mask 入口
  transforms.py # per-(series,target) target/feature transform + 目标变换栈（calendar/decomposition/scaling，2026-08-29 自 features/ 并入）
  trainer.py    # CanonicalTrainer / DirectMultiOutputRegressor / HorizonAlignedDirectRegressor（自 models/ 迁入）
  forecaster.py # CanonicalForecaster（自 models/ 迁入）
  runtime.py    # rolling backtest、final training、forecast 编排 + CanonicalBaseModelRunner（成员唯一通路）
  results.py    # canonical long result reader/writer + eval_mask 评估口径（D13 rewire）
```

包间分层（2026-08-29 架构收敛，权威规则见 AGENTS.md「包间分层规则」与
`docs/architecture_convergence_plan.md` §3.1）：L4 入口（main/run/config_loader）→
L3 能力扩展（probabilistic/、ensemble/，唯一豁免 `ensemble → probabilistic.types`）→
L2 预测（forecasting/）→ L1 基础设施（models/ 工厂+序列化、decomposition/、data_process/）→
L0 基础（utils/）。依赖只允许从上往下；`models/ModelTraining.py`、`models/ModelForecasting.py`
为兼容 shim（仅转发 forecasting/trainer.py、forecasting/forecaster.py）；
`features/`、`data_provider/` 已删除（P2/P4），目标变换与 outlier_handling 分别迁入
`forecasting/transforms.py` 与 `data_process/outlier_handling.py`（后者 legacy 身份，不接 canonical runtime）。
折生成双实现为有意分工（D9）：`validation.rolling_origin_folds`（外层回测：无训练样本 RAISE）
与 `ensemble/oof.py::oof_fold_origins`（OOF：gap/outer-cutoff 防泄漏特化、无样本跳过）共享
`TimeGeometry` 单一时间合同，不得合并。

依赖方向：`specs + tensors → data registry/information_set → feature compiler → strategy plan / estimator adapter → Trainer/Forecaster 编排 → results/artifact`。

审计入口：`scripts/audit_forecast_configs.py`、`scripts/check_model_configs.py`（迁移器与失效清单生成器已随 legacy 删除）。

## 8. Legacy 体系删除记录（历史迁移档案）

### 8.1 九种 legacy 方法的语义映射（历史记录）

2026-08 完成迁移、2026-08-29 只读层亦被删除。九种方法名不再存在于任何代码或配置，以下映射仅作历史档案：

| 缩写 | 旧 `pred_method` | 真实语义 | canonical 映射 |
|---|---|---|---|
| USMDP | `univariate-single-multistep-direct-pointwise` | 单变量，按未来时点逐点 direct | `direct` + target-history-only + single-model-horizon 布局 |
| USMD | `univariate-single-multistep-direct` | 单变量多输出 direct | `direct` + target-history-only |
| USMR | `univariate-single-multistep-recursive` | 单变量一步递归回填 | `recursive` + target-history-only |
| USMDR | `univariate-single-multistep-direct-recursive` | 单变量固定块模型重复推进（RecMO，非标准 DirRec） | `recmo` |
| MSMD | `multivariate-single-multistep-direct` | 多变量 direct | `direct` + 显式 observed-past provider |
| MSMR | `multivariate-single-multistep-recursive` | 多变量递归，其他内生回填 | `recursive` + 显式 observed-past provider |
| MSMDR | `multivariate-single-multistep-direct-recursive` | 多变量固定块重复推进（RecMO） | `recmo` + 显式 observed-past provider |
| USBR | `univariate-single-multistep-blend-direct-recursive` | 单变量 Direct+Recursive 加权融合 | `ensemble(direct, recursive)` |
| MSBR | `multivariate-single-multistep-blend-direct-recursive` | 多变量 Direct+Recursive 加权融合 | `ensemble(direct, recursive)` + 显式 provider |

迁移基线分布（831 个，含后续下线的 21 个 ETT-small 演示配置）：`msbr=8, msmd=21, msmdr=21, msmr=21, usbr=16, usmd=354, usmdp=124, usmdr=112, usmr=154`。现役分布（824 个）：`msbr=8, msmd=20, msmdr=20, msmr=20, usbr=16, usmd=353, usmdp=123, usmdr=111, usmr=153`；canonical identity 分布：`direct=496, ensemble=24, mimo=55, recmo=58, recursive=191`。

### 8.2 18 个非整除 RecMO resolution（已批准，历史记录）

18 个旧 MDR 配置因 `H % B != 0` 映射为 `recursive`：`aidc_power_month` 14 个（H=31,B=7）、`aidc_load_month` 4 个（H=34,B=7）。2026-08-29 wangzf 明确批准。迁移期 YAML 曾记录 `validation.legacy_origin.approved_resolution.name=recursive`；该溯源字段已随 legacy 清理删除，本节文字是批准记录的唯一档案。

### 8.3 legacy 层删除清单（2026-08-29 收口）

| 删除项 | 原职责 |
|---|---|
| `config/config_sections.py`、`univariate_config.py`、`multivariate_config.py` | legacy 扁平配置 dataclass 与模板 |
| `config/config_loader.py` legacy 分支、`load_model_config()`、overrides 机制 | 旧 YAML 只读解析与 override 应用 |
| `forecasting/legacy/`（由 `models/multistep/` 迁入后删除） | 九方法目录/解析/plans/panel/artifact 适配/methods 翻译 |
| `data_provider/data_loader.py`、`features/FeatureEngineering.py`、`models/AuxiliaryForecaster.py` | legacy 数据加载、特征工程、辅助内生预测 |
| `probabilistic/training.py::QuantileTrainer`、schema-1 `ProbabilisticModelBundle`、`BlendQuantileModel`、`migrate_legacy_quantile_bundle` | legacy 概率训练与 bundle |
| `forecasting/results.py::LegacyResultReader`、`ModelSaveLoad` 的 `LegacyArtifactAdapter` 回退 | 旧宽表/旧 pkl 只读适配 |
| `scripts/migrate_forecast_configs.py`、`scripts/generate_result_invalidation_manifest.py` | 迁移器与失效清单生成器 |
| 41 个 legacy 模型 YAML 与全部 YAML 的 `validation.legacy_origin` 块 | 旧配置本体与溯源标记 |

同时清理：legacy datetime 别名（`weekday`→`day_of_week`、`week`→`week_of_year`，824 YAML 中 304 行改写并去重）。结构门禁 `tests/test_canonical_only_runtime.py` 断言上述路径全部不存在。旧 pkl/旧宽表从此在本仓库不可读；生产侧如需消费历史产物必须自带独立解析器。

## 9. 结果与产物合同

canonical 输出目录：

```text
results/<scenario>/<result_identity>/
  pretrained_models/{model.pkl,resolved_model.json}
  results_test/{cv_plot_df.csv,test_scores_df.csv,result_metadata.json,...}
  results_forecast/{prediction.csv,resolved_config.json}
```

- `result_identity = <strategy|ensemble>-<model>-<scope>-k<K>-<fingerprint[:12]>`；fingerprint 只取语义 payload，并行度、日志、开关、输出目录不参与。
- `prediction.csv`/`cv_plot_df.csv` 使用 `(series_id,time,target[,window])` long 唯一键；`test_scores_df.csv` 写 per-target 与 aggregate 行。
- 目标训练变换顺序固定为 `calendar normalization → decomposition → target scaling`，point/quantile 严格逆序恢复；状态按 `(series_id,target)` 隔离并随 bundle 保存。
- 新模型只写 schema-2 `ForecastModelBundle`（含 canonical problem、strategy 或 ensemble spec、estimator capabilities、固定 output order、series order、feature/source lineage、transform 状态、config fingerprint）；普通策略保存 `CanonicalStrategyArtifact`，融合由顶层 `ensemble/` 包产出 method artifact 与 `OOFPredictionArtifact`/`EnsembleArtifact`（v4 引用式融合，见文档头部记录）。
- canonical 当前只输出 `predict_q*`；`probabilistic.conformal` 不属于 canonical 能力，canonical runtime 不执行 CQR、不输出 `predict_pi*`。

## 10. 结果失效结论（历史记录）

迁移导致全部 legacy 结果失效：配置 schema、strategy/ensemble、信息集、输出张量、result schema、artifact schema 和 identity 均已改变。旧结果已随用户清理移出（2026-08-29），`CanonicalResultReader` 只接受 canonical long schema，旧宽表/旧 pkl 从此不可读。

失效清单生成器（`scripts/generate_result_invalidation_manifest.py`）与迁移器（`scripts/migrate_forecast_configs.py`）已随 legacy 层删除；失效审计计数的历史结论（824 行、recorded=18、pending=0、68 个别名共享 run path）以 2026-08-29 再生校验为准，机器清单不再驻留仓库。

使用规则：

1. `results/` 只存在 canonical 布局；不得引入旧宽表或旧 pkl。
2. 新运行固定写 `results/<scenario>/<new_result_identity>/...`；fingerprint 不匹配即视为不同实验。
## 11. 实施完成记录

### 11.1 切片台账

| Slice | 内容 | 状态 | 证据 |
|---|---|---|---|
| R0 | 基线与设计文档 | completed | 2026-08-27 批准；831 配置审计基线 |
| R1 | 预测张量与问题 specs | completed | `forecasting/tensors.py`、`forecasting/specs/`；E1、E6 |
| R2 | 配置解析与 legacy adapter | completed | `forecasting/specs/config.py`、`forecasting/legacy.py`、`config/config_loader.py`；E1–E4 |
| R3 | source registry + information set | completed | `data_loading/`（2026-08-30 迁为顶层包）；E1、E6、E7 |
| R4 | feature visibility compiler | completed | `feature_engineering/compiler.py`（2026-08-30 迁为顶层包）；E1、E7 |
| R5 | 七策略 plan 与 executor | completed | `training/strategies/`（2026-08-30 迁为顶层包）；E1、E6 |
| R6 | multi-target estimator adapters | completed | `training/estimators/`（2026-08-30 迁为顶层包）；E1、E6 |
| R7 | Trainer/Forecaster 张量化 | completed | `CanonicalTrainer`、`CanonicalForecaster`、`forecasting/runtime.py`；E1、E6、E7 |
| R8 | transform/probabilistic 张量化 | completed | marginal quantile `(N,H,K,Q)`；831/831 transform/compiler constructible；**不含 CQR** |
| R9 | Global Panel K>1 | completed | N=2,K=2 七策略通过；E1、E6、E7 |
| R10 | Ensemble/旧 BR 迁移 | completed | `forecasting/ensemble.py`、nested stacking；E1、E6、E7 |
| R11 | result/artifact | completed | long CSV、schema-2 bundle、legacy 只读；E1、E6、E7 |
| R12 | 全量 YAML 一次迁移 | completed | 831/831 数量守恒；18 个 resolution 获批；E2–E4 |
| R13 | 全仓收口与删除门禁 | completed | legacy generator、旧运行类、旧 executor 删除；只读层保留；E1、E5–E7 |

### 11.2 最终验收证据

以下 E1–E7 是 2026-08-29 收口验收的原始记录（当时配置总量 831）；同日稍后 ETT-small 21 个演示配置下线、electricity_computility 14 个 Python 模板配置转为 canonical YAML，现役 824 个，checker 当前输出 `checked=824 passed=824 hard_failures=0 warnings=0`，其余证据不受配置增减影响。

**E1 — 全量 unittest（exit 0）**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run \
  python -m unittest discover -s tests -p "test_*.py"
```

结果：**617 tests，fail=0，error=0，skip=0**。

**E2 — 831 个现役模型配置 checker（exit 0）**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/check_model_configs.py
```

结果：`checked=831 passed=831 hard_failures=0 warnings=0`；另有 41 个聚合/检测 YAML 被精确排除。

**E3 — 配置 audit 与 transform 构造（exit 0）**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/audit_forecast_configs.py --root config \
  --output /tmp/tsproj_ml_forecast_config_audit.json
```

结果：模型 YAML 831、唯一路径 831、全部 canonical schema；按生产 ensemble member 构造路径复核 `model_configs=831 transform_constructible=831 compiler_instances=855 problems=0`。

**E4 — 从 HEAD 只读迁移复演（exit 0）**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/migrate_forecast_configs.py --root config --revision HEAD \
  --repository . --resolutions /tmp/tsproj_ml_recmo_resolutions_reconstructed.yaml \
  --report /tmp/tsproj_ml_migration_report_from_head.json
```

结果：831 路径与当前配置集合完全一致，canonical identity mismatch=0，legacy semantic coverage union=154；18 条 `non_divisible_recmo_tail` 均按批准 resolution 复演。

**E5 — compile/diff（均 exit 0）**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m compileall forecasting config data_provider features models probabilistic utils main.py run.py
git diff --check
```

**E6 — 执行矩阵（exit 0）**

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest \
  tests.test_config_entrypoints.Task27ExecutionMatrixTest.test_task27_full_execution_matrix_reports_exact_counts -v
```

结果：`TASK27_MATRIX cases=46 passed=46`，涵盖 Local K1/K2、Global N2/K2、point/quantile、七策略、weighted/stacking Ensemble、legacy config/artifact 只读。

**E7 — 独立 temp smoke（exit 0）**

真实 canonical runtime 四组：Local K1 point `(1,2,1)`、Local K1 quantile `(1,2,1,3)`、Global point `(2,4,2)`、Ensemble point `(1,2,1)`；每组核对 `prediction.csv/test_scores_df.csv/cv_plot_df.csv/model.pkl/resolved_model.json/resolved_config.json`，long key 唯一、bundle/config fingerprint 一致、result identity 与路径一致。CQR 不在验收范围，未运行。

### 11.3 偏差记录

| 编号 | 偏差 | 处置 |
|---|---|---|
| D1 | 旧实现将九方法视为稳定外部身份 | 九方法仅保留在 legacy mapping；新配置只用七 strategy |
| D2 | legacy `DirRecExecutor` 实际执行单模型分块递归 | 旧 MDR 标记 legacy RecMO；真正 DirRec 在 R5 实现 |
| D3 | BR 被纳入 rollout family | R10 迁入 `EnsembleSpec` |
| D4 | pointwise/horizon-feature 占据方法或策略表述 | 统一归入 Direct implementation |
| D5 | Global 由布尔开关混入运行配置 | 迁入 `TrainingScopeSpec` |
| D6 | DataLoader 可推断内生列并允许 legacy persistence | R3/R4 改为显式角色、provider 和 as-of 校验 |
| D7 | 旧输出以单目标 `(H,)`/`(H,Q)` 为中心 | R1/R7/R8 全链路保留 N/H/K/Q 维度 |
| D8 | 早期计划把 horizon 同时放入 ProblemSpec 与 StrategySpec | horizon 只归 `ForecastProblemSpec`；strategy resolve 生成私有 resolved plan |
| D9 | 18 个 legacy RecMO 的 H 不能被 B 整除 | 2026-08-29 批准正式映射为 `recursive`，见 §8.2 |
| D10 | 迁移器保留 138 个配置的 CQR 意图，但 canonical 未纳入 CQR runtime | 本轮非目标；后续需要 CQR 必须单独设计/TDD 接线 |
| D11 | 831 配置只有 761 个唯一 fingerprint、763 个唯一 run path | 同语义配置别名共享路径，不是 hash 碰撞 |
| D12 | 架构收敛 P4（2026-08-30）：D14 预判 decomposition/forecasters.py+composers.py 为死链 | **误诊撤销**——实为 DecompositionPipeline 主链包内活代码（pipeline.py:15-17,:42,:69-76,:116-132），文件保留；诊断 grep 曾排除包自身导致漏判 |
| D13 | 架构收敛（2026-08-29，P0–P4 五片完成，562 tests OK、check_model_configs=848 passed×每片、audit_ensemble 零失败） | Trainer/Forecaster 迁 forecasting/trainer.py、forecaster.py（models 留 shim）；features/、data_provider/ 整包删除（目标变换并入 transforms.py、outlier_handling 迁 data_process 带 legacy 身份）；回测原语公开化为 backtest.py（E-B 修复）；eval_mask rewire 进 results.py（掩码未配置=行为零变化，14 配置 fingerprint 不变）；ModelDeployPmml/PMML 依赖、utils/metrics+cv_plot、weighted 私有名等清扫；证据与偏差详见 docs/architecture_convergence_plan.md §7 |
| D15 | 流水线阶段重排 + 评估模块化（2026-08-30，wangzf 裁决方案 C） | forecasting/ 收缩为核心合同（specs/tensors/transforms）+ 结果 IO（results.py）+ 唯一编排器（runtime.py）；阶段顶层包：`data_loading/`（数据构造与组织）、`feature_engineering/`（特征编译）、`training/`（trainer+strategies+estimators）、`testing/`（validation+backtest 原语）、`evaluation/`（★独立评估模块：point/marginal/mask/metrics）、`prediction/`（forecaster）；`EstimatorCapabilities` 合同迁入 specs/estimator.py 消除反向依赖；轴校验 `require_matching_point_axes` 并入 tensors 合同。同轮完成：quantile 生产评估接线（`test_scores_probabilistic_df.csv`，eval_mask 与点评估同口径）、ensemble 融合 OOF 评分（`ensemble/evaluation.py`，run.py 日志输出）、models 双 shim 删除、分层门禁 v2（白名单制 13 项）。验证：581 tests OK、compileall 13 包 OK、848 passed、audit 零失败、main.py 真实冒烟（st_usmr）通过。行为零变化：848 配置 fingerprint 不变。**当日追加**：五包重命名为 `model_training/model_testing/model_forecasting/model_ensemble/model_evaluation`（wangzf 指示）；`prediction/` 收编回 `model_forecasting/forecaster.py`（方案 i：预测阶段重量在 executor 与编排，薄壳不独立成包）；重命名后 600 tests OK、848 passed、audit 零失败 |
| D14 | **【执行期发现 2026-08-30】v4 折合同与存量配置几何错位**：824 单模型配置中 618 个满足不了 E1 非重叠合同 `history_length > horizon + (max_windows-1)×stride`（load/ess 五场景 history_length=16/34 ≤ horizon=16/96），外层回测一律 RAISE `requires at least one non-overlapping training sample`；可运行的 206 个几乎全是 v4 验收实际跑过的 aidc_power_month | 归属 canonical 语义（收敛方案 §1.3 范围外），收敛轮未改动折生成逻辑（git diff 证明）；修复两选：① 存量配置 history_length 提到 ≥ horizon+1；② E1 合同加显式「仅 final forecast」出口（`max_test_windows: 0` 类语义）——待 wangzf 裁决后另立批次；D13-rewire 的 14 配置真实重跑验证随该批次执行 |

## 12. 开放风险与控制

| 风险 | 影响 | 控制措施 |
|---|---|---|
| H 与 K flatten 顺序不一致 | scaler、chain、metrics、artifact 错位 | 固定 time-major；shape 测试与 artifact 顺序双重校验 |
| independent 被误称为联合多目标 | 能力描述失真 | artifact 记录 `target_coupling=independent` |
| native multi-output 能力误判 | fit/predict shape 或权重能力不完整 | 显式能力探针；不支持即 RAISE |
| RegressorChain 顺序漂移 | 结果不可复现 | 固定输出顺序并随 artifact 保存 |
| MO 策略 chunk 退化或尾块截断 | strategy 名称失真 | 配置期强制 `1<B<H` 且 `H%B=0` |
| source registry 迁移引入泄漏 | 回测虚高 | 每个 origin 做 `available_at <= forecast_origin`；边界测试覆盖 |
| observed-past provider 缺失 | 运行期隐式冻结或中途失败 | 配置解析期要求 provider/target/known_future 三选一 |
| Global K>1 状态串线 | series/target 相互污染 | `(series_id,target,time)` 唯一键和 N=2,K=2 七策略矩阵 |
| 边际 quantile 被误当联合分布 | 风险结论错误 | samples 生成器显式 unsupported，metadata 无 dependence model |
| long result schema 破坏下游脚本 | 报告、筛选、绘图失败 | `LegacyResultReader` + 下游脚本定向升级 |
| legacy 长期残留形成双轨 | 新旧事实源再次分裂 | 生成路径与第二运行时已删除；结构测试禁止恢复 |

## 13. 历史溯源

- 2026-08-24 前：九方法 + 扁平字段实现，缺陷分析见 §2。
- 2026-08-25：中间态「九外部方法 → 五内部 executor（pointwise/direct/recursive/dirrec/blend）」收敛，修复 MSMD/MSMDR 目标时点、DirRec 训练宽度、safe-lag、auxiliary 回填、ridge-stacking 权重持久化（历史清单 A1–A5 与 DirRec 退化修复明细由 git 历史保存，均已被本次 canonical 迁移的失效标记覆盖）。
- 2026-08-27 至 29：正交架构实施（R0–R13），全量 YAML 迁移、legacy 运行路径删除，证据见 §11。
- 2026-08-29：ETT-small 演示配置（21 个 YAML）下线（演示数据保留在 `dataset/ETT-small/`），`results/` 下 aidc_ess_selfuse_load、aidc_load_15min_*、aidc_power_month、ETT-small 的旧结果及各 `tmp/` 暂存清除待重跑；`config/aidc_electricity_computility/electricity_computility/` 下 14 个 legacy Python 配置类转为 canonical YAML（`lgbm_<method>_a.yaml`，usmdr 退化块按规则映射为 mimo 并标记 `degenerate_alias`）；manifest 再生为 824 行。

当前只保留人工 review/commit gate；未经后续明确授权不 commit、不 push。
