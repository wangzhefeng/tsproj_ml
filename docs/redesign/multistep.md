# 多步预测 Canonical 架构与实施记录

> 文档状态：**当前唯一权威设计／C0–C7 全部完成并关闭（2026-08-31）**
> 当前实现以代码、活动 schema-2 YAML 和本次 fresh 验证为准；历史方案见 Git 及文末溯源，不再作为现状事实源。
> 七类运行、全量门禁、source header 合同和活动文档均有 fresh 证据；Global N2K2 fixture 已获批作为能力验收，`AGENTS.md` 当前合同已同步。

## 1. 当前边界

`tsproj_ml` 只保留 canonical 运行体系：

- 5,150 个活动模型 YAML 均为 `schema_version: 2`：5,066 个普通单模型、6 个其他场景保留的独立 Ensemble member、78 个引用式 Ensemble；按 typed spec 计为 5,072 个 `ForecastConfigSpec` + 78 个 `EnsembleConfigSpec`。另有 41 个数据工具 YAML，不计入模型数。三个 AIDC 15min 负荷场景共 4,689 份：4,617 份六组全因子单模型，另有 72 个 Ensemble 按三组 Latin-square × 四方法直接引用 baseline 成员，不维护 `ensemble_members/` 副本；3 个无有效资产配置原样保存在 `docs/archived_configs/`，不进入 loader/audit/runtime。
- `load_yaml_config()` 严格分派 `ForecastConfigSpec | EnsembleConfigSpec`。未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE。
- `main.py`、`run.py` 不接受 legacy 配置或旧产物。旧 `base_config + overrides`、九方法运行类、旧宽表、旧 pkl 和迁移兼容层均已删除。
- `docs/redesign/model_ensemble.md` 与 `docs/redesign/probabilistic.md` 仅作历史决策参考，本文和 `AGENTS.md` 才是当前事实源。

## 2. 正交预测合同

### 2.1 张量

| 类型 | Shape | 语义 |
|---|---|---|
| point | `(N,H,K)` | N 个 series、H 个未来步、K 个 target |
| marginal quantile | `(N,H,K,Q)` | 各 target/时点的边际条件分位数 |
| joint samples | `(N,S,H,K)` | 仅保留类型边界；生成器明确 unsupported |

内部二维展开固定为 time-major：`(h1,target1)...(hH,targetK)`。边际 quantile 不得表述为联合轨迹分布。

### 2.2 七种策略

`recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo` 只定义未来 H 步的训练与推进方式。Pointwise/horizon-feature 是 Direct layout，Local/Global 是 training scope，模型融合不是第八种 strategy。

MO 策略严格要求 `1 < B < H` 且 `H % B == 0`；不满足时在配置解析期 RAISE，不由运行时截断。

### 2.2.1 配置维度总表（2026-09-01 新增）

四个配置维度互相正交，各自独立取值、组合生效；理解任何一个配置先定位它在下表的位置：

| 维度 | 配置字段 | 取值 | 行为作用层 | 代码入口 |
|---|---|---|---|---|
| 策略 | `strategy.name`（+MO 的 `output_chunk_length`） | recursive / direct / mimo / recmo / dirrec / dirmo / dirrecmo | H 步输出如何在模型调用间分配、是否消费自身预测 | `forecasting_core/specs/strategy.py` 规则表；`model_training/strategies/` 执行 |
| Direct layout | `features.transformations.direct.layout` | independent_models（H 个独立模型）/ single_model_horizon（共享单模型=pointwise，必配 horizon_feature） | 同一 Direct 策略下训练样本的行×列组织 | `model_training/strategies/base.py::target_plan_for_config`；`feature_engineering/compiler.py`；`estimators/multi_target.py` |
| 训练 scope | `problem.training_scope` | local（逐序列独立建模）/ global（panel 池化共享模型，series_id 作 key 特征） | 序列间是否共享参数 | `forecasting_core/specs/problem.py` |
| 估计器耦合 | `estimator.target_adapter` | independent（标量模型组）/ regressor_chain（标量组+前序输出作输入，不支持 quantile）/ native（单实例多输出） | 语义输出块到物理估计器实例的映射 | `model_training/estimators/multi_target.py` |

维度命名属合同：不得把 layout/scope/adapter 取值登记为策略名（`tests/test_canonical_strategy_spec.py` 门禁钉住）；时间-major 压平合同唯一实现在 `forecasting_core/tensors.py::flatten_time_major/unflatten_time_major`。

### 2.3 数据角色

`DataSourceSpec.columns` 是入模投影视图；每个声明列必须显式且只声明一个角色：

`target / observed_past / known_future / static / key / ignored`

动态 source 通过 `SourceRegistry + InformationSetRequest` 执行 as-of：

- `target` 的未来真实值不可见；
- `observed_past` 只能使用 forecast origin 及以前的记录，未来轨迹必须绑定显式 provider；
- `known_future` 可以对应未来 target time，但 `available_at <= forecast_origin`；
- static 按 series key 唯一匹配；
- 非 ignored 声明列缺失、重复、越界、不可见或 schema 不匹配均 RAISE；物理文件其他列在 registry 边界丢弃，不会隐式入模。

信息可见性是运行时强制不变量，不再由公共 YAML 选择。特征编译只接受内部
`target_access=history_only` 请求；监督训练通过
`target_access=supervised_labels` 仅放开预测期 target 标签，known-future 等外生
source 仍必须满足 `available_at <= forecast_origin`。公共配置声明旧
`problem.information_mode` 一律 RAISE。

数据填补与异常清洗只允许发生在离线 `data_process/`，或未来以 config 驱动且满足 as-of 的 compiler 前置步骤中。训练窗口内清洗与评估掩码是两套独立语义。

## 3. 时间几何

### 3.1 Fixed-step

- `problem.horizon` 以 `freq` 为步数单位。
- fixed-step 使用 `validation.history_steps/train_window_steps/fold_count/stride_steps`，四者统一以监督 origin steps 保存。
- 训练标签结束必须严格早于 holdout 标签开始。
- final fit 使用与回测相同的最近训练窗口，不再隐式切为全部历史。

### 3.2 Calendar-month

- `horizon_mode=calendar_month` 时使用 `train_window_days/fold_count/stride_months`；训练窗口按原始日数计，折间距按自然月计。
- 每个历史折和最终目标月按真实月份动态解析 28/29/30/31 个日步。
- 生产折由 `model_testing.validation.calendar_month_folds()` 生成。
- final forecast 必须覆盖完整目标自然月。

### 3.3 月频

`1ME/1MS` 使用月步 offset；seasonal-naive 按月步定位，禁止近似转换为固定天数 Timedelta。

## 4. 当前模块架构

```text
main.py / run.py / config/config_loader.py
        ├── model_forecasting/      单模型生命周期
        └── model_ensemble/         引用式融合生命周期
                    │
                    ▼
probabilistic/                      quantile 训练与独立后处理能力
                    │
                    ▼
data_loading/ → feature_engineering/ → model_training/
                           ├── model_testing/
                           └── model_evaluation/
                    │
                    ▼
forecasting_core/                   稳定 specs/tensors/artifacts
models/ decomposition/ data_process/
                    │
                    ▼
utils/
```

### 4.1 `model_forecasting/` 职责

| 模块 | 当前职责 |
|---|---|
| `design.py` | information-set materialization、特征设计、监督样本和 fixed-step backtest geometry |
| `fit_service.py` | fold/final transform、point/quantile fit 与 predict 服务 |
| `backtest_runtime.py` | calendar-month backtest 生命周期 |
| `persistence.py` | schema-2 bundle 构造与持久化 |
| `forecaster.py` | point 与边际 quantile 推理 |
| `transforms.py` | feature/target transform 状态及严格逆变换 |
| `results.py` | canonical long result 读写 |
| `runtime.py` | `CanonicalBaseModelRunner` 与顶层单模型编排；不再内嵌上述实现 |

### 4.2 包依赖约束

- `forecasting_core/` 不 import 任何流水线、能力或编排包。
- `model_training/` 不 import `probabilistic/`。
- `probabilistic/` 不 import `model_forecasting/`。
- `model_ensemble/` 通过 `EnsembleRuntimeServices` 注入 runner factory 和 bundle persistence，不 import `model_forecasting.runtime`。
- 项目包内禁止函数级延迟 import 绕环，禁止跨包导入下划线私有实现。
- `tests/test_package_layering.py` 使用 AST 扫描顶层和函数内 import，并验证包依赖图无环。

## 5. 训练、变换与模型

`CanonicalTrainer/CanonicalForecaster` 支持 Local/Global、K1/K2、七策略、point 和 independent quantile。

目标变换固定顺序：

```text
calendar normalization → decomposition → target scaling
```

推理恢复严格逆序，状态按 `(series_id,target)` 隔离并随 bundle 保存。Feature scaling 与 feature selection 均在每个训练折内拟合，不能读取 holdout。

模型能力由 `model_training/estimators/capabilities.py` 显式探测；不支持的 target adapter、quantile 或并行组合直接 RAISE，不静默降级。

## 6. 引用式 Ensemble

顶层 `model_ensemble/` 是唯一融合实现，四种方法为：

- `averaging`
- `weighted`
- `linear_blending`
- `stacking`

成员必须引用现役单模型 YAML，并共享 problem/data/probabilistic/origin 合同。OOF cache 按有序成员语义 fingerprint、OOF geometry 与 `(member, source, path_role, file_sha256)` 内容寻址；fusion method 和 `config_ref` 路径不进入 OOF key，同一成员集跨方法共享。并发同 key miss 由线程锁与跨进程文件锁保证 single-flight，损坏或输入内容变化必须 RAISE/失效；成员 runner 与单模型共用 RawDesign cache。

Quantile linear blending 按 target 最小化 simplex pooled pinball，一组 target 权重跨 Q 共享。最终 Ensemble bundle 内嵌全部成员 bundle 和 method artifact；部署预测不读取成员 YAML 或 OOF cache。

### 6.1 OOF 标签隔离 gap 合同

`ensemble.oof.gap_steps` 是训练标签与 holdout 标签之间的隔离步数，规则为
`training_label_end < validation_label_start - gap_steps * freq_offset`。
gap 不平移预测 origin/标签时间，也不改变 outer cutoff 的原有筛选含义；月频使用
calendar offset，不转换为固定 Timedelta。单模型切折、holdout 审计和 OOF 共用
`model_testing.validation.is_label_safe`；两类选折策略仍各自保留，不能声称 OOF
直接调用了 `rolling_origin_folds`。

正 gap 使用内部 `label_embargo_v1` 语义标识参与 Ensemble fingerprint 和 OOF cache key，
避免读取旧版“正 gap 放宽标签边界”的缓存；该标识不是 YAML 开关。
OOF 逐折审计保存 `gap_steps/gap_semantics/training_label_end_max` 并随缓存持久化。
零 gap 的配置身份、OOF key 和选折保持不变，存量结果不删除、不自动重跑。

实施完成证据：2026-09-06 定向回归 **123 项、16.157 秒、OK**；覆盖小时/15min/月末
边界、等号排除、outer cutoff、历史不足、旧正 gap 缓存 miss 且文件保留、零 gap
golden，以及 point/quantile 小样本 OOF→final fit→预测表→缓存重用→bundle 重载推理。
读取现役全部 **78** 份 Ensemble YAML：全部 gap=0，修复前后配置 fingerprint 零变化，
无需 YAML 迁移；这不是存量生产结果重跑证明。未执行全量测试或生产配置训练。

验证命令：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -c 'import unittest; names=("test_oof_gap_*.py","test_ensemble_oof.py","test_ensemble_specs.py","test_ensemble_loader.py","test_ensemble_runtime.py","test_ensemble_parity.py","test_ensemble_persistence.py","test_canonical_base_model_runner.py","test_package_layering.py","test_compiled_feature_cache.py"); suite=unittest.TestSuite(unittest.defaultTestLoader.discover("tests",pattern=name) for name in names); result=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not result.wasSuccessful())'
```

日志与逐配置核对清单分别为 `.hermes/plans/oof-gap-regression.log`、
`.hermes/plans/oof-gap-config-audit.json`（本地实施证据，长期合同以本节和测试为准）。

### 6.2 原生模型参数校验（2026-09-06）

模型工厂拒绝未知参数，不把合法原生别名当作跨模型残留清扫：

- LightGBM 使用已安装 native alias 表校验名称；原生查询不可用即报错，不静默跳过。
- CatBoost 显式参数先按原生同义组归一化，再合并 wrapper 默认值；保留合法 `num_leaves`，不以默认 seed 覆盖 `random_state`，冲突的显式迭代别名报错。
- XGBoost 在 `models/xgb_validation.py` 的独立子进程中配置无数据 Booster，不调用训练；按真实参数、维度、特征名与版本做进程内有界 single-flight 缓存，未知/未使用参数报错。预检强制诊断开关，不修改父模型配置，也不在父线程捕获全局 warnings。RNG 使用副本，避免参数读取额外抽取 seed。wrapper 保存预检与实际拟合后的 native config，重载预测不依赖子进程；统一运行证据落盘仍属后续任务。

验收：**85 项定向测试，12.486 秒，OK**，含 point/quantile 训练、CatBoost 原生数值 parity、XGBoost RNG parity、并发校验复用、预检故障显式失败、wrapper 序列化重载预测、分层与编译缓存门禁。

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -c 'import unittest; names=("test_native_parameter_validation.py","test_xgb_parameter_preflight.py","test_package_layering.py","test_probabilistic_objectives.py","test_quantile_training_optimization.py","test_model_thread_runtime.py","test_runtime_array_fastpaths.py","test_model_parameter_validation.py","test_compiled_feature_cache.py"); groups=[unittest.defaultTestLoader.discover("tests",pattern=name) for name in names]; assert all(group.countTestCases() for group in groups); result=unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(groups)); raise SystemExit(not result.wasSuccessful())'
```

配置审计严格解析 5,072 份单模型和 78 份 Ensemble；其中 1,910 份原生模型归并为 7 组参数，构造/参数名预检全部通过。现役配置未发现受上述 CatBoost `num_leaves` 或非 42 `random_state` 覆盖修复影响的声明。审计没有启动训练，XGBoost 全集参数名检查使用两列占位维度，不能替代真实数据拟合预检。命令为 `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python .hermes/plans/audit-native-parameters.py`；本地证据为 `.hermes/plans/native-parameter-audit.json` 与 `.hermes/plans/native-preflight-regression.log`。

范围与代价：没有生产重跑、YAML 迁移或旧结果删除；未测量完整运行时中子进程预检的性能成本，不沿用旧性能基准宣称零开销。日内切折、目标变换窗口、统一运行元数据及自然月生命周期仍是独立待办，本项测试不构成这些事项的完成证据。

### 6.3 可靠性收口实现与受限验证

本节接续 §6.2 的历史待办。用户要求不运行配置完整生命周期和模型 fit/predict 测试；工程接线与运行验收分开记录。

| 项目 | 当前实现 | 验证边界 |
|---|---|---|
| intraday 调度 | 单模型/OOF 使用正式原点 stride 网格；保持标签 embargo/outer cutoff；批验收检查真实时间网格 | 纯时间几何通过，真实回测未执行 |
| 目标拟合窗口 | scaler 按唯一训练标签时间拟合；分解默认同窗，可声明 `features.transformations.target.decomposition.fit_history_steps`；确定性日历归一化没有虚构拟合状态 | 窗口选择通过，数值恢复/部署 parity 未执行 |
| 模型描述 | `models/catalog.py` 统一别名、quantile 类型、线程及能力字段，原 wrapper 路径保留 | 描述表/静态接线通过，原生模型数值 parity 未重跑 |
| 生命周期 | 自然月回测/CQR 收集先于 final fit；final bundle 与预测不再在 run 返回后重写；单模型完成状态由 runtime 写、批验收读 | 静态顺序/状态合同通过，真实训练失败恢复未执行 |
| 运行证据 | fixed-step/calendar-month 逐折和 Ensemble OOF/final 成员均接入公开 `execution_evidence`；final 和成员汇总写 readable JSON | 元数据 I/O/JSON 快照/静态接线通过，真实模型参数覆盖未执行验收 |

适用日内调度和 target decomposition/scaling 的配置语义分别包含 `formal_origin_grid_v1`、`unique_training_labels_v1` 内部版本。旧结果不删除、不自动重跑。上一实施轮严格解析全部活动配置，并对季节分解声明做规则网格窗口审计，未发现需增加独立上下文的配置；该审计不等于真实资产/分解验证，本次未重跑全仓配置检查。

运行证据约定：
- 单模型逐折证据在 holdout metadata；final 证据为 `run_evidence`。记录 config/实现摘要、依赖版本、目标变换窗口和已存在模型的参数；实现摘要沿用 checkpoint 实现文件集合，不是整个 Git 仓库快照。
- Ensemble 的 `run_evidence.member_oof` 与 `member_final` 分开；OOF 证据单独保存在 `OOFPredictionArtifact.execution_evidence` 和缓存 JSON，不进入 OOF 预测或配置语义指纹。缓存命中保留历史证据，并标记 cache_hit；旧缓存/自定义 runner 缺证据明确 unavailable，不补造也不为此重跑。
- NumPy 参数转为普通 JSON 值；合法 NaN/Infinity 参数哨兵显式标成 `nonfinite_float`，其他无法序列化对象标成 `unserialized_type`，不宣称这些字段可直接重建模型。证据采集不调用 fit/predict。
- 评估公式不变：point aggregate 是逐 target 指标加权，aggregate_horizon 和 marginal aggregate 是有效点池化，metadata 显式区分。
- `run_state.json` 的 completed 不是全目录原子事务；外部直接 pickle 读取不受它约束。本轮不扩展模型加载入口。

受限验证命令：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -B .hermes/plans/verify-architecture-no-models.py
git diff --check
```

本轮最终复验：**44 项通过，0 failures/errors/skips，模型调用拦截记录为空；52 个修改/新增 Python 文件 AST 解析通过，git diff --check 通过，命令退出码 0**。日志及 JSON 位于 `.hermes/plans/architecture-no-model-verification.*`。只运行纯逻辑/静态测试及临时元数据 I/O，真实训练、预测、CQR 数值、bundle 重载和失败恢复仍未验证，不能用这些检查替代。新增负向测试先复现原生 NaN 参数无法写严格 JSON，修复后通过；拦截器对 sklearn.base._fit_context 的导入期装饰器构造误报经源码核实修正，未放行实际 fit/predict。

提交收口另以 `tests/run_suite.py all` 执行完整仓库测试：发现 890 项（fast 283 / integration 482 / audit 125），**890 项全部通过，耗时 1265.407 秒**。该全集包含现有 runtime smoke、配置与资产审计，但仍不等于逐份活动配置的完整生产 fit/backtest/forecast 重跑；上表对生产级数值和资源验收的限制继续有效。

文档同步：首次 AGENTS.md 审批超时后未绕道写入；用户明确要求重启审批后，通过受保护 patch 入口成功追加“可靠性收口合同”，文档阻塞已解除。原补丁清单保留在 `.hermes/plans/agent-rules-pending.md` 供溯源；CLAUDE.md 仍为原有指针。此文档更新不改变上述运行验收边界。

## 7. 概率边界

生产能力覆盖 point、边际 quantile 与可选 CQR 校准区间（2026-09-01 激活）：

- quantile 模式写逐 level pinball、bias、central interval coverage/width/Winkler/coverage gap，并按 horizon 步拆分诊断行（2026-09-02）；
- crossing 后处理消费 `probabilistic.crossing.method` 配置（none/rearrangement/median_preserving_isotonic，缺省 median_preserving_isotonic=历史行为）；部署期从 bundle spec 读取（2026-09-01 裂缝修复：此前配置被静默忽略、runtime 无条件修复）；
- CQR：`probabilistic.calibration`（method=cqr）声明即启用。回测逐折 apply-before-collect，只消费 origin 严格更早且标签已可得的历史折（as-of 双门槛 min_windows/min_scores，pooled 分组跨 series/target）；产出 `predict_pi<coverage>_lower/upper` 列进 `cv_plot_df.csv` 与 `prediction.csv`；final 修正量冻结进 bundle `calibration_state`（`ForecastModelBundle.calibration_state`），部署期应用、不重校准，且经 `distribution_to_long` 消费 bundle metadata `prediction_intervals` 落成部署链 `prediction.csv` 的 pi 列（2026-09-02 接线，此前为死写；同批修复 deployment `_predict_quantiles` 的 `bundle_crossing_method` NameError、checker probabilistic 键白名单与 geometry manifest fingerprint 再生成）。不足门槛时诚实降级：不产出 pi 列，`result_metadata.json` 记 status/reason。fixed-step（`runtime.py`）与 calendar-month（`backtest_runtime.py`）共用 `probabilistic/calibration.py::ConformalCalibrationTracker`；
- legacy `crossing_method`/`conformal` 键已于 2026-09-01 从全部 YAML 清扫（379 个文件：`crossing_method: isotonic` → `crossing:` 块；`conformal: {method: none}` 删除；146 个 `method: cqr` 迁入 `intervals:` + `calibration:`），spec 层对 legacy 键一律 RAISE。注意 fingerprint 随之变化，存量 results 沿用旧 identity 前缀；
- quantile 训练优化（2026-09-01）：逐 level 训练默认线程并行（数值与串行一致）；xgboost≥2.0 自动走原生多分位共享 booster（训练成本 ≈1× 而非 Q×，与并行互斥；现役 YAML 零 xgb quantile 配置）；
- 仍不支持 joint sample generator、联合路径概率或 ensemble-of-ensemble。

### 7.1 统一运行资源规划（2026-09-04）

- `RuntimeWorkload` 描述策略布局、H/B/K/Q/N、fold/member、model group、logical/scalar/physical fit、可并行 output task、训练行数、特征宽度和设计体积；`RuntimeResourceBudget` 记录物理/逻辑核、总线程、可用内存和父级并发；`RuntimeExecutionPlan` 固化唯一外层并行轴及 config/member/window/quantile/output/model 预算。
- fixed-step、calendar-month、point、quantile 和 Ensemble 统一从 `resource_planner` 获取 plan。除 typed parser 与 planner 外，生产代码不得读取原始 `validation.performance`；显式多外层轴、线程乘积超预算、设计体积超内存或 estimator 线程冲突均直接 RAISE，不静默压制。
- LightGBM/XGBoost/RandomForest/CatBoost 的线程参数由 plan 唯一写入；HistGradientBoosting、NumPy/SciPy 的 BLAS/OpenMP 限额由 `threadpoolctl` 在任何线程池创建前应用。XGBoost shared multi-quantile 当前只允许串行共享位置训练，不宣称 output/quantile 并行。
- `validation.performance` 是非语义运行控制，不进入 canonical fingerprint/result identity。`resolved_config.json` 与 `result_metadata.json` 必须记录 workload、budget、execution plan、cache 状态、阶段 wall、profile 来源和 fallback/rejection 原因。
- 死字段 `step_logging`、`forecast_log_interval` 保持删除。`ensemble_parallel_workers` 已于 OPT-015 作为 typed 非语义字段重新接线：它只对 Ensemble 配置有效，单模型声明会在 planner RAISE；planner 选择 member 轴时把成员内部 window/quantile/output 轴压为 1，OOF 仅同 fold 成员并行、fold 仍按时间顺序，final refit 并行后按 member order 组装 bundle。serial Ensemble 保留成员各自最大 native thread；OOF cache miss 与 final fit 均在同一进程级 thread limit 内。point/quantile 三轮真实对照均显著变慢，因此活动 YAML 零声明，能力只供未来新拓扑重新标定，不设自动默认。

### 7.2 Catalog 批调度与 compiler 可观测性（2026-09-04）

- `batch_run.py` 是独立的多 physical-YAML 入口，不替代 `run.py`。L2 `batch_runtime` 建立 `RawDesignGroup → FoldTransformBranch → ModelFitTask → PerConfigPersist`：每组只加载/物化一份 raw X/Y，内存共享前由 runner 重新计算并校验 raw fingerprint；相同训练 indices 与 transformation 语义的分支由内存 single-flight cache 复用；每配置的 fingerprint、identity、bundle 和结果目录保持独立。
- 父级 config worker 持有统一 `RuntimeResourceBudget`，`for_children()` 将已有 `parent_concurrency` 与本次 worker 数相乘，线程不足直接拒绝，不用覆盖父并发或强行取至少一线程来放大预算。进程级 BLAS/OpenMP 限额在 config pool 前设置，绘图固定使用非交互 `Agg` backend。task state 原子落盘并由同 batch ID 的线程锁 + OS 文件锁串行化，batch ID 与输入路径顺序无关；resume 不覆盖已验证 completed task 的历史 provenance。
- 全批 preflight 顺序准备每组 raw design，检查显式计划、组合子预算、线程池同质性和估算内存准入，全部通过才开始任何组的模型执行；执行阶段逐组重读 durable raw cache，不在 preflight 中保留全批数组。transform cache 有 bytes/entry 上限，动态 raw cache 另有 bytes 上限及最多四个 horizon 条目，超大值不驻留，逐组释放所有权。RSS 每 0.02 秒采样并记录 batch 累积峰值；数组工作集估算与采样均不是 native estimator 的操作系统级硬内存隔离，不保证捕获采样间瞬时峰值。
- completed 必须通过 `batch_artifacts`：非空必需 manifest、distinct 文件路径、内容 SHA256、配置 fingerprint/result identity、schema-2 bundle、预测/回测 canonical 键与维度、评分表及 quantile 列网格。自然月按每折 label_start/label_end 推导真实步数，不能把 30 天折按最终 31 天 horizon 验收。预检、执行或产物验证失败均记录失败阶段；可确定的模型错误附 config/fold/model 坐标。旧空 manifest 或仅有文件存在性证据不能直接跳过。
- batch 自动向 runner 传入持久化 checkpoint 根目录；单运行 API 通过可选 `checkpoint_root` 启用。稳定合同 `forecasting_core.checkpoints.FitCheckpoint` 与 L2 `model_forecasting.checkpoints` 分离。完成的 scalar、multi-RHS、native、chain 节点及共享 quantile booster 按配置、source/raw、实现与依赖版本、实际 X/y/weights、训练索引、schema 和 group/level 身份寻址，原子 envelope 校验 schema/key/descriptor/SHA256 后读取；损坏 RAISE，输入变化新键 miss，未完成 estimator.fit 不保存。恢复重放预测与 CQR apply-before-collect 顺序，不重复拟合已完成单元。自然月 factory 显式接受并透传 checkpoint_root；部署 bundle 不依赖该目录。`--no-resume` 创建新的 checkpoint namespace，不删除旧数据。
- H=16、5-fold 的 XGBoost+CatBoost baseline pointwise 真实组中，两配置均固定 `model_threads=1`，父 `config_workers=2` 总模型线程乘积为 2；门禁修复后三轮复验，batch 相对逐 YAML median wall `40.470→33.999s`（改善 15.990%），p95 `41.867→35.087s`，swap used 增量均为 0；CPU median `40.032→42.580s`、最大 RSS `469,909,504→489,439,232 bytes`。每模型三类 CSV hash 唯一、identity 一致、bundle reload 最大差 `1.82e-12`。该数字只证明相同 RawDesign/transform 组的 catalog throughput，不替代单配置 latency profile，也不宣称正式 31-fold 结果已重跑。
- LGBM+Ridge 的 `/tmp` 5-fold 门禁夹具 output workers 从 8 改为 4 后三轮全部完成，median 仅改善 0.453%，不足采用门槛；它是正确性对照，不与上面的性能采用组混用。最终统一定向回归 222 项、123.089 秒通过（TASK27 46/46），命令与证据见 `docs/TODO_OPTIM.md` OPT-016 实施记录和 `/tmp/tsproj-opt016-benchmark/metrics/closeout-verification.json`。
- compiler 的 batch eligibility 统一为结构化单一事实源，runtime metadata 记录 batch/row reason、origin/call 规模、编译阶段与 cache load/compile/write wall。5,072 个活动单模型 inventory 中 4,265 个 batch eligible，807 个 fallback 全部是 `provider_dependent_lag`；活动集没有 percent-change/time-since/EWM fallback。真实 Recursive row 热点必须维持逐步 provider 语义，OPT-018 因此不新增向量化算法。
- 非 LightGBM 九类模型各完成三轮 profile。仅 short Route A baseline CatBoost Direct pointwise 的 `model_thread_count=8` 达到门槛并写入生成器/YAML（median 改善 19.56%）；其余保留默认。RandomForest 8 线程虽快 58.73%，但出现最大 `7.28e-12` 的跨轮并行归约差异，因 deterministic gate 不采用。profile 不跨 H/K/Q/rows/features/scope 外推。

2026-09-05 签名收口：上述唯一 CatBoost YAML 与生成器使用 `profile_ref: opt017-catboost-short-a-pointwise-v1`。`performance_profiles` 校验真实 raw workload、特征内容与顺序、数据 SHA256、CatBoost 版本/默认参数、训练窗口和预算；runtime/batch 必须传实际 base_dir 与 feature_schema，后续调度不能改写已标定计划后仍保留原 claim。resolved resources 区分 adopted_benchmark_profile、explicit_performance_override、automatic_default；签名/provenance 和其余 performance 一样不进入 semantic fingerprint。原实验为 5-fold，物理配置仍为 31-fold，metadata 明确 `formal_benchmark_verified=false`，不能将采用配置表述为已经测得正式 31-fold 收益。当前批次最终验证与正式产物验收进度见 `TODO_OPTIM.md`，上面的旧吞吐数字不作为新增 checkpoint 开销的复测结果。

2026-09-06 intraday 回测改为正式 forecast origin 的 stride 网格后，该历史 profile 的训练折语义已改变。活动 YAML 与确定性生成器撤下 `profile_ref`，旧签名保留并必须以 `semantic_sha256_at_5fold` mismatch fail-closed；不能只更新 SHA 冒充重测。未来如需恢复，必须按新时间网格重新执行 benchmark 与数值/资源验收。

2026-09-05 本批验收收口：新增预算/产物/checkpoint/profile及自然月门禁的统一定向回归为267项、225.300秒、OK（精确命令及日志见`docs/TODO_OPTIM.md`本批新增证据）；5150配置checker通过、4689矩阵生成计划零漂移。原正式24份清单全部通过31fold/final/forecast/摘要/身份/几何和历史四CSV及reload exact复验，其中22份新运行、2份复用。其后linear/STL96/MSTL96-672以独立5fold完成三拓扑各三轮，共27次运行；verifier独立回读108份CSV及27个bundle并复算统计一致。三个分解代表全部保持默认1x8：linear/STL最佳median改善仅9.867%/5.376%，MSTL候选更慢；未新增分解性能配置或扩大收益结论。记录到系统swap-out活动，不能宣称全局无swap。统计与验收证据在`results/_optimization_closeout/decomposition/`，完整实验产物在原`/tmp`隔离目录。最后收口遵照用户要求不跑全量测试、不重跑上述267项回归；完成证据验收、生成器13项定向测试、两次只读幂等和文档检查。OPT-001至OPT-019均已完成；历史测量、no-op与证据覆盖边界继续保留。

## 8. 结果与产物合同

```text
results/<scenario>/<result_identity>/
├── pretrained_models/
│   ├── model.pkl
│   └── resolved_model.json
├── results_test/
│   ├── cv_plot_df.csv
│   ├── test_scores_df.csv                  # 每窗汇总：scope ∈ {target, aggregate}
│   ├── test_scores_horizon_df.csv          # per-horizon 明细：scope ∈ {horizon, aggregate_horizon}
│   ├── test_scores_probabilistic_df.csv    # quantile only，每窗汇总
│   ├── test_scores_probabilistic_horizon_df.csv  # quantile only，per-horizon 明细
│   ├── test_prediction.png                 # 单 target 拼接总图（多 target 在 target_plots/）
│   ├── target_plots/                       # 多 target 拼接总图（per target 一张）
│   ├── windows_results/                    # per-window 图（window_<w>.png，多 target 子图）
│   └── result_metadata.json
└── results_forecast/
    ├── prediction.csv
    ├── forecast_prediction.png            # 预测图（单 target；多 target 在 prediction_plots/）
    └── resolved_config.json
```

- `prediction.csv` 唯一键：`(series_id,time,target)`。
- `cv_plot_df.csv` 唯一键：`(series_id,time,target,window)`。
- 可视化产物（2026-09-02，绘图消费未掩码原始值，掩码只用于指标）：回测总图按时间排序后整条拼接（现役 stride==horizon 契约下窗口首尾相接；同 series 时间戳重复即 stride<horizon 重叠配置，拼接 RAISE 并指向 windows_results/ 单窗图）；`windows_results/window_<w>.png` 每窗一文件、多 target 纵向子图；正式预测写 `forecast_prediction.png`（多 target → `prediction_plots/`），预测段前绘制 as-of origin 末段 `5×horizon` 步历史真实值参照（history 不可用时诚实降级为仅预测段并 warning），Preds 首点回接历史末点 + forecast origin 竖线，quantile 模式附 PI 区间带。线型：Trues 实线 / Preds 点划线（沿用旧版 `models/ModelTesting.py` 口径）。
- `test_scores_df.csv`（2026-09-03 方案 B 拆分）：**每窗汇总文件**，`scope ∈ {target, aggregate}`——`target` 一窗一 target 一行，`aggregate` 为 target 加权汇总；指标列含 `Bias`（= mean(pred−actual)，正=高估，含 `Naive Bias` 对照）。
- `test_scores_horizon_df.csv`（2026-09-02 新增明细文件）：per-horizon 诊断，`scope ∈ {horizon, aggregate_horizon}`，`horizon` 列 1-based，`aggregate_horizon` 跨 target 按有效点池化（proper score 语义）；列 schema 与汇总文件一致。
- `test_scores_probabilistic_df.csv`（quantile only，每窗汇总）：tidy long，metric ∈ {mae, bias, pinball, interval_coverage, interval_width, interval_winkler, coverage_gap, calibration_error}。
- `test_scores_probabilistic_horizon_df.csv`（quantile only，per-horizon 明细）：同 metric 集，`horizon` 列非空。
- 单模型和 Ensemble 均保存 schema-2 `ForecastModelBundle`。
- fingerprint 只取语义 payload；日志、并行度和输出目录不进入 fingerprint。
- resolved config/model 必须足以复算 training policy、轴顺序、source/feature lineage 和产物身份。正式预测的 `runtime.visibility_proof` 保留逐 lookup 明细；回测的 `runtime.holdout_visibility_proof` 使用 summary schema 1，按 `(forecast_origin, feature_name, source_name, role, provider)` 聚合 lookup 数及 horizon/target/source/available-at 范围，避免逐折逐 horizon JSON 膨胀。

## 9. 架构整改实施进度

| Slice | 状态 | 当前证据 |
|---|---|---|
| C0/C1 | completed | strict typed schema、生产 parser/constructor 门禁、5,150 个活动配置；3 个错配配置已批准归档 |
| C2 | completed | fixed-step/calendar-month typed geometry；5,150 YAML + 迁移 manifest；final fit 同训练窗；4,866 个子日单模型 sample-count 一致 |
| C3 | completed | proper quantile blending、自包含 bundle、bundle-only point/quantile 部署、优化审计与 canonical long result 已落地；用户批准的 H16/H31 两个代表均 exit 0 且完整产物核验通过 |
| C4 | completed | 稳定概率合同唯一位于 `forecasting_core/`；Protocol 注入、包 DAG、runtime 窄模块和重复合同清理完成 |
| C5 | completed | typed 5,150-row catalog、CLI/checker 单模型边界、legacy/test-only/零消费者链和 requirements 门禁已收口 |
| C6 | completed | README、权威设计、§17 与 AGENTS typed geometry 已同步；活动 Markdown 断链 0 |
| C7 | completed | 前轮 fresh 615 tests 与 H16/H31 代表已通过；V4 配置扩展另完成 5,150 typed parse、资产/Ensemble/manifest 审计、23 个静态定向 tests、compileall/diff，按用户要求未重跑模型 |

C7 的命令、计数和七类产物核验结果只在 `docs/redesign/architecture.md` 的最终执行记录维护；其他文档不复制测试数。

## 10. 历史决策与偏差

- 18 个旧 MDR 配置因 `H % B != 0` 于 2026-08-29 获批映射为 `recursive`：14 个 `aidc_power_month`、4 个 `aidc_load_month`。`validation.legacy_origin` 已删除，本条是批准记录。
- 旧 Direct/Recursive Blend 已迁入 Ensemble，不再作为 strategy。
- decomposition `forecasters.py/composers.py` 曾被误判为死链；包内消费者复核证明它们属于 `DecompositionPipeline` 活路径，因此保留。
- 2026-09-04 修正 AIDC 15min STL/MSTL 的趋势外推合同：972 份活动配置原误写 `trend_forecast: seasonal_naive`，而季节分量本就由内部 phase-template 独立外推；经裁决统一迁为 `polynomial`。公共值域严格为 `polynomial|damped`，非法值在配置解析/审计阶段 RAISE，不再延迟到首个 fold。迁移改变相关 canonical fingerprint/result identity，但不改变独立的 RawDesign cache key。
- 缺失日期类型未来资产时不伪造数据；受影响配置移除该 source。
- legacy runtime 曾在 `FeatureEngineering.extend_weather_feature()` 内由 `rt_tt2`/`rt_dt` 现场计算 `cal_rh`，随后线性插值、双向填充；canonical 删除隐式数据改写后，2026-08-31 经批准将同一公式迁到离线入口 `config/aidc_electricity_computility/derive_cal_rh.py`，为 23 个天气 CSV/27,180 行追加派生列，原有单元格 SHA 全部保持不变。
- 3 个 `add_training_inference_pod` 配置声明 `*_all_jobs_*` 列，但物理资产使用另一列族；经批准按字节原样迁至 `docs/archived_configs/`，活动模型数由 848 调整为 845。
- 活动 YAML 中没有 Global N2K2；2026-08-31 经批准接受完整生产 runtime fixture 作为该能力验收，不把它表述为现役生产配置。
- 配置语义改变后旧结果失效；旧 pkl/旧宽表在本仓库不可读。

## 11. 验证入口

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_runtime_assets.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_aidc_load_15min_designs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_ensemble_configs.py

env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m compileall -q \
  forecasting_core data_loading feature_engineering model_training model_testing \
  model_evaluation model_forecasting probabilistic model_ensemble models \
  decomposition data_process utils config scripts tests main.py run.py

git diff --check
```

前轮 C0–C7 的 fresh 命令、精确计数、七类产物和阻断项只在 `docs/redesign/architecture.md` §17 维护；本次 V4 配置矩阵扩展的静态证据见下节，其他文档中的旧测试数只代表历史环境。

## 12. V4 AIDC 15min 全因子配置矩阵（2026-09-03）

- 三场景共 4,617 个单模型：每场景 A/B 两路各 729，另有 K=2 joint 81；
- 独立保留 72 个 Latin-square Ensemble：每场景 A/B 两路各 12；
- AIDC 合计 4,689，全仓合计 5,150 = 5,072 `ForecastConfigSpec` + 78
  `EnsembleConfigSpec`；
- short 两种 pointwise 仅使用 `[16,96,192,672]` safe target lags；rolling 原点为
  `2026-07-31T14:00:00`；state 组使用资产中 10 个纯本路派生列，排除
  `state_route_diff_pct`；
- `scripts/generate_load_15min_matrix.py` 是确定性生成/验证入口，最终幂等结果为
  `expected=4689 create=0 rewrite=0 delete=0 unchanged=4689`；
- checker：`checked=5150 passed=5150 hard_failures=0 warnings=0`；资产审计缺失路径、
  声明列、成员引用均为 0；Ensemble 审计 228 refs、`failures=[]`；manifest 5,150
  entries / 4,866 subday single；
- 23 个配置相关静态定向测试、九模型默认构造、文件名/字段全量审计、compileall 和
  diff check 均通过。

用户明确要求不运行模型测试，因此本轮未执行 `audit_aidc_load_15min_designs.py`、任何
fit/backtest/forecast、融合 OOF 或 bundle smoke。配置结构与资产合同已验证；4,617 个机械
组合的真实运行能力和预测效果均未验证。

## 13. Recursive 真实运行效率收口（2026-09-03）

以 `config/aidc_load_15min_daily/route_A/baseline/lgbm_recursive.yaml` 的完整 6,336 origins、31 folds、H=96、35 features 为真实验收对象，不缩窗口、不改模型目标：

| 运行 | compiled cache | wall | 峰值 RSS | 关键结果 |
|---|---|---:|---:|---|
| 优化前冷运行 | miss | 2,313.33s | 3.77GiB | compile 1,909.35s |
| 优化后 2 windows × 4 model threads 冷运行 | miss | 325.55s | 1.298GiB | compile 214.05s，端到端 7.106× |
| 优化前热运行 | hit | 178.46s | 1.42GiB | 历史基线 |
| 优化后串行热运行 | hit | 155.42s | 0.949GiB | identity target transform 已启用 |
| 优化后现役 YAML 热运行 | hit | 108.81s | 1.126GiB | 相对优化前缩短 39.03%，相对当前串行缩短 29.99% |

实施项与硬证据：

- safe-lag `recursive/dirrec/dirrecmo` 不再因 `consumes_previous` 被 blanket 排除，仍由 source-time 检查决定 batch/fallback；目标配置冷编译提速 8.920×；
- 监督训练不再累计逐 call forecast audit，同一 `training_row()` 的 information set 仅 materialize 一次；
- calendar/decomposition/scaling 均为 `none` 时，target transform 只记录 fitted identity，训练数组、point/quantile tensor 和 recursive restore 走 no-copy 路径；32 次 transform 累计由 31.757s 降至 0.129s（99.60%）；
- `validation.performance` 同时从 canonical fingerprint 与 compiled-design cache geometry 排除；串行/并行共享同一 cache key，identity 仍为 `recursive-lightgbm-local-k1-bdc414c1ca20`；
- `holdout_visibility_proof` 将 105,245 个 lookup 聚合为 1,085 组，`resolved_config.json` 由 40,849,747 bytes 降至 2,403,890 bytes（94.12%）；
- 优化前后 `cv_plot_df.csv` SHA-256 均为 `47f3600fc57918c05faf232d0ce5c1bdf71e1d2ac4a1b23a8a67ab615d6276d1`，`prediction.csv` 均为 `9058902d5acb258c1db973bafe5e4cd8d596b586253f1c93cd5443eb27c90861`；拆分后的 62 行汇总分数与 5,952 行 horizon 分数逐值 exact。

2×4 预算只落到该已实测 YAML 及其确定性生成源，不外推到其他模型/策略。剩余最大冷启动瓶颈是 expanding mean/std 为保持 pandas 逐值 exact 而执行的 6,336 次 origin 统计；进一步并行或前缀算法必须先证明浮点 exact，不能用 tolerance 偷换合同。

定向验收命令覆盖 compiler batch/equivalence/visibility、compiled cache、target transform、fingerprint、window/model thread、配置矩阵、geometry manifest 与 canonical runtime smoke，共 101 tests / 94.453s / OK；另有 package layering 17 tests / OK、`py_compile` / OK、`git diff --check` / OK。配置门禁为 4,689 matrix 零漂移及 `checked=5150 passed=5150 hard_failures=0 warnings=0`。Pyright 对 15 个相关文件报告 87 个存量 pandas/Optional 类型诊断；将 JSON diagnostics 与 `git diff --unified=0` 新增行范围机械求交后，本 changeset 新增行诊断为 0。按用户要求未运行全量测试。

## 14. 公共配置固定值与死参数收口（2026-09-03）

- 公共 YAML 已删除 `problem.information_mode`、`output.setting_suffix`、
  `probabilistic.recursive_propagation`、`probabilistic.schema_version`，重新声明按未知字段
  RAISE；5,150 份活动配置对应引用均为 0，历史归档配置不迁移。
- 预测输入固定使用内部 `target_access=history_only`；监督训练仅通过
  `target_access=supervised_labels` 放开 `forecast_times` 上的 target 标签。known-future
  和历史 target revision 继续执行 `available_at <= forecast_origin`，故障注入测试钉住
  后者不会读取预测后修订值。
- quantile 递归传播在部署态内部固定为 `median_path`；概率产物内部
  `ProbabilisticSpec.schema_version=1` 继续保存并严格校验，但二者均不能由 YAML 覆盖。
- `output.setting_suffix` 删除不改变 fingerprint 或输出路由；其余三个语义 payload
  字段删除会产生新 canonical fingerprint。迁移 manifest 已重算，旧结果目录与 OOF
  cache 不重命名、不复用，后续按新 identity/cache key 生成；本次不删除既有结果。
- 验证：AIDC 4,689 矩阵零漂移；checker
  `checked=5150 passed=5150 hard_failures=0 warnings=0`；Ensemble 审计
  `failures=[]`；package layering 17 项通过；全量 unittest 741 项、1167.647 秒、OK；
  compileall 与 `git diff --check` 通过。

## 15. Direct pointwise 共享模型接线（2026-09-03）

真实运行发现 `single_model_horizon` 此前只在 FeatureCompiler 中增加 horizon feature，
训练计划仍沿用标准 Direct 的 `model_count=H`，导致
`lgbm_direct-pointwise.yaml` 实际保存 96 个模型、执行 3,072 次 scalar fit，违背
§2.2.1 的共享单模型合同。现已增加 config-aware `target_plan_for_config()`：

- 标准 `independent_models` Direct 继续使用 H 个模型；
- `single_model_horizon` 保留 H 个 call，但全部 `model_indices` 指向 model 0；训练按
  horizon 顺序纵向拼接 H 组 X/Y，K=1 时每个 fold 只 fit 一个 estimator；
- `lgbm_direct-pointwise.yaml` 使用原始 `forecast_horizon_idx`；
  `lgbm_direct-pointwise-horizon.yaml` 在原始索引之外增加 sin/cos 周期编码，两者只在
  horizon 编码上不同，训练/预测共享模型算法完全相同；
- Local/Global、K1/K2、point/quantile 共用同一计划；Forecaster 校验 artifact 的完整
  target plan，旧 96 模型 bundle 在新 layout 下明确 RAISE，不能静默按错误语义部署；
- 两份 route_A baseline LightGBM 配置经实测使用
  `window_parallel_workers=2 + model_thread_count=4`，性能字段不进入 canonical
  fingerprint，生成器再次检查 4,689 份矩阵零漂移。

真实完整验收不缩短 6,336 origins、31 folds、H=96：plain cache-hit physical run
179.72s；cyclical 冷运行 236.38s，其中 compile 53.11s、非编译阶段 183.27s。
plain bundle 从错误布局的 80,127,265 bytes 降至 1,605,994 bytes（缩小 98.00%）；
cyclical bundle 为 1,648,584 bytes。两份bundle均为 1 model group / 1 estimator，分别
保留 36/38 个特征；回测 2,976 行、汇总评分 62 行、horizon 评分 5,952 行、正式预测
96 行，visibility proof 与 bundle reload 均通过。

该修复改变模型训练语义但不改变配置文本的语义 fingerprint；旧 pointwise 结果不可与
新结果做数值 exact，也不能继续当作 pointwise 结果使用。当前仅重跑上述两份明确目标
配置，其他 `single_model_horizon` 存量结果需在使用前重训。

## 16. 历史溯源

- 2026-08-24：概率预测专项方案，现保留为历史参考。
- 2026-08-27 至 29：预测问题正交化、schema-2 迁移、legacy 删除。
- 2026-08-29：引用式 Ensemble v4 实施。
- 2026-08-30 至 31：包级 DAG、时间几何、配置资产、产物和文档收敛；完整执行记录见 `docs/redesign/architecture.md`。

未经 wangzf 明确指示，不 commit、不 push；模型效果消融不属于本架构验收。

## 17. 测试执行分层与重复生命周期精简

状态：completed（2026-09-06，测试与文档范围，不修改生产算法）。

- `tests/run_suite.py` 提供互斥的 fast/integration/audit 集合与 all 全集，保留原生 unittest discovery；未分类新测试进入 integration，发现失败、重复 ID、空选择均非零退出。分组和覆盖映射以 `tests/README.md` 为准。
- OOFSpec 两个完全重复测试仅保留 specs 版本；融合有限值/method 断言并入跨方法 cache 测试，减少三次重复生命周期。
- TASK27 保留 46 场景：单模型完整 runtime 从 35 次降至 23 次，trainer/forecaster 从 7 次增至 19 次；仅下沉 Local K2 非 Direct 的 independent/native 组合。对应十二组合不再各自验收落盘；Direct 两 adapter 接线、七策略其余 runtime 轴及精确策略/adapter 测试保留。
- 历史 geometry migration manifest 不再锁定当前配置 fingerprint 和路径全集；现役配置旧字段拒绝及真实时间轴训练窗计数检查保留。未删除 manifest 或冻结分解 fixture。
- 暂不迁移目录或全面抽取跨测试 fixture；避免仅为文件数引入测试框架重构。

完成证据：启动时发现的 **866 项**全部执行，**1260.613 秒，OK，零跳过**。独立 fast 入口 **282 项通过**；最后一次发现快照为 **873 项**（fast 282 / integration 466 / audit 125），差额为其他并发改动新增的 5 项原生参数测试及 2 项日内调度测试，单独补验 **7 项 / 3.669 秒 / OK**。这是一次全量快照加增量验证，不是最新工作树重新执行 873 项全量的声明。单项设计审计 802.182 秒、矩阵 21.511 秒；存在外部并发任务，不作为隔离性能基准。逐测试时间不包括类级 setup/teardown，不能把其求和当作完整 wall time。

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -B tests/run_suite.py all --report .hermes/plans/test-suite-all-report.json
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -B tests/run_suite.py fast --report .hermes/plans/test-suite-fast-report.json
git diff --check
```

本地证据：`.hermes/plans/test-suite-all.log`、`test-suite-all-report.json`、`test-suite-fast.log`、`test-suite-fast-report.json`。长期契约不依赖这些本地日志。

### 17.1 休眠 DataFrame 概率后处理链退出

状态：completed。经用户批准，删除 `probabilistic/pipeline.py`、`probabilistic/postprocessing.py` 及 `tests/test_probabilistic_pipeline.py`、`tests/test_probabilistic_postprocessing.py`，同步测试分组与包职责说明。仓库 Python 源码中旧模块/函数引用为零；历史设计文档中的旧验收记录保留，不代表现役 API。此项移除旧 DataFrame API，不提供兼容 shim；仓库外直接导入者须迁移。现役 `repair_marginal_quantile_crossing` 和 CQR calibration 内核未修改。

验证：删除前后 discovery 差集精确为上述两份测试中的 3 个用例，无意外丢失。定向 **96 tests / 59.279s / OK**，覆盖 `test_crossing_method_config`、`test_multitarget_probabilistic`、`test_conformal`、`test_conformal_tracker`、`test_probabilistic_eval_wiring`、`test_probabilistic_contracts`、`test_package_layering`、`test_suite_runner`、`test_canonical_runtime_smoke`、`test_calendar_month_runtime`；`git diff --check` 通过。未再运行全量或生产配置重跑。运行证据为 `.hermes/plans/retired-probability-verification.log`；可用原生 unittest 按上述模块构造 TestSuite 复验。

### 17.2 测试专用生产接口退出与算法测试接线

状态：completed（2026-09-06，定向回归范围；不是全量生产配置数值验收）。

- runtime/fit service 的五个测试专用资源探针退出；测试通过 `tests/fixtures/runtime_planning.py` 调用公开 workload/planner/参数解析接口。仍有真实调用的 `_runtime_execution_plan` 回退保留。
- `calibration_runtime_kwargs`、`validate_probabilistic_args` 和旧一维 `ForecastDistribution` 退出。旧空间/stage 枚举随旧类型退出；张量形状、时间、point-quantile 绑定、轴一致性及区间边界断言继续覆盖。`resolve_probabilistic_spec` 等现存消费者所需接口不扩大删除。
- 旧 NNLS 黄金参照迁入 `tests/fixtures/legacy_nnls.py`，不再从生产融合模块导出。剔除 docstring 后与 HEAD 原函数的 AST 完全一致；保留独立数值参照，不委托新实现。
- `combine_members` 校验成员顺序后调用三个 combine 函数；weighted/linear blending 共享预测期加权运算，权重学习仍独立。新增生产委托及多目标 point/quantile 逐值测试，四方法运行与 bundle 重载部署测试通过。
- `inject_quantile_objective` 及其重复 alias/injector 表退出。支持性查询复用训练层从 catalog 派生的 registry；参数注入唯一入口为 `models.catalog.quantile_parameters`。
- CQR tracker 整批调用 `compute_nonconformity_scores` 后追加记录，拒绝有效有限值行中的倒置区间且不留半批状态；原有非有限值/掩码筛选不变。
- 资产缺列测试改走真实 CSV 审计报告后删除 `_missing_required_columns`；选日尾部误差测试改走 `run_selection` 后删除 `select_best_scenario_day`，不改正式选日规则。
- 保留公开部署/缓存 API、性能证据、provider 工厂、joint-sample unsupported 边界，以及尚未裁决退出的离线分解/crossing 诊断能力。动态注册的七策略和节假日 generator 不作为死代码处理。

实施差异与兼容边界：

| 项目 | 实施裁决与影响 |
|---|---|
| 旧 objective 测试额外设置 metric/eval_metric | 不将旧规则迁入生产；保留现役参数覆盖语义。合法 quantile 的 objective 映射不变；直接调用 catalog 的非法分位点及未知模型改为明确 ValueError |
| CQR score 合并 | 倒置区间由原来的直接计算变为报错，属于错误输入语义收紧，不宣称完全零行为变化；新增负向测试先复现失败再修复 |
| 旧 API 退出 | 仓库外旧函数 import 及旧 ForecastDistribution pickle 不提供兼容 shim；未修改 canonical bundle schema、配置 fingerprint 或存量结果 |
| 验证范围 | 不重跑正式模型配置、不清理结果、不重跑全量套件；运行链测试产物位于临时目录 |

验证证据：修改前 fast **282 项通过**；最终 fast **283 项 / 2.822 秒 / OK**；定向 **203 项 / 90.833 秒 / OK**，均零失败、零错误、零跳过。两组按 ID 去重覆盖 **379 项**（重叠 107 项），不是相加后声称完整全集通过。最终 discovery **890 项**（fast 283 / integration 482 / audit 125）。定向包括融合四方法运行、point/quantile bundle 脱离 YAML/OOF 的重载部署、CQR fixed/calendar checkpoint replay、全仓资产审计、选日输出及包层级门禁；`git diff --check` 通过。

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python .hermes/plans/verify-test-only-cleanup.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python tests/run_suite.py fast --report .hermes/plans/test-only-fast.json
git diff --check
```

定向选择及逐项结果记录于 `.hermes/plans/verify-test-only-cleanup.py`、`test-only-targeted.json`；基线和 fast 结果分别为 `test-only-baseline.json`、`test-only-fast.json`。运行日志为 `/tmp/tsproj-test-only-targeted.log` 与 `/tmp/tsproj-test-only-fast.log`，只作本地执行证据；覆盖迁移的长期映射见 `tests/README.md`。
