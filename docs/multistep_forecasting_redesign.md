# 多步预测 Canonical 架构与实施记录

> 文档状态：**当前唯一权威设计／C0–C7 全部完成并关闭（2026-08-31）**
> 当前实现以代码、活动 schema-2 YAML 和本次 fresh 验证为准；历史方案见 Git 及文末溯源，不再作为现状事实源。
> 七类运行、全量门禁、source header 合同和活动文档均有 fresh 证据；Global N2K2 fixture 已获批作为能力验收，`AGENTS.md` 当前合同已同步。

## 1. 当前边界

`tsproj_ml` 只保留 canonical 运行体系：

- 5,150 个活动模型 YAML 均为 `schema_version: 2`：5,066 个普通单模型、6 个其他场景保留的独立 Ensemble member、78 个引用式 Ensemble；按 typed spec 计为 5,072 个 `ForecastConfigSpec` + 78 个 `EnsembleConfigSpec`。另有 41 个数据工具 YAML，不计入模型数。三个 AIDC 15min 负荷场景共 4,689 份：4,617 份六组全因子单模型，另有 72 个 Ensemble 按三组 Latin-square × 四方法直接引用 baseline 成员，不维护 `ensemble_members/` 副本；3 个无有效资产配置原样保存在 `docs/archived_configs/`，不进入 loader/audit/runtime。
- `load_yaml_config()` 严格分派 `ForecastConfigSpec | EnsembleConfigSpec`。未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE。
- `main.py`、`run.py` 不接受 legacy 配置或旧产物。旧 `base_config + overrides`、九方法运行类、旧宽表、旧 pkl 和迁移兼容层均已删除。
- `docs/model_ensemble_redesign.md` 与 `docs/time_series_probabilistic_forecasting_redesign.md` 仅作历史决策参考，本文和 `AGENTS.md` 才是当前事实源。

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

五个配置维度互相正交，各自独立取值、组合生效；理解任何一个配置先定位它在下表的位置：

| 维度 | 配置字段 | 取值 | 行为作用层 | 代码入口 |
|---|---|---|---|---|
| 策略 | `strategy.name`（+MO 的 `output_chunk_length`） | recursive / direct / mimo / recmo / dirrec / dirmo / dirrecmo | H 步输出如何在模型调用间分配、是否消费自身预测 | `forecasting_core/specs/strategy.py` 规则表；`model_training/strategies/` 执行 |
| Direct layout | `features.transformations.direct.layout` | independent_models（H 个独立模型）/ single_model_horizon（共享单模型=pointwise，必配 horizon_feature） | 同一 Direct 策略下训练样本的行×列组织 | `feature_engineering/compiler.py`；`estimators/multi_target.py` |
| 训练 scope | `problem.training_scope` | local（逐序列独立建模）/ global（panel 池化共享模型，series_id 作 key 特征） | 序列间是否共享参数 | `forecasting_core/specs/problem.py` |
| 估计器耦合 | `estimator.target_adapter` | independent（标量模型组）/ regressor_chain（标量组+前序输出作输入，不支持 quantile）/ native（单实例多输出） | 语义输出块到物理估计器实例的映射 | `model_training/estimators/multi_target.py` |
| 信息模式 | `problem.information_mode` | forecast / nowcast / oracle | 信息集 as-of 可见性 | `forecasting_core/specs/problem.py` + `SourceRegistry` |

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

成员必须引用现役单模型 YAML，并共享 problem/data/probabilistic/origin 合同。OOF cache 对 `(member, source, path_role, file_sha256)` 内容寻址；损坏或输入内容变化必须失效。

Quantile linear blending 按 target 最小化 simplex pooled pinball，一组 target 权重跨 Q 共享。最终 Ensemble bundle 内嵌全部成员 bundle 和 method artifact；部署预测不读取成员 YAML 或 OOF cache。

## 7. 概率边界

生产能力覆盖 point、边际 quantile 与可选 CQR 校准区间（2026-09-01 激活）：

- quantile 模式写逐 level pinball、bias、central interval coverage/width/Winkler/coverage gap，并按 horizon 步拆分诊断行（2026-09-02）；
- crossing 后处理消费 `probabilistic.crossing.method` 配置（none/rearrangement/median_preserving_isotonic，缺省 median_preserving_isotonic=历史行为）；部署期从 bundle spec 读取（2026-09-01 裂缝修复：此前配置被静默忽略、runtime 无条件修复）；
- CQR：`probabilistic.calibration`（method=cqr）声明即启用。回测逐折 apply-before-collect，只消费 origin 严格更早且标签已可得的历史折（as-of 双门槛 min_windows/min_scores，pooled 分组跨 series/target）；产出 `predict_pi<coverage>_lower/upper` 列进 `cv_plot_df.csv` 与 `prediction.csv`；final 修正量冻结进 bundle `calibration_state`（`ForecastModelBundle.calibration_state`），部署期应用、不重校准，且经 `distribution_to_long` 消费 bundle metadata `prediction_intervals` 落成部署链 `prediction.csv` 的 pi 列（2026-09-02 接线，此前为死写；同批修复 deployment `_predict_quantiles` 的 `bundle_crossing_method` NameError、checker probabilistic 键白名单与 geometry manifest fingerprint 再生成）。不足门槛时诚实降级：不产出 pi 列，`result_metadata.json` 记 status/reason。fixed-step（`runtime.py`）与 calendar-month（`backtest_runtime.py`）共用 `probabilistic/calibration.py::ConformalCalibrationTracker`；
- legacy `crossing_method`/`conformal` 键已于 2026-09-01 从全部 YAML 清扫（379 个文件：`crossing_method: isotonic` → `crossing:` 块；`conformal: {method: none}` 删除；146 个 `method: cqr` 迁入 `intervals:` + `calibration:`），spec 层对 legacy 键一律 RAISE。注意 fingerprint 随之变化，存量 results 沿用旧 identity 前缀；
- quantile 训练优化（2026-09-01）：逐 level 训练默认线程并行（数值与串行一致）；xgboost≥2.0 自动走原生多分位共享 booster（训练成本 ≈1× 而非 Q×，与并行互斥；现役 YAML 零 xgb quantile 配置）；
- 仍不支持 joint sample generator、联合路径概率或 ensemble-of-ensemble。

## 8. 结果与产物合同

```text
results/<scenario>/<result_identity>/
├── pretrained_models/
│   ├── model.pkl
│   └── resolved_model.json
├── results_test/
│   ├── cv_plot_df.csv
│   ├── test_scores_df.csv
│   ├── test_scores_probabilistic_df.csv   # quantile only
│   ├── test_prediction.png                # 单 target 拼接总图（多 target 在 target_plots/）
│   ├── target_plots/                      # 多 target 拼接总图（per target 一张）
│   ├── windows_results/                   # per-window 图（window_<w>.png，多 target 子图）
│   └── result_metadata.json
└── results_forecast/
    ├── prediction.csv
    ├── forecast_prediction.png            # 预测图（单 target；多 target 在 prediction_plots/）
    └── resolved_config.json
```

- `prediction.csv` 唯一键：`(series_id,time,target)`。
- `cv_plot_df.csv` 唯一键：`(series_id,time,target,window)`。
- 可视化产物（2026-09-02，绘图消费未掩码原始值，掩码只用于指标）：回测总图按时间排序后整条拼接（现役 stride==horizon 契约下窗口首尾相接；同 series 时间戳重复即 stride<horizon 重叠配置，拼接 RAISE 并指向 windows_results/ 单窗图）；`windows_results/window_<w>.png` 每窗一文件、多 target 纵向子图；正式预测写 `forecast_prediction.png`（多 target → `prediction_plots/`），quantile 模式附 PI 区间带。线型：Trues 实线 / Preds 点划线（沿用旧版 `models/ModelTesting.py` 口径）。
- `test_scores_df.csv`（2026-09-02 扩展）：`scope ∈ {target, aggregate, horizon, aggregate_horizon}`。`target`/`aggregate` 行与历史一致（aggregate 为 target 加权）；指标列含 `Bias`（= mean(pred−actual)，正=高估，含 `Naive Bias` 对照）；`horizon`/`aggregate_horizon` 行为 per-horizon 诊断（2026-09-02 新增），`horizon` 列 1-based，`aggregate_horizon` 跨 target 按有效点池化（proper score 语义）。
- `test_scores_probabilistic_df.csv`：tidy long，metric ∈ {mae, bias, pinball, interval_coverage, interval_width, interval_winkler, coverage_gap, calibration_error}；`horizon` 列仅 per-horizon 行非空；`scope ∈ {target, aggregate, horizon, aggregate_horizon}`。
- 单模型和 Ensemble 均保存 schema-2 `ForecastModelBundle`。
- fingerprint 只取语义 payload；日志、并行度和输出目录不进入 fingerprint。
- resolved config/model 必须足以复算 training policy、轴顺序、source/feature lineage 和产物身份。

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

C7 的命令、计数和七类产物核验结果只在 `docs/architecture_convergence_plan.md` 的最终执行记录维护；其他文档不复制测试数。

## 10. 历史决策与偏差

- 18 个旧 MDR 配置因 `H % B != 0` 于 2026-08-29 获批映射为 `recursive`：14 个 `aidc_power_month`、4 个 `aidc_load_month`。`validation.legacy_origin` 已删除，本条是批准记录。
- 旧 Direct/Recursive Blend 已迁入 Ensemble，不再作为 strategy。
- decomposition `forecasters.py/composers.py` 曾被误判为死链；包内消费者复核证明它们属于 `DecompositionPipeline` 活路径，因此保留。
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

前轮 C0–C7 的 fresh 命令、精确计数、七类产物和阻断项只在 `docs/architecture_convergence_plan.md` §17 维护；本次 V4 配置矩阵扩展的静态证据见下节，其他文档中的旧测试数只代表历史环境。

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

## 13. 历史溯源

- 2026-08-24：概率预测专项方案，现保留为历史参考。
- 2026-08-27 至 29：预测问题正交化、schema-2 迁移、legacy 删除。
- 2026-08-29：引用式 Ensemble v4 实施。
- 2026-08-30 至 31：包级 DAG、时间几何、配置资产、产物和文档收敛；完整执行记录见 `docs/architecture_convergence_plan.md`。

未经 wangzf 明确指示，不 commit、不 push；模型效果消融不属于本架构验收。
