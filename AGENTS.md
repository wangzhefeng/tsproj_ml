# AGENTS.md — tsproj_ml

Project-specific conventions shared by Claude Code and Codex CLI. `AGENTS.md` is the single source of truth; `CLAUDE.md` contains only an `@AGENTS.md` reference that imports this file. Edit here and both agents stay in sync — do not add project conventions to `CLAUDE.md`.

The general coding guidelines (Karpathy: think before coding, simplicity, surgical changes, goal-driven execution) live in the user-global config — `~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for Codex CLI — and are intentionally not duplicated here.

## Project Conventions

### 配置系统
- **现役模型配置统一为 canonical schema**：仓库内 5,150 个活动模型 YAML 均以 `schema_version: 2` 声明（5,072 个 `ForecastConfigSpec` + 78 个引用式 `EnsembleConfigSpec`，另有 41 个数据工具 YAML 不计模型数）；三个 AIDC 15min 负荷场景共 4,689 份（4,617 个六组全因子单模型 + 72 个引用式 Ensemble），不维护重复的 `ensemble_members/`。每路 baseline 为 9 模型 × 9 策略共 81 份，其中 ST/LightGBM/Ridge × Direct/Recursive/MIMO 九个普通成员供 `add_ensemble/` 的三组 Latin-square × 四方法直接引用。3 个数据列族错配配置已于 2026-08-31 经批准从活动集移除，历史内容由 Git 保留，不计入活动集。`load_yaml_config()` 按互斥字段集合分派返回 `ForecastConfigSpec | EnsembleConfigSpec`，未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE。
- **公共配置不暴露内部固定值或死参数**：`problem.information_mode`、`output.setting_suffix`、`probabilistic.recursive_propagation`、`probabilistic.schema_version` 已删除，重新声明一律 RAISE。预测输入始终执行严格 as-of；监督训练仅通过内部 `target_access=supervised_labels` 放开预测期 target 标签，不放开 known-future 或历史 target revision 的时间边界。递归 quantile 内部固定走 `median_path`；概率部署产物仍独立保存并校验其内部 schema 版本。
- legacy 配置体系（`base_config + overrides`、扁平 dataclass 模板、九方法名）已于 2026-08-29 彻底删除：config/ 下只存在 `schema_version: 2` canonical YAML 与数据工具 YAML，`load_yaml_config()` 只解析 canonical schema，其余一律 RAISE。
- canonical fingerprint 只取语义 payload；并行度、日志和输出目录相关字段不进入 fingerprint。结果目录使用可读 identity + 12 位 fingerprint，语义相同的配置别名可共享 identity，这不是 hash 碰撞。

### 时间边界约定
- `now_time` 配置值 = 最后一个已知数据点（如 `2026-06-11T23:55:00`）
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。日志/文件名的时间戳仍按 `now_time` 原值
- 内部分界点 = `schedule_mode=daily` 时 `floor("1D") + 1day` = 次日 00:00:00；`intraday` 时 = `now_time` 本身
- fixed-step 的 `validation.history_steps/train_window_steps/fold_count/stride_steps` 统一以监督 origin steps 计。calendar-month 使用独立的 `train_window_days/fold_count/stride_months`；其中训练窗按原始日数计、折间距按自然月计。
- **`predict_steps` 以 `freq` 为单位计步**（取代旧 `predict_days`）：15min 下 1 天 = 96、4 小时 = 16；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不再经 `n_per_day` 换算
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### Canonical 预测架构
- **权威状态**：`docs/multistep_forecasting_redesign.md` 是当前预测架构唯一权威文档与实施完成记录（历史分立设计文档已并入本文）。现役 canonical 配置由 `main.py::CanonicalModel` 调用 `model_forecasting.runtime.run_canonical_config()`，生产主链不再按九个 US/MS 名称分派。
- **七种标准策略**：`recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo` 只在 `forecasting_core/specs/strategy.py` 定义，由 `model_training/strategies/` 的七 executor 执行。MO 策略严格要求 `1 < B < H` 且 `H % B == 0`；Pointwise/horizon-feature 是 Direct layout，Local/Global 是 training scope，不得再扩充为策略名。
- **统一张量**：合同位于 `forecasting_core/tensors.py`；point 为 `(N,H,K)`，边际 quantile 为 `(N,H,K,Q)`；内部 flatten 固定 time-major。joint samples 仅保留 `(N,S,H,K)` 类型边界，生成器明确 unsupported，不得宣称联合概率能力。
- **数据与特征**：`SourceRegistry + InformationSetRequest + FeatureCompiler` 是 canonical 唯一通路。`DataSourceSpec.columns` 是进入模型的信息投影视图：每个声明列必须显式归为 target/observed_past/known_future/static/key/ignored，非 ignored 声明列在物理资产中缺失直接 RAISE；物理文件的其他列在 registry 边界丢弃，不会隐式入模。所有动态 source 执行严格 as-of，observed-past 只能使用显式 provider，不得隐式 persistence。**lag 深度与 provider 的分工规则**（2026-09-01 契约化）：默认目标日对齐（`align_to_target` 未设/false 语义见 Direct 条）下 `min(lags) >= horizon` 的声明（safe-lag，M5 forecast-date-aligned 模式）全程消费 `<= forecast_origin` 的真实历史值，provider 机制不介入、VisibilityProof 全部记录真实 source_time（有专测钉住）；浅于 horizon 的 lag 在 h 使 source_time 越过原点时必须显式选择 provider 三选一，此时 persistence（最后观测值冻结）是零假设兜底——两种声明服务不同信息需求，可同时声明由模型自学取舍。
- **训练与产物**：`CanonicalTrainer/CanonicalForecaster` 支持 Local/Global、K1/K2、七策略、point/independent quantile；target transform 按 `(series_id,target)` 隔离并执行 calendar normalization → decomposition → scaling，恢复严格逆序。回测与 final fit 使用同一配置训练窗口。新模型只写 schema-2 `ForecastModelBundle`，新结果只写 long schema 和 canonical fingerprint。
- **Ensemble（引用式）**：顶层 `model_ensemble/` 包唯一承载融合；四方法为 `averaging/weighted/linear_blending/stacking`。Quantile linear blending 按 target 最小化 simplex pooled pinball；OOF cache 对 `(member,source,path_role,file_sha256)` 内容寻址。Ensemble bundle 自包含成员 bundle，部署不读取成员 YAML/OOF cache，并写与单模型相同的 canonical long 结果布局。
- **CQR（2026-09-01 已激活）**：canonical 覆盖 point、边际 quantile 与可选 CQR 校准区间。`probabilistic.calibration.method=cqr` 启用严格 as-of 的逐折 apply-before-collect 校准；产出 `predict_pi<coverage>_lower/upper`，final 修正量写入 bundle `calibration_state` 并在部署期应用、不重校准。三个 AIDC 15min 负荷新矩阵按用户裁决全部使用 point 模式，因此不启用 CQR。
- **历史迁移记录**：18 个旧 MDR 配置曾因 `H % B != 0` 映射为 `recursive`（14 个 `aidc_power_month`、4 个 `aidc_load_month`，2026-08-29 批准）；迁移溯源字段 `validation.legacy_origin` 已随 legacy 清理从全部 YAML 删除，批准记录仅存于 `docs/multistep_forecasting_redesign.md`。
- **canonical 单运行时**：`main.py`、`run.py` 只接受 `ForecastConfigSpec`。旧 `Model/Trainer/Tester/Forecaster`、`models/multistep/`（含 executors 与只读语义层）和 legacy config generator 已全部删除（2026-08-29 收口），门禁测试断言相关目录不存在。
- **无 legacy 层**：旧 YAML/旧 pkl/旧宽表在本仓库不再可读；`ModelSaveLoad.load_model()` 直接返回加载对象，`CanonicalResultReader` 只接受 canonical long schema，非 canonical 输入直接 RAISE。

### 模型工厂（ModelFactory）
- **支持的模型类型**：LightGBM（`lgb`/`lightgbm`）、XGBoost（`xgb`/`xgboost`）、CatBoost（`cat`/`catboost`）、RandomForest（`rf`/`randomforest`）、HistGradientBoosting（`histgb`/`histgradientboosting`）、Ridge（`ridge`）、ElasticNet（`enet`/`elasticnet`）、Lasso（`lasso`）、QuantileRegressor（`qr`/`quantileregressor`）、SeasonalTemplate（`st`/`seasonaltemplate`，工作日/周末分组建模板 + NNLS 学权重）
- **融合训练数据一致性**：所有融合方法的成员 final fit 与对应单模型使用同一显式训练窗口；禁止某方法只训练部分历史或与回测窗口采用不同隐式策略。
- **多文件外生来源**（2026-09-01 文档漂移清理：legacy `custom_features`/`ExogenousFeatureConfig` 与 `plot_overlay_path`/`plot_overlay_col` 已随 canonical 收口删除，canonical 代码与全部现役 YAML 零引用）：多文件外生一律走 `data.sources` 多 source 声明——历史有真值、预测期无值的列挂 `observed_past` 角色 + 显式 provider 三选一（`persistence`/`auxiliary`/`provided_scenario`，禁止隐式 persistence）；未来可知的列（天气预报/计划表）挂 `known_future` 角色并用 `history_path + backtest_path + future_path` 三段路径。测试图叠加参考曲线需求未迁移，如需恢复按 canonical OutputConfig 重新设计。

### AIDC 负荷事件标签工具链（2026-08 新增）
- **共享检测核心** `data_process/load_event_detection.py`（有单测 `tests/test_load_event_detection.py`）：事件分类 shift_up/down（持久阶跃=集中上下架）、stress_up/down（1~21 天临时偏移=压测/临时操作）、burst_up/down（1.25h~24h 日内冲击）、spike_up/down（≤1h 功率突变）；三个探测器（自顶向下日级分段 + 短时偏移 + 15min 残差 MAD 突变）+ 边界伪影抑制 + 事件→逐点/逐日投影
- **场景入口**：`config/aidc_load_15min_daily/load_event_analysis.py`（15min 逐点标签 + 特征）、`config/aidc_load_month/load_event_analysis.py`（日频逐日标签 + 特征）；产物在 `dataset/<场景>/event_label_features/`（labeled_features.csv / events.csv / overview.png / report.md）。两频率事件明细完全一致（同一检测核心、日水平基线均取 15min 聚合逐日中位数）
- **列名前缀约定**：`feat_`=本频率 trailing 特征（无泄漏，可建模）、`xf_`=跨频率特征（15min↔日频互取）、`xr_`=跨 route 特征、`lbl_`=事件标签（检测含居中窗口=有未来信息，仅供离线分析/样本筛选，禁止直接作在线预测特征）
- 注意 `dataset/aidc_load_month/` 目录名沿用场景名，文件实际是 **1day 粒度**

### 低频（日/周/月）数据配置约定
- **中国节假日 builtin generator**（2026-09-01 新增，方案 A 默认 + B 审计兜底）：generator 名 `chinese_holiday`（chinese-calendar 后端，builtin 注册、无需 run_canonical_config 传参）。YAML 声明 `source_type: generated + generator: chinese_holiday + availability: generator_defined`（generated known_future 的 spec 层强约束），列 `is_holiday`（含调休连休）/`holiday_name`（categorical）/`next_holiday_days`（节前倒计时，日历日；超出已知年历取删失哨兵 400——次年年历通常年底发布，属有文档截断非编造值）。库覆盖 2004 起逐年扩展，覆盖外日期直接 RAISE 不静默降级；每年底国务院发布次年安排后需 `uv add chinese-calendar --upgrade`。审计兜底：`scripts/export_chinese_holiday_csv.py` 导出与在线逐点一致（有测试）。日频/日内频率适用；月频网格不适用（known_future 逐点精确匹配 RAISE）。三个 AIDC 15min 负荷场景共 972 份现役 YAML 已启用；其他场景启用仍属语义变更，按消融流程单独验证。
- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配
- **月频（`1ME`/`1MS`）已支持**：频率解析位于 `utils/frequency.py`；月频 seasonal-naive 使用月步 offset，不得转换为固定 Timedelta。
- **月度/日度气象严格信息集**（aidc_power_month）：现役载体是 canonical 通用机制——`DataSpec` 列角色 + `SourceRegistry` as-of 校验（`available_at <= forecast_origin`，目标时间戳缺失直接 RAISE）；legacy 字段 `strict_weather_information_set`、`weather_backtest_path`、`weather_*_source` 已废弃且无运行时语义（2026-08-29 收敛方案核实；当前 5,150 份现役模型 YAML 零处使用）。语义要求不变：历史训练天气为 actual；滑窗测试必须消费独立 ex-ante forecast/proxy 文件 source；正式 future 只允许 forecast|proxy；禁止把测试月实测天气当未来天气。2026-08 日频 future 使用 2025-08 同日纯代理（31 天），不含 2026-08 实测；回测 2026-01~07 使用上一年同日代理（212 天）。
- **Direct 目标日外生对齐**（2026-09-01 文档漂移清理：canonical 运行时无 `use_horizon_exogenous*` 消费点——known_future 外生天然按 `col(t+h)` 对目标时刻取值，历史锚点由 `features.transformations.direct.align_to_target` 控制，天气组配置为 `false` 即 lag/rolling/diff 冻结在预测原点；legacy `_h{h}` 宽表折叠通路（`HorizonAlignedDirectRegressor`，无主链调用方）已于 2026-09-01 从 `model_training/trainer.py` 删除，Direct 现役拟合通路唯一：`CanonicalTrainer → estimator.target_adapter` 三 adapter；`use_horizon_exogenous` 冗余字段已于 2026-09-01 从全部 519 个 YAML 清扫（wangzf 裁决不重跑，存量 results 按 identity 前缀对应））：天气组 source 用 `history_path`（历史实测）+ `backtest_path`（回测 ex-ante 代理）+ `future_path`（正式 future 预报/代理）三段路径。USBR 的 Direct/Recursive 共享 X 暂不支持两套外生时点，天气组保留为无天气 control（2026-09-01 核实：USBR 配置零 weather 引用）。USMDP 现役目标 lag（如 `[31,35,42,49,56]`）全部 ≥ horizon，安全距离由 YAML 直接表达。需要逐步消费自身预测时使用 USMR
- **calendar_month 训练窗口**：`horizon_mode=calendar_month` 时训练原始日数由 `train_window_days` 决定，回测折数与自然月间距分别由 `fold_count/stride_months` 表达；每个历史月和最终目标月动态解析 28/29/30/31 步。生产折在 `model_testing/validation.py::calendar_month_folds`，不使用 checker 私有实现。
- **回测/final 同合同**：fixed-step 与 calendar-month 的 final fit 使用各自配置训练窗口，不再隐式改成全部历史。
- **datetime 派生剔除子日粒度**：低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项

### Canonical 运行边界
- canonical rolling backtest、final training 和 forecast 均由 `model_forecasting/runtime.py` 顶层编排，design/fit/calendar-backtest/persistence 分别委托给同包窄模块；不得重新接入已删除的 `ModelTesting` 或旧 executor。
- estimator 并行能力必须由 `model_training/estimators/capabilities.py` 显式探测；不支持时 RAISE，不静默降级。
- 运行并行统一由 `model_forecasting/resource_planner.py` 生成 `RuntimeExecutionPlan`；fixed-step、calendar-month、point/quantile 与 Ensemble 不得自行读取 `validation.performance`。每次最多一个外层并行轴大于 1，线程/内存冲突直接 RAISE；performance 不进入 semantic fingerprint，resolved metadata 必须记录 workload、budget、plan、cache 与阶段 wall。
- 批调度与恢复合同：父并发乘法组合，不覆盖；全批 raw-design preflight 通过后才执行模型。缓存有容量上限，RSS 为采样观测而非硬内存隔离。completed 必须校验内容摘要、canonical 维度及 bundle 身份，calendar-month 按各折真实时间范围验收。`forecasting_core.checkpoints.FitCheckpoint` 是训练侧唯一持久化 Protocol，L2 `model_forecasting.checkpoints` 承载本地存储；只恢复已完成拟合单元，损坏 RAISE、变更身份 miss，CQR 依次重放而非跳过历史收集。动态 runner factory 必须显式透传 `checkpoint_root`，部署 bundle 不依赖 checkpoint。
- 已标定 performance 必须用可校验 `profile_ref`，与普通显式 override 区分；planner 输入真实 workload/base_dir/feature_schema，调度后仍须核对最终计划。唯一 CatBoost profile 的证据是 5-fold，物理 31-fold 不是已验证正式性能收益；签名及来源均不进入 semantic fingerprint。

### 包间分层规则（2026-08-29 架构收敛 P0 立规，详见 docs/architecture_convergence_plan.md §3.1）
- **分层结构（2026-09-05 批调度接线）**：L4 入口（`main.py`/`run.py`/`batch_run.py`/`config/config_loader.py`）→ L3 能力扩展（`probabilistic/`、`model_ensemble/`）→ L2 编排（`model_forecasting/runtime.py` 顶层编排 + `design.py` 信息集设计 + `fit_service.py` 训练服务 + `backtest_runtime.py` 自然月回测 + `resource_planner.py` 统一资源计划 + `batch_runtime.py` 跨配置批调度 + `transform_cache.py` 批内 fold transform 复用 + `forecaster.py` 推理 + `persistence.py` 产物持久化 + `{transforms,results}` 运行服务）→ 稳定合同层 `forecasting_core/{specs,tensors,artifacts,probabilistic_spec,runtime_resources}` → 阶段顶层包 `data_loading/`（数据构造与组织）→ `feature_engineering/`（特征工程）→ `model_training/`（训练）→ `model_testing/`（测试/回测），`model_evaluation/`（独立评估模块，只做指标计算）→ L1 基础设施（`models/`、`decomposition/`、`data_process/`）→ L0 基础（`utils/`）。
- **依赖只允许从上往下**（高层 import 低层），同层禁止互依；唯一豁免：L3 内 `ensemble → probabilistic` 单向（数据合同 types + 评估实现 evaluation 复用，不得触及 pipeline/training 执行面）。低层不得反向 import 高层，**包括函数内延迟 import**。
- 跨包复用的算法放 L1 并以**公开函数**暴露；**禁止下划线私有跨包导入**（ensemble 对 forecasting 的两处私有导入已随收敛 P3 修复为 `model_testing/backtest.py` 公开 API）。
- 任何循环依赖视为架构缺陷，出现即修；不允许用函数内延迟 import 长期绕开。`models/ModelTraining.py`、`models/ModelForecasting.py` 兼容 shim 已于 2026-08-30 删除，Trainer/Forecaster 直接从 `model_training/trainer.py`、`model_forecasting/forecaster.py` 导入；分层边界由 `tests/test_package_layering.py` 门禁固化（白名单制 13 个项目包全扫描，函数内 import 同罪）。
- 新增能力包一律按 ensemble 模式建：独立包 + Protocol 边界 + 单向依赖 + parity golden。ensemble 已知边界：`model_ensemble/loader.py` 直接操作 canonical YAML payload（成员身份锁定为现役单模型 YAML，属有意设计）。
- **流水线数据契约**：进入模型的信息集默认「缺失/异常 = RAISE」（预测性维护数据前置）；填补与清洗只允许发生在离线数据准备阶段（`data_process/`，场景维度）或以 config 驱动的 compiler 前置步骤引入（须满足 as-of 可得性）；训练窗口内清洗（改数据）与评估掩码（只改评估口径）严格分离。

### 关键约定
- 结果输出目录（2026-09-01 收口）：**全部模型配置统一写 `results/{pretrained_models,results_test,results_forecast}/<scenario_subpath>/<result_identity>/`** 三顶层目录；单模型与引用式 ensemble YAML 的 `output.directories` 均显式声明 `checkpoints: ./results/pretrained_models/`、`tests: ./results/results_test/`、`forecast: ./results/results_forecast/`（ensemble 路径分支在 `model_ensemble/runtime.py::_ensemble_output_paths`，`output_root` 显式覆写仍最优先）。`result_identity`：单模型 = `<strategy>-<model>-<scope>-k<K>-<fingerprint[:12]>`；ensemble = `ensemble-<method>-<scope>-k<K>-<fp[:12]>`。兼容形态：未声明 `directories` 时 runtime 回退 `results/<scenario>/<identity>/` 同 identity 三子树；OOF cache 仍写在 `results/_ensemble_oof/`。
- `prediction.csv` / `cv_plot_df.csv` 使用 `(series_id,time,target[,window])` long 唯一键；`test_scores_df.csv` 写 per-target 与 aggregate 行；**quantile 模式追加 `test_scores_probabilistic_df.csv`**（tidy long：mae/逐 level pinball/central 区间覆盖率·宽度·winkler·coverage gap，eval_mask 口径与点评估共用 `build_eval_mask_payload`，aggregate 行跨 target 按有效点池化）。ensemble 的无泄漏质量证据是**融合 OOF 评分**（`model_ensemble/evaluation.py`，复用同一评估实现）：结果挂在 `run_ensemble_config` 返回的 `fused_oof_scores` / audit，run.py 日志输出 aggregate MAE/RMSE/MAPE。
- 目标训练变换顺序固定为 `calendar normalization → decomposition → target scaling`，point/quantile 严格逆序恢复；状态按 `(series_id,target)` 隔离并随 bundle 保存。
- 本地入口只接受 canonical schema YAML：修改 `main.py::CONFIG_YAML` 后运行 `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py`。非 canonical YAML 一律 RAISE。
- `run.py` 只对 canonical payload 应用 CLI override；非 canonical 配置明确 RAISE（不再指向已删除的迁移脚本）。

- 日志目录：`logs/main/`（直接运行 `main.py`）
- `log_util.py` 在模块导入时执行，`LOG_NAME` 用 `os.environ.get('LOG_NAME', 'main')` 提供默认值
- 在这台机器上如果 `uv` 触发 `~/.cache/uv` 权限问题，优先使用 `UV_CACHE_DIR=.uv_cache`
- 算力房间数据：`df.csv`（215 列，全量合并）、`df_selected.csv`（多变量筛后建模数据）、`feature_selection_report.csv`（筛列报告）和稠密版 `df_filled.csv`（`config/aidc_electricity_computility/electricity/2026-06-11/scripts/fill_missing.py`：原始列补 0 + 重算派生）都在 `demand_load/<room>/` 下；算力预处理权威入口是 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py`，特征分原始聚合、状态/结构、时序动态三层；单变量配置继续使用 `df_power.csv`，4 个房间场景的多变量 LGBM 配置使用 `df_selected.csv`
- 算力天气 `cal_rh` 必须在离线数据准备阶段由 `rt_tt2`/`rt_dt` 按 Magnus–Tetens 公式计算并补齐，权威迁移入口为 `config/aidc_electricity_computility/derive_cal_rh.py`；canonical runtime 不恢复 legacy `FeatureEngineering` 的现场派生或插值。
- **2026-08-10 算力场景**（A2_IT / A3_IT / liantong_IT / yancheng_IT，各 7 个 YAML）：`dataset/.../2026-08-10/` 下房间目录当前为空，YAML 的 `data_dir` 指向 2026-06-11 数据（复用上批数据做配置模板）

### 仓库维护注意
- **Agent 工作文档分流**：Hermes Agent 生成的实施计划只放 `.hermes/plans/`；Codex/Superpowers 生成的设计与计划只放 `.agents/superpowers/{specs,plans}/`。不再使用 `docs/plans/` 或 `docs/superpowers/` 保存 Agent 过程文档；`docs/` 只保留面向项目使用者的长期有效文档。
- **`tests/` 已纳入版本控制**（2026-08-15 起移出 `.gitignore`）：改动被测模块时必须同步修复测试导入与接口断言，并运行 `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s tests -p "test_*.py"`；预测架构的最新精确计数与命令只在 `docs/multistep_forecasting_redesign.md` 维护。AIDC 专项脚本目录路径含日期段、不是合法 Python 包，相关测试用 `sys.path.insert` 引导后按模块名导入。
- `docs/feature_engineering/`（fft/wavelet/statistical/holiday/periodicity/weather 等）是实验性特征脚本集合，**未接入主流程**，2026-08-29 从 `features/feature_engineering/` 移入 `docs/` 归档，后续再决定处置
- `utils/`（L0）只保留 frequency/log_util/runtime_env 三个无领域语义工具；`utils/metrics.py`、`utils/cv_plot.py`、`utils/eval_mask.py` 已删除（指标在 `model_evaluation/`（point.py/marginal.py/metrics.py），评估掩码在 `model_evaluation/mask.py::build_eval_mask`）
