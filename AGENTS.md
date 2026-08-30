# AGENTS.md — tsproj_ml

Project-specific conventions shared by Claude Code and Codex CLI. `AGENTS.md` is the single source of truth; `CLAUDE.md` contains only an `@AGENTS.md` reference that imports this file. Edit here and both agents stay in sync — do not add project conventions to `CLAUDE.md`.

The general coding guidelines (Karpathy: think before coding, simplicity, surgical changes, goal-driven execution) live in the user-global config — `~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for Codex CLI — and are intentionally not duplicated here.

## Project Conventions

### 配置系统
- **现役模型配置统一为 canonical schema**：仓库内 848 个模型 YAML 均以 `schema_version: 2` 声明（800 单模型 + 24 引用式 ensemble + 24 个 `ensemble_members/` 独立基模型，另有 41 个数据工具 YAML 不计模型数）；`load_yaml_config()` 按互斥字段集合分派返回 `ForecastConfigSpec | EnsembleConfigSpec`，未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE。
- legacy 配置体系（`base_config + overrides`、扁平 dataclass 模板、九方法名）已于 2026-08-29 彻底删除：config/ 下只存在 `schema_version: 2` canonical YAML 与数据工具 YAML，`load_yaml_config()` 只解析 canonical schema，其余一律 RAISE。
- canonical fingerprint 只取语义 payload；并行度、日志和输出目录相关字段不进入 fingerprint。结果目录使用可读 identity + 12 位 fingerprint，语义相同的配置别名可共享 identity，这不是 hash 碰撞。

### 时间边界约定
- `now_time` 配置值 = 最后一个已知数据点（如 `2026-06-11T23:55:00`）
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。日志/文件名的时间戳仍按 `now_time` 原值
- 内部分界点 = `schedule_mode=daily` 时 `floor("1D") + 1day` = 次日 00:00:00；`intraday` 时 = `now_time` 本身
- `start_time = now_time - history_length`，`future_time = now_time + predict_steps × freq 步长`
- **`predict_steps` 以 `freq` 为单位计步**（取代旧 `predict_days`）：15min 下 1 天 = 96、4 小时 = 16；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不再经 `n_per_day` 换算
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### Canonical 预测架构
- **权威状态**：`docs/multistep_forecasting_redesign.md` 是当前预测架构唯一权威文档与实施完成记录（历史分立设计文档已并入本文）。现役 canonical 配置由 `main.py::CanonicalModel` 调用 `model_forecasting.runtime.run_canonical_config()`，生产主链不再按九个 US/MS 名称分派。
- **七种标准策略**：`recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo` 只在 `model_forecasting/specs/strategy.py` 定义，由 `model_forecasting/strategies/` 的七 executor 执行。MO 策略严格要求 `1 < B < H` 且 `H % B == 0`；Pointwise/horizon-feature 是 Direct layout，Local/Global 是 training scope，不得再扩充为策略名。
- **统一张量**：point 为 `(N,H,K)`，边际 quantile 为 `(N,H,K,Q)`；内部 flatten 固定 time-major。joint samples 仅保留 `(N,S,H,K)` 类型边界，生成器明确 unsupported，不得宣称联合概率能力。
- **数据与特征**：`SourceRegistry + InformationSetRequest + FeatureCompiler` 是 canonical 唯一通路。每列必须显式归为 target/observed_past/known_future/static/key/ignored；所有动态 source 执行严格 as-of，observed-past 只能使用显式 provider，不得隐式 persistence。
- **训练与产物**：`CanonicalTrainer/CanonicalForecaster` 支持 Local/Global、K1/K2、七策略、point/independent quantile；target transform 按 `(series_id,target)` 隔离并执行 calendar normalization → decomposition → scaling，恢复严格逆序。新模型只写 schema-2 `ForecastModelBundle`，新结果只写 long schema 和 canonical fingerprint。
- **Ensemble（v4 引用式）**：顶层 `model_ensemble/` 包唯一承载融合（specs/loader/oof/cache/artifacts/trainer/predictor/runtime/methods）；`load_yaml_config()` 按互斥字段分派 `ForecastConfigSpec | EnsembleConfigSpec`；成员必须是现役单模型 YAML（Direct 复用现役、Recursive 独立 YAML 在 `ensemble_members/`），四方法 `averaging/weighted/linear_blending/stacking`，OOF 按文件内容 SHA-256 fingerprint 缓存于 `results/_ensemble_oof/`，calendar_month 仅 `fold_count=1`。ensemble bundle `model_type="ensemble"`、top-level strategy_spec/estimator_spec 均 None。旧 `forecasting/ensemble.py`、`models/ModelEnsemble.py` 已删除，不得恢复。
- **CQR 边界**：canonical 当前只覆盖 point 与边际 quantile。迁移配置可保留 `probabilistic.conformal` 的历史意图，但 canonical runtime 不执行 CQR，也不输出 `predict_pi<coverage>_lower/upper`。
- **历史迁移记录**：18 个旧 MDR 配置曾因 `H % B != 0` 映射为 `recursive`（14 个 `aidc_power_month`、4 个 `aidc_load_month`，2026-08-29 批准）；迁移溯源字段 `validation.legacy_origin` 已随 legacy 清理从全部 YAML 删除，批准记录仅存于 `docs/multistep_forecasting_redesign.md`。
- **canonical 单运行时**：`main.py`、`run.py` 只接受 `ForecastConfigSpec`。旧 `Model/Trainer/Tester/Forecaster`、`models/multistep/`（含 executors 与只读语义层）和 legacy config generator 已全部删除（2026-08-29 收口），门禁测试断言相关目录不存在。
- **无 legacy 层**：旧 YAML/旧 pkl/旧宽表在本仓库不再可读；`ModelSaveLoad.load_model()` 直接返回加载对象，`CanonicalResultReader` 只接受 canonical long schema，非 canonical 输入直接 RAISE。

### 模型工厂（ModelFactory）
- **支持的模型类型**：LightGBM（`lgb`/`lightgbm`）、XGBoost（`xgb`/`xgboost`）、CatBoost（`cat`/`catboost`）、RandomForest（`rf`/`randomforest`）、HistGradientBoosting（`histgb`/`histgradientboosting`）、Ridge（`ridge`）、ElasticNet（`enet`/`elasticnet`）、Lasso（`lasso`）、QuantileRegressor（`qr`/`quantileregressor`）、SeasonalTemplate（`st`/`seasonaltemplate`，工作日/周末分组建模板 + NNLS 学权重）
- **融合训练数据一致性**：所有融合方法（averaging/weighted/blending/stacking）的基模型最终都在全量训练数据上重训，与单模型基线使用一致训练数据；`averaging` 此前只在 80% 数据上训练属于结构性缺陷，已修复
- **自定义外生特征注册表**（`custom_features`，`ExogenousFeatureConfig`）：多文件来源通路，每项 `{name, history_path, future_path, future_strategy, availability, ts_col, columns, categorical_columns}`。`future_strategy=freeze_last_observation + availability=end_of_period` 表示当期结束后可得的预测原点状态：Direct/DirRec 训练保留原点值且禁止 horizon shift；Recursive/Pointwise 训练按 1 个 freq 步向后对齐（行 t 使用上一期状态）；CV/final future 取 cutoff 前最后状态冻结到全 horizon。典型用途：完整日负荷状态；不得把目标日真实状态展开为 future 外生。
- **测试可视化叠加**（`plot_overlay_path`/`plot_overlay_col`，`OutputConfig`）：在测试图（test_prediction.png / window_plots）上以次坐标轴叠加一条参考曲线（如 PCS 功率），量级差异大时不压扁主曲线

### AIDC 负荷事件标签工具链（2026-08 新增）
- **共享检测核心** `data_process/load_event_detection.py`（有单测 `tests/test_load_event_detection.py`）：事件分类 shift_up/down（持久阶跃=集中上下架）、stress_up/down（1~21 天临时偏移=压测/临时操作）、burst_up/down（1.25h~24h 日内冲击）、spike_up/down（≤1h 功率突变）；三个探测器（自顶向下日级分段 + 短时偏移 + 15min 残差 MAD 突变）+ 边界伪影抑制 + 事件→逐点/逐日投影
- **场景入口**：`config/aidc_load_15min_daily/load_event_analysis.py`（15min 逐点标签 + 特征）、`config/aidc_load_month/load_event_analysis.py`（日频逐日标签 + 特征）；产物在 `dataset/<场景>/event_label_features/`（labeled_features.csv / events.csv / overview.png / report.md）。两频率事件明细完全一致（同一检测核心、日水平基线均取 15min 聚合逐日中位数）
- **列名前缀约定**：`feat_`=本频率 trailing 特征（无泄漏，可建模）、`xf_`=跨频率特征（15min↔日频互取）、`xr_`=跨 route 特征、`lbl_`=事件标签（检测含居中窗口=有未来信息，仅供离线分析/样本筛选，禁止直接作在线预测特征）
- 注意 `dataset/aidc_load_month/` 目录名沿用场景名，文件实际是 **1day 粒度**

### 低频（日/周/月）数据配置约定
- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配
- **月频（`1ME`/`1MS`）已支持**（`utils/frequency.is_monthly_freq`）：`resolve_freq_step_minutes` 近似 30 天、`resolve_samples_per_day` 特判返回 1；`main.py` 月频分界点 = `(now_time.to_period("M")+1).to_timestamp()` 推到下月月初（月频下 `history_length` 语义变为月数、`future_time` 用 `DateOffset(months=horizon)`）；`data_aggregate._freq_to_timedelta` 月频用 31 天上界做 target≥source 校验
- **月度/日度气象严格信息集**（aidc_power_month）：现役载体是 canonical 通用机制——`DataSpec` 列角色 + `SourceRegistry` as-of 校验（`available_at <= forecast_origin`，目标时间戳缺失直接 RAISE）；legacy 字段 `strict_weather_information_set`、`weather_backtest_path`、`weather_*_source` 已废弃且无运行时语义（2026-08-29 收敛方案核实：现役 848 YAML 零处使用）。语义要求不变：历史训练天气为 actual；滑窗测试必须消费独立 ex-ante forecast/proxy 文件 source；正式 future 只允许 forecast|proxy；禁止把测试月实测天气当未来天气。2026-08 日频 future 使用 2025-08 同日纯代理（31 天），不含 2026-08 实测；回测 2026-01~07 使用上一年同日代理（212 天）。
- **Direct 目标日外生对齐**：USMD/USMDR 天气配置启用 `use_horizon_exogenous_for_direct: true`，`col_h(h)=col(t+h)`；horizon-feature 训练 melt 按 h 从 `*_h{h}` 折叠回基础列名，lag/rolling/diff 保持预测原点值，推理端按未来日取外生。USBR 的 Direct/Recursive 共享 X 暂不支持两套外生时点，天气组保留为无天气 control。USMDP 默认不生成目标 lag；只有显式启用 safe-lag 且满足 `min(lags) >= horizon` 才生成。需要逐步消费自身预测时使用 USMR
- **calendar_month 训练窗口**：`horizon_mode=calendar_month` 时实际训练天数由 `train_window_length` 决定，最终 horizon 自动取目标月 28/29/30/31 天；setting 编码真实 `train_window_length` 并追加 `-calendar-month`。`window_length`/`predict_steps` 是遗留兼容字段，不作为自然月训练长度/最终跨度解释。固定步长模式仍按 `window_length × n_per_day − horizon > max(lags)` 校验。
- **history_length > window_length**：`main.py` 硬校验，否则 RAISE
- **datetime 派生剔除子日粒度**：低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项

### Canonical 运行边界
- canonical rolling backtest、final training 和 forecast 均由 `model_forecasting/runtime.py` 编排；不得重新接入已删除的 `ModelTesting` 或旧 executor。
- estimator 并行能力必须由 `model_training/estimators/capabilities.py` 显式探测；不支持时 RAISE，不静默降级。

### 包间分层规则（2026-08-29 架构收敛 P0 立规，详见 docs/architecture_convergence_plan.md §3.1）
- **分层结构（2026-08-30 流水线阶段重排）**：L4 入口（`main.py`/`run.py`/`config/config_loader.py`）→ L3 能力扩展（`probabilistic/`、`model_ensemble/`）→ L2 编排（`model_forecasting/runtime.py` 唯一编排器 + `forecaster.py` 推理执行 + 核心合同 `model_forecasting/{specs,tensors,transforms,results}`）→ 阶段顶层包 `data_loading/`（数据构造与组织）→ `feature_engineering/`（特征工程）→ `model_training/`（训练）→ `model_testing/`（测试/回测），`model_evaluation/`（独立评估模块，只做指标计算）→ L1 基础设施（`models/`、`decomposition/`、`data_process/`）→ L0 基础（`utils/`）。
- **依赖只允许从上往下**（高层 import 低层），同层禁止互依；唯一豁免：L3 内 `ensemble → probabilistic` 单向（数据合同 types + 评估实现 evaluation 复用，不得触及 pipeline/training 执行面）。低层不得反向 import 高层，**包括函数内延迟 import**。
- 跨包复用的算法放 L1 并以**公开函数**暴露；**禁止下划线私有跨包导入**（ensemble 对 forecasting 的两处私有导入已随收敛 P3 修复为 `model_forecasting.backtest` 公开 API）。
- 任何循环依赖视为架构缺陷，出现即修；不允许用函数内延迟 import 长期绕开。`models/ModelTraining.py`、`models/ModelForecasting.py` 兼容 shim 已于 2026-08-30 删除，Trainer/Forecaster 直接从 `model_training/trainer.py`、`model_forecasting/forecaster.py` 导入；分层边界由 `tests/test_package_layering.py` 门禁固化（白名单制 12 项全包扫描，函数内 import 同罪）。
- 新增能力包一律按 ensemble 模式建：独立包 + Protocol 边界 + 单向依赖 + parity golden。ensemble 已知边界：`model_ensemble/loader.py` 直接操作 canonical YAML payload（成员身份锁定为现役单模型 YAML，属有意设计）。
- **流水线数据契约**：进入模型的信息集默认「缺失/异常 = RAISE」（预测性维护数据前置）；填补与清洗只允许发生在离线数据准备阶段（`data_process/`，场景维度）或以 config 驱动的 compiler 前置步骤引入（须满足 as-of 可得性）；训练窗口内清洗（改数据）与评估掩码（只改评估口径）严格分离。

### 关键约定
- canonical 输出目录：`results/<scenario>/<result_identity>/{pretrained_models,results_test,results_forecast}`；`result_identity=<strategy>-<model>-<scope>-k<K>-<fingerprint[:12]>`（ensemble 产物 identity 由 `model_ensemble/` runtime 管理）。
- `prediction.csv` / `cv_plot_df.csv` 使用 `(series_id,time,target[,window])` long 唯一键；`test_scores_df.csv` 写 per-target 与 aggregate 行；**quantile 模式追加 `test_scores_probabilistic_df.csv`**（tidy long：mae/逐 level pinball/central 区间覆盖率·宽度·winkler·coverage gap，eval_mask 口径与点评估共用 `build_eval_mask_payload`，aggregate 行跨 target 按有效点池化）。ensemble 的无泄漏质量证据是**融合 OOF 评分**（`model_ensemble/evaluation.py`，复用同一评估实现）：结果挂在 `run_ensemble_config` 返回的 `fused_oof_scores` / audit，run.py 日志输出 aggregate MAE/RMSE/MAPE。
- 目标训练变换顺序固定为 `calendar normalization → decomposition → target scaling`，point/quantile 严格逆序恢复；状态按 `(series_id,target)` 隔离并随 bundle 保存。
- 本地入口只接受 canonical schema YAML：修改 `main.py::CONFIG_YAML` 后运行 `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py`。非 canonical YAML 一律 RAISE。
- `run.py` 只对 canonical payload 应用 CLI override；非 canonical 配置明确 RAISE（不再指向已删除的迁移脚本）。

- 日志目录：`logs/main/`（直接运行 `main.py`）
- `log_util.py` 在模块导入时执行，`LOG_NAME` 用 `os.environ.get('LOG_NAME', 'main')` 提供默认值
- 在这台机器上如果 `uv` 触发 `~/.cache/uv` 权限问题，优先使用 `UV_CACHE_DIR=.uv_cache`
- 算力房间数据：`df.csv`（215 列，全量合并）、`df_selected.csv`（多变量筛后建模数据）、`feature_selection_report.csv`（筛列报告）和稠密版 `df_filled.csv`（`config/aidc_electricity_computility/electricity/2026-06-11/scripts/fill_missing.py`：原始列补 0 + 重算派生）都在 `demand_load/<room>/` 下；算力预处理权威入口是 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py`，特征分原始聚合、状态/结构、时序动态三层；单变量配置继续使用 `df_power.csv`，4 个房间场景的多变量 LGBM 配置使用 `df_selected.csv`
- **2026-08-10 算力场景**（A2_IT / A3_IT / liantong_IT / yancheng_IT，各 7 个 YAML）：`dataset/.../2026-08-10/` 下房间目录当前为空，YAML 的 `data_dir` 指向 2026-06-11 数据（复用上批数据做配置模板）

### 仓库维护注意
- **`tests/` 已纳入版本控制**（2026-08-15 起移出 `.gitignore`）：改动被测模块时必须同步修复测试导入与接口断言，并运行 `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run python -m unittest discover -s tests -p "test_*.py"`；预测架构的最新精确计数与命令只在 `docs/multistep_forecasting_redesign.md` 维护。AIDC 专项脚本目录路径含日期段、不是合法 Python 包，相关测试用 `sys.path.insert` 引导后按模块名导入。
- `docs/feature_engineering/`（fft/wavelet/statistical/holiday/periodicity/weather 等）是实验性特征脚本集合，**未接入主流程**，2026-08-29 从 `features/feature_engineering/` 移入 `docs/` 归档，后续再决定处置
- `utils/`（L0）只保留 frequency/log_util/runtime_env 三个无领域语义工具；`utils/metrics.py`、`utils/cv_plot.py`、`utils/eval_mask.py` 已删除（指标在 `model_forecasting/results.py` + `probabilistic/metrics.py`，评估掩码在 `model_forecasting/backtest.py::build_eval_mask`）
