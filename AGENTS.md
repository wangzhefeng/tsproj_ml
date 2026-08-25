# AGENTS.md — tsproj_ml

Project-specific conventions shared by Claude Code and Codex CLI. `AGENTS.md` is the single source of truth; `CLAUDE.md` contains only an `@AGENTS.md` reference that imports this file. Edit here and both agents stay in sync — do not add project conventions to `CLAUDE.md`.

The general coding guidelines (Karpathy: think before coding, simplicity, surgical changes, goal-driven execution) live in the user-global config — `~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for Codex CLI — and are intentionally not duplicated here.

## Project Conventions

### 配置系统
- YAML 配置：`base_config` 指向 Python 模块（默认 `config.univariate_config`），`overrides` 按 `config_sections.py` 分组写一层嵌套（兼容旧平铺写法；两层以上嵌套 RAISE，未知字段 RAISE，避免拼写错误静默失效）
- 加载器无"值回退"机制（`apply_config_overrides` 直接 `setattr`，布尔值原样生效）。仍需注意两个 YAML 层面的坑：同一文件内重复键时 `yaml.safe_load` 静默 last-wins；值写 `null` 会把字段置为 `None`（对布尔字段等效 false）

### 时间边界约定
- `now_time` 配置值 = 最后一个已知数据点（如 `2026-06-11T23:55:00`）
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。日志/文件名的时间戳仍按 `now_time` 原值
- 内部分界点 = `schedule_mode=daily` 时 `floor("1D") + 1day` = 次日 00:00:00；`intraday` 时 = `now_time` 本身
- `start_time = now_time - history_length`，`future_time = now_time + predict_steps × freq 步长`
- **`predict_steps` 以 `freq` 为单位计步**（取代旧 `predict_days`）：15min 下 1 天 = 96、4 小时 = 16；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不再经 `n_per_day` 换算
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### 预测方法限制
- **多步策略单一事实源**：九种外部 `pred_method` 全名、短码和描述只在 `models/multistep/spec.py` 的 `STRATEGY_SPECS` 声明；运行期先解析为 `ResolvedStrategy`（`TargetPlan` / `TrainingPlan` / `FeaturePlan` / `RuntimePlan`），再由 Pointwise / Direct / Recursive / DirRec / Blend 五类 executor 执行。下游不得新增原始方法字符串目录或九方法代理分支
- **DirRec 块长契约**：USMDR/MSMDR 的有效块长 `B = min(horizon, block_size>0 ? block_size : min(lags))`，训练目标、模型输出宽度和推理推进统一使用 B；旧 H 宽模型仅由 `LegacyArtifactAdapter` 只读适配。受修复影响的存量结果见 `docs/multistep_forecasting_invalidation.md`
- **模型产物契约**：新模型统一保存 typed `ForecastModelBundle`，多步组合内部使用 `StrategyArtifact` / `BlendArtifact` / `AuxiliaryEndogenousArtifact`；旧裸估计器和 dict bundle 只在 `models/multistep/artifacts.py` 集中适配，不原地改写旧 pkl
- **Global panel**：`enable_global_training=true` 时历史/未来主键为 `(series_id, time)`，lag/shift/递归状态和输出均按 series 隔离，每个 series 必须恰好 H 行。当前支持 `global_incomplete_series_policy: raise|drop`，未知 series 只允许 `raise`；暂不与 calendar normalization、目标分解、auxiliary backfill、calendar_month 组合
- **USMDP + advanced_features 不兼容**：`add_rolling_statistics` 和 `add_diff_features` 依赖目标列 `y`，训练时存在但预测时（未来数据）不存在 → LightGBM Fatal: feature count mismatch
- 默认不得直接操作 `y`；可改为操作外生特征或已生成的 safe-lag 列。USMDP safe-lag 要求 `align_direct_features_to_target:true`、启用 lag 且 `min(lags) >= horizon`，只读取已知历史、不递归消费未来目标
- **时间衰减样本权重**（`enable_time_decay_sample_weight`）：默认 `multi_output_strategy: multioutput` 下 point/quantile 均生效——多输出统一走项目自定义 `DirectMultiOutputRegressor`，把同一 `sample_weight` 逐输出转发给基估计器；单输出 quantile 直接透传。仅 `regressor_chain` 与 ensemble 路径不支持，运行时 WARNING 安全跳过、不阻塞
- **datetime_features 有两对同义项**：`feature_map`（`features/FeatureEngineering.py`）中 `weekday` ≡ `day_of_week`（同 `.dt.weekday`）、`week` ≡ `week_of_year`（同 `.dt.isocalendar().week`），配置中只应保留后者（pandas 规范名）

### 预测增强策略（v1 新增，默认全关）
- **`direct_strategy: horizon_feature`**（仅 USMD/MSMD）：把 horizon 索引作为特征，训练 N×H 行长表（成本 H×→1×），推理展开 H 行，外生列按步取值解决 MIMO 滞后陈旧。DirRec（USMDR/MSMDR）暂不支持（遗留）
- **`endogenous_backfill_strategy: auxiliary`**（仅 MSMR/MSMDR）：为每个非目标内生变量训练独立递归模型（reduced-form：只用自身滞后+datetime 外生），替代持久性常量回填。辅助模型在 `models/AuxiliaryForecaster.py`，与主模型一起保存为 typed `AuxiliaryEndogenousArtifact`。注意：辅助模型每滑窗×每内生列各训练一次（成本 = 窗口数 × 内生列数），窗口并行时各窗口独立训练不共享
- **`pred_method: USBR/MSBR`**（Direct+Recursive Blend）：同时构造 Direct（shift_1..H）和 Recursive（shift_0）目标列，训练两个子模型后加权融合。`blend_weight_strategy: ridge_stacking` 在最近 N 个滑窗（`blend_weight_windows`）测试集上用 `Ridge(positive=True, fit_intercept=False)` 学凸组合权重；`blend_weights.csv` 保留审计视图，实际推理权重随 `BlendArtifact` 保存。**blend×quantile 已支持**：每个分位数训练 Direct（多输出 quantile）+ Recursive（单输出 quantile）子模型对，推理时逐分位数做 Direct+Recursive 加权融合，所有分位数共用同一组 ridge_stacking 权重（权重学自 median 分位数的 Direct/Recursive 分量）。当前硬性不支持（main.py 校验 raise）：blend×ensemble、blend×scale_target、blend×目标分解（`decomposition_method != none`）
- **概率预测统一契约**：运行期以 `ProbabilisticSpec` 为唯一语义来源（`overrides.probabilistic` canonical mapping 与 legacy 字段归一化，冲突 RAISE）。滑窗在 q50 锚定 crossing repair 后对 spec 指定的 quantile pair 记录 `conformal_score`；历史窗口 prequential 评价与 final forecast 共用按 `forecast_origin + label_available_at` 的 as-of 完整窗口 selector（不再按 window ID 取数）。模型分位数始终保留在 `predict_q*`；CQR 只新增 `predict_pi<coverage>_lower/upper`，不得覆盖 q10/q90。默认负 correction 截为 0（只扩不缩），canonical spec 可显式允许 shrink；score 少于 `conformal_min_scores`、完整窗口少于 `conformal_min_windows` 时 WARNING 跳过。时间序列只报告经验 coverage，不宣称无条件有限样本保证。CQR 可与目标分解/target scaling 组合（统一目标空间栈）。

### 模型工厂（ModelFactory）
- **支持的模型类型**：LightGBM（`lgb`/`lightgbm`）、XGBoost（`xgb`/`xgboost`）、CatBoost（`cat`/`catboost`）、RandomForest（`rf`/`randomforest`）、HistGradientBoosting（`histgb`/`histgradientboosting`）、Ridge（`ridge`）、ElasticNet（`enet`/`elasticnet`）、Lasso（`lasso`）、QuantileRegressor（`qr`/`quantileregressor`）、SeasonalTemplate（`st`/`seasonaltemplate`，工作日/周末分组建模板 + NNLS 学权重）
- **融合成员级规格**（`ensemble_model_specs`，非空时取代 `ensemble_models`）：每项 `{model, params?, scale?, impute?}`——`scale` 为该成员独立特征标准化（线性成员需要）、`impute` 为中位数填补（训练窗起始行长滞后特征为 NaN，GBDT 原生容忍，线性/KNN 等成员需要）。`ModelEnsemble.TimeSeriesEnsembleRegressor` 内部成员统一为三元组 `(name, model, preprocessor)`
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
- **月度/日度气象严格信息集**（aidc_power_month）：历史训练天气为 actual；滑窗测试必须使用独立 `weather_backtest_path` 的 ex-ante forecast/proxy；正式 future 只允许 `forecast|proxy`。启用天气的配置设 `strict_weather_information_set: true`、来源字段和 `available_at`，backtest 必须早于目标月、future 不晚于 `now_time`，目标时间戳缺失直接 RAISE。2026-08 日频 future 使用 2025-08 同日纯代理（31 天），不含 2026-08 实测；回测 2026-01~07 使用上一年同日代理（212 天）。禁止把测试月实测天气当未来天气。
- **Direct 目标日外生对齐**：USMD/USMDR 天气配置启用 `use_horizon_exogenous_for_direct: true`，`col_h(h)=col(t+h)`；horizon-feature 训练 melt 按 h 从 `*_h{h}` 折叠回基础列名，lag/rolling/diff 保持预测原点值，推理端按未来日取外生。USBR 的 Direct/Recursive 共享 X 暂不支持两套外生时点，天气组保留为无天气 control。USMDP 默认不生成目标 lag；只有显式启用 safe-lag 且满足 `min(lags) >= horizon` 才生成。需要逐步消费自身预测时使用 USMR
- **calendar_month 训练窗口**：`horizon_mode=calendar_month` 时实际训练天数由 `train_window_length` 决定，最终 horizon 自动取目标月 28/29/30/31 天；setting 编码真实 `train_window_length` 并追加 `-calendar-month`。`window_length`/`predict_steps` 是遗留兼容字段，不作为自然月训练长度/最终跨度解释。固定步长模式仍按 `window_length × n_per_day − horizon > max(lags)` 校验。
- **history_length > window_length**：`main.py` 硬校验，否则 RAISE
- **datetime 派生剔除子日粒度**：低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项

### 性能并行约定（PerformanceConfig）
- **窗口并行与内部并行不叠加**：`window_parallel_workers > 1` 时，`utils/parallel_budget.py` 会把每个窗口 payload 的 `multi_output_n_jobs`、`quantile_parallel_workers`、`model_thread_count`、`ensemble_parallel_workers` 全部改写为 1；故测试阶段总并行度≈`window_parallel_workers`。内部参数只在最终 forecast 的单模型上生效（forecast 用原始 `args`）——所以 `is_testing=true` 的场景可同时设高窗口并行与内部并行而不超额订阅。HistGB 走 OpenMP 无 `n_jobs` 构造参数，窗口并行时额外设 `OMP_NUM_THREADS=1` 兜底（`main.py` 给窗口 payload 注入 `force_single_thread_env` flag，`ModelTesting.Tester._window_test` 据此设环境变量）。
- **方法→有效杠杆**：多输出方法（USMD/USMDR/MSMD/MSMDR，走 `MultiOutputRegressor`/`DirectMultiOutputRegressor`）调 `multi_output_n_jobs`；单输出方法（USMDP/USMR/MSMR）只 `model_thread_count` 有效。融合方法调 `ensemble_parallel_workers`。
- **模型线程参数差异**：LightGBM/XGBoost/CatBoost/RandomForest 有 `n_jobs`/`thread_count`；HistGB 无（OpenMP 底层）；线性模型（Ridge/ElasticNet/Lasso）单线程/BLAS。`Trainer._thread_limited_params` 按类型注线程参数，无参数可注的原样返回。
- 容量上限：`multi_output_n_jobs × model_thread_count ≤ 物理核数`，优先「多并行单线程」。本机 M2 共 8 核（4 性能 + 4 能效）。

### 关键约定
- 输出目录：`results/`（不是 `saved_results/`）；三个子目录（pretrained_models/results_test/results_forecast）按 `<scenario>/<setting>` 嵌套，`<scenario>` 优先取 YAML `overrides.output.scenario_subpath`（与 config 目录对齐），未设时由 `data_dir` 去掉 `dataset/`/`demand_load` 段自动推导；多组配置共用同一 `data_dir` 时必须显式指定 `scenario_subpath`，否则结果混入同一目录
- 测试汇总指标：使用 median（中位数），不是 mean——单窗口 MAPE 爆炸会拖垮均值（在 `main.py` 汇总时追加「中位数」行）
- `MAPE Accuracy` 业务口径：按 `eval_mask` 掩码过滤后计算（默认 `mode: percentile` 即窗口正样本 `P5`；`absolute`/`combined` 模式启用 `min_value` 绝对下限，`max_value` 上限与 mode 正交、三模式均生效）；阈值落 `test_scores_df.csv` 的 `MAPE Threshold`/`MAPE Upper Threshold` 列，无有效点写 `NaN`；每窗口附带季节 naive（昨日同时刻）对照列 `Naive MAPE`/`Naive MAPE Accuracy`，与模型共用同一 eval_mask
- `prediction.png` 历史上下文绘制**原始 `y`**（保证线条连续，不再断线）；eval_mask 掩码结果保留在 `prediction_plot_concat.csv` 的 `plot_value`/`plot_valid` 列；未来预测原值在 `prediction.csv` 的 `predict_value`。（注：`test_prediction.png` 滑窗测试图仍按 eval_mask 断线显示无效点）
- 目标训练变换统一登记到 `TargetTransformPipeline`：训练顺序 `calendar normalization → decomposition → target scaling`，point/quantile 恢复严格逆序；decomposition 可与 scale_target、quantile、CQR 组合。Blend 与目标分解仍因分量空间未统一而 fail-fast
- `n_windows <= 0`（滑窗数为 0）不 RAISE，仅 WARNING 并跳过测试
- **消融矩阵的 no-op control**：矩阵中不消费目标特征的模型保留为对照组——SeasonalTemplate（`st`）不消费 custom/load-state/weather 特征、USBR 不接入天气特征；此类配置删除特征注入块，注释与 `setting_suffix` 显式标注 `-control-no-<feature>`（如 `-decomp-linear-control-no-state`、`-conformal-control-no-weather`），保证与实验组同目录可对照
- **baseline/weather-date 分组**：`aidc_load_15min_{daily,rolling,short}` 的 baseline 保留 USBR，且把“无模型融合”解释为 `enable_ensemble: false`，不把 USBR 策略视为 ModelEnsemble；baseline 显式关闭 datetime/date_type/weather/custom，只消费目标序列派生特征。`add_exogenous_weather_date` 不使用三个场景不存在的 date_type，仅在可消费外生变量的 baseline 子集上开启 datetime+weather；USBR 与 SeasonalTemplate 不进入该组，避免 no-op control 混入实验组。rolling/short 必须使用 `schedule_mode: intraday` + `setting_suffix: -intraday`，严格天气角色按预测原点切分：history 截止 `2026-07-31 13:45`，future 从 `2026-07-31 14:00` 开始；daily 仍按日界切分为 07-31 23:45 / 08-01 00:00
- **ESS self-use baseline/weather-date**：`aidc_ess_selfuse_load` baseline 同样关闭 datetime/date/weather/custom、分解和 ensemble，只保留目标自身派生特征与 USBR；weather-date 可使用该场景真实存在的 date_type，并增加 datetime+strict native weather，但 `decomposition_method` 必须保持 `none`，避免外生消融与目标分解混杂
- `EvalMaskConfig`（评估/绘图掩码，不改数据）≠ `TrainOutlierConfig`（滑窗训练窗口清洗，插值改数据）
- 本地默认入口：修改 `main.py` 中的 `CONFIG_YAML`（当前指向 `config/aidc_load_month/route_B/lgbm_usmd_prob_mean.yaml`），再直接运行 `uv run python main.py`
- `run.py` 保留为兼容入口，但当前文档和脚本不再把它作为推荐运行方式
- 日志目录：`logs/main/`（直接运行 `main.py`）
- `log_util.py` 在模块导入时执行，`LOG_NAME` 用 `os.environ.get('LOG_NAME', 'main')` 提供默认值
- 在这台机器上如果 `uv` 触发 `~/.cache/uv` 权限问题，优先使用 `UV_CACHE_DIR=.uv_cache`
- 算力房间数据：`df.csv`（215 列，全量合并）、`df_selected.csv`（多变量筛后建模数据）、`feature_selection_report.csv`（筛列报告）和稠密版 `df_filled.csv`（`config/aidc_electricity_computility/electricity/2026-06-11/scripts/fill_missing.py`：原始列补 0 + 重算派生）都在 `demand_load/<room>/` 下；算力预处理权威入口是 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py`，特征分原始聚合、状态/结构、时序动态三层；单变量配置继续使用 `df_power.csv`，4 个房间场景的多变量 LGBM 配置使用 `df_selected.csv`
- **2026-08-10 算力场景**（A2_IT / A3_IT / liantong_IT / yancheng_IT，各 7 个 YAML）：`dataset/.../2026-08-10/` 下房间目录当前为空，YAML 的 `data_dir` 指向 2026-06-11 数据（复用上批数据做配置模板）

### 仓库维护注意
- **`tests/` 已纳入版本控制**（2026-08-15 起移出 `.gitignore`）：改动被测模块（尤其是移动/重命名模块，如 AIDC 脚本目录）时必须同步修复测试导入与接口断言，并运行 `uv run python -m unittest discover -s tests -p "test_*.py"`（全套约 12~16s）。AIDC 专项脚本目录路径含日期段、不是合法 Python 包，相关测试用 `sys.path.insert` 引导后按模块名导入
- `features/feature_engineering/`（fft/wavelet/statistical/holiday/periodicity/weather 等）是实验性特征子包，**未接入主流程**（主流程只用 `features/FeatureEngineering.py`），修改它不影响模型运行
- `utils/metrics.py`、`utils/cv_plot.py` 是独立工具函数库，当前主流程未 import（主流程指标/绘图逻辑内联在 `models/` 中）
