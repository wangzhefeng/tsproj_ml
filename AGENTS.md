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
- `start_time = now_time - history_days`，`future_time = now_time + predict_steps × freq 步长`
- **`predict_steps` 以 `freq` 为单位计步**（取代旧 `predict_days`）：15min 下 1 天 = 96、4 小时 = 16；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不再经 `n_per_day` 换算
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### 预测方法限制
- **USMDP + advanced_features 不兼容**：`add_rolling_statistics` 和 `add_diff_features` 依赖目标列 `y`，训练时存在但预测时（未来数据）不存在 → LightGBM Fatal: feature count mismatch
- 可通过将操作列改为外生特征列来启用，但默认配置下必须禁用
- **时间衰减样本权重**（`enable_time_decay_sample_weight`）：默认路径（`multi_output_strategy: multioutput` + `predict_type: point` + 非融合）下所有 9 种预测方法均生效——多输出方法走项目自定义 `DirectMultiOutputRegressor`，其 `fit` 会把 `sample_weight` 逐输出转发给基估计器（`ModelTraining._prepare_multi_output_fit_data`）。仅在 `regressor_chain`、多输出分位数、ensemble 路径不支持（sklearn `MultiOutputRegressor`/`RegressorChain` 的 `fit` 不透传 `sample_weight`），运行时 WARNING 安全跳过、不阻塞
- **datetime_features 有两对同义项**：`feature_map`（`features/FeatureEngineering.py`）中 `weekday` ≡ `day_of_week`（同 `.dt.weekday`）、`week` ≡ `week_of_year`（同 `.dt.isocalendar().week`），配置中只应保留后者（pandas 规范名）

### 预测增强策略（v1 新增，默认全关）
- **`direct_strategy: horizon_feature`**（仅 USMD/MSMD）：把 horizon 索引作为特征，训练 N×H 行长表（成本 H×→1×），推理展开 H 行，外生列按步取值解决 MIMO 滞后陈旧。DirRec（USMDR/MSMDR）暂不支持（遗留）
- **`endogenous_backfill_strategy: auxiliary`**（仅 MSMR/MSMDR）：为每个非目标内生变量训练独立递归模型（reduced-form：只用自身滞后+datetime 外生），替代持久性常量回填。辅助模型在 `models/AuxiliaryForecaster.py`，与主模型打包为 `{"bundle_type": "auxiliary_endogenous", ...}` dict。注意：辅助模型每滑窗×每内生列各训练一次（成本 = 窗口数 × 内生列数），窗口并行时各窗口独立训练不共享
- **`pred_method: USBR/MSBR`**（Direct+Recursive Blend）：同时构造 Direct（shift_1..H）和 Recursive（shift_0）目标列，训练两个子模型后加权融合。`blend_weight_strategy: ridge_stacking` 在最近 N 个滑窗（`blend_weight_windows`）测试集上用 `Ridge(positive=True, fit_intercept=False)` 学凸组合权重，写入 `blend_weights.csv`。v1 硬性不支持（main.py 校验 raise）：blend×quantile、blend×ensemble、blend×scale_target、blend×detrend_target
- **`enable_conformal_calibration: true`**（CQR 后处理）：滑窗测试阶段记录 `conformal_score` 到 `cv_plot_df.csv`，forecast 阶段取最近 `conformal_calibration_windows`（默认 5）个窗口的 score 作校准集，对 `predict_q*` 首尾边界列对称膨胀，保证边际覆盖率 ≥ 1−α。纯后处理不重训模型，依赖 `is_testing=true` 产出的校准集；校准集 score 点数 < `conformal_min_scores`（默认 30）则跳过并 WARNING。v1 硬性不支持 conformal×detrend_target（main.py 校验 raise）

### 模型工厂（ModelFactory）
- **支持的模型类型**：LightGBM（`lgb`/`lightgbm`）、XGBoost（`xgb`/`xgboost`）、CatBoost（`cat`/`catboost`）、RandomForest（`rf`/`randomforest`）、HistGradientBoosting（`histgb`/`histgradientboosting`）、Ridge（`ridge`）、ElasticNet（`enet`/`elasticnet`）、Lasso（`lasso`）、QuantileRegressor（`qr`/`quantileregressor`）、SeasonalTemplate（`st`/`seasonaltemplate`，工作日/周末分组建模板 + NNLS 学权重）
- **融合成员级规格**（`ensemble_model_specs`，非空时取代 `ensemble_models`）：每项 `{model, params?, scale?, impute?}`——`scale` 为该成员独立特征标准化（线性成员需要）、`impute` 为中位数填补（训练窗起始行长滞后特征为 NaN，GBDT 原生容忍，线性/KNN 等成员需要）。`ModelEnsemble.TimeSeriesEnsembleRegressor` 内部成员统一为三元组 `(name, model, preprocessor)`
- **融合训练数据一致性**：所有融合方法（averaging/weighted/blending/stacking）的基模型最终都在全量训练数据上重训，与单模型基线使用一致训练数据；`averaging` 此前只在 80% 数据上训练属于结构性缺陷，已修复
- **自定义外生特征注册表**（`custom_features`，`ExogenousFeatureConfig`）：多文件来源的通用外生特征通路，每项 `{name, history_path, future_path, ts_col, columns, categorical_columns}`，按精确时间戳 merge（与 weather 同机制），无 `rt_/pred_` 列名映射（历史/未来列名一致）。典型用途：计划类外生特征
- **测试可视化叠加**（`plot_overlay_path`/`plot_overlay_col`，`OutputConfig`）：在测试图（test_prediction.png / window_plots）上以次坐标轴叠加一条参考曲线（如 PCS 功率），量级差异大时不压扁主曲线

### 低频（日/周）数据配置约定
- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配
- **window_days 在低频下 ≈ 每窗训练样本数**：`main.py` 滞后校验要求 `window_days × n_per_day − horizon > max(lags)`，否则 RAISE（"Lag features would be all-NaN"；USMDP 逐点法不构造多步目标列，豁免该校验）；日频即 `window_days > max(lags) + predict_steps`
- **history_days > window_days**：`main.py` 硬校验，否则 RAISE
- **datetime 派生剔除子日粒度**：低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项

### 性能并行约定（PerformanceConfig）
- **窗口并行与内部并行不叠加**：`window_parallel_workers > 1` 时，滑窗测试会把每个窗口 payload 的 `multi_output_n_jobs`、`model_thread_count`、`ensemble_parallel_workers` 强制改写为 1（见 `main.py` 的 `Model.test`）；故测试阶段总并行度≈`window_parallel_workers`。内部参数只在最终 forecast 的单模型上生效（forecast 用原始 `args`）——所以 `is_testing=true` 的场景可同时设高窗口并行与内部并行而不超额订阅。HistGB 走 OpenMP 无 `n_jobs` 构造参数，窗口并行时额外设 `OMP_NUM_THREADS=1` 兜底（`main.py` 给窗口 payload 注入 `force_single_thread_env` flag，`ModelTesting.Tester._window_test` 据此设环境变量）。
- **方法→有效杠杆**：多输出方法（USMD/USMDR/MSMD/MSMDR，走 `MultiOutputRegressor`/`DirectMultiOutputRegressor`）调 `multi_output_n_jobs`；单输出方法（USMDP/USMR/MSMR）只 `model_thread_count` 有效。融合方法调 `ensemble_parallel_workers`。
- **模型线程参数差异**：LightGBM/XGBoost/CatBoost/RandomForest 有 `n_jobs`/`thread_count`；HistGB 无（OpenMP 底层）；线性模型（Ridge/ElasticNet/Lasso）单线程/BLAS。`Trainer._thread_limited_params` 按类型注线程参数，无参数可注的原样返回。
- 容量上限：`multi_output_n_jobs × model_thread_count ≤ 物理核数`，优先「多并行单线程」。本机 M2 共 8 核（4 性能 + 4 能效）。

### 关键约定
- 输出目录：`results/`（不是 `saved_results/`）；三个子目录（pretrained_models/results_test/results_forecast）按 `<scenario>/<setting>` 嵌套，`<scenario>` 优先取 YAML `overrides.output.scenario_subpath`（与 config 目录对齐），未设时由 `data_dir` 去掉 `dataset/`/`demand_load` 段自动推导；多组配置共用同一 `data_dir` 时必须显式指定 `scenario_subpath`，否则结果混入同一目录
- 测试汇总指标：使用 median（中位数），不是 mean——单窗口 MAPE 爆炸会拖垮均值（在 `main.py` 汇总时追加「中位数」行）
- `MAPE Accuracy` 业务口径：按 `eval_mask` 掩码过滤后计算（默认 `mode: percentile` 即窗口正样本 `P5`；`absolute`/`combined` 模式启用 `min_value` 绝对下限，`max_value` 上限与 mode 正交、三模式均生效）；阈值落 `test_scores_df.csv` 的 `MAPE Threshold`/`MAPE Upper Threshold` 列，无有效点写 `NaN`；每窗口附带季节 naive（昨日同时刻）对照列 `Naive MAPE`/`Naive MAPE Accuracy`，与模型共用同一 eval_mask
- `prediction.png` 历史上下文绘制**原始 `y`**（保证线条连续，不再断线）；eval_mask 掩码结果保留在 `prediction_plot_concat.csv` 的 `plot_value`/`plot_valid` 列；未来预测原值在 `prediction.csv` 的 `predict_value`。（注：`test_prediction.png` 滑窗测试图仍按 eval_mask 断线显示无效点）
- `detrend_target` 与 `scale_target` 互斥：两者同开会双重加趋势且逆变换顺序错乱，`main.py` 校验 RAISE
- `n_windows <= 0`（滑窗数为 0）不 RAISE，仅 WARNING 并跳过测试
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
