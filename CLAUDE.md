# CLAUDE.md / AGENTS.md — tsproj_ml

Project-specific conventions shared by Claude Code and Codex CLI. `AGENTS.md` is a symlink to `CLAUDE.md` — this is the single source of truth; edit here and both agents stay in sync.

The general coding guidelines (Karpathy: think before coding, simplicity, surgical changes, goal-driven execution) live in the user-global config — `~/.claude/CLAUDE.md` for Claude Code, `~/.codex/AGENTS.md` for Codex CLI — and are intentionally not duplicated here.

## Project Conventions

### 配置系统
- YAML 配置：`base_config` 指向 Python 模块（默认 `config.univariate_config`），`overrides` 为扁平化键值覆盖
- 修改 YAML 后务必运行验证：配置值可能被回退（已知问题：`true` 会变成 `false`）

### 时间边界约定
- `now_time` 配置值 = 最后一个已知数据点（如 `2026-06-11T23:55:00`）
- 内部分界点 = `floor("1D") + 1day` = 次日 00:00:00
- `start_time = now_time - history_days`，`future_time = now_time + predict_steps × freq 步长`
- **`predict_steps` 以 `freq` 为单位计步**（取代旧 `predict_days`）：15min 下 1 天 = 96、4 小时 = 16；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不再经 `n_per_day` 换算
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——所以终日 23:55 是最后一个被包含的点

### 预测方法限制
- **USMDP + advanced_features 不兼容**：`add_rolling_statistics` 和 `add_diff_features` 依赖目标列 `y`，训练时存在但预测时（未来数据）不存在 → LightGBM Fatal: feature count mismatch
- 可通过将操作列改为外生特征列来启用，但默认配置下必须禁用
- **多输出方法的时间衰减样本权重不生效**：`enable_time_decay_sample_weight: true` 是全场景统一模板（单输出方法 USMDP/USMR 真实生效）。多输出方法（USMD/USMDR）走 sklearn `MultiOutputRegressor`/`DirectMultiOutputRegressor`，其 `fit` 接口不传递 `sample_weight`，运行时安全跳过（日志 WARNING），模型行为不变——这不是配置 bug，是 sklearn API 固有约束
- **datetime_features 有两对同义项**：`feature_map`（`features/FeatureEngineering.py`）中 `weekday` ≡ `day_of_week`（同 `.dt.weekday`）、`week` ≡ `week_of_year`（同 `.dt.isocalendar().week`），配置中只应保留后者（pandas 规范名）

### 低频（日/周）数据配置约定
- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配
- **window_days 在低频下 ≈ 每窗训练样本数**：`main.py` 滞后校验要求 `window_days × n_per_day − horizon > max(lags)`，否则 RAISE（"Lag features would be all-NaN"）；日频即 `window_days > max(lags) + predict_steps`
- **history_days > window_days**：`main.py` 硬校验，否则 RAISE
- **datetime 派生剔除子日粒度**：低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项

### 性能并行约定（PerformanceConfig）
- **窗口并行与内部并行不叠加**：`window_parallel_workers > 1` 时，滑窗测试会把每个窗口 payload 的 `multi_output_n_jobs`、`model_thread_count` 强制改写为 1（见 `main.py` 的 `Model.test`）；故测试阶段总并行度≈`window_parallel_workers`。内部参数只在最终 forecast 的单模型上生效（forecast 用原始 `args`）——所以 `is_testing=true` 的场景可同时设高窗口并行与内部并行而不超额订阅。
- **方法→有效杠杆**：多输出方法（USMD/USMDR/MSMD/MSMDR，走 `MultiOutputRegressor`/`DirectMultiOutputRegressor`）调 `multi_output_n_jobs`；单输出方法（USMDP/USMR/MSMR）只 `model_thread_count` 有效。
- 容量上限：`multi_output_n_jobs × model_thread_count ≤ 物理核数`，优先「多并行单线程」。本机 M2 共 8 核（4 性能 + 4 能效）。

### 关键约定
- 输出目录：`results/`（不是 `saved_results/`）；三个子目录（pretrained_models/results_test/results_forecast）按 `<scenario>/<setting>` 嵌套，`<scenario>` 优先取 YAML `overrides.output.scenario_subpath`（与 config 目录对齐），未设时由 `data_dir` 去掉 `dataset/`/`demand_load` 段自动推导；多组配置共用同一 `data_dir` 时必须显式指定 `scenario_subpath`，否则结果混入同一目录
- 测试汇总指标：使用 median（中位数），不是 mean——单窗口 MAPE 爆炸会拖垮均值
- `MAPE Accuracy` 业务口径：按 `eval_mask` 掩码过滤后计算（默认 `mode: percentile` 即窗口正样本 `P5`；`absolute`/`combined` 模式启用 `min_value` 绝对下限，`max_value` 上限与 mode 正交、三模式均生效）；阈值落 `test_scores_df.csv` 的 `MAPE Threshold`/`MAPE Upper Threshold` 列，无有效点写 `NaN`
- `prediction.png` 按 `eval_mask` 掩码历史上下文异常点（断线）；未来预测原值保留在 `prediction.csv` 和 `prediction_plot_concat.csv`
- `EvalMaskConfig`（评估/绘图掩码，不改数据）≠ `TrainOutlierConfig`（滑窗训练窗口清洗，插值改数据）
- 本地默认入口：修改 `main.py` 中的 `CONFIG_YAML`，再直接运行 `uv run python main.py`
- `run.py` 保留为兼容入口，但当前文档和脚本不再把它作为推荐运行方式
- 日志目录：`logs/main/`（直接运行 `main.py`）
- `log_util.py` 在模块导入时执行，`LOG_NAME` 用 `os.environ.get('LOG_NAME', 'main')` 提供默认值
- 在这台机器上如果 `uv` 触发 `~/.cache/uv` 权限问题，优先使用 `UV_CACHE_DIR=.uv_cache`
- 算力房间数据：`df.csv`（215 列，全量合并）、`df_selected.csv`（多变量筛后建模数据）、`feature_selection_report.csv`（筛列报告）和稠密版 `df_filled.csv`（`config/aidc_electricity_computility/electricity/2026-06-11/scripts/fill_missing.py`：原始列补 0 + 重算派生）都在 `demand_load/<room>/` 下；算力预处理权威入口是 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py`，特征分原始聚合、状态/结构、时序动态三层；单变量配置继续使用 `df_power.csv`，4 个房间场景的多变量 LGBM 配置使用 `df_selected.csv`
