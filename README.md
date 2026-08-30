# tsproj-ml

基于机器学习回归器的时间序列多步预测项目。现役模型配置统一为 canonical schema（`schema_version: 2`），canonical 主链以预测问题、数据角色、七种标准多步策略、估计器耦合、Local/Global scope 和 point/边际 quantile 为正交维度；模型工厂支持 LightGBM、XGBoost、CatBoost、RandomForest、HistGradientBoosting、Ridge/ElasticNet/Lasso、QuantileRegressor 和 SeasonalTemplate。

本文档按当前代码和权威 [`docs/multistep_forecasting_redesign.md`](docs/multistep_forecasting_redesign.md) 整理；旧设计已被该文档合并取代，历史细节通过 git 溯源。目录级细节可继续查阅 `config/README.md`、`dataset/README.md`、`data_provider/README.md`、`models/README.md`、`utils/README.md` 和 `docs/production_sync.md`。

## 当前能力

canonical strategy：

| Strategy | 推进语义 |
|---|---|
| Recursive | 单个一步模型逐步回填 |
| Direct | 每个 horizon 一个独立模型 |
| MIMO | 一个模型一次输出全部 horizon × target |
| RecMO | 一个模型按固定块输出并块间递归 |
| DirRec | 每个 horizon 一个模型，消费此前预测 |
| DirMO | 每个块一个独立模型 |
| DirRecMO | 每个块一个模型，块间递归 |

point 输出固定 `(N,H,K)`，边际 quantile 固定 `(N,H,K,Q)`；BR 使用 `EnsembleSpec`，不是第八种 strategy。数据列只允许 target/observed-past/known-future/static/key/ignored 六角色，并执行严格 as-of。joint samples 与 canonical CQR 不在当前已实现范围。

主要能力：

- 树模型主链路：`lightgbm` / `xgboost` / `catboost`。
- 工厂层额外支持：`randomforest` / `histgb` / `ridge` / `elasticnet` / `lasso` / `quantileregressor` / `seasonaltemplate`（共 10 类，均带别名，见 `models/ModelFactory.py`）。
- canonical point 与每目标边际 quantile；quantile crossing 独立按 target 修复，recursive 使用 median path。
- multi-target adapter：independent、RegressorChain、native capability gate；禁止静默降级。
- `EnsembleSpec`：weighted 与无泄漏 stacking；Local/Global、K1/K2、point/quantile 正交。
- source registry + FeatureCompiler：统一 date/weather/custom/static，输出 source lineage 与 visibility proof，observed-past 无隐式 persistence。
- 可选数据增强、特征选择、目标/特征缩放、窗口内因果目标分解、时间衰减样本权重、自动学习率、`RandomizedSearchCV` + `TimeSeriesSplit` 调参。
- 滑窗测试支持训练窗口异常值局部修复，测试真实值不被清洗。
- 滑窗测试中的 `MAPE Accuracy = 1 - MAPE` 按 `eval_mask` 掩码过滤后计算（默认 `mode: percentile` 即窗口正样本 `P5`；可设 `min_value`/`max_value` 绝对上下限），近零/越界异常点不再制造失真指标；汇总指标用 median，并附季节 naive 对照列。（legacy 测试链行为；canonical 结果层当前只保留 `plot_valid` 通道，掩码指标待接入。）
- 预测输出保存历史上下文和预测绘图拼接数据，便于生产问题定位。
- 滑窗测试图（`test_prediction.png`）按 `eval_mask` 对无效点断线显示；未来预测图（`prediction.png`）绘制原始值保证线条连续，掩码结果保留在 `prediction_plot_concat.csv` 的 `plot_value`/`plot_valid` 列。（同上，legacy 行为。）

## 项目结构

```text
tsproj_ml/
├── main.py                      # 主流程入口，本地修改 YAML 后直接运行
├── run.py                       # 保留的兼容入口，不作为本地直跑推荐方式
├── pyproject.toml               # uv 依赖定义，Python == 3.10.11
├── config/
│   ├── config_loader.py         # canonical schema strict parser（非 canonical 一律 RAISE）
│   ├── aidc_electricity_computility/
│   └── <场景配置目录>/
├── dataset/                     # ETT-small（演示数据保留）和 AIDC electricity 数据
├── data_provider/
│   └── outlier_handling.py      # 滑窗训练段异常处理
├── data_process/                # 离线/批处理脚本（不接入主流程模型代码）
│   ├── data_aggregate.py        # 单目标时序频率聚合 CLI（配置驱动，含缺失填充注册表）
│   ├── fill_method_backtest.py  # 缺失填充方法掩码回测（MAPE 分桶对比）
│   ├── outlier_process.py       # 原始负荷数据异常标记、清洗与可视化（配置驱动）
│   └── 其余工具脚本             # decomposition/difference/impute/normalization 等，当前未被模型代码引用
├── docs/
│   ├── multistep_forecasting_redesign.md
│   └── production_sync.md       # 生产包同步边界、可迁移模块和同步记录
├── model_forecasting/                 # canonical specs/data/features/strategies/transforms/runtime/results
├── features/
│   ├── FeatureScalering.py      # 特征/目标缩放与逆变换
│   ├── TargetTransformation.py  # calendar/decomposition/scaler 有序目标变换栈
│   ├── FeatureSelection.py      # 训练集特征选择
│   └── DataAugment.py           # 训练集数据增强
├── models/
│   ├── ModelFactory.py          # 模型工厂和模型封装（10 类回归器）
│   ├── ModelTraining.py         # CanonicalTrainer 与 estimator helper
│   ├── ModelForecasting.py      # CanonicalForecaster
│   ├── ModelEnsemble.py         # 融合模型
│   ├── AuxiliaryForecaster.py   # 非目标内生变量辅助递归预测器（auxiliary 回填）
│   ├── ModelSaveLoad.py         # pickle 保存/加载
│   └── losses.py, learning_rate.py
├── probabilistic/               # 概率预测 spec、类型、训练、后处理、校准与评估
├── scripts/                     # 项目级维护工具
│   └── check_model_configs.py   # 模型配置 dry-run 校验（复算 main.py 合法性校验，不训练不预测）
├── tests/                       # 已纳入版本控制的 unittest 回归测试
└── utils/
    ├── frequency.py             # pandas 频率解析
    ├── eval_mask.py             # 评估/绘图有效点掩码（独立工具，当前主流程未 import）
    ├── runtime_env.py           # 运行期环境变量辅助
    └── log_util.py              # 日志器
```

运行产物默认写入 `results/`，日志默认写入 `logs/`；这两个目录通常不作为源码事实来源。

## 核心流程

`main.py` 读取 YAML 后调用 `build_model()`：只有 canonical `ForecastConfigSpec` 能进入 `CanonicalModel`（其余直接 `TypeError` 拒绝），不存在第二运行时。canonical 流程：

1. `SourceRegistry` 读取显式 sources，按每个 origin 物化严格 information set。
2. `FeatureCompiler` 生成固定 schema、lineage 和 visibility proof。
3. `CanonicalTrainer` 按七策略与 estimator adapter teacher-forced 训练；rolling holdout 保证训练标签不跨测试边界。
4. `CanonicalForecaster` 按 series/target 隔离递归状态，返回 `(N,H,K[,Q])`。
5. per-series/per-target transform 严格逆序恢复；结果写 long CSV、schema-2 bundle、resolved config/model 与 fingerprint identity。

未来阶段不读取真实 target；observed-past 若需要推进期值，必须绑定显式 provider。

## 快速开始

安装依赖：

```bash
uv sync
```

本地直跑入口：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py
```

使用方式：

1. 打开 [`main.py`](main.py)。
2. 修改 `CONFIG_YAML`，指向要测试的本地 YAML 配置文件；当前默认值是 `config/aidc_load_15min_short/route_A/baseline/st_usmr_mean.yaml`。
3. 如需调整数据路径、模型类型、预测策略、测试与预测开关，直接修改对应 YAML。
4. 运行上述 `env -u PYTHONPATH ...` 命令。

`main.py` 不提供命令行参数覆盖；本地开发和测试统一采用“手动改 YAML + 直接运行”的方式。

## Legacy 清理结论

legacy 体系（config generator、`base_config + overrides + pred_method` YAML、九方法语义层、迁移器、只读适配层）已于 2026-08-29 全部删除：`main.py` / `run.py` 只执行 canonical `ForecastConfigSpec`，非 canonical 输入直接 RAISE；旧 pkl 与旧宽表结果在本仓库不可读。历史映射与删除清单见 `docs/multistep_forecasting_redesign.md` §8。

## 数据契约

核心配置字段：

| 字段 | 含义 |
|---|---|
| `data_dir` | 数据目录，会在运行时转为 `Path` |
| `data_path` | 目标序列 CSV 文件名 |
| `target_ts_feat` | 目标序列时间列 |
| `target` | 目标值列，进入主链路后映射为 `y` |
| `freq` | pandas 固定频率，如 `5min`、`15min`、`1h` |
| `now_time` | 预测基准时间 |
| `history_length` | 历史窗口天数 |
| `predict_steps` | 预测步数（以 `freq` 为单位，如 15min 下 1 天 = 96、4 小时 = 16） |
| `window_length` | 滑窗测试的训练+测试窗口天数 |

可选外生输入：

- 日期类型：`date_history_path`、`date_future_path`、`date_ts_feat`、`datetype_features`
- 天气：`weather_history_path`、`weather_future_path`、`weather_ts_feat`、`weather_features`
- 其他内生变量：由 `target_series_numeric_features` / `target_series_categorical_features` / `target_series_drop_features` 控制；历史处理时会自动排除时间列、目标列和 drop 列。

### AIDC 算力链路

`config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py` 是当前 AIDC 电力/算力融合的权威入口。它读取
`electricity/算力数据/<room>/training|inference|pod/` 下的原始指标，先生成
`df_computility.csv`，再与 `df_power.csv` 按 `count_data_time` 内连接生成 `df.csv`，
并继续产出 `df_selected.csv` 与 `feature_selection_report.csv`。

当前算力特征分三层：

- 原始聚合特征：按 5 分钟统计 `min/max/mean/std/count`，容量/功率类额外保留 `sum`
- 状态与结构特征：如 `training_active_jobs`、`*_gpu_busy_ratio`、
  `training_to_inference_power_ratio`、`pod_overhead_ratio`
- 时序动态特征：如 `*_diff_1`、`*_diff_3`、`*_roll_mean_3`、`*_roll_mean_12`

当前四个房间场景的 `df.csv` 均为 215 列：

- `count_data_time`
- `h_total_use`
- 213 个算力特征

多变量 LGBM 配置现优先使用筛后的 `df_selected.csv`；单变量配置继续使用 `df_power.csv`。

## 关键配置

运行模式：

- `is_testing`: 是否执行滑窗测试。
- `is_forecasting`: 是否训练全历史并预测未来。
- `max_test_windows`: 可选，只跑前 N 个测试窗口做快速验证。
- `test_window_stride`: 可选，滑窗抽样步长。

特征与预处理：

- `enable_datetime_features`
- `enable_date_features`
- `enable_weather_features`
- `enable_lags_features`
- `enable_advanced_features`
- `scale_features` / `feature_scaler_type`
- `scale_target` / `target_scaler_type` / `inverse_target`
- `encode_categorical_features`

训练增强：

- `enable_data_augmentation`
- `enable_feature_selection`
- `enable_auto_learning_rate`
- `perform_tuning`
- `enable_ensemble`
- `predict_type="quantile"` + `quantiles`
- `enable_global_training` + `series_id_feature`

并行与性能：

- `window_parallel_workers`
- `multi_output_n_jobs`
- `quantile_parallel_workers`
- `ensemble_parallel_workers`
- `model_thread_count`
- `block_size`

配置加载：

- `config/config_loader.py` 提供 `load_yaml_config(config_yaml)`，按 YAML 的 `base_config` 加载对应的 Python 模板并应用分组 `overrides`。
- `load_model_config()` 为内部工具函数，由 `load_yaml_config` 调用以实例化 `base_config` 指定的模板，不对外直接使用。
- 该加载器导入时不会解析 `sys.argv`；`main.py` 直接读取本地固定的 YAML 路径。

## 输出结构

canonical schema 输出：

```text
results/<scenario>/<result_identity>/
  pretrained_models/{model.pkl,resolved_model.json}
  results_test/{cv_plot_df.csv,test_scores_df.csv,result_metadata.json,...}
  results_forecast/{prediction.csv,resolved_config.json}
```

`result_identity=<strategy|ensemble>-<model>-<scope>-k<K>-<fingerprint[:12]>`；prediction/CV 使用 `(series_id,time,target[,window])` long 唯一键。canonical 当前只输出 `predict_q*`，不输出 CQR `predict_pi*`。

### 旧输出布局（已废弃）

`<scenario>` 由 YAML 中的 `scenario_subpath` 显式指定（如 `aidc_load_month/route_A`），使结果路径与 config 目录布局对齐；未指定时回退为从 `data_dir` 自动推导（去掉 `dataset/` 前缀和 `demand_load` 段）。多组配置共用同一 `data_dir` 时（如三个 `aidc_power_*` 场景共用 `dataset/aidc_power/`）必须显式指定，否则结果会混入同一目录。不同场景的结果因此互不覆盖。

`<setting>` 由 `{model_type}-{data_name}-{pred_method_code}-{window_token}[-quantile][setting_suffix][-calendar-month]` 组成；固定步长的 `window_token=window_length`，自然月模式取 `train_window_length`。

```text
results/
  pretrained_models/<scenario>/<setting>/
    model.pkl                     # 唯一 ForecastModelBundle
    probabilistic_schema.json     # 可读 schema/metadata，不含模型对象
  results_test/<scenario>/<setting>/
    test_scores_df.csv
    cv_plot_df.csv
    probabilistic_predictions_df.csv
    probabilistic_scores_df.csv
    horizon_scores_df.csv
    probabilistic_intervals_df.csv # 启用 CQR 且历史充分时
    calibration_report.csv         # prequential + final as-of 审计
    train_outlier_report.csv
    test_prediction.png            # 有有效预测列时保存
  results_forecast/<scenario>/<setting>/
    prediction.csv
    prediction.png
    history_context.csv
    prediction_plot_concat.csv
```

`prediction.csv` 基础列为 `time,predict_value`；分位数预测额外输出由数值 grid 编码的 `predict_q*`，且 `predict_value == configured point quantile`（本项目为 q50）。CQR 不覆盖模型分位数，而是追加独立的 `predict_pi<coverage>_lower/upper`。

滑窗测试与未来预测图的可视化契约：

- `test_scores_df.csv` 额外记录 `MAPE Threshold`、`MAPE Upper Threshold`、`MAPE Valid Points`、`MAPE Excluded Points`、`MAPE Excluded Ratio`。
- `cv_plot_df.csv` 额外记录测试图的有效性标记；被 `eval_mask` 判为异常的点在 `test_prediction.png` 中断线显示。
- `prediction_plot_concat.csv` 保留历史与未来拼接后的原始值 `raw_value`，并额外输出 `plot_value`（eval_mask 掩码后的绘图值，异常点为 NaN）、`plot_valid`、`series_type`；当前 `prediction.png` 的历史上下文直接绘制原始 `y`（保证线条连续），未来预测使用原始 `predict_value`，掩码列供排障和自定义绘图使用。

离线原始负荷异常处理由 `data_process/outlier_process.py` 提供（配置驱动，用法同 `data_aggregate.py`）。配置示例：`config/aidc_electricity_computility/electricity/2026-06-11/outlier_detect.yaml`。输出文件名从源文件名派生，保存到源数据目录：

```text
df_power_outlier_detection.csv
df_power_remove_outlier.csv
df_power_anomalies.png
```

## 常见边界

- 根 README 只做项目总览和主流程说明；配置映射、数据目录、目录职责请以对应子目录 README 为准。
- canonical 默认与校验由 `model_forecasting/specs/` 提供；新增模型配置只写 canonical schema YAML（`schema_version: 2`）。
- `main.py` 是本地开发与测试的主入口；切换任务时直接修改 `CONFIG_YAML` 和对应 YAML 内容。
- `now_time` 为预测锚点：`schedule_mode=daily`（默认）时规整到次日 00:00 作为历史结束/预测开始（预测下一完整自然日），`intraday` 时保留调度时刻；历史区间为 `[now_time - history_length, now_time)`，预测区间为 `[now_time, now_time + predict_steps × freq)`。
- canonical quantile 按 target/level 训练并保持 `(N,H,K,Q)`；Ensemble 与 probabilistic mode 正交。
- observed-past 不允许默认持久性回填，必须配置 explicit provider；需要共同预测时应提升为 target。
- Global 由 `problem.training_scope=global` 与 `series_id_cols` 声明，lag/shift/窗口/递归状态按 series 隔离，series order 写入 bundle。
- `.DS_Store`、`__pycache__/`、`logs/`、`results/` 不是源码或数据契约。

## 生产同步边界

`docs/production_sync.md` 是生产包同步边界和同步记录文档。本仓库 `tsproj_ml` 是时间序列预测核心库，生产包只作为 adapter。核心算法、特征工程、评估、损失函数和通用数据质量逻辑应先在本仓库优化，再按需同步到生产包。

允许从本仓库向生产包迁移的核心模块包括：

- `model_forecasting/`（必须按完整合同迁移）
- `models/ModelForecasting.py`
- `models/ModelTraining.py`
- `probabilistic/`
- `models/losses.py`
- `models/learning_rate.py`
- `data_provider/outlier_handling.py`


禁止直接回迁或反向污染本仓库的内容包括：

- 生产 API main class
- `BaseModelMainClass`
- 算力数据预处理
- 生产输出字段
- 生产路径 import
- dataset 和 results 文件

维护方式：

- 涉及生产同步时先更新或查阅 `docs/production_sync.md`，不要靠临时复制代码。
- 每次同步记录使用文档中的模板，说明变更摘要、是否需要迁移项目 2、项目 2 适配点和验证结果。
- 如果生产包中只有部署入口、平台父类、生产字段或路径适配变化，不应回迁到本仓库。
- 如果生产包中出现可复用的预测、评估、特征或数据质量逻辑，应先抽象成本仓库核心模块，再由生产包做最小 adapter 适配。

## 开发与维护注意事项

- 项目事实以当前代码为准；更新文档时不要把 `docs/` 或 `logs/` 中的历史内容当作当前实现。
- 运行入口、配置路径、输出目录和脚本说明容易漂移，修改前应重新核对 `main.py`、`config/`、`scripts/`。
- 配置字段、模型类型和预测策略名保持英文；中文用于解释语义和边界。
- 如果发现脚本、README、配置和代码不一致，优先按代码写清楚现状，不要为了文档一致性假定代码已经修好。
- 修改 README 时避免重复目录级 README 的全部细节；根 README 只保留全局流程、关键契约和常用命令。

## 日志

日志由 `utils/log_util.py` 创建。调用方通常先设置 `LOG_NAME`，日志写入 `logs/<LOG_NAME>/`；日志级别由 `SERVICE_LOG_LEVEL` 控制，默认 `INFO`。

`main.py` 会设置 `LOG_NAME=main`。脚本也可通过环境变量覆盖 `LOG_NAME`。

## 验证建议

文档更新后可用一致性搜索确认没有恢复旧入口模块名、旧配置类名或旧测试目录说明。

轻量入口运行检查：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py
```

静态和测试检查：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m compileall forecasting models/ModelTraining.py models/ModelForecasting.py main.py run.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run \
  python -m unittest discover -s tests -p "test_*.py"
```

批量 YAML 配置的合法性 dry-run 校验（不训练不预测）：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/<scenario>/**/*.yaml'
```

注意：`tests/` 已纳入版本控制；改动或迁移被测模块时需同步修复测试导入与接口断言，保持全套通过。

## 版本与依赖

- Python: `==3.10.11`
- 依赖来源：`pyproject.toml` 和 `uv.lock`
- 推荐使用 `uv sync` / `uv run`，避免手工 `pip install` 造成锁文件漂移。
