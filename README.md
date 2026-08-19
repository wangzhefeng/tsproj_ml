# tsproj-ml

基于机器学习回归器的时间序列多步预测项目。当前主链路围绕 LightGBM / XGBoost / CatBoost 构建，模型工厂层还支持 RandomForest、HistGradientBoosting、Ridge/ElasticNet/Lasso、QuantileRegressor 和 SeasonalTemplate（季节模板）；支持单变量/多变量、多步逐点 direct、direct、recursive、direct-recursive 分块和 direct+recursive 加权融合（blend）预测。

本文档按当前代码和当前目录整理，未使用 `docs/` 和 `logs/` 目录中的历史材料作为事实来源。目录级细节可继续查阅 `config/README.md`、`dataset/README.md`、`data_provider/README.md`、`models/README.md`、`utils/README.md` 和 `docs/production_sync.md`。

## 当前能力

预测策略由 `pred_method` 控制：

| 缩写 | `pred_method` | 说明 |
|---|---|---|
| USMDP | `univariate-single-multistep-direct-pointwise` | 单变量，按未来时点逐点 direct 预测 |
| USMD | `univariate-single-multistep-direct` | 单变量，一个多输出模型直接预测整个 horizon |
| USMR | `univariate-single-multistep-recursive` | 单变量，一步模型递归回填 |
| USMDR | `univariate-single-multistep-direct-recursive` | 单变量，按块 direct，块间递归回填 |
| MSMD | `multivariate-single-multistep-direct` | 多变量，一个多输出模型直接预测目标 horizon |
| MSMR | `multivariate-single-multistep-recursive` | 多变量递归预测目标，其他内生变量默认持久性回填 |
| MSMDR | `multivariate-single-multistep-direct-recursive` | 多变量按块 direct，块间递归回填 |
| USBR | `univariate-single-multistep-blend-direct-recursive` | 单变量，Direct+Recursive 加权融合 |
| MSBR | `multivariate-single-multistep-blend-direct-recursive` | 多变量，Direct+Recursive 加权融合 |

主要能力：

- 树模型主链路：`lightgbm` / `xgboost` / `catboost`。
- 工厂层额外支持：`randomforest` / `histgb` / `ridge` / `elasticnet` / `lasso` / `quantileregressor` / `seasonaltemplate`（共 10 类，均带别名，见 `models/ModelFactory.py`）。
- 点预测与分位数预测：`predict_type="point"` 或 `"quantile"`（可选 `quantile_monotone` 单调化）。
- 多输出训练策略：`multioutput`、`regressor_chain`，以及 direct 专用的逐 horizon 训练器；`direct_strategy: horizon_feature` 可把 horizon 索引作为特征将训练成本从 H× 降到 1×（仅 USMD/MSMD）。
- 可选模型融合：`averaging`、`weighted`、`stacking`、`blending`（支持成员级 `ensemble_model_specs`）。
- 可选预测增强：多变量内生变量 auxiliary 回填（MSMR/MSMDR）、Direct+Recursive blend（USBR/MSBR）、conformal（CQR）分位数校准。
- 可选数据增强、特征选择、目标/特征缩放、目标去趋势、时间衰减样本权重、自动学习率、`RandomizedSearchCV` + `TimeSeriesSplit` 调参。
- 滑窗测试支持训练窗口异常值局部修复，测试真实值不被清洗。
- 滑窗测试中的 `MAPE Accuracy = 1 - MAPE` 按 `eval_mask` 掩码过滤后计算（默认 `mode: percentile` 即窗口正样本 `P5`；可设 `min_value`/`max_value` 绝对上下限），近零/越界异常点不再制造失真指标；汇总指标用 median，并附季节 naive 对照列。
- 预测输出保存历史上下文和预测绘图拼接数据，便于生产问题定位。
- 滑窗测试图（`test_prediction.png`）按 `eval_mask` 对无效点断线显示；未来预测图（`prediction.png`）绘制原始值保证线条连续，掩码结果保留在 `prediction_plot_concat.csv` 的 `plot_value`/`plot_valid` 列。

## 项目结构

```text
tsproj_ml/
├── main.py                      # 主流程入口，本地修改 YAML 后直接运行
├── run.py                       # 保留的兼容入口，不作为本地直跑推荐方式
├── pyproject.toml               # uv 依赖定义，Python == 3.10.11
├── config/
│   ├── config_sections.py       # 共享分组 dataclass 和预测策略说明
│   ├── config_loader.py         # 无 import-time argparse 副作用的配置加载器
│   ├── generate_configs.py      # 根据 dataset 自动检测字段并批量生成模型配置
│   ├── univariate_config.py     # ETTm1 单变量示例配置
│   ├── multivariate_config.py   # ETTm1 多变量示例配置
│   ├── aidc_electricity_computility/
│   └── ETT-small/
├── dataset/                     # ETT-small 和 AIDC electricity 数据
├── data_provider/
│   ├── data_loader.py           # 数据读取、时间轴构造、历史/未来切片
│   └── outlier_handling.py      # 滑窗训练段异常处理
├── data_process/                # 离线/批处理脚本（不接入主流程模型代码）
│   ├── data_aggregate.py        # 单目标时序频率聚合 CLI（配置驱动，含缺失填充注册表）
│   ├── fill_method_backtest.py  # 缺失填充方法掩码回测（MAPE 分桶对比）
│   ├── outlier_process.py       # 原始负荷数据异常标记、清洗与可视化（配置驱动）
│   └── 其余工具脚本             # decomposition/difference/impute/normalization 等，当前未被模型代码引用
├── docs/
│   └── production_sync.md       # 生产包同步边界、可迁移模块和同步记录
├── features/
│   ├── FeatureEngineering.py    # 日期时间、日期类型、天气、自定义外生、滞后、高级特征
│   ├── FeatureScalering.py      # 特征/目标缩放与逆变换
│   ├── FeatureSelection.py      # 训练集特征选择
│   ├── DataAugment.py           # 训练集数据增强
│   └── feature_engineering/     # 实验性特征子包（fft/wavelet/statistical 等），未接入主流程
├── models/
│   ├── ModelFactory.py          # 模型工厂和模型封装（10 类回归器）
│   ├── ModelTraining.py         # 训练、调参、分位数、融合、blend 双子模型
│   ├── ModelTesting.py          # 滑窗测试与指标保存
│   ├── ModelForecasting.py      # 9 种预测策略的推理实现
│   ├── ModelEnsemble.py         # 融合模型
│   ├── AuxiliaryForecaster.py   # 非目标内生变量辅助递归预测器（auxiliary 回填）
│   ├── ModelSaveLoad.py         # pickle 保存/加载
│   └── losses.py, learning_rate.py
├── scripts/                     # 项目级维护工具
│   └── check_model_configs.py   # 模型配置 dry-run 校验（复算 main.py 合法性校验，不训练不预测）
├── tests/                       # 已纳入版本控制的 unittest 回归测试
└── utils/
    ├── frequency.py             # pandas 频率解析
    ├── eval_mask.py             # 评估/绘图有效点掩码
    ├── conformal.py             # CQR 分位数 conformal 校准
    ├── quantile.py              # 分位数列单调化
    ├── runtime_env.py           # 运行期环境变量辅助
    └── log_util.py              # 日志器
```

运行产物默认写入 `results/`，日志默认写入 `logs/`；这两个目录通常不作为源码事实来源。

## 核心流程

`main.Model.run()` 是当前主线：

1. `DataLoader.load_data()` 读取目标序列、日期外生、天气外生。
2. `DataLoader.process_history_data()` 根据 `now_time/history_length/freq` 构造历史时间轴，并把目标列统一映射为 `y`。
3. 单变量策略只保留 `time + y`；多变量策略保留配置中的其他内生变量。
4. `FeatureEngineer.create_features()` 生成外生、滞后和可选高级特征，并构造多步目标列。
5. 若 `is_testing=True`，`Tester._window_test()` 做滑窗验证，可按窗口并行。
6. 若 `is_forecasting=True`，`Trainer.train()` 用全历史训练，`Forecaster._predict_by_method()` 推理未来区间。
7. 测试、模型、预测结果分别保存到配置指定目录。

未来预测阶段只构造未来时间模板和未来外生特征，不读取未来真实目标值；递归类策略需要的目标值由预测结果逐步回填。

## 快速开始

安装依赖：

```bash
uv sync
```

本地直跑入口：

```bash
uv run python main.py
```

使用方式：

1. 打开 [main.py](/Users/wangzf/projects/tsproj_ml/main.py)。
2. 修改 `CONFIG_YAML`，指向要测试的本地 YAML 配置文件；当前默认值是 `config/aidc_load_month/route_B/lgbm_usmd_prob_mean.yaml`。
3. 如需调整数据路径、模型类型、预测策略、测试与预测开关，直接修改对应 YAML。
4. 运行 `uv run python main.py`。

`main.py` 不提供命令行参数覆盖；本地开发和测试统一采用“手动改 YAML + 直接运行”的方式。

## 配置生成

`config/generate_configs.py` 是批量生成分组 YAML 配置文件的工具，不是模型运行入口。它从一个 dataset 叶子目录读取目标 CSV，自动检测数据文件、时间列、频率、预测基准时间、内生特征列和外生文件，然后按模型 × 策略矩阵写入 `config/`。

YAML 按 `config_sections.py` 的分组语义保存覆盖项，例如 `target_series`、`exogenous_features`、`time_lag_features`、`model_strategy`。标准默认值由 YAML 内的 `base_config` 字段指定（通常为 `config.univariate_config` 或 `config.multivariate_config`）。

默认模型矩阵是 3 个树模型：

- `lightgbm` -> 文件名缩写 `lgbm`
- `xgboost` -> 文件名缩写 `xgb`
- `catboost` -> 文件名缩写 `cab`

默认单变量策略矩阵是 4 种方法：

- `usmdp`: `univariate-single-multistep-direct-pointwise`
- `usmd`: `univariate-single-multistep-direct`
- `usmr`: `univariate-single-multistep-recursive`
- `usmdr`: `univariate-single-multistep-direct-recursive`

脚本的关键输入：

| 参数 | 说明 |
|---|---|
| `--dataset` | 一个具体数据集叶子目录，目录内应包含目标 CSV |
| `--target` | 目标列名，例如 `h_total_use` |
| `--data-file` | 可选；不传时选目录中第一个非外生 CSV |
| `--target-ts-feat` | 可选；不传时使用目标 CSV 第一列 |
| `--models` | 逗号分隔模型列表，默认 `lightgbm,xgboost,catboost` |
| `--strategies` | 逗号分隔策略列表，默认 `usmdp,usmd,usmr,usmdr` |
| `--variant` | 输出子目录后缀，例如 `A1_201`、`A`、`B`；用于区分同数据集不同场景的配置目录，不写入文件名 |
| `--config-dir` | 输出目录；建议显式传入，避免默认路径和项目约定漂移 |
| `--base-config` | 可选；不传时按策略和内生特征自动选择 `config.univariate_config` 或 `config.multivariate_config` |
| `--dry-run` | 只打印将生成的文件，不写入磁盘 |

自动检测规则：

- 目标数据文件默认排除 `df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` 后取第一个 CSV。
- `target_ts_feat` 默认取目标 CSV 第一列。
- `freq` 优先用 `pandas.infer_freq` 推断，失败时按前两个时间戳间隔回退到 `1min`、`5min`、`15min`、`30min`、`1h`、`1D` 等固定频率。
- `now_time` 默认取目标 CSV 最后一行时间戳。
- 同时存在 `df_date.csv` 和 `df_date_future.csv` 时启用日期外生特征。
- 同时存在 `df_weather.csv` 和 `df_weather_future.csv` 时启用天气外生特征。
- 日期时间特征和滞后特征默认启用，分别可用 `--no-datetime`、`--no-lags` 关闭。

例如为 `2026-06-11` 的一个 demand load 数据集预览配置：

```bash
uv run python config/generate_configs.py \
  --dataset dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_201 \
  --target h_total_use \
  --models lightgbm,xgboost,catboost \
  --strategies usmdp,usmd,usmr,usmdr \
  --config-dir config/aidc_electricity_computility/electricity/2026-06-11/A1_201 \
  --dry-run
```

确认输出路径后去掉 `--dry-run` 即可生成实际配置。当前 `2026-06-11/demand_load` 已按 6 个叶子数据集生成了 84 个模型 YAML 配置（另含 1 个 `outlier_detect.yaml`）：

| 数据集 | 配置目录 | 文件数 |
|---|---|---:|
| `A1_201` | `config/aidc_electricity_computility/electricity/2026-06-11/A1_201` | 15 |
| `A1_01a` | `config/aidc_electricity_computility/electricity/2026-06-11/A1_01a` | 15 |
| `A1_IT` | `config/aidc_electricity_computility/electricity/2026-06-11/A1_IT` | 15 |
| `A3_01e` | `config/aidc_electricity_computility/electricity/2026-06-11/A3_01e` | 15 |
| `AIDC/route_A` | `config/aidc_electricity_computility/electricity/2026-06-11/AIDC/route_A` | 12 |
| `AIDC/route_B` | `config/aidc_electricity_computility/electricity/2026-06-11/AIDC/route_B` | 12 |

这些配置共用时间列 `count_data_time`、目标列 `h_total_use`、频率 `5min` 和 `now_time = 2026-06-11T23:55:00`。其中单变量 YAML 继续以 `df_power.csv` 为目标文件并使用 `config.univariate_config`，4 个房间场景的 `lgbm_msmd.yaml`、`lgbm_msmr.yaml`、`lgbm_msmdr.yaml` 为多变量配置，使用 `config.multivariate_config`，目标文件为筛后的 `df_selected.csv`。

另有 `2026-08-10` 场景目录（`A2_IT`、`A3_IT`、`liantong_IT`、`yancheng_IT`，各 7 个 YAML：4 单变量 + 3 多变量 lgbm）；对应 `dataset/.../2026-08-10/` 房间目录当前为空，这些 YAML 的 `data_dir` 指向 2026-06-11 数据，作为新房间的配置模板复用。

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

当前模板说明：

- `config/univariate_config.py` 和 `config/multivariate_config.py` 均默认指向 `dataset/ETT-small/ETTm1.csv`，目标列为 `OT`，使用 `dataset/ETT-small/ETTm1_exogenous/` 下的仿真外生数据。单变量策略共用同一数据源，区别在于只使用目标列自身历史。
- `config/multivariate_config.py` 默认指向 `dataset/ETT-small/ETTm1.csv`，目标列为 `OT`，多变量输入列为 `HUFL`、`HULL`、`MUFL`、`MULL`、`LUFL`、`LULL`。
- `config/multivariate_config.py` 默认启用 `dataset/ETT-small/ETTm1_exogenous/` 下的仿真日期类型和天气外生数据。
- 单变量默认 `pred_method` 为 `univariate-single-multistep-direct`；多变量默认 `pred_method` 为 `multivariate-single-multistep-direct-recursive`。

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

`<scenario>` 由 YAML 中的 `scenario_subpath` 显式指定（如 `aidc_load_month/route_A`），使结果路径与 config 目录布局对齐；未指定时回退为从 `data_dir` 自动推导（去掉 `dataset/` 前缀和 `demand_load` 段）。多组配置共用同一 `data_dir` 时（如三个 `aidc_power_*` 场景共用 `dataset/aidc_power/`）必须显式指定，否则结果会混入同一目录。不同场景的结果因此互不覆盖。

`<setting>` 由 `{model_type}-{data_name}-{pred_method_code}-{window_length}` 组成，例如 `lightgbm-df_power-usmr-15`。

```text
results/
  pretrained_models/<scenario>/<setting>/
    model.pkl
    target_scaler.pkl              # 仅目标缩放器已 fit 时保存
  results_test/<scenario>/<setting>/
    test_scores_df.csv
    cv_plot_df.csv
    train_outlier_report.csv
    test_prediction.png            # 有有效预测列时保存
  results_forecast/<scenario>/<setting>/
    prediction.csv
    prediction.png
    history_context.csv
    prediction_plot_concat.csv
```

`prediction.csv` 基础列为 `time,predict_value`；分位数预测会额外输出 `predict_q10`、`predict_q50`、`predict_q90` 这类列，实际列由 `quantiles` 决定。

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
- 标准默认值由 Python dataclass 模板提供；新增具体数据配置优先用分组 YAML 管理。
- `main.py` 是本地开发与测试的主入口；切换任务时直接修改 `CONFIG_YAML` 和对应 YAML 内容。
- `now_time` 为预测锚点：`schedule_mode=daily`（默认）时规整到次日 00:00 作为历史结束/预测开始（预测下一完整自然日），`intraday` 时保留调度时刻；历史区间为 `[now_time - history_length, now_time)`，预测区间为 `[now_time, now_time + predict_steps × freq)`。
- 分位数预测训练多个子模型；点预测融合只在 `predict_type="point"` 时生效。
- 多变量递归类方法（MSMR/MSMDR）对非目标内生变量默认持久性回填，可配置 `endogenous_backfill_strategy: auxiliary` 为每个内生变量训练独立递归辅助模型。
- `.DS_Store`、`__pycache__/`、`logs/`、`results/` 不是源码或数据契约。

## 生产同步边界

`docs/production_sync.md` 是生产包同步边界和同步记录文档。本仓库 `tsproj_ml` 是时间序列预测核心库，生产包只作为 adapter。核心算法、特征工程、评估、损失函数和通用数据质量逻辑应先在本仓库优化，再按需同步到生产包。

允许从本仓库向生产包迁移的核心模块包括：

- `models/ModelTesting.py`
- `models/ModelForecasting.py`
- `features/FeatureEngineering.py`
- `features/FeatureScalering.py`
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
uv run python main.py
```

静态和测试检查：

```bash
uv run python -m py_compile main.py run.py config/config_loader.py config/config_sections.py config/univariate_config.py config/multivariate_config.py config/generate_configs.py
uv run python -m unittest discover -s tests -p "test_*.py"
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
