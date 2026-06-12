# Config ↔ Dataset 映射说明

本说明按当前 `config/` 目录和配置模板整理。未参考 `docs/`、`logs/`
中的历史材料。

## 配置系统

配置文件是 Python `@dataclass`，运行时通过模块路径和类名加载：

```bash
uv run python run.py \
  --config-module config.templates.univariate_config \
  --config-class ModelConfig
```

当前模板：

| 模板 | 用途 |
|---|---|
| `config/templates/univariate_config.py` | 单变量策略模板，进入历史处理后只保留 `time + y` |
| `config/templates/multivariate_config.py` | 多变量策略模板，可保留其他内生变量 |

所有当前模板和批量生成配置的类名都是 `ModelConfig`。

注意：`run.py` 和部分 shell 脚本的默认值仍是旧路径
`config.univariate_config` / `ModelConfig_univariate`。实际运行时请显式传
`--config-module` / `--config-class`，或用脚本环境变量覆盖。

## 当前配置目录

```text
config/
├── templates/
│   ├── univariate_config.py
│   └── multivariate_config.py
├── aidc_electricity_computility/
│   ├── electricity/
│   │   ├── 2026-01-01/{route_A,route_B}
│   │   ├── 2026-01-07/{route_A,route_B}
│   │   └── 2026-03-12/{route_A,route_B}
│   ├── electricity_computility/
│   │   ├── add_training_inference_pod
│   │   └── add_training_jobs
│   ├── gaoweichao_compare
│   └── it_electricity_computility/
│       ├── A1_01a
│       ├── A1_IT
│       └── A3_01e
└── ETT-small/
    ├── ETTm1
    ├── ETTh1
    ├── ETTh2
    └── ETTm2
```

`ETTh1`、`ETTh2`、`ETTm2` 目录当前存在，但没有 `model_config*.py`。

## 配置数量

| 配置目录 | 配置数 | 说明 |
|---|---:|---|
| `ETT-small/ETTm1` | 21 | 3 模型 × 7 策略 |
| `aidc_electricity_computility/electricity/2026-01-01/route_A` | 12 | 3 模型 × 4 单变量策略 |
| `aidc_electricity_computility/electricity/2026-01-01/route_B` | 12 | 3 模型 × 4 单变量策略 |
| `aidc_electricity_computility/electricity/2026-01-07/route_A` | 12 | 3 模型 × 4 单变量策略 |
| `aidc_electricity_computility/electricity/2026-01-07/route_B` | 12 | 3 模型 × 4 单变量策略 |
| `aidc_electricity_computility/electricity/2026-03-12/route_A` | 12 | 3 模型 × 4 单变量策略 |
| `aidc_electricity_computility/electricity/2026-03-12/route_B` | 12 | 3 模型 × 4 单变量策略 |
| `aidc_electricity_computility/electricity_computility/add_training_inference_pod` | 7 | LightGBM × 7 策略 |
| `aidc_electricity_computility/electricity_computility/add_training_jobs` | 7 | LightGBM × 7 策略 |
| `aidc_electricity_computility/gaoweichao_compare` | 4 | CatBoost × 4 单变量策略 |
| `aidc_electricity_computility/it_electricity_computility/A1_01a` | 21 | 3 模型 × 7 策略 |
| `aidc_electricity_computility/it_electricity_computility/A1_IT` | 21 | 3 模型 × 7 策略 |
| `aidc_electricity_computility/it_electricity_computility/A3_01e` | 21 | 3 模型 × 7 策略 |

## 数据组说明

### electricity

路径：

```text
config/aidc_electricity_computility/electricity/<date>/route_A
config/aidc_electricity_computility/electricity/<date>/route_B
```

对应数据约定：

```text
dataset/aidc_electricity_computility/electricity/<date>/demand_load/lingang_A
dataset/aidc_electricity_computility/electricity/<date>/demand_load/lingang_B
```

当前日期：`2026-01-01`、`2026-01-07`、`2026-03-12`。

常见字段：

- `data_path`: `df_power.csv`
- `target_ts_feat`: `count_data_time`
- `target`: `h_total_use`
- `freq`: `5min`
- 日期外生：`df_date.csv`、`df_date_future.csv`
- 天气外生：`df_weather.csv`、`df_weather_future.csv`
- 策略：USMDO / USMD / USMR / USMDR

### electricity_computility

路径：

```text
config/aidc_electricity_computility/electricity_computility/add_training_inference_pod
config/aidc_electricity_computility/electricity_computility/add_training_jobs
```

特点：

- 当前每组 7 个 LightGBM 配置。
- 覆盖 7 种预测策略，包括多变量策略。
- 日期/天气外生默认关闭。
- 日期时间和滞后特征默认启用。

### gaoweichao_compare

路径：

```text
config/aidc_electricity_computility/gaoweichao_compare
```

特点：

- 当前 4 个 CatBoost 单变量配置。
- 常见目标文件：`aidc_df.csv`。
- 通常带日期/天气外生。

### it_electricity_computility

路径：

```text
config/aidc_electricity_computility/it_electricity_computility/{A1_01a,A1_IT,A3_01e}
```

特点：

- 每组 21 个配置，覆盖 3 模型 × 7 策略。
- 常见目标文件：`df_power.csv`。
- 常见字段：`target_ts_feat=count_data_time`，`target=h_total_use`，`freq=5min`。
- 日期/天气/日期时间/滞后特征通常启用。

### ETT-small

路径：

```text
config/ETT-small/ETTm1
```

特点：

- 当前 `ETTm1` 有 21 个配置。
- `ETTh1`、`ETTh2`、`ETTm2` 暂无配置文件。
- 常见字段：`data_path=ETTm1.csv`，`target_ts_feat=date`，`target=OT`，`freq=15min`。
- 多变量配置常见内生列来自 `HUFL/HULL/MUFL/MULL/LUFL/LULL`，具体保留/丢弃以配置文件为准。

## 核心字段契约

| 字段 | 说明 |
|---|---|
| `data_dir` | 数据目录，`main.Model.__init__` 会转为 `Path` |
| `data_path` | 目标序列 CSV 文件 |
| `target_ts_feat` | 目标序列时间戳列 |
| `target` | 目标值列，进入主链路后映射为 `y` |
| `freq` | pandas 固定频率字符串 |
| `now_time` | 预测基准时间 |
| `history_days` | 历史窗口天数 |
| `predict_days` | 预测未来天数 |
| `window_days` | 滑窗测试窗口天数 |
| `target_series_numeric_features` | 其他数值内生变量；为空时由历史目标表自动推断 |
| `target_series_categorical_features` | 其他类别内生变量 |
| `target_series_drop_features` | 从内生变量中排除的列 |

时间区间：

- 历史区间：`[now_time - history_days, now_time)`
- 预测区间：`[now_time, now_time + predict_days)`
- `now_time` 会在 `main.Model` 中规整到整点。

## 特征开关

| 字段 | 说明 |
|---|---|
| `enable_datetime_features` | 从统一 `time` 列生成分钟、小时、星期、月份等时间特征 |
| `enable_date_features` | 合并日期类型/节假日类外生特征 |
| `enable_weather_features` | 合并天气外生特征，并计算 `cal_rh` |
| `enable_lags_features` | 为目标或内生变量生成滞后特征 |
| `enable_advanced_features` | 总开关，高级特征还需要分别打开 rolling/diff 等子开关 |
| `use_horizon_exogenous_for_direct` | Direct 类方法是否使用 horizon-aware 外生展开 |

高级特征子开关包括：

- `enable_rolling_features`
- `enable_expanding_features`
- `enable_diff_features`
- `enable_pct_change_features`
- `enable_time_since_features`
- `enable_cyclical_features`
- `enable_interaction_features`
- `enable_polynomial_features`

## 训练与预测开关

| 字段 | 说明 |
|---|---|
| `model_type` | `lightgbm` / `xgboost` / `catboost`；工厂层也支持 `randomforest` |
| `model_params` | 透传到模型工厂，会覆盖默认参数 |
| `pred_method` | 7 种预测策略之一 |
| `multi_output_strategy` | `multioutput` 或 `regressor_chain` |
| `predict_type` | `point` 或 `quantile` |
| `quantiles` | 分位数列表，例如 `[0.1, 0.5, 0.9]` |
| `enable_ensemble` | 点预测下启用融合模型 |
| `ensemble_models` | 融合模型列表，如 `["lgb", "xgb", "cat"]` |
| `ensemble_method` | `averaging` / `weighted` / `stacking` / `blending` |
| `enable_global_training` | 透传 `series_id_feature` 的最小全局训练模式 |
| `block_size` | USMDR/MSMDR 分块大小；为 0 时使用最小 lag 或 1 |

## 数据预处理与增强

| 字段 | 说明 |
|---|---|
| `scale_features` | 是否缩放 X |
| `feature_scaler_type` | `standard` 或 `minmax` 等，见 `FeatureScalering.py` |
| `scale_target` | 是否缩放 Y |
| `target_scaler_type` | `none`、`standard`、`minmax`、`log1p`、`robust`、`yeo-johnson` |
| `inverse_target` | 预测/评估是否恢复到目标原始量纲 |
| `encode_categorical_features` | 是否编码类别特征 |
| `enable_data_augmentation` | 是否对训练集增强 |
| `enable_feature_selection` | 是否在训练集 fit 特征选择器 |
| `enable_auto_learning_rate` | 是否按样本量解析学习率 |
| `perform_tuning` | 是否执行超参数搜索 |

## 滑窗测试训练集异常处理

该能力只作用于每个测试窗口的训练段目标列，测试段真实 `y` 不被修改。

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `enable_train_outlier_handling` | `False` | 是否启用训练段异常处理 |
| `train_outlier_method` | `local_interpolate` | 当前支持局部插值修复 |
| `high_outlier_threshold` | `15000.0` | 短连续高值异常阈值 |
| `high_outlier_max_run_points` | `4` | 高值异常最大连续点数 |
| `drop_outlier_max_run_points` | `2` | 下探异常最大连续点数 |
| `drop_rebound_min_abs_diff` | `900.0` | 下探/回弹最小绝对差 |

启用后，测试输出目录会写出 `train_outlier_report.csv`。

## 并行与快速验证

| 字段 | 说明 |
|---|---|
| `window_parallel_workers` | 滑窗测试并行窗口数；大于 1 时会限制子任务模型线程 |
| `multi_output_n_jobs` | 多输出训练并行度 |
| `quantile_parallel_workers` | 分位数子模型训练并行度 |
| `ensemble_parallel_workers` | 融合模型训练并行度 |
| `model_thread_count` | 单模型内部线程数 |
| `max_test_windows` | 可选，只取前 N 个滑窗 |
| `test_window_stride` | 可选，滑窗抽样步长 |
| `enable_step_logging` | 递归预测逐步日志开关 |
| `forecast_log_interval` | 递归预测日志间隔 |

## 命名规范

```text
model_config_{model_short}_{method_short}[_{variant}].py
```

| 缩写 | 全称 |
|---|---|
| `lgbm` | LightGBM |
| `xgb` | XGBoost |
| `cab` | CatBoost |
| `usmdo` | `univariate-single-multistep-direct-output` |
| `usmd` | `univariate-single-multistep-direct` |
| `usmr` | `univariate-single-multistep-recursive` |
| `usmdr` | `univariate-single-multistep-direct-recursive` |
| `msmd` | `multivariate-single-multistep-direct` |
| `msmr` | `multivariate-single-multistep-recursive` |
| `msmdr` | `multivariate-single-multistep-direct-recursive` |

`variant` 用于区分站点或路线，例如 `A` / `B`。

## 添加新配置

推荐流程：

1. 从 `config/templates/univariate_config.py` 或 `config/templates/multivariate_config.py` 复制。
2. 修改 `data_dir`、`data_path`、`target_ts_feat`、`target`、`freq`。
3. 设置 `now_time`、`history_days`、`predict_days`、`window_days`。
4. 按数据实际情况设置日期、天气、滞后和高级特征开关。
5. 设置 `model_type`、`pred_method`、`model_params`。
6. 用命名规范保存到对应项目目录。

批量生成可用：

```bash
uv run python scripts/generate_configs.py --help
```

建议先用 `--dry-run` 预览，再写入配置文件。
