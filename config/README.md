# Config 说明

`config/` 保存模型运行配置、共享配置分组、YAML 配置、配置加载器和配置生成器。标准默认值由 Python `@dataclass`（`config/univariate_config.py` 或 `config/multivariate_config.py`）提供；具体数据和任务参数通过 YAML 覆盖。

## 加载规则

本地运行入口是 [main.py](/Users/wangzf/projects/tsproj_ml/main.py)。日常使用时直接在 `main.py` 中修改 `CONFIG_YAML`，然后运行 `uv run python main.py`。

加载边界：

- `config/config_loader.py` 提供 `load_yaml_config()`；`load_model_config()` 为内部工具函数，供 YAML 加载器读取 `base_config` 指定的 Python 模板。
- YAML 的 `base_config` 指向标准模板，例如 `config.univariate_config` 或 `config.multivariate_config`。
- `main.py` 通过 `CONFIG_YAML` 常量直接指定本地要加载的 YAML 配置文件。

YAML 基本结构：

```yaml
base_config: config.multivariate_config
overrides:
  target_series:
    data_dir: ./dataset/ETT-small/
    data_path: ETTm1.csv
    freq: 15min
    target_ts_feat: date
    target: OT
  exogenous_features:
    enable_date_features: true
    date_history_path: ETTm1_exogenous/df_date.csv
    date_future_path: ETTm1_exogenous/df_date_future.csv
    enable_weather_features: true
    weather_history_path: ETTm1_exogenous/df_weather.csv
    weather_future_path: ETTm1_exogenous/df_weather_future.csv
  model_strategy:
    model_type: lightgbm
    pred_method: multivariate-single-multistep-recursive
```

`config_class` 不写入 YAML；加载器默认使用 `ModelConfig`。`overrides` 可按 `config_sections.py` 的配置类语义分组，也兼容旧的平铺写法；最终叶子字段必须存在于模板中，未知字段会报错，避免拼写错误静默失效。

## 目录

```text
config/
├── config_sections.py
├── config_loader.py
├── generate_configs.py
├── univariate_config.py
├── multivariate_config.py
├── ETT-small/
│   └── ETTm1/
├── aidc_electricity_computility/
│   ├── electricity/
│   └── electricity_computility/
└── aidc_power_month/
    ├── route_A/
    └── route_B/
```

## 配置数量

| 配置目录 | 配置数 |
|---|---:|
| `ETT-small/ETTm1` | 21 YAML |
| `aidc_electricity_computility/electricity/2026-06-11/A1_01a` | 12 YAML |
| `aidc_electricity_computility/electricity/2026-06-11/A1_201` | 12 YAML |
| `aidc_electricity_computility/electricity/2026-06-11/A1_IT` | 12 YAML |
| `aidc_electricity_computility/electricity/2026-06-11/A3_01e` | 12 YAML |
| `aidc_electricity_computility/electricity/2026-06-11/AIDC/route_A` | 12 YAML |
| `aidc_electricity_computility/electricity/2026-06-11/AIDC/route_B` | 12 YAML |
| `aidc_power_month/route_A` | 8 YAML |
| `aidc_power_month/route_B` | 8 YAML |
| `aidc_electricity_computility/electricity_computility/add_training_inference_pod` | 7 `.py` 模块配置 |
| `aidc_electricity_computility/electricity_computility/add_training_jobs` | 7 `.py` 模块配置 |

## 数据映射

### ETT-small

- 数据目录：`dataset/ETT-small`
- 常见目标文件：`ETTm1.csv`
- 时间列：`date`
- 目标列：`OT`
- 频率：`15min`
- 示例外生目录：`dataset/ETT-small/ETTm1_exogenous`
- 示例日期外生：`ETTm1_exogenous/df_date.csv`、`ETTm1_exogenous/df_date_future.csv`
- 示例天气外生：`ETTm1_exogenous/df_weather.csv`、`ETTm1_exogenous/df_weather_future.csv`

### ETTm1 (单变量 & 多变量)

- 数据目录：`dataset/ETT-small`
- 目标文件：`ETTm1.csv`
- 时间列：`date`
- 目标列：`OT`
- 频率：`15min`
- 示例外生目录：`dataset/ETT-small/ETTm1_exogenous`
- 单变量策略使用 `config.univariate_config` 作为 base，多变量策略使用 `config.multivariate_config`

### AIDC electricity

旧日期批次：

```text
dataset/aidc_electricity_computility/electricity/<date>/demand_load/lingang_A
dataset/aidc_electricity_computility/electricity/<date>/demand_load/lingang_B
```

`2026-06-11` 批次：

```text
dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_201
dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_01a
dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_IT
dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A3_01e
dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/AIDC/route_A
dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/AIDC/route_B
```

常见字段：

- 目标文件：单变量默认 `df_power.csv`；4 个房间场景的多变量 LGBM 配置使用 `df_selected.csv`
- 时间列：`count_data_time`
- 目标列：`h_total_use`
- 频率：`5min`
- 日期外生：`df_date.csv`、`df_date_future.csv`
- 天气外生：`df_weather.csv`、`df_weather_future.csv`

### AIDC power month

日频（`1D`）算力房间月度负荷，与上面 5min 的 AIDC electricity 区分：

```text
dataset/aidc_power/
```

- 目标文件：`derived/{A,B}_Loads_1day_mean_20251001_20260728.csv`（日 mean 聚合，由 `data_process/data_aggregate.py` 从 5min 源数据生成，缺失点 linear 插值填充——回测依据见 `data_process/fill_method_backtest.py`）
- 时间列：`time`；目标列：`value`；频率：`1D`（301 天，2025-10-01 → 2026-07-28）
- 同目录 `weather_in_*`（小时粒度）、`date_in_*` 不作外生（粒度不匹配 / 无未来期），单变量配置默认仅用 datetime 派生
- 对应配置：`config/aidc_power_month/route_A/`、`route_B/`（各 8 个 lgbm 单变量方法，`data_path` 指向 `derived/` 下的 `_mean` 聚合文件）
- 数据聚合配置：`config/aidc_power_month/aggregate_A.yaml`、`aggregate_B.yaml`（由 `data_process/data_aggregate.py run_aggregation` 加载，schema 独立于模型配置，`config <yaml> [--force]` 运行）

### aidc_power_15min（5min -> 15min 场景）

15min 频率负荷，与日频共享 5min 源数据，聚合目标频率不同：

- 数据聚合配置：`config/aidc_power_15min/aggregate_A.yaml`、`aggregate_B.yaml`（`data_process/data_aggregate.py` 加载）
- 目标文件：`derived/{A,B}_Loads_15min_mean_20251001_20260728.csv`（15min mean 聚合，linear 插值填充）
- 时间列：`time`；目标列：`value`；频率：`15min`（301 天 × 96 点/天 = 28896 行）
- 对应配置：`config/aidc_power_15min/route_A/`、`route_B/`（各 8 个 lgbm 单变量方法，`data_path` 指向 `derived/` 下的 15min `_mean` 聚合文件）
- 与日频配置差异：`predict_steps` 96（34）、`window_days` 30（150）、`history_days` 60（281）、lags 为 1~7 天的 15min 步数 [96,192,288,384,480,576,672]、datetime 特征加回 `hour`/`minute`、rolling/diff 窗口与周期按 ×96 换算（如 rolling_windows [672,1344,2688] = 7/14/28 天）

### aidc_power_15min_short（4h 超短期场景）

复用 15min 聚合数据（不再重复 aggregate 配置），预测未来 4 小时 = 16 点：

- 对应配置：`config/aidc_power_15min_short/route_A/`、`route_B/`（各 8 个 lgbm 单变量方法）
- 与 15min 日预测配置差异：`predict_steps` 16（96）、`history_days` 30（60）、`window_days` 14（30）、lags 增加近邻步数 [1,2,3,4,8,12,16] 并保留日/周周期 [96,192,672]、rolling_windows [16,96,672]、diff_periods [1,96,672]、`decay_halflife_days` 14（60）
- 滑窗测试窗口数约 97 个（horizon 小则窗口多），可用 `window_parallel_workers` 并行

## 配置分组参考

YAML 的 `overrides` 按 `config_sections.py` 的 dataclass 分组。常用分组说明：

### training_enhancement

训练阶段的调参、增强与损失配置。所有字段均有 dataclass 默认值，YAML 只需写需要覆盖的字段。

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `patience` | int | `100` | 早停轮数 |
| `enable_time_decay_sample_weight` | bool | `false` | 时间衰减样本权重：近期样本权重更高，抑制概念漂移导致的远期噪声 |
| `decay_halflife_days` | float | `14.0` | 半衰期（天）：样本权重衰减到一半所需的样本年龄；仅在 `enable_time_decay_sample_weight: true` 时生效 |
| `perform_tuning` | bool | `false` | 是否执行超参数随机搜索 |
| `enable_data_augmentation` | bool | `false` | 是否启用数据增强 |
| `enable_feature_selection` | bool | `false` | 是否启用特征选择 |
| `enable_auto_learning_rate` | bool | `false` | 是否自动估算学习率 |

> 时间衰减权重兼容性：默认配置下（`multi_output_strategy: multioutput` + `predict_type: point` + `enable_ensemble: false`）所有 7 种预测方法均支持；`regressor_chain` 多输出、`ensemble`、多输出分位数路径不支持，开启时会 warning 跳过（不阻塞运行）。

启用示例：

```yaml
  training_enhancement:
    enable_time_decay_sample_weight: true
    decay_halflife_days: 14
```

## 配置生成

`config/generate_configs.py` 用于从 dataset 叶子目录批量生成分组 YAML 配置。生成前建议使用
`--dry-run` 检查输出路径和文件名。

```bash
uv run python config/generate_configs.py \
  --dataset dataset/ETT-small \
  --data-file ETTm1.csv \
  --target OT \
  --dry-run
```

默认模型矩阵：

- `lightgbm` -> `lgbm`
- `xgboost` -> `xgb`
- `catboost` -> `cab`

默认单变量策略矩阵：

- `usmdp`
- `usmd`
- `usmr`
- `usmdr`

多变量策略可显式传入：

- `msmd`
- `msmr`
- `msmdr`

默认输出格式为 YAML。
