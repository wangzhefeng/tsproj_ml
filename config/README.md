# Config 说明

`config/` 保存模型运行配置、共享配置分组、YAML 配置、配置加载器和配置生成器。标准默认值由 Python `@dataclass`（`config/univariate_config.py` 或 `config/multivariate_config.py`）提供；具体数据和任务参数通过 YAML 覆盖。

## 加载规则

正式运行入口是 `run.py`：

```bash
uv run python run.py \
  --config-yaml config/ETT-small/ETTm1/model_config_lgbm_usmd.yaml
```

加载边界：

- `config/config_loader.py` 提供 `load_yaml_config()`；`load_model_config()` 为内部工具函数，供 YAML 加载器读取 `base_config` 指定的 Python 模板。
- YAML 的 `base_config` 指向标准模板，例如 `config.univariate_config` 或 `config.multivariate_config`。
- `run.py` 加载配置实例后再应用 CLI 覆盖，优先级为：Python 模板默认值 < YAML `overrides` < CLI 参数。
- `main.py` 通过 `CONFIG_YAML` 常量直接硬编码配置文件路径，仅提供轻量直跑入口。

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
└── aidc_electricity_computility/
    ├── electricity/
    └── electricity_computility/
```

## 配置数量

| 配置目录 | 配置数 |
|---|---:|
| `ETT-small/ETTm1` | 21 YAML |
| `aidc_electricity_computility/electricity/2026-01-01/route_A` | 12 |
| `aidc_electricity_computility/electricity/2026-01-01/route_B` | 12 |
| `aidc_electricity_computility/electricity/2026-01-07/route_A` | 12 |
| `aidc_electricity_computility/electricity/2026-01-07/route_B` | 12 |
| `aidc_electricity_computility/electricity/2026-03-12/route_A` | 12 |
| `aidc_electricity_computility/electricity/2026-03-12/route_B` | 12 |
| `aidc_electricity_computility/electricity/2026-06-11/A1_01a` | 12 |
| `aidc_electricity_computility/electricity/2026-06-11/A1_201` | 12 |
| `aidc_electricity_computility/electricity/2026-06-11/A1_IT` | 12 |
| `aidc_electricity_computility/electricity/2026-06-11/A3_01e` | 12 |
| `aidc_electricity_computility/electricity/2026-06-11/AIDC/route_A` | 12 |
| `aidc_electricity_computility/electricity/2026-06-11/AIDC/route_B` | 12 |
| `aidc_electricity_computility/electricity/gaoweichao_compare` | 4 |
| `aidc_electricity_computility/electricity_computility/add_training_inference_pod` | 7 |
| `aidc_electricity_computility/electricity_computility/add_training_jobs` | 7 |

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

- 目标文件：`df_power.csv`
- 时间列：`count_data_time`
- 目标列：`h_total_use`
- 频率：`5min`
- 日期外生：`df_date.csv`、`df_date_future.csv`
- 天气外生：`df_weather.csv`、`df_weather_future.csv`

## CLI 覆盖优先级

配置文件提供默认值；`run.py` 参数在加载后覆盖对应字段。常用覆盖包括：

- 数据：`--data-dir`、`--data-path`、`--target`、`--target-ts-feat`、`--freq`
- 运行：`--is-testing`、`--is-forecasting`、`--now-time`
- 窗口：`--history-days`、`--predict-days`、`--window-days`、`--lags`
- 模型：`--model-type`、`--pred-method`、`--model-params`
- 输出：`--checkpoints-dir`、`--test-results-dir`、`--pred-results-dir`

布尔参数支持 `1/0`、`true/false`、`yes/no`、`on/off`。

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

- `usmdo`
- `usmd`
- `usmr`
- `usmdr`

多变量策略可显式传入：

- `msmd`
- `msmr`
- `msmdr`

默认输出格式为 YAML。
