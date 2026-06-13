# Config 说明

`config/` 保存模型运行配置、配置模板和配置加载器。当前配置文件均为 Python
`@dataclass`，类名统一为 `ModelConfig`。

## 加载规则

正式运行入口是 `run.py`：

```bash
uv run python run.py \
  --config-module config.templates.univariate_config \
  --config-class ModelConfig
```

加载边界：

- `config/config_loader.py` 只提供 `load_model_config()`，导入时不解析命令行。
- `run.py` 加载配置实例后再应用 CLI 覆盖。
- `main.py` 只支持 `--config-module` / `--config-class` 两个最小参数。
- `--config-module` 是传给 `importlib.import_module()` 的字符串模块路径。

## 目录

```text
config/
├── config_loader.py
├── templates/
│   ├── univariate_config.py
│   └── multivariate_config.py
├── ETT-small/
│   └── ETTm1/
└── aidc_electricity_computility/
    ├── electricity/
    └── electricity_computility/
```

## 配置数量

| 配置目录 | 配置数 |
|---|---:|
| `ETT-small/ETTm1` | 21 |
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

`scripts/generate_configs.py` 用于从 dataset 叶子目录批量生成配置。生成前建议使用
`--dry-run` 检查输出路径和文件名。

默认模型矩阵：

- `lightgbm` -> `lgbm`
- `xgboost` -> `xgb`
- `catboost` -> `cab`

默认单变量策略矩阵：

- `usmdo`
- `usmd`
- `usmr`
- `usmdr`
