# Config ↔ Dataset 映射说明

## 目录结构

```
config/
├── templates/
│   ├── univariate_config.py         # 单变量配置模板
│   └── multivariate_config.py       # 多变量配置模板
├── aidc_electricity_computility/
│   ├── electricity/                  # 用电需求预测 (lingang 站点)
│   │   ├── 2026-01-01/               # 数据集日期
│   │   │   ├── route_A/              # A 路线 (lingang_A)
│   │   │   └── route_B/              # B 路线 (lingang_B)
│   │   ├── 2026-01-07/
│   │   │   ├── route_A/
│   │   │   └── route_B/
│   │   └── 2026-03-12/
│   │       ├── route_A/
│   │       └── route_B/
│   ├── electricity_computility/      # 用电+算力联合预测
│   │   ├── add_training_inference_pod/  # 7 个 lgbm 配置
│   │   └── add_training_jobs/           # 7 个 lgbm 配置
│   ├── gaoweichao_compare/           # 高伟超对比数据
│   │   └── model_config_cab_*.py     # 4 个 catboost 配置
│   └── it_electricity_computility/   # IT 用电+算力预测
│       ├── A1_01a/                   # 21 个配置 (3模型×7策略)
│       ├── A1_IT/                    # 21 个配置
│       └── A3_01e/                   # 21 个配置
└── ETT-small/
    └── ETTm1/                        # 21 个配置 (3模型×7策略)
```

```
dataset/
└── aidc_electricity_computility/
    ├── electricity/
    │   ├── 2026-01-01/demand_load/{lingang_A,lingang_B}/
    │   ├── 2026-01-07/demand_load/{lingang_A,lingang_B}/
    │   └── 2026-03-12/demand_load/{lingang_A,lingang_B}/
    ├── electricity_computility/
    │   ├── add_training_inference_pod/
    │   └── add_training_jobs/
    ├── gaoweichao_compare/
    └── it_electricity_computility/
        ├── A1_01a/
        ├── A1_IT/
        └── A3_01e/
```

---

## 数据集 ↔ 配置对照表

### electricity 数据组（用电需求预测）

> 每个日期目录下按路线分为 `route_A/`（lingang_A）和 `route_B/`（lingang_B）两个子目录，各含 12 个配置文件。

| 数据集 | 路线 | 配置数 | 状态 |
|---|---|---|---|
| `electricity/2026-01-01/demand_load/lingang_A` | route_A | 12 | ✅ |
| `electricity/2026-01-01/demand_load/lingang_B` | route_B | 12 | ✅ |
| `electricity/2026-01-07/demand_load/lingang_A` | route_A | 12 | ✅ |
| `electricity/2026-01-07/demand_load/lingang_B` | route_B | 12 | ✅ |
| `electricity/2026-03-12/demand_load/lingang_A` | route_A | 12 | ✅ |
| `electricity/2026-03-12/demand_load/lingang_B` | route_B | 12 | ✅ |

- **数据文件**：`df_power.csv` + 日期外生 (`df_date.csv`, `df_date_future.csv`) + 气象外生 (`df_weather.csv`, `df_weather_future.csv`)
- **模型**：LightGBM / XGBoost / CatBoost
- **策略**：4 种单变量策略 (usmdo / usmd / usmr / usmdr)
- **特征开关**：日期 ✅ | 气象 ✅ | 日期时间 ✅ | 滞后 ✅

### electricity_computility 数据组（用电+算力联合预测）

| 数据集 | 配置数 | 状态 |
|---|---|---|
| `electricity_computility/add_training_inference_pod` | 7 | ✅ |
| `electricity_computility/add_training_jobs` | 7 | ✅ |

- **模型**：LightGBM
- **策略**：7 种全部策略 (含多变量)
- **特征开关**：日期 ❌ | 气象 ❌ | 日期时间 ✅ | 滞后 ✅
- **注意**：仅含 A 版数据配置（B 版暂无配置）

### gaoweichao_compare 数据组

| 数据集 | 配置数 | 状态 |
|---|---|---|
| `gaoweichao_compare/` | 4 | ✅ |

- **数据文件**：`aidc_df.csv` + 日期/气象外生
- **模型**：CatBoost
- **策略**：4 种单变量策略

### it_electricity_computility 数据组（IT 用电+算力预测）

| 数据集 | 配置数 | 状态 |
|---|---|---|
| `it_electricity_computility/A1_01a` | 21 | ✅ |
| `it_electricity_computility/A1_IT` | 21 | ✅ |
| `it_electricity_computility/A3_01e` | 21 | ✅ |

- **数据文件**：`df_power.csv` + 日期/气象外生
- **模型**：LightGBM / XGBoost / CatBoost
- **策略**：7 种全部策略
- **特征开关**：日期 ✅ | 气象 ✅ | 日期时间 ✅ | 滞后 ✅
- **now_time**：`2026-06-03T00:00:00`（数据范围 2026/4/8 ~ 2026/6/3）

### ETT-small 数据组

| 数据集 | 配置数 | 状态 |
|---|---|---|
| `ETT-small/ETTm1` | 21 | ✅ |
| `ETT-small/ETTh1` | 0 | ❌ |
| `ETT-small/ETTh2` | 0 | ❌ |
| `ETT-small/ETTm2` | 0 | ❌ |

- **ETTm1 数据文件**：`ETTm1.csv`，7 列多变量时序（15 分钟频率）
- **模型**：LightGBM / XGBoost / CatBoost
- **策略**：7 种全部策略
- **特征开关**：日期 ❌ | 气象 ❌ | 日期时间 ✅ | 滞后 ✅
- **内生变量**：`HUFL, HULL, MUFL, MULL, LUFL, LULL`
- **窗口参数**：`history_days: 365`, `window_days: 30`（适配长序列）

---

## 各数据组关键参数

| 参数 | electricity | electricity_computility | gaoweichao | it_electricity | ETTm1 |
|---|---|---|---|---|---|
| `target_ts_feat` | `count_data_time` | `time` | `time` | `count_data_time` | `date` |
| `target` | `h_total_use` | `y` | `y` | `h_total_use` | `OT` |
| `freq` | `5min` | `5min` | `5min` | `5min` | `15min` |
| `data_path` | `df_power.csv` | 各数据集不同 | `aidc_df.csv` | `df_power.csv` | `ETTm1.csv` |
| date 外生 | ✅ | ❌ | ✅ | ✅ | ❌ |
| weather 外生 | ✅ | ❌ | ✅ | ✅ | ❌ |
| datetime 特征 | ✅ | ✅ | ✅ | ✅ | ✅ |
| lags 特征 | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 配置命名规范

```
model_config_{model_short}_{method_short}[_{variant}].py
```

| 缩写 | 全称 |
|---|---|
| `lgbm` | LightGBM |
| `xgb` | XGBoost |
| `cab` | CatBoost |
| `usmdo` | univariate-single-multistep-direct-output |
| `usmd` | univariate-single-multistep-direct |
| `usmr` | univariate-single-multistep-recursive |
| `usmdr` | univariate-single-multistep-direct-recursive |
| `msmd` | multivariate-single-multistep-direct |
| `msmr` | multivariate-single-multistep-recursive |
| `msmdr` | multivariate-single-multistep-direct-recursive |

- `variant`：用于区分同一数据集的子站点（如 `A` / `B`）
- 所有配置类名统一为 `ModelConfig`，通过 `importlib` 动态加载

---

## 配置文件使用方式

```bash
# electricity 数据 (2026-01-01, route_A, LightGBM + USMD)
uv run python run.py \
  --config-module config.aidc_electricity_computility.electricity.2026-01-01.route_A.model_config_lgbm_usmd_A \
  --config-class ModelConfig \
  --is-testing 1 --is-forecasting 0

# it_electricity 数据 (A1_01a, CatBoost + MSMD)
uv run python run.py \
  --config-module config.aidc_electricity_computility.it_electricity_computility.A1_01a.model_config_cab_msmd \
  --config-class ModelConfig \
  --is-testing 1 --is-forecasting 0

# ETTm1 数据 (LightGBM + USMDO)
uv run python run.py \
  --config-module config.ETT-small.ETTm1.model_config_lgbm_usmdo \
  --config-class ModelConfig \
  --is-testing 1 --is-forecasting 0
```

---

## 添加新配置

1. 从 `config/templates/univariate_config.py`（单变量）或 `multivariate_config.py`（多变量）复制模板
2. 修改 `data_dir`、`data_path`、`target_ts_feat`、`target`、`freq` 指向实际数据
3. 设置 `now_time` 为预测基准时间
4. 开启/关闭需要的特征开关（`enable_date_features` 等）
5. 选择 `model_type` 和 `pred_method`
6. 按命名规范保存到对应目录

也可使用 `scripts/generate_configs.py` 脚本批量生成（修改脚本中的参数后运行 `uv run python scripts/generate_configs.py`）。
