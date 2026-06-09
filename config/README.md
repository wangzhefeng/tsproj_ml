# Config ↔ Dataset 映射分析

## 概览

- 配置目录：`config/aidc_electricitys/`，模板：`config/templates/`
- 数据目录：`dataset/aidc_electricity/`、`dataset/ETT-small/`

---

## 数据集对应配置文件情况

### 1. `dataset/aidc_electricity/2026-01-01/demand_load/lingang_A/`

| 文件 | 状态 |
|---|---|
| `df_power.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 2. `dataset/aidc_electricity/2026-01-01/demand_load/lingang_B/`

| 文件 | 状态 |
|---|---|
| `AIDC_B_dataset.csv`、`df_power.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 3. `dataset/aidc_electricity/2026-01-07/demand_load/lingang_A/`

| 文件 | 状态 |
|---|---|
| `AIDC_A_dataset.csv`、`df_power.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 4. `dataset/aidc_electricity/2026-01-07/demand_load/lingang_B/`

| 文件 | 状态 |
|---|---|
| `AIDC_B_dataset.csv`、`df_power.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 5. `dataset/aidc_electricity/2026-03-12/demand_load/lingang_A/`

| 文件 | 状态 |
|---|---|
| `df_power.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ✅ 有配置（12 个），但路径指向了不存在的 `electricity_work/` |

> **路径问题**：所有 12 个配置的 `data_dir` 写的是 `./dataset/electricity_work/2026-03-12/demand_load/lingang_A`，但 `electricity_work` 目录不存在。实际数据在 `./dataset/aidc_electricity/2026-03-12/demand_load/lingang_A/`。
>
> 已有配置覆盖的模型+方法组合：lgbm × (usmd/usmdo/usmdr/usmr)、xgb × (usmd/usmdo/usmdr/usmr)、catboost × (usmd/usmdo/usmdr/usmr)

### 6. `dataset/aidc_electricity/2026-03-12/demand_load/lingang_B/`

| 文件 | 状态 |
|---|---|
| `df_power.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 7. `dataset/aidc_electricity/2026-06-03/demand_load/A1_01a/`

| 文件 | 状态 |
|---|---|
| `df_power.csv`、`df_power_new.csv`、`df_power2.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 8. `dataset/aidc_electricity/2026-06-03/demand_load/A1_IT/`

| 文件 | 状态 |
|---|---|
| `df_power.csv`、`df_power_new.csv`、`df_power2.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 9. `dataset/aidc_electricity/2026-06-03/demand_load/A3_01e/`

| 文件 | 状态 |
|---|---|
| `df_power.csv`、`df_power_new.csv`、`df_power2.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 10. `dataset/aidc_electricity/aidc_electricity_computility/add_training_inference_pod/`

| 文件 | 状态 |
|---|---|
| `dataset_electricity_with_computility_A.csv` | ✅ 有配置（7 个），但路径有误 |
| `dataset_electricity_with_computility_B.csv` | ❌ 无配置 |

> **路径问题**：配置的 `data_dir` 写的是 `./dataset/aidc_electricity_computility/add_training_inference_pod/`，实际路径是 `./dataset/aidc_electricity/aidc_electricity_computility/add_training_inference_pod/`（缺少 `aidc_electricity/` 前缀）。
>
> 已有配置覆盖的模型+方法：lgbm × (msmd/msmdr/msmr/usmd/usmdo/usmdr/usmr)

### 11. `dataset/aidc_electricity/aidc_electricity_computility/add_training_jobs/`

| 文件 | 状态 |
|---|---|
| `A3_F2_201_A_dataset_new.csv` | ✅ 有配置（7 个），但路径有误 |
| `A3_F2_201_B_dataset_new.csv` | ❌ 无配置 |

> **路径问题**：配置的 `data_dir` 写的是 `./dataset/aidc_electricity_computility`，实际路径是 `./dataset/aidc_electricity/aidc_electricity_computility/add_training_jobs/`（缺少 `aidc_electricity/` 前缀和 `add_training_jobs/` 子目录）。
>
> 已有配置覆盖的模型+方法：lgbm × (msmd/msmdr/msmr/usmd/usmdo/usmdr/usmr)

### 12. `dataset/aidc_electricity/gaoweichao/2026-04-03/demand_load/`

| 文件 | 状态 |
|---|---|
| `aidc_df.csv`（仅有目标数据，无外生数据） | ❌ 无配置 |

### 13. `dataset/aidc_electricity/gaoweichao/2026-04-03/demand_load/both/`

| 文件 | 状态 |
|---|---|
| `aidc_df.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ❌ 无配置 |

### 14. `dataset/aidc_electricity/gaoweichao/2026-04-09/demand_load/both/`

| 文件 | 状态 |
|---|---|
| `aidc_df.csv`、`df_date.csv`、`df_date_future.csv`、`df_weather.csv`、`df_weather_future.csv` | ✅ 有配置（4 个），但路径指向了不存在的 `electricity_work/` |

> **路径问题**：配置的 `data_dir` 写的是 `./dataset/electricity_work/2026-04-09/demand_load/both`，但 `electricity_work` 目录不存在。实际数据在 `./dataset/aidc_electricity/gaoweichao/2026-04-09/demand_load/both/`。
>
> 已有配置覆盖的模型+方法：catboost × (usmd/usmdo/usmdr/usmr)

### 15. `dataset/ETT-small/`

| 文件 | 状态 |
|---|---|
| `ETTh1.csv`、`ETTh2.csv`、`ETTm1.csv`、`ETTm2.csv` | ❌ 无配置 |

---

## 汇总

### 已有配置的数据集（4 组，30 个配置文件）

| 数据集 | 配置文件数 | 路径是否正确 |
|---|---|---|
| `2026-03-12/demand_load/lingang_A` | 12 | ❌ 路径指向不存在的 `electricity_work/` |
| `aidc_electricity_computility/add_training_inference_pod` (A) | 7 | ❌ 缺少 `aidc_electricity/` 前缀 |
| `aidc_electricity_computility/add_training_jobs` (A) | 7 | ❌ 缺少 `aidc_electricity/` 前缀和子目录 |
| `gaoweichao/2026-04-09/demand_load/both` | 4 | ❌ 路径指向不存在的 `electricity_work/` |

### 缺少配置的数据集（24 个数据目录）

| 数据集 | 备注 |
|---|---|
| `2026-01-01/lingang_A` | 含 `df_power.csv` + 日期/气象外生 |
| `2026-01-01/lingang_B` | 含 `AIDC_B_dataset.csv` + 日期/气象外生 |
| `2026-01-07/lingang_A` | 含 `AIDC_A_dataset.csv` + 日期/气象外生 |
| `2026-01-07/lingang_B` | 含 `AIDC_B_dataset.csv` + 日期/气象外生 |
| `2026-03-12/lingang_B` | 含 `df_power.csv` + 日期/气象外生 |
| `2026-06-03/A1_01a` | 含 `df_power.csv` / `df_power_new.csv` / `df_power2.csv` + 外生 |
| `2026-06-03/A1_IT` | 含 `df_power.csv` / `df_power_new.csv` / `df_power2.csv` + 外生 |
| `2026-06-03/A3_01e` | 含 `df_power.csv` / `df_power_new.csv` / `df_power2.csv` + 外生 |
| `add_training_inference_pod` (B) | `dataset_electricity_with_computility_B.csv` |
| `add_training_jobs` (B) | `A3_F2_201_B_dataset_new.csv` |
| `gaoweichao/2026-04-03/demand_load` | 仅目标数据，无外生 |
| `gaoweichao/2026-04-03/demand_load/both` | 目标 + 日期/气象外生 |
| `ETT-small` (4 个文件) | ETTh1/ETTh2/ETTm1/ETTm2，经典基准数据集 |

### 路径问题（全部 30 个配置文件都存在路径错误）

所有现有配置引用的 `data_dir` 都指向实际不存在的目录：

| 配置位置 | 配置中的路径 | 实际路径 |
|---|---|---|
| `aidc_electricitys/aidc_electricity_computility/aidc_electricity/` | `./dataset/electricity_work/2026-03-12/...` | `./dataset/aidc_electricity/2026-03-12/...` |
| `aidc_electricitys/aidc_electricity_computility/add_training_inference_pod/` | `./dataset/aidc_electricity_computility/...` | `./dataset/aidc_electricity/aidc_electricity_computility/...` |
| `aidc_electricitys/aidc_electricity_computility/add_training_jobs/` | `./dataset/aidc_electricity_computility` | `./dataset/aidc_electricity/aidc_electricity_computility/add_training_jobs/` |
| `aidc_electricitys/gaoweichao_compare/` | `./dataset/electricity_work/2026-04-09/...` | `./dataset/aidc_electricity/gaoweichao/2026-04-09/...` |

> 推测原因：数据曾放在 `dataset/electricity_work/` 和 `dataset/aidc_electricity_computility/` 目录下，后来被移入 `dataset/aidc_electricity/` 作为子目录统一管理，但配置文件中的路径未同步更新。
