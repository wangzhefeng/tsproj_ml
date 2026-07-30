# AIDC 专项脚本说明

本目录（`config/aidc_electricity_computility/electricity/2026-06-11/scripts/`）集中保存 AIDC 电力/算力链路的 5 个专项脚本：算力指标聚合、算力 df 缺失稠密化、多模型结果融合、领导汇报样例挑选、汇报指标的一次性修整。

缺失填充方法回测脚本 `fill_method_backtest.py` 在 `data_process/`（回测对象是 `data_process.data_aggregate._FILLERS` 注册表）。

## 文件职责

| 文件 | 职责 |
|---|---|
| `computility_process.py` | 从 AIDC 算力原始指标聚合出 `df_computility.csv`，与 `df_power.csv` 合并生成 `df.csv`，并输出筛后的 `df_selected.csv` 与 `feature_selection_report.csv` |
| `fill_missing.py` | 对 `df.csv` 缺失补 0 + 重算派生列，产出稠密版 `df_filled.csv`（不改动原文件） |
| `select_aidc_leadership_days.py` | 基于 `results/results_test/aidc_electricity_computility/electricity/2026-06-11` 选择领导汇报候选日，重绘候选图/最终图，并导出 `leadership_day_summary.csv` 与 `leadership_day_summary_metrics.csv` |
| `fuse_aidc_results.py` | 将两个 AIDC 路由模型的 `cv_plot_df.csv` 按固定 `0.5/0.5` 权重融合，重算 `test_scores_df.csv` 和预测图 |
| `adjust_leadership_day_summary_metrics.py` | 对 `leadership_day_summary_metrics.csv` 做一次性数值调整，输出 `_new` 版本，不改原文件 |

## 运行方式

默认都从仓库根目录执行：

```bash
uv run python config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py
uv run python config/aidc_electricity_computility/electricity/2026-06-11/scripts/fill_missing.py
uv run python config/aidc_electricity_computility/electricity/2026-06-11/scripts/select_aidc_leadership_days.py --build-report
uv run python config/aidc_electricity_computility/electricity/2026-06-11/scripts/fuse_aidc_results.py
uv run python config/aidc_electricity_computility/electricity/2026-06-11/scripts/adjust_leadership_day_summary_metrics.py
```

## 输入输出约定

- `computility_process.py`
  - 默认根目录：`dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load`
  - 场景目录输入：`df_power.csv` + `算力数据/<room>/`
  - 输出：`df_computility.csv`、`df.csv`、`df_selected.csv`、`feature_selection_report.csv`
  - 当前 `df.csv` 为 215 列：`count_data_time`、`h_total_use` + 213 个算力特征
- `fill_missing.py`
  - 默认根目录：`dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load`
  - 输入：各房间目录下的 `df.csv`
  - 输出：同目录 `df_filled.csv`（原始聚合列补 0 + 重算派生，不改动原文件）
- `select_aidc_leadership_days.py`
  - 默认结果根目录：`results/results_test/aidc_electricity_computility/electricity/2026-06-11`
  - 输出目录：`results/leadership_selection/aidc_electricity_computility/2026-06-11`
  - `--build-report` 额外写入：`results/reports/aidc_electricity_computility/2026-06-11`
- `fuse_aidc_results.py`
  - 默认从 `AIDC/route_A`、`AIDC/route_B` 两个场景读取源模型结果
  - 输出目录名固定为 `lightgbm-df_power-usmd-15__lightgbm-df_power-usmdp-2__avg_0505`
- `adjust_leadership_day_summary_metrics.py`
  - 默认输入：`results/reports/aidc_electricity_computility/2026-06-11/leadership_day_summary_metrics.csv`
  - 默认输出：同目录下 `leadership_day_summary_metrics_new.csv`

## 注意事项

- 这 4 个脚本都面向当前 AIDC 数据目录和结果目录约定，不是通用 CLI。
- `computility_process.py` 当前特征分三层：原始聚合统计、状态/结构特征、时序动态特征。
- 多变量房间配置现优先使用 `df_selected.csv`；单变量配置继续使用 `df_power.csv`。
- `adjust_leadership_day_summary_metrics.py` 是一次性修整脚本，默认写 `_new` 文件，避免覆盖原汇报指标。
- `fill_missing.py` 与本目录 `computility_process.py` 同目录导入，复用其常量与派生重算函数，避免逻辑漂移。
