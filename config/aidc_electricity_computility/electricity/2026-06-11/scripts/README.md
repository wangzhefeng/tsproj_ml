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
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-06-11/scripts/computility_process.py
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-06-11/scripts/fill_missing.py
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-06-11/scripts/select_aidc_leadership_days.py --build-report
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-06-11/scripts/fuse_aidc_results.py
env -u PYTHONPATH .venv/bin/python config/aidc_electricity_computility/electricity/2026-06-11/scripts/adjust_leadership_day_summary_metrics.py
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

## 算力特征生成与缺失填充

`computility_process.py` 把 `算力数据/<room>/` 下 training / inference / pod 三类原始指标按 `count_data_time` 聚合（min/max/mean/std/count，可加量再加 sum，gpu_util 加 busy_*，gpu_power_usage 加 high_power_*），再叠加结构衍生列（占比/比值/present 标志）与时序衍生列（diff/rolling），产出 `df_computility.csv`；最后与 `df_power.csv` 按 `count_data_time` 内连接，写出 `df.csv`（215 列：时间 + `h_total_use` + 213 特征）。

特征分三层：

- **原始聚合统计**：按 5 分钟统计 `min/max/mean/std/count`，容量/功率类额外保留 `sum`。
- **状态/结构特征**：`training_active_jobs`、`inference_active_jobs`、`pod_active_count`、`*_gpu_busy_ratio`、`training_high_power_job_ratio`、`training_to_inference_power_ratio`、`training_minus_inference_power_sum`、`pod_overhead_ratio`、`pod_gpu_util_minus_jobs_gpu_util_mean` 等。
- **时序动态特征**：`*_diff_1`、`*_diff_3`、`*_roll_mean_3`、`*_roll_mean_12`。

在 `df.csv` 基础上，脚本会继续执行场景内特征筛选，额外产出两份文件：

- `df_selected.csv`：多变量建模使用的筛后数据，列顺序固定为 `count_data_time`、`h_total_use`、按综合分数降序排列的算力特征。
- `feature_selection_report.csv`：逐列统计和打分明细，包含 `missing_ratio`、`nunique_non_na`、`std_non_na`、`level_*`、`vol_*`、`final_score`、`keep` 和 `drop_reason`。

筛选分两步完成：

- 先做规则过滤：删除缺失率大于 30% 的列、常量列、`_min/_max`、`*_busy_count`、`*_high_power_count`、`*_diff_3`、`*_roll_std_12`。
- 再做相关性打分：同时考虑算力特征与目标负荷的水平相关性（Pearson + Spearman）和波动相关性（`abs(diff_1).rolling(12).mean()` 后的 Pearson + Spearman），按加权总分筛列。

最终保留规则为：先保留 `final_score >= 0.08` 的列；若不足 16 列则补齐到前 16，若超过 64 列则截断到前 64。

`df.csv` 的缺失几乎全部来自 `inference_*`：推理空窗期整段无数据（A1_201 最严重，约 33% 时刻缺推理；最长连续缺口 ~60h）。`pod_*` 全程完整，`training_*` 基本完整，`h_total_use` 与 `count_data_time` 100% 完整。

`fill_missing.py` 对 `df.csv` 做「原始列补 0 + 重算派生」产出稠密版 `df_filled.csv`：

- 162 个原始聚合列 `NaN→0`（语义：无负载 = 0）；
- 丢弃旧派生列，在补 0 的基础列上重算 51 个派生列（保证内部一致），残存滚动热身 NaN 一并补 0；
- `*_present` 标志保留区分：真无负载 → 0；仅功耗指标漏采但负载在 → 1（A3_01e 有约 24% 的推理"缺失"实为功耗漏采，可借 present 降权）；
- 时间列、目标列原样保留，列顺序与原 `df.csv` 一致。

`df_filled.csv` 未自动接入管线（当前配置以 `df_power.csv` 为目标）。多变量场景现应优先使用 `df_selected.csv`，单变量配置继续指向 `df_power.csv`。

## 注意事项

- 这 5 个脚本都面向当前 AIDC 数据目录和结果目录约定，不是通用 CLI。
- `computility_process.py` 当前特征分三层：原始聚合统计、状态/结构特征、时序动态特征。
- 多变量房间配置现优先使用 `df_selected.csv`；单变量配置继续使用 `df_power.csv`。
- `adjust_leadership_day_summary_metrics.py` 是一次性修整脚本，默认写 `_new` 文件，避免覆盖原汇报指标。
- `fill_missing.py` 与本目录 `computility_process.py` 同目录导入，复用其常量与派生重算函数，避免逻辑漂移。
