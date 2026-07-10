# data_provider 模块说明

`data_provider/` 负责把原始 CSV 转成主流程可用的历史/未来时间序列数据，并提供两类异常处理工具。

## 文件职责

| 文件 | 职责 |
|---|---|
| `data_loader.py` | 读取目标、日期、天气 CSV；构造历史/未来时间轴；统一目标列为 `y` |
| `fill_missing.py` | 对 `df.csv` 缺失补 0 + 重算派生列，产出稠密版 `df_filled.csv`（不改动原文件） |
| `outlier_handling.py` | 滑窗测试训练段异常处理，不修改测试真实值 |
| `outlier_process.py` | 离线负荷数据异常标记、清洗和图片输出 |

## DataLoader 输入契约

核心字段来自配置实例：

- `data_dir`：数据目录，`main.Model.__init__` 会转成 `Path`。
- `data_path`：目标序列 CSV。
- `target_ts_feat`：目标序列时间列。
- `target`：目标值列，历史处理后映射为 `y`。
- `freq`：固定频率，例如 `5min`、`15min`、`1h`。

可选外生文件：

- 日期：`date_history_path`、`date_future_path`、`date_ts_feat`
- 天气：`weather_history_path`、`weather_future_path`、`weather_ts_feat`

日期和天气外生文件同时存在历史/未来版本时，会先按时间列纵向拼接、排序、去重形成 canonical 全量表，再按 `forecast_start_time` 显式切成历史片段和未来片段供两个阶段分别使用。

## 历史与未来边界

`main.Model` 计算时间窗口后传给 `DataLoader`：

- 历史区间：`[now_time - history_days, now_time)`
- 预测区间：`[now_time, now_time + predict_days)`

`process_history_data()` 会按历史区间构造模板并映射真实目标；`process_future_data()`
只构造未来模板和外生特征，不读取未来真实目标。

## 异常处理边界

`outlier_handling.py` 用于滑窗测试训练段：

- 默认关闭。
- 只作用于每个测试窗口的训练片段。
- 输出 `train_outlier_report.csv` 所需的标准列。
- 不改变测试段真实值，避免评估泄漏。

`outlier_process.py` 用于离线处理原始负荷文件：

- 输入通常是一个 `df_power.csv`。
- 输出保存在源数据目录。
- 标记结果：`df_power_outlier_detection.csv`
- 清洗结果：`df_power_remove_outlier.csv`
- 图片结果：`df_power_anomalies.png`

离线清洗输出的文件名可变更，但 CSV 内容契约不应随意改变。

## 算力数据合并与缺失填充

算力预处理脚本已移动到 [scripts/aidc/computility_process.py](/Users/wangzf/projects/tsproj_ml/scripts/aidc/computility_process.py)。

该脚本把 `算力数据/<room>/` 下 training / inference / pod 三类原始指标
按 `count_data_time` 聚合（min/max/mean/std/count，可加量再加 sum，gpu_util 加 busy_*，
gpu_power_usage 加 high_power_*），再叠加结构衍生列（占比/比值/present 标志）与时序
衍生列（diff/rolling），产出 `df_computility.csv`；最后与 `df_power.csv` 按
`count_data_time` 内连接，写出 `df.csv`（215 列：时间 + `h_total_use` + 213 特征）。

增强后的代表性特征包括：

- 状态特征：`training_active_jobs`、`inference_active_jobs`、`pod_active_count`
- 负载占用特征：`training_gpu_busy_ratio`、`inference_gpu_busy_ratio`、
  `training_high_power_job_ratio`
- 结构耦合特征：`training_to_inference_power_ratio`、
  `training_minus_inference_power_sum`、`pod_overhead_ratio`、
  `pod_gpu_util_minus_jobs_gpu_util_mean`
- 时序动态特征：`*_diff_1`、`*_diff_3`、`*_roll_mean_3`、`*_roll_mean_12`

在 `df.csv` 基础上，脚本会继续执行场景内特征筛选，额外产出两份文件：

- `df_selected.csv`：多变量建模使用的筛后数据，列顺序固定为
  `count_data_time`、`h_total_use`、按综合分数降序排列的算力特征。
- `feature_selection_report.csv`：逐列统计和打分明细，包含
  `missing_ratio`、`nunique_non_na`、`std_non_na`、`level_*`、`vol_*`、
  `final_score`、`keep` 和 `drop_reason`。

筛选分两步完成：

- 先做规则过滤：删除缺失率大于 30% 的列、常量列、`_min/_max`、
  `*_busy_count`、`*_high_power_count`、`*_diff_3`、`*_roll_std_12`。
- 再做相关性打分：同时考虑算力特征与目标负荷的水平相关性
  （Pearson + Spearman）和波动相关性
  （`abs(diff_1).rolling(12).mean()` 后的 Pearson + Spearman），按加权总分筛列。

最终保留规则为：先保留 `final_score >= 0.08` 的列；若不足 16 列则补齐到前 16，
若超过 64 列则截断到前 64。

`df.csv` 的缺失几乎全部来自 `inference_*`：推理空窗期整段无数据（A1_201 最严重，
约 33% 时刻缺推理；最长连续缺口 ~60h）。`pod_*` 全程完整，`training_*` 基本完整，
`h_total_use` 与 `count_data_time` 100% 完整。

`fill_missing.py` 对 `df.csv` 做「原始列补 0 + 重算派生」产出稠密版 `df_filled.csv`：

- 162 个原始聚合列 `NaN→0`（语义：无负载 = 0）；
- 丢弃旧派生列，在补 0 的基础列上重算 51 个派生列（保证内部一致），残存滚动热身
  NaN 一并补 0；
- `*_present` 标志保留区分：真无负载 → 0；仅功耗指标漏采但负载在 → 1
  （A3_01e 有约 24% 的推理"缺失"实为功耗漏采，可借 present 降权）；
- 时间列、目标列原样保留，列顺序与原 `df.csv` 一致。

`df_filled.csv` 未自动接入管线（当前配置以 `df_power.csv` 为目标）。多变量场景现应
优先使用 `df_selected.csv`，单变量配置继续指向 `df_power.csv`。相关脚本默认根目录均为
`dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load`。
