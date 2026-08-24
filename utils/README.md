# utils 模块说明

`utils/` 保存项目级辅助工具，不承载模型训练或数据业务逻辑。

## 文件职责

| 文件 | 职责 |
|---|---|
| `frequency.py` | 解析 pandas 固定频率，计算步长和每日样本数 |
| `runtime_env.py` | 设置运行期环境变量，目前用于 Matplotlib 缓存目录 |
| `log_util.py` | 项目日志器，输出到控制台和 `logs/<LOG_NAME>/service*` |
| `eval_mask.py` | 评估/绘图阶段的有效点掩码（MAPE、MAPE Accuracy、预测图历史上下文共用） |
| `conformal.py` | CQR 兼容入口；权威实现与 as-of selector 位于 `probabilistic/calibration.py` / `evaluation.py` |
| `quantile.py` | 分位数后处理兼容入口；权威 crossing repair 位于 `probabilistic/postprocessing.py` |
| `parallel_budget.py` | 窗口并行启用时统一压低 quantile/output/model/ensemble 内部并行 |
| `metrics.py` | 回归指标函数库（独立工具，当前主流程未 import，指标计算内联在 `models/` 中） |
| `cv_plot.py` | 滑窗结果绘图辅助（独立工具，当前主流程未 import，绘图内联在 `models/` 中） |

## frequency.py

主要函数：

- `resolve_freq_step_minutes(freq)`：返回频率对应的分钟步长。
- `resolve_samples_per_day(freq)`：返回一天内样本数。

约束：

- 频率必须是固定时长。
- 步长必须为正。
- 当前项目要求频率不超过 1 天。
- 一天必须能被该频率整除。

常见结果：

| `freq` | 样本数/天 |
|---|---:|
| `5min` | 288 |
| `15min` | 96 |
| `1h` | 24 |
| `1D` | 1 |

## runtime_env.py

`ensure_runtime_environment()` 会设置 `MPLCONFIGDIR` 默认值到系统临时目录下的
`tsproj_ml_matplotlib`，避免受用户主目录权限或环境差异影响。

## log_util.py

日志配置：

- `LOG_NAME` 控制日志子目录，默认由入口文件设置。
- `SERVICE_LOG_LEVEL` 控制日志级别，默认 `INFO`。
- 控制台输出到 stderr。
- 文件输出到 `logs/<LOG_NAME>/service`，按天轮转，保留 10 个备份。

导入 `log_util.py` 会创建日志目录；测试或脚本中如需隔离日志，应先设置 `LOG_NAME`。

## eval_mask.py

`build_eval_mask(values, *, mode, percentile, min_value, max_value)`：构造有效点掩码，
MAPE / MAPE Accuracy 计算与预测图历史上下文掩码共用同一实现，避免两处分叉。

- `mode`：`percentile`（默认，正样本下分位）/ `absolute`（`min_value` 绝对下限）/ `combined`（分位与下限取大）。
- `min_value` 仅在 `absolute`/`combined` 模式生效；`max_value` 与 `mode` 正交，所有模式都生效。
- 仅用于评估/绘图排除点，不改数据；滑窗训练窗口清洗见 `data_provider/outlier_handling.py`。

## conformal.py

`compute_nonconformity_scores(y_true, q_low, q_high)`：CQR 逐点 nonconformity score（欠估/过估取大）；三个输入必须精确等长，负 score 合法。

`calibrate_quantile_band(...)` 保留旧调用兼容。主链通过 `CalibrationRecord` 按 forecast origin、标签可得时间、最少窗口和最少 score 双门槛选择最近完整历史窗口；CQR 输出独立 prediction interval，不把边界回写成 quantile。时间序列报告只声明经验 coverage。

## quantile.py

`monotonize_quantile_columns(df)`：迁移期兼容入口。当前三分位路径采用 q50 锚定 repair，修正上下分位数并同步 `predict_value`，q50 changed ratio 必须为 0。
