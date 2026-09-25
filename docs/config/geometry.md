# 时间几何

`problem.horizon` 以 `freq` 为步长单位。

## 严格原始历史窗口（显式启用）

fixed-step 的 `validation.train_history_steps: W` 表示每折仅读取原点前（含原点）W 个原始目标点；先截断再计算 lag/rolling/expanding。`train_window_steps` 必须等于 `W - minimum_history_rows(config) - horizon + 1`，不足或矛盾直接报错，`history_steps` 仍限制调度候选监督原点总数。缺省不改变原路径或配置身份。

当前支持 Local、point、无 target transform 的单模型 `--backtest-only`；calendar-month、Global、quantile、target transform、Ensemble（含引用该字段的成员）、完整生命周期和 bundle 导出明确拒绝。W 与原点共同确定逐折缓存/checkpoint 边界。离线已填充值按普通值使用，来源审计不自动触发评分排除。

## 回测窗口字段

Fixed-step validation 使用 `history_steps/train_window_steps/fold_count/stride_steps`，均以监督 origin steps 保存；calendar-month 使用 `train_window_days/fold_count/stride_months`，训练窗按原始日数计，每个历史月和最终目标月动态解析 28/29/30/31 步。

final fit 与回测使用相同训练窗口合同，不再隐式切换为全部历史。

## 时间边界

- `now_time` 配置值 = 最后一个已知数据点；日志/文件名的时间戳按 `now_time` 原值。
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。
- **`predict_steps` 以 `freq` 为单位计步**：15min 下 1 天 = 96；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不经 `n_per_day` 换算。
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——终日 23:55 是最后一个被包含的点。
- OOF `ensemble.oof.gap_steps` 承担验证/训练标签隔离：验证折与训练折的目标标签按 gap 隔离，不以训练折内标签充当验证标签。
