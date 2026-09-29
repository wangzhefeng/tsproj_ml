# 时间几何

`problem.horizon` 以 `freq` 为步长单位。

## 严格原始历史窗口（显式启用）

fixed-step 的 `validation.train_history_steps: W` 表示每折仅读取原点前（含原点）W 个原始目标点；先截断再计算 lag/rolling/expanding。`train_window_steps` 必须等于 `W - minimum_history_rows(config) - horizon + 1`，不足或矛盾直接报错，`history_steps` 仍限制调度候选监督原点总数。缺省不改变原路径或配置身份。

当前支持 Local、point、无 target transform 的单模型 `--backtest-only`；calendar-month、Global、quantile、target transform、Ensemble（含引用该字段的成员）、完整生命周期和 bundle 导出明确拒绝。W 与原点共同确定逐折缓存/checkpoint 边界。离线已填充值按普通值使用，来源审计不自动触发评分排除。

## 回测窗口字段

Fixed-step validation 使用 `history_steps/train_window_steps/fold_count/stride_steps`，均以监督 origin steps 保存；calendar-month 使用 `train_window_days/fold_count/stride_months`，训练窗按原始日数计，每个历史月和最终目标月动态解析 28/29/30/31 步。

rolling 系另有两种形态（2026-09-27 新增）：

- `horizon_mode: sliding_window`：字段同 fixed-step 四件，语义为 `stride_steps < horizon` 的重叠滑窗评估（同一时刻被多折预测）；窗口构造期校验 stride < horizon，否则 RAISE 指引改用 fixed_steps；产物不拼接总图（逐窗图与 csv 照常），禁用 `train_history_steps`。
- `horizon_mode: expanding_window`：字段为 `history_steps/fold_count/stride_steps` 三件，禁止 `train_window_steps` 与 `train_history_steps`；每折训练集取全部合格历史候选、随折扩大；无固定训练窗口语义，暂限 `--backtest-only`（final fit/bundle RAISE）。

final fit 与回测使用相同训练窗口合同，不再隐式切换为全部历史。

## 重训周期

`validation.refit_every` 为非负整数（不接受 bool）：省略或 1 表示每折重训，0 表示仅首折训练，N>1 表示第 1、1+N、1+2N… 折重训。仅 rolling 系支持非默认值；calendar-month 与显式 train_history_steps 拒绝非默认值。
Ensemble 的 OOF 仍独立管理共同拟合窗口，成员配置的非默认 refit_every 显式拒绝，不静默忽略。

复用时同时冻结模型、feature scaler、target transform 和 selector；每折仍按当前 origin 读取可见历史与未来已知输入，校准池仍按时间顺序 apply-before-collect。重训按当折配置训练窗口进行，final fit 无论周期如何都按原训练窗口重新执行。非默认值属于实验语义、进入 fingerprint；显式 1 与省略身份相同。执行证据记录 refitted、fit_origin 和实际拟合窗口，不能把计划窗口误报为已用于训练。

## 时间边界

- `now_time` 配置值 = 最后一个已知数据点；日志/文件名的时间戳按 `now_time` 原值。
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。
- **`predict_steps` 以 `freq` 为单位计步**：15min 下 1 天 = 96；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不经 `n_per_day` 换算。
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——终日 23:55 是最后一个被包含的点。
- OOF `ensemble.oof.gap_steps` 承担验证/训练标签隔离：验证折与训练折的目标标签按 gap 隔离，不以训练折内标签充当验证标签。
