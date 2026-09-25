# 因果特征与残差基线（历史来源：联通四组，场景已退役）

`features.transformations.advanced` 新增三个严格 as-of 变换：前两个只消费原点前（含原点）的目标/observed-past 历史，未来真值扰动不得影响特征值；第三个消费原点时已知的块内未来天气。历史不足或未来天气缺失时 RAISE，不静默缩窗：

- `same_slot`：`columns/period/days/stats`。对每个目标时刻锚点取过去 K 天同槽值统计（`mean/std` 等），特征名 `{col}_slot_{stat}_{k}d`。`period` 为每日周期步数（5min=288），槽位时间越过预测原点直接 RAISE；预热需求为 `max(days)×period`，不因减少其他 rolling 窗口而放宽。
- `recent_state`：`columns/windows/stats`。原点前固定步数窗口（5min 下 6/12/36 = 30min/1h/3h）的水平（`level`）、均值、波动（`std`）、变化（`diff` 首末差）与斜率（`slope`，每步），特征名 `{col}_rs_{stat}_{window}`。始终锚定预测原点，不随 horizon 移动。
- `block_weather`：`columns/stats`。用于块输出策略（MIMO/RecMO/DIRMO/DirRecMO），也支持单步退化；对当前调用块内全部目标时刻的 known_future 天气列聚合 `mean/min/max`，特征名 `{col}_blk_{stat}`。按真实块坐标聚合，禁止用块首值冒充；单步块退化为该步原值。训练读实测列、滑窗测试预测读预报列，沿用现有阶段切换；没有对应 known_future source 的声明直接 RAISE。沿用策略的整除合同，不引入不完整块的新策略语义。

`features.transformations.seasonal_baseline`（`column/period/days`）为独立残差通路：训练标签按各自监督原点 as-of 减去 K 天同槽基线，预测在目标变换恢复后原位加回；递归 provider 始终工作在原始目标空间。它不是 target transform、不走 decomposition，也不改变 `calendar normalization → decomposition → scaling` 顺序；启用后仍受严格原始历史窗口 backtest-only 限制。
