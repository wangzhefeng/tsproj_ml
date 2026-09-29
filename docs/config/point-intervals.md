# 点预测残差区间

## 合同

用户已确认：启用时返回独立 `PointIntervalForecast`（点预测＋区间）；未声明校准时仍返回原 `PointForecastTensor`。不制造 quantile 列或把区间边界当成模型分位数。

`probabilistic.mode: point` 下显式声明 `calibration.method: absolute_residual`。固定按 `(series_id, target, horizon_step)` 分组，不跨单位、序列或步长池化；字段包括 `target_coverage`、`calibration_windows`、`min_windows`、`min_scores`、`label_availability_delay_steps`，grouping 仅接受 `series_target_horizon`。不接受 CQR 的 interval 引用或区间收缩字段。

## 时序和数学定义

每折先 apply，再 collect。校准记录 origin 必须早于当前 origin，且对应标签加可得性延迟后不得晚于当前 origin；每组取最近的合格窗口。得分为原始单位的绝对残差，半径取 `ceil((n+1)*coverage)` 顺序统计量；超出样本数时明确 `insufficient_rank`，不把秩截断并伪称满足有限样本覆盖保证。

组样本不足时无区间：结果对象半径为 None，long 表区间两端为 NaN，并显式记录 `pi_available=false` 及原因；这不是输入缺失填补。启用且已有足够样本时，端点必须有限、有序。时序相关下名义覆盖率仍须实测，不能承诺无条件保证。

## 生命周期

区间在目标逆变换之后校准；训练器仍输出点预测。回测收集原始单位残差，最终 bundle 保存分组半径与轴身份，部署仅复用冻结状态。状态必须完整覆盖 bundle 的轴，不得把未知序列静默映射到已有序列。
状态保存频率；部署的隐含 forecast origin 不得早于校准原点，预测时间必须处在同一网格，不能仅凭首个预测时刻晚于校准原点放行。

评分单独输出 coverage、width、Winkler 和 coverage gap，并应用与点评估一致的 eval mask。无可用区间时报告零有效点，不伪造概率评分。

## 实施范围

优先接通单模型 fixed-step 完整生命周期、部署和结果校验；未支持的回测/融合组合显式拒绝，不只让 schema 接受却丢失区间。新增配置进入既有 fingerprint，不自动重训或覆盖存量结果。
