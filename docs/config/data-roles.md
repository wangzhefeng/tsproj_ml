# 数据角色与外生来源

- `DataSourceSpec.columns` 是进入模型的信息投影视图：每个声明列必须显式归为 target/observed_past/known_future/static/key/ignored；非 ignored 声明列在物理资产中缺失直接 RAISE；未声明列在 registry 边界丢弃，不会隐式入模。所有动态 source 执行严格 as-of。
- 历史有真值、预测期无值的列挂 `observed_past` 角色 + 显式 provider 三选一（`persistence`/`auxiliary`/`provided_scenario`），禁止隐式 persistence；未来可知的列（天气预报/计划表）挂 `known_future`。普通文件源保留既有路径合同；含 `inference_columns` 的天气源按 [weather.md](weather.md) 的历史/未来阶段合同选择文件。
- **lag 深度与 provider 分工**：`min(lags) >= horizon` 的 safe-lag 声明全程消费 `<= forecast_origin` 的真实历史值，provider 机制不介入；浅于 horizon 的 lag 在 source_time 越过原点时必须显式选择 provider，persistence 是零假设兜底。两种声明可同时存在，由模型自学取舍。
- 多文件外生一律走 `data.sources` 多 source 声明，不使用任何语义 hack 借道。

经用户裁决（2026-09-07 全量清除），全仓配置不再使用 `date_type` file source，`date_*.csv`/`df_date*.csv` 数据资产已删除，不生成伪标签；节假日特征唯一载体为 `chinese_holiday` generated source。

## 低频（日/周/月）约定

- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配。
- 月频（`1ME`/`1MS`）已支持，频率规范化位于 `forecasting_core/specs/problem.py`；月频 seasonal-naive 使用月步 offset，不得转换为固定 Timedelta。
- **中国节假日 builtin generator**：`source_type: generated` + `generator: chinese_holiday` + `availability: generator_defined`，列 `is_holiday`（含调休连休）/`holiday_name`（categorical）/`next_holiday_days`（节前倒计时，日历日；超出已知年历取删失哨兵 400，属有文档截断非编造值）。库覆盖 2004 起，覆盖外日期直接 RAISE 不静默降级；每年底国务院发布次年安排后需 `uv add chinese-calendar --upgrade`。审计兜底导出：`scripts/export_chinese_holiday_csv.py`。日频/日内频率适用；月频网格不适用（known_future 逐点精确匹配 RAISE）。新场景启用属语义变更，按消融流程单独验证。
- **Direct 目标日外生对齐**：known_future 外生按 `col(t+h)` 对目标时刻取值；历史锚点由 `features.transformations.direct.align_to_target` 控制（天气组配置为 `false` 即 lag/rolling/diff 冻结在预测原点）。需要逐步消费自身预测时使用 recursive 类策略。
- 低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项。
