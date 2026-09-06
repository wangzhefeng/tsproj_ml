# Config

`config/` 承载全部活动模型 YAML（`schema_version: 2` canonical）与数据工具 YAML。三个 AIDC 15min 负荷场景的 baseline 成员（ST/LightGBM/Ridge × Direct/Recursive/MIMO）由 `add_ensemble/` 的 Latin-square 组合直接引用，不维护重复 member。列族错配的历史配置已经批准移出活动集，内容由 Git 保留，不计入活动集。

## 唯一 schema

单模型顶层：

```text
schema_version/problem/data/features/strategy/estimator/probabilistic/validation/output
```

Ensemble 顶层：

```text
schema_version/problem/data/probabilistic/ensemble/validation/output
```

未知顶层和嵌套字段在 loader 阶段 RAISE。Transformation 只使用 `direct/advanced/feature_scaling/target/datetime_categorical/interactions` 新结构。

Fixed-step validation 使用 `history_steps/train_window_steps/fold_count/stride_steps`，均以监督 origin steps 保存；calendar-month 使用 `train_window_days/fold_count/stride_months`，训练窗按原始日数计并动态解析每月 H。

经用户裁决，无权威日期类型资产的配置已移除 `date_type` source，不生成伪标签。

## 加载严格性与 fingerprint

- `load_yaml_config()` 按互斥字段集合分派返回 `ForecastConfigSpec | EnsembleConfigSpec`；未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE；只接受 canonical schema，legacy 形态一律 RAISE。
- 已删除的公共配置字段（`problem.information_mode`、`output.setting_suffix`、`probabilistic.recursive_propagation`、`probabilistic.schema_version`）重新声明一律 RAISE。
- 预测输入始终执行严格 as-of；监督训练仅通过内部 `target_access=supervised_labels` 放开预测期 target 标签，不放开 known-future 或历史 target revision 的时间边界。递归 quantile 内部固定走 `median_path`。
- canonical fingerprint 只取语义 payload；并行度、日志和输出目录相关字段不进入 fingerprint。结果 identity = 可读前缀 + 12 位 fingerprint；语义相同的配置别名共享 identity，这不是 hash 碰撞。

## 时间边界

- `now_time` 配置值 = 最后一个已知数据点；日志/文件名的时间戳按 `now_time` 原值。
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。
- **`predict_steps` 以 `freq` 为单位计步**：15min 下 1 天 = 96；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不经 `n_per_day` 换算。
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——终日 23:55 是最后一个被包含的点。
- OOF `ensemble.oof.gap_steps` 承担验证/训练标签隔离：验证折与训练折的目标标签按 gap 隔离，不以训练折内标签充当验证标签。

## 数据角色与外生来源

- `DataSourceSpec.columns` 是进入模型的信息投影视图：每个声明列必须显式归为 target/observed_past/known_future/static/key/ignored；非 ignored 声明列在物理资产中缺失直接 RAISE；未声明列在 registry 边界丢弃，不会隐式入模。所有动态 source 执行严格 as-of。
- 历史有真值、预测期无值的列挂 `observed_past` 角色 + 显式 provider 三选一（`persistence`/`auxiliary`/`provided_scenario`），禁止隐式 persistence；未来可知的列（天气预报/计划表）挂 `known_future` 角色并用 `history_path + backtest_path + future_path` 三段路径。
- **lag 深度与 provider 分工**：`min(lags) >= horizon` 的 safe-lag 声明全程消费 `<= forecast_origin` 的真实历史值，provider 机制不介入；浅于 horizon 的 lag 在 source_time 越过原点时必须显式选择 provider，persistence 是零假设兜底。两种声明可同时存在，由模型自学取舍。
- 多文件外生一律走 `data.sources` 多 source 声明，不使用任何语义 hack 借道。

## 低频（日/周/月）约定

- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配。
- 月频（`1ME`/`1MS`）已支持，频率解析位于 `utils/frequency.py`；月频 seasonal-naive 使用月步 offset，不得转换为固定 Timedelta。
- **中国节假日 builtin generator**：`source_type: generated` + `generator: chinese_holiday` + `availability: generator_defined`，列 `is_holiday`（含调休连休）/`holiday_name`（categorical）/`next_holiday_days`（节前倒计时，日历日；超出已知年历取删失哨兵 400，属有文档截断非编造值）。库覆盖 2004 起，覆盖外日期直接 RAISE 不静默降级；每年底国务院发布次年安排后需 `uv add chinese-calendar --upgrade`。审计兜底导出：`scripts/export_chinese_holiday_csv.py`。日频/日内频率适用；月频网格不适用（known_future 逐点精确匹配 RAISE）。新场景启用属语义变更，按消融流程单独验证。
- **气象严格信息集**（aidc_power_month）：历史训练天气为 actual；滑窗测试必须消费独立 ex-ante forecast/proxy 文件 source；正式 future 只允许 forecast|proxy；禁止把测试期实测天气当未来天气。语义由 `DataSpec` 列角色 + SourceRegistry as-of 校验承载。
- **Direct 目标日外生对齐**：known_future 外生按 `col(t+h)` 对目标时刻取值；历史锚点由 `features.transformations.direct.align_to_target` 控制（天气组配置为 `false` 即 lag/rolling/diff 冻结在预测原点）。需要逐步消费自身预测时使用 recursive 类策略。
- 低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项。

## 场景数据备注

- 算力房间数据文件、预处理权威入口与特征分层见 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/README.md`。算力天气 `cal_rh` 在离线数据准备阶段由 `rt_tt2`/`rt_dt` 按 Magnus–Tetens 公式派生，权威迁移入口为 `config/aidc_electricity_computility/derive_cal_rh.py`；canonical runtime 不做现场派生或插值。
- 2026-08-10 算力场景（A2_IT / A3_IT / liantong_IT / yancheng_IT）YAML 的 `data_dir` 指向 2026-06-11 数据（复用上批数据做配置模板），对应 `dataset/` 下房间目录为空。
