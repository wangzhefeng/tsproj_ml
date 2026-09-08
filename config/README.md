# config

## 严格原始历史窗口（显式启用）

fixed-step 的 `validation.train_history_steps: W` 表示每折仅读取原点前（含原点）W 个原始目标点；先截断再计算 lag/rolling/expanding。`train_window_steps` 必须等于 `W - minimum_history_rows(config) - horizon + 1`，不足或矛盾直接报错，`history_steps` 仍限制调度候选监督原点总数。缺省不改变原路径或配置身份。

当前支持 Local、point、无 target transform 的单模型 `--backtest-only`；calendar-month、Global、quantile、target transform、Ensemble（含引用该字段的成员）、完整生命周期和 bundle 导出明确拒绝。W 与原点共同确定逐折缓存/checkpoint 边界。离线已填充值按普通值使用，来源审计不自动触发评分排除。

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

经用户裁决（2026-09-07 全量清除），全仓配置不再使用 `date_type` file source，`date_*.csv`/`df_date*.csv` 数据资产已删除，不生成伪标签；节假日特征唯一载体为 `chinese_holiday` generated source。

## 加载严格性与 fingerprint

- `load_yaml_config()` 按互斥字段集合分派返回 `ForecastConfigSpec | EnsembleConfigSpec`；未知字段、重复 YAML key、角色冲突和非法 strategy/chunk 均 RAISE；只接受 canonical schema，legacy 形态一律 RAISE。
- 已删除的公共配置字段（`problem.information_mode`、`output.setting_suffix`、`probabilistic.recursive_propagation`、`probabilistic.schema_version`）重新声明一律 RAISE。
- 预测输入始终执行严格 as-of；监督训练仅通过内部 `target_access=supervised_labels` 放开预测期 target 标签，不放开 known-future 或历史 target revision 的时间边界。递归 quantile 内部固定走 `median_path`。
- canonical fingerprint 只取语义 payload；并行度、日志和输出目录相关字段不进入 fingerprint。结果 identity = 可读前缀 + 12 位 fingerprint；语义相同的配置别名共享 identity，这不是 hash 碰撞。
- 启用目标分解时语义 payload 带 `decomposition_semantics: component_fit_v2`，区分已修复的 STL/MSTL 外推参数语义。分解别名与严格参数校验唯一入口是 `decomposition/configuration/spec.py`；不修改 YAML、不自动重跑或清除旧结果。

## 时间边界

- `now_time` 配置值 = 最后一个已知数据点；日志/文件名的时间戳按 `now_time` 原值。
- **`schedule_mode`**（`RuntimeConfig`，默认 `daily`）：`daily` = 日界对齐（`floor("1D") + 1day` → 次日 00:00，预测下一完整自然日）；`intraday` = 保留调度时刻（从 `now_time` 起 `predict_steps` 步）。
- **`predict_steps` 以 `freq` 为单位计步**：15min 下 1 天 = 96；5min 下 1 天 = 288；日频下 = 天数。`horizon = predict_steps`，不经 `n_per_day` 换算。
- `pd.date_range` 使用 `inclusive="left"`，end 为排除边界——终日 23:55 是最后一个被包含的点。
- OOF `ensemble.oof.gap_steps` 承担验证/训练标签隔离：验证折与训练折的目标标签按 gap 隔离，不以训练折内标签充当验证标签。

## 数据角色与外生来源

- `DataSourceSpec.columns` 是进入模型的信息投影视图：每个声明列必须显式归为 target/observed_past/known_future/static/key/ignored；非 ignored 声明列在物理资产中缺失直接 RAISE；未声明列在 registry 边界丢弃，不会隐式入模。所有动态 source 执行严格 as-of。
- 历史有真值、预测期无值的列挂 `observed_past` 角色 + 显式 provider 三选一（`persistence`/`auxiliary`/`provided_scenario`），禁止隐式 persistence；未来可知的列（天气预报/计划表）挂 `known_future`。普通文件源保留既有路径合同；含 `inference_columns` 的天气源按下述历史/未来阶段合同选择文件。
- **lag 深度与 provider 分工**：`min(lags) >= horizon` 的 safe-lag 声明全程消费 `<= forecast_origin` 的真实历史值，provider 机制不介入；浅于 horizon 的 lag 在 source_time 越过原点时必须显式选择 provider，persistence 是零假设兜底。两种声明可同时存在，由模型自学取舍。
- 多文件外生一律走 `data.sources` 多 source 声明，不使用任何语义 hack 借道。

## 低频（日/周/月）约定

- **freq 必须写 `1D` 而非 `D`**：`default_lags_for_freq` 只认 `1D`，写 `D` 会落回 5min 基准 lags（`[288,576,...]`），与低频数据错配。
- 月频（`1ME`/`1MS`）已支持，频率解析位于 `utils/frequency.py`；月频 seasonal-naive 使用月步 offset，不得转换为固定 Timedelta。
- **中国节假日 builtin generator**：`source_type: generated` + `generator: chinese_holiday` + `availability: generator_defined`，列 `is_holiday`（含调休连休）/`holiday_name`（categorical）/`next_holiday_days`（节前倒计时，日历日；超出已知年历取删失哨兵 400，属有文档截断非编造值）。库覆盖 2004 起，覆盖外日期直接 RAISE 不静默降级；每年底国务院发布次年安排后需 `uv add chinese-calendar --upgrade`。审计兜底导出：`scripts/export_chinese_holiday_csv.py`。日频/日内频率适用；月频网格不适用（known_future 逐点精确匹配 RAISE）。新场景启用属语义变更，按消融流程单独验证。
- **气象文件与列分流**：`history_path` 覆盖完整历史训练/测试区间，并保留实测和预报列。训练读 rt_/cal_rh，滑窗测试预测读同一 history 的 pred_；日历/datetime 按对应时刻生成。`future_path` 仅用于真正未来推理，对应实测列即使存在也忽略。文件选择由显式 `data_phase=historical|future` 决定，不能通过 `target_access` 或文件日期猜测；`inference_columns` 仍声明模型面列到预报物理列的映射，pred_ 声明为 ignored。无未来任务时允许只配置 history，当前活动场景均采用此方式（history 截至 2026-08-31）；不拼接 future，不用 future 补历史覆盖，不隐式回退实测。
- **天气证据边界**：活动文件的 `availability: forecast_origin` 是现行可得性假设，不是供应商发布时间证据。既有离线补值及 ERA5 替代来源保留审计，不能仅凭 pred_ 前缀宣称完全真实 ex-ante 回放；预报缺失/非有限仍 RAISE，不缩窗。
- **Direct 目标日外生对齐**：known_future 外生按 `col(t+h)` 对目标时刻取值；历史锚点由 `features.transformations.direct.align_to_target` 控制（天气组配置为 `false` 即 lag/rolling/diff 冻结在预测原点）。需要逐步消费自身预测时使用 recursive 类策略。
- 低频下 `datetime_features` 去掉 `minute`/`hour`，`datetime_categorical_features` 同步去对应 `dt_*` 项。

## 天气生成合同（分批迁移中）

### 活动天气特征选择

三个 `aidc_load_15min_{daily,rolling,short}`、`aidc_ess_selfuse_load` 与 `aidc_electricity_computility/electricity/2026-08-31/liantong_IT` 的天气组统一使用六项：`rt_tt2`（温度）、`cal_rh`（相对湿度）、`rt_ssr`（辐射）、`rt_ws10`（风速）、`rt_ps`（气压）、`rt_rain`（降雨）。对应预报为 `pred_tt2/pred_rh/pred_ssrd/pred_ws10/pred_ps/pred_rain`；露点只作为湿度派生原料，不再独立入模。无天气基线不添加天气，其他 `aidc_electricity_computility` 场景及 `aidc_power_month` 不在此次特征调整范围。

共享 `extracted/actual` 中上述特征所需原料及预报列的缺口先离线填补，原非缺失值保持不变；ERA5 气压采用 `surface_pressure`（hPa→Pa），不是海平面气压。逐格来源见源 CSV 旁的 `.six_features_repair.json`；再分析替代不等于站点实测，既有预报缺口修补也不构成真实发布时间证据。本次气压已补齐，未启用用户授权的“无法补齐则跳过气压”例外；该例外不得实现为运行时静默降级。未入模的其他原始列不承诺完整。

### Generated weather 配方

`generator: weather` 的目标合同是 `source_type: generated`、
`availability: generator_defined`，使用强类型 `generator_options`；禁止三段文件路径。
模型进程只读取固定本地资产，不联网、不寻找 skill 或凭据，不隐式降级。

- `inputs`：非空列表，每项仅 `manifest` 与 `sha256`，固定资产清单及完整内容身份。
- `location_map`：非空列表，每项 `series_id`（键值列表；无序列键为 `[]`）与 `location_id`。请求地点必须显式覆盖，不广播未知序列。
- `variables`：每项 `name`（输出列）、`input`（规范变量）、`unit` 与 `aggregation`（`point/mean/min/max/integral/sum`）。物理单位须明确；积分与累计量不能当普通均值。
- `native_features`：显式原生网格特征，每项 `name/inputs/operation/window`；操作为 `relative_humidity/rolling_mean/difference`。湿度输入顺序是气温、露点；湿度 window 为 null，滚动/差分使用带单位正时间长度。先在原生时间轴计算，再重采样。
- `temporal`：`timezone/freq/label/closed/upsampling/max_age`；freq 支持 `5min/15min/1h/1D/1ME/1MS`，label 为 `left/right`、closed 为 `left/right`，升频为 `exact/hold/linear`。hold/linear 必须显式正 max_age，exact 为 null；不能借升频填测量缺口。
- `scenario`：`forecast/prior_year_proxy`。forecast 使用 `vintage_policy: latest_complete_snapshot`、proxy 为 null；代理则 vintage_policy 为 null，`proxy` 显式声明 `years: 1`、`leap_day: reject/feb28`、`data_kind: observation/reanalysis`。
- `semantics_version: weather_v1`。未知字段、单位、情景、版本及输出投影不匹配直接拒绝；处理配方只在模型 YAML，manifest 仅记录事实。
- 独立研究回放使用 `semantics_version: weather_research_v1`，必须显式声明 `research.release_delay` 与 `research.rationale`；不改写真实接收/发布事实，bundle 默认部署拒绝。候选配方的校验通过、目标网格覆盖通过、完整生命周期覆盖通过必须分开记录，不能以候选代替活动配置切换验收。

此合同正在按阶段实现；只有配置校验通过不代表生成器或真实资产已经可用。
历史训练样本的未来天气也必须按各自监督原点选择，不能使用该目标期 actual。
日/月使用完整自然区间、派生可得性追溯全部输入；ESS 的小时统计不得变成 5min 点数窗口。
缺少证据、资产或覆盖保持 blocked，不缩短 horizon、不删除特征。已有天气文件尚未迁移，不作为新链路黄金参照。

## 场景数据备注

- 红太阳2025全年滚动回测见 `hongtaiyang_cesuan/README.md`：主矩阵五任务×四个非递归方法，共20份配置；实验另置experiments目录。1月接入真值，2月由版本化配方采用日历条件基线（假设不等于企业复工事实），3月起按采用profile训练；结果复用通用评分/总图/逐窗图。仅经日频门槛的3份负荷配置推广到15min，未运行15min模型；不把参考窗口 `run.py` 结果当作全年产物。

- 算力房间数据文件、预处理权威入口与特征分层见 `config/aidc_electricity_computility/electricity/2026-06-11/scripts/README.md`。算力天气 `cal_rh` 在离线数据准备阶段由 `rt_tt2`/`rt_dt` 按 Magnus–Tetens 公式派生，权威迁移入口为 `config/aidc_electricity_computility/derive_cal_rh.py`；canonical runtime 不做现场派生或插值。
- 2026-08-31 算力场景（A2_IT / A3_IT / liantong_IT / yancheng_IT）YAML 的 `data_dir` 指向 2026-06-11 数据（复用上批数据做配置模板），对应 `dataset/` 下房间目录为空。
