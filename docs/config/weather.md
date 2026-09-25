# 天气合同

## 气象文件与列分流

`history_path` 覆盖完整历史训练/测试区间，并保留实测和预报列。训练读 rt_/cal_rh，滑窗测试预测读同一 history 的 pred_；日历/datetime 按对应时刻生成。`future_path` 仅用于真正未来推理，对应实测列即使存在也忽略。文件选择由显式 `data_phase=historical|future` 决定，不能通过 `target_access` 或文件日期猜测；`inference_columns` 仍声明模型面列到预报物理列的映射，pred_ 声明为 ignored。无未来任务时允许只配置 history，当前活动场景均采用此方式（history 截至 2026-08-31）；不拼接 future，不用 future 补历史覆盖，不隐式回退实测。

**天气证据边界**：活动文件的 `availability: forecast_origin` 是现行可得性假设，不是供应商发布时间证据。既有离线补值及 ERA5 替代来源保留审计，不能仅凭 pred_ 前缀宣称完全真实 ex-ante 回放；预报缺失/非有限仍 RAISE，不缩窗。

## 天气生成合同（分批迁移中）

### 活动天气特征选择

活动场景 `aidc_load_15min_short` 的天气组统一使用六项：`rt_tt2`（温度）、`cal_rh`（相对湿度）、`rt_ssr`（辐射）、`rt_ws10`（风速）、`rt_ps`（气压）、`rt_rain`（降雨）。对应预报为 `pred_tt2/pred_rh/pred_ssrd/pred_ws10/pred_ps/pred_rain`；露点只作为湿度派生原料，不再独立入模。无天气基线不添加天气。

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
