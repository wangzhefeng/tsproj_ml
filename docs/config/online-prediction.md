# 有界历史增量预测

`pipeline.online.RollingForecastSession` 管理部署期历史，不训练、不调用 partial_fit。调用方显式提供已观测数据和当前 origin，使用已有 canonical config 与同身份 bundle。

## 首个支持面

- Local、固定频率、单一 target 文件源（source_time 可得性）；无 key 身份列、外生源、天气、研究回放或原始历史训练模式。
- 配置必须显式声明 final `forecast_origin`；在线 origin 不得早于它，且必须保持同一频率网格，避免用未来拟合的模型回看过去。
- 有限历史特征：lag、rolling、difference/percent_change、same-slot、recent-state、trailing Fourier/wavelet，及 datetime。沿用 `minimum_history_rows` 得到保留长度，并以时间网格而非单纯行数截断。
- 全前缀 EWM、expanding、time_since 尚未接入本入口，明确拒绝；`feature_engineering/streaming_statistics.py` 已有独立状态内核，但不等于部署支持。不能通过尾部截断伪装支持。
- 模型、scaler、selector、目标变换和已冻结校准保持不变。预测仍使用 `SupervisedDesignBuilder`、`SourceRegistry`、`predict_strategy_bundle`；没有第二套特征公式。

## 更新与恢复

- `update(observations, origin=...)` 只接受严格追加、连续、有限的新观测；重复、历史修订、乱序、断档、未来标签直接 RAISE，失败不改变已有状态。
- 历史修订或补采须显式 `rebuild(history, origin=...)`，重新验证完整输入后替换状态。不存在自动覆盖或隐式填补。
- `predict()` 只预测当前 origin 的下一段 horizon；更新历史不代表模型已重新训练，也不自动更新校准。
- `state()` 返回含 schema、config fingerprint、origin 和有限历史副本的状态；`from_state(config, bundle, state)` 验证身份并重建。序列化由调用方负责，pickle 只可读取可信来源；状态不包含模型副本。
- 内存数据的 source lineage 使用内容摘要、`path=None`，不把更新后的数据假称为原 YAML 路径内的文件内容。返回 `last_audit` 保留现有特征可见性证据。

状态内存合同按算子声明：可有界维护的统计使用增量状态；精确中位数等允许状态增长，不能改用近似计算。当前统计提供器仍待主链接线，扩展在线测试尚未通过；本入口没有在线模型刷新、任意外生发布版本缓冲或跨序列在线更新，也没有对吞吐/RSS 作未经测量的承诺。
