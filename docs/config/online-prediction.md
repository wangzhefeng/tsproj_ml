# 有限历史与全前缀统计的增量预测

`pipeline.online.RollingForecastSession` 管理部署期历史，不训练、不调用 partial_fit。调用方显式提供已观测数据和当前 origin，使用已有 canonical config 与同身份 bundle。

## 首个支持面

- Local、固定频率、单一 target 文件源（source_time 可得性）；无 key 身份列、外生源、天气、研究回放或原始历史训练模式。
- 配置必须显式声明 final `forecast_origin`；在线 origin 不得早于它，且必须保持同一频率网格，避免用未来拟合的模型回看过去。
- 有限历史特征：lag、rolling、difference/percent_change、same-slot、recent-state、trailing Fourier/wavelet，及 datetime。沿用 `minimum_history_rows` 得到保留长度，并以时间网格而非单纯行数截断。
- EWM、expanding、time_since 通过 `HistoryStatisticsProvider` 接入 canonical compiler。初始化/重建先用调用方提供的完整可见前缀构建 `StreamingStatistics`，再截取有限历史；追加只累计新的真实观测，递归预测不会更新观测统计。
- 模型、scaler、selector、目标变换和已冻结校准保持不变。预测仍使用 `SupervisedDesignBuilder`、`SourceRegistry`、`predict_strategy_bundle`；没有第二套特征公式。

## 更新与恢复

- `update(observations, origin=...)` 只接受严格追加、连续、有限的新观测；重复、历史修订、乱序、断档直接 RAISE。候选统计、保留历史与预测可用性全部验证后才一起发布；编译或预测失败也不改变已有状态。origin 为调用方显式声明的观测截止点，不推断现实时间。
- 历史修订或补采须显式 `rebuild(history, origin=...)`，重新验证完整输入后替换状态。不存在自动覆盖或隐式填补。
- `predict()` 只预测当前 origin 的下一段 horizon；更新历史不代表模型已重新训练，也不自动更新校准。
- `state()` 返回 schema 2、config fingerprint、origin、有限历史副本及 statistics 深拷贝（有限历史配置为 None）；`from_state()` 校验配置/原点绑定、算子布局及统计计数/时间网格，再验证预测。schema 1 或缺失统计状态明确拒绝，须从完整历史显式重建，不能仅由尾部数据恢复全前缀语义。序列化由调用方负责，pickle 只可读取可信来源；状态不包含模型副本。
- 内存数据的 source lineage 使用内容摘要、`path=None`；统计额外保存完整前缀哈希链证据。同样的尾部历史但不同的早期历史不会拥有相同统计来源身份。`last_audit` 保留特征可见性证据。

## 精度与资源边界

- EWM 保持 pandas `adjust=True` 与样本步数半衰期，部署环境的 pandas 乘加运算规则不兼容时 RAISE；time_since 仍须右侧真实观测确认峰谷。
- 状态按算子声明 `state_policy`：EWM/time_since 有界；expanding 仅 min/max/min_diff/max_diff 时有界，其他精确统计保留增长前缀并复用原统计内核。保留 DataFrame 有界不代表整个会话内存有界，快照复制与部分统计查询成本会随前缀增长；无吞吐/RSS 性能承诺。
- 显式统计 provider 走 single compiler，Direct 与 recursive 均与完整历史 single 特征/预测作逐位对照。默认 batch 的 rolling/expanding 数值内核存在既有浮点差异（不仅 mean/std，还包括高阶矩），不宣称与 batch 位级相等、不改训练算法；独立测试验证没有引入相对完整历史 single 的额外差异。
- 不改变 canonical 模型配置 fingerprint 或 bundle schema；改变的是在线状态格式及先前拒绝的特征组合。无 provider 的普通训练/编译链不变。
- 不支持在线重训、任意外生发布版本缓冲或 Global 更新。完整前缀的起点由调用方负责提供；历史修订用完整前缀 rebuild，不把有限尾部当成完整历史。
