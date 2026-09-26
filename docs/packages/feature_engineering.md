# feature_engineering

`feature_engineering/` 是 canonical 唯一特征编译层，含特征与目标变换三件套。

- `compiler.py`：lag、known-future、static、datetime、advanced transformation、visibility proof 与 lineage。
- `seasonal.py`：同槽统计与近期状态的纯函数内核（按步长周期定位槽位、窗口统计），供 compiler 的 `advanced.same_slot`/`advanced.recent_state` 与残差基线（`transformations.seasonal_baseline`）共用；槽位越界或窗口不足由调用方 RAISE，不在内核静默截断。
- `spectral.py`：FFT/小波/熵特征纯函数（trailing 窗），供 compiler 的 `advanced.fourier`/`advanced.wavelet` 与 rolling `entropy` 调用。
- `selection.py`：每个训练窗独立拟合的监督特征选择。
- `transform_specs.py`：feature/target transformation 严格配置归一化。
- `transforms/`：目标/特征变换子包（详见下文 transforms 子包节）：
  - `pipeline.py`：目标变换栈（calendar normalization → decomposition → scaling，point/quantile 严格逆序恢复，状态按 `(series_id,target)` 隔离）；
  - `scaling.py`：`CanonicalFeatureScaler`，在训练设计上拟合并将同一状态用于预测设计；
  - `windows.py`：按唯一监督标签时间选取 scaler 窗口；分解默认同窗，允许显式较长上下文。

特征只能使用预测原点可见信息。频域/小波特征由 `spectral.py` 提供纯函数实现，经 `features.transformations.advanced.fourier`（trailing 窗 FFT：top-k 振幅/频率/相位 + 谱质心 + 按周期区间的频带能量占比）与 `advanced.wavelet`（trailing 窗 DWT 各分量能量占比）声明启用；两者只在 trailing 可见窗上计算，as-of 由编译器合同保证，可见历史不足窗口长度时 RAISE。rolling/expanding 的 stats 白名单为 `{mean, std, min, max, median, skew, kurt, entropy, max_diff, min_diff}`（entropy = 香农熵，p=|y|/Σ|y|），ewm 仅 `{mean, std}`；拼错统计名在编译期 RAISE 并列明合法集（2026-09-26 规范化，此前错误延迟到编译中段）。现役中国节假日 source 由 `data_loading/calendar_generator/chinese_holiday.py` 提供，两者不要混淆。

## 缓存与训练态

- `cache.py`：raw-design 内容寻址缓存、进程锁、元数据与载荷校验。源文件/生成器哈希由 `data_loading.sources.provenance` 提供，设计语义、依赖环境及编译链身份仍在本包组合；不等同于配置语义 fingerprint。递归编译链哈希覆盖数据层的日历计算文件，不能只以薄生成器适配函数代表完整实现。
- `CanonicalFeatureSelector` 只在本训练窗拟合，保留的特征索引随训练 artifact 使用；不在全量数据上先选列再做回测。

## 输入输出与对齐

联通因果通路：`same_slot` 与 `recent_state` 在 single/batch 中共享纯数值内核，并以独立黄金值测试；批编译每个块天气请求单独绑定信息集作用域，不能复用其他原点的缓存帧。`block_weather` 对真实调用块全部 known_future 时刻计算固定 mean/min/max schema，proof 使用整块最晚 available_at。`seasonal.py` 与全部新规格参数自动进入既有 raw-design 编译链哈希，不新增平行缓存身份。

single/batch 均先生成同槽、近期状态及块摘要，再计算 cyclical、interaction 和 polynomial，允许后者引用前者。块天气摘要只在当前信息集的编译作用域内按 `(forecast_origin, series identity, block start)` 复用；每次 single 调用及每个 batch item 都重新建立作用域，同原点实测/预报也不能混用。只编译块内部分 horizon 时仍聚合完整块。完整 H 步的块摘要取数总量为 O(H)，不按每行重复读取整块；每行保留自身的可见性证据。

`FeatureCompiler.compile()` 消费物化信息集，返回 `CompiledFeatures`，包含设计值、`FeatureSchema`、lineage 和 `VisibilityProof`。`batch_eligibility()` 检查批编译能力，`compile_batch()` 提供受支持设计的批量编译。single 与 batch 保留不同执行路径，共用规则解析；不能把 provider 依赖设计强制改走 batch，也不能把同一实现自比较当成独立黄金值验证。

known-future 按目标时刻取值；Direct 历史 lag/rolling/diff 的锚点由 `features.transformations.direct.align_to_target` 决定。目标日对齐且 lag 足够深时消费原点前真实历史；越过原点的 observed-past 访问必须显式 provider，不能隐式填补。

single/batch 都保留请求时间的时区；`generator_defined` 的 proof 必须使用生成帧逐行 `available_at`，不可替换为请求原点。天气的 manifest/raw/normalized 及计算内核进入设计缓存身份，完整天气依赖证据随 source lineage 传递。

目标变换规格归一化（`transform_specs.py`）与实际拟合恢复（`transforms/pipeline.py`）都在本包；runtime 只按 fold 调用。raw-design 缓存不是 `.uv_cache/`：删除依赖下载缓存不应清理模型设计缓存。

## transforms 子包（原 transforms/README.md 并入）

本子包承载训练窗内拟合、预测期复用的目标与特征变换。编排层负责提供显式窗口，本包不决定回测折或输出目录。

- `pipeline.py`：`CanonicalTargetTransform` 及目标变换实现。顺序固定为 calendar normalization → decomposition → scaling；point/quantile 严格逆序恢复，状态按 `(series_id, target)` 隔离。
- `scaling.py`：`CanonicalFeatureScaler`，只在训练设计上拟合，预测设计使用同一状态。
- `windows.py`：`select_transform_history()`，按唯一监督标签时间选取 scaler 窗口；分解默认使用同窗，可依配置显式扩展上下文。

变换组合规格归一化属于父包 `transform_specs.py`，其中分解别名与参数校验委托 `decomposition/configuration/spec.py` 的唯一入口。训练与预测共用已拟合状态；不得在预测输入上重新拟合或以隐式填补掩盖缺失。分解算法内核属于 `decomposition/`，本子包负责按目标/序列接线与严格恢复。

无消费者的迁移方法 `attach_fitted_target_scaler()` 与 `restore_quantile_matrix()` 已退役。调用方使用 `CanonicalTargetTransform` 的训练/恢复接口；quantile 恢复仍按 `(series_id, target)` 隔离并复用 point 的逆变换，不提供旧方法 shim，不改变已保存对象的类路径或状态字段。
