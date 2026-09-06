# feature_engineering

`feature_engineering/` 是 canonical 唯一特征编译层，含特征与目标变换三件套。

- `compiler.py`：lag、known-future、static、datetime、advanced transformation、visibility proof 与 lineage。
- `spectral.py`：FFT/小波/熵特征纯函数（trailing 窗），供 compiler 的 `advanced.fourier`/`advanced.wavelet` 与 rolling `entropy` 调用。
- `selection.py`：每个训练窗独立拟合的监督特征选择。
- `transform_specs.py`：feature/target transformation 严格配置归一化。
- `transforms/`：目标/特征变换子包，详见 `transforms/README.md`：
  - `pipeline.py`：目标变换栈（calendar normalization → decomposition → scaling，point/quantile 严格逆序恢复，状态按 `(series_id,target)` 隔离）；
  - `scaling.py`：`CanonicalFeatureScaler`，在训练设计上拟合并将同一状态用于预测设计；
  - `windows.py`：按唯一监督标签时间选取 scaler 窗口；分解默认同窗，允许显式较长上下文。

特征只能使用预测原点可见信息。频域/小波特征由 `spectral.py` 提供纯函数实现，经 `features.transformations.advanced.fourier`（trailing 窗 FFT：top-k 振幅/频率/相位 + 谱质心 + 按周期区间的频带能量占比）与 `advanced.wavelet`（trailing 窗 DWT 各分量能量占比）声明启用；两者只在 trailing 可见窗上计算，as-of 由编译器合同保证，可见历史不足窗口长度时 RAISE。rolling.stats 额外支持 `entropy`（香农熵，p=|y|/Σ|y|）。现役中国节假日 source 由 `data_loading/calendar_generator/chinese_holiday.py` 提供，两者不要混淆。

## 缓存与训练态

- `cache.py`：raw-design 内容寻址缓存、进程锁、元数据与载荷校验。源文件/生成器哈希由 `data_loading.sources.provenance` 提供，设计语义、依赖环境及编译链身份仍在本包组合；不等同于配置语义 fingerprint。递归编译链哈希覆盖数据层的日历计算文件，不能只以薄生成器适配函数代表完整实现。
- `CanonicalFeatureSelector` 只在本训练窗拟合，保留的特征索引随训练 artifact 使用；不在全量数据上先选列再做回测。

## 输入输出与对齐

`FeatureCompiler.compile()` 消费物化信息集，返回 `CompiledFeatures`，包含设计值、`FeatureSchema`、lineage 和 `VisibilityProof`。`batch_eligibility()` 检查批编译能力，`compile_batch()` 提供受支持设计的批量编译。single 与 batch 保留不同执行路径，共用规则解析；不能把 provider 依赖设计强制改走 batch，也不能把同一实现自比较当成独立黄金值验证。

known-future 按目标时刻取值；Direct 历史 lag/rolling/diff 的锚点由 `features.transformations.direct.align_to_target` 决定。目标日对齐且 lag 足够深时消费原点前真实历史；越过原点的 observed-past 访问必须显式 provider，不能隐式填补。

目标变换规格归一化（`transform_specs.py`）与实际拟合恢复（`transforms/pipeline.py`）都在本包；runtime 只按 fold 调用。raw-design 缓存不是 `.uv_cache/`：删除依赖下载缓存不应清理模型设计缓存。
