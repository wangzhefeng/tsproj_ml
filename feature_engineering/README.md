# feature_engineering

`feature_engineering/` 是 canonical 唯一特征编译层，含特征与目标变换三件套。

- `compiler.py`：lag、known-future、static、datetime、advanced transformation、visibility proof 与 lineage。
- `seasonal.py`：同槽统计与近期状态的纯函数内核（按步长周期定位槽位、窗口统计），供 compiler 的 `advanced.same_slot`/`advanced.recent_state` 与残差基线（`transformations.seasonal_baseline`）共用；槽位越界或窗口不足由调用方 RAISE，不在内核静默截断。
- `spectral.py`：FFT/小波/熵特征纯函数（trailing 窗），供 compiler 的 `advanced.fourier`/`advanced.wavelet` 与 rolling `entropy` 调用。
- `selection.py`：每个训练窗独立拟合的监督特征选择。
- `transform_specs.py`：feature/target transformation 严格配置归一化。
- `transforms/`：目标/特征变换子包，详见 `transforms/README.md`：
  - `pipeline.py`：目标变换栈（calendar normalization → decomposition → scaling，point/quantile 严格逆序恢复，状态按 `(series_id,target)` 隔离）；
  - `scaling.py`：`CanonicalFeatureScaler`，在训练设计上拟合并将同一状态用于预测设计；
  - `windows.py`：按唯一监督标签时间选取 scaler 窗口；分解默认同窗，允许显式较长上下文。

特征只能使用预测原点可见信息。频域/小波特征由 `spectral.py` 提供纯函数实现，经 `features.transformations.advanced.fourier`（trailing 窗 FFT：top-k 振幅/频率/相位 + 谱质心 + 按周期区间的频带能量占比）与 `advanced.wavelet`（trailing 窗 DWT 各分量能量占比）声明启用；两者只在 trailing 可见窗上计算，as-of 由编译器合同保证，可见历史不足窗口长度时 RAISE。rolling.stats 额外支持 `entropy`（香农熵，p=|y|/Σ|y|）。现役中国节假日 source 由 `data_loading/calendar_generator/chinese_holiday.py` 提供，两者不要混淆。

## 设计身份与训练态

磁盘编译缓存已彻底退役：无缓存参数、读写API、块存储或库存CLI，不再创建`_compiled_features/`。设计仅在内存共享；运行身份与来源证据仍保留。runner计划/准备边界见`model_pipeline/README.md`。

- `design_identity.py`：仅计算原始设计身份及来源证据，供内存共享、checkpoint和运行审计使用；不序列化数组、不创建目录。源内容/生成器/依赖环境/编译代码变化仍改变身份，不等于配置语义fingerprint。
- `indexed.py`：规则 local/source-time 数值历史的紧凑编译。lag 以目标时刻/原点偏移取值，rolling/difference 绑定原点，expanding 还绑定本折历史下界；生成 `forecasting_core.design.IndexedDesign`，不持久化 horizon 广播。类别、global、版本化外生、provider 越界和未支持变换继续走严格 single/batch 路径；这不是绕过信息集的另一套输入入口。
- `CanonicalFeatureSelector` 只在本训练窗拟合，保留的特征索引随训练 artifact 使用；不在全量数据上先选列再做回测。

## 输入输出与对齐

联通因果通路：`same_slot` 与 `recent_state` 在 single/batch 中共享纯数值内核，并以独立黄金值测试；批编译每个块天气请求单独绑定信息集作用域，不能复用其他原点的缓存帧。`block_weather` 对真实调用块全部 known_future 时刻计算固定 mean/min/max schema，proof 使用整块最晚 available_at。`seasonal.py` 与全部新规格参数自动进入既有 raw-design 编译链哈希，不新增平行缓存身份。

single/batch 均先生成同槽、近期状态及块摘要，再计算 cyclical、interaction 和 polynomial，允许后者引用前者。块天气摘要只在当前信息集的编译作用域内按 `(forecast_origin, series identity, block start)` 复用；每次 single 调用及每个 batch item 都重新建立作用域，同原点实测/预报也不能混用。只编译块内部分 horizon 时仍聚合完整块。完整 H 步的块摘要取数总量为 O(H)，不按每行重复读取整块；每行保留自身的可见性证据。

`FeatureCompiler.compile()` 消费物化信息集，返回 `CompiledFeatures`，包含设计值、`FeatureSchema`、lineage 和 `VisibilityProof`。`batch_eligibility()` 检查批编译能力，`compile_batch()` 提供受支持设计的批量编译。single 与 batch 保留不同执行路径，共用规则解析；不能把 provider 依赖设计强制改走 batch，也不能把同一实现自比较当成独立黄金值验证。

known-future 按目标时刻取值；Direct 历史 lag/rolling/diff 的锚点由 `features.transformations.direct.align_to_target` 决定。目标日对齐且 lag 足够深时消费原点前真实历史；越过原点的 observed-past 访问必须显式 provider，不能隐式填补。

single/batch 都保留请求时间的时区；`generator_defined` 的 proof 必须使用生成帧逐行 `available_at`，不可替换为请求原点。天气的 manifest/raw/normalized 及计算内核进入设计缓存身份，完整天气依赖证据随 source lineage 传递。

批量 rolling/expanding 使用数组/序列统计，不再逐原点精确复算旧 pandas 标量结果；允许已验证的浮点误差，不承诺位级相同，更不能据特征 allclose 推断树模型预测等价。历史下界、标签、列顺序和 as-of 仍是严格合同。

single/batch/indexed的统计定义一致：恒定窗口峰度为0，不能沿用pandas窗口内核的-3；用向量化恒定窗掩码修正，不恢复逐原点重扫。全部支持统计量以独立窗口切片覆盖恒定、全零、近恒定、短窗及状态切换边界。

目标变换规格归一化（`transform_specs.py`）与实际拟合恢复（`transforms/pipeline.py`）都在本包；runtime 只按 fold 调用。设计身份校验不属于缓存持久化；模型checkpoint和融合OOF具有独立生命周期。
