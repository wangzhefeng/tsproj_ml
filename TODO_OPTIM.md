# 项目待优化问题

本文档记录尚未解决的工程问题、技术债和配置体系优化事项。业务实验与场景研究事项继续记录在 [`docs/TODO.md`](docs/TODO.md)。

## 维护约定

- 状态使用：`待处理`、`进行中`、`阻塞`、`已完成`。
- 每个问题必须记录当前事实、影响、建议方向、风险和验收标准。
- `已完成` 必须补充实际执行的验证命令及结果，不能只记录实现计划。
- 涉及预测语义、数据口径或产物兼容性的修改，实施前需要单独确认。
- 问题开始实施后，如方案发生偏差，在原条目中追加实施记录，不回改已确认的历史结论。

---

## OPT-001：移除公共配置中的 `problem.information_mode`

- **状态**：已完成
- **优先级**：P2
- **类型**：配置 schema 简化 / 信息泄漏边界加固
- **涉及范围**：canonical 配置、信息集合同、fingerprint、配置生成器、测试与文档

### 当前事实

1. `problem.information_mode` 是 canonical schema 的必填字段；缺失时配置解析直接 RAISE，见 `forecasting_core/specs/config.py:351-359`。
2. 该字段允许 `forecast`、`nowcast`、`oracle` 三个值，见 `forecasting_core/specs/problem.py:108-157`。
3. 正常训练、回测和正式预测配置均使用 `forecast`，该字段没有承担现役配置间的语义区分。
4. 当前 `nowcast` 与 `forecast` 使用相同的数据可见性规则：二者都会屏蔽预测期目标真实值，见 `tests/test_information_set_registry.py:467-488`。
5. `oracle` 用于框架内部读取训练标签：它允许目标数据覆盖预测期，并放开带 `available_at` 数据的预测原点限制，见 `data_loading/registry.py:240-247`、`data_loading/registry.py:414-434` 和 `model_forecasting/design.py:356-362`。
6. 该字段进入 canonical semantic payload，因此会影响配置 fingerprint 和结果目录 identity，见 `forecasting_core/specs/problem.py:168-177` 与 `forecasting_core/specs/config.py:245-273`。

### 问题

- 数量众多的 YAML 重复声明固定值 `forecast`，增加配置噪声和批量维护成本。
- `nowcast` 看似提供独立能力，但当前没有独立运行语义，容易误导配置使用者。
- `oracle` 是读取监督标签所需的内部权限，却可以从公共 YAML 选择，形成不必要的未来信息泄漏入口。
- 预测信息边界属于系统必须强制保证的不变量，不宜由每份模型配置自行选择。

### 建议方向

1. 从公共 YAML schema 和 `ForecastProblemSpec` 中移除 `information_mode`。
2. 正常训练特征、回测输入和正式预测统一强制执行严格的 `forecast` as-of 边界。
3. 保留内部“构造预测输入”和“读取监督标签”的权限区分，但不要再由用户配置；内部命名应直接表达访问目的，而不是继续暴露 `oracle` 模式。
4. 删除没有独立语义的 `nowcast`。未来确有实时估计需求时，重新设计目标时点、数据到达延迟和回测模拟合同，不复用当前空壳选项。
5. 批量清扫配置、生成器、测试和长期文档中的该字段，并保持 canonical schema 对未知字段严格 RAISE。

### 实施风险

- 删除字段会改变所有单模型配置的 semantic fingerprint，现有结果目录、OOF cache 和引用式 Ensemble 的身份关联需要统一评估和处理。
- 训练阶段仍必须取得预测期真实标签；收紧公共接口时不能破坏内部标签读取链路。
- 配置数量大，必须通过生成器零漂移检查和全量配置审计确认没有漏改。
- 如果直接给字段增加默认值而不删除，虽然可减少 YAML 内容，但仍会保留无效选项和双重表达，不是建议的最终方案。

### 验收标准

- [x] canonical 单模型与引用式 Ensemble YAML 不再声明 `problem.information_mode`。
- [x] 公共配置无法启用 `oracle` 或绕过预测原点的 as-of 限制。
- [x] 训练标签仍能通过仅限内部的明确接口读取。
- [x] `forecast` 与 `oracle` 的现有信息可见性测试被重写为“预测输入”与“训练标签”合同测试。
- [x] 配置生成器、配置审计脚本及长期文档同步完成。
- [x] 明确记录 fingerprint、结果目录和 OOF cache 的迁移或失效策略。
- [x] 全量配置审计、架构门禁及 unittest 测试全部通过。

### 实施记录

- `ForecastProblemSpec`、单模型 parser 与 Ensemble parser 已删除公共字段；重新声明按未知字段 RAISE。
- `InformationSetRequest` 改为内部 `target_access=history_only|supervised_labels`。后者只放开 `forecast_times` 上的 target 标签；known-future 和历史 target revision 仍受 `available_at <= forecast_origin` 约束。故障注入测试曾复现历史修订泄漏 `[99, 2, 3]`，修复后恢复为 `[1, 2, 3]`。
- 5,150 份活动 YAML 与 AIDC 生成器已清扫。该语义字段删除会使全部活动配置获得新 fingerprint；`docs/validation_geometry_migration_manifest.yaml` 已全量重算。旧结果目录与 OOF cache 不重命名、不复用，后续运行按新 identity/cache key 重新生成；物理清理旧结果不在本项范围内。
- 定向合同测试：`python -m unittest tests.test_forecast_problem_spec tests.test_information_set_registry tests.test_ensemble_loader`，40 项相关信息集/Ensemble 测试通过；共同全量证据见 OPT-004。

---

## OPT-002：移除公共配置中的 `output.setting_suffix`

- **状态**：已完成
- **优先级**：P1
- **类型**：死参数清理
- **涉及范围**：输出配置 schema、现役 YAML、配置生成器、配置测试与结果元数据

### 当前事实与问题

1. `output.setting_suffix` 仍被 `OutputSpec` 接受，见 `forecasting_core/specs/output.py:16-33`。
2. 现役配置中既有空值也有非空值，但生产输出路径只使用 `scenario_subpath` 和 `result_identity`，没有读取 `setting_suffix`，见 `model_forecasting/runtime.py:266-315` 与 `model_ensemble/runtime.py:57-95`。
3. 该字段目前只被配置生成器写入、被部分测试当作配置标签断言；它不会改变模型身份或结果目录，容易让使用者误以为后缀已经生效。

### 处理方式

1. 从 `OutputSpec` 的公共字段及 `output.identity` 嵌套字段中删除 `setting_suffix`，删除后再次声明必须按未知字段 RAISE。
2. 清扫全部单模型与 Ensemble YAML、配置生成器和测试中的该字段。
3. 原来依赖 `setting_suffix` 区分配置的测试，改为断言真实语义字段，例如 decomposition、Direct layout、CQR 配置或配置文件名。
4. 不为该字段增加兼容读取或静默忽略逻辑，避免继续保留无运行语义的入口。

### 实施风险

- resolved config 等结果元数据将不再包含该字段，但预测、训练和输出路径行为不应变化。
- `output` 不进入单模型或 Ensemble fingerprint，因此删除该字段不应改变模型 fingerprint；需要用测试固定这一点。

### 验收标准

- [x] 全部现役 YAML 与配置生成器零引用 `setting_suffix`。
- [x] 公共配置声明 `output.setting_suffix` 时明确 RAISE。
- [x] 单模型与 Ensemble 的输出路径在删除前后保持一致。
- [x] 删除前后的单模型与 Ensemble fingerprint 分别保持一致。
- [x] 原有配置标签测试已改为断言真实模型语义。
- [x] 全量配置审计及 unittest 测试通过。

### 实施记录

- `OutputSpec` 顶层及 `output.identity` 嵌套合同均已删除该字段；5,150 份活动 YAML 和生成器零引用，重新声明时严格 RAISE。
- 原标签测试已改为断言 decomposition、Direct layout、CQR 和实际配置文件集合。生产输出路径代码未因本项修改；单模型既有 output 非语义测试及新增 Ensemble output 非语义测试均通过。
- `output` 本来就不进入单模型或 Ensemble fingerprint，因此本项自身不改变 fingerprint；本批次出现的新 identity 来自 OPT-001/003/004 的语义 payload 删除，而不是输出后缀清理。
- 定向合同测试：`python -m unittest tests.test_public_config_field_removal tests.test_calendar_month_horizon tests.test_ess_quantile_config_matrix` 通过；共同全量证据见 OPT-004。

---

## OPT-003：移除公共配置中的 `probabilistic.recursive_propagation`

- **状态**：已完成
- **优先级**：P2
- **类型**：伪配置项清理
- **涉及范围**：概率配置 schema、quantile 配置、部署态概率合同、配置生成器、测试与文档

### 当前事实与问题

1. 现役 quantile 配置声明的值全部是 `median_path`。
2. 运行时也只支持 `median_path`；其他值直接 RAISE，见 `forecasting_core/probabilistic_spec.py:211-220`。
3. 只有一个合法值的字段没有形成真实配置能力，却让使用者误以为可以选择其他递归传播方式。

### 处理方式

1. 从 YAML 级 `ProbabilisticConfigSpec` 允许字段和部署映射解析入口中删除 `recursive_propagation`；公共配置再次声明时按未知字段 RAISE。
2. 清扫全部 quantile YAML、配置生成器、配置审计和长期文档中的该字段。
3. 内部概率合同若仍需记录递归传播机制，可固定为内部常量 `median_path`，不得重新暴露给 YAML。
4. 未来实现第二种传播算法后，再以经过设计和测试的新公共合同引入，不预留无效参数。

### 实施风险

- 清扫会改变所有原先显式声明该字段的 quantile 配置 fingerprint，相关结果目录和缓存需要按新 fingerprint 处理。
- 必须覆盖 recursive、RECMO 等消费递归路径的策略，确认字段删除后仍固定使用中位数路径。

### 验收标准

- [x] 全部现役 YAML 与配置生成器零引用 `recursive_propagation`。
- [x] 公共配置声明该字段时明确 RAISE。
- [x] quantile 递归预测继续固定使用 `median_path`，并有策略级测试证明结果等价。
- [x] 新 fingerprint 及相关结果、缓存处理方式已经记录。
- [x] 全量配置审计及 unittest 测试通过。

### 实施记录

- YAML 级 `ProbabilisticConfigSpec` 与部署 mapping 解析入口已删除该字段；176 份活动 quantile 配置完成清扫，重新声明时严格 RAISE。
- 部署态 `ProbabilisticSpec` 仍保存内部 `recursive_propagation="median_path"`，point/quantile mapping 均由代码固定，不允许 YAML 覆盖。
- 原先显式声明该字段的 quantile 配置 fingerprint 已随新 canonical payload 失效；旧结果和缓存不复用，迁移 manifest 已重算。
- `tests.test_public_config_field_removal`、概率合同测试及覆盖 recursive/RECMO 的全量测试通过；共同全量证据见 OPT-004。

---

## OPT-004：移除公共配置中的 `probabilistic.schema_version`

- **状态**：已完成
- **优先级**：P2
- **类型**：内部产物字段与公共配置解耦
- **涉及范围**：概率配置 schema、部署态概率合同、少量现役 YAML、测试与文档

### 当前事实与问题

1. 只有少量现役配置显式声明 `probabilistic.schema_version: 1`，其余配置依赖相同的默认值。
2. 当前代码只接受版本 `1`，见 `forecasting_core/probabilistic_spec.py:205-212` 和 `forecasting_core/probabilistic_spec.py:392-410`。
3. 顶层 `schema_version: 2` 已负责 YAML 配置格式分派；概率部署产物可以继续拥有独立内部版本，但该内部版本不应由模型 YAML 选择。

### 处理方式

1. 从 YAML 级 `ProbabilisticConfigSpec` 允许字段和概率映射解析入口中删除 `schema_version`；公共配置再次声明时按未知字段 RAISE。
2. 清扫仍显式声明该字段的 YAML，并同步配置审计、测试和长期文档。
3. 在部署态 `ProbabilisticSpec` 或 bundle 中继续由代码固定并持久化内部版本 `1`，加载产物时仍严格校验版本。
4. 将“公共配置版本”和“部署产物内部版本”分别测试，避免后续再次混用。

### 实施风险

- 清扫会改变原先显式声明该字段的少量配置 fingerprint，相关结果目录和缓存需要按新 fingerprint 处理。
- 删除公共入口时不能误删 bundle 反序列化所需的内部版本校验。

### 验收标准

- [x] 现役 YAML 零引用 `probabilistic.schema_version`。
- [x] 公共配置声明该字段时明确 RAISE。
- [x] 新生成的概率部署产物仍写入内部版本，已有合法产物仍可按当前合同加载。
- [x] 顶层配置版本与概率产物版本的测试边界清晰且互不依赖。
- [x] 新 fingerprint 及相关结果、缓存处理方式已经记录。
- [x] 全量配置审计及 unittest 测试通过。

### 实施记录

- YAML 级概率合同和部署 mapping 解析入口已删除该字段，11 份活动配置完成清扫；顶层 canonical `schema_version: 2` 保持不变。
- 部署态 `ProbabilisticSpec.schema_version=1` 及反序列化校验完整保留，并由代码固定。公共旧字段拒绝与内部版本保留由独立测试同时覆盖。
- fingerprint 变化按 OPT-001 的统一策略处理：迁移 manifest 重算，旧结果目录与缓存不重命名、不复用、不在本项中删除。
- 最终验证证据：
  - `python scripts/generate_load_15min_matrix.py --json`：4,689 份矩阵 `rewrite=0`、`delete=0`，其中 4,617 个单模型、72 个 Ensemble；
  - `python scripts/check_model_configs.py`：`checked=5150 passed=5150 hard_failures=0 warnings=0`；
  - `python scripts/audit_ensemble_configs.py`：`failures=[]`；
  - `python -m unittest tests.test_package_layering`：17 项通过；
  - `python -m unittest discover -s tests -p "test_*.py"`：741 项，耗时 1167.647 秒，全部通过；
  - `python -m compileall -q forecasting_core data_loading feature_engineering model_forecasting model_ensemble scripts tests` 与 `git diff --check`：均通过。

---

## OPT-005：消除监督设计拆分中的 608,256 次 pandas 筛选

- **状态**：待处理
- **优先级**：P0
- **类型**：冷启动性能 / 批编译数据布局
- **涉及范围**：`model_forecasting/design.py::_supervised_arrays()`、`_RegistryDesignBuilder.training_rows()`、compiler batch 合同及等价性测试

### 当前事实

1. `lgbm_recursive.yaml` 会构造 6,336 个监督 origin、96 个 Recursive call，每个 call 的设计矩阵为 `(6336, 35)`。
2. 2026-09-03 真实冷运行耗时 313.93s，其中 `compile_supervised_arrays` 为 205.59s，占 65.49%；cache hit 后 runner 初始化只需 0.20s。
3. `_supervised_arrays()` 按 128 个 origin 分成 50 批；`training_rows()` 对每个 origin 的 96 行 frame，逐 call 执行一次 `frame.loc[frame["horizon_step"] == call_step, schema].to_numpy(copy=True)`，见 `model_forecasting/design.py:567-577`、`model_forecasting/design.py:750-765`。
4. 因此单次冷编译会执行 `6336 × 96 = 608,256` 次 pandas 布尔筛选。
5. compile-only 边界计时中，`training_rows()` 共 230.95s，其中 batch 后拆 call 的残差耗时 130.26s；`compiler.compile_batch()` 为 88.24s、标签抽取为 12.28s、registry materialize 仅为 0.16s。计时包装会放大绝对值，但调用次数和热点排序明确。

### 问题与影响

- batch compiler 已一次生成完整 frame，却在返回训练数组前重新进入高频 pandas 索引路径，抵消了批编译的大部分收益。
- 该开销只在 cache miss 时出现，但 schema、数据、compiler 版本或 cache key 变化都会重新触发；首次部署和新实验组运行仍需承担数分钟冷启动。
- 608,256 个小 DataFrame 操作还会制造大量短命对象和解释器调度，无法有效使用多核。

### 修复方法

1. 在每个 batch item 上只执行一次 feature matrix 的 NumPy 转换，不再为每个 call 重复调用 pandas `.loc`。
2. 将 compiler 已有的 `identity-major / call-minor` 行序写成显式合同；按 `selected_steps` 建立 call position 映射，用 reshape 或固定步长 slice 返回各 call 视图。
3. 第一阶段保持 `training_rows()` 返回类型不变，只替换内部拆分方式；验证稳定后再评估让 batch API 直接返回 call-major 数组，减少 per-origin tuple 和最终 `np.concatenate` 中间对象。
4. 不通过缩短 `history_steps`、fold 或 horizon 获得性能数字，也不改变 feature schema、行序和 float 值。

### 实施风险

- Global/K2 下每个 origin 包含多个 series，错误 reshape 容易交换 series 与 horizon 轴。
- Direct/MO 策略的 call step 不一定等同于连续位置，不能把 Recursive 的 `0..95` 假设硬编码为通用合同。
- 返回 NumPy view 后，下游若原地修改数组可能污染同批其他 call；必须确认只读边界或在必要位置做一次明确 copy。

### 验收标准

- [ ] Local K1、Global K2及七种策略的 batch-vs-row设计矩阵逐值 exact，shape、dtype、feature schema和行序一致。
- [ ] 监督 origin、sample series ID、Y tensor及最终 bundle合同保持不变。
- [ ] `cv_plot_df.csv`、`prediction.csv`及两张评分表与修改前逐值或SHA-256一致。
- [ ] 相同机器独占运行下，608,256次pandas筛选归零；batch后拆call阶段耗时至少下降80%。
- [ ] cache miss/hit、source失效及package layering定向测试通过。

### 实施记录

尚未实施。2026-09-03 已用真实配置和两轮 compile-only 计时完成根因定位。

---

## OPT-006：训练批编译避免物化 443.52 万个 `VisibilityProof` 对象

- **状态**：待处理
- **优先级**：P1
- **类型**：冷启动性能 / 信息可见性审计
- **涉及范围**：`feature_engineering/compiler.py`、监督训练编译入口、visibility proof合同与泄漏门禁

### 当前事实

1. 当前配置有 7 个 lag 特征和 28 个 derived 特征。每个 origin 构造 `7 × 96 + 28 = 700` 条 proof，6,336 个 origin 合计 4,435,200 个Python对象。
2. `collect_audit=False` 只阻止 `_RegistryDesignBuilder` 累计 `CompiledFeatures`，不能阻止 compiler 先创建proof、执行 `_validate_visibility_proofs()`，再随临时对象释放。
3. compile-only计时中，`_finish_batch_item()` 累计约24s；proof column时间戳tuple、对象构造与校验还分散在其他未单列区段中。
4. 正式预测和holdout仍需要可持久化、可审计的proof；这里的问题仅限监督训练批编译中的一次性对象物化。

### 问题与影响

- 数百万个dataclass和Timestamp引用产生明显CPU、内存分配及GC成本。
- 直接关闭proof会破坏严格as-of安全合同；性能优化不能以跳过泄漏检查为代价。

### 修复方法

1. 为 compiler 增加明确的训练校验模式，例如 `proof_mode="validate_only"`，不得使用含糊的全局开关或静默禁用。
2. validate-only模式使用列式数组执行 `source_time/available_at <= visibility_cutoff`、provider和角色边界校验，不逐条创建 `VisibilityProof`。
3. forecast/holdout继续走现有proof产出路径；训练模式只能由内部监督编译入口选择，不能暴露为YAML选项。
4. 对非法未来source time、非法available_at、浅lag缺provider等案例做故障注入，证明validate-only仍按原异常类型和边界RAISE。

### 实施风险

- 这是信息泄漏安全边界，遗漏任一role/provider分支都比性能回退更严重。
- batch与row路径的derived proof布局不同，不能用“当前不持久化”作为删除校验的理由。
- 需要区分“不要返回对象”和“不要验证”；前者可优化，后者禁止。

### 验收标准

- [ ] 监督batch编译不创建逐行 `VisibilityProof`，但所有现有及新增泄漏测试继续RAISE。
- [ ] forecast `visibility_proof` 和holdout summary的schema、lookup计数及as-of边界不变。
- [ ] batch-vs-row feature值和合法输入判定保持一致。
- [ ] 真实冷运行记录proof对象数量、compiler时间和峰值RSS的修改前后对比。
- [ ] 最终预测、回测、评分及bundle逐值等价。

### 实施记录

尚未实施。此前“禁止训练期累计audit”已经解决长期持有问题，本条处理仍存在的短命proof构造成本。

---

## OPT-007：compiled feature cache改用 `RawDesignFingerprint`

- **状态**：待处理
- **优先级**：P2
- **类型**：缓存复用 / 内容寻址边界
- **涉及范围**：`feature_engineering/cache.py`、canonical fingerprint、source/generator hash、compiler schema与缓存测试

### 当前事实

1. `compute_compiled_fingerprint()` 当前直接包含完整 `config.fingerprint()`，见 `feature_engineering/cache.py:87-112`。
2. 已单独排除 `validation.performance`，因此2×4和4×2可以共享cache；但 estimator、probabilistic、output及其他不参与raw design的字段仍会阻断复用。
3. 本次清除 `information_mode` 后 canonical fingerprint 从 `bdc414...` 变为 `8f932...`，compiled cache key从 `f51b750...` 变为 `a05fe115...`。特征和最终CSV保持一致，但正式结果目录仍发生一次223.42s冷编译。
4. 当前单份cache约167MiB，其中96个 `(6336, 35)` float64 X理论体积162.42MiB，Y约4.64MiB。

### 问题与影响

- raw design未变化时仍可能重复执行昂贵冷编译并保存大型重复cache。
- 配置矩阵中只改变模型、概率模式或输出布局的配置无法安全共享原始设计，放大磁盘和批跑时间。
- 反过来，过度排除字段也会导致不兼容设计错误共用，属于缓存正确性风险。

### 修复方法

1. 定义独立、版本化的 `RawDesignFingerprint`，只包含problem、data、compiler实际消费的feature spec、strategy call layout、origin、backtest geometry、training scope/series order、source SHA-256、generator实现、环境锁文件及compiler schema版本。
2. 排除estimator、probabilistic、output、eval mask、日志/并行参数，以及仅在fit阶段消费的feature scaling/selection。
3. canonical fingerprint、结果identity和bundle fingerprint继续独立保留，不用raw key替代模型身份。
4. 同key并发miss增加single-flight或原子等待，避免多个配置同时编译和写入同一cache。

### 实施风险

- 漏纳入call layout、series order、source内容或generator实现会产生错误cache hit。
- raw feature与fold-time transform边界必须清晰；target transform、scaling和selection不得错误固化进raw payload，也不得在应失效时复用。
- 变更key结构需要提升cache schema版本，旧cache应明确失效，不做模糊兼容。

### 验收标准

- [ ] 仅改变estimator/probabilistic/output/performance的配置得到相同raw key，第二次运行cache hit且磁盘只有一份payload。
- [ ] 改变data/features/strategy layout/origin/backtest geometry/source内容/generator实现时raw key必然变化。
- [ ] 每个配置自己的canonical fingerprint、identity、resolved config和bundle保持独立正确。
- [ ] 并发同key miss只允许一个writer，其余读取完整且hash校验通过的payload。
- [ ] 跨模型复用后的设计数组、预测结果和独立冷编译基线逐值exact。

### 实施记录

尚未实施。`validation.performance`双重排除已完成，本条是进一步拆分raw design身份。

---

## OPT-008：评估LightGBM回测fit的下一阶段优化

- **状态**：阻塞
- **优先级**：P3
- **类型**：热运行性能 / 并行预算
- **涉及范围**：回测fold调度、LightGBM线程参数、训练数据容器与bundle兼容

### 当前事实

1. 当前physical配置采用2个window worker × 每模型4线程；cache-hit真实运行为108.03s，峰值RSS 1.138GiB，平均使用约4.27个CPU核。
2. 32次fit累计worker时间179.55s；按2窗口并行估算关键路径约91.92s，占热运行约85.08%，是热运行主瓶颈。
3. 临时4窗口×2线程对照为107.11s，只快0.86%，处于单轮运行波动范围；单fit均值却从5.61s恶化到10.47s，慢86.62%，final fit从4.28s增至6.51s。
4. 4×2的平均CPU仅约4.40核，没有解决利用率问题；当前没有证据支持修改现役2×4配置。

### 问题与影响

- 热运行进一步优化必须触及模型训练本身，但简单移动“外层fold数×模型线程数”不能获得稳定收益。
- 直接切换LightGBM原生Dataset/backend、跨fold warm start或共享booster可能改变模型对象、训练轨迹和bundle合同，不是透明优化。

### 后续方法

1. 保留现役2×4，不将4×2写入生成器或YAML。
2. 若继续评估，需在独占机器上交错运行2×4、4×2及必要的1×8，每组至少3次，比较median/p95、CPU、RSS和逐值产物，不以单轮小于5%的差异裁决。
3. 单独计时NumPy→LightGBM Dataset构造和booster训练；只有容器构造占端到端比例足够高，才评估公开API下的Dataset复用。
4. 禁止通过减少fold、训练窗、树数或改变objective伪造加速。

### 阻塞原因与风险

- 当前候选4×2已证伪，尚无达到稳定收益且保证逐值等价的下一实现方案。
- LightGBM线程调度受数据规模、系统负载和OpenMP影响，微小差异不能外推为默认策略。
- 原生backend切换可能破坏sklearn wrapper序列化和部署预测接口。

### 验收标准

- [ ] 新候选在交错重复实验中median端到端至少稳定改善10%，且p95不恶化。
- [ ] `cv_plot_df.csv`、`prediction.csv`、两张评分表和bundle reload预测逐值exact。
- [ ] 并行预算不超过物理核，不出现嵌套线程池或RSS不可控增长。
- [ ] 明确记录适用拓扑，仅对Recursive单scalar fit生效时不得外推到其他策略/模型。

### 实施记录

2026-09-03 已完成2×4与4×2真实对照；4×2无显著收益，因此本条阻塞，不修改现役配置。

---

## OPT-009：减少Recursive 96步串行预测的固定开销

- **状态**：待处理
- **优先级**：P4
- **类型**：推理性能 / 递归状态更新
- **涉及范围**：Recursive executor、预测provider、逐步特征更新及visibility proof

### 当前事实与问题

1. 当前配置每个fold及final forecast都执行96步递推，32次 `predict()` 累计7.94s，平均0.248s，占108.03s热运行约7.35%。
2. 后一步lag依赖前一步预测，不能像Direct一样把96步改成完全并行；盲目向量化会改变策略语义。
3. 该阶段明显小于模型fit和冷编译拆分，当前不是首要瓶颈。

### 修复方法

1. 先细分每步成本为estimator predict、provider状态更新、动态lag替换、DataFrame/ndarray转换和proof生成。
2. 保留horizon顺序依赖，只优化循环内固定工作：预分配预测数组、缓存不随step变化的列/索引、仅更新依赖预测值的lag位置。
3. 数值模型继续使用ndarray快路径；不得为了减少调用把96步改成不同的多输出模型。
4. 如果优化后的绝对收益不足以影响端到端时间，则保留现状，不为低优先级热点增加复杂状态机。

### 实施风险

- lag位置或provider可用时间更新错误会造成隐蔽的数据泄漏或递推错位。
- point与quantile递归传播、Local/Global和多target的状态布局不同，不能只验证当前K1 point配置。
- proof减少必须与OPT-006一样保留严格as-of校验。

### 验收标准

- [ ] Recursive/RecMO/DirRec/DirRecMO涉及预测依赖的策略级测试全部通过。
- [ ] 96步预测及bundle reload预测与修改前逐值exact，provider usage和proof边界不变。
- [ ] 独立计时证明 `predict()` 阶段有稳定、可复现的改善，并报告对端到端的真实贡献。
- [ ] 不改变训练模型数、objective、策略定义或概率传播合同。

### 实施记录

尚未实施。优先等待OPT-005、OPT-006及热运行fit方向收口后再评估。
