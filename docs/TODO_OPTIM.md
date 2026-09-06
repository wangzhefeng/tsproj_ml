# 项目待优化问题

本文档记录尚未解决的工程问题、技术债和配置体系优化事项。业务实验与场景研究事项继续记录在 [`docs/TODO_AIDC.md`](TODO_AIDC.md)。

## 维护约定

- 状态使用：`待处理`、`进行中`、`阻塞`、`已完成`。
- 每个问题必须记录当前事实、影响、建议方向、风险和验收标准。
- `已完成` 必须补充实际执行的验证命令及结果，不能只记录实现计划。
- 涉及预测语义、数据口径或产物兼容性的修改，实施前需要单独确认。
- 问题开始实施后，如方案发生偏差，在原条目中追加实施记录，不回改已确认的历史结论。

---

## 当前待办总览（2026-09-05 实现核对）

初次复核重新打开 OPT-010、OPT-016、OPT-017；经本次获准实施及最终证据验收，19个条目全部完成，当前无未完成项。历史事实和测量保留，新增证据另记，不将历史基准冒充当前代码重测。最后收口遵守用户“不进行全量测试”的要求，没有重跑已通过的267项定向回归。

| 条目 | 当前状态 | 剩余工作 | 执行边界 |
| --- | --- | --- | --- |
| OPT-010 | 已完成 | 正式24份及27次分解标定均通过证据验收；三个分解代表保留默认 | 正式31fold与开发5fold证据分开，串行排队 |
| OPT-016 | 已完成 | 预算/完整性/checkpoint通过统一定向回归，规范已同步 | 估算内存准入+有限缓存+采样RSS，不宣称OS硬内存隔离 |
| OPT-017 | 已完成 | 唯一CatBoost签名、metadata/checker接线通过 | 不扩大采用范围；原5fold证据不冒充31fold性能 |

**执行顺序**：用户已确认完整实施方案和正式长跑。预算/产物/checkpoint/profile先通过门禁；正式24份清单验收完成后才开始分解标定，最后汇总产物及文档。不启动互抢CPU的并行性能实验。

**不重新打开**：OPT-009、OPT-015、OPT-018 的 no-op 是有证据的结论，不是漏做；OPT-019 的配置/运行合同修复已完成，不因尚未重新标定 LightGBM 分解性能而回退。未测 route、K2、宽特征或其他模型拓扑只是证据覆盖边界，不自动扩展为本批次必须全量重跑的任务。旧 identity 产物清理仍需单独授权。

### 初次核对证据与限制（以下为实施前快照）

- 接手实查 `git status --short`、`git diff --stat`：`dev`，22 modified + 6 untracked；保留全部已有更改，本次只修改本文档。
- `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest tests.test_batch_runtime tests.test_decomposition_spec tests.test_generate_load_15min_matrix tests.test_package_layering`：57 项、9.273 秒、全部通过。通过的是既有测试，不代表新增缺口已修复。
- 用项目 `.venv` 执行最小故障探针：`_artifacts_complete({"artifacts": {}})` 实际返回 `True`；将该空字典放入一项 completed task 后，`verify_batch_results()` 实际返回 `verified_count=1`。此为人工构造的故障输入，不是真实模型产物。
- 父预算探针：8 核、`parent_concurrency=4` 时 `available_threads=2`；按 batch 当前做法替换为 `parent_concurrency=2` 后变为 4，证明嵌套父预算可被放大。该探针验证预算运算，不宣称已经实测造成 CPU 超卖。
- 本次没有重跑 222 项统一回归、全配置审计、三轮性能基准或正式 31-fold 模型；下文原有数字仍是对应历史批次证据，不能标注为本次新验证。

### 2026-09-05 获准实施后的新增证据（已收口）

- 最新统一定向回归：267 tests / 225.300s / OK，日志 `/tmp/tsproj-opt-final-regression.log`。与历史222项不是同一次运行；覆盖原21模块，加 artifact/resource/checkpoint/profile/calendar-batch/decomposition-spec 六模块。命令如下，统一使用项目根 `.venv`：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest tests.test_runtime_resource_planner tests.test_ensemble_oof tests.test_ensemble_quantile_blending tests.test_ensemble_runtime tests.test_audit_ensemble_configs tests.test_ensemble_loader tests.test_ensemble_specs tests.test_ensemble_methods tests.test_ensemble_parity tests.test_ensemble_persistence tests.test_compiler_batch_design tests.test_compiler_batch_equivalence tests.test_feature_visibility_compiler tests.test_compiled_feature_cache tests.test_batch_eligibility_audit tests.test_batch_runtime tests.test_generate_load_15min_matrix tests.test_config_entrypoints tests.test_canonical_runtime_smoke tests.test_runtime_window_parallelism tests.test_package_layering tests.test_batch_artifact_gate tests.test_batch_resource_limits tests.test_runtime_checkpoints tests.test_performance_profiles tests.test_batch_calendar_checkpoint tests.test_decomposition_spec
```

- 全配置 checker：同一前缀运行 `scripts/check_model_configs.py`，`checked=5150 passed=5150 hard_failures=0 warnings=0`；日志 `/tmp/tsproj-opt-final-config-audit.log`。
- OPT-016：父预算组合、全批preflight、bounded cache、采样RSS、严格产物摘要/身份门禁和完成拟合单元checkpoint已接通。自然月factory关键字透传通过RED→GREEN；真实自然月quantile batch又复现验收错误使用最终月horizon，改为逐折label范围后完成/resume测试通过。checkpoint覆盖scalar/native/multi-RHS/chain/shared-quantile，CQR按时间重放，不保存未完成fit。合同细节见权威设计文档§7.2。
- OPT-017：`profile_ref`绑定实际H/K/Q、rows/features/schema、scope、源数据SHA256、CatBoost版本/默认参数、训练窗和资源预算；runtime/batch以真实设计解析，metadata区分benchmark profile与普通override。唯一physical31fold配置真实runner规划探针通过，`formal_benchmark_verified=false`；没有重新跑九模型实验。
- 正式清单24份现已全部验收：2份复用、22份新运行。单队列 `/tmp/tsproj-opt-formal-closeout.py` 按原physical YAML跑31fold/final/forecast并做reload及历史四CSV对照；`results/_optimization_closeout/summary.json`为24/24，另起只读进程全量复核通过。分解5fold标定在正式队列结束后启动。
- 根`AGENTS.md`同步首次审批超时后没有绕过；用户随后明确“审批：AGENTS.md 的同步补丁”，两条规则已通过受保护patch正常落盘。权威设计文档同步完成。
- 分解三拓扑各三轮全部完成，共27次5fold运行；独立verifier回读108份CSV、27个bundle，并复算统计一致。三个代表均不采用新performance，完整证据归档到`results/_optimization_closeout/decomposition/`；不是用no-op代替未运行实验。

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
- 5,150 份活动 YAML 与 AIDC 生成器已清扫。该语义字段删除会使全部活动配置获得新 fingerprint；迁移清单 `docs/redesign/architecture.md` 当时已全量重算（该 manifest 已于 2026-09-06 清除，历史内容从 Git 溯源）。旧结果目录与 OOF cache 不重命名、不复用，后续运行按新 identity/cache key 重新生成；物理清理旧结果不在本项范围内。
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

- **状态**：已完成
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

- [x] Local K1、Global K2及七种策略的 batch-vs-row设计矩阵逐值 exact，shape、dtype、feature schema和行序一致。
- [x] 监督 origin、sample series ID、Y tensor及最终 bundle合同保持不变。
- [x] `cv_plot_df.csv`、`prediction.csv`及两张评分表与修改前逐值或SHA-256一致。
- [x] 相同机器独占运行下，608,256次pandas筛选归零；batch后拆call阶段耗时至少下降80%。
- [x] cache miss/hit、source失效及package layering定向测试通过。

### 实施记录

- 新增 `_split_batch_designs()`，将每个 batch item 的 feature frame 只转换一次 NumPy，按显式 `identity-major / call-minor` 合同 reshape，再以 call position 返回视图；非连续 MO call step 不按数值连续性假设处理，行数和 `horizon_step` 顺序不符时 RAISE。
- Local K1 与 Global N2/K2 的七策略 batch-vs-row 测试逐值 exact；监督 origins、sample series IDs、Y、feature/categorical schema 合同保持不变。
- 真实 `lgbm_recursive.yaml` 未插桩冷编译由 205.59s 降至 49.551s；边界插桩中 batch 后拆 call 残差由 130.26s 降至 2.401s，下降 98.16%，608,256 次 pandas 布尔筛选归零。
- OPT-005 隔离对照时，Direct/MIMO/Recursive 的 `cv_plot_df.csv`、`test_scores_df.csv`、`test_scores_horizon_df.csv`、`prediction.csv` 与 `model.pkl` 均与修改前正式结果 SHA-256 一致；后续 OPT-008 改变 Recursive 持久化线程参数后，语义 CSV 与 bundle reload prediction 仍 exact，但 `model.pkl` 字节按该项说明不再要求相同。共同测试证据见 OPT-011。

---

## OPT-006：训练批编译避免物化 443.52 万个 `VisibilityProof` 对象

- **状态**：已完成
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

- [x] 监督batch编译不创建逐行 `VisibilityProof`，但所有现有及新增泄漏测试继续RAISE。
- [x] forecast `visibility_proof` 和holdout summary的schema、lookup计数及as-of边界不变。
- [x] batch-vs-row feature值和合法输入判定保持一致。
- [x] 真实冷运行记录proof对象数量、compiler时间和峰值RSS的修改前后对比。
- [x] 最终预测、回测、评分及bundle逐值等价。

### 实施记录

- `FeatureCompiler.compile_batch()` 新增内部 `proof_mode="materialize"|"validate_only"`；仅 `_RegistryDesignBuilder.training_rows()` 选择 validate-only，fallback 若误用该模式会明确 RAISE，公共 YAML 无入口。
- validate-only 保留 lag source time、role/provider 的原编译边界，并对列式 `available_at` 执行 NaT 与 `<= visibility_cutoff` 校验，不创建逐行 `VisibilityProof`；晚到 available-at 故障注入继续按原异常 RAISE。普通 batch、holdout 和 forecast 仍走 materialize，proof 数量、顺序和字段与 row 路径逐条相等。
- 真实 MIMO 冷编译在 detached 旧版基线为 32.009s、峰值 RSS 249,905,152 bytes；修改后为 24.694s、235,896,832 bytes，RSS 下降 14,008,320 bytes（5.61%），两者均无 swap，X/Y shape 分别保持 `(6336,35)` 与 `(6336,96,1)`。
- 训练批路径通过构造器故障注入确认 `VisibilityProof` 创建数为 0；Direct/MIMO/Recursive 在后续线程拓扑调整前的正式产物 SHA-256 等价证据见 OPT-011，最终 Recursive 的语义等价边界见 OPT-008。

---

## OPT-007：compiled feature cache改用 `RawDesignFingerprint`

- **状态**：已完成
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

- [x] 仅改变estimator/probabilistic/output/performance的配置得到相同raw key，第二次运行cache hit且磁盘只有一份payload。
- [x] 改变data/features/strategy layout/origin/backtest geometry/source内容/generator实现时raw key必然变化。
- [x] 每个配置自己的canonical fingerprint、identity、resolved config和bundle保持独立正确。
- [x] 并发同key miss只允许一个writer，其余读取完整且hash校验通过的payload。
- [x] 跨模型复用后的设计数组、预测结果和独立冷编译基线逐值exact。

### 实施记录

- compiled cache schema 提升为 2，运行时改用独立 `compute_raw_design_fingerprint()`；payload 包含 problem、data、去除 fit-time scaling/target transform/selection 后的 compiler feature spec、strategy layout、origin、回测几何、training scope/series order、source SHA-256、generator 实现及 `pyproject.toml`/`uv.lock` hash。旧 schema-1 cache 明确失效，不模糊兼容。
- estimator、probabilistic、output、eval mask、performance 和 fit-time transformations 不进入 raw key。Ridge point 与 LightGBM quantile 的测试配置共享同一 cache，第二个 runner 命中且 X/Y 逐值 exact；两者 canonical fingerprint 与 result identity 仍不同。
- data/features/strategy/origin/backtest geometry/source 内容/generator 实现的逐项变异测试均改变 raw key。generator digest 同时覆盖 callable 源码和定义模块文件 SHA-256，因此入口不变但同模块 helper 改变也会失效。运行时仍按每份配置写独立 resolved config、identity 和 bundle，raw key 不替代模型身份。
- cache 目录内增加同 key 锁：线程内 `threading.Lock` + 跨进程文件锁；POSIX 使用 `fcntl.flock`，Windows 使用 `msvcrt.locking`，不再引入平台导入回归。线程并发 miss 测试中仅一个 runner 编译/原子写入，另一个等待后通过 metadata 与 payload SHA-256 校验命中完整产物；spawn 双进程测试另行证明进程间互斥。
- Direct/MIMO configured-output 与 Recursive 隔离运行在后续线程拓扑调整前的五类产物均和修改前正式结果 SHA-256 一致；MIMO 独立冷编译 X/Y shape 与旧版逐值合同一致。最终 Recursive 的 bundle 字节差异与 reload prediction 等价边界见 OPT-008。
- 统一 changeset 独立 review 首轮发现 `fcntl` 无条件导入会造成 Windows import 回归；按 RED→GREEN 改为双平台锁后，第二轮独立 review 返回 `passed=true`、零 security/logic finding。回归测试同时覆盖入口函数不变但同模块 helper 改变时 raw key 必须变化、无条件 POSIX import 门禁和 spawn 双进程真实互斥。

---

## OPT-008：评估LightGBM回测fit的下一阶段优化

- **状态**：已完成
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

- [x] 新候选在交错重复实验中median端到端至少稳定改善10%，且p95不恶化。
- [x] `cv_plot_df.csv`、`prediction.csv`、两张评分表和bundle reload预测逐值exact。
- [x] 并行预算不超过物理核，不出现嵌套线程池或RSS不可控增长。
- [x] 明确记录适用拓扑，仅对Recursive单scalar fit生效时不得外推到其他策略/模型。

### 实施记录

2026-09-03 已完成2×4与4×2真实对照；4×2无显著收益，因此本条阻塞，不修改现役配置。

- 后续在同一 compiled cache 上按 `2×4→4×2→1×8` 交错三轮完整运行。2×4 median/p95 为 `112.276s/112.932s`；4×2 为 `94.737s/95.812s`，median 改善15.62%且p95改善；1×8为 `154.301s/156.206s`，比2×4慢37.43%，明确淘汰。新重复实验推翻了早期单轮4×2结论，以三轮median/p95为最终裁决。
- 4×2 median CPU time 为460.665s、平均使用4.863核，2×4分别为467.562s/4.169核；4×2最大RSS 1,351,401,472 bytes，全部运行swap=0、无OOM。并行预算固定为4个window worker×每模型2线程，不叠加output或quantile线程池。
- 九轮的 `cv_plot_df.csv`、`test_scores_df.csv`、`test_scores_horizon_df.csv`、`prediction.csv` 与 bundle reload prediction 逐值一致。不同 `model_thread_count` 会持久化不同线程参数，因此 `model.pkl` 文件字节不要求相同；三个拓扑各自内部稳定，且所有 bundle 均可加载并给出相同预测。
- final fit 三轮拆分中 `Dataset.construct` 为0.180～0.207s，只占 LightGBM fit 的4.02%～4.54%；即使完全消除也达不到10%端到端门槛，因此不切换 `lgb.train` 原生backend，不引入第二套模型对象/序列化合同。
- 仅 `aidc_load_15min_daily/route_A/baseline/lgbm_recursive.yaml` 改为 `window_parallel_workers=4 + model_thread_count=2`，pointwise、其他route/模型不外推。生成器应用计划为 `rewrite=1/create=0/delete=0`，应用后4,689份矩阵零漂移；physical YAML configured-output完成31折、final fit和正式预测，resolved config回读及bundle reload通过。

---

## OPT-009：减少Recursive 96步串行预测的固定开销

- **状态**：已完成
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

- [x] Recursive/RecMO/DirRec/DirRecMO涉及预测依赖的策略级测试全部通过。
- [x] 96步预测及bundle reload预测与修改前逐值exact，provider usage和proof边界不变。
- [x] 独立计时确认可优化固定开销仅占端到端 0.50%，不足以形成稳定有效改善；已报告真实贡献并按既定规则作 no-op 裁决。
- [x] 不改变训练模型数、objective、策略定义或概率传播合同。

### 实施记录

- 2026-09-03 在 OPT-005/006/011 收口后完成真实 `lgbm_recursive.yaml` 全生命周期分段：32 次 Recursive forecast 共 8.910s；其中 3,104 次 feature compile 共 6.702s，3,072 次 adapter/estimator predict 共 1.437s；按调用数扣除首步预编译后，剩余 executor 固定开销约 0.840s，仅占 168.876s 端到端的 0.50%。该 safe-lag 配置 provider value 调用为 0，不存在递归状态更新热点。
- 验收标准第 3 项未达到：可优化固定开销的绝对上限不足以影响端到端，继续引入预分配状态机或增量 proof 路径不具备真实收益。按“修复方法”第 4 条采用 no-op 裁决并关闭本项，不把“没有加速”伪写成性能改善。
- 未修改 executor、策略定义、模型数、objective 或概率传播。完整 Recursive 隔离运行的五类产物与修改前正式结果 SHA-256 一致，provider usage 仍为 100% history；策略级与 bundle reload 验证命令记录于 OPT-011。

---

## OPT-010：评估Direct/MIMO LightGBM的8路output并行

- **状态**：已完成
- **优先级**：P1
- **类型**：热运行性能 / 多scalar输出任务并行预算
- **涉及范围**：`model_forecasting/fit_service.py`、AIDC 15min配置生成器、Direct/MIMO LightGBM配置与性能验收

### 当前事实

1. `lgbm_direct.yaml`与`lgbm_mimo.yaml`均未声明`validation.performance`。两者的independent adapter最终都产生96个scalar output任务，运行时默认使用4个worker，并把每个LightGBM固定为`n_jobs=1`，见`model_forecasting/fit_service.py:123-138`、`model_forecasting/fit_service.py:161-169`、`model_forecasting/fit_service.py:189-204`和`model_training/estimators/multi_target.py:219-270`。
2. Direct有96个单输出模型组；MIMO虽然只有1个模型组，但该组的96步输出仍由independent adapter拆成96个scalar estimator。两种配置均为H=96、K=1、31个回测窗口，加1次final fit，共执行`(31 + 1) × 96 = 3,072`次scalar estimator fit。
3. Direct正式4-worker cache-hit运行耗时201.73s，平均使用3.687核，峰值RSS为1.298GiB；临时8-worker隔离运行耗时154.24s，平均使用6.022核，峰值RSS为1.128GiB，wall缩短47.49s（23.54%），加速1.308×。
4. MIMO正式4-worker cache-hit运行耗时203.23s，平均使用3.693核，峰值RSS为0.623GiB；31个回测窗口在cache加载后耗时190.91s，占热运行93.94%。临时8-worker隔离运行耗时143.81s，平均使用6.300核，峰值RSS为0.699GiB，wall缩短59.42s（29.24%），加速1.413×。
5. 两种8-worker试验均继续命中各自原compiled cache，canonical fingerprint保持不变；各自的`cv_plot_df.csv`、`test_scores_df.csv`、`test_scores_horizon_df.csv`、`prediction.csv`和`model.pkl`SHA-256均与正式4-worker结果完全一致。
6. MIMO 8-worker将平均CPU利用率从46.17%提高到78.75%，但总CPU时间增加20.70%、峰值RSS增加12.32%；这证明单配置延迟下降，不证明多配置并发吞吐同步提高。
7. 当前每种策略都只有一次正式4-worker暖运行与一次8-worker隔离对照，足以确认候选方向，但不足以直接修改物理YAML或生成器。OPT-008记录的是Recursive单scalar模型的窗口并行与模型内线程问题，其结论不覆盖本项多scalar输出拓扑。

### 问题与影响

- 当前4-worker上限使Direct和MIMO在8核机器上长期空置约一半CPU容量，单次完整热运行分别多消耗约47s和59s wall time。
- 只看`strategy.model_count`会把MIMO误判为单模型任务，漏掉independent adapter内部96个可并行scalar输出；性能拓扑必须统计底层estimator数量。
- 沿用Recursive的window/model-thread结论会错过现成的无语义output并行空间；反过来，把8-worker单配置低延迟结果直接用于配置目录并发也会造成CPU争用。

### 修复方法

1. 在独占机器上分别以Direct和MIMO physical YAML、同一seed和同一compiled cache，交错重复运行4-worker与8-worker各至少3次，记录wall、user/sys、平均CPU、峰值RSS及median/p95。
2. 所有对照均保持`window_parallel_workers=1`和LightGBM `n_jobs=1`，只改变`multi_output_n_jobs`，禁止开启嵌套线程池；正式configured-output运行与`/tmp`性能隔离结果必须明确区分。
3. 某一策略独立达到验收阈值后，在AIDC 15min配置生成器中仅对已验证的LightGBM拓扑写入`validation.performance.multi_output_n_jobs: 8`并重生成对应物理YAML；不要求Direct与MIMO绑定采用，也不修改canonical fingerprint或结果identity。
4. 若后续扩展到K2、不同H、不同训练窗或其他模型，必须按各自scalar任务数和内存重新标定，不把本项数值写成跨模型默认常量。

### 实施风险

- worker增多会增加同时构造LightGBM Dataset和直方图的CPU争用；单配置wall下降不等于多配置批跑总吞吐提高。
- 8-worker若再叠加窗口并行或模型内部多线程，会超过8核预算并产生oversubscription。
- MIMO单轮峰值RSS已增加12.32%；更宽特征、K2或更大训练窗可能继续放大，必须用重复运行p95裁决。
- 性能参数虽不进入semantic fingerprint，但生成器与物理YAML必须同步，否则下一次幂等生成会覆盖手工配置。

### 验收标准

- [x] Direct与MIMO分别完成4-worker/8-worker各至少3次交错独占运行；拟采用8-worker的策略median wall稳定改善至少10%，p95不恶化。
- [x] 所有对照运行的回测CSV、两张评分表、正式预测和bundle reload预测逐值exact，模型bundle可正常加载。
- [x] 运行时解析结果固定为window worker 1、output worker 8、LightGBM `n_jobs=1`，全局并行预算不超过8个物理核。
- [x] 峰值RSS无失控增长、无swap/OOM；同时报告单配置延迟与CPU总消耗，不把两者混为配置目录吞吐。
- [x] 若正式采用，配置生成器幂等检查、目标YAML解析、compiled cache复用和相关线程调度测试通过。

### 实施记录

2026-09-03 已分别完成Direct与MIMO的一次正式4-worker暖运行和一次8-worker隔离候选运行；候选wall分别改善23.54%和29.24%，各自五类关键产物SHA-256完全一致。按验收标准完成交错重复实验后，再按策略独立决定是否写入生成器和物理YAML。

- 后续在 `/tmp/tsproj-opt010-1788425949032083000` 独占串行完成 `4→8→4→8→4→8` 三轮交错。Direct median wall `221.499s→157.645s`（改善28.83%），p95 `234.563s→187.548s`；MIMO median wall `224.806s→162.090s`（改善27.90%），p95 `236.965s→185.405s`。两者均达到采用阈值。
- Direct median CPU time `775.543s→918.834s`、平均核数 `3.510→5.809`；MIMO median CPU time `800.068s→896.997s`、平均核数 `3.569→5.525`。结论仅是单配置延迟改善，CPU总消耗上升，不外推为配置目录吞吐改善。
- Direct 4/8-worker 最大 RSS 分别为 1,074,741,248 / 939,540,480 bytes；MIMO 分别为 771,063,808 / 744,751,104 bytes；全部运行 swap=0、无OOM。六轮各自的五类文件 SHA-256 和 bundle reload prediction SHA-256 完全一致，compiled cache 全部命中，canonical fingerprint/identity 不变。
- 仅 `aidc_load_15min_daily/route_A/baseline/lgbm_{direct,mimo}.yaml` 写入 `window_parallel_workers=1`、`multi_output_n_jobs=8`、`model_thread_count=1`；不外推到 route_B、K2、更宽特征或其他模型。两份 physical YAML configured-output 已完成31折、final fit与正式预测，resolved config 回读确认 1×8×1，正式产物仍与修改前 SHA-256 一致。
- 生成器写盘计划为 `rewrite=2/create=0/delete=0`，应用后幂等复查 `rewrite=0/delete=0`、4,689份矩阵通过；线程解析/调度和共同定向测试见 OPT-011。

### 2026-09-04 P0/P1 后续扩面裁决

- P0 在共享 pointwise 语义修复后，对 `daily/route_A` 的 plain/cyclical 两份配置完成 `2×4→4×2→1×8` 三轮交错。plain median `114.706s→98.125s`（4×2 相对2×4改善14.46%），cyclical 为 `112.206s→96.091s`（改善14.36%）；两者 p95 均改善，1×8 明显更慢。故两份配置由2×4改为4×2。
- P1 对 `daily/route_B`、`rolling/route_A`、`rolling/route_B` 的15份 H=96 baseline 镜像逐配置重复标定。9份 pointwise/recursive 的4×2改善11.65%～15.94%；6份 Direct/MIMO 的output8改善30.50%～31.51%。所有配置均达到median至少改善10%、p95不恶化、四张语义CSV与bundle reload prediction等价、零swap门槛。
- 结合 OPT-008/010 已测的 `daily/route_A` 三份配置，最终采用边界扩展为 **daily+rolling × route_A/B baseline 共20份 LightGBM**：Recursive/两个pointwise使用 `window_parallel_workers=4 + model_thread_count=2`；Direct/MIMO使用 `window_parallel_workers=1 + multi_output_n_jobs=8 + model_thread_count=1`。本记录以新扩面证据替代上文“仅route_A、不外推”的旧边界，但不回改历史实施结论。
- 15份 P1 physical YAML 均按正式目录完成31折、final fit和预测；`resolved_config.json` 回读线程计划正确，四张语义CSV与 benchmark candidate SHA-256 一致，bundle 可加载且 reload prediction 一致。这15个新 identity 运行前不存在正式产物，因此没有把空基线误写为“修改前正式SHA一致”。基准与正式断点分别在 `/tmp/tsproj-performance-matrix-20260903/`、`/tmp/tsproj-configured-p1-20260904/`。
- 采用参数按**单配置独占8核**标定；catalog 批跑必须保持 `P_config=1`，不得把单配置 latency 改善表述为多配置吞吐改善。未覆盖 short、非baseline、K2、非LightGBM和Ensemble。

### 2026-09-04 P2/P3/P4 分层标定

- **P2（H=96其他策略）**：`daily/route_A/baseline` 的 DirRec、DirMO、DirRecMO、RecMO 完成 output4/8 各三轮交错；median 分别为 `220.065→154.149s`（29.95%）、`192.706→134.920s`（29.99%）、`190.154→130.486s`（31.38%）、`121.326→80.746s`（33.45%）。四者均选择1×8×1；p95、语义产物、reload prediction、RSS/swap门禁通过。采用范围只含这4份实测配置，不外推到未测 H=96 镜像或其他group。
- **P3（short H=16）**：A/B两路18份 baseline LightGBM 均完成三轮交错。Recursive与两个pointwise的4×2相对2×4改善27.67%～31.90%；Direct/MIMO/DirRec/DirMO/DirRecMO的output8相对output4改善15.86%～22.32%。RecMO只有4个实际output任务：route_A 的output8慢1.03%，route_B虽快4.13%但未达10%门槛，故两路均保留默认output4且不写冗余performance。最终 short 仅16份配置写显式预算。
- **P4（代表配置）**：三份output代表完成三轮。weather+holiday Direct 为 `235.232→172.802s`（26.54%，output8最大RSS 1,225,359,360 bytes）；105特征 state Direct 为 `524.957→354.899s`（32.39%，最大RSS 1,577,320,448 bytes）；K2 joint Direct 为 `632.031→464.177s`（26.56%，p95 `690.368→511.503s`，最大RSS 1,590,149,120 bytes）。三者均选择1×8×1、零swap、语义/reload等价；只写这3份代表配置，不按单个策略结果推广整个group。
- cross-route Recursive 三轮选择2×4：median/p95为 `126.992s/131.344s`，明显快于近似现役默认的1×8（`176.436s/177.047s`）；4×2虽进一步降到116.855s，但相对2×4改善不足10%，故不选4×2。仅 `daily/route_A/add_endogenous_cross_route/lgbm_recursive.yaml` 写2×4，不推广同组其他策略。linear decomposition 仅完成两轮后按用户要求停止，证据不足不采用；STL/MSTL代表在fit时暴露既存合同冲突：生成器写 `trend_forecast: seasonal_naive`，运行时只接受 `polynomial|damped`。该项涉及分解语义，不在性能任务中擅改，P4按阻塞且不推广收口。
- P2/P3/P4 基准目录分别为 `/tmp/tsproj-performance-p2-20260904/`、`/tmp/tsproj-performance-p3*-20260904/`、`/tmp/tsproj-performance-p4*-20260904/`。生成器落地14份新增profile后，4,689份矩阵复查为 `create=0/rewrite=0/delete=0`；矩阵测试12项通过。
- configured-output 原计划覆盖 P2/P3/P4 的24份新增/已确认profile，用户要求停止后台任务后未恢复：仅 P2 的 `lgbm_dirrec.yaml`、`lgbm_dirmo.yaml` 2份完成正式31折、final fit、预测、resolved performance与benchmark hash/reload核验；其余22份只有三轮隔离benchmark parity和静态配置证据，不宣称正式结果已落盘。断点在 `/tmp/tsproj-configured-p234-20260904/`。

---

### 2026-09-05 完成边界复核与剩余待办

原 Direct/MIMO 采用实验及已完成的扩面配置保持验收通过；本项后续纳入 P2/P3/P4 后，正式产物和分解标定没有全部收口，因此总状态改为进行中，不撤销已经完成的 profile。

1. **剩余正式产物验收（阻塞：待用户恢复长跑授权）**：历史清单为 24 份、已验收 2 份、剩余 22 份。本次回读 `/tmp/tsproj-configured-p234-20260904/`，仍只有 `lgbm_dirrec` 与 `lgbm_dirmo` 两份 `result.json`，状态均为 `passed`；这证明原断点没有新增验收记录，不等于已全盘证明其他位置不存在产物。恢复前先按当前 fingerprint/identity 逐配置清点正式 results，已有完整产物先验收，不盲目重跑。验收须包含实际 fold 数、final fit、预测、resolved plan、语义 CSV 与 bundle reload；5-fold 开发产物不替代正式 31-fold 证据。
2. **分解代表性能标定（待处理）**：linear 断点有 `run-01` 至 `run-06`，对应三个拓扑各两轮，不能充作各三轮。STL/MSTL 的 `seasonal_naive` 冲突已由 OPT-019 的 `polynomial` 合同修复，技术阻塞解除；但 OPT-019 验证的是 Ridge linear/STL/MSTL 的正确性，不是 LightGBM 性能。后续用当前实现及最多 5-fold 派生配置重建同口径三轮交错证据，不把旧 31-fold 与新 5-fold 计时混合计算；只有 median 改善至少 10%、p95 不恶化、parity 与 RSS/swap 达标才采用，否则记录 no-op。

**风险**：沿用旧 identity 或不一致验证几何会导致误验收；以 5-fold 延迟推断正式 31-fold 收益亦不成立。禁止为补证据临时截断 tracked YAML，禁止未经确认启动正式长跑。

- [x] 将原剩余22份逐项核对并获准重跑，全部补齐正式31fold产物；原有2份通过复验，没有取消项。
- [x] 完成linear/STL/MSTL代表三轮标定，独立验收后全部保留默认；没有新采用项，生成器及物理YAML未改，无需新增正式运行。

### 2026-09-05 正式产物补齐验收

本节正式产物与下节分解5fold性能实验是两个独立证据层级。

用户确认恢复长跑后，`/tmp/tsproj-opt-formal-closeout.py`按24份清单单队列执行，在正式`results/{pretrained_models,results_test,results_forecast}`写入/验收。`results/_optimization_closeout/summary.json`记录`expected=processed=passed=24`，其中2份复用、22份新运行。全部保持物理YAML的31fold，final bundle、forecast和回测CSV齐全，4类语义CSV和bundle reload预测对历史对照均exact；未删除旧产物。队列结束后另起只读进程逐条加载当前YAML、核对fingerprint/fold_count，调用`model_forecasting.batch_artifacts.validate_artifacts(record['task'])`复核全部摘要/身份/几何，输出`independently_verified=24`。cross-route对照使用实际P4-window基准路径与2x4拓扑，而不是缺失对照时跳过parity。上段“待授权”是此前诊断断点，本记录替代其当前状态。

---

### 2026-09-05 分解三轮标定最终裁决

三个daily Route A LightGBM Direct pointwise-horizon代表各用独立5fold派生配置，按`2x4→4x2→1x8`交错三轮，共27个独立子进程。compiler cache在计时外预热，物理31fold YAML未改，不混入旧31fold计时。拓扑记号为window workers × model threads。

| 分解 | 1x8 median / p95 秒 | 2x4 median / p95 秒 | 4x2 median / p95 秒 | 裁决 |
| --- | --- | --- | --- | --- |
| linear | 38.952 / 42.428 | 35.109 / 35.757 | 35.867 / 38.129 | 最佳改善9.867%，不足10%，保留默认 |
| STL96 | 81.734 / 82.990 | 77.340 / 77.806 | 78.136 / 78.929 | 最佳改善5.376%，不足10%，保留默认 |
| MSTL96-672 | 697.125 / 701.506 | 706.339 / 731.867 | 707.539 / 720.344 | 两候选分别变慢1.322%/1.494%，保留默认 |

所有组的四类语义CSV及reload prediction均exact；内存bundle与重载bundle预测逐值相等，CSV十进制往返仅使用`rtol=0, atol=2e-12`。每次记录CPU、最大RSS及swap；进程swap计数、系统swap-used增长为0，但系统swap-out有增加，不能声称全局无swap，也不能把全局计数归因于单个模型。没有候选满足全套采用门槛，因此不改生成器/YAML，不触发额外正式31fold重跑。

独立验收命令：`env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python /tmp/tsproj-opt-decomposition-closeout/verify_evidence.py`。真实返回`status=passed, variants=3, runs=27, csv_files_verified=108, bundles_reloaded=27, each_run_folds=5, canonical_physical_yaml_unchanged=true, summary_recomputed_exact=true`；全部运行代码签名相同。逐次指标、汇总、fixtures及verification共33份JSON已复制到`results/_optimization_closeout/decomposition/`并逐文件核对SHA256；完整实验模型/CSV保留在原`/tmp`隔离目录。原始summary中的`benchmark_complete_adoption_pending`是控制器阶段标签，最终no-adoption裁决见归档README，不改写原始测量。

最后收口按用户要求不运行全量测试、不重跑267项定向回归；本阶段仅新增统计与文档，没有新的模型配置或运行代码改动。子任务另完成`tests.test_generate_load_15min_matrix`定向13项（0.684s、OK），两次只读`generate_load_15min_matrix.py --json`均为4689 unchanged、create/rewrite/delete=0；日志和`validation-checks.json`均已回读并归档。最终机器状态`closeout.json`为`completed_no_adoption`。此前5150配置审计、Ensemble/eligibility审计及统一定向回归证据保持独立记录。

---

## OPT-011：批量提取监督标签，消除逐时点pandas取值

- **状态**：已完成
- **优先级**：P2
- **类型**：冷启动性能 / 监督标签张量构造
- **涉及范围**：`model_forecasting/design.py::_labels_from_information_set()`、监督batch入口、Local/Global与多target等价性测试

### 当前事实

1. MIMO只有1个`(6336, 35)`特征矩阵，但仍需为6,336个监督origin构造`(96, 1)`标签块，最终`Y_all`形状为`(6336, 96, 1)`。
2. `_labels_from_information_set()`当前对每个target、series和forecast step先查询时间位置，再执行`series_frame.iloc[position][target]`，见`model_forecasting/design.py:369-430`。
3. `lgbm_mimo.yaml`因此执行`6336 × 96 = 608,256`次逐时点标签读取。compile-only cProfile中pandas `fast_xs`恰好调用608,256次；`_labels_from_information_set()`累计24.28s，占插桩后监督编译62.66s的38.75%。
4. 未插桩的MIMO正式冷运行中，compiled feature cache miss耗时32.106s，占236.09s总wall的13.60%；cache hit加载仅0.010s。cProfile将compile-only wall放大到64.69s，因此只用于调用次数和热点排序，不把插桩绝对值当生产耗时。
5. Direct同一路径在compile-only profile中也累计约23.05s，但被OPT-005记录的逐call特征frame拆分遮蔽；本项是跨策略的Y标签构造问题，不是OPT-005的X设计拆分问题。

### 问题与影响

- row-position lookup已经提供批量定位所需信息，随后仍逐时点进入pandas `.iloc`，制造数十万次Series构造、索引和Python `float()`调用。
- MIMO的X设计只有一个call，OPT-005优化后收益有限；标签提取反而成为其更明确的冷启动固定成本。
- 该成本只在cache miss时发生，但新数据、配置fingerprint、cache schema或依赖锁变化都会重新触发；多策略、多模型实验会重复承担。

### 修复方法

1. 对每个`(source, series_id, target)`一次取得目标列NumPy视图，并把当前request的全部forecast time转换为位置数组；完成边界校验后用一次`take`或高级索引填入标签块。
2. 第一阶段保持`_labels_from_information_set()`返回的`values`与`trajectories`合同不变，只消除step循环内的pandas `.iloc`；稳定后再评估在128-origin batch层一次组装完整标签张量。
3. 保留“目标时点必须恰好存在一行”、series identity、target顺序和严格`target_access=supervised_labels`边界；不得用重采样、插值或容错填充换性能。
4. 对Local K1、Global K2、多target及非连续forecast time分别验证位置映射，不能假设所有标签恒为连续96行。

### 实施风险

- Global场景若复用未纳入series identity的行位置数组，会发生跨序列标签错位。
- 多target列顺序或MIMO time-major展平顺序错误，会生成shape正确但监督语义错误的Y张量。
- NumPy高级索引通常会复制；优化目标是消除数十万次小pandas对象，而不是强行追求零拷贝。
- 旧compiled cache会绕过新标签提取路径；性能验收必须显式执行cache miss，不能只跑暖缓存。

### 验收标准

- [x] Local K1、Global K2、多target及七种策略的Y tensor、trajectory和异常边界与修改前逐值exact。
- [x] 真实MIMO compile-only profile中608,256次pandas `fast_xs`归零，标签提取阶段耗时至少下降80%。
- [x] `X_all`、`Y_all`、supervised origins、sample series IDs和feature schema合同保持不变。
- [x] `cv_plot_df.csv`、`prediction.csv`、两张评分表及bundle reload预测与修改前逐值或SHA-256一致。
- [x] compiled cache miss/hit、source失效、信息集泄漏故障注入、Global/multitarget及package layering定向测试通过。

### 实施记录

- `_labels_from_information_set()` 改为每个 `(source,series_id,target)` 一次取得 numeric NumPy 列，以 request forecast time 的位置数组执行 `take()`，一次写入 `(N,H,K)` 标签块并从同一数组构造 trajectory；不再逐 step 进入 `DataFrame.iloc`。
- Local K1、Global N2/K2、多target与七策略 batch-vs-row 的 Y/trajectory 逐值 exact；缺时点、重复时点、target access、series order 和 time-major 顺序继续由原合同严格校验。
- 真实 MIMO compile-only cProfile 的 pandas `fast_xs` 由 608,256 次降为 0；Recursive 同口径边界计时中标签阶段由 12.28s 降至 0.756s，下降 93.84%。MIMO 未插桩冷编译由旧版复现的 32.009s 降至 24.694s。
- Direct/MIMO physical YAML 已完成 configured-output；Recursive 在 OPT-008 调整线程拓扑前完成隔离全生命周期。该阶段三者的 `cv_plot_df.csv`、`test_scores_df.csv`、`test_scores_horizon_df.csv`、`prediction.csv`、`model.pkl` 均与修改前正式结果 SHA-256 一致；Direct/MIMO bundle reload prediction 在所有 4/8-worker 对照中逐值 exact。最终 Recursive 仍保持四张语义 CSV 与 reload prediction exact，`model.pkl` 字节差异见 OPT-008。
- 此前实施定向验证覆盖上述模块并追加 `tests.test_canonical_runtime_smoke`：152 项、38.920s、全部通过；其中再次单跑 compiler batch + package layering 共34项通过（package layering 17项）。
- 统一 changeset 收口使用 `env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m unittest` 定向运行 compiler batch、visibility、compiled cache、information set、七策略、Global/K2、线程预算、canonical runtime、矩阵生成、calendar month、ESS quantile/strategy 与 package layering 共13个模块（含强化后的跨进程锁测试）：162项、22.815s、全部通过。`scripts/check_model_configs.py` 为 `checked=5150 passed=5150 hard_failures=0 warnings=0`，Ensemble audit `failures=[]`，矩阵生成器为 `expected=4689 rewrite=0 delete=0`；compileall、added-line安全扫描与 `git diff --check` 通过。当前项目环境未声明或安装 pyright/mypy/ruff，`uv run --no-sync pyright ...` 明确以 `No such file or directory` 退出，未将其伪报为通过；按用户约定未运行约25分钟的全量 unittest。

---

## OPT-012：接通Direct `single_model_horizon`共享模型语义

- **状态**：已完成
- **优先级**：P0
- **类型**：算法语义修复 / 训练拓扑 / 部署兼容
- **涉及范围**：Direct target plan、trainer、runtime fit计数、forecaster、AIDC pointwise配置及bundle

### 当前事实与问题

1. 权威合同规定`independent_models`为H个独立模型，`single_model_horizon`为共享单模型pointwise。
2. 修复前`single_model_horizon`只在FeatureCompiler增加horizon特征，训练计划仍由标准Direct规则产生`model_count=H`；H=96时每次训练96个模型，31折加final共3,072次scalar fit。
3. `lgbm_direct-pointwise.yaml`旧bundle实测为96 model groups / 96 estimators、80,127,265 bytes，不能代表共享pointwise算法。
4. `lgbm_direct-pointwise-horizon.yaml`与plain变体共享同一算法；前者除原始`forecast_horizon_idx`外增加sin/cos周期编码，后者只使用原始索引。

### 修复方法

1. 增加config-aware `target_plan_for_config()`：保留H个Direct call，但`single_model_horizon`将全部`model_indices`映射到model 0并设置`model_count=1`；标准Direct不变。
2. 复用trainer现有model-group池化机制，按horizon顺序纵向拼接H组X和对应单步Y；sample weight按相同顺序tile。K=1每个fold训练一个estimator，K2 independent训练两个target estimator。
3. builder、trainer和runtime scalar fit计数统一消费同一个计划，防止设计、训练和线程预算再次漂移。
4. Forecaster校验artifact完整target plan；旧96模型artifact即使fingerprint相同，也会明确RAISE并要求重训。
5. 两份route_A baseline LightGBM目标配置采用实测有效的2 window workers × 4 model threads；生成器为唯一配置源，其他模型/路由不外推。

### 实施风险

- 这是模型语义修复，旧新预测不应要求数值exact；旧结果不能继续标注为pointwise。
- canonical fingerprint由配置语义决定，本次代码修复不改变fingerprint，因此必须依赖artifact plan校验阻止旧bundle静默加载。
- Global/K2、quantile和cyclical编码必须保持time-major池化顺序；错误reshape会产生shape正确但目标错位的模型。

### 验收标准

- [x] 两个pointwise配置均解析为1 model group；标准Direct仍解析为H个模型。
- [x] 共享训练X/Y和sample weight按horizon顺序池化，Local K1与Global K2预测正确。
- [x] point/quantile、plain/cyclical的runtime、bundle持久化与reload均通过。
- [x] 旧Direct layout artifact在共享layout配置下明确RAISE。
- [x] 两份真实H=96配置完成31折、final fit、正式预测和bundle核验。
- [x] 生成器4,689份矩阵零漂移，相关定向测试、静态编译与分层门禁通过。

### 实施记录

- `lgbm_direct-pointwise.yaml`新bundle为1 model group / 1 estimator / 1,605,994 bytes，保留36个特征；cache-hit physical run为179.72s。
- `lgbm_direct-pointwise-horizon.yaml`新bundle为1 model group / 1 estimator / 1,648,584 bytes，保留38个特征；冷运行236.38s，其中compiled design 53.11s。
- 两份产物均包含回测2,976行、汇总评分62行、horizon评分5,952行、正式预测96行；visibility proof、唯一键、有限值和bundle reload通过。
- plain旧bundle到新bundle缩小98.00%。旧新训练语义不同，不声明预测数值等价。
- 定向回归共123项、128.435秒、全部通过；其中package layering 17项通过；按用户约定未运行全量测试。

---

## OPT-013：建立统一运行资源规划与性能控制模块

- **状态**：已完成
- **优先级**：P0
- **类型**：性能治理 / 资源规划合同 / 并行预算
- **依赖**：无；OPT-015、OPT-016、OPT-017 依赖本项
- **涉及范围**：typed performance contract、单模型 fixed-step/calendar-month、point/quantile 训练计划、estimator 线程控制、resolved metadata、死配置清理、架构文档与分层门禁

### 当前事实与问题

1. estimator 线程、output worker、quantile worker 和两类 backtest window worker 分散在 `model_forecasting/fit_service.py`、`model_forecasting/runtime.py`、`model_forecasting/backtest_runtime.py`、`model_training/` 与 `probabilistic/` 中解析和执行，一次运行没有唯一的 resolved plan。
2. 当前只有局部互斥：window 并行会压制内部 output/quantile worker，quantile level 与 output worker 互斥；但没有探测物理核、统一限制 OpenMP/BLAS，也没有机械校验 `P_config × P_ensemble × P_window × P_quantile × P_output × P_model <= P_available`。
3. `validation.performance` 不进入 canonical fingerprint，适合承载非语义运行控制；但 `ensemble_parallel_workers`、`step_logging`、`forecast_log_interval` 目前被 schema 接受却没有生产消费点，与“不暴露死参数”约定冲突。
4. 已测结果证明单大 fit、多 scalar output、quantile 和不同 estimator 的最优拓扑不同；不能用一个固定 `n_jobs` 代表全部工作量，也不能把单配置 latency 参数直接当 catalog throughput 参数。

### 目标方案

1. 在稳定合同层增加 immutable typed contract：
   - `RuntimeWorkload`：strategy topology、Direct layout、H/B/K/Q/N、fold/member/model-group/scalar-fit 数、adapter、estimator、训练行数、特征宽度、动态 horizon 和设计体积估计；
   - `RuntimeResourceBudget`：physical/logical cores、总线程上限、内存上限、父级 config/member 并发预算、确定性要求和显式 override；
   - `RuntimeExecutionPlan`：selected axis、config/member/window/quantile/output/model worker、预算乘积、任务数、profile 来源和 rejection/fallback 原因。
2. 在 L2 编排层建立唯一 `resource_planner`。fixed-step、calendar-month、point/quantile 和 Ensemble 只消费 plan；除 parser/planner 外，生产代码不再直接读取原始 `validation.performance`。
3. 每次运行只允许一个主要外层并行轴大于 1；regressor-chain、CQR apply-before-collect、native adapter、Ridge multi-RHS、XGBoost shared multi-quantile 等约束在 planner 中显式表达，不由调用方临时特判。
4. LightGBM/XGBoost/RandomForest/CatBoost 的内部线程参数由 plan 唯一解析；HistGradientBoosting、NumPy/SciPy 的 OpenMP/BLAS 限额必须在创建线程池前统一应用，不能在线程 worker 内竞态修改进程级状态。
5. `resolved_config.json`/result metadata 写入版本化 workload、budget、execution plan、任务数、cache 状态、阶段 wall 和 rejection/fallback 原因；performance 继续不进入 semantic fingerprint。
6. `ensemble_parallel_workers` 由 OPT-015 接线；`step_logging`、`forecast_log_interval` 若无明确生产需求则从 schema、活动 YAML、checker、测试和文档彻底删除，不为保留字段而补空壳。

### 去重边界

- 本项只建立资源合同、预算解析和消费接口；不实现 Ensemble OOF cache（OPT-014）、Ensemble 线程池（OPT-015）、catalog 批调度（OPT-016）、非 LightGBM benchmark/profile（OPT-017）或 compiler fallback 优化（OPT-018）。
- 不重复“运行效率-3”正在处理的 LightGBM H=96 其他策略、H=16 short、宽特征/decomposition/cross-route/K2 标定及其 YAML/文档收口。
- 不建立机器学习自动调优器，不迁移 FeatureCompiler/RawDesign cache/模型算法，不切换第二套 LightGBM/XGBoost backend，不在单配置内用 ProcessPool 拆 horizon/output。

### 实施风险

- 若只迁移部分调用点，会形成 raw performance mapping 与 resolved plan 双重事实源。
- estimator 线程字段会持久化进模型对象；线程计划变化时应验证语义产物与 reload prediction，不错误要求跨线程参数的 `model.pkl` 字节一致。
- 可用核数与内存探测具有平台差异；探测失败必须使用显式保守预算或 RAISE，不能默认为占满机器。
- typed plan 若错误进入 canonical payload 会改变模型 identity；必须保持 semantic fingerprint 排除、resolved metadata 完整记录的双层边界。

### 验收标准

- [x] fixed-step、calendar-month、point、quantile、Local/Global、K1/K2、七策略和 Ensemble 均从同一个 planner 获得执行计划；parser/planner 外零处直接读取 `validation.performance`。
- [x] workload 的 model group、scalar estimator、fold、quantile 和 member 任务数与真实 artifact/fit 调用计数一致，覆盖 Direct pointwise、MIMO、RecMO、Ridge multi-RHS、chain、native 和 XGBoost shared multi-quantile。
- [x] 任一 plan 最多一个主要外层并行轴大于 1；预算乘积不超过可用核数/父级预算，不可控 OpenMP 或显式冲突直接 RAISE，无 silent fallback。
- [x] 各 estimator 的 resolved 线程参数与 plan 一致，OpenMP/BLAS 限额在进入 pool 前生效，并有进程级测试证明不存在嵌套超卖。
- [x] `step_logging`、`forecast_log_interval` 和尚未接线的 `ensemble_parallel_workers` 已彻底清扫；OPT-015 如需成员并行，必须重新引入 typed 合同并由 planner 消费。
- [x] metadata 可直接读出 workload、budget、selected axis、worker/thread、cache 状态、阶段 wall 与失败原因；只改 performance/budget 时 fingerprint/result identity 不变。
- [x] 已验证配置在 planner 接线前后保持回测、预测、评分与 bundle reload 语义；package layering、定向测试、配置审计、compileall 和 `git diff --check` 通过；项目未安装独立 pyright/mypy/ruff CLI，未虚报静态类型检查结果。

### 实施记录

- 新增稳定合同 `RuntimeWorkload`、`RuntimeResourceBudget`、`RuntimeExecutionPlan` 与 L2 唯一 `resource_planner`。workload 区分 model group、logical/scalar/physical fit 和当前真实可并行 output task；Direct pointwise、MIMO、RecMO、Ridge multi-RHS、chain、native 及 XGBoost shared multi-quantile 均有计数门禁。后者当前 pool 在 `fit_position()` 上持锁，因此物理 fit 为 H、可并行 output task 为 1；显式请求 output/quantile 并行直接 RAISE，不伪报并行能力。
- `RuntimeValidationSpec.performance` 改为 immutable typed spec；单模型 fixed-step/calendar-month、point/quantile 与 Ensemble 串行阶段只消费 resolved plan。LightGBM/XGBoost/RandomForest/CatBoost 的线程参数由 plan 写入，HistGB/NumPy/SciPy 通过 `threadpoolctl` 在创建线程池前统一限额。多外层并行轴、显式超预算、内存预算和 estimator 线程冲突均 RAISE；预算 provenance 重复解析保持幂等。
- `resolved_config.json` 与 `result_metadata.json` 现统一写 workload、budget、execution plan、任务数、cache hit/miss/fingerprint、阶段 wall、profile 来源及 fallback/rejection 原因。`validation.performance` 仍完全排除在单模型和 Ensemble semantic fingerprint/result identity 之外。
- 清扫活动 YAML、checker 与测试中的 `step_logging`、`forecast_log_interval`、`ensemble_parallel_workers`；11 份活动 YAML 完成清扫。`add_training_jobs/lgbm_msmd_a.yaml` 原先同时声明 window=2 与 output=4，本项收敛为单一 window 外层轴，避免嵌套超卖。新增项目依赖 `psutil`（核数/可用内存探测）与 `threadpoolctl`（进程级 BLAS/OpenMP 限额）。
- 按开发期限折协议，从现役 short Route A LightGBM Direct physical YAML 派生 5-fold 配置，隔离运行于 `/tmp/tsproj-opt013-5fold-20260904-191408/`，未修改正式 31-fold 几何。实际完成 5 折、final fit、16 步预测和 schema-2 bundle reload；resolved plan 为 `window=1/output=8/model=1`、预算乘积 8，RawDesign 首次 miss、reload hit，预测与 CSV 最大绝对差 `1.8189894035458565e-12`。
- 最终验证：16 个定向模块 168 项、157.665 秒全部通过（TASK27 完整执行矩阵 46/46）；`scripts/check_model_configs.py` 为 `checked=5150 passed=5150 hard_failures=0 warnings=0`；Ensemble audit 为 `failures=[]`；矩阵生成器为 `expected=4689 create=0 rewrite=0 delete=0`；compileall 与 `git diff --check` 通过。项目未安装独立 pyright/mypy/ruff CLI，未虚报静态类型检查结果；按用户约定未运行约 25 分钟的全量 unittest。

---

## OPT-014：修正 Ensemble OOF 缓存作用域并接通成员 RawDesign 缓存

- **状态**：已完成
- **优先级**：P0（本会话待办中最先执行）
- **类型**：Ensemble 缓存复用 / 内容寻址 / single-flight
- **依赖**：无；完成后再执行 OPT-015
- **涉及范围**：`model_ensemble/runtime.py`、`model_ensemble/cache.py`、Ensemble runtime services、成员 runner 构造、OOF cache 测试与审计

### 当前事实与问题

1. `model_ensemble/runtime.py` 构造 OOF fingerprint 时把 fusion `method` 及其参数放入 ensemble payload；但融合方法只消费已经生成的 OOF，不改变成员训练、fold 或成员预测。同一成员集的 averaging/weighted/linear blending/stacking 因此无法共享 OOF cache。
2. OOF cache 逐文件使用原子替换并校验 manifest/hash，但没有同 key 的线程锁和跨进程锁；多个方法并发共享新 key 后可能重复生成，或在一个 writer 尚未完成全部文件时被另一个 reader 视为不完整缓存。
3. 单模型入口会向 `CanonicalBaseModelRunner` 传入 `compiled_cache_root`；Ensemble 的 `runner_factory` 当前只接收 config/registry/origin，成员监督设计不能复用 `results/_compiled_features/`。OOF cache 命中虽能跳过 OOF fit，但 final refit 仍可能重复冷编译设计。

### 修复方法

1. 将 OOF fingerprint 限定为成员顺序与成员 semantic fingerprint、OOF geometry、problem/probabilistic axes、实际 source SHA-256 和 OOF schema/实现版本；fusion method/params 只进入 Ensemble artifact/config fingerprint，不进入 OOF key。
2. 复用 compiled feature cache 的可靠模式，为 OOF cache 增加同 key 的线程内锁、跨进程文件锁、持锁后二次 load、单 writer 和完整 payload/hash 校验。
3. 扩展 Ensemble runtime service/factory 合同，显式传入与单模型一致的 compiled cache root；不同 estimator 但 raw design 相同的成员命中同一 RawDesignFingerprint，output-root 覆写时两类 cache root 保持一致。
4. 保持成员 config/source 校验、source hash、OOF fold、member order、实际预测和 bundle 身份不变；不在本项增加任何训练并行。

### 实施风险

- 过度排除 fingerprint 字段会错误复用不兼容 OOF；method 可排除，但成员、fold、problem/probabilistic 轴和实际 source 内容不得遗漏。
- 旧 method-qualified cache 与新共享 key 不兼容；应按新 schema/key 自然失效，不扫描或猜测迁移旧目录。
- 延迟或重排 runner 构造不能绕过成员引用、数据源和 forecast origin 校验；final refit 所需设计仍必须存在，只是应从 RawDesign cache 加载。
- OOF CSV 十进制往返与内存 ndarray 可能有末位差异；hash 校验口径必须沿用当前 cache 合同，不能混用字节 exact 与语义 tolerance。

### 验收标准

- [x] 同一成员/fold/problem/probabilistic/source 的四种 fusion method 得到同一 OOF key；改变任一成员 fingerprint、fold geometry、source byte、目标轴或 quantile grid 时 key 必须变化。
- [x] 第一种方法生成 OOF 后，其余方法直接命中同一 cache，`generate_oof()` 不再调用；各自 Ensemble fingerprint、融合 artifact 和结果目录仍保持独立。
- [x] 线程与 spawn 双进程同 key miss 只生成一次，其他调用方等待后加载完整 cache；损坏、缺文件、hash/schema 不符继续 RAISE。
- [x] Ensemble 成员 runner 使用与单模型一致的 RawDesign cache；跨 estimator 的相同设计可命中，改变 data/features/strategy/origin 时正确失效。
- [x] 修改前后 OOF tensor、融合结果、四类语义 CSV、成员 bundle 与 ensemble bundle reload prediction 保持既定等价合同。
- [x] OOF/compiled cache 审计、Ensemble runtime/parity/persistence、package layering、静态检查和 `git diff --check` 通过。

### 实施记录

- OOF cache schema 提升为 2；key 只包含有序成员名与成员 semantic fingerprint、problem/probabilistic、OOF geometry、实际 source SHA-256 和 schema 版本。fusion method/params 与 `config_ref` 路径不再进入 OOF key；Ensemble 自身 fingerprint、artifact 和结果目录仍保留方法/引用差异。四方法端到端测试实际得到同一 OOF fingerprint，cache 状态为 `[miss, hit, hit, hit]`。
- 新增 `get_or_create_oof_cache()`：同 key 使用进程内 `threading.Lock` 与 POSIX/Windows 条件导入的跨进程文件锁，锁内二次读取并由单 writer 生成。线程竞争和 macOS spawn 双进程竞争均实测只调用一次 factory；已有部分 payload、metadata/schema/hash 损坏继续 RAISE，不静默重算。旧 schema-1 目录按新 key 自然失效，不扫描、迁移或删除。
- `EnsembleRuntimeServices.runner_factory` 改为显式 `BaseModelRunnerFactory` Protocol，并强制接收 `compiled_cache_root`。两个同 data/features/strategy、不同 estimator 的真实成员 runner 实测得到同一 RawDesign fingerprint：第一个 miss、第二个 hit，磁盘仅一份 `_compiled_features` payload；data/features/strategy/origin 的失效边界继续由 compiled-cache 合同测试覆盖。
- OOF 生成收口到 `generate_oof_for_config()` 公共窄入口；runtime 只在 cache miss 的 single-flight 临界区生成 OOF，fusion fit、final member refit 和持久化保持锁外串行，本项没有新增成员线程池或统一资源规划模块。并发两个融合方法的 RED 基线执行 8 次成员 OOF fit，接线后固定为 `2 folds × 2 members = 4` 次。
- 结果返回值、`resolved_config.json` 和 `result_metadata.json` 新增 `oof_cache_hit`，可直接审计本次运行是生成还是复用。权威设计文档同步记录 method-independent key、single-flight 和成员 RawDesign cache 合同。
- 验证证据：
  - `python -m unittest tests.test_ensemble_oof tests.test_ensemble_quantile_blending tests.test_ensemble_runtime tests.test_audit_ensemble_configs tests.test_ensemble_loader tests.test_ensemble_specs tests.test_ensemble_methods tests.test_ensemble_parity tests.test_ensemble_persistence tests.test_config_entrypoints tests.test_compiled_feature_cache tests.test_package_layering`：122 项，93.683 秒，全部通过；
  - `python scripts/audit_ensemble_configs.py`：`model_yaml_total=5150`、`ensemble=78`、`member_reference_count=228`、`orphan_oof_cache_dirs=[]`、`failures=[]`；
  - `python scripts/check_model_configs.py`：`checked=5150 passed=5150 hard_failures=0 warnings=0`；
  - `python -m compileall -q model_ensemble tests/test_ensemble_oof.py tests/test_ensemble_runtime.py run.py` 与 `git diff --check`：均通过。
- 按用户要求暂不建立 OPT-013 统一资源规划模块，也未实施 OPT-015 Ensemble 成员/final-refit 并行；本次只完成无新增并行轴的缓存复用优化。按常规改动策略未运行约 25 分钟的全量 unittest。

---

## OPT-015：在统一预算下实现 Ensemble 成员与 final refit 并行

- **状态**：已完成
- **优先级**：P1
- **类型**：Ensemble 热运行性能 / 单层并行
- **依赖**：OPT-013、OPT-014
- **涉及范围**：`model_ensemble/oof.py`、`model_ensemble/trainer.py`、Ensemble runtime、resource plan 消费、bundle/结果 parity 与真实性能基准

### 当前事实与问题

1. OOF 当前按 `fold → member` 双层循环串行 fit/predict；final refit 也逐成员串行。成员在同一 fold 内相互独立，但当前没有利用该并行空间。
2. 直接新增 member pool 会与成员自身的 window/quantile/output/model thread 叠加；在 OPT-013 之前实施会形成第五个分散并行轴并放大 CPU/RSS。
3. OOF fold、CQR 和结果审计具有时间顺序要求；可以并行成员计算，但不能让 future 完成顺序改变 fold/member/bundle 顺序。

### 实施方法

1. planner 选择 `ensemble_member` 为主并行轴时，在同一 fold 内并行成员 fit/predict，并把每个成员内部 window/quantile/output worker 压为 1；estimator thread 由剩余预算决定。
2. fold 继续按时间顺序执行和归并；worker 返回不可变 member result，主线程按配置 member order 重组 OOF，异常必须携带 fold/member 坐标。
3. final refit 可按成员并行，复用 OPT-014 的 RawDesign cache；主线程按固定 member order 构建 self-contained Ensemble bundle。
4. `ensemble_parallel_workers` 由 typed spec/planner 正式消费；若 planner 判定内存或模型能力不安全则明确 RAISE，不静默回退到串行后仍报告已并行。

### 实施风险

- 同时驻留多个成员设计、transform 和模型会放大 RSS；QuantileRegressor、CatBoost、宽特征、多目标不能套用 LightGBM K1 的 worker 数。
- 线程 worker 内修改 OpenMP/BLAS 全局限制会竞态，必须由 OPT-013 在建 pool 前完成。
- Learned fusion 的 outer-holdout 隔离、CQR apply-before-collect 和 OOF chronological order 不得因并行完成顺序改变。

### 验收标准

- [x] OOF 同 fold 成员与 final refit 确实并发，成员内部 worker/线程预算与 `RuntimeExecutionPlan` 一致，总预算不超限。
- [x] 串行/并行的 OOF tensor、fold/member order、四种融合 artifact、四类语义 CSV、成员 bundle 和 ensemble bundle reload prediction 保持既定等价合同。
- [x] worker 异常包含 fold/member 坐标，无 silent fallback；CQR、outer cutoff 和时间顺序门禁全部通过。
- [x] 代表性 point/quantile Ensemble 完成至少三轮交错独占基准；无候选达到采用门槛，已如实记录 no-op 裁决、CPU/RSS/swap 与 parity。
- [x] `ensemble_parallel_workers` 在 resolved metadata 中记录配置值、有效值、selected axis 和降级/拒绝原因；全量活动配置不存在“声明但不消费”。
- [x] Ensemble runtime/oof/parity/persistence、资源规划、package layering、静态检查和 `git diff --check` 通过。

### 实施记录

- `RuntimePerformanceSpec` 重新以 typed 非语义字段接入 `ensemble_parallel_workers`；planner 选择 `ensemble_member` 时按父预算生成成员子计划，成员内部 window/quantile/output 轴固定为 1，剩余预算分配给 estimator thread。显式超成员数、线程或保守内存门槛直接 RAISE。
- OOF 保持 fold 串行，只在同 fold 内并行成员；final refit 同样并行。主线程按配置 member order 重组结果与自包含 bundle，worker 异常写入 fold/member 坐标；OOF cache miss 与 final fit 均在成员 pool 创建前的同一 `threadpool_limits` 内执行。serial Ensemble 保留成员各自已标定的最大 native thread，不再统一压成 1；单模型 YAML 声明 `ensemble_parallel_workers` 会在 planner 直接 RAISE。resolved metadata 同时持久化 Ensemble plan 与各 member workload/budget/plan/cache/training-compile。
- `/tmp/tsproj-opt015-benchmark/` 使用显式派生且最多 5-fold 的配置、关闭 OOF cache 并复用 RawDesign cache，按 `serial→parallel` 三轮交错。point median/p95 为 `12.483/12.547s → 54.598/60.080s`；quantile 为 `3.012/3.076s → 14.721/15.305s`，两类 member 并行均显著变慢。原因是异构成员不均衡且 member 轴会压制已标定的 output/quantile 内部并行，因此活动 YAML 保持串行，不为“必须采用”写负收益 profile。
- 两组全部运行 swap 增量为 0；各自语义 CSV、combined prediction 与 bundle reload hash 唯一，reload 最大绝对差为 0。最终统一定向回归 222 项、123.089 秒通过，TASK27 为 46/46。

---

## OPT-016：建立跨配置 RawDesignGroup 批调度器

- **状态**：已完成
- **优先级**：P1
- **类型**：catalog throughput / 跨配置 DAG / 断点续跑
- **依赖**：OPT-013；复用已完成的 OPT-007
- **涉及范围**：新批运行入口、RawDesignGroup/FoldTransform 分组、父级资源预算、任务状态、独立结果持久化和批量性能验收

### 当前事实与问题

1. `run.py` 一次只执行一个 physical YAML；RawDesign cache 已能避免跨 estimator/probabilistic/output/performance 的重复冷编译，但每份配置仍独立执行 cache load、fold slicing、transform 和模型任务调度。
2. 多个 `run.py` 并发时，每个子运行都按单配置预算决定 worker，无法协调总 CPU/RSS；单配置 latency 最优参数可能降低整个配置目录的吞吐。
3. 大矩阵中大量配置共享 data/features/strategy/fold geometry，只改变 estimator 或 fit-time transform；当前没有把这些配置组织成共享设计 DAG，也没有 model-task 粒度的可恢复状态。

### 目标方案

1. 建立显式 `RawDesignGroup → FoldTransformBranch → ModelFitTask → PerConfigPersist` DAG：raw X/Y 每组加载一次，连续 fold slice 一次，相同 feature/target transform branch 复用，再调度不同模型任务。
2. 父调度器持有总 `RuntimeResourceBudget`，向每个子任务传递剩余预算；任何子配置不得自行占满机器。只允许有限数量的大 design 常驻内存，并记录峰值 resident design/RSS。
3. 保留每个 physical YAML 的独立 canonical fingerprint、result identity、resolved config、bundle、回测和预测结果；批执行不能合并或省略任何配置，也不能抽代表组合替代全量。
4. 状态按 model task 原子落盘，支持进程中断后跳过已通过任务、重跑失败任务和最终独立聚合复核；失败必须保留 config/fold/model 坐标。

### 明确边界

- 本项优化目录级吞吐，不替代单配置 `run.py`，不改变模型语义或验证几何。
- 本项不重新实现 RawDesign fingerprint/cache（OPT-007）或资源计划（OPT-013），只消费其合同。
- 不纳入“运行效率-3”正在做的具体 LightGBM profile/YAML 标定；该会话产出的 profile 作为输入，而不是本项工作内容。

### 实施风险

- 同时驻留多个百 MiB 设计矩阵可能比逐配置更慢或触发 swap；必须优先按内存限流，不能只按 CPU 排队。
- fold transform、decomposition 和 feature selection 只可在完全相同的训练索引与语义配置间复用，错误分组会形成跨配置污染。
- 部分模型失败时仍要保证其他结果与单配置路径一致，不能为续跑引入半成品 bundle 或不完整 CSV。

### 验收标准

- [x] 分组键逐项覆盖 RawDesign 与 fold-time transform 的全部语义输入；改变任一输入会分组，只有 estimator/output/performance 等非 raw 字段变化时才安全复用。
- [x] 同一设计组只加载/物化一次 raw payload，相同 fold transform branch 只拟合一次；调用计数测试与运行 metadata 可审计。
- [x] 按本次获准方案约束父级CPU乘积、估算工作集准入与有限缓存，记录执行期采样RSS，无嵌套预算放大；不把进程内native模型宣称为OS硬内存隔离（验收边界澄清见新增收口记录）。
- [x] 中断后可按完成的fold/model拟合单元续跑；最终verifier验证清单数量、唯一配置与必需产物的内容/身份/几何，不仅检查文件存在。
- [x] 批路径与逐 YAML 路径的语义 CSV、bundle reload prediction、canonical config 和 identity 一致；运行时不改写 fold/horizon/quantile/model 数。
- [x] 真实多模型设计组完成三轮独占对照；最终复验 batch median 改善 15.990%、p95 改善、无 swap 增量/OOM，并记录 CPU 与峰值 RSS。
- [x] 单配置入口回归、批调度定向测试、package layering、配置审计、静态检查和 `git diff --check` 通过。

### 实施记录

- 新增独立入口 `batch_run.py` 与 L2 `model_forecasting/batch_runtime.py`。物理 YAML 先按 `RawDesignFingerprint` 分组；首个 runner 编译/加载 raw X/Y，组内后续 runner 直接共享同一内存 payload。precompiled payload 必须由 runner 重新计算 fingerprint 并校验匹配。`FoldTransformCache` 的键包含 raw fingerprint、训练 origin indices、problem 轴和完整 transformation 语义；不同 scaling/decomposition 不会误命中。
- DAG 执行为 `RawDesignGroup → FoldTransformBranch → ModelFitTask → PerConfigPersist`。每组只驻留一份 raw design；父 `RuntimeResourceBudget` 通过 `parent_concurrency` 分配给 config worker，线程乘积机械受限。并行配置在父级 `threadpool_limits` 内调用 prelimited runtime，避免 worker 竞态修改进程级 BLAS/OpenMP；文件绘图固定使用非交互 `Agg` backend。
- task state 以 JSON 临时文件 + 原子替换落盘，并由同 batch ID 的线程锁 + OS 文件锁串行化；batch ID 与输入路径顺序无关。中断后只跳过产物完整的 completed task，失败项单独重跑；resume 只更新本轮实际执行任务的 batch provenance，不覆盖历史 completed task。`verify_batch_results()` 校验输入/task 数、路径唯一和五类核心产物；resolved config 追加 batch id、group、父预算、transform cache、RSS provenance。单配置 `run.py` 未被替代。
- calendar-month 的动态 horizon runner 通过 `_SharedBatchRunnerFactory` 继承批子预算、compiled cache root 和 fold-transform cache；动态 raw fingerprint 不同则不复用 payload。组 metadata 的 raw load 数按真实成功构造计数，构造全失败时记录 0。
- 真实采用组为 short Route A baseline 的 XGBoost + CatBoost、H=16、Direct pointwise，二者都显式限制 `model_thread_count=1`，父 `config_workers=2` 的总模型线程乘积为 2；按开发协议仅派生 `fold_count=5`，其余模型/数据/特征/horizon 不变，目录 `/tmp/tsproj-opt016-benchmark/`。门禁修复后最终三轮 sequential median/p95 为 `40.470/41.867s`，batch 为 `33.999/35.087s`，median 改善 15.990%（此前为 19.502%，本轮覆盖旧性能结论）；CPU median `40.032→42.580s`，最大 RSS `469,909,504→489,439,232 bytes`，六次 swap used 增量均为 0。每轮 batch raw payload 只加载一次、transform cache `6 miss + 6 hit`；两模型三类 CSV hash 唯一，identity 一致，bundle reload 最大绝对差 `1.8189894035458565e-12`。证据为 `metrics/closeout-xc1-{sequential,batch}-{1,2,3}.json` 与 `metrics/closeout-summary.json`，收益仅指 wall，CPU 与 RSS 本轮略升。
- 子预算门禁收口：runner 先按父预算构造，当前组全部 runnable 的显式计划在任何模型执行/线程池创建前统一检查，再重新解析子预算计划；避免构造期超预算被捕获成 task failed，以及 available 被重复分割。测试使用 H=8、安全 lag、固定 8-thread 父预算，保留真实 compile，并断言 `run/run_prelimited` 均未调用；validation 变体从 mapping 重建，避免 typed 字段与序列化 payload 脱节。真实 LGBM+Ridge 5-fold 夹具得到 `budget_product=8, available=4 (config_workers=2)`、执行次数 0；Ridge 单独成功，reload 最大差同上。
- 仅把 `/tmp/tsproj-opt016-benchmark/lgbm_direct_5fold.yaml` 的 `multi_output_n_jobs` 从 8 改为 4，未修改正式 YAML。该 LGBM+Ridge 组最终三轮 median `29.725→29.591s`（改善 0.453%）、p95 `29.782→29.624s`；未达 10% 门槛，不作为性能采用组。六次运行无 swap 增量、三类 CSV hash 唯一、identity 一致，reload 最大差同上；两模型 scaling 不同，transform cache 为 `12 miss + 0 hit`。证据为 `metrics/closeout-lr4-{sequential,batch}-{1,2,3}.json`。
- 最终统一定向回归 222 项、123.089 秒通过，TASK27 为 46/46；完整命令与逐门禁退出码记录于 `/tmp/tsproj-opt016-benchmark/metrics/closeout-verification.json`。两组每次实际回读均为每模型 5 fold、80 行回测、16 行预测，并完成 bundle reload；5-fold 只用于开发性能与 parity 证据，不宣称正式 31-fold 结果已落盘。

---

### 2026-09-05 实现偏差与剩余待办

RawDesign 内存共享、transform single-flight、同 batch 锁、普通两配置子预算和 XGBoost+CatBoost 三轮收益均保留。重新打开本项是因为通用调度验收大于当前实现，不是推翻 15.990% 的历史基准。

#### A. 父预算与内存限流（P1，待处理）

- **事实**：`model_forecasting/batch_runtime.py:416-420,507-510` 用 `effective_workers` 覆盖已有 `parent_concurrency`，没有组合上层并发约束；`RuntimeResourceBudget.available_threads`（`forecasting_core/runtime_resources.py:108-111`）又把不足一线程的除法结果抬到 1，不能单独证明父层总乘积合法。普通 `parent_concurrency=1/config_workers=2` 的既有测试不覆盖这些输入。
- **事实**：`resource_planner.py:289-294` 校验单 runner 的 raw design 估计，而 batch 在 `batch_runtime.py:428-463` 先构造全部 pending runner；`FoldTransformCache._values` 和动态 `_SharedBatchRunnerFactory._payloads` 都没有容量/淘汰约束。动态月份可能驻留多个不同 fingerprint，故“每组只驻留一份 raw design”仅适用于当前 fixed-step 共享路径。`peak_rss_bytes` 在构造后和整组结束采样，不是持续峰值测量或内存硬限制。
- **影响/方向**：嵌套调用可能放大 CPU 预算；长窗口、多 transform 分支和动态月份可能累积内存。补齐父预算组合及最小并发可行性校验；按共享 raw、transform branch 与并发任务估计做组级 admission/缓存限额，并明确实测 RSS 与估计预算的区别。不能用一次无 swap 的窄基准证明所有 catalog 不超限。
- **风险**：计数不能重复计算共享 raw；回收仍被模型使用的 transform 会破坏结果。先写超预算拒绝、缓存生命周期和动态月份测试，再实施。
- [x] 嵌套父并发、可用线程少于 config worker 时拒绝或按显式合同限流；任何路径不得放大父预算。
- [x] 有限内存下多分支/动态月份不无界缓存；峰值指标与实际测量方法一致，并验证 batch-vs-single parity。

#### B. completed 产物真实性门禁（P1，待处理）

- **事实**：`_core_artifacts()` 声明五类文件，但 `_artifacts_complete()`（`batch_runtime.py:277-291`）只遍历调用方提供的值检查 `is_file()`，没有验证必需键、非空内容、hash 或模型身份。本次空 `artifacts={}` 探针通过 helper，进一步通过 `verify_batch_results()`，返回 `verified_count=1`。空文件/被替换的内容也不会仅因存在性检查而被拒绝。
- **影响/方向**：resume 可跳过伪 completed 任务，最终 verifier 可误报完整。先强制验证必需字段和非空有效文件，再核对配置身份、CSV schema/唯一键/期望几何及 bundle 合同；记录可复核摘要或 hash。按 point/quantile 等配置补充必要产物，不把空字典当完整结果。
- **风险**：hash 写入时机须晚于 metadata 最终更新；加强状态 schema 后旧 state 如何重验收必须明确，不能静默继续相信旧 completed 标记。
- [x] 空/缺键 manifest、缺文件、空文件、损坏 CSV/bundle、错误 identity 都不能通过 verifier 或被 resume 跳过。
- [x] 正常完成/恢复后输入清单与真实完整产物一一对应，损坏项按明确策略拒绝或重新执行。

#### C. 失败状态与恢复粒度（P2，待处理）

- **事实**：task ID 仅取物理 YAML 路径（`batch_runtime.py:117-118`），每次 `execute()` 调用完整 `runner.run/run_prelimited`，所以恢复粒度是**配置**，不是 fold/scalar model 的持久化 checkpoint。异常 state 目前记录 config task 下的 type/message，没有统一结构化 fold/model 坐标。
- **事实**：显式预算检查发生在当前 group 构造后、执行前（`batch_runtime.py:399-523`），不是全 batch 开始前；后续组拒绝时，前面的组可能已经完成。预算拒绝、异构线程拒绝位于 task 执行异常捕获之外，已写入 `running` 的任务不会在本轮统一转为 failed 并记录原因；下次可重试，不应把它说成不可恢复。
- **方向/风险**：补齐 preflight 失败的原子终态和结构化坐标。对原目标中的 model-task 粒度恢复，需要明确实施 fold/model checkpoint，或经用户裁决将合同收窄为配置级恢复；不能仅因配置级 resume 测试通过就勾选更强能力。细粒度 checkpoint 涉及模型状态、CQR 顺序及缓存有效性，不能当文档任务顺手实施。
- [x] 后续组预算拒绝/异构线程拒绝后，state保留准确失败原因且无遗留伪running；本次已实现全批preflight后才执行模型。
- [x] 原子恢复单位、异常坐标和验收用例一致；补齐原model-task合同，没有收窄为仅配置级resume。

### 2026-09-05 新增收口记录

上面A/B/C事实为修复前诊断，保留溯源；对应缺口已由`for_children`、全批preflight、bounded cache、batch_artifacts和FitCheckpoint修复。用户确认的方案明确区分“资源估算/准入/缓存限制”和“操作系统硬内存隔离”，因此替换原无法由线程池保证的RSS绝不超限表述，而不是宣称一次窄基准证明全catalog硬内存安全。最新统一定向267项及5150配置checker证据见页首；自然月quantile batch完成/resume、损坏manifest及checkpoint、嵌套预算、后续组提前拒绝、final故障后复用fold均有真实测试。生成器4689项零漂移、Ensemble审计无失败、eligibility仍为4265/807；静态compileall与diff检查通过。AGENTS.md和权威设计文档§7.2已同步。原吞吐三轮数字属于新增checkpoint之前的历史实测，不将其改写成checkpoint开销复测。

---

## OPT-017：标定非 LightGBM 模型的资源策略

- **状态**：已完成
- **优先级**：P2
- **类型**：模型感知性能 profile / 线程与内存标定
- **依赖**：OPT-013
- **涉及范围**：XGBoost、CatBoost、RandomForest、HistGradientBoosting、Ridge、ElasticNet、Lasso、QuantileRegressor、SeasonalTemplate 的代表性拓扑、profile provenance、配置生成器与性能测试

### 当前事实与问题

1. 现有真实性能标定主要覆盖 LightGBM；其他模型多数依赖 `_DEFAULT_OUTPUT_WORKERS_BY_MODEL` 和 estimator 默认线程，缺少与 H/K/Q、训练行数、特征宽度、adapter 和内存等级绑定的交错 benchmark。
2. 各模型并行语义不同：XGBoost 有 shared multi-quantile，CatBoost 使用 `thread_count`，RandomForest 有树内并行，HistGB 由 OpenMP 控制，Ridge 已有 multi-RHS，Lasso/ElasticNet 不可替换为不同目标函数的 MultiTask 版本，QuantileRegressor 为 LP 重内存，SeasonalTemplate 的 NNLS 很轻且可能受日志 I/O 主导。
3. 将 LightGBM 的 `4×2` 或 `1×8×1` 复制到其他 estimator 没有证据，也可能造成 oversubscription、负收益或 RSS 放大。

### 实施方法

1. 按真实 scalar task 拓扑而非策略名分组：单大 fit、H groups×1 output、1 group×H outputs、MO groups×B outputs、quantile level 和 Global/K2。
2. 每类模型选择当前活动配置中的代表性窄/宽设计，固定 seed、fold 和模型参数，只改变 planner 允许的 `{outer_workers, model_threads}`；每个候选至少三轮交错独占运行。
3. profile 记录 estimator/version、strategy topology、H/K/Q、rows/features、scope、机器核数/内存、median/p95、CPU、RSS、swap 和 bundle parity，不满足适用签名时回到保守默认或 RAISE。
4. 只采用不改变模型耦合、objective、backend 和序列化接口的调度优化；原生容器复用、warm start 或算法替换必须另立语义任务。

### 去重边界

- 不重复“运行效率-3”的 LightGBM H=96/H=16/宽特征/K2 标定。
- 本项不实现 planner（OPT-013）或批调度（OPT-016），只生成并验证非 LightGBM profile。

### 实施风险

- 小 Ridge/ST 任务线程池开销可能超过 fit；不能把占满物理核当作目标。
- RandomForest 原生 multi-output、MultiTaskLasso/ElasticNet 等会改变模型耦合或目标函数，不属于透明性能优化。
- QuantileRegressor、CatBoost 和宽特征多目标可能先受内存限制，必须以 RSS/swap 门禁高于 CPU 利用率。

### 验收标准

- [x] 九类非 LightGBM 模型均有至少一个真实代表拓扑基准；未达到收益门槛的模型明确记录保留默认值，不为“必须有改动”强行写 profile。
- [x] 采用项三轮 median wall 改善至少 10%、p95 不恶化、无 swap/OOM，并同时报告 CPU 总时间、平均核和峰值 RSS。
- [x] 同一模型不同H/K/Q/rows/features超出签名范围时不会错误命中profile；真实上下文、版本、source内容与调度后计划均可核验，metadata不再仅为来源标签。
- [x] 串行/候选的语义 CSV和 bundle reload prediction 保持既定等价合同；RandomForest 因 exact deterministic gate 失败未采用。
- [x] 生成器/YAML 只写入已验证范围，幂等检查零漂移；模型线程、量化训练、adapter、配置审计和静态检查通过。

### 实施记录

- `/tmp/tsproj-opt017-benchmark/` 对九类模型各完成 `A→B` 三轮交错，共 54 个独立进程。八类活动模型使用 short Route A、H=16、5-fold 派生配置；仓库活动集没有 QuantileRegressor YAML，故 QR 明确以现役月频数据/geometry 构造 H=1、3-fold quantile 代表，不把结果推广为活动 profile。全部运行 swap 增量为 0，resolved metadata 回读的 worker/thread 与候选一致。
- median wall（A→B）与裁决：XGBoost `40.166→39.162s`（+2.50%，保留）；CatBoost `44.191→35.546s`（+19.56%，采用 `model_thread_count=8`）；RandomForest `81.393→33.593s`（+58.73%，但 n_jobs=8 在不同轮产生最多 `7.28e-12` 的预测归约差异，违反 deterministic gate，保留）；HistGB `27.350→35.536s`、ElasticNet `23.655→23.744s`、Lasso `23.710→23.739s`、QR `0.935→0.988s` 均变慢；Ridge `18.283→18.215s`、ST `18.247→18.176s` 均不足 1%，全部保留默认。
- CatBoost p95 `44.197→35.667s`，CPU median `40.454→54.738s`、平均核 `0.919→1.547`、最大 RSS `417,988,608→451,936,256 bytes`；三类语义 CSV hash 和 bundle reload prediction 保持 exact。只为 `aidc_load_15min_short/route_A/baseline/cab_direct-pointwise.yaml` 及唯一生成器规则写入该非语义 profile，不推广 route_B、其他 H/K/Q/features/scope。
- 生成器写盘前 `rewrite=1/create=0/delete=0`，写盘后 4,689 份矩阵为 `rewrite=0/delete=0`。全量 checker 为 `checked=5150 passed=5150 hard_failures=0 warnings=0`；最终统一定向测试 222 项、123.089 秒通过。

---

### 2026-09-05 适用签名复核与剩余待办

- **已完成边界**：九模型三轮实验、CatBoost 单份 YAML/生成器采用规则、其他模型保留默认的裁决保持有效；不重新跑 54 个进程，也不把 RF 的浮点差异改成容忍后强行采用。
- **当前事实**：`scripts/generate_load_15min_matrix.py:427-434` 仅按 model/scenario/route/group/strategy_variant 写 `model_thread_count=8`，不比较 H/K/Q、rows/features、scope、数据版本或 estimator 版本。`resource_planner.py` 的 `profile_source` 仅区分 `automatic_default`、`validation.performance` 等来源，当前无 benchmark signature 匹配器。元数据能审计实际 workload，不等于能拒绝超出已测签名的性能采用规则。
- **影响**：原验收“超出签名不会错误命中”尚无实现；在相同命名场景下改变 horizon、特征或数据规模后，生成器仍可能沿用旧规则。这是性能适用性证据缺口，不是已经证明当前 CatBoost 预测错误。
- **建议方向**：为已采用规则绑定最小、可核验的 benchmark provenance 与适用签名，生成或执行时机械检查；签名失配时不自动复用已标定规则。用户显式 performance override 与“系统已验证 profile”须区分，不建立无需求的自动调参系统。如果只保留按物理场景人工维护，需要用户明确同意收窄原签名验收合同，不能直接勾选。
- **风险**：过细签名使数据更新后频繁失配，过粗签名仍会错误外推；签名与性能 provenance 不得进入 semantic fingerprint。五折标定和正式 31-fold 是不同证据层级，不能补写不存在的正式产物验收。
- [x] 对唯一采用的 CatBoost 规则，改变 H/K/Q/rows/features/scope/版本时，测试证明不会自动宣称命中原已验证 profile；保留用户显式 override 的明确边界。
- [x] 采用签名、实验来源与 resolved workload 可互相核对；生成器幂等、模型线程与配置审计通过，没有收窄为人工维护合同。

### 2026-09-05 新增收口记录

上段为签名实现前的诊断，保留历史证据。`model_forecasting/performance_profiles.py`现为唯一profile实现；物理YAML和生成器只为原CatBoost采用项声明`profile_ref`，不扩大配置范围。planner消费实际base_dir、workload和feature_schema，核验来源内容、模型版本、语义签名与最终执行计划；无profile的人工线程设置标为显式override。新机制不进入semantic fingerprint，也不把历史5-fold标定升级为正式31-fold收益。`tests.test_performance_profiles`已纳入页首267项定向回归；5150配置checker通过、4689生成计划零漂移。AGENTS.md和权威设计文档§7.2已同步。

2026-09-06 后续可靠性收口将 intraday 回测改为正式 forecast origin 的 stride 网格，原 5-fold benchmark 的折语义因此失效。活动 CatBoost YAML 和生成器已撤下 `profile_ref`；旧签名保留作为 fail-closed 历史证据，重新 benchmark 前不得更新 SHA 或恢复活动引用。

---

## OPT-018：补齐 compiler batch fallback 可观测性并按证据优化剩余冷路径

- **状态**：已完成
- **优先级**：P3
- **类型**：冷路径可观测 / fallback 治理 / 选择性向量化
- **依赖**：无；若由批调度统一采集阶段指标，可在 OPT-016 后实施
- **涉及范围**：`model_forecasting/design.py`、`feature_engineering/compiler.py`、运行 metadata、fallback 故障注入与真实性能 profile

### 当前事实与问题

1. `FeatureCompiler.compile_batch()` 自身 fallback 时会 warning；但 `_RegistryDesignBuilder._can_batch_training()` 会在调用前直接改走逐 origin `training_row()`，生产运行无法统一看到是否 fallback、由哪个 transformation/lag/provider 条件触发以及损失多少 wall。
2. `percent_change`、`time_since`、`ewm` 当前未实现训练 batch；浅 lag/provider 依赖也可能要求逐步语义。`expanding` 已有 batch 路径，但为保持 pandas 逐值 exact 仍在 origin 粒度执行统计。
3. OPT-005/006/007/011 已消除主要 pandas 拆分、proof 对象、标签提取和重复冷编译成本；剩余 compiler 优化的绝对上限必须重新 profile，不能沿用旧热点排序。

### 实施方法

1. 将 batch eligibility 解析为结构化结果，而不是 bool：记录 `eligible`、reason code、strategy/call geometry、触发字段和预计 origin×call 次数；row 路径必须在日志与 resolved metadata 中可见。
2. 增加 cold compile 分段计时：materialize、labels、compile_batch/row、batch finish、split/concatenate、proof validation 和 cache load/write；细粒度插桩只用于热点排序，端到端基线使用未插桩新进程。
3. 先用活动配置 inventory 统计 fallback 覆盖范围和真实 wall，再决定逐项实现 percent-change/time-since/EWM batch 或 expanding 前缀算法。
4. 任何新 batch 算法必须保持 feature 值、NaN mask、VisibilityProof 字段/顺序和异常边界；浮点 exact 无法满足时保留当前路径或单独申请 tolerance 合同，不能静默放宽。

### 实施风险

- 仅增加 warning 而无结构化 metadata，批运行时仍无法聚合；但把高频逐 origin 信息全部写日志又会制造新的 I/O 瓶颈，应按配置/阶段聚合。
- provider/浅 lag 的 row 路径可能是策略语义必需，不能为了 batch 命中率改成 persistence 或跳过 as-of proof。
- cProfile 和细粒度 wrapper 会放大绝对耗时，不能把插桩数字当生产 wall。

### 验收标准

- [x] 每次监督设计构造明确记录 batch/row、reason code、origin/call 数和阶段 wall；合法 fallback 无 silent path，批量汇总不产生逐 origin 日志洪泛。
- [x] inventory 能列出全部活动配置的 eligibility/fallback 原因，并与实际运行路径一致；改变相关 transformation/lag/provider 时 reason 正确变化。
- [x] 未新增无证据 batch 算法；既有 batch/row reference 的 Local/Global、K1/K2、七策略等价与故障注入继续 RAISE。
- [x] 剩余 row 阶段虽超过端到端 10%，但全部属于 provider-dependent lag 的顺序语义，已记录 no-op 裁决并保留现状。
- [x] compiler/design/cache 定向测试、package layering、静态检查和 `git diff --check` 通过。

### 实施记录

- 新增 immutable `BatchEligibility`，结构化记录 `eligible/reason_codes/trigger_fields/origin_count/call_count/estimated_origin_call_count`。`FeatureCompiler.compile_batch()` 与 `_RegistryDesignBuilder.training_rows()` 改为消费同一判定，修复 Recursive 配置含 `direct.align_to_target=false` 时 builder 错误绕过浅 lag fallback 的双事实源问题。
- runtime metadata 与每 runner 单条聚合日志记录 `materialize/labels/compile_batch/compile_row/batch_prepare/batch_feature_columns/finish_and_proof_validation/split_concatenate`，RawDesign cache 另记 load/compile/write；cache hit 明确为 `mode=cache_hit`，不伪造上次 miss 的 timing。Ensemble 顶层 metadata 同步写每个 member 的完整资源/编译摘要。
- 新增 `scripts/audit_batch_eligibility.py`。全量结果：5,150 canonical YAML = 5,072 Forecast + 78 Ensemble；4,265 个单模型 batch eligible，807 个 fallback，reason 全部为 `provider_dependent_lag`，活动集没有 percent-change/time-since/EWM fallback。最终 inventory 在 `/tmp/tsproj-opt018-batch-eligibility-final.json`。
- `/tmp/tsproj-opt018-quantile-row-evidence/` 的真实 quantile Ensemble cold run 验证 Direct 为 batch（245 origins×31 calls，compile_batch `0.755s`），Recursive 为 row（同规模，reason=`provider_dependent_lag`，compile_row `10.545s`），端到端 `14.698s`、reload exact、swap 增量 0。该 row 热点必须逐步消费 provider，不能透明 batch 化；因此本项按设计作算法 no-op，而不是用信息语义变化换加速。
- compiler/design/cache 定向回归已并入最终统一收口 222 项、123.089 秒通过，compileall 与 `git diff --check` 通过。

---

## OPT-019：修正 STL/MSTL `trend_forecast` 配置与运行时合同冲突

- **状态**：已完成
- **优先级**：P0
- **类型**：配置正确性 / 目标分解语义 / 运行前校验
- **涉及范围**：`scripts/generate_load_15min_matrix.py::_decomposition_spec()`、target transform normalization、`decomposition/` forecaster、972份活动 STL/MSTL YAML、配置审计与真实fit门禁

### 当前事实与问题

1. AIDC 三个15min场景的生成器对 STL/MSTL 固定写入 `trend_forecast: seasonal_naive`；活动配置中共有972份该声明。
2. canonical parser、target transformation normalizer和 `scripts/check_model_configs.py` 只接受该键，没有把值域收窄到实际运行时能力，因此972份配置均可通过静态配置审计。
3. `DecompositionPipeline` 构造的趋势外推器为 `PolyTrendForecastForecaster`，其运行时只接受 `polynomial|damped`。P4 真实 MSTL fit 在首个回测窗口稳定 RAISE：`decomposition_trend_forecast must be 'polynomial' or 'damped'`。
4. `seasonal_naive` 从名称上属于季节分量外推，而当前字段控制趋势分量；直接把它静默映射为任一趋势算法会改变972份配置的模型语义和fingerprint，不能作为性能任务内的顺手修复。

### 建议方向

1. 先裁决 STL/MSTL 趋势分量应使用 `polynomial` 还是 `damped`；若原意是季节分量 seasonal-naive，则建立独立且类型化的 seasonal forecast 合同，不复用 `trend_forecast`。
2. 在 canonical spec/normalizer 层严格校验实际可运行值域，使非法值在 YAML 加载或审计阶段 RAISE，而不是延迟到第一个fold。
3. 按裁决同步生成器与972份 YAML，并记录 fingerprint、旧结果目录和缓存失效策略；不提供静默兼容映射。
4. 增加真实 target-transform 构造/fit 门禁，覆盖 linear/STL/MSTL，不再以“typed parse通过”代替运行合同验证。

### 实施风险

- 选择趋势外推算法会改变回测、正式预测、bundle状态与全部相关identity，属于明确语义变更。
- 如果只放宽 `PolyTrendForecastForecaster` 接受 `seasonal_naive` 而没有定义其趋势语义，会把配置错误伪装成新能力。
- 972份配置的批量迁移必须保持生成器幂等，并处理已有结果/OOF/cache不复用边界。

### 验收标准

- [x] 用户已明确选择 STL/MSTL 的趋势与季节外推合同，并记录迁移影响。
- [x] 非法 `trend_forecast` 在配置加载/审计阶段直接 RAISE；linear/STL/MSTL 的合法配置均能完成真实fit。
- [x] 生成器与972份活动 YAML 零漂移，配置审计不再出现“静态通过、运行失败”。
- [x] 代表配置完成回测、final fit、正式预测、bundle reload及分解恢复顺序验证。
- [x] fingerprint/结果/缓存处理策略、定向测试、package layering、compileall和 `git diff --check` 全部通过。

### 阻塞原因与实施记录

2026-09-04 P4 decomposition 代表基准首次暴露该问题。按用户要求停止后续性能运行；由于修复需要选择模型语义，本轮只登记问题并阻止 performance 参数推广，不修改972份配置。

- 2026-09-04 用户裁决 STL/MSTL 的趋势分量统一使用 `polynomial`；季节分量继续由内部 `PhaseTemplateForecaster` 处理，不新增只有一个合法值的公共 seasonal 配置字段。
- `decomposition.spec` 新增 `TREND_FORECAST_MODES={polynomial,damped}` 并在 preset 解析期严格校验；`seasonal_naive` 现在于 YAML 加载/配置审计阶段直接 RAISE，不再延迟到首个 fold。checker 复用同一公开值域。
- 生成器与 972 份活动 STL/MSTL YAML 已由 `seasonal_naive` 迁为 `polynomial`。写盘前计划为 `create=0/rewrite=972/delete=0`；写盘后复查 4,689 份矩阵为 `create=0/rewrite=0/delete=0`。这 972 份配置获得新 canonical fingerprint/result identity；旧结果不复用、不在本项删除。target decomposition 不进入 RawDesignFingerprint，因此既有 raw design cache 可继续按其独立合同复用。
- 按用户要求，真实模型测试使用显式派生的 5-fold 配置并隔离在 `/tmp/tsproj-opt019-5fold-OizTwp/`，不修改活动 YAML 的 31-fold 几何。Ridge Direct 的 linear/STL/MSTL 三份代表均完成 5 折回测、final fit、16 步预测和 schema-2 bundle 加载；metadata 回读均为 5 个窗口。bundle reload prediction 与 `prediction.csv` 的最大绝对差均为 `1.8189894035458565e-12`（仅 CSV 十进制往返）。
- 验证：`python -m unittest tests.test_decomposition_spec tests.test_decomposition_equivalence tests.test_canonical_transforms tests.test_generate_load_15min_matrix tests.test_check_model_configs tests.test_package_layering` 共 75 项通过；`scripts/check_model_configs.py` 为 `checked=5150 passed=5150 hard_failures=0 warnings=0`；目标 changeset 的 `git diff --check` 通过。真实模型运行按本项 5-fold 协议验收，不宣称 31 折正式结果已重跑。
