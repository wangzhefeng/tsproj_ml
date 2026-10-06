# 覆盖映射

`test_direct_result_identity` 默认 integration：三类 Direct 方法（含 horizon 的有/无周期编码两种特征变体）的真实小样本训练/回测/预测落盘及元数据、原始语义 hash、文件别名不变性和非法配置拒绝；`test_forecast_config_fingerprint` 在 fast 中覆盖三类前缀及 cyclical 仅影响特征元数据和语义 hash、不新增方法类型。迁移清单只读，不移动存量结果。

天气阶段隔离定向测试：`test_inference_columns` 验证历史训练实测/测试预报、future 不参与历史请求及未来实测列忽略；`test_weather_phase_runtime` 用真实编译、Ridge 生命周期验证文件隔离和缓存不依赖 future；`test_weather_history_coverage` 验证完整 history 覆盖。均默认纳入 integration，不自动运行正式业务模型。

严格原始历史窗口的定向验证：`test_raw_history_window` 覆盖请求下界、single/batch、expanding、窗口扰动、serial/parallel、cache/checkpoint 及入口拒绝；默认 integration 发现，不加入 skip 或 fast 白名单。

rolling 系回测形态（2026-09-27 新增）：`test_window_backtest_modes` 默认 integration，覆盖 sliding_window/expanding_window 的 spec 解析合同（字段缺失/禁字段/非法 horizon_mode 拒绝）、expanding 几何训练集逐折扩大、sliding 重叠折与 stride≥horizon 拒绝、`stitch_overview=False` 产物合同（不拼总图、逐窗图与 csv 照常），以及两种形态的合成数据端到端 backtest-only smoke（metadata mode 标签、训练样本数递增）与 expanding final fit 拒绝。

## 精简与覆盖映射

在线统计：原 `test_online_prediction` 的扩展用例保留原精确断言，`test_online_statistics` 默认 integration，分别覆盖 Direct/recursive 全前缀 single 特征与预测精确对照、默认 batch 既有差异归因、逐点/分批追加等价、同尾不同前缀的数值与 lineage、失败发布原子性、状态防别名、旧 schema/缺统计/计数和 origin 错配拒绝、有限历史模式。EWM 与事件算子的独立 pandas 对照继续由 `test_streaming_statistics` 覆盖。

引用式融合闭环新增测试默认 integration：`test_ensemble_regressions`（成员/原点身份、非整数缓存精确往返、quantile weighted/pinball、顶层 point_quantile 部署），`test_ensemble_strict_contract`（重复 YAML 键、源语义、未知参数、成员未消费选项拟合前拒绝），`test_ensemble_outer_evaluation`（外层标签扰动不改变融合权重/预测、成员窗口、掩码与季节基线传递），`test_ensemble_lifecycle_contract`（缓存写中断、bundle 写失败恢复、真实模型参数/visibility/lineage、CQR），`test_ensemble_panel_alignment`（异构 lag、Global direct+recursive、point/quantile/逐 horizon/动态权重的冷暖及 bundle 对照），`test_ensemble_adaptive_weights`（独立解析期望、时间衰减和未来标签拒绝）。`test_ensemble_methods` 的常量目标 stacking 断言精确恢复目标均值，NNLS 用独立解析权重，避免恒零/固定等权错误实现蒙混通过。

- 核心残留清理：`test_core_cleanup_contract` 默认 integration，验证退役定义无 shim、无效结果筛选/事件 config 参数拒绝、完整多序列多目标 long 表读回、非 canonical 拒绝及 pickle IO 新进程导入不初始化日志。
- `test_probabilistic_contracts` / `test_probabilistic_objectives` 改走生产 `probabilistic_spec_from_mapping()`；保留 quantile/interval/CQR 数值与错误断言，旧 args 新旧冲突检查改为 legacy 字段及 args 对象拒绝，不再维持生产兼容适配器。目标变换往返、融合四方法与 bundle 恢复继续由既有集成测试覆盖；未删测试文件或冻结 fixture。
- 全仓模型数量不锁定历史快照：`fixtures/config_inventory.py` 独立读取物理 YAML，按 estimator/ensemble 建立相对路径及类型清单，不调用生产 loader 的发现规则。catalog 与 checker 核对完整路径集合、类型（catalog）及无重复，runtime grammar 核对单模型集合；资产审计总数对照独立清单，保留零缺文件、零缺列、零天气资产错误的全部断言。具体场景矩阵仍保留明确文件集合与数量门禁；`test_config_inventory` 默认 integration，覆盖新文件、非法版本不漏扫及类型歧义拒绝。
- CLI 测试同时验证 `backtest_only=False` 默认值与显式开关；MSTL 单周期仍必须被拒绝，断言同步现行错误文本。配置入口测试核对当前版本数据文件及真实存在性。活动天气矩阵按六项实测/预报映射、history-only 文件及 forecast_origin 可得性假设校验；逐份配置使用临时数据验证训练读实测、历史预测读预报及 history lineage，不把该假设当真实发布时间证据。未来文件隔离/缺预报拒绝继续由 `test_inference_columns`、`test_weather_phase_runtime` 覆盖。

| 原重复/历史检查 | 当前保留位置与变化 |
|---|---|
| `test_ensemble_oof.CalendarMonthOOFBoundaryTest` 两个 OOFSpec 构造测试 | 同正文断言保留在 `test_ensemble_specs.OOFSpecTest`；OOF 的切分和缓存测试不删 |
| `test_ensemble_runtime.test_learned_methods_end_to_end` 的三次独立运行 | 有限值与 method 断言合入 `test_oof_cache_reused_across_fusion_methods`；四方法仍真实执行，减少三次重复生命周期 |
| TASK27 Local K2 的七策略 × independent/native 完整 runtime | Direct 保留两种 adapter 的完整接线；其余十二组合改为真实 Ridge/RF 的 trainer → forecaster 测试；regressor-chain 原有七组合保留。矩阵仍覆盖原组合，并非跳过策略 |
| TASK27 的其他轴 | Local K1、Local K2 quantile、Global K2 仍逐策略完整 runtime；两种融合及非法配置/产物检查保留。策略依赖与 time-major 精确值另由 `test_standard_strategy_executors`、`test_multi_target_adapters` 保护 |
| 历史 migration manifest 与当前 fingerprint/路径全集的永久相等锁 | `test_validation_geometry_manifest` 不读取历史 manifest；继续扫描当前活动配置，拒绝旧字段，并基于真实时间轴验证实际训练 origin 数等于当前 `train_window_steps`。历史 manifest 已于 2026-09-06 清除（连同 `_regen_validation_geometry_manifest.py`），内容从 Git 溯源 |

上述改变只影响测试执行与历史快照验收，不修改预测、评估、模型文件或配置解析。下沉的十二组合不再各自单独验证落盘：这是有意收窄重复的接线覆盖，保留 Direct adapter 接线及七策略其他 runtime 轴作为补偿，不宣称逐组合端到端覆盖完全不变。

暂保留冻结分解 fixture（仍是新旧算法独立数值参照）、时间边界/OOF gap、缓存损坏和部署恢复的全部不同层次断言。暂不搬目录或批量抽象共享 fixture：现有跨测试导入需要逐个解耦，不能仅为减少文件数扩大本次改动。

旧 DataFrame 概率后处理链已退出：`probabilistic/pipeline.py`、`probabilistic/postprocessing.py` 及对应的 `test_probabilistic_pipeline.py`、`test_probabilistic_postprocessing.py` 一并删除，不留无测试的休眠实现。现役张量 crossing 由 `test_crossing_method_config.py`、`test_multitarget_probabilistic.py` 验证，CQR 保留 `test_conformal.py`、`test_conformal_tracker.py` 与 runtime 集成测试。仓库外直接导入旧 DataFrame API 的代码须迁移；不提供兼容 shim。

测试专用生产代码退出后的覆盖映射：

- 资源兼容探针移出 runtime/fit service；`fixtures/runtime_planning.py` 只构造最小 workload 并调用正式 planner，不复制决策规则，也不作为性能证据。原线程、策略几何与窗口并行断言保留。
- 旧 NNLS 冻结在 `fixtures/legacy_nnls.py`，parity/runtime 测试仍对照独立实现；不能改成委托新算法的“黄金参照”。
- `test_probabilistic_types` 改测 canonical 张量分布的形状、时间、point 绑定、轴一致性；区间边界断言保留。旧分布专属的 space/stage 枚举随旧类型退出，不伪造 canonical 对应字段。
- `test_probabilistic_objectives` 直接测现役 catalog 参数注入，支持性检查走训练层能力注册表；旧额外 metric/eval_metric 注入不迁入生产。
- `test_ensemble_methods.PredictorDelegationTest` 验证三类 combine 的生产委托、多目标 point/quantile 精确值和成员顺序；现有四方法运行及 bundle 重载测试保留。
- `test_conformal_tracker` 增加倒置区间整批拒绝且不污染历史池的负向断言；CQR 合法输入、as-of 和恢复链继续验证。
- 资产缺列测试改走实际 `audit_runtime_assets` 的 CSV 审计与报告，选日尾部误差测试改走 `run_selection`，不再只测休眠包装。

## 训练设计执行重构（2026-10-05，自 stable 移植）

磁盘编译特征缓存退役（`test_compiled_feature_cache` 一并退役），新增测试默认 integration：`test_indexed_design`/`test_indexed_training`（IndexedDesign 紧凑视图、估计器边界展开、规划期预算前置、native 免编译、与独立编译器数值对照）、`test_numeric_snapshot`（数据源只读数值快照与运行中修订拒绝）、`test_compiler_array_kernels`（全统计量 single/batch/indexed 与独立窗口参照，批路径禁止逐原点标量重算）、`test_design_identity`（身份隔离与退役参数 RAISE）、`test_in_memory_design_execution`（规划不物化全量设计、并发准备仅一次、批量同组编译一次、组间释放）、`test_bounded_execution`/`test_calendar_bounded_execution`（有界在途与按序消费）、`test_forecast_schedule_windows`（next_day 多天/延迟目标、rolling/expanding training_window、重叠窗口、final 一致性）、`test_training_workload`（origin_sampling 契约与训练证据计数/计时）。stable 侧的 refit 复用用例未移植：dev 的 refit_every 语义（非负整数、rolling 几何、禁 train_history_steps、兼容 target transform）与 stable 不同，以 `test_refit_schedule` 为 dev 侧覆盖。

## 日历生成器回归

`test_holiday_generator` 使用包级公开入口，独立钉住 2024 年年初四节气日期并与依赖库公开 API 对照；验证日内最后支持日不多读下一年、每帧仅构建一次节气表，以及 registry → compiler → VisibilityProof 接线和真实 CSV 导出。旧 `data_loading/holiday_generator.py` 不再是兼容合同；前轮信息集冻结 fixture 保留，不按修复后算法重生成。

## 行为不变重构覆盖映射

- Ensemble 子包化：`test_ensemble_package_structure` 默认 integration，固定根入口/共享类型与五个子包的职责清单，检查算法/预测不得反向依赖训练/IO，注入违规导入验证门禁确实拒绝；持久化类型仍定义于 `model_ensemble.artifacts`。既有测试的导入、mock目标和AST文件路径同步新位置，不减少原行为断言。

- 数据层实现归入 `sources/processing/information`；根信息集/provider 兼容文件及 registry 旧 provider 别名已删除。冻结 provider fixture 不重生成，改为验证旧路径拒绝加载；新路径类往返、实际 provider 值及可得性往返仍有回归覆盖。

- `test_data_loading_boundaries` 对照迁移前独立冻结的 `fixtures/data_loading_materialization.json`，覆盖 local/global、vintage、标签访问、provider、日历及既有异常；不得用改后实现重生成参照。另验证读取缓存/拷贝、覆盖发现、角色限定索引、旧 provider pickle，以及包含赋值别名和属性链的生产 registry 私有访问 AST 门禁。信息集原本不能 pickle 往返的失败行为保留，不误记为支持。
- `test_data_loading_assets` 验证真实单模型/融合配置分派、缺资产/缺列/空文件、typed 表头预检查及来源哈希；`test_runtime_asset_audit` 仍在 audit 集合。生成器计算文件的实现指纹保护和源码变化失效由资产边界测试及 `test_compiled_feature_cache` 共同覆盖。

- 编译器保留 single/batch 双执行路径，共用规则解析。`test_compiler_batch_equivalence` 继续覆盖受支持路径等价；`test_compiler_shared_rules` 对照迁移前冻结的 `fixtures/compiler_shared_rules.json`，独立钉住 frame、schema、lineage、visibility 和错误行为。黄金值不得改为委托现实现生成。
- 目标/特征变换测试直接导入 `feature_engineering.transforms`；quantile 训练测试直接导入 `model_training.quantile`。回测评分、pipeline runner 与预测器的静态点名检查跟随新职责位置。
- 模型测试使用 `models.factory`、`models.wrappers.<family>` 与 `models.pickle_io`。`test_model_wrapper_persistence` 验证各 family 的新路径 pickle 往返预测、旧路径无 shim/拒绝加载，以及 pickle IO 导入不修改 `sys.path`；参数校验和多目标 adapter 的原有断言保留。
- layering 的 `PROJECT_PACKAGES` 包含 `pipeline` 与 `model_performance`；新包必须注册，不能靠漏扫换取门禁通过。
- 依赖白名单门禁遍历全部 `PROJECT_PACKAGES` 并核对注册集合；新增包自动进入检查。两个新包的非法依赖探针验证门禁确实报错，而不仅验证当前仓库为绿。
- `test_backtest_lifecycle_split` 覆盖 runner 的显式协议能力、串行/并行窗口评分顺序与历史传参、running/completed/failed 状态以及 BaseException 原样传播；生命周期静态检查位于 `pipeline/lifecycle.py`，逐折证据接线检查指向 `model_testing/loops/fixed_step.py` 与 calendar-month。这些接缝测试不替代真实配置产物 diff。

```bash
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_compiler_shared_rules
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_model_wrapper_persistence
```

真实配置的数值 diff、wall 与 bundle 部署往返证据记录在实施计划，不以单元测试或产物交集相等替代完整结果验收。
