# tests

`tests/` 是版本控制内的 unittest 套件，覆盖 core contracts、信息集、特征、七策略、Local/Global、point/quantile、fixed/calendar/monthly runtime、Ensemble、结果 schema、场景数据链和包间结构。

## 执行集合

保留平铺目录及原生 unittest discovery，不移动现有测试或修改 CI。分组在 `run_suite.py` 单点维护，三个集合互斥，其并集等于原生发现全集：

| 集合 | 用途 | 何时执行 |
|---|---|---|
| `fast` | 显式选入的轻量合同、指标、七策略轨迹、结构门禁 | 日常修改；不能替代相关集成验证 |
| `integration` | 真实后端、信息集/编译器、小样本运行链、缓存与部署恢复 | 按改动范围运行；未分类新测试默认进入这里 |
| `audit` | 全仓 YAML、真实资产、场景特征生产链 | 配置/数据/场景模块修改及全量收口 |
| `all` | 三组全集 | 全量收口，串行运行，避免多套测试争抢 CPU |

```bash
# 以下命令均从项目根运行；run_suite 只接受项目 .venv 解释器，其他环境直接拒绝
env -u PYTHONPATH .venv/bin/python tests/run_suite.py fast
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_ensemble_runtime
env -u PYTHONPATH .venv/bin/python tests/run_suite.py audit --match test_validation_geometry_manifest
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --list
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --report .hermes/plans/test-suite-report.json

# 原有入口仍发现并执行所有测试，不是快测入口
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

测试分组入口只接受根目录 `.venv/`（按 `sys.prefix` 校验，错误环境返回 2）；该门禁不拦截绕过入口的原生 unittest 或库调用，直接运行它们时也必须显式使用 `.venv/bin/python`。测试不经过 uv 或 `.uv_cache/`；依赖仍由 `pyproject.toml` + `uv.lock` 管理，必要时用 `uv sync --locked --no-cache` 同步已有 `.venv`，不新建其他环境。

`--match` 是完整测试 ID 的子串过滤；无匹配、重复 ID 或任一模块导入失败返回非零，不允许因为选择 fast 而隐藏其他模块的发现错误。`--list` 仅导入和发现，不执行测试正文；仍会发生项目已有的 import 初始化。`--report` 记录本次实际 testsRun、失败/错误/跳过 ID 和逐测试耗时，不作未经测量的速度承诺。场景模块需要独立选择，不代表其测试可以删除。

## 精简与覆盖映射

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

## 行为不变重构覆盖映射

- 编译器保留 single/batch 双执行路径，共用规则解析。`test_compiler_batch_equivalence` 继续覆盖受支持路径等价；`test_compiler_shared_rules` 对照迁移前冻结的 `fixtures/compiler_shared_rules.json`，独立钉住 frame、schema、lineage、visibility 和错误行为。黄金值不得改为委托现实现生成。
- 目标/特征变换测试直接导入 `feature_engineering.transforms`；quantile 训练测试直接导入 `model_training.quantile`。回测评分、pipeline runner 与预测器的静态点名检查跟随新职责位置。
- 模型测试使用 `models.factory`、`models.wrappers.<family>` 与 `models.pickle_io`。`test_model_wrapper_persistence` 验证各 family 的新路径 pickle 往返预测、旧路径无 shim/拒绝加载，以及 pickle IO 导入不修改 `sys.path`；参数校验和多目标 adapter 的原有断言保留。
- layering 的 `PROJECT_PACKAGES` 包含 `model_pipeline` 与 `model_performance`；新包必须注册，不能靠漏扫换取门禁通过。
- 依赖白名单门禁遍历全部 `PROJECT_PACKAGES` 并核对注册集合；新增包自动进入检查。两个新包的非法依赖探针验证门禁确实报错，而不仅验证当前仓库为绿。
- `test_backtest_lifecycle_split` 覆盖 runner 的显式协议能力、串行/并行窗口评分顺序与历史传参、running/completed/failed 状态以及 BaseException 原样传播；生命周期静态检查位于 `model_pipeline/lifecycle.py`，逐折证据接线检查指向 `model_testing/fixed_step.py` 与 calendar-month。这些接缝测试不替代真实配置产物 diff。

```bash
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_compiler_shared_rules
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_model_wrapper_persistence
```

真实配置的数值 diff、wall 与 bundle 部署往返证据记录在实施计划，不以单元测试或产物交集相等替代完整结果验收。

执行纪律：

- 新行为先 RED 再实现；
- AIDC 专项脚本目录路径含日期段、不是合法 Python 包，相关测试用 `sys.path.insert` 引导后按模块名导入；
- runtime 产物使用临时目录；
- `tests/test_package_layering.py` AST 扫描所有 import，包括函数内 import；
- `tests/test_active_config_runtime_contract.py` 保证活动 YAML 与生产 grammar/时间几何一致；
- `tests/test_runtime_asset_audit.py` 要求活动 source 零缺失；
- 删除测试前必须证明生产链已删除，或将具体断言映射到保留测试；高层冒烟不能替代低层负向合同。

最新精确测试数以本次全量命令输出为准，不在多份文档重复维护。

## 禁止模型执行时的定向检查

不要把 `fast` 当作“绝无模型调用”的保证。可靠性收口仅选择以下已审阅的测试：

- `test_intraday_schedule_contract.py`、`test_oof_gap_contract.py`：纯调度与标签边界。
- `test_transform_window_geometry.py`、`test_batch_time_grid.py`：纯标签窗口与 CSV 时间网格。
- `test_model_catalog_contract.py`、`test_package_layering.py`、`test_lifecycle_static_contract.py`：描述表、依赖门禁及生命周期静态接线/状态合同。
- `test_execution_evidence_contract.py`：合成元数据的缓存读写、旧缓存缺证据、公开能力错误传播、JSON 参数快照和静态调用接线；临时测试数组不是模型结果或运行验收证据。

本地复验入口为 `.hermes/plans/verify-architecture-no-models.py`，安装调用拦截器后执行上述白名单，遇到 fit/predict 类调用立即中止。各文件仍纳入原生 discovery，不从完整测试集合排除。
