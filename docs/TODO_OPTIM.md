# 项目待优化问题

本文档记录尚未解决的工程问题、技术债和配置体系优化事项。业务实验与场景研究事项记录在 [`docs/scenarios/aidc_load_15min_short/TODO_AIDC.md`](scenarios/aidc_load_15min_short/TODO_AIDC.md)。

## 维护约定

- 状态使用：`待处理`、`进行中`、`阻塞`、`已完成`。
- 每个问题必须记录当前事实、影响、建议方向、风险和验收标准。
- `已完成` 必须补充实际执行的验证命令及结果，不能只记录实现计划。
- 涉及预测语义、数据口径或产物兼容性的修改，实施前需要单独确认。
- 问题开始实施后，如方案发生偏差，在原条目中追加实施记录，不回改已确认的历史结论。
- 条目编号自 OPT-023 起继续分配（OPT-001~022 历史条目见下方历史记录）。

## 历史记录

- 2026-09-26：OPT-001~022 全部条目（含「进行中」的 OPT-021 月频正式结果重建、「待处理」的 OPT-022 天气插补严格 as-of 资格核实）随文档系统重构清空退役，完整内容从 Git 溯源：`git show 6d2729f0:docs/TODO_OPTIM.md`。OPT-021/022 如仍需推进，按维护约定重新登记条目，不整段回搬。

## 待处理

### OPT-023 estimator 适配层完整下沉 models/ 的第二消费方触发条件

- **状态**：待处理
- **登记日期**：2026-09-26
- **当前事实**：2026-09-26 已完成适配层部分下沉（方案 A）：`ModelFactoryEstimator` / `make_model_factory` / `probe_native_multioutput` / `supports_native_multi_quantile` 自 `model_training/estimators/capabilities.py` 下沉至 `model_building/adapters/canonical.py`（零 forecasting_core 依赖）；`NATIVE_HISTORY_MODELS` 注册表自 `model_pipeline/runner.py` 下沉至 `model_building/adapters/native_registry.py`。未下沉件：`SharedMultiQuantilePool`（依赖 `forecasting_core.checkpoints.FitCheckpoint`）、三个 MultiTargetAdapter（依赖 checkpoint / tensor 几何 / `TargetCoordinate` 策略坐标），仍留 `model_training/estimators/`。消费结构核实（2026-09-26 grep）：适配器调用方仅 `model_pipeline/{runner,fold_fit}.py`；`model_ensemble/` 对 models / model_training 零 import，经 `BaseModelRunner` Protocol（`model_ensemble/contracts.py:19`）间接消费成员模型，不感知适配层。
- **影响**：当前单一消费方（model_pipeline）下无实际影响；若未来 ensemble 需绕过 runner 直接控制成员模型构造粒度（按成员重训某层、拆特征子集等），需要直接依赖适配器，届时「能力注册表 + Pool + MultiTargetAdapter」分散在训练层会造成跨层调用或接口复制。
- **建议方向**：不立项预建。触发条件 = 出现第二个真实消费方（首个候选：model_ensemble 直接调用 adapter）。触发后再议：将 `SharedMultiQuantilePool` 与 MultiTargetAdapter 一并下沉 `model_building/adapters/`，前提是先把 `FitCheckpoint` 依赖抽象为注入接口、`TargetCoordinate`/tensor 几何与 strategies 解耦。
- **风险**：提前下沉需为单一消费方造中间接口（checkpoint 抽象层），纯增维护负担且无真实场景约束接口设计；滞后处理的风险是触发时迁移面变大，但迁移路径已被方案 A 验证（分层门禁 + 兼容 re-export 模式可复用）。
- **验收标准**：触发时按方案 B 执行——全量下沉后 `model_building/adapters/` 无 forecasting_core / model_training 依赖（`tests.test_package_layering` 通过）；`model_training/estimators/` 保留兼容 re-export；fast 套件与定向测试（adapters / quantile / multitarget / ensemble）全过；`docs/packages/model_building.md` 与 `model_training.md` 同步包边界描述。

### OPT-025 strategy executor 与 artifact 合同的包归属（预测侧消费训练包）

- **状态**：待处理
- **登记日期**：2026-09-26
- **当前事实**：`model_predicting` 六处 import `model_training`（2026-09-26 grep 核实）：`predictor.py:11,21`（strategies executor + `CanonicalMarginalQuantileArtifact`）、`deployment.py:22,27`（同上）、`persistence.py:15-16`（`CanonicalStrategyArtifact` + `CanonicalTrainer`，后者仅类型标注）。另有 `model_performance/resource_planner.py:16-17` 消费 `supports_native_multi_quantile` 与 `target_plan_for_config`。即「加载 bundle 做预测」的部署路径必须 import 训练包；策略 executor 同时是训练产物与推理合同，知识住在 `model_training/strategies/base.py`（TargetCoordinate + 共享预测循环）。分层 DAG 合法（`model_predicting → model_training` 边在 ALLOWED_PACKAGES 中），非违规，是语义纯度问题。
- **影响**：当前零行为影响。语义上「训练」成为「预测」的前置；若未来出现纯推理部署形态（不装训练环境）或 ensemble 直接消费成员模型（OPT-023 触发条件），该边界会在两条线上重复出现。
- **建议方向**：不立项预建。与 OPT-023 绑定同一触发条件（第二个真实消费方出现）：触发时先做 TargetCoordinate / tensor 几何与 strategies 解耦（即 OPT-023 Step 2 前置），随后将 executor 与 artifact 合同上收（候选位置：`forecasting_core/specs/strategy.py` 已有策略 spec，或独立 strategies 包），`model_predicting` 只认合同层类型。解耦成本与 OPT-023 共摊一次，不重复付费。
- **风险**：提前上收需为单消费方预造合同层接口；滞后处理的风险是触发时迁移面变大，但迁移路径已被 OPT-023 方案 A 验证（分层门禁 + 兼容 re-export 模式可复用）。
- **验收标准**：触发时 `model_predicting` 对 `model_training` 的 import 归零（或仅剩合同层类型再导出）；`tests.test_package_layering` 通过；fast / integration 全过；`docs/packages/model_training.md` 与 `model_predicting.md` 同步边界描述。

### OPT-028 refit_every 语义分歧裁决与「有界历史 + 间隔重训」组合缺口

- **状态**：待处理
- **登记日期**：2026-10-05
- **当前事实**：stable 训练设计重构（2026-10-05 移植入 dev，快照 `.hermes/plans/migration_stable_wip_20261004/`）携带的 refit_every 语义与 dev 演进版分歧：stable 要求 refit_every ≥ 1、限 fixed_steps、允许搭配 train_history_steps、禁止搭配 target transform；dev 允许 refit_every=0（仅首折拟合）、要求 rolling 系几何、禁止搭配 train_history_steps、兼容 target transform（scaler 随 artifact 冻结复用，`tests/test_refit_schedule.py` 覆盖）。移植裁决（已执行）：保留 dev 语义，不引入 stable 的「refit_every > 1 需无 target transform」守卫；WIP 两个 refit 用例按 dev 语义改写/移除（`test_training_workload.py` 中 `test_refit_reuses_artifact_but_updates_prediction_context` 删除、`test_refit_contract_and_unsupported_modes` 改写）。同时新 temporal 合同（`forecasting_core/specs/temporal.py`）要求显式 training_window/forecast_window 必须 refit_every=1。
- **影响**：dev 当前不存在「有界历史（train_history_steps 或 training_window）+ refit_every ≠ 1」的合法配置组合——stable 侧该组合的「复用 artifact 但逐折更新只读信息集/历史下界」能力未带入 dev。回测引擎已为此预留结构：`run_rolling_backtest` 的 strict 分支在 refit_every ≠ 1 时逐折 `for_backtest_window` 更新信息集（stable 语义已并入），仅缺 spec 层放行与合同测试。
- **建议方向**：不立项预建。触发条件 = 出现「长历史 bounded 回测 + 控制重训频率」的真实需求（如 training_window 长窗逐日发报的批量回测成本压力）。触发时：放开 `validation.py` 的 refit_every/train_history_steps 互斥与 temporal.py 的 refit_every=1 约束，恢复 stable 版用例语义（快照中 `new_files/tests/test_training_workload.py` 的 refit 复用测试可作参照），并裁定与「scaler 冻结复用」合同的交互。
- **风险**：两分支语义已显式分化，未来再从 stable 移植回测相关改动时需逐条对照 refit 规则，不能默认 stable 行为；若长期无人触发，stable 侧的 bounded+refit 证据字段（did_refit/model_fit_origin）与 dev 字段名（refitted/fit_origin）的差异固化，跨分支结果对比需人工映射。
- **验收标准**：触发实施时——spec 层放行组合的解析与校验测试通过；恢复 bounded+refit 的折级信息集更新测试（参照快照）；`tests/run_suite.py fast` 与 integration 全过；`docs/packages/model_testing.md` 与本文档同步。

### OPT-029 model_evaluation 走读遗留：池化一致性、scope 集对齐与掩码全排除护栏

- **状态**：进行中（①② 已完成，③ 待处理）
- **登记日期**：2026-10-05
- **实施记录（2026-10-05，用户点名 §三 6/7/8 后执行）**：① 已完成——`marginal.py` aggregate_horizon 池化段改为复用 target 行 `_emit` 返回的 valid（同一掩码只算一遍），旧/新程序化对照（git HEAD 版 vs 现版，含/不含掩码各 150 行）逐值一致；② 已完成——`point_intervals.py` 补齐 aggregate/aggregate_horizon 池化行（proper-score 口径与 marginal 对齐），旧 scope 行逐值一致（含/不含掩码各 40 行），新增 20 行/场景；同批完成指标补充（原 §三.6）：point 增 SMAPE/MASE/RMSSE（MASE/RMSSE 以 `FoldScoringRunner.target_history` 的 in-sample 季节差分为缩放，lag 经 `primitives.resolve_seasonal_naive_lag` 与 naive 基线同口径，未提供时 NaN 不伪造；aggregate_horizon 行 MASE/RMSSE 恒 NaN），marginal 增 CRPS 分位梯形积分近似行（单 level 记 NaN）。验证：定向 71 tests OK；fast 300 passed（298+2）；layering 23 OK；`git diff --check` 干净；marginal/point_intervals 旧/新对照脚本逐值一致（见上）。③ 待处理（语义敏感，实施前单独确认）。
- **当前事实**：2026-10-05 model_evaluation 模块走读（§一/§二 已同批处理：过期 docstring 引用修正、`point_intervals.py` 补 docstring/`__all__`/门面导出、`normalized_width` 死输出删除、`crossing.report_raw` 断链激活至回测逐窗 execution_evidence、`eval_mask.mode` 解析期白名单前置）。遗留项③：掩码全排除时 `excluded_ratio=1.0` 不 RAISE，评分全 NaN 正常落盘——「掩码配置错误」与「数据真异常」产物不可区分。
- **影响**：③ 掩码误配的失败信号被推迟到人工读结果阶段。
- **建议方向**：③ 在评分接缝处加护栏：`excluded_ratio == 1.0` 时 RAISE 或至少在评分帧 attrs / 回测 evidence 中记录告警字段。语义敏感，实施前单独确认。
- **风险**：③ 护栏 RAISE 会改变「掩码全排除」场景的失败面（从静默 NaN 变显式失败），存量研究配置若有依赖该行为的会被打断。
- **验收标准**：①② 已达成（验证命令与结果见实施记录）；③ 触发实施时——fast 套件与定向测试全过，差异逐行解释，`docs/packages/model_evaluation.md` 同步。

## 已完成

### OPT-024 sample_weight 功能激活、子包收口与 stats 白名单规范化

- **状态**：已完成
- **登记日期**：2026-09-26
- **当前事实**：原 `model_training/sample_weight.py`（27 行指数衰减函数）自 2026-09-25 红太阳脚本退役后无生产消费者；配置合同（`forecasting_core/specs/validation.py:171` `validation.training.sample_weight`）与训练管道两端（trainer/adapter/wrapper `sample_weight=`）早已就绪，但 `fold_fit` 接线缺失——schema 接受该字段不代表权重真实到达估计器。
- **实施内容**：
  1. 子包收口：`sample_weight.py` → `model_training/weights/`（`temporal.py` 算法本体 + `resolve.py` 配置接线与能力前置校验），旧文件删除（内部文件、零引用）。
  2. 功能完善：`anchor`（`cutoff` 默认 / `latest_origin`）与 `normalization`（`mean` 默认 / `sum` / `none`）两个语义维度显式化。
  3. 接线：`fold_fit.py::_fit_point/_fit_quantile` 增加 `sample_weight` 参数透传 `trainer.train`；`runner.fit()` 按训练 origins + 标签末端 cutoff 计算权重；`fit_final()` 防护——声明加权但跳过 `final_bundle_inputs` 时 RAISE；`resolve.py` 对 catalog `sample_weight=False` 的模型（seasonal_template）前置 RAISE。
  4. 关联规范化（EWM 特征检查中发现）：`feature_engineering/compiler.py` 的 rolling/expanding/ewm `stats` 此前无解析期白名单，拼错统计名要到编译中段才 RAISE；新增 `ROLLING_STATS`（10 项全集）/`EWM_STATS`（mean/std）frozenset 与 `_validated_stats`，5 处调用点替换，报错列明非法项与合法集。
- **验证命令及结果**：
  - 新增 `tests/test_training_sample_weight.py`（12 项：数值正确性/防御合同/配置接线/能力 RAISE）与 `tests/test_sample_weight_wiring.py`（4 项端到端：加权前后预测差异严格为正证明权重到达估计器、未声明时行为不变、fit_final 防护、final fit 加权完整跑通）——全过。
  - `tests/test_feature_visibility_compiler.py` 新增 `test_unknown_stats_rejected_at_compile_time`（rolling/expanding/ewm 三子用例）——全过。
  - `env -u PYTHONPATH .venv/bin/python tests/run_suite.py fast`：290 项 OK；`... integration`：683 项 OK（233.046s）。
  - `env -u PYTHONPATH .venv/bin/python -m compileall -q model_training model_pipeline feature_engineering` OK。
- **风险**：声明 `validation.training.sample_weight` 的配置产生新语义身份（fingerprint 变化、结果目录新建）；存量 171 份活动配置（全 LightGBM，能力兼容）均未声明，不受影响。compiler stats 白名单为行为变更——存量配置若有拼错统计名会在编译期报错（这正是目的）；活动配置实测全过。

### OPT-026 model_pipeline 点/分位拟合分派模板收敛（4 处同构）

- **状态**：待处理
- **登记日期**：2026-09-27
- **当前事实**：`model_pipeline/fold_fit.py::_fit_point`（138-184）与 `_fit_quantile`（187-274）头部同构约 30 行（checkpoint.child → capabilities → trainer 构造 → 调度参数解析）；`runner.py::fit`（540-565）与 `fit_final`（802-826）各自再写一次 mode 二分分派。同一「构造+checkpoint+调度」形状出现 4 处。2026-09-27 model_pipeline 优化会话已做项：B1 preflight 单任务隔离、A 批卫生（batch_size 常量化、死赋值清理、私有别名公开化、回测几何显式分派）、证据四函数迁出、B4 native+quantile 构造期前置、B5 实现指纹进程内缓存 + 包清单补 model_building + layering 门禁补盲；A1 模板合并因改动面大、且与并发会话（model_building 重命名）同场，降级为本条目登记。
- **影响**：新增 probabilistic mode 或调整 checkpoint/调度接线时需同步 4 处；漏改一处只在特定路径生效，属 RAISE 可见工程债而非静默错误。
- **建议方向**：合并为单一分派入口（「构造+checkpoint+组装」统一函数，仅调度入口不同），对齐 model_training 会话已沉淀的「统一 fit 调度」模式。
- **风险**：控制流改写面大（两个文件、4 个调用点），需 fast + 定向（runtime_checkpoints / canonical_transforms / sample_weight_wiring / batch_runtime）四件套验证；行为零变化验证靠既有 parity 测试。
- **验收标准**：4 处分派收敛为 1 个实现；`compileall` + `tests/run_suite.py fast` + 上述定向全过；`git diff --check` 干净；本条目标记已完成并附验证命令。

### OPT-027 lifecycle 证据组装函数迁出与批调度 preflight 隔离（已完成批次记录）

- **状态**：已完成（2026-09-27）
- **登记日期**：2026-09-27
- **当前事实**：model_pipeline 优化批次落地五项：(1) B1——`batch_runtime._preflight_groups` 补传 `checkpoint_root`，构造期 ValueError 被包装为 FitCheckpointError 后按单任务 failed 隔离（旧行为：整批 171 任务标 failed）；(2) A 批——`TRAINING_COMPILE_BATCH_SIZE` 常量化、`_runner=None` 死赋值删除、`read_text` 补 encoding、`actual_tensor` 别名公开化、回测几何按 spec 类型显式分派（backtest 缺失前置 RAISE，替代 fixed_step 返回 None 的探测回退协议）；(3) 证据域聚集——`proof_payload`/`holdout_proof_summary`/`source_lineage_payload`/`compiled_lineage` 四函数自 lifecycle.py 迁入新建 `model_predicting/evidence_assembly.py`（实现逐字保真），lifecycle 543→362 行；(4) B4——runner 构造期对 native 历史模型 + `probabilistic.mode=quantile` 前置 RAISE（原到折拟合期才报）；(5) B5 + 并发遗留修复——`implementation_fingerprint` 进程内缓存，包清单补 `model_building`（models/ 更名后实现文件曾不再进指纹），layering 门禁 PROJECT_PACKAGES/ALLOWED 同步补 model_building 并新增 `test_model_building_stays_infra`。
- **影响**：preflight 错误隔离符合 docstring 声明语义；证据组装与 evidence.py 同域维护；native quantile 错误前移到构造期（含 batch preflight 阶段）；批跑指纹计算从 171 次全仓哈希降为 1 次。实现指纹值因包清单修正而变化（属预期——原值漏掉了 model_building 实现文件，是身份失真修正）。
- **验证命令及结果**：`compileall` model_pipeline/model_predicting/model_performance/tests 改动文件 OK；`git diff --check` 干净；`tests/run_suite.py fast` 291 项 OK（含 3 项新增：preflight ValueError 隔离、native quantile 构造期拒绝、model_building 门禁）；`tests.test_package_layering` 23 项 OK；定向 `test_batch_runtime`(13)/`test_batch_calendar_checkpoint`(1)/`test_runtime_checkpoints`+`test_compiled_feature_cache`(36 总)/`test_backtest_only`+`test_backtest_lifecycle_split`+`test_lifecycle_static_contract`+`test_canonical_transforms`+`test_sample_weight_wiring`(29 总)/`test_runtime_array_fastpaths`(8)/`test_raw_history_window`(9) 全过；`implementation_fingerprint()` 双调用一致性断言通过。
- **风险**：A1 模板合并拆至 OPT-026。追加（2026-09-27 用户点名后执行）：`_output_paths` legacy 三键分支与 spec `OUTPUT_FIELDS` 三键字段已删除（零活动消费，语义变更已授权）；`probe_training_design`/`TrainingDesignProbe` 零调用方已删除；验证 `compileall` + fast + 定向全过。
