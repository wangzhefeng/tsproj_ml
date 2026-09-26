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
- **当前事实**：2026-09-26 已完成适配层部分下沉（方案 A）：`ModelFactoryEstimator` / `make_model_factory` / `probe_native_multioutput` / `supports_native_multi_quantile` 自 `model_training/estimators/capabilities.py` 下沉至 `models/adapters/canonical.py`（零 forecasting_core 依赖）；`NATIVE_HISTORY_MODELS` 注册表自 `model_pipeline/runner.py` 下沉至 `models/adapters/native_registry.py`。未下沉件：`SharedMultiQuantilePool`（依赖 `forecasting_core.checkpoints.FitCheckpoint`）、三个 MultiTargetAdapter（依赖 checkpoint / tensor 几何 / `TargetCoordinate` 策略坐标），仍留 `model_training/estimators/`。消费结构核实（2026-09-26 grep）：适配器调用方仅 `model_pipeline/{runner,fold_fit}.py`；`model_ensemble/` 对 models / model_training 零 import，经 `BaseModelRunner` Protocol（`model_ensemble/contracts.py:19`）间接消费成员模型，不感知适配层。
- **影响**：当前单一消费方（model_pipeline）下无实际影响；若未来 ensemble 需绕过 runner 直接控制成员模型构造粒度（按成员重训某层、拆特征子集等），需要直接依赖适配器，届时「能力注册表 + Pool + MultiTargetAdapter」分散在训练层会造成跨层调用或接口复制。
- **建议方向**：不立项预建。触发条件 = 出现第二个真实消费方（首个候选：model_ensemble 直接调用 adapter）。触发后再议：将 `SharedMultiQuantilePool` 与 MultiTargetAdapter 一并下沉 `models/adapters/`，前提是先把 `FitCheckpoint` 依赖抽象为注入接口、`TargetCoordinate`/tensor 几何与 strategies 解耦。
- **风险**：提前下沉需为单一消费方造中间接口（checkpoint 抽象层），纯增维护负担且无真实场景约束接口设计；滞后处理的风险是触发时迁移面变大，但迁移路径已被方案 A 验证（分层门禁 + 兼容 re-export 模式可复用）。
- **验收标准**：触发时按方案 B 执行——全量下沉后 `models/adapters/` 无 forecasting_core / model_training 依赖（`tests.test_package_layering` 通过）；`model_training/estimators/` 保留兼容 re-export；fast 套件与定向测试（adapters / quantile / multitarget / ensemble）全过；`docs/packages/models.md` 与 `model_training.md` 同步包边界描述。

### OPT-025 strategy executor 与 artifact 合同的包归属（预测侧消费训练包）

- **状态**：待处理
- **登记日期**：2026-09-26
- **当前事实**：`model_forecasting` 六处 import `model_training`（2026-09-26 grep 核实）：`predictor.py:11,21`（strategies executor + `CanonicalMarginalQuantileArtifact`）、`deployment.py:22,27`（同上）、`persistence.py:15-16`（`CanonicalStrategyArtifact` + `CanonicalTrainer`，后者仅类型标注）。另有 `model_performance/resource_planner.py:16-17` 消费 `supports_native_multi_quantile` 与 `target_plan_for_config`。即「加载 bundle 做预测」的部署路径必须 import 训练包；策略 executor 同时是训练产物与推理合同，知识住在 `model_training/strategies/base.py`（TargetCoordinate + 共享预测循环）。分层 DAG 合法（`model_forecasting → model_training` 边在 ALLOWED_PACKAGES 中），非违规，是语义纯度问题。
- **影响**：当前零行为影响。语义上「训练」成为「预测」的前置；若未来出现纯推理部署形态（不装训练环境）或 ensemble 直接消费成员模型（OPT-023 触发条件），该边界会在两条线上重复出现。
- **建议方向**：不立项预建。与 OPT-023 绑定同一触发条件（第二个真实消费方出现）：触发时先做 TargetCoordinate / tensor 几何与 strategies 解耦（即 OPT-023 Step 2 前置），随后将 executor 与 artifact 合同上收（候选位置：`forecasting_core/specs/strategy.py` 已有策略 spec，或独立 strategies 包），`model_forecasting` 只认合同层类型。解耦成本与 OPT-023 共摊一次，不重复付费。
- **风险**：提前上收需为单消费方预造合同层接口；滞后处理的风险是触发时迁移面变大，但迁移路径已被 OPT-023 方案 A 验证（分层门禁 + 兼容 re-export 模式可复用）。
- **验收标准**：触发时 `model_forecasting` 对 `model_training` 的 import 归零（或仅剩合同层类型再导出）；`tests.test_package_layering` 通过；fast / integration 全过；`docs/packages/model_training.md` 与 `model_forecasting.md` 同步边界描述。

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
