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
