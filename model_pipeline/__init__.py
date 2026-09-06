"""model_pipeline

`model_pipeline/` 是全流程编排包（方案 v3，2026-09-06 R6 自 model_forecasting 迁入，
`.hermes/plans/2026-09-06_162605`）：把「数据管道 → 特征工程 → 模型训练 → 模型测试 →
模型预测」串成完整生命周期，`run.py` / `batch_run.py` 只调用本包。

- `runner.py`：`CanonicalBaseModelRunner` 执行能力与 `run_canonical_config()`。
- `lifecycle.py`：阶段编排、完成态调用与产物元数据；回测循环位于 model_testing。
- `supervised_design.py`：information set 消费、监督样本构造与 fixed-step 回测几何。
- `fold_fit.py`：折/final 拟合与预测服务（fit_service）。
- `run_state.py`：running/completed/failed 完成态证据。
- `batch_runtime.py` / `batch_artifacts.py`：可恢复批调度与产物验收（`batch_run.py` 调用）。

依赖方向：本包向下调用全部阶段模块（data_loading/feature_engineering/model_training/
model_testing/model_forecasting/model_ensemble 服务注入）；阶段模块不反向 import 本包。
"""
