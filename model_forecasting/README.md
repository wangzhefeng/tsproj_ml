# model_forecasting

`model_forecasting/` 是单模型生命周期编排与推理层。

- `design.py`：information set、特征设计、监督样本和 fixed-step 回测几何。
- `fit_service.py`：fold/final transform、训练与预测服务。
- `backtest_runtime.py`：calendar-month 回测生命周期。
- `persistence.py`：schema-2 bundle 构造与持久化。
- `runtime.py`：`CanonicalBaseModelRunner` 与顶层单模型生命周期编排。
- `forecaster.py`：point 与 marginal quantile 推理，recursive quantile 使用 median path。
- `transforms.py`：目标/特征变换状态与严格逆变换。
- `transform_windows.py`：按唯一监督标签时间选取 scaler 窗口；分解默认同窗，允许显式较长上下文。
- `evidence.py`：现有模型状态的只读参数快照及 JSON 安全转换，不执行拟合/预测。
- `lifecycle.py`：单模型 running/completed/failed 状态；批产物验收检查完成状态，不是目录事务。
- `results.py`：canonical long result 读写和绘图。

稳定类型全部来自 `forecasting_core/`。本包可以向下编排各流水线阶段，但不得 import `model_ensemble`。

`CanonicalBaseModelRunner.execution_evidence(artifact, target_transform)` 为公开只读能力，供 fixed-step、calendar-month 和通过 Protocol 边界调用的 Ensemble 成员复用。单模型逐折证据位于 holdout metadata，final 证据位于 `runtime.run_evidence` 与 `result_metadata.json`。自然月回测/CQR 收集在 final fit 之前完成，不再于 run 返回后改写 final 产物。

运行证据和验证边界见 [权威设计 §6.3](../docs/multistep_forecasting_redesign.md#63-可靠性收口实现与受限验证)。外部直接加载 pickle 不会自动消费完成状态。
