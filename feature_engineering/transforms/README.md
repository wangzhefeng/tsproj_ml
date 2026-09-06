# feature_engineering/transforms

本子包承载训练窗内拟合、预测期复用的目标与特征变换。编排层负责提供显式窗口，本包不决定回测折或输出目录。

- `pipeline.py`：`CanonicalTargetTransform` 及目标变换实现。顺序固定为 calendar normalization → decomposition → scaling；point/quantile 严格逆序恢复，状态按 `(series_id, target)` 隔离。
- `scaling.py`：`CanonicalFeatureScaler`，只在训练设计上拟合，预测设计使用同一状态。
- `windows.py`：`select_transform_history()`，按唯一监督标签时间选取 scaler 窗口；分解默认使用同窗，可依配置显式扩展上下文。

变换组合规格归一化属于父包 `transform_specs.py`，其中分解别名与参数校验委托 `decomposition/configuration/spec.py` 的唯一入口。训练与预测共用已拟合状态；不得在预测输入上重新拟合或以隐式填补掩盖缺失。分解算法内核属于 `decomposition/`，本子包负责按目标/序列接线与严格恢复。
