"""feature_engineering.transforms

特征与目标变换三件套（2026-09-06 R3 自 model_forecasting + feature_engineering/scaling
归位为子包，方案 v3，.hermes/plans/2026-09-06_162605）。

- `pipeline.py`：目标变换栈（calendar normalization → decomposition → scaling，
  point/quantile 严格逆序恢复，状态按 (series_id,target) 隔离）。
- `scaling.py`：canonical 特征缩放器。
- `windows.py`：按唯一监督标签时间选取 scaler 窗口；分解默认同窗，允许显式较长上下文。
"""
from feature_engineering.transforms.pipeline import (  # noqa: F401
    CalendarDayTargetNormalizer,
    CanonicalTargetTransform,
    PerSeriesTargetTransformPipeline,
    TargetTransformPipeline,
)
from feature_engineering.transforms.scaling import CanonicalFeatureScaler  # noqa: F401
from feature_engineering.transforms.windows import select_transform_history  # noqa: F401
