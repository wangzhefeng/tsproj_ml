"""预测张量轴匹配与 time-major 数值布局。"""

from forecasting_core.tensors.point import PointForecastTensor
import numpy as np


def require_matching_point_axes(
    actual: PointForecastTensor, prediction: PointForecastTensor
) -> None:
    """逐轴校验两个 point 张量对齐（series/targets/forecast_times）。

    轴对齐是张量合同的一部分，消费方（评估 `model_evaluation/point.py`、结果读写
    `model_predicting/artifacts/results.py`）共用本函数，不各自实现（2026-08-30 评估模块化）。
    """
    if (
        actual.series_ids != prediction.series_ids
        or actual.targets != prediction.targets
        or not actual.forecast_times.equals(prediction.forecast_times)
    ):
        raise ValueError("actual and prediction axes must match")


def flatten_time_major(
    values: np.ndarray,
    *,
    steps: int,
    width: int,
) -> np.ndarray:
    """``(N, steps, width)`` -> ``(N, steps*width)``（time-major 合同唯一实现）。

    列序固定 time-major/target-minor：``(t0,k0), (t0,k1), ..., (t1,k0), ...``，
    与 ``PointForecastTensor.to_time_major_matrix`` 同一合同。训练端把单次模型
    调用负责的 ``(N, steps, K)`` 目标块压平成监督矩阵、执行器把 ``(N, H, K)``
    预测张量压平成 2D 合同，均须走本函数，不得各自 reshape（2026-09-01 收敛）。
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (steps, width):
        raise ValueError(
            f"expected shape (N, {steps}, {width}), got {array.shape}"
        )
    return array.reshape(array.shape[0], steps * width)


def unflatten_time_major(
    matrix: np.ndarray,
    *,
    steps: int,
    width: int,
) -> np.ndarray:
    """``(N, steps*width)`` -> ``(N, steps, width)``（flatten_time_major 逆变换）。"""
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[1] != steps * width:
        raise ValueError(
            f"expected shape (N, {steps * width}), got {array.shape}"
        )
    return array.reshape(array.shape[0], steps, width)
