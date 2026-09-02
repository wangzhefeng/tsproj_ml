# -*- coding: utf-8 -*-
"""Conformalized Quantile Regression 的严格数学内核。"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from forecasting_core.probabilistic_spec import ProbabilisticSpec, validate_cqr_params
from forecasting_core.artifacts import QuantileGrid


@dataclass(frozen=True)
class CalibrationRecord:
    """一条带完整时序可得性信息的 CQR 校准记录。

    ``series_id``/``target`` 为 2026-09-01 新增的可选字段：pooled 分组下
    同一 target_time 可对应多条（series, target）记录，选择时按
    ``(target_time, series_id, target)`` 去重；缺省 None 时退化为旧的
    单序列单目标语义（按 target_time 去重）。
    """

    forecast_origin: pd.Timestamp
    target_time: pd.Timestamp
    label_available_at: pd.Timestamp
    horizon_step: int
    window: int
    interval_name: str
    lower: float
    upper: float
    y_true: float
    score: float
    stage: str
    series_id: Optional[str] = None
    target: Optional[str] = None


@dataclass(frozen=True)
class CalibrationResult:
    """一次 as-of CQR 校准的结果和审计状态。"""

    lower: Optional[np.ndarray]
    upper: Optional[np.ndarray]
    correction: Optional[float]
    selected_records: Tuple[CalibrationRecord, ...]
    selected_windows: int
    selected_scores: int
    status: str
    reason: str


def select_calibration_records(
    records: Sequence[CalibrationRecord],
    forecast_origin: pd.Timestamp,
    n_windows: int,
) -> List[CalibrationRecord]:
    """按 as-of 语义选择最新的完整历史窗口，并按 target time 去重。"""
    window_limit = int(n_windows)
    if window_limit <= 0:
        raise ValueError("n_windows must be > 0")
    current_origin = pd.Timestamp(forecast_origin)
    if bool(pd.isna(current_origin)):
        raise ValueError("forecast_origin must be a finite timestamp")
    grouped = {}
    for record in records:
        key = (int(record.window), pd.Timestamp(record.forecast_origin))
        grouped.setdefault(key, []).append(record)

    eligible = []
    for (_, origin), window_records in grouped.items():
        latest_label = max(pd.Timestamp(record.label_available_at) for record in window_records)
        if bool(pd.isna(origin)) or bool(pd.isna(latest_label)):
            raise ValueError("calibration records require finite timestamps")
        if origin.value < current_origin.value and latest_label.value <= current_origin.value:
            eligible.append((origin, window_records))
    eligible.sort(key=lambda item: item[0], reverse=True)

    selected: List[CalibrationRecord] = []
    seen_targets = set()
    for _, window_records in eligible[:window_limit]:
        for record in sorted(window_records, key=lambda item: pd.Timestamp(item.target_time)):
            # pooled 语义：去重键含 series/target，缺省 None 时退化为 target_time
            dedup_key = (
                pd.Timestamp(record.target_time),
                record.series_id,
                record.target,
            )
            if dedup_key in seen_targets:
                continue
            selected.append(record)
            seen_targets.add(dedup_key)
    return selected


def calibrate_with_records(
    q_low: np.ndarray,
    q_high: np.ndarray,
    records: Sequence[CalibrationRecord],
    forecast_origin: pd.Timestamp,
    n_windows: int,
    min_windows: int,
    min_scores: int,
    alpha: float,
    allow_interval_shrink: bool = False,
) -> CalibrationResult:
    """用满足 as-of 约束的完整历史窗口执行 CQR 双门槛校准。"""
    required_windows = int(min_windows)
    if required_windows <= 0:
        raise ValueError("min_windows must be > 0")
    validate_cqr_params(alpha, min_scores)
    selected = select_calibration_records(records, forecast_origin, n_windows)
    selected_window_count = len(
        {(record.window, pd.Timestamp(record.forecast_origin)) for record in selected}
    )
    finite_scores = np.asarray(
        [record.score for record in selected if np.isfinite(record.score)],
        dtype=float,
    )
    if selected_window_count < required_windows:
        return CalibrationResult(
            lower=None,
            upper=None,
            correction=None,
            selected_records=tuple(selected),
            selected_windows=selected_window_count,
            selected_scores=len(finite_scores),
            status="insufficient_windows",
            reason=f"selected {selected_window_count} windows < min_windows={required_windows}",
        )
    if len(finite_scores) < int(min_scores):
        return CalibrationResult(
            lower=None,
            upper=None,
            correction=None,
            selected_records=tuple(selected),
            selected_windows=selected_window_count,
            selected_scores=len(finite_scores),
            status="insufficient_scores",
            reason=f"selected {len(finite_scores)} scores < min_scores={int(min_scores)}",
        )

    lower, upper, correction = calibrate_quantile_band(
        q_low,
        q_high,
        finite_scores,
        alpha=alpha,
        min_scores=min_scores,
        allow_interval_shrink=allow_interval_shrink,
    )
    return CalibrationResult(
        lower=lower,
        upper=upper,
        correction=correction,
        selected_records=tuple(selected),
        selected_windows=selected_window_count,
        selected_scores=len(finite_scores),
        status="applied",
        reason="",
    )


def _as_1d_finite_array(values, name: str, allow_nonfinite: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not allow_nonfinite and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def compute_nonconformity_scores(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    """逐点计算 CQR score；区间内的 score 可以为负。"""
    y = _as_1d_finite_array(y_true, "y_true")
    lower = _as_1d_finite_array(q_low, "q_low")
    upper = _as_1d_finite_array(q_high, "q_high")
    if not (len(y) == len(lower) == len(upper)):
        raise ValueError(
            "CQR score length mismatch: "
            f"y_true={len(y)}, q_low={len(lower)}, q_high={len(upper)}"
        )
    if np.any(lower > upper):
        raise ValueError("CQR score requires q_low <= q_high at every point")
    return np.maximum(lower - y, y - upper)


def calibrate_quantile_band(
    q_low: np.ndarray,
    q_high: np.ndarray,
    scores: np.ndarray,
    alpha: float,
    min_scores: int = 30,
    allow_interval_shrink: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """用历史 CQR score 对基础区间做对称校准。"""
    alpha_value, min_score_count = validate_cqr_params(alpha, min_scores)
    lower = np.asarray(q_low, dtype=float).reshape(-1)
    upper = np.asarray(q_high, dtype=float).reshape(-1)
    if len(lower) != len(upper):
        raise ValueError(
            f"CQR calibration length mismatch: q_low={len(lower)}, q_high={len(upper)}"
        )
    score_values = np.asarray(scores, dtype=float).reshape(-1)
    score_values = score_values[np.isfinite(score_values)]
    if len(score_values) < min_score_count:
        return None, None, None

    n_scores = len(score_values)
    quantile_level = min(
        1.0,
        np.ceil((1.0 - alpha_value) * (n_scores + 1)) / n_scores,
    )
    correction = float(np.quantile(score_values, quantile_level, method="higher"))
    if not allow_interval_shrink:
        correction = max(0.0, correction)
    return lower - correction, upper + correction, correction


def pi_column_names(target_coverage: float) -> Tuple[str, str]:
    """``predict_pi<coverage>_lower/upper`` 列名的唯一生成入口。"""
    coverage = float(target_coverage)
    validate_cqr_params(alpha=1.0 - coverage, min_scores=1)
    coverage_percent = coverage * 100.0
    if np.isclose(coverage_percent, round(coverage_percent), atol=1e-10):
        token = str(int(round(coverage_percent)))
    else:
        token = f"{coverage_percent:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"predict_pi{token}_lower", f"predict_pi{token}_upper"


def attach_cqr_interval_columns(
    frame: pd.DataFrame,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
) -> pd.DataFrame:
    """追加独立 CQR prediction interval 列，不覆盖模型 quantile。"""
    lower_column, upper_column = pi_column_names(target_coverage)
    lower_values = _as_1d_finite_array(lower, "lower")
    upper_values = _as_1d_finite_array(upper, "upper")
    if not (len(frame) == len(lower_values) == len(upper_values)):
        raise ValueError(
            "CQR interval output length mismatch: "
            f"frame={len(frame)}, lower={len(lower_values)}, upper={len(upper_values)}"
        )
    if np.any(lower_values > upper_values):
        raise ValueError("CQR interval output requires lower <= upper at every point")

    result = frame.copy()
    result[lower_column] = lower_values
    result[upper_column] = upper_values
    return result


class ConformalCalibrationTracker:
    """canonical rolling backtest 的 as-of CQR 校准追踪器（pooled 分组）。

    每折先用 origin 严格更早、且标签已可得的历史折校准当前折区间，再把
    当前折记入校准池（apply-before-collect）；不足 min_windows/min_scores
    时诚实降级：不产出 ``predict_pi*`` 列，audit 记录原因。

    标签可得性约定：``label_available_at = target_time +
    label_availability_delay_steps * freq_offset``（delay=0 时即目标时刻
    本身——回测中前一折的目标期整体早于当前折 origin 时才参与校准）。
    """

    def __init__(self, spec: ProbabilisticSpec, *, freq_offset) -> None:
        if not isinstance(spec, ProbabilisticSpec):
            raise TypeError("spec must be a ProbabilisticSpec")
        calibration = spec.calibration
        interval = spec.calibration_interval
        if calibration is None or interval is None:
            raise ValueError(
                "ConformalCalibrationTracker requires spec.calibration referencing "
                "a configured interval"
            )
        grid = QuantileGrid(spec.quantiles, point_level=spec.point_quantile)
        self._lower_column = grid.column_name(interval.lower_quantile)
        self._upper_column = grid.column_name(interval.upper_quantile)
        self._calibration = calibration
        self._interval = interval
        self._alpha = round(1.0 - calibration.target_coverage, 15)
        self._freq_offset = freq_offset
        self._records: List[CalibrationRecord] = []

    @property
    def interval_name(self) -> str:
        return self._interval.name

    @property
    def target_coverage(self) -> float:
        return self._calibration.target_coverage

    @property
    def pi_columns(self) -> Tuple[str, str]:
        return pi_column_names(self._calibration.target_coverage)

    @property
    def lower_column(self) -> str:
        return self._lower_column

    @property
    def upper_column(self) -> str:
        return self._upper_column

    def _check_frame(self, frame: pd.DataFrame) -> None:
        required = {
            "series_id",
            "time",
            "target",
            "actual_value",
            self._lower_column,
            self._upper_column,
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"CQR calibration frame is missing columns: {sorted(missing)}"
            )

    def _calibrate(self, forecast_origin: pd.Timestamp) -> CalibrationResult:
        return calibrate_with_records(
            # 修正量只由历史 score 决定；这里只需要 correction，不需要逐点边界
            q_low=np.zeros(1),
            q_high=np.zeros(1),
            records=self._records,
            forecast_origin=forecast_origin,
            n_windows=self._calibration.calibration_windows,
            min_windows=self._calibration.min_windows,
            min_scores=self._calibration.min_scores,
            alpha=self._alpha,
            allow_interval_shrink=self._calibration.allow_interval_shrink,
        )

    def _audit(self, result: CalibrationResult) -> dict:
        return {
            "interval": self._interval.name,
            "target_coverage": self._calibration.target_coverage,
            "status": result.status,
            "reason": result.reason,
            "correction": result.correction,
            "selected_windows": result.selected_windows,
            "selected_scores": result.selected_scores,
        }

    def apply_to_frame(
        self,
        frame: pd.DataFrame,
        *,
        forecast_origin: pd.Timestamp,
    ) -> Tuple[pd.DataFrame, dict]:
        """用校准池修正当前折区间；未 applied 时原样返回。"""
        self._check_frame(frame)
        result = self._calibrate(pd.Timestamp(forecast_origin))
        audit = self._audit(result)
        if result.status != "applied":
            return frame, audit
        corrected = attach_cqr_interval_columns(
            frame,
            lower=frame[self._lower_column].to_numpy(dtype=float)
            - float(result.correction),
            upper=frame[self._upper_column].to_numpy(dtype=float)
            + float(result.correction),
            target_coverage=self._calibration.target_coverage,
        )
        return corrected, audit

    def collect_from_frame(
        self,
        frame: pd.DataFrame,
        *,
        forecast_origin: pd.Timestamp,
        window: int,
        stage: str = "backtest",
    ) -> int:
        """把一折的（有限 actual）行记入校准池；返回新增记录数。"""
        self._check_frame(frame)
        origin = pd.Timestamp(forecast_origin)
        delay = int(self._calibration.label_availability_delay_steps)
        frame = frame.copy()
        frame["cqr_actual"] = pd.to_numeric(frame["actual_value"], errors="coerce")
        frame["cqr_lower"] = pd.to_numeric(frame[self._lower_column], errors="coerce")
        frame["cqr_upper"] = pd.to_numeric(frame[self._upper_column], errors="coerce")
        finite = (
            np.isfinite(frame["cqr_actual"])
            & np.isfinite(frame["cqr_lower"])
            & np.isfinite(frame["cqr_upper"])
        )
        sub = frame.loc[finite].sort_values("time")
        if sub.empty:
            return 0
        # horizon_step：每个 (series_id, target) 内按 time 的排名（1 起）
        sub["cqr_hstep"] = (
            sub.groupby(["series_id", "target"], sort=False).cumcount() + 1
        )
        count = 0
        for row in sub.itertuples(index=False):
            target_time = pd.Timestamp(row.time)
            count += 1
            self._records.append(
                CalibrationRecord(
                    forecast_origin=origin,
                    target_time=target_time,
                    label_available_at=target_time + delay * self._freq_offset,
                    horizon_step=int(row.cqr_hstep),
                    window=int(window),
                    interval_name=self._interval.name,
                    lower=float(row.cqr_lower),
                    upper=float(row.cqr_upper),
                    y_true=float(row.cqr_actual),
                    score=float(
                        max(row.cqr_lower - row.cqr_actual, row.cqr_actual - row.cqr_upper)
                    ),
                    stage=stage,
                    series_id=str(row.series_id),
                    target=str(row.target),
                )
            )
        return count

    def final_correction(
        self,
        forecast_origin: pd.Timestamp,
    ) -> Tuple[CalibrationResult, dict]:
        """最终预测的修正量：消费全部满足 as-of 的历史折。"""
        result = self._calibrate(pd.Timestamp(forecast_origin))
        return result, self._audit(result)

    def correction_bounds(
        self,
        quantile_values: np.ndarray,
        levels: Sequence[float],
        correction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """把修正量应用到任意形状的 quantile values 末轴（levels 维）。"""
        levels = tuple(float(level) for level in levels)
        lower_index = levels.index(self._interval.lower_quantile)
        upper_index = levels.index(self._interval.upper_quantile)
        lower = quantile_values[..., lower_index] - float(correction)
        upper = quantile_values[..., upper_index] + float(correction)
        return lower, upper
