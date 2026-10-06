"""概率配置的唯一运行期语义解析。"""
from __future__ import annotations

from dataclasses import dataclass
from forecasting_core.probability._validation import (
    _is_close,
    _require_mapping,
    _strict_bool,
    _strict_int,
    _strict_number,
    _validate_unknown_keys,
)
from forecasting_core.probability.calibration import CalibrationSpec, ResidualCalibrationSpec
from forecasting_core.probability.grid import (
    _quantile_token,
    validate_interval_quantiles,
    validate_quantile_grid,
)
from typing import Any, Mapping, Optional, Tuple
import math
import warnings


_SUPPORTED_CROSSING_METHODS = {
    "none",
    "rearrangement",
    "median_preserving_isotonic",
}


_DEFAULT_CROSSING_METHOD = "median_preserving_isotonic"


_SUPPORTED_RECURSIVE_PROPAGATION = {"median_path"}


PROBABILISTIC_FIELDS = frozenset({
    "mode",
    "quantiles",
    "point_quantile",
    "crossing",
    "intervals",
    "calibration",
})


_CROSSING_KEYS = {"method", "report_raw"}


_INTERVAL_KEYS = {"name", "lower_quantile", "upper_quantile"}


_CALIBRATION_KEYS = {
    "method",
    "interval",
    "target_coverage",
    "calibration_windows",
    "min_windows",
    "min_scores",
    "label_availability_delay_steps",
    "allow_interval_shrink",
    "grouping",
}


@dataclass(frozen=True)
class IntervalSpec:
    """由两个模型 quantile 定义的基础边际区间。"""

    name: str
    lower_quantile: float
    upper_quantile: float

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        lower = _strict_number(self.lower_quantile, "lower_quantile")
        upper = _strict_number(self.upper_quantile, "upper_quantile")
        if not name:
            raise ValueError("interval name must not be empty")
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError("interval quantiles must be finite")
        if lower >= upper:
            raise ValueError("lower_quantile must be < upper_quantile")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "lower_quantile", lower)
        object.__setattr__(self, "upper_quantile", upper)

    @property
    def nominal_coverage(self) -> float:
        return self.upper_quantile - self.lower_quantile


@dataclass(frozen=True)
class ProbabilisticSpec:
    """主链唯一消费的概率预测配置。"""

    mode: str
    quantiles: tuple[float, ...]
    point_quantile: float
    recursive_propagation: str
    crossing_method: str
    crossing_report_raw: bool
    intervals: tuple[IntervalSpec, ...]
    calibration: CalibrationSpec | ResidualCalibrationSpec | None
    schema_version: int = 1

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in {"point", "quantile"}:
            raise ValueError(f"Unsupported probabilistic mode={mode}; expected point or quantile")
        if _strict_int(self.schema_version, "schema_version") != 1:
            raise ValueError(f"Unsupported probabilistic schema_version={self.schema_version}")
        crossing_method = str(self.crossing_method).lower()
        if crossing_method not in _SUPPORTED_CROSSING_METHODS:
            raise ValueError(f"Unsupported crossing method={crossing_method}")
        recursive_propagation = str(self.recursive_propagation).lower()
        if recursive_propagation not in _SUPPORTED_RECURSIVE_PROPAGATION:
            raise ValueError(
                f"Unsupported recursive_propagation={recursive_propagation}; "
                "only median_path is implemented"
            )

        if mode == "point":
            if self.intervals or (self.calibration is not None and not isinstance(self.calibration, ResidualCalibrationSpec)):
                raise ValueError("point mode forbids intervals and calibration")
            levels: tuple[float, ...] = ()
        else:
            if isinstance(self.calibration, ResidualCalibrationSpec):
                raise ValueError("absolute_residual requires point mode")
            levels = validate_quantile_grid(self.quantiles, self.point_quantile)
            names = [interval.name for interval in self.intervals]
            if len(set(names)) != len(names):
                raise ValueError("interval names must be unique")
            for interval in self.intervals:
                validate_interval_quantiles(
                    interval.lower_quantile,
                    interval.upper_quantile,
                    levels,
                )
            if self.calibration is not None:
                intervals_by_name = {interval.name: interval for interval in self.intervals}
                if self.calibration.interval_name not in intervals_by_name:
                    raise ValueError(
                        f"calibration interval={self.calibration.interval_name!r} "
                        "must reference a configured interval"
                    )
                interval = intervals_by_name[self.calibration.interval_name]
                if not _is_close(
                    interval.nominal_coverage,
                    self.calibration.target_coverage,
                ):
                    warnings.warn(
                        "CQR target coverage differs from the base interval nominal coverage; "
                        "calibrated bounds remain prediction-interval bounds, not quantiles",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "quantiles", levels)
        object.__setattr__(self, "point_quantile", _strict_number(self.point_quantile, "point_quantile"))
        object.__setattr__(self, "recursive_propagation", recursive_propagation)
        object.__setattr__(self, "crossing_method", crossing_method)
        object.__setattr__(self, "crossing_report_raw", _strict_bool(self.crossing_report_raw, "report_raw"))
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "schema_version", 1)

    def interval_by_name(self, name: str) -> IntervalSpec:
        for interval in self.intervals:
            if interval.name == name:
                return interval
        raise ValueError(f"Unknown interval name={name!r}")

    @property
    def calibration_interval(self) -> Optional[IntervalSpec]:
        if self.calibration is None or isinstance(self.calibration, ResidualCalibrationSpec):
            return None
        return self.interval_by_name(self.calibration.interval_name)


def resolve_crossing_settings(
    probabilistic: Mapping[str, Any],
) -> Tuple[str, bool]:
    """从 canonical probabilistic mapping 解析 crossing 设置（运行时唯一入口）。

    返回 ``(method, report_raw)``；未声明 ``crossing`` 块时回落到
    ``_DEFAULT_CROSSING_METHOD``（保持历史硬编码行为）。
    """
    if not isinstance(probabilistic, Mapping):
        raise TypeError("probabilistic must be a mapping")
    raw = probabilistic.get("crossing")
    if raw is None:
        return _DEFAULT_CROSSING_METHOD, True
    crossing = _require_mapping(raw, "probabilistic.crossing")
    _validate_unknown_keys(crossing, _CROSSING_KEYS, "probabilistic.crossing")
    method = str(crossing.get("method", _DEFAULT_CROSSING_METHOD)).lower()
    if method not in _SUPPORTED_CROSSING_METHODS:
        raise ValueError(f"Unsupported crossing method={method}")
    return method, _strict_bool(crossing.get("report_raw", True), "report_raw")


def _new_spec(raw_mapping: Mapping[str, Any]) -> ProbabilisticSpec:
    if not isinstance(raw_mapping, Mapping):
        raise TypeError("probabilistic must be a mapping")
    # 解析不修改嵌套值，兼容不可 pickle 的冻结 Mapping。
    mapping = dict(raw_mapping)
    _validate_unknown_keys(mapping, PROBABILISTIC_FIELDS, "probabilistic")
    if mapping.get("calibration") is not None:
        calibration_mapping = _require_mapping(mapping["calibration"], "probabilistic.calibration")
        _validate_unknown_keys(calibration_mapping, _CALIBRATION_KEYS, "probabilistic.calibration")
    mode = str(mapping.get("mode", "point") or "point").lower()
    if mode == "point":
        unused = sorted(set(mapping) & {"quantiles", "point_quantile", "crossing"})
        if unused:
            raise ValueError(f"point mode forbids quantile-only fields: {unused}")
        raw_calibration = mapping.get("calibration")
        calibration = None
        if isinstance(raw_calibration, Mapping) and raw_calibration.get("method") == "absolute_residual":
            calibration = ResidualCalibrationSpec.from_mapping(raw_calibration)
        forbidden = [key for key in ("intervals", "calibration")
                     if mapping.get(key) and not (key == "calibration" and calibration is not None)]
        if forbidden:
            raise ValueError(f"point mode forbids {', '.join(forbidden)}")
        return ProbabilisticSpec(
            mode="point",
            quantiles=(),
            point_quantile=_strict_number(mapping.get("point_quantile", 0.5), "point_quantile"),
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=calibration,
            schema_version=1,
        )
    if mode != "quantile":
        raise ValueError(f"Unsupported probabilistic mode={mode}; expected point or quantile")

    point_quantile = _strict_number(mapping.get("point_quantile", 0.5), "point_quantile")
    levels = validate_quantile_grid(mapping.get("quantiles", ()), point_quantile)
    crossing_method, crossing_report_raw = resolve_crossing_settings(mapping)

    raw_intervals = mapping.get("intervals")
    if raw_intervals is None:
        raw_intervals = [] if len(levels) == 1 else [
            {
                "name": f"q{_quantile_token(levels[0])}_q{_quantile_token(levels[-1])}",
                "lower_quantile": levels[0],
                "upper_quantile": levels[-1],
            }
        ]
    if not isinstance(raw_intervals, (list, tuple)):
        raise ValueError("probabilistic.intervals must be a list")
    intervals = []
    for index, raw_interval in enumerate(raw_intervals):
        interval_mapping = _require_mapping(
            raw_interval,
            f"probabilistic.intervals[{index}]",
        )
        _validate_unknown_keys(
            interval_mapping,
            _INTERVAL_KEYS,
            f"probabilistic.intervals[{index}]",
        )
        missing = [
            key
            for key in ("name", "lower_quantile", "upper_quantile")
            if key not in interval_mapping
        ]
        if missing:
            raise ValueError(
                f"probabilistic.intervals[{index}] missing required key(s): {missing}"
            )
        intervals.append(
            IntervalSpec(
                name=interval_mapping["name"],
                lower_quantile=interval_mapping["lower_quantile"],
                upper_quantile=interval_mapping["upper_quantile"],
            )
        )

    calibration = None
    raw_calibration = mapping.get("calibration")
    if raw_calibration:
        calibration_mapping = _require_mapping(
            raw_calibration,
            "probabilistic.calibration",
        )
        _validate_unknown_keys(
            calibration_mapping,
            _CALIBRATION_KEYS,
            "probabilistic.calibration",
        )
        method = str(calibration_mapping.get("method", "cqr") or "cqr").lower()
        missing = [
            key
            for key in ("interval", "target_coverage")
            if key not in calibration_mapping
        ]
        if missing:
            raise ValueError(
                f"probabilistic.calibration missing required key(s): {missing}"
            )
        calibration = CalibrationSpec(
            method=method,
            interval_name=calibration_mapping["interval"],
            target_coverage=calibration_mapping["target_coverage"],
            calibration_windows=calibration_mapping.get("calibration_windows", 5),
            min_windows=calibration_mapping.get("min_windows", 3),
            min_scores=calibration_mapping.get("min_scores", 30),
            label_availability_delay_steps=calibration_mapping.get(
                "label_availability_delay_steps",
                0,
            ),
            allow_interval_shrink=calibration_mapping.get(
                "allow_interval_shrink",
                False,
            ),
            grouping=calibration_mapping.get("grouping", "pooled"),
        )

    return ProbabilisticSpec(
        mode=mode,
        quantiles=levels,
        point_quantile=point_quantile,
        recursive_propagation="median_path",
        crossing_method=crossing_method,
        crossing_report_raw=crossing_report_raw,
        intervals=tuple(intervals),
        calibration=calibration,
        schema_version=1,
    )


def probabilistic_spec_from_mapping(
    raw_mapping: Mapping[str, Any],
) -> ProbabilisticSpec:
    """从 canonical probabilistic mapping 构建部署态 spec（运行时唯一入口）。

    只接受新版键集合（mode/quantiles/point_quantile/crossing/intervals/
    calibration）；legacy ``crossing_method``
    与 ``conformal`` 键一律 RAISE（已于 2026-09-01 从全部现役 YAML 清扫）。
    """
    return _new_spec(raw_mapping)
