# -*- coding: utf-8 -*-
"""概率预测配置归一化与严格语义校验。"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Tuple


_SUPPORTED_CROSSING_METHODS = {
    "none",
    "rearrangement",
    "median_preserving_isotonic",
}
# 运行时唯一默认（2026-09-01 裂缝修复）：与 forecaster 历史硬编码行为一致——
# 排序 + 钳制到 point level 锚点。配置缺省时必须保持这一行为，否则全部存量
# quantile 结果的语义静默改变。
_DEFAULT_CROSSING_METHOD = "median_preserving_isotonic"
_SUPPORTED_RECURSIVE_PROPAGATION = {"median_path"}
_SUPPORTED_CALIBRATION_METHODS = {"cqr"}
_SUPPORTED_CALIBRATION_GROUPINGS = {"pooled"}
_TOP_LEVEL_KEYS = {
    "mode",
    "schema_version",
    "quantiles",
    "point_quantile",
    "recursive_propagation",
    "crossing",
    "intervals",
    "calibration",
}
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


def _is_close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _validate_unknown_keys(mapping: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"Unknown {path} key(s): {unknown}")


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def validate_quantile_grid(
    quantiles: Iterable[float],
    point_quantile: float = 0.5,
) -> Tuple[float, ...]:
    """将合法分位数网格归一化为严格递增的 float tuple。"""
    levels = tuple(float(level) for level in quantiles)
    if not levels:
        raise ValueError("quantiles must not be empty")
    if any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in levels):
        raise ValueError("quantiles must be finite and inside (0, 1)")
    if len(set(levels)) != len(levels):
        raise ValueError("quantiles must be unique")
    if any(left >= right for left, right in zip(levels, levels[1:])):
        raise ValueError("quantiles must be strictly increasing")
    point = float(point_quantile)
    if not any(_is_close(level, point) for level in levels):
        raise ValueError(f"point_quantile={point:g} must be present in quantiles")
    return levels


def validate_interval_quantiles(
    lower_quantile: float,
    upper_quantile: float,
    quantiles: Iterable[float],
) -> Tuple[float, float]:
    """校验区间边界引用已配置且严格有序的 quantile。"""
    lower = float(lower_quantile)
    upper = float(upper_quantile)
    levels = tuple(float(level) for level in quantiles)
    if lower >= upper:
        raise ValueError("lower_quantile must be < upper_quantile")
    for name, value in (("lower_quantile", lower), ("upper_quantile", upper)):
        if not any(_is_close(level, value) for level in levels):
            raise ValueError(f"{name}={value:g} must be present in quantiles")
    return lower, upper


def validate_cqr_params(alpha: float, min_scores: int) -> Tuple[float, int]:
    """校验 CQR 的误覆盖率和最小 score 数。"""
    alpha_value = float(alpha)
    min_score_count = int(min_scores)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be finite and inside (0, 1)")
    if min_score_count <= 0:
        raise ValueError("min_scores must be > 0")
    return alpha_value, min_score_count


@dataclass(frozen=True)
class IntervalSpec:
    """由两个模型 quantile 定义的基础边际区间。"""

    name: str
    lower_quantile: float
    upper_quantile: float

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        lower = float(self.lower_quantile)
        upper = float(self.upper_quantile)
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
class CalibrationSpec:
    """一个 prediction interval 的 CQR 校准配置。"""

    method: str
    interval_name: str
    target_coverage: float
    calibration_windows: int
    min_windows: int
    min_scores: int
    label_availability_delay_steps: int = 0
    allow_interval_shrink: bool = False
    grouping: str = "pooled"

    def __post_init__(self) -> None:
        method = str(self.method).lower()
        grouping = str(self.grouping).lower()
        coverage = float(self.target_coverage)
        calibration_windows = int(self.calibration_windows)
        min_windows = int(self.min_windows)
        min_scores = int(self.min_scores)
        delay_steps = int(self.label_availability_delay_steps)
        if method not in _SUPPORTED_CALIBRATION_METHODS:
            raise ValueError(f"Unsupported calibration method={method}")
        if grouping not in _SUPPORTED_CALIBRATION_GROUPINGS:
            raise ValueError(f"Unsupported calibration grouping={grouping}")
        if not math.isfinite(coverage) or not 0.0 < coverage < 1.0:
            raise ValueError("target_coverage must be finite and inside (0, 1)")
        if calibration_windows <= 0:
            raise ValueError("calibration_windows must be > 0")
        if min_windows <= 0:
            raise ValueError("min_windows must be > 0")
        if min_windows > calibration_windows:
            raise ValueError("min_windows must be <= calibration_windows")
        if min_scores <= 0:
            raise ValueError("min_scores must be > 0")
        if delay_steps < 0:
            raise ValueError("label_availability_delay_steps must be >= 0")
        interval_name = str(self.interval_name).strip()
        if not interval_name:
            raise ValueError("calibration interval must not be empty")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "interval_name", interval_name)
        object.__setattr__(self, "target_coverage", coverage)
        object.__setattr__(self, "calibration_windows", calibration_windows)
        object.__setattr__(self, "min_windows", min_windows)
        object.__setattr__(self, "min_scores", min_scores)
        object.__setattr__(self, "label_availability_delay_steps", delay_steps)
        object.__setattr__(self, "allow_interval_shrink", bool(self.allow_interval_shrink))
        object.__setattr__(self, "grouping", grouping)


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
    calibration: Optional[CalibrationSpec]
    schema_version: int = 1

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in {"point", "quantile"}:
            raise ValueError(f"Unsupported probabilistic mode={mode}; expected point or quantile")
        if int(self.schema_version) != 1:
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
            if self.intervals or self.calibration is not None:
                raise ValueError("point mode forbids intervals and calibration")
            levels: tuple[float, ...] = ()
        else:
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
        object.__setattr__(self, "point_quantile", float(self.point_quantile))
        object.__setattr__(self, "recursive_propagation", recursive_propagation)
        object.__setattr__(self, "crossing_method", crossing_method)
        object.__setattr__(self, "crossing_report_raw", bool(self.crossing_report_raw))
        object.__setattr__(self, "intervals", tuple(self.intervals))
        object.__setattr__(self, "schema_version", 1)

    def interval_by_name(self, name: str) -> IntervalSpec:
        for interval in self.intervals:
            if interval.name == name:
                return interval
        raise ValueError(f"Unknown interval name={name!r}")

    @property
    def calibration_interval(self) -> Optional[IntervalSpec]:
        if self.calibration is None:
            return None
        return self.interval_by_name(self.calibration.interval_name)


def calibration_runtime_kwargs(spec: ProbabilisticSpec) -> dict[str, Any]:
    """把 canonical calibration spec 转成 evaluator/final 共享运行参数。"""
    calibration = spec.calibration
    interval = spec.calibration_interval
    if calibration is None or interval is None:
        return {"enable_cqr": False}
    return {
        "enable_cqr": True,
        "alpha": round(1.0 - calibration.target_coverage, 15),
        "calibration_windows": calibration.calibration_windows,
        "min_windows": calibration.min_windows,
        "min_scores": calibration.min_scores,
        "label_availability_delay_steps": calibration.label_availability_delay_steps,
        "interval_name": interval.name,
        "lower_quantile": interval.lower_quantile,
        "upper_quantile": interval.upper_quantile,
        "allow_interval_shrink": calibration.allow_interval_shrink,
    }


def _quantile_token(level: float) -> str:
    percent = float(level) * 100.0
    if _is_close(percent, round(percent)):
        return str(int(round(percent)))
    return f"{percent:.12f}".rstrip("0").rstrip(".").replace(".", "p")


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
    return method, bool(crossing.get("report_raw", True))


def _legacy_spec(args: Any) -> ProbabilisticSpec:
    mode = str(getattr(args, "predict_type", "point") or "point").lower()
    if mode == "point":
        if bool(getattr(args, "enable_conformal_calibration", False)):
            raise ValueError("enable_conformal_calibration requires predict_type=quantile")
        return ProbabilisticSpec(
            mode="point",
            quantiles=(),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )
    levels = validate_quantile_grid(getattr(args, "quantiles", ()), point_quantile=0.5)
    interval_name = f"q{_quantile_token(levels[0])}_q{_quantile_token(levels[-1])}"
    interval = IntervalSpec(interval_name, levels[0], levels[-1])
    calibration = None
    if bool(getattr(args, "enable_conformal_calibration", False)):
        alpha, min_scores = validate_cqr_params(
            getattr(args, "conformal_alpha", 0.1),
            getattr(args, "conformal_min_scores", 30),
        )
        calibration_windows = int(getattr(args, "conformal_calibration_windows", 5))
        min_windows = int(getattr(args, "conformal_min_windows", 3))
        delay_steps = int(
            getattr(args, "conformal_label_availability_delay_steps", 0)
        )
        if calibration_windows <= 0:
            raise ValueError("conformal_calibration_windows must be > 0")
        if min_windows <= 0:
            raise ValueError("conformal_min_windows must be > 0")
        if min_windows > calibration_windows:
            raise ValueError(
                "conformal_min_windows must be <= conformal_calibration_windows"
            )
        if delay_steps < 0:
            raise ValueError(
                "conformal_label_availability_delay_steps must be >= 0"
            )
        calibration = CalibrationSpec(
            method="cqr",
            interval_name=interval_name,
            target_coverage=1.0 - alpha,
            calibration_windows=calibration_windows,
            min_windows=min_windows,
            min_scores=min_scores,
            label_availability_delay_steps=delay_steps,
            allow_interval_shrink=False,
            grouping="pooled",
        )
    return ProbabilisticSpec(
        mode="quantile",
        quantiles=levels,
        point_quantile=0.5,
        recursive_propagation="median_path",
        crossing_method=(
            "rearrangement" if bool(getattr(args, "quantile_monotone", False)) else "none"
        ),
        crossing_report_raw=True,
        intervals=(interval,),
        calibration=calibration,
    )


def _new_spec(raw_mapping: Mapping[str, Any]) -> ProbabilisticSpec:
    mapping = copy.deepcopy(dict(raw_mapping))
    _validate_unknown_keys(mapping, _TOP_LEVEL_KEYS, "probabilistic")
    mode = str(mapping.get("mode", "point") or "point").lower()
    schema_version = int(mapping.get("schema_version", 1))
    if mode == "point":
        forbidden = [key for key in ("intervals", "calibration") if mapping.get(key)]
        if forbidden:
            raise ValueError(f"point mode forbids {', '.join(forbidden)}")
        return ProbabilisticSpec(
            mode="point",
            quantiles=(),
            point_quantile=float(mapping.get("point_quantile", 0.5)),
            recursive_propagation=str(mapping.get("recursive_propagation", "median_path")),
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
            schema_version=schema_version,
        )
    if mode != "quantile":
        raise ValueError(f"Unsupported probabilistic mode={mode}; expected point or quantile")

    point_quantile = float(mapping.get("point_quantile", 0.5))
    levels = validate_quantile_grid(mapping.get("quantiles", ()), point_quantile)
    crossing_raw = mapping.get("crossing", {}) or {}
    crossing = _require_mapping(crossing_raw, "probabilistic.crossing")
    _validate_unknown_keys(crossing, _CROSSING_KEYS, "probabilistic.crossing")
    crossing_method = str(
        crossing.get("method", _DEFAULT_CROSSING_METHOD)
    ).lower()
    crossing_report_raw = bool(crossing.get("report_raw", True))

    raw_intervals = mapping.get("intervals")
    if raw_intervals is None:
        raw_intervals = [
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
        recursive_propagation=str(mapping.get("recursive_propagation", "median_path")),
        crossing_method=crossing_method,
        crossing_report_raw=crossing_report_raw,
        intervals=tuple(intervals),
        calibration=calibration,
        schema_version=schema_version,
    )


def probabilistic_spec_from_mapping(
    raw_mapping: Mapping[str, Any],
) -> ProbabilisticSpec:
    """从 canonical probabilistic mapping 构建部署态 spec（运行时唯一入口）。

    只接受新版键集合（mode/quantiles/point_quantile/recursive_propagation/
    crossing/intervals/calibration/schema_version）；legacy ``crossing_method``
    与 ``conformal`` 键一律 RAISE（已于 2026-09-01 从全部现役 YAML 清扫）。
    """
    return _new_spec(raw_mapping)


def _legacy_fields_are_explicit(args: Any) -> bool:
    """判断新 mapping 旁是否同时存在有语义的 legacy 概率配置。"""
    mode = str(getattr(args, "predict_type", "point") or "point").lower()
    return (
        mode != "point"
        or bool(getattr(args, "enable_conformal_calibration", False))
        or bool(getattr(args, "quantile_monotone", False))
    )


def resolve_probabilistic_spec(args: Any) -> ProbabilisticSpec:
    """把 legacy 字段或新版 mapping 归一化为唯一不可变 spec。"""
    raw_mapping = getattr(args, "probabilistic", {}) or {}
    if raw_mapping and not isinstance(raw_mapping, Mapping):
        raise ValueError("probabilistic must be a mapping")
    if not raw_mapping:
        return _legacy_spec(args)

    canonical = _new_spec(raw_mapping)
    if _legacy_fields_are_explicit(args):
        legacy = _legacy_spec(args)
        if legacy != canonical:
            raise ValueError(
                "legacy and probabilistic config conflict: "
                f"legacy={legacy!r}, probabilistic={canonical!r}"
            )
    return canonical


def apply_probabilistic_spec_to_args(args: Any, spec: ProbabilisticSpec) -> None:
    """迁移期薄适配器：让 legacy 主链字段与已解析 spec 保持同义。"""
    args.probabilistic_spec = spec
    args.predict_type = spec.mode
    args.quantiles = list(spec.quantiles)
    args.quantile_monotone = spec.crossing_method != "none"
    calibration = spec.calibration
    args.enable_conformal_calibration = calibration is not None
    if calibration is None:
        return
    args.conformal_alpha = 1.0 - calibration.target_coverage
    args.conformal_calibration_windows = calibration.calibration_windows
    args.conformal_min_windows = calibration.min_windows
    args.conformal_min_scores = calibration.min_scores
    args.conformal_label_availability_delay_steps = (
        calibration.label_availability_delay_steps
    )
    args.conformal_allow_interval_shrink = calibration.allow_interval_shrink


def validate_probabilistic_args(args: Any) -> Tuple[float, ...]:
    """主流程兼容入口：解析完整 spec 并返回 quantile grid。"""
    return resolve_probabilistic_spec(args).quantiles
