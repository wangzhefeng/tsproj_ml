#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tsproj_ml 模型配置 dry-run 校验：只加载配置 + 复算 main.py 合法性校验，不训练不预测。

用法（项目根，注意 env -u PYTHONPATH 防 Hermes 注入遮蔽项目 utils/ 包）：
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/<scenario>/route_*/lgbm_*.yaml'
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/aidc_power_15min/**/*.yaml'
    （无参数时默认 glob config/**/*.yaml，仅检查含 base_config/overrides 的模型 schema）

复算的校验（与 main.py.__init__ 一致）：
  - window_length < history_length
  - window_len - horizon > max(lags)（滞后特征非全 NaN；USMDP 仅在未启用 safe-lag 时除外）
  - USMDP 多步 safe-lag：align_direct_features_to_target=true 时必须启用 lag，且 min(lags) >= horizon
  - 目标分解方法/周期合法；概率模型具备原生 quantile objective
  - n_windows > 0（滑窗数）
  - advanced_features：USMDP 不能直接依赖 y；仅操作历史/未来都存在的列才可用
    （USMDP 仅在 align_direct_features_to_target=true 时生成 y_lag_*）；预测上下文取
    max(lags, 已启用 rolling_windows/diff_periods/pct_change_periods)，且不得超过
    history_length × n_per_day

退出码：0 = 全部通过（可能有提示性警告），1 = 存在硬校验失败。
"""
from __future__ import annotations

import glob
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from config.config_loader import is_model_yaml, load_yaml_config  # noqa: E402
from model_forecasting.specs import ForecastConfigSpec  # noqa: E402
from probabilistic.objectives import validate_quantile_model_support  # noqa: E402
from probabilistic.spec import resolve_probabilistic_spec  # noqa: E402
from utils.frequency import resolve_samples_per_day  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent


_CANONICAL_NESTED_FIELDS = {
    "features.transformations": {
        "advanced",
        "direct",
        "direct_layout",
        "feature_scaling",
        "target",
        "target_transform",
        "datetime_categorical",
    },
    "features.transformations.advanced": {
        "rolling",
        "expanding",
        "difference",
        "percent_change",
        "time_since",
        "cyclical",
        "interaction",
        "polynomial",
        "rolling_windows",
        "diff_periods",
        "pct_change_periods",
    },
    "features.transformations.advanced.rolling": {"columns", "windows", "stats"},
    "features.selection": {
        "enabled",
        "method",
        "max_features",
        "min_features",
        "force_keep",
    },
    "features.transformations.advanced.expanding": {"columns", "stats"},
    "features.transformations.advanced.difference": {"columns", "periods"},
    "features.transformations.advanced.percent_change": {"columns", "periods"},
    "features.transformations.advanced.time_since": {"columns", "events"},
    "features.transformations.advanced.cyclical": {"columns", "period"},
    "features.transformations.advanced.interaction": {"column_pairs", "operations"},
    "features.transformations.advanced.polynomial": {"columns", "degree"},
    "features.transformations.direct": {
        "layout",
        "use_horizon_exogenous",
        "align_to_target",
        "horizon_feature",
    },
    "features.transformations.direct.horizon_feature": {"name", "cyclical"},
    "features.transformations.feature_scaling": {
        "method",
        "grouped",
        "encode_categorical",
    },
    "features.transformations.target": {
        "calendar_normalization",
        "decomposition",
        "scaling",
    },
    "features.transformations.target.calendar_normalization": {"method"},
    "features.transformations.target.decomposition": {
        "method",
        "composition",
        "periods",
        "robust",
        "trend_degree",
        "trend_forecast",
        "damping",
        "trend_lookback",
        "seasonal_cycles",
    },
    "features.transformations.target.scaling": {"method", "inverse"},
    "validation": {
        "forecast_origin",
        "schedule_mode",
        "horizon_mode",
        "history_length",
        "window_length",
        "train_window_length",
        "training_scope",
        "training",
        "train_outlier",
        "eval_mask",
        "performance",
    },
    "validation.training_scope": {
        "incomplete_series_policy",
        "unknown_series_policy",
    },
    "validation.training": {
        "early_stopping_patience",
        "sample_weight",
        "tuning",
        "augmentation",
        "feature_selection",
        "learning_rate",
        "huber_delta",
        "blend_weight_windows",
        "estimator_ensemble",
    },
    "validation.training.sample_weight": {"method", "halflife_days"},
    "validation.training.tuning": {"method", "metric", "n_splits"},
    "validation.training.augmentation": {
        "method",
        "ratio",
        "feature_noise_std",
        "target_noise_std",
        "random_state",
    },
    "validation.training.feature_selection": {
        "method",
        "max_features",
        "min_features",
    },
    "validation.training.learning_rate": {"method", "min", "max"},
    "validation.training.estimator_ensemble": {
        "method",
        "members",
        "member_specs",
        "validation_ratio",
    },
    "validation.train_outlier": {"method", "high", "rise", "low", "drop"},
    "validation.train_outlier.high": {"threshold", "max_run_points"},
    "validation.train_outlier.rise": {"max_run_points", "rebound_min_abs_diff"},
    "validation.train_outlier.low": {"threshold", "max_run_points"},
    "validation.train_outlier.drop": {"max_run_points", "rebound_min_abs_diff"},
    "validation.eval_mask": {"mode", "percentile", "min_value", "max_value"},
    "validation.performance": {
        "window_parallel_workers",
        "max_test_windows",
        "test_window_stride",
        "multi_output_n_jobs",
        "quantile_parallel_workers",
        "ensemble_parallel_workers",
        "model_thread_count",
        "step_logging",
        "forecast_log_interval",
    },
    "output": {
        "identity",
        "directories",
        "overlay",
        "scenario_subpath",
        "setting_suffix",
        "results_root",
    },
    "output.identity": {"scenario_subpath", "setting_suffix"},
    "output.directories": {"checkpoints", "tests", "forecast"},
    "output.overlay": {"path", "column"},
    "probabilistic": {
        "mode",
        "schema_version",
        "quantiles",
        "point_quantile",
        "recursive_propagation",
        "crossing",
        "crossing_method",
        "intervals",
        "conformal",
    },
    "probabilistic.crossing": {"method", "report_raw"},
    "probabilistic.conformal": {
        "method",
        "interval",
        "target_coverage",
        "alpha",
        "calibration_windows",
        "min_windows",
        "min_scores",
        "label_availability_delay_steps",
        "allow_interval_shrink",
        "grouping",
        "coverage",
    },
}

_CANONICAL_FREQUENCIES = frozenset({"5min", "15min", "1h", "1D", "1ME", "1MS"})
_CANONICAL_DATETIME_FEATURES = frozenset(
    {
        "minute",
        "hour",
        "day",
        "day_of_week",
        "week_of_year",
        "month",
        "days_in_month",
        "quarter",
        "day_of_year",
        "year",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
    }
)
_DIRECT_LAYOUTS = frozenset({"independent_models", "single_model_horizon"})
_CALENDAR_NORMALIZATION_METHODS = frozenset({"none", "per_calendar_day"})
_DECOMPOSITION_METHODS = frozenset({"none", "linear", "quadratic", "damped", "stl", "mstl"})
_TARGET_SCALING_METHODS = frozenset({"none", "minmax", "standard", "robust"})


def _nested_value(root: Mapping[str, Any], path: str) -> Any:
    value: Any = root
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _check_canonical_nested_fields(cfg: ForecastConfigSpec) -> list[str]:
    payload = cfg.canonical_payload()
    problems = []
    for path, allowed in _CANONICAL_NESTED_FIELDS.items():
        value = _nested_value(payload, path)
        if value is None:
            continue
        if not isinstance(value, Mapping):
            problems.append(f"{path} 必须是 mapping")
            continue
        unknown = sorted(set(value) - allowed)
        if unknown:
            problems.append(f"{path} 包含未知字段: {unknown}")
    return problems


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _check_positive_integer_sequence(value: Any, path: str) -> list[str]:
    if not _is_sequence(value) or not value:
        return [f"{path} 必须是非空正整数序列"]
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        return [f"{path} 必须是非空正整数序列"]
    return []


def _check_nonempty_string_sequence(value: Any, path: str) -> list[str]:
    if not _is_sequence(value) or not value:
        return [f"{path} 必须是非空字符串序列"]
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return [f"{path} 必须是非空字符串序列"]
    return []


def _check_problem_data_contract(cfg: ForecastConfigSpec) -> list[str]:
    problems = []
    if cfg.problem.freq not in _CANONICAL_FREQUENCIES:
        problems.append(
            f"problem.freq 必须使用 canonical 集合: {sorted(_CANONICAL_FREQUENCIES)}"
        )
    if isinstance(cfg.problem.horizon, bool) or not isinstance(cfg.problem.horizon, int) or cfg.problem.horizon <= 0:
        problems.append("problem.horizon 必须是正整数")
    if cfg.problem.targets != cfg.data.target_columns:
        problems.append("problem.targets 必须与 data target columns 顺序完全一致")

    for source in cfg.data.sources:
        roles = {column.role.value for column in source.columns}
        if "target" in roles:
            if source.time_col != cfg.problem.time_col:
                problems.append(
                    f"target source {source.name!r} time_col 必须等于 problem.time_col"
                )
            availability = getattr(source.availability, "value", source.availability)
            if availability != "source_time":
                problems.append(
                    f"target source {source.name!r} availability 必须是 source_time"
                )
        if "observed_past" in roles and source.provider is None:
            problems.append(
                f"observed_past source {source.name!r} 缺少显式 provider"
            )
        if "known_future" in roles:
            availability = getattr(source.availability, "value", source.availability)
            valid = (
                source.source_type == "file"
                and availability in {"column", "forecast_origin"}
            ) or (
                source.source_type == "generated"
                and availability == "generator_defined"
            )
            if not valid:
                problems.append(
                    f"known_future source {source.name!r} availability 与 source_type 不合法"
                )
    return problems


def _check_feature_contract(cfg: ForecastConfigSpec) -> list[str]:
    problems = []
    for field_name, lag_mapping in (
        ("target_lags", cfg.features.target_lags),
        ("observed_past_lags", cfg.features.observed_past_lags),
    ):
        for name, lags in lag_mapping.items():
            problems.extend(
                _check_positive_integer_sequence(lags, f"features.{field_name}.{name}")
            )

    unknown_datetime = sorted(
        set(cfg.features.datetime_features) - set(_CANONICAL_DATETIME_FEATURES)
    )
    if unknown_datetime:
        problems.append(
            f"features.datetime_features 包含非 canonical 字段: {unknown_datetime}"
        )
    return problems


def _check_direct_transform(value: Any) -> list[str]:
    if value is None:
        return []
    path = "features.transformations.direct"
    if not isinstance(value, Mapping):
        return [f"{path} 必须是 mapping"]
    problems = []
    layout = value.get("layout")
    if layout not in _DIRECT_LAYOUTS:
        problems.append(f"{path}.layout 不合法: {layout!r}")
    for field_name in ("use_horizon_exogenous", "align_to_target"):
        if field_name in value and not isinstance(value[field_name], bool):
            problems.append(f"{path}.{field_name} 必须是 bool")
    horizon_feature = value.get("horizon_feature")
    if horizon_feature is not None:
        if not isinstance(horizon_feature, Mapping):
            problems.append(f"{path}.horizon_feature 必须是 mapping")
        else:
            name = horizon_feature.get("name")
            if not isinstance(name, str) or not name.strip():
                problems.append(f"{path}.horizon_feature.name 必须是非空字符串")
            if "cyclical" in horizon_feature and not isinstance(
                horizon_feature["cyclical"], bool
            ):
                problems.append(f"{path}.horizon_feature.cyclical 必须是 bool")
    return problems


def _check_advanced_transform(value: Any) -> list[str]:
    if value is None:
        return []
    path = "features.transformations.advanced"
    if not isinstance(value, Mapping):
        return [f"{path} 必须是 mapping"]
    problems = []
    for name, period_field in (
        ("rolling", "windows"),
        ("difference", "periods"),
        ("percent_change", "periods"),
    ):
        spec = value.get(name)
        if spec is None:
            continue
        spec_path = f"{path}.{name}"
        if not isinstance(spec, Mapping):
            problems.append(f"{spec_path} 必须是 mapping")
            continue
        if "columns" in spec:
            problems.extend(
                _check_nonempty_string_sequence(spec["columns"], f"{spec_path}.columns")
            )
        if period_field in spec:
            problems.extend(
                _check_positive_integer_sequence(
                    spec[period_field],
                    f"{spec_path}.{period_field}",
                )
            )
        if name == "rolling" and "stats" in spec:
            problems.extend(
                _check_nonempty_string_sequence(spec["stats"], f"{spec_path}.stats")
            )
    polynomial = value.get("polynomial")
    if polynomial is not None:
        polynomial_path = f"{path}.polynomial"
        if not isinstance(polynomial, Mapping):
            problems.append(f"{polynomial_path} 必须是 mapping")
        else:
            degree = polynomial.get("degree")
            if isinstance(degree, bool) or not isinstance(degree, int) or degree <= 0:
                problems.append(f"{polynomial_path}.degree 必须是正整数")
    return problems


def _check_target_transform(value: Any) -> list[str]:
    if value is None:
        return []
    path = "features.transformations.target"
    if not isinstance(value, Mapping):
        return [f"{path} 必须是 mapping"]
    problems = []
    calendar = value.get("calendar_normalization")
    if isinstance(calendar, Mapping):
        method = calendar.get("method", "none")
        if method not in _CALENDAR_NORMALIZATION_METHODS:
            problems.append(f"{path}.calendar_normalization.method 不合法: {method!r}")
    decomposition = value.get("decomposition")
    if isinstance(decomposition, Mapping):
        method = decomposition.get("method", "none")
        if method not in _DECOMPOSITION_METHODS:
            problems.append(f"{path}.decomposition.method 不合法: {method!r}")
        composition = decomposition.get("composition", "additive")
        if composition != "additive":
            problems.append(f"{path}.decomposition.composition 不合法: {composition!r}")
        periods = decomposition.get("periods") or []
        if method == "stl" and len(periods) != 1:
            problems.append(f"{path}.decomposition stl 需要且只能配置一个周期")
        if method == "mstl" and len(periods) < 2:
            problems.append(f"{path}.decomposition mstl 至少需要两个周期")
        if decomposition.get("periods"):
            problems.extend(
                _check_positive_integer_sequence(
                    decomposition["periods"],
                    f"{path}.decomposition.periods",
                )
            )
        if "robust" in decomposition and not isinstance(decomposition["robust"], bool):
            problems.append(f"{path}.decomposition.robust 必须是 bool")
    scaling = value.get("scaling")
    if isinstance(scaling, Mapping):
        method = scaling.get("method", "none")
        if method not in _TARGET_SCALING_METHODS:
            problems.append(f"{path}.scaling.method 不合法: {method!r}")
        if "inverse" in scaling and not isinstance(scaling["inverse"], bool):
            problems.append(f"{path}.scaling.inverse 必须是 bool")
    return problems


def _check_canonical_transform_values(cfg: ForecastConfigSpec) -> list[str]:
    transformations = cfg.features.transformations
    problems = [
        *_check_direct_transform(transformations.get("direct")),
        *_check_advanced_transform(transformations.get("advanced")),
        *_check_target_transform(transformations.get("target_transform")),
    ]
    problems.extend(_check_advanced_target_leakage(cfg))
    return problems


def _check_advanced_target_leakage(cfg: ForecastConfigSpec) -> list[str]:
    """advanced 序列特征引用 target 列时，提示仅限 origin 布局使用。"""
    problems: list[str] = []
    advanced = cfg.features.transformations.get("advanced")
    if not isinstance(advanced, Mapping):
        return problems
    targets = set(cfg.problem.targets)
    lagged = {f"{name}_lag_{lag}" for name in targets for lag in (
        *(cfg.features.target_lags.get(name) or ()),
    )}
    for section in ("rolling", "expanding", "difference", "percent_change"):
        spec = advanced.get(section)
        if not isinstance(spec, Mapping):
            continue
        columns = spec.get("columns") or []
        overlap = sorted(set(columns) & targets)
        if overlap:
            problems.append(
                f"features.transformations.advanced.{section} 不能依赖目标列 y（预测 df_future 无 y）："
                + ", ".join(overlap)
            )
        y_lag_overlap = sorted(set(columns) & lagged)
        if section == "rolling" and y_lag_overlap:
            problems.append(
                "提示：target lag 列仅在 align_direct_features_to_target=true 时进入 rolling；"
                f"当前 {section}_columns 中的 {y_lag_overlap} 会被跳过。"
            )
    return problems


def _build_calendar_month_folds(
    df_history: pd.DataFrame,
    train_window_len: int,
) -> list[dict[str, Any]]:
    """Build complete 1D calendar-month folds for legacy config checks."""
    if train_window_len <= 0:
        raise ValueError("train_window_len must be > 0 for calendar_month folds")
    times = pd.DatetimeIndex(pd.to_datetime(df_history["time"]))
    if len(times) < 2:
        return []
    if not times.is_monotonic_increasing or times.has_duplicates:
        raise ValueError("calendar_month folds require ordered unique timestamps")
    if any(diff != pd.Timedelta(days=1) for diff in times[1:] - times[:-1]):
        raise ValueError("calendar_month folds require a complete regular 1D index")

    last_time = pd.Timestamp(int(times.asi8[-1]))
    current_end = cast(pd.Timestamp, last_time + pd.offsets.MonthBegin(1))
    folds = []
    while True:
        test_end_time = cast(pd.Timestamp, pd.Timestamp(current_end))
        test_start_time = cast(
            pd.Timestamp,
            (test_end_time - pd.offsets.MonthBegin(1)).normalize(),
        )
        test_start = int(times.asi8.searchsorted(test_start_time.value, side="left"))
        test_end = int(times.asi8.searchsorted(test_end_time.value, side="left"))
        expected_horizon = int(test_start_time.days_in_month)
        if test_start >= len(times) or times[test_start] != test_start_time:
            break
        if test_end - test_start != expected_horizon:
            raise ValueError(f"incomplete calendar_month fold: {test_start_time:%Y-%m}")
        train_end = test_start
        train_start = train_end - int(train_window_len)
        if train_start < 0:
            break
        folds.append(
            {
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "horizon": expected_horizon,
            }
        )
        current_end = test_start_time
    return folds


def _resolve_runtime_shape(cfg) -> tuple[str, int, int, int, int]:
    """返回 horizon_mode、最终 horizon、训练行数、窗口数、有效训练天数。"""
    horizon_mode = str(getattr(cfg, "horizon_mode", "fixed_steps") or "fixed_steps").lower()
    n_per_day = resolve_samples_per_day(cfg.freq)
    if horizon_mode == "calendar_month":
        if str(cfg.freq) != "1D":
            raise ValueError("horizon_mode=calendar_month currently requires freq=1D")
        if str(getattr(cfg, "schedule_mode", "daily")).lower() != "daily":
            raise ValueError("horizon_mode=calendar_month requires schedule_mode=daily.")
        train_window_length = getattr(cfg, "train_window_length", None)
        if train_window_length is None or int(train_window_length) <= 0:
            raise ValueError("calendar_month requires train_window_length > 0")
        now_time = cast(pd.Timestamp, pd.Timestamp(cfg.now_time))
        forecast_start = now_time.normalize() + pd.Timedelta(days=1)
        if forecast_start.day != 1:
            raise ValueError("calendar_month requires forecast_start at month start")
        horizon = int(forecast_start.days_in_month)
        train_rows = int(train_window_length) * n_per_day
        history_start = forecast_start - pd.Timedelta(days=int(cfg.history_length))
        theoretical_history = pd.DataFrame(
            {"time": pd.date_range(history_start, forecast_start, freq="1D", inclusive="left")}
        )
        n_windows = len(
            _build_calendar_month_folds(
                theoretical_history,
                train_window_len=train_rows,
            )
        )
        return horizon_mode, horizon, train_rows, n_windows, int(train_window_length)
    if horizon_mode != "fixed_steps":
        raise ValueError(f"unknown horizon_mode={horizon_mode}")
    window_len = int(cfg.window_length) * n_per_day
    horizon = int(cfg.predict_steps)
    train_rows = window_len - horizon
    n_windows = (int(cfg.history_length) * n_per_day - window_len) // horizon + 1
    return horizon_mode, horizon, train_rows, n_windows, int(cfg.window_length)


def check_model_yaml(f: str) -> tuple[Any, list[str]]:
    """加载单个模型 YAML，返回 (cfg, problems)。problems 空 = 通过。"""
    from model_ensemble.specs import EnsembleConfigSpec

    cfg: Any = load_yaml_config(f)
    if isinstance(cfg, ForecastConfigSpec):
        return _check_canonical_config(cfg)
    if isinstance(cfg, EnsembleConfigSpec):
        return _check_ensemble_config(cfg)
    raise TypeError(f"non-canonical model config rejected: {f}")


def _check_ensemble_config(
    cfg: Any,
) -> tuple[Any, list[str]]:
    """Ensemble 配置检查（v4 §5.2）：几何与概率合同。"""
    problems: list[str] = []
    mode = str(cfg.probabilistic.get("mode", "point"))
    if mode not in {"point", "quantile"}:
        problems.append(f"probabilistic mode 不支持: {mode}")
    if mode == "quantile":
        levels = tuple(float(value) for value in cfg.probabilistic.get("quantiles", ()))
        point_level = float(cfg.probabilistic.get("point_quantile", 0.5))
        if not levels or tuple(sorted(set(levels))) != levels:
            problems.append("quantiles 必须非空、唯一且递增")
        if point_level not in levels:
            problems.append("point_quantile 必须包含在 quantiles 中")
    horizon_mode = str(cfg.validation.get("horizon_mode", "fixed_steps"))
    if horizon_mode == "calendar_month" and cfg.oof.fold_count > 1:
        problems.append("calendar_month 场景 OOF 只允许 fold_count=1 (v4)")
    if cfg.method.name == "stacking" and mode == "quantile":
        problems.append("stacking 只支持 point 模式")
    return cfg, problems


def _check_canonical_config(
    cfg: ForecastConfigSpec,
) -> tuple[ForecastConfigSpec, list[str]]:
    problems: list[str] = []
    problems.extend(_check_canonical_nested_fields(cfg))
    problems.extend(_check_problem_data_contract(cfg))
    problems.extend(_check_feature_contract(cfg))
    problems.extend(_check_canonical_transform_values(cfg))
    try:
        if cfg.strategy is not None:
            cfg.strategy.resolve(cfg.problem.horizon)
        else:
            if cfg.ensemble is None:
                raise ValueError("canonical config requires strategy or ensemble")
            for member in cfg.ensemble.members:
                member.strategy.resolve(cfg.problem.horizon)
    except ValueError as exc:
        problems.append(f"strategy 配置错误: {exc}")

    mode = str(cfg.probabilistic.get("mode", "point"))
    if mode not in {"point", "quantile"}:
        problems.append(f"probabilistic mode 不支持: {mode}")
    if mode == "quantile":
        levels = tuple(float(value) for value in cfg.probabilistic.get("quantiles", ()))
        point_level = float(cfg.probabilistic.get("point_quantile", 0.5))
        if not levels or tuple(sorted(set(levels))) != levels:
            problems.append("quantiles 必须非空、唯一且递增")
        if point_level not in levels:
            problems.append("point_quantile 必须包含在 quantiles 中")

    target_lags = set(cfg.features.target_lags)
    observed_lags = set(cfg.features.observed_past_lags)
    if target_lags - set(cfg.problem.targets):
        problems.append("target_lags 包含未声明目标")
    if observed_lags - set(cfg.data.role_columns("observed_past")):
        problems.append("observed_past_lags 包含未声明列")
    return cfg, problems


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "config/**/*.yaml"
    files = sorted(
        f for f in glob.glob(pattern, recursive=True)
        if is_model_yaml(Path(f))
    )
    if not files:
        print(f"no files matched: {pattern}")
        return 1

    hard_failures: list[tuple[str, list[str]]] = []
    warnings: list[tuple[str, list[str]]] = []
    for f in files:
        cfg, problems = check_model_yaml(f)
        if isinstance(cfg, ForecastConfigSpec):
            identity = cfg.strategy.name.value
            lags = sorted(
                {
                    int(lag)
                    for values in cfg.features.target_lags.values()
                    for lag in values
                }
            )
            print(f"{f}")
            print(
                f"    schema=2 identity={identity} freq={cfg.problem.freq} "
                f"horizon={cfg.problem.horizon} targets={list(cfg.problem.targets)} "
                f"max_lag={max(lags or [0])} fingerprint={cfg.fingerprint()[:12]}"
            )
            if problems:
                hard_failures.append((f, problems))

            print("\n=== 汇总 ===")
    if hard_failures:
        print(f"发现 {len(hard_failures)} 个硬校验失败:")
        for f, probs in hard_failures:
            print(f"  {f}:")
            for p in probs:
                print(f"    - {p}")
    if warnings:
        print(f"{len(warnings)} 个提示（不崩溃但建议修）:")
        for f, probs in warnings:
            print(f"  {f}: {probs}")
    if not hard_failures and not warnings:
        print("全部配置通过校验")
    print(
        f"checked={len(files)} passed={len(files) - len(hard_failures)} "
        f"hard_failures={len(hard_failures)} warnings={len(warnings)}"
    )
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
