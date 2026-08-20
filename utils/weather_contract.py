# -*- coding: utf-8 -*-
"""气象外生数据的信息集与时间覆盖契约。"""
from typing import Any, Iterable

import pandas as pd


_ALLOWED_SOURCES = {"", "actual", "forecast", "proxy", "mixed"}
_ALLOWED_MODES = {"forecast", "nowcast", "oracle"}


def validate_weather_information_contract(args: Any) -> None:
    """验证严格模式下历史、回测和未来气象来源是否合法。"""
    if not bool(getattr(args, "enable_weather_features", False)):
        return
    if not bool(getattr(args, "strict_weather_information_set", False)):
        return

    mode = str(getattr(args, "forecast_mode", "forecast") or "forecast").lower()
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported forecast_mode='{mode}'; expected one of {sorted(_ALLOWED_MODES)}.")

    sources = {
        "history": str(getattr(args, "weather_history_source", "") or "").lower(),
        "backtest": str(getattr(args, "weather_backtest_source", "") or "").lower(),
        "future": str(getattr(args, "weather_future_source", "") or "").lower(),
    }
    unknown = {name: value for name, value in sources.items() if value not in _ALLOWED_SOURCES}
    if unknown:
        raise ValueError(f"Unsupported weather source declaration(s): {unknown}.")
    if sources["history"] != "actual":
        raise ValueError("Strict weather contract requires weather_history_source='actual'.")

    if bool(getattr(args, "is_testing", False)):
        if not getattr(args, "weather_backtest_path", None):
            raise ValueError("Strict weather contract requires weather_backtest_path when is_testing=true.")
        if sources["backtest"] in {"", "actual", "mixed"}:
            raise ValueError(
                "Strict forecast backtest weather source must be 'forecast' or 'proxy'; "
                f"got backtest weather source '{sources['backtest']}'."
            )

    if bool(getattr(args, "is_forecasting", False)):
        if not getattr(args, "weather_future_path", None):
            raise ValueError("Strict weather contract requires weather_future_path when is_forecasting=true.")
        if mode == "forecast" and sources["future"] in {"", "actual", "mixed"}:
            raise ValueError(
                "Forecast mode future weather source must be 'forecast' or 'proxy'; "
                f"got future weather source '{sources['future']}'."
            )


def validate_weather_coverage(
    df: pd.DataFrame,
    expected_times: Iterable,
    ts_col: str,
    label: str,
) -> None:
    """要求外生文件精确覆盖全部目标时间戳，缺失时 fail-fast。"""
    if df is None or df.empty:
        raise ValueError(f"{label} is empty; expected weather coverage for target timestamps.")
    if not ts_col or ts_col not in df.columns:
        raise ValueError(f"{label} missing timestamp column '{ts_col}'.")
    available = set(pd.to_datetime(df[ts_col]))
    expected = pd.DatetimeIndex(pd.to_datetime(list(expected_times)))
    missing = [ts for ts in expected if ts not in available]
    if missing:
        preview = ", ".join(str(ts) for ts in missing[:10])
        raise ValueError(f"{label} missing {len(missing)} target timestamp(s): {preview}")


def validate_weather_availability(
    df: pd.DataFrame,
    ts_col: str,
    label: str,
    forecast_origin=None,
    require_before_target_month: bool = False,
) -> None:
    """验证每条天气记录在预测原点前确实可获得。"""
    if df is None or df.empty:
        raise ValueError(f"{label} is empty; availability cannot be validated.")
    if ts_col not in df.columns:
        raise ValueError(f"{label} missing timestamp column '{ts_col}'.")
    if "available_at" not in df.columns:
        raise ValueError(f"{label} missing required provenance column 'available_at'.")

    target_ts = pd.Series(pd.to_datetime(df[ts_col]))
    available_at = pd.Series(pd.to_datetime(df["available_at"]))
    if available_at.isna().any():
        raise ValueError(f"{label} contains invalid available_at timestamp(s).")

    if forecast_origin is not None:
        origin = pd.Timestamp(forecast_origin)
        bad = available_at > origin
        if bad.any():
            bad_positions = [i for i, flag in enumerate(bad.tolist()) if flag]
            preview = ", ".join(str(available_at.iloc[i]) for i in bad_positions[:10])
            raise ValueError(
                f"{label} contains data available after forecast origin {origin}: {preview}"
            )

    if require_before_target_month:
        target_month_start = target_ts.dt.to_period("M").dt.start_time
        bad = available_at >= target_month_start
        if bad.any():
            bad_positions = [i for i, flag in enumerate(bad.tolist()) if flag]
            pairs = [
                f"target={target_ts.iloc[i]}, available_at={available_at.iloc[i]}"
                for i in bad_positions[:10]
            ]
            raise ValueError(
                f"{label} must be available before target month starts: " + "; ".join(pairs)
            )
