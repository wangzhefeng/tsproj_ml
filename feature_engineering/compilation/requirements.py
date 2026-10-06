"""特征历史需求的唯一入口；编排层只消费结果，不重新解释窗口。"""
from collections.abc import Mapping, Sequence
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.feature_transformations import STAT_MIN_SAMPLES
from feature_engineering.kernels.seasonal import normalize_seasonal_baseline_spec


def minimum_history_rows(config: ForecastConfigSpec) -> int:
    """Return visible rows required before one supervised origin is valid."""
    configured_lags = tuple(
        lag
        for lag_mapping in (
            config.features.target_lags,
            config.features.observed_past_lags,
        )
        for lags in lag_mapping.values()
        for lag in lags
    )
    required = max((*configured_lags, 1))
    baseline = config.features.transformations.get("seasonal_baseline")
    if baseline is not None:
        spec = normalize_seasonal_baseline_spec(baseline)
        required = max(required, spec["period"] * spec["days"])
    if config.strategy is None:
        raise ValueError("minimum_history_rows requires a single-model strategy")
    resolved_strategy = config.strategy.resolve(config.problem.horizon)
    direct = config.features.transformations.get("direct")
    if (
        configured_lags
        and not resolved_strategy.consumes_previous
        and isinstance(direct, Mapping)
        and direct.get("align_to_target") is False
    ):
        required = max(required, max(configured_lags) + 1)
    advanced = config.features.transformations.get("advanced", {})
    if not isinstance(advanced, Mapping):
        return required
    rolling = advanced.get("rolling")
    same_slot = advanced.get("same_slot")
    if isinstance(same_slot, Mapping):
        required = max(required, same_slot["period"] * max(same_slot["days"]))
    recent = advanced.get("recent_state")
    if isinstance(recent, Mapping):
        required = max(required, max(recent["windows"]))
    if isinstance(rolling, Mapping):
        windows = rolling.get("windows", ())
        if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
            required = max(required, *(int(value) for value in windows))
    for kind in ("fourier", "wavelet", "rolling_quantile"):
        spec = advanced.get(kind)
        if not isinstance(spec, Mapping):
            continue
        windows = spec.get("windows", ())
        if isinstance(windows, Sequence) and not isinstance(windows, (str, bytes)):
            required = max(required, *(int(value) for value in windows))
    for kind in ("difference", "percent_change"):
        spec = advanced.get(kind)
        if not isinstance(spec, Mapping):
            continue
        periods = spec.get("periods", ())
        if isinstance(periods, Sequence) and not isinstance(periods, (str, bytes)):
            required = max(required, *(int(value) + 1 for value in periods))
    lagged = advanced.get("lagged_rolling")
    if lagged is not None:
        required = max(required, max(lagged["windows"]) + max(lagged["offsets"]))
    for kind in ("expanding", "ewm"):
        spec = advanced.get(kind, {})
        required = max(required, *(STAT_MIN_SAMPLES.get(stat, 1) for stat in spec.get("stats", ())), 1)
    return required
