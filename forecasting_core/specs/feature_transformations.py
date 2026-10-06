"""特征变换语法合同；无数据、算法库或运行编排依赖。"""
from collections.abc import Mapping, Sequence
import math


HISTORY_STATS = frozenset({"mean", "std", "min", "max", "median", "skew", "kurt", "entropy", "max_diff", "min_diff"})
STAT_MIN_SAMPLES = {"std": 2, "skew": 3, "kurt": 4, "max_diff": 2, "min_diff": 2}
ADVANCED_FIELDS = {
    "rolling_quantile": ({"columns", "windows", "quantiles"}, set()),
    "lagged_rolling": ({"columns", "windows", "offsets", "stats"}, set()),
    "rolling": ({"columns", "windows", "stats"}, set()),
    "expanding": ({"columns", "stats"}, set()),
    "difference": ({"columns", "periods"}, set()),
    "percent_change": ({"columns", "periods"}, set()),
    "time_since": ({"columns", "events"}, set()),
    "ewm": ({"columns", "halflives", "stats"}, set()),
    "fourier": ({"columns", "windows"}, {"top_k", "band_periods"}),
    "wavelet": ({"columns", "windows"}, {"wavelet", "level"}),
    "same_slot": ({"columns", "period", "days", "stats"}, set()),
    "recent_state": ({"columns", "windows", "stats"}, set()),
    "block_weather": ({"columns", "stats"}, set()),
    "cyclical": ({"columns", "period"}, set()),
    "interaction": ({"column_pairs", "operations"}, set()),
    "polynomial": ({"columns"}, {"degree"}),
}


def require_fields(value, required, optional, path):
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    unknown = set(value) - required - optional
    if unknown:
        raise ValueError(f"unknown {path} keys: {sorted(unknown)}")
    missing = required - set(value)
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")


def names(value, path, *, allow_empty=False):
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence of strings")
    if not value and not allow_empty:
        raise ValueError(f"{path} must not be empty")
    if any(not isinstance(v, str) or not v or v.strip() != v for v in value):
        raise ValueError(f"{path} requires nonblank string entries")
    if len(set(value)) != len(value):
        raise ValueError(f"{path} entries must be unique")
    return tuple(value)


def positive(value, path, *, integer=True, minimum: float = 1):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise TypeError(f"{path} must be a {'positive integer' if integer else 'positive number'}")
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{path} must be finite and >= {minimum}")


def validate_transformations(value):
    require_fields(value, set(), {"advanced", "direct", "target", "feature_scaling", "datetime_categorical", "interactions", "seasonal_baseline"}, "transformations")
    advanced = value.get("advanced", {})
    require_fields(advanced, set(), set(ADVANCED_FIELDS), "advanced")
    for kind, spec in advanced.items():
        required, optional = ADVANCED_FIELDS[kind]
        path = f"advanced.{kind}"
        require_fields(spec, required, optional, path)
        if "columns" in spec:
            names(spec["columns"], path + ".columns")
        for field in ("windows", "periods", "days", "halflives"):
            if field not in spec:
                continue
            sequence = spec[field]
            if isinstance(sequence, (str, bytes)) or not isinstance(sequence, Sequence) or not sequence:
                raise ValueError(f"{path}.{field} must be a nonempty sequence")
            for entry in sequence:
                positive(entry, f"{path}.{field}", integer=field != "halflives", minimum=0.0 if field == "halflives" else 1)
                if entry == 0:
                    raise ValueError(f"{path}.{field} must be positive")
            if len(set(sequence)) != len(sequence):
                raise ValueError(f"{path}.{field} entries must be unique")
        if "stats" in spec:
            allowed = {"ewm": {"mean", "std"}, "same_slot": {"mean", "std"},
                       "recent_state": {"level", "mean", "std", "diff", "slope"},
                       "block_weather": {"mean", "min", "max"}}.get(kind, HISTORY_STATS)
            stats = names(spec["stats"], path + ".stats")
            if kind in {"rolling", "lagged_rolling"}:
                minimum = max(STAT_MIN_SAMPLES.get(stat, 1) for stat in stats)
                if min(spec["windows"]) < minimum:
                    raise ValueError(f"rolling window requires at least {minimum} samples for configured statistics")
            if set(stats) - allowed:
                raise ValueError(f"unsupported {kind}.stats entries {sorted(set(stats) - allowed)}; expected subset of {sorted(allowed)}")
        if kind == "time_since" and set(names(spec["events"], path + ".events")) - {"peak", "trough"}:
            raise ValueError("unsupported time_since.events")
        if kind in {"same_slot", "cyclical"}:
            positive(spec["period"], path + ".period", integer=kind == "same_slot", minimum=0.0)
            if spec["period"] == 0:
                raise ValueError(f"{path}.period must be positive")
        if kind == "rolling_quantile":
            levels = spec["quantiles"]
            if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence) or not levels:
                raise ValueError("rolling_quantile.quantiles requires a nonempty sequence")
            if any(isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1 for q in levels):
                raise ValueError("rolling_quantile.quantiles must be finite in [0,1]")
            if len(set(levels)) != len(levels):
                raise ValueError("rolling_quantile.quantiles must be unique")
        if kind == "lagged_rolling":
            offsets = spec["offsets"]
            if isinstance(offsets, (str, bytes)) or not isinstance(offsets, Sequence) or not offsets:
                raise ValueError("lagged_rolling.offsets requires a nonempty sequence")
            for offset in offsets:
                positive(offset, "lagged_rolling.offsets", minimum=0)
            if len(set(offsets)) != len(offsets):
                raise ValueError("lagged_rolling.offsets must be unique")
        if kind == "fourier":
            positive(spec.get("top_k", 5), path + ".top_k")
        if kind == "wavelet":
            positive(spec.get("level", 3), path + ".level")
            names([spec.get("wavelet", "db4")], path + ".wavelet")
        if kind == "polynomial":
            positive(spec.get("degree", 2), path + ".degree", minimum=2)
        if kind == "interaction":
            operations = names(spec["operations"], path + ".operations")
            if set(operations) - {"add", "subtract", "multiply", "divide"}:
                raise ValueError("unsupported interaction.operations")
            pairs = spec["column_pairs"]
            if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence) or not pairs:
                raise ValueError("interaction.column_pairs must be nonempty")
            for pair in pairs:
                # a*a 是合法交互，故不要求 pair 内名称唯一。
                if isinstance(pair, (str, bytes)) or not isinstance(pair, Sequence) or len(pair) != 2:
                    raise ValueError("interaction.column_pairs requires pairs")
                for name in pair:
                    names([name], "interaction.column_pairs")
    if "feature_scaling" in value:
        require_fields(value["feature_scaling"], set(), {"method", "encode_categorical"}, "feature_scaling")
    if "target" in value:
        target = value["target"]
        require_fields(target, set(), {"calendar_normalization", "decomposition", "scaling"}, "target")
        if "scaling" in target:
            require_fields(target["scaling"], set(), {"method"}, "target.scaling")
    if "direct" in value:
        direct = value["direct"]
        require_fields(direct, {"layout"}, {"align_to_target", "horizon_feature"}, "direct")
        if direct["layout"] not in {"independent_models", "single_model_horizon"}:
            raise ValueError("unsupported direct.layout")
        if "align_to_target" in direct and not isinstance(direct["align_to_target"], bool):
            raise TypeError("direct.align_to_target must be bool")
        if "horizon_feature" in direct:
            horizon = direct["horizon_feature"]
            require_fields(horizon, set(), {"name", "enabled", "cyclical"}, "direct.horizon_feature")
            for flag in ("enabled", "cyclical"):
                if flag in horizon and not isinstance(horizon[flag], bool):
                    raise TypeError(f"direct.horizon_feature.{flag} must be bool")
            names([horizon.get("name", "forecast_horizon_idx")], "direct.horizon_feature.name")
    if "datetime_categorical" in value:
        names(value["datetime_categorical"], "datetime_categorical", allow_empty=True)
    if "interactions" in value:
        interactions = value["interactions"]
        if not isinstance(interactions, Mapping):
            raise TypeError("interactions must be a mapping")
        for name, members in interactions.items():
            names([name], "interactions.name")
            if isinstance(members, (str, bytes)) or not isinstance(members, Sequence) or len(members) < 2:
                raise ValueError("interactions requires at least two feature names")
            for member in members:
                names([member], "interactions.members")
