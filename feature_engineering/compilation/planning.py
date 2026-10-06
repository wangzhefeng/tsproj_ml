"""无数据编译规划：输出名称注册与特征依赖验证。"""
from forecasting_core.specs import ColumnRole
import hashlib
import json


def feature_plan(config):
    """按执行顺序登记名称；碰撞/未知引用在物化数据前失败。"""
    output = {}
    reserved = {*config.problem.series_id_cols, "target_time", "horizon_step"}

    def add(name, operation="input", inputs=(), parameters=None, input_kind="feature"):
        if name in reserved or name in output:
            raise ValueError(f"feature name collision or reserved name: {name!r}")
        rule = {"operation": operation, "inputs": list(inputs), "input_kind": input_kind, "parameters": parameters or {}}
        rule["identity"] = hashlib.sha256(json.dumps(rule, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        output[name] = rule

    def require(name):
        if name not in output:
            raise ValueError(f"unknown feature reference: {name!r}")
        return name

    for mapping in (config.features.target_lags, config.features.observed_past_lags):
        for column, lags in mapping.items():
            for lag in lags:
                add(f"{column}__lag_{lag}", "lag", (column,), {"lag": lag}, "history")
    for role in (ColumnRole.KNOWN_FUTURE, ColumnRole.STATIC):
        for source in config.data.sources:
            for column in source.columns:
                if column.role is role:
                    add(column.name, role.value, (column.name,), {"source": source.name}, "raw")
    for name in config.features.datetime_features:
        add(f"dt_{name}", "datetime", (), {"part": name})
    transformations = config.features.canonical_payload()["transformations"]
    direct = transformations.get("direct", {})
    horizon = direct.get("horizon_feature", {})
    if direct.get("layout") == "single_model_horizon" and horizon.get("enabled", True):
        name = horizon.get("name", "forecast_horizon_idx")
        add(name)
        if horizon.get("cyclical", False):
            add(name + "_sin")
            add(name + "_cos")
    advanced = transformations.get("advanced", {})
    history_columns = {column.name for source in config.data.sources for column in source.columns
                       if column.role in {ColumnRole.TARGET, ColumnRole.OBSERVED_PAST} and not column.categorical}
    for kind in ("rolling", "expanding", "difference", "percent_change", "time_since", "ewm", "fourier", "wavelet", "same_slot", "recent_state", "rolling_quantile", "lagged_rolling"):
        spec = advanced.get(kind)
        if spec is None:
            continue
        for column in spec["columns"]:
            if column not in history_columns:
                raise ValueError(f"{kind} requires numeric history column: {column!r}")
            if kind == "rolling_quantile":
                suffixes = [f"rolling_quantile_{float(level)}_{window}" for window in spec["windows"] for level in spec["quantiles"]]
            elif kind == "lagged_rolling":
                suffixes = [f"rolling_{stat}_{window}_offset_{offset}" for offset in spec["offsets"] for window in spec["windows"] for stat in spec["stats"]]
            elif kind == "rolling":
                suffixes = [f"rolling_{stat}_{window}" for window in spec["windows"] for stat in spec["stats"]]
            elif kind == "expanding":
                suffixes = [f"expanding_{stat}" for stat in spec["stats"]]
            elif kind in {"difference", "percent_change"}:
                prefix = "diff" if kind == "difference" else "pct_change"
                suffixes = [f"{prefix}_{period}" for period in spec["periods"]]
            elif kind == "time_since":
                suffixes = [f"time_since_{event}" for event in spec["events"]]
            elif kind == "ewm":
                suffixes = [f"ewm_{stat}_{float(half)}" for half in spec["halflives"] for stat in spec["stats"]]
            elif kind == "fourier":
                fft_names = [f"{part}_{k}" for k in range(1, spec.get("top_k", 5) + 1) for part in ("amp", "freq", "phase")]
                fft_names += ["centroid"] + [f"bandenergy_{k}" for k in range(1, len(spec.get("band_periods", ())) + 1)]
                suffixes = [f"fft_{name}_{window}" for window in spec["windows"] for name in fft_names]
            elif kind == "wavelet":
                level = spec.get("level", 3)
                parts = [f"a{level}"] + [f"d{k}" for k in range(level, 0, -1)]
                suffixes = [f"wavelet_energy_{part}_{window}" for window in spec["windows"] for part in parts]
            elif kind == "same_slot":
                suffixes = [f"slot_{stat}_{days}d" for days in sorted(spec["days"]) for stat in sorted(spec["stats"])]
            else:
                suffixes = [f"rs_{stat}_{window}" for window in sorted(spec["windows"]) for stat in sorted(spec["stats"])]
            for suffix in suffixes:
                add(f"{column}_{suffix}", kind, (column,), spec, "history")
    if "block_weather" in advanced:
        spec = advanced["block_weather"]
        for column in spec["columns"]:
            require(column)
            for stat in spec["stats"]:
                add(f"{column}_blk_{stat}", "block_weather", (column,), {"stat": stat})
    if "cyclical" in advanced:
        for name in advanced["cyclical"]["columns"]:
            column = require(name if name in output else f"dt_{name}")
            add(column + "_sin", "sin", (column,), advanced["cyclical"])
            add(column + "_cos", "cos", (column,), advanced["cyclical"])
    if "interaction" in advanced:
        spec = advanced["interaction"]
        for left, right in spec["column_pairs"]:
            require(left)
            require(right)
            for operation, label in (("add", "add"), ("subtract", "substract"), ("multiply", "multiply"), ("divide", "divide")):
                if operation in spec["operations"]:
                    add(f"{left}_{label}_{right}", operation, (left, right))
    if "polynomial" in advanced:
        spec = advanced["polynomial"]
        for column in spec["columns"]:
            require(column)
            for degree in range(2, spec.get("degree", 2) + 1):
                add(f"{column}_pow_{degree}", "power", (column,), {"degree": degree})
    for name, members in transformations.get("interactions", {}).items():
        for member in members:
            require(member)
        add(name, "multiply", tuple(members))
    return output


def planned_feature_names(config):
    return tuple(feature_plan(config))
