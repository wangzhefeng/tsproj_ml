# -*- coding: utf-8 -*-
"""确定性生成并校验三个 AIDC 15min 负荷场景的全因子配置矩阵。

默认模式只报告配置漂移；传 ``--write`` 才会重建预期 YAML。生成范围限于
route_A/route_B 的五个单路组和独立 ``add_ensemble``，以及 route_AB 的联合组。
本脚本不运行模型训练、回测或预测。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_yaml_config  # noqa: E402
from forecasting_core.specs import ForecastConfigSpec  # noqa: E402
from model_ensemble.specs import EnsembleConfigSpec  # noqa: E402

MODEL_TYPES = {
    "st": "st",
    "ridge": "ridge",
    "lasso": "lasso",
    "enet": "enet",
    "lgbm": "lightgbm",
    "xgb": "xgboost",
    "cab": "catboost",
    "rf": "randomforest",
    "histgb": "histgb",
}
STRATEGY_VARIANTS = (
    "direct-pointwise",
    "direct-pointwise-horizon",
    "direct",
    "recursive",
    "dirrec",
    "dirmo",
    "recmo",
    "dirrecmo",
    "mimo",
)
EXOGENOUS_VARIANTS = ("holiday", "weather", "holiday-weather")
DECOMPOSITION_VARIANTS = ("linear", "mstl96-672", "stl96")
ENSEMBLE_METHODS = ("averaging", "weighted", "linear_blending", "stacking")
LATIN_GROUPS = {
    "latin-a": ("st_recursive", "lgbm_mimo", "ridge_direct"),
    "latin-b": ("st_direct", "lgbm_recursive", "ridge_mimo"),
    "latin-c": ("st_mimo", "lgbm_direct", "ridge_recursive"),
}
ROUTES = ("route_A", "route_B")
SINGLE_ROUTE_GROUPS = (
    "baseline",
    "add_exogenous",
    "add_endogenous_cross_route",
    "add_endogenous_state",
    "add_decomposition",
)
DATETIME_FEATURES = (
    "hour",
    "minute",
    "day",
    "day_of_week",
    "week_of_year",
    "month",
    "days_in_month",
    "quarter",
    "day_of_year",
    "year",
)
WEATHER_COLUMNS = ("rt_tt2", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt")
STATE_COLUMNS = (
    "state_roll_1h_mean",
    "state_roll_1h_std",
    "state_roll_4h_mean",
    "state_roll_4h_std",
    "state_roll_24h_range",
    "state_diff_15min",
    "state_diff_1h",
    "state_diff_24h_pct",
    "state_robust_z_7d",
    "state_weekly_base_dev_pct",
)
SCENARIO_SPECS: dict[str, dict[str, Any]] = {
    "aidc_load_15min_daily": {
        "horizon": 96,
        "forecast_origin": "2026-07-31T23:45:00",
        "schedule_mode": "daily",
        "history_steps": 6336,
        "train_window_steps": 2784,
        "fold_count": 31,
        "stride_steps": 96,
        "target_lags": (96, 192, 288, 384, 480, 576, 672),
        "rolling_windows": (96, 192, 384, 672),
        "mo_chunk": 24,
        "oof_fold_count": 5,
        "weather_history": "weather_15min_20250101_20260731.csv",
        "weather_future": "weather_15min_future_proxy_20260801_20260831.csv",
    },
    "aidc_load_15min_rolling": {
        "horizon": 96,
        "forecast_origin": "2026-07-31T14:00:00",
        "schedule_mode": "intraday",
        "history_steps": 6336,
        "train_window_steps": 2784,
        "fold_count": 31,
        "stride_steps": 96,
        "target_lags": (96, 192, 288, 384, 480, 576, 672),
        "rolling_windows": (96, 192, 384, 672),
        "mo_chunk": 24,
        "oof_fold_count": 5,
        "weather_history": "weather_15min_20250101_20260731T1345.csv",
        "weather_future": "weather_15min_future_proxy_20260731T1400_20260831.csv",
    },
    "aidc_load_15min_short": {
        "horizon": 16,
        "forecast_origin": "2026-07-31T14:00:00",
        "schedule_mode": "intraday",
        "history_steps": 5072,
        "train_window_steps": 1424,
        "fold_count": 31,
        "stride_steps": 96,
        "target_lags": (1, 2, 3, 4, 8, 12, 16, 96, 192, 672),
        "rolling_windows": (16, 96, 672),
        "mo_chunk": 4,
        "oof_fold_count": 7,
        "weather_history": "weather_15min_20250101_20260731T1345.csv",
        "weather_future": "weather_15min_future_proxy_20260731T1400_20260831.csv",
    },
}
SCENARIOS = tuple(SCENARIO_SPECS)


def _column(name: str, role: str) -> dict[str, Any]:
    return {"name": name, "role": role, "categorical": False}


def _file_source(
    *,
    name: str,
    columns: list[dict[str, Any]],
    history_path: str,
    time_col: str = "time",
    availability: str = "source_time",
    provider: str | None = None,
    backtest_path: str | None = None,
    future_path: str | None = None,
    available_at_col: str | None = None,
) -> dict[str, Any]:
    source: dict[str, Any] = {
        "name": name,
        "source_type": "file",
        "columns": columns,
        "history_path": history_path,
        "time_col": time_col,
        "series_id_cols": [],
        "availability": availability,
    }
    if provider is not None:
        source["provider"] = provider
    if backtest_path is not None:
        source["backtest_path"] = backtest_path
    if future_path is not None:
        source["future_path"] = future_path
    if available_at_col is not None:
        source["available_at_col"] = available_at_col
    return source


def _target_path(scenario: str, route: str) -> str:
    route_name = "A" if route == "route_A" else "B"
    return (
        f"dataset/{scenario}/{route_name}_Loads_15min_mean_"
        "20251001_20260731.csv"
    )


def _joint_path(scenario: str) -> str:
    return (
        f"dataset/{scenario}/forecasting_data/"
        "AB_Loads_15min_mean_20251001_20260731.csv"
    )


def _single_target_source(scenario: str, route: str) -> list[dict[str, Any]]:
    return [
        _file_source(
            name="target_history",
            columns=[_column("value", "target")],
            history_path=_target_path(scenario, route),
        )
    ]


def _cross_route_sources(
    scenario: str,
    route: str,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    target = "A_load" if route == "route_A" else "B_load"
    peer = "B_load" if route == "route_A" else "A_load"
    path = _joint_path(scenario)
    sources = [
        _file_source(
            name="target_history",
            columns=[_column(target, "target")],
            history_path=path,
        ),
        _file_source(
            name="peer_history",
            columns=[_column(peer, "observed_past")],
            history_path=path,
            provider="persistence",
        ),
    ]
    return [target], [peer], sources


def _joint_target_sources(scenario: str) -> list[dict[str, Any]]:
    return [
        _file_source(
            name="target_history",
            columns=[
                _column("A_load", "target"),
                _column("B_load", "target"),
            ],
            history_path=_joint_path(scenario),
        )
    ]


def _state_source(scenario: str, route: str) -> dict[str, Any]:
    route_name = "A" if route == "route_A" else "B"
    return _file_source(
        name="load_state",
        columns=[_column(column, "observed_past") for column in STATE_COLUMNS],
        history_path=(
            f"dataset/{scenario}/load_state_features/"
            f"{route_name}_load_state_history.csv"
        ),
        provider="persistence",
    )


def _holiday_source() -> dict[str, Any]:
    return {
        "name": "chinese_holiday",
        "source_type": "generated",
        "generator": "chinese_holiday",
        "columns": [
            _column("is_holiday", "known_future"),
            _column("next_holiday_days", "known_future"),
        ],
        "time_col": "time",
        "series_id_cols": [],
        "availability": "generator_defined",
    }


def _weather_source(scenario: str) -> dict[str, Any]:
    spec = SCENARIO_SPECS[scenario]
    return _file_source(
        name="weather",
        columns=[_column(column, "known_future") for column in WEATHER_COLUMNS],
        history_path=f"dataset/{scenario}/{spec['weather_history']}",
        backtest_path=(
            f"dataset/{scenario}/"
            "weather_15min_backtest_proxy_20260101_20260731.csv"
        ),
        future_path=f"dataset/{scenario}/{spec['weather_future']}",
        time_col="ts",
        availability="column",
        available_at_col="available_at",
    )


def _strategy_spec(
    scenario: str,
    variant: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if variant not in STRATEGY_VARIANTS:
        raise ValueError(f"unknown strategy variant: {variant}")
    if variant.startswith("direct-pointwise"):
        return (
            {"name": "direct"},
            {
                "layout": "single_model_horizon",
                "align_to_target": True,
                "horizon_feature": {
                    "name": "forecast_horizon_idx",
                    "cyclical": variant == "direct-pointwise-horizon",
                },
            },
        )
    strategy_name = variant
    strategy: dict[str, Any] = {"name": strategy_name}
    if strategy_name in {"dirmo", "recmo", "dirrecmo"}:
        strategy["output_chunk_length"] = SCENARIO_SPECS[scenario]["mo_chunk"]
    direct = None
    if strategy_name in {"direct", "dirmo", "mimo"}:
        direct = {"layout": "independent_models", "align_to_target": False}
    return strategy, direct


def _decomposition_spec(variant: str) -> dict[str, Any]:
    if variant == "linear":
        return {
            "method": "linear",
            "trend_degree": 1,
            "trend_forecast": "polynomial",
            "damping": 0.98,
            "trend_lookback": 28,
        }
    if variant == "stl96":
        return {
            "method": "stl",
            "periods": [96],
            "robust": True,
            "trend_forecast": "polynomial",
            "damping": 0.98,
        }
    if variant == "mstl96-672":
        return {
            "method": "mstl",
            "periods": [96, 672],
            "robust": True,
            "trend_forecast": "polynomial",
            "damping": 0.98,
        }
    raise ValueError(f"unknown decomposition variant: {variant}")


def _target_lags(scenario: str, strategy_variant: str) -> list[int]:
    spec = SCENARIO_SPECS[scenario]
    lags = list(spec["target_lags"])
    if scenario == "aidc_load_15min_short" and strategy_variant.startswith(
        "direct-pointwise"
    ):
        return [lag for lag in lags if lag >= spec["horizon"]]
    return lags


def _features(
    *,
    scenario: str,
    strategy_variant: str,
    model: str,
    targets: list[str],
    observed_columns: list[str] | None = None,
    decomposition: str | None = None,
) -> dict[str, Any]:
    spec = SCENARIO_SPECS[scenario]
    target_lags = _target_lags(scenario, strategy_variant)
    strategy, direct = _strategy_spec(scenario, strategy_variant)
    del strategy
    transformations: dict[str, Any] = {}
    if direct is not None:
        transformations["direct"] = direct
    if model in {"ridge", "lasso", "enet"}:
        transformations["feature_scaling"] = {
            "method": "minmax",
            "grouped": False,
            "encode_categorical": False,
        }
    if decomposition is not None:
        transformations["target"] = {
            "calendar_normalization": {"method": "none"},
            "decomposition": _decomposition_spec(decomposition),
            "scaling": {"method": "none", "inverse": False},
        }
    transformations["advanced"] = {
        "rolling": {
            "columns": list(targets),
            "windows": list(spec["rolling_windows"]),
            "stats": ["mean", "std", "min", "max"],
        },
        "expanding": {
            "columns": list(targets),
            "stats": ["mean", "std"],
        },
    }
    observed = observed_columns or []
    return {
        "target_lags": {target: list(target_lags) for target in targets},
        "observed_past_lags": {
            column: list(spec["target_lags"]) for column in observed
        },
        "datetime_features": list(DATETIME_FEATURES),
        "transformations": transformations,
    }


def _validation(scenario: str) -> dict[str, Any]:
    spec = SCENARIO_SPECS[scenario]
    return {
        "forecast_origin": spec["forecast_origin"],
        "schedule_mode": spec["schedule_mode"],
        "horizon_mode": "fixed_steps",
        "history_steps": spec["history_steps"],
        "train_window_steps": spec["train_window_steps"],
        "fold_count": spec["fold_count"],
        "stride_steps": spec["stride_steps"],
    }


def _payload(
    *,
    scenario: str,
    output_route: str,
    group: str,
    model: str,
    strategy_variant: str,
    targets: list[str],
    sources: list[dict[str, Any]],
    observed_columns: list[str] | None = None,
    decomposition: str | None = None,
) -> dict[str, Any]:
    strategy, _ = _strategy_spec(scenario, strategy_variant)
    validation = _validation(scenario)
    is_lgbm = model == "lgbm"
    is_profiled_baseline = group == "baseline" and is_lgbm
    source_names = {str(source.get("name", "")) for source in sources}
    if is_profiled_baseline and strategy_variant in {
        "recursive",
        "direct-pointwise",
        "direct-pointwise-horizon",
    }:
        validation["performance"] = {
            "window_parallel_workers": 4,
            "model_thread_count": 2,
        }
    elif is_profiled_baseline and (
        strategy_variant in {"direct", "mimo"}
        or (
            scenario == "aidc_load_15min_short"
            and strategy_variant in {"dirrec", "dirmo", "dirrecmo"}
        )
        or (
            scenario == "aidc_load_15min_daily"
            and output_route == "route_A"
            and strategy_variant in {"dirrec", "dirmo", "dirrecmo", "recmo"}
        )
    ):
        validation["performance"] = {
            "window_parallel_workers": 1,
            "multi_output_n_jobs": 8,
            "model_thread_count": 1,
        }
    elif (
        is_lgbm
        and scenario == "aidc_load_15min_daily"
        and output_route == "route_A"
        and group == "add_endogenous_cross_route"
        and strategy_variant == "recursive"
    ):
        validation["performance"] = {
            "window_parallel_workers": 2,
            "model_thread_count": 4,
        }
    elif is_lgbm and (
        (
            scenario == "aidc_load_15min_daily"
            and output_route == "route_A"
            and group == "add_exogenous"
            and strategy_variant == "direct"
            and {"chinese_holiday", "weather"}.issubset(source_names)
        )
        or (
            scenario == "aidc_load_15min_daily"
            and output_route == "route_A"
            and group == "add_endogenous_state"
            and strategy_variant == "direct"
        )
        or (
            scenario == "aidc_load_15min_daily"
            and output_route == "route_AB"
            and group == "add_endogenous_joint"
            and strategy_variant == "direct"
        )
    ):
        validation["performance"] = {
            "window_parallel_workers": 1,
            "multi_output_n_jobs": 8,
            "model_thread_count": 1,
        }
    return {
        "schema_version": 2,
        "problem": {
            "time_col": "time",
            "freq": "15min",
            "horizon": SCENARIO_SPECS[scenario]["horizon"],
            "targets": list(targets),
            "training_scope": "local",
            "series_id_cols": [],
        },
        "data": {"sources": sources},
        "features": _features(
            scenario=scenario,
            strategy_variant=strategy_variant,
            model=model,
            targets=targets,
            observed_columns=observed_columns,
            decomposition=decomposition,
        ),
        "strategy": strategy,
        "estimator": {
            "model_type": MODEL_TYPES[model],
            "target_adapter": "independent",
            "params": {},
        },
        "probabilistic": {"mode": "point"},
        "validation": validation,
        "output": {
            "scenario_subpath": f"{scenario}/{output_route}/{group}",
            "results_root": "results",
            "directories": {
                "checkpoints": "./results/pretrained_models/",
                "tests": "./results/results_test/",
                "forecast": "./results/results_forecast/",
            },
        },
    }


def _ensemble_payload(
    *,
    scenario: str,
    route: str,
    latin_group: str,
    method: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    spec = SCENARIO_SPECS[scenario]
    return {
        "schema_version": 2,
        "problem": copy.deepcopy(baseline["problem"]),
        "data": copy.deepcopy(baseline["data"]),
        "probabilistic": copy.deepcopy(baseline["probabilistic"]),
        "ensemble": {
            "members": [
                {
                    "name": member,
                    "config_ref": f"../baseline/{member}.yaml",
                }
                for member in LATIN_GROUPS[latin_group]
            ],
            "oof": {
                "train_window_steps": spec["train_window_steps"],
                "fold_count": spec["oof_fold_count"],
                "stride_steps": spec["stride_steps"],
            },
            "method": {"name": method},
        },
        "validation": copy.deepcopy(baseline["validation"]),
        "output": {
            "scenario_subpath": f"{scenario}/{route}/add_ensemble",
            "results_root": "results",
            "directories": {
                "checkpoints": "./results/pretrained_models/",
                "tests": "./results/results_test/",
                "forecast": "./results/results_forecast/",
            },
        },
    }


def build_expected_configs(scenario: str) -> dict[Path, dict[str, Any]]:
    """构造一个场景的 1,539 单模型 + 24 Ensemble 映射。"""
    if scenario not in SCENARIO_SPECS:
        raise ValueError(f"unknown scenario: {scenario}")
    root = ROOT / "config" / scenario
    configs: dict[Path, dict[str, Any]] = {}

    for route in ROUTES:
        for model in MODEL_TYPES:
            for strategy_variant in STRATEGY_VARIANTS:
                filename = f"{model}_{strategy_variant}.yaml"
                configs[root / route / "baseline" / filename] = _payload(
                    scenario=scenario,
                    output_route=route,
                    group="baseline",
                    model=model,
                    strategy_variant=strategy_variant,
                    targets=["value"],
                    sources=_single_target_source(scenario, route),
                )

                for feature_variant in EXOGENOUS_VARIANTS:
                    sources = _single_target_source(scenario, route)
                    if feature_variant in {"holiday", "holiday-weather"}:
                        sources.append(_holiday_source())
                    if feature_variant in {"weather", "holiday-weather"}:
                        sources.append(_weather_source(scenario))
                    exogenous_name = (
                        f"{model}_{strategy_variant}_{feature_variant}.yaml"
                    )
                    configs[root / route / "add_exogenous" / exogenous_name] = (
                        _payload(
                            scenario=scenario,
                            output_route=route,
                            group="add_exogenous",
                            model=model,
                            strategy_variant=strategy_variant,
                            targets=["value"],
                            sources=sources,
                        )
                    )

                targets, peers, sources = _cross_route_sources(scenario, route)
                configs[
                    root / route / "add_endogenous_cross_route" / filename
                ] = _payload(
                    scenario=scenario,
                    output_route=route,
                    group="add_endogenous_cross_route",
                    model=model,
                    strategy_variant=strategy_variant,
                    targets=targets,
                    sources=sources,
                    observed_columns=peers,
                )

                state_sources = _single_target_source(scenario, route)
                state_sources.append(_state_source(scenario, route))
                configs[root / route / "add_endogenous_state" / filename] = (
                    _payload(
                        scenario=scenario,
                        output_route=route,
                        group="add_endogenous_state",
                        model=model,
                        strategy_variant=strategy_variant,
                        targets=["value"],
                        sources=state_sources,
                        observed_columns=list(STATE_COLUMNS),
                    )
                )

                for decomposition in DECOMPOSITION_VARIANTS:
                    decomp_name = (
                        f"{model}_{strategy_variant}_decomp-{decomposition}.yaml"
                    )
                    configs[
                        root / route / "add_decomposition" / decomp_name
                    ] = _payload(
                        scenario=scenario,
                        output_route=route,
                        group="add_decomposition",
                        model=model,
                        strategy_variant=strategy_variant,
                        targets=["value"],
                        sources=_single_target_source(scenario, route),
                        decomposition=decomposition,
                    )

        baseline = configs[root / route / "baseline/st_recursive.yaml"]
        for latin_group in LATIN_GROUPS:
            for method in ENSEMBLE_METHODS:
                method_slug = method.replace("_", "-")
                filename = f"ensemble_{latin_group}_{method_slug}.yaml"
                configs[root / route / "add_ensemble" / filename] = (
                    _ensemble_payload(
                        scenario=scenario,
                        route=route,
                        latin_group=latin_group,
                        method=method,
                        baseline=baseline,
                    )
                )

    for model in MODEL_TYPES:
        for strategy_variant in STRATEGY_VARIANTS:
            filename = f"{model}_{strategy_variant}.yaml"
            configs[
                root / "route_AB" / "add_endogenous_joint" / filename
            ] = _payload(
                scenario=scenario,
                output_route="route_AB",
                group="add_endogenous_joint",
                model=model,
                strategy_variant=strategy_variant,
                targets=["A_load", "B_load"],
                sources=_joint_target_sources(scenario),
            )

    if len(configs) != 1563:
        raise AssertionError(
            f"{scenario}: expected 1563 configs, built {len(configs)}"
        )
    return configs


def _all_expected_configs() -> dict[Path, dict[str, Any]]:
    configs: dict[Path, dict[str, Any]] = {}
    for scenario in SCENARIOS:
        configs.update(build_expected_configs(scenario))
    if len(configs) != 4689:
        raise AssertionError(f"expected 4689 configs, built {len(configs)}")
    return configs


def _actual_model_paths() -> set[Path]:
    paths: set[Path] = set()
    for scenario in SCENARIOS:
        scenario_root = ROOT / "config" / scenario
        for route in (*ROUTES, "route_AB"):
            route_root = scenario_root / route
            if route_root.exists():
                paths.update(route_root.rglob("*.yaml"))
    return paths


def _load_raw(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: YAML root must be a mapping")
    return payload


def build_write_plan() -> dict[str, Any]:
    expected = _all_expected_configs()
    actual = _actual_model_paths()
    expected_paths = set(expected)
    common = actual & expected_paths
    rewrite = sorted(path for path in common if _load_raw(path) != expected[path])
    unchanged = sorted(common - set(rewrite))
    return {
        "expected": expected,
        "create": sorted(expected_paths - actual),
        "rewrite": rewrite,
        "delete": sorted(actual - expected_paths),
        "unchanged": unchanged,
    }


def _dump(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )


def apply_write_plan(plan: dict[str, Any]) -> None:
    expected: dict[Path, dict[str, Any]] = plan["expected"]
    for path in plan["delete"]:
        path.unlink()
    for path in [*plan["create"], *plan["rewrite"]]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(expected[path]), encoding="utf-8")


def validate_matrix() -> dict[str, int]:
    expected = _all_expected_configs()
    actual = _actual_model_paths()
    if actual != set(expected):
        missing = sorted(str(path.relative_to(ROOT)) for path in set(expected) - actual)
        unexpected = sorted(str(path.relative_to(ROOT)) for path in actual - set(expected))
        raise AssertionError(
            f"matrix path drift: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    forecast_count = 0
    ensemble_count = 0
    for path, payload in expected.items():
        if _load_raw(path) != payload:
            raise AssertionError(f"payload drift: {path.relative_to(ROOT)}")
        loaded = load_yaml_config(path)
        expected_type = (
            EnsembleConfigSpec if "ensemble" in payload else ForecastConfigSpec
        )
        if not isinstance(loaded, expected_type):
            raise TypeError(
                f"{path.relative_to(ROOT)}: expected {expected_type.__name__}, "
                f"got {type(loaded).__name__}"
            )
        if isinstance(loaded, EnsembleConfigSpec):
            ensemble_count += 1
        else:
            forecast_count += 1

    return {
        "scenarios": len(SCENARIOS),
        "per_scenario": 1563,
        "forecast_configs": forecast_count,
        "ensemble_configs": ensemble_count,
    }


def _summary(plan: dict[str, Any]) -> dict[str, int]:
    return {
        "expected": len(plan["expected"]),
        "create": len(plan["create"]),
        "rewrite": len(plan["rewrite"]),
        "delete": len(plan["delete"]),
        "unchanged": len(plan["unchanged"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="重建 4,617 份单模型与 72 份 AIDC Ensemble YAML",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出计划/验证摘要",
    )
    args = parser.parse_args()

    plan = build_write_plan()
    summary = _summary(plan)
    if not args.write and any(summary[key] for key in ("create", "rewrite", "delete")):
        output = {"status": "drift", **summary}
        print(json.dumps(output, ensure_ascii=False) if args.json else output)
        return 1

    if args.write:
        apply_write_plan(plan)
    validation = validate_matrix()
    output = {"status": "ok", **summary, **validation}
    print(json.dumps(output, ensure_ascii=False) if args.json else output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
