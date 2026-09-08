"""红太阳全年严格 as-of 回测；复用 canonical compiler/trainer/executor。"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from config.config_loader import load_yaml_config
from data_loading import BUILTIN_GENERATORS, SourceRegistry
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.strategy import ForecastStrategySpec
from model_pipeline.supervised_design import SupervisedDesignBuilder, minimum_history_rows
from model_training.estimators import make_model_factory, resolve_model_capabilities
from model_training.trainer import CanonicalTrainer
from model_training.sample_weight import temporal_sample_weight
from model_training.strategies import get_standard_executor, target_plan_for_config
from model_performance.checkpoints import implementation_fingerprint
from prepare import validate_frame
from annual_reporting import write_annual_results
from cold_start import DEFAULT_RECIPE, load_recipe, validate_recipe, calendar_baseline


def schedule(freq: str) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]]:
    if freq not in ("15min", "1D"):
        raise ValueError("only 15min and 1D are supported")
    step = pd.tseries.frequencies.to_offset(freq)
    starts = pd.date_range("2025-02-01", "2025-12-31", freq="MS" if freq == "1D" else "1D")
    result = []
    for start in starts:
        stop = start + (pd.offsets.MonthBegin(1) if freq == "1D" else pd.Timedelta(days=1))
        history_start = (pd.Timestamp("2025-01-01") if freq == "1D" and start.month > 2
                         else start - pd.Timedelta(days=30))
        result.append((history_start, start - step, pd.date_range(start, stop, freq=freq, inclusive="left")))
    return result


def window_config(config: ForecastConfigSpec, horizon: int, strategy: str) -> ForecastConfigSpec:
    # 年度调度拥有窗口；动态对象仍严格解析为 canonical fixed-step 配置。
    transformations = config.features.canonical_payload()["transformations"]
    if strategy != "direct":
        transformations.pop("direct", None)
    validation = {"horizon_mode": "fixed_steps", "history_steps": 35040,
                  "train_window_steps": 2, "fold_count": 1, "stride_steps": horizon}
    if config.validation.get('training') is not None:
        validation['training'] = config.validation['training']
    return replace(config, problem=replace(config.problem, horizon=horizon),
                   features=replace(config.features, transformations=transformations),
                   strategy=ForecastStrategySpec(name=strategy),
                   validation=validation)


def forecast_window(config: ForecastConfigSpec, actual: pd.DataFrame, window, *, recipe: dict | None = None) -> tuple[pd.DataFrame, dict]:
    started = perf_counter()
    history_start, origin, times = window
    if config.strategy is None:
        raise ValueError("single model strategy required")
    if config.strategy.name.value not in ('direct', 'mimo'):
        raise ValueError('hongtaiyang supports direct layouts and mimo only')
    recipe = load_recipe() if recipe is None else validate_recipe(recipe)
    history = actual.loc[(actual.time >= history_start) & (actual.time <= origin)].copy()
    expected = pd.date_range(history_start, origin, freq=config.problem.freq)
    if not pd.DatetimeIndex(history.time).equals(expected) or not np.isfinite(history.value).all():
        raise ValueError("incomplete bounded training history")
    short = config.problem.freq == "1D" and times[0] == pd.Timestamp("2025-02-01")
    if short and recipe['cold_start']['method'] == 'calendar_baseline':
        values, evidence = calendar_baseline(history, times, recipe['cold_start'])
        audit = {'history_start': history_start.isoformat(), 'origin': origin.isoformat(),
                 'forecast_start': times[0].isoformat(), 'forecast_end': times[-1].isoformat(),
                 'history_rows': len(history), 'training_samples': 0, 'training_horizon': 0,
                 'forecast_horizon': len(times), 'training_label_end_max': origin.isoformat(),
                 'effective_strategy': 'calendar_baseline', 'model_count': 0,
                 'direct_layout': None, 'feature_schema': [], 'cold_start': evidence,
                 'sample_weight': None, 'design_seconds': 0., 'fit_seconds': 0.,
                 'predict_seconds': perf_counter() - started, 'seconds': perf_counter() - started}
        return pd.DataFrame({'time': times, 'y_pred': values}), audit
    strategy = 'direct' if short else config.strategy.name.value
    if short:
        transformations = config.features.canonical_payload()['transformations']
        transformations['direct'] = {'layout': 'single_model_horizon', 'align_to_target': False,
                                     'horizon_feature': {'enabled': False, 'name': 'forecast_horizon_idx', 'cyclical': False}}
        config = replace(config, features=replace(config.features, datetime_features=('day_of_week',),
                                                   transformations=transformations))
    training = window_config(config, 1 if short else len(times), strategy)
    prediction_config = window_config(config, len(times), strategy)
    # 物理 reader 也只提供该窗历史；即使上层请求错误，也无法读取未来真实目标。
    target_path = (ROOT / config.data.sources[0].history_path).resolve()

    def read_history(path, **kwargs):
        if Path(path).resolve() != target_path:
            raise ValueError("annual runner accepts exactly one target file")
        return history.copy()

    registry = SourceRegistry(training.data, ROOT, reader=read_history, generators=BUILTIN_GENERATORS)
    builder = SupervisedDesignBuilder(training, registry, history_start=history_start)
    warmup = minimum_history_rows(training)
    origins = tuple(expected[warmup - 1:len(expected) - training.problem.horizon])
    if len(origins) < 2:
        raise ValueError("window has fewer than two safe supervised samples")
    rows = []
    for start in range(0, len(origins), 64):
        rows.extend(builder.training_rows(origins[start:start + 64]))
    X = tuple(np.concatenate([row[0][i] for row in rows], axis=0) for i in range(len(rows[0][0])))
    Y = np.stack([row[1] for row in rows])
    weight_spec = None if short else training.validation.get('training', {}).get('sample_weight')
    weights = temporal_sample_weight(origins, origin, weight_spec)
    design_seconds = perf_counter() - started
    params = dict(config.estimator.params)
    trainer = CanonicalTrainer(training,
        estimator_factory=make_model_factory(config.estimator.model_type, params, feature_names=builder.feature_schema),
        capabilities=resolve_model_capabilities(config.estimator.model_type, params, feature_names=builder.feature_schema),
        feature_schema=builder.feature_schema)
    with threadpool_limits(limits=1):
        fit_started = perf_counter()
        artifact = trainer.train(X, Y, sample_weight=weights, n_series=1, max_workers=1)
        fit_seconds = perf_counter() - fit_started
        predict_started = perf_counter()
        # 短样本仅复用单步 predictor，明确构造长预测计划，不伪造长H训练artifact。
        prediction_builder = SupervisedDesignBuilder(prediction_config, registry, history_start=history_start)
        designs, provider = prediction_builder.forecast_designs(origin)
        if prediction_builder.feature_schema != builder.feature_schema:
            raise ValueError("training/prediction feature schemas differ")
        plan = target_plan_for_config(prediction_config)
        executor = get_standard_executor(prediction_config.strategy)(prediction_config.strategy, plan, artifact.predictors)
        tensor = executor.predict(designs[0], series_ids=prediction_builder.series_ids,
                                  forecast_times=times, feature_provider=provider)
    values = np.asarray(tensor.values).reshape(-1)
    if len(values) != len(times) or not np.isfinite(values).all():
        raise ValueError("invalid model prediction")
    audit = {"history_start": history_start.isoformat(), "origin": origin.isoformat(),
             "forecast_start": times[0].isoformat(), "forecast_end": times[-1].isoformat(),
             "history_rows": len(history), "training_samples": len(origins),
             "training_horizon": training.problem.horizon, "forecast_horizon": len(times),
             "training_label_end_max": (origins[-1] + training.problem.horizon * builder.offset).isoformat(),
             "effective_strategy": "calendar_pointwise" if short else strategy,
             "sample_weight": None if weights is None else {
                 'spec': dict(weight_spec), 'min': float(weights.min()), 'max': float(weights.max()),
                 'mean': float(weights.mean()), 'count': len(weights)},
             "direct_layout": prediction_config.features.canonical_payload()["transformations"].get("direct"),
             "model_count": artifact.model_count, "design_seconds": design_seconds,
             "fit_seconds": fit_seconds, "predict_seconds": perf_counter() - predict_started,
             "feature_schema": list(builder.feature_schema), "seconds": perf_counter() - started}
    return pd.DataFrame({"time": times, "y_pred": values}), audit


def assemble_year(actual: pd.DataFrame, windows: list[pd.DataFrame], freq: str) -> pd.DataFrame:
    actual = validate_frame(actual, freq)
    january = actual.loc[actual.time < "2025-02-01", ["time", "value"]].rename(columns={"value": "y_pred"})
    predictions = pd.concat([january, *windows], ignore_index=True).sort_values("time").reset_index(drop=True)
    if not pd.DatetimeIndex(predictions.time).equals(pd.DatetimeIndex(actual.time)):
        raise ValueError("annual predictions must match every actual timestamp exactly once")
    if not np.isfinite(predictions.y_pred).all():
        raise ValueError("nonfinite annual prediction")
    return pd.DataFrame({"time": actual.time, "y_true": actual.value, "y_pred": predictions.y_pred})


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_config(path: Path, max_windows: int | None = None, *, rerun: bool = False,
               recipe_path: Path = DEFAULT_RECIPE, months: tuple[int, ...] | None = None) -> dict:
    config = load_yaml_config(path)
    if not isinstance(config, ForecastConfigSpec):
        raise ValueError("single canonical model required")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max_windows must be positive")
    source = ROOT / config.data.sources[0].history_path
    actual = validate_frame(pd.read_csv(source), config.problem.freq)
    annual_recipe = load_recipe(recipe_path)
    if months is not None and (not months or len(set(months)) != len(months) or any(m not in range(2, 13) for m in months)):
        raise ValueError('months must be unique months in [2,12]')
    recipe = {"config": config.fingerprint(), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "implementation": implementation_fingerprint(),
              "annual_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "reporting_code_sha256": hashlib.sha256(Path(__file__).with_name("annual_reporting.py").read_bytes()).hexdigest(),
              "cold_start_code_sha256": hashlib.sha256(Path(__file__).with_name('cold_start.py').read_bytes()).hexdigest(),
              'annual_recipe': annual_recipe, 'evaluation_months': months, 'evaluation_limit': max_windows,
              "january": "actual values included in annual scoring", "year": 2025}
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:12]
    output = ROOT / "results/results_test" / config.output["scenario_subpath"] / config.result_identity() / f"annual_{identity}"
    windows_dir = output / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)
    expected = schedule(config.problem.freq)
    frames, audits = [], []
    write_json(output / "status.json", {"status": "running", "expected_windows": len(expected)})
    try:
        selected = [w for w in expected if months is None or w[2][0].month in months]
        for index, window in enumerate(selected[:max_windows]):
            file = windows_dir / f"{window[2][0]:%Y%m%d}.csv"
            metadata = file.with_suffix(".json")
            if not rerun and file.exists() and metadata.exists():
                frame = pd.read_csv(file, parse_dates=["time"], float_precision="round_trip")
                audit = json.loads(metadata.read_text())
                if (list(frame.columns) != ["time", "y_pred"]
                        or not pd.DatetimeIndex(frame.time).equals(window[2])
                        or not np.isfinite(frame.y_pred).all()
                        or audit["csv_sha256"] != hashlib.sha256(file.read_bytes()).hexdigest()):
                    raise ValueError(f"corrupt checkpoint: {file}")
            else:
                frame, audit = forecast_window(config, actual, window, recipe=annual_recipe)
                temporary = file.with_suffix(".tmp")
                frame.to_csv(temporary, index=False)
                temporary.replace(file)
                audit["csv_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                write_json(metadata, audit)
            frames.append(frame)
            audits.append(audit)
            print(f"{path.relative_to(ROOT)} {index + 1}/{len(expected)} {window[2][0]:%Y-%m-%d} {audit['seconds']:.1f}s", flush=True)
            write_json(output / "status.json", {"status": "running", "completed_windows": len(frames), "expected_windows": len(expected)})
        completed = len(frames) == len(expected)
        write_json(output / "audit.json", {**recipe, "windows": audits, "complete": completed})
        if completed:
            annual = assemble_year(actual, frames, config.problem.freq)
            temporary = output / "prediction.tmp"
            annual.to_csv(temporary, index=False)
            temporary.replace(output / "prediction.csv")
            scores = write_annual_results(output, annual, config.problem.freq, recipe)
            write_json(output / "scores.json", scores)
            # 重新加载实际交付文件，验收行数、对齐、有限值、1月回填。
            saved = pd.read_csv(output / "prediction.csv", parse_dates=["time"], float_precision="round_trip")
            pd.testing.assert_frame_equal(saved, annual, check_exact=False, rtol=1e-14, atol=1e-14)
        state = {"status": "completed" if completed else "partial", "completed_windows": len(frames),
                 "expected_windows": len(expected), "output": str(output)}
        write_json(output / "status.json", state)
        return state
    except BaseException as exc:
        write_json(output / "status.json", {"status": "failed", "completed_windows": len(frames), "error": str(exc)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--config-yaml", type=Path)
    selection.add_argument("--freq", choices=("1D", "15min"))
    parser.add_argument("--max-windows", type=int)
    parser.add_argument('--recipe', type=Path, default=DEFAULT_RECIPE)
    parser.add_argument('--months', type=int, nargs='+')
    parser.add_argument("--rerun", action="store_true", help="Refit every selected window, do not reuse window CSVs.")
    args = parser.parse_args()
    paths = [args.config_yaml.resolve()] if args.config_yaml else sorted(Path(__file__).parent.glob("*/*/freq_*/lgbm_*.yaml"))
    if not args.config_yaml:
        if len(paths) != 20:
            raise ValueError("expected exactly 20 non-recursive physical model YAML files")
        paths = [path for path in paths if load_yaml_config(path).problem.freq == args.freq]
    results = [run_config(path, args.max_windows, rerun=args.rerun, recipe_path=args.recipe,
                          months=tuple(args.months) if args.months else None) for path in paths]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
