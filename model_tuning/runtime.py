"""串行调参、选优后独立 holdout、final 产物验收；不绕过 canonical 主链。"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import optuna
import pandas as pd
import yaml

from data_loading import BUILTIN_GENERATORS, SourceRegistry
from forecasting_core.specs.config import ForecastConfigSpec, parse_model_config
from forecasting_core.specs.validation import FixedStepBacktestSpec
from forecasting_core.specs.weather import WeatherGenerationSpec
from model_testing.contracts.windows import rolling_backtest_windows
from model_tuning.specs import TuningSpec
from pipeline.batch_artifacts import artifact_digests, artifact_paths, validate_artifacts
from pipeline.lifecycle import BacktestRuntimeResult, CanonicalRuntimeResult
from pipeline.runner import CanonicalBaseModelRunner
from pipeline.run_state import require_completed_state


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_config(path: Path, config: ForecastConfigSpec) -> None:
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(config.canonical_payload(), handle, allow_unicode=True, sort_keys=False)
    reloaded = parse_model_config(yaml.safe_load(path.read_text(encoding="utf-8")), source=path)
    if reloaded.fingerprint() != config.fingerprint():
        raise ValueError("exported canonical YAML fingerprint mismatch")


def _output_config(config: ForecastConfigSpec, root: Path, stage: str) -> ForecastConfigSpec:
    payload = config.canonical_payload()
    payload["output"] = {
        "scenario_subpath": f"tuning/{stage}", "results_root": str(root),
        "directories": {
            "checkpoints": str(root / "pretrained_models"),
            "tests": str(root / "results_test"),
            "forecast": str(root / "results_forecast"),
        },
    }
    return parse_model_config(payload, source="<tuning output>")


def _holdout_config(config: ForecastConfigSpec, spec: TuningSpec) -> ForecastConfigSpec:
    payload = config.canonical_payload()
    validation = dict(config.validation.canonical_payload())
    validation.update(forecast_origin=spec.holdout_origin, fold_count=spec.holdout_fold_count)
    payload["validation"] = validation
    return parse_model_config(payload, source="<tuning holdout>")


def _window_signature(config: ForecastConfigSpec) -> tuple[str, ...]:
    """仅根据声明的规则时间网格固定评分原点，不读取 holdout 值。"""
    backtest = config.validation.backtest
    if not isinstance(backtest, FixedStepBacktestSpec) or backtest.train_history_steps is not None:
        raise ValueError("tuning currently requires fixed-step without train_history_steps")
    raw_origin = config.validation.get("forecast_origin")
    if not isinstance(raw_origin, str) or not raw_origin.strip():
        raise ValueError("tuning requires an explicit forecast_origin")
    origin = pd.Timestamp(raw_origin)
    if pd.isna(origin):
        raise ValueError("forecast_origin cannot be NaT")
    origin = cast(pd.Timestamp, origin)
    offset = pd.tseries.frequencies.to_offset(config.problem.freq)
    # 只接受固定步长，避免把自然月跨度当固定 Tick。
    if not isinstance(offset, pd.tseries.offsets.Tick):
        raise ValueError("tuning requires a fixed-duration frequency")
    candidates = tuple(cast(pd.Timestamp, value) for value in pd.date_range(
        end=origin - config.problem.horizon * offset, periods=backtest.history_steps, freq=offset,
    ))
    windows = rolling_backtest_windows(
        candidates, offset=offset, horizon=config.problem.horizon, backtest=backtest,
        schedule_origin=origin if config.validation.get("schedule_mode") == "intraday" else None,
    )
    if len(windows) != backtest.fold_count:
        raise ValueError("tuning cannot provide the declared fold_count")
    return tuple(window.origin.isoformat() for window in windows)


def _runner(config: ForecastConfigSpec, root: Path, signature: tuple[str, ...]) -> CanonicalBaseModelRunner:
    registry = SourceRegistry(config.data, Path.cwd(), generators=BUILTIN_GENERATORS)
    runner = CanonicalBaseModelRunner(
        config, registry, cast(pd.Timestamp, pd.Timestamp(config.validation["forecast_origin"])), compiled_cache_root=root,
    )
    windows = runner.backtest_windows()
    if tuple(window.origin.isoformat() for window in windows) != signature:
        raise ValueError("candidate changes fixed tuning windows")
    expected_rows = config.validation["train_window_steps"]
    if any(window.metadata["training_sample_count"] != expected_rows for window in windows):
        raise ValueError("candidate cannot supply the complete declared training window")
    return runner


def _score(test_dir: Path, metric: str, config: ForecastConfigSpec, *, backtest_only: bool) -> tuple[float, list[dict]]:
    if backtest_only:
        state = json.loads((test_dir / "run_state.json").read_text(encoding="utf-8"))
        require_completed_state(state, config.fingerprint())
    frame = pd.read_csv(test_dir / "test_scores_df.csv")
    rows = frame.loc[frame["scope"] == "aggregate"].sort_values("window")
    expected_windows = list(range(1, config.validation["fold_count"] + 1))
    if rows["window"].tolist() != expected_windows:
        raise ValueError("tuning scores missing/duplicating aggregate windows")
    values = rows[metric].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (rows["Valid Points"] <= 0).any():
        raise ValueError("tuning requires finite scores and positive valid points in every fold")
    return float(values.mean()), [
        {"window": int(window), "score": float(value)}
        for window, value in zip(rows["window"], values)
    ]


def run_tuning(base: ForecastConfigSpec, spec: TuningSpec, *, study_dir: str | Path) -> dict[str, Any]:
    if not isinstance(base, ForecastConfigSpec) or not isinstance(spec, TuningSpec):
        raise TypeError("tuning requires ForecastConfigSpec and TuningSpec")
    if base.probabilistic.get("mode", "point") != "point":
        raise ValueError("tuning currently supports point models only")
    if any(isinstance(source.generator_options, WeatherGenerationSpec)
           and source.generator_options.research is not None for source in base.data.sources):
        raise ValueError("research replay is not supported by tuning")
    search_signature = _window_signature(base)
    holdout_base = _holdout_config(base, spec)
    holdout_signature = _window_signature(holdout_base)
    search_cutoff = pd.Timestamp(base.validation["forecast_origin"])
    if pd.Timestamp(holdout_signature[0]) < search_cutoff:
        raise ValueError("holdout must start after all search labels: its first origin must be >= search cutoff")
    root = Path(study_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "trials").mkdir()
    _write_config(root / "base.yaml", base)
    _write_json(root / "search_spec.json", asdict(spec))
    report: dict[str, Any] = {
        "status": "running", "selection_metric": spec.metric,
        "aggregation": "equal mean of per-window aggregate scores", "seed": spec.seed,
        "optuna_version": optuna.__version__,
        "trial_count": spec.trials, "trials": [], "search_cutoff": search_cutoff.isoformat(),
        "search_origins": list(search_signature), "holdout_origins": list(holdout_signature),
    }
    state_path = root / "study.json"
    _write_json(state_path, report)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=spec.seed),
                                pruner=optuna.pruners.NopPruner())

    def objective(trial: optuna.trial.Trial) -> float:
        started = perf_counter()
        record: dict[str, Any] = {"number": trial.number, "status": "running", "score": None}
        trial_dir = root / "trials" / f"{trial.number:04d}"
        trial_dir.mkdir()
        try:
            candidate, selected = spec.sample_config(base, trial)
            candidate = _output_config(candidate, root, f"trial_{trial.number:04d}")
            path = trial_dir / "config.yaml"
            _write_config(path, candidate)
            record.update(parameters=selected, config_yaml=str(path), config_fingerprint=candidate.fingerprint())
            runner = _runner(candidate, root, search_signature)
            result = runner.run(backtest_only=True)
            if not isinstance(result, BacktestRuntimeResult):
                raise TypeError("tuning trial must stop after backtest")
            score, fold_scores = _score(result.test_dir, spec.metric, candidate, backtest_only=True)
            record.update(status="completed", score=score, fold_scores=fold_scores, test_dir=str(result.test_dir))
            return score
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["suggested_values"] = dict(trial.params)
            record["wall_seconds"] = perf_counter() - started
            report["trials"].append(record)
            _write_json(trial_dir / "trial.json", record)
            _write_json(state_path, report)

    try:
        study.optimize(objective, n_trials=spec.trials, n_jobs=1, catch=(ValueError, TypeError, RuntimeError))
        successful = [trial for trial in report["trials"] if trial["status"] == "completed"]
        if not successful:
            raise RuntimeError("tuning produced no successful trials")
        winner = min(successful, key=lambda trial: (trial["score"], trial["number"]))
        report.update(status="selected", best_trial=winner["number"], best_search_score=winner["score"])
        _write_json(state_path, report)  # 先固定选优证据，再读取 holdout 标签。
        payload = yaml.safe_load(Path(winner["config_yaml"]).read_text(encoding="utf-8"))
        selected = parse_model_config(payload, source=winner["config_yaml"])
        final_config = _output_config(_holdout_config(selected, spec), root, "holdout_final")
        _write_config(root / "validated.yaml", final_config)
        runner = _runner(final_config, root, holdout_signature)
        result = runner.run()
        if not isinstance(result, CanonicalRuntimeResult):
            raise TypeError("selected model must complete final fit and forecast")
        if result.bundle.execution_mode != "strict":
            raise ValueError("tuning cannot produce a research bundle")
        paths = artifact_paths(result)
        manifest = {
            "config_fingerprint": final_config.fingerprint(), "result_identity": final_config.result_identity(),
            "artifacts": paths, "artifact_sha256": artifact_digests(paths),
        }
        validate_artifacts(manifest)
        holdout_score, holdout_folds = _score(result.test_dir, spec.metric, final_config, backtest_only=False)
        exported = _output_config(final_config, root / "exported_run", "best")
        _write_config(root / "best.yaml", exported)
        offset = pd.tseries.frequencies.to_offset(base.problem.freq)
        report.update(status="completed", holdout_score=holdout_score, holdout_fold_scores=holdout_folds,
                      holdout_label_start=(pd.Timestamp(holdout_signature[0]) + offset).isoformat(),
                      best_yaml=str(root / "best.yaml"), final_artifacts=manifest)
        _write_json(state_path, report)
        return report
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_json(state_path, report)
        raise
