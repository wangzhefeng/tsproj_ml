"""融合产物路径、训练诊断与外层回测报告；不训练任何模型。"""
import json
from collections.abc import Mapping
from pathlib import Path
from threading import Lock

import pandas as pd

from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from forecasting_core.tensors import PointForecastTensor
from model_ensemble.inference.forecast import build_ensemble_forecast
from model_ensemble.inference.predictor import combine_members
from model_evaluation.point import resolve_aggregate_weighting
from model_predicting.artifacts.results import write_forecast_results
from model_testing.artifacts.reporting import write_backtest_results
from model_testing.artifacts.tensor_frames import backtest_tensors_to_long
from utils.log_util import logger

# pyplot 有进程级状态；并发方法共享 OOF 时，绘图仍必须串行。
_PLOT_LOCK = Lock()


def ensemble_output_paths(config, output_root):
    identity = config.output.get("identity", {})
    scenario = str(identity.get("scenario_subpath", "canonical") if isinstance(identity, Mapping)
                   else config.output.get("scenario_subpath", "canonical"))
    if not scenario or scenario == "canonical":
        scenario = str(config.output.get("scenario_subpath", "canonical"))
    scenario = scenario.strip("/") or "canonical"
    if output_root is not None:
        run_dir = Path(output_root) / scenario / config.result_identity()
        return run_dir, run_dir / "pretrained_models", run_dir / "results_test", run_dir / "results_forecast"
    directories = config.output.get("directories", {})
    if isinstance(directories, Mapping) and {"checkpoints", "tests", "forecast"}.issubset(directories):
        model_dir = Path(str(directories["checkpoints"])) / scenario / config.result_identity()
        test_dir = Path(str(directories["tests"])) / scenario / config.result_identity()
        forecast_dir = Path(str(directories["forecast"])) / scenario / config.result_identity()
        return forecast_dir, model_dir, test_dir, forecast_dir
    run_dir = Path(str(config.output.get("results_root", "results"))) / scenario / config.result_identity()
    return run_dir, run_dir / "pretrained_models", run_dir / "results_test", run_dir / "results_forecast"


def meta_train_frame(config, artifact, oof, runner):
    """诊断表保留真实 fold/series/time；其标签参与过融合器训练。"""
    combined = combine_members(artifact, oof.values_by_member)
    probability = probabilistic_spec_from_mapping(config.probabilistic.canonical_payload())
    positions = {origin.isoformat(): index for index, origin in enumerate(runner.supervised_origins)}
    frames = []
    count = len(runner.series_ids)
    for index, fold in enumerate(oof.folds):
        position = positions[fold["origin"]]
        times = runner.forecast_times(runner.supervised_origins[position])
        actual = runner.actual(position, times)
        prediction = build_ensemble_forecast(
            combined[index * count:(index + 1) * count], probability=probability,
            series_ids=runner.series_ids, forecast_times=times, targets=config.problem.targets,
            method_name=config.method.name,
        )
        frames.append(backtest_tensors_to_long(actual, prediction, window=int(fold["fold"])))
    return pd.concat(frames, ignore_index=True)


def write_predictions_and_scores(config, *, forecast, runner, backtest, artifact, oof, audit, forecast_dir, test_dir):
    history = None
    try:
        full = runner.target_history(runner.origin)
        steps = max(1, config.problem.horizon * 5)
        history = PointForecastTensor(full.values[:, -steps:, :], full.series_ids, full.forecast_times[-steps:], full.targets)
    except (ValueError, KeyError) as exc:
        logger.warning("[ensemble forecast plot] target history unavailable: %s", exc)
    with _PLOT_LOCK:
        write_forecast_results(forecast_dir, forecast, history=history)
        write_backtest_results(
            test_dir, backtest.frame, backtest.point_scores,
            aggregate_weighting=resolve_aggregate_weighting(config.problem.targets, config.validation.get("aggregate_weighting")),
            probabilistic_scores_df=backtest.probabilistic_scores,
            metadata={**backtest.metadata, "run_evidence": audit["run_evidence"], "method": config.method.name},
        )
    meta_train_frame(config, artifact, oof, runner).to_csv(test_dir / "meta_train_predictions.csv", index=False, encoding="utf_8_sig")
    for kind, name in (("point", "meta_train_scores.csv"), ("probabilistic", "meta_train_scores_probabilistic.csv")):
        scores = audit["fused_oof_scores"][kind]
        if scores is not None:
            scores.to_csv(test_dir / name, index=False, encoding="utf_8_sig")


def write_run_metadata(config, *, document_fingerprint, artifact, oof, audit, cache_hit, resources, forecast_dir, test_dir):
    runtime = {
        "run_evidence": audit["run_evidence"], "resources": resources,
        "oof_fingerprint": artifact.oof_fingerprint, "oof_cache_hit": cache_hit,
        "member_order": list(artifact.member_order), "folds": list(oof.folds),
        "oof_score_role": "meta_train_diagnostic",
    }
    resolved = {
        **config.canonical_payload(), "config_fingerprint": config.fingerprint(),
        "document_fingerprint": document_fingerprint,
        "resolved_member_fingerprints": dict(config.resolved_member_fingerprints),
        "resolved_origin": config.resolved_origin, "runtime": runtime,
    }
    (forecast_dir / "resolved_config.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    path = test_dir / "result_metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update({"config_fingerprint": config.fingerprint(), "oof_fingerprint": artifact.oof_fingerprint,
                    "oof_cache_hit": cache_hit, "meta_train_folds": list(oof.folds), "runtime_resources": resources})
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
