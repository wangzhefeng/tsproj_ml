"""逐配置验收当前实现、当前数据对应的年度文件；未完成退出非零。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from model_performance.checkpoints import implementation_fingerprint
from prepare import validate_frame
from cold_start import load_recipe


def verify(freq: str | None = None) -> dict:
    paths = sorted(Path(__file__).parent.glob("*/*/freq_*/lgbm_*.yaml"))
    if len(paths) != 20:
        raise ValueError("expected all 20 non-recursive physical configurations")
    if freq is not None:
        paths = [path for path in paths if load_yaml_config(path).problem.freq == freq]
    implementation = implementation_fingerprint()
    code_hash = hashlib.sha256(Path(__file__).with_name("annual_backtest.py").read_bytes()).hexdigest()
    reporting_hash = hashlib.sha256(Path(__file__).with_name("annual_reporting.py").read_bytes()).hexdigest()
    cold_hash = hashlib.sha256(Path(__file__).with_name('cold_start.py').read_bytes()).hexdigest()
    recipe = load_recipe()
    reports = []
    for path in paths:
        cfg = load_yaml_config(path)
        source = ROOT / cfg.data.sources[0].history_path
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        base = ROOT / "results/results_test" / cfg.output["scenario_subpath"] / cfg.result_identity()
        eligible = []
        for audit_path in base.glob("annual_*/audit.json"):
            audit = json.loads(audit_path.read_text())
            if (audit["implementation"] == implementation and audit["annual_code_sha256"] == code_hash
                    and audit.get("reporting_code_sha256") == reporting_hash
                    and audit.get('cold_start_code_sha256') == cold_hash
                    and audit.get('annual_recipe') == recipe
                    and audit.get('evaluation_months') is None and audit.get('evaluation_limit') is None
                    and audit["config"] == cfg.fingerprint() and audit["source_sha256"] == source_hash):
                eligible.append((audit_path.parent, audit))
        if len(eligible) > 1:
            raise ValueError(f"ambiguous current output: {path}")
        report = {"config": str(path.relative_to(ROOT)), "completed": False}
        if eligible:
            output, audit = eligible[0]
            report.update(output=str(output), windows=len(audit["windows"]))
            state = json.loads((output / "status.json").read_text())
            if state["status"] == "completed":
                actual = validate_frame(pd.read_csv(source), cfg.problem.freq)
                frame = pd.read_csv(output / "prediction.csv", parse_dates=["time"], float_precision="round_trip")
                if list(frame.columns) != ["time", "y_true", "y_pred"]:
                    raise ValueError("wrong annual CSV schema")
                if not pd.DatetimeIndex(frame.time).equals(pd.DatetimeIndex(actual.time)):
                    raise ValueError("annual grid does not match actual")
                np.testing.assert_allclose(frame.y_true, actual.value, rtol=1e-14, atol=1e-14)
                if not np.isfinite(frame.y_pred).all():
                    raise ValueError("nonfinite predictions")
                january = frame.time < "2025-02-01"
                np.testing.assert_allclose(frame.loc[january, "y_pred"], frame.loc[january, "y_true"], rtol=1e-14, atol=1e-14)
                expected_windows = 11 if cfg.problem.freq == "1D" else 334
                if len(audit["windows"]) != expected_windows or not audit["complete"]:
                    raise ValueError("incomplete annual audit")
                for window in audit["windows"]:
                    if window['effective_strategy'] not in ('direct', 'mimo', 'calendar_baseline', 'calendar_pointwise'):
                        raise ValueError('unsupported strategy in current result')
                    if pd.Timestamp(window["training_label_end_max"]) > pd.Timestamp(window["origin"]):
                        raise ValueError("training labels exceed origin")
                    if window['forecast_start'].startswith('2025-02-01') and cfg.problem.freq == '1D':
                        if window['effective_strategy'] != recipe['cold_start']['method']:
                            raise ValueError('cold-start policy differs from selected recipe')
                    elif cfg.validation.get('training', {}).get('sample_weight') is not None:
                        weight = window.get('sample_weight')
                        if weight is None or weight['count'] != window['training_samples'] or not np.isclose(weight['mean'], 1.):
                            raise ValueError('configured sample weighting not applied')
                scores = json.loads((output / "scores.json").read_text())
                error = frame.y_pred.to_numpy() - frame.y_true.to_numpy()
                np.testing.assert_allclose(scores["MAE"], np.abs(error).mean())
                np.testing.assert_allclose(scores["RMSE"], np.sqrt(np.square(error).mean()))
                canonical = pd.read_csv(output / "cv_plot_df.csv", parse_dates=["time"])
                if canonical.duplicated(["series_id", "time", "target", "window"]).any():
                    raise ValueError("duplicate canonical result keys")
                if not pd.DatetimeIndex(canonical.time).equals(pd.DatetimeIndex(frame.time)):
                    raise ValueError("canonical and annual timestamps differ")
                np.testing.assert_allclose(canonical.actual_value, frame.y_true, rtol=1e-14, atol=1e-14)
                np.testing.assert_allclose(canonical.predict_value, frame.y_pred, rtol=1e-14, atol=1e-14)
                plots = [output / "test_prediction.png"] + [output / "windows_results" / f"window_{w:02d}.png"
                                                             for w in range(expected_windows + 1)]
                if any(not p.is_file() or p.stat().st_size == 0 for p in plots):
                    raise ValueError("missing annual/window plot")
                for name in ("test_scores_df.csv", "test_scores_horizon_df.csv", "annual_scores_df.csv", "result_metadata.json", 'diagnostic_scores_df.csv'):
                    if not (output / name).is_file():
                        raise ValueError(f"missing canonical artifact: {name}")
                report["plots"] = len(plots)
                report.update(completed=True, rows=len(frame), MAE=scores["MAE"], RMSE=scores["RMSE"])
        reports.append(report)
    return {"expected": len(paths), "completed": sum(item["completed"] for item in reports), "results": reports}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freq", choices=("1D", "15min"))
    report = verify(parser.parse_args().freq)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["completed"] == report["expected"] else 1)
