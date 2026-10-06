"""融合 bundle 组装与完成清单；保留既有 artifact 类的 pickle 路径。"""
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from forecasting_core.artifacts import ForecastModelBundle
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from data_loading.sources.provenance import file_sha256
from model_ensemble.artifacts import method_artifact_audit_payload


def build_ensemble_bundle(config, artifact, member_bundles, *, series_ids, dimensions, calibration_state=None):
    """融合自有概率规格、成员 lineage 和状态，独立于训练编排。"""
    return ForecastModelBundle(
        schema_version=2,
        model={"ensemble_artifact": artifact, "member_bundles": member_bundles},
        feature_scaler=None, target_transform=None, selected_features=(),
        input_schema={
            "members": list(artifact.member_order),
            "panel": {"series_id_cols": list(config.problem.series_id_cols),
                      "known_series_ids": [list(value) if isinstance(value, tuple) else value for value in series_ids],
                      "unknown_series_policy": "raise"},
        },
        probabilistic_spec=probabilistic_spec_from_mapping(config.probabilistic.canonical_payload()),
        model_type="ensemble", pred_method=None, canonical_problem=config.problem.canonical_payload(),
        strategy_spec=None, estimator_spec=None,
        ensemble_spec={**config.payload(), "oof_fingerprint": artifact.oof_fingerprint,
                       "folds": list(artifact.fold_summary),
                       "method_artifact": method_artifact_audit_payload(artifact.method_artifact)},
        dimensions=dimensions, series_ids=series_ids, target_order=config.problem.targets,
        feature_lineage=tuple({"member": name, **item} for name, member in member_bundles.items() for item in member.feature_lineage),
        source_lineage=tuple({"member": name, **item} for name, member in member_bundles.items() for item in member.source_lineage),
        training_scope=config.problem.training_scope, result_schema_version=2,
        config_fingerprint=config.fingerprint(), calibration_state=calibration_state,
        execution_mode="research_replay" if any(member.execution_mode == "research_replay" for member in member_bundles.values()) else "strict",
    )


class RunCompletion:
    """running→completed/failed；全组读回验收前绝不发布 completed。"""
    def __init__(self):
        self.path: Path | None = None
        self.fingerprint: str | None = None

    def _write(self, status: str, **payload):
        if self.path is None:
            return
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps({"status": status, "config_fingerprint": self.fingerprint, **payload}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.path)

    def start(self, model_dir: Path, fingerprint: str):
        model_dir.mkdir(parents=True, exist_ok=True)
        self.path = model_dir / "run_state.json"
        self.fingerprint = fingerprint
        self._write("running")

    def fail(self, error: Exception):
        self._write("failed", error_type=type(error).__name__, error=str(error))

    def complete(self, result: dict[str, Any]):
        model, test, forecast = result["model_dir"], result["test_dir"], result["forecast_dir"]
        paths = [model / "model.pkl", model / "resolved_model.json", forecast / "prediction.csv",
                 forecast / "resolved_config.json", test / "cv_plot_df.csv", test / "test_scores_df.csv",
                 test / "result_metadata.json", test / "meta_train_scores.csv", test / "meta_train_predictions.csv"]
        if result["bundle"].probabilistic_spec.mode == "quantile":
            paths.append(test / "test_scores_probabilistic_df.csv")
        for window in result["backtest"].frame.window.unique():
            paths.append(test / "windows_results" / f"window_{int(window):02d}.png")
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty ensemble artifact: {path}")
        for path in (model / "resolved_model.json", forecast / "resolved_config.json"):
            if json.loads(path.read_text())["config_fingerprint"] != self.fingerprint:
                raise ValueError("ensemble artifact fingerprint mismatch")
        predictions = pd.read_csv(forecast / "prediction.csv")
        n, h, k = result["bundle"].dimensions
        if len(predictions) != n * h * k or predictions.duplicated(["series_id", "time", "target"]).any():
            raise ValueError("ensemble forecast artifact has an invalid coordinate grid")
        backtest = pd.read_csv(test / "cv_plot_df.csv")
        if len(backtest) != len(result["backtest"].frame) or backtest.duplicated(["series_id", "time", "target", "window"]).any():
            raise ValueError("ensemble backtest artifact has an invalid coordinate grid")
        self._write("completed", artifacts={str(path): file_sha256(path) for path in paths})
