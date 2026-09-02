"""Canonical model bundle construction and persistence.

This module owns the durable schema-2 artifact boundary. Runtime orchestration
builds training state and delegates serialization here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from forecasting_core.artifacts import ForecastModelBundle
from forecasting_core.probabilistic_spec import probabilistic_spec_from_mapping
from model_training.strategies import CanonicalStrategyArtifact
from model_training.trainer import CanonicalTrainer
from models.ModelSaveLoad import ModelDeployPkl


def build_strategy_model_bundle(
    trainer: CanonicalTrainer,
    artifact: CanonicalStrategyArtifact,
    *,
    feature_scaler: Any = None,
    target_transform: Any = None,
    input_schema: Mapping[str, Any] | None = None,
    feature_lineage: Sequence[Mapping[str, Any]] = (),
    source_lineage: Sequence[Mapping[str, Any]] = (),
    series_ids: tuple[Any, ...] = (),
    calibration_state: Mapping[str, Any] | None = None,
) -> ForecastModelBundle:
    """Build a schema-2 bundle without making the training layer own IO types."""
    if not isinstance(trainer, CanonicalTrainer):
        raise TypeError("trainer must be a CanonicalTrainer")
    if not isinstance(artifact, CanonicalStrategyArtifact):
        raise TypeError("artifact must be a CanonicalStrategyArtifact")
    config = trainer.config
    # 部署态概率规格由唯一解析入口从 canonical YAML 段构建（2026-09-01：
    # 取代此前 mode 分支硬编码 crossing_method 的写法，配置即事实）。
    probabilistic_spec = probabilistic_spec_from_mapping(
        config.probabilistic.canonical_payload()
    )
    estimator_payload = config.estimator.canonical_payload()
    estimator_payload["capabilities"] = trainer.capabilities.canonical_payload()
    assert config.strategy is not None
    return ForecastModelBundle(
        schema_version=2,
        model=artifact,
        feature_scaler=feature_scaler,
        target_transform=target_transform,
        selected_features=trainer.feature_schema,
        input_schema=dict(
            input_schema or {"columns": list(trainer.feature_schema)}
        ),
        probabilistic_spec=probabilistic_spec,
        model_type=config.estimator.model_type,
        pred_method=None,
        canonical_problem=config.problem.canonical_payload(),
        strategy_spec=config.strategy.canonical_payload(),
        estimator_spec=estimator_payload,
        dimensions=(artifact.N, artifact.H, artifact.K),
        series_ids=tuple(series_ids),
        target_order=config.problem.targets,
        feature_lineage=tuple(dict(item) for item in feature_lineage),
        source_lineage=tuple(dict(item) for item in source_lineage),
        training_scope=config.problem.training_scope,
        result_schema_version=2,
        config_fingerprint=config.fingerprint(),
        calibration_state=(
            dict(calibration_state) if calibration_state is not None else None
        ),
    )


def persist_model_bundle(
    bundle: ForecastModelBundle,
    model_dir: str | Path,
) -> tuple[Path, Path]:
    """Persist one schema-2 model bundle and its readable schema metadata."""
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    schema_path = output_dir / "resolved_model.json"
    ModelDeployPkl(str(model_path)).save_model(bundle)
    bundle.write_schema_json(schema_path)
    return model_path, schema_path


__all__ = ["build_strategy_model_bundle", "persist_model_bundle"]
