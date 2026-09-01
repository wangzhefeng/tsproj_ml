"""Executable canonical runtime without legacy configuration translation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from model_testing import validation
from data_loading import (
    BUILTIN_GENERATORS,
    EndogenousFutureProvider,
    InformationSetRequest,
    SourceRegistry,
)
from model_training.estimators import (
    make_model_factory,
    resolve_model_capabilities,
)
from feature_engineering import CompiledFeatures, FeatureCompiler
from model_evaluation.marginal import evaluate_marginal_distribution
from model_evaluation.point import (
    build_eval_mask_payload,
    evaluate_point_forecasts,
    resolve_aggregate_weighting,
)
from feature_engineering.selection import (
    CanonicalFeatureSelector,
    normalize_feature_selection,
    selected_indices_for_artifact,
)
from utils.log_util import logger
from model_forecasting.results import (
    backtest_tensors_to_long,
    write_backtest_results,
    write_forecast_results,
)
from forecasting_core.specs import (
    CalendarMonthBacktestSpec,
    ColumnRole,
    FixedStepBacktestSpec,
    ForecastConfigSpec,
    TargetAdapter,
)
from model_testing.backtest import (
    actual_tensor,
    positive_validation_int,
    resolve_origin,
    seasonal_naive_tensor,
)
from model_forecasting.forecaster import (
    CanonicalForecaster,
    CanonicalMarginalQuantileForecaster,
)
from model_forecasting.persistence import (
    build_strategy_model_bundle,
    persist_model_bundle,
)
from model_training.strategies import (
    CanonicalStrategyArtifact,
    StrategyTargetPlan,
    TargetCoordinate,
)
from forecasting_core.tensors import PointForecastTensor
from model_forecasting.transforms import CanonicalFeatureScaler, CanonicalTargetTransform
from model_training.trainer import CanonicalTrainer
from models.ModelFactory import ModelFactory
from probabilistic.training import CanonicalMarginalQuantileTrainer
from forecasting_core.artifacts import ForecastModelBundle, MarginalForecastDistribution
from model_forecasting.backtest_runtime import overwrite_calendar_month_backtest
from model_forecasting.design import (
    _RegistryDesignBuilder,
    _actual_at_origin,
    _holdout_training_indices,
    _label_end,
    _label_start,
    _rolling_backtest_windows,
    _sample_indices,
    _supervised_arrays,
)
from model_forecasting.fit_service import (
    _fit_point,
    _fit_quantile,
    _fit_runtime_transforms,
    _forecast_designs_with_scaler,
    _predict,
    _restore_prediction,
)

# P3/D3：回测原语已公开化至 model_testing/backtest.py；保留私有别名转发（历史调用点零改动）。
_positive_validation_int = positive_validation_int
_seasonal_naive_tensor = seasonal_naive_tensor
_actual_tensor = actual_tensor
_resolve_origin = resolve_origin


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeResult:
    run_dir: Path
    model_dir: Path
    test_dir: Path
    forecast_dir: Path
    fingerprint: str
    bundle: ForecastModelBundle






































def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _proof_payload(compiled_items: tuple[CompiledFeatures, ...]) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for compiled in compiled_items:
        for proof in compiled.visibility_proof:
            item = asdict(proof)
            key = tuple(item.items())
            if key in seen:
                continue
            seen.add(key)
            item["target_time"] = proof.target_time.isoformat()
            item["source_time"] = (
                proof.source_time.isoformat() if proof.source_time is not None else None
            )
            item["forecast_origin"] = proof.forecast_origin.isoformat()
            item["available_at"] = (
                proof.available_at.isoformat() if proof.available_at is not None else None
            )
            payload.append(item)
    return payload


def _source_lineage_payload(
    compiled_items: tuple[CompiledFeatures, ...],
) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for compiled in compiled_items:
        for lineage in compiled.source_lineage:
            item = {
                "source": lineage.source_name,
                "path_version": lineage.path_version,
                "path": lineage.path,
                "availability": lineage.availability_policy,
                "oracle": lineage.oracle,
            }
            key = tuple(item.items())
            if key not in seen:
                seen.add(key)
                payload.append(item)
    return payload


def _compiled_lineage(
    feature_schema: tuple[str, ...],
    proof_payload: list[dict[str, Any]],
    config: ForecastConfigSpec,
) -> tuple[tuple[dict[str, Any], ...], dict[str, list[str]]]:
    by_feature = {}
    for item in proof_payload:
        by_feature.setdefault(item["feature_name"], item)
    feature_lineage = []
    availability_summary: dict[str, list[str]] = {}
    source_availability = {
        source.name: (
            source.availability.value if source.availability is not None else "static"
        )
        for source in config.data.sources
    }
    for feature in feature_schema:
        if feature in config.problem.series_id_cols:
            feature_lineage.append(
                {
                    "feature": feature,
                    "source": "series_identity",
                    "role": "key",
                    "source_time": None,
                    "provider": None,
                    "availability": "static",
                }
            )
            availability_summary.setdefault("static", []).append(feature)
            continue
        proof = by_feature[feature]
        availability = (
            "known_future"
            if proof["source_name"] == "calendar"
            else source_availability.get(proof["source_name"], proof["role"])
        )
        feature_lineage.append(
            {
                "feature": feature,
                "source": proof["source_name"],
                "role": proof["role"],
                "source_time": proof["source_time"],
                "provider": proof["provider"],
                "availability": availability,
            }
        )
        availability_summary.setdefault(availability, []).append(feature)
    return (
        tuple(feature_lineage),
        {key: availability_summary[key] for key in sorted(availability_summary)},
    )


def _output_paths(
    config: ForecastConfigSpec,
    fingerprint: str,
    output_root: str | Path | None,
) -> tuple[Path, Path, Path, Path]:
    output = config.output
    identity = output.get("identity", {})
    scenario = str(
        identity.get("scenario_subpath", output.get("scenario_subpath", "canonical"))
        if isinstance(identity, Mapping)
        else output.get("scenario_subpath", "canonical")
    ).strip("/") or "canonical"
    result_identity = config.result_identity()
    if output_root is not None:
        run_dir = Path(output_root) / scenario / result_identity
        return (
            run_dir,
            run_dir / "pretrained_models",
            run_dir / "results_test",
            run_dir / "results_forecast",
        )
    directories = output.get("directories", {})
    if isinstance(directories, Mapping) and {
        "checkpoints",
        "tests",
        "forecast",
    }.issubset(directories):
        model_dir = Path(str(directories["checkpoints"])) / scenario / result_identity
        test_dir = Path(str(directories["tests"])) / scenario / result_identity
        forecast_dir = Path(str(directories["forecast"])) / scenario / result_identity
        return forecast_dir, model_dir, test_dir, forecast_dir
    legacy_directories = {
        "checkpoints_dir": "model",
        "test_results_dir": "test",
        "pred_results_dir": "forecast",
    }
    if set(legacy_directories).issubset(output):
        model_dir = Path(str(output["checkpoints_dir"])) / scenario / result_identity
        test_dir = Path(str(output["test_results_dir"])) / scenario / result_identity
        forecast_dir = Path(str(output["pred_results_dir"])) / scenario / result_identity
        return forecast_dir, model_dir, test_dir, forecast_dir
    legacy_scenario = str(output.get("scenario_subpath", scenario)).strip("/") or scenario
    legacy_root = Path(str(output.get("results_root", "results")))
    run_dir = legacy_root / legacy_scenario / result_identity
    return (
        run_dir,
        run_dir / "pretrained_models",
        run_dir / "results_test",
        run_dir / "results_forecast",
    )


class CanonicalBaseModelRunner:
    """Public narrow facade over the validated single-model runtime path.

    Extracted in E1 (v4 §6.1). Single-model `run_canonical_config` delegates to
    `run`; ensemble members (E5+) depend only on this facade, never on runtime
    private helpers.
    """

    def __init__(
        self,
        config: ForecastConfigSpec,
        registry: SourceRegistry,
        origin: pd.Timestamp,
    ) -> None:
        if not isinstance(config, ForecastConfigSpec):
            raise TypeError("config must be a ForecastConfigSpec")
        if config.strategy is None:
            raise ValueError("CanonicalBaseModelRunner requires a strategy")
        self.config = config
        self.registry = registry
        self.origin = origin
        self.builder = _RegistryDesignBuilder(config, registry)
        (
            self.X_all,
            self.Y_all,
            self.supervised_origins,
            self.supervised_sample_origins,
            self.supervised_sample_series_ids,
        ) = _supervised_arrays(self.builder, origin)

    @property
    def geometry(self) -> validation.TimeGeometry:
        return validation.TimeGeometry(
            offset=self.builder.offset,
            horizon=self.config.problem.horizon,
        )

    @property
    def series_ids(self) -> tuple[Any, ...]:
        return self.builder.series_ids

    @property
    def feature_schema(self) -> tuple[str, ...]:
        return self.builder.feature_schema

    def backtest_windows(self) -> tuple[_BacktestWindow, ...]:
        if isinstance(self.config.validation.backtest, CalendarMonthBacktestSpec):
            return ()
        return _rolling_backtest_windows(self.builder, self.supervised_origins)

    def fit(
        self,
        train_indices: tuple[int, ...],
    ) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
        Any,
    ]:
        """Fit this member's own transforms and model on given train origins.

        Returns ``(feature_scaler, target_transform, X_train_transformed,
        Y_train_transformed, artifact)``; the artifact is a point
        `CanonicalStrategyArtifact` or a quantile
        `CanonicalMarginalQuantileArtifact` depending on the config mode.
        """
        mode = self._mode()
        train_sample_indices = _sample_indices(
            train_indices,
            self.builder.n_series,
        )
        X_train = tuple(
            design[list(train_sample_indices)] for design in self.X_all
        )
        Y_train = self.Y_all[list(train_sample_indices)]
        training_origins = tuple(
            self.supervised_sample_origins[index]
            for index in train_sample_indices
        )
        training_series_ids = tuple(
            self.supervised_sample_series_ids[index]
            for index in train_sample_indices
        )
        training_history_cutoff = max(
            _label_end(self.builder, self.supervised_origins[index])
            for index in train_indices
        )
        (
            feature_scaler,
            target_transform,
            X_train_transformed,
            Y_train_transformed,
        ) = _fit_runtime_transforms(
            self.config,
            self.builder,
            X_train,
            Y_train,
            training_origins,
            training_series_ids,
            training_history_cutoff,
        )
        # 监督特征选择（2026-08-30 专项）：有监督步骤挂在训练 fit 边界，
        # 每个回测窗口/最终训练各自重拟合，只消费当前训练窗 (X, Y)，无泄漏；
        # 选中集写入 artifact.feature_schema，预测端按同名子集对齐。
        X_train_transformed, feature_schema = self._apply_feature_selection(
            X_train_transformed, Y_train_transformed
        )
        if mode == "point":
            _, artifact = _fit_point(
                self.config,
                feature_schema,
                X_train_transformed,
                Y_train_transformed,
                n_series=self.builder.n_series,
            )
        else:
            _, artifact, _ = _fit_quantile(
                self.config,
                feature_schema,
                X_train_transformed,
                Y_train_transformed,
                n_series=self.builder.n_series,
            )
        return (
            feature_scaler,
            target_transform,
            X_train_transformed,
            Y_train_transformed,
            artifact,
        )

    def forecast_designs(
        self,
        origin: pd.Timestamp,
        feature_scaler: CanonicalFeatureScaler,
        target_transform: CanonicalTargetTransform,
    ) -> tuple[tuple[np.ndarray, ...], Any]:
        return _forecast_designs_with_scaler(
            self.builder,
            origin,
            feature_scaler,
            target_transform,
        )

    def predict(
        self,
        artifact: Any,
        designs: tuple[np.ndarray, ...],
        provider: Any,
        forecast_times: pd.DatetimeIndex,
        target_transform: CanonicalTargetTransform,
    ) -> PointForecastTensor | MarginalForecastDistribution:
        """Predict at the given origin and restore to the original target space."""
        base_design = designs[0]
        # 特征选择对齐（2026-08-30 专项）：artifact.feature_schema 是训练期选中集，
        # 预测端把全 schema 设计矩阵按同名子集对齐（provider 输出同为全 schema 宽）。
        artifact_schema = (
            artifact.feature_schema
            if isinstance(artifact, CanonicalStrategyArtifact)
            else next(iter(artifact.artifacts_by_level.values())).feature_schema
        )
        indices = selected_indices_for_artifact(
            self.builder.feature_schema, tuple(artifact_schema)
        )
        if indices is not None:
            base_design = np.asarray(base_design, dtype=float)[:, indices]
            full_provider = provider

            def selected_provider(call_index, coordinates, dependencies, predicted):
                return np.asarray(
                    full_provider(call_index, coordinates, dependencies, predicted),
                    dtype=float,
                )[:, indices]

            provider = selected_provider

        raw = _predict(
            self.config,
            artifact,
            base_design,
            provider,
            forecast_times,
            self.builder.series_ids,
        )
        return _restore_prediction(raw, target_transform)

    def actual(
        self,
        origin_index: int,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        return _actual_at_origin(
            self.config,
            self.Y_all,
            origin_index,
            forecast_times,
            self.builder.series_ids,
        )

    def seasonal_naive(
        self,
        origin: pd.Timestamp,
        forecast_times: pd.DatetimeIndex,
    ) -> PointForecastTensor:
        return _seasonal_naive_tensor(
            self.builder,
            origin,
            forecast_times,
        )

    def forecast_times(
        self,
        origin: pd.Timestamp,
    ) -> pd.DatetimeIndex:
        return pd.date_range(
            origin,
            periods=self.config.problem.horizon + 1,
            freq=self.config.problem.freq,
        )[1:]

    def final_bundle_inputs(self) -> tuple[
        CanonicalFeatureScaler,
        CanonicalTargetTransform,
        tuple[np.ndarray, ...],
        np.ndarray,
    ]:
        """Fit final transforms under the same explicit window as backtesting."""
        backtest = self.config.validation.backtest
        if isinstance(backtest, FixedStepBacktestSpec):
            first_origin_index = max(
                0,
                len(self.supervised_origins) - backtest.train_window_steps,
            )
            origin_indices = tuple(
                range(first_origin_index, len(self.supervised_origins))
            )
        elif isinstance(backtest, CalendarMonthBacktestSpec):
            raw_history_times = self.builder.target_history_times(self.origin)
            if len(raw_history_times) < backtest.train_window_days:
                raise ValueError(
                    "calendar-month final fit has fewer raw history days than "
                    "validation.train_window_days"
                )
            train_start_time = pd.Timestamp(
                raw_history_times.to_numpy()[-backtest.train_window_days]
            )
            forecast_start = self.geometry.label_start(self.origin)
            origin_indices = tuple(
                index
                for index, candidate in enumerate(self.supervised_origins)
                if candidate >= train_start_time
                and self.geometry.label_end(candidate) < forecast_start
            )
        else:
            raise TypeError("canonical final fit requires typed backtest geometry")
        if not origin_indices:
            raise ValueError("canonical final fit has no safe supervised samples")
        sample_indices = _sample_indices(origin_indices, self.builder.n_series)
        X_window = tuple(
            design[list(sample_indices)] for design in self.X_all
        )
        Y_window = self.Y_all[list(sample_indices)]
        sample_origins = tuple(
            self.supervised_sample_origins[index] for index in sample_indices
        )
        sample_series_ids = tuple(
            self.supervised_sample_series_ids[index] for index in sample_indices
        )
        history_cutoff = max(
            _label_end(self.builder, self.supervised_origins[index])
            for index in origin_indices
        )
        (
            feature_scaler,
            target_transform,
            X_all_transformed,
            Y_all_transformed,
        ) = _fit_runtime_transforms(
            self.config,
            self.builder,
            X_window,
            Y_window,
            sample_origins,
            sample_series_ids,
            history_cutoff,
        )
        return feature_scaler, target_transform, X_all_transformed, Y_all_transformed

    def fit_final(
        self,
        X_transformed: tuple[np.ndarray, ...],
        Y_transformed: np.ndarray,
    ) -> tuple[CanonicalTrainer, Any, Any]:
        """Train the final artifact and return (trainer, artifact, capabilities)."""
        mode = self._mode()
        X_transformed, feature_schema = self._apply_feature_selection(
            X_transformed, Y_transformed
        )
        if mode == "point":
            trainer, artifact = _fit_point(
                self.config,
                feature_schema,
                X_transformed,
                Y_transformed,
                n_series=self.builder.n_series,
            )
            capabilities = trainer.capabilities
        else:
            trainer, artifact, capabilities = _fit_quantile(
                self.config,
                feature_schema,
                X_transformed,
                Y_transformed,
                n_series=self.builder.n_series,
            )
        return trainer, artifact, capabilities

    def build_final_bundle(
        self,
        feature_scaler: CanonicalFeatureScaler,
        target_transform: CanonicalTargetTransform,
        trainer: Any,
        artifact: Any,
        capabilities: Any,
    ) -> ForecastModelBundle:
        """Build a self-contained schema-2 bundle from a completed final fit."""
        mode = self._mode()
        if mode == "point":
            bundle_builder = trainer
            bundle_artifact = artifact
        else:
            point_level = float(self.config.probabilistic.get("point_quantile", 0.5))
            bundle_artifact = artifact.artifacts_by_level[point_level]
            selected_schema = tuple(bundle_artifact.feature_schema)
            bundle_builder = CanonicalTrainer(
                self.config,
                estimator_factory=make_model_factory(
                    self.config.estimator.model_type,
                    self.config.estimator.params,
                    feature_names=selected_schema,
                    quantile=point_level,
                ),
                capabilities=capabilities,
                feature_schema=selected_schema,
            )
        bundle = build_strategy_model_bundle(
            bundle_builder,
            bundle_artifact,
            feature_scaler=feature_scaler,
            target_transform=target_transform,
            input_schema={
                "columns": list(self.feature_schema),
                "panel": {
                    "series_id_cols": list(self.config.problem.series_id_cols),
                    "known_series_ids": [
                        list(value) if isinstance(value, tuple) else value
                        for value in self.series_ids
                    ],
                    "unknown_series_policy": "raise",
                },
            },
            series_ids=self.series_ids,
        )
        if mode == "quantile":
            bundle.model = artifact
        return bundle

    def _apply_feature_selection(
        self,
        X_by_call: tuple[np.ndarray, ...],
        Y: np.ndarray,
    ) -> tuple[tuple[np.ndarray, ...], tuple[str, ...]]:
        """监督特征选择（features.selection，2026-08-30 专项）。

        有监督步骤挂在训练 fit 边界：每个回测窗口与最终训练各自重拟合选择器，
        只消费当前训练窗的 (X, Y)，无泄漏；未配置/未启用时原样直通。
        选中集进入 artifact.feature_schema，预测端按同名子集对齐。
        """
        feature_schema = self.builder.feature_schema
        spec = normalize_feature_selection(self.config.features.selection)
        if spec is None or not spec.enabled:
            return X_by_call, feature_schema
        selector = CanonicalFeatureSelector(spec, feature_schema)
        y_signal = Y.reshape(Y.shape[0], -1).mean(axis=1)
        selector.fit(X_by_call[0], y_signal)
        assert selector.selected_names_ is not None  # fit 后必有选中集
        logger.info(
            "[FeatureSelection] %d -> %d features (method=%s)",
            len(feature_schema),
            len(selector.selected_names_),
            spec.method,
        )
        return (
            tuple(selector.transform(design) for design in X_by_call),
            selector.selected_names_,
        )

    def run(
        self,
        output_root: str | Path | None = None,
    ) -> CanonicalRuntimeResult:
        """Full single-model lifecycle: rolling backtest, final fit, persist."""
        builder = self.builder
        config = self.config
        origin = self.origin
        mode = self._mode()

        fingerprint = config.fingerprint()
        run_dir, model_dir, test_dir, forecast_dir = _output_paths(
            config,
            fingerprint,
            output_root,
        )
        aggregate_weights = resolve_aggregate_weighting(
            config.problem.targets,
            config.validation.get("aggregate_weighting"),
        )
        cv_frames = []
        score_frames = []
        prob_score_frames = []
        eval_mask_config = (
            config.validation.get("eval_mask")
            if isinstance(config.validation.get("eval_mask"), Mapping)
            else None
        )
        holdout_audits = []
        for backtest_window in self.backtest_windows():
            (
                holdout_feature_scaler,
                holdout_target_transform,
                _X_train_transformed,
                _Y_train_transformed,
                holdout_artifact,
            ) = self.fit(backtest_window.train_indices)

            builder.reset_audit()
            holdout_designs, holdout_provider = self.forecast_designs(
                backtest_window.origin,
                holdout_feature_scaler,
                holdout_target_transform,
            )
            holdout_times = self.forecast_times(backtest_window.origin)
            holdout_prediction = self.predict(
                holdout_artifact,
                holdout_designs,
                holdout_provider,
                holdout_times,
                holdout_target_transform,
            )
            actual = self.actual(
                backtest_window.origin_index,
                holdout_times,
            )
            seasonal_naive = self.seasonal_naive(
                backtest_window.origin,
                holdout_times,
            )
            holdout_point = (
                holdout_prediction
                if isinstance(holdout_prediction, PointForecastTensor)
                else holdout_prediction.point
            )
            cv_frames.append(
                backtest_tensors_to_long(
                    actual,
                    holdout_prediction,
                    window=backtest_window.window,
                )
            )
            score_frames.append(
                evaluate_point_forecasts(
                    actual,
                    holdout_point,
                    aggregate_weighting=aggregate_weights,
                    seasonal_naive=seasonal_naive,
                    window=backtest_window.window,
                    eval_mask=eval_mask_config,
                )
            )
            # 概率评估接线（2026-08-30）：quantile 模式逐窗产出 pinball/central 区间
            # 指标；掩码口径与点评估共用同一 eval_mask payload（同一业务口径）。
            if not isinstance(holdout_prediction, PointForecastTensor):
                prob_mask_payload = build_eval_mask_payload(eval_mask_config, actual)
                prob_score_frames.append(
                    evaluate_marginal_distribution(
                        actual,
                        holdout_prediction,
                        valid_masks=(
                            {
                                target: payload["valid_mask"]
                                for target, payload in prob_mask_payload.items()
                            }
                            if prob_mask_payload is not None
                            else None
                        ),
                        window=backtest_window.window,
                    )
                )
            holdout_audits.extend(builder.audit)
        holdout_audit = tuple(holdout_audits)
        if cv_frames:
            backtest = config.validation.backtest
            if not isinstance(backtest, FixedStepBacktestSpec):
                raise TypeError("fixed backtest results require FixedStepBacktestSpec")
            windows = self.backtest_windows()
            holdout_metadata = {
                **windows[-1].metadata,
                "mode": "fixed_steps",
                "history_steps": backtest.history_steps,
                "train_window_steps": backtest.train_window_steps,
                "fold_count": backtest.fold_count,
                "stride_steps": backtest.stride_steps,
                "windows": [window.metadata for window in windows],
            }
            write_backtest_results(
                test_dir,
                pd.concat(cv_frames, ignore_index=True),
                pd.concat(score_frames, ignore_index=True),
                aggregate_weighting=aggregate_weights,
                metadata={"backtest": holdout_metadata},
                probabilistic_scores_df=(
                    pd.concat(prob_score_frames, ignore_index=True)
                    if prob_score_frames
                    else None
                ),
            )
        else:
            holdout_metadata = {
                "mode": "calendar_month",
                "windows": [],
            }

        (
            final_feature_scaler,
            final_target_transform,
            X_all_transformed,
            Y_all_transformed,
        ) = self.final_bundle_inputs()
        final_trainer, final_artifact, final_capabilities = self.fit_final(
            X_all_transformed,
            Y_all_transformed,
        )

        builder.reset_audit()
        final_designs, final_provider = self.forecast_designs(
            origin,
            final_feature_scaler,
            final_target_transform,
        )
        forecast_times = self.forecast_times(origin)
        forecast = self.predict(
            final_artifact,
            final_designs,
            final_provider,
            forecast_times,
            final_target_transform,
        )
        final_audit = builder.audit
        visibility_proof = _proof_payload(final_audit)
        holdout_visibility_proof = _proof_payload(holdout_audit)
        source_lineage = _source_lineage_payload(final_audit)
        feature_lineage, availability_summary = _compiled_lineage(
            builder.feature_schema,
            visibility_proof,
            config,
        )
        if mode == "point":
            bundle_builder = final_trainer
            bundle_artifact = final_artifact
        else:
            point_level = float(config.probabilistic.get("point_quantile", 0.5))
            bundle_builder = CanonicalTrainer(
                config,
                estimator_factory=make_model_factory(
                    config.estimator.model_type,
                    config.estimator.params,
                    feature_names=builder.feature_schema,
                    quantile=point_level,
                ),
                capabilities=final_capabilities,
                feature_schema=builder.feature_schema,
            )
            bundle_artifact = final_artifact.artifacts_by_level[point_level]
        bundle = build_strategy_model_bundle(
            bundle_builder,
            bundle_artifact,
            feature_scaler=final_feature_scaler,
            target_transform=final_target_transform,
            input_schema={
                "columns": list(builder.feature_schema),
                "panel": {
                    "series_id_cols": list(config.problem.series_id_cols),
                    "known_series_ids": [
                        list(value) if isinstance(value, tuple) else value
                        for value in builder.series_ids
                    ],
                    "unknown_series_policy": str(
                        builder._training_scope_validation().get(
                            "unknown_series_policy",
                            "raise",
                        )
                    ).lower(),
                },
                "availability_summary": availability_summary,
                "visibility_proof": visibility_proof,
            },
            feature_lineage=feature_lineage,
            source_lineage=source_lineage,
            series_ids=builder.series_ids,
        )
        if mode == "quantile":
            bundle.model = final_artifact

        persist_model_bundle(bundle, model_dir)

        write_forecast_results(forecast_dir, forecast)
        _write_json(
            forecast_dir / "resolved_config.json",
            {
                **config.canonical_payload(),
                "config_fingerprint": fingerprint,
                "runtime": {
                    "series_order": [
                        list(value) if isinstance(value, tuple) else value
                        for value in builder.series_ids
                    ],
                    "feature_schema": list(builder.feature_schema),
                    "availability_summary": availability_summary,
                    "source_lineage": source_lineage,
                    "visibility_proof": visibility_proof,
                    "holdout_visibility_proof": holdout_visibility_proof,
                    "capability_probe": {
                        "native_multioutput_probed": (
                            config.estimator.target_adapter is TargetAdapter.NATIVE
                        ),
                        "resolved": final_capabilities.canonical_payload(),
                    },
                    "strategy": {
                        "model_count": builder.plan.model_count,
                        "dependencies": [
                            [
                                {
                                    "target": coordinate.target,
                                    "horizon_step": coordinate.horizon_step,
                                }
                                for coordinate in dependencies
                            ]
                            for dependencies in builder.plan.dependencies
                        ],
                    },
                    "holdout": holdout_metadata,
                },
            },
        )
        return CanonicalRuntimeResult(
            run_dir=run_dir,
            model_dir=model_dir,
            test_dir=test_dir,
            forecast_dir=forecast_dir,
            fingerprint=fingerprint,
            bundle=bundle,
        )

    def _mode(self) -> str:
        mode = str(self.config.probabilistic.get("mode", "point"))
        if mode not in {"point", "quantile"}:
            raise ValueError(
                f"unsupported canonical probabilistic mode: {mode!r}"
            )
        return mode




def run_canonical_config(
    config: ForecastConfigSpec,
    output_root: str | Path | None = None,
    *,
    generators: Mapping[str, Any] | None = None,
) -> CanonicalRuntimeResult:
    """Train, backtest, forecast, and persist one canonical config."""
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("config must be a ForecastConfigSpec")
    if config.strategy is None:
        raise ValueError("run_canonical_config requires a strategy or ensemble")
    # builtin generators（chinese_holiday）默认可用；调用方同名注入时覆盖。
    merged_generators: dict[str, Any] = {**BUILTIN_GENERATORS, **(generators or {})}
    registry = SourceRegistry(config.data, Path.cwd(), generators=merged_generators)
    origin = _resolve_origin(registry, config.validation.get("forecast_origin"))
    runner = CanonicalBaseModelRunner(config, registry, origin)
    result = runner.run(output_root)
    if str(config.validation.get("horizon_mode", "fixed_steps")) == "calendar_month":
        overwrite_calendar_month_backtest(
            config, registry, runner, result, runner_factory=CanonicalBaseModelRunner
        )
    return result


__all__ = [
    "CanonicalBaseModelRunner",
    "CanonicalRuntimeResult",
    "persist_model_bundle",
    "run_canonical_config",
]
