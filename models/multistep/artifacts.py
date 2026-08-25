# -*- coding: utf-8 -*-
"""多步策略模型产物与旧产物兼容适配。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Sequence

from models.multistep.plans import ResolvedStrategy
from models.multistep.weights import BlendWeights


@dataclass(frozen=True)
class MultistepArtifactMetadata:
    method: str
    method_code: str
    horizon: int
    target_steps: tuple[int, ...]
    model_output_width: int
    training_layout: str
    feature_schema: tuple[str, ...] = ()

    @classmethod
    def from_strategy(
        cls,
        strategy: ResolvedStrategy,
        feature_schema: Sequence[str] = (),
    ) -> "MultistepArtifactMetadata":
        return cls(
            method=strategy.spec.method,
            method_code=strategy.spec.code,
            horizon=strategy.horizon,
            target_steps=tuple(strategy.target_plan.label_steps),
            model_output_width=strategy.runtime_plan.model_output_width,
            training_layout=strategy.training_plan.layout.value,
            feature_schema=tuple(str(column) for column in feature_schema),
        )

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyArtifact:
    """普通单模型的自描述容器。"""

    model: Any
    metadata: MultistepArtifactMetadata
    schema_version: int = 1


@dataclass(frozen=True)
class BlendArtifact:
    direct_model: Any
    recursive_model: Any
    weights: Optional[BlendWeights]
    metadata: Optional[MultistepArtifactMetadata] = None
    schema_version: int = 1


@dataclass(frozen=True)
class AuxiliaryEndogenousArtifact:
    main_model: Any
    auxiliary_model: Any
    endogenous_features: tuple[str, ...]
    metadata: Optional[MultistepArtifactMetadata] = None
    schema_version: int = 1


class LegacyArtifactAdapter:
    """把历史裸模型/dict bundle 转为只读运行期产物，不改写原 pkl。"""

    @classmethod
    def adapt(
        cls,
        value: Any,
        strategy: Optional[ResolvedStrategy] = None,
        feature_schema: Sequence[str] = (),
    ) -> Any:
        adapted = cls._adapt_shape(value)
        if strategy is None:
            return adapted

        metadata = MultistepArtifactMetadata.from_strategy(
            strategy,
            feature_schema=feature_schema,
        )
        actual_width = cls._infer_output_width(adapted)
        if actual_width is not None:
            metadata = replace(metadata, model_output_width=actual_width)
        return StrategyArtifact(model=adapted, metadata=metadata)

    @classmethod
    def _adapt_shape(cls, value: Any) -> Any:
        from probabilistic.types import (
            ForecastModelBundle,
            ProbabilisticModelBundle,
            migrate_legacy_quantile_bundle,
        )

        if isinstance(value, ForecastModelBundle):
            cls._require_schema_version(value, "ForecastModelBundle")
            return value
        if isinstance(value, ProbabilisticModelBundle):
            cls._require_schema_version(value, "ProbabilisticModelBundle")
            return value
        if not isinstance(value, dict):
            return value

        bundle_type = value.get("bundle_type")
        if bundle_type == "auxiliary_endogenous":
            main_model = value.get("main")
            auxiliary_model = value.get("aux")
            if main_model is None or auxiliary_model is None:
                raise ValueError(
                    "legacy auxiliary_endogenous bundle requires main and aux models."
                )
            return AuxiliaryEndogenousArtifact(
                main_model=cls._adapt_shape(main_model),
                auxiliary_model=auxiliary_model,
                endogenous_features=tuple(
                    str(column) for column in value.get("endogenous_features", ())
                ),
            )
        if bundle_type == "blend_direct_recursive":
            direct_model = value.get("direct")
            recursive_model = value.get("recursive")
            if direct_model is None or recursive_model is None:
                raise ValueError(
                    "legacy blend_direct_recursive bundle requires direct and recursive models."
                )
            return BlendArtifact(
                direct_model=direct_model,
                recursive_model=recursive_model,
                weights=cls._legacy_blend_weights(value),
            )
        return migrate_legacy_quantile_bundle(value)

    @staticmethod
    def _require_schema_version(value: Any, label: str) -> None:
        version = int(getattr(value, "schema_version", 0))
        if version != 1:
            raise ValueError(f"Unsupported {label} schema_version={version}")

    @staticmethod
    def _legacy_blend_weights(value: dict[str, Any]) -> Optional[BlendWeights]:
        raw = value.get("blend_weights", value.get("weights"))
        if raw is None:
            return None
        if isinstance(raw, BlendWeights):
            return raw
        if isinstance(raw, dict):
            return BlendWeights(
                direct=float(raw["direct"]),
                recursive=float(raw["recursive"]),
                strategy=str(raw.get("strategy", "legacy")),
                calibration_windows=int(raw.get("calibration_windows", 0)),
            )
        values = tuple(float(item) for item in raw)
        if len(values) != 2:
            raise ValueError("legacy blend weights must contain exactly two values.")
        total = values[0] + values[1]
        if total <= 0:
            raise ValueError("legacy blend weights must have a positive sum.")
        return BlendWeights(
            direct=values[0] / total,
            recursive=values[1] / total,
            strategy="legacy",
        )

    @classmethod
    def _infer_output_width(cls, value: Any) -> Optional[int]:
        from probabilistic.types import BlendQuantileModel, ProbabilisticModelBundle

        if isinstance(value, AuxiliaryEndogenousArtifact):
            return cls._infer_output_width(value.main_model)
        if isinstance(value, BlendArtifact):
            return None
        if isinstance(value, ProbabilisticModelBundle):
            model = value.models_by_quantile.get(value.spec.point_quantile)
            if isinstance(model, BlendQuantileModel):
                return None
            return cls._infer_output_width(model)

        n_outputs = getattr(value, "n_outputs_", None)
        if n_outputs is not None:
            width = int(n_outputs)
            return width if width > 0 else None

        class_name = type(value).__name__.lower()
        estimators = getattr(value, "estimators_", None)
        if estimators is not None and (
            "multioutput" in class_name or "regressorchain" in class_name
        ):
            width = len(estimators)
            return width if width > 0 else None

        nested = getattr(value, "model", None)
        if nested is not None and nested is not value:
            return cls._infer_output_width(nested)
        return None
