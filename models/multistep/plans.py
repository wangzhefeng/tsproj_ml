# -*- coding: utf-8 -*-
"""多步策略解析后的不可变计划对象。"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from models.multistep.spec import InputScope, StrategySpec


class TargetLayout(str, Enum):
    POINTWISE = "pointwise"
    SINGLE_OUTPUT = "single_output"
    MULTI_OUTPUT = "multi_output"
    HORIZON_LONG = "horizon_long"
    COMPOSITE = "composite"


class TrainingLayout(str, Enum):
    SINGLE_OUTPUT = "single_output"
    MULTI_OUTPUT = "multi_output"
    HORIZON_LONG = "horizon_long"
    COMPOSITE = "composite"


class LagPolicy(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    SAFE_TARGET_ROW = "safe_target_row"


class RowAlignment(str, Enum):
    FORECAST_ORIGIN = "forecast_origin"
    TARGET_TIME = "target_time"


class ExogenousTiming(str, Enum):
    FORECAST_ORIGIN = "forecast_origin"
    BY_HORIZON = "by_horizon"


class BackfillPolicy(str, Enum):
    PERSISTENCE = "persistence"
    AUXILIARY = "auxiliary"


@dataclass(frozen=True)
class TargetPlan:
    label_steps: Tuple[int, ...]
    column_names: Tuple[str, ...]
    layout: TargetLayout
    direct: Optional["TargetPlan"] = None
    recursive: Optional["TargetPlan"] = None

    @property
    def output_width(self) -> int:
        return len(self.column_names)


@dataclass(frozen=True)
class FeaturePlan:
    input_scope: InputScope
    lag_policy: LagPolicy
    row_alignment: RowAlignment
    exogenous_timing: ExogenousTiming
    backfill_policy: Optional[BackfillPolicy]
    panel_key: Optional[str]


@dataclass(frozen=True)
class TrainingPlan:
    layout: TrainingLayout
    output_width: int
    direct: Optional["TrainingPlan"] = None
    recursive: Optional["TrainingPlan"] = None


@dataclass(frozen=True)
class RuntimePlan:
    executor_key: str
    forecast_length: int
    model_output_width: int
    block_size: Optional[int] = None
    direct: Optional["RuntimePlan"] = None
    recursive: Optional["RuntimePlan"] = None


@dataclass(frozen=True)
class ResolvedStrategy:
    spec: StrategySpec
    horizon: int
    target_plan: TargetPlan
    feature_plan: FeaturePlan
    training_plan: TrainingPlan
    runtime_plan: RuntimePlan

    def __post_init__(self) -> None:
        if self.target_plan.output_width != self.training_plan.output_width:
            raise ValueError("TargetPlan and TrainingPlan output widths differ.")
        if self.training_plan.output_width != self.runtime_plan.model_output_width:
            raise ValueError("TrainingPlan and RuntimePlan output widths differ.")
        if self.runtime_plan.forecast_length != self.horizon:
            raise ValueError("RuntimePlan forecast length differs from horizon.")
