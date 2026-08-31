"""Canonical multi-step forecasting strategy specifications."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class StrategyName(str, Enum):
    RECURSIVE = "recursive"
    DIRECT = "direct"
    MIMO = "mimo"
    RECMO = "recmo"
    DIRREC = "dirrec"
    DIRMO = "dirmo"
    DIRRECMO = "dirrecmo"


class _PlanShape(Enum):
    SINGLE_STEP_SINGLE_MODEL = "single_step_single_model"
    SINGLE_STEP_PER_HORIZON = "single_step_per_horizon"
    FULL_HORIZON_SINGLE_MODEL = "full_horizon_single_model"
    CHUNK_SINGLE_MODEL = "chunk_single_model"
    CHUNK_PER_BLOCK = "chunk_per_block"


@dataclass(frozen=True, slots=True)
class _StrategyRule:
    plan_shape: _PlanShape
    consumes_previous: bool


_STRATEGY_RULES = MappingProxyType(
    {
        StrategyName.RECURSIVE: _StrategyRule(_PlanShape.SINGLE_STEP_SINGLE_MODEL, True),
        StrategyName.DIRECT: _StrategyRule(_PlanShape.SINGLE_STEP_PER_HORIZON, False),
        StrategyName.MIMO: _StrategyRule(_PlanShape.FULL_HORIZON_SINGLE_MODEL, False),
        StrategyName.RECMO: _StrategyRule(_PlanShape.CHUNK_SINGLE_MODEL, True),
        StrategyName.DIRREC: _StrategyRule(_PlanShape.SINGLE_STEP_PER_HORIZON, True),
        StrategyName.DIRMO: _StrategyRule(_PlanShape.CHUNK_PER_BLOCK, False),
        StrategyName.DIRRECMO: _StrategyRule(_PlanShape.CHUNK_PER_BLOCK, True),
    }
)
_CHUNK_SHAPES = frozenset({_PlanShape.CHUNK_SINGLE_MODEL, _PlanShape.CHUNK_PER_BLOCK})


def _normalize_name(value: StrategyName | str) -> StrategyName:
    if isinstance(value, StrategyName):
        return value
    if not isinstance(value, str):
        raise TypeError("name must be a StrategyName or string")
    try:
        return StrategyName(value)
    except ValueError as exc:
        raise ValueError(f"unknown strategy name: {value!r}") from exc


@dataclass(frozen=True, slots=True, init=False)
class _ResolvedStrategyPlan:
    name: StrategyName
    horizon: int
    output_chunk_length: int | None = None

    def __init__(self) -> None:
        raise TypeError("resolved strategy plans are created by ForecastStrategySpec.resolve()")

    @property
    def model_count(self) -> int:
        plan_shape = _STRATEGY_RULES[self.name].plan_shape
        if plan_shape in {
            _PlanShape.SINGLE_STEP_SINGLE_MODEL,
            _PlanShape.FULL_HORIZON_SINGLE_MODEL,
            _PlanShape.CHUNK_SINGLE_MODEL,
        }:
            return 1
        if plan_shape is _PlanShape.SINGLE_STEP_PER_HORIZON:
            return self.horizon
        return self.n_calls

    @property
    def steps_per_call(self) -> int:
        plan_shape = _STRATEGY_RULES[self.name].plan_shape
        if plan_shape is _PlanShape.FULL_HORIZON_SINGLE_MODEL:
            return self.horizon
        if plan_shape in _CHUNK_SHAPES:
            assert self.output_chunk_length is not None
            return self.output_chunk_length
        return 1

    @property
    def consumes_previous(self) -> bool:
        return _STRATEGY_RULES[self.name].consumes_previous

    @property
    def n_calls(self) -> int:
        return self.horizon // self.steps_per_call


@dataclass(frozen=True, slots=True, init=False)
class ForecastStrategySpec:
    name: StrategyName
    output_chunk_length: int | None = None

    def __init__(
        self,
        name: StrategyName | str,
        output_chunk_length: int | None = None,
    ) -> None:
        normalized_name = _normalize_name(name)
        requires_chunk = _STRATEGY_RULES[normalized_name].plan_shape in _CHUNK_SHAPES
        if requires_chunk:
            if output_chunk_length is None:
                raise ValueError(f"{normalized_name.value} requires output_chunk_length")
            if isinstance(output_chunk_length, bool) or not isinstance(output_chunk_length, int):
                raise TypeError("output_chunk_length must be an integer")
        elif output_chunk_length is not None:
            raise ValueError(f"{normalized_name.value} forbids output_chunk_length")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "output_chunk_length", output_chunk_length)

    def resolve(self, horizon: int) -> _ResolvedStrategyPlan:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon must be an integer")
        if horizon <= 0:
            raise ValueError("horizon must be positive")

        if self.output_chunk_length is not None:
            if not 1 < self.output_chunk_length < horizon:
                raise ValueError("output_chunk_length must satisfy 1 < B < horizon")
            if horizon % self.output_chunk_length != 0:
                raise ValueError("horizon must be divisible by output_chunk_length")

        plan = object.__new__(_ResolvedStrategyPlan)
        object.__setattr__(plan, "name", self.name)
        object.__setattr__(plan, "horizon", horizon)
        object.__setattr__(plan, "output_chunk_length", self.output_chunk_length)
        return plan

    def canonical_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {"name": self.name.value}
        if self.output_chunk_length is not None:
            payload["output_chunk_length"] = self.output_chunk_length
        return payload
