"""Typed specs for reference-based ensemble configurations (v4 §5.2).

Ensemble YAML allowed top-level fields: `schema_version`, `problem`, `data`,
`probabilistic`, `ensemble` (members/oof/method), `validation`, `output`.
`features`, `strategy` and `estimator` are FORBIDDEN (raising on any of them,
including `strategy: null`) — those dimensions belong to each referenced base
model YAML.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from forecasting_core.specs.data import DataSpec
from forecasting_core.specs.output import OutputSpec
from forecasting_core.specs.problem import ForecastProblemSpec
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec
from forecasting_core.specs.validation import RuntimeValidationSpec

ENSEMBLE_ALLOWED_TOP_LEVEL = frozenset(
    {
        "schema_version",
        "problem",
        "data",
        "probabilistic",
        "ensemble",
        "validation",
        "output",
    }
)
ENSEMBLE_FORBIDDEN_TOP_LEVEL = frozenset({"features", "strategy", "estimator"})
ENSEMBLE_ALLOWED_FIELDS = frozenset({"members", "oof", "method"})
OOF_ALLOWED_FIELDS = frozenset(
    {"train_window_steps", "fold_count", "stride_steps", "gap_steps"}
)
METHOD_NAMES = frozenset(
    {"averaging", "weighted", "linear_blending", "stacking"}
)


class EnsembleSpecError(ValueError):
    """Raised for any ensemble schema/contract violation."""


def _require_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise EnsembleSpecError(f"model_ensemble.{label} must be a nonblank string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EnsembleSpecError(f"model_ensemble.oof.{label} must be a positive integer")
    return value


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EnsembleSpecError(
            f"model_ensemble.oof.{label} must be a non-negative integer"
        )
    return value


@dataclass(frozen=True, slots=True)
class MemberRef:
    """Reference to one standalone single-model YAML."""

    name: str
    config_ref: str

    def __post_init__(self) -> None:
        _require_str(self.name, "member.name")
        _require_str(self.config_ref, "member.config_ref")

    def payload(self) -> dict[str, str]:
        return {"name": self.name, "config_ref": self.config_ref}


@dataclass(frozen=True, slots=True)
class OOFSpec:
    """Shared rolling-origin fold geometry for the fuser (v4 §7.1)."""

    train_window_steps: int
    fold_count: int
    stride_steps: int
    gap_steps: int = 0
    calendar_month: bool = False

    def __post_init__(self) -> None:
        _positive_int(self.train_window_steps, "train_window_steps")
        _positive_int(self.fold_count, "fold_count")
        _positive_int(self.stride_steps, "stride_steps")
        _non_negative_int(self.gap_steps, "gap_steps")
        # v4 B2: variable-length calendar-month horizons only define the
        # single-holdout geometry; multi-fold is explicitly unsupported
        if self.calendar_month and self.fold_count > 1:
            raise EnsembleSpecError(
                "model_ensemble.oof.fold_count must be 1 for calendar_month "
                "horizon_mode; multi-fold variable-length geometry is not "
                "supported"
            )

    def payload(self) -> dict[str, int | bool]:
        return {
            "train_window_steps": self.train_window_steps,
            "fold_count": self.fold_count,
            "stride_steps": self.stride_steps,
            "gap_steps": self.gap_steps,
            "calendar_month": self.calendar_month,
        }


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """Fusion method and its parameters (v4 §3)."""

    name: str
    params: dict[str, Any]

    def __post_init__(self) -> None:
        _require_str(self.name, "method.name")
        if self.name not in METHOD_NAMES:
            raise EnsembleSpecError(
                f"model_ensemble.method.name must be one of {sorted(METHOD_NAMES)}; "
                f"got {self.name!r}"
            )
        if not isinstance(self.params, dict):
            raise EnsembleSpecError("model_ensemble.method.params must be a mapping")

    def payload(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


@dataclass(frozen=True, slots=True)
class EnsembleConfigSpec:
    """Parsed reference-based ensemble configuration."""

    schema_version: int
    members: tuple[MemberRef, ...]
    oof: OOFSpec
    method: MethodSpec
    problem: ForecastProblemSpec
    data: DataSpec
    probabilistic: ProbabilisticConfigSpec
    validation: RuntimeValidationSpec
    output: OutputSpec

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise EnsembleSpecError(
                "ensemble config schema_version must be 2"
            )
        expected_types = (
            ("problem", self.problem, ForecastProblemSpec),
            ("data", self.data, DataSpec),
            ("probabilistic", self.probabilistic, ProbabilisticConfigSpec),
            ("validation", self.validation, RuntimeValidationSpec),
            ("output", self.output, OutputSpec),
        )
        for field_name, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(f"{field_name} must be {expected_type.__name__}")
        if self.problem.targets != self.data.target_columns:
            raise EnsembleSpecError(
                "problem.targets must exactly match data.target_columns in the same order"
            )
        if not isinstance(self.members, tuple) or len(self.members) < 2:
            raise EnsembleSpecError("ensemble requires at least two members")
        names = [member.name for member in self.members]
        refs = [member.config_ref for member in self.members]
        if len(names) != len(set(names)):
            raise EnsembleSpecError("ensemble member names must be unique")
        if len(refs) != len(set(refs)):
            raise EnsembleSpecError("ensemble member config_ref values must be unique")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "members": [member.payload() for member in self.members],
            "oof": self.oof.payload(),
            "method": self.method.payload(),
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "problem": self.problem.canonical_payload(),
            "data": self.data.canonical_payload(),
            "probabilistic": self.probabilistic.canonical_payload(),
            "ensemble": {
                "members": [member.payload() for member in self.members],
                "oof": self.oof.payload(),
                "method": self.method.payload(),
            },
            "validation": self.validation.canonical_payload(),
            "output": self.output.canonical_payload(),
        }

    def fingerprint(self) -> str:
        payload = self.canonical_payload()
        payload.pop("output", None)
        payload["validation"] = self.validation.semantic_payload()
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def result_identity(self, hash_length: int = 12) -> str:
        if isinstance(hash_length, bool) or not isinstance(hash_length, int):
            raise TypeError("hash_length must be an integer")
        if not 8 <= hash_length <= 64:
            raise ValueError("hash_length must be between 8 and 64")
        scope = self.problem.training_scope
        target_count = len(self.problem.targets)
        return "-".join(
            (
                "ensemble",
                self.method.name,
                scope,
                f"k{target_count}",
                self.fingerprint()[:hash_length],
            )
        )


def parse_ensemble_member(payload: Any) -> MemberRef:
    if not isinstance(payload, Mapping):
        raise EnsembleSpecError("model_ensemble.members entries must be mappings")
    unknown = set(payload) - {"name", "config_ref"}
    if unknown:
        raise EnsembleSpecError(
            f"model_ensemble.members has unknown fields: {sorted(unknown)}"
        )
    missing = {"name", "config_ref"} - set(payload)
    if missing:
        raise EnsembleSpecError(
            f"model_ensemble.members missing fields: {sorted(missing)}"
        )
    return MemberRef(
        name=str(payload["name"]),
        config_ref=str(payload["config_ref"]),
    )


def parse_oof_spec(payload: Any, *, calendar_month: bool) -> OOFSpec:
    if not isinstance(payload, Mapping):
        raise EnsembleSpecError("model_ensemble.oof must be a mapping")
    unknown = set(payload) - OOF_ALLOWED_FIELDS
    if unknown:
        raise EnsembleSpecError(
            f"model_ensemble.oof has unknown fields: {sorted(unknown)}"
        )
    missing = {
        "train_window_steps",
        "fold_count",
        "stride_steps",
    } - set(payload)
    if missing:
        raise EnsembleSpecError(
            f"model_ensemble.oof missing fields: {sorted(missing)}"
        )
    return OOFSpec(
        train_window_steps=payload["train_window_steps"],
        fold_count=payload["fold_count"],
        stride_steps=payload["stride_steps"],
        gap_steps=payload.get("gap_steps", 0),
        calendar_month=calendar_month,
    )


def parse_method_spec(payload: Any) -> MethodSpec:
    if isinstance(payload, str):
        return MethodSpec(name=payload, params={})
    if not isinstance(payload, Mapping):
        raise EnsembleSpecError("model_ensemble.method must be a mapping or string")
    unknown = set(payload) - {"name", "params"}
    if unknown:
        raise EnsembleSpecError(
            f"model_ensemble.method has unknown fields: {sorted(unknown)}"
        )
    if "name" not in payload:
        raise EnsembleSpecError("model_ensemble.method missing field: name")
    params = payload.get("params", {})
    if not isinstance(params, Mapping):
        raise EnsembleSpecError("model_ensemble.method.params must be a mapping")
    return MethodSpec(name=payload["name"], params=dict(params))


def parse_ensemble_section(payload: Any, *, calendar_month: bool) -> tuple[
    tuple[MemberRef, ...],
    OOFSpec,
    MethodSpec,
]:
    if not isinstance(payload, Mapping):
        raise EnsembleSpecError("ensemble section must be a mapping")
    unknown = set(payload) - ENSEMBLE_ALLOWED_FIELDS
    if unknown:
        raise EnsembleSpecError(
            f"ensemble section has unknown fields: {sorted(unknown)}"
        )
    members_payload = payload.get("members")
    if not isinstance(members_payload, list) or len(members_payload) < 2:
        raise EnsembleSpecError("model_ensemble.members must be a list with >= 2 entries")
    members = tuple(parse_ensemble_member(item) for item in members_payload)
    names = [member.name for member in members]
    refs = [member.config_ref for member in members]
    if len(names) != len(set(names)):
        raise EnsembleSpecError("ensemble member names must be unique")
    if len(refs) != len(set(refs)):
        raise EnsembleSpecError(
            "ensemble member config_ref values must be unique"
        )
    if "oof" not in payload:
        raise EnsembleSpecError("model_ensemble.oof is required")
    oof = parse_oof_spec(payload["oof"], calendar_month=calendar_month)
    if "method" not in payload:
        raise EnsembleSpecError("model_ensemble.method is required")
    method = parse_method_spec(payload["method"])
    return members, oof, method


def enforce_forbidden_top_level(raw: Mapping[str, Any]) -> None:
    """Raise when any single-model-only top-level field is present (v4 §5.2)."""
    present_forbidden = sorted(set(raw) & ENSEMBLE_FORBIDDEN_TOP_LEVEL)
    if present_forbidden:
        raise EnsembleSpecError(
            "ensemble YAML forbids single-model top-level fields "
            f"{present_forbidden}; these belong to each member base-model YAML "
            "(strategy: null is not accepted — remove the field)"
        )


__all__ = [
    "ENSEMBLE_ALLOWED_TOP_LEVEL",
    "ENSEMBLE_FORBIDDEN_TOP_LEVEL",
    "EnsembleConfigSpec",
    "EnsembleSpecError",
    "MemberRef",
    "MethodSpec",
    "OOFSpec",
    "enforce_forbidden_top_level",
    "parse_ensemble_section",
]
