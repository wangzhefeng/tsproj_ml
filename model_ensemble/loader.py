"""Loader: resolve member config_refs and enforce the shared contract (v4 §5.3).

The loader never mutates the production `load_yaml_config()` path in E2 —
it is an independent entrypoint used by the ensemble runtime (E5+) and the
config switch in E6.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from forecasting_core.specs.config import (
    parse_data_spec,
    parse_output_spec,
    parse_probabilistic_config_spec,
    parse_problem_spec,
    parse_runtime_validation_spec,
)
from model_ensemble.specs import (
    EnsembleConfigSpec,
    EnsembleSpecError,
    MemberRef,
    enforce_forbidden_top_level,
    parse_ensemble_section,
)


def load_raw_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise EnsembleSpecError(f"config_ref does not exist: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, Mapping):
        raise EnsembleSpecError(f"config is not a mapping: {file_path}")
    return dict(raw)


def parse_ensemble_document(
    raw: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> EnsembleConfigSpec:
    """Parse a raw ensemble YAML mapping into an EnsembleConfigSpec."""
    unknown = set(raw) - {
        "schema_version",
        "problem",
        "data",
        "probabilistic",
        "ensemble",
        "validation",
        "output",
    }
    if unknown:
        raise EnsembleSpecError(
            f"ensemble YAML has unknown top-level fields: {sorted(unknown)}"
        )
    enforce_forbidden_top_level(raw)
    missing = {"schema_version", "problem", "data", "ensemble", "output"} - set(raw)
    if missing:
        raise EnsembleSpecError(
            f"ensemble YAML missing top-level fields: {sorted(missing)}"
        )
    if raw["schema_version"] != 2:
        raise EnsembleSpecError("ensemble YAML schema_version must be 2")

    source = source_path or "<ensemble>"
    problem = parse_problem_spec(raw.get("problem"), source)
    data = parse_data_spec(raw.get("data"), source)
    probabilistic = parse_probabilistic_config_spec(
        raw.get("probabilistic") or {}, source
    )
    validation = parse_runtime_validation_spec(
        raw.get("validation") or {}, source, require_geometry=False
    )
    output = parse_output_spec(raw.get("output") or {}, source)
    calendar_month = str(validation.get("horizon_mode", "fixed_steps")) == "calendar_month"
    members, oof, method = parse_ensemble_section(
        raw["ensemble"], calendar_month=calendar_month
    )
    return EnsembleConfigSpec(
        schema_version=2,
        members=members,
        oof=oof,
        method=method,
        problem=problem,
        data=data,
        probabilistic=probabilistic,
        validation=validation,
        output=output,
    )


def load_ensemble_config(path: str | Path) -> EnsembleConfigSpec:
    return parse_ensemble_document(
        load_raw_yaml(path),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# Member reference contract (v4 §5.3)
# ---------------------------------------------------------------------------

_SHARED_PROBLEM_FIELDS = (
    "time_col",
    "freq",
    "horizon",
    "targets",
    "training_scope",
    "series_id_cols",
)


def _normalized(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_normalized(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalized(value[key]) for key in sorted(value)}
    return value


def _member_problem(member_raw: Mapping[str, Any], ref: MemberRef) -> dict[str, Any]:
    strategy = member_raw.get("strategy")
    if not isinstance(strategy, Mapping) or not strategy.get("name"):
        raise EnsembleSpecError(
            f"member {ref.name!r} ({ref.config_ref}) is not a single-model "
            "canonical config (missing strategy)"
        )
    if member_raw.get("ensemble") is not None:
        raise EnsembleSpecError(
            f"member {ref.name!r} ({ref.config_ref}) is itself an ensemble; "
            "ensemble-of-ensemble is not supported"
        )
    problem = member_raw.get("problem")
    if not isinstance(problem, Mapping):
        raise EnsembleSpecError(
            f"member {ref.name!r} ({ref.config_ref}) has no problem section"
        )
    return dict(problem)


def _quantile_grid(probabilistic: Mapping[str, Any]) -> tuple[Any, ...]:
    mode = str(probabilistic.get("mode", "point"))
    if mode == "point":
        return ()
    levels = probabilistic.get("quantiles") or []
    return tuple(float(level) for level in levels)


def resolve_members(
    ensemble: EnsembleConfigSpec,
    *,
    base_dir: str | Path = ".",
) -> dict[str, dict[str, Any]]:
    """Load each member YAML and enforce the shared contract.

    Returns ``{member_name: raw_member_document}`` after validation.
    """
    root = Path(base_dir)
    problem_payload = ensemble.problem.canonical_payload()
    ensemble_problem = {
        field: _normalized(problem_payload.get(field))
        for field in _SHARED_PROBLEM_FIELDS
    }
    ensemble_quantiles = _quantile_grid(ensemble.probabilistic)
    reference_problem: dict[str, Any] | None = None
    reference_quantiles: tuple[Any, ...] | None = None
    resolved: dict[str, dict[str, Any]] = {}
    for ref in ensemble.members:  # order defines member order everywhere
        if ref.name in resolved:
            raise EnsembleSpecError(f"duplicate member name: {ref.name!r}")
        member_path = (root / ref.config_ref).resolve()
        # cycle guard: an ensemble YAML must never reference itself directly
        if _is_same_file(member_path, root):
            raise EnsembleSpecError(
                f"member {ref.name!r} config_ref cycle detected: {ref.config_ref}"
            )
        member_raw = load_raw_yaml(member_path)
        member_problem = _member_problem(member_raw, ref)
        for field in _SHARED_PROBLEM_FIELDS:
            if _normalized(member_problem.get(field)) != ensemble_problem[field]:
                raise EnsembleSpecError(
                    f"member {ref.name!r} problem.{field} differs from the "
                    "top-level ensemble problem"
                )
        if reference_problem is None:
            reference_problem = {
                field: _normalized(member_problem.get(field))
                for field in _SHARED_PROBLEM_FIELDS
            }
        else:
            for field in _SHARED_PROBLEM_FIELDS:
                if _normalized(member_problem.get(field)) != reference_problem[field]:
                    raise EnsembleSpecError(
                        f"member {ref.name!r} problem.{field} differs from the "
                        "ensemble problem; all members must share the prediction "
                        "problem and information boundary"
                    )
        quantiles = _quantile_grid(
            dict(member_raw.get("probabilistic") or {})
        )
        if quantiles != ensemble_quantiles:
            raise EnsembleSpecError(
                f"member {ref.name!r} probabilistic quantile grid differs from "
                "the top-level ensemble config"
            )
        if reference_quantiles is None:
            reference_quantiles = quantiles
        elif quantiles != reference_quantiles:
            raise EnsembleSpecError(
                f"member {ref.name!r} probabilistic quantile grid differs from "
                "other members"
            )
        resolved[ref.name] = member_raw
    return resolved


def _is_same_file(candidate: Path, base_dir: Path) -> bool:
    try:
        return candidate.exists() and candidate.resolve() == base_dir.resolve()
    except OSError:
        return False


def validate_member_sources(
    ensemble: EnsembleConfigSpec,
    members: Mapping[str, Mapping[str, Any]],
) -> None:
    """Ensemble data is the allowed superset; member sources must be subsets.

    Target/key sources must match the ensemble definition exactly; every other
    member source must be defined identically inside the ensemble data section.
    """
    data_payload = ensemble.data.canonical_payload()
    raw_sources = data_payload["sources"]
    if not isinstance(raw_sources, list):  # DataSpec canonical contract
        raise TypeError("ensemble data.sources payload must be a list")
    ens_sources = {
        str(source["name"]): dict(source)
        for source in raw_sources
        if isinstance(source, Mapping)
    }

    def source_signature(source: Mapping[str, Any]) -> Any:
        return _normalized(source)

    for ref in ensemble.members:
        member_raw = members[ref.name]
        member_sources = {
            str(source.get("name")): dict(source)
            for source in ((member_raw.get("data") or {}).get("sources") or [])
        }
        for name, source in member_sources.items():
            if name not in ens_sources:
                raise EnsembleSpecError(
                    f"member {ref.name!r} source {name!r} is not defined in the "
                    "ensemble data section"
                )
            roles = {
                column.get("name"): column.get("role")
                for column in source.get("columns", [])
            }
            target_key_roles = {
                column_name: role
                for column_name, role in roles.items()
                if role in {"target", "key"}
            }
            ens_source = ens_sources[name]
            ens_roles = {
                column.get("name"): column.get("role")
                for column in ens_source.get("columns", [])
            }
            ens_target_key = {
                column_name: role
                for column_name, role in ens_roles.items()
                if role in {"target", "key"}
            }
            if target_key_roles != ens_target_key:
                raise EnsembleSpecError(
                    f"member {ref.name!r} source {name!r} target/key columns "
                    "differ from the ensemble definition"
                )
            if source_signature(source) != source_signature(ens_source):
                # definitions differ: allowed only if the member uses a strict
                # column subset of the ensemble source (same file/time/roles)
                if not _is_column_subset(source, ens_source):
                    raise EnsembleSpecError(
                        f"member {ref.name!r} source {name!r} is not an identical "
                        "subset of the ensemble data definition"
                    )


def _is_column_subset(
    member_source: Mapping[str, Any], ens_source: Mapping[str, Any]
) -> bool:
    shared = {
        "source_type",
        "history_path",
        "time_col",
        "series_id_cols",
        "availability",
        "available_at_col",
        "backtest_path",
        "future_path",
        "provider",
    }
    for field in shared:
        if member_source.get(field) != ens_source.get(field):
            return False
    member_cols = {
        (column.get("name"), column.get("role"), column.get("categorical"))
        for column in member_source.get("columns", [])
    }
    ens_cols = {
        (column.get("name"), column.get("role"), column.get("categorical"))
        for column in ens_source.get("columns", [])
    }
    return member_cols.issubset(ens_cols)


__all__ = [
    "EnsembleConfigSpec",
    "EnsembleSpecError",
    "MemberRef",
    "load_ensemble_config",
    "parse_ensemble_document",
    "resolve_members",
    "validate_member_sources",
]
