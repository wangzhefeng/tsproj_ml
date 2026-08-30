"""ensemble: reference-based model fusion on top of canonical base models.

Public typed surface (v4 §4); algorithm implementations live in
`model_ensemble.methods`, orchestration in `model_ensemble.runtime` (E5+).
"""

from model_ensemble.specs import (
    ENSEMBLE_ALLOWED_TOP_LEVEL,
    ENSEMBLE_FORBIDDEN_TOP_LEVEL,
    EnsembleConfigSpec,
    EnsembleSpecError,
    MemberRef,
    MethodSpec,
    OOFSpec,
    enforce_forbidden_top_level,
    parse_ensemble_section,
)
from model_ensemble.loader import (
    load_ensemble_config,
    parse_ensemble_document,
    resolve_members,
    validate_member_sources,
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
    "load_ensemble_config",
    "parse_ensemble_document",
    "parse_ensemble_section",
    "resolve_members",
    "validate_member_sources",
]
