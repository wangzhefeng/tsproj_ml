"""In-memory fold-transform cache for canonical batch execution."""
from __future__ import annotations

import hashlib
import json
from typing import Sequence

from forecasting_core.specs import ForecastConfigSpec
from model_forecasting.batch_memory import BoundedPayloadCache


class FoldTransformCache(BoundedPayloadCache):
    """Single-flight cache keyed by raw design, train rows, and transform semantics."""


def fold_transform_fingerprint(
    config: ForecastConfigSpec,
    *,
    raw_design_fingerprint: str,
    origin_indices: Sequence[int],
) -> str:
    """Hash every semantic input consumed before estimator fitting."""
    payload = {
        "schema_version": 1,
        "raw_design_fingerprint": raw_design_fingerprint,
        "origin_indices": [int(value) for value in origin_indices],
        "problem": {
            "freq": config.problem.freq,
            "targets": list(config.problem.targets),
            "training_scope": config.problem.training_scope,
            "series_id_cols": list(config.problem.series_id_cols),
        },
        "transformations": config.features.canonical_payload().get(
            "transformations", {}
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["FoldTransformCache", "fold_transform_fingerprint"]
