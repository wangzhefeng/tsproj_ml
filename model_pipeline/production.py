from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_loading.registry import SourceRegistry
from forecasting_core.specs.config import ForecastConfigSpec


@dataclass(frozen=True, slots=True)
class ProductionForecastRequest:
    issue_date: str

    def __post_init__(self) -> None:
        if not isinstance(self.issue_date, str) or re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", self.issue_date) is None:
            raise ValueError("production issue_date must be YYYY-MM-DD")
        try:
            pd.Timestamp(self.issue_date).normalize()
        except ValueError as exc:
            raise ValueError("production issue_date must be a valid calendar date") from exc

    @property
    def issue_time(self) -> pd.Timestamp:
        return pd.Timestamp(self.issue_date).normalize()

    @property
    def target_month(self) -> str:
        return str(self.issue_time.to_period("M") + 1)

    @property
    def run_scope(self) -> str:
        return f"target_month={self.target_month}"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_production_origin(
    config: ForecastConfigSpec,
    registry: SourceRegistry,
    request: ProductionForecastRequest,
) -> tuple[pd.Timestamp, dict[str, Any]]:
    if not isinstance(request, ProductionForecastRequest):
        raise TypeError("request must be a ProductionForecastRequest")
    coverage = registry.target_history_coverage()
    latest = []
    source_evidence = []
    for item in coverage:
        series_evidence = []
        for series_id, times in item.times_by_series.items():
            if times.empty or times.hasnans:
                raise ValueError(f"target source {item.source_name!r} has empty series coverage")
            if times.has_duplicates or not times.is_monotonic_increasing:
                raise ValueError(f"target source {item.source_name!r} has duplicate or unordered timestamps")
            expected = pd.date_range(times[0], times[-1], freq=config.problem.freq)
            if not times.equals(expected):
                raise ValueError(
                    f"target source {item.source_name!r} has incomplete time-grid coverage"
                )
            latest.append(times[-1])
            series_evidence.append({
                "series_id": str(series_id),
                "first_time": times[0].isoformat(),
                "last_time": times[-1].isoformat(),
                "rows": len(times),
                "grid_complete": True,
            })
        source = next(source for source in config.data.sources if source.name == item.source_name)
        if source.history_path is None:
            raise ValueError(f"target source {item.source_name!r} has no history_path")
        path = Path(source.history_path)
        if not path.is_absolute():
            path = registry.base_dir / path
        path = path.resolve()
        source_evidence.append({
            "source_name": item.source_name,
            "history_path": str(path),
            "sha256": _source_sha256(path),
            "series": series_evidence,
        })
    if not latest:
        raise ValueError("production requires target history")
    origin = min(latest)
    if config.validation.get("horizon_mode") == "calendar_month" and config.problem.freq == "1D":
        target_start = pd.Period(request.target_month, freq="M").start_time
        if origin >= target_start:
            raise ValueError(
                "production data origin has entered the requested target month; "
                "a complete target-month forecast is no longer as-of"
            )
    evidence = {
        "mode": "production",
        "origin_policy": "last_complete_target",
        "issue_date": request.issue_date,
        "target_month": request.target_month,
        "resolved_origin": origin.isoformat(),
        "run_scope": request.run_scope,
        "sources": source_evidence,
    }
    return origin, evidence


__all__ = ["ProductionForecastRequest", "resolve_production_origin"]
