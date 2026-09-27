"""Canonical evidence assembly for lifecycle artifacts.

自 ``pipeline/lifecycle.py`` 迁入（2026-09-27 证据域聚集）：四个纯
组装函数只消费 ``CompiledFeatures`` / feature_schema / config，输出产物
元数据 dict，不触碰 runner 状态；lifecycle 保留调用点，实现与合同在
证据域唯一维护。函数实现逐字保真，行为零变化。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, cast

from data_loading.information.information_set import WeatherSourceLineage
from feature_engineering import CompiledFeatures
from forecasting_core.specs import ForecastConfigSpec


def proof_payload(compiled_items: tuple[CompiledFeatures, ...]) -> list[dict[str, Any]]:
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


def holdout_proof_summary(
    compiled_items: tuple[CompiledFeatures, ...],
) -> dict[str, Any]:
    group_by = (
        "forecast_origin",
        "feature_name",
        "source_name",
        "role",
        "provider",
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    total_lookups = 0
    for compiled in compiled_items:
        for proof in compiled.visibility_proof:
            total_lookups += 1
            key = (
                proof.forecast_origin,
                proof.feature_name,
                proof.source_name,
                proof.role,
                proof.provider,
            )
            item = grouped.get(key)
            if item is None:
                item = {
                    "forecast_origin": proof.forecast_origin,
                    "feature_name": proof.feature_name,
                    "source_name": proof.source_name,
                    "role": proof.role,
                    "provider": proof.provider,
                    "lookup_count": 0,
                    "horizon_step_min": proof.horizon_step,
                    "horizon_step_max": proof.horizon_step,
                    "target_time_min": proof.target_time,
                    "target_time_max": proof.target_time,
                    "source_time_count": 0,
                    "source_time_min": proof.source_time,
                    "source_time_max": proof.source_time,
                    "available_at_min": proof.available_at,
                    "available_at_max": proof.available_at,
                }
                grouped[key] = item
            item["lookup_count"] += 1
            item["horizon_step_min"] = min(
                item["horizon_step_min"], proof.horizon_step
            )
            item["horizon_step_max"] = max(
                item["horizon_step_max"], proof.horizon_step
            )
            item["target_time_min"] = min(item["target_time_min"], proof.target_time)
            item["target_time_max"] = max(item["target_time_max"], proof.target_time)
            item["available_at_min"] = min(
                item["available_at_min"], proof.available_at
            )
            item["available_at_max"] = max(
                item["available_at_max"], proof.available_at
            )
            if proof.source_time is not None:
                item["source_time_count"] += 1
                item["source_time_min"] = (
                    proof.source_time
                    if item["source_time_min"] is None
                    else min(item["source_time_min"], proof.source_time)
                )
                item["source_time_max"] = (
                    proof.source_time
                    if item["source_time_max"] is None
                    else max(item["source_time_max"], proof.source_time)
                )

    serialized = []
    timestamp_fields = (
        "forecast_origin",
        "target_time_min",
        "target_time_max",
        "source_time_min",
        "source_time_max",
        "available_at_min",
        "available_at_max",
    )
    for grouped_item in grouped.values():
        item = dict(grouped_item)
        for field in timestamp_fields:
            value = item[field]
            item[field] = value.isoformat() if value is not None else None
        serialized.append(item)
    return {
        "group_by": list(group_by),
        "total_lookups": total_lookups,
        "group_count": len(serialized),
        "groups": serialized,
    }


def source_lineage_payload(
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
                "includes_target_labels": lineage.includes_target_labels,
            }
            if isinstance(lineage, WeatherSourceLineage):
                item["weather_evidence"] = lineage.weather_evidence
            key = tuple(item.items())
            if key not in seen:
                seen.add(key)
                payload.append(item)
    return payload


def compiled_lineage(
    feature_schema: tuple[str, ...],
    proof_payload_items: list[dict[str, Any]],
    config: ForecastConfigSpec,
) -> tuple[tuple[dict[str, Any], ...], dict[str, list[str]]]:
    by_feature = {}
    for item in proof_payload_items:
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
        availability = cast(
            str,
            "known_future"
            if proof["source_name"] == "calendar"
            else source_availability.get(proof["source_name"], proof["role"]),
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
