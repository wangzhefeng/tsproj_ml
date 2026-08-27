from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_START_TIME = "2025-10-01 00:00:00"
DEFAULT_END_TIME = "2026-07-31 23:55:00"
DEFAULT_FREQ = "5min"
AUDIT_FILENAME = "A1_A2_A3_points_aggregate_audit.json"
MAIN_HVAC_REMARKS = frozenset({"冷水机组", "冷却塔", "冷冻水一次泵", "冷冻水二次泵"})
ROUTES = ("A", "B")


@dataclass(frozen=True)
class OutputSpec:
    name: str
    file_stem: str
    sheet_name: str
    data_types: tuple[str, ...]
    route: str
    remarks_include: frozenset[str] | None = None
    remarks_exclude: frozenset[str] | None = None


@dataclass(frozen=True)
class LoadedPoint:
    data_type: str
    spot_id: str
    source_path: Path
    series: pd.Series
    stats: dict[str, Any]


def build_output_specs() -> list[OutputSpec]:
    """返回按楼栋、A/B 路和三楼合计划分的 24 个输出规格。"""
    buildings = ("A1", "A2", "A3")
    specs: list[OutputSpec] = []

    for route in ROUTES:
        for building in buildings:
            specs.append(
                OutputSpec(
                    name=f"{building}楼{route}路暖通电力总负荷-主要设备",
                    file_stem=f"{building.lower()}_route_{route.lower()}_hvac_main_load",
                    sheet_name="暖通负荷",
                    data_types=(f"{building}楼暖通电力负荷",),
                    route=route,
                    remarks_include=MAIN_HVAC_REMARKS,
                )
            )
        specs.append(
            OutputSpec(
                name=f"A1+A2+A3 {route}路暖通电力总负荷-主要设备",
                file_stem=f"a1_a2_a3_route_{route.lower()}_hvac_main_load",
                sheet_name="暖通负荷",
                data_types=tuple(f"{building}楼暖通电力负荷" for building in buildings),
                route=route,
                remarks_include=MAIN_HVAC_REMARKS,
            )
        )

    for route in ROUTES:
        for building in buildings:
            specs.append(
                OutputSpec(
                    name=f"{building}楼{route}路UPS总负荷",
                    file_stem=f"{building.lower()}_route_{route.lower()}_ups_load",
                    sheet_name="UPS负荷",
                    data_types=(f"{building}楼UPS负荷",),
                    route=route,
                    remarks_exclude=frozenset({"一楼"}),
                )
            )
        specs.append(
            OutputSpec(
                name=f"A1+A2+A3 {route}路UPS总负荷",
                file_stem=f"a1_a2_a3_route_{route.lower()}_ups_load",
                sheet_name="UPS负荷",
                data_types=tuple(f"{building}楼UPS负荷" for building in buildings),
                route=route,
                remarks_exclude=frozenset({"一楼"}),
            )
        )

    for route in ROUTES:
        for building in buildings:
            specs.append(
                OutputSpec(
                    name=f"{building}楼{route}路列头柜总负荷",
                    file_stem=f"{building.lower()}_route_{route.lower()}_rpp_load",
                    sheet_name="列头柜负荷",
                    data_types=(f"{building}楼列头柜负荷",),
                    route=route,
                )
            )
        specs.append(
            OutputSpec(
                name=f"A1+A2+A3 {route}路列头柜总负荷",
                file_stem=f"a1_a2_a3_route_{route.lower()}_rpp_load",
                sheet_name="列头柜负荷",
                data_types=tuple(f"{building}楼列头柜负荷" for building in buildings),
                route=route,
            )
        )
    return specs


def _build_legacy_output_names() -> frozenset[str]:
    categories = (
        "暖通电力总负荷",
        "暖通电力总负荷-主要设备",
        "UPS总负荷",
        "列头柜总负荷",
    )
    names = {
        f"{building}楼{category}"
        for building in ("A1", "A2", "A3")
        for category in categories
    }
    names.update(f"A1+A2+A3 {category}" for category in categories)
    return frozenset(names)


LEGACY_OUTPUT_NAMES = _build_legacy_output_names()


def _select_metadata(frame: pd.DataFrame, spec: OutputSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = {"data_type", "deviceId", "spot_id", "备注", "route"}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Sheet '{spec.sheet_name}' missing columns: {missing_columns}")
    if "spotName" not in frame.columns and "spot_name" not in frame.columns:
        raise ValueError(f"Sheet '{spec.sheet_name}' must contain spotName or spot_name")

    selected = frame.loc[frame["data_type"].isin(spec.data_types)].copy()
    if spec.remarks_include is not None:
        selected = selected.loc[selected["备注"].isin(spec.remarks_include)]
    if spec.remarks_exclude is not None:
        selected = selected.loc[~selected["备注"].isin(spec.remarks_exclude)]
    selected = selected.loc[selected["spot_id"].notna()]
    selected["route"] = selected["route"].astype("string").str.strip().str.upper()
    invalid_routes = selected.loc[~selected["route"].isin(ROUTES), "route"].fillna("<NA>").unique().tolist()
    if invalid_routes:
        raise ValueError(f"Sheet '{spec.sheet_name}' contains invalid route values: {invalid_routes}")
    selected = selected.loc[selected["route"] == spec.route]
    selected["data_type"] = selected["data_type"].astype(str)
    selected["deviceId"] = selected["deviceId"].astype(str)
    selected["spot_id"] = selected["spot_id"].astype(str)

    reference = selected.reset_index(drop=True)
    labels = pd.Series("", index=reference.index, dtype="string")
    for column in ("spotName", "spot_name"):
        if column in reference.columns:
            labels = labels.str.cat(reference[column].fillna("").astype(str), sep=" ")
    total_power = labels.str.contains("总", regex=False) & labels.str.contains("功率", regex=False)
    phase_power = labels.str.contains(r"[ABC]相.*功率", regex=True)
    devices_with_total = set(reference.loc[total_power, "deviceId"])
    excluded_mask = phase_power & reference["deviceId"].isin(devices_with_total)
    excluded_phase = reference.loc[excluded_mask].reset_index(drop=True)
    preferred = reference.loc[~excluded_mask].reset_index(drop=True)
    return preferred, excluded_phase, reference


def _load_point(
    *,
    data_type: str,
    spot_id: str,
    source_path: Path,
    full_index: pd.DatetimeIndex,
    source_unit: str,
    unit_scale_to_kw: float,
) -> LoadedPoint:
    frame = pd.read_csv(source_path)
    missing_columns = sorted({"time", "value"} - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Point CSV '{source_path}' missing columns: {missing_columns}")

    source_rows = len(frame)
    parsed_time = pd.Series(pd.to_datetime(frame["time"], errors="coerce"), index=frame.index)
    parsed_value = pd.Series(pd.to_numeric(frame["value"], errors="coerce"), index=frame.index, dtype=float)
    parsed_value = parsed_value.where(np.isfinite(parsed_value), np.nan)
    invalid_timestamp_count = int(parsed_time.isna().sum())
    invalid_value_count = int(parsed_value.isna().sum())
    parsed_value = parsed_value * unit_scale_to_kw

    valid_time = parsed_time.notna()
    in_range = valid_time & parsed_time.between(full_index[0], full_index[-1], inclusive="both")
    out_of_range_count = int((valid_time & ~in_range).sum())
    point = pd.DataFrame({"time": parsed_time[in_range], "value": parsed_value[in_range]})
    duplicate_timestamp_count = int(point["time"].duplicated(keep="last").sum())
    point = point.drop_duplicates(subset="time", keep="last").sort_values("time")
    off_grid_timestamp_count = int((~point["time"].isin(pd.Series(full_index))).sum())
    series = pd.Series(point.set_index("time")["value"].reindex(full_index), dtype=float)

    stats = {
        "data_type": data_type,
        "spot_id": spot_id,
        "source_file": str(source_path),
        "source_unit": source_unit,
        "unit_scale_to_kw": unit_scale_to_kw,
        "source_rows": int(source_rows),
        "invalid_timestamp_count": invalid_timestamp_count,
        "invalid_value_count": invalid_value_count,
        "out_of_range_count": out_of_range_count,
        "duplicate_timestamp_count": duplicate_timestamp_count,
        "off_grid_timestamp_count": off_grid_timestamp_count,
        "observed_value_count": int(series.notna().sum()),
        "missing_value_count": int(series.isna().sum()),
    }
    return LoadedPoint(
        data_type=data_type,
        spot_id=spot_id,
        source_path=source_path,
        series=series,
        stats=stats,
    )


def _atomic_to_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    frame.to_csv(temporary, index=False, date_format="%Y-%m-%d %H:%M:%S", na_rep="")
    os.replace(temporary, destination)


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)


def _iter_unique_metadata(rows: pd.DataFrame) -> Iterable[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for data_type, spot_id in rows[["data_type", "spot_id"]].itertuples(index=False, name=None):
        key = (str(data_type), str(spot_id))
        if key not in seen:
            seen.add(key)
            yield key


def _resolve_unit_scale_to_kw(unit: Any) -> tuple[str, float]:
    if pd.isna(unit) or str(unit).strip() == "":
        return "", 1.0
    source_unit = str(unit).strip()
    normalized = source_unit.casefold()
    if normalized == "kw":
        return source_unit, 1.0
    if normalized == "w":
        return source_unit, 0.001
    raise ValueError(f"Unsupported power unit: {source_unit}")


def _build_output_filename(spec: OutputSpec, full_index: pd.DatetimeIndex, freq: str) -> str:
    normalized_freq = pd.tseries.frequencies.to_offset(freq).freqstr.lower()
    freq_token = re.sub(r"[^a-z0-9]+", "_", normalized_freq).strip("_")
    start_token = full_index[0].strftime("%Y%m%d")
    end_token = full_index[-1].strftime("%Y%m%d")
    return f"{spec.file_stem}_{freq_token}_{start_token}_{end_token}.csv"


def run_point_load_aggregation(
    *,
    reference_path: str | Path,
    points_root: str | Path,
    output_dir: str | Path,
    start_time: str = DEFAULT_START_TIME,
    end_time: str = DEFAULT_END_TIME,
    freq: str = DEFAULT_FREQ,
) -> dict[str, Any]:
    """按点位表生成暖通、UPS、列头柜宽表及总负荷，并写审计 JSON。"""
    reference = Path(reference_path)
    point_root = Path(points_root)
    destination_dir = Path(output_dir)
    if not reference.exists():
        raise FileNotFoundError(f"Reference workbook not found: {reference}")
    if not point_root.exists():
        raise FileNotFoundError(f"Points root not found: {point_root}")

    start = pd.Timestamp(start_time)
    end = pd.Timestamp(end_time)
    if start is pd.NaT or end is pd.NaT:
        raise ValueError("start_time and end_time must be valid timestamps")
    if end < start:
        raise ValueError("end_time must be >= start_time")
    full_index = pd.date_range(start, end, freq=freq)
    if len(full_index) == 0:
        raise ValueError("The requested time range produced an empty index")

    specs = build_output_specs()
    required_sheets = sorted({spec.sheet_name for spec in specs})
    with pd.ExcelFile(reference) as workbook:
        missing_sheets = sorted(set(required_sheets) - set(workbook.sheet_names))
        if missing_sheets:
            raise ValueError(f"Reference workbook missing sheets: {missing_sheets}")
        sheet_frames = {
            sheet_name: pd.read_excel(workbook, sheet_name=sheet_name, dtype=str).dropna(how="all")
            for sheet_name in required_sheets
        }

    audit: dict[str, Any] = {
        "reference_path": str(reference),
        "points_root": str(point_root),
        "output_dir": str(destination_dir),
        "start_time": full_index[0].strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": full_index[-1].strftime("%Y-%m-%d %H:%M:%S"),
        "freq": freq,
        "output_unit": "kW",
        "output_count": len(specs),
        "routes": list(ROUTES),
        "outputs": [],
    }

    for sheet_name in required_sheets:
        sheet_specs = [spec for spec in specs if spec.sheet_name == sheet_name]
        selections = {spec.name: _select_metadata(sheet_frames[sheet_name], spec) for spec in sheet_specs}
        selected_by_name = {name: selection[0] for name, selection in selections.items()}
        point_cache: dict[tuple[str, str], LoadedPoint] = {}
        all_rows = pd.concat(selected_by_name.values(), ignore_index=True)
        raw_units = all_rows["unit_spot"] if "unit_spot" in all_rows.columns else pd.Series("", index=all_rows.index)
        point_units: dict[tuple[str, str], tuple[str, float]] = {}
        for data_type, spot_id, raw_unit in zip(all_rows["data_type"], all_rows["spot_id"], raw_units):
            key = (str(data_type), str(spot_id))
            resolved_unit = _resolve_unit_scale_to_kw(raw_unit)
            previous_unit = point_units.get(key)
            if previous_unit is not None and previous_unit != resolved_unit:
                raise ValueError(f"Point {key} has conflicting power units: {previous_unit[0]}, {resolved_unit[0]}")
            point_units[key] = resolved_unit
        for data_type, spot_id in _iter_unique_metadata(all_rows):
            source_path = point_root / data_type / f"{spot_id}.csv"
            if source_path.exists():
                source_unit, unit_scale_to_kw = point_units[(data_type, spot_id)]
                point_cache[(data_type, spot_id)] = _load_point(
                    data_type=data_type,
                    spot_id=spot_id,
                    source_path=source_path,
                    full_index=full_index,
                    source_unit=source_unit,
                    unit_scale_to_kw=unit_scale_to_kw,
                )

        for spec in sheet_specs:
            selected = selected_by_name[spec.name]
            excluded_phase = selections[spec.name][1]
            reference = selections[spec.name][2]
            duplicate_reference_count = int(reference.duplicated(subset=["data_type", "spot_id"]).sum())
            missing_files: list[dict[str, str]] = []
            included: list[LoadedPoint] = []
            used_spot_ids: set[str] = set()
            duplicate_column_spot_ids: list[dict[str, str]] = []

            for data_type, spot_id in _iter_unique_metadata(selected):
                source_path = point_root / data_type / f"{spot_id}.csv"
                loaded = point_cache.get((data_type, spot_id))
                if loaded is None:
                    missing_files.append(
                        {"data_type": data_type, "spot_id": spot_id, "source_file": str(source_path)}
                    )
                    continue
                if spot_id in used_spot_ids:
                    duplicate_column_spot_ids.append(
                        {"data_type": data_type, "spot_id": spot_id, "source_file": str(source_path)}
                    )
                    continue
                used_spot_ids.add(spot_id)
                included.append(loaded)

            point_columns = [point.spot_id for point in included]
            output = pd.DataFrame(
                {
                    "time": full_index,
                    **{point.spot_id: point.series.to_numpy(dtype=float) for point in included},
                }
            )
            if point_columns:
                output["value"] = output[point_columns].sum(axis=1, min_count=1)
            else:
                output["value"] = np.nan

            output_path = destination_dir / f"route_{spec.route}" / _build_output_filename(
                spec, full_index, freq
            )
            _atomic_to_csv(output, output_path)
            output_audit = {
                "name": spec.name,
                "file_stem": spec.file_stem,
                "sheet_name": spec.sheet_name,
                "data_types": list(spec.data_types),
                "route": spec.route,
                "remarks_include": sorted(spec.remarks_include) if spec.remarks_include is not None else [],
                "remarks_exclude": sorted(spec.remarks_exclude) if spec.remarks_exclude is not None else [],
                "output_path": str(output_path),
                "reference_row_count": int(len(reference)),
                "reference_point_count": int(len(list(_iter_unique_metadata(reference)))),
                "selected_point_count": int(len(list(_iter_unique_metadata(selected)))),
                "excluded_phase_point_count": int(len(list(_iter_unique_metadata(excluded_phase)))),
                "excluded_phase_points": [
                    {"data_type": str(data_type), "deviceId": str(device_id), "spot_id": str(spot_id)}
                    for data_type, device_id, spot_id in excluded_phase[
                        ["data_type", "deviceId", "spot_id"]
                    ].drop_duplicates().itertuples(index=False, name=None)
                ],
                "duplicate_reference_count": duplicate_reference_count,
                "included_point_count": int(len(included)),
                "unit_converted_point_count": int(
                    sum(point.stats["unit_scale_to_kw"] != 1.0 for point in included)
                ),
                "missing_file_count": int(len(missing_files)),
                "missing_files": missing_files,
                "duplicate_output_spot_id_count": int(len(duplicate_column_spot_ids)),
                "duplicate_output_spot_ids": duplicate_column_spot_ids,
                "output_rows": int(len(output)),
                "value_non_missing_count": int(output["value"].notna().sum()),
                "value_missing_count": int(output["value"].isna().sum()),
                "points": [point.stats for point in included],
            }
            audit["outputs"].append(output_audit)

        del point_cache

    audit["outputs"].sort(key=lambda item: [spec.name for spec in specs].index(item["name"]))
    removed_legacy_outputs: list[str] = []
    legacy_paths = {destination_dir / f"{name}.csv" for name in LEGACY_OUTPUT_NAMES}
    for spec in specs:
        legacy_paths.add(destination_dir / f"{spec.name}.csv")
        legacy_paths.add(destination_dir / f"route_{spec.route}" / f"{spec.name}.csv")
    for legacy_path in sorted(legacy_paths):
        if legacy_path.exists():
            legacy_path.unlink()
            removed_legacy_outputs.append(str(legacy_path))
    audit["removed_legacy_outputs"] = removed_legacy_outputs
    _atomic_write_json(audit, destination_dir / AUDIT_FILENAME)
    return audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 all_ids.xlsx 聚合 A1/A2/A3 点位负荷宽表")
    parser.add_argument(
        "--reference-path",
        default="dataset/aidc_load_5min/A1_A2_A3_points/all_ids.xlsx",
        help="点位参考表路径",
    )
    parser.add_argument(
        "--points-root",
        default="dataset/aidc_load_5min/A1_A2_A3_points",
        help="点位 CSV 根目录",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/aidc_load_5min/A1_A2_A3_data",
        help="A/B 路聚合 CSV 子目录与审计 JSON 的输出根目录",
    )
    parser.add_argument("--start-time", default=DEFAULT_START_TIME)
    parser.add_argument("--end-time", default=DEFAULT_END_TIME)
    parser.add_argument("--freq", default=DEFAULT_FREQ)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    audit = run_point_load_aggregation(
        reference_path=args.reference_path,
        points_root=args.points_root,
        output_dir=args.output_dir,
        start_time=args.start_time,
        end_time=args.end_time,
        freq=args.freq,
    )
    missing_file_count = sum(item["missing_file_count"] for item in audit["outputs"])
    print(
        f"Generated {audit['output_count']} tables under {audit['output_dir']}; "
        f"reported {missing_file_count} missing file references."
    )


if __name__ == "__main__":
    main()
