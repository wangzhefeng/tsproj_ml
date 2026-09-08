"""联通 IT 原始 Excel → 5min 点位功率、总功率及信号映射。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUTPUT = ROOT / "dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"
SOURCE = OUTPUT / "aidc_load_liantong_5min"
TOTAL = "输入总有功功率"
PHASES = {f"A(1)路输入{phase}相有功功率" for phase in "ABC"}
GRID = pd.date_range("2026-08-01 00:00:00", "2026-08-31 23:55:00", freq="5min", name="time")


def build_mapping(reference: pd.DataFrame) -> pd.DataFrame:
    """按参考表首次出现顺序编号，三相信号共用一个点位列。"""
    columns = ["SignalID", "RoomID", "DevAssestID", "DeviceName", "SignalName"]
    mapping = reference[columns].copy()
    if mapping.isna().any().any() or mapping.SignalID.duplicated().any():
        raise ValueError("参考表关键字段缺失或 SignalID 重复")
    units = reference["Describe"].fillna("KW")
    if not units.eq("KW").all():
        raise ValueError("参考表包含未授权的单位")
    for number, (_, group) in enumerate(mapping.groupby(["RoomID", "DevAssestID"], sort=False), 1):
        signals = group.SignalName.tolist()
        if not (signals == [TOTAL] or (len(signals) == 3 and set(signals) == PHASES)):
            raise ValueError(f"点位信号定义异常: {signals}")
        if group.DeviceName.nunique() != 1:
            raise ValueError("同一设备资产对应多个名称")
        mapping.loc[group.index, "point_column"] = f"point_{number}_value"
    mapping["unit"] = "kW"
    return mapping[["SignalID", "point_column", "RoomID", "DevAssestID", "DeviceName", "SignalName", "unit"]]


def aggregate(data: pd.DataFrame, mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """显式 floor 对齐，缺失不填补，各级求和忽略空值但不将全空变零。"""
    data = data.copy()
    data["time"] = pd.to_datetime(data.time, format="%Y-%m-%d %H:%M:%S", errors="raise")
    if data.time.isna().any():
        raise ValueError("信号时间缺失")
    data["value"] = pd.to_numeric(data.value, errors="raise")
    if np.isinf(data.value.to_numpy(dtype=float)).any():
        raise ValueError("功率包含非有限值")
    data["time"] = data.time.dt.floor("5min")
    if not data.time.isin(GRID).all():
        raise ValueError("信号时间超出 2026 年 8 月")
    if data.duplicated(["signalid", "time"]).any():
        raise ValueError("同一信号在同一 5min 时间格内重复")
    wide = data.pivot(index="time", columns="signalid", values="value").reindex(index=GRID, columns=mapping.SignalID)
    points = {}
    partial_phases = {}
    for column, group in mapping.groupby("point_column", sort=False):
        values = wide[group.SignalID]
        points[column] = values.sum(axis=1, min_count=1)
        count = values.notna().sum(axis=1)
        if len(group) == 3:
            partial_phases[column] = int(((count > 0) & (count < 3)).sum())
    result = pd.DataFrame(points, index=GRID)
    available = result.notna().sum(axis=1)
    result["value"] = result.sum(axis=1, min_count=1)
    audit = {
        "rows": len(result), "points": len(points), "signals": len(mapping),
        "missing_by_signal": {key: int(value) for key, value in wide.isna().sum().items()},
        "partial_phase_rows_by_point": partial_phases,
        "partial_device_rows": int(((available > 0) & (available < len(points))).sum()),
        "all_empty_rows": int(available.eq(0).sum()),
        "policy": {"alignment": "floor_5min", "unit": "kW", "sum": "skipna_min_count_1", "imputation": "none"},
    }
    return result.reset_index(), audit


def run(source_dir: Path = SOURCE, output_dir: Path = OUTPUT) -> dict:
    """读取白名单信号后再解析时间，排除表外数据及 Excel 临时文件。"""
    reference = pd.read_excel(source_dir / "联通IT测点筛选表.xlsx", dtype=str)
    mapping = build_mapping(reference)
    files = sorted(source_dir.glob("data-*.xlsx"))
    if not files:
        raise ValueError("未找到 data-*.xlsx")
    frames = []
    inventory = []
    for path in files:
        data = pd.read_excel(path, dtype=str)
        selected = data.signalid.isin(mapping.SignalID)
        inventory.append({"file": path.name, "rows": len(data), "selected_rows": int(selected.sum()),
                          "excluded_signals": data.loc[~selected, "signalid"].value_counts(dropna=False).to_dict()})
        frames.append(data.loc[selected, ["time", "signalid", "value"]])
        print(f"读取 {path.name}: {len(data)} 行，保留 {int(selected.sum())} 行", flush=True)
    result, audit = aggregate(pd.concat(frames, ignore_index=True), mapping)
    audit["inputs"] = inventory
    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "df_power.csv", index=False, encoding="utf-8", na_rep="", date_format="%Y-%m-%d %H:%M:%S")
    mapping.to_csv(output_dir / "point_mapping.csv", index=False, encoding="utf-8")
    (output_dir / "processing_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"输出 {output_dir}: {audit['rows']} 行，{audit['points']} 点位，全空 {audit['all_empty_rows']} 行", flush=True)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.source_dir, args.output_dir)


if __name__ == "__main__":
    main()
