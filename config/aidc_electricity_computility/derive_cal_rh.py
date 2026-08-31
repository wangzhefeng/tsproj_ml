"""为算力场景天气资产补齐 legacy `cal_rh` 派生列。

历史 `FeatureEngineering.extend_weather_feature()` 在运行时按 Magnus–Tetens
公式由 `rt_tt2`/`rt_dt` 计算相对湿度，并对缺口做线性插值与双向填充。
canonical SourceRegistry 不再隐式改写输入，因此该派生应在离线数据准备阶段完成。

默认只预览；显式传入 ``--write`` 才原子更新 CSV。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = PROJECT_ROOT / "config/aidc_electricity_computility"


class MigrationResult(NamedTuple):
    path: Path
    status: str
    rows: int
    minimum: float
    maximum: float


def calculate_cal_rh(tt2_k: pd.Series, dt_k: pd.Series) -> pd.Series:
    """复刻 legacy Magnus–Tetens 计算及线性/双向填充合同。"""
    t_air = pd.Series(
        pd.to_numeric(tt2_k, errors="coerce"),
        index=tt2_k.index,
        dtype=float,
    ) - 273.15
    t_dew = pd.Series(
        pd.to_numeric(dt_k, errors="coerce"),
        index=dt_k.index,
        dtype=float,
    ) - 273.15
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        e_s_td = 6.1078 * np.exp((17.2693 * t_dew) / (237.29 + t_dew))
        e_s_t = 6.1078 * np.exp((17.2693 * t_air) / (237.29 + t_air))
        values = pd.Series(
            np.clip((e_s_td / e_s_t) * 100, 0, 100),
            index=tt2_k.index,
            dtype=float,
        )
    values = values.replace([np.inf, -np.inf], np.nan)
    if not values.notna().any():
        raise ValueError("cal_rh cannot be derived because no finite rt_tt2/rt_dt pair exists")
    values = values.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if values.isna().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("cal_rh derivation left non-finite values")
    return values


def _read_rows(path: Path) -> tuple[list[list[str]], str]:
    raw = path.read_bytes()
    line_ending = "\r\n" if b"\r\n" in raw else "\n"
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError(f"weather CSV is empty: {path}")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"weather CSV has ragged rows: {path}")
    return rows, line_ending


def migrate_path(path: Path, *, write: bool) -> MigrationResult:
    """校验并按需向一个 CSV 追加 `cal_rh`，不改写旧字段文本。"""
    path = Path(path)
    rows, line_ending = _read_rows(path)
    header = rows[0]
    required = {"rt_tt2", "rt_dt"}
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"weather CSV missing source columns {missing}: {path}")

    tt2_index = header.index("rt_tt2")
    dt_index = header.index("rt_dt")
    tt2 = pd.Series([row[tt2_index] for row in rows[1:]], dtype="object")
    dt = pd.Series([row[dt_index] for row in rows[1:]], dtype="object")
    expected = calculate_cal_rh(tt2, dt)
    minimum = float(expected.min())
    maximum = float(expected.max())

    if "cal_rh" in header:
        actual_index = header.index("cal_rh")
        actual = pd.Series(
            pd.to_numeric(
                pd.Series([row[actual_index] for row in rows[1:]]),
                errors="coerce",
            ),
            dtype=float,
        )
        if actual.isna().any() or not np.allclose(
            actual.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=5e-10,
        ):
            raise ValueError(f"existing cal_rh does not match the legacy formula: {path}")
        return MigrationResult(path, "unchanged", len(rows) - 1, minimum, maximum)

    if not write:
        return MigrationResult(path, "would_write", len(rows) - 1, minimum, maximum)

    migrated = [header + ["cal_rh"]]
    migrated.extend(
        row + [format(float(value), ".10f")]
        for row, value in zip(rows[1:], expected, strict=True)
    )
    mode = path.stat().st_mode
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            writer = csv.writer(stream, lineterminator=line_ending)
            writer.writerows(migrated)
        os.chmod(temp_path, mode)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return MigrationResult(path, "written", len(rows) - 1, minimum, maximum)


def discover_missing_paths(project_root: Path = PROJECT_ROOT) -> tuple[Path, ...]:
    """发现本场景活动 schema-2 配置引用且缺少 `cal_rh` 的物理文件。"""
    config_root = project_root / "config/aidc_electricity_computility"
    paths: set[Path] = set()
    for config_path in config_root.rglob("*.yaml"):
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            continue
        for source in payload.get("data", {}).get("sources", []):
            names = {
                column.get("name")
                for column in source.get("columns", [])
                if isinstance(column, dict)
            }
            if "cal_rh" not in names:
                continue
            for key in ("history_path", "backtest_path", "future_path"):
                raw_path = source.get(key)
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = project_root / path
                if not path.exists():
                    raise FileNotFoundError(path)
                rows, _ = _read_rows(path)
                if "cal_rh" not in rows[0]:
                    paths.add(path)
    return tuple(sorted(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="原子写入 cal_rh；省略时只预览",
    )
    args = parser.parse_args()
    results = [
        migrate_path(path, write=args.write)
        for path in discover_missing_paths()
    ]
    print(
        json.dumps(
            {
                "write": args.write,
                "files": len(results),
                "rows": sum(result.rows for result in results),
                "results": [
                    {
                        "path": str(result.path.relative_to(PROJECT_ROOT)),
                        "status": result.status,
                        "rows": result.rows,
                        "cal_rh_min": result.minimum,
                        "cal_rh_max": result.maximum,
                    }
                    for result in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
