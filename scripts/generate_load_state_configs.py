# -*- coding: utf-8 -*-
"""由 add_exogenous_weather_date 生成无目标分解的 load-state 消融配置。

生成契约：
- 只处理 15min daily/rolling/short 与 power_month freq_1day；不处理 ESS；
- 每个 add_load_state 配置与 weather-date 同名配置保持同构；
- 唯一行为增量是 origin-frozen load_state custom source；
- preprocessing.decomposition_method 必须保持 none。
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "config"

STATE_COLUMNS_15MIN = [
    "state_roll_1h_mean",
    "state_roll_1h_std",
    "state_roll_4h_mean",
    "state_roll_4h_std",
    # roll_24h_mean/std 与基线 rolling(y, 96) 精确重复，不重复输入。
    "state_roll_24h_range",
    # roll_7d_mean/std 与基线 rolling(y, 672) 精确重复，不重复输入。
    "state_diff_15min",
    "state_diff_1h",
    "state_diff_24h_pct",
    "state_robust_z_7d",
    "state_weekly_base_dev_pct",
    "state_route_diff_pct",
]

STATE_COLUMNS_1DAY = [
    "state_z30_robust",
    "state_z30_ready",
    "state_slope30",
    "state_slope30_ready",
    "state_intraday_std",
    "state_intraday_range",
    "state_intraday_p95_p5_gap",
    "state_intraday_cv",
    "state_intraday_max_abs_step",
    "state_intraday_peak_time_frac",
    "state_intraday_range_pct",
    "state_route_diff_pct",
    "state_last_day_volatile",
    "state_volatile_count_7d",
    "state_volatile_count_30d",
]

SCENARIOS = {
    "aidc_load_15min_daily": {
        "route_parent": CONFIG_ROOT / "aidc_load_15min_daily",
        "columns": STATE_COLUMNS_15MIN,
    },
    "aidc_load_15min_rolling": {
        "route_parent": CONFIG_ROOT / "aidc_load_15min_rolling",
        "columns": STATE_COLUMNS_15MIN,
    },
    "aidc_load_15min_short": {
        "route_parent": CONFIG_ROOT / "aidc_load_15min_short",
        "columns": STATE_COLUMNS_15MIN,
    },
    "aidc_power_month": {
        "route_parent": CONFIG_ROOT / "aidc_power_month",
        "route_suffix": Path("freq_1day"),
        "columns": STATE_COLUMNS_1DAY,
    },
}


def _custom_block(route: str, columns: list[str]) -> str:
    lines = [
        "    custom_features:",
        "    - name: load_state",
        f"      history_path: load_state_features/{route}_load_state_history.csv",
        "      future_path: null",
        "      future_strategy: freeze_last_observation",
        "      availability: end_of_period  # 当期结束后可得；预测期冻结原点最后状态",
        "      ts_col: time",
        "      columns:",
    ]
    lines.extend(f"      - {column}" for column in columns)
    lines.append("      categorical_columns: []")
    return "\n".join(lines)


def _append_load_state_suffix(text: str) -> str:
    pattern = re.compile(r"(?m)^(    setting_suffix:\s*)([^#\n]*?)(\s*(?:#.*)?)$")
    match = pattern.search(text)
    if match:
        old_value = match.group(2).strip().strip("'\"")
        new_value = f"{old_value}-load-state" if old_value else "-load-state"
        return pattern.sub(rf"\g<1>{new_value}\g<3>", text, count=1)

    scenario_pattern = re.compile(r"(?m)^(    scenario_subpath:.*)$")
    if not scenario_pattern.search(text):
        raise ValueError("scenario_subpath not found")
    return scenario_pattern.sub(r"\1\n    setting_suffix: -load-state", text, count=1)


def _transform(source: Path, scene: str, route: str, columns: list[str]) -> str:
    text = source.read_text()
    text = text.replace("add_exogenous_weather_date", "add_load_state")
    text = _append_load_state_suffix(text)
    marker = "    custom_features: []"
    if text.count(marker) != 1:
        raise ValueError(f"{source}: expected exactly one '{marker}'")
    text = text.replace(marker, _custom_block(route, columns), 1)

    lines = text.splitlines()
    if lines and "预测原点负荷状态" not in lines[0]:
        lines[0] += " · 预测原点负荷状态"
    text = "\n".join(lines) + "\n"

    data = yaml.safe_load(text)
    overrides = data["overrides"]
    preprocessing = overrides.get("preprocessing", {})
    if preprocessing.get("decomposition_method", "none") != "none":
        raise ValueError(f"{source}: weather-date base must use decomposition_method=none")
    custom = overrides["exogenous_features"]["custom_features"]
    if len(custom) != 1 or custom[0]["columns"] != columns:
        raise ValueError(f"{source}: generated custom feature contract mismatch")
    return text


def generate(*, replace: bool) -> tuple[int, int]:
    removed = 0
    written = 0
    for scene, spec in SCENARIOS.items():
        route_parent: Path = spec["route_parent"]
        route_suffix: Path = spec.get("route_suffix", Path())
        columns: list[str] = spec["columns"]
        for route in ("route_A", "route_B"):
            parent = route_parent / route / route_suffix
            source_dir = parent / "add_exogenous_weather_date"
            target_dir = parent / "add_load_state"
            sources = sorted(source_dir.glob("*.yaml"))
            if len(sources) != 8:
                raise ValueError(f"{source_dir}: expected 8 weather-date configs, got {len(sources)}")
            target_dir.mkdir(parents=True, exist_ok=True)
            if replace:
                for existing in target_dir.glob("*.yaml"):
                    existing.unlink()
                    removed += 1
            for source in sources:
                target = target_dir / source.name
                target.write_text(_transform(source, scene, route[-1], columns))
                written += 1
    return removed, written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replace",
        action="store_true",
        help="删除目标目录现有 YAML 后按 weather-date 基线重建。",
    )
    args = parser.parse_args()
    if not args.replace:
        raise SystemExit("Refusing to overwrite without --replace")
    removed, written = generate(replace=True)
    print(f"removed={removed} written={written}")


if __name__ == "__main__":
    main()
