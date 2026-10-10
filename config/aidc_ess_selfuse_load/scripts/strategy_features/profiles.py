"""按 22:00 调度周期汇总 ESS 运行画像。"""

import math

import numpy as np
import pandas as pd

from .windows import SLOT_MINUTES, dispatch_cycle_slot, dispatch_cycle_start


STATE_NAMES = ("charge", "standby", "discharge")


def _segment_count(mask: pd.Series) -> int:
    return int((mask & ~mask.shift(fill_value=False)).sum())


def _first_slot(slots: pd.Index, mask: pd.Series):
    if not mask.any():
        return pd.NA
    return int(slots[mask.to_numpy()][0])


def _slot_cyclic_features(slot) -> tuple[float, float]:
    if pd.isna(slot):
        return 0.0, 0.0
    angle = 2 * math.pi * int(slot) / 288
    return math.sin(angle), math.cos(angle)


def summarize_dispatch_profiles(
    frame: pd.DataFrame,
    time_col: str = "time",
    power_col: str = "power_kw",
    state_prefix: str = "actual_operating",
) -> pd.DataFrame:
    """计算每个调度周期的状态时长、分段、能量与变化统计。"""
    state_columns = [f"{state_prefix}_{state}" for state in STATE_NAMES]
    required_columns = [time_col, power_col, *state_columns]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(f"missing profile columns: {missing_columns}")

    work = frame.loc[:, required_columns].copy()
    try:
        work[power_col] = pd.to_numeric(work[power_col], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("profile power must contain only numeric values") from exc
    if not np.isfinite(work[power_col].to_numpy(dtype=float)).all():
        raise ValueError("profile power must contain only finite values")
    state_values = work[state_columns]
    if not state_values.isin([0, 1]).all().all() or not (
        state_values.sum(axis=1) == 1
    ).all():
        raise ValueError("profile state columns must be exactly one-hot per row")
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.sort_values(time_col).reset_index(drop=True)
    work["cycle_start"] = dispatch_cycle_start(work[time_col])

    rows = []
    for cycle_start_value, group in work.groupby("cycle_start", sort=True):
        group = group.reset_index(drop=True)
        power = group[power_col]
        slots = dispatch_cycle_slot(group[time_col])
        state_masks = {
            state: group[f"{state_prefix}_{state}"].astype(bool)
            for state in STATE_NAMES
        }
        state_labels = group[state_columns].idxmax(axis=1)

        row = {"cycle_start": pd.Timestamp(cycle_start_value)}
        for state, mask in state_masks.items():
            row[f"{state}_hours"] = float(mask.sum() * SLOT_MINUTES / 60)
            row[f"{state}_segment_count"] = _segment_count(mask)

        charge_mask = state_masks["charge"]
        discharge_mask = state_masks["discharge"]
        row["charge_energy_kwh"] = float(power[charge_mask].sum() * SLOT_MINUTES / 60)
        row["discharge_energy_kwh"] = float(power[discharge_mask].sum() * SLOT_MINUTES / 60)
        row["switch_count"] = int(state_labels.ne(state_labels.shift()).sum() - 1)
        row["max_ramp_kw"] = float(power.diff().abs().max()) if len(power) > 1 else 0.0
        first_charge_slot = _first_slot(slots, charge_mask)
        first_discharge_slot = _first_slot(slots, discharge_mask)
        first_charge_slot_sin, first_charge_slot_cos = _slot_cyclic_features(
            first_charge_slot
        )
        first_discharge_slot_sin, first_discharge_slot_cos = _slot_cyclic_features(
            first_discharge_slot
        )
        row["first_charge_slot"] = first_charge_slot
        row["first_discharge_slot"] = first_discharge_slot
        row["first_charge_slot_sin"] = first_charge_slot_sin
        row["first_charge_slot_cos"] = first_charge_slot_cos
        row["first_discharge_slot_sin"] = first_discharge_slot_sin
        row["first_discharge_slot_cos"] = first_discharge_slot_cos
        row["has_charge"] = bool(charge_mask.any())
        row["has_discharge"] = bool(discharge_mask.any())
        rows.append(row)

    return pd.DataFrame(rows)
