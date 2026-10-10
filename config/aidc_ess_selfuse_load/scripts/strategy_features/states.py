"""计划方向与实际运行状态编码。"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OperatingThresholds:
    """实际 PCS 功率的充放电判定阈值，单位 kW。"""

    charge_power: float = -1500.0
    discharge_power: float = 5000.0

    def __post_init__(self) -> None:
        if self.charge_power >= self.discharge_power:
            raise ValueError("charge_power must be smaller than discharge_power")


def _encode_one_hot(
    charge: pd.Series,
    discharge: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    standby = ~(charge | discharge)
    return pd.DataFrame(
        {
            f"{prefix}_charge": charge.astype(int),
            f"{prefix}_standby": standby.astype(int),
            f"{prefix}_discharge": discharge.astype(int),
        },
        index=charge.index,
    )


def _validate_power(power: pd.Series) -> pd.Series:
    values = pd.Series(power, copy=False)
    try:
        converted = pd.to_numeric(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("power must contain only numeric values") from exc
    numeric_values = pd.Series(converted, index=values.index, dtype="float64")
    if not np.isfinite(numeric_values.to_numpy()).all():
        raise ValueError("power must contain only finite values")
    return numeric_values


def encode_plan_direction(plan_power: pd.Series) -> pd.DataFrame:
    """按符号把计划功率编码为严格互斥的充电、待机、放电 one-hot。"""
    values = _validate_power(plan_power)
    return _encode_one_hot(values < 0, values > 0, prefix="plan_direction")


def encode_actual_operating_state(
    actual_power: pd.Series,
    thresholds: OperatingThresholds | None = None,
) -> pd.DataFrame:
    """按可配置阈值编码实际运行状态；阈值边界归入待机。"""
    values = _validate_power(actual_power)
    limits = thresholds or OperatingThresholds()
    return _encode_one_hot(
        values < limits.charge_power,
        values > limits.discharge_power,
        prefix="actual_operating",
    )
