"""有限尾窗与显式偏移统计；输入仅为调用方授权的历史。"""
import pandas as pd
from feature_engineering.kernels.history import history_statistic


def window_features(history: pd.Series, kind: str, spec) -> dict[str, float]:
    result = {}
    offsets = spec.get("offsets", (0,))
    for offset in offsets:
        for window in spec["windows"]:
            stop = len(history) - offset
            if stop < window:
                raise ValueError(f"insufficient visible history for {kind} window={window} offset={offset}")
            values = history.iloc[stop - window:stop]
            if kind == "rolling_quantile":
                for level in spec["quantiles"]:
                    result[f"rolling_quantile_{float(level)}_{window}"] = float(values.quantile(level, interpolation="linear"))
            elif kind == "lagged_rolling":
                for stat in spec["stats"]:
                    result[f"rolling_{stat}_{window}_offset_{offset}"] = history_statistic(values, stat)
            else:
                raise ValueError(f"unsupported finite window kind: {kind}")
    return result
