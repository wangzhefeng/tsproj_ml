# -*- coding: utf-8 -*-
"""点预测评估：MAE/RMSE/MAPE/Accuracy + seasonal-naive 对照 + 聚合加权 + 评估掩码。

自 `model_predicting/artifacts/results.py` 迁入（2026-08-30 evaluation 模块化），实现逐字保真；
掩码与概率评估（`model_evaluation/marginal.py`）共用 `build_eval_mask_payload`，
保证同一业务口径（D13 rewire 纪律的延续）。
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from model_evaluation.mask import build_eval_mask
from model_evaluation.metrics import seasonal_insample_scales
from forecasting_core.tensors import PointForecastTensor, require_matching_point_axes


EVALUATION_AGGREGATION = {
    "point.aggregate": "weighted_mean_of_per_target_metrics",
    "point.aggregate_horizon": "pool_valid_points_across_targets_at_horizon",
    "marginal.aggregate": "pool_valid_points_across_targets",
    "mask": "configured_eval_mask_only; training_data_unchanged",
}

# 点评分列指标全集（target/aggregate/horizon 行与 naive 对照共用；2026-10-05
# 增加 SMAPE 与 MASE/RMSSE，顺带收口此前三处重复的键元组与占位字典）。
_POINT_METRIC_KEYS = ("MAE", "RMSE", "Bias", "MAPE", "Accuracy", "SMAPE", "MASE", "RMSSE")


def _nan_metric_values() -> dict[str, float | int]:
    return {**{key: float("nan") for key in _POINT_METRIC_KEYS}, "Valid Points": 0, "n_points": 0}


def resolve_aggregate_weighting(
    targets: tuple[str, ...],
    aggregate_weighting: Mapping[str, float] | None,
) -> dict[str, float]:
    if aggregate_weighting is None:
        weight = 1.0 / len(targets)
        return {target: weight for target in targets}
    weights = {
        str(target): float(weight) for target, weight in aggregate_weighting.items()
    }
    if set(weights) != set(targets):
        raise ValueError("aggregate weighting keys must exactly match forecast targets")
    values = np.asarray(tuple(weights[target] for target in targets), dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("aggregate weights must be finite and nonnegative")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("aggregate weights must sum to 1")
    return {target: weights[target] for target in targets}


def _metric_values(
    actual: np.ndarray,
    prediction: np.ndarray,
    eval_mask: dict | None = None,
    *,
    scales: tuple[float, float] | None = None,
) -> dict[str, float | int]:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if eval_mask is not None:
        # D13 rewire：业务评估掩码（与历史「掩码后中位数」口径对齐）。
        # 掩码只依赖 actual，与 isfinite 正交组合；未配置时保持原行为。
        valid = valid & eval_mask["valid_mask"]
    mape_valid = valid & (actual != 0.0)
    if not valid.any():
        mae = rmse = bias = float("nan")
    else:
        error = actual[valid] - prediction[valid]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        # bias（2026-09-02）：mean(pred - actual)，正值 = 系统性高估。
        bias = float(np.mean(prediction[valid] - actual[valid]))
    if not mape_valid.any():
        mape = accuracy = float("nan")
    else:
        mape = float(
            np.mean(
                np.abs(
                    (actual[mape_valid] - prediction[mape_valid])
                    / actual[mape_valid]
                )
            )
        )
        accuracy = 1.0 - mape
    # SMAPE（2026-10-05）：2|a−p|/(|a|+|p|)，分母为零的点不计入口径。
    smape_valid = valid & ((np.abs(actual) + np.abs(prediction)) > 0.0)
    if not smape_valid.any():
        smape = float("nan")
    else:
        smape = float(
            np.mean(
                2.0
                * np.abs(actual[smape_valid] - prediction[smape_valid])
                / (np.abs(actual[smape_valid]) + np.abs(prediction[smape_valid]))
            )
        )
    # MASE/RMSSE（2026-10-05）：以 in-sample 季节差分尺度为分母；尺度缺失或
    # 近似零（常数历史）时 NaN，不伪造数值。
    mase = rmsse = float("nan")
    if scales is not None:
        mae_scale, rmse_scale = scales
        if np.isfinite(mae) and np.isfinite(mae_scale):
            mase = mae / mae_scale
        if np.isfinite(rmse) and np.isfinite(rmse_scale):
            rmsse = rmse / rmse_scale
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Bias": bias,
        "MAPE": mape,
        "Accuracy": accuracy,
        "SMAPE": smape,
        "MASE": mase,
        "RMSSE": rmsse,
        "Valid Points": int(mape_valid.sum()),
        "n_points": int(valid.sum()),
    }


def build_eval_mask_payload(
    eval_mask: Mapping[str, Any] | None,
    actual: PointForecastTensor,
    actual_full: PointForecastTensor | None = None,
) -> dict[str, dict] | None:
    """按 validation.eval_mask 配置构造逐 target 掩码（作用于 actual 值）。

    未配置（默认 None）时返回 None，掩码逻辑完全不参与，行为与历史逐值一致。
    点评估（evaluate_point_forecasts）与概率评估（evaluate_marginal_distribution）
    共用本函数，保证 MAPE 与 pinball/区间指标同一业务口径（2026-08-30 接线）。
    """
    if eval_mask is None:
        return None
    reference = actual_full if actual_full is not None else actual
    return {
        target: build_eval_mask(
            reference.values[:, :, target_index].reshape(-1),
            mode=str(eval_mask.get("mode", "percentile")),
            percentile=float(eval_mask.get("percentile", 5.0)),
            min_value=eval_mask.get("min_value"),
            max_value=eval_mask.get("max_value"),
        )
        for target_index, target in enumerate(actual.targets)
    }


def evaluate_point_forecasts(
    actual: PointForecastTensor,
    prediction: PointForecastTensor,
    *,
    aggregate_weighting: Mapping[str, float] | None = None,
    seasonal_naive: PointForecastTensor | None = None,
    window: int = 1,
    eval_mask: Mapping[str, Any] | None = None,
    actual_full: PointForecastTensor | None = None,
    insample_history: PointForecastTensor | None = None,
    naive_lag: int | None = None,
) -> pd.DataFrame:
    """点预测评分帧（scope ∈ {target, aggregate, horizon, aggregate_horizon}）。

    ``insample_history`` + ``naive_lag``（必须同给或同不给，2026-10-05）启用
    MASE/RMSSE：缩放因子为逐 target 的 in-sample 季节差分尺度（原始域、与
    seasonal-naive 同一 lag）；未提供时 MASE/RMSSE 列为 NaN。aggregate_horizon
    行的 MASE/RMSSE 为 NaN（跨 target 池化无单一尺度，不伪造）。
    """
    if not isinstance(actual, PointForecastTensor) or not isinstance(
        prediction, PointForecastTensor
    ):
        raise TypeError("actual and prediction must be PointForecastTensor values")
    require_matching_point_axes(actual, prediction)
    if seasonal_naive is not None:
        require_matching_point_axes(actual, seasonal_naive)
    if actual_full is not None:
        require_matching_point_axes(actual, actual_full)
    if (insample_history is None) != (naive_lag is None):
        raise ValueError("insample_history and naive_lag must be provided together")
    insample_scales: dict[str, tuple[float, float]] | None = None
    if insample_history is not None:
        if not isinstance(insample_history, PointForecastTensor):
            raise TypeError("insample_history must be a PointForecastTensor")
        if tuple(insample_history.targets) != tuple(actual.targets):
            raise ValueError("insample_history targets must match actual targets")
        assert naive_lag is not None  # 上面已校验两者同给
        insample_scales = {
            target: seasonal_insample_scales(
                insample_history.values[:, :, target_index], naive_lag
            )
            for target_index, target in enumerate(actual.targets)
        }
    weights = resolve_aggregate_weighting(actual.targets, aggregate_weighting)

    # D13 rewire：validation.eval_mask 配置时构造掩码（作用于 actual 值）；
    # 未配置（默认 None）时掩码逻辑完全不参与，行为与历史逐值一致。
    mask_payload = build_eval_mask_payload(eval_mask, actual, actual_full)

    rows: list[dict[str, Any]] = []
    target_metrics: dict[str, dict[str, float | int]] = {}
    target_naive_metrics: dict[str, dict[str, float | int]] = {}
    for target_index, target in enumerate(actual.targets):
        scales = insample_scales[target] if insample_scales is not None else None
        metrics = _metric_values(
            actual.values[:, :, target_index].reshape(-1),
            prediction.values[:, :, target_index].reshape(-1),
            mask_payload[target] if mask_payload is not None else None,
            scales=scales,
        )
        naive_metrics = (
            _metric_values(
                actual.values[:, :, target_index].reshape(-1),
                seasonal_naive.values[:, :, target_index].reshape(-1),
                mask_payload[target] if mask_payload is not None else None,
                scales=scales,
            )
            if seasonal_naive is not None
            else _nan_metric_values()
        )
        target_metrics[target] = metrics
        target_naive_metrics[target] = naive_metrics
        rows.append(
            {
                "window": int(window),
                "scope": "target",
                "target": target,
                **metrics,
                **{f"Naive {key}": value for key, value in naive_metrics.items()},
            }
        )

    aggregate = {
        key: float(
            sum(float(target_metrics[target][key]) * weights[target] for target in actual.targets)
        )
        for key in _POINT_METRIC_KEYS
    }
    aggregate["Valid Points"] = int(
        sum(int(target_metrics[target]["Valid Points"]) for target in actual.targets)
    )
    aggregate["n_points"] = int(
        sum(int(target_metrics[target]["n_points"]) for target in actual.targets)
    )
    aggregate_naive = {
        key: float(
            sum(
                float(target_naive_metrics[target][key]) * weights[target]
                for target in actual.targets
            )
        )
        for key in _POINT_METRIC_KEYS
    }
    aggregate_naive["Valid Points"] = int(
        sum(
            int(target_naive_metrics[target]["Valid Points"])
            for target in actual.targets
        )
    )
    aggregate_naive["n_points"] = int(
        sum(
            int(target_naive_metrics[target]["n_points"])
            for target in actual.targets
        )
    )
    rows.append(
        {
            "window": int(window),
            "scope": "aggregate",
            "target": "__aggregate__",
            **aggregate,
            **{f"Naive {key}": value for key, value in aggregate_naive.items()},
        }
    )

    # per-horizon 诊断（2026-09-02）：逐 horizon 步的指标衰减曲线。
    # - ``scope="horizon"``：per-target，掩码切片到同一 horizon（与 target 行同口径）；
    # - ``scope="aggregate_horizon"``：跨 target 按有效点池化（proper score 语义，
    #   与 target 加权 aggregate 语义不同，勿混用），MASE/RMSSE 因无单一尺度记 NaN；
    # - ``horizon`` 列 1-based（h=1 即第一个预测步）。
    horizon_keys = _POINT_METRIC_KEYS
    n_series, n_horizons, _ = prediction.values.shape

    def _horizon_row(scope: str, label: str, h: int, values: dict, naive: dict) -> dict:
        return {
            "window": int(window),
            "scope": scope,
            "target": label,
            "horizon": int(h) + 1,
            **{key: values[key] for key in horizon_keys},
            **{
                "Valid Points": int(values["Valid Points"]),
                "n_points": int(values["n_points"]),
            },
            **{f"Naive {key}": naive[key] for key in horizon_keys},
            "Naive Valid Points": int(naive["Valid Points"]),
            "Naive n_points": int(naive["n_points"]),
        }

    for h in range(n_horizons):
        pooled_actual: list[np.ndarray] = []
        pooled_prediction: list[np.ndarray] = []
        pooled_naive: list[np.ndarray] = []
        for target_index, target in enumerate(actual.targets):
            actual_h = actual.values[:, h, target_index]
            prediction_h = prediction.values[:, h, target_index]
            naive_h = (
                seasonal_naive.values[:, h, target_index]
                if seasonal_naive is not None
                else None
            )
            scales = insample_scales[target] if insample_scales is not None else None
            h_mask = (
                {
                    "valid_mask": mask_payload[target]["valid_mask"].reshape(
                        n_series, n_horizons
                    )[:, h]
                }
                if mask_payload is not None
                else None
            )
            rows.append(
                _horizon_row(
                    "horizon",
                    str(target),
                    h,
                    _metric_values(actual_h, prediction_h, h_mask, scales=scales),
                    (
                        _metric_values(actual_h, naive_h, h_mask, scales=scales)
                        if naive_h is not None
                        else _nan_metric_values()
                    ),
                )
            )
            # 池化：按各 target 掩码后有效点拼接（掩码已作用，重算时不再传）。
            flat_mask = (
                mask_payload[target]["valid_mask"] if mask_payload is not None else None
            )
            valid = np.isfinite(actual_h) & np.isfinite(prediction_h)
            if flat_mask is not None:
                valid = valid & flat_mask.reshape(n_series, n_horizons)[:, h]
            pooled_actual.append(actual_h[valid])
            pooled_prediction.append(prediction_h[valid])
            if naive_h is not None:
                pooled_naive.append(naive_h[valid])
        pooled_metrics = _metric_values(
            np.concatenate(pooled_actual), np.concatenate(pooled_prediction)
        )
        pooled_naive_metrics = (
            _metric_values(
                np.concatenate(pooled_actual), np.concatenate(pooled_naive)
            )
            if seasonal_naive is not None
            else _nan_metric_values()
        )
        rows.append(
            _horizon_row(
                "aggregate_horizon",
                "__aggregate__",
                h,
                pooled_metrics,
                pooled_naive_metrics,
            )
        )

    result = pd.DataFrame(rows)
    ordered_columns = [
        "window",
        "scope",
        "target",
        "horizon",
        *horizon_keys,
        "Valid Points",
        "n_points",
        *[f"Naive {key}" for key in horizon_keys],
        "Naive Valid Points",
        "Naive n_points",
    ]
    present = [column for column in ordered_columns if column in result.columns]
    remaining = [
        column for column in result.columns if column not in present
    ]
    result = result.loc[:, present + remaining]
    result.attrs["aggregate_weighting"] = weights
    return result


__all__ = [
    "build_eval_mask_payload",
    "evaluate_point_forecasts",
    "resolve_aggregate_weighting",
]
