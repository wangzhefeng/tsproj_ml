"""基于自然日计划相似度生成因果 ESS 模板。"""

from dataclasses import dataclass
import math
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SLOTS_PER_DAY = 288
SLOT_HOURS = 5 / 60


@dataclass(frozen=True)
class SimilarDayConfig:
    """相似日模板参数。"""

    lookback_days: int = 180
    k_neighbors: int = 5
    min_history_days: int = 14
    robust_template_days: int = 7
    q75: float = 0.75
    q95: float = 0.95
    curve_weight: float = 0.60
    duration_energy_weight: float = 0.25
    transition_weight: float = 0.15
    power_scale: float = 9000.0
    count_scale: float = 10.0
    min_effective_samples: float = 2.0

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.k_neighbors <= 0:
            raise ValueError("k_neighbors must be positive")
        if self.min_history_days <= 0:
            raise ValueError("min_history_days must be positive")
        if self.robust_template_days <= 0:
            raise ValueError("robust_template_days must be positive")
        if not 0 <= self.q75 < self.q95 <= 1:
            raise ValueError("novelty quantiles must satisfy 0 <= q75 < q95 <= 1")
        weights = (
            self.curve_weight,
            self.duration_energy_weight,
            self.transition_weight,
        )
        if any(weight < 0 for weight in weights) or not math.isclose(
            sum(weights), 1.0
        ):
            raise ValueError("similar-day block weights must be non-negative and sum to 1")
        if self.power_scale <= 0:
            raise ValueError("power_scale must be positive")
        if self.count_scale <= 0:
            raise ValueError("count_scale must be positive")
        if self.min_effective_samples <= 0:
            raise ValueError("min_effective_samples must be positive")


@dataclass(frozen=True)
class NaturalDayPlan:
    """自然日计划的三块距离表示。"""

    curve: np.ndarray
    duration_energy: np.ndarray
    transition: np.ndarray


@dataclass(frozen=True)
class SimilarDayMatch:
    """入选相似日及其计划距离。"""

    day: pd.Timestamp
    distance: float


@dataclass(frozen=True)
class SimilarDayResult:
    """相似日模板及可审计的就绪、回退和匹配信息。"""

    ready: bool
    method: str
    reason: Optional[str]
    template: Optional[np.ndarray]
    similar_template: Optional[np.ndarray]
    similar_std: Optional[np.ndarray]
    robust_template: Optional[np.ndarray]
    matches: tuple[SimilarDayMatch, ...]
    temperature: Optional[float]
    n_effective: float
    nearest_distance: Optional[float]
    knn_mean_distance: Optional[float]
    novelty_distance: Optional[float]
    novelty_q75: Optional[float]
    novelty_q95: Optional[float]
    gate: float


def _complete_day_values(values) -> Optional[np.ndarray]:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size != SLOTS_PER_DAY:
        return None
    if not np.isfinite(array).all():
        return None
    return array.copy()


def _require_complete_day(values, name: str) -> np.ndarray:
    array = _complete_day_values(values)
    if array is None:
        raise ValueError(f"{name} must contain exactly 288 finite numeric values")
    return array


def _normalize_day(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("day must not be NaT")
    return timestamp.normalize()


def _normalize_history(history: Mapping) -> dict[pd.Timestamp, object]:
    normalized = {}
    for day, values in history.items():
        normalized[_normalize_day(day)] = values
    return normalized


def _segment_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask & ~np.r_[False, mask[:-1]]))


def _first_slot_features(mask: np.ndarray) -> tuple[float, float, float]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return 0.0, 0.0, 0.0
    angle = 2 * math.pi * int(indices[0]) / SLOTS_PER_DAY
    return math.sin(angle), math.cos(angle), 1.0


def build_natural_day_plan(
    pcs_plan,
    config: SimilarDayConfig = SimilarDayConfig(),
) -> NaturalDayPlan:
    """把 288 点有符号 PCS 计划转换为三块归一化表示。"""
    values = _require_complete_day(pcs_plan, "pcs_plan")
    charge = values < 0
    discharge = values > 0
    standby = ~(charge | discharge)

    duration_energy = np.array(
        [
            charge.sum() * SLOT_HOURS / 24,
            standby.sum() * SLOT_HOURS / 24,
            discharge.sum() * SLOT_HOURS / 24,
            -values[charge].sum() * SLOT_HOURS / (config.power_scale * 24),
            values[discharge].sum() * SLOT_HOURS / (config.power_scale * 24),
        ],
        dtype=float,
    )

    states = np.where(charge, -1, np.where(discharge, 1, 0))
    switch_count = int(np.count_nonzero(states[1:] != states[:-1]))
    charge_sin, charge_cos, has_charge = _first_slot_features(charge)
    discharge_sin, discharge_cos, has_discharge = _first_slot_features(discharge)
    transition = np.array(
        [
            _segment_count(charge) / config.count_scale,
            _segment_count(standby) / config.count_scale,
            _segment_count(discharge) / config.count_scale,
            switch_count / config.count_scale,
            charge_sin,
            charge_cos,
            discharge_sin,
            discharge_cos,
            has_charge,
            has_discharge,
        ],
        dtype=float,
    )
    return NaturalDayPlan(
        curve=values / config.power_scale,
        duration_energy=duration_energy,
        transition=transition,
    )


def plan_distance(
    left: NaturalDayPlan,
    right: NaturalDayPlan,
    config: SimilarDayConfig = SimilarDayConfig(),
) -> float:
    """计算三块表示的加权均方根距离。"""
    curve_rms = math.sqrt(float(np.mean(np.square(left.curve - right.curve))))
    duration_energy_rms = math.sqrt(
        float(np.mean(np.square(left.duration_energy - right.duration_energy)))
    )
    transition_rms = math.sqrt(
        float(np.mean(np.square(left.transition - right.transition)))
    )
    return float(
        config.curve_weight * curve_rms
        + config.duration_energy_weight * duration_energy_rms
        + config.transition_weight * transition_rms
    )


def _candidate_matches(
    target_day: pd.Timestamp,
    history_cutoff_day: pd.Timestamp,
    target_plan: NaturalDayPlan,
    plan_history: Mapping[pd.Timestamp, object],
    ess_history: Mapping[pd.Timestamp, object],
    config: SimilarDayConfig,
) -> tuple[SimilarDayMatch, ...]:
    earliest_day = target_day - pd.Timedelta(days=config.lookback_days)
    candidates = []
    for day, plan_values in plan_history.items():
        if day < earliest_day or day >= target_day or day > history_cutoff_day:
            continue
        if _complete_day_values(ess_history.get(day)) is None:
            continue
        candidate_values = _complete_day_values(plan_values)
        if candidate_values is None:
            continue
        distance = plan_distance(
            target_plan,
            build_natural_day_plan(candidate_values, config),
            config,
        )
        candidates.append(SimilarDayMatch(day=day, distance=distance))
    candidates.sort(key=lambda match: (match.distance, match.day))
    return tuple(candidates[: config.k_neighbors])


def _causal_knn_mean_distances(
    target_day: pd.Timestamp,
    history_cutoff_day: pd.Timestamp,
    plan_history: Mapping[pd.Timestamp, object],
    ess_history: Mapping[pd.Timestamp, object],
    config: SimilarDayConfig,
) -> np.ndarray:
    """逐日回放历史 kNN，只使用当日之前的信息形成校准样本。"""
    means = []
    for day in sorted(plan_history):
        if day >= target_day or day > history_cutoff_day:
            break
        plan_values = _complete_day_values(plan_history[day])
        if plan_values is None or _complete_day_values(ess_history.get(day)) is None:
            continue
        matches = _candidate_matches(
            day,
            day,
            build_natural_day_plan(plan_values, config),
            plan_history,
            ess_history,
            config,
        )
        if matches:
            means.append(float(np.mean([match.distance for match in matches])))
    return np.asarray(means, dtype=float)


def _temperature(causal_distances: np.ndarray) -> float:
    if causal_distances.size:
        median_distance = float(np.median(causal_distances))
        if median_distance > 0:
            return median_distance
    return 1.0


def _weighted_template(
    matches: Sequence[SimilarDayMatch],
    ess_history: Mapping[pd.Timestamp, object],
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    distances = np.asarray([match.distance for match in matches], dtype=float)
    raw_weights = np.exp(-(distances - distances.min()) / temperature)
    weights = raw_weights / raw_weights.sum()
    samples = np.vstack([_complete_day_values(ess_history[match.day]) for match in matches])
    template = np.sum(samples * weights[:, None], axis=0)
    variance = np.sum(weights[:, None] * np.square(samples - template), axis=0)
    n_effective = float(1.0 / np.sum(np.square(weights)))
    return template, np.sqrt(np.maximum(variance, 0.0)), n_effective


def _robust_template(
    target_day: pd.Timestamp,
    history_cutoff_day: pd.Timestamp,
    ess_history: Mapping[pd.Timestamp, object],
    config: SimilarDayConfig,
) -> Optional[np.ndarray]:
    complete_days = []
    for day in sorted(ess_history):
        if day >= target_day or day > history_cutoff_day:
            break
        values = _complete_day_values(ess_history[day])
        if values is not None:
            complete_days.append((day, values))
    if len(complete_days) < config.min_history_days:
        return None
    selected = complete_days[-config.robust_template_days :]
    return np.median(np.vstack([values for _, values in selected]), axis=0)


def _distance_gate(
    novelty_distance: float,
    q75: Optional[float],
    q95: Optional[float],
) -> float:
    if q75 is None or q95 is None:
        return 0.0
    if q95 <= q75:
        return float(novelty_distance > q95)
    return float(np.clip((novelty_distance - q75) / (q95 - q75), 0.0, 1.0))


def estimate_similar_day_template(
    target_day,
    target_plan,
    plan_history: Mapping,
    ess_history: Mapping,
    config: SimilarDayConfig = SimilarDayConfig(),
    history_cutoff_day=None,
) -> SimilarDayResult:
    """生成仅依赖目标日前历史的相似日/稳健混合 ESS 模板。"""
    normalized_target_day = _normalize_day(target_day)
    normalized_history_cutoff_day = (
        normalized_target_day
        if history_cutoff_day is None
        else _normalize_day(history_cutoff_day)
    )
    normalized_plans = _normalize_history(plan_history)
    normalized_ess = _normalize_history(ess_history)
    robust_template = _robust_template(
        normalized_target_day,
        normalized_history_cutoff_day,
        normalized_ess,
        config,
    )
    if robust_template is None:
        return SimilarDayResult(
            ready=False,
            method="not_ready",
            reason="insufficient_complete_history",
            template=None,
            similar_template=None,
            similar_std=None,
            robust_template=None,
            matches=(),
            temperature=None,
            n_effective=0.0,
            nearest_distance=None,
            knn_mean_distance=None,
            novelty_distance=None,
            novelty_q75=None,
            novelty_q95=None,
            gate=1.0,
        )

    target_representation = build_natural_day_plan(target_plan, config)
    matches = _candidate_matches(
        normalized_target_day,
        normalized_history_cutoff_day,
        target_representation,
        normalized_plans,
        normalized_ess,
        config,
    )
    causal_distances = _causal_knn_mean_distances(
        normalized_target_day,
        normalized_history_cutoff_day,
        normalized_plans,
        normalized_ess,
        config,
    )
    novelty_q75 = (
        float(np.quantile(causal_distances, config.q75))
        if causal_distances.size
        else None
    )
    novelty_q95 = (
        float(np.quantile(causal_distances, config.q95))
        if causal_distances.size
        else None
    )

    if not matches:
        return SimilarDayResult(
            ready=True,
            method="robust_fallback",
            reason="no_complete_candidates",
            template=robust_template.copy(),
            similar_template=None,
            similar_std=None,
            robust_template=robust_template,
            matches=(),
            temperature=None,
            n_effective=0.0,
            nearest_distance=None,
            knn_mean_distance=None,
            novelty_distance=None,
            novelty_q75=novelty_q75,
            novelty_q95=novelty_q95,
            gate=1.0,
        )

    temperature = _temperature(causal_distances)
    similar_template, similar_std, n_effective = _weighted_template(
        matches, normalized_ess, temperature
    )
    nearest_distance = float(matches[0].distance)
    knn_mean_distance = float(np.mean([match.distance for match in matches]))
    novelty_distance = nearest_distance
    gate = max(
        _distance_gate(novelty_distance, novelty_q75, novelty_q95),
        float(n_effective < config.min_effective_samples),
    )
    template = (1.0 - gate) * similar_template + gate * robust_template
    return SimilarDayResult(
        ready=True,
        method="blended",
        reason=None,
        template=template,
        similar_template=similar_template,
        similar_std=similar_std,
        robust_template=robust_template,
        matches=matches,
        temperature=temperature,
        n_effective=n_effective,
        nearest_distance=nearest_distance,
        knn_mean_distance=knn_mean_distance,
        novelty_distance=novelty_distance,
        novelty_q75=novelty_q75,
        novelty_q95=novelty_q95,
        gate=gate,
    )
