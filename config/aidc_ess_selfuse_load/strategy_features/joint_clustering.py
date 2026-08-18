"""因果安全的 ESS/实际 PCS/计划 PCS 三视图联合聚类。"""

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


VIEW_NAMES = ("ess", "actual", "plan")


@dataclass(frozen=True)
class JointClusteringConfig:
    """联合聚类训练参数。"""

    pca_variance_ratio: float = 0.90
    candidate_clusters: tuple[int, ...] = (2, 3, 4, 5)
    max_clusters: int = 5
    rare_cluster_min_days: int = 10
    random_state: int = 42
    n_init: int = 20

    def __post_init__(self) -> None:
        if not 0.0 < self.pca_variance_ratio <= 1.0:
            raise ValueError("pca_variance_ratio must be in (0, 1]")
        if not self.candidate_clusters:
            raise ValueError("candidate_clusters must not be empty")
        if tuple(sorted(set(self.candidate_clusters))) != self.candidate_clusters:
            raise ValueError("candidate_clusters must be unique and increasing")
        if min(self.candidate_clusters) < 2:
            raise ValueError("candidate_clusters must start at 2 or greater")
        if self.max_clusters < max(self.candidate_clusters):
            raise ValueError("max_clusters must cover candidate_clusters")
        if self.rare_cluster_min_days < 1:
            raise ValueError("rare_cluster_min_days must be positive")
        if self.n_init < 1:
            raise ValueError("n_init must be positive")


@dataclass
class ConstantPCA:
    """常量 block 的一维零投影，避免 PCA 产生 NaN 解释率。"""

    n_features_in_: int
    n_components_: int = 1
    explained_variance_ratio_: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0], dtype=float)
    )

    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.zeros((len(values), 1), dtype=float)


@dataclass
class JointClusterArtifact:
    """固定 reference 期拟合的可序列化联合聚类 artifact。"""

    config: JointClusteringConfig
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    reference_days: tuple[pd.Timestamp, ...]
    scalers: dict[str, StandardScaler]
    pcas: dict[str, PCA | ConstantPCA]
    block_scales: dict[str, float]
    kmeans: KMeans
    selected_k: int
    silhouette_scores: dict[int, float]
    raw_to_canonical: dict[int, int]
    cluster_counts: dict[int, int]
    rare_clusters: tuple[int, ...]


@dataclass(frozen=True)
class JointClusterResult:
    """单个完整自然日的联合聚类结果。"""

    cluster_id: int
    distance: float
    rare: int


def _complete_curve(values, name: str) -> np.ndarray:
    curve = np.asarray(values, dtype=float)
    if curve.shape != (288,):
        raise ValueError(f"{name} curve must contain exactly 288 points")
    if not np.isfinite(curve).all():
        raise ValueError(f"{name} curve must contain only finite values")
    return curve


def _normalize_days(values: Mapping) -> dict[pd.Timestamp, np.ndarray]:
    return {
        pd.Timestamp(day).normalize(): np.asarray(curve, dtype=float)
        for day, curve in values.items()
    }


def _reference_matrices(
    ess_days: Mapping,
    actual_days: Mapping,
    plan_days: Mapping,
    fit_end: pd.Timestamp,
) -> tuple[tuple[pd.Timestamp, ...], dict[str, np.ndarray]]:
    normalized = {
        "ess": _normalize_days(ess_days),
        "actual": _normalize_days(actual_days),
        "plan": _normalize_days(plan_days),
    }
    common_days = sorted(
        set(normalized["ess"])
        & set(normalized["actual"])
        & set(normalized["plan"])
    )
    reference_days = tuple(day for day in common_days if day <= fit_end)
    if len(reference_days) < 3:
        raise ValueError("joint clustering requires at least 3 complete reference days")

    matrices = {}
    for view in VIEW_NAMES:
        matrices[view] = np.vstack(
            [_complete_curve(normalized[view][day], view) for day in reference_days]
        )
    return reference_days, matrices


def _fit_view_transforms(
    matrices: dict[str, np.ndarray],
    config: JointClusteringConfig,
) -> tuple[
    dict[str, StandardScaler],
    dict[str, PCA | ConstantPCA],
    dict[str, float],
    np.ndarray,
]:
    scalers = {}
    pcas = {}
    block_scales = {}
    blocks = []
    for view in VIEW_NAMES:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrices[view])
        if float(np.var(scaled, axis=0).sum()) <= 1e-12:
            pca = ConstantPCA(n_features_in_=scaled.shape[1])
            transformed = pca.transform(scaled)
        else:
            pca = PCA(n_components=config.pca_variance_ratio, svd_solver="full")
            transformed = pca.fit_transform(scaled)
        block_scale = math.sqrt(transformed.shape[1])
        scalers[view] = scaler
        pcas[view] = pca
        block_scales[view] = block_scale
        blocks.append(transformed / block_scale)
    return scalers, pcas, block_scales, np.hstack(blocks)


def _canonical_label_map(centers: np.ndarray) -> dict[int, int]:
    ordered = sorted(range(len(centers)), key=lambda index: tuple(centers[index].tolist()))
    return {raw: canonical for canonical, raw in enumerate(ordered)}


def fit_joint_cluster_artifact(
    ess_days: Mapping,
    actual_days: Mapping,
    plan_days: Mapping,
    *,
    fit_end,
    config: JointClusteringConfig = JointClusteringConfig(),
) -> JointClusterArtifact:
    """只用 ``day <= fit_end`` 的完整三视图自然日拟合 artifact。"""
    normalized_fit_end = pd.Timestamp(fit_end).normalize()
    reference_days, matrices = _reference_matrices(
        ess_days, actual_days, plan_days, normalized_fit_end
    )
    scalers, pcas, block_scales, joint = _fit_view_transforms(matrices, config)

    unique_count = len(np.unique(np.round(joint, decimals=12), axis=0))
    valid_candidates = [
        clusters
        for clusters in config.candidate_clusters
        if clusters < len(reference_days) and clusters <= unique_count
    ]
    if not valid_candidates:
        raise ValueError("no candidate cluster count is valid for the reference data")

    fitted = {}
    scores = {}
    for clusters in valid_candidates:
        model = KMeans(
            n_clusters=clusters,
            random_state=config.random_state,
            n_init=config.n_init,
        ).fit(joint)
        labels = model.labels_
        if len(np.unique(labels)) < 2:
            continue
        fitted[clusters] = model
        scores[clusters] = float(silhouette_score(joint, labels))
    if not scores:
        raise ValueError("joint clustering could not produce at least two clusters")

    selected_k = min(scores, key=lambda clusters: (-scores[clusters], clusters))
    kmeans = fitted[selected_k]
    raw_to_canonical = _canonical_label_map(kmeans.cluster_centers_)
    canonical_labels = np.asarray(
        [raw_to_canonical[int(label)] for label in kmeans.labels_], dtype=int
    )
    cluster_counts = {
        cluster: int(np.sum(canonical_labels == cluster))
        for cluster in range(selected_k)
    }
    rare_clusters = tuple(
        cluster
        for cluster, count in cluster_counts.items()
        if count < config.rare_cluster_min_days
    )
    return JointClusterArtifact(
        config=config,
        fit_start=reference_days[0],
        fit_end=normalized_fit_end,
        reference_days=reference_days,
        scalers=scalers,
        pcas=pcas,
        block_scales=block_scales,
        kmeans=kmeans,
        selected_k=selected_k,
        silhouette_scores=scores,
        raw_to_canonical=raw_to_canonical,
        cluster_counts=cluster_counts,
        rare_clusters=rare_clusters,
    )


def _transform_views(
    artifact: JointClusterArtifact,
    ess_curve,
    actual_curve,
    plan_curve,
) -> np.ndarray:
    values = {
        "ess": _complete_curve(ess_curve, "ess"),
        "actual": _complete_curve(actual_curve, "actual"),
        "plan": _complete_curve(plan_curve, "plan"),
    }
    blocks = []
    for view in VIEW_NAMES:
        scaled = artifact.scalers[view].transform(values[view][None, :])
        transformed = artifact.pcas[view].transform(scaled)
        blocks.append(transformed / artifact.block_scales[view])
    return np.hstack(blocks)


def transform_joint_day(
    artifact: JointClusterArtifact,
    ess_curve,
    actual_curve,
    plan_curve,
) -> JointClusterResult:
    """用固定 artifact 变换一个完整自然日。"""
    joint = _transform_views(artifact, ess_curve, actual_curve, plan_curve)
    raw_label = int(artifact.kmeans.predict(joint)[0])
    cluster_id = artifact.raw_to_canonical[raw_label]
    distance = float(np.linalg.norm(joint[0] - artifact.kmeans.cluster_centers_[raw_label]))
    return JointClusterResult(
        cluster_id=cluster_id,
        distance=distance,
        rare=int(cluster_id in artifact.rare_clusters),
    )


def joint_cluster_feature_columns(max_clusters: int) -> list[str]:
    """返回固定顺序的 lag1 联合聚类模型列。"""
    return [
        *[f"joint_cluster_lag1_c{cluster}" for cluster in range(max_clusters)],
        "joint_cluster_lag1_distance",
        "joint_cluster_lag1_rare",
        "joint_cluster_feature_ready",
    ]


def _has_complete_day(values: Mapping[pd.Timestamp, np.ndarray], day: pd.Timestamp) -> bool:
    if day not in values:
        return False
    curve = np.asarray(values[day], dtype=float)
    return curve.shape == (288,) and bool(np.isfinite(curve).all())


def build_joint_lag_features(
    grid: pd.DatetimeIndex,
    artifact: JointClusterArtifact,
    ess_days: Mapping,
    actual_days: Mapping,
    plan_days: Mapping,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将 D-1 完整自然日的联合聚类结果广播到 D 日每个时间点。"""
    normalized = {
        "ess": _normalize_days(ess_days),
        "actual": _normalize_days(actual_days),
        "plan": _normalize_days(plan_days),
    }
    columns = joint_cluster_feature_columns(artifact.config.max_clusters)
    features = pd.DataFrame(0.0, index=grid, columns=columns)
    assignment_rows = []
    for target_day in pd.DatetimeIndex(grid.normalize().unique()):
        source_day = target_day - pd.Timedelta(days=1)
        ready = all(
            _has_complete_day(normalized[view], source_day) for view in VIEW_NAMES
        )
        cluster_id = -1
        distance = 0.0
        rare = 0
        if ready:
            result = transform_joint_day(
                artifact,
                normalized["ess"][source_day],
                normalized["actual"][source_day],
                normalized["plan"][source_day],
            )
            cluster_id = result.cluster_id
            distance = result.distance
            rare = result.rare
            day_mask = grid.normalize() == target_day
            features.loc[day_mask, f"joint_cluster_lag1_c{cluster_id}"] = 1.0
            features.loc[day_mask, "joint_cluster_lag1_distance"] = distance
            features.loc[day_mask, "joint_cluster_lag1_rare"] = rare
            features.loc[day_mask, "joint_cluster_feature_ready"] = 1
        assignment_rows.append(
            {
                "target_day": target_day,
                "source_day": source_day,
                "cluster_id": cluster_id,
                "distance": distance,
                "rare": rare,
                "ready": int(ready),
                "fit_end": artifact.fit_end,
            }
        )
    return features, pd.DataFrame(assignment_rows)
