"""ESS 策略特征 v2 公共 API。"""

from .profiles import summarize_dispatch_profiles
from .states import (
    OperatingThresholds,
    encode_actual_operating_state,
    encode_plan_direction,
)
from .windows import (
    HistoryTimestampAudit,
    audit_history_timestamps,
    calendar_day_slot,
    dispatch_cycle_slot,
    dispatch_cycle_start,
    validate_future_timestamps,
)

from .contracts import (
    CRITICAL_FUTURE_COLUMNS,
    FORBIDDEN_FUTURE_NAMES,
    FORBIDDEN_FUTURE_PATTERNS,
    JOINT_CLUSTER_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from .joint_clustering import (
    JointClusterArtifact,
    JointClusteringConfig,
    build_joint_lag_features,
    fit_joint_cluster_artifact,
    transform_joint_day,
)
from .pipeline import build_strategy_features, load_strategy_config

__all__ = [
    "CRITICAL_FUTURE_COLUMNS",
    "FORBIDDEN_FUTURE_NAMES",
    "FORBIDDEN_FUTURE_PATTERNS",
    "HistoryTimestampAudit",
    "JOINT_CLUSTER_FEATURE_COLUMNS",
    "JointClusterArtifact",
    "JointClusteringConfig",
    "MODEL_FEATURE_COLUMNS",
    "OperatingThresholds",
    "audit_history_timestamps",
    "build_strategy_features",
    "build_joint_lag_features",
    "calendar_day_slot",
    "dispatch_cycle_slot",
    "dispatch_cycle_start",
    "encode_actual_operating_state",
    "encode_plan_direction",
    "fit_joint_cluster_artifact",
    "load_strategy_config",
    "summarize_dispatch_profiles",
    "transform_joint_day",
    "validate_future_timestamps",
]
