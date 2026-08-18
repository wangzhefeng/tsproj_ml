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
    MODEL_FEATURE_COLUMNS,
)
from .pipeline import build_strategy_features, load_strategy_config

__all__ = [
    "CRITICAL_FUTURE_COLUMNS",
    "FORBIDDEN_FUTURE_NAMES",
    "FORBIDDEN_FUTURE_PATTERNS",
    "HistoryTimestampAudit",
    "MODEL_FEATURE_COLUMNS",
    "OperatingThresholds",
    "audit_history_timestamps",
    "build_strategy_features",
    "calendar_day_slot",
    "dispatch_cycle_slot",
    "dispatch_cycle_start",
    "encode_actual_operating_state",
    "encode_plan_direction",
    "load_strategy_config",
    "summarize_dispatch_profiles",
    "validate_future_timestamps",
]
