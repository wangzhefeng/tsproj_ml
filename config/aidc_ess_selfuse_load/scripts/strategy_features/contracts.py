"""ESS 策略特征 v2 的模型列与未来泄漏契约。"""

import re


PLAN_CYCLE_FEATURE_COLUMNS = [
    "plan_cycle_charge_hours",
    "plan_cycle_standby_hours",
    "plan_cycle_discharge_hours",
    "plan_cycle_charge_segment_count",
    "plan_cycle_standby_segment_count",
    "plan_cycle_discharge_segment_count",
    "plan_cycle_charge_energy_kwh",
    "plan_cycle_discharge_energy_kwh",
    "plan_cycle_switch_count",
    "plan_cycle_max_ramp",
    "plan_cycle_first_charge_slot_sin",
    "plan_cycle_first_charge_slot_cos",
    "plan_cycle_first_discharge_slot_sin",
    "plan_cycle_first_discharge_slot_cos",
    "plan_cycle_has_charge",
    "plan_cycle_has_discharge",
]

LAG_FEATURE_COLUMNS = [
    "ess_lag_288",
    "pcs_actual_lag_288",
    "actual_operating_charge_lag_288",
    "actual_operating_standby_lag_288",
    "actual_operating_discharge_lag_288",
    "last_completed_cycle_charge_hours",
    "last_completed_cycle_standby_hours",
    "last_completed_cycle_discharge_hours",
    "last_completed_cycle_plan_state_agreement",
    "lag_feature_ready",
]

SIMILAR_DAY_FEATURE_COLUMNS = [
    "plan_nearest_day_distance",
    "plan_knn_mean_distance",
    "plan_novelty_score",
    "plan_is_novel",
    "similar_day_effective_samples",
    "plan_similar_day_ess_template",
    "plan_similar_day_template_std",
    "robust_recent_ess_template",
    "template_gate_weight",
    "gated_ess_template",
    "template_feature_ready",
]

JOINT_CLUSTER_FEATURE_COLUMNS = [
    "joint_cluster_lag1_c0",
    "joint_cluster_lag1_c1",
    "joint_cluster_lag1_c2",
    "joint_cluster_lag1_c3",
    "joint_cluster_lag1_c4",
    "joint_cluster_lag1_distance",
    "joint_cluster_lag1_rare",
    "joint_cluster_feature_ready",
]

MODEL_FEATURE_COLUMNS = [
    "time",
    "pcs_plan",
    "plan_direction_charge",
    "plan_direction_standby",
    "plan_direction_discharge",
    "plan_power_abs",
    *PLAN_CYCLE_FEATURE_COLUMNS,
    *LAG_FEATURE_COLUMNS,
    *SIMILAR_DAY_FEATURE_COLUMNS,
    *JOINT_CLUSTER_FEATURE_COLUMNS,
]

CRITICAL_FUTURE_COLUMNS = set(MODEL_FEATURE_COLUMNS) - {"time"}

FORBIDDEN_FUTURE_NAMES = {
    "value",
    "target",
    "ess_power",
    "pcs_power",
    "actual_charge_binary",
    "actual_discharge_binary",
    "schedule_pattern_id",
}

FORBIDDEN_FUTURE_PATTERNS = (
    re.compile(r"^(?:value|target|y|ess_power|pcs_power)$"),
    re.compile(r"^actual_(?!operating_(?:charge|standby|discharge)_lag_288$)"),
    re.compile(r"(?:future|current)_(?:ess|actual|pcs)"),
    re.compile(r"(?:same_day|posthoc|cluster)_pattern"),
)
