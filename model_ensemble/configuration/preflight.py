"""融合成员的静态预检；全部通过后才构造任何 runner。"""
from dataclasses import replace

from forecasting_core.specs.config import parse_model_config
from model_ensemble.configuration.specs import EnsembleSpecError


def parse_member_configs(config, resolved):
    members = {}
    shared_validation = config.validation.canonical_payload()
    for ref in config.members:
        try:
            member = parse_model_config(resolved[ref.name], source=ref.config_ref)
        except (TypeError, ValueError) as exc:
            raise EnsembleSpecError(f"invalid member {ref.name!r} ({ref.config_ref}): {exc}") from exc
        validation = member.validation
        for key in ("train_history_steps", "training_window", "forecast_window"):
            if validation.get(key) is not None:
                raise EnsembleSpecError(f"Ensemble members do not support {key}")
        if validation.get("refit_every", 1) != 1:
            raise EnsembleSpecError("Ensemble members do not support non-default refit_every")
        if validation.get("horizon_mode", "fixed_steps") not in {"fixed_steps", "sliding_window"}:
            raise EnsembleSpecError("Ensemble members require fixed-step horizons")
        unused = set(validation.get("training", {})) - {"sample_weight", "origin_sampling"}
        if unused or "train_outlier" in validation:
            raise EnsembleSpecError(f"Ensemble member has unconsumed training options: {sorted(unused | ({'train_outlier'} if 'train_outlier' in validation else set()))}")
        if member.probabilistic.get("calibration"):
            raise EnsembleSpecError("Ensemble members do not support calibration; configure fusion-level calibration")
        if member.features.transformations.get("seasonal_baseline") is not None:
            raise EnsembleSpecError("Ensemble members do not support seasonal_baseline deployment")
        # 发报调度与 panel 口径由融合顶层拥有；训练窗口/变换仍由成员拥有。
        effective = validation.canonical_payload()
        effective["schedule_mode"] = shared_validation.get("schedule_mode", "daily")
        for key in ("training_scope", "seasonal_naive_lag"):
            if key in shared_validation:
                effective[key] = shared_validation[key]
        members[ref.name] = replace(member, validation=effective)
    return members
