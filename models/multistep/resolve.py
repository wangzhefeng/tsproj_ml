# -*- coding: utf-8 -*-
"""旧配置到唯一 ResolvedStrategy 的集中解析与校验。"""

from typing import Any, Iterable

from models.multistep.plans import (
    BackfillPolicy,
    ExogenousTiming,
    FeaturePlan,
    LagPolicy,
    ResolvedStrategy,
    RowAlignment,
    RuntimePlan,
    TargetLayout,
    TargetPlan,
    TrainingLayout,
    TrainingPlan,
)
from models.multistep.spec import InputScope, RolloutFamily, get_strategy_spec


def _int_lags(values: Iterable[Any]) -> tuple[int, ...]:
    lags = tuple(int(value) for value in (values or ()))
    invalid = tuple(value for value in lags if value <= 0)
    if invalid:
        raise ValueError(f"lags must contain positive integers; got {list(invalid)}.")
    return lags


def _target_columns(target_feature: str, steps: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(f"{target_feature}_shift_{step}" for step in steps)


def _resolve_block_size(args: Any, horizon: int, lags: tuple[int, ...]) -> int:
    block_size = int(getattr(args, "block_size", 0) or 0)
    if block_size < 0:
        raise ValueError(f"block_size must be >= 0; got {block_size}.")
    if block_size > horizon:
        raise ValueError(
            f"explicit block_size must not exceed horizon; got block_size={block_size}, horizon={horizon}."
        )
    candidate = block_size if block_size > 0 else (min(lags) if lags else 1)
    return min(horizon, candidate)


def _validate_options(args: Any, spec, horizon: int, lags: tuple[int, ...]) -> None:
    direct_strategy = str(getattr(args, "direct_strategy", "multioutput") or "multioutput").lower()
    if direct_strategy not in {"multioutput", "horizon_feature"}:
        raise ValueError(f"Unsupported direct_strategy={direct_strategy!r}.")
    if direct_strategy == "horizon_feature" and spec.rollout != RolloutFamily.DIRECT:
        raise ValueError("direct_strategy=horizon_feature is only valid for USMD/MSMD.")

    block_size = int(getattr(args, "block_size", 0) or 0)
    if spec.rollout != RolloutFamily.DIRREC and block_size != 0:
        raise ValueError("block_size>0 is only valid for USMDR/MSMDR.")
    if spec.rollout == RolloutFamily.DIRREC:
        _resolve_block_size(args, horizon, lags)

    backfill = str(getattr(args, "endogenous_backfill_strategy", "persistence") or "persistence").lower()
    if backfill not in {"persistence", "auxiliary"}:
        raise ValueError(f"Unsupported endogenous_backfill_strategy={backfill!r}.")
    if backfill != "persistence" and not (
        spec.input_scope == InputScope.ALL_ENDOGENOUS
        and spec.rollout in {RolloutFamily.RECURSIVE, RolloutFamily.DIRREC}
    ):
        raise ValueError("endogenous_backfill_strategy=auxiliary is only valid for MSMR/MSMDR.")

    blend_strategy = str(getattr(args, "blend_weight_strategy", "fixed") or "fixed").lower()
    if blend_strategy not in {"fixed", "ridge_stacking"}:
        raise ValueError(f"Unsupported blend_weight_strategy={blend_strategy!r}.")
    blend_weights = list(getattr(args, "blend_weights", [0.5, 0.5]) or [])
    if spec.rollout != RolloutFamily.BLEND:
        if blend_strategy != "fixed" or blend_weights != [0.5, 0.5]:
            raise ValueError("blend weight options are only valid for USBR/MSBR.")
    elif len(blend_weights) != 2:
        raise ValueError("blend_weights must contain exactly two values.")

    if bool(getattr(args, "use_horizon_exogenous_for_direct", False)) and spec.rollout not in {
        RolloutFamily.DIRECT,
        RolloutFamily.DIRREC,
    }:
        raise ValueError("use_horizon_exogenous_for_direct is only valid for Direct/DirRec methods.")

    align_to_target = bool(getattr(args, "align_direct_features_to_target", False))
    if align_to_target:
        if spec.rollout == RolloutFamily.POINTWISE:
            if not bool(getattr(args, "enable_lags_features", False)):
                raise ValueError("USMDP safe-lag requires enable_lags_features=true.")
            if not lags:
                raise ValueError("USMDP safe-lag requires at least one positive lag.")
            if min(lags) < horizon:
                raise ValueError(
                    f"USMDP safe-lag requires min(lags) >= horizon; got min_lag={min(lags)}, horizon={horizon}."
                )
        elif spec.rollout in {RolloutFamily.DIRECT, RolloutFamily.DIRREC}:
            if horizon != 1:
                raise ValueError(
                    "align_direct_features_to_target currently requires predict_steps=1 "
                    f"for pred_method={spec.method}."
                )
        else:
            raise ValueError("align_direct_features_to_target is only valid for USMDP/Direct/DirRec.")

    if bool(getattr(args, "enable_global_training", False)):
        series_id_feature = str(
            getattr(args, "series_id_feature", "series_id") or ""
        ).strip()
        if not series_id_feature:
            raise ValueError("Global panel requires a non-empty series_id_feature.")
        incomplete_policy = str(
            getattr(args, "global_incomplete_series_policy", "raise") or "raise"
        ).lower()
        if incomplete_policy not in {"raise", "drop"}:
            raise ValueError(
                "global_incomplete_series_policy must be 'raise' or 'drop'."
            )
        unknown_policy = str(
            getattr(args, "global_unknown_series_policy", "raise") or "raise"
        ).lower()
        if unknown_policy != "raise":
            raise ValueError(
                "global_unknown_series_policy currently supports only 'raise'."
            )
        if str(
            getattr(args, "target_calendar_normalization", "none") or "none"
        ).lower() != "none":
            raise ValueError(
                "Global panel does not support target_calendar_normalization; "
                "normalization state must be series-specific."
            )
        if str(getattr(args, "decomposition_method", "none") or "none").lower() != "none":
            raise ValueError(
                "Global panel does not support target decomposition; "
                "decomposition state must be series-specific."
            )
        if str(getattr(args, "endogenous_backfill_strategy", "persistence") or "persistence").lower() == "auxiliary":
            raise ValueError(
                "Global panel does not support auxiliary endogenous backfill; "
                "auxiliary models must be series-specific."
            )
        if str(getattr(args, "horizon_mode", "fixed_steps") or "fixed_steps").lower() != "fixed_steps":
            raise ValueError("Global panel currently requires horizon_mode=fixed_steps.")


def resolve_strategy(args: Any, horizon: int, target_feature: str = "y") -> ResolvedStrategy:
    horizon = int(horizon)
    if horizon <= 0:
        raise ValueError(f"forecast horizon must be positive; got {horizon}.")

    spec = get_strategy_spec(getattr(args, "pred_method", ""))
    lags = _int_lags(getattr(args, "lags", ()) or ())
    _validate_options(args, spec, horizon, lags)

    direct_strategy = str(getattr(args, "direct_strategy", "multioutput") or "multioutput").lower()
    if spec.rollout == RolloutFamily.POINTWISE:
        steps = (0,)
        target_layout = TargetLayout.POINTWISE
        training_layout = TrainingLayout.SINGLE_OUTPUT
        model_width = 1
    elif spec.rollout == RolloutFamily.DIRECT:
        steps = tuple(range(1, horizon + 1))
        if direct_strategy == "horizon_feature":
            target_layout = TargetLayout.HORIZON_LONG
            training_layout = TrainingLayout.HORIZON_LONG
            model_width = 1
        else:
            target_layout = TargetLayout.MULTI_OUTPUT
            training_layout = TrainingLayout.MULTI_OUTPUT
            model_width = horizon
    elif spec.rollout == RolloutFamily.RECURSIVE:
        steps = (0,)
        target_layout = TargetLayout.SINGLE_OUTPUT
        training_layout = TrainingLayout.SINGLE_OUTPUT
        model_width = 1
    elif spec.rollout == RolloutFamily.DIRREC:
        block_size = _resolve_block_size(args, horizon, lags)
        steps = tuple(range(1, block_size + 1))
        target_layout = TargetLayout.MULTI_OUTPUT
        training_layout = TrainingLayout.MULTI_OUTPUT
        model_width = block_size
    else:
        direct_steps = tuple(range(1, horizon + 1))
        recursive_steps = (0,)
        direct_target = TargetPlan(direct_steps, _target_columns(target_feature, direct_steps), TargetLayout.MULTI_OUTPUT)
        recursive_target = TargetPlan(recursive_steps, _target_columns(target_feature, recursive_steps), TargetLayout.SINGLE_OUTPUT)
        target_plan = TargetPlan(
            direct_steps + recursive_steps,
            direct_target.column_names + recursive_target.column_names,
            TargetLayout.COMPOSITE,
            direct=direct_target,
            recursive=recursive_target,
        )
        direct_training = TrainingPlan(TrainingLayout.MULTI_OUTPUT, horizon)
        recursive_training = TrainingPlan(TrainingLayout.SINGLE_OUTPUT, 1)
        training_plan = TrainingPlan(
            TrainingLayout.COMPOSITE,
            horizon + 1,
            direct=direct_training,
            recursive=recursive_training,
        )
        direct_runtime = RuntimePlan(RolloutFamily.DIRECT.value, horizon, horizon)
        recursive_runtime = RuntimePlan(RolloutFamily.RECURSIVE.value, horizon, 1)
        runtime_plan = RuntimePlan(
            RolloutFamily.BLEND.value,
            horizon,
            horizon + 1,
            direct=direct_runtime,
            recursive=recursive_runtime,
        )
        return ResolvedStrategy(
            spec=spec,
            horizon=horizon,
            target_plan=target_plan,
            feature_plan=_feature_plan(args, spec),
            training_plan=training_plan,
            runtime_plan=runtime_plan,
        )

    columns = _target_columns(target_feature, steps)
    if target_layout == TargetLayout.HORIZON_LONG:
        columns = (columns[0],)
    target_plan = TargetPlan(steps, columns, target_layout)
    runtime_plan = RuntimePlan(
        spec.rollout.value,
        horizon,
        model_width,
        block_size=_resolve_block_size(args, horizon, lags) if spec.rollout == RolloutFamily.DIRREC else None,
    )
    return ResolvedStrategy(
        spec=spec,
        horizon=horizon,
        target_plan=target_plan,
        feature_plan=_feature_plan(args, spec),
        training_plan=TrainingPlan(training_layout, model_width),
        runtime_plan=runtime_plan,
    )


def _feature_plan(args: Any, spec) -> FeaturePlan:
    align_to_target = bool(getattr(args, "align_direct_features_to_target", False))
    if spec.rollout == RolloutFamily.POINTWISE:
        lag_policy = LagPolicy.SAFE_TARGET_ROW if align_to_target else LagPolicy.NONE
        # USMDP safe-lag 在目标行上直接构造 lag（不 shift 外生列），
        # 行对齐保持 FORECAST_ORIGIN，避免外生特征被错误前移。
        row_alignment = RowAlignment.FORECAST_ORIGIN
    else:
        lag_policy = LagPolicy.STANDARD
        row_alignment = RowAlignment.TARGET_TIME if align_to_target else RowAlignment.FORECAST_ORIGIN
    exogenous_timing = (
        ExogenousTiming.BY_HORIZON
        if bool(getattr(args, "use_horizon_exogenous_for_direct", False))
        else ExogenousTiming.FORECAST_ORIGIN
    )
    backfill_policy = None
    if spec.input_scope == InputScope.ALL_ENDOGENOUS and spec.rollout in {
        RolloutFamily.RECURSIVE,
        RolloutFamily.DIRREC,
    }:
        backfill_policy = BackfillPolicy(
            str(getattr(args, "endogenous_backfill_strategy", "persistence") or "persistence").lower()
        )
    panel_key = (
        str(getattr(args, "series_id_feature", "series_id"))
        if bool(getattr(args, "enable_global_training", False))
        else None
    )
    return FeaturePlan(
        input_scope=spec.input_scope,
        lag_policy=lag_policy,
        row_alignment=row_alignment,
        exogenous_timing=exogenous_timing,
        backfill_policy=backfill_policy,
        panel_key=panel_key,
    )
