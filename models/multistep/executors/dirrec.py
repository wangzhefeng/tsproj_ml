# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from models.multistep.contracts import require_dirrec_output, require_endogenous_history
from models.multistep.runtime import (
    context_horizon,
    ensure_resolved_strategy,
    log_execution_summary,
)
from models.multistep.spec import InputScope


def iter_block_ranges(horizon: int, block_size: int):
    horizon = int(horizon)
    block_size = int(block_size)
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if block_size < 1:
        raise ValueError("block_size must be >= 1.")
    for start in range(0, horizon, block_size):
        yield start, min(start + block_size, horizon)


def resolve_runtime_block_size(context, resolved) -> int:
    """旧 DirRec 模型按实际输出宽度执行；新模型使用解析块长。"""
    width = getattr(context, "model_output_width", None)
    block_size = int(width or resolved.runtime_plan.block_size or 1)
    if block_size < 1 or block_size > int(resolved.horizon):
        raise ValueError(
            "DirRec runtime block size must be within forecast horizon; "
            f"got block_size={block_size}, horizon={resolved.horizon}."
        )
    return block_size


class DirRecExecutor:
    def execute(self, context):
        if ensure_resolved_strategy(context).spec.input_scope == InputScope.ALL_ENDOGENOUS:
            require_endogenous_history(context.df_history_for_lags, context.endogenous_features)
            return self._execute_multivariate(context)
        return self._execute_univariate(context)

    def _execute_univariate(self, context):
        resolved = ensure_resolved_strategy(context)
        horizon = context_horizon(context)
        block_size = resolve_runtime_block_size(context, resolved)
        predictions = []
        quantile_store = {}
        block_calls = 0
        for start, stop in iter_block_ranges(horizon, block_size):
            block_calls += 1
            future_remain = context.df_future.iloc[start:].copy()
            forecast_frame = pd.concat(
                [context.df_history_for_lags, future_remain], ignore_index=True, copy=False
            )
            date_future, weather_future = context._slice_future_aux_by_forecast(forecast_frame)
            featured, predictor_features, _, categorical_features = context.feature_engineer.create_features(
                df_series=forecast_frame,
                df_date_history=None,
                df_date_future=date_future,
                df_weather_history=None,
                df_weather_future=weather_future,
                df_custom_future=context.df_custom_future,
                endogenous_features_with_target=context.endogenous_features,
                target_feature=context.target_feature,
                horizon=horizon,
            )
            predictor_features, categorical_features = context._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            anchor = max(len(context.df_history_for_lags) - 1, 0)
            x_forecast = featured.reindex(columns=predictor_features).iloc[anchor : anchor + 1]
            x_processed = context._transform_features(x_forecast, categorical_features)
            point_pred, quantile_preds = context._predict_point_and_quantiles(x_processed)
            block_pred = require_dirrec_output(point_pred, block_size)
            take = stop - start
            predictions.extend(block_pred[:take])
            if quantile_preds:
                for level, prediction in quantile_preds.items():
                    quantile_block = require_dirrec_output(prediction, block_size)
                    quantile_store.setdefault(level, []).extend(quantile_block[:take])
            for offset, value in enumerate(block_pred[:take]):
                new_row = future_remain.iloc[offset : offset + 1].copy()
                new_row[context.target_feature] = float(value)
                context._append_history_row(new_row)
        context._finalize_recursive_quantiles(quantile_store)
        log_execution_summary(
            context,
            "dirrec",
            model_calls=block_calls * (len(quantile_store) if quantile_store else 1),
            block_size=block_size,
        )
        return np.asarray(predictions, dtype=float)

    def _execute_multivariate(self, context):
        resolved = ensure_resolved_strategy(context)
        horizon = context_horizon(context)
        block_size = resolve_runtime_block_size(context, resolved)
        use_feature_cache = bool(getattr(context.args, "enable_feature_cache", False))
        runtime_cache = context._prepare_msmdr_runtime() if use_feature_cache else None
        predictions = []
        quantile_store = {}
        backfill_sources = set()
        block_calls = 0
        other_endogenous = [
            feature for feature in context.endogenous_features if feature != context.target_feature
        ]
        for start, stop in iter_block_ranges(horizon, block_size):
            block_calls += 1
            future_remain = context.df_future.iloc[start:].copy()
            if use_feature_cache:
                assert runtime_cache is not None
                x_forecast = context._build_msmr_step_input(runtime_cache, start)
                categorical_features = runtime_cache.categorical_features
            else:
                forecast_frame = pd.concat(
                    [context.df_history_for_lags, future_remain], ignore_index=True, copy=False
                )
                date_future, weather_future = context._slice_future_aux_by_forecast(forecast_frame)
                featured, predictor_features, _, categorical_features = context.feature_engineer.create_features(
                    df_series=forecast_frame,
                    df_date_history=None,
                    df_date_future=date_future,
                    df_weather_history=None,
                    df_weather_future=weather_future,
                    df_custom_future=context.df_custom_future,
                    endogenous_features_with_target=context.endogenous_features,
                    target_feature=context.target_feature,
                    horizon=horizon,
                )
                predictor_features, categorical_features = context._apply_selected_feature_subset(
                    predictor_features, categorical_features
                )
                anchor = max(len(context.df_history_for_lags) - 1, 0)
                x_forecast = featured.reindex(columns=predictor_features).iloc[anchor : anchor + 1]
            x_processed = context._transform_features(x_forecast, categorical_features)
            point_pred, quantile_preds = context._predict_point_and_quantiles(x_processed)
            block_pred = require_dirrec_output(point_pred, block_size)
            take = stop - start
            predictions.extend(block_pred[:take])
            if quantile_preds:
                for level, prediction in quantile_preds.items():
                    quantile_block = require_dirrec_output(prediction, block_size)
                    quantile_store.setdefault(level, []).extend(quantile_block[:take])

            for offset, value in enumerate(block_pred[:take]):
                step = start + offset
                new_row = future_remain.iloc[offset : offset + 1].copy()
                new_row[context.target_feature] = float(value)
                if use_feature_cache:
                    assert runtime_cache is not None
                    runtime_cache.append_endogenous(context.target_feature, float(value))
                for feature in other_endogenous:
                    if context.endogenous_future_provider is None:
                        raise ValueError(
                            "Multivariate DirRec forecast requires an endogenous future provider."
                        )
                    resolved_value = context.endogenous_future_provider.value_at(feature, step)
                    feature_value = resolved_value.value
                    backfill_sources.add(resolved_value.source)
                    new_row[feature] = feature_value
                    if use_feature_cache:
                        assert runtime_cache is not None
                        runtime_cache.append_endogenous(feature, feature_value)
                for column in context.df_history_for_lags.columns:
                    if column not in new_row.columns:
                        new_row[column] = context.df_history_for_lags[column].iloc[-1]
                context._append_history_row(new_row)
        context._finalize_recursive_quantiles(quantile_store)
        log_execution_summary(
            context,
            "dirrec",
            model_calls=block_calls * (len(quantile_store) if quantile_store else 1),
            backfill_source=",".join(sorted(backfill_sources)),
            block_size=block_size,
        )
        return np.asarray(predictions, dtype=float)
