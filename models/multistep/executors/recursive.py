# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from models.multistep.contracts import require_endogenous_history, require_recursive_output
from models.multistep.runtime import (
    context_horizon,
    ensure_resolved_strategy,
    log_execution_summary,
)
from models.multistep.spec import InputScope


class RecursiveExecutor:
    def execute(self, context):
        if ensure_resolved_strategy(context).spec.input_scope == InputScope.ALL_ENDOGENOUS:
            return self._execute_multivariate(context)
        return self._execute_univariate(context)

    def _execute_univariate(self, context):
        horizon = context_horizon(context)
        predictions = []
        quantile_store = {}
        for step in range(horizon):
            future_row = context.df_future.iloc[step : step + 1].copy()
            forecast_frame = pd.concat(
                [context.df_history_for_lags, future_row], ignore_index=True, copy=False
            )
            date_future, weather_future = context._slice_future_aux_by_forecast(forecast_frame)
            (
                featured,
                predictor_features,
                target_output_features,
                categorical_features,
            ) = context.feature_engineer.create_features(
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
            schema = context._get_recursive_schema("usmr")
            if schema is None:
                context._set_recursive_schema(
                    "usmr", predictor_features, categorical_features, target_output_features
                )
            else:
                predictor_features = schema["predictor_features"]
                categorical_features = schema["categorical_features"]
            predictor_features, categorical_features = context._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            x_forecast = featured.reindex(columns=predictor_features).iloc[-1:]
            x_processed = context._transform_features(x_forecast, categorical_features)
            point_pred, quantile_preds = context._predict_point_and_quantiles(x_processed)
            value = require_recursive_output(point_pred)
            predictions.append(value)
            context._record_quantile_recursive_step(quantile_store, quantile_preds)
            new_row = future_row.iloc[-1:].copy()
            new_row[context.target_feature] = value
            context._append_history_row(new_row)
        context._finalize_recursive_quantiles(quantile_store)
        log_execution_summary(
            context,
            "recursive",
            model_calls=horizon * (len(quantile_store) if quantile_store else 1),
        )
        return np.asarray(predictions, dtype=float)

    def _execute_multivariate(self, context):
        require_endogenous_history(context.df_history_for_lags, context.endogenous_features)
        runtime_cache = context._prepare_msmr_runtime()
        horizon = context_horizon(context)
        predictions = []
        quantile_store = {}
        backfill_sources = set()
        for step in range(horizon):
            future_row = context.df_future.iloc[step : step + 1].copy()
            x_forecast = context._build_msmr_step_input(runtime_cache, step)
            x_processed = context._transform_features(
                x_forecast, runtime_cache.categorical_features
            )
            point_pred, quantile_preds = context._predict_point_and_quantiles(x_processed)
            value = require_recursive_output(point_pred)
            predictions.append(value)
            context._record_quantile_recursive_step(quantile_store, quantile_preds)

            new_row = future_row.iloc[-1:].copy()
            new_row[context.target_feature] = value
            runtime_cache.append_endogenous(context.target_feature, value)
            for feature in context.endogenous_features:
                if feature == context.target_feature:
                    continue
                if context.endogenous_future_provider is None:
                    raise ValueError(
                        "Multivariate recursive forecast requires an endogenous future provider."
                    )
                resolved_value = context.endogenous_future_provider.value_at(feature, step)
                feature_value = resolved_value.value
                backfill_sources.add(resolved_value.source)
                new_row[feature] = feature_value
                runtime_cache.append_endogenous(feature, feature_value)
            for column in context.df_history_for_lags.columns:
                if column not in new_row.columns:
                    if column in future_row.columns:
                        new_row[column] = future_row[column].iloc[-1]
                    else:
                        new_row[column] = context.df_history_for_lags[column].iloc[-1]
            context._append_history_row(new_row)
        context._finalize_recursive_quantiles(quantile_store)
        log_execution_summary(
            context,
            "recursive",
            model_calls=horizon * (len(quantile_store) if quantile_store else 1),
            backfill_source=",".join(sorted(backfill_sources)),
        )
        return np.asarray(predictions, dtype=float)
