# -*- coding: utf-8 -*-
import numpy as np

from models.multistep.contracts import require_pointwise_output
from models.multistep.runtime import context_horizon, log_execution_summary


class PointwiseExecutor:
    def execute(self, context):
        horizon = context_horizon(context)
        if bool(getattr(context.args, "align_direct_features_to_target", False)):
            from utils.multistep_contract import validate_direct_feature_alignment

            validate_direct_feature_alignment(context.args, horizon)
            df_pointwise = context._concat_history_and_future()
            df_date_future, df_weather_future = context._slice_future_aux_by_forecast(df_pointwise)
        else:
            df_pointwise = context.df_future
            df_date_future = context.df_date_future
            df_weather_future = context.df_weather_future

        (
            df_future_featured,
            predictor_features,
            _,
            categorical_features,
        ) = context.feature_engineer.create_features(
            df_series=df_pointwise,
            df_date_history=None,
            df_date_future=df_date_future,
            df_weather_history=None,
            df_weather_future=df_weather_future,
            df_custom_future=context.df_custom_future,
            endogenous_features_with_target=context.endogenous_features,
            target_feature=context.target_feature,
            horizon=horizon,
        )
        if bool(getattr(context.args, "align_direct_features_to_target", False)):
            df_future_featured = df_future_featured.iloc[-horizon:].copy()
        if not predictor_features:
            raise ValueError("pointwise predictor feature set is empty.")
        x_future = df_future_featured.reindex(columns=predictor_features)
        if context.selected_features:
            selected = [column for column in context.selected_features if column in x_future.columns]
            if selected:
                x_future = x_future[selected]
                categorical_features = [column for column in categorical_features if column in selected]
        x_processed = context._transform_features(x_future, categorical_features)
        point_pred, quantile_preds = context._predict_point_and_quantiles(x_processed)
        result = require_pointwise_output(point_pred, horizon)
        if quantile_preds:
            context._quantile_outputs = {
                level: require_pointwise_output(prediction, horizon)
                for level, prediction in quantile_preds.items()
            }
        log_execution_summary(
            context,
            "pointwise",
            model_calls=len(quantile_preds) if quantile_preds else 1,
        )
        return np.asarray(result, dtype=float)
