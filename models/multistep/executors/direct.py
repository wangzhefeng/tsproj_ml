# -*- coding: utf-8 -*-
import numpy as np

from models.multistep.contracts import require_direct_output
from models.multistep.runtime import context_horizon, log_execution_summary


class DirectExecutor:
    def execute(self, context):
        x_forecast, categorical_features = context._build_direct_forecast_input(
            endogenous_features=context.endogenous_features
        )
        x_processed = context._transform_features(x_forecast, categorical_features)
        point_pred, quantile_preds = context._predict_point_and_quantiles(x_processed)
        horizon = context_horizon(context)
        result = require_direct_output(point_pred, horizon, label="point")
        if quantile_preds:
            context._quantile_outputs = {
                level: require_direct_output(prediction, horizon, label=f"quantile q={level}")
                for level, prediction in quantile_preds.items()
            }
        log_execution_summary(
            context,
            "direct",
            model_calls=len(quantile_preds) if quantile_preds else 1,
        )
        return np.asarray(result, dtype=float)
