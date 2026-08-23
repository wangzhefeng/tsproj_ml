# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelForecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer
from utils.eval_mask import build_eval_mask
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Forecaster:
    """
    预测辅助类-所有预测方法的公共逻辑
    """

    @staticmethod
    def _resolve_history_context_length(args: Dict) -> int:
        """返回预测阶段构造启用特征所需的最大固定回看步数。"""
        lengths = [max(getattr(args, "lags", []) or [1])]
        if not getattr(args, "enable_advanced_features", False):
            return max(lengths)

        fixed_lookbacks = (
            ("enable_rolling_features", "rolling_windows"),
            ("enable_diff_features", "diff_periods"),
            ("enable_pct_change_features", "pct_change_periods"),
        )
        for enabled_attr, periods_attr in fixed_lookbacks:
            if getattr(args, enabled_attr, False):
                lengths.extend(getattr(args, periods_attr, []) or [])
        return max(lengths)
    
    def __init__(self,
                 args: Dict,
                 horizon: int,
                 model: Any,
                 feature_scaler,
                 target_scaler,
                 df_history: pd.DataFrame,
                 df_future: pd.DataFrame,
                 df_date_future: pd.DataFrame,
                 df_weather_future: Optional[pd.DataFrame],
                 endogenous_features: List[str],
                 target_feature: str,
                 target_output_features: List[str],
                 categorical_features: List[str],
                 selected_features: List[str] = None,
                 target_decomposer=None,
                 df_custom_future=None,
                 log_prefix: str = "[Forecaster]"):
        self.args = args
        self.horizon = horizon
        self.model = model
        # 解包 auxiliary bundle（MSMR/MSMDR + endogenous_backfill_strategy=auxiliary）
        self.aux_forecaster = None
        self.aux_trajectories = None
        if isinstance(model, dict) and model.get("bundle_type") == "auxiliary_endogenous":
            self.aux_forecaster = model.get("aux")
            self.model = model.get("main")
        # 解包 blend bundle（USBR/MSBR = Direct+Recursive 融合）
        self.blend_direct_model = None
        self.blend_recursive_model = None
        self.blend_direct_pred = None
        self.blend_recursive_pred = None
        if isinstance(self.model, dict) and self.model.get("bundle_type") == "blend_direct_recursive":
            self.blend_direct_model = self.model.get("direct")
            self.blend_recursive_model = self.model.get("recursive")
            self.model = self.blend_direct_model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.df_history = df_history
        self.df_future = df_future
        self.df_date_future = df_date_future
        self.df_weather_future = df_weather_future
        self.df_custom_future = df_custom_future or []
        self.endogenous_features = endogenous_features
        self.target_feature = target_feature
        self.target_output_features = target_output_features
        self.categorical_features = categorical_features
        self.selected_features = selected_features
        self.target_decomposer = target_decomposer
        self.log_prefix = log_prefix
        logger.info(f"{self.log_prefix} Forecaster params init...")
        logger.info(f"{self.log_prefix} {'-' * 71}")
        # 最大滞后数量（仅用于 lag 特征和递归 lag state）
        self.max_lag = max(self.args.lags) if self.args.lags else 1
        self.history_context_length = self._resolve_history_context_length(self.args)
        logger.info(f"{self.log_prefix} Forecaster max_lag: {self.max_lag}")
        logger.info(
            f"{self.log_prefix} Forecaster history context length: "
            f"{self.history_context_length}"
        )
        
        # 保留构造启用 lag/rolling/diff 等特征所需的完整历史上下文。
        self.df_history_for_lags = self.df_history.iloc[-self.history_context_length:].copy()
        logger.info(f"{self.log_prefix} Forecaster df_history_for_lags shape: {self.df_history_for_lags.shape}")
        logger.info(f"{self.log_prefix} Forecaster df_history_for_lags columns: {self.df_history_for_lags.columns.tolist()}")
        # 复用特征工程器，避免递归预测中反复实例化
        self.feature_engineer = FeatureEngineer(self.args, self.log_prefix, verbose=False)
        # 递归预测 schema 缓存
        self._recursive_schema_cache = {}
        # 分位数输出缓存（仅 predict_type=quantile 时）
        self.quantile_outputs = None
        # 未来辅助特征索引（日期/天气），用于按步裁剪，减少重复 merge 开销
        self._prepare_future_aux_index()
        # MSMR 递归缓存
        self._msmr_runtime_cache = None
        # MSMDR 分块缓存
        self._msmdr_runtime_cache = None
        # 预先预测非目标内生变量的未来轨迹（auxiliary 策略，供 MSMR/MSMDR 回填）
        if self.aux_forecaster is not None:
            self.aux_trajectories = self.aux_forecaster.predict_horizon(
                self.df_history, self.df_future, self.horizon
            )
            logger.info(f"{self.log_prefix} Auxiliary trajectories predicted for {len(self.aux_trajectories)} endogenous cols.")

    def _should_log_step(self, step: int) -> bool:
        if not bool(getattr(self.args, "enable_step_logging", False)):
            return False
        interval = max(1, int(getattr(self.args, "forecast_log_interval", 1) or 1))
        return step == 0 or (step + 1) % interval == 0 or (step + 1) == self.horizon
    
    @staticmethod
    def _to_1d(pred: Any) -> np.ndarray:
        pred_arr = np.asarray(pred)
        if pred_arr.ndim == 0:
            return np.asarray([float(pred_arr)])
        return pred_arr.reshape(-1)

    @staticmethod
    def _to_scalar(pred: Any) -> float:
        pred_arr = Forecaster._to_1d(pred)
        return float(pred_arr[0]) if pred_arr.size > 0 else np.nan

    def _prepare_future_aux_index(self):
        """预处理未来日期/气象数据索引，便于按步快速切片。"""
        self._df_date_future_indexed = None
        self._df_weather_future_indexed = None

        if (
            self.df_date_future is not None
            and getattr(self.args, "date_ts_feat", None)
            and self.args.date_ts_feat in self.df_date_future.columns
        ):
            df_date = self.df_date_future.copy()
            df_date[self.args.date_ts_feat] = pd.to_datetime(df_date[self.args.date_ts_feat]).dt.normalize()
            df_date = df_date.drop_duplicates(subset=[self.args.date_ts_feat], keep="last")
            self._df_date_future_indexed = df_date.set_index(self.args.date_ts_feat).sort_index()

        if (
            self.df_weather_future is not None
            and getattr(self.args, "weather_ts_feat", None)
            and self.args.weather_ts_feat in self.df_weather_future.columns
        ):
            df_weather = self.df_weather_future.copy()
            df_weather[self.args.weather_ts_feat] = pd.to_datetime(df_weather[self.args.weather_ts_feat])
            df_weather = df_weather.drop_duplicates(subset=[self.args.weather_ts_feat], keep="last")
            self._df_weather_future_indexed = df_weather.set_index(self.args.weather_ts_feat).sort_index()

    def _append_history_row(self, row_df: pd.DataFrame):
        """
        将新预测步回填到固定长度历史窗口，避免在递归预测中反复 concat。
        """
        if row_df is None or row_df.empty:
            return

        row = row_df.iloc[-1:].copy()
        history_columns = self.df_history_for_lags.columns.tolist()
        if history_columns:
            row = row.reindex(columns=history_columns)
            if len(self.df_history_for_lags) > 0:
                row = row.fillna(self.df_history_for_lags.iloc[-1])

        self.df_history_for_lags.loc[len(self.df_history_for_lags)] = row.iloc[0]
        if len(self.df_history_for_lags) > self.history_context_length:
            self.df_history_for_lags = self.df_history_for_lags.iloc[-self.history_context_length:].reset_index(drop=True)

    def _slice_future_aux_by_forecast(self, df_forecast: pd.DataFrame):
        """根据当前预测窗口切出必要的日期/天气特征子集。"""
        df_date_slice = None
        df_weather_slice = None
        if "time" not in df_forecast.columns:
            return df_date_slice, df_weather_slice

        if self._df_date_future_indexed is not None:
            needed_dates = pd.to_datetime(df_forecast["time"]).dt.normalize().unique()
            date_idx = self._df_date_future_indexed.index.intersection(pd.DatetimeIndex(needed_dates))
            if len(date_idx) > 0:
                df_date_slice = self._df_date_future_indexed.loc[date_idx].reset_index()
                if self.args.date_ts_feat not in df_date_slice.columns and "index" in df_date_slice.columns:
                    df_date_slice = df_date_slice.rename(columns={"index": self.args.date_ts_feat})

        if self._df_weather_future_indexed is not None:
            needed_times = pd.to_datetime(df_forecast["time"]).unique()
            weather_idx = self._df_weather_future_indexed.index.intersection(pd.DatetimeIndex(needed_times))
            if len(weather_idx) > 0:
                df_weather_slice = self._df_weather_future_indexed.loc[weather_idx].reset_index()
                if self.args.weather_ts_feat not in df_weather_slice.columns and "index" in df_weather_slice.columns:
                    df_weather_slice = df_weather_slice.rename(columns={"index": self.args.weather_ts_feat})

        return df_date_slice, df_weather_slice

    def _get_recursive_schema(self, schema_key: str):
        return self._recursive_schema_cache.get(schema_key)

    def _set_recursive_schema(self, schema_key: str, predictor_features, categorical_features, target_output_features):
        self._recursive_schema_cache[schema_key] = {
            "predictor_features": predictor_features,
            "categorical_features": categorical_features,
            "target_output_features": target_output_features,
        }

    def _apply_selected_feature_subset(self, predictor_features: List[str], categorical_features: List[str]):
        """
        训练阶段做了特征选择时，推理阶段使用同一子集，避免特征空间不一致。
        """
        if not self.selected_features:
            return predictor_features, categorical_features

        selected_set = set(self.selected_features)
        selected_predictor_features = [f for f in predictor_features if f in selected_set]
        if not selected_predictor_features:
            logger.warning(
                f"{self.log_prefix} selected_features does not overlap current predictor_features, fallback to all predictors."
            )
            selected_predictor_features = predictor_features
        selected_categorical_features = [f for f in categorical_features if f in selected_predictor_features]
        return selected_predictor_features, selected_categorical_features

    def _transform_features(self, X: pd.DataFrame, categorical_features: List[str]):
        """
        baseline 训练路径没有 feature scaler；此时预测阶段直接使用原始特征。

        无 scaler 时仍需保证列 schema 与训练一致：未来帧外生列可能缺失
        （如 weather_future 未覆盖预测期），LightGBM 会因特征数不一致直接报错。
        从模型的 feature_name_ / 训练列恢复 schema，缺列用 NaN→dropna 前中位数兜底
        与 _align_feature_schema 同语义（保持列序一致）。
        """
        if self.feature_scaler is None:
            X_out = X.copy()
            # 恢复训练期列 schema（LightGBM booster 持有 feature_name_；
            # 其他模型按传入列原样）
            training_columns = None
            model = getattr(self, "model", None)
            if isinstance(model, dict):
                inner = model.get("models")
                if isinstance(inner, dict) and inner:
                    first = next(iter(inner.values()))
                    model = first
            booster = getattr(model, "booster_", None)
            if booster is not None:
                try:
                    training_columns = list(booster.feature_name())
                except Exception:
                    training_columns = None
            inner_est = getattr(model, "estimators_", None)
            if training_columns is None and inner_est:
                leaf = inner_est[0]
                while hasattr(leaf, "estimator"):
                    leaf = leaf.estimator
                leaf_booster = getattr(leaf, "booster_", None)
                if leaf_booster is not None:
                    try:
                        training_columns = list(leaf_booster.feature_name())
                    except Exception:
                        training_columns = None
            if training_columns:
                missing_cols = [c for c in training_columns if c not in X_out.columns]
                if missing_cols:
                    logger.warning(
                        f"{self.log_prefix} Missing columns at inference (no-scaler baseline): {missing_cols}"
                    )
                    # 缺列兜底：与 _align_feature_schema 一致用 0.0
                    for col in missing_cols:
                        X_out[col] = 0.0
                extra_cols = [c for c in X_out.columns if c not in training_columns]
                if extra_cols:
                    X_out = X_out.drop(columns=extra_cols)
                X_out = X_out[training_columns]
            return X_out
        return self.feature_scaler.transform(X, categorical_features)

    @staticmethod
    def _resolve_block_size(args) -> int:
        explicit_block_size = int(getattr(args, "block_size", 0) or 0)
        if explicit_block_size > 0:
            return explicit_block_size
        lags = list(getattr(args, "lags", []) or [])
        return min(lags) if lags else 1

    def _prepare_msmr_runtime(self):
        """
        为 MSMR 递归预测准备静态外生特征、schema 与滞后状态。
        """
        if self._msmr_runtime_cache is not None:
            return self._msmr_runtime_cache

        df_future_exog, exogenous_features, exogenous_categorical = self.feature_engineer.create_exogenouse_features(
            df=self.df_future.copy(),
            df_date_history=None,
            df_date_future=self.df_date_future,
            df_weather_history=None,
            df_weather_future=self.df_weather_future,
            df_custom_future=self.df_custom_future,
        )
        if "time" in df_future_exog.columns:
            df_future_exog = df_future_exog.set_index("time", drop=False)

        schema_key = "msmr"
        schema = self._get_recursive_schema(schema_key)
        if schema is None:
            df_forecast_seed = pd.concat(
                [self.df_history_for_lags, self.df_future.iloc[:1].copy()],
                ignore_index=True,
                copy=False,
            )
            df_date_future_step, df_weather_future_step = self._slice_future_aux_by_forecast(df_forecast_seed)
            (_, predictor_features, target_output_features, categorical_features) = self.feature_engineer.create_features(
                df_series=df_forecast_seed,
                df_date_history=None,
                df_date_future=df_date_future_step,
                df_weather_history=None,
                df_weather_future=df_weather_future_step,
                df_custom_future=self.df_custom_future,
                endogenous_features_with_target=self.endogenous_features,
                target_feature=self.target_feature,
                horizon=self.horizon,
            )
            self._set_recursive_schema(schema_key, predictor_features, categorical_features, target_output_features)
            schema = self._get_recursive_schema(schema_key)

        predictor_features = schema["predictor_features"]
        categorical_features = schema["categorical_features"]
        target_output_features = schema["target_output_features"]

        if exogenous_features:
            predictor_features = exogenous_features + [
                feature for feature in predictor_features if feature not in exogenous_features
            ]
            categorical_features = sorted(
                set(categorical_features + exogenous_categorical),
                key=(categorical_features + exogenous_categorical).index,
            )

        predictor_features, categorical_features = self._apply_selected_feature_subset(
            predictor_features,
            categorical_features,
        )
        lags = list(getattr(self.args, "lags", []) or [])
        lag_feature_names = [
            f"{col}_lag_{lag}"
            for col in self.endogenous_features
            for lag in lags
            if f"{col}_lag_{lag}" in predictor_features
        ]

        lag_state = {}
        for col in self.endogenous_features:
            if col in self.df_history_for_lags.columns:
                values = self.df_history_for_lags[col].tolist()
            elif col in self.df_history.columns:
                values = self.df_history[col].iloc[-self.max_lag:].tolist()
            else:
                values = []
            values = [0.0 if pd.isna(v) else v for v in values]
            if not values:
                values = [0.0]
            if len(values) < self.max_lag:
                values = [values[0]] * (self.max_lag - len(values)) + values
            lag_state[col] = deque(values[-self.max_lag:], maxlen=self.max_lag)

        self._msmr_runtime_cache = {
            "df_future_exog": df_future_exog,
            "exogenous_features": exogenous_features,
            "predictor_features": predictor_features,
            "categorical_features": categorical_features,
            "target_output_features": target_output_features,
            "lag_feature_names": lag_feature_names,
            "lag_state": lag_state,
            "lags": lags,
        }
        return self._msmr_runtime_cache

    @staticmethod
    def _read_lag_value(buffer: deque, lag: int) -> float:
        values = list(buffer)
        if not values:
            return 0.0
        if lag <= len(values):
            return values[-lag]
        return values[0]

    def _build_msmr_step_input(self, runtime_cache: Dict[str, Any], step: int) -> pd.DataFrame:
        row_data = {}
        df_future_exog = runtime_cache["df_future_exog"]
        if not df_future_exog.empty:
            future_row = df_future_exog.iloc[step]
            for feature in runtime_cache["exogenous_features"]:
                if feature in future_row.index:
                    row_data[feature] = future_row[feature]

        for col in self.endogenous_features:
            buffer = runtime_cache["lag_state"][col]
            for lag in runtime_cache["lags"]:
                lag_feature = f"{col}_lag_{lag}"
                if lag_feature in runtime_cache["lag_feature_names"]:
                    row_data[lag_feature] = self._read_lag_value(buffer, lag)

        X_forecast_input = pd.DataFrame([row_data])
        return X_forecast_input.reindex(columns=runtime_cache["predictor_features"])

    def _prepare_msmdr_runtime(self):
        """
        为 MSMDR 分块预测准备静态外生特征、schema 与滞后状态。
        """
        if self._msmdr_runtime_cache is not None:
            return self._msmdr_runtime_cache

        df_future_exog, exogenous_features, exogenous_categorical = self.feature_engineer.create_exogenouse_features(
            df=self.df_future.copy(),
            df_date_history=None,
            df_date_future=self.df_date_future,
            df_weather_history=None,
            df_weather_future=self.df_weather_future,
            df_custom_future=self.df_custom_future,
        )
        if "time" in df_future_exog.columns:
            df_future_exog = df_future_exog.set_index("time", drop=False)

        schema_key = "msmdr"
        schema = self._get_recursive_schema(schema_key)
        if schema is None:
            df_forecast_seed = pd.concat(
                [self.df_history_for_lags, self.df_future.iloc[:1].copy()],
                ignore_index=True,
                copy=False,
            )
            df_date_future_step, df_weather_future_step = self._slice_future_aux_by_forecast(df_forecast_seed)
            (_, predictor_features, target_output_features, categorical_features) = self.feature_engineer.create_features(
                df_series=df_forecast_seed,
                df_date_history=None,
                df_date_future=df_date_future_step,
                df_weather_history=None,
                df_weather_future=df_weather_future_step,
                df_custom_future=self.df_custom_future,
                endogenous_features_with_target=self.endogenous_features,
                target_feature=self.target_feature,
                horizon=self.horizon,
            )
            self._set_recursive_schema(schema_key, predictor_features, categorical_features, target_output_features)
            schema = self._get_recursive_schema(schema_key)

        predictor_features = schema["predictor_features"]
        categorical_features = schema["categorical_features"]
        target_output_features = schema["target_output_features"]

        if exogenous_features:
            predictor_features = exogenous_features + [
                feature for feature in predictor_features if feature not in exogenous_features
            ]
            categorical_features = sorted(
                set(categorical_features + exogenous_categorical),
                key=(categorical_features + exogenous_categorical).index,
            )

        predictor_features, categorical_features = self._apply_selected_feature_subset(
            predictor_features,
            categorical_features,
        )
        lags = list(getattr(self.args, "lags", []) or [])
        lag_feature_names = [
            f"{col}_lag_{lag}"
            for col in self.endogenous_features
            for lag in lags
            if f"{col}_lag_{lag}" in predictor_features
        ]

        lag_state = {}
        for col in self.endogenous_features:
            if col in self.df_history_for_lags.columns:
                values = self.df_history_for_lags[col].tolist()
            elif col in self.df_history.columns:
                values = self.df_history[col].iloc[-self.max_lag:].tolist()
            else:
                values = []
            values = [0.0 if pd.isna(v) else v for v in values]
            if not values:
                values = [0.0]
            if len(values) < self.max_lag:
                values = [values[0]] * (self.max_lag - len(values)) + values
            lag_state[col] = deque(values[-self.max_lag:], maxlen=self.max_lag)

        self._msmdr_runtime_cache = {
            "df_future_exog": df_future_exog,
            "exogenous_features": exogenous_features,
            "predictor_features": predictor_features,
            "categorical_features": categorical_features,
            "target_output_features": target_output_features,
            "lag_feature_names": lag_feature_names,
            "lag_state": lag_state,
            "lags": lags,
        }
        return self._msmdr_runtime_cache

    def _is_quantile_bundle(self) -> bool:
        return isinstance(self.model, dict) and self.model.get("predict_type") == "quantile" and "models" in self.model

    def _predict_point_and_quantiles(self, X_processed: pd.DataFrame) -> Tuple[np.ndarray, Optional[Dict[float, np.ndarray]]]:
        """
        统一预测入口：
        - 点预测模型: 返回(point_pred, None)
        - 分位数模型: 返回(中位分位点预测, {q: pred_q})
        """
        if not self._is_quantile_bundle():
            return np.asarray(self.model.predict(X_processed)), None

        quantile_models = self.model.get("models", {})
        quantiles = [float(q) for q in self.model.get("quantiles", list(quantile_models.keys()))]
        quantile_preds = {}
        for q in quantiles:
            q_key = q if q in quantile_models else str(q)
            model_q = quantile_models.get(q_key)
            if model_q is None and q in quantile_models:
                model_q = quantile_models[q]
            if model_q is None:
                continue
            quantile_preds[float(q)] = np.asarray(model_q.predict(X_processed))
        if not quantile_preds:
            raise ValueError(f"{self.log_prefix} quantile model bundle has no valid sub-models.")

        median_q = float(self.model.get("median_quantile", min(quantile_preds.keys(), key=lambda x: abs(x - 0.5))))
        point_pred = quantile_preds.get(median_q)
        if point_pred is None:
            median_q = min(quantile_preds.keys(), key=lambda x: abs(x - 0.5))
            point_pred = quantile_preds[median_q]
        return point_pred, quantile_preds

    def _record_quantile_direct(self, quantile_preds: Optional[Dict[float, np.ndarray]], n_required: int):
        if not quantile_preds:
            return
        self.quantile_outputs = {}
        for q, pred in quantile_preds.items():
            pred_1d = self._to_1d(pred[0] if np.asarray(pred).ndim > 1 else pred)
            if len(pred_1d) >= n_required:
                self.quantile_outputs[q] = pred_1d[:n_required]
            else:
                self.quantile_outputs[q] = np.pad(pred_1d, (0, n_required - len(pred_1d)), mode="edge")

    def _record_quantile_recursive_step(self, store: Dict[float, List[float]], quantile_preds: Optional[Dict[float, np.ndarray]]):
        if not quantile_preds:
            return
        for q, pred in quantile_preds.items():
            store.setdefault(q, []).append(self._to_scalar(pred))

    def _finalize_recursive_quantiles(self, store: Dict[float, List[float]]):
        if not store:
            return
        self.quantile_outputs = {q: np.asarray(v, dtype=float) for q, v in store.items()}

    def _is_horizon_feature_mode(self) -> bool:
        """USMD/MSMD 且 direct_strategy=horizon_feature 时推理走多行展开。"""
        if str(getattr(self.args, "pred_method", "")).lower() not in (
            "univariate-single-multistep-direct",
            "multivariate-single-multistep-direct",
        ):
            return False
        return str(getattr(self.args, "direct_strategy", "multioutput")).lower() == "horizon_feature"

    def _resolve_forecast_horizon_period(self) -> int:
        """horizon sin/cos 编码周期：子日频用 n_per_day（日周期），日频用 7（周周期）。"""
        n_per_day = int(getattr(self.args, "n_per_day", 1) or 1)
        if n_per_day > 1:
            return n_per_day
        return 7

    _DERIVED_SUFFIX_MARKERS = ("lag_", "rolling_", "diff_")

    def _classify_endogenous_derived(self, predictor_features: List[str], endogenous_features: List[str]) -> set:
        """
        识别由内生变量派生的列（滞后/滚动/差分/三角编码等）。
        horizon_feature 模式下这些列用 anchor 值（MIMO 约束），外生列保留 future 按步值（horizon-aware）。
        用已知派生模式白名单匹配，避免误伤以 {base}_ 开头的外生列（如 power_forecast）。
        """
        endo_set = set(endogenous_features)
        derived = set()
        for col in predictor_features:
            if col in endo_set:
                derived.add(col)
                continue
            for base in endo_set:
                if not col.startswith(base + "_"):
                    continue
                tail = col[len(base) + 1:]
                if (
                    any(tail.startswith(m) for m in self._DERIVED_SUFFIX_MARKERS)
                    or tail in ("sin", "cos")
                ):
                    derived.add(col)
                    break
        return derived

    def _append_horizon_features(self, X_input: pd.DataFrame) -> pd.DataFrame:
        """向推理输入追加 horizon 索引特征（+ 可选 sin/cos）。"""
        h_name = str(getattr(self.args, "horizon_feature_name", "forecast_horizon_idx"))
        h_vals = np.arange(1, len(X_input) + 1, dtype=float)
        X_input = X_input.copy()
        X_input[h_name] = h_vals
        if bool(getattr(self.args, "enable_horizon_cyclical", True)):
            period = float(self._resolve_forecast_horizon_period())
            X_input[f"{h_name}_sin"] = np.sin(2 * np.pi * h_vals / period)
            X_input[f"{h_name}_cos"] = np.cos(2 * np.pi * h_vals / period)
        return X_input

    def _build_direct_forecast_input(self, endogenous_features: List[str]):
        """
        构建 Direct 策略输入：
        - 历史 max_lag 行 + 全部未来外生行
        - 取锚点为最后一个历史行，使 horizon-aware 外生展开可用
        """
        for endo_feat in endogenous_features:
            if endo_feat not in self.df_history_for_lags.columns and endo_feat in self.df_history.columns:
                self.df_history_for_lags[endo_feat] = self.df_history[endo_feat].iloc[-self.history_context_length:]

        df_forecast = pd.concat([self.df_history_for_lags, self.df_future.copy()], ignore_index=True, copy=False)
        df_date_future_slice, df_weather_future_slice = self._slice_future_aux_by_forecast(df_forecast)
        (df_forecast_featured,
         predictor_features,
         target_output_features,
         categorical_features) = self.feature_engineer.create_features(
            df_series=df_forecast,
            df_date_history=None,
            df_date_future=df_date_future_slice,
            df_weather_history=None,
            df_weather_future=df_weather_future_slice,
            df_custom_future=self.df_custom_future,
            endogenous_features_with_target=endogenous_features,
            target_feature=self.target_feature,
            horizon=self.horizon,
        )
        predictor_features, categorical_features = self._apply_selected_feature_subset(
            predictor_features, categorical_features
        )
        anchor_idx = max(len(self.df_history_for_lags) - 1, 0)
        if self._is_horizon_feature_mode():
            # horizon_feature 模式：展开 H 行，外生列按步取值（horizon-aware），
            # 内生派生列用 anchor 值覆盖（MIMO 约束），追加 horizon 索引特征
            H = min(self.horizon, len(self.df_future))
            X_input = df_forecast_featured.reindex(columns=predictor_features).iloc[anchor_idx:anchor_idx + H].copy()
            anchor_row = df_forecast_featured.reindex(columns=predictor_features).iloc[[anchor_idx]]
            endo_derived = self._classify_endogenous_derived(predictor_features, endogenous_features)
            for col in endo_derived:
                if col in X_input.columns and col in anchor_row.columns:
                    X_input[col] = anchor_row[col].values[0]
            X_forecast_input = self._append_horizon_features(X_input)
            logger.info(
                f"{self.log_prefix} horizon_feature forecast input: {X_forecast_input.shape[0]} rows, "
                f"endo_derived cols masked={len([c for c in endo_derived if c in X_input.columns])}"
            )
        else:
            X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[anchor_idx:anchor_idx + 1]
        return X_forecast_input, categorical_features
    # ------------------------------
    # 单变量（目标变量滞后特征）预测单变量（目标变量）
    # ------------------------------
    def univariate_single_multi_step_direct_pointwise_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步逐点 direct 预测(USMDP)
        """
        # 多步预测值收集器
        Y_preds = np.array([])
        # 预测阶段始终使用未来日期/天气进行特征工程，避免被 is_testing 分支误跳过
        if bool(getattr(self.args, "align_direct_features_to_target", False)):
            if self.horizon != 1:
                raise ValueError(
                    "USMDP align_direct_features_to_target currently supports horizon=1 only."
                )
            df_pointwise = pd.concat(
                [self.df_history_for_lags, self.df_future.copy()],
                ignore_index=True,
                copy=False,
            )
            df_date_future, df_weather_future = self._slice_future_aux_by_forecast(df_pointwise)
        else:
            df_pointwise = self.df_future
            df_date_future = self.df_date_future
            df_weather_future = self.df_weather_future
        (df_future_featured,
         predictor_features,
         target_output_features,
         categorical_features) = self.feature_engineer.create_features(
            df_series=df_pointwise,
            df_date_history=None,
            df_date_future=df_date_future,
            df_weather_history=None,
            df_weather_future=df_weather_future,
            df_custom_future=self.df_custom_future,
            endogenous_features_with_target=self.endogenous_features,
            target_feature=self.target_feature,
            horizon=self.horizon,
        )
        if bool(getattr(self.args, "align_direct_features_to_target", False)):
            df_future_featured = df_future_featured.iloc[-len(self.df_future):].copy()
        if predictor_features:
            X_test_future = df_future_featured[predictor_features].copy()
        else:
            logger.warning(f"{self.log_prefix} predictor_features is empty in USMDP pointwise forecast; fallback to raw future frame.")
            X_test_future = self.df_future.copy()
            categorical_features = self.categorical_features
        logger.info(f"{self.log_prefix} after feature engineering df_future_featured shape: {df_future_featured.shape}")
        if self.selected_features:
            selected_cols = [c for c in self.selected_features if c in X_test_future.columns]
            if selected_cols:
                X_test_future = X_test_future[selected_cols]
                categorical_features = [c for c in categorical_features if c in selected_cols]
        logger.info(f"{self.log_prefix} after feature engineering X_test_future: \n{X_test_future.head()}")
        logger.info(f"{self.log_prefix} after feature engineering X_test_future shape: {X_test_future.shape}")
        logger.info(f"{self.log_prefix} after feature engineering categorical_features: {categorical_features}")
        # 特征预处理
        X_test_processed = self._transform_features(X_test_future, categorical_features)
        # 模型推理
        if len(X_test_processed) > 0:
            point_pred, quantile_preds = self._predict_point_and_quantiles(X_test_processed)
            Y_preds = np.asarray(point_pred)
            if quantile_preds:
                # 直接输出方法通常每行对应一步预测，按行记录分位数
                self.quantile_outputs = {
                    q: np.asarray(pred).reshape(-1) for q, pred in quantile_preds.items()
                }
        if Y_preds.size == 0:
            return np.array([])
        if Y_preds.ndim == 2 and Y_preds.shape[1] == 1:
            return Y_preds[:, 0]
        if Y_preds.ndim == 2 and Y_preds.shape[0] == 1:
            return Y_preds[0]

        return Y_preds

    def univariate_single_multi_step_direct_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接预测(USMD)
        """
        X_forecast_input, categorical_features = self._build_direct_forecast_input(
            endogenous_features=self.endogenous_features
        )
        X_test_processed = self._transform_features(X_forecast_input, categorical_features)
        point_pred, quantile_preds = self._predict_point_and_quantiles(X_test_processed)
        y_pred_multi_step = self._to_1d(point_pred[0] if np.asarray(point_pred).ndim > 1 else point_pred)
        if len(y_pred_multi_step) >= len(self.df_future):
            y_preds = y_pred_multi_step[:len(self.df_future)]
        else:
            y_preds = np.pad(y_pred_multi_step, pad_width=(0, len(self.df_future) - len(y_pred_multi_step)), mode='edge')
        self._record_quantile_direct(quantile_preds, n_required=len(self.df_future))
        return np.asarray(y_preds)

    def univariate_single_multi_step_recursive_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步递归预测(USMR)
        """
        # 多步预测值收集器
        Y_preds = []
        quantile_store = {}
        for step in range(self.horizon):
            if self._should_log_step(step):
                logger.info(f"{self.log_prefix} recursive forecast step: {step}...")
                logger.info(f"{self.log_prefix} {'=' * 31}")
            # 0.Prepare current features for prediction
            if step >= len(self.df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            # 1.构建预测特征数据
            df_future_step = self.df_future.iloc[step:step+1].copy()
            # 2.合并历史数据和当前步数据
            df_forecast = pd.concat([self.df_history_for_lags, df_future_step], ignore_index=True, copy=False)
            # 3.特征工程（按步裁剪辅助特征，避免每步处理完整未来表）
            df_date_future_step, df_weather_future_step = self._slice_future_aux_by_forecast(df_forecast)
            (df_forecast_featured,
             predictor_features,
             target_output_features,
             categorical_features) = self.feature_engineer.create_features(
                df_series = df_forecast,
                df_date_history = None,
                df_date_future = df_date_future_step,
                df_weather_history = None,
                df_weather_future = df_weather_future_step,
                df_custom_future = self.df_custom_future,
                endogenous_features_with_target = self.endogenous_features,
                target_feature = self.target_feature,
                horizon = self.horizon,
            )
            schema_key = "usmr"
            schema = self._get_recursive_schema(schema_key)
            if schema is None:
                self._set_recursive_schema(schema_key, predictor_features, categorical_features, target_output_features)
            else:
                predictor_features = schema["predictor_features"]
                categorical_features = schema["categorical_features"]
            predictor_features, categorical_features = self._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            # 4.提取出当前预测步所需要的特征（最后一行）
            X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[-1:]
            # 5.特征预处理（预测模式）
            X_forecast_processed = self._transform_features(X_forecast_input, categorical_features)
            # self.feature_scaler.validate_features(X_forecast_processed, stage="prediction")
            # 6.模型预测
            point_pred, quantile_preds = self._predict_point_and_quantiles(X_forecast_processed)
            y_pred_step = self._to_scalar(point_pred)
            Y_preds.append(y_pred_step)
            self._record_quantile_recursive_step(quantile_store, quantile_preds)
            # 7.将预测值更新回 df_future_step，以便为下一步预测提供滞后特征
            df_future_step_new_row = df_future_step.copy().iloc[-1:]
            df_future_step_new_row[self.target_feature] = y_pred_step
            # 8.将新行添加到历史数据中，进行下一次循环
            self._append_history_row(df_future_step_new_row)

        self._finalize_recursive_quantiles(quantile_store)
        return np.array(Y_preds)

    def univariate_single_multi_step_direct_recursive_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接+递归预测(USMDR)
        
        - 核心思想：
            1. 将预测 horizon 分成多个块（block_size = min(lags)）
            2. 在每个块内进行递归预测
            3. 块与块之间也是递归的（使用前一块的预测值）
        - 与其他方法的区别
            - 与 USMD 的区别：
                - USMD: 完全直接，为每步训练独立模型
                - USMDR: 只训练一个模型，但采用分块策略
            - 与 USMR 的区别：
                - USMR: 完全递归，每步都用上一步的预测
                - USMDR: 分块递归，块内递归，减少误差累积
        - 特征构成：
            - 内生变量：目标变量的滞后特征
            - 外生变量：日期时间特征+节假日类型特征+气象特征
        
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        # 严格分块直接：每个块仅调用一次模型，取块长输出
        block_size = self._resolve_block_size(self.args)
        logger.info(f"{self.log_prefix} block_size: {block_size}")
        y_preds = []
        quantile_store = {}
        while len(y_preds) < len(self.df_future):
            produced = len(y_preds)
            remain = len(self.df_future) - produced
            df_future_remain = self.df_future.iloc[produced:].copy()
            df_forecast = pd.concat([self.df_history_for_lags, df_future_remain], ignore_index=True, copy=False)
            df_date_future_slice, df_weather_future_slice = self._slice_future_aux_by_forecast(df_forecast)
            (df_forecast_featured,
             predictor_features,
             target_output_features,
             categorical_features) = self.feature_engineer.create_features(
                df_series=df_forecast,
                df_date_history=None,
                df_date_future=df_date_future_slice,
                df_weather_history=None,
                df_weather_future=df_weather_future_slice,
                df_custom_future=self.df_custom_future,
                endogenous_features_with_target=self.endogenous_features,
                target_feature=self.target_feature,
                horizon=self.horizon,
            )
            predictor_features, categorical_features = self._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            anchor_idx = max(len(self.df_history_for_lags) - 1, 0)
            X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[anchor_idx:anchor_idx + 1]
            X_forecast_processed = self._transform_features(X_forecast_input, categorical_features)
            point_pred, quantile_preds = self._predict_point_and_quantiles(X_forecast_processed)
            pred_vec = self._to_1d(point_pred[0] if np.asarray(point_pred).ndim > 1 else point_pred)
            take = min(block_size, remain, len(pred_vec))
            block_pred = pred_vec[:take]
            y_preds.extend(block_pred.tolist())
            if quantile_preds:
                for q, pred in quantile_preds.items():
                    q_vec = self._to_1d(pred[0] if np.asarray(pred).ndim > 1 else pred)
                    quantile_store.setdefault(q, []).extend(q_vec[:take].tolist())
            # 逐点更新历史（块内直接，不再逐点重复调模型）
            for i in range(take):
                df_new = df_future_remain.iloc[i:i+1].copy()
                df_new[self.target_feature] = float(block_pred[i])
                self._append_history_row(df_new)
        self._finalize_recursive_quantiles(quantile_store)
        return np.asarray(y_preds[:len(self.df_future)])
    # ------------------------------
    # 多变量（除目标变量外的内生变量）预测单变量（目标变量）
    # ------------------------------
    def multivariate_single_multi_step_direct_forecast(self):
        """
        多变量(内生变量)预测单变量(目标变量)多步直接预测(MSMD)
        - 方法特点：
            1. 特征：所有内生变量(target + 其他内生变量)的滞后 + 外生变量
            2. 训练：为每个未来步 H 创建目标列 target_shift_0, target_shift_1, ..., target_shift_H-1
            3. 预测：一次性输出所有 H 步的预测值
        - 与 USMD 的区别：
            - USMD: 只使用目标变量的滞后特征
            - MSMD: 使用所有内生变量的滞后特征（更多信息）
         
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        X_forecast_input, categorical_features = self._build_direct_forecast_input(
            endogenous_features=self.endogenous_features
        )
        X_test_processed = self.feature_scaler.transform(X_forecast_input, categorical_features)
        point_pred, quantile_preds = self._predict_point_and_quantiles(X_test_processed)
        y_pred_multi_step = self._to_1d(point_pred[0] if np.asarray(point_pred).ndim > 1 else point_pred)
        if len(y_pred_multi_step) >= len(self.df_future):
            y_preds = y_pred_multi_step[:len(self.df_future)]
        else:
            y_preds = np.pad(y_pred_multi_step, (0, len(self.df_future) - len(y_pred_multi_step)), 'edge')
        self._record_quantile_direct(quantile_preds, n_required=len(self.df_future))
        return np.asarray(y_preds)

    def multivariate_single_multi_step_recursive_forecast(self):
        """
        多变量(内生变量)预测单变量(目标变量)多步递归预测(MSMR)
        - 方法特点：
            1. 特征：所有内生变量(target + 其他内生变量)的滞后 + 外生变量
            2. 训练：训练单个 1 步预测模型，输入为所有内生变量(目标 + 其他内生变量)的滞后 + 外生变量，目标为下一时点的目标值
            3. 预测：逐步递归，每步用缓存的外生特征 + 当前滞后状态预测目标值，并把预测值回填为下一步的滞后输入
        - 与 USMD 的区别：
            - USMR: 只使用目标变量的滞后特征
            - MSMR: 使用所有内生变量的滞后特征（更多信息）
        
        Returns:
            目标变量的预测结果数组，形状为 (horizon,)
        """
        runtime_cache = self._prepare_msmr_runtime()
        # 多步预测值收集器
        Y_preds = []
        quantile_store = {}
        
        # Iterate for each step in the forecast horizon
        for step in range(self.horizon):
            if self._should_log_step(step):
                logger.info(f"{self.log_prefix} multivariate-recursive forecast step: {step}...")
            # 0.Prepare current features for prediction
            if step >= len(self.df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            
            df_future_exogenous = self.df_future.iloc[step:step+1].copy()
            # 1. 直接使用缓存的外生特征 + lag 状态组装当前步输入
            X_forecast_input = self._build_msmr_step_input(runtime_cache, step)

            # 2.特征预处理（预测模式）
            X_forecast_processed = self.feature_scaler.transform(
                X_forecast_input,
                runtime_cache["categorical_features"],
            )
            # self.feature_scaler.validate_features(X_forecast_processed, stage="prediction")

            # 3.模型预测（MSMR 当前训练目标为 target 的一步，因此取标量）
            point_pred, quantile_preds = self._predict_point_and_quantiles(X_forecast_processed)
            y_pred_target = self._to_scalar(point_pred)
            Y_preds.append(y_pred_target)
            self._record_quantile_recursive_step(quantile_store, quantile_preds)

            # 4.将预测值更新回缓存与历史，以便为下一步预测提供滞后特征
            df_future_exogenous_new_row = df_future_exogenous.copy().iloc[-1:]
            df_future_exogenous_new_row[self.target_feature] = y_pred_target
            runtime_cache["lag_state"][self.target_feature].append(y_pred_target)
            # 5. 其他内生变量回填：优先用 aux 轨迹，回退持久性
            for feat in self.endogenous_features:
                if feat == self.target_feature:
                    continue
                if (
                    self.aux_trajectories is not None
                    and feat in self.aux_trajectories
                    and step < len(self.aux_trajectories[feat])
                    and np.isfinite(self.aux_trajectories[feat][step])
                ):
                    val = float(self.aux_trajectories[feat][step])
                elif feat in self.df_history_for_lags.columns:
                    val = float(self.df_history_for_lags[feat].iloc[-1])
                else:
                    val = 0.0
                df_future_exogenous_new_row[feat] = val
                runtime_cache["lag_state"][feat].append(val)

            # 6.补齐其余列
            for col in self.df_history_for_lags.columns:
                if col not in df_future_exogenous_new_row.columns:
                    # If it's an exogenous feature in current_step_df, prefer that
                    if col in df_future_exogenous.columns:
                        df_future_exogenous_new_row[col] = df_future_exogenous[col].iloc[-1]
                    else: # Otherwise, take from the last known data point
                        df_future_exogenous_new_row[col] = self.df_history_for_lags[col].iloc[-1]

            # 7.将新行添加到历史数据中，进行下一次循环
            self._append_history_row(df_future_exogenous_new_row)

        self._finalize_recursive_quantiles(quantile_store)
        return np.array(Y_preds)

    def multivariate_single_multi_step_direct_recursive_forecast(self):
        """
        多变量(内生变量)预测单变量(目标变量)多步直接+递归预测(MSMDR)
        
        - 核心思想：
            1. 使用所有内生变量的滞后特征（不只是目标变量）
            2. 分块递归预测目标变量
            3. 对于其他内生变量，使用持久性预测或简单方法估计
        - 与其他方法的区别：
            - 与 USMDR 的核心区别：
                - USMDR: 只用目标变量的滞后 → 特征少
                - MSMDR: 用所有内生变量的滞后 → 特征多，信息丰富
            - 与 MSMR 的区别：
                - MSMR: 递归预测所有内生变量
                - MSMDR: 只递归预测目标变量，其他内生变量用简化方法
        - 特征构成示例：
            假设 endogenous_features = ['load', 'temperature', 'humidity']
                target_feature = 'load'
                lags = [1, 2, 7]
            
            特征 = [load_lag_1, load_lag_2, load_lag_7,           # 目标变量的滞后
                temperature_lag_1, temperature_lag_2, temperature_lag_7,  # 其他内生变量的滞后
                humidity_lag_1, humidity_lag_2, humidity_lag_7,
                hour, day_of_week, ...]  # 外生变量
        
        Returns:
            目标变量的预测结果数组，形状为 (horizon,)
        """
        # 严格分块直接：每个块仅调用一次模型，取块长输出
        block_size = self._resolve_block_size(self.args)
        logger.info(f"{self.log_prefix} block_size: {block_size}")
        y_preds = []
        quantile_store = {}
        use_feature_cache = bool(getattr(self.args, "enable_feature_cache", False))
        runtime_cache = self._prepare_msmdr_runtime() if use_feature_cache else None

        # 确保所有内生变量都在历史数据中
        for endo_feat in self.endogenous_features:
            if endo_feat not in self.df_history_for_lags.columns and endo_feat in self.df_history.columns:
                self.df_history_for_lags[endo_feat] = self.df_history[endo_feat].iloc[-self.history_context_length:]

        other_endogenous = [feat for feat in self.endogenous_features if feat != self.target_feature]

        while len(y_preds) < len(self.df_future):
            produced = len(y_preds)
            remain = len(self.df_future) - produced
            df_future_remain = self.df_future.iloc[produced:].copy()
            if use_feature_cache:
                X_forecast_input = self._build_msmr_step_input(runtime_cache, produced)
                categorical_features = runtime_cache["categorical_features"]
            else:
                df_forecast = pd.concat([self.df_history_for_lags, df_future_remain], ignore_index=True, copy=False)
                df_date_future_slice, df_weather_future_slice = self._slice_future_aux_by_forecast(df_forecast)
                (df_forecast_featured,
                 predictor_features,
                 target_output_features,
                 categorical_features) = self.feature_engineer.create_features(
                    df_series=df_forecast,
                    df_date_history=None,
                    df_date_future=df_date_future_slice,
                    df_weather_history=None,
                    df_weather_future=df_weather_future_slice,
                    df_custom_future=self.df_custom_future,
                    endogenous_features_with_target=self.endogenous_features,
                    target_feature=self.target_feature,
                    horizon=self.horizon,
                )
                predictor_features, categorical_features = self._apply_selected_feature_subset(
                    predictor_features, categorical_features
                )
                anchor_idx = max(len(self.df_history_for_lags) - 1, 0)
                X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[anchor_idx:anchor_idx + 1]
            X_forecast_processed = self.feature_scaler.transform(X_forecast_input, categorical_features)
            point_pred, quantile_preds = self._predict_point_and_quantiles(X_forecast_processed)

            pred_vec = self._to_1d(point_pred[0] if np.asarray(point_pred).ndim > 1 else point_pred)
            take = min(block_size, remain, len(pred_vec))
            block_pred = pred_vec[:take]
            y_preds.extend(block_pred.tolist())
            if quantile_preds:
                for q, pred in quantile_preds.items():
                    q_vec = self._to_1d(pred[0] if np.asarray(pred).ndim > 1 else pred)
                    quantile_store.setdefault(q, []).extend(q_vec[:take].tolist())

            # 按步回填目标+其他内生变量(持久性)到历史窗口，供下一块构造滞后特征
            for i in range(take):
                df_new = df_future_remain.iloc[i:i+1].copy()
                df_new[self.target_feature] = float(block_pred[i])
                if use_feature_cache:
                    runtime_cache["lag_state"][self.target_feature].append(float(block_pred[i]))
                for feat in other_endogenous:
                    if feat in df_new.columns and pd.notna(df_new[feat].iloc[0]):
                        if use_feature_cache:
                            runtime_cache["lag_state"][feat].append(float(df_new[feat].iloc[0]))
                        continue
                    # 优先用 aux 轨迹，回退持久性
                    global_step = produced + i
                    if (
                        self.aux_trajectories is not None
                        and feat in self.aux_trajectories
                        and global_step < len(self.aux_trajectories[feat])
                        and np.isfinite(self.aux_trajectories[feat][global_step])
                    ):
                        val = float(self.aux_trajectories[feat][global_step])
                    elif feat in self.df_history_for_lags.columns:
                        val = float(self.df_history_for_lags[feat].iloc[-1])
                    else:
                        val = 0.0
                    df_new[feat] = val
                    if use_feature_cache:
                        runtime_cache["lag_state"][feat].append(val)
                for col in self.df_history_for_lags.columns:
                    if col not in df_new.columns:
                        df_new[col] = self.df_history_for_lags[col].iloc[-1]
                self._append_history_row(df_new)

        self._finalize_recursive_quantiles(quantile_store)
        return np.asarray(y_preds[:len(self.df_future)])
    # ------------------------------
    # forecasting
    # ------------------------------
    def _resolve_blend_weights(self) -> List[float]:
        """解析 blend 权重：ridge_stacking 读 blend_weights.csv，否则用固定 blend_weights。"""
        strategy = str(getattr(self.args, "blend_weight_strategy", "fixed")).lower()
        if strategy == "ridge_stacking":
            w_path = self.args.test_results_dir.joinpath("blend_weights.csv")
            if w_path.exists():
                df_w = pd.read_csv(w_path)
                if len(df_w) > 0:
                    w_d = float(df_w["direct_weight"].iloc[-1])
                    w_r = float(df_w["recursive_weight"].iloc[-1])
                    total = w_d + w_r
                    if total > 0:
                        return [w_d / total, w_r / total]
                logger.warning(f"{self.log_prefix} blend_weights.csv empty or invalid; fallback to fixed.")
            else:
                logger.warning(f"{self.log_prefix} blend_weights.csv not found; fallback to fixed (need is_testing=True).")
        weights = list(getattr(self.args, "blend_weights", [0.5, 0.5]))[:2]
        total = sum(weights) or 1.0
        return [w / total for w in weights]

    def _blend_forecast(self) -> np.ndarray:
        """Direct+Recursive 加权融合预测（USBR/MSBR）。"""
        is_multi = str(self.args.pred_method).startswith("multivariate")
        # 1. Direct 子预测
        self.model = self.blend_direct_model
        if is_multi:
            d_pred = self.multivariate_single_multi_step_direct_forecast()
        else:
            d_pred = self.univariate_single_multi_step_direct_forecast()
        d_pred = np.asarray(d_pred, dtype=float).flatten()
        # 2. 重置 recursive 状态（Direct 推理可能改过 df_history_for_lags 列）
        self.df_history_for_lags = self.df_history.iloc[-self.history_context_length:].copy()
        self._recursive_schema_cache = {}
        # 3. Recursive 子预测
        self.model = self.blend_recursive_model
        if is_multi:
            r_pred = self.multivariate_single_multi_step_recursive_forecast()
        else:
            r_pred = self.univariate_single_multi_step_recursive_forecast()
        r_pred = np.asarray(r_pred, dtype=float).flatten()
        # 保存分预测（供 cv_plot 记录和 ridge_stacking）
        self.blend_direct_pred = d_pred
        self.blend_recursive_pred = r_pred
        # 4. 加权融合
        n = min(len(d_pred), len(r_pred))
        weights = self._resolve_blend_weights()
        raw_pred = weights[0] * d_pred[:n] + weights[1] * r_pred[:n]
        # 复位 main 指向 Direct 子模型（保持状态一致，避免后续复用异常）
        self.model = self.blend_direct_model
        logger.info(f"{self.log_prefix} Blend weights: direct={weights[0]:.4f}, recursive={weights[1]:.4f}")
        return raw_pred

    def _blend_forecast_quantile(self) -> np.ndarray:
        """USBR/MSBR 的 quantile 融合预测：每个分位数独立做 Direct+Recursive 加权融合。

        权重（direct_weight/recursive_weight）学自 cv_plot_df 中 median 分位数的
        Direct/Recursive 分量，所有分位数共用同一组权重（权重语义是子模型贡献比例，
        对分位数边界同样适用）。
        """
        if not self._is_quantile_bundle():
            raise ValueError(f"{self.log_prefix} blend quantile requires a quantile bundle.")
        is_multi = str(self.args.pred_method).startswith("multivariate")
        original_model = self.model
        quantile_models = self.model.get("models", {})
        weights = self._resolve_blend_weights()
        median_q = float(self.model.get("median_quantile", 0.5))
        quantile_preds = {}
        blend_direct_ref = None
        blend_recursive_ref = None
        for q_key, blend_bundle in quantile_models.items():
            q = float(q_key)
            direct_q = blend_bundle.get("direct")
            recursive_q = blend_bundle.get("recursive")
            if direct_q is None or recursive_q is None:
                raise ValueError(
                    f"{self.log_prefix} blend quantile bundle for q={q} missing direct/recursive sub-model."
                )
            # 1. Direct 子预测
            self.model = direct_q
            if is_multi:
                d_pred = self.multivariate_single_multi_step_direct_forecast()
            else:
                d_pred = self.univariate_single_multi_step_direct_forecast()
            d_pred = np.asarray(d_pred, dtype=float).flatten()
            # 2. 重置 recursive 状态（Direct 推理可能改过 df_history_for_lags / schema）
            self.df_history_for_lags = self.df_history.iloc[-self.history_context_length:].copy()
            self._recursive_schema_cache = {}
            # 3. Recursive 子预测
            self.model = recursive_q
            if is_multi:
                r_pred = self.multivariate_single_multi_step_recursive_forecast()
            else:
                r_pred = self.univariate_single_multi_step_recursive_forecast()
            r_pred = np.asarray(r_pred, dtype=float).flatten()
            n = min(len(d_pred), len(r_pred))
            quantile_preds[q] = weights[0] * d_pred[:n] + weights[1] * r_pred[:n]
            if abs(q - median_q) < 1e-9:
                blend_direct_ref = d_pred
                blend_recursive_ref = r_pred
        # 记录 median 分位数的 Direct/Recursive 分量，供 cv_plot 与 ridge_stacking
        if blend_direct_ref is not None:
            self.blend_direct_pred = blend_direct_ref
            self.blend_recursive_pred = blend_recursive_ref
        self.quantile_outputs = quantile_preds
        # point = median 分位数融合结果
        point_pred = quantile_preds.get(median_q)
        if point_pred is None:
            median_q = min(quantile_preds.keys(), key=lambda x: abs(x - median_q))
            point_pred = quantile_preds[median_q]
        # 复位 main 指向 quantile bundle，避免后续复用异常
        self.model = original_model
        logger.info(
            f"{self.log_prefix} Blend quantile weights: direct={weights[0]:.4f}, "
            f"recursive={weights[1]:.4f} (n_quantiles={len(quantile_preds)})"
        )
        return point_pred

    def _restore_target_decomposition(self, values) -> np.ndarray:
        """给点预测和全部分位数加回同一确定性趋势/季节分量。"""
        result = np.asarray(values).reshape(-1)
        decomposer = getattr(self, "target_decomposer", None)
        if decomposer is None or not getattr(decomposer, "is_fitted", False):
            return result
        n = min(len(result), len(self.df_future))
        future_times = self.df_future["time"].iloc[:n]
        restored = decomposer.restore(result[:n], future_times)
        if self.quantile_outputs:
            self.quantile_outputs = {
                q: decomposer.restore(np.asarray(pred).reshape(-1)[:n], future_times)
                for q, pred in self.quantile_outputs.items()
            }
        return restored

    def _predict_by_method(self) -> np.ndarray:
        """
        根据配置分发预测策略并返回一维预测数组
        """
        # 每次预测前重置，避免复用同一 Forecaster 实例时污染
        self.quantile_outputs = None
        perf_start = time.perf_counter()
        if self.args.pred_method == "univariate-single-multistep-direct-pointwise":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_direct_pointwise_forecast(USMDP)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_direct_pointwise_forecast()
            logger.info(f"{self.log_prefix} USMDP forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "univariate-single-multistep-direct":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_direct_forecast(USMD)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_direct_forecast()
            logger.info(f"{self.log_prefix} USMD forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_recursive_forecast(USMR)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_recursive_forecast()
            logger.info(f"{self.log_prefix} USMR forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_direct_recursive_forecast(USMDR)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_direct_recursive_forecast()
            logger.info(f"{self.log_prefix} USMDR forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            logger.info(f"{self.log_prefix} Forecast method: multivariate_single_multi_step_direct_forecast(MSMD)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.multivariate_single_multi_step_direct_forecast()
            logger.info(f"{self.log_prefix} MSMD forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            logger.info(f"{self.log_prefix} Forecast method: multivariate_single_multi_step_recursive_forecast(MSMR)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.multivariate_single_multi_step_recursive_forecast()
            logger.info(f"{self.log_prefix} MSMR forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            logger.info(f"{self.log_prefix} Forecast method: multivariate_single_multi_step_direct_recursive_forecast(MSMDR)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.multivariate_single_multi_step_direct_recursive_forecast()
            logger.info(f"{self.log_prefix} MSMDR forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method in (
            "univariate-single-multistep-blend-direct-recursive",
            "multivariate-single-multistep-blend-direct-recursive",
        ):
            if str(getattr(self.args, "predict_type", "point")).lower() == "quantile":
                logger.info(f"{self.log_prefix} Forecast method: blend_direct_recursive_quantile(USBR/MSBR)")
                logger.info(f"{self.log_prefix} {'-' * 60}")
                raw_pred = self._blend_forecast_quantile()
                logger.info(f"{self.log_prefix} USBR/MSBR quantile forecast completed, predicted {len(raw_pred)} steps.")
            else:
                logger.info(f"{self.log_prefix} Forecast method: blend_direct_recursive(USBR/MSBR)")
                logger.info(f"{self.log_prefix} {'-' * 60}")
                raw_pred = self._blend_forecast()
                logger.info(f"{self.log_prefix} USBR/MSBR forecast completed, predicted {len(raw_pred)} steps.")
        else:
            raise ValueError(f"{self.log_prefix} Unsupported pred_method: {self.args.pred_method}")

        pred_arr = np.asarray(raw_pred)
        if pred_arr.ndim == 0:
            result = np.asarray([float(pred_arr)])
        elif pred_arr.ndim == 1:
            result = pred_arr
        elif pred_arr.shape[0] == 1:
            result = pred_arr[0]
        elif pred_arr.shape[1] == 1:
            result = pred_arr[:, 0]
        else:
            result = pred_arr[:, 0]

        # 目标分解还原：给点预测和全部分位数加回相同的确定性分量。
        result = self._restore_target_decomposition(result)

        logger.info(
            f"{self.log_prefix} Forecast method runtime: "
            f"{self.args.pred_method} took {time.perf_counter() - perf_start:.3f}s"
        )
        return result

    def forecast_results_save(self, df_history, df_future, n_per_day):
        """
        输出结果处理
        """
        # 预测结果保存
        df_future = df_future.copy()
        df_future["time"] = pd.to_datetime(df_future["time"])
        df_future = df_future.sort_values(by=["time"]).reset_index(drop=True)
        df_future.to_csv(self.args.pred_results_dir.joinpath("prediction.csv"), encoding="utf_8_sig", index=False)
        # 历史上下文截取：以未来预测起点为边界，取其前最近若干历史真值
        # 上下文长度与 horizon 挂钩，避免低频(日/周)下 2*n_per_day 退化为极少点
        plot_context_len = max(2 * n_per_day, int(getattr(self, "horizon", 2 * n_per_day)))
        y_trues_df_plot = pd.DataFrame()
        if df_history is not None and not df_history.empty and "time" in df_history.columns and "y" in df_history.columns:
            df_history_plot = df_history.copy()
            df_history_plot["time"] = pd.to_datetime(df_history_plot["time"])
            df_history_plot = df_history_plot.sort_values(by=["time"]).dropna(subset=["y"]).reset_index(drop=True)
            if not df_future.empty:
                future_start = df_future["time"].iloc[0]
                history_before_future = df_history_plot[df_history_plot["time"] < future_start]
                if history_before_future.empty:
                    logger.warning(
                        f"{self.log_prefix} No history before forecast start ({future_start}); "
                        f"fallback to latest available {plot_context_len}-point history for plotting."
                    )
                    y_trues_df_plot = df_history_plot.tail(plot_context_len).copy()
                else:
                    y_trues_df_plot = history_before_future.tail(plot_context_len).copy()
            else:
                y_trues_df_plot = df_history_plot.tail(plot_context_len).copy()
        # 保留历史上下文，便于生产问题定位
        if df_history is not None and not df_history.empty:
            df_history.copy().to_csv(
                self.args.pred_results_dir.joinpath("history_context.csv"),
                encoding="utf_8_sig",
                index=False,
            )
        history_plot_meta = None
        if not y_trues_df_plot.empty and "y" in y_trues_df_plot.columns:
            history_plot_meta = build_eval_mask(
                y_trues_df_plot["y"].values,
                mode=self.args.mode,
                percentile=self.args.percentile,
                min_value=self.args.min_value,
                max_value=self.args.max_value,
            )
            logger.info(
                f"{self.log_prefix} History plot mask threshold={history_plot_meta['threshold']}, "
                f"valid_points={history_plot_meta['valid_points']}, "
                f"excluded_points={history_plot_meta['excluded_points']}, "
                f"excluded_ratio={history_plot_meta['excluded_ratio']:.6f}"
            )
            y_trues_df_plot = y_trues_df_plot.copy()
            y_trues_df_plot["plot_valid"] = history_plot_meta["valid_mask"]
            y_trues_df_plot["plot_value"] = np.where(
                history_plot_meta["valid_mask"],
                y_trues_df_plot["y"],
                np.nan,
            )
        # 拼接可视化数据：最近若干历史真值 + 未来预测(含分位数)
        history_part = pd.DataFrame()
        if not y_trues_df_plot.empty:
            history_part = y_trues_df_plot[["time", "y", "plot_value", "plot_valid"]].rename(columns={"y": "value"})
            history_part["raw_value"] = history_part["value"]
            history_part["series_type"] = "history_true"
        future_part = pd.DataFrame()
        if not df_future.empty and "predict_value" in df_future.columns:
            # 分位数列一并写入,使 plot_concat 数据完整
            q_cols = [c for c in df_future.columns if str(c).startswith("predict_q")]
            future_part = df_future[["time", "predict_value"] + q_cols].rename(columns={"predict_value": "value"})
            future_part["raw_value"] = future_part["value"]
            future_part["plot_value"] = future_part["value"]
            future_part["plot_valid"] = True
            future_part["series_type"] = "future_pred"
        if not history_part.empty or not future_part.empty:
            plot_concat_df = pd.concat([history_part, future_part], axis=0, ignore_index=True)
            plot_concat_df = plot_concat_df.sort_values(by=["time"]).reset_index(drop=True)
            plot_concat_df.to_csv(
                self.args.pred_results_dir.joinpath("prediction_plot_concat.csv"),
                encoding="utf_8_sig",
                index=False,
            )
        import matplotlib.pyplot as plt
        plt.figure(figsize=(25, 8))
        # 历史真值:用原始 y 而非 masked plot_value,保证线条连续不断
        if not y_trues_df_plot.empty and "y" in y_trues_df_plot.columns:
            plt.plot(y_trues_df_plot["time"], y_trues_df_plot["y"], label="Trues", lw=2.0)
        # 未来预测(点)
        if not df_future.empty and "predict_value" in df_future.columns:
            plt.plot(df_future["time"], df_future["predict_value"], label="Preds", lw=2.0, ls="-.")
        # 分位数预测区间带(若启用):填充 q_low~q_high
        qcols = sorted(c for c in df_future.columns if str(c).startswith("predict_q"))
        if len(qcols) >= 2:
            plt.fill_between(
                df_future["time"],
                df_future[qcols[0]].astype(float).values,
                df_future[qcols[-1]].astype(float).values,
                color="tab:blue", alpha=0.15, label=f"PI [{qcols[0]},{qcols[-1]}]",
            )
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"模型预测--{self.args.pred_method}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=1.0)
        plt.tight_layout()
        # plt.xticks(rotation=45)
        plt.savefig(self.args.pred_results_dir.joinpath("prediction.png"), dpi=300, bbox_inches="tight")
        plt.close()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
