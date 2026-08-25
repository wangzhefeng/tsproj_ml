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
from copy import copy
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer
from models.multistep.contracts import (
    require_endogenous_history,
    require_future_horizon,
)
from models.multistep.executors import get_executor
from models.multistep.artifacts import (
    AuxiliaryEndogenousArtifact,
    BlendArtifact,
    LegacyArtifactAdapter,
    StrategyArtifact,
)
from models.multistep.backfill import build_endogenous_future_provider
from models.multistep.plans import TrainingLayout
from models.multistep.resolve import resolve_strategy
from models.multistep.runtime import ensure_resolved_strategy
from models.multistep.state import RecursiveFeatureCache
from models.multistep.weights import BlendWeights
from models.multistep.spec import InputScope
from probabilistic.types import (
    BlendQuantileModel,
    ForecastDistribution,
    ProbabilisticModelBundle,
    QuantileGrid,
)
from utils.eval_mask import build_eval_mask
from utils.log_util import logger
from utils.multistep_contract import validate_direct_feature_alignment

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
                 target_transform=None,
                 df_custom_future=None,
                 log_prefix: str = "[Forecaster]"):
        self.args = args
        self.horizon = horizon
        self.target_feature = target_feature
        # 策略解析：__init__ 全路径解析并缓存；裸构造（__new__）在首次使用时惰性解析。
        pred_method = getattr(args, "pred_method", None)
        self.resolved_strategy = (
            resolve_strategy(args, horizon, target_feature=target_feature)
            if pred_method
            else None
        )
        if len(df_future) != horizon:
            raise ValueError(
                f"{log_prefix} future frame length mismatch: expected horizon={horizon}, "
                f"got {len(df_future)}."
            )
        adapted_model = LegacyArtifactAdapter.adapt(
            model,
            strategy=self.resolved_strategy,
            feature_schema=selected_features or (),
        )
        self.multistep_metadata = None
        if isinstance(adapted_model, StrategyArtifact):
            self.multistep_metadata = adapted_model.metadata
            self.model = adapted_model.model
        else:
            self.model = adapted_model
        self.model_output_width = (
            self.multistep_metadata.model_output_width
            if self.multistep_metadata is not None
            else None
        )
        # 解包 auxiliary bundle（MSMR/MSMDR + endogenous_backfill_strategy=auxiliary）
        self.aux_forecaster = None
        self.aux_trajectories = None
        if isinstance(self.model, AuxiliaryEndogenousArtifact):
            self.aux_forecaster = self.model.auxiliary_model
            self.model = self.model.main_model
        # 解包 blend bundle（USBR/MSBR = Direct+Recursive 融合）
        self.blend_direct_model = None
        self.blend_recursive_model = None
        self.blend_direct_pred = None
        self.blend_recursive_pred = None
        self.blend_weights = None
        if isinstance(self.model, BlendArtifact):
            self.blend_direct_model = self.model.direct_model
            self.blend_recursive_model = self.model.recursive_model
            self.blend_weights = self.model.weights
        elif isinstance(self.model, ProbabilisticModelBundle):
            blend_metadata = self.model.metadata.get("blend_weights")
            if blend_metadata is not None:
                self.blend_weights = BlendWeights(
                    direct=float(blend_metadata["direct"]),
                    recursive=float(blend_metadata["recursive"]),
                    strategy=str(blend_metadata["strategy"]),
                    calibration_windows=int(blend_metadata.get("calibration_windows", 0)),
                )
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.df_history = df_history
        self.df_future = df_future
        self.df_date_future = df_date_future
        self.df_weather_future = df_weather_future
        self.df_custom_future = df_custom_future or []
        self.endogenous_features = endogenous_features
        self.target_output_features = target_output_features
        self.categorical_features = categorical_features
        self.selected_features = selected_features
        self.target_decomposer = target_decomposer
        self.target_transform = target_transform
        if target_scaler is not None:
            self.prediction_target_columns = target_scaler.get_prediction_target_columns(
                self.args.pred_method,
                target_output_features,
                direct_strategy=str(getattr(self.args, "direct_strategy", "multioutput")),
            )
        else:
            self.prediction_target_columns = list(target_output_features or [target_feature])
        if self.target_transform is not None and target_scaler is not None:
            self.target_transform.attach_fitted_target_scaler(
                target_scaler,
                target_columns=self.prediction_target_columns,
            )
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
        self._quantile_outputs = None
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

        self.endogenous_future_provider = build_endogenous_future_provider(self)

        if (
            self.resolved_strategy is not None
            and self.resolved_strategy.spec.input_scope == InputScope.ALL_ENDOGENOUS
        ):
            require_endogenous_history(self.df_history_for_lags, self.endogenous_features)

    def _concat_history_and_future(self) -> pd.DataFrame:
        return pd.concat(
            [self.df_history_for_lags, self.df_future.copy()],
            ignore_index=True,
            copy=False,
        )

    def _fork_for_model(self, model):
        """为 Blend 子路径创建隔离运行上下文，不改写共享模型和递归历史。"""
        child = copy(self)
        child.model = model
        child.blend_direct_model = None
        child.blend_recursive_model = None
        child.blend_direct_pred = None
        child.blend_recursive_pred = None
        child.df_history_for_lags = self.df_history_for_lags.copy(deep=True)
        child._recursive_schema_cache = {}
        child._msmr_runtime_cache = None
        child._msmdr_runtime_cache = None
        child._quantile_outputs = None
        return child

    @property
    def quantile_outputs(self) -> Optional[Dict[float, np.ndarray]]:
        """迁移期只读兼容视图；主链应消费 ForecastDistribution。"""
        if self._quantile_outputs is None:
            return None
        return {
            float(level): np.asarray(values, dtype=float).copy()
            for level, values in self._quantile_outputs.items()
        }

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
    def _require_direct_prediction_length(pred: Any, n_required: int, label: str) -> np.ndarray:
        """Direct 输出必须与请求 horizon 精确一致，禁止截断或复制尾值补齐。"""
        pred_1d = Forecaster._to_1d(pred)
        if len(pred_1d) != n_required:
            raise ValueError(
                f"{label} direct prediction length mismatch: "
                f"expected {n_required}, got {len(pred_1d)}."
            )
        return pred_1d

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
                    raise ValueError(
                        f"{self.log_prefix} Missing required inference feature columns "
                        f"(no-scaler baseline): {missing_cols}."
                    )
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

        lag_state = self._build_endogenous_lag_state()
        cache_future_features = self._resolve_recursive_future_features(
            exogenous_features,
            predictor_features,
            df_future_exog,
        )

        self._msmr_runtime_cache = RecursiveFeatureCache(
            df_future_exog=df_future_exog.reset_index(drop=True),
            exogenous_features=tuple(cache_future_features),
            predictor_features=tuple(predictor_features),
            categorical_features=tuple(categorical_features),
            target_output_features=tuple(target_output_features),
            lag_feature_names=frozenset(lag_feature_names),
            lag_state=lag_state,
            lags=tuple(lags),
        )
        return self._msmr_runtime_cache

    def _resolve_recursive_future_features(
        self,
        exogenous_features: List[str],
        predictor_features: List[str],
        df_future_exog: pd.DataFrame,
    ) -> List[str]:
        """返回递归缓存需从 future frame 原样携带的特征。"""
        result = list(exogenous_features)
        panel_key = ensure_resolved_strategy(self).feature_plan.panel_key
        if (
            panel_key
            and panel_key in predictor_features
            and panel_key in df_future_exog.columns
            and panel_key not in result
        ):
            result.append(panel_key)
        return result

    @staticmethod
    def _read_lag_value(buffer: deque, lag: int) -> float:
        values = list(buffer)
        if not values:
            raise ValueError("recursive lag state is empty.")
        if lag <= len(values):
            return values[-lag]
        return values[0]

    def _build_endogenous_lag_state(self) -> Dict[str, deque]:
        require_endogenous_history(self.df_history_for_lags, self.endogenous_features)
        lag_state = {}
        for column in self.endogenous_features:
            values = pd.to_numeric(
                self.df_history_for_lags[column], errors="coerce"
            ).ffill().bfill().tolist()
            if not values or not np.isfinite(values).all():
                raise ValueError(
                    f"history has no complete finite endogenous lag state: {column}."
                )
            if len(values) < self.max_lag:
                values = [values[0]] * (self.max_lag - len(values)) + values
            lag_state[column] = deque(values[-self.max_lag :], maxlen=self.max_lag)
        return lag_state

    def _build_msmr_step_input(
        self,
        runtime_cache: RecursiveFeatureCache,
        step: int,
    ) -> pd.DataFrame:
        return runtime_cache.build_step_input(step)

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

        lag_state = self._build_endogenous_lag_state()
        cache_future_features = self._resolve_recursive_future_features(
            exogenous_features,
            predictor_features,
            df_future_exog,
        )

        self._msmdr_runtime_cache = RecursiveFeatureCache(
            df_future_exog=df_future_exog.reset_index(drop=True),
            exogenous_features=tuple(cache_future_features),
            predictor_features=tuple(predictor_features),
            categorical_features=tuple(categorical_features),
            target_output_features=tuple(target_output_features),
            lag_feature_names=frozenset(lag_feature_names),
            lag_state=lag_state,
            lags=tuple(lags),
        )
        return self._msmdr_runtime_cache

    def _is_quantile_bundle(self) -> bool:
        return isinstance(self.model, ProbabilisticModelBundle)

    def _predict_point_and_quantiles(self, X_processed: pd.DataFrame) -> Tuple[np.ndarray, Optional[Dict[float, np.ndarray]]]:
        """
        统一预测入口：
        - 点预测模型: 返回(point_pred, None)
        - 分位数模型: 返回(中位分位点预测, {q: pred_q})
        """
        if not self._is_quantile_bundle():
            return np.asarray(self.model.predict(X_processed)), None

        quantile_models = self.model.models_by_quantile
        quantiles = list(self.model.spec.quantiles)
        median_q = self.model.spec.point_quantile
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

        point_pred = quantile_preds.get(median_q)
        if point_pred is None:
            raise ValueError(
                f"{self.log_prefix} typed quantile bundle is missing "
                f"point_quantile={median_q:g}"
            )
        return point_pred, quantile_preds

    def _record_quantile_direct(self, quantile_preds: Optional[Dict[float, np.ndarray]], n_required: int):
        if not quantile_preds:
            return
        self._quantile_outputs = {}
        for q, pred in quantile_preds.items():
            pred_for_horizon = pred[0] if np.asarray(pred).ndim > 1 else pred
            self._quantile_outputs[q] = self._require_direct_prediction_length(
                pred_for_horizon,
                n_required,
                label=f"quantile q={q}",
            )

    def _record_quantile_recursive_step(self, store: Dict[float, List[float]], quantile_preds: Optional[Dict[float, np.ndarray]]):
        if not quantile_preds:
            return
        for q, pred in quantile_preds.items():
            store.setdefault(q, []).append(self._to_scalar(pred))

    def _finalize_recursive_quantiles(self, store: Dict[float, List[float]]):
        if not store:
            return
        self._quantile_outputs = {q: np.asarray(v, dtype=float) for q, v in store.items()}

    def _is_horizon_feature_mode(self) -> bool:
        """USMD/MSMD 且 direct_strategy=horizon_feature 时推理走多行展开。"""
        resolved = ensure_resolved_strategy(self)
        return resolved.training_plan.layout == TrainingLayout.HORIZON_LONG

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
    def _resolve_blend_weights(self) -> BlendWeights:
        """返回模型产物内权重；固定权重可直接由配置构造。"""
        if self.blend_weights is not None:
            return self.blend_weights
        return BlendWeights.from_args(self.args)


    def _restore_target_decomposition(self, values) -> np.ndarray:
        """给点预测和全部分位数加回同一确定性趋势/季节分量。"""
        result = np.asarray(values).reshape(-1)
        decomposer = getattr(self, "target_decomposer", None)
        if decomposer is None or not getattr(decomposer, "is_fitted", False):
            return result
        n = min(len(result), len(self.df_future))
        future_times = self.df_future["time"].iloc[:n]
        restored = decomposer.restore(result[:n], future_times)
        if self._quantile_outputs:
            self._quantile_outputs = {
                q: decomposer.restore(np.asarray(pred).reshape(-1)[:n], future_times)
                for q, pred in self._quantile_outputs.items()
            }
        return restored

    def _restore_target_transform(self, values) -> np.ndarray:
        """通过同一变换栈恢复 point 和全部 quantile；无新栈时兼容旧分解入口。"""
        pipeline = getattr(self, "target_transform", None)
        if pipeline is None:
            return self._restore_target_decomposition(values)

        result = np.asarray(values, dtype=float).reshape(-1)
        if len(result) != len(self.df_future):
            raise ValueError(
                f"{self.log_prefix} target restore length mismatch: "
                f"prediction={len(result)}, future={len(self.df_future)}"
            )
        future_times = self.df_future["time"]
        restored = pipeline.restore(
            result,
            future_times,
            target_columns=self.prediction_target_columns,
        )
        if self._quantile_outputs:
            levels = sorted(self._quantile_outputs, key=float)
            columns = []
            for level in levels:
                column = np.asarray(self._quantile_outputs[level], dtype=float).reshape(-1)
                if len(column) != len(result):
                    raise ValueError(
                        f"{self.log_prefix} quantile q={float(level):g} restore length mismatch: "
                        f"prediction={len(column)}, expected={len(result)}"
                    )
                columns.append(column)
            restored_matrix = pipeline.restore_quantile_matrix(
                np.column_stack(columns),
                future_times,
                target_columns=self.prediction_target_columns,
            )
            self._quantile_outputs = {
                level: restored_matrix[:, index]
                for index, level in enumerate(levels)
            }
        return restored

    def _predict_by_method(self) -> Any:
        """按解析后的 rollout family 通过执行器 catalog 分发预测。"""
        self._quantile_outputs = None
        perf_start = time.perf_counter()
        resolved = ensure_resolved_strategy(self)
        executor = get_executor(resolved)
        raw_pred = executor.execute(self)
        result = np.asarray(raw_pred, dtype=float).reshape(-1)
        if result.shape != (self.horizon,):
            raise ValueError(
                f"{self.log_prefix} forecast output shape mismatch: "
                f"expected ({self.horizon},), got {result.shape}."
            )
        if not np.isfinite(result).all():
            raise ValueError(f"{self.log_prefix} forecast output contains non-finite values.")

        result = self._restore_target_transform(result)
        logger.info(
            f"{self.log_prefix} Forecast method runtime: "
            f"{resolved.spec.method} took {time.perf_counter() - perf_start:.3f}s"
        )
        if self._quantile_outputs:
            if isinstance(self.model, ProbabilisticModelBundle):
                grid = self.model.quantile_grid
                recursive_propagation = self.model.recursive_propagation
            else:
                levels = tuple(sorted((float(q) for q in self._quantile_outputs), key=float))
                grid = QuantileGrid(levels, point_level=0.5)
                recursive_propagation = "median_path"
            quantile_matrix = np.column_stack(
                [
                    np.asarray(self._quantile_outputs[level], dtype=float).reshape(-1)
                    for level in grid.levels
                ]
            )
            if quantile_matrix.shape != (self.horizon, len(grid.levels)):
                raise ValueError(
                    f"{self.log_prefix} quantile output shape mismatch: "
                    f"expected ({self.horizon}, {len(grid.levels)}), got {quantile_matrix.shape}."
                )
            target_transform = getattr(self, "target_transform", None)
            target_decomposer = getattr(self, "target_decomposer", None)
            space = (
                "target"
                if target_transform is not None
                or bool(getattr(target_decomposer, "is_fitted", False))
                else "model"
            )
            return ForecastDistribution(
                point=np.asarray(result, dtype=float).reshape(-1),
                quantile_grid=grid,
                quantile_values=quantile_matrix,
                intervals={},
                space=space,
                quantile_stage="raw",
                forecast_times=pd.DatetimeIndex(self.df_future["time"]),
                metadata={"recursive_propagation": recursive_propagation},
            )
        return result

    def forecast_results_save(self, df_history, df_future, n_per_day):
        """
        输出结果处理
        """
        # 预测结果保存
        df_future = df_future.copy()
        df_future["time"] = pd.to_datetime(df_future["time"])
        series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
        is_panel = bool(getattr(self.args, "enable_global_training", False))
        sort_columns = [series_id_col, "time"] if is_panel else ["time"]
        if is_panel and series_id_col not in df_future.columns:
            raise ValueError(
                f"panel forecast result missing series ID column '{series_id_col}'."
            )
        df_future = df_future.sort_values(by=sort_columns).reset_index(drop=True)
        df_future.to_csv(self.args.pred_results_dir.joinpath("prediction.csv"), encoding="utf_8_sig", index=False)
        if is_panel:
            if df_history is not None and not df_history.empty:
                df_history.copy().to_csv(
                    self.args.pred_results_dir.joinpath("history_context.csv"),
                    encoding="utf_8_sig",
                    index=False,
                )
            logger.info(
                f"{self.log_prefix} Panel forecast saved without a single-series plot; "
                "prediction.csv preserves (series_id, time)."
            )
            return
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
