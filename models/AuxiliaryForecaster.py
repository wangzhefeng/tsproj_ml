# -*- coding: utf-8 -*-

# ***************************************************
# * File        : AuxiliaryForecaster.py
# * Description : 多变量递归（MSMR/MSMDR）非目标内生变量的辅助预测器。
# *               为每个非目标内生变量训练独立 1 步递归模型（reduced-form：
# *               只用自身滞后 + datetime 派生外生），替代持久性常量回填。
# * ***************************************************

from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from models.ModelFactory import ModelFactory
from utils.log_util import logger

LOGGING_LABEL = Path(__file__).name[:-3]


class AuxiliaryEndogenousForecaster:
    """
    非目标内生变量的辅助递归预测器。

    训练：对每个非目标内生变量 col，用 [col_lag_{lag} for lag in lags] + datetime
    派生外生作为特征，col.shift(-1) 作为目标，训练一个 1 步回归器。
    推理：逐步递归预测每个 col 的未来轨迹，供 MSMR/MSMDR 回填目标滞后特征。

    设计决策（reduced-form）：
    - 辅助模型只用"自身滞后 + datetime 外生"，不用其他内生变量的滞后。
    - 避免为每个内生建模→又需要其他内生未来→无限递归。
    - weather/date_type 等需 merge 的外生不在 aux 特征内（aux 从 time 列自派生 datetime）；
      aux 的目标是"比持久性常量好"，datetime 周期已能捕获主要日间模式。
    """

    def __init__(
        self,
        args: Any,
        endogenous_cols: List[str],
        target_feature: str,
        log_prefix: str = "[AuxiliaryForecaster]",
    ):
        self.args = args
        self.endogenous_cols = [c for c in endogenous_cols if c != target_feature]
        self.target_feature = target_feature
        self.log_prefix = log_prefix
        self.models: Dict[str, Any] = {}
        self.feature_cols: Dict[str, List[str]] = {}
        self.lags = [int(l) for l in (getattr(args, "lags", []) or []) if int(l) > 0]
        self.datetime_features: List[str] = list(getattr(args, "datetime_features", []) or [])
        self.aux_model_type = str(getattr(args, "auxiliary_model_type", "lightgbm"))

    def _build_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 time 列派生 datetime 特征（与 FeatureEngineer 对齐的子集）。"""
        result = pd.DataFrame(index=df.index)
        if "time" not in df.columns:
            return result
        dt = pd.to_datetime(df["time"])
        feature_map = {
            "minute": dt.dt.minute,
            "hour": dt.dt.hour,
            "day": dt.dt.day,
            "weekday": dt.dt.weekday,
            "day_of_week": dt.dt.day_of_week,
            "week_of_year": dt.dt.isocalendar().week.astype(int),
            "week": dt.dt.isocalendar().week.astype(int),
            "month": dt.dt.month,
            "quarter": dt.dt.quarter,
            "day_of_year": dt.dt.day_of_year,
            "year": dt.dt.year,
            "days_in_month": dt.dt.days_in_month,
        }
        for feat_name in self.datetime_features:
            if feat_name in feature_map:
                result[feat_name] = feature_map[feat_name]
        return result

    def _build_training_frame(self, df_history: pd.DataFrame, col: str) -> Optional[pd.DataFrame]:
        """为单个内生变量构造训练帧：滞后特征 + datetime 外生 + 下一时点目标。"""
        df = df_history.copy()
        if "time" not in df.columns or col not in df.columns:
            return None
        df = df.set_index("time")
        # 滞后特征
        for lag in self.lags:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        # datetime 外生：df_history 带 time 列（RangeIndex），df 已 set_index(time)（DatetimeIndex），
        # 赋值时用 .values 绕过 index 对齐，避免全 NaN
        df_dt = self._build_datetime_features(df_history)
        for dt_col in df_dt.columns:
            df[dt_col] = df_dt[dt_col].values
        # 目标 = 下一时点
        df[f"__{col}_next"] = df[col].shift(-1)
        # 特征列
        feature_cols = [f"{col}_lag_{lag}" for lag in self.lags] + [
            c for c in df_dt.columns if c in df.columns
        ]
        self.feature_cols[col] = feature_cols
        # dropna
        needed = feature_cols + [f"__{col}_next"]
        df = df.dropna(subset=needed)
        return df

    def fit(self, df_history: pd.DataFrame) -> "AuxiliaryEndogenousForecaster":
        """训练每个非目标内生变量的 1 步递归模型。"""
        if not self.endogenous_cols:
            logger.info(f"{self.log_prefix} No endogenous cols to train aux models (single-output target only).")
            return self
        if not self.lags:
            logger.warning(f"{self.log_prefix} No lags configured; aux models cannot be trained. Falling back to persistence.")
            return self

        factory = ModelFactory(log_prefix=self.log_prefix)
        aux_params = dict(getattr(self.args, "auxiliary_model_params", {}) or {})
        model_threads = int(getattr(self.args, "model_thread_count", 1) or 1)
        mt = self.aux_model_type.lower()
        if mt in ["lightgbm", "lgb", "xgboost", "xgb", "randomforest", "rf"]:
            aux_params["n_jobs"] = model_threads
        elif mt in ["catboost", "cat"]:
            aux_params["thread_count"] = model_threads

        trained = 0
        for col in self.endogenous_cols:
            df_train = self._build_training_frame(df_history, col)
            if df_train is None or df_train.empty:
                logger.warning(f"{self.log_prefix} Skip aux model for '{col}': empty training frame.")
                continue
            X_aux = df_train[self.feature_cols[col]]
            y_aux = df_train[f"__{col}_next"].values
            try:
                wrapper = factory.create_model(model_type=self.aux_model_type, model_params=aux_params)
                wrapper.fit(X_aux, y_aux)
                self.models[col] = wrapper
                trained += 1
            except Exception as e:
                logger.warning(f"{self.log_prefix} Failed to train aux model for '{col}': {e}")
        logger.info(f"{self.log_prefix} Auxiliary models trained: {trained}/{len(self.endogenous_cols)} cols.")
        # 设计断言：配置了 datetime_features 但没有一个辅助模型用到 datetime 特征 → 大概率 time 列缺失或特征名不匹配
        if self.datetime_features and self.models:
            dt_used = any(
                any(feat in dt_feats for feat in self.datetime_features)
                for dt_feats in self.feature_cols.values()
            )
            if not dt_used:
                logger.warning(
                    f"{self.log_prefix} datetime_features configured ({self.datetime_features}) but none "
                    f"were used by aux models; check 'time' column availability in history/future frames."
                )
        return self

    def predict_horizon(
        self,
        df_history: pd.DataFrame,
        df_future: pd.DataFrame,
        horizon: int,
    ) -> Dict[str, np.ndarray]:
        """
        逐步递归预测每个非目标内生变量的未来轨迹。

        返回 {col: np.array(horizon)}；未训练成功的 col 返回 NaN 数组（调用方回退持久性）。
        """
        trajectories: Dict[str, np.ndarray] = {}
        max_lag = max(self.lags) if self.lags else 1
        # 未来 datetime 外生
        df_future_dt = self._build_datetime_features(df_future) if "time" in df_future.columns else pd.DataFrame()

        for col in self.endogenous_cols:
            if col not in self.models:
                trajectories[col] = np.full(horizon, np.nan)
                continue
            # 初始化 lag buffer（从 df_history 尾部取 max_lag 个值）
            if col in df_history.columns:
                seed = df_history[col].iloc[-max_lag:].tolist()
            else:
                seed = [0.0]
            seed = [0.0 if pd.isna(v) else float(v) for v in seed]
            if len(seed) < max_lag:
                seed = [seed[0]] * (max_lag - len(seed)) + seed
            buffer = deque(seed[-max_lag:], maxlen=max_lag)

            preds: List[float] = []
            feat_cols = self.feature_cols.get(col, [])
            for h in range(horizon):
                row: Dict[str, float] = {}
                # 滞后取值
                for lag in self.lags:
                    idx = len(buffer) - lag
                    row[f"{col}_lag_{lag}"] = float(buffer[idx]) if idx >= 0 else float(buffer[0])
                # datetime 外生取未来第 h 行
                for dt_col in feat_cols:
                    if dt_col.startswith(f"{col}_lag_"):
                        continue
                    if dt_col in df_future_dt.columns and h < len(df_future_dt):
                        row[dt_col] = float(df_future_dt[dt_col].iloc[h])
                    else:
                        row[dt_col] = 0.0
                X = pd.DataFrame([row]).reindex(columns=feat_cols)
                pred = float(np.asarray(self.models[col].predict(X)).reshape(-1)[0])
                if not np.isfinite(pred):
                    pred = float(buffer[-1]) if buffer else 0.0
                preds.append(pred)
                buffer.append(pred)
            trajectories[col] = np.array(preds)
        return trajectories

    def is_empty(self) -> bool:
        return len(self.models) == 0


def maybe_build_auxiliary_bundle(
    args: Any,
    model: Any,
    df_history: pd.DataFrame,
    endogenous_features_with_target: List[str],
    target_feature: str,
    log_prefix: str,
) -> Any:
    """
    如果启用 auxiliary 回填策略（MSMR/MSMDR），训练辅助模型并返回 bundle dict；
    否则原样返回 model。调用方（Model.train / _window_test）在 Trainer.train 返回后调用。

    bundle 格式: {"bundle_type": "auxiliary_endogenous", "main": model, "aux": AuxiliaryEndogenousForecaster}
    Forecaster 解包时识别 bundle_type，aux 预先预测全部内生变量轨迹供回填使用。
    """
    pred_method = str(getattr(args, "pred_method", "")).lower()
    backfill = str(getattr(args, "endogenous_backfill_strategy", "persistence")).lower()
    if backfill != "auxiliary":
        return model
    if pred_method not in (
        "multivariate-single-multistep-recursive",
        "multivariate-single-multistep-direct-recursive",
    ):
        return model
    other_endo = [c for c in endogenous_features_with_target if c != target_feature]
    if not other_endo:
        logger.info(f"{log_prefix} auxiliary: no non-target endogenous cols, skip aux bundle.")
        return model

    aux = AuxiliaryEndogenousForecaster(args, other_endo, target_feature, log_prefix=log_prefix)
    aux.fit(df_history)
    if aux.is_empty():
        logger.warning(f"{log_prefix} auxiliary: all aux models failed, keeping raw model (will fall back to persistence).")
        return model
    logger.info(f"{log_prefix} auxiliary: wrapping model into aux bundle (trained cols: {list(aux.models.keys())}).")
    return {"bundle_type": "auxiliary_endogenous", "main": model, "aux": aux}
