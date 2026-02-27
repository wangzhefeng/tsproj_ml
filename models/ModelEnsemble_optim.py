# -*- coding: utf-8 -*-

"""Time-series friendly ensemble models for regression."""
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

import lightgbm as lgb

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _to_2d(arr: Any) -> np.ndarray:
    pred = np.asarray(arr)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    return pred


@dataclass
class EnsembleConfig:
    method: str = "averaging"
    val_ratio: float = 0.2
    random_state: int = 42


class TimeSeriesEnsembleRegressor:
    """
    Ensemble model for time-series regression.

    Supported methods:
    - averaging: simple mean of base models
    - weighted: weighted average (weights from validation MAE)
    - blending: linear blend on validation split
    - stacking: meta-model on base predictions
    """

    def __init__(self, base_models: Sequence[Tuple[str, Any]], config: EnsembleConfig):
        self.base_models = list(base_models)
        self.config = config
        self.weights_: np.ndarray | None = None
        self.meta_model_: Any | None = None
        self._single_output = True

    def _split_train_val(self, X, y):
        n = len(X)
        if n < 10:
            return X, y, None, None
        val_ratio = min(max(self.config.val_ratio, 0.05), 0.4)
        split_idx = int(n * (1.0 - val_ratio))
        split_idx = min(max(split_idx, 1), n - 1)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    def _fit_base_models(self, X_train, y_train):
        for name, model in self.base_models:
            logger.info(f"[Ensemble] Training base model: {name}")
            model.fit(X_train, y_train)

    def _collect_predictions(self, X) -> np.ndarray:
        preds = []
        for _, model in self.base_models:
            pred = _to_2d(model.predict(X))
            preds.append(pred)
        # shape: (n_models, n_samples, n_outputs)
        return np.stack(preds, axis=0)

    def _fit_weighted(self, val_preds: np.ndarray, y_val_2d: np.ndarray):
        # MAE per model on validation split.
        maes = []
        for i in range(val_preds.shape[0]):
            maes.append(mean_absolute_error(y_val_2d, val_preds[i]))
        maes = np.asarray(maes)
        inv = 1.0 / np.clip(maes, 1e-8, None)
        self.weights_ = inv / inv.sum()
        logger.info(f"[Ensemble] Weighted averaging weights: {self.weights_}")

    def _fit_blending(self, val_preds: np.ndarray, y_val_2d: np.ndarray):
        # Solve linear blend w >= 0, sum(w)=1 by least squares + projection.
        n_models = val_preds.shape[0]
        Z = val_preds.transpose(1, 0, 2).reshape(-1, n_models)  # (n_samples*n_out, n_models)
        y = y_val_2d.reshape(-1)
        w, *_ = np.linalg.lstsq(Z, y, rcond=None)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 1e-8:
            w = np.ones(n_models) / n_models
        else:
            w = w / w.sum()
        self.weights_ = w
        logger.info(f"[Ensemble] Blending weights: {self.weights_}")

    def _fit_stacking(self, val_preds: np.ndarray, y_val_2d: np.ndarray):
        # Meta features: concat all base predictions.
        n_models, n_samples, n_outputs = val_preds.shape
        X_meta = val_preds.transpose(1, 0, 2).reshape(n_samples, n_models * n_outputs)
        if n_outputs == 1:
            meta = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=self.config.random_state,
                verbose=-1,
            )
            meta.fit(X_meta, y_val_2d.ravel())
        else:
            meta_base = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=self.config.random_state,
                verbose=-1,
            )
            meta = MultiOutputRegressor(meta_base)
            meta.fit(X_meta, y_val_2d)
        self.meta_model_ = meta
        logger.info("[Ensemble] Stacking meta-model fitted.")

    def fit(self, X, y):
        y_2d = _to_2d(y)
        self._single_output = y_2d.shape[1] == 1

        X_train, y_train, X_val, y_val = self._split_train_val(X, y_2d)
        self._fit_base_models(X_train, y_train.ravel() if self._single_output else y_train)

        method = self.config.method.lower()
        needs_val = method in {"weighted", "blending", "stacking"}
        if needs_val and X_val is not None and len(X_val) > 0:
            val_preds = self._collect_predictions(X_val)
            if method == "weighted":
                self._fit_weighted(val_preds, y_val)
            elif method == "blending":
                self._fit_blending(val_preds, y_val)
            elif method == "stacking":
                self._fit_stacking(val_preds, y_val)
            # Refit base models on full training after estimating ensemble params.
            self._fit_base_models(X, y_2d.ravel() if self._single_output else y_2d)
        elif needs_val:
            logger.warning("[Ensemble] Validation split unavailable, fallback to averaging.")
            self.config.method = "averaging"

        return self

    def predict(self, X) -> np.ndarray:
        method = self.config.method.lower()
        preds = self._collect_predictions(X)  # (n_models, n_samples, n_outputs)

        if method == "averaging":
            out = preds.mean(axis=0)
        elif method in {"weighted", "blending"}:
            w = self.weights_
            if w is None:
                w = np.ones(preds.shape[0]) / preds.shape[0]
            out = np.tensordot(w, preds, axes=(0, 0))
        elif method == "stacking":
            if self.meta_model_ is None:
                out = preds.mean(axis=0)
            else:
                n_models, n_samples, n_outputs = preds.shape
                X_meta = preds.transpose(1, 0, 2).reshape(n_samples, n_models * n_outputs)
                out = _to_2d(self.meta_model_.predict(X_meta))
        else:
            raise ValueError(f"Unsupported ensemble method: {self.config.method}")

        if self._single_output:
            return out.ravel()
        return out
