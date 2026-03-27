# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelEnsemble.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-27
# * Version     : 2.0.032700
# * Description : 时间序列回归模型融合模块
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error

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
    parallel_workers: int = 1
    stacking_alphas: Tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


class TimeSeriesEnsembleRegressor:
    """
    时间序列回归融合器。

    支持的常用融合方法:
    - averaging: 简单平均
    - weighted: 基于验证集 MAE 的加权平均
    - blending: 基于验证集最小二乘的非负加权融合
    - stacking: 基于验证集预测的二层 RidgeCV 融合
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
        if self.config.parallel_workers <= 1 or len(self.base_models) <= 1:
            for name, model in self.base_models:
                logger.info(f"[Ensemble] Training base model: {name}")
                model.fit(X_train, y_train)
            return

        def _fit_one(name, model):
            logger.info(f"[Ensemble] Training base model: {name}")
            model.fit(X_train, y_train)
            return name, model

        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            futures = [executor.submit(_fit_one, name, model) for name, model in self.base_models]
            fitted_models = []
            for future in as_completed(futures):
                fitted_models.append(future.result())
        fitted_map = {name: model for name, model in fitted_models}
        self.base_models = [(name, fitted_map[name]) for name, _ in self.base_models]

    def _collect_predictions(self, X) -> np.ndarray:
        preds = []
        for _, model in self.base_models:
            preds.append(_to_2d(model.predict(X)))
        # shape: (n_models, n_samples, n_outputs)
        return np.stack(preds, axis=0)

    def _fit_weighted(self, val_preds: np.ndarray, y_val_2d: np.ndarray):
        maes = []
        for i in range(val_preds.shape[0]):
            maes.append(mean_absolute_error(y_val_2d, val_preds[i]))
        maes = np.asarray(maes, dtype=float)
        inv = 1.0 / np.clip(maes, 1e-8, None)
        self.weights_ = inv / inv.sum()
        logger.info(f"[Ensemble] Weighted averaging weights: {self.weights_}")

    def _fit_blending(self, val_preds: np.ndarray, y_val_2d: np.ndarray):
        n_models = val_preds.shape[0]
        Z = val_preds.transpose(1, 0, 2).reshape(-1, n_models)
        y = y_val_2d.reshape(-1)
        w, *_ = np.linalg.lstsq(Z, y, rcond=None)
        w = np.clip(w, 0.0, None)
        if w.sum() <= 1e-8:
            w = np.ones(n_models, dtype=float) / n_models
        else:
            w = w / w.sum()
        self.weights_ = w
        logger.info(f"[Ensemble] Blending weights: {self.weights_}")

    def _fit_stacking(self, val_preds: np.ndarray, y_val_2d: np.ndarray):
        n_models, n_samples, n_outputs = val_preds.shape
        X_meta = val_preds.transpose(1, 0, 2).reshape(n_samples, n_models * n_outputs)
        meta = RidgeCV(alphas=self.config.stacking_alphas)
        meta.fit(X_meta, y_val_2d.ravel() if self._single_output else y_val_2d)
        self.meta_model_ = meta
        logger.info("[Ensemble] Stacking meta-model fitted with RidgeCV.")

    def fit(self, X, y):
        y_2d = _to_2d(y)
        self._single_output = y_2d.shape[1] == 1

        X_train, y_train, X_val, y_val = self._split_train_val(X, y_2d)
        self._fit_base_models(X_train, y_train.ravel() if self._single_output else y_train)

        method = str(self.config.method).lower()
        needs_val = method in {"weighted", "blending", "stacking"}
        if needs_val and X_val is not None and len(X_val) > 0:
            val_preds = self._collect_predictions(X_val)
            if method == "weighted":
                self._fit_weighted(val_preds, y_val)
            elif method == "blending":
                self._fit_blending(val_preds, y_val)
            elif method == "stacking":
                self._fit_stacking(val_preds, y_val)
            self._fit_base_models(X, y_2d.ravel() if self._single_output else y_2d)
        elif needs_val:
            logger.warning("[Ensemble] Validation split unavailable, fallback to averaging.")
            self.config.method = "averaging"

        return self

    def predict(self, X) -> np.ndarray:
        method = str(self.config.method).lower()
        preds = self._collect_predictions(X)

        if method == "averaging":
            out = preds.mean(axis=0)
        elif method in {"weighted", "blending"}:
            w = self.weights_
            if w is None:
                w = np.ones(preds.shape[0], dtype=float) / preds.shape[0]
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




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
