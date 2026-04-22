# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureSelection.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021111
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

from utils.log_util import logger


class FeatureSelector:
    """
    训练阶段特征选择器（fit on train, transform on inference）
    """

    def __init__(
        self,
        enabled: bool = False,
        method: str = "f_regression",
        max_features: int = 80,
        min_features: int = 10,
        force_keep_features: Optional[List[str]] = None,
        log_prefix: str = "[FeatureSelection]",
    ):
        self.enabled = enabled
        self.method = method
        self.max_features = int(max_features)
        self.min_features = int(min_features)
        self.force_keep_features = force_keep_features or []
        self.log_prefix = log_prefix
        self.selected_features_: Optional[List[str]] = None

    def _y_for_selection(self, y: pd.DataFrame) -> np.ndarray:
        if isinstance(y, pd.Series):
            return y.values
        if y.shape[1] == 1:
            return y.iloc[:, 0].values
        # 多输出场景使用各 horizon 的行均值作为筛选信号，避免只偏向第一个预测步。
        return y.mean(axis=1).values

    def _prepare_numeric(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X.copy()
        for col in X_num.columns:
            if pd.api.types.is_object_dtype(X_num[col]) or pd.api.types.is_categorical_dtype(X_num[col]):
                X_num[col] = X_num[col].astype("category").cat.codes
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return X_num

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        if not self.enabled:
            self.selected_features_ = list(X.columns)
            return X, self.selected_features_

        categorical_features = categorical_features or []
        n_features = X.shape[1]
        if n_features <= self.min_features:
            self.selected_features_ = list(X.columns)
            return X, self.selected_features_

        k = min(self.max_features, n_features)
        k = max(k, self.min_features)
        k = min(k, n_features)

        X_num = self._prepare_numeric(X)
        y_vec = self._y_for_selection(y)
        score_func = mutual_info_regression if self.method == "mutual_info" else f_regression
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X_num, y_vec)

        support = selector.get_support()
        selected = [col for col, keep in zip(X.columns, support) if keep]
        for col in self.force_keep_features + categorical_features:
            if col in X.columns and col not in selected:
                selected.append(col)

        if len(selected) < self.min_features:
            remain = [c for c in X.columns if c not in selected]
            selected.extend(remain[: max(0, self.min_features - len(selected))])

        self.selected_features_ = selected
        logger.info(
            f"{self.log_prefix} Feature selection enabled: {n_features} -> {len(self.selected_features_)} "
            f"(method={self.method})"
        )
        return X[self.selected_features_], self.selected_features_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_features_:
            return X
        cols = [c for c in self.selected_features_ if c in X.columns]
        if not cols:
            return X
        return X[cols]


# ------------------------------
# 向后兼容函数
# ------------------------------
def feature_selection(X_train, y_train, k: int = 50):
    selector = SelectKBest(score_func=f_regression, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    selected_cols = [col for col, keep in zip(X_train.columns, selector.get_support()) if keep]
    return X_train[selected_cols]


def feature_importance_analysis(model, X_train, y_train, top_k: int = 50):
    if not hasattr(model, "feature_importances_"):
        return X_train
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": X_train.columns, "importance": feature_importance}
    ).sort_values("importance", ascending=False)
    top_features = importance_df.head(min(top_k, len(importance_df)))["feature"].tolist()
    return X_train[top_features]




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
