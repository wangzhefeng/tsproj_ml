# -*- coding: utf-8 -*-

# ***************************************************
# * File        : DataAugment.py
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

from utils.log_util import logger


class TimeSeriesAugmenter:
    """
    面向表格时间序列特征的轻量增强器（训练阶段使用）
    """

    def __init__(
        self,
        enabled: bool = False,
        augmentation_ratio: float = 0.2,
        feature_noise_std: float = 0.01,
        target_noise_std: float = 0.005,
        random_state: int = 42,
        log_prefix: str = "[DataAugment]",
    ):
        self.enabled = enabled
        self.augmentation_ratio = max(0.0, float(augmentation_ratio))
        self.feature_noise_std = max(0.0, float(feature_noise_std))
        self.target_noise_std = max(0.0, float(target_noise_std))
        self.rng = np.random.default_rng(random_state)
        self.log_prefix = log_prefix

    def _numeric_cols(self, df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
        exclude_set = set(exclude or [])
        return [c for c in df.columns if c not in exclude_set and pd.api.types.is_numeric_dtype(df[c])]

    def augment(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        categorical_features: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        在训练数据上做 bootstrap + numeric jitter，返回增广后的 (X, y)
        """
        if not self.enabled or self.augmentation_ratio <= 0 or len(X) < 10:
            return X, y

        categorical_features = categorical_features or []
        n_samples = len(X)
        n_aug = int(n_samples * self.augmentation_ratio)
        if n_aug <= 0:
            return X, y

        idx = self.rng.choice(n_samples, size=n_aug, replace=True)
        X_aug = X.iloc[idx].copy().reset_index(drop=True)
        y_aug = y.iloc[idx].copy().reset_index(drop=True)

        # 仅对数值特征加噪，类别特征保持不变
        numeric_cols = self._numeric_cols(X_aug, exclude=categorical_features)
        for col in numeric_cols:
            col_std = float(X[col].std()) if col in X.columns else 0.0
            if not np.isfinite(col_std) or col_std == 0.0:
                continue
            noise = self.rng.normal(0.0, self.feature_noise_std * col_std, size=n_aug)
            X_aug[col] = X_aug[col].astype(float) + noise

        # 目标做更小幅度噪声扰动，增强鲁棒性
        for col in y_aug.columns:
            if not pd.api.types.is_numeric_dtype(y_aug[col]):
                continue
            col_std = float(y[col].std()) if col in y.columns else 0.0
            if not np.isfinite(col_std) or col_std == 0.0:
                continue
            noise = self.rng.normal(0.0, self.target_noise_std * col_std, size=n_aug)
            y_aug[col] = y_aug[col].astype(float) + noise

        X_out = pd.concat([X.reset_index(drop=True), X_aug], axis=0, ignore_index=True)
        y_out = pd.concat([y.reset_index(drop=True), y_aug], axis=0, ignore_index=True)
        logger.info(
            f"{self.log_prefix} Data augmentation enabled: {n_samples} -> {len(X_out)} "
            f"(+{n_aug}, ratio={self.augmentation_ratio:.2f})"
        )

        return X_out, y_out


def augment_time_series(df, target_feature, noise_level=0.01):
    """
    向后兼容的简单增强函数（仅对目标列加噪）
    """
    if target_feature not in df.columns:
        return df
    df_augmented = df.copy()
    noise = np.random.normal(0, noise_level, len(df))
    df_augmented[target_feature] = df[target_feature] + df[target_feature] * noise

    return df_augmented




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
