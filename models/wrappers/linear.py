"""linear: estimator wrappers extracted from the model factory."""

import copy
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet, Lasso, QuantileRegressor
from utils.log_util import logger
from models.wrappers.base import BaseModel, _filter_valid_params


class _LinearModelBase(BaseModel):
    """
    线性模型封装基类（Ridge / ElasticNet / LASSO / QuantileRegressor）

    四者接口完全一致（fit(X, y, sample_weight)），差异仅在估计器类与默认参数。

    注意:
    - 线性模型对特征量纲敏感，建议开启 scale_features / scale_target
    - 不识别类别特征语义，类别列需先编码（encode_categorical_features）
    - 不支持验证集与早停，相关 fit kwargs 静默忽略
    """

    ESTIMATOR_CLS = None
    DEFAULT_PARAMS: Dict[str, Any] = {}

    def __init__(self, params: Dict[str, Any], log_prefix: str="LinearModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        # 参数合并（用户参数优先，避免被默认值覆盖）
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        # 模型参数
        self.params = _filter_valid_params(merged_params, self.ESTIMATOR_CLS)
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 模型构建
        if self.ESTIMATOR_CLS is None:
            raise NotImplementedError(f"{self.log_prefix} 子类必须声明 ESTIMATOR_CLS")
        self.model = self.ESTIMATOR_CLS(**self.params)

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            sample_weight: Optional[Any] = None,
            **kwargs):
        """
        训练线性模型

        Args:
            X: 训练特征
            y: 训练目标
            sample_weight: 训练样本权重(例如时间衰减权重)
            **kwargs: 为跨模型统一接口而保留（categorical_feature / eval_set /
                early_stopping_rounds 等），线性模型不支持，静默忽略

        注意: 线性估计器不容忍 NaN——fit 丢弃含 NaN 的行（训练窗起始行的长滞后
        特征为 NaN，GBDT 原生容忍、线性模型必须处理），并记录列中位数供
        predict 端填补。
        """
        if hasattr(X, "notna"):
            yv = np.asarray(y, dtype=float).ravel()
            mask = X.notna().all(axis=1).to_numpy() & np.isfinite(yv)
            self._col_medians = X.median().to_numpy(dtype=float)
            self._columns = list(X.columns)
            X_fit = X[mask]
            y_fit = yv[mask]
            sw_fit = np.asarray(sample_weight, dtype=float)[mask] if sample_weight is not None else None
        else:
            X_fit, y_fit, sw_fit = X, y, sample_weight
        self.model.fit(X_fit, y_fit, sample_weight=sw_fit)
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测（NaN 用训练期列中位数填补；正常预测路径无 NaN，属防御性处理）
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

        if hasattr(X, "isna") and getattr(self, "_col_medians", None) is not None and X.isna().any().any():
            X = X.fillna(pd.Series(self._col_medians, index=self._columns))

        return self.model.predict(X)

class RidgeModel(_LinearModelBase):
    """Ridge 岭回归（L2 正则线性基线）"""

    ESTIMATOR_CLS = Ridge
    DEFAULT_PARAMS = {
        "alpha": 1.0,
        "fit_intercept": True,
        "random_state": 42,
    }

class ElasticNetModel(_LinearModelBase):
    """ElasticNet 回归（L1 + L2 混合正则线性基线）"""

    ESTIMATOR_CLS = ElasticNet
    DEFAULT_PARAMS = {
        "alpha": 1.0,
        "l1_ratio": 0.5,
        "max_iter": 2000,
        "random_state": 42,
    }

class LassoModel(_LinearModelBase):
    """LASSO 回归（L1 正则线性基线，带特征选择效果）"""

    ESTIMATOR_CLS = Lasso
    DEFAULT_PARAMS = {
        "alpha": 1.0,
        "max_iter": 2000,
        "random_state": 42,
    }

class QuantileRegressorModel(_LinearModelBase):
    """
    QuantileRegressor 线性分位数回归基线

    默认 quantile=0.5 + alpha=1e-3。alpha 不可取 0：alpha=0 时 LP 解不唯一
    （退化多面体），HiGHS 返回任意顶点，滑窗小样本和强共线特征下容易产生
    巨大系数并在外推期预测爆炸。1e-3 是日频 baseline 六个完整自然月的生产
    消融最优值；sklearn 官方默认 1.0 在该场景过度收缩。分位数预测路径由
    Trainer._inject_quantile_params 注入目标 quantile。
    """

    ESTIMATOR_CLS = QuantileRegressor
    DEFAULT_PARAMS = {
        "quantile": 0.5,
        "alpha": 1e-3,
        "solver": "highs",
    }
