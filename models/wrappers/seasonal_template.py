"""seasonal_template: estimator wrappers extracted from the model factory."""

import copy
import re
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from utils.log_util import logger
from models.wrappers.base import BaseModel


class SeasonalTemplateModel(BaseModel):
    """
    季节模板基线（seasonal template / climatology / 相似日法）

    不学习一般的 特征→目标 映射，而是把「历史周期同一时刻的实测值」
    （特征工程生成的 {target}_lag_{lag} 滞后列）按非负权重重放：

        ŷ = Σ_j w_j · y_lag_j,  w_j >= 0

    - 权重由 NNLS 在训练集上学习（equal_weight=True 时退化为等权 climatology）；
    - day_type_split=True 且存在 dt_day_of_week 列时，工作日/周末各学一组权重；
    - 特例 w=[1,0,...,0] 即 seasonal naive（昨日同时刻，对应 ModelTesting 的
      _build_seasonal_naive 评估对照）。

    误差来源（形态持续性假设：平常日强、节假日/突变日失效）与 GBDT/线性成员
    结构性不同，主要用作模型融合的多样性成员；也可作为单模型基线。
    在 USMR 递归且 horizon <= 最短滞后阶数时，滞后列全程引用实测值，
    模板不会被模型自身预测污染。

    注意:
    - 依赖特征矩阵中的滞后列（自动识别 ^.+_lag_\\d+$，按滞后阶数排序），
      无滞后列时 raise；
    - fit 自动丢弃含 NaN 的行（训练窗起始行的长滞后特征为 NaN）；
    - predict 端 NaN 用训练期滞后列中位数填补（防御性，正常不出现）；
    - 无线程参数；不支持分位数目标（Trainer 注入时走 warning 分支）。
    """

    DEFAULT_PARAMS = {
        "day_type_split": True,   # 工作日/周末分组建模板
        "equal_weight": False,    # True=等权 climatology；False=NNLS 学权重
        "min_group_samples": 10,  # 分组样本不足该数时回退全局权重
    }

    LAG_COL_PATTERN = r"^.+_lag_\d+$"
    DOW_COL = "dt_day_of_week"

    def __init__(self, params: Dict[str, Any], log_prefix: str="SeasonalTemplateModel", log_params: bool = True):
        super().__init__(params, log_prefix=log_prefix, log_params=log_params)
        # 参数合并（用户参数优先，避免被默认值覆盖）
        merged_params = {**copy.deepcopy(self.DEFAULT_PARAMS), **(params or {})}
        self.params = merged_params
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        # 无底层估计器：非负模板权重即模型
        self.model = None
        self.lag_columns_ = None
        self.weights_ = None
        self.group_weights_ = None
        self.lag_medians_ = None

    def _detect_lag_columns(self, X: pd.DataFrame) -> list:
        """自动识别滞后列并按滞后阶数排序，保证 fit/predict 列序一致"""
        cols = [c for c in X.columns if re.match(self.LAG_COL_PATTERN, str(c))]
        cols.sort(key=lambda c: int(str(c).rsplit("_lag_", 1)[1]))
        return cols

    def _fit_weights(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        """NNLS 学非负模板权重；equal_weight=True 时等权；数值异常时退化等权"""
        n_lags = Z.shape[1]
        if self.params.get("equal_weight"):
            return np.full(n_lags, 1.0 / n_lags)
        w, _ = nnls(Z, y)
        if not np.isfinite(w).all() or w.sum() <= 1e-12:
            logger.warning(f"{self.log_prefix} NNLS 权重异常，退化为等权模板。")
            w = np.full(n_lags, 1.0 / n_lags)
        return w

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            **kwargs):
        """
        训练季节模板基线

        Args:
            X: 训练特征（DataFrame，需含 {target}_lag_{lag} 滞后列）
            y: 训练目标
            **kwargs: 为跨模型统一接口而保留（sample_weight / eval_set 等），静默忽略
        """
        if not hasattr(X, "columns"):
            raise ValueError(f"{self.log_prefix} 需要 DataFrame 输入以识别滞后列，当前为 {type(X).__name__}")
        lag_cols = self._detect_lag_columns(X)
        if not lag_cols:
            raise ValueError(
                f"{self.log_prefix} 特征中未找到滞后列（模式 {self.LAG_COL_PATTERN}），"
                f"无法构建季节模板。"
            )
        self.lag_columns_ = lag_cols
        Z_df = X[lag_cols]
        yv = np.asarray(y, dtype=float).ravel()
        # fit 端丢弃含 NaN 的行（训练窗起始行的长滞后特征为 NaN）
        mask = Z_df.notna().all(axis=1).to_numpy() & np.isfinite(yv)
        Z_c = Z_df.to_numpy(dtype=float)[mask]
        y_c = yv[mask]
        if len(y_c) == 0:
            raise ValueError(f"{self.log_prefix} 滞后列全部含 NaN，无有效训练行。")
        # predict 端 NaN 填补用的训练期中位数
        self.lag_medians_ = Z_df.median().to_numpy(dtype=float)
        # 全局权重
        self.weights_ = self._fit_weights(Z_c, y_c)
        if self.log_params:
            logger.info(
                f"{self.log_prefix} template weights (global): "
                f"{np.round(self.weights_, 4)}"
            )
        # 工作日/周末分组权重（两组样本都足够才启用，否则回退全局）
        self.group_weights_ = None
        if self.params.get("day_type_split") and self.DOW_COL in X.columns:
            dow = X[self.DOW_COL].to_numpy()[mask]
            min_n = int(self.params.get("min_group_samples", 10))
            group_weights = {}
            for group_name, group_mask in [("weekday", dow <= 4), ("weekend", dow >= 5)]:
                if int(group_mask.sum()) >= min_n:
                    group_weights[group_name] = self._fit_weights(Z_c[group_mask], y_c[group_mask])
            if len(group_weights) == 2:
                self.group_weights_ = group_weights
                if self.log_params:
                    logger.info(
                        f"{self.log_prefix} day-type split enabled: "
                        f"weekday={np.round(group_weights['weekday'], 4)}, "
                        f"weekend={np.round(group_weights['weekend'], 4)}"
                    )
        self.is_fitted = True

        return self

    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测
        """
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")
        assert self.lag_columns_ is not None and self.weights_ is not None
        assert self.lag_medians_ is not None
        Z = X[self.lag_columns_].to_numpy(dtype=float)
        # 防御性 NaN 填补（horizon <= 最短滞后时预测端 lag 全为实测，不会出现）
        if np.isnan(Z).any():
            nan_rows, nan_cols = np.where(np.isnan(Z))
            Z[nan_rows, nan_cols] = self.lag_medians_[nan_cols]
        if self.group_weights_ is not None and self.DOW_COL in X.columns:
            dow = X[self.DOW_COL].to_numpy()
            is_weekday = dow <= 4
            out = np.empty(len(Z), dtype=float)
            out[is_weekday] = Z[is_weekday] @ self.group_weights_["weekday"]
            out[~is_weekday] = Z[~is_weekday] @ self.group_weights_["weekend"]
            return out

        return Z @ self.weights_

    def get_feature_importance(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """模板权重即各滞后列的重要性"""
        if self.weights_ is None or self.lag_columns_ is None:
            return None
        lines = [f"{name}: {w:.4f}" for name, w in zip(self.lag_columns_, self.weights_)]
        logger.info(f"{self.log_prefix} template weights:\n" + "\n".join(lines))

        return self.weights_
