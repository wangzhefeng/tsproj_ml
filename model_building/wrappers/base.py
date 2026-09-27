"""base: estimator wrappers extracted from the model factory."""

import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from utils.log_util import logger

# 各模型 fit 接口统一的早停轮数默认值
DEFAULT_EARLY_STOPPING_ROUNDS = 50


class BaseModel(ABC):
    """
    模型基类 (Base Model Class)

    所有具体模型必须继承此类并实现抽象方法。

    构造走统一模板（2026-09-26 模板收敛）：
    1. ``_resolve_params(supplied)``：DEFAULT_PARAMS 深拷贝合并用户参数，
       子类覆写以插入家族特定校验（别名全集 / 签名白名单 / synonym 归一化）；
    2. 参数日志；
    3. ``_build_estimator()``：构造底层估计器（或无估计器成员的拟合状态）。

    实例属性 ``params`` / ``model`` / ``is_fitted`` 与历史行为逐项一致，
    pickle 路径与属性名不变，存量 bundle 不受影响。
    """

    DEFAULT_PARAMS: Dict[str, Any] = {}

    def __init__(self, params: Dict[str, Any], log_prefix: str="BaseModel", log_params: bool = True):
        """
        初始化模型

        Args:
            params (Dict[str, Any]): 模型参数字典
        """
        self.log_prefix = log_prefix
        self.params = params
        self.model = None
        self.is_fitted = False
        self.log_params = log_params
        # 模板：合并默认参数（用户参数优先）→ 家族校验钩子 → 构造钩子
        self.params = self._resolve_params(dict(params or {}))
        if self.log_params:
            logger.info(f"{log_prefix} model parameters: \n{self.params}")
        self.model = self._build_estimator()

    def _resolve_params(self, supplied: Dict[str, Any]) -> Dict[str, Any]:
        """默认参数合并（用户参数优先）；子类覆写以插入家族特定校验。"""
        return {**copy.deepcopy(self.DEFAULT_PARAMS), **copy.deepcopy(supplied)}

    def _build_estimator(self) -> Any:
        """构造底层估计器；默认无估计器（由子类覆写）。"""
        return None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "BaseModel":
        """
        训练模型

        Args:
            X: 特征数据
            y: 目标数据
            **kwargs: 其他参数（如验证集、类别特征等）

        Returns:
            模型实例自身（支持链式调用）
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测

        Args:
            X: 特征数据
            **kwargs: 其他参数

        Returns:
            预测结果
        """
        pass

    def _require_fitted(self) -> None:
        """predict 前置检查：未训练直接报错（各封装共用，消息逐字一致）"""
        if not self.is_fitted:
            raise ValueError(f"{self.log_prefix} 模型尚未训练(Model not fitted yet).")

    def get_params(self) -> Dict[str, Any]:
        """获取模型参数"""
        return self.params

    def get_feature_importance(self, X: Optional[pd.DataFrame] = None) -> Optional[np.ndarray]:
        """
        获取特征重要性

        Args:
            X: 特征数据（可选，仅用于取列名；非 DataFrame 时回退为 feature_{idx}）

        Returns:
            特征重要性数组，如果模型不支持则返回 None
        """
        importance = getattr(self.model, "feature_importances_", None)
        if importance is None:
            return None
        importance = np.asarray(importance)
        columns = getattr(X, "columns", None)
        top_features = np.argsort(importance)[-5:][::-1]
        lines = []
        for rank, idx in enumerate(top_features, 1):
            if columns is not None and idx < len(columns):
                name = str(columns[idx])
            else:
                name = f"feature_{idx}"
            lines.append(f"{rank}. {name}: {importance[idx]:.4f}")
        logger.info(f"{self.log_prefix} top-5 important features:\n" + "\n".join(lines))

        return importance


def nan_defense_fit_state(
    X: pd.DataFrame,
    y,
    columns: Optional[list] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    fit 端 NaN 防御共享实现（线性族 / 季节模板共用）。

    在 ``columns`` 指定列集（缺省全列）上构建有效行掩码
    （列集全 notna 且 y 有限）并记录列集中位数，供 predict 端
    防御性填补。正常预测路径无 NaN，填补仅防御性兜底。

    Returns:
        (mask, yv, medians)：有效行布尔掩码、展平的 float 目标、
        列集中位数数组（顺序与 ``columns`` / X 列序一致）
    """
    frame = X if columns is None else X[columns]
    if not hasattr(frame, "notna"):
        raise TypeError("nan_defense_fit_state requires a pandas DataFrame input")
    yv = np.asarray(y, dtype=float).ravel()
    mask = frame.notna().all(axis=1).to_numpy() & np.isfinite(yv)
    medians = frame.median().to_numpy(dtype=float)
    return mask, yv, medians
