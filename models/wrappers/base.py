"""base: estimator wrappers extracted from the model factory."""

import inspect
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

    所有具体模型必须继承此类并实现抽象方法
    """

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

def _filter_valid_params(params: Dict[str, Any], estimator_cls) -> Dict[str, Any]:
    """
    按估计器 ``__init__`` 签名校验模型参数，未知参数直接报错。

    对于通过 ``**kwargs`` 透传原生参数的封装（签名含 VAR_KEYWORD），
    无法用显式签名做白名单，直接原样返回，避免误删合法配置。
    """
    signature = inspect.signature(estimator_cls.__init__)
    # 部分 sklearn 风格封装通过 **kwargs 接收额外原生参数，
    # 这类模型不能用显式签名做白名单过滤，否则会错误丢弃合法配置。
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(params)
    valid = set(signature.parameters.keys())
    valid.discard("self")
    unknown = sorted(set(params) - valid)
    if unknown:
        raise ValueError(f"Unknown {estimator_cls.__name__} parameters: {unknown}")
    return dict(params)

def _filter_fit_params(model, fit_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    按底层估计器 ``fit`` 签名过滤训练参数，兼容不同版本 sklearn API 的参数差异
    （例如 lightgbm >= 4.x 的 fit 不再接受 verbose）。
    """
    supported = set(inspect.signature(model.fit).parameters.keys())
    return {k: v for k, v in fit_params.items() if k in supported}
