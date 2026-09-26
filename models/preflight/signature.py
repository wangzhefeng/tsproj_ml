"""通用签名级参数校验：估计器构造签名白名单 + fit 签名版本过滤。"""

import inspect
from typing import Any, Dict


def filter_valid_params(params: Dict[str, Any], estimator_cls) -> Dict[str, Any]:
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


def filter_fit_params(model, fit_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    按底层估计器 ``fit`` 签名过滤训练参数，兼容不同版本 sklearn API 的参数差异
    （例如 lightgbm >= 4.x 的 fit 不再接受 verbose）。
    """
    supported = set(inspect.signature(model.fit).parameters.keys())
    return {k: v for k, v in fit_params.items() if k in supported}
