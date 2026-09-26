"""引擎参数预检收口：未知参数 RAISE 合同的统一执行层。

各引擎的原生参数表面探测（签名白名单 / 别名全集 / 子进程试探 /
同义词归一化）统一在本包实现，wrappers 只保留训练/预测本体。
第三方私有 API 依赖清单见 docs/packages/models.md。

门面约定：包根只导出跨引擎通用的 signature 一对（纯标准库，零引擎依赖）；
引擎专属函数一律全路径导入（如 ``models.preflight.lightgbm.validate_lgbm_params``），
调用点自证引擎归属，且不强制无关节点传递加载引擎重依赖。
"""

from models.preflight.signature import filter_fit_params, filter_valid_params

__all__ = [
    "filter_fit_params",
    "filter_valid_params",
]
