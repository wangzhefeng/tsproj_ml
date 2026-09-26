"""LightGBM 原生参数校验：别名全集 introspection（进程内，无子进程）。"""

from typing import Any, Dict

import lightgbm as lgb


def validate_lgbm_params(params: Dict[str, Any]) -> None:
    """
    按已安装 LightGBM 原生参数及别名表做严格参数名校验。

    LGBMRegressor 通过 ``**kwargs`` 透传原生 booster 参数，签名白名单对其无效，
    拼写错误的参数会被静默忽略。合法名全集 = sklearn 封装显式参数 +
    原生参数及其别名（``_ConfigAliases``）。内省失败必须显式报错。
    """
    try:
        import inspect

        explicit = {p for p in inspect.signature(lgb.LGBMRegressor.__init__).parameters if p != "self"}
        alias_map = lgb.basic._ConfigAliases._get_all_param_aliases()
        valid = explicit | set(alias_map) | {a for aliases in alias_map.values() for a in aliases}
    except Exception as exc:
        raise RuntimeError("LightGBM parameter validation is unavailable") from exc
    unknown = sorted(k for k in params if k not in valid)
    if unknown:
        raise ValueError(f"Unknown LightGBM parameters: {unknown}")
