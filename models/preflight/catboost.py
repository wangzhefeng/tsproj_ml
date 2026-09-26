"""CatBoost 同义参数归一化（进程内薄封装）。

私有 API：``catboost.core._process_synonyms``——把 ``iterations``/``n_estimators``
等同义参数归一化为规范名。升级 catboost 时必须复核（见 docs/packages/models.md
私有 API 依赖清单）。
"""

from typing import Any, Dict

import catboost as cab


def process_synonym_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """原地归一化同义参数，返回同一字典（与既有调用语义一致）。"""
    cab.core._process_synonyms(params)
    return params
