# -*- coding: utf-8 -*-
"""九个外部预测方法的稳定身份与内部正交语义。"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple


class InputScope(str, Enum):
    TARGET_ONLY = "target_only"
    ALL_ENDOGENOUS = "all_endogenous"


class RolloutFamily(str, Enum):
    POINTWISE = "pointwise"
    DIRECT = "direct"
    RECURSIVE = "recursive"
    DIRREC = "dirrec"
    BLEND = "blend"


@dataclass(frozen=True)
class StrategySpec:
    method: str
    code: str
    description: str
    input_scope: InputScope
    rollout: RolloutFamily


_SPECS: Tuple[StrategySpec, ...] = (
    StrategySpec(
        "univariate-single-multistep-direct-pointwise",
        "usmdp",
        "单变量输入，按未来时点逐点 direct 预测",
        InputScope.TARGET_ONLY,
        RolloutFamily.POINTWISE,
    ),
    StrategySpec(
        "univariate-single-multistep-direct",
        "usmd",
        "单变量输入，多步直接预测",
        InputScope.TARGET_ONLY,
        RolloutFamily.DIRECT,
    ),
    StrategySpec(
        "univariate-single-multistep-recursive",
        "usmr",
        "单变量输入，多步递归预测",
        InputScope.TARGET_ONLY,
        RolloutFamily.RECURSIVE,
    ),
    StrategySpec(
        "univariate-single-multistep-direct-recursive",
        "usmdr",
        "单变量输入，多步直接递归预测",
        InputScope.TARGET_ONLY,
        RolloutFamily.DIRREC,
    ),
    StrategySpec(
        "multivariate-single-multistep-direct",
        "msmd",
        "多变量输入，多步直接预测",
        InputScope.ALL_ENDOGENOUS,
        RolloutFamily.DIRECT,
    ),
    StrategySpec(
        "multivariate-single-multistep-recursive",
        "msmr",
        "多变量输入，多步递归预测",
        InputScope.ALL_ENDOGENOUS,
        RolloutFamily.RECURSIVE,
    ),
    StrategySpec(
        "multivariate-single-multistep-direct-recursive",
        "msmdr",
        "多变量输入，多步直接递归预测",
        InputScope.ALL_ENDOGENOUS,
        RolloutFamily.DIRREC,
    ),
    StrategySpec(
        "univariate-single-multistep-blend-direct-recursive",
        "usbr",
        "单变量输入，Direct+Recursive 加权融合",
        InputScope.TARGET_ONLY,
        RolloutFamily.BLEND,
    ),
    StrategySpec(
        "multivariate-single-multistep-blend-direct-recursive",
        "msbr",
        "多变量输入，Direct+Recursive 加权融合",
        InputScope.ALL_ENDOGENOUS,
        RolloutFamily.BLEND,
    ),
)

STRATEGY_SPECS: Dict[str, StrategySpec] = {spec.method: spec for spec in _SPECS}
STRATEGY_ALIASES: Dict[str, StrategySpec] = {
    alias: spec
    for spec in _SPECS
    for alias in (spec.method, spec.code)
}


def get_strategy_spec(method_or_code: str) -> StrategySpec:
    key = str(method_or_code or "").strip().lower()
    try:
        return STRATEGY_ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(spec.code for spec in _SPECS)
        raise ValueError(
            f"Unsupported pred_method={method_or_code!r}; supported methods/codes: {supported}."
        ) from exc
