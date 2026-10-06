"""训练选项的纯配置合同；权重数值计算仍属于 model_training。"""
from collections.abc import Mapping
import math


SAMPLE_WEIGHT_FIELDS = frozenset({"method", "halflife_days", "anchor", "normalization"})


def resolve_sample_weight_spec(spec: Mapping) -> tuple[float, str, str]:
    """返回半衰期/可得性锚点/归一化；不写回默认值或改变配置身份。"""
    if not isinstance(spec, Mapping) or set(spec) - SAMPLE_WEIGHT_FIELDS:
        raise ValueError("sample_weight requires method/halflife_days with optional anchor/normalization only")
    half = spec.get("halflife_days")
    if spec.get("method", "exponential") != "exponential" or isinstance(half, bool) or not isinstance(half, (int, float)):
        raise ValueError("sample_weight requires exponential method and numeric halflife_days")
    if not math.isfinite(half) or half <= 0:
        raise ValueError("sample_weight halflife_days must be finite and positive")
    anchor = spec.get("anchor", "cutoff")
    if anchor not in {"cutoff", "latest_origin"}:
        raise ValueError("sample_weight anchor must be cutoff or latest_origin")
    normalization = spec.get("normalization", "mean")
    if normalization not in {"mean", "sum", "none"}:
        raise ValueError("sample_weight normalization must be mean, sum or none")
    return float(half), anchor, normalization
