"""模型静态描述唯一入口；不导入训练层或构造任何模型。"""
from dataclasses import dataclass
import math
from types import MappingProxyType


@dataclass(frozen=True)
class ModelDescriptor:
    aliases: tuple[str, ...]
    wrapper: str
    quantile_style: str | None = None
    categorical: bool = False
    nan_support: bool = False
    sample_weight: bool = True
    thread_param: str | None = None
    output_workers: int | None = None
    dataframe_input: bool = False
    native_multi_quantile: bool = False


DESCRIPTORS = (
    ModelDescriptor(("lightgbm", "lgb"), "LightGBMModel", "lightgbm", True, True, True, "n_jobs", 4, True),
    ModelDescriptor(("xgboost", "xgb"), "XGBoostModel", "xgboost", False, True, True, "n_jobs", 4, False, True),
    ModelDescriptor(("catboost", "cat"), "CatBoostModel", "catboost", True, True, True, "thread_count", 2),
    ModelDescriptor(("randomforest", "rf"), "RandomForestModel", thread_param="n_jobs", output_workers=2),
    ModelDescriptor(("histgb", "histgradientboosting"), "HistGBModel", "histgb", True, True),
    ModelDescriptor(("ridge",), "RidgeModel", output_workers=4),
    ModelDescriptor(("elasticnet", "enet"), "ElasticNetModel", output_workers=4),
    ModelDescriptor(("lasso",), "LassoModel", output_workers=4),
    ModelDescriptor(("quantileregressor", "qr"), "QuantileRegressorModel", "qr", output_workers=2),
    ModelDescriptor(("seasonaltemplate", "st"), "SeasonalTemplateModel", sample_weight=False, dataframe_input=True),
)
MODEL_CATALOG = MappingProxyType({alias: descriptor for descriptor in DESCRIPTORS for alias in descriptor.aliases})


def quantile_parameters(model_type: str, params: dict, quantile: float) -> dict:
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be finite and inside (0, 1)")
    descriptor = MODEL_CATALOG.get(model_type.strip().lower())
    if descriptor is None:
        raise ValueError(f"unknown model_type: {model_type!r}")
    style = descriptor.quantile_style
    resolved = dict(params)
    if style == "lightgbm":
        resolved.update(objective="quantile", alpha=quantile)
    elif style == "xgboost":
        resolved.update(objective="reg:quantileerror", quantile_alpha=quantile)
    elif style == "catboost":
        resolved["loss_function"] = f"Quantile:alpha={quantile}"
    elif style == "histgb":
        resolved.update(loss="quantile", quantile=quantile)
    elif style == "qr":
        resolved["quantile"] = quantile
    else:
        raise ValueError(f"model_type {model_type!r} does not declare scalar quantile support")
    return resolved
