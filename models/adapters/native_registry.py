"""原生序列模型注册表：ets / naive / theta 的统一接线。

catalog 声明 ``native_history=True`` 的成员必须在此注册，漏接线 RAISE
（防「descriptor 声明了能力但无实现」的静默缺口）。成员合同：
``fit_history(history, as_of, freq) / forecast(steps) / execution_evidence()``
三件套，见 models/wrappers/ets.py 等。

原 ``model_pipeline/runner.py::NATIVE_HISTORY_MODELS``（2026-09-26 建
注册表时误置于编排层，本文件为其模型层归位；runner 4 处分发点改为
import 此处）。
"""

from models.catalog import MODEL_CATALOG
from models.wrappers.ets import ETSModel
from models.wrappers.naive import NaiveModel
from models.wrappers.theta import ThetaModel

NATIVE_HISTORY_MODELS: dict[str, type] = {
    "ets": ETSModel,
    "naive": NaiveModel,
    "theta": ThetaModel,
}


def native_history_cls(model_type: str) -> type | None:
    """model_type 是否为原生序列模型；别名归一化与 catalog 对齐。"""
    normalized = str(model_type).strip().lower()
    descriptor = MODEL_CATALOG.get(normalized)
    if descriptor is not None and descriptor.native_history:
        if normalized not in NATIVE_HISTORY_MODELS:
            raise ValueError(
                f"native_history model {normalized!r} is not wired in "
                "models.adapters.native_registry.NATIVE_HISTORY_MODELS"
            )
        return NATIVE_HISTORY_MODELS[normalized]
    return None
