"""Catalog-driven estimator construction; implementations live in wrappers/."""

import copy
from typing import Any, Dict, Optional
from models.catalog import MODEL_CATALOG
from models.wrappers.base import BaseModel
from models.wrappers.lightgbm import LightGBMModel
from models.wrappers.xgboost import XGBoostModel
from models.wrappers.catboost import CatBoostModel
from models.wrappers.sklearn_tree import RandomForestModel, HistGBModel
from models.wrappers.linear import RidgeModel, ElasticNetModel, LassoModel, QuantileRegressorModel
from models.wrappers.seasonal_template import SeasonalTemplateModel


class ModelFactory:
    """
    模型工厂 (Model Factory)

    用于创建不同类型的模型实例
    """
    # 支持的模型映射
    _models = {
        alias: globals()[descriptor.wrapper]
        for alias, descriptor in MODEL_CATALOG.items()
    }

    def __init__(self, log_prefix: str = "ModelFactory"):
        self.log_prefix = log_prefix

    @staticmethod
    def _normalize_model_type(model_type: str) -> str:
        model_type = str(model_type).lower()
        if model_type not in ModelFactory._models:
            supported = ", ".join(ModelFactory._models.keys())
            raise ValueError(
                f"不支持的模型类型: {model_type}\n"
                f"支持的模型: {supported}"
            )
        return model_type

    @classmethod
    def get_default_model_params(cls, model_type: str) -> Dict[str, Any]:
        normalized_type = cls._normalize_model_type(model_type)
        model_class = cls._models[normalized_type]

        return copy.deepcopy(getattr(model_class, "DEFAULT_PARAMS", {}))

    @classmethod
    def resolve_model_params(cls, model_type: str, model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        resolved = cls.get_default_model_params(model_type)
        resolved.update(copy.deepcopy(model_params or {}))

        return resolved

    def create_model(self, model_type: str, model_params: Dict[str, Any], log_params: bool = True) -> BaseModel:
        """
        创建模型实例

        Args:
            model_type: 模型类型 ('lightgbm', 'xgboost', 'catboost', 'randomforest')
            model_params: 模型参数字典（缺省参数由各模型封装的 DEFAULT_PARAMS 补齐）
            log_params: 是否记录模型参数日志

        Returns:
            模型实例

        Raises:
            ValueError: 如果模型类型不支持

        Examples:
            >>> params = {'n_estimators': 1000, 'learning_rate': 0.05}
            >>> model = ModelFactory().create_model('lightgbm', params)
            >>> model.fit(X_train, y_train)
            >>> y_pred = model.predict(X_test)
        """
        # 模型类型
        model_type = self._normalize_model_type(model_type)
        # 创建模型实例（默认参数在模型封装内部合并，此处不做重复 merge）
        model_class = ModelFactory._models[model_type]

        return model_class(
            model_params,
            log_prefix=f"{self.log_prefix} {model_type.capitalize()}",
            log_params=log_params,
        )

    @staticmethod
    def list_models() -> list:
        """
        列出所有支持的模型类型
        """
        return list(ModelFactory._models.keys())
