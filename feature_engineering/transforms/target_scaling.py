"""目标变换职责拆分；旧持久化类路径不提供兼容别名。"""
from __future__ import annotations
from typing import Optional, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from utils.log_util import logger


class TargetScaler:
    """
    目标变量缩放器。

    与 FeatureScaler 分离，专门处理训练目标 Y 的缩放与逆变换。
    """

    def __init__(self, scaler_type="standard", log_prefix: str="[TargetScaler]", verbose: bool = False):
        self.log_prefix = log_prefix
        self.verbose = verbose
        # target scaling 默认启用（canonical 配置无旧版 scale 开关残留）
        self.scaler_type = str(scaler_type).lower()
        self.column_transformers = {}
        self.column_names = []
        self.is_fitted = False

    def _resolve_columns(self, columns: Optional[List[str]] = None) -> List[str]:
        if columns is not None:
            return list(columns)
        return list(self.column_names)

    def _validate_columns(self, columns: List[str]):
        if not self.column_names:
            raise ValueError(f"{self.log_prefix} Target scaler has not been fitted yet.")
        missing = [col for col in columns if col not in self.column_names]
        if missing:
            raise ValueError(f"{self.log_prefix} Unknown target columns for scaling: {missing}")

    def _create_column_transformer(self):
        if self.scaler_type == "none":
            return None
        if self.scaler_type == "standard":
            return StandardScaler()
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        if self.scaler_type == "robust":
            return RobustScaler()
        raise ValueError(
            f"{self.log_prefix} Unsupported target_scaler_type={self.scaler_type}. "
            f"Supported: none, standard, minmax, robust."
        )

    def _fit_transform_column(self, values: np.ndarray, column_name: str) -> np.ndarray:
        transformer = self._create_column_transformer()
        self.column_transformers[column_name] = transformer

        if transformer is None:
            return values

        return transformer.fit_transform(values)

    def _apply_column_transform(self, values: np.ndarray, column_name: str, inverse: bool) -> np.ndarray:
        transformer = self.column_transformers.get(column_name)
        if transformer is None:
            return values
        return transformer.inverse_transform(values) if inverse else transformer.transform(values)

    @staticmethod
    def _ensure_2d_array(y, columns: List[str]):
        original_type = "array"
        original_shape = np.asarray(y).shape
        original_index = None
        original_columns = columns

        if isinstance(y, pd.DataFrame):
            original_type = "dataframe"
            original_index = y.index
            original_columns = y.columns.tolist()
            arr = y.values
        elif isinstance(y, pd.Series):
            original_type = "series"
            original_index = y.index
            original_columns = [y.name if y.name is not None else columns[0]]
            arr = y.to_frame().values
        else:
            arr = np.asarray(y)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                if len(columns) > 1:
                    arr = arr.reshape(1, -1)
                else:
                    arr = arr.reshape(-1, 1)

        return arr.astype(float), original_type, original_shape, original_index, original_columns

    @staticmethod
    def _restore_type(arr: np.ndarray, original_type: str, original_shape, original_index, original_columns):
        if original_type == "dataframe":
            return pd.DataFrame(arr, index=original_index, columns=original_columns)
        if original_type == "series":
            return pd.Series(arr.reshape(-1), index=original_index, name=original_columns[0])

        if len(original_shape) == 0:
            return np.asarray(arr).reshape(())
        if len(original_shape) == 1:
            return np.asarray(arr).reshape(-1)
        return np.asarray(arr)

    def fit_transform(self, y):
        if isinstance(y, pd.DataFrame):
            self.column_names = y.columns.tolist()
        elif isinstance(y, pd.Series):
            self.column_names = [y.name if y.name is not None else "target"]
        else:
            arr = np.asarray(y)
            width = arr.shape[1] if arr.ndim == 2 else 1
            self.column_names = [f"target_{i}" for i in range(width)]


        arr, original_type, original_shape, original_index, original_columns = self._ensure_2d_array(
            y,
            self.column_names,
        )
        transformed = np.zeros_like(arr, dtype=float)
        self.column_transformers = {}
        for idx, column_name in enumerate(self.column_names):
            transformed[:, [idx]] = self._fit_transform_column(arr[:, [idx]], column_name)
        self.is_fitted = True

        if self.verbose:
            logger.info(f"{self.log_prefix} Fitted target scaler ({self.scaler_type}) on columns: {self.column_names}")

        return self._restore_type(transformed, original_type, original_shape, original_index, original_columns)

    def transform(self, y, columns: Optional[List[str]] = None):

        resolved_columns = self._resolve_columns(columns)
        self._validate_columns(resolved_columns)
        arr, original_type, original_shape, original_index, original_columns = self._ensure_2d_array(
            y,
            resolved_columns,
        )
        transformed = np.zeros_like(arr, dtype=float)
        for idx, column_name in enumerate(resolved_columns):
            transformed[:, [idx]] = self._apply_column_transform(arr[:, [idx]], column_name, inverse=False)

        return self._restore_type(transformed, original_type, original_shape, original_index, original_columns)

    def inverse_transform(self, y, columns: Optional[List[str]] = None):

        resolved_columns = self._resolve_columns(columns)
        self._validate_columns(resolved_columns)
        arr, original_type, original_shape, original_index, original_columns = self._ensure_2d_array(
            y,
            resolved_columns,
        )
        restored = np.zeros_like(arr, dtype=float)
        for idx, column_name in enumerate(resolved_columns):
            restored[:, [idx]] = self._apply_column_transform(arr[:, [idx]], column_name, inverse=True)

        return self._restore_type(restored, original_type, original_shape, original_index, original_columns)
