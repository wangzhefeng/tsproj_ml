# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureScalering.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-18
# * Version     : 1.0.021816
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from models.multistep.spec import RolloutFamily, get_strategy_spec
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.base import clone

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def resolve_scale_features_enabled(args) -> bool:
    """优先使用新字段，兼容旧字段 `scale`。"""
    return bool(getattr(args, "scale_features", getattr(args, "scale", False)))


def resolve_scale_target_enabled(args) -> bool:
    """优先使用新字段，兼容旧字段 `scale`。"""
    return bool(getattr(args, "scale_target", getattr(args, "scale", False)))


def resolve_inverse_target_enabled(args) -> bool:
    """优先使用新字段，兼容旧字段 `inverse`。"""
    return bool(getattr(args, "inverse_target", getattr(args, "inverse", False)))


def resolve_feature_scaler_type(args) -> str:
    """获取预测特征 X 的缩放方法。"""
    return str(getattr(args, "feature_scaler_type", "standard")).lower()


def resolve_target_scaler_type(args) -> str:
    """获取目标变量 Y 的缩放方法。"""
    return str(getattr(args, "target_scaler_type", "standard")).lower()


class TargetScaler:
    """
    目标变量缩放器。

    与 FeatureScaler 分离，专门处理训练目标 Y 的缩放与逆变换。
    """

    def __init__(self, args, scaler_type="standard", log_prefix: str="[TargetScaler]", verbose: bool = False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        self.enabled = resolve_scale_target_enabled(self.args)
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
        if self.scaler_type in {"yeo-johnson", "yeojohnson"}:
            return PowerTransformer(method="yeo-johnson", standardize=True)
        if self.scaler_type == "log1p":
            return "log1p"
        raise ValueError(
            f"{self.log_prefix} Unsupported target_scaler_type={self.scaler_type}. "
            f"Supported: none, standard, minmax, log1p, robust, yeo-johnson."
        )

    def _fit_transform_column(self, values: np.ndarray, column_name: str) -> np.ndarray:
        transformer = self._create_column_transformer()
        self.column_transformers[column_name] = transformer

        if transformer is None:
            return values
        if transformer == "log1p":
            if np.any(values < 0):
                raise ValueError(f"{self.log_prefix} log1p target scaling requires non-negative values, but column '{column_name}' contains negatives.")
            return np.log1p(values)

        return transformer.fit_transform(values)

    def _apply_column_transform(self, values: np.ndarray, column_name: str, inverse: bool) -> np.ndarray:
        transformer = self.column_transformers.get(column_name)
        if transformer is None:
            return values
        if transformer == "log1p":
            if not inverse and np.any(values < 0):
                raise ValueError(f"{self.log_prefix} log1p target scaling requires non-negative values, but column '{column_name}' contains negatives.")
            return np.expm1(values) if inverse else np.log1p(values)
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

        if not self.enabled:
            return y.copy() if hasattr(y, "copy") else np.asarray(y).copy()

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
        if not self.enabled:
            return y.copy() if hasattr(y, "copy") else np.asarray(y).copy()

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
        if not self.enabled:
            return y.copy() if hasattr(y, "copy") else np.asarray(y).copy()

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

    @staticmethod
    def get_prediction_target_columns(
        pred_method: str,
        target_output_features: List[str],
        direct_strategy: str = "multioutput",
    ) -> List[str]:
        """
        根据预测方法确定预测结果对应的目标列。
        horizon_feature 模式下训练只 fit 了单列目标，推理 restore 也按单列。
        """
        rollout = get_strategy_spec(pred_method).rollout
        if rollout in {RolloutFamily.DIRECT, RolloutFamily.DIRREC}:
            if str(direct_strategy).lower() == "horizon_feature":
                return [target_output_features[0]]
            return list(target_output_features)
        # Blend（Direct+Recursive）：预测混合了两策略，用 shift_0（=target 本身）近似 restore
        if rollout == RolloutFamily.BLEND:
            return [target_output_features[-1]]
        return [target_output_features[0]]

    def restore_predictions(self, values, target_columns: List[str]):
        """
        在需要时将预测结果从目标缩放空间恢复到原始量纲。
        """
        values_arr = np.asarray(values)
        if not self.enabled:
            return values_arr
        if not resolve_inverse_target_enabled(self.args):
            return values_arr
        return np.asarray(self.inverse_transform(values_arr, columns=target_columns))

    def prepare_eval_target(self, values, target_columns: List[str]):
        """
        统一评估阶段的目标尺度：
        - inverse_target=True: 使用原始量纲
        - inverse_target=False: 将真实值映射到目标缩放空间
        """
        values_arr = np.asarray(values)
        if not self.enabled:
            return values_arr
        if resolve_inverse_target_enabled(self.args):
            return values_arr
        return np.asarray(self.transform(values_arr, columns=target_columns))

    def prepare_history_target_for_plot(self, df_history: pd.DataFrame, target_columns: List[str]):
        """
        预测图保存前，按输出尺度对历史目标列做对齐。
        """
        if (
            not self.enabled
            or resolve_inverse_target_enabled(self.args)
            or df_history is None
            or df_history.empty
            or "y" not in df_history.columns
        ):
            return df_history

        df_history_plot = df_history.copy()
        df_history_plot["y"] = self.transform(df_history_plot["y"], columns=target_columns)
        return df_history_plot


class FeatureScaler:
    """
    统一的特征预处理器: 处理归一化和类别特征编码
    """
    
    def __init__(self, args, scaler_type="standard", log_prefix: str="[FeatureScaler]", verbose: bool=False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        self.enabled = resolve_scale_features_enabled(self.args)
        self.scaler_type = str(scaler_type).lower()
        # 归一化器
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"{self.log_prefix} Unsupported feature_scaler_type={self.scaler_type}. "
                f"Supported: standard, minmax."
            )
        # 类别特征信息
        self.category_mappings = {}  # 类别到编码的映射
        self.category_info = {}      # 类别特征的元信息
        # 特征分组信息
        self.feature_groups = {}
        # 分组归一化器
        self.feature_groups_scalers = {}
        # 训练特征 schema
        self.training_columns = []
        self.training_fill_values = {}
    
    def __identify_feature_groups(self, X: pd.DataFrame, categorical_features: List[str]) -> Dict[str, List[str]]:
        """
        识别特征分组
        
        Args:
            X: 输入特征 DataFrame
            categorical_features: 类别特征列表
        
        Returns:
            特征分组字典
        """
        groups = {
            'lag_features': [col for col in X.columns if '_lag_' in col],
            'datetime_features': [col for col in X.columns if 'dt_' in col or col.startswith('hour') or col.startswith('day')],
            'weather_features': [],
            'categorical_features': [col for col in categorical_features if col in X.columns],
            'other_numeric': []
        }
        # 识别天气特征
        weather_keywords = ['temp', 'humidity', 'wind', 'rain', 'pressure', 'weather', 'rt_', 'cal_']
        for col in X.columns:
            if any(keyword in col.lower() for keyword in weather_keywords):
                groups['weather_features'].append(col)
        # 其余数值特征
        all_special = (
            groups['lag_features'] + 
            groups['datetime_features'] + 
            groups['weather_features'] + 
            groups['categorical_features']
        )
        groups['other_numeric'] = [col for col in X.columns if col not in all_special]
        # logger.info(f"{self.log_prefix} groups: \n{groups}")
        # 打印分组信息
        if self.enabled:
            logger.info(f"{self.log_prefix} Feature groups identified:")
            for group_name, features in groups.items():
                logger.info(f"{self.log_prefix} {group_name}: {len(features)} features")
        # 保存分组信息
        self.feature_groups = groups

    def _capture_training_schema(self, X: pd.DataFrame, categorical_features: List[str]):
        """
        记录训练阶段的特征 schema 与缺失值回填默认值
        """
        self.training_columns = X.columns.tolist()
        self.training_fill_values = {}
        categorical_set = set(categorical_features)
        for col in self.training_columns:
            series = X[col]
            if col in categorical_set:
                mode = series.mode(dropna=True)
                self.training_fill_values[col] = mode.iloc[0] if not mode.empty else "__MISSING__"
            else:
                if pd.api.types.is_numeric_dtype(series):
                    median = series.median(skipna=True)
                    self.training_fill_values[col] = 0.0 if pd.isna(median) else float(median)
                else:
                    mode = series.mode(dropna=True)
                    self.training_fill_values[col] = mode.iloc[0] if not mode.empty else "__MISSING__"

    def _align_feature_schema(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        对齐预测阶段特征 schema（缺列失败、去多余、按训练顺序重排）
        """
        if not self.training_columns:
            return X

        X_aligned = X.copy()
        missing_cols = [c for c in self.training_columns if c not in X_aligned.columns]
        extra_cols = [c for c in X_aligned.columns if c not in self.training_columns]

        if missing_cols:
            raise ValueError(
                f"{self.log_prefix} Missing required inference feature columns: {missing_cols}."
            )

        if extra_cols:
            logger.warning(f"{self.log_prefix} Extra columns at inference (dropped): {extra_cols}")
            X_aligned = X_aligned.drop(columns=extra_cols)

        X_aligned = X_aligned[self.training_columns]

        # 对齐后再次兜底填充
        for col in self.training_columns:
            if X_aligned[col].isna().any():
                X_aligned[col] = X_aligned[col].fillna(self.training_fill_values.get(col, 0.0))

        return X_aligned
    
    def _fit_transform_categorical(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        训练模式：拟合并转换类别特征
        """
        X_processed = X.copy()
        for col in categorical_features:
            if col not in X_processed.columns:
                continue
            # 转换为 category 类型
            X_processed[col] = X_processed[col].astype('category')
            # 保存类别信息
            categories = X_processed[col].cat.categories.tolist()
            codes = X_processed[col].cat.codes.values
            self.category_mappings[col] = {
                'categories': categories,
                'cat_to_code': {cat: code for code, cat in enumerate(categories)},
                'code_to_cat': {code: cat for code, cat in enumerate(categories)}
            }
            # 编码为整数
            X_processed[col] = codes
            logger.info(f"{self.log_prefix} {col}: {len(categories)} categories -> [0, {len(categories)-1}]")
        
        return X_processed
    
    def _transform_categorical(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        预测模式：转换类别特征（使用已保存的映射）
        """
        X_processed = X.copy()
        for col in categorical_features:
            if col not in X_processed.columns:
                continue
            if col not in self.category_mappings:
                logger.warning(f"{self.log_prefix} No mapping for {col}, skipping encoding.")
                continue
            mapping = self.category_mappings[col]
            cat_to_code = mapping['cat_to_code']
            # 应用映射（处理未知类别）
            def encode_value(val):
                if val in cat_to_code:
                    return cat_to_code[val]
                else:
                    # 未知类别：映射到最常见的类别（索引0）
                    logger.warning(f"{self.log_prefix} Unknown category '{val}' in {col}, mapping to 0")
                    return 0
            X_processed[col] = X_processed[col].apply(encode_value)
        
        return X_processed

    def _validate_global_series_ids(self, X: pd.DataFrame) -> None:
        """拒绝把训练期未见序列静默映射成其他序列。"""
        if not bool(getattr(self.args, "enable_global_training", False)):
            return
        series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
        if series_id_col not in X.columns:
            raise ValueError(
                f"{self.log_prefix} global panel input missing series ID column "
                f"'{series_id_col}'."
            )
        mapping = self.category_mappings.get(series_id_col, {})
        known = list(mapping.get("categories", ()))
        if not known:
            known = list(self.category_info.get(series_id_col, ()))
        if not known:
            raise ValueError(
                f"{self.log_prefix} no training series IDs recorded for '{series_id_col}'."
            )
        known_set = set(known)
        unknown = [
            value
            for value in pd.unique(X[series_id_col].dropna())
            if value not in known_set
        ]
        if unknown:
            raise ValueError(
                f"{self.log_prefix} unknown series IDs for '{series_id_col}': {unknown}. "
                "global_unknown_series_policy='raise'."
            )
    
    def _fit_transform_numeric(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        训练模式：拟合并转换数值特征
        """
        X_processed = X.copy()
        # 选择归一化器类型
        if self.args.use_grouped_scaling:
            # 分组归一化
            logger.info(f"{self.log_prefix} Using grouped scaling strategy...")
            for group_name, features in self.feature_groups.items():
                if group_name == 'categorical_features':
                    continue
                
                if not features:
                    continue
                
                # 过滤掉不存在的特征
                existing_features = [f for f in features if f in X_processed.columns]
                if not existing_features:
                    continue
                # 为每组创建独立的归一化器
                self.feature_groups_scalers[group_name] = clone(self.scaler)
                for col in existing_features:
                    X_processed[col] = X_processed[col].astype(float)
                X_processed.loc[:, existing_features] = self.feature_groups_scalers[group_name].fit_transform(X_processed[existing_features])
                logger.info(f"{self.log_prefix} Scaled {group_name}: {len(existing_features)} features")
        else:
            # 统一归一化所有数值特征
            logger.info(f"{self.log_prefix} Using unified scaling strategy...")
            numeric_features = [col for col in X_processed.columns if col not in categorical_features]
            if numeric_features:
                for col in numeric_features:
                    X_processed[col] = X_processed[col].astype(float)
                X_processed.loc[:, numeric_features] = self.scaler.fit_transform(X_processed[numeric_features])
                logger.info(f"{self.log_prefix} Scaled {len(numeric_features)} numeric features")
        
        return X_processed
    
    def _transform_numeric(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        预测模式：转换数值特征（使用已拟合的参数）
        """
        X_processed = X.copy()
        if self.args.use_grouped_scaling:
            # 分组归一化
            for group_name, features in self.feature_groups.items():
                if group_name == 'categorical_features':
                    continue
                if group_name not in self.feature_groups_scalers:
                    continue
                # 过滤掉不存在的特征
                existing_features = [f for f in features if f in X_processed.columns]
                if not existing_features:
                    continue
                for col in existing_features:
                    X_processed[col] = X_processed[col].astype(float)
                X_processed.loc[:, existing_features] = self.feature_groups_scalers[group_name].transform(X_processed[existing_features])
        else:
            # 统一归一化
            if self.scaler is not None:
                numeric_features = [col for col in X_processed.columns if col not in categorical_features]
                if numeric_features:
                    for col in numeric_features:
                        X_processed[col] = X_processed[col].astype(float)
                    X_processed.loc[:, numeric_features] = self.scaler.transform(X_processed[numeric_features])
        
        return X_processed
    
    def validate_features(self, X: pd.DataFrame, stage: str = "unknown"):
        """
        验证特征质量
        
        Args:
            X: 特征DataFrame
            stage: 阶段名称（用于日志）
        """
        logger.info(f"{self.log_prefix} === Feature Validation ({stage}) ===")
        logger.info(f"{self.log_prefix} Shape: {X.shape}")
        # 检查缺失值
        missing = X.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"{self.log_prefix} Missing values detected:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"{self.log_prefix} {col}: {count} ({count/len(X)*100:.2f}%)")
        else:
            logger.info(f"{self.log_prefix} No missing values.")
        # 检查无穷值
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(X[col]).sum()
            if inf_count > 0:
                logger.error(f"{self.log_prefix} Infinite values in {col}: {inf_count}")
        # 数值特征统计
        if len(numeric_cols) > 0:
            logger.info(f"{self.log_prefix} Numeric features range:")
            for col in numeric_cols[:5]:  # 只显示前5个
                min_val, max_val = X[col].min(), X[col].max()
                logger.info(f"{self.log_prefix} {col}: [{min_val:.4f}, {max_val:.4f}]")
        # 类别特征统计
        categorical_cols = X.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            logger.info(f"{self.log_prefix} Categorical features:")
            for col in categorical_cols:
                n_unique = X[col].nunique()
                logger.info(f"{self.log_prefix} {col}: {n_unique} unique values")

    def fit_transform(self, X: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        训练模式：拟合并转换特征
        
        Args:
            X: 输入特征 DataFrame
            categorical_features: 类别特征列表
        
        Returns:
            转换后的特征 DataFrame, 实际使用的类别特征列表
        """
        logger.info(f"{self.log_prefix} Fitting and transforming features start (training)...")
        logger.info(f"{self.log_prefix} {'-' * 60}")
        X_processed = X.copy()
        # 记录训练 schema（原始特征）
        self._capture_training_schema(X_processed, categorical_features)
        # 1. 识别特征分组
        self.__identify_feature_groups(X_processed, categorical_features)
        # 2. 确定实际存在的类别特征
        actual_categorical = [f for f in categorical_features if f in X_processed.columns]
        # 3. 处理类别特征
        if self.args.encode_categorical_features and actual_categorical:
            logger.info(f"{self.log_prefix} Scaling categorical features...")
            X_processed = self._fit_transform_categorical(X_processed, actual_categorical)
        else:
            # 即使不编码，也转换为 'category' 类型（LightGBM 原生支持）
            if actual_categorical:
                logger.info(f"{self.log_prefix} Encoding categorical features...")
                for col in actual_categorical:
                    X_processed[col] = X_processed[col].astype('category')
                    self.category_info[col] = X_processed[col].cat.categories.tolist()
        # 4. 数值特征归一化
        if self.enabled:
            logger.info(f"{self.log_prefix} Scaling numeric features...")
            X_processed = self._fit_transform_numeric(X_processed, actual_categorical)
            logger.info(f"{self.log_prefix} Feature preprocessing completed.")
            if self.verbose:
                logger.info(f"{self.log_prefix} after fit_transform X_processed: \n{X_processed.head()}")
                logger.info(f"{self.log_prefix} after fit_transform X_processed shape: {X_processed.shape}")
                logger.info(f"{self.log_prefix} after fit_transform actual_categorical: {actual_categorical}")
        
        return X_processed, actual_categorical
    
    def transform(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        预测模式：仅转换特征（使用已拟合的参数）
        
        Args:
            X: 输入特征 DataFrame
            categorical_features: 类别特征列表
        
        Returns:
            转换后的特征 DataFrame
        """
        logger.info(f"{self.log_prefix} Transforming features start (forecasting)...")
        logger.info(f"{self.log_prefix} {'-' * 69}")
        # 先按训练阶段 schema 对齐
        X_processed = self._align_feature_schema(X.copy())
        self._validate_global_series_ids(X_processed)
        # 1. 确定实际存在的类别特征
        actual_categorical = [f for f in categorical_features if f in X_processed.columns]
        # 2. 处理类别特征
        if self.args.encode_categorical_features and actual_categorical:
            logger.info(f"{self.log_prefix} Scaling categorical features...")
            X_processed = self._transform_categorical(X_processed, actual_categorical)
        else:
            # 即使不编码，转换为 'category' 类型（使用训练时的类别）
            if actual_categorical:
                logger.info(f"{self.log_prefix} Encoding categorical features...")
                for col in actual_categorical:
                    if col in self.category_info:
                        X_processed[col] = pd.Categorical(X_processed[col], categories=self.category_info[col])
                    else:
                        logger.warning(f"{self.log_prefix} No category info for {col}, using as is.")
                        X_processed[col] = X_processed[col].astype('category')
        # 3. 数值特征归一化
        if self.enabled:
            logger.info(f"{self.log_prefix} Scaling numeric features...")
            X_processed = self._transform_numeric(X_processed, actual_categorical)
            logger.info(f"{self.log_prefix} Feature preprocessing completed.")
            if self.verbose:
                logger.info(f"{self.log_prefix} after transform X_processed: \n{X_processed.head()}")
                logger.info(f"{self.log_prefix} after transform X_processed shape: {X_processed.shape}")
                logger.info(f"{self.log_prefix} after transform actual_categorical: {actual_categorical}")
        
        return X_processed




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
