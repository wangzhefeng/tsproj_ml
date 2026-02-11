# -*- coding: utf-8 -*-

# ***************************************************
# * File        : advanced_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : 特征工程：
# *               基本特征：
# *                 - 内生变量特征
# *                     - 日期时间特征(小时、星期、月份、季度等)、周期性编码(sin/cos)
# *                     - 天气特征(天气数据集成)
# *                     - 节假日 标记特征(日期类型数据集成)
# *                 - 内生变量特征：
# *                     - 滞后特征：单变量(目标变量)滞后特征、多变量(目标变量、其他内生变量)滞后特征
# *               高级特征：
# *                 - 内生变量特征
# *                     - 滑动窗口统计特征 (Rolling Window Statistics)
# *                         - load_rolling_mean_3   # 最近3步平均值
# *                         - load_rolling_std_7    # 最近7步标准差
# *                         - load_rolling_min_12   # 最近12步最小值
# *                         - load_rolling_max_12   # 最近12步最大值
# *                     - 扩展窗口统计特征 (Expanding Window Statistics)
# *                         - load_expanding_mean   # 累积平均值
# *                         - load_expanding_std    # 累积标准差
# *                     - 差分特征 (Difference Features)
# *                         - load_diff_1          # 一阶差分
# *                         - load_diff_seasonal   # 季节性差分
# *                     - 时间距离特征 (Time-based Features)
# *                         - time_since_peak      # 距离峰值的时间
# *                         - time_since_low      # 距离谷值的时间
# *                 - 交叉特征
# *                     - hour_x_load_lag_1 = hour * load_lag_1     # 时间 × 滞后值
# *                     - temp_x_humidity = temperature * humidity  # 温度 × 湿度
# *                     - load_lag_1_squared = load_lag_1 ** 2      # 多项式特征
# *               目标编码
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import Dict, List, Tuple, Union, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class FeaturePreprocessor:
    """
    统一的特征预处理器
    处理归一化和类别特征编码
    """
    
    def __init__(self, args, log_prefix="[FeaturePreprocessor]"):
        """
        初始化
        
        Args:
            args: 模型配置对象
            log_prefix: 日志前缀
        """
        self.args = args
        self.log_prefix = log_prefix
        # 归一化器
        self.scaler = None
        self.feature_scalers = {}  # 分组归一化器
        # 类别特征信息
        self.category_mappings = {}  # 类别到编码的映射
        self.category_info = {}       # 类别特征的元信息
        # 特征分组信息
        self.feature_groups = {}
    
    def identify_feature_groups(self, X: pd.DataFrame, categorical_features: List[str]) -> Dict[str, List[str]]:
        """
        识别特征分组
        
        Args:
            X: 输入特征DataFrame
            categorical_features: 类别特征列表
        
        Returns:
            特征分组字典
        """
        groups = {
            'lag_features': [col for col in X.columns if '_lag_' in col],
            'datetime_features': [col for col in X.columns if 'datetime_' in col or col.startswith('hour') or col.startswith('day')],
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
        logger.info(f"{self.log_prefix} === Feature groups identified:===")
        for group_name, features in groups.items():
            logger.info(f"{self.log_prefix} {group_name}: {len(features)} features")
        self.feature_groups = groups
    
    def fit_transform(self, X: pd.DataFrame, categorical_features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        """
        训练模式：拟合并转换特征
        
        Args:
            X: 输入特征DataFrame
            categorical_features: 类别特征列表
        
        Returns:
            转换后的特征DataFrame, 实际使用的类别特征列表
        """
        logger.info(f"{self.log_prefix} === Fitting and transforming features (training) ===")
        X_processed = X.copy()
        # 1. 识别特征分组
        self.identify_feature_groups(X_processed, categorical_features)
        # 2. 确定实际存在的类别特征
        actual_categorical = [f for f in categorical_features if f in X_processed.columns]
        # 3. 处理类别特征
        if self.args.encode_categorical_features and actual_categorical:
            logger.info(f"{self.log_prefix} Encoding categorical features...")
            X_processed = self._fit_transform_categorical(X_processed, actual_categorical)
        else:
            # 即使不编码，也转换为 category 类型（LightGBM 原生支持）
            if actual_categorical:
                logger.info(f"{self.log_prefix} Transfomer categorical features...")
                for col in actual_categorical:
                    X_processed[col] = X_processed[col].astype('category')
                    self.category_info[col] = X_processed[col].cat.categories.tolist()
        # 4. 数值特征归一化
        if self.args.scale:
            logger.info(f"{self.log_prefix} Scaling numeric features...")
            X_processed = self._fit_transform_numeric(X_processed, actual_categorical)
        logger.info(f"{self.log_prefix} Feature preprocessing completed.")
        logger.info(f"{self.log_prefix} Processed shape: {X_processed.shape}")
        
        return X_processed, actual_categorical
    
    def transform(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        预测模式：仅转换特征（使用已拟合的参数）
        
        Args:
            X: 输入特征DataFrame
            categorical_features: 类别特征列表
        
        Returns:
            转换后的特征DataFrame
        """
        logger.info(f"{self.log_prefix} === Transforming features (prediction mode) ===")
        X_processed = X.copy()
        # 确定实际存在的类别特征
        actual_categorical = [f for f in categorical_features if f in X_processed.columns]
        # 1. 处理类别特征
        if self.args.encode_categorical_features and actual_categorical:
            X_processed = self._transform_categorical(X_processed, actual_categorical)
        else:
            # 转换为 category 类型（使用训练时的类别）
            for col in actual_categorical:
                if col in self.category_info:
                    X_processed[col] = pd.Categorical(X_processed[col], categories=self.category_info[col])
                else:
                    logger.warning(f"{self.log_prefix} No category info for {col}, using as is.")
                    X_processed[col] = X_processed[col].astype('category')
        # 2. 数值特征归一化
        if self.args.scale:
            X_processed = self._transform_numeric(X_processed, actual_categorical)
        
        return X_processed
    
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
    
    def _fit_transform_numeric(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """
        训练模式：拟合并转换数值特征
        """
        X_processed = X.copy()
        # 选择归一化器类型
        scaler_class = StandardScaler if self.args.scaler_type == "standard" else MinMaxScaler
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
                self.feature_scalers[group_name] = scaler_class()
                X_processed.loc[:, existing_features] = self.feature_scalers[group_name].fit_transform(X_processed[existing_features])
                logger.info(f"{self.log_prefix} Scaled {group_name}: {len(existing_features)} features")
        else:
            # 统一归一化所有数值特征
            logger.info(f"{self.log_prefix} Using unified scaling strategy...")
            numeric_features = [col for col in X_processed.columns if col not in categorical_features]
            if numeric_features:
                self.scaler = scaler_class()
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
                if group_name not in self.feature_scalers:
                    continue
                # 过滤掉不存在的特征
                existing_features = [f for f in features if f in X_processed.columns]
                if not existing_features:
                    continue
                X_processed.loc[:, existing_features] = self.feature_scalers[group_name].transform(X_processed[existing_features])
        else:
            # 统一归一化
            if self.scaler is not None:
                numeric_features = [col for col in X_processed.columns if col not in categorical_features]
                if numeric_features:
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


class EnhancedFeaturePreprocessor(FeaturePreprocessor):
    """增强的特征预处理器"""
    
    def add_lag_statistics(self, df: pd.DataFrame, column: str, windows: List[int]):
        """
        添加滞后统计特征
        
        Args:
            df: 数据框
            column: 列名
            windows: 窗口大小列表 [3, 7, 14]
        
        Returns:
            增强后的数据框
        """
        df_enhanced = df.copy()
        
        for window in windows:
            # 滑动平均
            df_enhanced[f'{column}_rolling_mean_{window}'] = (
                df[column].rolling(window=window, min_periods=1).mean()
            )
            
            # 滑动标准差
            df_enhanced[f'{column}_rolling_std_{window}'] = (
                df[column].rolling(window=window, min_periods=1).std()
            )
            
            # 滑动最小值
            df_enhanced[f'{column}_rolling_min_{window}'] = (
                df[column].rolling(window=window, min_periods=1).min()
            )
            
            # 滑动最大值
            df_enhanced[f'{column}_rolling_max_{window}'] = (
                df[column].rolling(window=window, min_periods=1).max()
            )
            
            # 滑动中位数
            df_enhanced[f'{column}_rolling_median_{window}'] = (
                df[column].rolling(window=window, min_periods=1).median()
            )
        
        return df_enhanced
    
    def add_diff_features(self, df: pd.DataFrame, column: str, periods: List[int]):
        """
        添加差分特征
        
        Args:
            df: 数据框
            column: 列名
            periods: 差分周期列表 [1, 7, 24]
        """
        df_enhanced = df.copy()
        
        for period in periods:
            df_enhanced[f'{column}_diff_{period}'] = df[column].diff(period)
        
        return df_enhanced
    
    def add_expanding_features(self, df: pd.DataFrame, column: str):
        """添加扩展窗口特征"""
        df_enhanced = df.copy()
        
        df_enhanced[f'{column}_expanding_mean'] = df[column].expanding().mean()
        df_enhanced[f'{column}_expanding_std'] = df[column].expanding().std()
        df_enhanced[f'{column}_expanding_min'] = df[column].expanding().min()
        df_enhanced[f'{column}_expanding_max'] = df[column].expanding().max()
        
        return df_enhanced
    
    def add_time_based_features(self, df: pd.DataFrame, column: str, time_column: str):
        """添加基于时间的特征"""
        df_enhanced = df.copy()
        
        # 距离峰值的时间
        peak_idx = df[column].idxmax()
        df_enhanced[f'{column}_time_since_peak'] = (df.index - peak_idx).total_seconds() / 3600
        
        # 距离谷值的时间
        low_idx = df[column].idxmin()
        df_enhanced[f'{column}_time_since_low'] = (df.index - low_idx).total_seconds() / 3600
        
        return df_enhanced


class AdvancedFeatureEngine:
    """
    高级特征工程器
    
    新增特征类型:
    1. 滑窗统计特征（rolling statistics）
    2. 差分特征（differencing）
    3. 周期性特征（cyclical encoding）
    4. 交叉特征（interaction features）
    """
    
    def add_rolling_features(self, df, columns, windows=[3, 7, 14]):
        """
        添加滑窗统计特征
        
        为每个指定列和窗口大小计算:
        - rolling_mean: 滑窗均值
        - rolling_std: 滑窗标准差
        - rolling_min: 滑窗最小值
        - rolling_max: 滑窗最大值
        - rolling_median: 滑窗中位数
        """
        for col in columns:
            for window in windows:
                df[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                df[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                df[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
                df[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
                df[f'{col}_roll_median_{window}'] = df[col].rolling(window).median()
        
        return df
    
    def add_diff_features(self, df, columns, periods=[1, 7]):
        """
        添加差分特征
        
        计算:
        - diff: 差分值
        - pct_change: 百分比变化
        """
        for col in columns:
            for period in periods:
                df[f'{col}_diff_{period}'] = df[col].diff(period)
                df[f'{col}_pct_change_{period}'] = df[col].pct_change(period)
        
        return df
    
    def add_cyclical_features(self, df, col, period):
        """
        添加周期性特征（正弦余弦编码）
        
        避免周期性特征的边界问题
        例如: 23点和0点在数值上差距大，但实际很接近
        """
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
        return df
    
    def add_interaction_features(self, df, pairs):
        """
        添加交叉特征
        
        创建特征对的乘积，捕捉特征间的交互作用
        """
        for col1, col2 in pairs:
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        return df



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
