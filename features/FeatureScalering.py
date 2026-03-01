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
from typing import List, Dict, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class FeatureScaler:
    """
    统一的特征预处理器: 处理归一化和类别特征编码
    """
    
    def __init__(self, args, scaler_type="standard", log_prefix: str="[FeatureScaler]"):
        self.args = args
        self.log_prefix = log_prefix
        # 归一化器
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        # 类别特征信息
        self.category_mappings = {}  # 类别到编码的映射
        self.category_info = {}      # 类别特征的元信息
        # 特征分组信息
        self.feature_groups = {}
        # 分组归一化器
        self.feature_groups_scalers = {}
    
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
        if self.args.scale:
            logger.info(f"{self.log_prefix} Feature groups identified:")
            for group_name, features in groups.items():
                logger.info(f"{self.log_prefix} {group_name}: {len(features)} features")
        # 保存分组信息
        self.feature_groups = groups
    
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
                self.feature_groups_scalers[group_name] = self.scaler
                X_processed.loc[:, existing_features] = self.feature_groups_scalers[group_name].fit_transform(X_processed[existing_features])
                logger.info(f"{self.log_prefix} Scaled {group_name}: {len(existing_features)} features")
        else:
            # 统一归一化所有数值特征
            logger.info(f"{self.log_prefix} Using unified scaling strategy...")
            numeric_features = [col for col in X_processed.columns if col not in categorical_features]
            if numeric_features:
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
                X_processed.loc[:, existing_features] = self.feature_groups_scalers[group_name].transform(X_processed[existing_features])
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
        if self.args.scale:
            logger.info(f"{self.log_prefix} Scaling numeric features...")
            X_processed = self._fit_transform_numeric(X_processed, actual_categorical)
            logger.info(f"{self.log_prefix} Feature preprocessing completed.")

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
        logger.info(f"{self.log_prefix} {'-' * 68}")
        X_processed = X.copy()
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
        if self.args.scale:
            logger.info(f"{self.log_prefix} Scaling numeric features...")
            X_processed = self._transform_numeric(X_processed, actual_categorical)
        logger.info(f"{self.log_prefix} Feature preprocessing completed.")

        logger.info(f"{self.log_prefix} after transform X_processed: \n{X_processed.head()}")
        logger.info(f"{self.log_prefix} after transform X_processed shape: {X_processed.shape}")
        logger.info(f"{self.log_prefix} after transform actual_categorical: {actual_categorical}")
        
        return X_processed




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
