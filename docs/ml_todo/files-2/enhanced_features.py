# -*- coding: utf-8 -*-
"""
增强特征工程模块 (Enhanced Feature Engineering)
===========================================

提供时间序列预测所需的高级特征工程功能

主要功能:
1. 滞后统计特征 (Rolling/Expanding Statistics)
2. 差分特征 (Difference Features)
3. 时间距离特征 (Time-based Features)
4. 交互特征 (Interaction Features)
5. 周期性特征 (Cyclical Features)

作者: Zhefeng Wang
日期: 2026-02-11
版本: 3.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from utils.log_util import logger

class FeatureScaler:
    """统一的特征缩放器"""
    
    def __init__(self, scaler, encode_categorical: bool = False):
        self.scaler = scaler
        self.encode_categorical = encode_categorical
        self.category_encoders = {}
    
    def fit_transform(self, X: pd.DataFrame, categorical_features: List[str]):
        """训练并转换"""
        X_scaled = X.copy()
        
        # 分离数值和类别特征
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # 缩放数值特征
        if numeric_features and self.scaler is not None:
            X_scaled[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        # 编码类别特征
        if self.encode_categorical:
            for col in categorical_features:
                if col in X.columns:
                    X_scaled[col] = X[col].astype('category')
                    self.category_encoders[col] = {
                        'categories': X_scaled[col].cat.categories.tolist()
                    }
                    X_scaled[col] = X_scaled[col].cat.codes
        
        return X_scaled
    
    def transform(self, X: pd.DataFrame, categorical_features: List[str]):
        """仅转换"""
        X_scaled = X.copy()
        
        # 分离数值和类别特征
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # 缩放数值特征
        if numeric_features and self.scaler is not None:
            X_scaled[numeric_features] = self.scaler.transform(X[numeric_features])
        
        # 编码类别特征
        if self.encode_categorical:
            for col in categorical_features:
                if col in X.columns and col in self.category_encoders:
                    encoder_info = self.category_encoders[col]
                    X_scaled[col] = pd.Categorical(
                        X[col],
                        categories=encoder_info['categories']
                    )
                    X_scaled[col] = X_scaled[col].cat.codes
        
        return X_scaled


class AdvancedFeatureEngineer:
    """
    高级特征工程器
    
    提供丰富的特征生成方法，提升模型预测能力
    """
    
    def __init__(self, log_prefix: str = "[FeatureEng]"):
        self.log_prefix = log_prefix
        self.generated_features = []
    
    def add_lag_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        stats: List[str] = ['mean', 'std', 'min', 'max', 'median']
    ) -> pd.DataFrame:
        """
        添加滞后统计特征 (滑动窗口统计)
        
        Args:
            df: 数据框
            columns: 需要计算统计特征的列
            windows: 窗口大小列表，如 [3, 7, 14, 30]
            stats: 统计量列表 ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurt']
        
        Returns:
            增强后的数据框
        
        Examples:
            >>> fe = AdvancedFeatureEngineer()
            >>> df = fe.add_lag_statistics(df, ['load'], [3, 7], ['mean', 'std'])
            # 生成: load_rolling_mean_3, load_rolling_std_3, load_rolling_mean_7, load_rolling_std_7
        """
        logger.info(f"{self.log_prefix} 添加滞后统计特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过")
                continue
            
            for window in windows:
                # 滑动均值
                if 'mean' in stats:
                    feature_name = f'{col}_rolling_mean_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).mean()
                    self.generated_features.append(feature_name)
                
                # 滑动标准差
                if 'std' in stats:
                    feature_name = f'{col}_rolling_std_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).std()
                    self.generated_features.append(feature_name)
                
                # 滑动最小值
                if 'min' in stats:
                    feature_name = f'{col}_rolling_min_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).min()
                    self.generated_features.append(feature_name)
                
                # 滑动最大值
                if 'max' in stats:
                    feature_name = f'{col}_rolling_max_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).max()
                    self.generated_features.append(feature_name)
                
                # 滑动中位数
                if 'median' in stats:
                    feature_name = f'{col}_rolling_median_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).median()
                    self.generated_features.append(feature_name)
                
                # 滑动偏度
                if 'skew' in stats:
                    feature_name = f'{col}_rolling_skew_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).skew()
                    self.generated_features.append(feature_name)
                
                # 滑动峰度
                if 'kurt' in stats:
                    feature_name = f'{col}_rolling_kurt_{window}'
                    df_enhanced[feature_name] = df[col].rolling(
                        window=window, min_periods=1, center=False
                    ).kurt()
                    self.generated_features.append(feature_name)
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个滞后统计特征")
        return df_enhanced
    
    def add_expanding_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        stats: List[str] = ['mean', 'std', 'min', 'max']
    ) -> pd.DataFrame:
        """
        添加扩展窗口统计特征
        
        扩展窗口从数据开始到当前位置，窗口大小递增
        
        Args:
            df: 数据框
            columns: 列名列表
            stats: 统计量列表
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加扩展窗口统计特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if 'mean' in stats:
                feature_name = f'{col}_expanding_mean'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).mean()
                self.generated_features.append(feature_name)
            
            if 'std' in stats:
                feature_name = f'{col}_expanding_std'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).std()
                self.generated_features.append(feature_name)
            
            if 'min' in stats:
                feature_name = f'{col}_expanding_min'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).min()
                self.generated_features.append(feature_name)
            
            if 'max' in stats:
                feature_name = f'{col}_expanding_max'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).max()
                self.generated_features.append(feature_name)
        
        return df_enhanced
    
    def add_diff_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 7, 24]
    ) -> pd.DataFrame:
        """
        添加差分特征
        
        差分可以去除趋势，使数据更平稳
        
        Args:
            df: 数据框
            columns: 列名列表
            periods: 差分周期列表
                - 1: 一阶差分（相邻差分）
                - 7: 周差分（去除周周期）
                - 24: 日差分（去除日周期，针对小时数据）
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加差分特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for period in periods:
                diff_feature_name = f'{col}_diff_{period}'
                pct_change_feature_name = f'{col}_pct_change_{period}'
                df_enhanced[diff_feature_name] = df[col].diff(period)
                df[pct_change_feature_name] = df[col].pct_change(period)
                
                self.generated_features.append(diff_feature_name)
                self.generated_features.append(pct_change_feature_name)
        
        return df_enhanced
    
    def add_pct_change_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 7]
    ) -> pd.DataFrame:
        """
        添加百分比变化特征
        
        计算相对于前N期的百分比变化
        
        Args:
            df: 数据框
            columns: 列名列表
            periods: 周期列表
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加百分比变化特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for period in periods:
                feature_name = f'{col}_pct_change_{period}'
                df_enhanced[feature_name] = df[col].pct_change(period)
                self.generated_features.append(feature_name)
        
        return df_enhanced
    
    def add_time_since_features(
        self,
        df: pd.DataFrame,
        column: str,
        events: List[str] = ['peak', 'trough']
    ) -> pd.DataFrame:
        """
        添加距离关键事件的时间特征
        
        Args:
            df: 数据框
            column: 列名
            events: 事件列表 ['peak', 'trough']
                - 'peak': 距离峰值的时间
                - 'trough': 距离谷值的时间
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加时间距离特征...")
        df_enhanced = df.copy()
        
        if column not in df.columns:
            return df_enhanced
        
        if 'peak' in events:
            # 找到最近的峰值位置
            peaks = (df[column].shift(1) < df[column]) & (df[column] > df[column].shift(-1))
            peak_indices = df.index[peaks]
            
            # 计算距离最近峰值的时间
            feature_name = f'{column}_time_since_peak'
            df_enhanced[feature_name] = 0
            for i in range(len(df)):
                if i == 0:
                    df_enhanced.loc[i, feature_name] = 0
                else:
                    recent_peaks = peak_indices[peak_indices < i]
                    if len(recent_peaks) > 0:
                        df_enhanced.loc[i, feature_name] = i - recent_peaks[-1]
                    else:
                        df_enhanced.loc[i, feature_name] = i
            self.generated_features.append(feature_name)
        
        if 'trough' in events:
            # 找到最近的谷值位置
            troughs = (df[column].shift(1) > df[column]) & (df[column] < df[column].shift(-1))
            trough_indices = df.index[troughs]
            
            # 计算距离最近谷值的时间
            feature_name = f'{column}_time_since_trough'
            df_enhanced[feature_name] = 0
            for i in range(len(df)):
                if i == 0:
                    df_enhanced.loc[i, feature_name] = 0
                else:
                    recent_troughs = trough_indices[trough_indices < i]
                    if len(recent_troughs) > 0:
                        df_enhanced.loc[i, feature_name] = i - recent_troughs[-1]
                    else:
                        df_enhanced.loc[i, feature_name] = i
            self.generated_features.append(feature_name)
        
        return df_enhanced
    
    def add_interaction_features(
        self,
        df: pd.DataFrame,
        column_pairs: List[tuple],
        operations: List[str] = ['multiply', 'divide', 'add', 'subtract']
    ) -> pd.DataFrame:
        """
        添加交互特征
        
        Args:
            df: 数据框
            column_pairs: 列对列表 [('col1', 'col2'), ...]
            operations: 操作列表 ['multiply', 'divide', 'add', 'subtract']
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加交互特征...")
        df_enhanced = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            if 'multiply' in operations:
                feature_name = f'{col1}_x_{col2}'
                df_enhanced[feature_name] = df[col1] * df[col2]
                self.generated_features.append(feature_name)
            
            if 'divide' in operations:
                feature_name = f'{col1}_div_{col2}'
                df_enhanced[feature_name] = df[col1] / (df[col2] + 1e-8)  # 避免除零
                self.generated_features.append(feature_name)
            
            if 'add' in operations:
                feature_name = f'{col1}_add_{col2}'
                df_enhanced[feature_name] = df[col1] + df[col2]
                self.generated_features.append(feature_name)
            
            if 'subtract' in operations:
                feature_name = f'{col1}_sub_{col2}'
                df_enhanced[feature_name] = df[col1] - df[col2]
                self.generated_features.append(feature_name)
        
        return df_enhanced
    
    def add_polynomial_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        degree: int = 2
    ) -> pd.DataFrame:
        """
        添加多项式特征
        
        Args:
            df: 数据框
            columns: 列名列表
            degree: 多项式阶数
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加多项式特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for d in range(2, degree + 1):
                feature_name = f'{col}_pow_{d}'
                df_enhanced[feature_name] = df[col] ** d
                self.generated_features.append(feature_name)
        
        return df_enhanced
    
    def get_generated_features(self) -> List[str]:
        """获取所有生成的特征列表"""
        return self.generated_features
    
    def reset(self):
        """重置生成的特征列表"""
        self.generated_features = []


# 使用示例
if __name__ == "__main__":
    # 创建示例时间序列数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    df = pd.DataFrame({
        'time': dates,
        'load': np.random.randn(1000).cumsum() + 100,
        'temperature': 20 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000),
    })
    
    # 创建特征工程器
    fe = AdvancedFeatureEngineer()
    
    # 添加各种特征
    df = fe.add_lag_statistics(df, ['load'], windows=[3, 7, 24], stats=['mean', 'std'])
    df = fe.add_diff_features(df, ['load'], periods=[1, 24])
    df = fe.add_interaction_features(df, [('load', 'temperature')], operations=['multiply'])
    
    print(f"原始特征数: 3")
    print(f"生成特征数: {len(fe.get_generated_features())}")
    print(f"总特征数: {len(df.columns)}")
    print(f"\n生成的特征列表:")
    for feat in fe.get_generated_features()[:10]:
        print(f"  - {feat}")
