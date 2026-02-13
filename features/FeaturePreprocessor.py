# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeaturePreprocessor.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : 特征工程：
# *               - 一、基本特征：
# *                 - 1.外生变量特征
# *                     - 1.1 日期时间特征(小时、星期、月份、季度等)、周期性编码(sin/cos)
# *                     - 1.2 天气特征(天气数据集成)
# *                     - 1.3 节假日 标记特征(日期类型数据集成)
# *                 - 2.内生变量特征：
# *                     - 2.1 滞后特征：单变量(目标变量)滞后特征、多变量(目标变量、其他内生变量)滞后特征
# *               - 二、高级特征：
# *                 - 1.内生变量特征
# *                     - 1.1 滑动窗口统计特征 (Rolling Window Statistics)
# *                         - load_rolling_mean_3   # 最近3步平均值
# *                         - load_rolling_std_7    # 最近7步标准差
# *                         - load_rolling_min_12   # 最近12步最小值
# *                         - load_rolling_max_12   # 最近12步最大值
# *                     - 1.2 扩展窗口统计特征 (Expanding Window Statistics)
# *                         - load_expanding_mean   # 累积平均值
# *                         - load_expanding_std    # 累积标准差
# *                     - 1.3 差分特征 (Difference Features)
# *                         - load_diff_1          # 一阶差分
# *                         - load_diff_seasonal   # 季节性差分
# *                     - 1.4 百分比变化特征 (Percentage Change Features)
# *                     - 1.5 距离关键事件的时间特征 (Time-based Features)
# *                         - time_since_peak      # 距离峰值的时间
# *                         - time_since_low      # 距离谷值的时间
# *                     - 1.6 周期性特征（正弦余弦编码）(cyclical encoding)
# *                 - 2.交叉(交互)特征 (interaction features)
# *                     - hour_x_load_lag_1 = hour * load_lag_1     # 时间 × 滞后值
# *                     - temp_x_humidity = temperature * humidity  # 温度 × 湿度
# *                     - load_lag_1_squared = load_lag_1 ** 2      # 多项式特征
# *                 - 3.多项式特征 (polynomial features)
# *               - 三、目标编码
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class FeatureScaler:
    """
    统一的特征预处理器
    处理归一化和类别特征编码
    """
    
    def __init__(self, args, scaler_type="standard", log_prefix="[FeatureScaler]"):
        """
        初始化
        
        Args:
            args: 模型配置对象
            log_prefix: 日志前缀
        """
        self.args = args
        self.log_prefix = log_prefix
        # 归一化器
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
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
                self.feature_scalers[group_name] = self.scaler
                X_processed.loc[:, existing_features] = self.feature_scalers[group_name].fit_transform(X_processed[existing_features])
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


class FeatureEngineer:
    """
    特征预处理器（包含原有的所有特征工程功能）
    """
    def __init__(self, args, log_prefix="[FeatureEngineer]"):
        self.args = args
        self.log_prefix = log_prefix
        # 高级特征工程
        self.advanced_feature_engineer = AdvancedFeatureEngineer(log_prefix)
    # ------------------------------
    # exogenous features engineering
    # ------------------------------
    def extend_datetime_feature(self, df: pd.DataFrame, freq_minutes: int, n_per_day):
        """
        增加时间特征
        """
        df_copy = df.copy()
        df_copy["time"] = pd.to_datetime(df_copy["time"])

        datetime_features_list = []
        # 时间基础特征
        feature_map = {
            "minute": lambda x: x.minute,
            "hour": lambda x: x.hour,
            "day": lambda x: x.day,
            "weekday": lambda x: x.weekday(),
            # "week": lambda x: x.week,
            "week": lambda x: x.isocalendar().week, # Use isocalendar().week for consistency
            "day_of_week": lambda x: x.dayofweek,
            # "week_of_year": lambda x: x.weekofyear,
            "week_of_year": lambda x: x.isocalendar().week,
            "month": lambda x: x.month,
            "days_in_month": lambda x: x.daysinmonth,
            "quarter": lambda x: x.quarter,
            "day_of_year": lambda x: x.dayofyear,
            "year": lambda x: x.year,
        }
        for feature_name in self.args.datetime_features:
            if feature_name in feature_map:
                col_name = f"dt_{feature_name}"
                df_copy[col_name] = df_copy["time"].apply(feature_map[feature_name])
                datetime_features_list.append(col_name)
        # 周期性特征 (将时间转换为可循环的 sin/cos 形式)
        if 'dt_hour' in df_copy.columns and 'dt_minute' in df_copy.columns:
            df_copy["dt_minute_in_day"] = df_copy["dt_hour"] * (60 / freq_minutes) + df_copy["dt_minute"] / freq_minutes
            df_copy["dt_minute_in_day_sin"] = np.sin(df_copy["dt_minute_in_day"] * (2 * np.pi / n_per_day))
            df_copy["dt_minute_in_day_cos"] = np.cos(df_copy["dt_minute_in_day"] * (2 * np.pi / n_per_day))
        datetime_features_list.append("dt_minute_in_day_sin")
        datetime_features_list.append("dt_minute_in_day_cos")

        return df_copy, datetime_features_list

    def extend_datetype_feature(self, df: pd.DataFrame, df_date: pd.DataFrame, col_ts: str="date", col_type: str="date_type"):
        """
        增加日期类型特征：
        1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
        """
        df_copy = df.copy()
        if df_date is not None and not df_date.empty:
            # data map
            df_copy["date"] = df_copy["time"].dt.normalize() # Use .dt.normalize() to get date part
            df_copy["date_type"] = df_copy["date"].map(df_date.set_index(col_ts)[col_type])
            del df_copy["date"]
            # date features
            date_features = ["date_type"]
        else:
            date_features = []

        return df_copy, date_features

    def extend_weather_feature(self, df: pd.DataFrame, df_weather: pd.DataFrame, col_ts: str):
        """
        处理天气特征
        """
        df_copy = df.copy()
        if df_weather is not None and not df_weather.empty:
            weather_features_raw = ["rt_ssr", "rt_ws10", "rt_tt2", "rt_dt", "rt_ps", "rt_rain"]
            # Ensure df_weather has these columns
            df_weather_filtered = df_weather[[col for col in [col_ts] + weather_features_raw if col in df_weather.columns]].copy()
            # 删除含空值的行
            df_weather_filtered.dropna(inplace=True, ignore_index=True)
            if df_weather_filtered.empty:
                logger.warning(f"{self.log_prefix} df_weather became empty after dropping NaNs.")
                return df_copy, []

            # 将除了timeStamp的列转为float类型
            for col in weather_features_raw:
                if col in df_weather_filtered.columns:
                    # df_weather_filtered[col] = df_weather_filtered[col].apply(lambda x: float(x))
                    df_weather_filtered[col] = pd.to_numeric(df_weather_filtered[col], errors='coerce')

            # 计算相对湿度
            df_weather_filtered["cal_rh"] = np.nan
            # This calculation is for specific units (Kelvin), ensure consistency
            # Assuming rt_tt2 and rt_dt are in Kelvin based on the calculation method
            # If not, convert to Kelvin first (e.g., Celsius + 273.15)
            # This needs to be vectorized for efficiency
            valid_idx = df_weather_filtered["rt_tt2"].notna() & df_weather_filtered["rt_dt"].notna()
            
            # Constants for Tetens formula variation
            A = 17.27
            B = 237.7
            
            # Convert temperature from Kelvin to Celsius for the formula if needed, or adjust formula
            # Assuming values are in Celsius and formula used is for Celsius, then converting to K-like internally
            # Given the original code: (df_weather.loc[i, "rt_dt"] - 273.15) and (df_weather.loc[i, "rt_tt2"] - 273.15)
            # This suggests rt_dt and rt_tt2 are in Kelvin, but the 35.86 value is unusual.
            # I will use a more standard formula for relative humidity from dew point and temperature.
            # Using August-Roche-Magnus approximation (values in Celsius)
            # es = a * exp((b*T) / (c+T))
            
            # Simplified for existing structure:
            # Check for non-NaN values for rt_tt2 and rt_dt
            idx_to_calc = df_weather_filtered.index[valid_idx]
            if not idx_to_calc.empty:
                # Assuming rt_tt2 and rt_dt are already in Celsius or will be used direct as is
                # This formula seems to be a variation of the Tetens equation, adapted for specific units
                # Original formula was: exp(17.2693 * (T_dew - 273.15) / (T_dew - 35.86)) / exp(17.2693 * (T_air - 273.15) / (T_air - 35.86)) * 100
                # Let's adjust this to be more robust. The '35.86' looks like a typo for something like 237.7 or 243.0 for Celsius.
                # If rt_tt2 and rt_dt are indeed in Kelvin, the formula is highly unusual.
                # For a standard RH calculation, we need T_air (air temp) and T_dew (dew point temp) in Celsius.
                
                # Let's assume rt_tt2 and rt_dt are in Celsius for now, or the provided constants are based on this.
                # If the values are actual Kelvin, they need to be converted to Celsius first for standard formulas.
                # Given no explicit conversion, I will use the provided structure and assume the original values.
                
                # If these are Kelvin, convert to Celsius first:
                T_air_C = df_weather_filtered.loc[valid_idx, "rt_tt2"] - 273.15
                T_dew_C = df_weather_filtered.loc[valid_idx, "rt_dt"] - 273.15

                # Standard Magnus-Tetens formula for saturation vapor pressure (in hPa)
                # Ps(T) = 6.1078 * exp((17.27 * T) / (237.3 + T))
                # For RH = (Ps(T_dew) / Ps(T_air)) * 100
                e_s_Td = 6.1078 * np.exp((17.2693 * T_dew_C) / (237.29 + T_dew_C))
                e_s_T = 6.1078 * np.exp((17.2693 * T_air_C) / (237.29 + T_air_C))
                
                rh_values = (e_s_Td / e_s_T) * 100
                rh_values = np.clip(rh_values, 0, 100) # Clip between 0 and 100
                df_weather_filtered.loc[valid_idx, "cal_rh"] = rh_values

            # 特征筛选
            weather_features = [
                "rt_ssr",   # 太阳总辐射
                "rt_ws10",  # 10m 风速
                "rt_tt2",   # 2M 气温
                "cal_rh",   # 相对湿度
                "rt_ps",    # 气压
                "rt_rain",  # 降雨量
            ]
            # Keep only features that exist in the dataframe
            weather_features = [f for f in weather_features if f in df_weather_filtered.columns]
            df_weather_filtered = df_weather_filtered[[col_ts] + weather_features]
            
            # 合并目标数据和天气数据
            df_copy = pd.merge(df_copy, df_weather_filtered, left_on="time", right_on=col_ts, how="left")
            # 插值填充缺失值
            df_copy = df_copy.interpolate(method="linear", limit_direction="both")
            df_copy.dropna(inplace=True, ignore_index=True)
            # 删除无用特征
            if col_ts in df_copy.columns:
                del df_copy[col_ts]
        else:
            weather_features = []
        
        return df_copy, weather_features

    def extend_future_weather_feature(self, df: pd.DataFrame, df_weather: pd.DataFrame, col_ts: str):
        """
        未来天气数据特征构造
        """
        df_copy = df.copy()
        if df_weather is not None and not df_weather.empty:
            # 筛选天气预测数据
            pred_weather_features_map = {
                "pred_ssrd": "rt_ssr",
                "pred_ws10": "rt_ws10",
                "pred_tt2": "rt_tt2",
                "pred_rh": "cal_rh",
                "pred_ps": "rt_ps",
                "pred_rain": "rt_rain"
            }
            # Filter df_weather for relevant columns and dropna
            df_weather_filtered = df_weather[[col for col in [col_ts] + list(pred_weather_features_map.keys()) if col in df_weather.columns]].copy()
            df_weather_filtered.dropna(inplace=True, ignore_index=True)
            if df_weather_filtered.empty:
                logger.warning(f"{self.log_prefix} df_weather_future became empty after dropping NaNs.")
                return df_copy, []

            # 数据类型转换
            for pred_col in pred_weather_features_map.keys():
                if pred_col in df_weather_filtered.columns:
                    # df_weather_filtered[pred_col] = df_weather_filtered[pred_col].apply(lambda x: float(x))
                    df_weather_filtered[pred_col] = pd.to_numeric(df_weather_filtered[pred_col], errors='coerce')

            # 将预测天气数据整理到预测df中
            for pred_col, target_col in pred_weather_features_map.items():
                if pred_col in df_weather_filtered.columns:
                    # Apply specific transformations if defined
                    if pred_col == "pred_ps":
                        df_weather_filtered[pred_col] = df_weather_filtered[pred_col].apply(lambda x: x - 50.0)
                    elif pred_col == "pred_rain":
                        df_weather_filtered[pred_col] = df_weather_filtered[pred_col].apply(lambda x: x - 2.5)
                    df_copy[target_col] = df_copy["time"].map(df_weather_filtered.set_index(col_ts)[pred_col])
            
            # features to return
            weather_features = list(pred_weather_features_map.values())
            # Ensure to return only features that were actually added
            weather_features = [f for f in weather_features if f in df_copy.columns]
        else:
            weather_features = []
        
        return df_copy, weather_features
    # ------------------------------
    # endogenous features engineering
    # ------------------------------
    def extend_direct_multi_step_targets(self, df: pd.DataFrame, target: str, horizon: int):
        """
        为直接多步预测创建未来多步目标
        
        Args:
            df: 数据框
            target: 目标变量名
            horizon: 预测horizon
        
        Returns:
            (扩展后的数据框, 目标特征列表)
        """
        df_copy = df.copy()
        shift_target_features = []
        
        # shift features building
        for h in range(0, horizon):
            shifted_col_name = f"{target}_shift_{h}"
            df_copy[shifted_col_name] = df_copy[target].shift(-h)
            shift_target_features.append(shifted_col_name)
        
        return df_copy, shift_target_features

    def extend_lag_feature_univariate(self, df: pd.DataFrame, target: str, lags: List[int]):
        """
        扩展单变量滞后特征(for univariate time series)
        
        Args:
            df: 数据框
            target: 目标变量名
            lags: 滞后期列表
        
        Returns:
            (扩展后的数据框, 新增的滞后特征列表)
        """
        df_lags = df.copy()
        lag_features = []

        for lag in lags:
            col_name = f'{target}_lag_{lag}'
            df_lags[col_name] = df_lags[target].shift(lag)
            lag_features.append(col_name)
        
        return df_lags, lag_features

    def extend_lag_feature_multivariate(self, df: pd.DataFrame, endogenous_cols: List[str], lags: List[int]):
        """
        扩展多变量滞后特征
        
        Args:
            df: 数据框
            endogenous_cols: 内生变量列表
            n_lags: 最大滞后期
            horizon: 预测horizon
        
        Returns:
            (扩展后的数据框, 滞后特征列表, 目标特征列表)
        """
        df_copy = df.copy()
        # 将 time 作为索引
        df_copy = df_copy.set_index("time").copy()
        
        all_lag_features = []
        for col in endogenous_cols:
            if col not in df_copy.columns:
                continue
            
            # 创建滞后特征
            lags_X = [df_copy[col].shift(i) for i in lags]
            lag_col_names_X = [f'{col}_lag_{i}' for i in lags]
            for i, name in enumerate(lag_col_names_X):
                df_copy[name] = lags_X[i].values
                all_lag_features.append(name)
        
        return df_copy, all_lag_features
    # ------------------------------
    # create features
    # ------------------------------
    def create_features(self, df_series: pd.DataFrame, endogenous_features_with_target: List[str], exogenous_features: List[str], target_feature: str, categorical_features: List[str]):
        """
        创建特征（集成高级特征工程）
        
        Args:
            df_series: 时间序列数据
            endogenous_features_with_target: 内生变量（含目标）
            exogenous_features: 外生变量
            target_feature: 目标变量
            categorical_features: 类别特征
        
        Returns:
            (特征化后的数据, 预测特征列表, 目标输出特征列表, 类别特征列表)
        """
        # df_series copy
        df_series_copy = df_series.copy()
        # For multi-output recursive, we need lags for ALL endogenous variables.
        endogenous_for_lags = endogenous_features_with_target
        # Clear and re-populate categorical_features for each run to avoid duplicates
        categorical_features_copy = categorical_features.copy() 
        # ------------------------------
        # 特征工程：日期时间特征
        # ------------------------------
        df_series_copy, datetime_features = self.extend_datetime_feature(
            df=df_series_copy,
            freq_minutes=self.args.freq_minutes,
            n_per_day=self.args.n_per_day,
        )
        logger.info(f"{self.log_prefix} after extend_datetime_feature df_series_copy: \n{df_series_copy.head()}")
        logger.info(f"{self.log_prefix} after extend_datetime_feature datetime_features: {datetime_features}")
        # 类别特征更新
        categorical_features_copy.extend(self.args.datetime_categorical_features)
        categorical_features_copy = sorted(set(categorical_features_copy), key=categorical_features_copy.index)
        # ------------------------------
        # 特征工程：滞后特征
        # ------------------------------
        lag_features = []
        target_output_features = []
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            target_output_features.append(target_feature)
            logger.info(f"{self.log_prefix} after target_output_features: {target_output_features}")
        elif self.args.pred_method == "univariate-single-multistep-direct":
            # For univariate, only target lags are features
            df_series_copy, uni_lag_features = self.extend_lag_feature_univariate(
                df = df_series_copy,
                target = target_feature,
                lags = self.args.lags,
            )
            lag_features.extend(uni_lag_features)
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate target_output_features: {target_output_features}")
            # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
            df_series_copy, shift_target_features = self.extend_direct_multi_step_targets(
                df = df_series_copy,
                target = target_feature,
                horizon = self.horizon,
            )
            target_output_features.extend(shift_target_features)
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {target_output_features}")
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            # predictor features
            df_series_copy, uni_lag_features = self.extend_lag_feature_univariate(
                df = df_series_copy,
                target = target_feature,
                lags = self.args.lags,
            )
            lag_features.extend(uni_lag_features)
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate target_output_features: {target_output_features}")
            # target features(For recursive, target is target_t+1)
            df_series_copy, shift_target_features = self.extend_direct_multi_step_targets(
                df = df_series_copy,
                target = target_feature,
                horizon = 1,
            )
            target_output_features.extend(shift_target_features)
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {target_output_features}")
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            # For univariate, only target lags are features
            df_series_copy, uni_lag_features = self.extend_lag_feature_univariate(
                df = df_series_copy,
                target = target_feature,
                lags = self.args.lags,
            )
            lag_features.extend(uni_lag_features)
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate target_output_features: {target_output_features}")
            # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
            df_series_copy, shift_target_features = self.extend_direct_multi_step_targets(
                df = df_series_copy,
                target = target_feature,
                horizon = self.horizon,
            )
            target_output_features.extend(shift_target_features)
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {target_output_features}")
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
            df_series_copy, shift_target_features = self.extend_direct_multi_step_targets(
                df = df_series_copy,
                target = target_feature,
                horizon = self.horizon,
            )
            target_output_features.extend(shift_target_features)
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {target_output_features}")
            # For multivariate, target and other endogenous lags are features
            df_series_copy, multi_lag_features = self.extend_lag_feature_multivariate(
                df = df_series_copy,
                endogenous_features_with_target = endogenous_for_lags,
                lags = self.args.lags,
            )
            lag_features.extend(multi_lag_features)
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate target_output_features: {target_output_features}")
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            # For multivariate recursive, lags of target and other endogenous are features
            df_series_copy, multi_lag_features = self.extend_lag_feature_multivariate(
                df = df_series_copy,
                endogenous_features_with_target = endogenous_for_lags,
                lags = self.args.lags,
            )
            lag_features.extend(multi_lag_features)
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate target_output_features: {target_output_features}")
            # target features(For recursive, target is target_t+1)
            df_series_copy, shift_target_features = self.extend_direct_multi_step_targets(
                df = df_series_copy,
                target = target_feature,
                horizon = 1,
            )
            target_output_features.extend(shift_target_features)
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {target_output_features}")
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            # For multivariate recursive, lags of target and other endogenous are features
            df_series_copy, multi_lag_features = self.extend_lag_feature_multivariate(
                df = df_series_copy,
                endogenous_features_with_target = endogenous_for_lags,
                lags = self.args.lags,
            )
            lag_features.extend(multi_lag_features)
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate target_output_features: {target_output_features}")
            # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
            logger.info(f"{self.log_prefix} self.horizon: {self.horizon}")
            df_series_copy, shift_target_features = self.extend_direct_multi_step_targets(
                df = df_series_copy,
                target = target_feature,
                horizon = self.horizon,
            )
            target_output_features.extend(shift_target_features)
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_copy: \n{df_series_copy.head()}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets lag_features: {lag_features}")
            logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {target_output_features}")
        # ------------------------------
        # 添加高级特征（新增）
        # ------------------------------
        if self.args.enable_advanced_features:
            logger.info(f"{self.log_prefix} 添加高级特征...")
            # 添加滞后统计特征
            if target_feature in df_series_copy.columns:
                df_series_copy = self.advanced_feature_engineer.add_lag_statistics(
                    df_series_copy,
                    columns=[target_feature],
                    windows=self.args.rolling_windows,
                    stats=self.args.rolling_stats
                )
            # 添加差分特征
            if target_feature in df_series_copy.columns:
                df_series_copy = self.advanced_feature_engineer.add_diff_features(
                    df_series_copy,
                    columns=[target_feature],
                    periods=self.args.diff_periods
                )
            # 将生成的特征添加到特征列表
            lag_features.extend(self.advanced_feature_engineer.get_generated_features())
        # ------------------------------
        # Feature ordering
        # ------------------------------
        # 内生变量特征 = 内生变量滞后特征
        endgenous_features_all = lag_features
        # 外生变量特征 = 外生变量特征 + 日期时间特征(特征工程)
        exogenous_features_all = exogenous_features + datetime_features
        # 预测特征 = 内生变量特征（滞后特征） + 外生变量特征
        predictor_features = endgenous_features_all + exogenous_features_all
        # 除内生变量特征(滞后特征)、内生变量特征(shift特征)、目标内生变量外的内生变量
        current_endogenous_as_features = [
            col for col in endogenous_features_with_target
            if col not in target_output_features + lag_features + endogenous_for_lags
        ]
        predictor_features.extend(current_endogenous_as_features)
        # Filter df_series_copy to only include necessary columns to avoid errors later
        predictor_features = sorted(set(predictor_features), key=predictor_features.index)
        all_cols_needed = ["time"] + predictor_features + target_output_features
        df_series_copy = df_series_copy[[col for col in all_cols_needed if col in df_series_copy.columns]]

        return df_series_copy, predictor_features, target_output_features, categorical_features_copy

class AdvancedFeatureEngineer:
    """
    高级特征工程器
    
    新增特征类型:
    1. 滑动窗口统计特征 (rolling window statistics)
    2. 滚动(扩展)窗口统计特征 (expanding window statistics)
    3. 差分特征 (difference features）
    4. 百分比变化特征 (percentage change features)
    5. 距离关键事件的时间特征 (Time-based Features)
    6. 周期性特征 (cyclical encoding)
    7. 交叉(交互)特征 (interaction features)
    8. 多项式特征 (polynomial features)
    """
    
    def __init__(self, log_prefix: str = "[FeatureEngineer]"):
        self.log_prefix = log_prefix
        self.generated_features = []
    
    def add_rolling_statistics(self, df: pd.DataFrame, columns: List[str], windows: List[int], stats: List[str] = ["mean", "std", "min", "max", "median", "skew", "kurt"]) -> pd.DataFrame:
        """
        添加滑动窗口统计特征
        
        Args:
            df: 数据框
            columns: 需要计算统计特征的列
            windows: 窗口大小列表，如 [3, 7, 14, 30]
            stats: 统计量列表 ["mean", "std", "min", "max", "median", "skew", "kurt"]
        
        Returns:
            增强后的数据框
        
        Examples:
            >>> fe = AdvancedFeatureEngineer()
            >>> df = fe.add_lag_statistics(df, ['load'], [3, 7], ['mean', 'std'])
            # 生成: load_rolling_mean_3, load_rolling_std_3, load_rolling_mean_7, load_rolling_std_7
        """
        logger.info(f"{self.log_prefix} 添加滑动窗口统计特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
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
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个滑动窗口统计特征。")
        return df_enhanced

    def add_expanding_statistics(self, df: pd.DataFrame, columns: List[str], stats: List[str] = ["mean", "std", "min", "max", "median", "skew", "kurt"]) -> pd.DataFrame:
        """
        添加扩展窗口统计特征
        
        扩展窗口从数据开始到当前位置，窗口大小递增
        
        Args:
            df: 数据框
            columns: 需要计算统计特征的列
            stats: 统计量列表 ["mean", "std", "min", "max", "median", "skew", "kurt"]
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加扩展窗口统计特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
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

            if 'median' in stats:
                feature_name = f'{col}_expanding_median'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).median()
                self.generated_features.append(feature_name)
            
            if 'skew' in stats:
                feature_name = f'{col}_expanding_skew'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).skew()
                self.generated_features.append(feature_name)       
            
            if 'kurt' in stats:
                feature_name = f'{col}_expanding_kurt'
                df_enhanced[feature_name] = df[col].expanding(min_periods=1).kurt()
                self.generated_features.append(feature_name)
            
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个扩展窗口统计特征。")
        return df_enhanced
  
    def add_diff_features(self, df: pd.DataFrame, columns: List[str], periods: List[int] = [1, 7, 24]) -> pd.DataFrame:
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
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for period in periods:
                feature_name = f'{col}_diff_{period}'
                df_enhanced[feature_name] = df[col].diff(period)
                self.generated_features.append(feature_name)
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个差分特征。")
        return df_enhanced
    
    def add_pct_change_features(self, df: pd.DataFrame, columns: List[str], periods: List[int] = [1, 7]) -> pd.DataFrame:
        """
        添加百分比变化特征
        
        计算相对于前 N 期的百分比变化
        
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
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for period in periods:
                feature_name = f'{col}_pct_change_{period}'
                df_enhanced[feature_name] = df[col].pct_change(period)
                self.generated_features.append(feature_name)
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个百分比变化特征。")
        return df_enhanced
    
    def add_time_since_features(self, df: pd.DataFrame, column: str, events: List[str] = ['peak', 'trough']) -> pd.DataFrame:
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
            logger.warning(f"{self.log_prefix} 列 {column} 不存在。")
            return df_enhanced
        
        # 距离最近峰值的时间
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
        
        # 距离最近谷值的时间
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
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个距离关键事件的时间特征。")
        return df_enhanced
    
    def add_cyclical_features(self, df: pd.DataFrame, column: str="minute", period: int=15) -> pd.DataFrame:
        """
        添加周期性特征（正弦余弦编码）
        
        避免周期性特征的边界问题
        例如: 23 点和 0 点在数值上差距大，但实际很接近

        Args:
            df (pd.DataFrame): 数据框
            column (str): 列名
            period (int): 周期内的样本数量

        Returns:
            增强后的数据框
        """
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / period)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / period)
        self.generated_features.append(f"{column}_sin")
        self.generated_features.append(f"{column}_cos")
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个交互(交叉)特征。")
        return df
    
    def add_interaction_features(self, df: pd.DataFrame, column_pairs: List[tuple], operations: List[str] = ["add", "subtract", "multiply", "divide"]) -> pd.DataFrame:
        """
        添加交互(交叉)特征
        
        Args:
            df: 数据框
            column_pairs: 列对列表 [('col1', 'col2'), ...]
            operations: 操作列表 ["add", "subtract", "multiply", "divide"]
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加交互(交叉)特征...")
        df_enhanced = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col1} 或者 {col2} 不存在，跳过。")
                continue

            if 'add' in operations:
                feature_name = f'{col1}_add_{col2}'
                df_enhanced[feature_name] = df[col1] + df[col2]
                self.generated_features.append(feature_name)
            
            if 'subtract' in operations:
                feature_name = f'{col1}_substract_{col2}'
                df_enhanced[feature_name] = df[col1] - df[col2]
                self.generated_features.append(feature_name)
            
            if 'multiply' in operations:
                feature_name = f'{col1}_multiply_{col2}'
                df_enhanced[feature_name] = df[col1] * df[col2]
                self.generated_features.append(feature_name)
            
            if 'divide' in operations:
                feature_name = f'{col1}_divide_{col2}'
                df_enhanced[feature_name] = df[col1] / (df[col2] + 1e-8)  # 避免除零
                self.generated_features.append(feature_name)
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个交互(交叉)特征。")
        return df_enhanced
    
    def add_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
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
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for d in range(2, degree + 1):
                feature_name = f'{col}_pow_{d}'
                df_enhanced[feature_name] = df[col] ** d
                self.generated_features.append(feature_name)
        
        logger.info(f"{self.log_prefix} 生成 {len(self.generated_features)} 个多项式特征。")
        return df_enhanced
    
    def get_generated_features(self) -> List[str]:
        """
        获取所有生成的特征列表
        """
        return self.generated_features
    
    def reset(self):
        """
        重置生成的特征列表
        """
        self.generated_features = []



# 测试代码 main 函数
def main():
    # ------------------------------
    # 创建示例时间序列数据
    # ------------------------------
    np.random.seed(42)
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'load': np.random.randn(1000).cumsum() + 100,
        'temperature': 20 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000),
    })
    print(df)

    # ------------------------------
    # 创建特征工程器
    # ------------------------------
    fe = AdvancedFeatureEngineer()
    # ------------------------------
    # 添加各种特征
    # ------------------------------
    df = fe.add_rolling_statistics(df, ["load", "temperature"], windows=[3, 7, 24], stats=['mean', 'std', 'min', 'max', 'median', "skew", "kurt"])
    fe.reset()
    df = fe.add_expanding_statistics(df, ["load", "temperature"], stats=['mean', 'std', 'min', 'max', 'median', "skew", "kurt"])
    fe.reset()
    df = fe.add_diff_features(df, ['load'], periods=[1, 24])
    df = fe.add_interaction_features(df, [('load', 'temperature')], operations=['multiply'])
    
    print(f"原始特征数: 3")
    print(f"生成特征数: {len(fe.get_generated_features())}")
    print(f"总特征数: {len(df.columns)}")
    print(f"\n生成的特征列表:")
    for feat in fe.get_generated_features():#[:10]:
        print(f"  - {feat}")

    print(df)

if __name__ == "__main__":
    main()
