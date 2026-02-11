# -*- coding: utf-8 -*-

# ***************************************************
# * File        : basic_features.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
from typing import List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ------------------------------
# exogenous features engineering
# ------------------------------
def extend_datetime_feature(df: pd.DataFrame, feature_names: List[str], freq_minutes: int, n_per_day):
    """
    增加时间特征
    """
    df_copy = df.copy()
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
    for feature_name in feature_names:
        if feature_name in feature_map:
            df_copy[f"datetime_{feature_name}"] = df_copy["time"].apply(feature_map[feature_name])
    # 周期性特征 (将时间转换为可循环的 sin/cos 形式)
    if 'datetime_hour' in df_copy.columns and 'datetime_minute' in df_copy.columns:
        df_copy["datetime_minute_in_day"] = df_copy["datetime_hour"] * (60 / freq_minutes) + df_copy["datetime_minute"] / freq_minutes
        df_copy["datetime_minute_in_day_sin"] = np.sin(df_copy["datetime_minute_in_day"] * (2 * np.pi / n_per_day))
        df_copy["datetime_minute_in_day_cos"] = np.cos(df_copy["datetime_minute_in_day"] * (2 * np.pi / n_per_day))
    
    datetime_features = [col for col in df_copy.columns if col.startswith("datetime")]

    return df_copy, datetime_features

def extend_datetype_feature(df: pd.DataFrame, df_date: pd.DataFrame, col_ts: str="date", col_type: str="date_type"):
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

def extend_weather_feature(df: pd.DataFrame, df_weather: pd.DataFrame, col_ts: str):
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
            logger.warning(f"df_weather became empty after dropping NaNs.")
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

def extend_future_weather_feature(df: pd.DataFrame, df_weather: pd.DataFrame, col_ts: str):
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
            logger.warning(f"df_weather_future became empty after dropping NaNs.")
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
def extend_direct_multi_step_targets(df: pd.DataFrame, target: str, horizon: int):
    """
    Generates H shifted target columns for direct multi-step forecasting.
    """
    df_shifts = df.copy()
    shift_target_features = []
    # shift features building
    for h in range(0, horizon):
        shifted_col_name = f"{target}_shift_{h}"
        df_shifts[shifted_col_name] = df_shifts[target].shift(-h)
        shift_target_features.append(shifted_col_name)
    
    return df_shifts, shift_target_features

def extend_lag_feature_univariate(df: pd.DataFrame, target: str, lags: List[int]):
    """
    添加滞后特征(for univariate time series)
    """
    df_lags = df.copy()
    lag_features = []
    # lag features building
    for lag in lags:
        lag_col_name = f'{target}_lag_{lag}'
        df_lags[lag_col_name] = df_lags[target].shift(lag)
        lag_features.append(lag_col_name)
    
    return df_lags, lag_features

def extend_lag_feature_multivariate(df: pd.DataFrame, endogenous_features_with_target: List[str], lags: List[int]):
    """
    添加滞后特征 
    for multivariate time series, including targets for direct forecasting.
    endogenous_cols should include the primary target 'y' and other endogenous numeric features.
    """
    df_copy = df.copy()
    # 将 date 作为索引: Ensure 'time' is not in endogenous_cols before setting as index temporarily
    temp_df = df_copy.set_index("time").copy()
    
    all_lag_features = []
    available_endogenous_features = [col for col in endogenous_features_with_target if col in temp_df.columns]
    for col in available_endogenous_features:
        # Generate lags for X
        lags_X = [temp_df[col].shift(i) for i in lags]
        lag_col_names_X = [f'{col}_lag_{i}' for i in lags]
        
        # Add to df_copy
        for i, name in enumerate(lag_col_names_X):
            # Re-align to original df_copy index
            df_copy[name] = lags_X[i].values
            all_lag_features.append(name)
    
    return df_copy, all_lag_features




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
