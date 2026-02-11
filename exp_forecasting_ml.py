# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_forecasting_ml.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 2.0
# * Description : 基于LightGBM的时间序列预测框架
# *               支持以下预测方法:
# *               1. USMDO - 单变量多步直接输出预测
# *               2. USMD  - 单变量多步直接预测
# *               3. USMR  - 单变量多步递归预测
# *               4. USMDR - 单变量多步直接递归预测
# *               5. MSMD  - 多变量多步直接预测
# *               6. MSMR  - 多变量多步递归预测
# *               7. MSMDR - 多变量多步直接递归预测
# * Requirement : lightgbm, xgboost, catboost, scikit-learn, pandas, numpy
# ***************************************************

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import copy
import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# model
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
# model evaluation
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)
# data processing
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

from config.model_config import ModelConfig_univariate, ModelConfig_multivariate
from features.AdvancedFeatures import FeaturePreprocessor
from models.ModelFactory import ModelFactory

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


class Model:
    """
    基于机器学习模型的时间序列预测模型类
    """
    def __init__(self, args):
        """
        初始化模型
        """
        self.args = args
        self.setting = f"{self.args.model_name}-{self.args.data}-{self.args.pred_method}"
        self.log_prefix = f"[{self.args.model_name}-{self.args.data}]"
        # ------------------------------
        # 数据参数
        # ------------------------------
        # 数据读取路径
        self.args.data_dir = Path(self.args.data_dir)
        # 目标时间序列每天样本数量
        self.n_per_day = int(24 * 60 / self.args.freq_minutes)
        # 时间序列历史数据开始时刻
        start_time = self.args.now_time.replace(hour=0) - datetime.timedelta(days=self.args.history_days)
        # 时间序列当前时刻(模型预测的日期时间)
        now_time = self.args.now_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
        # 时间序列未来结束时刻
        future_time = self.args.now_time + datetime.timedelta(days=self.args.predict_days)
        # 数据划分时间戳
        self.train_start_time = start_time
        self.train_end_time = now_time
        self.forecast_start_time = now_time
        self.forecast_end_time = future_time
        # ------------------------------
        # 特征工程
        # ------------------------------
        # 特征滞后数个数(1,2,...)
        self.n_lags = len(self.args.lags)
        # ------------------------------
        # 模型训练
        # ------------------------------
        self.model_factory = ModelFactory()
        # ------------------------------
        # 模型测试
        # ------------------------------ 
        # 测试窗口数据长度(训练+测试)
        self.window_len = int(self.args.window_days * self.n_per_day)
        # 测试滑动窗口数量, >=1, 1: 单个窗口
        self.n_windows = int(self.args.history_days * self.n_per_day - self.window_len - self.horizon + 1) // self.horizon
        # ------------------------------
        # 模型预测
        # ------------------------------
        # 预测未来 1 天(24小时)的数据/数据划分长度/预测数据长度
        self.horizon = int(self.args.predict_days * self.n_per_day)
        # ------------------------------
        # 模型训练、测试、预测结果保存路径
        # ------------------------------
        self.args.checkpoints_dir = Path(self.args.checkpoints_dir).joinpath(self.setting)
        self.args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.args.test_results_dir = Path(self.args.test_results_dir).joinpath(self.setting)
        self.args.test_results_dir.mkdir(parents=True, exist_ok=True)
        self.args.pred_results_dir = Path(self.args.pred_results_dir).joinpath(self.setting)
        self.args.pred_results_dir.mkdir(parents=True, exist_ok=True)
        # ------------------------------
        # 日志打印
        # ------------------------------ 
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Prepare params...")
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} history data range: {self.train_start_time}~{self.train_end_time}")
        logger.info(f"{self.log_prefix} predict data range: {self.forecast_start_time}~{self.forecast_end_time}")
    # ##############################
    # Data load
    # ##############################
    def load_data(self) -> Dict:
        """
        加载所有必要的数据
        
        Returns:
            包含目标序列、日期类型、天气等数据的字典
        """
        logger.info(f"{self.log_prefix} Loading data from {self.args.data_dir}")
        
        input_data = {}
        
        # 加载目标时间序列数据
        target_data_path = self.args.data_dir / self.args.data_path
        if target_data_path.exists():
            df_target = pd.read_csv(target_data_path)
            # df_target[self.args.target_ts_feat] = pd.to_datetime(df_target[self.args.target_ts_feat])
            input_data["target_series"] = df_target
            logger.info(f"{self.log_prefix} Target series loaded: {df_target.shape}")
        else:
            logger.error(f"{self.log_prefix} Target data not found at {target_data_path}")
            raise FileNotFoundError(f"Target data not found at {target_data_path}")
        
        # 加载历史日期类型数据
        if self.args.date_history_path:
            date_history_path = self.args.data_dir / self.args.date_history_path
            if date_history_path.exists():
                df_date_history = pd.read_csv(date_history_path)
                # df_date_history[self.args.date_ts_feat] = pd.to_datetime(df_date_history[self.args.date_ts_feat])
                input_data["date_history"] = df_date_history
                logger.info(f"{self.log_prefix} Date history loaded: {df_date_history.shape}")
        
        # 加载未来日期类型数据
        if self.args.date_future_path:
            date_future_path = self.args.data_dir / self.args.date_future_path
            if date_future_path.exists():
                df_date_future = pd.read_csv(date_future_path)
                # df_date_future[self.args.date_ts_feat] = pd.to_datetime(df_date_future[self.args.date_ts_feat])
                input_data["date_future"] = df_date_future
                logger.info(f"{self.log_prefix} Date future loaded: {df_date_future.shape}")
        
        # 加载历史天气数据
        if self.args.weather_history_path:
            weather_history_path = self.args.data_dir / self.args.weather_history_path
            if weather_history_path.exists():
                df_weather_history = pd.read_csv(weather_history_path)
                # df_weather_history[self.args.weather_ts_feat] = pd.to_datetime(df_weather_history[self.args.weather_ts_feat])
                input_data["weather_history"] = df_weather_history
                logger.info(f"{self.log_prefix} Weather history loaded: {df_weather_history.shape}")
        
        # 加载未来天气数据
        if self.args.weather_future_path:
            weather_future_path = self.args.data_dir / self.args.weather_future_path
            if weather_future_path.exists():
                df_weather_future = pd.read_csv(weather_future_path)
                # df_weather_future[self.args.weather_ts_feat] = pd.to_datetime(df_weather_future[self.args.weather_ts_feat])
                input_data["weather_future"] = df_weather_future
                logger.info(f"{self.log_prefix} Weather future loaded: {df_weather_future.shape}")
        # ------------------------------
        # 数据合并
        # ------------------------------
        if self.args.date_history_path and self.args.date_future_path:
            df_date_all = pd.concat([df_date_history.iloc[:-1,], df_date_future], axis=0)
        else:
            df_date_all = None
        
        if self.args.weather_history_path and self.args.weather_future_path:
            df_weather_all = pd.concat([df_weather_history.iloc[:-1,], df_weather_future], axis=0)
        else:
            df_weather_all = None
        input_data["date_history"] = df_date_all
        input_data["date_future"] = df_date_all
        input_data["weather_history"] = df_weather_all
        input_data["weather_future"] = df_weather_all
        
        return input_data
    # ##############################
    # Data Preprocessing
    # ##############################
    def __process_df_timestamp(self, df: pd.DataFrame, col_ts: str):
        """
        时序数据时间特征预处理

        Args:
            df (pd.DataFrame): 时间序列数据
            col_ts (str): 原时间戳列
        """
        if df is not None:
            # 数据拷贝
            df_processed = copy.deepcopy(df)
            # 转换时间戳类型
            df_processed[col_ts] = pd.to_datetime(df_processed[col_ts])
            # del df_processed[ts_col]
            # 去除重复时间戳
            df_processed.drop_duplicates(subset=col_ts, keep="last", inplace=True, ignore_index=True)
            return df_processed
        else:
            return df

    def __process_target_series(self, df_template: pd.DataFrame, df_series: pd.DataFrame, col_ts: str, col_numeric: List, col_categorical: List, col_drop: List):
        """
        目标特征数据预处理
        df_template: ["time"]
        """
        df_template_copy = df_template.copy()
        if df_series is not None:
            # 目标特征数据转换为浮点数
            if self.args.target in df_series.columns:
                df_series[self.args.target] = df_series[self.args.target].apply(lambda x: float(x))
                df_template_copy["y"] = df_template_copy["time"].map(df_series.set_index(col_ts)[self.args.target])
                target_feature = "y"
            else:
                target_feature = None
            # 除目标特征外的其他数值类型的内生变量处理
            filtered_col_numeric = [col for col in col_numeric if col not in [col_ts, self.args.target] + col_categorical + col_drop]
            for col in filtered_col_numeric:
                if col in df_series.columns:
                    df_series[col] = df_series[col].apply(lambda x: float(x))
                    df_template_copy[col] = df_template_copy["time"].map(df_series.set_index(col_ts)[col])
            # TODO 类别类型的内生变量处理
            filtered_col_categorical = [col for col in col_categorical if col not in [col_ts, self.args.target] + col_numeric + col_drop]
            for col in filtered_col_categorical:
                if col in df_series.columns:
                    df_series[col] = df_series[col].apply(lambda x: str(x))
                    df_template_copy[col] = df_template_copy["time"].map(df_series.set_index(col_ts)[col])
            # 内生变量(Endogenous variable)
            endogenous_features = [col for col in df_template_copy.columns if col not in ["time"]]
            if target_feature and target_feature in endogenous_features:
                 # Remove target from here for consistency, will be handled separately
                 endogenous_features.remove(target_feature)
        else:
            endogenous_features = []
            target_feature = None
        
        return df_template_copy, endogenous_features, target_feature
    
    def process_history_data(self, input_data: Dict):
        """
        历史数据预处理
        """
        # ------------------------------
        # 类别特征收集器
        # ------------------------------
        categorical_features = []
        # ------------------------------
        # 历史数据时间戳
        # ------------------------------
        df_history_template = pd.DataFrame({
            "time": pd.date_range(self.train_start_time, self.train_end_time, freq=self.args.freq, inclusive="left"),
        })
        logger.info(f"{self.log_prefix} template df_history_template: \n{df_history_template.head()}")
        # ------------------------------
        # 数据预处理：目标时间序列特征
        # ------------------------------
        df_history_series = self.__process_df_timestamp(df=input_data["target_series"], col_ts=self.args.target_ts_feat)
        df_history, other_endogenous_features, target_feature = self.__process_target_series(
            df_template=df_history_template,
            df_series=df_history_series,
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        logger.info(f"{self.log_prefix} after process_target_series df_history: \n{df_history.head()}")
        # 所有内生变量(包含目标特征 y)
        endogenous_features_with_target = [target_feature] + other_endogenous_features if target_feature else other_endogenous_features
        # ------------------------------
        # 特征工程：日期类型(节假日、特殊事件)特征
        # ------------------------------
        df_date_history = self.__process_df_timestamp(df=input_data[f"date_history"], col_ts=self.args.date_ts_feat)
        df_history, date_features = self.extend_datetype_feature(
            df=df_history,
            df_date=df_date_history,
            col_ts=self.args.date_ts_feat,
        )
        if date_features:
            categorical_features.extend(self.args.datetype_categorical_features)
        logger.info(f"{self.log_prefix} after extend_datetype_feature df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} after extend_datetype_feature date_features: {date_features}")
        # ------------------------------
        # 特征工程：天气特征
        # ------------------------------
        df_weather_history = self.__process_df_timestamp(df=input_data[f"weather_history"], col_ts=self.args.weather_ts_feat)
        df_history, weather_features = self.extend_weather_feature(
            df=df_history,
            df_weather=df_weather_history,
            col_ts=self.args.weather_ts_feat,
        )
        # Assuming weather features are numeric
        if weather_features:
            categorical_features.extend(self.args.weather_categorical_features)
        logger.info(f"{self.log_prefix} after extend_weather_feature df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} after extend_weather_feature weather_features: {weather_features}")
        # ------------------------------
        # 外生变量(包含：日期类型(节假日、特殊事件)特征、气象特征)
        # ------------------------------
        exogenous_features = date_features + weather_features
        # ------------------------------
        # 插值填充预测缺失值
        # ------------------------------
        # Interpolate existing data, then drop any remaining NaNs (e.g., at boundaries)
        df_history = df_history.interpolate(method="linear", limit_direction="both")
        df_history.dropna(inplace=True, ignore_index=True) # Drops rows where even interpolation couldn't help
        logger.info(f"{self.log_prefix} after interpolate and dropna df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} endogenous_features_with_target: {endogenous_features_with_target}")
        logger.info(f"{self.log_prefix} exogenous_features: {exogenous_features}")
        logger.info(f"{self.log_prefix} target_features: {target_feature}")
        logger.info(f"{self.log_prefix} categorical_features: {categorical_features}")
        
        return (df_history, endogenous_features_with_target, exogenous_features, target_feature, categorical_features)
    
    def process_future_data(self, input_data: Dict):
        """
        处理未来数据
        """
        # ------------------------------
        # 未来数据时间戳
        # ------------------------------
        df_future_template = pd.DataFrame({
            "time": pd.date_range(self.forecast_start_time, self.forecast_end_time, freq=self.args.freq, inclusive="left")
        })
        logger.info(f"{self.log_prefix} template df_future_template: \n{df_future_template.head()}")
        """
        # ------------------------------
        # 数据预处理：目标时间序列特征
        # ------------------------------
        df_future_series = self.__process_df_timestamp(
            df=input_data["df_future_series"],
            col_ts=self.args.target_ts_feat
        )
        df_future, other_endogenous_features, target_feature = self.__process_target_series(
            df_template=df_future_template,
            df_series=df_future_series, # This will be None, so df_future_template is returned.
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        logger.info(f"{self.log_prefix} after process_target_series df_future: \n{df_future.head()}")
        # 所有内生变量(没有目标特征 y及其衍生特征)
        future_endogenous_cols_for_lag = other_endogenous_features
        """
        # ------------------------------
        # 特征工程：日期类型(节假日、特殊事件)特征
        # ------------------------------
        df_date_future = self.__process_df_timestamp(df=input_data[f"date_future"], col_ts=self.args.date_ts_feat)
        df_future, date_features = self.extend_datetype_feature(
            df=df_future_template,
            df_date=df_date_future,
            col_ts=self.args.date_ts_feat,
        )
        logger.info(f"{self.log_prefix} after extend_datetype_feature df_future: \n{df_future.head()}")
        logger.info(f"{self.log_prefix} after extend_datetype_feature date_features: {date_features}")
        # ------------------------------
        # 特征工程：天气特征
        # ------------------------------
        df_weather_future = self.__process_df_timestamp(df=input_data[f"weather_future"], col_ts=self.args.weather_ts_feat)
        df_future, weather_features = self.extend_future_weather_feature(
            df=df_future,
            df_weather=df_weather_future,
            col_ts=self.args.weather_ts_feat,
        )
        logger.info(f"{self.log_prefix} after extend_future_weather_feature df_future: \n{df_future.head()}")
        logger.info(f"{self.log_prefix} after extend_future_weather_feature weather_features: {weather_features}")
        # ------------------------------
        # 外生变量(包含：日期类型(节假日、特殊事件)特征、气象特征)
        # ------------------------------
        future_exogenous_features = date_features + weather_features
        # ------------------------------
        # 插值填充预测缺失值
        # ------------------------------
        # Interpolate existing data, then drop any remaining NaNs (e.g., at boundaries)
        df_future = df_future.interpolate(method="linear", limit_direction="both")
        df_future.dropna(inplace=True, ignore_index=True) # Drops rows where even interpolation couldn't help
        logger.info(f"{self.log_prefix} after interpolate and dropna df_future: \n{df_future.head()}")
        # logger.info(f"{self.log_prefix} future_endogenous_cols_for_lag: {future_endogenous_cols_for_lag}")
        logger.info(f"{self.log_prefix} future_exogenous_features: {future_exogenous_features}")
        # logger.info(f"{self.log_prefix} target_feature: {target_feature}")

        return df_future#, future_endogenous_cols_for_lag, future_exogenous_features
    # ##############################
    # Feature Engineering
    # ##############################
    # ------------------------------
    # exogenous features engineering
    # ------------------------------
    def extend_datetime_feature(self, df: pd.DataFrame, feature_names: List[str], freq_minutes: int):
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
            df_copy["datetime_minute_in_day_sin"] = np.sin(df_copy["datetime_minute_in_day"] * (2 * np.pi / self.n_per_day))
            df_copy["datetime_minute_in_day_cos"] = np.cos(df_copy["datetime_minute_in_day"] * (2 * np.pi / self.n_per_day))
        
        datetime_features = [col for col in df_copy.columns if col.startswith("datetime")]

        return df_copy, datetime_features

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
    
    def extend_lag_feature_univariate(self, df: pd.DataFrame, target: str, lags: List[int]):
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

    def extend_lag_feature_multivariate(self, df: pd.DataFrame, endogenous_features_with_target: List[str], lags: List[int]):
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
    # ------------------------------
    # feature engineering summary
    # ------------------------------
    def create_features(self, df_series: pd.DataFrame, endogenous_features_with_target: List[str], exogenous_features: List[str], target_feature: str, categorical_features: List[str]):
        """
        特征工程
        """
        # df_series copy
        df_series_copy = df_series.copy()
        # For multi-output recursive, we need lags for ALL endogenous variables.
        all_endogenous_for_lags = endogenous_features_with_target
        # Clear and re-populate categorical_features for each run to avoid duplicates
        categorical_features_copy = categorical_features.copy() 
        # ------------------------------
        # 特征工程：日期时间特征
        # ------------------------------
        df_series_copy, datetime_features = self.extend_datetime_feature(
            df=df_series_copy,
            feature_names=self.args.datetime_features,
            freq_minutes=self.args.freq_minutes,
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
                endogenous_features_with_target = all_endogenous_for_lags,
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
                endogenous_features_with_target = all_endogenous_for_lags,
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
                endogenous_features_with_target = all_endogenous_for_lags,
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
            if col not in target_output_features + lag_features + all_endogenous_for_lags
        ]
        predictor_features.extend(current_endogenous_as_features)
        # Filter df_series_copy to only include necessary columns to avoid errors later
        predictor_features = sorted(set(predictor_features), key=predictor_features.index)
        all_cols_needed = ["time"] + predictor_features + target_output_features
        df_series_copy = df_series_copy[[col for col in all_cols_needed if col in df_series_copy.columns]]

        return df_series_copy, predictor_features, target_output_features, categorical_features_copy
    # ##############################
    # Model Testing
    # ##############################
    # ------------------------------
    # Model sliding window testing
    # ------------------------------
    def _evaluate_split_index(self, window: int, total_data_points: int):
        """
        数据分割索引构建
        Calculates train/test split indices for a sliding window.
        Assumes total_data_points is the length of `df_history_featured` after dropna,
        so `self.horizon` refers to the number of samples in the test set.
        The window slides from the most recent data backwards.
        """
        # Calculate test start/end index
        test_end = total_data_points - 1 - (self.horizon * (window - 1))
        test_start = test_end - self.horizon + 1
        # Calculate train start/end index
        train_end = test_start
        train_start = train_end - (self.window_len - self.horizon)
        train_start = max(0, train_start)

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X: pd.DataFrame, data_Y: pd.Series, df_history: pd.DataFrame, window: int):
        """
        训练、测试数据集分割
        """
        logger.info(f"{self.log_prefix} Model Testing sliding window...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        # 数据分割指标
        total_data_points = len(data_X)
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window, total_data_points)
        logger.info(f"{self.log_prefix} split indexes:: [train_start:train_end]: [{train_start}:{train_end}]")
        logger.info(f"{self.log_prefix} split indexes:: [test_start:test_end]: [{test_start}:{test_end+1}]")

        # 数据分割(Data slicing, handle cases where indices might be out of bounds for the window)
        if train_start >= train_end or test_start >= test_end + 1 or train_start < 0 or test_end >= total_data_points:
            logger.warning(f"{self.log_prefix} Insufficient data for window {window} (train_start={train_start}, train_end={train_end}, test_start={test_start}, test_end={test_end}). Skipping this window.")
            return None, None, None, None, None, None

        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        X_test = data_X.iloc[test_start:test_end+1] # +1 to include test_end
        Y_test = data_Y.iloc[test_start:test_end+1]
        df_history_train = df_history.iloc[train_start:train_end]
        df_history_test = df_history.iloc[test_start:test_end+1]
        logger.info(f"{self.log_prefix} X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}")
        logger.info(f"{self.log_prefix} X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}")
        logger.info(f"{self.log_prefix} df_history_train.shape: {df_history_train.shape}, df_history_test.shape: {df_history_test.shape}")

        if X_train.empty or Y_train.empty or X_test.empty or Y_test.empty:
            logger.warning(f"{self.log_prefix} Empty dataframe in window {window} split. Skipping.")
            return None, None, None, None, None, None
        
        return X_train, Y_train, X_test, Y_test, df_history_train, df_history_test

    def _evaluate_score(self, y_test: np.ndarray, y_pred: np.ndarray, window: int, df_history_test: pd.DataFrame):
        """
        模型评估
        计算模型的性能指标
        """
        # Ensure y_test and y_pred are 1D arrays for metrics
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        # Handle potential division by zero in MAPE if y_test contains zeros
        y_test_mape = np.where(y_test == 0, 0.01, y_test) # Avoid division by zero, small epsilon
        # Calculate the model's performance metrics
        test_scores = {
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test_mape, y_pred),
            "MAPE Accuracy": 1 - mean_absolute_percentage_error(y_test_mape, y_pred),
        }
        test_scores_df = pd.DataFrame(test_scores, index=[window])
        test_scores_df["time_range"] = f"{df_history_test['time'].min()}~{df_history_test['time'].max()}"
        test_scores_df = test_scores_df[["time_range"] + list(test_scores.keys())]
        logger.info(f"{self.log_prefix} test_scores_df: \n{test_scores_df}")
        
        return test_scores_df

    def _evaluate_result(self, y_test: np.ndarray, y_pred: np.ndarray, window: int, cv_timestamp_df: pd.DataFrame):
        """
        测试集预测数据
        """
        # Ensure y_test and y_pred are 1D arrays
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()

        # Data collection for plot
        cv_plot_df_window = pd.DataFrame()
        
        total_data_points_ts_df = len(cv_timestamp_df)
        _, _, test_start_ts_idx, test_end_ts_idx = self._evaluate_split_index(window, total_data_points_ts_df)
        
        # Ensure the slice is valid and matches the length of y_pred/y_test
        time_slice = cv_timestamp_df["time"].iloc[test_start_ts_idx:test_end_ts_idx + 1]
        if len(time_slice) != len(y_pred):
            logger.warning(f"Length mismatch for plotting data: time_slice ({len(time_slice)}) vs y_pred ({len(y_pred)}). Adjusting to min length.")
            min_len = min(len(time_slice), len(y_pred))
            cv_plot_df_window["time"] = time_slice.iloc[:min_len].values
            cv_plot_df_window["Y_trues"] = y_test[:min_len]
            cv_plot_df_window["Y_preds"] = y_pred[:min_len]
        else:
            cv_plot_df_window["time"] = time_slice.values
            cv_plot_df_window["Y_trues"] = y_test
            cv_plot_df_window["Y_preds"] = y_pred
        
        return cv_plot_df_window

    def _calc_features_corr(self, df: pd.DataFrame, train_features: List[str]):
        """
        分析预测特征与目标特征的相关性
        """
        # Ensure 'load' is target_feature for this function, assuming it's the target.
        if self.args.target in df.columns:
            features_corr = df[train_features + [self.args.target]].corr()
        else:
            logger.warning(f"{self.log_prefix} Target feature '{self.args.target}' not found in DataFrame for correlation calculation.")
            features_corr = df[train_features].corr()
        
        return features_corr
    # ------------------------------
    # Model results save
    # ------------------------------
    def test_results_save(self, test_scores_df, cv_plot_df):
        # 测试结果数据保存
        test_scores_df.to_csv(self.args.test_results_dir.joinpath("test_scores_df.csv"), index=False, encoding="utf-8")
        cv_plot_df.to_csv(self.args.test_results_dir.joinpath("cv_plot_df.csv"), index=False, encoding="utf-8")
        # 测试结果数据可视化
        if len(cv_plot_df["Y_preds"].values) == 0 or len(cv_plot_df["Y_trues"].values) == 0:
            logger.warning(f"{self.log_prefix} No data to visualize for test prediction.")
            return
        # 画布
        plt.figure(figsize=(25, 8))
        # 创建折线图
        plt.plot(cv_plot_df["Y_trues"].values, label='Trues', lw=1.7, )
        plt.plot(cv_plot_df["Y_preds"].values, label='Preds', lw=1.7, ls="-.")
        # 增强视觉效果
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title('Trues and Preds Timeseries Plot')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.args.test_results_dir.joinpath("test_prediction.png"), bbox_inches='tight', dpi=300)
        # plt.show();
    # ------------------------------
    # Model testing
    # ------------------------------
    def _window_test(self, 
                     X_train, Y_train, 
                     X_test, Y_test, 
                     df_history_train, df_history_test, 
                     endogenous_features, exogenous_features, 
                     target_feature, target_output_features, 
                     categorical_features):
        """
        模型滑窗测试
        """
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{self.log_prefix} Model Testing training start...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        # 创建特征预处理器
        self.preprocessor_testing = FeaturePreprocessor(self.args, log_prefix=self.log_prefix)
        model = self.train(X_train, Y_train, self.preprocessor_testing, categorical_features)
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{self.log_prefix} Model Testing forecasting start...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        Y_pred = None
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            Y_pred = self.univariate_single_multi_step_direct_output_forecast(
                model = model,
                X_test = X_test.copy(),
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_testing,
            )
        elif self.args.pred_method == "univariate-single-multistep-direct":
            Y_pred = self.univariate_single_multi_step_direct_forecast(
                model = model,
                df_history = df_history_train,
                df_future = df_history_test,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                categorical_features = categorical_features,
                # target_output_features = target_output_features,
                preprocessor = self.preprocessor_testing,
            )
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            Y_pred = self.univariate_single_multi_step_recursive_forecast(
                model = model,
                df_history = df_history_train,
                df_future = df_history_test,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features = target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_testing,
            )
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            Y_pred = self.univariate_single_multi_step_direct_recursive_forecast(
                model = model,
                df_history = df_history_train,
                df_future = df_history_test,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features = target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_testing,
            )
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            Y_pred = self.multivariate_single_multi_step_direct_forecast(
                model = model,
                df_history = df_history_train,
                df_future = df_history_test,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features = target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_testing,
            )
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            Y_pred = self.multivariate_single_multi_step_recursive_forecast(
                model = model,
                df_history = df_history_train,
                df_future = df_history_test,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                target_output_features = target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_testing,
            )
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            Y_pred = self.multivariate_single_multi_step_direct_recursive_forecast(
                model = model,
                df_history = df_history_train,
                df_future = df_history_test,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features = target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_testing,
            )
        # Return empty array if prediction fails or is empty
        if Y_pred is None or len(Y_pred) == 0:
            logger.error(f"{self.log_prefix} Prediction failed or returned empty for method: {self.args.pred_method}. Returning empty array.")
            return np.array([])

        return Y_pred

    def test(self, 
             df_history, X_train_history, Y_train_history, 
             endogenous_features_with_target, exogenous_features, 
             target_feature, target_output_features, categorical_features):
        """
        模型滑窗测试
        """
        # 模型测试结果收集
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        # Max number of windows to run, ensuring enough data for at least one full test horizon
        if self.n_windows <= 0:
            logger.warning(f"{self.log_prefix} Not enough data for testing with current window configuration (Total X points: {len(X_train_history)}")
            logger.warning(f"{self.log_prefix} Window length: {self.window_len}, Horizon: {self.horizon}). No tests will be performed.")
            return test_scores_df, cv_plot_df
        # Create full timestamp df once for evaluation plotting
        cv_timestamp_full_df = pd.DataFrame({"time": pd.date_range(self.train_start_time, self.train_end_time, freq=self.args.freq, inclusive="left")})
        # 模型滑窗测试(Model sliding window test)
        for window in range(1, int(self.n_windows + 1)):
            logger.info(f"{self.log_prefix} {'-' * 40}")
            logger.info(f"{self.log_prefix} Model Testing window: {window}...")
            logger.info(f"{self.log_prefix} {'-' * 40}")
            # 数据分割: 训练集、测试集
            (X_train, Y_train, X_test, Y_test, df_history_train, df_history_test) = self._evaluate_split(
                X_train_history, Y_train_history, df_history, window
            )
            if X_train is None: continue # Skip if split was invalid
            # 目标特征处理(Ensure Y_train is DataFrame for MultiOutputRegressor)
            Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
            Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
            # 模型测试
            Y_pred = self._window_test(
                X_train, Y_train,
                X_test, Y_test,
                df_history_train, df_history_test,
                endogenous_features_with_target, exogenous_features, 
                target_feature, target_output_features, 
                categorical_features
            )
            if len(Y_pred) == 0: # If _window_test returned empty predictions
                logger.warning(f"{self.log_prefix} Skipping evaluation for window {window} due to empty predictions.")
                continue
            # Process Y_test and Y_pred for evaluation. We always evaluate the primary target's first step prediction.
            # Y_test for evaluation should always be the actuals for target_t+1.
            # Assuming the primary target (y) shifted by 1 is always the first column of Y_test
            Y_test_for_eval = Y_test.iloc[:, 0].values
            # Ensure Y_pred matches length of Y_test_for_eval
            if len(Y_pred) != len(Y_test_for_eval):
                logger.warning(f"Length mismatch: Y_pred ({len(Y_pred)}) vs Y_test_for_eval ({len(Y_test_for_eval)}) in window {window}. Truncating Y_pred.")
                Y_pred = Y_pred[:len(Y_test_for_eval)]
            # 测试集评价指标
            eval_scores_window = self._evaluate_score(Y_test_for_eval, Y_pred, window, df_history_test)
            test_scores_df = pd.concat([test_scores_df, eval_scores_window], axis=0)
            # 测试集预测数据
            cv_plot_df_window = self._evaluate_result(Y_test_for_eval, Y_pred, window, cv_timestamp_full_df)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis=0)
            # TODO localtest
            break
        # 模型测试评价指标数据处理
        if not test_scores_df.empty:
            test_scores_df_mean = test_scores_df.drop(columns=["time_range"]).mean()
            test_scores_df_mean = test_scores_df_mean.to_frame().T.reset_index(drop=True, inplace=False)
            test_scores_df_mean["time_range"] = "均值"
            test_scores_df = pd.concat([test_scores_df, test_scores_df_mean], axis=0)
        # 模型结果保存
        logger.info(f"{self.log_prefix} {'-' * 40}")
        logger.info(f"{self.log_prefix} Model Testing result...")
        logger.info(f"{self.log_prefix} {'-' * 40}")
        logger.info(f"{self.log_prefix} Model Testing test_scores_df: \n{test_scores_df}")
        logger.info(f"{self.log_prefix} Model Testing cv_plot_df: \n{cv_plot_df.head()}")
        
        return test_scores_df, cv_plot_df
    # ##############################
    # Model Hyperparameters tuning and Model training
    # ##############################
    # Model Hyperparameters tuning
    def _hyperparameters_tuning(self, X_train, Y_train):
        """
        模型超参数调优 (Grid Search / Randomized Search with TimeSeriesSplit)
        """
        logger.info(f"{self.log_prefix} Starting hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            'estimator__num_leaves': [15, 31, 63],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__feature_fraction': [0.7, 0.8, 0.9],
            'estimator__lambda_l1': [0.1, 0.5, 1.0],
            'estimator__lambda_l2': [0.1, 0.5, 1.0],
            'estimator__min_child_samples': [20, 50, 100], # Corresponds to min_data_in_leaf
        }

        # Base LightGBM estimator
        lgbm_base = self.model_factory.create_model(self.args.model_type, self.args.model_params)

        # Wrap in MultiOutputRegressor if the method is multi-output
        if self.args.pred_method in ["univariate-multi-step-direct", "multivariate-multi-step-direct", "multivariate-multi-step-recursive"]:
            model_for_tuning = MultiOutputRegressor(lgbm_base)
        else: # "univariate-multi-step-recursive"
            model_for_tuning = lgbm_base

        # TimeSeriesSplit for cross-validation
        # n_splits determines how many train-test splits to generate.
        # The test set size will be at least self.horizon.
        tscv = TimeSeriesSplit(n_splits=self.args.tuning_n_splits)

        # Use GridSearchCV for exhaustive search or RandomizedSearchCV for faster search
        # RandomizedSearchCV is generally preferred for larger search spaces
        search = RandomizedSearchCV(
            estimator=model_for_tuning,
            param_distributions=param_grid,
            n_iter=10, # Number of parameter settings that are sampled
            scoring=self.args.tuning_metric,
            cv=tscv,
            verbose=1,
            n_jobs=-1, # Use all available cores
            random_state=42
        )
        
        search.fit(X_train, Y_train)

        logger.info(f"{self.log_prefix} Best hyperparameters found: {search.best_params_}")
        logger.info(f"{self.log_prefix} Best score: {search.best_score_}")

        # Update model_params with the best ones
        best_params_estimator = {k.replace('estimator__', ''): v for k, v in search.best_params_.items()}
        self.model_params.update(best_params_estimator)
        logger.info(f"{self.log_prefix} Model parameters updated with best tuning results.")
        
        return search.best_estimator_ # Return the best model direct
    
    # Model train results save
    def model_save(self, model):
        """
        模型保存
        """
        # model_deploy = ModelDeployPkl(save_file_path=self.args.checkpoints.joinpath("model.pkl"))
        # model_deploy.save_model(model)
        # logger.info(f"{self.log_prefix} Model saved to {model_dir.joinpath('model.pkl')}")
        pass
    # ------------------------------
    # Model training
    # ------------------------------
    def train(self, X_train, Y_train, preprocessor, categorical_features):
        """
        模型训练
        """
        # 训练集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        # ------------------------------
        # 归一化/标准化
        # ------------------------------
        # 特征预处理（训练模式）
        X_train_df_processed, actual_categorical = preprocessor.fit_transform(X_train_df, categorical_features)
        preprocessor.validate_features(X_train_df_processed, stage="training")
        # 根据编码策略决定是否传递 categorical_feature
        if self.args.encode_categorical_features:
            # 已编码为整数，不传递 categorical_feature
            lgbm_categorical = None
        else:
            # 未编码，传递 categorical_feature 让 LightGBM 处理
            lgbm_categorical = actual_categorical
        logger.info(f"{self.log_prefix} lgbm_categorical: {lgbm_categorical}")
        # ------------------------------
        # TODO Hyperparameter tuning (if enabled)
        # ------------------------------
        if self.args.perform_tuning:
            best_model = self._hyperparameters_tuning(X_train_df_processed, Y_train_df)
            return best_model
        # ------------------------------
        # 模型训练
        # ------------------------------
        # model estimator
        lgbm_estimator = self.model_factory.create_model(self.args.model_type, self.args.model_params)
        # model training
        if Y_train_df.shape[1] == 1:
            model = lgbm_estimator
            model.fit(X_train_df_processed, Y_train_df)
            logger.info(f"{self.log_prefix} Training single output LGBMRegressor")
        elif Y_train_df.shape[1] > 1:
            model = MultiOutputRegressor(estimator=lgbm_estimator)
            model.fit(X_train_df_processed, Y_train_df)
            logger.info(f"{self.log_prefix} Training MultiOutputRegressor with {Y_train.shape[1]} outputs")
        logger.info(f"{self.log_prefix} Model training completed!")

        return model
    # ##############################
    # Model Forecast(Model Inference)
    # ##############################
    # ------------------------------
    # 单变量（目标变量滞后特征）预测单变量（目标变量）
    # ------------------------------
    def univariate_single_multi_step_direct_output_forecast(self, model, X_test, categorical_features, preprocessor):
        """
        单变量多步直接输出预测
        [datetype, datetime, weather] -> [Y]
        """
        # 特征预处理（预测模式）
        X_test_processed = preprocessor.transform(X_test, categorical_features)
        preprocessor.validate_features(X_test_processed, stage="prediction")
        # 模型推理
        if len(X_test) > 0:
            Y_pred = model.predict(X_test_processed)
        else:
            Y_pred = []
            logger.info(f"{self.log_prefix} X_future length is 0!")
        
        return Y_pred

    def univariate_single_multi_step_direct_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, preprocessor):
        """
        单变量多步直接预测(USMD): 使用目标变量滞后 + 外生变量预测未来多步
        [Y, datetime, weather, datetype]
        """
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        # Need enough history to form the maximum lag
        max_lag = max(self.args.lags) if self.args.lags else 1
        # Combine last relevant history for lags and the first future point for exogenous/datetime
        relevant_history_for_lags = df_history.iloc[-max_lag:].copy()
        # Use the first point of df_future_for_prediction for exogenous features
        forecast_start_point_exogenous = df_future.iloc[0:1].copy()
        # For lag generation, combined_for_features needs a 'time' column for features
        combined_for_features = pd.concat([relevant_history_for_lags, forecast_start_point_exogenous], ignore_index=True)
        # Generate features for this combined data, then take the last row as input for prediction
        combined_featured, predictor_features_for_forecast, _, categorical_features = self.create_features(
            df_series = combined_for_features,
            endogenous_features_with_target = endogenous_features,
            exogenous_features = available_exogenous,
            target_feature = target_feature,
            categorical_features = categorical_features,
        )
        # The actual features for prediction are the last row (corresponding to the moment before forecast starts)
        X_forecast_input = combined_featured[predictor_features_for_forecast].iloc[-1:]
        
        # 特征预处理（预测模式）
        X_forecast_input_processed = preprocessor.transform(X_forecast_input, categorical_features)
        preprocessor.validate_features(X_forecast_input_processed, stage="prediction")
        
        # Expects (1, H) output -> [H] array
        Y_pred_multi_step = model.predict(X_forecast_input_processed)[0]
        
        # Assign predictions to df_future_for_prediction
        if len(Y_pred_multi_step) >= len(df_future):
            Y_pred = Y_pred_multi_step[:len(df_future)]
        else:
            logger.warning(f"Predicted {len(Y_pred_multi_step)} steps but df_future requires {len(df_future)} rows. Padding predictions.")
            Y_pred = np.pad(Y_pred_multi_step, (0, len(df_future) - len(Y_pred_multi_step)), 'edge')
        
        return Y_pred

    def univariate_single_multi_step_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, preprocessor):
        """
        单变量递归多步预测(USMR)
        """
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        # 多步预测值收集器
        y_preds = []
        # Need enough history to form the maximum lag.
        max_lag = max(self.args.lags) if self.args.lags else 1
        # Start with the latest available data to construct initial features
        last_known_data = df_history.iloc[-max_lag:].copy()
        # Ensure target_feature is present in last_known_data
        if target_feature not in last_known_data.columns and target_feature in df_history.columns:
            last_known_data[target_feature] = df_history[target_feature].iloc[-max_lag:]
        
        for step in range(self.horizon):
            logger.info(f"{self.log_prefix} univariate-recursive forecast step: {step}...")
            # 1. Prepare current features for prediction
            if step >= len(df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            # Future exogenous and datetime for this step
            current_step_df = df_future.iloc[step:step+1].copy()
            # 2.将历史数据和当前步的数据合并，以便创建特征
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
            # 3.为合并后的数据创建特征
            temp_df_featured, predictor_features, _, categorical_features = self.create_features(
                df_series=combined_df,
                endogenous_features_with_target=endogenous_features,
                exogenous_features=available_exogenous,
                target_feature=target_feature,
                categorical_features=categorical_features,
            )
            # 4.提取出当前预测步所需要的特征（最后一行）
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
            # 5.特征预处理（预测模式）
            current_features_for_pred_processed = preprocessor.transform(current_features_for_pred, categorical_features)
            preprocessor.validate_features(current_features_for_pred_processed, stage="prediction")
            # 5.进行预测
            y_pred_step = model.predict(current_features_for_pred_processed)[0]
            y_preds.append(y_pred_step)
            # 6.将预测值更新回 df_history，以便为下一步预测提供滞后特征
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            new_row_for_last_known[target_feature] = y_pred_step # Add the prediction as the new 'actual' for lag calculation
            # 将新行添加到历史数据中，进行下一次循环
            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]

        return np.array(y_preds)

    def univariate_single_multi_step_direct_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, preprocessor):
        """
        单变量多步直接递归预测 (USMDR)
    
        核心思想：
        1. 将预测horizon分成多个块（block_size = min(lags)）
        2. 在每个块内进行递归预测
        3. 块与块之间也是递归的（使用前一块的预测值）
        
        与 USMR 的区别：
        - USMR: 完全递归，每步都用上一步的预测
        - USMDR: 分块递归，块内递归，减少误差累积
        
        与 USMD 的区别：
        - USMD: 完全直接，为每步训练独立模型
        - USMDR: 只训练一个模型，但采用分块策略
        
        特征构成：
        - 只使用目标变量的滞后特征
        - 加上外生变量
        
        Args:
            model: 训练好的单输出模型
            df_history: 历史数据
            df_future: 未来数据(包含外生变量)
            endogenous_features: 内生变量列表(但只使用目标变量)
            exogenous_features: 外生变量列表
            target_feature: 目标变量名
            categorical_features: 类别特征列表
            scaler_features: 归一化器
        
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} univariate_single_multi_step_directly_recursive_forecast (USMDR)")
        logger.info(f"{self.log_prefix} Target feature: {target_feature}")
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        logger.info(f"{self.log_prefix} Available exogenous features: {available_exogenous}")
        
        # 初始化
        max_lag = max(self.args.lags) if self.args.lags else 1
        block_size = min(self.args.lags) if self.args.lags else 1  # 分块大小
        
        logger.info(f"{self.log_prefix} Max lag: {max_lag}, Block size: {block_size}")
        logger.info(f"{self.log_prefix} Forecast horizon: {self.horizon}")
        
        # 预测值收集器
        y_preds = []
        
        # 获取历史数据
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # 确保目标特征在历史数据中
        if target_feature not in last_known_data.columns and target_feature in df_history.columns:
            last_known_data[target_feature] = df_history[target_feature].iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} Last known data shape: {last_known_data.shape}")
        
        # 分块递归预测
        num_blocks = int(np.ceil(self.horizon / block_size))
        logger.info(f"{self.log_prefix} Number of blocks: {num_blocks}")
        
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, self.horizon)
            
            logger.info(f"{self.log_prefix} Processing block {block_idx + 1}/{num_blocks}: steps {block_start} to {block_end-1}")
            
            # 在当前块内进行递归预测
            for step in range(block_start, block_end):
                if step >= len(df_future):
                    logger.warning(f"{self.log_prefix} Exhausted df_future at step {step}. Stopping.")
                    break
                
                # 1. 获取当前步的外生变量
                current_step_df = df_future.iloc[step:step+1].copy()
                
                # 2. 合并历史数据和当前步数据
                combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
                
                # 3. 创建特征（只为目标变量创建滞后特征）
                temp_df_featured, predictor_features, _, categorical_features_updated = self.create_features(
                    df_series=combined_df,
                    endogenous_features_with_target=[target_feature],  # 只使用目标变量！
                    exogenous_features=available_exogenous,
                    target_feature=target_feature,
                    categorical_features=categorical_features,
                )
                
                # 4. 提取预测特征（最后一行）
                current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
                
                # 5. 特征缩放
                current_features_for_pred_processed = preprocessor.transform(current_features_for_pred, categorical_features)
                preprocessor.validate_features(current_features_for_pred_processed, stage="prediction")
                
                # 6. 预测
                y_pred_step = model.predict(current_features_for_pred_processed)[0]
                y_preds.append(y_pred_step)
                
                logger.info(f"{self.log_prefix}   Step {step}: predicted {y_pred_step:.4f}")
                
                # 7. 更新历史数据（用于下一步预测）
                new_row_for_last_known = current_step_df.copy().iloc[-1:]
                new_row_for_last_known[target_feature] = y_pred_step
                
                last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
                last_known_data = last_known_data.iloc[-max_lag:]  # 只保留需要的历史长度
        
        logger.info(f"{self.log_prefix} USMDR forecast completed, predicted {len(y_preds)} steps")
        
        return np.array(y_preds)
    # ------------------------------
    # 多变量（除目标变量外的内生变量）预测单变量（目标变量）
    # ------------------------------
    def multivariate_single_multi_step_direct_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, preprocessor):
        """
        多变量多步直接预测 (MSMD)
        
        使用所有内生变量的滞后特征 + 外生变量，一次性预测未来多步的目标变量
        
        方法特点：
        1. 特征：所有内生变量(target + 其他内生变量)的滞后 + 外生变量
        2. 训练：为每个未来步h创建目标列 target_shift_0, target_shift_1, ..., target_shift_H-1
        3. 预测：一次性输出所有H步的预测值
        
        与 USMD 的区别：
        - USMD: 只使用目标变量的滞后特征
        - MSMD: 使用所有内生变量的滞后特征（更多信息）
        
        Args:
            model: 训练好的模型(MultiOutputRegressor)
            df_history: 历史数据(包含所有内生变量的原始值)
            df_future: 未来数据(包含外生变量)
            endogenous_features: 所有内生变量列表(包含目标变量)
            exogenous_features: 外生变量列表
            target_feature: 目标变量名
            categorical_features: 类别特征列表
            scaler_features: 归一化器
        
        Returns:
            预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_direct_forecast")
        logger.info(f"{self.log_prefix} Endogenous features: {endogenous_features}")
        logger.info(f"{self.log_prefix} Target feature: {target_feature}")
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        logger.info(f"{self.log_prefix} Available exogenous features: {available_exogenous}")
        
        # 1. 获取足够的历史数据以构建滞后特征
        max_lag = max(self.args.lags) if self.args.lags else 1
        relevant_history_for_lags = df_history.iloc[-max_lag:].copy()
        
        # 确保所有内生变量都在历史数据中
        for endo_feat in endogenous_features:
            if endo_feat not in relevant_history_for_lags.columns and endo_feat in df_history.columns:
                relevant_history_for_lags[endo_feat] = df_history[endo_feat].iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} Relevant history shape: {relevant_history_for_lags.shape}")
        logger.info(f"{self.log_prefix} Relevant history columns: {relevant_history_for_lags.columns.tolist()}")
        
        # 2. 使用未来第一个时间点的外生变量
        forecast_start_point_exogenous = df_future.iloc[0:1].copy()
        
        # 3. 合并历史数据(用于滞后特征)和未来第一个点(用于外生变量)
        combined_for_features = pd.concat([relevant_history_for_lags, forecast_start_point_exogenous], ignore_index=True)
        
        logger.info(f"{self.log_prefix} Combined data shape: {combined_for_features.shape}")
        logger.info(f"{self.log_prefix} Combined data columns: {combined_for_features.columns.tolist()}")
        
        # 4. 创建特征
        # 注意：这里 endogenous_features_with_target 包含所有内生变量
        # create_features 会为所有这些变量创建滞后特征
        combined_featured, predictor_features_for_forecast, _, categorical_features_updated = self.create_features(
            df_series=combined_for_features,
            endogenous_features_with_target=endogenous_features,  # 所有内生变量(包括目标变量)
            exogenous_features=available_exogenous,
            target_feature=target_feature,
            categorical_features=categorical_features
        )
        
        logger.info(f"{self.log_prefix} Featured data shape: {combined_featured.shape}")
        logger.info(f"{self.log_prefix} Predictor features: {predictor_features_for_forecast}")
        
        # 5. 提取预测特征(最后一行)
        X_forecast_input = combined_featured[predictor_features_for_forecast].iloc[-1:]
        
        logger.info(f"{self.log_prefix} X_forecast_input shape: {X_forecast_input.shape}")
        logger.info(f"{self.log_prefix} X_forecast_input columns: {X_forecast_input.columns.tolist()}")
        
        # 6. 特征缩放
        X_forecast_input_processed = preprocessor.transform(X_forecast_input, categorical_features)
        preprocessor.validate_features(X_forecast_input_processed, stage="prediction")
        
        # 7. 预测
        # 模型期望输出 (1, H) 的形状，其中 H 是预测horizon
        Y_pred_multi_step = model.predict(X_forecast_input_processed)[0]
        
        logger.info(f"{self.log_prefix} Raw prediction shape: {Y_pred_multi_step.shape}")
        logger.info(f"{self.log_prefix} Raw prediction: {Y_pred_multi_step[:10]}...")  # 显示前10个值
        
        # 8. 处理预测结果
        if len(Y_pred_multi_step) >= len(df_future):
            Y_pred = Y_pred_multi_step[:len(df_future)]
        else:
            logger.warning(
                f"{self.log_prefix} Predicted {len(Y_pred_multi_step)} steps "
                f"but df_future requires {len(df_future)} rows. Padding predictions."
            )
            Y_pred = np.pad(Y_pred_multi_step, (0, len(df_future) - len(Y_pred_multi_step)), 'edge')
        
        logger.info(f"{self.log_prefix} Final prediction shape: {Y_pred.shape}")
        
        return Y_pred

    def multivariate_single_multi_step_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, target_output_features, categorical_features, preprocessor):
        """
        多变量多步递归预测 (predicts all endogenous variables recursively)
        """
        # target_output_features contains names like "target_feature_shift_1", "endogenous1_shift_1".
        # model predicts target_t+1, endogenous1_t+1, ...
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        y_preds_primary_target = []
        
        # Start with the latest available data to construct initial features
        # `last_known_data` will hold values (actuals or predictions) for all endogenous variables needed for lags.
        all_endogenous_original_cols = [col.replace("_shift_1", "") for col in target_output_features] # original names of predicted endogenous
        
        max_lag = max(self.args.lags) if self.args.lags else 1
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # Ensure all original endogenous_features are present in last_known_data for lag creation
        for col in all_endogenous_original_cols:
            if col not in last_known_data.columns and col in df_history.columns:
                last_known_data[col] = df_history[col].iloc[-max_lag:]
        
        # Iterate for each step in the forecast horizon
        for step in range(self.horizon):
            logger.info(f"{self.log_prefix} multivariate-recursive forecast step: {step}...")

            if step >= len(df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break

            # 1. Prepare current features for prediction
            current_step_df = df_future.iloc[step:step+1].copy() # Future exogenous and datetime features for this step
            
            # Combine last_known_data (with actuals or previous predictions for lags) and current_step_df
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)

            # Create features based on the combined data (lags will be based on last_known_data)
            temp_df_featured, predictor_features, _, categorical_features = self.create_features(
                df_series=combined_df,
                endogenous_features_with_target=endogenous_features, # All actual endogenous cols including target
                exogenous_features=available_exogenous,
                target_feature=target_feature,
                categorical_features=categorical_features,
            )
            # The last row of temp_df_featured contains the features for the current step's prediction
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]

            # 特征预处理（预测模式）
            current_features_for_pred_processed = preprocessor.transform(current_features_for_pred, categorical_features)
            preprocessor.validate_features(current_features_for_pred_processed, stage="prediction")
            
            # 2. Make prediction for the next step for all target_output_features
            next_pred_values_array = model.predict(current_features_for_pred_processed)[0] # [0] because predict returns [[val1, val2, ...]]
            # Map predictions back to their shifted column names
            next_pred_dict = dict(zip(target_output_features, next_pred_values_array))

            # Store the prediction for the primary target (assuming it's the first in target_output_features)
            if target_feature:
                primary_target_shifted_name = f"{target_feature}_shift_1"
                if primary_target_shifted_name in next_pred_dict:
                    y_preds_primary_target.append(next_pred_dict[primary_target_shifted_name])
                else: # Fallback if primary target not in output cols
                    y_preds_primary_target.append(next_pred_values_array[0]) # Default to first predicted output
            else:
                y_preds_primary_target.append(next_pred_values_array[0]) # No target_feature, take first predicted

            # 3. Update `last_known_data` for the next recursive step
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            
            # Update endogenous variables (target and other endogenous) with predicted values
            for shifted_col_name, pred_val in next_pred_dict.items():
                original_col_name = shifted_col_name.replace("_shift_1", "")
                new_row_for_last_known[original_col_name] = pred_val
            
            # Carry over any other features from previous last_known_data that are needed but not predicted
            # This is crucial for exogenous features that remain constant or are known in future
            # And also other endogenous that are not predicted but are part of the lag features
            for col in last_known_data.columns:
                if col not in new_row_for_last_known.columns:
                    # If it's an exogenous feature in current_step_df, prefer that
                    if col in current_step_df.columns:
                        new_row_for_last_known[col] = current_step_df[col].iloc[-1]
                    else: # Otherwise, take from the last known data point
                        new_row_for_last_known[col] = last_known_data[col].iloc[-1]

            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]

        return np.array(y_preds_primary_target)

    def multivariate_single_multi_step_direct_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, categorical_features, preprocessor):
        """
        多变量多步直接递归预测 (MSMDR)
        
        核心思想：
        1. 使用所有内生变量的滞后特征（不只是目标变量）
        2. 分块递归预测目标变量
        3. 对于其他内生变量，使用持久性预测或简单方法估计
        
        与 USMDR 的核心区别：
        - USMDR: 只用目标变量的滞后 → 特征少
        - MSMDR: 用所有内生变量的滞后 → 特征多，信息丰富
        
        与 MSMR 的区别：
        - MSMR: 递归预测所有内生变量
        - MSMDR: 只递归预测目标变量，其他内生变量用简化方法
        
        特征构成示例：
        假设 endogenous_features = ['load', 'temperature', 'humidity']
            target_feature = 'load'
            lags = [1, 2, 7]
        
        特征 = [load_lag_1, load_lag_2, load_lag_7,           # 目标变量的滞后
            temperature_lag_1, temperature_lag_2, temperature_lag_7,  # 其他内生变量的滞后
            humidity_lag_1, humidity_lag_2, humidity_lag_7,
            hour, day_of_week, ...]  # 外生变量
        
        Args:
            model: 训练好的单输出模型
            df_history: 历史数据（包含所有内生变量）
            df_future: 未来数据（包含外生变量）
            endogenous_features: 所有内生变量列表（包括目标变量）
            exogenous_features: 外生变量列表
            target_feature: 目标变量名
            categorical_features: 类别特征列表
            scaler_features: 归一化器
        
        Returns:
            目标变量的预测结果数组，形状为 (horizon,)
        """
        logger.info(f"{self.log_prefix} multivariate_single_multi_step_directly_recursive_forecast (MSMDR)")
        logger.info(f"{self.log_prefix} Endogenous features: {endogenous_features}")
        logger.info(f"{self.log_prefix} Target feature: {target_feature}")
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        logger.info(f"{self.log_prefix} Available exogenous features: {available_exogenous}")
        
        # 初始化
        max_lag = max(self.args.lags) if self.args.lags else 1
        block_size = min(self.args.lags) if self.args.lags else 1
        
        logger.info(f"{self.log_prefix} Max lag: {max_lag}, Block size: {block_size}")
        logger.info(f"{self.log_prefix} Forecast horizon: {self.horizon}")
        
        # 预测值收集器
        y_preds_target = []
        
        # 获取历史数据
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # 确保所有内生变量都在历史数据中
        for endo_feat in endogenous_features:
            if endo_feat not in last_known_data.columns and endo_feat in df_history.columns:
                last_known_data[endo_feat] = df_history[endo_feat].iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} Last known data shape: {last_known_data.shape}")
        logger.info(f"{self.log_prefix} Last known data columns: {last_known_data.columns.tolist()}")
        
        # 为其他内生变量准备持久性预测
        # （假设其他内生变量在未来保持最后观测值不变，或使用简单趋势）
        other_endogenous = [feat for feat in endogenous_features if feat != target_feature]
        last_values_other_endogenous = {}
        
        for feat in other_endogenous:
            if feat in last_known_data.columns:
                last_values_other_endogenous[feat] = last_known_data[feat].iloc[-1]
            else:
                last_values_other_endogenous[feat] = 0
                logger.warning(f"{self.log_prefix} Feature {feat} not found, using 0")
        
        logger.info(f"{self.log_prefix} Last values for other endogenous: {last_values_other_endogenous}")
        
        # 分块递归预测
        num_blocks = int(np.ceil(self.horizon / block_size))
        logger.info(f"{self.log_prefix} Number of blocks: {num_blocks}")
        
        for block_idx in range(num_blocks):
            block_start = block_idx * block_size
            block_end = min(block_start + block_size, self.horizon)
            
            logger.info(f"{self.log_prefix} Processing block {block_idx + 1}/{num_blocks}: steps {block_start} to {block_end-1}")
            
            # 在当前块内进行递归预测
            for step in range(block_start, block_end):
                if step >= len(df_future):
                    logger.warning(f"{self.log_prefix} Exhausted df_future at step {step}. Stopping.")
                    break
                
                # 1. 获取当前步的外生变量
                current_step_df = df_future.iloc[step:step+1].copy()
                
                # 2. 合并历史数据和当前步数据
                combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
                
                # 3. 创建特征（为所有内生变量创建滞后特征）
                temp_df_featured, predictor_features, _, categorical_features_updated = self.create_features(
                    df_series=combined_df,
                    endogenous_features_with_target=endogenous_features,  # 所有内生变量！
                    exogenous_features=available_exogenous,
                    target_feature=target_feature,
                    categorical_features=categorical_features,
                )
                
                # 4. 提取预测特征（最后一行）
                current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
                
                # 5. 特征缩放
                current_features_for_pred_processed = preprocessor.transform(current_features_for_pred, categorical_features)
                preprocessor.validate_features(current_features_for_pred_processed, stage="prediction")
                
                # 6. 预测目标变量
                y_pred_target = model.predict(current_features_for_pred_processed)[0]
                y_preds_target.append(y_pred_target)
                
                logger.info(f"{self.log_prefix}   Step {step}: predicted target = {y_pred_target:.4f}")
                
                # 7. 更新历史数据
                new_row_for_last_known = current_step_df.copy().iloc[-1:]
                
                # 更新目标变量的值（使用预测值）
                new_row_for_last_known[target_feature] = y_pred_target
                
                # 更新其他内生变量的值（使用持久性预测）
                # 策略1: 保持最后观测值不变（最简单）
                for feat in other_endogenous:
                    new_row_for_last_known[feat] = last_values_other_endogenous[feat]
                
                # 策略2: 也可以使用简单的移动平均或趋势（更复杂但可能更准确）
                # for feat in other_endogenous:
                #     # 计算最近几个值的平均作为预测
                #     if feat in last_known_data.columns:
                #         recent_mean = last_known_data[feat].tail(3).mean()
                #         new_row_for_last_known[feat] = recent_mean
                
                last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
                last_known_data = last_known_data.iloc[-max_lag:]
        
        logger.info(f"{self.log_prefix} MSMDR forecast completed, predicted {len(y_preds_target)} steps")
        
        return np.array(y_preds_target)
    # ------------------------------
    # Model Forecast results save
    # ------------------------------
    def forecast_results_save(self, df_history, df_future):
        """
        输出结果处理
        """
        # 预测结果保存
        df_future["time"] = pd.to_datetime(df_future["time"])
        df_future = df_future.sort_values(by=["time"])
        df_future.to_csv(self.args.pred_results_dir.joinpath("prediction.csv"), encoding="utf_8_sig", index=False)
        # 预测结果可视化
        # Only plot the last 2 days of true history for context, if available
        if not df_history.empty:
            y_trues_df_plot = df_history.iloc[-2 * self.n_per_day:]
        else:
            y_trues_df_plot = pd.DataFrame()
        plt.figure(figsize=(25, 8))
        if not y_trues_df_plot.empty and 'y' in y_trues_df_plot.columns:
            plt.plot(y_trues_df_plot["time"], y_trues_df_plot["y"], label='Trues', lw=2.0)
        if not df_future.empty and 'predict_value' in df_future.columns:
            plt.plot(df_future["time"], df_future["predict_value"], label='Preds', lw=2.0, ls="-.")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"模型预测预测--{self.args.pred_method}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=1.0)
        plt.tight_layout()
        # plt.xticks(rotation=45)
        plt.savefig(self.args.pred_results_dir.joinpath('prediction.png'), dpi=300, bbox_inches='tight')
        # plt.show();
    # ------------------------------
    # Model forecasting
    # ------------------------------
    def forecast(self, df_history, X_train_history, Y_train_history, df_future, endogenous_features, exogenous_features, target_feature, target_output_features, categorical_features):
        """
        模型预测
        """
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Training start...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        # 创建特征预处理器
        self.preprocessor_forecasting = FeaturePreprocessor(self.args, log_prefix=self.log_prefix)
        # 模型训练
        model = self.train(X_train_history, Y_train_history, self.preprocessor_forecasting, categorical_features)
        # 模型保存
        self.model_save(model)
        logger.info(f"{self.log_prefix} Model Training result saved in: {self.args.checkpoints_dir}")
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Forecasting start...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        # Initialize Y_pred
        Y_pred = np.array([])
        # Use a copy of raw future data
        df_future_for_prediction = df_future.copy()
        
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            # 特征工程
            df_future_featured, predictor_features, target_output_features, categorical_features = self.create_features(
                df_series=df_future_for_prediction, 
                endogenous_features_with_target=endogenous_features,
                exogenous_features=exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features
            )
            df_future_featured = df_future_featured.dropna()
            logger.info(f"{self.log_prefix} df_future_featured: \n{df_future_featured.head()}")
            logger.info(f"{self.log_prefix} predictor_features: {predictor_features}")
            logger.info(f"{self.log_prefix} target_output_features: {target_output_features}")
            logger.info(f"{self.log_prefix} categorical_features: {categorical_features}")
            # 特征选择
            X_test_future = df_future_featured[predictor_features]
            X_test_future_df = X_test_future.copy()
            logger.info(f"{self.log_prefix} X_test_future_df: \n{X_test_future_df}")
            # 模型预测
            Y_pred = self.univariate_single_multi_step_direct_output_forecast(
                model = model,
                X_test = X_test_future_df,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        elif self.args.pred_method == "univariate-single-multistep-direct":
            self.univariate_single_multi_step_direct_forecast(
                model = model,
                df_history = df_history,
                df_future = df_future_for_prediction,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features=target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            Y_pred = self.univariate_single_multi_step_recursive_forecast(
                model = model,
                df_history = df_history,
                df_future = df_future_for_prediction,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features=target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            Y_pred = self.univariate_single_multi_step_direct_recursive_forecast(
                model = model,
                df_history = df_history,
                df_future = df_future_for_prediction,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features=target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            Y_pred = self.multivariate_single_multi_step_direct_forecast(
                model = model,
                df_history = df_history,
                df_future = df_future_for_prediction,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features=target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            Y_pred = self.multivariate_single_multi_step_recursive_forecast(
                model = model,
                df_history = df_history,
                df_future = df_future_for_prediction,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                target_output_features = target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            Y_pred = self.multivariate_single_multi_step_direct_recursive_forecast(
                model = model,
                df_history = df_history,
                df_future = df_future_for_prediction,
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                # target_output_features=target_output_features,
                categorical_features = categorical_features,
                preprocessor = self.preprocessor_forecasting,
            )
        
        # 预测结果收集
        df_future_for_prediction["predict_value"] = Y_pred
        df_future_for_prediction = df_future_for_prediction[["time", "predict_value"]]
        logger.info(f"{self.log_prefix} after forecast df_future: \n{df_future_for_prediction.head()}")
        logger.info(f"{self.log_prefix} after forecast df_future.shape: {df_future_for_prediction.shape}")
        
        return df_future_for_prediction
    # ##############################
    # 运行
    # ##############################
    def run(self):
        # ------------------------------
        # 数据加载
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history and future data loading...")
        logger.info(f"{self.log_prefix} {80*'='}")
        input_data = self.load_data()
        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history data preprocessing...")
        logger.info(f"{self.log_prefix} {80*'='}")
        (df_history, 
         endogenous_features_with_target, 
         exogenous_features, 
         target_feature, 
         categorical_features
        ) = self.process_history_data(input_data = input_data)
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {80*'='}")
        df_history_featured, predictor_features, target_output_features, categorical_features = self.create_features(
            df_series = df_history,
            endogenous_features_with_target = endogenous_features_with_target,
            exogenous_features = exogenous_features,
            target_feature = target_feature,
            categorical_features = categorical_features,
        )
        # Drop rows with NaNs after feature/target generation
        df_history_featured = df_history_featured.dropna()
        logger.info(f"{self.log_prefix} df_history_featured: \n{df_history_featured.head()}")
        # df_history_featured.to_csv(f"./exp/ml_todo/df_history_{self.args.pred_method}.csv")
        logger.info(f"{self.log_prefix} predictor_features: {predictor_features}")
        logger.info(f"{self.log_prefix} target_output_features: {target_output_features}")
        logger.info(f"{self.log_prefix} categorical_features: {categorical_features}")
        # ------------------------------
        # 特征分割：预测特征和目标特征
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history data feature split...")
        logger.info(f"{self.log_prefix} {80*'='}")
        # Select X and Y
        X_train_history = df_history_featured[predictor_features]
        Y_train_history = df_history_featured[target_output_features]
        # Drop any remaining NaNs from X and Y that might occur after specific feature/target generation(This is critical to ensure training data is clean).
        combined_xy = pd.concat([X_train_history, Y_train_history], axis=1)
        combined_xy.dropna(inplace=True)
        X_train_history = combined_xy[X_train_history.columns]
        Y_train_history = combined_xy[Y_train_history.columns]
        logger.info(f"{self.log_prefix} X_train_history shape after NaN drop: {X_train_history.shape}")
        logger.info(f"{self.log_prefix} Y_train_history shape after NaN drop: {Y_train_history.shape}")
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_testing:
            logger.info(f"{self.log_prefix} {80*'='}")
            logger.info(f"{self.log_prefix} Model Testing...")
            logger.info(f"{self.log_prefix} {80*'='}")
            # 模型滑窗测试(Model sliding window test)
            test_scores_df, cv_plot_df = self.test(
                df_history,
                X_train_history,
                Y_train_history,
                endogenous_features_with_target,
                exogenous_features,
                target_feature,
                target_output_features,
                categorical_features, 
            )
            # 测试结果保存
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Testing result save...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            self.test_results_save(test_scores_df, cv_plot_df)
            logger.info(f"{self.log_prefix} Model Testing result saved in: {self.args.test_results_dir}")
        # ------------------------------
        # 模型预测
        # ------------------------------
        if self.args.is_forecasting:
            logger.info(f"{self.log_prefix} {80*'='}")
            logger.info(f"{self.log_prefix} Model Forecasting...")
            logger.info(f"{self.log_prefix} {80*'='}")
            # 未来数据处理(用来推理)
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Forecasting future data preprocessing...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            # df_future contains time, exogenous features for the forecast horizon
            df_future = self.process_future_data(input_data = input_data)
            # 模型预测
            df_future_predicted = self.forecast(
                df_history,
                X_train_history,
                Y_train_history,
                df_future,
                endogenous_features_with_target,
                exogenous_features,
                target_feature,
                target_output_features,
                categorical_features, 
            )
            # 模型预测结果保存
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Forecasting result save...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            self.forecast_results_save(df_history, df_future_predicted)
            logger.info(f"{self.log_prefix} Model Forecasting result saved in: {self.args.pred_results_dir}")




# 测试代码 main 函数
def main():
    """
    主函数入口
    """
    # 模型配置
    args = ModelConfig_univariate()
    # args = ModelConfig_multivariate()
    
    # 创建模型实例
    model = Model(args)
    
    # 运行模型
    model.run()

if __name__ == "__main__":
    main()
