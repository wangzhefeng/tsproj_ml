# -*- coding: utf-8 -*-

# ***************************************************
# * File        : LightGBM_forecast.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 1.0.121116
# * Description : 1.单变量多步直接预测(数据标准化)
# *               2.单变量多步递归预测(滞后特征，数据标准化)
# *               3.多变量多步直接预测(滞后特征，数据标准化)
# *               4.多变量多步递归预测(滞后特征，数据标准化)
# * Link        : 1.https://mp.weixin.qq.com/s/haCeJW9wamtXkBjX3oUvdQ
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)增加 log;
# ***************************************************

__all__ = []

# python libraries
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import math
import copy
import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# model
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.multioutput import (
    MultiOutputRegressor,
    RegressorChain,
)
# model evaluation
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)
# data processing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

# feature engineering
# from utils.ts.feature_engine import (
#     extend_datetime_feature,
#     extend_datetype_feature,
#     extend_weather_feature,
#     extend_future_weather_feature,
#     extend_lag_feature,
#     extend_lag_feature_univariate,
#     extend_lag_feature_multivariate,
# )
# utils
from utils.model_save_load import ModelDeployPkl
# from utils.plot_results import (
#     forecast_results_visual,
#     predict_result_visual
# )

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


@dataclass
class ModelConfig:
    # ------------------------------
    # model data
    # ------------------------------
    # 数据路径
    data_dir = Path("./dataset/electricity_work/demand_load/lingang_A")
    # target time series
    # -----------------
    data_path = "AIDC_A_dataset.csv"              # 目标时间序列数据路径
    data = "AIDC_A_dataset"                       # 目标时间序列数据名称
    freq = "5min"                                 # 目标时间序列数据频率
    freq_minutes = 5                              # 目标时间序列数据频率(分钟)
    target_ts_feat = "count_data_time"            # 目标时间序列时间戳特征名称
    target_series_numeric_features = []           # 目标时间序列的数值特征 (other endogenous variables)
    target_series_categorical_features = []       # 目标时间序列的类别特征
    target_series_drop_features = []              # 目标时间序列的删除特征
    target = "h_total_use"                        # 目标时间序列预测目标变量名称
    # date type
    # -----------------
    date_history_path = "df_date.csv"
    # date_history_path = None
    date_future_path = "df_date_future.csv"
    # date_future_path = None
    date_ts_feat = "date"                         # 日期数据时间戳名称
    # weather
    # -----------------
    weather_history_path = "df_weather.csv"
    # weather_history_path = None
    weather_future_path = "df_weather_future.csv"
    # weather_future_path = None
    weather_ts_feat = "ts"                        # 天气数据时间戳名称
    # ------------------------------
    # model name
    # ------------------------------
    model_name = "LightGBM"
    # ------------------------------
    # model data preprocessing
    # ------------------------------
    scale = True                                  # 是否进行归一化/标准化
    inverse = True                               # TODO 是否进行归一化/标准化逆变换
    target_transform = False                      # TODO 预测目标是否需要转换
    target_transform_predict = None               # TODO 预测目标的转换特征是否需要预测
    threshold = 100000                            # TODO 异常值处理阈值
    # ------------------------------
    # feature engineering
    # ------------------------------
    lags = [
        1 * 288, # Daily lag
        2 * 288,
        3 * 288,
        4 * 288,
        5 * 288,
        6 * 288,
        7 * 288, # Weekly lag
    ]                                             # 特征滞后数列表
    datetime_features = [
        'minute', 'hour', 'day', 'weekday', 'week',
        'day_of_week', 'week_of_year', 'month', 'days_in_month',
        'quarter', 'day_of_year', 'year',
    ]                                             # 日期时间特征
    # categorical_features = [ # These will be collected dynamically
    #     "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week",
    #     "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month",
    #     "datetime_quarter", "datetime_day_of_year", "datetime_year",
    #     "date_type",
    # ]
    # ------------------------------
    # model define
    # ------------------------------
    # 预测方法
    # pred_method = "univariate-multi-step-directly"     # UMSD 单变量(内生变量)多步直接预测
    pred_method = "univariate-multi-step-recursive"      # UMSR 单变量(内生变量)多步递归预测
    # pred_method = "multivariate-multi-step-directly"   # MMSD 多变量(内生变量)多步直接预测
    # pred_method = "multivariate-multi-step-recursive"  # MMSR 多变量(内生变量)多步递归预测
    featuresd = "MS"                              # TODO 模型预测方式 (This seems unused, keeping for now)
    objective = "regression_l1"                   # 训练目标
    loss = "mae"                                  # 训练损失函数
    learning_rate = 0.05                          # 模型学习率
    patience = 100                                # 早停步数
    # ------------------------------
    # model training
    # ------------------------------
    history_days = 30                             # 历史数据天数
    predict_days = 1                              # 预测未来 1 天的数据
    window_days = 15                              # 滑动窗口天数
    train_ratio = 0.7                             # TODO 训练数据样本比例 (Not used in current sliding window)
    test_ratio = 0.2                              # TODO 测试数据样本比例 (Not used in current sliding window)
    date_type = None                              # 日期类型，用于区分工作日("workday")，非工作日("offday") (This seems unused, keeping for now)
    encode_categorical_features = False           # 是否对类别特征进行编码
    # ------------------------------
    # model testing
    # ------------------------------
    is_testing = True
    # ------------------------------
    # model forecasting
    # ------------------------------
    is_forecasting = True
    # now time(预测推理开始的时间)
    now_time = datetime.datetime(2025, 12, 27, 0, 0, 0)
    # ------------------------------
    # result saved
    # ------------------------------
    checkpoints = "./saved_results/pretrained_models/"
    test_results = "./saved_results/test_results/"
    pred_results = "./saved_results/predict_results/"
    # ------------------------------
    # hyperparameter tuning
    # ------------------------------
    perform_tuning = False # Set to True to enable tuning
    tuning_metric = "neg_mean_absolute_error" # Metric for hyperparameter tuning
    tuning_n_splits = 3 # Number of splits for TimeSeriesSplit cross-validation


class Model:

    def __init__(self, args: ModelConfig) -> None:
        self.args = args
        # ------------------------------
        # 数据参数
        # ------------------------------
        # 目标时间序列每天样本数量
        self.n_per_day = int(24 * 60 / args.freq_minutes)
        # ------------------------------
        # 特征工程
        # ------------------------------
        # 特征滞后数个数(1,2,...)
        self.n_lags = len(args.lags)
        # 类别特征收集器
        self.categorical_features = []
        # ------------------------------
        # 模型预测
        # ------------------------------
        # 预测未来 1 天(24小时)的数据/数据划分长度/预测数据长度
        self.horizon = int(args.predict_days * self.n_per_day)
        # ------------------------------
        # 数据窗口
        # ------------------------------
        # 测试滑动窗口数量, >=1, 1: 单个窗口
        self.n_windows = int(args.history_days - (args.window_days - 1))
        # 测试窗口数据长度(训练+测试)
        self.window_len = int(args.window_days * self.n_per_day if self.n_windows > 1 else args.history_days * self.n_per_day)
        # ------------------------------
        # 数据划分时间
        # ------------------------------
        # 时间序列历史数据开始时刻
        self.start_time = args.now_time.replace(hour=0) - datetime.timedelta(days=args.history_days)
        # 时间序列当前时刻(模型预测的日期时间)
        self.now_time = args.now_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
        # 时间序列未来结束时刻
        self.future_time = args.now_time + datetime.timedelta(days=args.predict_days)
        # datetime index
        self.train_start_time = self.start_time
        self.train_end_time = self.now_time
        self.forecast_start_time = self.now_time
        self.forecast_end_time = self.future_time
        # ------------------------------
        # 模型参数
        # ------------------------------
        self.model_params = {
            "boosting_type": "gbdt",
            "objective": self.args.objective,  # "regression_l1": L1 loss or MAE, "regression": L2 loss or MSE
            "metric": self.args.loss,  # if objective=="regression_l1": mae, if objective=="regression": rmse
            "n_estimators": 1000,
            "learning_rate": self.args.learning_rate,
            "max_bin": 31,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.5,
            "lambda_l2": 0.5,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
        # ------------------------------
        # log
        # ------------------------------
        logger.info(f"{75*'='}")
        logger.info(f"Prepare params...")
        logger.info(f"{75*'='}")
        logger.info(f"history data range: {self.train_start_time}~{self.train_end_time}")
        logger.info(f"predict data range: {self.forecast_start_time}~{self.forecast_end_time}")
        self.log_prefix = f"model: {args.model_name}, data: {args.data}::"
        self.setting = f"{args.model_name}-{args.data}-{args.pred_method}"
    # ##############################
    # Data load
    # ##############################
    def load_data(self):
        """
        数据加载
        """
        # ------------------------------
        # 历史数据
        # ------------------------------
        # series
        if self.args.data_path is not None:
            df_history_series = pd.read_csv(self.args.data_dir.joinpath(self.args.data_path), encoding="utf-8")
            logger.info(f"{self.log_prefix} df_history_series: \n{df_history_series.head()}")
        else:
            df_history_series = None
            logger.info(f"{self.log_prefix} df_history_series: {df_history_series}")
        # date type
        if self.args.date_history_path is not None:
            df_date_history = pd.read_csv(self.args.data_dir.joinpath(self.args.date_history_path), encoding="utf-8")
            logger.info(f"{self.log_prefix} df_date_history: \n{df_date_history.head()}")
        else:
            df_date_history = None
            logger.info(f"{self.log_prefix} df_date_history: {df_date_history}")
        # weather
        if self.args.weather_history_path is not None:
            df_weather_history = pd.read_csv(self.args.data_dir.joinpath(self.args.weather_history_path), encoding="utf-8")
            logger.info(f"{self.log_prefix} df_weather_history: \n{df_weather_history.head()}")
        else:
            df_weather_history = None
            logger.info(f"{self.log_prefix} df_weather_history: {df_weather_history}")
        # ------------------------------
        # 未来数据
        # ------------------------------
        # series
        df_future_series = None
        logger.info(f"{self.log_prefix} df_future_series: {df_future_series}")
        # date type
        if self.args.date_future_path is not None:
            df_date_future = pd.read_csv(self.args.data_dir.joinpath(self.args.date_future_path), encoding="utf-8")
            logger.info(f"{self.log_prefix} df_date_future: \n{df_date_future.head()}")
        else:
            df_date_future = None
            logger.info(f"{self.log_prefix} df_date_future: {df_date_future}")
        # weather
        if self.args.weather_future_path is not None:
            df_weather_future = pd.read_csv(self.args.data_dir.joinpath(self.args.weather_future_path), encoding="utf-8")
            logger.info(f"{self.log_prefix} df_weather_future: \n{df_weather_future.head()}")
        else:
            df_weather_future = None
            logger.info(f"{self.log_prefix} df_weather_future: {df_weather_future}")
        # ------------------------------
        # 数据合并
        # ------------------------------
        if df_date_history is None or df_date_future is None:
            df_date_all = None
        else:
            df_date_all = pd.concat([df_date_history.iloc[:-1,], df_date_future], axis=0) # .iloc[:-1] to avoid duplicate last entry
        
        if df_weather_history is None or df_weather_future is None:
            df_weather_all = None
        else:
            df_weather_all = pd.concat([df_weather_history.iloc[:-1,], df_weather_future], axis=0) # .iloc[:-1] to avoid duplicate last entry
        # ------------------------------
        # 输入数据以字典形式整理
        # ------------------------------
        input_data = {
            "df_history_series": df_history_series,
            "df_date_history": df_date_all,
            "df_weather_history": df_weather_all,
            "df_future_series": df_future_series,
            "df_date_future": df_date_all,
            "df_weather_future": df_weather_all,
        }
        
        return input_data
    # ##############################
    # Data Preprocessing
    # ##############################
    def __process_df_timestamp(self, df, col_ts: str):
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

    def __process_target_series(self, df_template, df_series, col_ts: str, col_numeric: List, col_categorical: List, col_drop: List):
        """
        目标特征数据预处理
        df_template: ["time"]
        """
        df_template_copy = df_template.copy()
        if df_series is not None:
            # 目标特征数据转换为浮点数
            if self.args.target in df_series.columns:
                df_series[self.args.target] = df_series[self.args.target].apply(lambda x: float(x))
                # 将原始数据映射到时间戳完整的 df_template 中
                df_template_copy["y"] = df_template_copy["time"].map(df_series.set_index(col_ts)[self.args.target])
                target_feature = "y"
            else:
                target_feature = None
            # 数值特征处理 (other endogenous variables)
            # Ensure col_numeric includes only actual numeric columns, and not the target
            filtered_col_numeric = [col for col in col_numeric if col not in [col_ts, self.args.target] + col_categorical + col_drop]
            for col in filtered_col_numeric:
                if col in df_series.columns:
                    # 将数据转换为浮点数类型
                    df_series[col] = df_series[col].apply(lambda x: float(x))
                    # 将时序特征映射到时间戳完整的 df_template 中, 特征包括[ds, y, feature_numeric]
                    df_template_copy[col] = df_template_copy["time"].map(df_series.set_index(col_ts)[col])
            # 类别特征处理
            filtered_col_categorical = [col for col in col_categorical if col not in [col_ts, self.args.target] + col_numeric + col_drop]
            for col in filtered_col_categorical:
                if col in df_series.columns:
                    # 类别特征处理
                    df_series[col] = df_series[col].apply(lambda x: str(x))
                    # 将时序特征映射到时间戳完整的 df_template 中, 特征包括[ds, y, feature_categorical]
                    df_template_copy[col] = df_template_copy["time"].map(df_series.set_index(col_ts)[col])
            
            # Endogenous features (including the primary target 'y' if present)
            # These are features whose values are determined within the system being modeled.
            # They might be used as lags or as targets for multi-output recursive prediction.
            endogenous_features = [col for col in df_template_copy.columns if col not in ["time"]]
            if target_feature and target_feature in endogenous_features:
                 endogenous_features.remove(target_feature) # Remove target from here for consistency, will be handled separately

            return df_template_copy, endogenous_features, target_feature
        else:
            return df_template_copy, [], None
    
    def process_history_data(self, input_data: Dict=None):
        """
        历史数据预处理
        """
        # ------------------------------
        # 历史数据时间戳
        # ------------------------------
        df_history_template = pd.DataFrame({
            "time": pd.date_range(self.train_start_time, self.train_end_time, freq=self.args.freq, inclusive="left")
        })
        logger.info(f"{self.log_prefix} template df_history_template: \n{df_history_template.head()}")
        # ------------------------------
        # 数据预处理：目标时间序列特征
        # ------------------------------
        df_history_series = self.__process_df_timestamp(
            df=input_data["df_history_series"],
            col_ts=self.args.target_ts_feat
        )
        # endogenous_features here are the 'target_series_numeric_features' plus 'target' (as 'y')
        df_history, other_endogenous_features, target_feature = self.__process_target_series(
            df_template=df_history_template,
            df_series=df_history_series,
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        logger.info(f"{self.log_prefix} after process_target_series df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} after process_target_series other_endogenous_features: {other_endogenous_features}")
        logger.info(f"{self.log_prefix} after process_target_series target_features: {target_feature}")
        
        endogenous_features_with_target = [target_feature] + other_endogenous_features if target_feature else other_endogenous_features

        # ------------------------------
        # 特征工程：日期类型(节假日、特殊事件)特征
        # ------------------------------
        df_date_history = self.__process_df_timestamp(
            df=input_data[f"df_date_history"],
            col_ts=self.args.date_ts_feat
        )
        df_history, date_features = self.extend_datetype_feature(
            df=df_history,
            df_date=df_date_history,
            col_ts=self.args.date_ts_feat,
        )
        self.categorical_features.append("date_type")
        logger.info(f"{self.log_prefix} after extend_datetype_feature df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} after extend_datetype_feature date_features: {date_features}")
        # ------------------------------
        # 特征工程：天气特征
        # ------------------------------
        df_weather_history = self.__process_df_timestamp(
            df=input_data[f"df_weather_history"],
            col_ts=self.args.weather_ts_feat
        )
        df_history, weather_features = self.extend_weather_feature(
            df=df_history,
            df_weather=df_weather_history,
            col_ts=self.args.weather_ts_feat,
        )
        # self.categorical_features += [] # Assuming weather features are numeric or handled by feature_engine
        logger.info(f"{self.log_prefix} after extend_weather_feature df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} after extend_weather_feature weather_features: {weather_features}")
        # ------------------------------
        # 外生特征收集
        # ------------------------------
        exogenous_features = date_features + weather_features
        logger.info(f"{self.log_prefix} exogenous_features: {exogenous_features}")
        # ------------------------------
        # 插值填充预测缺失值
        # ------------------------------
        # Interpolate existing data, then drop any remaining NaNs (e.g., at boundaries)
        df_history = df_history.interpolate(method="linear", limit_direction="both")
        df_history.dropna(inplace=True, ignore_index=True) # Drops rows where even interpolation couldn't help
        logger.info(f"{self.log_prefix} after interpolate and dropna df_history: \n{df_history.head()}")
        
        return (df_history, endogenous_features_with_target, exogenous_features, target_feature)

    def process_future_data(self, input_data):
        """
        处理未来数据
        """
        # 未来数据格式
        df_future_template = pd.DataFrame({
            "time": pd.date_range(self.forecast_start_time, self.forecast_end_time, freq=self.args.freq, inclusive="left")
        })
        logger.info(f"{self.log_prefix} template df_future_template: \n{df_future_template.head()}")
        # 数据预处理：除目标特征外的其他内生特征
        df_future_series = self.__process_df_timestamp(
            df=input_data["df_future_series"],
            col_ts=self.args.target_ts_feat
        )
        # For future data, target is usually unknown, other endogenous features might be unknown or based on past.
        # So we only pass the template. The primary target_feature will be None here.
        df_future, other_endogenous_features, target_feature_dummy = self.__process_target_series(
            df_template=df_future_template,
            df_series=df_future_series, # This will be None, so df_future_template is returned.
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        logger.info(f"{self.log_prefix} after process_target_series df_future: \n{df_future.head()}")
        logger.info(f"{self.log_prefix} after process_target_series other_endogenous_features: {other_endogenous_features}")
        logger.info(f"{self.log_prefix} after process_target_series target_feature (dummy): {target_feature_dummy}") # Expecting None or no 'y'

        # In forecast, we might need to carry forward some of these if they are to be predicted recursively.
        # For now, let's just use the `other_endogenous_features` as a reference.
        future_endogenous_cols_for_lag = other_endogenous_features

        # ------------------------------
        # 特征工程：日期类型(节假日、特殊事件)特征
        # ------------------------------
        df_date_future = self.__process_df_timestamp(
            df=input_data[f"df_date_future"],
            col_ts=self.args.date_ts_feat
        )
        df_future, date_features = self.extend_datetype_feature(
            df=df_future,
            df_date=df_date_future,
            col_ts=self.args.date_ts_feat,
        )
        # self.categorical_features.append("date_type") # Already collected in history
        logger.info(f"{self.log_prefix} after extend_datetype_feature df_future: \n{df_future.head()}")
        logger.info(f"{self.log_prefix} after extend_datetype_feature date_features: {date_features}")
        # ------------------------------
        # 特征工程：天气特征
        # ------------------------------
        df_weather_future = self.__process_df_timestamp(
            df=input_data[f"df_weather_future"],
            col_ts=self.args.weather_ts_feat
        )
        df_future, weather_features = self.extend_future_weather_feature(
            df=df_future,
            df_weather=df_weather_future,
            col_ts=self.args.weather_ts_feat,
        )
        # self.categorical_features += []
        logger.info(f"{self.log_prefix} after extend_future_weather_feature df_future: \n{df_future.head()}")
        logger.info(f"{self.log_prefix} after extend_future_weather_feature weather_features: {weather_features}")
        # ------------------------------
        # 外生特征收集
        # ------------------------------
        exogenous_features = date_features + weather_features
        logger.info(f"{self.log_prefix} exogenous_features: {exogenous_features}")
        # ------------------------------
        # 插值填充预测缺失值
        # ------------------------------
        df_future = df_future.interpolate(method="linear", limit_direction="both")
        df_future.dropna(inplace=True, ignore_index=True) # Drops rows where even interpolation couldn't help

        logger.info(f"{self.log_prefix} after interpolate and dropna df_future: \n{df_future.head()}")
        # target_feature_dummy is None
        return df_future, future_endogenous_cols_for_lag, exogenous_features, target_feature_dummy
    
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
            "week": lambda x: x.isocalendar().week, # Use isocalendar().week for consistency
            "day_of_week": lambda x: x.dayofweek,
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
        else:
            df_copy["datetime_minute_in_day"] = (df_copy["time"].dt.hour * (60 / freq_minutes)) + (df_copy["time"].dt.minute / freq_minutes)

        df_copy["datetime_minute_in_day_sin"] = np.sin(df_copy["datetime_minute_in_day"] * (2 * np.pi / self.n_per_day))
        df_copy["datetime_minute_in_day_cos"] = np.cos(df_copy["datetime_minute_in_day"] * (2 * np.pi / self.n_per_day))
        
        datetime_features = [col for col in df_copy.columns if col.startswith("datetime")]

        return df_copy, datetime_features

    def extend_datetype_feature(self, df: pd.DataFrame, df_date: pd.DataFrame, col_ts: str="date", col_type: str="date_type") -> Tuple[pd.DataFrame, List[str]]:
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
                # Assuming rt_tt2 and rt_dt are already in Celsius or will be used directly as is
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

                e_s_Td = 6.1078 * np.exp((17.27 * T_dew_C) / (237.3 + T_dew_C))
                e_s_T = 6.1078 * np.exp((17.27 * T_air_C) / (237.3 + T_air_C))
                
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

    def time_delay_embedding(self, series: pd.Series, n_lags: int, horizon: int, return_Xy: bool = False):
        """
        Time delay embedding
        Time series for supervised learning

        Args:
            series: time series as pd.Series
            n_lags: number of past values to used as explanatory variables
            horizon: how many values to forecast (used to separate X and Y)
            return_Xy: whether to return the lags split from future observations

        Return: pd.DataFrame with reconstructed time series
        """
        assert isinstance(series, pd.Series)
        # series name
        name = "Series" if series.name is None else series.name
        
        # Lags for X (past values)
        X_lags_list = [series.shift(i) for i in range(n_lags, 0, -1)]
        X_lags_df = pd.concat(X_lags_list, axis=1)
        X_lags_df.columns = [f'{name}_lag_{i}' for i in range(n_lags, 0, -1)]

        if not return_Xy:
            return X_lags_df
        
        # Targets for Y (future values)
        Y_targets_list = [series.shift(-i) for i in range(1, horizon + 1)]
        Y_targets_df = pd.concat(Y_targets_list, axis=1)
        Y_targets_df.columns = [f'{name}_shift_{i}' for i in range(1, horizon + 1)]

        # Combine and drop NaNs
        combined_df = pd.concat([X_lags_df, Y_targets_df], axis=1).dropna()
        
        # Split X and Y from combined_df
        X = combined_df[X_lags_df.columns]
        Y = combined_df[Y_targets_df.columns]

        return X, Y

    def extend_lag_feature_multivariate(self, df: pd.DataFrame, endogenous_cols: List[str], n_lags: int, horizon: int):
        """
        添加滞后特征 
        for multivariate time series, including targets for direct forecasting.
        endogenous_cols should include the primary target 'y' and other endogenous numeric features.
        """
        df_copy = df.copy()
        all_lag_features = []
        all_target_shifted_cols = []

        # 将 date 作为索引: Ensure 'time' is not in endogenous_cols before setting as index temporarily
        temp_df = df_copy.set_index("time").copy()
        
        for col in endogenous_cols:
            if col in temp_df.columns:
                # Generate lags for X
                lags_X = [temp_df[col].shift(i) for i in range(n_lags, 0, -1)]
                lag_col_names_X = [f'{col}_lag_{i}' for i in range(n_lags, 0, -1)]
                
                # Generate shifted targets for Y
                shifted_Y = [temp_df[col].shift(-i) for i in range(1, horizon + 1)]
                shifted_Y_names = [f'{col}_shift_{i}' for i in range(1, horizon + 1)]
                
                # Add to df_copy
                for i, name in enumerate(lag_col_names_X):
                    df_copy[name] = lags_X[i].values # Re-align to original df_copy index
                    all_lag_features.append(name)

                for i, name in enumerate(shifted_Y_names):
                    df_copy[name] = shifted_Y[i].values # Re-align to original df_copy index
                    all_target_shifted_cols.append(name)
        
        # Drop rows with NaNs generated by shifting
        # This will be done later during general NaN handling or just before model training/testing
        
        return df_copy, all_lag_features, all_target_shifted_cols

    def _generate_direct_multi_step_targets(self, df: pd.DataFrame, target_col: str, horizon: int):
        """
        Generates H shifted target columns for direct multi-step forecasting.
        """
        df_copy = df.copy()
        target_cols = []
        for h in range(1, horizon + 1):
            shifted_col_name = f"{target_col}_shift_{h}"
            df_copy[shifted_col_name] = df_copy[target_col].shift(-h)
            target_cols.append(shifted_col_name)
        return df_copy, target_cols

    def create_features(self, df_series: pd.DataFrame, endogenous_features: List[str], exogenous_features: List[str], target_feature: str):
        """
        特征工程
        Returns: df_series_copy, predictor_features, target_output_cols
        """
        df_series_copy = df_series.copy()
        # Clear and re-populate categorical_features for each run to avoid duplicates
        self.categorical_features = [] # Reset
        # ------------------------------
        # 特征工程：日期时间特征
        # ------------------------------
        df_series_copy, datetime_features = self.extend_datetime_feature(
            df=df_series_copy,
            feature_names=self.args.datetime_features,
            freq_minutes=self.args.freq_minutes,
        )
        self.categorical_features.extend([
            "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week",
            "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month",
            "datetime_quarter", "datetime_day_of_year", "datetime_year",
        ])
        self.categorical_features = list(set(self.categorical_features)) # Ensure uniqueness
        # ------------------------------
        # 特征工程：滞后特征
        # ------------------------------
        lag_features = []
        target_output_cols = []
        
        # For multi-output recursive, we need lags for ALL endogenous variables.
        # endogenous_features list passed to this function already includes 'y' (primary target).
        all_endogenous_for_lags = endogenous_features # This contains 'y' and other numeric features from ModelConfig

        if self.args.pred_method == "univariate-multi-step-directly":
            # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
            df_series_copy, target_output_cols = self._generate_direct_multi_step_targets(
                df=df_series_copy,
                target_col=target_feature,
                horizon=self.horizon
            )
            # For univariate, only target lags are features
            df_series_copy, uni_lag_features = self.extend_lag_feature_univariate(
                df = df_series_copy,
                target = target_feature,
                lags = self.args.lags,
            )
            lag_features.extend(uni_lag_features)
        elif self.args.pred_method == "univariate-multi-step-recursive":
            df_series_copy, uni_lag_features = self.extend_lag_feature_univariate(
                df = df_series_copy,
                target = target_feature,
                lags = self.args.lags,
            )
            lag_features.extend(uni_lag_features)
            # For recursive, target is target_t+1
            df_series_copy[f"{target_feature}_shift_1"] = df_series_copy[target_feature].shift(-1)
            target_output_cols = [f"{target_feature}_shift_1"]
        elif self.args.pred_method == "multivariate-multi-step-directly":
            # Direct multi-step: create H target columns (Y_t+1, ..., Y_t+H)
            df_series_copy, target_output_cols = self._generate_direct_multi_step_targets(
                df=df_series_copy,
                target_col=target_feature,
                horizon=self.horizon
            )
            # For multivariate, target and other endogenous lags are features
            # endogenous_features already contains the primary target
            df_series_copy, multi_lag_features, _ = self.extend_lag_feature_multivariate( # _ for targets, handled by _generate_direct_multi_step_targets
                df = df_series_copy,
                endogenous_cols = all_endogenous_for_lags,
                n_lags = max(self.args.lags), # Use max lag to define n_lags for TDE
                horizon = 1 # Not used for target_output_cols here, just for TDE structure
            )
            lag_features.extend(multi_lag_features)
        elif self.args.pred_method == "multivariate-multi-step-recursive":
            # For multivariate recursive, lags of target and other endogenous are features
            # The 'horizon' for extend_lag_feature_multivariate here is 1 because we are creating y_t+1 for *all* endogenous variables
            # which will then be passed to the model as a multi-output target.
            df_series_copy, multi_lag_features, multi_shifted_targets = self.extend_lag_feature_multivariate(
                df = df_series_copy,
                endogenous_cols = all_endogenous_for_lags,
                n_lags = max(self.args.lags), # Use max lag to define n_lags for TDE
                horizon = 1 # We want target_t+1 for all endogenous for training
            )
            lag_features.extend(multi_lag_features)
            # The target_output_cols are the shifted values for ALL endogenous variables
            # Ensure the primary target is first for consistent evaluation
            primary_target_shifted_name = f"{target_feature}_shift_1"
            if primary_target_shifted_name in multi_shifted_targets:
                target_output_cols.append(primary_target_shifted_name)
                # Add other shifted targets, excluding the primary if already added
                target_output_cols.extend([col for col in multi_shifted_targets if col != primary_target_shifted_name])
            else:
                target_output_cols.extend(multi_shifted_targets)

        # Feature ordering
        exogenous_features_all = exogenous_features + datetime_features
        
        # Predictor features are lags, exogenous, datetime.
        # Ensure 'time' column is not included in predictor_features.
        predictor_features = lag_features + exogenous_features_all

        # Add original endogenous features as predictors if they are not part of targets (e.g. for MMSD, MMSR)
        # Only add current values of endogenous features that are not primary target and not having their lags already included.
        # For MMSR, current values of endogenous features might be needed if they are not already covered by lags or exogenous.
        # For simplicity, let's include non-lagged non-target endogenous_features as current predictors.
        current_endogenous_as_features = [col for col in endogenous_features if col not in target_output_cols and col not in lag_features and col != target_feature]
        predictor_features.extend(current_endogenous_as_features)

        # Remove potential duplicates and 'time'
        predictor_features = list(set(predictor_features))
        if "time" in predictor_features:
            predictor_features.remove("time")
        
        # Filter df_series_copy to only include necessary columns to avoid errors later
        all_cols_needed = ["time"] + predictor_features + target_output_cols
        # Filter existing columns
        existing_cols = [col for col in all_cols_needed if col in df_series_copy.columns]
        df_series_copy = df_series_copy[existing_cols]

        return df_series_copy, predictor_features, target_output_cols
    # ##############################
    # forecast
    # ##############################
    def univariate_multi_step_directly_forecast(self, model, X_test_input, scaler_features=None):
        """
        模型预测 - 单变量多步直接预测
        """
        if not X_test_input.empty:
            # Scale features
            if scaler_features is not None:
                if self.args.encode_categorical_features:
                    categorical_features = list(set(self.categorical_features))
                    numeric_features = [col for col in X_test_input.columns if col not in categorical_features]
                    X_test_input_scaled = X_test_input.copy()
                    if numeric_features:
                        X_test_input_scaled.loc[:, numeric_features] = scaler_features.transform(X_test_input_scaled[numeric_features])
                    for col in categorical_features:
                        if col in X_test_input_scaled.columns:
                            X_test_input_scaled.loc[:, col] = X_test_input_scaled[col].apply(lambda x: int(x))
                    X_test_input_processed = X_test_input_scaled
                else:
                    X_test_input_processed = scaler_features.transform(X_test_input)
            else:
                X_test_input_processed = X_test_input
            
            Y_pred_multi_step = model.predict(X_test_input_processed)
            
            # Y_pred_multi_step will be (num_test_samples, H). We need the first column for evaluation.
            # In `forecast` phase, X_test_input is typically one row, returning (1, H).
            # The result for evaluation is always the first step.
            return Y_pred_multi_step[:, 0] # Return predictions for target_t+1
        else:
            return np.array([])

    def univariate_multi_step_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, target_output_cols, scaler_features=None):
        """
        递归多步预测 - 单变量
        """
        y_preds = []

        # Start with the latest available data to construct initial features
        # This `last_known_data` will hold values (actuals or predictions) for lag features.
        # We need enough history to form the maximum lag.
        max_lag = max(self.args.lags) if self.args.lags else 1
        last_known_data = df_history.iloc[-max_lag:].copy()
        
        # Ensure target_feature is present in last_known_data
        if target_feature not in last_known_data.columns and target_feature in df_history.columns:
            last_known_data[target_feature] = df_history[target_feature].iloc[-max_lag:]
            
        for step in range(self.horizon):
            logger.info(f"{self.log_prefix} univariate-recursive forecast step: {step}...")
            # 1. Prepare current features for prediction
            if step >= len(df_future):
                # If df_future is exhausted, we cannot get new exogenous features
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            # Future exogenous and datetime for this step
            current_step_df = df_future.iloc[step:step+1].copy()
            # Combine last known data (for lags) and current step data (for exogenous/datetime)
            # Need to align the 'time' column for proper feature creation
            combined_df = pd.concat([last_known_data, current_step_df], ignore_index=True)
            # Create features for prediction. We only need the last row from this for X_test_input.
            temp_df_featured, predictor_features, _ = self.create_features(
                df_series=combined_df,
                endogenous_features=endogenous_features, # These include 'y'
                exogenous_features=exogenous_features,
                target_feature=target_feature,
            )
            # The actual features for prediction are the last row
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]
            # Feature scaling
            if scaler_features is not None:
                if self.args.encode_categorical_features:
                    categorical_features = list(set(self.categorical_features))
                    numeric_features = [col for col in current_features_for_pred.columns if col not in categorical_features]
                    current_features_for_pred_scaled = current_features_for_pred.copy()
                    if numeric_features:
                        current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(current_features_for_pred_scaled[numeric_features])
                    for col in categorical_features:
                        if col in current_features_for_pred_scaled.columns:
                            current_features_for_pred_scaled.loc[:, col] = current_features_for_pred_scaled[col].apply(lambda x: int(x))
                    current_features_for_pred_processed = current_features_for_pred_scaled
                else:
                    current_features_for_pred_processed = scaler_features.transform(current_features_for_pred)
            else:
                current_features_for_pred_processed = current_features_for_pred
            # 2. Make prediction
            y_pred_step = model.predict(current_features_for_pred_processed)[0] # single value
            y_preds.append(y_pred_step)

            # 3. Update `last_known_data` for the next recursive step
            # Create a new row based on current_step_df (future time, exogenous features)
            new_row_for_last_known = current_step_df.copy().iloc[-1:]
            new_row_for_last_known[target_feature] = y_pred_step # Add the prediction as the new 'actual' for lag calculation
            # Concatenate and keep only the necessary number of past points for the next lag calculation
            last_known_data = pd.concat([last_known_data, new_row_for_last_known], ignore_index=True)
            last_known_data = last_known_data.iloc[-max_lag:]

        return np.array(y_preds)
    
    def multivariate_multi_step_directly_forecast(self, model, X_test_input, scaler_features=None):
        """
        模型预测 - 多变量多步直接预测
        """
        if not X_test_input.empty:
            # Scale features
            if scaler_features is not None:
                if self.args.encode_categorical_features:
                    categorical_features = list(set(self.categorical_features))
                    numeric_features = [col for col in X_test_input.columns if col not in categorical_features]
                    X_test_input_scaled = X_test_input.copy()
                    if numeric_features:
                        X_test_input_scaled.loc[:, numeric_features] = scaler_features.transform(X_test_input_scaled[numeric_features])
                    for col in categorical_features:
                        if col in X_test_input_scaled.columns:
                            X_test_input_scaled.loc[:, col] = X_test_input_scaled[col].apply(lambda x: int(x))
                    X_test_input_processed = X_test_input_scaled
                else:
                    X_test_input_processed = scaler_features.transform(X_test_input)
            else:
                X_test_input_processed = X_test_input
            
            Y_pred_multi_step = model.predict(X_test_input_processed)
            # Y_pred_multi_step will be (num_test_samples, H). We need the first column for evaluation.
            # In `forecast` phase, X_test_input is typically one row, returning (1, H).
            # The result for evaluation is always the first step.
            return Y_pred_multi_step[:, 0] # Return predictions for target_t+1
        else:
            return np.array([])

    def multivariate_multi_step_recursive_forecast(self, model, df_history, df_future, endogenous_features, exogenous_features, target_feature, target_output_cols: List, scaler_features=None):
        """
        多变量多步递归预测 (predicts all endogenous variables recursively)
        """
        # target_output_cols contains names like "target_feature_shift_1", "endogenous1_shift_1".
        # model predicts target_t+1, endogenous1_t+1, ...
        
        y_preds_primary_target = []
        
        # Start with the latest available data to construct initial features
        # `last_known_data` will hold values (actuals or predictions) for all endogenous variables needed for lags.
        all_endogenous_original_cols = [col.replace("_shift_1", "") for col in target_output_cols] # original names of predicted endogenous
        
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
            temp_df_featured, predictor_features, _ = self.create_features(
                df_series=combined_df,
                endogenous_features=endogenous_features, # All actual endogenous cols including target
                exogenous_features=exogenous_features,
                target_feature=target_feature,
            )
            # The last row of temp_df_featured contains the features for the current step's prediction
            current_features_for_pred = temp_df_featured[predictor_features].iloc[-1:]

            # Feature scaling
            if scaler_features is not None:
                if self.args.encode_categorical_features:
                    categorical_features = list(set(self.categorical_features))
                    numeric_features = [col for col in current_features_for_pred.columns if col not in categorical_features]
                    current_features_for_pred_scaled = current_features_for_pred.copy()
                    if numeric_features:
                        current_features_for_pred_scaled.loc[:, numeric_features] = scaler_features.transform(current_features_for_pred_scaled[numeric_features])
                    for col in categorical_features:
                        if col in current_features_for_pred_scaled.columns:
                            current_features_for_pred_scaled.loc[:, col] = current_features_for_pred_scaled[col].apply(lambda x: int(x))
                    current_features_for_pred_processed = current_features_for_pred_scaled
                else:
                    current_features_for_pred_processed = scaler_features.transform(current_features_for_pred)
            else:
                current_features_for_pred_processed = current_features_for_pred
            
            # 2. Make prediction for the next step for all target_output_cols
            next_pred_values_array = model.predict(current_features_for_pred_processed)[0] # [0] because predict returns [[val1, val2, ...]]
            # Map predictions back to their shifted column names
            next_pred_dict = dict(zip(target_output_cols, next_pred_values_array))

            # Store the prediction for the primary target (assuming it's the first in target_output_cols)
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
    # ##############################
    # Testing: sliding windows
    # ##############################
    def _evaluate_split_index(self, window: int, total_data_points: int):
        """
        数据分割索引构建
        Calculates train/test split indices for a sliding window.
        Assumes total_data_points is the length of `df_history_featured` after dropna,
        so `self.horizon` refers to the number of samples in the test set.
        The window slides from the most recent data backwards.
        """
        # Calculate test end index from the end of the data.
        # For window 1, test_end is -1 (last element of data_X)
        # For window 2, test_end is -1 - horizon (last element - horizon)
        test_end = total_data_points - 1 - (self.horizon * (window - 1))
        # Calculate test start index based on horizon
        test_start = test_end - self.horizon + 1
        # Calculate train end index (just before test_start)
        train_end = test_start
        # Calculate train start index based on window_len
        train_start = train_end - self.window_len
        # Ensure indices are not negative
        train_start = max(0, train_start)

        return train_start, train_end, test_start, test_end

    def _evaluate_split(self, data_X, data_Y, df_history, window: int):
        """
        训练、测试数据集分割
        """
        logger.info(f"{self.log_prefix} Model Testing sliding window...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        total_data_points = len(data_X) # Use X length as reference for available data
        # 数据分割指标
        train_start, train_end, test_start, test_end = self._evaluate_split_index(window, total_data_points)

        # Data slicing, handle cases where indices might be out of bounds for the window
        if train_start >= train_end or test_start >= test_end + 1 or train_start < 0 or test_end >= total_data_points:
            logger.warning(f"{self.log_prefix} Insufficient data for window {window} (train_start={train_start}, train_end={train_end}, test_start={test_start}, test_end={test_end}). Skipping this window.")
            return None, None, None, None, None, None

        X_train = data_X.iloc[train_start:train_end]
        Y_train = data_Y.iloc[train_start:train_end]
        X_test = data_X.iloc[test_start:test_end+1] # +1 to include test_end
        Y_test = data_Y.iloc[test_start:test_end+1]
        
        df_history_train = df_history.iloc[train_start:train_end]
        df_history_test = df_history.iloc[test_start:test_end+1]

        logger.info(f"{self.log_prefix} split indexes:: [train_start:train_end]: [{train_start}:{train_end}]")
        logger.info(f"{self.log_prefix} split indexes:: [test_start:test_end]: [{test_start}:{test_end+1}]")
        logger.info(f"{self.log_prefix} X_train.shape: {X_train.shape}, Y_train.shape: {Y_train.shape}")
        logger.info(f"{self.log_prefix} X_test.shape: {X_test.shape}, Y_test.shape: {Y_test.shape}")

        if X_train.empty or Y_train.empty or X_test.empty or Y_test.empty:
            logger.warning(f"{self.log_prefix} Empty dataframe in window {window} split. Skipping.")
            return None, None, None, None, None, None

        return X_train, Y_train, X_test, Y_test, df_history_train, df_history_test

    def _evaluate_score(self, y_test, y_pred, df_history_test, window: int=1):
        """
        模型评估
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

    def _evaluate_result(self, y_test, y_pred, window: int, cv_timestamp_df: pd.DataFrame):
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
        time_slice = cv_timestamp_df["time"].iloc[test_start_ts_idx : test_end_ts_idx + 1]
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
        lgbm_base = lgb.LGBMRegressor(**self.model_params)

        # Wrap in MultiOutputRegressor if the method is multi-output
        if self.args.pred_method in ["univariate-multi-step-directly", "multivariate-multi-step-directly", "multivariate-multi-step-recursive"]:
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
        
        return search.best_estimator_ # Return the best model directly
    
    # TODO
    def _calc_features_corr(self, df, train_features):
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
    
    def _window_test(self, X_train, Y_train, X_test, Y_test, df_history_train, df_history_test, endogenous_features, exogenous_features, target_feature, target_output_cols):
        """
        模型滑窗测试
        """
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{self.log_prefix} Model Testing training start...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        model, scaler_features = self.train(X_train, Y_train)
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{self.log_prefix} Model Testing forecasting start...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        Y_pred = None
        if self.args.pred_method == "univariate-multi-step-directly":
            Y_pred = self.univariate_multi_step_directly_forecast(
                model = model,
                X_test_input = X_test, # X_test could be multiple rows here
                scaler_features = scaler_features,
            )
        elif self.args.pred_method == "univariate-multi-step-recursive":
            Y_pred = self.univariate_multi_step_recursive_forecast(
                model = model,
                df_history = df_history_train, # Used for last actuals
                df_future = df_history_test, # Used for exogenous/datetime features for recursive prediction
                endogenous_features=endogenous_features,
                exogenous_features=exogenous_features,
                target_feature=target_feature,
                target_output_cols=target_output_cols, # Not strictly needed for univariate recursive, but good for consistency
                scaler_features = scaler_features,
            )
        elif self.args.pred_method == "multivariate-multi-step-directly":
            Y_pred = self.multivariate_multi_step_directly_forecast(
                model = model,
                X_test_input = X_test, # X_test could be multiple rows here
                scaler_features = scaler_features,
            )
        elif self.args.pred_method == "multivariate-multi-step-recursive":
            Y_pred = self.multivariate_multi_step_recursive_forecast(
                model = model,
                df_history = df_history_train, # Used for last actuals of all endogenous
                df_future = df_history_test, # Used for exogenous/datetime features for recursive prediction
                endogenous_features = endogenous_features,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                target_output_cols = target_output_cols,
                scaler_features = scaler_features,
            )
        
        if Y_pred is None or len(Y_pred) == 0:
            logger.error(f"{self.log_prefix} Prediction failed or returned empty for method: {self.args.pred_method}. Returning empty array.")
            return np.array([]) # Return empty array if prediction fails or is empty

        return Y_pred
    # ##############################
    # Model testm train, forecast
    # ##############################
    def test(self, df_history, endogenous_features, exogenous_features, target_feature, X_train_history, Y_train_history, target_output_cols):
        """
        模型滑窗测试
        """
        # Model testing results collection
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()

        # Max number of windows to run, ensuring enough data for at least one full test horizon
        total_available_points_for_test = len(X_train_history)
        max_windows_to_run = (total_available_points_for_test - self.window_len - self.horizon + 1) // self.horizon
        max_windows_to_run = min(self.n_windows, max_windows_to_run) # Cap by user defined n_windows

        if max_windows_to_run <= 0:
            logger.warning(f"{self.log_prefix} Not enough data for testing with current window configuration (Total X points: {total_available_points_for_test}, Window length: {self.window_len}, Horizon: {self.horizon}). No tests will be performed.")
            return test_scores_df, cv_plot_df
        
        # Create full timestamp df once for evaluation plotting
        cv_timestamp_full_df = pd.DataFrame({"time": pd.date_range(self.train_start_time, self.train_end_time, freq=self.args.freq, inclusive="left")})

        # Model sliding window test
        for window in range(1, int(max_windows_to_run + 1)):
            logger.info(f"{self.log_prefix} {'-' * 40}")
            logger.info(f"{self.log_prefix} Model Testing window: {window}...")
            logger.info(f"{self.log_prefix} {'-' * 40}")
            # Data split: training set, test set
            (X_train, Y_train, 
             X_test, Y_test, 
             df_history_train, df_history_test) = self._evaluate_split(X_train_history, Y_train_history, df_history, window)
            
            if X_train is None: # Skip if split was invalid
                continue
            
            # Target feature processing (ensure Y_train is DataFrame for MultiOutputRegressor)
            Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
            Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
            
            # Model test prediction
            Y_pred = self._window_test(
                X_train, Y_train,
                X_test, Y_test,
                df_history_train, df_history_test,
                endogenous_features, exogenous_features, target_feature,
                target_output_cols
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

            # Testing set evaluation metrics
            eval_scores_window = self._evaluate_score(Y_test_for_eval, Y_pred, df_history_test, window)
            test_scores_df = pd.concat([test_scores_df, eval_scores_window], axis=0)

            # Testing set prediction data for visualization
            cv_plot_df_window = self._evaluate_result(Y_test_for_eval, Y_pred, window, cv_timestamp_full_df)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis=0)
            
        # Model testing evaluation metrics data processing
        if not test_scores_df.empty:
            test_scores_df_mean = test_scores_df.drop(columns=["time_range"]).mean()
            test_scores_df_mean = test_scores_df_mean.to_frame().T.reset_index(drop=True, inplace=False)
            test_scores_df_mean["time_range"] = "均值"
            test_scores_df = pd.concat([test_scores_df, test_scores_df_mean], axis=0)

        logger.info(f"{self.log_prefix} {'-' * 40}")
        logger.info(f"{self.log_prefix} Model Testing result...")
        logger.info(f"{self.log_prefix} {'-' * 40}")
        logger.info(f"{self.log_prefix} Model Testing test_scores_df: \n{test_scores_df}")
        logger.info(f"{self.log_prefix} Model Testing cv_plot_df: \n{cv_plot_df.head()}")
        
        return test_scores_df, cv_plot_df
    
    def train(self, data_X, data_Y):
        """
        模型训练
        """
        # Training set
        X_train_df = data_X.copy()
        Y_train_df = data_Y.copy()
        # Scaling/Standardization
        categorical_features = []
        if self.args.encode_categorical_features:
            categorical_features = [f for f in list(set(self.categorical_features)) if f in X_train_df.columns]
        
        scaler_features = None
        if self.args.scale:
            scaler_features = MinMaxScaler()
            # If categorical features are distinguished
            numeric_features = [col for col in X_train_df.columns if col not in categorical_features]
            if not X_train_df[numeric_features].empty: # Only scale if there are numeric features
                X_train_df.loc[:, numeric_features] = scaler_features.fit_transform(X_train_df[numeric_features])
            # Categorical features are converted to int if encoded
            # LightGBM can handle categorical features directly if specified, no need to convert to int if not one-hot encoding.
            # But for consistency with scaling logic if encode_categorical_features is True:
            for col in categorical_features:
                if col in X_train_df.columns:
                    X_train_df.loc[:, col] = X_train_df[col].astype('category').cat.codes # Convert to integer codes for LightGBM
        else:
            # If no scaling, still ensure categorical are int if encoded
            if self.args.encode_categorical_features:
                for col in categorical_features:
                    if col in X_train_df.columns:
                        X_train_df.loc[:, col] = X_train_df[col].astype('category').cat.codes

        # Hyperparameter tuning (if enabled)
        if self.args.perform_tuning:
            best_model = self._hyperparameters_tuning(X_train_df, Y_train_df)
            return best_model, scaler_features

        lgbm_estimator = lgb.LGBMRegressor(**self.model_params)
        
        # Model training
        if self.args.pred_method == "univariate-multi-step-recursive":
            # Single output for recursive methods (predicts next single step of primary target)
            model = lgbm_estimator
            model.fit(
                X_train_df,
                Y_train_df,
                categorical_feature=categorical_features,
                eval_set=[(X_train_df, Y_train_df)],
                eval_metric="mae",
                callbacks=[lgb.early_stopping(self.args.patience, verbose=False)],
            )
        else: # "univariate-multi-step-directly", "multivariate-multi-step-directly", "multivariate-multi-step-recursive"
            # Multi-output for direct methods (predicts H future steps of single target)
            # Multi-output for multivariate recursive (predicts next step of multiple endogenous targets)
            model = MultiOutputRegressor(estimator=lgbm_estimator)
            # Note: MultiOutputRegressor does not directly support `eval_set`, `eval_metric`, `callbacks`
            # for the outer estimator. These should be passed to the base estimator (`lgbm_estimator`) if needed.
            # The current setup passes them via `self.model_params` to the `lgbm_estimator` constructor.
            model.fit(
                X_train_df,
                Y_train_df,
            )
        logger.info(f"{self.log_prefix} model: \n{model}")

        return model, scaler_features

    def forecast(self, df_history, df_future_raw, endogenous_features, exogenous_features, target_feature, X_train_history, Y_train_history, target_output_cols):
        """
        模型预测
        """
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Training start...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        # Model training
        model, scaler_features_train = self.train(X_train_history, Y_train_history)
        # Model saving
        model_dir = self.model_save(model)
        logger.info(f"{self.log_prefix} Model Training result saved in: {model_dir}")
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Forecasting start...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        Y_pred = np.array([]) # Initialize Y_pred
        df_future_for_prediction = df_future_raw.copy() # Use a copy of raw future data
        
        # Determine the endogenous features for lag creation for future data
        # This includes the primary target, and other numeric endogenous features from ModelConfig
        endogenous_cols_for_future_lags = endogenous_features # This list already contains 'y' and others.
                                                              # For df_future_raw, 'y' will be NaN.

        if self.args.pred_method in ["univariate-multi-step-directly", "multivariate-multi-step-directly"]:
            # For direct forecasting, we need to construct a single feature vector
            # from the last known data point in history to predict the entire horizon.
            
            # Combine last relevant history for lags and the first future point for exogenous/datetime
            max_lag_val = max(self.args.lags) if self.args.lags else 1
            relevant_history_for_lags = df_history.iloc[-max_lag_val:].copy()
            
            # Use the first point of df_future_for_prediction for exogenous features
            forecast_start_point_exogenous = df_future_for_prediction.iloc[0:1].copy()

            # For lag generation, combined_for_features needs a 'time' column for features.
            combined_for_features = pd.concat([relevant_history_for_lags, forecast_start_point_exogenous], ignore_index=True)
            
            # Generate features for this combined data, then take the last row as input for prediction
            # The 'target_feature' here refers to 'y' in df_history.
            # 'endogenous_features' for create_features (lag generation) should be all those that might have lags.
            combined_featured, predictor_features_for_forecast, _ = self.create_features(
                df_series=combined_for_features,
                endogenous_features=endogenous_cols_for_future_lags,
                exogenous_features=exogenous_features,
                target_feature=target_feature, # This 'y' is used to define target-specific lags
            )
            
            # The actual features for prediction are the last row (corresponding to the moment before forecast starts)
            X_forecast_input = combined_featured[predictor_features_for_forecast].iloc[-1:]
            
            # Scale features
            if scaler_features_train is not None:
                if self.args.encode_categorical_features:
                    categorical_features = list(set(self.categorical_features))
                    numeric_features = [col for col in X_forecast_input.columns if col not in categorical_features]
                    X_forecast_input_scaled = X_forecast_input.copy()
                    if numeric_features:
                        X_forecast_input_scaled.loc[:, numeric_features] = scaler_features_train.transform(X_forecast_input_scaled[numeric_features])
                    for col in categorical_features:
                        if col in X_forecast_input_scaled.columns:
                            X_forecast_input_scaled.loc[:, col] = X_forecast_input_scaled[col].apply(lambda x: int(x))
                    X_forecast_input_processed = X_forecast_input_scaled
                else:
                    X_forecast_input_processed = scaler_features_train.transform(X_forecast_input)
            else:
                X_forecast_input_processed = X_forecast_input

            Y_pred_multi_step = model.predict(X_forecast_input_processed)[0] # Expects (1, H) output -> [H] array
            
            # Assign predictions to df_future_for_prediction
            if len(Y_pred_multi_step) >= len(df_future_for_prediction):
                Y_pred = Y_pred_multi_step[:len(df_future_for_prediction)]
            else:
                logger.warning(f"Predicted {len(Y_pred_multi_step)} steps but df_future requires {len(df_future_for_prediction)} rows. Padding predictions.")
                Y_pred = np.pad(Y_pred_multi_step, (0, len(df_future_for_prediction) - len(Y_pred_multi_step)), 'edge')
        elif self.args.pred_method == "univariate-multi-step-recursive":
            Y_pred = self.univariate_multi_step_recursive_forecast(
                model = model,
                df_history = df_history, # Original history for lags
                df_future = df_future_for_prediction, # Processed future exogenous/datetime
                endogenous_features=endogenous_features, # All endogenous features
                exogenous_features=exogenous_features,
                target_feature=target_feature,
                target_output_cols=target_output_cols, # Not strictly needed here
                scaler_features = scaler_features_train,
            )
        elif self.args.pred_method == "multivariate-multi-step-recursive":
            Y_pred = self.multivariate_multi_step_recursive_forecast(
                model = model,
                df_history = df_history, # Original history for lags
                df_future = df_future_for_prediction, # Processed future exogenous/datetime
                endogenous_features = endogenous_features, # All endogenous features
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                target_output_cols = target_output_cols, # For recursive methods to know which variables to predict
                scaler_features = scaler_features_train,
            )
        
        df_future_for_prediction["predict_value"] = Y_pred
        
        logger.info(f"{self.log_prefix} after forecast df_future: \n{df_future_for_prediction.head()}")
        logger.info(f"{self.log_prefix} after forecast df_future.shape: {df_future_for_prediction.shape}")
        
        return df_future_for_prediction
    # ##############################
    # Model results save
    # ##############################
    def forecast_results_visual(self, y_trues_df, y_preds_df, n_per_day, path):
        # Only plot the last 2 days of true history for context, if available
        if not y_trues_df.empty:
            y_trues_df_plot = y_trues_df.iloc[-2 * n_per_day:]
        else:
            y_trues_df_plot = pd.DataFrame()

        plt.figure(figsize=(25, 8))
        if not y_trues_df_plot.empty and 'y' in y_trues_df_plot.columns:
            plt.plot(y_trues_df_plot["time"], y_trues_df_plot["y"], label='Trues', lw=2.0)
        
        if not y_preds_df.empty and 'predict_value' in y_preds_df.columns:
            plt.plot(y_preds_df["time"], y_preds_df["predict_value"], label='Preds', lw=2.0, ls="-.")
        
        plt.legend()
        plt.title("模型预测预测", fontsize=14)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path.joinpath('prediction.png'), bbox_inches='tight', dpi=300)
        # plt.show();

    def predict_result_visual(self, preds: np.array, trues: np.array, path: Path):
        """
        Results visualization for test set
        """
        if len(preds) == 0 or len(trues) == 0:
            logger.warning(f"{self.log_prefix} No data to visualize for test prediction.")
            return

        # 画布
        plt.figure(figsize=(25, 8))
        # 创建折线图
        plt.plot(trues, label='Trues', lw=1.7, )
        plt.plot(preds, label='Preds', lw=1.7, ls="-.")
        # 增强视觉效果
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title('Trues and Preds Timeseries Plot')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight', dpi=300)
        # plt.show();

    def model_save(self, model):
        """
        模型保存
        """
        # model path
        model_dir = Path(self.args.checkpoints).joinpath(self.setting)
        model_dir.mkdir(parents=True, exist_ok=True)
        # model save
        model_deploy = ModelDeployPkl(save_file_path=model_dir.joinpath("model.pkl"))
        model_deploy.save_model(model)
        logger.info(f"{self.log_prefix} Model saved to {model_dir.joinpath('model.pkl')}")
        
        return model_dir

    def test_results_save(self, test_scores_df, cv_plot_df):
        # 保存路径
        test_results_dir = Path(self.args.test_results).joinpath(self.setting + "-1")
        test_results_dir.mkdir(parents=True, exist_ok=True)
        # 结果保存
        test_scores_df.to_csv(test_results_dir.joinpath("test_scores_df.csv"), index=False, encoding="utf-8")
        cv_plot_df.to_csv(test_results_dir.joinpath("cv_plot_df.csv"), index=False, encoding="utf-8")
        self.predict_result_visual(
            preds=cv_plot_df["Y_preds"].values,
            trues=cv_plot_df["Y_trues"].values,
            path=test_results_dir.joinpath("test_prediction.png")
        )
        
        return test_results_dir

    def forecast_results_save(self, df_history, df_future):
        """
        输出结果处理
        """
        # 保存路径
        pred_results_dir = Path(self.args.pred_results).joinpath(self.setting)
        pred_results_dir.mkdir(parents=True, exist_ok=True)
        # 预测结果保存
        df_future["time"] = pd.to_datetime(df_future["time"])
        df_future_clean = df_future[["time", "predict_value"]]
        df_future_clean = df_future_clean.sort_values(by=["time"])
        df_future_clean.to_csv(pred_results_dir.joinpath("prediction.csv"), encoding="utf_8_sig", index=False)
        # 预测结果可视化
        self.forecast_results_visual(
            y_trues_df=df_history,
            y_preds_df=df_future_clean,
            n_per_day=self.n_per_day,
            path=pred_results_dir
        )
        
        return pred_results_dir
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
        # endogenous_features now includes the primary target
        (df_history, endogenous_features, exogenous_features, target_feature) = self.process_history_data(input_data = input_data)
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {80*'='}")
        df_history_featured, predictor_features, target_output_cols = self.create_features( # target_output_cols added
            df_series = df_history,
            endogenous_features = endogenous_features, # All endogenous, including target
            exogenous_features = exogenous_features,
            target_feature = target_feature,
        )
        df_history_featured = df_history_featured.dropna() # Drop rows with NaNs after feature/target generation
        logger.info(f"{self.log_prefix} df_history_featured: \n{df_history_featured.head()}")
        logger.info(f"{self.log_prefix} predictor_features: \n{predictor_features}")
        logger.info(f"{self.log_prefix} target_output_cols: \n{target_output_cols}")
        
        # Ensure target_feature is not None
        if target_feature is None:
            logger.error(f"{self.log_prefix} Primary target feature could not be identified. Cannot proceed with training or forecasting.")
            return

        # Ensure target_output_cols is not empty
        if not target_output_cols:
             logger.error(f"{self.log_prefix} Target output columns (Y) are empty. Cannot proceed with training or forecasting. Check create_features logic.")
             return
        
        # Select X and Y
        X_train_history = df_history_featured[predictor_features]
        Y_train_history = df_history_featured[target_output_cols] # Y_train_history uses target_output_cols

        # Drop any remaining NaNs from X and Y that might occur after specific feature/target generation.
        # This is critical to ensure training data is clean.
        combined_xy = pd.concat([X_train_history, Y_train_history], axis=1)
        combined_xy.dropna(inplace=True)
        X_train_history = combined_xy[X_train_history.columns]
        Y_train_history = combined_xy[Y_train_history.columns]

        logger.info(f"{self.log_prefix} X_train_history shape after NaN drop: {X_train_history.shape}")
        logger.info(f"{self.log_prefix} Y_train_history shape after NaN drop: {Y_train_history.shape}")

        if X_train_history.empty or Y_train_history.empty:
            logger.error(f"{self.log_prefix} Training data is empty after feature engineering and NaN removal. Cannot proceed.")
            return
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_testing:
            logger.info(f"{self.log_prefix} {80*'='}")
            logger.info(f"{self.log_prefix} Model Testing...")
            logger.info(f"{self.log_prefix} {80*'='}")
            # Model sliding window test
            test_scores_df, cv_plot_df = self.test(
                df_history, # original history without feature engineering for plotting and split reference
                endogenous_features, # All endogenous, including target
                exogenous_features,
                target_feature,
                X_train_history, # X and Y already processed
                Y_train_history,
                target_output_cols # Pass for recursive methods
            )
            # Test results saving
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Testing result save...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            test_results_dir = self.test_results_save(test_scores_df, cv_plot_df)
            logger.info(f"{self.log_prefix} Model Testing result saved in: {test_results_dir}")
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
            # df_future_raw contains time, exogenous features for the forecast horizon
            df_future_raw, future_endogenous_cols_for_lag, future_exogenous, future_target_dummy = self.process_future_data(input_data = input_data)
            # Model forecasting
            df_future_predicted = self.forecast(
                df_history, # Original history for lags
                df_future_raw, # Processed future exogenous/datetime
                endogenous_features, # All endogenous features (used to generate features for future)
                exogenous_features,
                target_feature,
                X_train_history, # From training phase
                Y_train_history, # From training phase
                target_output_cols # For recursive methods
            )
            # Model forecast results saving
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Forecasting result save...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            pred_results_dir = self.forecast_results_save(df_history, df_future_predicted)
            logger.info(f"{self.log_prefix} Model Forecasting result saved in: {pred_results_dir}")


# 测试代码 main 函数
def main():
    # model config
    args = ModelConfig()
    # model instance
    model = Model(args)
    model.run()

if __name__ == "__main__":
    main()
