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
# model selection
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
)

from config.model_config import (
    ModelConfig_univariate, ModelConfig_multivariate
)
from features.FeaturePreprocessor import (
    FeatureScaler, FeatureEngineer,
)
from models.ModelFactory import ModelFactory
from strategies.PredictionStrategy import PredictionHelper

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
        # 预测未来 1 天(24小时)的数据/数据划分长度/预测数据长度
        self.horizon = int(self.args.predict_days * self.n_per_day)
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
        logger.info(f"{self.log_prefix} 模型类型: {self.args.model_type}")
        logger.info(f"{self.log_prefix} 高级特征: {'启用' if self.args.enable_advanced_features else '禁用'}")
        logger.info(f"{self.log_prefix} 模型融合: {'启用' if self.args.enable_ensemble else '禁用'}")
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
        df_history, date_features = self.preprocessor.extend_datetype_feature(
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
        df_history, weather_features = self.preprocessor.extend_weather_feature(
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
        df_future, date_features = self.preprocessor.extend_datetype_feature(
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
        df_future, weather_features = self.preprocessor.extend_future_weather_feature(
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
    def predictor_target_split(self, df_history_featured, predictor_features, target_output_features):
        """
        历史数据预测特征、目标特征分离
        """
        X_train_history = df_history_featured[predictor_features]
        Y_train_history = df_history_featured[target_output_features]
        combined_xy = pd.concat([X_train_history, Y_train_history], axis=1)
        combined_xy.dropna(inplace=True)
        X_train_history = combined_xy[X_train_history.columns]
        Y_train_history = combined_xy[Y_train_history.columns]
        logger.info(f"{self.log_prefix} X_train_history shape after NaN drop: {X_train_history.shape}")
        logger.info(f"{self.log_prefix} Y_train_history shape after NaN drop: {Y_train_history.shape}")
        
        return X_train_history, Y_train_history

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
        self.scaler_testing = FeatureScaler(self.args, self.args.scaler_type, log_prefix=self.log_prefix)
        model = self.train(X_train, Y_train, self.scaler_testing, categorical_features)
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{self.log_prefix} Model Testing forecasting start...")
        logger.info(f"{self.log_prefix} {30*'-'}")
        Y_pred = None
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            Y_pred = self.univariate_single_multi_step_direct_output_forecast(
                model = model,
                model = model, 
                df_future = X_test.copy(), 
                endogenous_features = endogenous_features, 
                exogenous_features = exogenous_features, 
                target_feature = target_feature, 
                categorical_features = categorical_features,
                feature_scaler = self.scaler_testing
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
                feature_scaler = self.scaler_testing,
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
                feature_scaler = self.scaler_testing,
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
                feature_scaler = self.scaler_testing,
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
                feature_scaler = self.scaler_testing,
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
                feature_scaler = self.scaler_testing,
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
                feature_scaler = self.scaler_testing,
            )
        # Return empty array if prediction fails or is empty
        if Y_pred is None or len(Y_pred) == 0:
            logger.error(f"{self.log_prefix} Prediction failed or returned empty for method: {self.args.pred_method}. Returning empty array.")
            return np.array([])

        return Y_pred

    def test(self, df_history, X_train_history, Y_train_history, 
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
        lgbm_base = self.model_factory.create_model(model_type=self.args.model_type, model_params=self.args.model_params)

        # Wrap in MultiOutputRegressor if the method is multi-output
        if Y_train.shape[1] == 1:
            model_for_tuning = lgbm_base
        else:
            model_for_tuning = MultiOutputRegressor(lgbm_base)

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
    def train(self, X_train, Y_train, feature_scaler, categorical_features):
        """
        模型训练
        """
        logger.info(f"{self.log_prefix} 开始训练模型...")
        # 训练集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        # ------------------------------
        # 归一化/标准化
        # ------------------------------
        # 特征预处理（训练模式）
        X_train_df_processed, actual_categorical = feature_scaler.fit_transform(X_train_df, categorical_features)
        feature_scaler.validate_features(X_train_df_processed, stage="training")
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
        if self.args.enable_ensemble:
            # 模型融合
            logger.info(f"{self.log_prefix} 使用模型融合: {self.args.ensemble_models}")
            models = []
            for model_type in self.args.ensemble_models:
                base_model = self.model_factory.create_model(model_type=model_type, model_params=self.model_params)
                models.append(base_model)
            
            # 简化的融合：平均法
            for i, model in enumerate(models):
                logger.info(f"{self.log_prefix} 训练模型 {i+1}/{len(models)}: {self.args.ensemble_models[i]}")
                if Y_train_df.shape[1] == 1:
                    model.fit(X_train_df_processed, Y_train_df)
                else:
                    wrapped_model = MultiOutputRegressor(estimator=model.model)
                    wrapped_model.fit(X_train_df, Y_train_df)
                    model.model = wrapped_model
            # 返回模型列表（在预测时平均）
            return models
        else:
            # 单模型
            lgbm_estimator = self.model_factory.create_model(
                model_type=getattr(self.args, "model_type", "lightgbm"), 
                model_params=self.args.model_params
            )
            if Y_train_df.shape[1] == 1:
                # 单输出
                model = lgbm_estimator
                model.fit(X_train_df_processed, Y_train_df)
                logger.info(f"{self.log_prefix} Training single output LGBMRegressor")
            elif Y_train_df.shape[1] > 1:
                # 多输出
                model = MultiOutputRegressor(estimator=lgbm_estimator)
                model.fit(X_train_df_processed, Y_train_df)
                logger.info(f"{self.log_prefix} Training MultiOutputRegressor with {Y_train.shape[1]} outputs")
            logger.info(f"{self.log_prefix} Model training completed!")

            return model
    # ##############################
    # Model Forecast(Model Inference)
    # ##############################
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
    def forecast(self, df_history, X_train_history, Y_train_history, df_future, 
                 endogenous_features, exogenous_features, 
                 target_feature, target_output_features, categorical_features):
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
        self.scaler_forecasting = FeatureScaler(self.args, log_prefix=self.log_prefix)
        # 模型训练
        model = self.train(X_train_history, Y_train_history, self.scaler_forecasting, categorical_features)
        # 模型保存
        self.model_save(model)
        logger.info(f"{self.log_prefix} Model Training result saved in: {self.args.checkpoints_dir}")
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Forecasting start...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        predictor = PredictionHelper(
            args = self.args,
            model = model, 
            df_history = df_history, 
            df_future = df_future_for_prediction, 
            endogenous_features = endogenous_features, 
            exogenous_features = exogenous_features, 
            target_feature = target_feature, 
            target_output_features = target_output_features, 
            categorical_features = categorical_features,
            feature_scaler = self.scaler_forecasting,
            log_prefix = self.log_prefix,
        )
        # Initialize Y_pred
        Y_pred = np.array([])
        Y_preds = []
        # Use a copy of raw future data
        df_future_for_prediction = df_future.copy()
        
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            Y_pred = predictor.univariate_single_multi_step_direct_output_forecast()
        elif self.args.pred_method == "univariate-single-multistep-direct":
            Y_pred = predictor.univariate_single_multi_step_direct_forecast()
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            Y_pred = predictor.univariate_single_multi_step_recursive_forecast()
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            Y_pred = predictor.univariate_single_multi_step_direct_recursive_forecast()
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            Y_pred = predictor.multivariate_single_multi_step_direct_forecast()
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            Y_pred = predictor.multivariate_single_multi_step_recursive_forecast()
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            Y_pred = predictor.multivariate_single_multi_step_direct_recursive_forecast()
        
        # TODO 模型融合
        if isinstance(model, List):
            pass
        else:
            pass
        
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
        (
            df_history, 
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
        # 特征预处理器
        preprocessor_history = FeatureEngineer(self.args, self.log_prefix)
        (
            df_history_featured, 
            predictor_features, 
            target_output_features, 
            categorical_features
        ) = preprocessor_history.create_features(
            df_series = df_history,
            endogenous_features_with_target = endogenous_features_with_target,
            exogenous_features = exogenous_features,
            target_feature = target_feature,
            categorical_features = categorical_features,
        )
        # Drop rows with NaNs after feature/target generation
        df_history_featured = df_history_featured.dropna()
        logger.info(f"{self.log_prefix} df_history_featured: \n{df_history_featured.head()}")
        logger.info(f"{self.log_prefix} predictor_features: {predictor_features}")
        logger.info(f"{self.log_prefix} target_output_features: {target_output_features}")
        logger.info(f"{self.log_prefix} categorical_features: {categorical_features}")
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_testing:
            logger.info(f"{self.log_prefix} {80*'='}")
            logger.info(f"{self.log_prefix} Model Testing...")
            logger.info(f"{self.log_prefix} {80*'='}")
            # 历史数据预测特征、目标特征分离
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model history data feature split...")
            logger.info(f"{self.log_prefix} {40*'='}")
            X_train_history, Y_train_history = self.predictor_target_split(
                df_history_featured = df_history_featured, 
                predictor_features = predictor_features, 
                target_output_features = target_output_features,
            )
            
            # 模型滑窗测试
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Testing start...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            test_scores_df, cv_plot_df = self.test(
                df_history = df_history,
                X_train_history = X_train_history,
                Y_train_history = Y_train_history,
                endogenous_features_with_target = endogenous_features_with_target,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                target_output_features = target_output_features,
                categorical_features = categorical_features, 
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
            df_future = self.process_future_data(input_data = input_data)
            
            # 模型预测
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Forecasting start...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            df_future_predicted = self.forecast(
                df_history = df_history,
                X_train_history = X_train_history,
                Y_train_history = Y_train_history,
                df_future = df_future,
                endogenous_features = endogenous_features_with_target,
                exogenous_features = exogenous_features,
                target_feature = target_feature,
                target_output_features = target_output_features,
                categorical_features = categorical_features, 
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
    logger.info("预测流程完成！")

if __name__ == "__main__":
    main()
