# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_forecasting_ml_v2.py
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
import catboost as cab
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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


@dataclass
class ModelConfig:
    """
    模型配置类
    包含数据路径、特征设置、模型参数等所有配置项
    """
    # ------------------------------
    # 数据配置
    # ------------------------------
    # 数据路径
    data_dir = Path("./dataset/electricity_work/demand_load/lingang_A")

    # 目标时间序列配置
    data_path = "AIDC_A_dataset.csv"              # 目标时间序列数据路径
    data = "AIDC_A_dataset"                       # 目标时间序列数据名称
    freq = "5min"                                 # 目标时间序列数据频率
    freq_minutes = 5                              # 目标时间序列数据频率(分钟)
    target_ts_feat = "count_data_time"            # 目标时间序列时间戳特征名称
    target_series_numeric_features = []           # 目标时间序列的数值特征 (其他内生变量)
    target_series_categorical_features = []       # 目标时间序列的类别特征
    target_series_drop_features = []              # 目标时间序列的删除特征
    target = "h_total_use"                        # 目标时间序列预测目标变量名称

    # 日期类型数据配置
    date_history_path = None
    date_future_path = None
    date_ts_feat = "date"                         # 日期数据时间戳名称
    
    # 天气数据配置
    weather_history_path = None
    weather_future_path = None
    weather_ts_feat = "ts"                        # 天气数据时间戳名称
    
    # ------------------------------
    # 模型配置
    # ------------------------------
    model_name = "LightGBM"
    
    # 数据预处理
    scale = False                                 # 是否进行归一化/标准化
    inverse = False                               # 是否进行归一化/标准化逆变换
    
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    lags = [
        1 * 288,  # Daily lag
        2 * 288,
        3 * 288,
        4 * 288,
        5 * 288,
        6 * 288,
        7 * 288,  # Weekly lag
    ]                                             # 特征滞后数列表
    
    datetype_features = ["date_type"]
    weather_features = []
    datetime_features = []                        # 日期时间特征
    
    # 类别特征
    datetype_categorical_features = ["date_type"]
    weather_categorical_features = []
    datetime_categorical_features = []
    
    # ------------------------------
    # 预测方法配置
    # ------------------------------
    # 可选预测方法:
    # - "univariate-single-multistep-direct-output"      # USMDO
    # - "univariate-single-multistep-direct"             # USMD
    # - "univariate-single-multistep-recursive"          # USMR
    # - "univariate-single-multistep-direct-recursive"   # USMDR
    # - "multivariate-single-multistep-direct"           # MSMD
    # - "multivariate-single-multistep-recursive"        # MSMR
    # - "multivariate-single-multistep-direct-recursive" # MSMDR
    pred_method = "univariate-single-multistep-direct-output"
    
    featuresd = "MS"                              # 模型预测方式
    objective = "regression_l1"                   # 训练目标
    loss = "mae"                                  # 训练损失函数
    learning_rate = 0.05                          # 模型学习率
    patience = 100                                # 早停步数
    
    # ------------------------------
    # 训练和预测配置
    # ------------------------------
    history_days = 31                             # 历史数据天数
    predict_days = 1                              # 预测未来天数
    window_days = 15                              # 滑动窗口天数
    forecast_horizon_day = predict_days           # 预测时间范围(天)
    forecast_horizon_in_steps = 288 * forecast_horizon_day  # 预测时间范围(步数)
    
    # ------------------------------
    # 模型运行模式
    # ------------------------------
    is_testing = True                             # 是否进行模型测试
    is_forecasting = True                         # 是否进行模型预测
    
    # ------------------------------
    # 结果保存路径
    # ------------------------------
    test_results_dir = Path("./saved_results/results_test")
    forecast_results_dir = Path("./saved_results/results_forecast")


class Model:
    """
    时间序列预测模型类
    支持多种预测策略，包括单变量和多变量、直接和递归预测方法
    """
    
    def __init__(self, args: ModelConfig):
        """
        初始化模型
        
        Args:
            args: 模型配置对象
        """
        self.args = args
        self.log_prefix = f"[{self.args.model_name}]"
        
        # 创建结果保存目录
        self.args.test_results_dir.mkdir(parents=True, exist_ok=True)
        self.args.forecast_results_dir.mkdir(parents=True, exist_ok=True)
    
    # ==============================
    # 数据加载模块
    # ==============================
    
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
            df_target[self.args.target_ts_feat] = pd.to_datetime(df_target[self.args.target_ts_feat])
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
                df_date_history[self.args.date_ts_feat] = pd.to_datetime(df_date_history[self.args.date_ts_feat])
                input_data["date_history"] = df_date_history
                logger.info(f"{self.log_prefix} Date history loaded: {df_date_history.shape}")
        
        # 加载未来日期类型数据
        if self.args.date_future_path:
            date_future_path = self.args.data_dir / self.args.date_future_path
            if date_future_path.exists():
                df_date_future = pd.read_csv(date_future_path)
                df_date_future[self.args.date_ts_feat] = pd.to_datetime(df_date_future[self.args.date_ts_feat])
                input_data["date_future"] = df_date_future
                logger.info(f"{self.log_prefix} Date future loaded: {df_date_future.shape}")
        
        # 加载历史天气数据
        if self.args.weather_history_path:
            weather_history_path = self.args.data_dir / self.args.weather_history_path
            if weather_history_path.exists():
                df_weather_history = pd.read_csv(weather_history_path)
                df_weather_history[self.args.weather_ts_feat] = pd.to_datetime(df_weather_history[self.args.weather_ts_feat])
                input_data["weather_history"] = df_weather_history
                logger.info(f"{self.log_prefix} Weather history loaded: {df_weather_history.shape}")
        
        # 加载未来天气数据
        if self.args.weather_future_path:
            weather_future_path = self.args.data_dir / self.args.weather_future_path
            if weather_future_path.exists():
                df_weather_future = pd.read_csv(weather_future_path)
                df_weather_future[self.args.weather_ts_feat] = pd.to_datetime(df_weather_future[self.args.weather_ts_feat])
                input_data["weather_future"] = df_weather_future
                logger.info(f"{self.log_prefix} Weather future loaded: {df_weather_future.shape}")
        
        return input_data
    
    # ==============================
    # 数据预处理模块
    # ==============================
    
    def process_history_data(self, input_data: Dict) -> Tuple:
        """
        处理历史数据，合并目标序列与外生变量
        
        Args:
            input_data: 包含各类数据的字典
            
        Returns:
            处理后的历史数据DataFrame及特征列表
        """
        df_target = input_data["target_series"].copy()
        df_target = df_target.rename(columns={self.args.target_ts_feat: "time"})
        df_target = df_target.sort_values("time").reset_index(drop=True)
        
        logger.info(f"{self.log_prefix} Initial target series shape: {df_target.shape}")
        
        # 删除指定的特征列
        if self.args.target_series_drop_features:
            df_target = df_target.drop(columns=self.args.target_series_drop_features, errors='ignore')
        
        # 合并日期类型数据
        if "date_history" in input_data:
            df_date = input_data["date_history"].copy()
            df_date = df_date.rename(columns={self.args.date_ts_feat: "time"})
            df_target = pd.merge(df_target, df_date, on="time", how="left")
            logger.info(f"{self.log_prefix} After merging date history: {df_target.shape}")
        
        # 合并天气数据
        if "weather_history" in input_data:
            df_weather = input_data["weather_history"].copy()
            df_weather = df_weather.rename(columns={self.args.weather_ts_feat: "time"})
            df_target = pd.merge(df_target, df_weather, on="time", how="left")
            logger.info(f"{self.log_prefix} After merging weather history: {df_target.shape}")
        
        # 确定内生变量（包括目标变量）
        endogenous_features_with_target = [self.args.target] + self.args.target_series_numeric_features
        
        # 确定外生变量
        exogenous_features = (
            self.args.datetype_features +
            self.args.weather_features +
            self.args.datetime_features
        )
        
        # 确定目标变量
        target_feature = self.args.target
        
        # 确定类别特征
        categorical_features = (
            self.args.target_series_categorical_features +
            self.args.datetype_categorical_features +
            self.args.weather_categorical_features +
            self.args.datetime_categorical_features
        )
        
        logger.info(f"{self.log_prefix} Endogenous features (with target): {endogenous_features_with_target}")
        logger.info(f"{self.log_prefix} Exogenous features: {exogenous_features}")
        logger.info(f"{self.log_prefix} Target feature: {target_feature}")
        logger.info(f"{self.log_prefix} Categorical features: {categorical_features}")
        
        return df_target, endogenous_features_with_target, exogenous_features, target_feature, categorical_features
    
    def process_future_data(self, input_data: Dict) -> Tuple:
        """
        处理未来数据，用于预测
        
        Args:
            input_data: 包含各类数据的字典
            
        Returns:
            未来数据DataFrame及相关特征列表
        """
        df_target = input_data["target_series"].copy()
        df_target = df_target.rename(columns={self.args.target_ts_feat: "time"})
        
        # 生成未来时间序列
        last_time = df_target["time"].max()
        future_times = pd.date_range(
            start=last_time + pd.Timedelta(minutes=self.args.freq_minutes),
            periods=self.args.forecast_horizon_in_steps,
            freq=self.args.freq
        )
        df_future = pd.DataFrame({"time": future_times})
        
        logger.info(f"{self.log_prefix} Future time range: {df_future['time'].min()} to {df_future['time'].max()}")
        
        # 合并未来日期类型数据
        if "date_future" in input_data:
            df_date_future = input_data["date_future"].copy()
            df_date_future = df_date_future.rename(columns={self.args.date_ts_feat: "time"})
            df_future = pd.merge(df_future, df_date_future, on="time", how="left")
            logger.info(f"{self.log_prefix} After merging date future: {df_future.shape}")
        
        # 合并未来天气数据
        if "weather_future" in input_data:
            df_weather_future = input_data["weather_future"].copy()
            df_weather_future = df_weather_future.rename(columns={self.args.weather_ts_feat: "time"})
            df_future = pd.merge(df_future, df_weather_future, on="time", how="left")
            logger.info(f"{self.log_prefix} After merging weather future: {df_future.shape}")
        
        # 确定未来内生变量（用于计算滞后特征）
        future_endogenous_features_for_lag = [self.args.target] + self.args.target_series_numeric_features
        
        # 确定未来外生变量
        future_exogenous_features = (
            self.args.datetype_features +
            self.args.weather_features +
            self.args.datetime_features
        )
        
        logger.info(f"{self.log_prefix} Future endogenous features for lag: {future_endogenous_features_for_lag}")
        logger.info(f"{self.log_prefix} Future exogenous features: {future_exogenous_features}")
        
        return df_future, future_endogenous_features_for_lag, future_exogenous_features
    
    # ==============================
    # 特征工程模块
    # ==============================
    
    def create_lag_features(self, df: pd.DataFrame, features: List[str], lags: List[int]) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入数据框
            features: 需要创建滞后特征的列名列表
            lags: 滞后步数列表
            
        Returns:
            添加了滞后特征的数据框
        """
        df_lagged = df.copy()
        
        for feature in features:
            if feature in df.columns:
                for lag in lags:
                    df_lagged[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
        
        return df_lagged
    
    def create_features(self, df_series: pd.DataFrame, 
                       endogenous_features_with_target: List[str],
                       exogenous_features: List[str],
                       target_feature: str,
                       categorical_features: List[str]) -> Tuple:
        """
        根据预测方法创建相应的特征
        
        Args:
            df_series: 时间序列数据
            endogenous_features_with_target: 内生变量列表（包含目标变量）
            exogenous_features: 外生变量列表
            target_feature: 目标变量名
            categorical_features: 类别特征列表
            
        Returns:
            添加特征后的数据框、预测特征列表、目标输出特征列表、类别特征列表
        """
        df_featured = df_series.copy()
        predictor_features = []
        target_output_features = []
        
        # 确定实际存在的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_featured.columns]
        
        # 根据不同的预测方法创建特征
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            # USMDO: 单变量多步直接输出
            # 特征: 目标变量的滞后特征 + 外生变量
            # 目标: 未来多步的目标变量（同时输出）
            
            # 创建滞后特征（仅目标变量）
            df_featured = self.create_lag_features(df_featured, [target_feature], self.args.lags)
            
            # 预测特征 = 滞后特征 + 外生变量
            lag_features = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1到horizon步的目标变量
            for h in range(1, self.args.forecast_horizon_in_steps + 1):
                target_col = f"{target_feature}_future_{h}"
                df_featured[target_col] = df_featured[target_feature].shift(-h)
                target_output_features.append(target_col)
        
        elif self.args.pred_method == "univariate-single-multistep-direct":
            # USMD: 单变量多步直接预测
            # 特征: 目标变量的滞后特征 + 外生变量
            # 目标: 未来某一步的目标变量（每步单独训练模型）
            
            # 创建滞后特征（仅目标变量）
            df_featured = self.create_lag_features(df_featured, [target_feature], self.args.lags)
            
            # 预测特征 = 滞后特征 + 外生变量
            lag_features = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1步的目标变量（训练时只需要1步）
            target_output_features = [target_feature]
        
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            # USMR: 单变量多步递归预测
            # 特征: 目标变量的滞后特征 + 外生变量
            # 目标: 未来1步的目标变量（递归预测）
            
            # 创建滞后特征（仅目标变量）
            df_featured = self.create_lag_features(df_featured, [target_feature], self.args.lags)
            
            # 预测特征 = 滞后特征 + 外生变量
            lag_features = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1步的目标变量
            target_output_features = [target_feature]
        
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            # USMDR: 单变量多步直接递归预测
            # 这是USMD和USMR的混合策略
            # 特征: 目标变量的滞后特征 + 外生变量
            # 目标: 未来1步的目标变量
            
            # 创建滞后特征（仅目标变量）
            df_featured = self.create_lag_features(df_featured, [target_feature], self.args.lags)
            
            # 预测特征 = 滞后特征 + 外生变量
            lag_features = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1步的目标变量
            target_output_features = [target_feature]
        
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            # MSMD: 多变量多步直接预测
            # 特征: 所有内生变量的滞后特征 + 外生变量
            # 目标: 未来某一步的目标变量
            
            # 创建滞后特征（所有内生变量）
            df_featured = self.create_lag_features(df_featured, endogenous_features_with_target, self.args.lags)
            
            # 预测特征 = 所有内生变量的滞后特征 + 外生变量
            lag_features = []
            for feature in endogenous_features_with_target:
                lag_features.extend([f"{feature}_lag_{lag}" for lag in self.args.lags])
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1步的目标变量
            target_output_features = [target_feature]
        
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            # MSMR: 多变量多步递归预测
            # 特征: 所有内生变量的滞后特征 + 外生变量
            # 目标: 未来1步的所有内生变量（递归预测）
            
            # 创建滞后特征（所有内生变量）
            df_featured = self.create_lag_features(df_featured, endogenous_features_with_target, self.args.lags)
            
            # 预测特征 = 所有内生变量的滞后特征 + 外生变量
            lag_features = []
            for feature in endogenous_features_with_target:
                lag_features.extend([f"{feature}_lag_{lag}" for lag in self.args.lags])
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1步的所有内生变量
            target_output_features = endogenous_features_with_target
        
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            # MSMDR: 多变量多步直接递归预测
            # 这是MSMD和MSMR的混合策略
            # 特征: 所有内生变量的滞后特征 + 外生变量
            # 目标: 未来1步的所有内生变量
            
            # 创建滞后特征（所有内生变量）
            df_featured = self.create_lag_features(df_featured, endogenous_features_with_target, self.args.lags)
            
            # 预测特征 = 所有内生变量的滞后特征 + 外生变量
            lag_features = []
            for feature in endogenous_features_with_target:
                lag_features.extend([f"{feature}_lag_{lag}" for lag in self.args.lags])
            predictor_features = lag_features + available_exogenous
            
            # 目标输出特征 = 未来1步的所有内生变量
            target_output_features = endogenous_features_with_target
        
        # 确保所有预测特征都存在于数据框中
        predictor_features = [f for f in predictor_features if f in df_featured.columns]
        
        logger.info(f"{self.log_prefix} Created {len(predictor_features)} predictor features")
        logger.info(f"{self.log_prefix} Created {len(target_output_features)} target output features")
        
        return df_featured, predictor_features, target_output_features, categorical_features
    
    # ==============================
    # 模型训练模块
    # ==============================
    
    def train_model(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, 
                   categorical_features: List[str]) -> object:
        """
        训练LightGBM模型
        
        Args:
            X_train: 训练特征
            Y_train: 训练目标
            categorical_features: 类别特征列表
            
        Returns:
            训练好的模型
        """
        # 确定实际存在的类别特征
        actual_categorical = [f for f in categorical_features if f in X_train.columns]
        
        # 根据目标输出的数量决定是否使用多输出模型
        if Y_train.shape[1] > 1:
            # 多输出回归
            base_model = lgb.LGBMRegressor(
                objective=self.args.objective,
                learning_rate=self.args.learning_rate,
                n_estimators=1000,
                random_state=42,
                verbose=-1
            )
            model = MultiOutputRegressor(base_model)
            logger.info(f"{self.log_prefix} Training MultiOutputRegressor with {Y_train.shape[1]} outputs")
        else:
            # 单输出回归
            model = lgb.LGBMRegressor(
                objective=self.args.objective,
                learning_rate=self.args.learning_rate,
                n_estimators=1000,
                random_state=42,
                verbose=-1
            )
            logger.info(f"{self.log_prefix} Training single output LGBMRegressor")
        
        # 训练模型
        if actual_categorical and not isinstance(model, MultiOutputRegressor):
            model.fit(
                X_train, 
                Y_train.values.ravel() if Y_train.shape[1] == 1 else Y_train,
                categorical_feature=actual_categorical
            )
        else:
            model.fit(
                X_train, 
                Y_train.values.ravel() if Y_train.shape[1] == 1 else Y_train
            )
        
        logger.info(f"{self.log_prefix} Model training completed")
        
        return model
    
    # ==============================
    # 模型评估模块
    # ==============================
    
    def evaluate_model(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> Dict:
        """
        评估模型性能
        
        Args:
            Y_true: 真实值
            Y_pred: 预测值
            
        Returns:
            包含各项评估指标的字典
        """
        metrics = {}
        
        try:
            metrics['MAE'] = mean_absolute_error(Y_true, Y_pred)
            metrics['RMSE'] = root_mean_squared_error(Y_true, Y_pred)
            metrics['MSE'] = mean_squared_error(Y_true, Y_pred)
            metrics['R2'] = r2_score(Y_true, Y_pred)
            
            # 避免MAPE计算时除以零
            mask = Y_true != 0
            if mask.sum() > 0:
                metrics['MAPE'] = mean_absolute_percentage_error(Y_true[mask], Y_pred[mask])
            else:
                metrics['MAPE'] = np.nan
        except Exception as e:
            logger.error(f"{self.log_prefix} Error in model evaluation: {e}")
            metrics = {'MAE': np.nan, 'RMSE': np.nan, 'MSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
        
        return metrics
    
    # ==============================
    # 预测方法实现模块
    # ==============================
    
    def univariate_single_multi_step_direct_output_forecast(self, model, df_history, df_future,
                                                            endogenous_features, exogenous_features,
                                                            target_feature, categorical_features,
                                                            scaler_features) -> np.ndarray:
        """
        USMDO: 单变量多步直接输出预测
        一次性预测未来所有步
        
        Args:
            model: 训练好的模型
            df_history: 历史数据
            df_future: 未来数据（包含外生变量）
            endogenous_features: 内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            categorical_features: 类别特征
            scaler_features: 特征缩放器
            
        Returns:
            预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting USMDO forecast")
        
        # 创建滞后特征（使用历史数据）
        df_history_lagged = self.create_lag_features(df_history, [target_feature], self.args.lags)
        
        # 获取最后一行的滞后特征值
        last_row = df_history_lagged.iloc[-1]
        
        # 准备预测特征
        lag_feature_names = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        # 准备预测数据（只需要第一行，因为是一次性输出所有步）
        X_pred_row = {}
        
        # 添加滞后特征
        for lag_feat in lag_feature_names:
            if lag_feat in last_row.index:
                X_pred_row[lag_feat] = last_row[lag_feat]
        
        # 添加外生变量（使用未来第一步的外生变量）
        for exo_feat in available_exogenous:
            X_pred_row[exo_feat] = df_future[exo_feat].iloc[0]
        
        # 转换为DataFrame
        X_pred = pd.DataFrame([X_pred_row])
        
        # 确保特征顺序与训练时一致
        predictor_features = lag_feature_names + available_exogenous
        X_pred = X_pred[predictor_features]
        
        # 预测（一次性输出所有步）
        Y_pred_all_steps = model.predict(X_pred)
        
        # Y_pred_all_steps的形状应该是(1, forecast_horizon_in_steps)
        # 我们需要取出第一行
        if Y_pred_all_steps.ndim == 2:
            Y_pred = Y_pred_all_steps[0, :]
        else:
            Y_pred = Y_pred_all_steps
        
        logger.info(f"{self.log_prefix} USMDO forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    def univariate_single_multi_step_directly_forecast(self, model, df_history, df_future,
                                                       endogenous_features, exogenous_features,
                                                       target_feature, categorical_features,
                                                       scaler_features) -> np.ndarray:
        """
        USMD: 单变量多步直接预测
        为每个预测步训练单独的模型，然后逐步预测
        
        注意：此方法需要为每个步长训练不同的模型，这里简化为使用同一个模型
        实际应用中应该为每个步长h训练一个模型
        
        Args:
            model: 训练好的模型列表或单个模型
            df_history: 历史数据
            df_future: 未来数据
            endogenous_features: 内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            categorical_features: 类别特征
            scaler_features: 特征缩放器
            
        Returns:
            预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting USMD forecast")
        
        # 创建滞后特征
        df_history_lagged = self.create_lag_features(df_history, [target_feature], self.args.lags)
        
        # 初始化预测结果
        Y_pred = np.zeros(len(df_future))
        
        # 准备滞后特征名称
        lag_feature_names = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
        
        # 确定实际可用的外生变量（只使用df_future中存在的）
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        predictor_features = lag_feature_names + available_exogenous
        
        # 逐步预测
        for h in range(len(df_future)):
            # 准备当前步的预测特征
            if h == 0:
                # 第一步使用历史数据的滞后特征
                last_row = df_history_lagged.iloc[-1]
            else:
                # 后续步需要更新滞后特征（但这里我们使用固定的历史滞后，因为是直接预测）
                last_row = df_history_lagged.iloc[-1]
            
            X_pred_row = {}
            
            # 添加滞后特征
            for lag_feat in lag_feature_names:
                if lag_feat in last_row.index:
                    X_pred_row[lag_feat] = last_row[lag_feat]
            
            # 添加当前步的外生变量（只添加可用的）
            for exo_feat in available_exogenous:
                X_pred_row[exo_feat] = df_future[exo_feat].iloc[h]
            
            # 转换为DataFrame
            X_pred = pd.DataFrame([X_pred_row])
            X_pred = X_pred[predictor_features]
            
            # 预测当前步
            Y_pred[h] = model.predict(X_pred)[0]
        
        logger.info(f"{self.log_prefix} USMD forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    def univariate_single_multi_step_recursive_forecast(self, model, df_history, df_future,
                                                        endogenous_features, exogenous_features,
                                                        target_feature, categorical_features,
                                                        scaler_features) -> np.ndarray:
        """
        USMR: 单变量多步递归预测
        递归地预测每一步，将预测值作为下一步的输入
        
        Args:
            model: 训练好的模型
            df_history: 历史数据
            df_future: 未来数据
            endogenous_features: 内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            categorical_features: 类别特征
            scaler_features: 特征缩放器
            
        Returns:
            预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting USMR forecast")
        
        # 创建历史数据的滞后特征
        df_history_lagged = self.create_lag_features(df_history, [target_feature], self.args.lags)
        
        # 初始化预测结果
        Y_pred = np.zeros(len(df_future))
        
        # 准备滞后特征名称
        lag_feature_names = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        predictor_features = lag_feature_names + available_exogenous
        
        # 获取历史数据的目标变量值（用于更新滞后特征）
        history_values = df_history[target_feature].values
        
        # 递归预测
        for h in range(len(df_future)):
            # 准备当前步的预测特征
            X_pred_row = {}
            
            # 添加滞后特征（考虑已预测的值）
            for lag in self.args.lags:
                lag_feat = f"{target_feature}_lag_{lag}"
                
                if h < lag:
                    # 如果滞后步数大于当前预测步数，从历史数据中获取
                    idx = len(history_values) - lag + h
                    if idx >= 0:
                        X_pred_row[lag_feat] = history_values[idx]
                    else:
                        X_pred_row[lag_feat] = 0
                else:
                    # 使用之前预测的值
                    X_pred_row[lag_feat] = Y_pred[h - lag]
            
            # 添加当前步的外生变量（只添加可用的）
            for exo_feat in available_exogenous:
                X_pred_row[exo_feat] = df_future[exo_feat].iloc[h]
            
            # 转换为DataFrame
            X_pred = pd.DataFrame([X_pred_row])
            X_pred = X_pred[predictor_features]
            
            # 预测当前步
            Y_pred[h] = model.predict(X_pred)[0]
        
        logger.info(f"{self.log_prefix} USMR forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    def univariate_single_multi_step_direct_recursive_forecast(self, model, df_history, df_future,
                                                               endogenous_features, exogenous_features,
                                                               target_feature, categorical_features,
                                                               scaler_features) -> np.ndarray:
        """
        USMDR: 单变量多步直接递归预测
        结合直接预测和递归预测的策略
        将预测horizon分成多个块，每个块使用直接预测，块之间使用递归
        
        Args:
            model: 训练好的模型
            df_history: 历史数据
            df_future: 未来数据
            endogenous_features: 内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            categorical_features: 类别特征
            scaler_features: 特征缩放器
            
        Returns:
            预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting USMDR forecast")
        
        # 创建历史数据的滞后特征
        df_history_lagged = self.create_lag_features(df_history, [target_feature], self.args.lags)
        
        # 初始化预测结果
        Y_pred = np.zeros(len(df_future))
        
        # 准备滞后特征名称
        lag_feature_names = [f"{target_feature}_lag_{lag}" for lag in self.args.lags]
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        predictor_features = lag_feature_names + available_exogenous
        
        # 获取历史数据的目标变量值
        history_values = df_history[target_feature].values
        
        # 定义块大小（可以根据需要调整）
        # 这里使用最小滞后作为块大小
        block_size = min(self.args.lags) if self.args.lags else 1
        
        # 分块递归预测
        for block_start in range(0, len(df_future), block_size):
            block_end = min(block_start + block_size, len(df_future))
            
            # 对当前块进行预测
            for h in range(block_start, block_end):
                # 准备当前步的预测特征
                X_pred_row = {}
                
                # 添加滞后特征
                for lag in self.args.lags:
                    lag_feat = f"{target_feature}_lag_{lag}"
                    
                    # 计算滞后位置
                    lag_pos = len(history_values) + h - lag
                    
                    if lag_pos >= len(history_values):
                        # 使用已预测的值
                        pred_idx = lag_pos - len(history_values)
                        X_pred_row[lag_feat] = Y_pred[pred_idx]
                    elif lag_pos >= 0:
                        # 使用历史值
                        X_pred_row[lag_feat] = history_values[lag_pos]
                    else:
                        X_pred_row[lag_feat] = 0
                
                # 添加当前步的外生变量（只添加可用的）
                for exo_feat in available_exogenous:
                    X_pred_row[exo_feat] = df_future[exo_feat].iloc[h]
                
                # 转换为DataFrame
                X_pred = pd.DataFrame([X_pred_row])
                X_pred = X_pred[predictor_features]
                
                # 预测当前步
                Y_pred[h] = model.predict(X_pred)[0]
        
        logger.info(f"{self.log_prefix} USMDR forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    def multivariate_single_multi_step_direct_forecast(self, model, df_history, df_future,
                                                       endogenous_features, exogenous_features,
                                                       target_feature, categorical_features,
                                                       scaler_features) -> np.ndarray:
        """
        MSMD: 多变量多步直接预测
        使用所有内生变量的滞后特征，直接预测目标变量
        
        Args:
            model: 训练好的模型
            df_history: 历史数据
            df_future: 未来数据
            endogenous_features: 所有内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            categorical_features: 类别特征
            scaler_features: 特征缩放器
            
        Returns:
            预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting MSMD forecast")
        
        # 创建所有内生变量的滞后特征
        df_history_lagged = self.create_lag_features(df_history, endogenous_features, self.args.lags)
        
        # 初始化预测结果
        Y_pred = np.zeros(len(df_future))
        
        # 准备滞后特征名称
        lag_feature_names = []
        for feature in endogenous_features:
            lag_feature_names.extend([f"{feature}_lag_{lag}" for lag in self.args.lags])
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        predictor_features = lag_feature_names + available_exogenous
        
        # 逐步预测
        for h in range(len(df_future)):
            # 准备当前步的预测特征
            last_row = df_history_lagged.iloc[-1]
            
            X_pred_row = {}
            
            # 添加所有内生变量的滞后特征
            for lag_feat in lag_feature_names:
                if lag_feat in last_row.index:
                    X_pred_row[lag_feat] = last_row[lag_feat]
            
            # 添加当前步的外生变量（只添加可用的）
            for exo_feat in available_exogenous:
                X_pred_row[exo_feat] = df_future[exo_feat].iloc[h]
            
            # 转换为DataFrame
            X_pred = pd.DataFrame([X_pred_row])
            X_pred = X_pred[predictor_features]
            
            # 预测当前步
            Y_pred[h] = model.predict(X_pred)[0]
        
        logger.info(f"{self.log_prefix} MSMD forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    def multivariate_single_multi_step_recursive_forecast(self, model, df_history, df_future,
                                                          endogenous_features, exogenous_features,
                                                          target_feature, target_output_features,
                                                          scaler_features) -> np.ndarray:
        """
        MSMR: 多变量多步递归预测
        递归预测所有内生变量，将预测值作为下一步的输入
        
        Args:
            model: 训练好的模型
            df_history: 历史数据
            df_future: 未来数据
            endogenous_features: 所有内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            target_output_features: 目标输出特征列表（所有内生变量）
            scaler_features: 特征缩放器
            
        Returns:
            目标变量的预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting MSMR forecast")
        
        # 创建所有内生变量的滞后特征
        df_history_lagged = self.create_lag_features(df_history, endogenous_features, self.args.lags)
        
        # 初始化预测结果（所有内生变量）
        predictions_dict = {feat: np.zeros(len(df_future)) for feat in endogenous_features}
        
        # 准备滞后特征名称
        lag_feature_names = []
        for feature in endogenous_features:
            lag_feature_names.extend([f"{feature}_lag_{lag}" for lag in self.args.lags])
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        predictor_features = lag_feature_names + available_exogenous
        
        # 获取历史数据的所有内生变量值
        history_values = {feat: df_history[feat].values for feat in endogenous_features}
        
        # 递归预测
        for h in range(len(df_future)):
            # 准备当前步的预测特征
            X_pred_row = {}
            
            # 添加所有内生变量的滞后特征
            for feature in endogenous_features:
                for lag in self.args.lags:
                    lag_feat = f"{feature}_lag_{lag}"
                    
                    if h < lag:
                        # 从历史数据中获取
                        idx = len(history_values[feature]) - lag + h
                        if idx >= 0:
                            X_pred_row[lag_feat] = history_values[feature][idx]
                        else:
                            X_pred_row[lag_feat] = 0
                    else:
                        # 使用之前预测的值
                        X_pred_row[lag_feat] = predictions_dict[feature][h - lag]
            
            # 添加当前步的外生变量（只添加可用的）
            for exo_feat in available_exogenous:
                X_pred_row[exo_feat] = df_future[exo_feat].iloc[h]
            
            # 转换为DataFrame
            X_pred = pd.DataFrame([X_pred_row])
            X_pred = X_pred[predictor_features]
            
            # 预测当前步（所有内生变量）
            Y_pred_step = model.predict(X_pred)
            
            # 存储预测结果
            if Y_pred_step.ndim == 2:
                Y_pred_step = Y_pred_step[0]
            
            for i, feat in enumerate(endogenous_features):
                predictions_dict[feat][h] = Y_pred_step[i]
        
        # 返回目标变量的预测结果
        Y_pred = predictions_dict[target_feature]
        
        logger.info(f"{self.log_prefix} MSMR forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    def multivariate_single_multi_step_direct_recursive_forecast(self, model, df_history, df_future,
                                                                 endogenous_features, exogenous_features,
                                                                 target_feature, categorical_features,
                                                                 scaler_features) -> np.ndarray:
        """
        MSMDR: 多变量多步直接递归预测
        结合多变量直接预测和递归预测的策略
        
        Args:
            model: 训练好的模型
            df_history: 历史数据
            df_future: 未来数据
            endogenous_features: 所有内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            categorical_features: 类别特征
            scaler_features: 特征缩放器
            
        Returns:
            目标变量的预测结果数组
        """
        logger.info(f"{self.log_prefix} Starting MSMDR forecast")
        
        # 创建所有内生变量的滞后特征
        df_history_lagged = self.create_lag_features(df_history, endogenous_features, self.args.lags)
        
        # 初始化预测结果
        Y_pred = np.zeros(len(df_future))
        
        # 为其他内生变量也准备预测数组（用于更新滞后特征）
        predictions_dict = {feat: np.zeros(len(df_future)) for feat in endogenous_features}
        
        # 准备滞后特征名称
        lag_feature_names = []
        for feature in endogenous_features:
            lag_feature_names.extend([f"{feature}_lag_{lag}" for lag in self.args.lags])
        
        # 确定实际可用的外生变量
        available_exogenous = [feat for feat in exogenous_features if feat in df_future.columns]
        
        predictor_features = lag_feature_names + available_exogenous
        
        # 获取历史数据的所有内生变量值
        history_values = {feat: df_history[feat].values for feat in endogenous_features}
        
        # 定义块大小
        block_size = min(self.args.lags) if self.args.lags else 1
        
        # 分块递归预测
        for block_start in range(0, len(df_future), block_size):
            block_end = min(block_start + block_size, len(df_future))
            
            # 对当前块进行预测
            for h in range(block_start, block_end):
                # 准备当前步的预测特征
                X_pred_row = {}
                
                # 添加所有内生变量的滞后特征
                for feature in endogenous_features:
                    for lag in self.args.lags:
                        lag_feat = f"{feature}_lag_{lag}"
                        
                        # 计算滞后位置
                        lag_pos = len(history_values[feature]) + h - lag
                        
                        if lag_pos >= len(history_values[feature]):
                            # 使用已预测的值
                            pred_idx = lag_pos - len(history_values[feature])
                            if feature == target_feature:
                                X_pred_row[lag_feat] = Y_pred[pred_idx]
                            else:
                                # 对于其他内生变量，使用简单的持久性预测
                                # （假设保持最后观测值不变）
                                X_pred_row[lag_feat] = history_values[feature][-1]
                        elif lag_pos >= 0:
                            # 使用历史值
                            X_pred_row[lag_feat] = history_values[feature][lag_pos]
                        else:
                            X_pred_row[lag_feat] = 0
                
                # 添加当前步的外生变量（只添加可用的）
                for exo_feat in available_exogenous:
                    X_pred_row[exo_feat] = df_future[exo_feat].iloc[h]
                
                # 转换为DataFrame
                X_pred = pd.DataFrame([X_pred_row])
                X_pred = X_pred[predictor_features]
                
                # 预测当前步
                Y_pred[h] = model.predict(X_pred)[0]
        
        logger.info(f"{self.log_prefix} MSMDR forecast completed, predicted {len(Y_pred)} steps")
        
        return Y_pred
    
    # ==============================
    # 模型测试模块
    # ==============================
    
    def test(self, df_history, X_train_history, Y_train_history,
            endogenous_features_with_target, exogenous_features,
            target_feature, target_output_features, categorical_features) -> Tuple:
        """
        使用时间序列交叉验证测试模型
        
        Args:
            df_history: 历史数据
            X_train_history: 训练特征
            Y_train_history: 训练目标
            endogenous_features_with_target: 内生变量列表
            exogenous_features: 外生变量列表
            target_feature: 目标变量
            target_output_features: 目标输出特征
            categorical_features: 类别特征
            
        Returns:
            测试评分DataFrame和交叉验证绘图数据
        """
        logger.info(f"{self.log_prefix} Starting time series cross-validation testing")
        
        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=5)
        
        test_scores = []
        cv_plot_data = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_history)):
            logger.info(f"{self.log_prefix} Processing fold {fold + 1}/5")
            
            # 分割训练集和测试集
            X_train_fold = X_train_history.iloc[train_idx]
            Y_train_fold = Y_train_history.iloc[train_idx]
            X_test_fold = X_train_history.iloc[test_idx]
            Y_test_fold = Y_train_history.iloc[test_idx]
            
            # 训练模型
            model = self.train_model(X_train_fold, Y_train_fold, categorical_features)
            
            # 预测
            Y_pred_fold = model.predict(X_test_fold)
            
            # 评估
            if Y_test_fold.shape[1] == 1:
                metrics = self.evaluate_model(Y_test_fold.values.ravel(), Y_pred_fold)
            else:
                # 对于多输出，计算平均指标
                metrics = self.evaluate_model(Y_test_fold.values, Y_pred_fold)
            
            metrics['fold'] = fold + 1
            test_scores.append(metrics)
            
            # 保存CV绘图数据
            cv_plot_data.append({
                'fold': fold + 1,
                'y_true': Y_test_fold.values,
                'y_pred': Y_pred_fold
            })
            
            logger.info(f"{self.log_prefix} Fold {fold + 1} - MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}")
        
        # 转换为DataFrame
        test_scores_df = pd.DataFrame(test_scores)
        
        logger.info(f"{self.log_prefix} Average test scores:")
        logger.info(f"{self.log_prefix} {test_scores_df.mean()}")
        
        return test_scores_df, cv_plot_data
    
    # ==============================
    # 模型预测模块
    # ==============================
    
    def forecast(self, df_history, X_train_history, Y_train_history, df_future,
                future_endogenous_features_for_lag, future_exogenous_features,
                target_feature, target_output_features, categorical_features) -> pd.DataFrame:
        """
        使用训练好的模型进行预测
        
        Args:
            df_history: 历史数据
            X_train_history: 训练特征
            Y_train_history: 训练目标
            df_future: 未来数据
            future_endogenous_features_for_lag: 未来内生变量（用于滞后）
            future_exogenous_features: 未来外生变量
            target_feature: 目标变量
            target_output_features: 目标输出特征
            categorical_features: 类别特征
            
        Returns:
            包含预测结果的DataFrame
        """
        logger.info(f"{self.log_prefix} Starting forecast on full training data")
        
        # 在全部训练数据上训练最终模型
        model = self.train_model(X_train_history, Y_train_history, categorical_features)
        
        # 准备预测数据
        df_future_for_prediction = df_future.copy()
        
        # 特征缩放器（如果需要）
        scaler_features_train = None
        if self.args.scale:
            scaler = StandardScaler()
            scaler_features_train = scaler.fit(X_train_history)
        
        # 根据预测方法调用相应的预测函数
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            Y_pred = self.univariate_single_multi_step_direct_output_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features,
                scaler_features=scaler_features_train,
            )
        
        elif self.args.pred_method == "univariate-single-multistep-direct":
            Y_pred = self.univariate_single_multi_step_directly_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features,
                scaler_features=scaler_features_train,
            )
        
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            Y_pred = self.univariate_single_multi_step_recursive_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features,
                scaler_features=scaler_features_train,
            )
        
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            Y_pred = self.univariate_single_multi_step_direct_recursive_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features,
                scaler_features=scaler_features_train,
            )
        
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            Y_pred = self.multivariate_single_multi_step_direct_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features,
                scaler_features=scaler_features_train,
            )
        
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            Y_pred = self.multivariate_single_multi_step_recursive_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                target_output_features=target_output_features,
                scaler_features=scaler_features_train,
            )
        
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            Y_pred = self.multivariate_single_multi_step_direct_recursive_forecast(
                model=model,
                df_history=df_history,
                df_future=df_future_for_prediction,
                endogenous_features=future_endogenous_features_for_lag,
                exogenous_features=future_exogenous_features,
                target_feature=target_feature,
                categorical_features=categorical_features,
                scaler_features=scaler_features_train,
            )
        
        else:
            logger.error(f"{self.log_prefix} Unknown prediction method: {self.args.pred_method}")
            Y_pred = np.zeros(len(df_future))
        
        # 收集预测结果
        df_future_for_prediction["predict_value"] = Y_pred
        df_future_for_prediction = df_future_for_prediction[["time", "predict_value"]]
        
        logger.info(f"{self.log_prefix} Forecast completed")
        logger.info(f"{self.log_prefix} Predicted shape: {df_future_for_prediction.shape}")
        
        return df_future_for_prediction
    
    # ==============================
    # 结果保存模块
    # ==============================
    
    def test_results_save(self, test_scores_df, cv_plot_df) -> Path:
        """
        保存测试结果
        
        Args:
            test_scores_df: 测试评分DataFrame
            cv_plot_df: 交叉验证绘图数据
            
        Returns:
            结果保存路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.args.test_results_dir / f"test_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存测试评分
        test_scores_path = results_dir / "test_scores.csv"
        test_scores_df.to_csv(test_scores_path, index=False)
        logger.info(f"{self.log_prefix} Test scores saved to {test_scores_path}")
        
        return results_dir
    
    def forecast_results_save(self, df_history, df_future_predicted) -> Path:
        """
        保存预测结果
        
        Args:
            df_history: 历史数据
            df_future_predicted: 预测结果
            
        Returns:
            结果保存路径
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.args.forecast_results_dir / f"forecast_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存预测结果
        forecast_path = results_dir / "forecast_results.csv"
        df_future_predicted.to_csv(forecast_path, index=False)
        logger.info(f"{self.log_prefix} Forecast results saved to {forecast_path}")
        
        # 可视化（如果需要）
        try:
            plt.figure(figsize=(15, 6))
            
            # 绘制历史数据（最后一部分）
            history_to_plot = df_history.tail(288 * 3)  # 最后3天
            plt.plot(history_to_plot['time'], history_to_plot[self.args.target], 
                    label='History', color='blue', alpha=0.7)
            
            # 绘制预测数据
            plt.plot(df_future_predicted['time'], df_future_predicted['predict_value'], 
                    label='Forecast', color='red', alpha=0.7)
            
            plt.xlabel('Time')
            plt.ylabel(self.args.target)
            plt.title(f'Forecast Results - {self.args.pred_method}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_path = results_dir / "forecast_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"{self.log_prefix} Forecast plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"{self.log_prefix} Could not create forecast plot: {e}")
        
        return results_dir
    
    # ==============================
    # 主运行流程
    # ==============================
    
    def run(self):
        """
        执行完整的模型训练和预测流程
        """
        # 数据加载
        logger.info(f"{self.log_prefix} {'=' * 80}")
        logger.info(f"{self.log_prefix} Model history and future data loading...")
        logger.info(f"{self.log_prefix} {'=' * 80}")
        input_data = self.load_data()
        
        # 历史数据处理
        logger.info(f"{self.log_prefix} {'=' * 80}")
        logger.info(f"{self.log_prefix} Model history data preprocessing...")
        logger.info(f"{self.log_prefix} {'=' * 80}")
        (df_history, 
         endogenous_features_with_target, 
         exogenous_features, 
         target_feature, 
         categorical_features
        ) = self.process_history_data(input_data=input_data)
        
        # 特征工程
        logger.info(f"{self.log_prefix} {'=' * 80}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {'=' * 80}")
        df_history_featured, predictor_features, target_output_features, categorical_features = self.create_features(
            df_series=df_history,
            endogenous_features_with_target=endogenous_features_with_target,
            exogenous_features=exogenous_features,
            target_feature=target_feature,
            categorical_features=categorical_features,
        )
        
        # 删除NaN行
        df_history_featured = df_history_featured.dropna()
        logger.info(f"{self.log_prefix} df_history_featured shape: {df_history_featured.shape}")
        
        # 特征分割
        logger.info(f"{self.log_prefix} {'=' * 80}")
        logger.info(f"{self.log_prefix} Model history data feature split...")
        logger.info(f"{self.log_prefix} {'=' * 80}")
        X_train_history = df_history_featured[predictor_features]
        Y_train_history = df_history_featured[target_output_features]
        
        # 删除任何剩余的NaN
        combined_xy = pd.concat([X_train_history, Y_train_history], axis=1)
        combined_xy.dropna(inplace=True)
        X_train_history = combined_xy[X_train_history.columns]
        Y_train_history = combined_xy[Y_train_history.columns]
        
        logger.info(f"{self.log_prefix} X_train_history shape: {X_train_history.shape}")
        logger.info(f"{self.log_prefix} Y_train_history shape: {Y_train_history.shape}")
        
        # 模型测试
        if self.args.is_testing:
            logger.info(f"{self.log_prefix} {'=' * 80}")
            logger.info(f"{self.log_prefix} Model Testing...")
            logger.info(f"{self.log_prefix} {'=' * 80}")
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
            
            # 保存测试结果
            test_results_dir = self.test_results_save(test_scores_df, cv_plot_df)
            logger.info(f"{self.log_prefix} Model Testing result saved in: {test_results_dir}")
        
        # 模型预测
        if self.args.is_forecasting:
            logger.info(f"{self.log_prefix} {'=' * 80}")
            logger.info(f"{self.log_prefix} Model Forecasting...")
            logger.info(f"{self.log_prefix} {'=' * 80}")
            
            # 未来数据处理
            (df_future, 
             future_endogenous_features_for_lag, 
             future_exogenous_features) = self.process_future_data(input_data=input_data)
            
            # 模型预测
            df_future_predicted = self.forecast(
                df_history,
                X_train_history,
                Y_train_history,
                df_future,
                future_endogenous_features_for_lag,
                future_exogenous_features,
                target_feature,
                target_output_features,
                categorical_features,
            )
            
            # 保存预测结果
            pred_results_dir = self.forecast_results_save(df_history, df_future_predicted)
            logger.info(f"{self.log_prefix} Model Forecasting result saved in: {pred_results_dir}")


# 测试代码 main 函数
def main():
    """
    主函数入口
    """
    # 模型配置
    args = ModelConfig()
    
    # 创建模型实例
    model = Model(args)
    
    # 运行模型
    model.run()


if __name__ == "__main__":
    main()
