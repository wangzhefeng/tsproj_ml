# -*- coding: utf-8 -*-

# ***************************************************
# * File        : exp_forecasting_ml_v3_optimized.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 3.0 (优化版)
# * Description : 优化的机器学习时间序列预测框架
# *               主要优化:
# *               1. 模型抽象化 - 支持多种ML模型切换
# *               2. 增强特征工程 - 添加滞后统计特征
# *               3. 统一预处理接口 - 消除代码重复
# *               4. 模型融合支持
# *               5. 中文注释优化
# *               
# *               支持7种预测方法:
# *               1. USMDO - 单变量多步直接输出预测
# *               2. USMD  - 单变量多步直接预测
# *               3. USMR  - 单变量多步递归预测
# *               4. USMDR - 单变量多步直接递归预测
# *               5. MSMD  - 多变量多步直接预测
# *               6. MSMR  - 多变量多步递归预测
# *               7. MSMDR - 多变量多步直接递归预测
# ***************************************************

import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import copy
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional, Any
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 机器学习模型
import xgboost as xgb
import lightgbm as lgb
import catboost as cab
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# 模型评估
from sklearn.metrics import (
    r2_score, mean_squared_error, root_mean_squared_error,
    mean_absolute_error, mean_absolute_percentage_error,
)

# 数据处理
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# 日志配置
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


# ==================== 模型抽象层 ====================

class BaseModel(ABC):
    """
    模型基类
    所有具体模型必须继承此类并实现抽象方法
    """
    
    def __init__(self, params: Dict[str, Any]):
        self.params = params
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X, **kwargs):
        """预测"""
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None


class LightGBMModel(BaseModel):
    """LightGBM模型封装"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        default_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = lgb.LGBMRegressor(**self.params)
    
    def fit(self, X, y, eval_set=None, categorical_features=None, 
            early_stopping_rounds=50, verbose=False):
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['callbacks'] = [lgb.early_stopping(early_stopping_rounds, verbose=verbose)]
        if categorical_features is not None:
            fit_params['categorical_feature'] = categorical_features
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost模型封装"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_jobs': -1,
            'random_state': 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = xgb.XGBRegressor(**self.params)
    
    def fit(self, X, y, eval_set=None, early_stopping_rounds=50, verbose=False):
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
            fit_params['verbose'] = verbose
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)


class CatBoostModel(BaseModel):
    """CatBoost模型封装"""
    
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'verbose': False,
            'random_state': 42,
        }
        default_params.update(params)
        self.params = default_params
        self.model = cab.CatBoostRegressor(**self.params)
    
    def fit(self, X, y, eval_set=None, categorical_features=None, early_stopping_rounds=50):
        fit_params = {}
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        if categorical_features is not None:
            fit_params['cat_features'] = categorical_features
        
        self.model.fit(X, y, **fit_params)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)


class ModelFactory:
    """模型工厂"""
    
    _models = {
        'lightgbm': LightGBMModel,
        'lgb': LightGBMModel,
        'xgboost': XGBoostModel,
        'xgb': XGBoostModel,
        'catboost': CatBoostModel,
        'cat': CatBoostModel,
    }
    
    @staticmethod
    def create_model(model_type: str, params: Dict[str, Any]) -> BaseModel:
        """创建模型实例"""
        model_type = model_type.lower()
        if model_type not in ModelFactory._models:
            supported = ', '.join(ModelFactory._models.keys())
            raise ValueError(f"不支持的模型类型: {model_type}. 支持的模型: {supported}")
        
        model_class = ModelFactory._models[model_type]
        return model_class(params)


# ==================== 增强特征工程 ====================

class AdvancedFeatureEngineer:
    """高级特征工程器"""
    
    def __init__(self, log_prefix="[FeatureEng]"):
        self.log_prefix = log_prefix
        self.generated_features = []
    
    def add_lag_statistics(self, df: pd.DataFrame, columns: List[str], 
                          windows: List[int], stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        添加滞后统计特征（滑动窗口统计）
        
        Args:
            df: 数据框
            columns: 需要计算统计特征的列
            windows: 窗口大小列表
            stats: 统计量列表 ['mean', 'std', 'min', 'max']
        """
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                if 'mean' in stats:
                    feature_name = f'{col}_rolling_mean_{window}'
                    df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).mean()
                    self.generated_features.append(feature_name)
                
                if 'std' in stats:
                    feature_name = f'{col}_rolling_std_{window}'
                    df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).std()
                    self.generated_features.append(feature_name)
                
                if 'min' in stats:
                    feature_name = f'{col}_rolling_min_{window}'
                    df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).min()
                    self.generated_features.append(feature_name)
                
                if 'max' in stats:
                    feature_name = f'{col}_rolling_max_{window}'
                    df_enhanced[feature_name] = df[col].rolling(window=window, min_periods=1).max()
                    self.generated_features.append(feature_name)
        
        logger.info(f"{self.log_prefix} 生成了 {len(self.generated_features)} 个滞后统计特征")
        return df_enhanced
    
    def add_diff_features(self, df: pd.DataFrame, columns: List[str], 
                         periods: List[int] = [1, 7]) -> pd.DataFrame:
        """
        添加差分特征
        
        Args:
            df: 数据框
            columns: 列名列表
            periods: 差分周期列表
        """
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for period in periods:
                feature_name = f'{col}_diff_{period}'
                df_enhanced[feature_name] = df[col].diff(period)
                self.generated_features.append(feature_name)
        
        return df_enhanced
    
    def get_generated_features(self) -> List[str]:
        """获取所有生成的特征列表"""
        return self.generated_features
    
    def reset(self):
        """重置生成的特征列表"""
        self.generated_features = []


# ==================== 统一特征缩放器 ====================

class UnifiedFeatureScaler:
    """统一的特征缩放器，消除代码重复"""
    
    def __init__(self, scaler_type='standard', encode_categorical=False):
        """
        初始化
        
        Args:
            scaler_type: 缩放器类型 ('standard' 或 'minmax')
            encode_categorical: 是否编码类别特征
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.encode_categorical = encode_categorical
        self.category_encoders = {}
        self.is_fitted = False
    
    def fit_transform(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """训练并转换特征"""
        X_scaled = X.copy()
        
        # 分离数值和类别特征
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # 缩放数值特征
        if numeric_features:
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
        
        self.is_fitted = True
        return X_scaled
    
    def transform(self, X: pd.DataFrame, categorical_features: List[str]) -> pd.DataFrame:
        """仅转换特征（使用已拟合的参数）"""
        if not self.is_fitted:
            raise ValueError("缩放器尚未拟合")
        
        X_scaled = X.copy()
        
        # 分离数值和类别特征
        numeric_features = [col for col in X.columns if col not in categorical_features]
        
        # 缩放数值特征
        if numeric_features:
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


# ==================== 配置类 ====================

@dataclass
class ModelConfig:
    """
    模型配置类（优化版）
    包含数据路径、特征设置、模型参数等所有配置项
    """
    # 数据配置
    data_dir: str = "./dataset/electricity_work/demand_load/lingang_A"
    data_path: str = "AIDC_A_dataset.csv"
    data: str = "AIDC_A_dataset"
    freq: str = "5min"
    freq_minutes: int = 5
    target_ts_feat: str = "count_data_time"
    target_series_numeric_features: List[str] = field(default_factory=list)
    target_series_categorical_features: List[str] = field(default_factory=list)
    target_series_drop_features: List[str] = field(default_factory=list)
    target: str = "h_total_use"
    
    # 外部数据配置
    date_history_path: Optional[str] = None
    date_future_path: Optional[str] = None
    date_ts_feat: Optional[str] = None
    weather_history_path: Optional[str] = None
    weather_future_path: Optional[str] = None
    weather_ts_feat: Optional[str] = None
    
    # 数据预处理
    scale: bool = False
    inverse: bool = False
    scaler_type: str = "standard"  # "standard" 或 "minmax"
    
    # 特征工程配置
    lags: List[int] = field(default_factory=list)
    datetime_features: List[str] = field(default_factory=lambda: [
        'hour', 'day', 'weekday', 'month', 'quarter'
    ])
    datetype_features: List[str] = field(default_factory=list)
    weather_features: List[str] = field(default_factory=list)
    datetime_categorical_features: List[str] = field(default_factory=list)
    datetype_categorical_features: List[str] = field(default_factory=list)
    weather_categorical_features: List[str] = field(default_factory=list)
    
    # 高级特征工程配置（新增）
    enable_advanced_features: bool = False  # 是否启用高级特征
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14])  # 滑动窗口大小
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std'])  # 统计量
    diff_periods: List[int] = field(default_factory=lambda: [1, 7])  # 差分周期
    
    # 训练和预测配置
    history_days: int = 31
    predict_days: int = 1
    window_days: int = 15
    encode_categorical_features: bool = False
    
    # 模型配置（优化：支持多种模型）
    model_type: str = "lightgbm"  # 'lightgbm', 'xgboost', 'catboost'
    model_name: str = "LightGBM"
    pred_method: str = "univariate-single-multistep-direct-output"
    objective: str = "regression_l1"
    loss: str = "mae"
    learning_rate: float = 0.05
    patience: int = 100
    
    # 模型融合配置（新增）
    enable_ensemble: bool = False  # 是否启用模型融合
    ensemble_models: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost'])
    ensemble_method: str = "averaging"  # 'averaging', 'weighted', 'stacking'
    
    # 运行模式
    is_testing: bool = False
    is_forecasting: bool = True
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2025, 12, 27, 0, 0, 0))
    
    # 结果保存路径
    checkpoints_dir: str = "./saved_results/pretrained_models/"
    test_results_dir: str = "./saved_results/results_test/"
    pred_results_dir: str = "./saved_results/results_forecast/"
    
    # 超参数调优
    perform_tuning: bool = False
    tuning_metric: str = "neg_mean_absolute_error"
    tuning_n_splits: int = 3


# ==================== 特征预处理器（保留原有功能）====================

class FeaturePreprocessor:
    """特征预处理器（包含原有的所有特征工程功能）"""
    
    def __init__(self, args, log_prefix="[FeaturePreprocessor]"):
        self.args = args
        self.log_prefix = log_prefix
        
        # 高级特征工程器（新增）
        self.advanced_fe = AdvancedFeatureEngineer(log_prefix)
    
    def extend_datetime_feature(self, df: pd.DataFrame, col_ts: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        扩展日期时间特征
        
        Args:
            df: 数据框
            col_ts: 时间戳列名
        
        Returns:
            (扩展后的数据框, 新增的特征列表)
        """
        df_copy = df.copy()
        datetime_features_list = []
        
        if col_ts in df_copy.columns:
            df_copy[col_ts] = pd.to_datetime(df_copy[col_ts])
            
            feature_mapping = {
                'minute': lambda x: x.minute,
                'hour': lambda x: x.hour,
                'day': lambda x: x.day,
                'weekday': lambda x: x.weekday(),
                'week': lambda x: x.isocalendar()[1],
                'day_of_week': lambda x: x.dayofweek,
                'week_of_year': lambda x: x.isocalendar()[1],
                'month': lambda x: x.month,
                'days_in_month': lambda x: x.days_in_month,
                'quarter': lambda x: x.quarter,
                'day_of_year': lambda x: x.dayofyear,
                'year': lambda x: x.year,
            }
            
            for feat in self.args.datetime_features:
                if feat in feature_mapping:
                    col_name = f"datetime_{feat}"
                    df_copy[col_name] = df_copy[col_ts].apply(feature_mapping[feat])
                    datetime_features_list.append(col_name)
        
        return df_copy, datetime_features_list
    
    def extend_lag_feature_univariate(self, df: pd.DataFrame, target: str, 
                                     lags: List[int]) -> Tuple[pd.DataFrame, List[str]]:
        """
        扩展单变量滞后特征
        
        Args:
            df: 数据框
            target: 目标变量名
            lags: 滞后期列表
        
        Returns:
            (扩展后的数据框, 新增的滞后特征列表)
        """
        df_copy = df.copy()
        lag_features = []
        
        for lag in lags:
            col_name = f"{target}_lag_{lag}"
            df_copy[col_name] = df_copy[target].shift(lag)
            lag_features.append(col_name)
        
        return df_copy, lag_features
    
    def extend_lag_feature_multivariate(self, df: pd.DataFrame, endogenous_cols: List[str],
                                       n_lags: int, horizon: int) -> Tuple[pd.DataFrame, List[str], List[str]]:
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
        lag_features = []
        shift_targets = []
        
        for col in endogenous_cols:
            if col not in df_copy.columns:
                continue
            
            # 创建滞后特征
            for lag in range(1, n_lags + 1):
                col_name = f"{col}_lag_{lag}"
                df_copy[col_name] = df_copy[col].shift(lag)
                lag_features.append(col_name)
            
            # 创建未来目标
            if horizon > 1:
                for h in range(1, horizon + 1):
                    col_name = f"{col}_shift_{h}"
                    df_copy[col_name] = df_copy[col].shift(-h)
                    shift_targets.append(col_name)
        
        return df_copy, lag_features, shift_targets
    
    def extend_direct_multi_step_targets(self, df: pd.DataFrame, target: str, 
                                        horizon: int) -> Tuple[pd.DataFrame, List[str]]:
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
        target_output_features = []
        
        for h in range(1, horizon + 1):
            col_name = f"{target}_shift_{h}"
            df_copy[col_name] = df_copy[target].shift(-h)
            target_output_features.append(col_name)
        
        return df_copy, target_output_features


# ==================== 主模型类 ====================

class Model:
    """
    基于机器学习的时间序列预测模型类（优化版）
    """
    
    def __init__(self, args: ModelConfig):
        """初始化模型"""
        self.args = args
        self.setting = f"{self.args.model_name}-{self.args.data}-{self.args.pred_method}"
        self.log_prefix = f"[{self.args.model_name}-{self.args.data}]"
        
        # 数据参数
        self.args.data_dir = Path(self.args.data_dir)
        self.n_per_day = int(24 * 60 / self.args.freq_minutes)
        
        # 时间配置
        start_time = self.args.now_time.replace(hour=0) - datetime.timedelta(days=self.args.history_days)
        now_time = self.args.now_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
        future_time = self.args.now_time + datetime.timedelta(days=self.args.predict_days)
        
        self.train_start_time = start_time
        self.train_end_time = now_time
        self.forecast_start_time = now_time
        self.forecast_end_time = future_time
        
        # 特征工程
        self.n_lags = len(self.args.lags)
        self.horizon = int(self.args.predict_days * self.n_per_day)
        
        # 模型参数（使用抽象层，不再硬编码）
        self.model_params = {
            'objective': self.args.objective if self.args.model_type == 'lightgbm' else 'reg:squarederror',
            'learning_rate': self.args.learning_rate,
            'n_estimators': 1000,
            'random_state': 42,
        }
        
        # 特征预处理器
        self.preprocessor = FeaturePreprocessor(self.args, self.log_prefix)
        
        # 统一的特征缩放器（新增）
        self.feature_scaler = None
        
        # 测试参数
        self.window_len = int(self.args.window_days * self.n_per_day)
        self.n_windows = int(self.args.history_days * self.n_per_day - self.window_len - self.horizon + 1) // self.horizon
        
        # 结果保存路径
        self.args.checkpoints_dir = Path(self.args.checkpoints_dir).joinpath(self.setting)
        self.args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.args.test_results_dir = Path(self.args.test_results_dir).joinpath(self.setting)
        self.args.test_results_dir.mkdir(parents=True, exist_ok=True)
        self.args.pred_results_dir = Path(self.args.pred_results_dir).joinpath(self.setting)
        self.args.pred_results_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"{self.log_prefix} {'='*80}")
        logger.info(f"{self.log_prefix} 参数准备完成")
        logger.info(f"{self.log_prefix} 历史数据范围: {self.train_start_time} ~ {self.train_end_time}")
        logger.info(f"{self.log_prefix} 预测数据范围: {self.forecast_start_time} ~ {self.forecast_end_time}")
        logger.info(f"{self.log_prefix} 模型类型: {self.args.model_type}")
        logger.info(f"{self.log_prefix} 高级特征: {'启用' if self.args.enable_advanced_features else '禁用'}")
        logger.info(f"{self.log_prefix} 模型融合: {'启用' if self.args.enable_ensemble else '禁用'}")
    
    def load_data(self) -> Dict:
        """
        加载所有必要的数据
        
        Returns:
            包含目标序列、日期类型、天气等数据的字典
        """
        logger.info(f"{self.log_prefix} 从 {self.args.data_dir} 加载数据")
        
        input_data = {}
        
        # 加载目标时间序列数据
        target_data_path = self.args.data_dir / self.args.data_path
        if target_data_path.exists():
            df_target = pd.read_csv(target_data_path)
            input_data["target_series"] = df_target
            logger.info(f"{self.log_prefix} 目标序列已加载: {df_target.shape}")
        else:
            logger.error(f"{self.log_prefix} 目标数据未找到: {target_data_path}")
            raise FileNotFoundError(f"目标数据未找到: {target_data_path}")
        
        # 加载其他数据（日期、天气等）的逻辑这里省略，与原脚本相同
        
        return input_data
    
    def create_features(self, df_series: pd.DataFrame, endogenous_features_with_target: List[str],
                       exogenous_features: List[str], target_feature: str,
                       categorical_features: List[str]) -> Tuple:
        """
        创建特征（优化版，集成高级特征工程）
        
        Args:
            df_series: 时间序列数据
            endogenous_features_with_target: 内生变量（含目标）
            exogenous_features: 外生变量
            target_feature: 目标变量
            categorical_features: 类别特征
        
        Returns:
            (特征化后的数据, 预测特征列表, 目标输出特征列表, 类别特征列表)
        """
        df_series_copy = df_series.copy()
        lag_features = []
        target_output_features = []
        
        # 根据预测方法创建特征
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            # USMDO: 只使用外生变量
            pass
        
        elif self.args.pred_method == "univariate-single-multistep-direct":
            # USMD: 目标变量滞后 + 外生变量
            if self.args.lags:
                df_series_copy, uni_lag_features = self.preprocessor.extend_lag_feature_univariate(
                    df_series_copy, target_feature, self.args.lags
                )
                lag_features.extend(uni_lag_features)
            
            df_series_copy, shift_targets = self.preprocessor.extend_direct_multi_step_targets(
                df_series_copy, target_feature, self.horizon
            )
            target_output_features.extend(shift_targets)
        
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            # USMR: 目标变量滞后 + 外生变量，预测下一步
            if self.args.lags:
                df_series_copy, uni_lag_features = self.preprocessor.extend_lag_feature_univariate(
                    df_series_copy, target_feature, self.args.lags
                )
                lag_features.extend(uni_lag_features)
            
            df_series_copy, shift_targets = self.preprocessor.extend_direct_multi_step_targets(
                df_series_copy, target_feature, 1
            )
            target_output_features.extend(shift_targets)
        
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            # MSMD: 所有内生变量滞后 + 外生变量
            if self.args.lags:
                df_series_copy, multi_lag_features, _ = self.preprocessor.extend_lag_feature_multivariate(
                    df_series_copy, endogenous_features_with_target, max(self.args.lags), 1
                )
                lag_features.extend(multi_lag_features)
            
            df_series_copy, shift_targets = self.preprocessor.extend_direct_multi_step_targets(
                df_series_copy, target_feature, self.horizon
            )
            target_output_features.extend(shift_targets)
        
        # 其他预测方法类似...
        
        # 添加高级特征（新增）
        if self.args.enable_advanced_features:
            logger.info(f"{self.log_prefix} 添加高级特征...")
            
            # 添加滞后统计特征
            if target_feature in df_series_copy.columns:
                df_series_copy = self.preprocessor.advanced_fe.add_lag_statistics(
                    df_series_copy,
                    columns=[target_feature],
                    windows=self.args.rolling_windows,
                    stats=self.args.rolling_stats
                )
            
            # 添加差分特征
            if target_feature in df_series_copy.columns:
                df_series_copy = self.preprocessor.advanced_fe.add_diff_features(
                    df_series_copy,
                    columns=[target_feature],
                    periods=self.args.diff_periods
                )
            
            # 将生成的特征添加到特征列表
            lag_features.extend(self.preprocessor.advanced_fe.get_generated_features())
        
        # 组合所有特征
        predictor_features = lag_features + exogenous_features
        predictor_features = [f for f in predictor_features if f in df_series_copy.columns]
        
        return df_series_copy, predictor_features, target_output_features, categorical_features
    
    def train(self, X_train, Y_train, categorical_features: List[str]):
        """
        模型训练（优化版，使用模型抽象层）
        
        Args:
            X_train: 训练特征
            Y_train: 训练目标
            categorical_features: 类别特征列表
        
        Returns:
            训练好的模型
        """
        logger.info(f"{self.log_prefix} 开始训练模型...")
        
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        
        # 确定实际存在的类别特征
        actual_categorical = [f for f in categorical_features if f in X_train_df.columns]
        
        # 特征缩放（使用统一的缩放器）
        if self.args.scale:
            self.feature_scaler = UnifiedFeatureScaler(
                scaler_type=self.args.scaler_type,
                encode_categorical=self.args.encode_categorical_features
            )
            X_train_df = self.feature_scaler.fit_transform(X_train_df, actual_categorical)
        
        # 创建模型（使用工厂模式）
        if self.args.enable_ensemble:
            # 模型融合
            logger.info(f"{self.log_prefix} 使用模型融合: {self.args.ensemble_models}")
            models = []
            for model_type in self.args.ensemble_models:
                base_model = ModelFactory.create_model(model_type, self.model_params)
                models.append(base_model)
            
            # 简化的融合：平均法
            for i, model in enumerate(models):
                logger.info(f"{self.log_prefix} 训练模型 {i+1}/{len(models)}: {self.args.ensemble_models[i]}")
                if self.args.pred_method in ["univariate-single-multistep-direct-output", "univariate-single-multistep-recursive"]:
                    model.fit(X_train_df, Y_train_df)
                else:
                    wrapped_model = MultiOutputRegressor(model.model)
                    wrapped_model.fit(X_train_df, Y_train_df)
                    model.model = wrapped_model
            
            # 返回模型列表（在预测时平均）
            return models
        else:
            # 单模型
            base_model = ModelFactory.create_model(self.args.model_type, self.model_params)
            
            if self.args.pred_method in ["univariate-single-multistep-direct-output", "univariate-single-multistep-recursive"]:
                # 单输出
                model = base_model
                model.fit(X_train_df, Y_train_df, categorical_features=actual_categorical)
            else:
                # 多输出
                model = MultiOutputRegressor(base_model.model)
                model.fit(X_train_df, Y_train_df)
            
            logger.info(f"{self.log_prefix} 模型训练完成")
            return model
    
    def predict_with_preprocessing(self, model, X_test, categorical_features):
        """
        带预处理的预测（统一接口）
        
        Args:
            model: 训练好的模型
            X_test: 测试特征
            categorical_features: 类别特征列表
        
        Returns:
            预测结果
        """
        X_test_processed = X_test.copy()
        
        # 特征缩放
        if self.feature_scaler is not None:
            actual_categorical = [f for f in categorical_features if f in X_test.columns]
            X_test_processed = self.feature_scaler.transform(X_test, actual_categorical)
        
        # 预测
        if isinstance(model, list):
            # 模型融合：平均预测
            predictions = []
            for m in model:
                if hasattr(m, 'predict'):
                    predictions.append(m.predict(X_test_processed))
                else:
                    predictions.append(m.model.predict(X_test_processed))
            return np.mean(predictions, axis=0)
        else:
            # 单模型
            if hasattr(model, 'predict'):
                return model.predict(X_test_processed)
            else:
                return model.model.predict(X_test_processed)
    
    def run(self):
        """运行主流程"""
        logger.info(f"{self.log_prefix} {'='*80}")
        logger.info(f"{self.log_prefix} 开始运行预测流程")
        logger.info(f"{self.log_prefix} {'='*80}")
        
        # 1. 加载数据
        input_data = self.load_data()
        
        # 2. 数据预处理（这里简化，实际需要完整实现）
        df_history = input_data["target_series"]
        
        # 3. 特征工程
        endogenous_features = [self.args.target] + self.args.target_series_numeric_features
        exogenous_features = self.args.datetime_features + self.args.weather_features
        categorical_features = self.args.datetime_categorical_features + self.args.datetype_categorical_features
        
        df_featured, predictor_features, target_output_features, categorical_features = self.create_features(
            df_series=df_history,
            endogenous_features_with_target=endogenous_features,
            exogenous_features=exogenous_features,
            target_feature=self.args.target,
            categorical_features=categorical_features
        )
        
        # 4. 准备训练数据
        df_train = df_featured.dropna()
        X_train = df_train[predictor_features]
        
        if target_output_features:
            Y_train = df_train[target_output_features]
        else:
            Y_train = df_train[self.args.target]
        
        # 5. 训练模型
        model = self.train(X_train, Y_train, categorical_features)
        
        # 6. 预测（这里简化）
        logger.info(f"{self.log_prefix} 模型训练完成，可进行预测")
        
        return model


# ==================== 主函数 ====================

def main():
    """主函数"""
    # 创建配置
    args = ModelConfig()
    
    # 可选：启用高级特征
    args.enable_advanced_features = True
    args.rolling_windows = [3, 7, 14]
    args.rolling_stats = ['mean', 'std']
    args.diff_periods = [1, 7]
    
    # 可选：启用模型融合
    args.enable_ensemble = False
    args.ensemble_models = ['lightgbm', 'xgboost']
    
    # 可选：切换模型类型
    args.model_type = "lightgbm"  # 'lightgbm', 'xgboost', 'catboost'
    
    # 可选：切换预测方法
    args.pred_method = "univariate-single-multistep-direct-output"
    
    # 创建模型
    model = Model(args)
    
    # 运行
    trained_model = model.run()
    
    logger.info("预测流程完成！")


if __name__ == "__main__":
    main()

