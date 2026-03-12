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
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.model_selection import (
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
)

from config.univariate_config import (
    ModelConfig_univariate, 
    ModelConfig_multivariate
)
from data_provider.data_loader import DataLoader
from features.FeatureEngineering import FeatureEngineer
from features.FeatureScalering import FeatureScaler
from models.ModelTesting import ModelTesting
from models.ModelFactory import ModelFactory
from models.ModelEnsemble import TimeSeriesEnsembleRegressor, EnsembleConfig
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
        self.setting = f"{self.args.model_type}-{self.args.data}-{self.args.pred_method}"
        self.log_prefix = f"[{self.args.model_type}-{self.args.data}]"
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
        self.model_params = copy.deepcopy(self.args.model_params)
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

    @staticmethod
    def _align_pred_length(y_pred: np.ndarray, target_len: int, fill_value: float = np.nan) -> np.ndarray:
        pred = np.asarray(y_pred).reshape(-1)
        if target_len <= 0:
            return np.asarray([])
        if len(pred) == target_len:
            return pred
        if len(pred) == 0:
            return np.full(shape=(target_len,), fill_value=fill_value)
        if len(pred) > target_len:
            return pred[:target_len]
        return np.pad(pred, pad_width=(0, target_len - len(pred)), mode="edge")
    # ##############################
    # Model Testing
    # ##############################
    def test(self, 
             df_history, 
             X_train_history, 
             Y_train_history, 
             endogenous_features_with_target, 
             target_feature, 
             predictor_features,
             target_output_features, 
             categorical_features):
        """
        模型滑窗测试
        """
        model_testing = ModelTesting(args=self.args, log_prefix=self.log_prefix, horizon=self.horizon, window_len=self.window_len)
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
            (X_train, Y_train, X_test, Y_test, df_history_train, df_history_test) = model_testing._evaluate_split(
                X_train_history, Y_train_history, df_history, window
            )
            if X_train is None:
                continue
            # 目标特征处理(Ensure Y_train is DataFrame for MultiOutputRegressor)
            Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
            Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
            # 窗口训练
            scaler_testing = FeatureScaler(self.args, self.args.scaler_type, log_prefix=self.log_prefix)
            model = self.train(X_train, Y_train, scaler_testing, categorical_features)
            # 窗口预测
            predictor = PredictionHelper(
                args=self.args,
                model=model,
                horizon=len(X_test),
                df_history=df_history_train,
                df_future=X_test.copy() if self.args.pred_method == "univariate-single-multistep-direct-output" else df_history_test,
                df_date_future=None,
                df_weather_future=None,
                endogenous_features=endogenous_features_with_target,
                target_feature=target_feature,
                target_output_features=target_output_features,
                categorical_features=categorical_features,
                feature_scaler=scaler_testing,
                log_prefix=self.log_prefix,
            )
            Y_pred = predictor._predict_by_method()
            if len(Y_pred) == 0: # If _window_test returned empty predictions
                logger.warning(f"{self.log_prefix} Skipping evaluation for window {window} due to empty predictions.")
                continue
            # Process Y_test and Y_pred for evaluation. We always evaluate the primary target's first step prediction.
            # Y_test for evaluation should always be the actuals for target_t+1.
            # Assuming the primary target (y) shifted by 1 is always the first column of Y_test
            Y_test_for_eval = Y_test.iloc[:, 0].values
            # Ensure Y_pred matches length of Y_test_for_eval
            if len(Y_pred) != len(Y_test_for_eval):
                logger.warning(
                    f"Length mismatch: Y_pred ({len(Y_pred)}) vs Y_test_for_eval ({len(Y_test_for_eval)}) "
                    f"in window {window}. Truncating both to the minimum length."
                )
                min_len = min(len(Y_pred), len(Y_test_for_eval))
                Y_pred = np.asarray(Y_pred)[:min_len]
                Y_test_for_eval = np.asarray(Y_test_for_eval)[:min_len]
            # 测试集评价指标
            eval_scores_window = model_testing._evaluate_score(Y_test_for_eval, Y_pred, window, df_history_test)
            test_scores_df = pd.concat([test_scores_df, eval_scores_window], axis=0)
            # 测试集预测数据
            cv_plot_df_window = model_testing._evaluate_result(Y_test_for_eval, Y_pred, window, cv_timestamp_full_df)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis=0)
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
        # 测试结果保存
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Testing result save...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        model_testing.test_results_save(test_scores_df, cv_plot_df)
        logger.info(f"{self.log_prefix} Model Testing result saved in: {self.args.test_results_dir}")
        
        return test_scores_df, cv_plot_df
    # ##############################
    # Model Hyperparameters tuning and Model training
    # ##############################
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
        lgbm_base = self.model_factory.create_model(
            model_type=self.args.model_type,
            model_params=self.model_params
        )

        # Wrap in MultiOutputRegressor if the method is multi-output
        if Y_train.shape[1] == 1:
            model_for_tuning = lgbm_base.model
            tuned_param_grid = {k.replace("estimator__", ""): v for k, v in param_grid.items()}
        else:
            model_for_tuning = MultiOutputRegressor(lgbm_base.model)
            tuned_param_grid = param_grid

        # TimeSeriesSplit for cross-validation
        # n_splits determines how many train-test splits to generate.
        # The test set size will be at least self.horizon.
        tscv = TimeSeriesSplit(n_splits=self.args.tuning_n_splits)

        # Use GridSearchCV for exhaustive search or RandomizedSearchCV for faster search
        # RandomizedSearchCV is generally preferred for larger search spaces
        search = RandomizedSearchCV(
            estimator=model_for_tuning,
            param_distributions=tuned_param_grid,
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
     
    def model_save(self, model):
        """
        模型保存
        """
        # model_deploy = ModelDeployPkl(save_file_path=self.args.checkpoints.joinpath("model.pkl"))
        # model_deploy.save_model(model)
        # logger.info(f"{self.log_prefix} Model saved to {model_dir.joinpath('model.pkl')}")
        pass
    
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
        # Hyperparameter tuning (if enabled)
        # ------------------------------
        if self.args.perform_tuning:
            best_model = self._hyperparameters_tuning(X_train_df_processed, Y_train_df)
            return best_model
        # ------------------------------
        # 模型训练
        # ------------------------------
        if self.args.enable_ensemble:
            logger.info(f"{self.log_prefix} 使用模型融合: {self.args.ensemble_models}, 方法: {self.args.ensemble_method}")
            base_models = []
            for model_type in self.args.ensemble_models:
                model_wrapper = self.model_factory.create_model(model_type=model_type, model_params=self.model_params)
                estimator = model_wrapper.model
                if Y_train_df.shape[1] > 1:
                    estimator = MultiOutputRegressor(estimator)
                base_models.append((model_type, estimator))
            ensemble = TimeSeriesEnsembleRegressor(
                base_models=base_models,
                config=EnsembleConfig(
                    method=getattr(self.args, "ensemble_method", "averaging"),
                    val_ratio=float(getattr(self.args, "ensemble_val_ratio", 0.2)),
                    random_state=42,
                ),
            )
            y_train_input = np.ravel(Y_train_df.values) if Y_train_df.shape[1] == 1 else Y_train_df.values
            ensemble.fit(X_train_df_processed, y_train_input)
            logger.info(f"{self.log_prefix} Ensemble training completed!")
            return ensemble
        else:
            # 单模型
            lgbm_estimator = self.model_factory.create_model(
                model_type=getattr(self.args, "model_type", "lightgbm"), 
                model_params=self.model_params
            )
            if Y_train_df.shape[1] == 1:
                # 单输出
                model = lgbm_estimator
                model.fit(X_train_df_processed, np.ravel(Y_train_df.values))
                logger.info(f"{self.log_prefix} Training single output LGBMRegressor")
            elif Y_train_df.shape[1] > 1:
                # 多输出
                model = MultiOutputRegressor(estimator=lgbm_estimator.model)
                model.fit(X_train_df_processed, Y_train_df)
                logger.info(f"{self.log_prefix} Training MultiOutputRegressor with {Y_train.shape[1]} outputs")
            logger.info(f"{self.log_prefix} Model training completed!")

            return model
    # ##############################
    # Model Forecast(Model Inference)
    # ##############################
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
    
    def forecast(self, 
                 df_history, 
                 X_train_history, 
                 Y_train_history, 
                 df_future, 
                 df_date_future,
                 df_weather_future,
                 endogenous_features, 
                 target_feature, 
                 predictor_features,
                 target_output_features, 
                 categorical_features):
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
        self.scaler_forecasting = FeatureScaler(self.args, scaler_type=self.args.scaler_type, log_prefix=self.log_prefix)
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
            horizon = self.horizon,
            df_history = df_history, 
            df_future = df_future, 
            df_date_future=df_date_future,
            df_weather_future=df_weather_future,
            endogenous_features = endogenous_features, 
            target_feature = target_feature, 
            target_output_features = target_output_features, 
            categorical_features = categorical_features,
            feature_scaler = self.scaler_forecasting,
            log_prefix = self.log_prefix,
        )
        # Initialize Y_pred
        Y_pred = np.array([])
        # Use a copy of raw future data
        df_future_for_prediction = df_future.copy()

        Y_pred = predictor._predict_by_method()
        
        Y_pred = self._align_pred_length(Y_pred, len(df_future_for_prediction), fill_value=np.nan)
        # 预测结果收集
        df_future_for_prediction["predict_value"] = Y_pred
        df_future_for_prediction = df_future_for_prediction[["time", "predict_value"]]
        logger.info(f"{self.log_prefix} after forecast df_future: \n{df_future_for_prediction.head()}")
        logger.info(f"{self.log_prefix} after forecast df_future.shape: {df_future_for_prediction.shape}")
        # 模型预测结果保存
        logger.info(f"{self.log_prefix} {40*'-'}")
        logger.info(f"{self.log_prefix} Model Forecasting result save...")
        logger.info(f"{self.log_prefix} {40*'-'}")
        self.forecast_results_save(df_history, df_future_for_prediction)
        logger.info(f"{self.log_prefix} Model Forecasting result saved in: {self.args.pred_results_dir}")
        
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
        dataloader = DataLoader(
            args=self.args, 
            train_start_time=self.train_start_time,
            train_end_time=self.train_end_time,
            forecast_start_time=self.forecast_start_time,
            forecast_end_time=self.forecast_end_time,
            log_prefix=self.log_prefix,
        )
        input_data = dataloader.load_data()
        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history data preprocessing...")
        logger.info(f"{self.log_prefix} {80*'='}")
        (df_history, 
         df_date_history, 
         df_weather_history, 
         endogenous_features_with_target, 
         target_feature) = dataloader.process_history_data(input_data = input_data)
        if target_feature is None:
            raise ValueError(f"{self.log_prefix} Target feature '{self.args.target}' not found in dataset.")
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{self.log_prefix} {80*'='}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {80*'='}")
        # 特征预处理器
        feature_engineer_history = FeatureEngineer(self.args, self.log_prefix)
        (df_history_featured, 
         predictor_features, 
         target_output_features, 
         categorical_features) = feature_engineer_history.create_features(
            df_series = df_history,
            df_date_history=df_date_history,
            df_date_future=None,
            df_weather_history=df_weather_history,
            df_weather_future=None,
            endogenous_features_with_target = endogenous_features_with_target,
            target_feature = target_feature,
            horizon = self.horizon,
        )
        # 删除在构建滞后特征时产生的缺失值
        df_history_featured = df_history_featured.dropna()
        logger.info(f"{self.log_prefix} after dropna df_history_featured: \n{df_history_featured.head()}")
        
        # 预测特征、目标特征分离
        X_train_history, Y_train_history = feature_engineer_history.predictor_target_split(
            df_series_featured = df_history_featured, 
            predictor_features = predictor_features, 
            target_output_features = target_output_features,
        )
        if X_train_history.empty or Y_train_history.empty:
            raise ValueError(f"{self.log_prefix} Empty training matrix after feature engineering.")
        logger.info(f"{self.log_prefix} after predictor_target_split X_train_history: \n{X_train_history.head()}")
        logger.info(f"{self.log_prefix} after predictor_target_split Y_train_history: \n{Y_train_history.head()}")
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
            logger.info(f"{self.log_prefix} {40*'-'}")
            # 模型滑窗测试
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Testing start...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            test_scores_df, cv_plot_df = self.test(
                df_history = df_history,
                X_train_history = X_train_history,
                Y_train_history = Y_train_history,
                endogenous_features_with_target = endogenous_features_with_target,
                target_feature = target_feature,
                predictor_features = predictor_features,
                target_output_features = target_output_features,
                categorical_features = categorical_features, 
            )
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
            (df_future, df_date_future, df_weather_future) = dataloader.process_future_data(input_data = input_data)
            # 模型预测
            logger.info(f"{self.log_prefix} {40*'-'}")
            logger.info(f"{self.log_prefix} Model Forecasting start...")
            logger.info(f"{self.log_prefix} {40*'-'}")
            df_future_predicted = self.forecast(
                df_history = df_history,
                X_train_history = X_train_history,
                Y_train_history = Y_train_history,
                df_future = df_future,
                df_date_future = df_date_future,
                df_weather_future = df_weather_future,
                endogenous_features = endogenous_features_with_target,
                target_feature = target_feature,
                predictor_features = predictor_features,
                target_output_features = target_output_features,
                categorical_features = categorical_features, 
            )




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
