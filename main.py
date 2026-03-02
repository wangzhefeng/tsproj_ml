# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
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
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from data_provider.data_loader import DataLoader
from features.FeatureScalering import FeatureScaler
from features.FeatureEngineering import FeatureEngineer
from models.ModelTraining import Trainer
from models.ModelTesting import Tester
from models.ModelForecasting import Forecaster

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
        # 模型测试、预测
        # ------------------------------ 
        # 预测未来 1 天(24小时)的数据/数据划分长度/预测数据长度
        self.horizon = int(self.args.predict_days * self.n_per_day)
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
        logger.info(f"{self.log_prefix} {'#' * 85}")
        logger.info(f"{self.log_prefix} Prepare params...")
        logger.info(f"{self.log_prefix} {'#' * 85}")
        logger.info(f"{self.log_prefix} history data range: {self.train_start_time}~{self.train_end_time}")
        logger.info(f"{self.log_prefix} predict data range: {self.forecast_start_time}~{self.forecast_end_time}")
        logger.info(f"{self.log_prefix} 模型类型: {self.args.model_type}")
        logger.info(f"{self.log_prefix} 预测方法: {self.args.pred_method}")
        logger.info(f"{self.log_prefix} 事件(date type features)特征: {'启用' if self.args.enable_date_features else '禁用'}")
        logger.info(f"{self.log_prefix} 气象(weather   features)特征: {'启用' if self.args.enable_weather_features else '禁用'}")
        logger.info(f"{self.log_prefix} 时间(date time features)特征: {'启用' if self.args.enable_datetime_features else '禁用'}")
        logger.info(f"{self.log_prefix} 滞后(lags      features)特征: {'启用' if self.args.enable_lags_features else '禁用'}")
        logger.info(f"{self.log_prefix} 高级(advanced  features)特征: {'启用' if self.args.enable_advanced_features else '禁用'}")
        logger.info(f"{self.log_prefix} 特征变换: {'启用' if self.args.scale else '禁用'}")
        logger.info(f"{self.log_prefix} 类别特征: {'启用' if self.args.encode_categorical_features else '禁用'}")
        logger.info(f"{self.log_prefix} 模型融合: {'启用' if self.args.enable_ensemble else '禁用'}")
        logger.info(f"{self.log_prefix} 模型测试: {'启用' if self.args.is_testing else '禁用'}")
        logger.info(f"{self.log_prefix} 模型预测: {'启用' if self.args.is_forecasting else '禁用'}")

    def train(self, X_train, Y_train, categorical_features, mode="forecast", verbose=False):
        """
        模型训练
        """ 
        # 创建特征预处理器
        scaler = FeatureScaler(self.args, scaler_type=self.args.scaler_type, log_prefix=self.log_prefix, verbose=verbose)
        # 模型训练类
        model_trainer = Trainer(args=self.args, log_prefix=self.log_prefix)
        # 模型训练
        model, scaler = model_trainer.train(
            X_train = X_train, 
            Y_train = Y_train, 
            feature_scaler = scaler, 
            categorical_features = categorical_features,
        )
        # 模型保存
        if mode == "forecast":
            model_trainer.model_save(model)

        return model, scaler

    def test(self, 
             df_history, 
             X_train_history, 
             Y_train_history, 
             df_date_history,
             df_weather_history,
             endogenous_features_with_target, 
             target_feature, 
             predictor_features,
             target_output_features, 
             categorical_features):
        """
        模型滑窗测试
        """
        # ------------------------------
        # 判断是否有足够的历史数据保证至少一个完整的测试窗口
        # ------------------------------
        if self.n_windows <= 0:
            logger.warning(f"{self.log_prefix} Not enough data for testing with current window configuration (Total X points: {len(X_train_history)}")
            logger.warning(f"{self.log_prefix} Window length: {self.window_len}, Horizon: {self.horizon}). No tests will be performed.")
            return test_scores_df, cv_plot_df
        # ------------------------------
        # 模型滑窗测试结果收集
        # ------------------------------
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        # 训练数据集的完整时间戳
        cv_timestamp_full_df = pd.DataFrame({"time": pd.date_range(self.train_start_time, self.train_end_time, freq=self.args.freq, inclusive="left")})
        # ------------------------------
        # 模型滑窗测试过程
        # ------------------------------
        for window in range(1, int(self.n_windows + 1)):
            logger.info(f"{self.log_prefix} {'=' * 48}")
            logger.info(f"{self.log_prefix} Model Testing window: {window}...")
            logger.info(f"{self.log_prefix} {'=' * 48}")
            # 模型测试类
            model_tester = Tester(args=self.args, log_prefix=self.log_prefix, horizon=self.horizon, window_len=self.window_len)
            # ------------------------------
            # 数据分割: 训练集、测试集
            # ------------------------------
            logger.info(f"{self.log_prefix} Model Testing sliding window data split...")
            logger.info(f"{self.log_prefix} {'=' * 48}")
            (X_train, Y_train, 
             X_test, Y_test, 
             df_history_train, df_history_test) = model_tester._evaluate_split(
                 X_train_history, Y_train_history, df_history, window
            )
            if X_train is None:
                continue
            # 窗口目标特征处理
            Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
            Y_test = Y_test.to_frame() if isinstance(Y_test, pd.Series) else Y_test
            # ------------------------------
            # 窗口训练
            # ------------------------------
            logger.info(f"{self.log_prefix} {'=' * 48}")
            logger.info(f"{self.log_prefix} Model Testing sliding window training...")
            logger.info(f"{self.log_prefix} {'=' * 48}")
            model, scaler_testing = self.train(
                X_train = X_train_history, 
                Y_train = Y_train_history, 
                categorical_features = categorical_features,
                mode = "test",
                verbose = False,
            )
            # ------------------------------
            # 窗口预测
            # ------------------------------
            logger.info(f"{self.log_prefix} {'=' * 48}")
            logger.info(f"{self.log_prefix} Model Testing sliding window forecasting...")
            logger.info(f"{self.log_prefix} {'=' * 48}")
            # 未来数据
            if self.args.pred_method == "univariate-single-multistep-direct-output":
                df_future_prediction = X_test.copy()
            else:
                df_future_prediction = df_history_test
            # 模型预测
            predictor = Forecaster(
                args = self.args,
                horizon = min(self.horizon, len(X_test)),
                model = model,
                feature_scaler = scaler_testing,
                df_history = df_history_train,
                df_future = df_future_prediction,
                df_date_future = df_date_history,
                df_weather_future = df_weather_history,
                endogenous_features = endogenous_features_with_target,
                target_feature = target_feature,
                target_output_features = target_output_features,
                categorical_features = categorical_features,
                log_prefix = self.log_prefix,
            )
            Y_pred = predictor._predict_by_method()
            # ------------------------------
            # 模型滑窗预测结果收集
            # ------------------------------
            logger.info(f"{self.log_prefix} {'=' * 48}")
            logger.info(f"{self.log_prefix} Model Testing sliding window forecasting results collecting...")
            logger.info(f"{self.log_prefix} {'=' * 48}")
            # If window test returned empty predictions
            if len(Y_pred) == 0:
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
            eval_scores_window = model_tester._evaluate_score(Y_test_for_eval, Y_pred, window, df_history_test)
            test_scores_df = pd.concat([test_scores_df, eval_scores_window], axis=0)
            # 测试集预测数据
            cv_plot_df_window = model_tester._evaluate_result(Y_test_for_eval, Y_pred, window, cv_timestamp_full_df)
            cv_plot_df = pd.concat([cv_plot_df, cv_plot_df_window], axis=0)
            # ------------------------------
            # localtest
            # ------------------------------
            # break
        # ------------------------------
        # 模型测试结果保存
        # ------------------------------
        logger.info(f"{self.log_prefix} {'=' * 48}")
        logger.info(f"{self.log_prefix} Model Testing result saving...")
        logger.info(f"{self.log_prefix} {'=' * 48}")
        # 模型测试评价指标数据处理
        if not test_scores_df.empty:
            test_scores_df_mean = test_scores_df.drop(columns=["time_range"]).mean()
            test_scores_df_mean = test_scores_df_mean.to_frame().T.reset_index(drop=True, inplace=False)
            test_scores_df_mean["time_range"] = "均值"
            test_scores_df = pd.concat([test_scores_df, test_scores_df_mean], axis=0)
        logger.info(f"{self.log_prefix} Model Testing test_scores_df: \n{test_scores_df}")
        logger.info(f"{self.log_prefix} Model Testing cv_plot_df: \n{cv_plot_df.head()}")
        logger.info(f"{self.log_prefix} Model Testing cv_plot_df shape: {cv_plot_df.shape}")
        # 模型测试结果保存
        model_tester.test_results_save(test_scores_df, cv_plot_df)
        logger.info(f"{self.log_prefix} Model Testing result saved in: {self.args.test_results_dir}")
        
        return test_scores_df, cv_plot_df 

    def forecast(self, 
                 model, 
                 scaler_forecasting,
                 df_history, 
                 df_future, 
                 df_date_future,
                 df_weather_future,
                 endogenous_features_with_target, 
                 target_feature, 
                 target_output_features, 
                 categorical_features):
        """
        模型预测
        """
        # 未来数据复制
        df_future_prediction = df_future.copy()
        # 模型预测
        predictor = Forecaster(
            args = self.args,
            horizon = self.horizon,
            model = model, 
            feature_scaler = scaler_forecasting,
            df_history = df_history, 
            df_future = df_future_prediction, 
            df_date_future = df_date_future,
            df_weather_future = df_weather_future,
            endogenous_features = endogenous_features_with_target, 
            target_feature = target_feature, 
            target_output_features = target_output_features, 
            categorical_features = categorical_features,
            log_prefix = self.log_prefix,
        )
        Y_pred = predictor._predict_by_method()
        # ------------------------------
        # 模型预测结果收集和保存
        # ------------------------------
        logger.info(f"{self.log_prefix} {'=' * 87}")
        logger.info(f"{self.log_prefix} Model Forecasting result save...")
        logger.info(f"{self.log_prefix} {'=' * 87}")
        # 模型预测结果收集
        df_future_prediction["predict_value"] = Y_pred
        df_future_prediction = df_future_prediction[["time", "predict_value"]]
        logger.info(f"{self.log_prefix} after forecast df_future_prediction: \n{df_future_prediction.head()}")
        logger.info(f"{self.log_prefix} after forecast df_future_prediction.shape: {df_future_prediction.shape}")
        # 模型预测结果保存
        predictor.forecast_results_save(df_history, df_future_prediction, self.n_per_day)
        logger.info(f"{self.log_prefix} Model Forecasting result saved in: {self.args.pred_results_dir}")
        
        return df_future_prediction

    def run(self):
        # ------------------------------
        # 数据加载和处理
        # ------------------------------
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} Model history and future data loading...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
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
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} Model history data preprocessing...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
        (df_history, 
         df_date_history, 
         df_weather_history, 
         endogenous_features_with_target, 
         target_feature) = dataloader.process_history_data(input_data = input_data)
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} {'=' * 87}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {'=' * 87}")
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
        logger.info(f"{self.log_prefix} after dropna df_history_featured.shape: {df_history_featured.shape}")
        
        # 历史数据预测特征、目标特征分离
        logger.info(f"{self.log_prefix} {'=' * 87}")
        logger.info(f"{self.log_prefix} Model history data feature split...")
        logger.info(f"{self.log_prefix} {'=' * 87}")
        X_train_history, Y_train_history = feature_engineer_history.predictor_target_split(
            df_series_featured = df_history_featured, 
            predictor_features = predictor_features, 
            target_output_features = target_output_features, 
        )
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_testing:
            logger.info(f"{self.log_prefix} {'#' * 90}")
            logger.info(f"{self.log_prefix} Model Testing...")
            logger.info(f"{self.log_prefix} {'#' * 90}")
            test_scores_df, cv_plot_df = self.test(
                df_history = df_history,
                X_train_history = X_train_history,
                Y_train_history = Y_train_history,
                df_date_history = df_date_history,
                df_weather_history = df_weather_history,
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
            logger.info(f"{self.log_prefix} {'#' * 90}")
            logger.info(f"{self.log_prefix} Model Forecasting...")
            logger.info(f"{self.log_prefix} {'#' * 90}")
            # 未来数据处理(用来推理)
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Forecasting future data preprocessing...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            (df_future, \
             df_date_future, 
             df_weather_future) = dataloader.process_future_data(input_data = input_data)
            
            # 模型训练
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Training start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            model, scaler_forecasting = self.train(
                X_train = X_train_history, 
                Y_train = Y_train_history, 
                categorical_features = categorical_features,
                mode = "forecast",
                verbose = True
            )
            
            # 模型预测
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Forecasting start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            df_future_predicted = self.forecast(
                model = model,
                scaler_forecasting = scaler_forecasting,
                df_history = df_history,
                df_future = df_future,
                df_date_future = df_date_future,
                df_weather_future = df_weather_future,
                endogenous_features_with_target = endogenous_features_with_target,
                target_feature = target_feature,
                target_output_features = target_output_features,
                categorical_features = categorical_features, 
            )




# 测试代码 main 函数
def main():
    """
    主函数入口
    """
    from config.model_config import (
        ModelConfig_univariate, 
        ModelConfig_multivariate
    )
    # 模型配置
    args = ModelConfig_univariate()
    # args = ModelConfig_multivariate()
    # 创建模型实例
    model = Model(args)
    # 运行模型
    model.run()
    logger.info(f"{model.log_prefix} {'#' * 85}")
    logger.info(f"{model.log_prefix} 模型预测流程完成！")
    logger.info(f"{model.log_prefix} {'#' * 85}")

if __name__ == "__main__":
    main()
