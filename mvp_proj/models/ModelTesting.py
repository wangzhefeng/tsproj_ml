# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelTesting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-25
# * Version     : 1.0.022509
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# model evaluation
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)

from features.FeatureScalering import FeatureScaler
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelTesting:
    
    def __init__(self, args, log_prefix: str, horizon: int, window_len: int):
        self.args = args
        self.log_prefix = log_prefix
        self.horizon = horizon
        self.window_len = window_len

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
        if getattr(self.args, "disable_plotting", False):
            logger.info(f"{self.log_prefix} Skip plotting because disable_plotting=True.")
            return
        # 测试结果数据可视化
        required_cols = {"Y_preds", "Y_trues"}
        if cv_plot_df.empty or not required_cols.issubset(set(cv_plot_df.columns)):
            logger.warning(f"{self.log_prefix} No valid prediction columns found for visualization.")
            return
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



# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
