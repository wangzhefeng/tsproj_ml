# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelTesting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-29
# * Version     : 1.0.032909
# * Description : 生产环境滑窗测试模块
# * Link        : link
# * Requirement : pandas, numpy, scikit-learn
# ***************************************************

# python libraries
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
# model evaluation
from sklearn.metrics import (
    r2_score,                        # R2
    mean_squared_error,              # MSE
    root_mean_squared_error,         # RMSE
    mean_absolute_error,             # MAE
    mean_absolute_percentage_error,  # MAPE
)

from features.FeatureEngineering import FeatureEngineer
from features.FeatureScalering import (
    FeatureScaler,
    TargetScaler,
    resolve_feature_scaler_type,
    resolve_target_scaler_type,
)
from models.ModelTraining import Trainer
from models.ModelForecasting import Forecaster
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Tester:
    
    def __init__(self, args, log_prefix: str, horizon: int, window_len: int):
        self.args = args
        self.log_prefix = log_prefix
        self.horizon = horizon
        self.window_len = window_len

    @staticmethod
    def _window_test(payload):
        """
        单个滑动窗口测试任务
        """
        args = payload["args"]
        log_prefix = payload["log_prefix"]
        horizon = payload["horizon"]
        window_len = payload["window_len"]
        window = payload["window"]

        # 滑窗数据分割：先切原始历史，再在窗口内构造训练标签，避免 Direct 标签跨入测试期
        split_result = Tester._evaluate_split(
            payload["df_history"],
            window,
            horizon=horizon,
            window_len=window_len,
            log_prefix=log_prefix,
        )
        if split_result is None:
            return {"window": window, "test_scores_df": None, "cv_plot_df": None}
        df_history_train, df_history_test = split_result
        build_result = Tester._build_window_train_xy(
            args=args,
            log_prefix=log_prefix,
            df_history_train=df_history_train,
            df_date_history=payload["df_date_history"],
            df_weather_history=payload["df_weather_history"],
            endogenous_features_with_target=payload["endogenous_features_with_target"],
            target_feature=payload["target_feature"],
            horizon=horizon,
        )
        if build_result is None:
            return {"window": window, "test_scores_df": None, "cv_plot_df": None}
        X_train, Y_train, target_output_features, categorical_features = build_result
        # 窗口目标特征处理
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        y_test_raw = df_history_test[payload["target_feature"]].to_numpy()
        # ------------------------------
        # 窗口训练
        # ------------------------------
        scaler = FeatureScaler(
            args,
            scaler_type=resolve_feature_scaler_type(args),
            log_prefix=log_prefix,
            verbose=False,
        )
        target_scaler = TargetScaler(
            args,
            scaler_type=resolve_target_scaler_type(args),
            log_prefix=log_prefix,
            verbose=False,
        )
        model_trainer = Trainer(args=args, log_prefix=log_prefix)
        model, scaler_testing, target_scaler_testing, selected_features = model_trainer.train(
            X_train=X_train,
            Y_train=Y_train,
            feature_scaler=scaler,
            target_scaler=target_scaler,
            categorical_features=categorical_features,
        )
        # ------------------------------
        # 窗口预测
        # ------------------------------
        df_future_for_test = Tester._build_test_future_frame(df_history_test)
        predictor = Forecaster(
            args=args,
            horizon=min(horizon, len(df_future_for_test)),
            model=model,
            feature_scaler=scaler_testing,
            target_scaler=target_scaler_testing,
            df_history=df_history_train,
            df_future=df_future_for_test,
            df_date_future=payload["df_date_history"],
            df_weather_future=payload["df_weather_history"],
            endogenous_features=payload["endogenous_features_with_target"],
            target_feature=payload["target_feature"],
            target_output_features=target_output_features,
            categorical_features=categorical_features,
            selected_features=selected_features,
            log_prefix=log_prefix,
        )
        y_pred = predictor._predict_by_method()
        # ------------------------------
        # 模型滑窗预测结果收集
        # ------------------------------
        if len(y_pred) == 0:
            return {"window": window, "test_scores_df": None, "cv_plot_df": None}
        # 预测结果恢复到目标空间，用于评估
        if target_scaler_testing is not None:
            pred_target_columns = target_scaler_testing.get_prediction_target_columns(
                args.pred_method,
                target_output_features,
            )
            y_pred = target_scaler_testing.restore_predictions(
                y_pred,
                pred_target_columns,
            )
            # 始终评估主目标的一步预测
            y_test_for_eval = target_scaler_testing.prepare_eval_target(
                y_test_raw,
                [target_output_features[0]],
            )
        else:
            y_test_for_eval = np.asarray(y_test_raw).reshape(-1)
        # 对齐预测结果与评估标签长度
        if len(y_pred) != len(y_test_for_eval):
            min_len = min(len(y_pred), len(y_test_for_eval))
            y_pred = np.asarray(y_pred)[:min_len]
            y_test_for_eval = np.asarray(y_test_for_eval)[:min_len]
        # 测试集评价指标
        eval_scores_window = Tester._evaluate_score(y_test_for_eval, y_pred, window, df_history_test, log_prefix=log_prefix)
        # 测试集预测数据
        cv_plot_df_window = Tester._evaluate_result(
            y_test_for_eval,
            y_pred,
            df_history_test,
            log_prefix=log_prefix,
        )

        return {"window": window, "test_scores_df": eval_scores_window, "cv_plot_df": cv_plot_df_window}

    # ------------------------------
    # Model sliding window testing
    # ------------------------------
    @staticmethod
    def _evaluate_split_index(window: int, total_data_points: int, horizon: int, window_len: int):
        """
        数据分割索引构建
        """
        # Calculate test start/end index
        test_end = total_data_points - 1 - (horizon * (window - 1))
        test_start = test_end - horizon + 1
        # Calculate train start/end index
        train_end = test_start
        train_start = train_end - (window_len - horizon)
        train_start = max(0, train_start)

        return train_start, train_end, test_start, test_end

    @staticmethod
    def _evaluate_split(
        df_history: pd.DataFrame,
        window: int,
        horizon: int,
        window_len: int,
        log_prefix: str,
    ):
        """
        训练、测试数据集分割
        """
        # 滑窗数据分割索引
        total_data_points = len(df_history)
        train_start, train_end, test_start, test_end = Tester._evaluate_split_index(
            window, total_data_points, horizon, window_len
        )
        logger.info(f"{log_prefix} split indexes:: [train_start:train_end]: [{train_start}:{train_end}]")
        logger.info(f"{log_prefix} split indexes:: [test_start:test_end]: [{test_start}:{test_end+1}]")
        if train_start >= train_end or test_start >= test_end + 1 or train_start < 0 or test_end >= total_data_points:
            logger.warning(
                f"{log_prefix} Insufficient data for window {window} "
                f"(train_start={train_start}, train_end={train_end}, "
                f"test_start={test_start}, test_end={test_end}). "
                f"Skipping this window."
            )
            return None

        # 滑窗数据分割
        df_history_train = df_history.iloc[train_start:train_end]
        df_history_test = df_history.iloc[test_start:test_end+1]
        logger.info(f"{log_prefix} df_history_train.shape: {df_history_train.shape}, df_history_test.shape: {df_history_test.shape}")

        if df_history_train.empty or df_history_test.empty:
            logger.warning(f"{log_prefix} Empty dataframe in window {window} split. Skipping.")
            return None
        
        return df_history_train, df_history_test

    @staticmethod
    def _build_window_train_xy(
        args,
        log_prefix: str,
        df_history_train: pd.DataFrame,
        df_date_history: pd.DataFrame,
        df_weather_history: pd.DataFrame,
        endogenous_features_with_target: List[str],
        target_feature: str,
        horizon: int,
    ):
        """
        在单个训练窗口内部构造特征和多步标签，避免标签跨入测试窗口。
        """
        feature_engineer = FeatureEngineer(args, log_prefix, verbose=False)
        (
            df_history_featured,
            predictor_features,
            target_output_features,
            categorical_features,
        ) = feature_engineer.create_features(
            df_series=df_history_train,
            df_date_history=df_date_history,
            df_date_future=None,
            df_weather_history=df_weather_history,
            df_weather_future=None,
            endogenous_features_with_target=endogenous_features_with_target,
            target_feature=target_feature,
            horizon=horizon,
        )
        df_history_featured = df_history_featured.dropna(subset=target_output_features)
        if df_history_featured.empty:
            logger.warning(f"{log_prefix} Empty featured training dataframe after target dropna. Skipping.")
            return None

        X_train, Y_train = feature_engineer.predictor_target_split(
            df_series_featured=df_history_featured,
            predictor_features=predictor_features,
            target_output_features=target_output_features,
        )
        if X_train.empty or Y_train.empty:
            logger.warning(f"{log_prefix} Empty X/Y after window feature split. Skipping.")
            return None

        return X_train, Y_train, target_output_features, categorical_features

    @staticmethod
    def _build_test_future_frame(df_history_test: pd.DataFrame):
        """
        测试预测阶段只能看到未来时间模板，不透传测试期真实 y。
        """
        return df_history_test[["time"]].copy()

    @staticmethod
    def _evaluate_score(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        window: int,
        df_history_test: pd.DataFrame,
        log_prefix: str,
    ):
        """
        模型评估
        计算模型的性能指标
        """
        # Ensure y_test and y_pred are 1D arrays for metrics
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        # Handle potential division by zero in MAPE if y_test contains zeros
        y_test_mape = np.where(y_test == 0, 0.01, y_test)
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
        logger.info(f"{log_prefix} test_scores_df: \n{test_scores_df}")
        
        return test_scores_df

    @staticmethod
    def _evaluate_result(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        df_history_test: pd.DataFrame,
        log_prefix: str,
    ):
        """
        测试集预测数据
        """
        # Ensure y_test and y_pred are 1D arrays
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()

        # Data collection for plot
        cv_plot_df_window = pd.DataFrame()
        
        # Ensure the slice is valid and matches the length of y_pred/y_test
        time_slice = df_history_test["time"]
        if len(time_slice) != len(y_pred):
            logger.warning(f"{log_prefix} Length mismatch for plotting data: time_slice ({len(time_slice)}) vs y_pred ({len(y_pred)}). Adjusting to min length.")
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
    @staticmethod
    def test_results_save(args, log_prefix: str, test_scores_df, cv_plot_df):
        # 测试结果数据保存
        test_scores_df.to_csv(args.test_results_dir.joinpath("test_scores_df.csv"), index=False, encoding="utf-8")
        cv_plot_df.to_csv(args.test_results_dir.joinpath("cv_plot_df.csv"), index=False, encoding="utf-8")
        # if getattr(self.args, "disable_plotting", False):
        #     logger.info(f"{log_prefix} Skip plotting because disable_plotting=True.")
        #     return
        # 测试结果数据可视化
        required_cols = {"Y_preds", "Y_trues"}
        if cv_plot_df.empty or not required_cols.issubset(set(cv_plot_df.columns)):
            logger.warning(f"{log_prefix} No valid prediction columns found for visualization.")
            return
        if len(cv_plot_df["Y_preds"].values) == 0 or len(cv_plot_df["Y_trues"].values) == 0:
            logger.warning(f"{log_prefix} No data to visualize for test prediction.")
            return
        import matplotlib.pyplot as plt
        # 画布
        plt.figure(figsize=(25, 8))
        # 创建折线图
        plt.plot(cv_plot_df["Y_trues"].values, label="Trues", lw=1.7)
        plt.plot(cv_plot_df["Y_preds"].values, label="Preds", lw=1.7, ls="-.")
        # 增强视觉效果
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Trues and Preds Timeseries Plot")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(args.test_results_dir.joinpath("test_prediction.png"), bbox_inches="tight", dpi=300)
        # plt.show();




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
