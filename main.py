# -*- coding: utf-8 -*-

# ***************************************************
# * File        : main.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-12-11
# * Version     : 2.0
# * Description : 基于机器学习回归器的时间序列预测框架
# *               支持以下预测方法:
# *               1. USMDP - 单变量多步逐点 direct 预测
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
import copy
import time
import datetime
import warnings
from typing import List
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from config.config_sections import PRED_METHOD_CODE
from data_provider.data_loader import DataLoader
from features.FeatureScalering import (
    FeatureScaler,
    TargetScaler,
    resolve_feature_scaler_type,
    resolve_inverse_target_enabled,
    resolve_scale_features_enabled,
    resolve_scale_target_enabled,
    resolve_target_scaler_type,
)
from features.FeatureEngineering import FeatureEngineer
from models.ModelTraining import Trainer
from models.ModelTesting import Tester
from models.ModelForecasting import Forecaster
from data_provider.outlier_handling import empty_train_outlier_report
from utils.frequency import resolve_freq_step_minutes, resolve_samples_per_day

warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ.setdefault('LOG_NAME', LOGGING_LABEL)
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
        data_name = Path(self.args.data_path).stem if getattr(self.args, "data_path", None) else "unknown_data"
        pred_method_code = PRED_METHOD_CODE.get(self.args.pred_method, str(self.args.pred_method).lower())
        self.setting = f"{self.args.model_type}-{data_name}-{pred_method_code}-{self.args.window_days}"
        self.log_prefix = f"[{self.setting}]"
        # ------------------------------
        # 数据参数
        # ------------------------------
        # 数据读取路径
        self.args.data_dir = Path(self.args.data_dir)
        # 场景子路径:由 data_dir 解析,使结果保存路径与 config 目录布局对齐
        self.scenario_subpath = self._resolve_scenario_subpath(self.args.data_dir)
        self.step_minutes = resolve_freq_step_minutes(self.args.freq)
        # 目标时间序列每天样本数量
        self.n_per_day = resolve_samples_per_day(self.args.freq)
        # 时间序列当前时刻（历史/未来分界点 = 次日 00:00:00）
        now_time = pd.Timestamp(self.args.now_time).replace(tzinfo=None).floor("1D") + datetime.timedelta(days=1)
        # 时间序列历史数据开始时刻（= now_time - history_days）
        start_time = now_time - datetime.timedelta(days=self.args.history_days)
        # 时间序列未来结束时刻（= now_time + predict_days）
        future_time = now_time + datetime.timedelta(days=self.args.predict_days)
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
        # self.n_windows = int(self.args.history_days * self.n_per_day - self.window_len - self.horizon + 1) // self.horizon
        self.n_windows = int(self.args.history_days * self.n_per_day - self.window_len) // self.horizon + 1
        self.args.horizon = self.horizon
        self.args.n_windows = self.n_windows
        self.args.n_per_day = self.n_per_day
        # ------------------------------
        # 模型训练、测试、预测结果保存路径
        # ------------------------------
        self.args.checkpoints_dir = Path(self.args.checkpoints_dir).joinpath(self.scenario_subpath, self.setting)
        self.args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.args.test_results_dir = Path(self.args.test_results_dir).joinpath(self.scenario_subpath, self.setting)
        self.args.test_results_dir.mkdir(parents=True, exist_ok=True)
        self.args.pred_results_dir = Path(self.args.pred_results_dir).joinpath(self.scenario_subpath, self.setting)
        self.args.pred_results_dir.mkdir(parents=True, exist_ok=True)
        # ------------------------------
        # 参数合法性校验
        # ------------------------------
        if self.args.window_days >= self.args.history_days:
            raise ValueError(
                f"{self.log_prefix} window_days ({self.args.window_days}) must be less than "
                f"history_days ({self.args.history_days})."
            )
        if self.n_windows <= 0:
            logger.warning(
                f"{self.log_prefix} n_windows={self.n_windows} (<= 0). Testing will be skipped."
            )
        block_size = int(getattr(self.args, 'block_size', 0) or 0)
        if block_size < 0:
            raise ValueError(f"{self.log_prefix} block_size ({block_size}) must be >= 0.")
        # 滞后特征可用性校验:滑窗训练行数 = window_len - horizon 必须 > max(lags),
        # 否则 shift(lag) 产出的滞后列全 NaN,模型无声退化(仅对真正构造 lag 列的方法校验)。
        if self.args.pred_method != "univariate-single-multistep-direct-pointwise":
            effective_lags = [int(l) for l in (getattr(self.args, "lags", []) or []) if int(l) > 0]
            if effective_lags:
                max_lag = max(effective_lags)
                min_train_rows = self.window_len - self.horizon
                if min_train_rows <= max_lag:
                    min_window_days = -(-(max_lag + self.horizon) // self.n_per_day)
                    raise ValueError(
                        f"{self.log_prefix} window_days ({self.args.window_days}) too small for lags: "
                        f"sliding-window train rows = window_len - horizon = {self.window_len} - {self.horizon} "
                        f"= {min_train_rows}, but max(lags) = {max_lag}. "
                        f"Lag features would be all-NaN. Need window_days >= {min_window_days}."
                    )
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
        logger.info(f"{self.log_prefix} 特征缩放: {'启用' if resolve_scale_features_enabled(self.args) else '禁用'}")
        logger.info(f"{self.log_prefix} 目标缩放: {'启用' if resolve_scale_target_enabled(self.args) else '禁用'}")
        logger.info(f"{self.log_prefix} 目标逆变换: {'启用' if resolve_inverse_target_enabled(self.args) else '禁用'}")
        logger.info(f"{self.log_prefix} 类别特征: {'启用' if self.args.encode_categorical_features else '禁用'}")
        logger.info(f"{self.log_prefix} 模型融合: {'启用' if self.args.enable_ensemble else '禁用'}")
        logger.info(f"{self.log_prefix} 模型测试: {'启用' if self.args.is_testing else '禁用'}")
        logger.info(f"{self.log_prefix} 模型预测: {'启用' if self.args.is_forecasting else '禁用'}")
        logger.info(f"{self.log_prefix} 窗口并行数: {int(getattr(self.args, 'window_parallel_workers', 1) or 1)}")
        logger.info(f"{self.log_prefix} 多输出并行数: {int(getattr(self.args, 'multi_output_n_jobs', 1) or 1)}")
        logger.info(f"{self.log_prefix} 分位数并行数: {int(getattr(self.args, 'quantile_parallel_workers', 1) or 1)}")
        logger.info(f"{self.log_prefix} 集成并行数: {int(getattr(self.args, 'ensemble_parallel_workers', 1) or 1)}")
        logger.info(f"{self.log_prefix} 模型线程数: {int(getattr(self.args, 'model_thread_count', 1) or 1)}")
        logger.info(f"{self.log_prefix} 分块大小: {int(getattr(self.args, 'block_size', 0) or 0)}")
        logger.info(f"{self.log_prefix} 快速测试窗口上限: {getattr(self.args, 'max_test_windows', None)}")
        logger.info(f"{self.log_prefix} 测试窗口步长: {int(getattr(self.args, 'test_window_stride', 1) or 1)}")
        logger.info(f"{self.log_prefix} Horizon: {self.horizon}")
        logger.info(f"{self.log_prefix} Window length: {self.window_len}")
        logger.info(f"{self.log_prefix} Number of windows: {self.n_windows}")

    @staticmethod
    def _resolve_scenario_subpath(data_dir) -> Path:
        """
        由 data_dir 解析场景子路径,使结果保存路径与 config 目录布局对齐。

        去掉 data_dir 中 config 侧不存在的段:
        - "dataset":     数据集根目录
        - "demand_load": 数据集侧的分组目录(config 路径中无此段)

        例:
          ./dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_01a/
          -> aidc_electricity_computility/electricity/2026-06-11/A1_01a
        """
        _DATASET_NOISE_SEGMENTS = {"dataset", "demand_load"}
        parts = [p for p in Path(data_dir).parts if p not in ("", ".")]
        scenario_parts = [p for p in parts if p not in _DATASET_NOISE_SEGMENTS]
        return Path(*scenario_parts) if scenario_parts else Path()

    def train(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, categorical_features: List, mode: str="forecast", verbose: bool=False):
        """
        模型训练
        """
        train_start = time.perf_counter()
        # 创建特征预处理器
        scaler = FeatureScaler(
            self.args,
            scaler_type=resolve_feature_scaler_type(self.args),
            log_prefix=self.log_prefix,
            verbose=verbose,
        )
        target_scaler = TargetScaler(
            self.args,
            scaler_type=resolve_target_scaler_type(self.args),
            log_prefix=self.log_prefix,
            verbose=verbose,
        )
        # 模型训练类
        model_trainer = Trainer(args=self.args, log_prefix=self.log_prefix)
        # 模型训练
        model, scaler, target_scaler, selected_features = model_trainer.train(
            X_train = X_train,
            Y_train = Y_train,
            feature_scaler = scaler,
            target_scaler = target_scaler,
            categorical_features = categorical_features,
        )
        # 模型保存
        if mode == "forecast":
            model_trainer.model_save(model, target_scaler)
        logger.info(f"{self.log_prefix} Model Training runtime: {time.perf_counter() - train_start:.3f}s")

        return model, scaler, target_scaler, selected_features

    def test(self,
             df_history,
             df_date_history,
             df_weather_history,
             endogenous_features_with_target,
             target_feature,
             categorical_features):
        """
        模型滑窗测试
        """
        test_start = time.perf_counter()
        # ------------------------------
        # 模型滑窗测试结果收集
        # ------------------------------
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        train_outlier_report = empty_train_outlier_report()
        # ------------------------------
        # 判断是否有足够的历史数据保证至少一个完整的测试窗口
        # ------------------------------
        if self.n_windows <= 0:
            logger.warning(f"{self.log_prefix} Not enough data for testing with current window configuration (Total history points: {len(df_history)}")
            logger.warning(f"{self.log_prefix} Window length: {self.window_len}, Horizon: {self.horizon}). No tests will be performed.")
            return test_scores_df, cv_plot_df
        # ------------------------------
        # 模型滑窗测试过程
        # ------------------------------
        window_stride = max(1, int(getattr(self.args, "test_window_stride", 1) or 1))
        window_indices = list(range(1, int(self.n_windows + 1), window_stride))
        max_test_windows = getattr(self.args, "max_test_windows", None)
        if max_test_windows is not None:
            max_test_windows = max(1, int(max_test_windows))
            window_indices = window_indices[:max_test_windows]
        logger.info(f"{self.log_prefix} Testing windows selected: {window_indices}")
        
        window_workers = int(getattr(self.args, "window_parallel_workers", 1) or 1)
        payload_args = copy.deepcopy(self.args)
        if window_workers > 1:
            payload_args.multi_output_n_jobs = 1
            payload_args.model_thread_count = 1
        payloads = [
            {
                "args": payload_args,
                "log_prefix": self.log_prefix,
                "horizon": self.horizon,
                "window_len": self.window_len,
                "window": window,
                "df_history": df_history,
                "df_date_history": df_date_history,
                "df_weather_history": df_weather_history,
                "endogenous_features_with_target": endogenous_features_with_target,
                "target_feature": target_feature,
                "categorical_features": categorical_features,
                "train_start_time": self.train_start_time,
                "train_end_time": self.train_end_time,
            }
            for window in window_indices
        ]
        window_results = []
        if window_workers > 1 and len(payloads) > 1:
            logger.info(f"{self.log_prefix} Model Testing window parallel workers: {window_workers}")
            executor_cls = ProcessPoolExecutor
            executor_name = "process"
            try:
                executor = executor_cls(max_workers=window_workers)
            except (PermissionError, OSError) as exc:
                executor_cls = ThreadPoolExecutor
                executor_name = "thread"
                logger.warning(
                    f"{self.log_prefix} ProcessPoolExecutor unavailable, fallback to ThreadPoolExecutor: {exc}"
                )
                executor = executor_cls(max_workers=window_workers)
            with executor:
                logger.info(f"{self.log_prefix} Model Testing executor backend: {executor_name}")
                futures = [executor.submit(Tester._window_test, payload) for payload in payloads]
                for future in as_completed(futures):
                    window_results.append(future.result())
        else:
            for payload in payloads:
                window_results.append(Tester._window_test(payload))
        # ------------------------------
        # 滑窗测试结果解析
        # ------------------------------
        for result in sorted(window_results, key=lambda x: x["window"]):
            if "train_outlier_report" in result and not result["train_outlier_report"].empty:
                train_outlier_report = pd.concat([train_outlier_report, result["train_outlier_report"]], axis=0)
            
            if result["test_scores_df"] is None or result["cv_plot_df"] is None:
                continue
            test_scores_df = pd.concat([test_scores_df, result["test_scores_df"]], axis=0)
            cv_plot_df = pd.concat([cv_plot_df, result["cv_plot_df"]], axis=0)
        # 模型测试评价指标数据处理
        if not test_scores_df.empty:
            test_scores_df_median = test_scores_df.drop(columns=["time_range"]).median()
            test_scores_df_median = test_scores_df_median.to_frame().T.reset_index(drop=True, inplace=False)
            test_scores_df_median["time_range"] = "中位数"
            test_scores_df = pd.concat([test_scores_df, test_scores_df_median], axis=0)
        logger.info(f"{self.log_prefix} Model Testing train_outlier_report shape: {train_outlier_report.shape}")
        logger.info(f"{self.log_prefix} Model Testing cv_plot_df shape: {cv_plot_df.shape}")
        logger.info(f"{self.log_prefix} Model Testing test_scores_df: \n{test_scores_df}")
        # ------------------------------
        # 模型测试结果保存
        # ------------------------------
        logger.info(f"{self.log_prefix} {'=' * 48}")
        logger.info(f"{self.log_prefix} Model Testing result saving...")
        logger.info(f"{self.log_prefix} {'=' * 48}")
        Tester.test_results_save(self.args, self.log_prefix, test_scores_df, cv_plot_df, train_outlier_report)
        logger.info(f"{self.log_prefix} Model Testing result saved in: {self.args.test_results_dir}")

        logger.info(f"{self.log_prefix} Model Testing runtime: {time.perf_counter() - test_start:.3f}s")

        return test_scores_df, cv_plot_df

    def forecast(self,
                 model,
                 scaler_forecasting,
                 target_scaler_forecasting,
                 df_history,
                 df_future,
                 df_date_future,
                 df_weather_future,
                 endogenous_features_with_target,
                 target_feature,
                 target_output_features,
                 categorical_features,
                 selected_features=None):
        """
        模型预测
        """
        forecast_start = time.perf_counter()
        # 未来数据复制
        df_future_prediction = df_future.copy()
        # Global 模式下，未来数据补齐 series_id（若缺失）
        if getattr(self.args, "enable_global_training", False):
            series_id_col = getattr(self.args, "series_id_feature", "series_id")
            if series_id_col not in df_future_prediction.columns and series_id_col in df_history.columns:
                last_series_id = df_history[series_id_col].dropna()
                if not last_series_id.empty:
                    df_future_prediction[series_id_col] = last_series_id.iloc[-1]
        # 模型预测
        predictor = Forecaster(
            args=self.args,
            horizon=self.horizon,
            model=model,
            feature_scaler=scaler_forecasting,
            target_scaler=target_scaler_forecasting,
            df_history=df_history,
            df_future=df_future_prediction,
            df_date_future=df_date_future,
            df_weather_future=df_weather_future,
            endogenous_features=endogenous_features_with_target,
            target_feature=target_feature,
            target_output_features=target_output_features,
            categorical_features=categorical_features,
            selected_features=selected_features,
            log_prefix=self.log_prefix,
        )
        Y_pred = predictor._predict_by_method()
        # ------------------------------
        # 模型预测结果收集和保存
        # ------------------------------
        logger.info(f"{self.log_prefix} {'=' * 87}")
        logger.info(f"{self.log_prefix} Model Forecasting result save...")
        logger.info(f"{self.log_prefix} {'=' * 87}")
        # 模型预测结果收集
        if target_scaler_forecasting is None:
            pred_target_columns = list(target_output_features or [target_feature])
            Y_pred = np.asarray(Y_pred).reshape(-1)
            if len(Y_pred) != len(df_future_prediction):
                logger.warning(
                    f"{self.log_prefix} Y_pred length ({len(Y_pred)}) "
                    f"!= df_future_prediction length ({len(df_future_prediction)}); truncating."
                )
                Y_pred = Y_pred[:len(df_future_prediction)]
        else:
            pred_target_columns = target_scaler_forecasting.get_prediction_target_columns(
                self.args.pred_method,
                target_output_features,
            )
            Y_pred = target_scaler_forecasting.restore_predictions(Y_pred, pred_target_columns)
        df_future_prediction["predict_value"] = np.asarray(Y_pred).reshape(-1)[:len(df_future_prediction)]
        # 分位数预测结果（若启用）
        if getattr(predictor, "quantile_outputs", None):
            for q, q_pred in sorted(predictor.quantile_outputs.items(), key=lambda x: float(x[0])):
                q_col = f"predict_q{int(round(float(q) * 100)):02d}"
                if target_scaler_forecasting is None:
                    q_arr = np.asarray(q_pred).reshape(-1)
                else:
                    q_arr = target_scaler_forecasting.restore_predictions(
                        np.asarray(q_pred).reshape(-1),
                        pred_target_columns,
                    ).reshape(-1)
                if len(q_arr) != len(df_future_prediction):
                    min_len = min(len(q_arr), len(df_future_prediction))
                    df_future_prediction.loc[df_future_prediction.index[:min_len], q_col] = q_arr[:min_len]
                else:
                    df_future_prediction[q_col] = q_arr
        quantile_cols = [c for c in df_future_prediction.columns if c.startswith("predict_q")]
        if quantile_cols:
            df_future_prediction = df_future_prediction[["time", "predict_value"] + quantile_cols]
        else:
            df_future_prediction = df_future_prediction[["time", "predict_value"]]
        logger.info(f"{self.log_prefix} after forecast df_future_prediction: \n{df_future_prediction.head()}")
        logger.info(f"{self.log_prefix} after forecast df_future_prediction.shape: {df_future_prediction.shape}")
        # 模型预测结果保存
        if target_scaler_forecasting is None:
            history_target = target_feature if target_feature in df_history.columns else "y"
            df_history_for_plot = df_history[["time", history_target]].rename(columns={history_target: "y"}).copy()
        else:
            df_history_for_plot = target_scaler_forecasting.prepare_history_target_for_plot(
                df_history,
                [target_output_features[0]],
            )
        predictor.forecast_results_save(df_history_for_plot, df_future_prediction, self.n_per_day)
        logger.info(f"{self.log_prefix} Model Forecasting result saved in: {self.args.pred_results_dir}")
        logger.info(f"{self.log_prefix} Model Forecasting runtime: {time.perf_counter() - forecast_start:.3f}s")

        return df_future_prediction

    def run(self):
        run_start = time.perf_counter()
        # ------------------------------
        # 数据加载和处理
        # ------------------------------
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} Model history and future data loading...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
        # 数据加载
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
         target_feature) = dataloader.process_history_data(input_data=input_data)
        # ------------------------------
        # 特征工程
        # ------------------------------
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} Model history data feature engineering...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
        # 特征预处理器
        verbose_logging = bool(getattr(self.args, "enable_step_logging", False))
        feature_engineer_history = FeatureEngineer(self.args, self.log_prefix, verbose=verbose_logging)
        (df_history_featured,
         predictor_features,
         target_output_features,
         categorical_features) = feature_engineer_history.create_features(
            df_series=df_history,
            df_date_history=df_date_history,
            df_date_future=None,
            df_weather_history=df_weather_history,
            df_weather_future=None,
            endogenous_features_with_target=endogenous_features_with_target,
            target_feature=target_feature,
            horizon=self.horizon,
        )
        # 删除在构建目标输出时产生的缺失值（仅按目标列过滤，避免外生缺失导致样本被清空）
        df_history_featured = df_history_featured.dropna(subset=target_output_features)
        logger.info(f"{self.log_prefix} after dropna df_history_featured: \n{df_history_featured.head()}")
        logger.info(f"{self.log_prefix} after dropna df_history_featured.shape: {df_history_featured.shape}")
        # 历史数据预测特征、目标特征分离
        logger.info(f"{self.log_prefix} {'=' * 87}")
        logger.info(f"{self.log_prefix} Model history data feature split...")
        logger.info(f"{self.log_prefix} {'=' * 87}")
        X_train_history, Y_train_history = feature_engineer_history.predictor_target_split(
            df_series_featured=df_history_featured,
            predictor_features=predictor_features,
            target_output_features=target_output_features,
        )
        # ------------------------------
        # 模型测试
        # ------------------------------
        if self.args.is_testing:
            logger.info(f"{self.log_prefix} {'#' * 90}")
            logger.info(f"{self.log_prefix} Model Testing...")
            logger.info(f"{self.log_prefix} {'#' * 90}")
            test_scores_df, cv_plot_df = self.test(
                df_history=df_history,
                df_date_history=df_date_history,
                df_weather_history=df_weather_history,
                endogenous_features_with_target=endogenous_features_with_target,
                target_feature=target_feature,
                categorical_features=categorical_features,
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
            (df_future,
             df_date_future,
             df_weather_future) = dataloader.process_future_data(input_data=input_data)

            # 模型训练
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Training start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            model, scaler_forecasting, target_scaler_forecasting, selected_features = self.train(
                X_train=X_train_history,
                Y_train=Y_train_history,
                categorical_features=categorical_features,
                mode="forecast",
                verbose=verbose_logging,
            )

            # 模型预测
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Forecasting start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            df_future_predicted = self.forecast(
                model=model,
                scaler_forecasting=scaler_forecasting,
                target_scaler_forecasting=target_scaler_forecasting,
                df_history=df_history,
                df_future=df_future,
                df_date_future=df_date_future,
                df_weather_future=df_weather_future,
                endogenous_features_with_target=endogenous_features_with_target,
                target_feature=target_feature,
                target_output_features=target_output_features,
                categorical_features=categorical_features,
                selected_features=selected_features,
            )
        logger.info(f"{self.log_prefix} Total runtime: {time.perf_counter() - run_start:.3f}s")




# 测试代码 main 函数
def main():
    """
    主函数入口

    配置文件切换：修改 CONFIG_YAML 即可切换不同的模型配置。
    YAML 内部的 base_config 字段指定基础 Python 配置模块入口。
    """
    # ensuer runtime environment
    from utils.runtime_env import ensure_runtime_environment
    ensure_runtime_environment()
    # ------------------------------
    # 配置文件切换区域
    # ------------------------------
    CONFIG_YAML = "config/aidc_electricity_computility/electricity/2026-06-11/A1_01a/cab_usmdp.yaml"
    # ------------------------------
    # 创建模型配置参数
    # ------------------------------
    args = load_yaml_config(CONFIG_YAML)
    # 创建模型实例
    model = Model(args)
    # 运行模型
    try:
        model.run()
    except Exception as e:
        logger.error(
            f"{model.log_prefix} Pipeline FAILED: {e}",
            exc_info=True,
        )
        raise
    logger.info(f"{model.log_prefix} {'#' * 85}")
    logger.info(f"{model.log_prefix} 模型预测流程完成！")
    logger.info(f"{model.log_prefix} {'#' * 85}")

if __name__ == "__main__":
    main()
