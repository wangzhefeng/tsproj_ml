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
    TargetDetrender,
    resolve_detrend_target_enabled,
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
from utils.frequency import resolve_freq_step_minutes, resolve_samples_per_day, is_monthly_freq
from utils.quantile import monotonize_quantile_columns

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
        # 概率预测(quantile)用独立 setting 后缀,避免与点预测版本的结果目录/模型撞车
        _predict_suffix = "-quantile" if str(getattr(self.args, "predict_type", "point")).lower() == "quantile" else ""
        # 可选自定义后缀（如 "-intraday"），用于同配置不同语义版本的结果隔离
        _custom_suffix = str(getattr(self.args, "setting_suffix", "") or "").strip()
        self.setting = f"{self.args.model_type}-{data_name}-{pred_method_code}-{self.args.window_length}{_predict_suffix}{_custom_suffix}"
        self.log_prefix = f"[{self.setting}]"
        # ------------------------------
        # 数据参数
        # ------------------------------
        # 数据读取路径
        self.args.data_dir = Path(self.args.data_dir)
        # 场景子路径:优先用 YAML 显式指定的 scenario_subpath(多组配置共用同一 data_dir 时用);
        # 为空时由 data_dir 自动推导,使结果保存路径与 config 目录布局对齐。
        explicit_scenario = str(getattr(self.args, "scenario_subpath", "") or "").strip().strip("/")
        self.scenario_subpath = (
            Path(*explicit_scenario.split("/")) if explicit_scenario
            else self._resolve_scenario_subpath(self.args.data_dir)
        )
        self.step_minutes = resolve_freq_step_minutes(self.args.freq)
        # 目标时间序列每天样本数量
        self.n_per_day = resolve_samples_per_day(self.args.freq)
        # 时间序列当前时刻（历史/未来分界点）
        #   schedule_mode=daily(默认): floor("1D")+1day 对齐到次日 00:00, 预测下一完整自然日
        #   schedule_mode=intraday   : 保留调度时刻, 预测从调度时刻起 predict_steps 步
        #   月频(freq=1ME): 分界点推到下月月初(类比 daily 的 +1day), 使历史含完整月末点
        now_ts = pd.Timestamp(self.args.now_time).replace(tzinfo=None)
        is_monthly = is_monthly_freq(self.args.freq)
        if is_monthly:
            # 月频分界点 = 下月月初（类比 daily 的 +1day），使历史含完整月末点
            now_time = (now_ts.to_period("M") + 1).to_timestamp()
            # 历史开始时刻 = now - history_length 月
            start_time = now_time - pd.DateOffset(months=self.args.history_length)
        elif str(getattr(self.args, "schedule_mode", "daily")).lower() == "intraday":
            now_time = now_ts
            start_time = now_time - datetime.timedelta(days=self.args.history_length)
        else:
            now_time = now_ts.floor("1D") + datetime.timedelta(days=1)
            start_time = now_time - datetime.timedelta(days=self.args.history_length)
        # 时间序列历史数据开始时刻（= now_time - history_length）
        # 预测数据长度（= predict_steps 个 freq 步长）
        self.horizon = int(self.args.predict_steps)
        # 时间序列未来结束时刻（= now_time + predict_steps × freq 步长）
        if is_monthly:
            future_time = now_time + pd.DateOffset(months=self.horizon)
        else:
            future_time = now_time + datetime.timedelta(minutes=self.step_minutes * self.horizon)
        # 数据划分时间戳
        self.train_start_time = start_time
        self.train_end_time = now_time
        self.forecast_start_time = now_time
        self.forecast_end_time = future_time
        # ------------------------------
        # 模型测试、预测
        # ------------------------------
        # 测试窗口数据长度(训练+测试)
        self.window_len = int(self.args.window_length * self.n_per_day)
        # 测试滑动窗口数量, >=1, 1: 单个窗口
        # self.n_windows = int(self.args.history_length * self.n_per_day - self.window_len - self.horizon + 1) // self.horizon
        self.n_windows = int(self.args.history_length * self.n_per_day - self.window_len) // self.horizon + 1
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
        if self.args.window_length >= self.args.history_length:
            raise ValueError(
                f"{self.log_prefix} window_length ({self.args.window_length}) must be less than "
                f"history_length ({self.args.history_length})."
            )
        if self.n_windows <= 0:
            logger.warning(
                f"{self.log_prefix} n_windows={self.n_windows} (<= 0). Testing will be skipped."
            )
        block_size = int(getattr(self.args, 'block_size', 0) or 0)
        if block_size < 0:
            raise ValueError(f"{self.log_prefix} block_size ({block_size}) must be >= 0.")
        # detrend_target 与 scale_target 互斥:两者同开会双重加趋势且逆变换顺序错乱
        if resolve_detrend_target_enabled(self.args) and resolve_scale_target_enabled(self.args):
            raise ValueError(
                f"{self.log_prefix} detrend_target and scale_target are mutually exclusive "
                f"(both enabled would double-apply the trend). Keep scale_target=false when detrending."
            )
        # 滞后特征可用性校验:滑窗训练行数 = window_len - horizon 必须 > max(lags),
        # 否则 shift(lag) 产出的滞后列全 NaN,模型无声退化(仅对真正构造 lag 列的方法校验)。
        if self.args.pred_method != "univariate-single-multistep-direct-pointwise":
            effective_lags = [int(l) for l in (getattr(self.args, "lags", []) or []) if int(l) > 0]
            if effective_lags:
                max_lag = max(effective_lags)
                min_train_rows = self.window_len - self.horizon
                if min_train_rows <= max_lag:
                    min_window_days = (max_lag + self.horizon) // self.n_per_day + 1
                    raise ValueError(
                        f"{self.log_prefix} window_length ({self.args.window_length}) too small for lags: "
                        f"sliding-window train rows = window_len - horizon = {self.window_len} - {self.horizon} "
                        f"= {min_train_rows}, but max(lags) = {max_lag}. "
                        f"Lag features would be all-NaN. To keep at least one valid lag row, "
                        f"need window_length >= {min_window_days}."
                    )
        # ------------------------------
        # 预测增强策略(v1)组合校验:不支持的模式必须显式拒绝,避免裸奔崩溃或静默错配
        # ------------------------------
        pred_method_l = str(self.args.pred_method).lower()
        blend_methods = {
            "univariate-single-multistep-blend-direct-recursive",
            "multivariate-single-multistep-blend-direct-recursive",
        }
        if pred_method_l in blend_methods:
            if bool(getattr(self.args, "enable_ensemble", False)):
                raise ValueError(
                    f"{self.log_prefix} USBR/MSBR blend + enable_ensemble is not supported in v1 "
                    f"(ensemble path would train on the H+1-column blend target table and crash at forecast). "
                    f"Disable ensemble, or use a non-blend method."
                )
            if resolve_scale_target_enabled(self.args):
                raise ValueError(
                    f"{self.log_prefix} USBR/MSBR blend + scale_target is not supported in v1 "
                    f"(Direct/Recursive sub-models live in different scaled spaces; "
                    f"blend weights and restore become ill-defined). Keep scale_target=false."
                )
            if resolve_detrend_target_enabled(self.args):
                raise ValueError(
                    f"{self.log_prefix} USBR/MSBR blend + detrend_target is not supported in v1 "
                    f"(test-time blend sub-predictions stay in detrend space while eval targets are "
                    f"restored to level space; ridge_stacking weights would be learned on mismatched spaces). "
                    f"Keep detrend_target=false."
                )
        if bool(getattr(self.args, "enable_conformal_calibration", False)) and resolve_detrend_target_enabled(self.args):
            raise ValueError(
                f"{self.log_prefix} enable_conformal_calibration + detrend_target is not supported: "
                f"test-time quantiles are not detrend-restored (calibration scores in detrend space) while "
                f"forecast bands are in level space. Keep detrend_target=false when calibrating."
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

    def train(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, categorical_features: List, mode: str="forecast", verbose: bool=False,
              df_history=None, endogenous_features_with_target=None):
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
        # 多变量递归辅助预测器包装（MSMR/MSMDR + endogenous_backfill_strategy=auxiliary）
        if df_history is not None and endogenous_features_with_target is not None:
            from models.AuxiliaryForecaster import maybe_build_auxiliary_bundle
            model = maybe_build_auxiliary_bundle(
                self.args, model, df_history,
                endogenous_features_with_target, self.args.target, self.log_prefix,
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
             categorical_features,
             df_custom_history=None):
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
        if (
            str(getattr(self.args, "pred_method", "")) == "multivariate-single-multistep-direct"
            and window_workers > 1
        ):
            logger.warning(
                f"{self.log_prefix} pred_method=multivariate-single-multistep-direct with "
                f"window_parallel_workers={window_workers} will force multi_output_n_jobs=1 "
                f"and model_thread_count=1 during testing. This usually slows msmd testing; "
                f"prefer window_parallel_workers=1."
            )
        payload_args = copy.deepcopy(self.args)
        if window_workers > 1:
            payload_args.multi_output_n_jobs = 1
            payload_args.model_thread_count = 1
            payload_args.ensemble_parallel_workers = 1
        payloads = [
            {
                "args": payload_args,
                "force_single_thread_env": window_workers > 1,
                "log_prefix": self.log_prefix,
                "horizon": self.horizon,
                "window_len": self.window_len,
                "window": window,
                "df_history": df_history,
                "df_date_history": df_date_history,
                "df_weather_history": df_weather_history,
                "df_custom_history": df_custom_history,
                "endogenous_features_with_target": endogenous_features_with_target,
                "target_feature": target_feature,
                "categorical_features": categorical_features,
                "train_start_time": self.train_start_time,
                "train_end_time": self.train_end_time,
                "target_detrender": getattr(self, "target_detrender", None),
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
                 selected_features=None,
                 df_custom_future=None):
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
            df_custom_future=df_custom_future,
            endogenous_features=endogenous_features_with_target,
            target_feature=target_feature,
            target_output_features=target_output_features,
            categorical_features=categorical_features,
            selected_features=selected_features,
            target_detrender=getattr(self, "target_detrender", None),
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
                direct_strategy=str(getattr(self.args, "direct_strategy", "multioutput")),
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
        # Conformal CQR 校准（可选）：用滑窗 nonconformity scores 对 q_low/q_high 对称膨胀
        if bool(getattr(self.args, "enable_conformal_calibration", False)):
            quantile_cols_cal = [c for c in df_future_prediction.columns if str(c).startswith("predict_q")]
            if len(quantile_cols_cal) >= 2:
                from utils.conformal import calibrate_quantile_band
                cv_path = self.args.test_results_dir.joinpath("cv_plot_df.csv")
                if cv_path.exists():
                    cv_df = pd.read_csv(cv_path)
                    if "conformal_score" in cv_df.columns:
                        n_cal_windows = int(getattr(self.args, "conformal_calibration_windows", 5))
                        if "window" in cv_df.columns:
                            recent_windows = sorted(cv_df["window"].unique())[-n_cal_windows:]
                            cal_scores = cv_df[cv_df["window"].isin(recent_windows)]["conformal_score"].dropna().values
                        else:
                            cal_scores = cv_df["conformal_score"].dropna().values
                        alpha = float(getattr(self.args, "conformal_alpha", 0.1))
                        min_scores = int(getattr(self.args, "conformal_min_scores", 30))
                        cal_low, cal_high, E_alpha = calibrate_quantile_band(
                            df_future_prediction[quantile_cols_cal[0]].values,
                            df_future_prediction[quantile_cols_cal[-1]].values,
                            cal_scores, alpha, min_scores,
                        )
                        if cal_low is not None:
                            df_future_prediction[quantile_cols_cal[0]] = cal_low
                            df_future_prediction[quantile_cols_cal[-1]] = cal_high
                            logger.info(
                                f"{self.log_prefix} Conformal CQR calibrated: E_alpha={E_alpha:.4f}, "
                                f"target coverage={1 - alpha:.2f}, n_scores={len(cal_scores)}"
                            )
                        else:
                            logger.warning(
                                f"{self.log_prefix} Conformal CQR skipped: only {len(cal_scores)} scores "
                                f"(< {min_scores}); need more test windows"
                            )
                    else:
                        logger.warning(
                            f"{self.log_prefix} Conformal CQR skipped: no conformal_score column in cv_plot_df.csv "
                            f"(need enable_conformal_calibration=true during testing)"
                        )
                else:
                    logger.warning(
                        f"{self.log_prefix} Conformal CQR skipped: cv_plot_df.csv not found "
                        f"(need is_testing=True to produce calibration scores)"
                    )
        # 分位数单调化(可选):逐行排序保证 q10<=q50<=q90
        df_future_prediction = monotonize_quantile_columns(
            df_future_prediction, bool(getattr(self.args, "quantile_monotone", False))
        )
        quantile_cols = [c for c in df_future_prediction.columns if c.startswith("predict_q")]
        if quantile_cols:
            df_future_prediction = df_future_prediction[["time", "predict_value"] + quantile_cols]
        else:
            df_future_prediction = df_future_prediction[["time", "predict_value"]]
        logger.info(f"{self.log_prefix} after forecast df_future_prediction: \n{df_future_prediction.head()}")
        logger.info(f"{self.log_prefix} after forecast df_future_prediction.shape: {df_future_prediction.shape}")
        # 模型预测结果保存
        # 绘图历史用真电平(detrend 关闭时 df_history_levels 即 df_history 本身)
        df_history_plot_src = getattr(self, "df_history_levels", df_history)
        if target_scaler_forecasting is None:
            history_target = target_feature if target_feature in df_history_plot_src.columns else "y"
            df_history_for_plot = df_history_plot_src[["time", history_target]].rename(columns={history_target: "y"}).copy()
        else:
            df_history_for_plot = target_scaler_forecasting.prepare_history_target_for_plot(
                df_history_plot_src,
                [target_output_features[0]],
            )
        predictor.forecast_results_save(df_history_for_plot, df_future_prediction, self.n_per_day)
        logger.info(f"{self.log_prefix} Model Forecasting result saved in: {self.args.pred_results_dir}")
        logger.info(f"{self.log_prefix} Model Forecasting runtime: {time.perf_counter() - forecast_start:.3f}s")

        return df_future_prediction

    def _learn_blend_weights(self):
        """Blend ridge_stacking：从 cv_plot_df 学 Direct/Recursive 最优权重，写 blend_weights.csv。"""
        cv_path = self.args.test_results_dir.joinpath("cv_plot_df.csv")
        if not cv_path.exists():
            logger.warning(f"{self.log_prefix} ridge_stacking: cv_plot_df.csv not found; using fixed blend_weights.")
            return
        cv = pd.read_csv(cv_path)
        needed = ["Y_trues", "blend_direct_pred", "blend_recursive_pred"]
        if not all(c in cv.columns for c in needed):
            logger.warning(f"{self.log_prefix} ridge_stacking: cv_plot_df missing blend columns; using fixed blend_weights.")
            return
        cv_clean = cv.dropna(subset=needed)
        if len(cv_clean) < 10:
            logger.warning(f"{self.log_prefix} ridge_stacking: only {len(cv_clean)} valid rows; using fixed blend_weights.")
            return
        # 与 conformal 校准一致：只用最近 N 个窗口的数据学权重（分布更贴合当前预测任务）
        n_cal_windows = int(getattr(self.args, "blend_weight_windows", 5))
        if "window" in cv_clean.columns:
            recent_windows = sorted(cv_clean["window"].unique())[-n_cal_windows:]
            cv_clean = cv_clean[cv_clean["window"].isin(recent_windows)]
        if len(cv_clean) < 10:
            logger.warning(f"{self.log_prefix} ridge_stacking: only {len(cv_clean)} rows in recent {n_cal_windows} windows; using fixed blend_weights.")
            return
        from sklearn.linear_model import Ridge
        X_stack = cv_clean[["blend_direct_pred", "blend_recursive_pred"]].values
        y_stack = cv_clean["Y_trues"].values
        # 无截距凸组合：截距在归一化时会被丢弃，导致权重偏离最优（系统偏差大时尤其明显）
        ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(X_stack, y_stack)
        w = ridge.coef_
        total = float(w.sum())
        if total <= 0:
            w = np.array([0.5, 0.5])
            total = 1.0
        w_norm = w / total
        w_df = pd.DataFrame([{
            "direct_weight": float(w_norm[0]),
            "recursive_weight": float(w_norm[1]),
            "n_samples": len(cv_clean),
        }])
        w_df.to_csv(self.args.test_results_dir.joinpath("blend_weights.csv"), index=False)
        logger.info(
            f"{self.log_prefix} ridge_stacking weights: direct={w_norm[0]:.4f}, "
            f"recursive={w_norm[1]:.4f} (n_samples={len(cv_clean)})"
        )

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
         target_feature,
         df_custom_history) = dataloader.process_history_data(input_data=input_data)
        # ------------------------------
        # 目标去趋势(可选):特征工程前对整条 y 线性去趋势,
        # 使 target/lag/rolling/diff 一致落在 detrended 空间;Forecaster 输出时点对点还原电平
        # ------------------------------
        self.df_history_levels = df_history.copy()
        self.target_detrender = TargetDetrender(
            self.args,
            log_prefix=self.log_prefix,
            verbose=bool(getattr(self.args, "enable_step_logging", False)),
        )
        if self.target_detrender.enabled:
            self.target_detrender.fit(df_history, time_col="time", target_col="y")
            df_history = self.target_detrender.detrend(df_history)
            logger.info(f"{self.log_prefix} 目标去趋势(detrend_target): 启用")
        else:
            logger.info(f"{self.log_prefix} 目标去趋势(detrend_target): 禁用")
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
            df_custom_history=df_custom_history,
            df_custom_future=None,
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
                df_custom_history=df_custom_history,
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
             df_weather_future,
             df_custom_future) = dataloader.process_future_data(input_data=input_data)

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
                df_history=df_history,
                endogenous_features_with_target=endogenous_features_with_target,
            )

            # 模型预测
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Forecasting start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            # Blend ridge_stacking：forecast 前从测试结果学权重（需 is_testing=True 先产出 cv_plot_df）
            if (
                str(getattr(self.args, "pred_method", "")).lower()
                in (
                    "univariate-single-multistep-blend-direct-recursive",
                    "multivariate-single-multistep-blend-direct-recursive",
                )
                and str(getattr(self.args, "blend_weight_strategy", "fixed")).lower() == "ridge_stacking"
            ):
                self._learn_blend_weights()
            df_future_predicted = self.forecast(
                model=model,
                scaler_forecasting=scaler_forecasting,
                target_scaler_forecasting=target_scaler_forecasting,
                df_history=df_history,
                df_future=df_future,
                df_date_future=df_date_future,
                df_weather_future=df_weather_future,
                df_custom_future=df_custom_future,
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
    # CONFIG_YAML = "config/aidc_electricity_computility/electricity/2026-06-11/A1_01a/lgbm_msmd.yaml"
    CONFIG_YAML = "config/aidc_load_month/route_B/lgbm_usmd_prob_mean.yaml"
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
