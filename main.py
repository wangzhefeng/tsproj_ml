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
from typing import List, cast
from pathlib import Path
ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
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
from decomposition.spec import resolve_decomposition_spec
from features.TargetTransformation import TargetTransformPipeline
from models.ModelTraining import Trainer
from models.ModelTesting import Tester
from models.ModelForecasting import Forecaster
from models.multistep.panel import PanelSeriesSlice, execute_panel
from models.multistep.plans import LagPolicy, RowAlignment
from models.multistep.resolve import resolve_strategy
from models.multistep.spec import InputScope, RolloutFamily, get_strategy_spec
from models.multistep.weights import BlendWeights
from probabilistic.calibration import attach_cqr_interval_columns
from probabilistic.evaluation import (
    append_final_calibration_report,
    calibrate_final_from_cv,
)
from probabilistic.pipeline import finalize_quantile_forecast
from probabilistic.objectives import validate_quantile_model_support
from probabilistic.types import ForecastDistribution, QuantileGrid
from probabilistic.spec import (
    apply_probabilistic_spec_to_args,
    calibration_runtime_kwargs,
    resolve_probabilistic_spec,
)
from data_provider.outlier_handling import empty_train_outlier_report
from utils.frequency import resolve_freq_step_minutes, resolve_samples_per_day, is_monthly_freq
from utils.multistep_contract import validate_direct_feature_alignment
from utils.parallel_budget import apply_window_parallel_budget
from utils.weather_contract import validate_weather_information_contract

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
        self.probabilistic_spec = resolve_probabilistic_spec(self.args)
        validate_quantile_model_support(
            getattr(self.args, "model_type", ""),
            self.probabilistic_spec,
        )
        apply_probabilistic_spec_to_args(self.args, self.probabilistic_spec)
        self.horizon_mode = str(
            getattr(self.args, "horizon_mode", "fixed_steps") or "fixed_steps"
        ).lower()
        if self.horizon_mode not in {"fixed_steps", "calendar_month"}:
            raise ValueError(
                f"Unsupported horizon_mode={self.horizon_mode}. "
                "Supported: fixed_steps, calendar_month."
            )
        data_name = Path(self.args.data_path).stem if getattr(self.args, "data_path", None) else "unknown_data"
        pred_method_code = get_strategy_spec(self.args.pred_method).code
        # 概率预测(quantile)用独立 setting 后缀,避免与点预测版本的结果目录/模型撞车
        _predict_suffix = "-quantile" if str(getattr(self.args, "predict_type", "point")).lower() == "quantile" else ""
        # 可选自定义后缀（如 "-intraday"），用于同配置不同语义版本的结果隔离
        _custom_suffix = str(getattr(self.args, "setting_suffix", "") or "").strip()
        _horizon_suffix = "-calendar-month" if self.horizon_mode == "calendar_month" else ""
        # calendar_month 模式下 window_length 只是遗留兼容字段，真实训练长度是
        # train_window_length；setting 编码真实值，避免结果目录名误导（150→120）。
        _window_token = (
            str(int(getattr(self.args, "train_window_length", 0) or 0))
            if self.horizon_mode == "calendar_month"
            else str(self.args.window_length)
        )
        self.setting = (
            f"{self.args.model_type}-{data_name}-{pred_method_code}-{_window_token}"
            f"{_predict_suffix}{_custom_suffix}{_horizon_suffix}"
        )
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
        # 历史数据开始时刻（= now_time - history_length）
        # 预测跨度：默认固定 predict_steps；calendar_month 自动覆盖目标自然月。
        future_time = None
        if self.horizon_mode == "calendar_month":
            if str(self.args.freq) != "1D":
                raise ValueError("horizon_mode=calendar_month currently requires freq=1D.")
            if str(getattr(self.args, "schedule_mode", "daily")).lower() != "daily":
                raise ValueError("horizon_mode=calendar_month requires schedule_mode=daily.")
            if now_time.day != 1:
                raise ValueError(
                    f"horizon_mode=calendar_month requires forecast_start at month start; got {now_time}."
                )
            train_window_length = getattr(self.args, "train_window_length", None)
            if train_window_length is None or int(train_window_length) <= 0:
                raise ValueError(
                    "horizon_mode=calendar_month requires train_window_length > 0."
                )
            self.horizon = int(now_time.days_in_month)
            future_time = now_time + pd.DateOffset(months=1)
        else:
            self.horizon = int(self.args.predict_steps)
        self.resolved_strategy = resolve_strategy(self.args, self.horizon)
        # 时间序列未来结束时刻
        if is_monthly:
            future_time = now_time + pd.DateOffset(months=self.horizon)
        elif self.horizon_mode != "calendar_month":
            future_time = now_time + datetime.timedelta(minutes=self.step_minutes * self.horizon)
        if future_time is None:
            raise RuntimeError("Failed to resolve forecast_end_time.")
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
        if self.horizon_mode == "calendar_month":
            self.train_window_len = int(self.args.train_window_length * self.n_per_day)
            theoretical_history = pd.DataFrame(
                {"time": pd.date_range(start_time, now_time, freq="1D", inclusive="left")}
            )
            self.n_windows = len(
                Tester._build_calendar_month_folds(
                    theoretical_history,
                    train_window_len=self.train_window_len,
                )
            )
        else:
            self.train_window_len = self.window_len - self.horizon
            # 测试滑动窗口数量, >=1, 1: 单个窗口
            self.n_windows = int(self.args.history_length * self.n_per_day - self.window_len) // self.horizon + 1
        self.args.horizon = self.horizon
        self.args.n_windows = self.n_windows
        self.args.n_per_day = self.n_per_day
        self.args.train_window_len = self.train_window_len
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
        effective_train_length = (
            int(self.args.train_window_length)
            if self.horizon_mode == "calendar_month"
            else int(self.args.window_length)
        )
        if effective_train_length >= self.args.history_length:
            raise ValueError(
                f"{self.log_prefix} effective training length ({effective_train_length}) must be less than "
                f"history_length ({self.args.history_length})."
            )
        target_calendar_normalization = str(
            getattr(self.args, "target_calendar_normalization", "none") or "none"
        ).lower()
        if target_calendar_normalization == "per_calendar_day" and not is_monthly:
            raise ValueError(
                f"{self.log_prefix} target_calendar_normalization=per_calendar_day is only valid for monthly freq."
            )
        validate_direct_feature_alignment(self.args, self.horizon)
        validate_weather_information_contract(self.args)
        if self.n_windows <= 0:
            logger.warning(
                f"{self.log_prefix} n_windows={self.n_windows} (<= 0). Testing will be skipped."
            )
        block_size = int(getattr(self.args, 'block_size', 0) or 0)
        if block_size < 0:
            raise ValueError(f"{self.log_prefix} block_size ({block_size}) must be >= 0.")
        spec = resolve_decomposition_spec(self.args)
        decomposition_method = spec.method
        # 周期从 spec 读取（新写法 overrides.decomposition 的 periods 不落在 legacy 属性上）
        decomposition_periods = list(spec.preset.periods) if spec.preset else []
        if decomposition_method == "stl" and len(decomposition_periods) != 1:
            raise ValueError(f"{self.log_prefix} decomposition_method=stl requires exactly one period.")
        if decomposition_method == "mstl" and len(decomposition_periods) < 2:
            raise ValueError(f"{self.log_prefix} decomposition_method=mstl requires at least two periods.")
        if decomposition_method in {"stl", "mstl"}:
            decomposition_train_rows = self.train_window_len
            if any(
                period < 2 or 2 * period > decomposition_train_rows
                for period in decomposition_periods
            ):
                raise ValueError(
                    f"{self.log_prefix} decomposition_periods={decomposition_periods} require at least "
                    f"two full cycles in each {decomposition_train_rows}-point training window."
                )
        # 滞后特征可用性校验:滑窗训练行数必须 > max(lags),
        # 否则 shift(lag) 产出的滞后列全 NaN,模型无声退化(仅对真正构造 lag 列的方法校验)。
        constructs_lags = self.resolved_strategy.feature_plan.lag_policy != LagPolicy.NONE
        if constructs_lags:
            effective_lags = [int(l) for l in (getattr(self.args, "lags", []) or []) if int(l) > 0]
            if effective_lags:
                max_lag = max(effective_lags)
                if self.resolved_strategy.feature_plan.row_alignment == RowAlignment.TARGET_TIME:
                    max_lag -= 1
                min_train_rows = self.train_window_len
                if min_train_rows <= max_lag:
                    min_window_days = (max_lag + self.horizon) // self.n_per_day + 1
                    raise ValueError(
                        f"{self.log_prefix} window_length ({self.args.window_length}) too small for lags: "
                        f"sliding-window train rows = {min_train_rows}, but max(lags) = {max_lag}. "
                        f"Lag features would be all-NaN. To keep at least one valid lag row, "
                        f"need window_length >= {min_window_days}."
                    )
        # ------------------------------
        # 预测增强策略(v1)组合校验:不支持的模式必须显式拒绝,避免裸奔崩溃或静默错配
        # ------------------------------
        if self.resolved_strategy.spec.rollout == RolloutFamily.BLEND:
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
            if decomposition_method != "none":
                raise ValueError(
                    f"{self.log_prefix} USBR/MSBR blend + target decomposition is not supported: "
                    f"ridge_stacking component predictions would remain in residual space. "
                    f"Keep decomposition_method=none."
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
        target_transform = getattr(self, "target_transform", None)
        if target_transform is not None:
            target_transform.attach_fitted_target_scaler(target_scaler)
        # 多变量递归辅助预测器包装（MSMR/MSMDR + endogenous_backfill_strategy=auxiliary）
        if df_history is not None and endogenous_features_with_target is not None:
            from models.AuxiliaryForecaster import maybe_build_auxiliary_bundle
            model = maybe_build_auxiliary_bundle(
                self.args, model, df_history,
                endogenous_features_with_target, self.args.target, self.log_prefix,
            )
        # 模型保存
        if mode == "forecast":
            model_trainer.model_save(
                model,
                feature_scaler=scaler,
                target_transform=getattr(self, "target_transform", None),
                selected_features=selected_features,
                input_schema={
                    "columns": list(getattr(scaler, "training_columns", ()) or ()),
                },
            )
        logger.info(f"{self.log_prefix} Model Training runtime: {time.perf_counter() - train_start:.3f}s")

        return model, scaler, target_scaler, selected_features

    def test(self,
             df_history,
             df_date_history,
             df_weather_history,
             df_weather_backtest,
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
        if self.horizon_mode == "calendar_month":
            calendar_folds = Tester._build_calendar_month_folds(
                df_history,
                train_window_len=self.train_window_len,
            )
            self.n_windows = len(calendar_folds)
            self.args.n_windows = self.n_windows
            window_specs = [
                {
                    "window": fold["window"],
                    "horizon": fold["horizon"],
                    "window_len": self.train_window_len + fold["horizon"],
                    "split_indices": {
                        "train_start": fold["train_start"],
                        "train_end": fold["train_end"],
                        "test_start": fold["test_start"],
                        "test_end": fold["test_end"],
                    },
                }
                for fold in calendar_folds[::window_stride]
            ]
        else:
            window_specs = [
                {
                    "window": window,
                    "horizon": self.horizon,
                    "window_len": self.window_len,
                    "split_indices": None,
                }
                for window in range(1, int(self.n_windows + 1), window_stride)
            ]
        max_test_windows = getattr(self.args, "max_test_windows", None)
        if max_test_windows is not None:
            max_test_windows = max(1, int(max_test_windows))
            window_specs = window_specs[:max_test_windows]
        logger.info(
            f"{self.log_prefix} Testing windows selected: "
            f"{[(spec['window'], spec['horizon']) for spec in window_specs]}"
        )
        
        window_workers = int(getattr(self.args, "window_parallel_workers", 1) or 1)
        if (
            self.resolved_strategy.spec.input_scope == InputScope.ALL_ENDOGENOUS
            and self.resolved_strategy.spec.rollout == RolloutFamily.DIRECT
            and window_workers > 1
        ):
            logger.warning(
                f"{self.log_prefix} pred_method=multivariate-single-multistep-direct with "
                f"window_parallel_workers={window_workers} will force multi_output_n_jobs=1 "
                f"and model_thread_count=1 during testing. This usually slows msmd testing; "
                f"prefer window_parallel_workers=1."
            )
        payload_args = copy.deepcopy(self.args)
        apply_window_parallel_budget(payload_args, window_workers)
        if self.resolved_strategy.spec.rollout == RolloutFamily.BLEND:
            payload_args.resolved_blend_weights = BlendWeights.for_backtest(payload_args)
        payloads = [
            {
                "args": payload_args,
                "force_single_thread_env": window_workers > 1,
                "log_prefix": self.log_prefix,
                "horizon": spec["horizon"],
                "window_len": spec["window_len"],
                "window": spec["window"],
                "split_indices": spec["split_indices"],
                "df_history": df_history,
                "df_date_history": df_date_history,
                "df_weather_history": df_weather_history,
                "df_weather_backtest": df_weather_backtest,
                "df_custom_history": df_custom_history,
                "endogenous_features_with_target": endogenous_features_with_target,
                "target_feature": target_feature,
                "categorical_features": categorical_features,
                "train_start_time": self.train_start_time,
                "train_end_time": self.train_end_time,
            }
            for spec in window_specs
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
            test_scores_df_median = test_scores_df.drop(columns=["time_range"]).median(numeric_only=True)
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
        Tester.test_results_save(self.args, self.log_prefix, test_scores_df, cv_plot_df, train_outlier_report, window_results)
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
                 df_custom_future=None,
                 _panel_child: bool = False,
                 _save_results: bool = True):
        """
        模型预测
        """
        forecast_start = time.perf_counter()
        if bool(getattr(self.args, "enable_global_training", False)) and not _panel_child:
            series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))

            def execute_one(series_slice: PanelSeriesSlice) -> pd.DataFrame:
                return self.forecast(
                    model=model,
                    scaler_forecasting=scaler_forecasting,
                    target_scaler_forecasting=target_scaler_forecasting,
                    df_history=series_slice.history,
                    df_future=series_slice.future,
                    df_date_future=df_date_future,
                    df_weather_future=df_weather_future,
                    df_custom_future=df_custom_future,
                    endogenous_features_with_target=endogenous_features_with_target,
                    target_feature=target_feature,
                    target_output_features=target_output_features,
                    categorical_features=categorical_features,
                    selected_features=selected_features,
                    _panel_child=True,
                    _save_results=False,
                )

            panel_prediction = execute_panel(
                df_history,
                df_future,
                series_id_col=series_id_col,
                horizon=self.horizon,
                execute_one=execute_one,
            )
            if _save_results:
                panel_history = df_history[
                    [series_id_col, "time", target_feature]
                ].rename(columns={target_feature: "y"})
                self._last_panel_predictor.forecast_results_save(
                    panel_history,
                    panel_prediction,
                    self.n_per_day,
                )
            return panel_prediction
        # 未来数据复制
        df_future_prediction = df_future.copy()
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
            target_decomposer=getattr(self, "target_decomposer", None),
            target_transform=getattr(self, "target_transform", None),
            log_prefix=self.log_prefix,
        )
        self._last_panel_predictor = predictor
        forecast_result = predictor._predict_by_method()
        if isinstance(forecast_result, ForecastDistribution):
            Y_pred = forecast_result.point
            output_quantile_grid = forecast_result.quantile_grid
            quantile_outputs = {
                level: forecast_result.quantile_values[:, index]
                for index, level in enumerate(forecast_result.quantile_grid.levels)
            }
        else:
            Y_pred = forecast_result
            quantile_outputs = getattr(predictor, "quantile_outputs", None)
            output_quantile_grid = (
                QuantileGrid(tuple(sorted(quantile_outputs)), point_level=0.5)
                if quantile_outputs
                else None
            )
        # ------------------------------
        # 模型预测结果收集和保存
        # ------------------------------
        logger.info(f"{self.log_prefix} {'=' * 87}")
        logger.info(f"{self.log_prefix} Model Forecasting result save...")
        logger.info(f"{self.log_prefix} {'=' * 87}")
        # Forecaster 已通过共享 TargetTransformPipeline 恢复到 target space。
        Y_pred = np.asarray(Y_pred, dtype=float).reshape(-1)
        if len(Y_pred) != len(df_future_prediction):
            raise ValueError(
                f"{self.log_prefix} forecast length mismatch after target restore: "
                f"prediction={len(Y_pred)}, future={len(df_future_prediction)}"
            )
        df_future_prediction["predict_value"] = Y_pred
        # 分位数预测结果（若启用）
        if quantile_outputs:
            for q, q_pred in sorted(quantile_outputs.items(), key=lambda x: float(x[0])):
                q_col = output_quantile_grid.column_name(float(q))
                q_arr = np.asarray(q_pred, dtype=float).reshape(-1)
                if len(q_arr) != len(df_future_prediction):
                    raise ValueError(
                        f"{self.log_prefix} quantile q={float(q):g} forecast length mismatch "
                        f"after target restore: prediction={len(q_arr)}, "
                        f"future={len(df_future_prediction)}"
                    )
                df_future_prediction[q_col] = q_arr
        # 先以 q50 为锚点修复 crossing；CQR 随后通过 as-of selector 生成独立 PI。
        calibration_kwargs = calibration_runtime_kwargs(self.probabilistic_spec)
        alpha = float(calibration_kwargs.get("alpha", 0.1))
        min_scores = int(calibration_kwargs.get("min_scores", 30))
        df_future_prediction, _ = finalize_quantile_forecast(
            df_future_prediction,
            monotone_enabled=bool(getattr(self.args, "quantile_monotone", False)),
            conformal_scores=None,
            alpha=alpha,
            min_scores=min_scores,
        )
        if bool(calibration_kwargs["enable_cqr"]):
            interval = self.probabilistic_spec.calibration_interval
            if interval is None or output_quantile_grid is None:
                raise RuntimeError("CQR requires a calibration interval and quantile grid")
            q_low_col = output_quantile_grid.column_name(interval.lower_quantile)
            q_high_col = output_quantile_grid.column_name(interval.upper_quantile)
            cv_path = self.args.test_results_dir.joinpath("cv_plot_df.csv")
            if cv_path.exists():
                cv_df = pd.read_csv(cv_path)
                forecast_origin = cast(
                    pd.Timestamp,
                    pd.Timestamp(df_future_prediction["time"].iloc[0]),
                )
                calibration_windows = int(calibration_kwargs["calibration_windows"])
                calibration_result = calibrate_final_from_cv(
                    cv_df,
                    q_low=df_future_prediction[q_low_col].to_numpy(dtype=float),
                    q_high=df_future_prediction[q_high_col].to_numpy(dtype=float),
                    forecast_origin=forecast_origin,
                    freq=str(getattr(self.args, "freq", "1D")),
                    calibration_windows=calibration_windows,
                    min_windows=int(calibration_kwargs["min_windows"]),
                    min_scores=min_scores,
                    alpha=alpha,
                    label_availability_delay_steps=int(
                        calibration_kwargs["label_availability_delay_steps"]
                    ),
                    interval_name=str(calibration_kwargs["interval_name"]),
                    lower_quantile=float(calibration_kwargs["lower_quantile"]),
                    upper_quantile=float(calibration_kwargs["upper_quantile"]),
                    allow_interval_shrink=bool(
                        calibration_kwargs["allow_interval_shrink"]
                    ),
                )
                append_final_calibration_report(
                    self.args.test_results_dir.joinpath("calibration_report.csv"),
                    result=calibration_result,
                    forecast_origin=forecast_origin,
                    interval_name=str(calibration_kwargs["interval_name"]),
                    target_coverage=1.0 - alpha,
                    calibration_windows=calibration_windows,
                    allow_interval_shrink=bool(
                        calibration_kwargs["allow_interval_shrink"]
                    ),
                )
                if (
                    calibration_result.status == "applied"
                    and calibration_result.lower is not None
                    and calibration_result.upper is not None
                    and calibration_result.correction is not None
                ):
                    df_future_prediction = attach_cqr_interval_columns(
                        df_future_prediction,
                        calibration_result.lower,
                        calibration_result.upper,
                        target_coverage=1.0 - alpha,
                    )
                    logger.info(
                        f"{self.log_prefix} Conformal CQR calibrated: "
                        f"correction={calibration_result.correction:.4f}, "
                        f"target coverage={1 - alpha:.2f}, "
                        f"n_windows={calibration_result.selected_windows}, "
                        f"n_scores={calibration_result.selected_scores}"
                    )
                else:
                    logger.warning(
                        f"{self.log_prefix} Conformal CQR skipped: "
                        f"status={calibration_result.status}, reason={calibration_result.reason}"
                    )
            elif not cv_path.exists():
                logger.warning(
                    f"{self.log_prefix} Conformal CQR skipped: cv_plot_df.csv not found "
                    f"(need is_testing=True to produce calibration scores)"
                )

        probability_cols = [
            c
            for c in df_future_prediction.columns
            if c.startswith("predict_q") or c.startswith("predict_pi")
        ]
        identity_columns = []
        if bool(getattr(self.args, "enable_global_training", False)):
            series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
            if series_id_col not in df_future_prediction.columns:
                raise ValueError(
                    f"{self.log_prefix} panel forecast output missing '{series_id_col}'."
                )
            identity_columns = [series_id_col]
        if probability_cols:
            df_future_prediction = df_future_prediction[
                identity_columns + ["time", "predict_value"] + probability_cols
            ]
        else:
            df_future_prediction = df_future_prediction[
                identity_columns + ["time", "predict_value"]
            ]
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
        if _save_results:
            predictor.forecast_results_save(
                df_history_for_plot,
                df_future_prediction,
                self.n_per_day,
            )
            logger.info(
                f"{self.log_prefix} Model Forecasting result saved in: "
                f"{self.args.pred_results_dir}"
            )
        logger.info(f"{self.log_prefix} Model Forecasting runtime: {time.perf_counter() - forecast_start:.3f}s")

        return df_future_prediction

    def _learn_blend_weights(self) -> BlendWeights:
        """从 CV 分量预测学习权重；CSV 仅作诊断，产物中的权重才是权威。"""
        cv_path = self.args.test_results_dir.joinpath("cv_plot_df.csv")
        if not cv_path.exists():
            raise ValueError(
                f"{self.log_prefix} ridge_stacking requires cv_plot_df.csv; "
                "enable testing before forecasting."
            )
        cv = pd.read_csv(cv_path)
        needed = ["Y_trues", "blend_direct_pred", "blend_recursive_pred"]
        if not all(c in cv.columns for c in needed):
            missing = [column for column in needed if column not in cv.columns]
            raise ValueError(
                f"{self.log_prefix} ridge_stacking CV artifact missing columns: {missing}."
            )
        cv_clean = cv.dropna(subset=needed)
        if len(cv_clean) < 10:
            raise ValueError(
                f"{self.log_prefix} ridge_stacking requires at least 10 valid rows; "
                f"got {len(cv_clean)}."
            )
        # 与 conformal 校准一致：只用最近 N 个窗口的数据学权重（分布更贴合当前预测任务）
        n_cal_windows = int(getattr(self.args, "blend_weight_windows", 5))
        if "window" in cv_clean.columns:
            recent_windows = sorted(cv_clean["window"].unique())[-n_cal_windows:]
            cv_clean = cv_clean[cv_clean["window"].isin(recent_windows)]
        if len(cv_clean) < 10:
            raise ValueError(
                f"{self.log_prefix} ridge_stacking requires at least 10 rows in recent "
                f"{n_cal_windows} windows; got {len(cv_clean)}."
            )
        from sklearn.linear_model import Ridge
        X_stack = cv_clean[["blend_direct_pred", "blend_recursive_pred"]].values
        y_stack = cv_clean["Y_trues"].values
        # 无截距凸组合：截距在归一化时会被丢弃，导致权重偏离最优（系统偏差大时尤其明显）
        ridge = Ridge(alpha=1.0, positive=True, fit_intercept=False).fit(X_stack, y_stack)
        w = ridge.coef_
        total = float(w.sum())
        if total <= 0:
            raise ValueError("ridge_stacking produced non-positive total weight.")
        w_norm = w / total
        weights = BlendWeights(
            direct=float(w_norm[0]),
            recursive=float(w_norm[1]),
            strategy="ridge_stacking",
            calibration_windows=n_cal_windows,
        )
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
        return weights

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
        df_weather_backtest = dataloader.process_weather_backtest_data(input_data=input_data)
        # ------------------------------
        # 目标分解（可选）：最终 forecast 在全部已知历史上拟合；滑窗测试会在
        # 各自训练段内重新拟合独立分解器，禁止未来窗口参与预处理。
        # ------------------------------
        self.df_history_levels = df_history.copy()
        assert isinstance(df_history, pd.DataFrame)
        self.target_transform = TargetTransformPipeline.from_args(self.args)
        df_history = self.target_transform.fit_transform_history(
            df_history,
            time_col="time",
            target_col="y",
        )
        # 迁移期兼容属性；新主链只通过 target_transform restore。
        self.target_calendar_normalizer = self.target_transform.calendar_normalizer
        self.target_decomposer = self.target_transform.decomposition
        if self.target_decomposer.enabled:
            logger.info(
                f"{self.log_prefix} 目标分解: 启用 "
                f"(method={self.target_decomposer.method})"
            )
        else:
            logger.info(f"{self.log_prefix} 目标分解: 禁用")
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
                df_history=self.df_history_levels,
                df_date_history=df_date_history,
                df_weather_history=df_weather_history,
                df_weather_backtest=df_weather_backtest,
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

            if (
                self.resolved_strategy.spec.rollout == RolloutFamily.BLEND
                and str(getattr(self.args, "blend_weight_strategy", "fixed")).lower()
                == "ridge_stacking"
            ):
                learned_weights = self._learn_blend_weights()
                self.args.resolved_blend_weights = learned_weights

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
