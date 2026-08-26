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
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

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
from features.TargetTransformation import TargetTransformPipeline
from models.ModelTraining import Trainer
from models.ModelForecasting import Forecaster
from models.multistep.panel import (
    PanelSeriesSlice,
    execute_panel,
    split_panel_window,
)
from data_provider.data_loader import materialize_custom_future_sources
from data_provider.outlier_handling import (
    empty_train_outlier_report,
    handle_train_outliers,
)
from utils.eval_mask import build_eval_mask
from utils.quantile import monotonize_quantile_columns
from utils.conformal import compute_nonconformity_scores
from probabilistic.evaluation import write_probabilistic_artifacts
from probabilistic.spec import calibration_runtime_kwargs
from probabilistic.postprocessing import repair_quantile_crossing
from probabilistic.types import ForecastDistribution, QuantileGrid
from utils.weather_contract import validate_weather_coverage
from utils.log_util import logger
from utils.exogenous_contract import select_asof_rows

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _load_plot_overlay_df(args, log_prefix: str) -> Optional[pd.DataFrame]:
    """加载测试可视化叠加参考序列（time + plot_overlay_col 两列）。

    plot_overlay_path 相对 data_dir 解析（绝对路径直接用）；路径或列名任一为空、
    文件不存在、缺列时返回 None（叠加关闭，不影响原有绘图）。
    """
    overlay_path_raw = str(getattr(args, "plot_overlay_path", "") or "").strip()
    overlay_col = str(getattr(args, "plot_overlay_col", "") or "").strip()
    if not overlay_path_raw or not overlay_col:
        return None
    overlay_path = Path(overlay_path_raw)
    if not overlay_path.is_absolute():
        overlay_path = Path(args.data_dir) / overlay_path
    if not overlay_path.exists():
        logger.warning(f"{log_prefix} Plot overlay file not found, overlay skipped: {overlay_path}")
        return None
    try:
        overlay_df = pd.read_csv(overlay_path)
    except Exception as exc:
        logger.warning(f"{log_prefix} Plot overlay file read failed, overlay skipped: {exc}")
        return None
    if "time" not in overlay_df.columns or overlay_col not in overlay_df.columns:
        logger.warning(
            f"{log_prefix} Plot overlay file missing 'time'/'{overlay_col}' column(s), overlay skipped: {overlay_path}"
        )
        return None
    overlay_df["time"] = pd.to_datetime(overlay_df["time"])
    overlay_df[overlay_col] = pd.to_numeric(overlay_df[overlay_col], errors="coerce")
    overlay_df = (
        overlay_df.loc[overlay_df[overlay_col].notna(), ["time", overlay_col]]
        .drop_duplicates(subset="time", keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    logger.info(f"{log_prefix} Plot overlay loaded: {overlay_path} column='{overlay_col}' rows={len(overlay_df)}")
    return overlay_df


class Tester:
    @staticmethod
    def _attach_window_target_scaler(
        args,
        target_output_features,
        target_scaler,
        target_transform,
    ):
        if target_scaler is None:
            return list(target_output_features)
        try:
            pred_target_columns = target_scaler.get_prediction_target_columns(
                args.pred_method,
                target_output_features,
                direct_strategy=str(getattr(args, "direct_strategy", "multioutput")),
            )
        except TypeError:
            # 测试/旧 adapter 的兼容签名；生产 TargetScaler 支持 direct_strategy。
            pred_target_columns = target_scaler.get_prediction_target_columns(
                args.pred_method,
                target_output_features,
            )
        target_transform.attach_fitted_target_scaler(
            target_scaler,
            target_columns=pred_target_columns,
        )
        return pred_target_columns

    
    def __init__(self, args, log_prefix: str, horizon: int, window_len: int):
        self.args = args
        self.log_prefix = log_prefix
        self.horizon = horizon
        self.window_len = window_len

    @staticmethod
    def _resolve_window_weather_future(
        args,
        df_history_test: pd.DataFrame,
        df_weather_history: Optional[pd.DataFrame],
        df_weather_backtest: Optional[pd.DataFrame],
        fold_origin: pd.Timestamp,
        log_prefix: str,
    ) -> Optional[pd.DataFrame]:
        """选择滑窗预测可用的气象信息集。

        严格模式必须使用独立的 ex-ante backtest 文件，禁止把含测试月整月
        实测值的 weather_history 当成未来气象。旧配置未启用严格模式时保持
        原行为，避免影响其它场景。
        """
        if not bool(getattr(args, "enable_weather_features", False)):
            return None
        if not bool(getattr(args, "strict_weather_information_set", False)):
            return df_weather_history
        assert df_weather_backtest is not None
        return select_asof_rows(
            df_weather_backtest,
            expected_times=df_history_test["time"],
            forecast_origin=fold_origin,
            ts_col=getattr(args, "weather_ts_feat", "ts"),
            available_at_col="available_at",
            label="Backtest weather",
        )

    @staticmethod
    def _window_test(payload):
        """
        单个滑动窗口测试任务
        """
        # 窗口并行（进程池）时每个 worker 单线程纪律：payload 的 n_jobs/thread_count
        # 已被强制为 1，但 HistGB 走 OpenMP 无 n_jobs 参数，需 OMP_NUM_THREADS=1 兜底，
        # 避免 window_workers × OMP 线程超额订阅（串行路径不带此标记，不受影响）。
        if payload.get("force_single_thread_env"):
            os.environ["OMP_NUM_THREADS"] = "1"
        args = payload["args"]
        series_id_col = str(getattr(args, "series_id_feature", "series_id"))
        log_prefix = payload["log_prefix"]
        horizon = payload["horizon"]
        window_len = payload["window_len"]
        window = payload["window"]
        train_outlier_report = empty_train_outlier_report()

        # 滑窗数据分割：先切原始历史，再在窗口内构造训练标签，避免 Direct 标签跨入测试期
        if bool(getattr(args, "enable_global_training", False)):
            series_id_col = str(getattr(args, "series_id_feature", "series_id"))
            split_result = split_panel_window(
                payload["df_history"],
                series_id_col=series_id_col,
                window=window,
                horizon=horizon,
                window_len=window_len,
                incomplete_policy=str(
                    getattr(args, "global_incomplete_series_policy", "raise") or "raise"
                ),
            )
        else:
            split_result = Tester._evaluate_split(
                payload["df_history"],
                window,
                horizon=horizon,
                window_len=window_len,
                log_prefix=log_prefix,
                split_indices=payload.get("split_indices"),
            )
        if split_result is None:
            return {
                "window": window,
                "test_scores_df": None,
                "cv_plot_df": None,
                "train_outlier_report": train_outlier_report,
                "residual_diag_row": None,
            }
        df_history_train, df_history_test = split_result
        if bool(getattr(args, "enable_global_training", False)):
            cleaned_parts = []
            report_parts = []
            for series_id, series_frame in df_history_train.groupby(
                series_id_col,
                sort=False,
                observed=True,
            ):
                cleaned, report = handle_train_outliers(
                    args=args,
                    df_history_train=series_frame.copy(),
                    target_feature=payload["target_feature"],
                    window=window,
                    log_prefix=f"{log_prefix}[series={series_id}]",
                )
                cleaned_parts.append(cleaned)
                if report is not None and not report.empty:
                    report = report.copy()
                    report[series_id_col] = series_id
                    report_parts.append(report)
            df_history_train = pd.concat(cleaned_parts, ignore_index=True)
            train_outlier_report = (
                pd.concat(report_parts, ignore_index=True)
                if report_parts
                else empty_train_outlier_report()
            )
        else:
            df_history_train, train_outlier_report = handle_train_outliers(
                args=args,
                df_history_train=df_history_train,
                target_feature=payload["target_feature"],
                window=window,
                log_prefix=log_prefix,
            )
        target_transform = TargetTransformPipeline.from_args(args)
        df_history_train = target_transform.fit_transform_history(
            df_history_train,
            time_col="time",
            target_col=payload["target_feature"],
        )
        # 每个滑窗只在训练段拟合目标分解器，禁止测试段及更晚数据参与预处理。
        residual_diag_row = None
        target_decomposer = target_transform.decomposition
        if target_decomposer.enabled:
            # 分解诊断报告（按窗口序号命名，不互相覆盖；无输出目录时跳过）
            from decomposition.diagnostics import write_diagnostics_report

            diag_dir = getattr(args, "test_results_dir", None)
            if diag_dir is not None:
                write_diagnostics_report(
                    target_decomposer,
                    df_history_train,
                    time_col="time",
                    target_col=payload["target_feature"],
                    output_dir=diag_dir,
                    suffix=f"_win{window}",
                )
                # B1 残差频谱诊断：fit_transform 后的 y 即 residual，随结果返回供主进程汇总
                from decomposition.residual_diagnostics import diagnose_window_residual

                residual = df_history_train[payload["target_feature"]].to_numpy(dtype=float)
                residual_diag_row = diagnose_window_residual(residual, window_idx=window)
        build_result = Tester._build_window_train_xy(
            args=args,
            log_prefix=log_prefix,
            df_history_train=df_history_train,
            df_date_history=payload["df_date_history"],
            df_weather_history=payload["df_weather_history"],
            df_custom_history=payload.get("df_custom_history"),
            endogenous_features_with_target=payload["endogenous_features_with_target"],
            target_feature=payload["target_feature"],
            horizon=horizon,
        )
        if build_result is None:
            return {
                "window": window,
                "test_scores_df": None,
                "cv_plot_df": None,
                "train_outlier_report": train_outlier_report,
                "residual_diag_row": None,
            }
        X_train, Y_train, target_output_features, categorical_features = build_result
        # 窗口目标特征处理
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        # 测试标签始终保留原始电平，不经过分解，避免测试信息进入分解器。
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
        # calendar_month 每个测试 fold 的天数可能不同于全局 forecast horizon。
        # 测试替身可能不实现该可选 setter，保持旧 Trainer 接口兼容。
        set_train_horizon = getattr(model_trainer, "set_train_horizon", None)
        if callable(set_train_horizon):
            set_train_horizon(horizon)
        model, scaler_testing, target_scaler_testing, selected_features = model_trainer.train(
            X_train=X_train,
            Y_train=Y_train,
            feature_scaler=scaler,
            target_scaler=target_scaler,
            categorical_features=categorical_features,
        )
        Tester._attach_window_target_scaler(
            args=args,
            target_output_features=target_output_features,
            target_scaler=target_scaler_testing,
            target_transform=target_transform,
        )
        # 多变量递归辅助预测器包装（滑窗测试段同样需要 aux 轨迹回填）
        from models.AuxiliaryForecaster import maybe_build_auxiliary_bundle
        model = maybe_build_auxiliary_bundle(
            args, model, df_history_train,
            payload["endogenous_features_with_target"],
            payload["target_feature"], log_prefix,
        )
        # ------------------------------
        # 窗口预测
        # ------------------------------
        df_future_for_test = Tester._build_test_future_frame(
            df_history_test,
            series_id_col=(
                series_id_col
                if bool(getattr(args, "enable_global_training", False))
                else None
            ),
        )
        weather_test_frame = (
            df_history_test.drop_duplicates(subset="time", keep="last")
            if bool(getattr(args, "enable_global_training", False))
            else df_history_test
        )
        df_weather_future_for_test = Tester._resolve_window_weather_future(
            args=args,
            df_history_test=weather_test_frame,
            df_weather_history=payload["df_weather_history"],
            df_weather_backtest=payload.get("df_weather_backtest"),
            fold_origin=df_history_train["time"].max(),
            log_prefix=log_prefix,
        )
        df_custom_future_for_test = materialize_custom_future_sources(
            custom_history=payload.get("df_custom_history"),
            # explicit 策略维持既有 CV 语义：历史归档按测试期时间戳对齐；
            # freeze 策略会忽略 cutoff 之后的行并冻结训练末状态。
            custom_future=payload.get("df_custom_history"),
            future_times=pd.unique(df_future_for_test["time"]),
            cutoff=df_history_train["time"].max(),
        )
        predictor = None
        if bool(getattr(args, "enable_global_training", False)):
            direct_components: list[np.ndarray] = []
            recursive_components: list[np.ndarray] = []
            quantile_column_levels: dict[str, float] = {}

            def execute_one(series_slice: PanelSeriesSlice) -> pd.DataFrame:
                nonlocal predictor
                predictor = Forecaster(
                    args=args,
                    horizon=horizon,
                    model=model,
                    feature_scaler=scaler_testing,
                    target_scaler=target_scaler_testing,
                    df_history=series_slice.history,
                    df_future=series_slice.future,
                    df_date_future=payload["df_date_history"],
                    df_weather_future=df_weather_future_for_test,
                    df_custom_future=df_custom_future_for_test,
                    endogenous_features=payload["endogenous_features_with_target"],
                    target_feature=payload["target_feature"],
                    target_output_features=target_output_features,
                    categorical_features=categorical_features,
                    selected_features=selected_features,
                    target_decomposer=target_decomposer,
                    target_transform=target_transform,
                    log_prefix=f"{log_prefix}[series={series_slice.series_id}]",
                )
                result = predictor._predict_by_method()
                output = series_slice.future[[series_id_col, "time"]].copy()
                if isinstance(result, ForecastDistribution):
                    output["predict_value"] = result.point
                    for index, level in enumerate(result.quantile_grid.levels):
                        column = result.quantile_grid.column_name(level)
                        quantile_column_levels[column] = float(level)
                        output[column] = (
                            result.quantile_values[:, index]
                        )
                else:
                    output["predict_value"] = np.asarray(result).reshape(-1)
                    for level, values in (predictor.quantile_outputs or {}).items():
                        column = QuantileGrid((float(level),)).column_name(float(level))
                        quantile_column_levels[column] = float(level)
                        output[column] = np.asarray(values).reshape(-1)
                if predictor.blend_direct_pred is not None:
                    direct_components.append(
                        np.asarray(predictor.blend_direct_pred).reshape(-1)
                    )
                    recursive_components.append(
                        np.asarray(predictor.blend_recursive_pred).reshape(-1)
                    )
                return output

            panel_output = execute_panel(
                df_history_train,
                df_future_for_test,
                series_id_col=series_id_col,
                horizon=horizon,
                execute_one=execute_one,
            )
            y_pred = panel_output["predict_value"].to_numpy(dtype=float)
            quantile_columns = sorted(
                column
                for column in panel_output.columns
                if str(column).startswith("predict_q")
            )
            quantile_outputs = {
                quantile_column_levels[column]: panel_output[column].to_numpy(dtype=float)
                for column in quantile_columns
            }
            output_quantile_grid = (
                QuantileGrid(tuple(sorted(quantile_outputs)), point_level=0.5)
                if quantile_outputs
                else None
            )
            assert predictor is not None
            if direct_components:
                predictor.blend_direct_pred = np.concatenate(direct_components)
                predictor.blend_recursive_pred = np.concatenate(recursive_components)
        else:
            predictor = Forecaster(
                args=args,
                horizon=min(horizon, len(df_future_for_test)),
                model=model,
                feature_scaler=scaler_testing,
                target_scaler=target_scaler_testing,
                df_history=df_history_train,
                df_future=df_future_for_test,
                df_date_future=payload["df_date_history"],
                df_weather_future=df_weather_future_for_test,
                df_custom_future=df_custom_future_for_test,
                endogenous_features=payload["endogenous_features_with_target"],
                target_feature=payload["target_feature"],
                target_output_features=target_output_features,
                categorical_features=categorical_features,
                selected_features=selected_features,
                target_decomposer=target_decomposer,
                target_transform=target_transform,
                log_prefix=log_prefix,
            )
            forecast_result = predictor._predict_by_method()
            if isinstance(forecast_result, ForecastDistribution):
                y_pred = forecast_result.point
                output_quantile_grid = forecast_result.quantile_grid
                quantile_outputs = {
                    level: forecast_result.quantile_values[:, index]
                    for index, level in enumerate(forecast_result.quantile_grid.levels)
                }
            else:
                y_pred = forecast_result
                quantile_outputs = getattr(predictor, "quantile_outputs", None)
                output_quantile_grid = (
                    QuantileGrid(tuple(sorted(quantile_outputs)), point_level=0.5)
                    if quantile_outputs
                    else None
                )
        # ------------------------------
        # 模型滑窗预测结果收集
        # ------------------------------
        if len(y_pred) == 0:
            return {
                "window": window,
                "test_scores_df": None,
                "cv_plot_df": None,
                "train_outlier_report": train_outlier_report,
            }
        # Forecaster 已通过共享 TargetTransformPipeline 恢复到原始 target space；
        # 测试标签始终保留同一原始电平。
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y_test_for_eval = np.asarray(y_test_raw, dtype=float).reshape(-1)
        # 对齐预测结果与评估标签长度
        if len(y_pred) != len(y_test_for_eval):
            min_len = min(len(y_pred), len(y_test_for_eval))
            y_pred = np.asarray(y_pred)[:min_len]
            y_test_for_eval = np.asarray(y_test_for_eval)[:min_len]
        # 季节 naive 对照（昨日同时刻实际值），与评估标签对齐
        y_naive = None
        if not bool(getattr(args, "enable_global_training", False)):
            y_naive = Tester._build_seasonal_naive(
                df_history=payload["df_history"],
                window=window,
                horizon=horizon,
                window_len=window_len,
                target_feature=payload["target_feature"],
                n_per_day=int(getattr(args, "n_per_day", 1) or 1),
                split_indices=payload.get("split_indices"),
            )
        if y_naive is not None:
            y_naive = np.asarray(y_naive).reshape(-1)
            if len(y_naive) >= len(y_test_for_eval):
                y_naive = y_naive[: len(y_test_for_eval)]
            else:
                # 长度不足无法对齐，放弃本窗口 naive 对照
                y_naive = None
        # 测试集评价指标
        eval_scores_window = Tester._evaluate_score(
            y_test_for_eval,
            y_pred,
            window,
            df_history_test,
            log_prefix=log_prefix,
            mode=args.mode,
            percentile=args.percentile,
            min_value=args.min_value,
            max_value=args.max_value,
            y_naive=y_naive,
            horizon_mode=str(getattr(args, "horizon_mode", "fixed_steps")),
        )
        # 测试集预测数据
        cv_plot_df_window = Tester._evaluate_result(
            y_test_for_eval,
            y_pred,
            df_history_test,
            log_prefix=log_prefix,
            mode=args.mode,
            percentile=args.percentile,
            min_value=args.min_value,
            max_value=args.max_value,
        )
        if bool(getattr(args, "enable_global_training", False)):
            cv_plot_df_window.insert(
                0,
                series_id_col,
                df_history_test[series_id_col].to_numpy()[: len(cv_plot_df_window)],
            )
        # 分位数预测(若启用):补入 cv_plot,使回测也体现分位数区间
        if quantile_outputs:
            if output_quantile_grid is None:
                raise RuntimeError("quantile outputs require a QuantileGrid")
            n = len(cv_plot_df_window)
            for q, q_pred in sorted(quantile_outputs.items(), key=lambda x: float(x[0])):
                q_col = output_quantile_grid.column_name(float(q))
                q_values = np.asarray(q_pred).reshape(-1)
                if len(q_values) != n:
                    raise ValueError(
                        f"{log_prefix} quantile q={float(q):g} test prediction length mismatch: "
                        f"expected {n}, got {len(q_values)}"
                    )
                cv_plot_df_window[f"{q_col}_raw"] = q_values
                cv_plot_df_window[q_col] = q_values
            cv_plot_df_window = repair_quantile_crossing(
                cv_plot_df_window,
                enabled=bool(getattr(args, "quantile_monotone", False)),
                point_column="Y_preds",
            )
            # conformal score 记录（若启用）：逐点 nonconformity score，供 forecast 阶段 CQR 校准
            probabilistic_spec = getattr(args, "probabilistic_spec", None)
            if probabilistic_spec is not None:
                calibration_kwargs = calibration_runtime_kwargs(probabilistic_spec)
            else:
                calibration_kwargs = {
                    "enable_cqr": bool(
                        getattr(args, "enable_conformal_calibration", False)
                    )
                }
            if bool(calibration_kwargs["enable_cqr"]):
                if probabilistic_spec is not None:
                    interval = probabilistic_spec.calibration_interval
                    if interval is None:
                        raise ValueError("CQR is enabled without a calibration interval")
                    q_low_col = output_quantile_grid.column_name(
                        interval.lower_quantile
                    )
                    q_high_col = output_quantile_grid.column_name(
                        interval.upper_quantile
                    )
                else:
                    q_cols_sorted = [
                        output_quantile_grid.column_name(float(q))
                        for q in sorted(quantile_outputs.keys(), key=float)
                    ]
                    q_low_col, q_high_col = q_cols_sorted[0], q_cols_sorted[-1]
                q_low_pred = cv_plot_df_window[q_low_col].to_numpy(dtype=float)
                q_high_pred = cv_plot_df_window[q_high_col].to_numpy(dtype=float)
                # point/quantile 已经由同一变换栈恢复；score 直接在保存的
                # processed target-space boundaries 上计算，可逐点复算。
                cv_plot_df_window["conformal_score"] = compute_nonconformity_scores(
                    y_test_for_eval,
                    q_low_pred,
                    q_high_pred,
                )
        # blend 分预测记录（供 ridge_stacking 在 forecast 阶段学权重）
        if getattr(predictor, "blend_direct_pred", None) is not None:
            n_blend = len(cv_plot_df_window)
            cv_plot_df_window["blend_direct_pred"] = np.asarray(predictor.blend_direct_pred).reshape(-1)[:n_blend]
            cv_plot_df_window["blend_recursive_pred"] = np.asarray(predictor.blend_recursive_pred).reshape(-1)[:n_blend]

        # 注入 naive 对照（若可对齐），供概率 horizon 聚合输出 point/naive 指标
        if y_naive is not None:
            n_rows = len(cv_plot_df_window)
            if len(y_naive) >= n_rows:
                cv_plot_df_window["Y_naive"] = np.asarray(y_naive[:n_rows], dtype=float)

        # 注入窗口编号，供后续 per-window 绘图使用
        if not cv_plot_df_window.empty:
            cv_plot_df_window["window"] = window

        return {
            "window": window,
            "test_scores_df": eval_scores_window,
            "cv_plot_df": cv_plot_df_window,
            "train_outlier_report": train_outlier_report,
            "residual_diag_row": residual_diag_row,
        }

    # ------------------------------
    # Model sliding window testing
    # ------------------------------
    @staticmethod
    def _build_calendar_month_folds(
        df_history: pd.DataFrame,
        train_window_len: int,
    ) -> List[Dict[str, Any]]:
        """按完整自然月构造由近到远的滑窗，并固定每窗训练行数。"""
        if train_window_len <= 0:
            raise ValueError("train_window_len must be > 0 for calendar_month folds.")
        if "time" not in df_history.columns:
            raise ValueError("calendar_month folds require a time column.")

        times = pd.DatetimeIndex(pd.to_datetime(df_history["time"]))
        if len(times) < 2:
            return []
        if not times.is_monotonic_increasing or times.has_duplicates:
            raise ValueError("calendar_month folds require strictly increasing unique timestamps.")
        expected_step = pd.Timedelta(days=1)
        if any(diff != expected_step for diff in times[1:] - times[:-1]):
            raise ValueError("calendar_month folds currently require a complete regular 1D index.")

        last_time = pd.Timestamp(int(times.asi8[-1]))
        current_end = cast(pd.Timestamp, last_time + pd.offsets.MonthBegin(1))
        folds: List[Dict[str, Any]] = []
        while True:
            test_end_time = cast(pd.Timestamp, pd.Timestamp(current_end))
            test_start_time = cast(
                pd.Timestamp,
                (test_end_time - pd.offsets.MonthBegin(1)).normalize(),
            )
            test_start = int(times.asi8.searchsorted(test_start_time.value, side="left"))
            test_end = int(times.asi8.searchsorted(test_end_time.value, side="left"))
            expected_horizon = int(test_start_time.days_in_month)

            if test_start >= len(times) or times[test_start] != test_start_time:
                break
            if test_end - test_start != expected_horizon:
                raise ValueError(
                    f"calendar_month fold {test_start_time:%Y-%m} is incomplete: "
                    f"expected {expected_horizon} daily rows, got {test_end - test_start}."
                )

            train_end = test_start
            train_start = train_end - int(train_window_len)
            if train_start < 0:
                break
            folds.append(
                {
                    "window": len(folds) + 1,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "horizon": expected_horizon,
                    "train_start_time": times[train_start],
                    "train_end_time": test_start_time,
                    "test_start_time": test_start_time,
                    "test_end_time": test_end_time,
                }
            )
            current_end = test_start_time
        return folds

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
        split_indices: Optional[Dict[str, int]] = None,
    ):
        """
        训练、测试数据集分割
        """
        # 滑窗数据分割索引
        total_data_points = len(df_history)
        if split_indices is None:
            train_start, train_end, test_start, test_end_inclusive = Tester._evaluate_split_index(
                window, total_data_points, horizon, window_len
            )
            test_end = test_end_inclusive + 1
        else:
            train_start = int(split_indices["train_start"])
            train_end = int(split_indices["train_end"])
            test_start = int(split_indices["test_start"])
            test_end = int(split_indices["test_end"])
        logger.info(f"{log_prefix} split indexes:: [train_start:train_end]: [{train_start}:{train_end}]")
        logger.info(f"{log_prefix} split indexes:: [test_start:test_end]: [{test_start}:{test_end}]")
        if train_start >= train_end or test_start >= test_end or train_start < 0 or test_end > total_data_points:
            logger.warning(
                f"{log_prefix} Insufficient data for window {window} "
                f"(train_start={train_start}, train_end={train_end}, "
                f"test_start={test_start}, test_end={test_end}). "
                f"Skipping this window."
            )
            return None

        # 滑窗数据分割
        df_history_train = df_history.iloc[train_start:train_end]
        df_history_test = df_history.iloc[test_start:test_end]
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
        df_custom_history=None,
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
            df_custom_history=df_custom_history,
            df_custom_future=None,
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
    def _build_test_future_frame(
        df_history_test: pd.DataFrame,
        series_id_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        测试预测阶段只能看到未来时间模板，不透传测试期真实 y。
        """
        columns = ["time"]
        if series_id_col is not None:
            if series_id_col not in df_history_test.columns:
                raise ValueError(
                    f"Panel test frame missing series ID column '{series_id_col}'."
                )
            columns.insert(0, series_id_col)
        return pd.DataFrame(df_history_test.loc[:, columns]).copy()

    @staticmethod
    def _build_seasonal_naive(
        df_history: pd.DataFrame,
        window: int,
        horizon: int,
        window_len: int,
        target_feature: str,
        n_per_day: int,
        split_indices: Optional[Dict[str, int]] = None,
    ):
        """
        季节 naive 对照序列：测试期第 i 点的 naive 值 = 该点 n_per_day 步前
        （昨日同时刻）的实际值，全部取自预测原点之前/之中的已知实际数据。
        历史不足一天时返回 None（naive 指标列记 NaN）。
        """
        if split_indices is None:
            _, _, test_start, test_end_inclusive = Tester._evaluate_split_index(
                window, len(df_history), horizon, window_len
            )
            test_end = test_end_inclusive + 1
        else:
            test_start = int(split_indices["test_start"])
            test_end = int(split_indices["test_end"])
        naive_start = test_start - n_per_day
        if naive_start < 0:
            return None
        y_naive = df_history[target_feature].iloc[naive_start: test_end - n_per_day].to_numpy()
        return y_naive

    @staticmethod
    def _evaluate_score(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        window: int,
        df_history_test: pd.DataFrame,
        log_prefix: str,
        mode: str = "percentile",
        percentile: float = 5.0,
        min_value: float = None,
        max_value: float = None,
        y_naive: Optional[np.ndarray] = None,
        horizon_mode: str = "fixed_steps",
    ):
        """
        模型评估
        计算模型的性能指标
        """
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        mape_meta = build_eval_mask(
            y_test, mode=mode, percentile=percentile, min_value=min_value, max_value=max_value
        )
        valid_mask = mape_meta["valid_mask"]
        if mape_meta["valid_points"] > 0:
            mape_value = mean_absolute_percentage_error(y_test[valid_mask], y_pred[valid_mask])
            mape_accuracy = 1 - mape_value
        else:
            mape_value = np.nan
            mape_accuracy = np.nan
        # 季节 naive 对照指标：与模型共用同一 eval_mask，保证口径可比
        if y_naive is not None and mape_meta["valid_points"] > 0:
            y_naive = np.asarray(y_naive).flatten()
            naive_mape = mean_absolute_percentage_error(y_test[valid_mask], y_naive[valid_mask])
            naive_mape_accuracy = 1 - naive_mape
        else:
            naive_mape = np.nan
            naive_mape_accuracy = np.nan

        test_scores = {
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE": mape_value,
            "MAPE Accuracy": mape_accuracy,
            "Naive MAPE": naive_mape,
            "Naive MAPE Accuracy": naive_mape_accuracy,
            "MAPE Threshold": mape_meta["threshold"],
            "MAPE Upper Threshold": mape_meta["upper_threshold"],
            "MAPE Valid Points": mape_meta["valid_points"],
            "MAPE Excluded Points": mape_meta["excluded_points"],
            "MAPE Excluded Ratio": mape_meta["excluded_ratio"],
        }
        if str(horizon_mode).lower() == "calendar_month":
            actual_total = float(np.sum(y_test))
            predicted_total = float(np.sum(y_pred))
            monthly_total_mape = (
                abs(actual_total - predicted_total) / abs(actual_total)
                if abs(actual_total) > 1e-12
                else np.nan
            )
            month_start = cast(
                pd.Timestamp,
                pd.Timestamp(str(df_history_test["time"].iloc[0])),
            )
            test_scores.update(
                {
                    "Calendar Month": month_start.strftime("%Y-%m"),
                    "Forecast Steps": len(y_test),
                    "Monthly Actual Total": actual_total,
                    "Monthly Predicted Total": predicted_total,
                    "Monthly Total MAPE": monthly_total_mape,
                }
            )
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
        mode: str = "percentile",
        percentile: float = 5.0,
        min_value: float = None,
        max_value: float = None,
    ):
        """
        测试集预测数据
        """
        y_test = np.array(y_test).flatten()
        y_pred = np.array(y_pred).flatten()
        mape_meta = build_eval_mask(
            y_test, mode=mode, percentile=percentile, min_value=min_value, max_value=max_value
        )
        valid_mask = mape_meta["valid_mask"]

        cv_plot_df_window = pd.DataFrame()
        time_slice = df_history_test["time"]
        if len(time_slice) != len(y_pred):
            logger.warning(f"{log_prefix} Length mismatch for plotting data: time_slice ({len(time_slice)}) vs y_pred ({len(y_pred)}). Adjusting to min length.")
            min_len = min(len(time_slice), len(y_pred))
            cv_plot_df_window["time"] = time_slice.iloc[:min_len].values
            cv_plot_df_window["Y_trues"] = y_test[:min_len]
            cv_plot_df_window["Y_preds"] = y_pred[:min_len]
            valid_mask = valid_mask[:min_len]
        else:
            cv_plot_df_window["time"] = time_slice.values
            cv_plot_df_window["Y_trues"] = y_test
            cv_plot_df_window["Y_preds"] = y_pred
        cv_plot_df_window["mape_valid"] = valid_mask
        cv_plot_df_window["Y_trues_plot"] = np.where(valid_mask, cv_plot_df_window["Y_trues"], np.nan)
        cv_plot_df_window["Y_preds_plot"] = np.where(valid_mask, cv_plot_df_window["Y_preds"], np.nan)

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
    def test_results_save(args, log_prefix: str, test_scores_df, cv_plot_df, train_outlier_report=None, window_results=None):
        # 分位数单调化(可选):q50 锚定修复，csv/绘图/概率指标使用同一 processed stage。
        cv_plot_df = monotonize_quantile_columns(cv_plot_df, bool(getattr(args, "quantile_monotone", False)))
        if any(str(column).startswith("predict_q") for column in cv_plot_df.columns):
            probabilistic_spec = getattr(args, "probabilistic_spec", None)
            if probabilistic_spec is not None:
                calibration_kwargs = calibration_runtime_kwargs(probabilistic_spec)
                calibration_kwargs["interval_specs"] = probabilistic_spec.intervals
            else:
                calibration_kwargs = {
                    "enable_cqr": bool(
                        getattr(args, "enable_conformal_calibration", False)
                    ),
                    "calibration_windows": int(
                        getattr(args, "conformal_calibration_windows", 5)
                    ),
                    "min_windows": int(getattr(args, "conformal_min_windows", 3)),
                    "min_scores": int(getattr(args, "conformal_min_scores", 30)),
                    "alpha": float(getattr(args, "conformal_alpha", 0.1)),
                    "label_availability_delay_steps": int(
                        getattr(args, "conformal_label_availability_delay_steps", 0)
                    ),
                }
            cv_plot_df = write_probabilistic_artifacts(
                cv_plot_df,
                args.test_results_dir,
                freq=str(getattr(args, "freq", "1D")),
                **calibration_kwargs,
            )
        test_scores_df.to_csv(args.test_results_dir.joinpath("test_scores_df.csv"), index=False, encoding="utf-8")
        cv_plot_df.to_csv(args.test_results_dir.joinpath("cv_plot_df.csv"), index=False, encoding="utf-8")
        if train_outlier_report is None:
            train_outlier_report = empty_train_outlier_report()
        train_outlier_report.to_csv(
            args.test_results_dir.joinpath("train_outlier_report.csv"),
            index=False,
            encoding="utf-8",
        )
        required_cols = {"Y_preds", "Y_trues"}
        # B1 残差频谱诊断：汇总全部窗口的 residual FFT/ACF，输出跨窗口稳定性
        residual_rows = [
            result["residual_diag_row"]
            for result in (window_results or [])
            if result.get("residual_diag_row") is not None
        ]
        if residual_rows:
            from decomposition.residual_diagnostics import (
                summarize_window_residuals,
                write_residual_diagnostics,
            )

            summary = summarize_window_residuals(residual_rows)
            write_residual_diagnostics(
                summary,
                args.test_results_dir.joinpath("residual_diagnostics.csv"),
            )
            fft_cv_text = (
                f"{summary.fft_period_cv:.3f}"
                if summary.fft_period_cv is not None
                and np.isfinite(summary.fft_period_cv)
                else "N/A"
            )
            logger.info(
                f"{log_prefix} residual_diagnostics: "
                f"fft_period_median={summary.fft_period_median}, "
                f"cv={fft_cv_text}, "
                f"stable_band={summary.stable_band_detected}"
            )
        if cv_plot_df.empty or not required_cols.issubset(set(cv_plot_df.columns)):
            logger.warning(f"{log_prefix} No valid prediction columns found for visualization.")
            return
        if bool(getattr(args, "enable_global_training", False)):
            logger.info(
                f"{log_prefix} Panel test artifacts saved without a single-series plot; "
                "cv_plot_df.csv preserves series identity."
            )
            return
        if len(cv_plot_df["Y_preds"].values) == 0 or len(cv_plot_df["Y_trues"].values) == 0:
            logger.warning(f"{log_prefix} No data to visualize for test prediction.")
            return
        # 加载叠加参考序列（若已配置）
        overlay_df = _load_plot_overlay_df(args, log_prefix)
        overlay_col = str(getattr(args, "plot_overlay_col", "") or "").strip()
        import matplotlib.pyplot as plt
        fig_main, ax_main = plt.subplots(figsize=(25, 8))
        # 用未掩码的原始列绘制,保证线条连续不断(eval_mask 掩码仅用于 MAPE 计算,不参与绘图)
        plot_true_col = "Y_trues"
        plot_pred_col = "Y_preds"
        # 按时间排序后再绘制:滑窗 CV 结果默认按 window(最新优先)拼接,直接画会出现
        # 窗口倒序、边界时间倒退造成的"拼接错乱/真值递减"视觉假象(底层预测与指标无误)
        plot_df = cv_plot_df.sort_values("time").reset_index(drop=True) if "time" in cv_plot_df.columns else cv_plot_df
        plot_x = plot_df["time"] if "time" in plot_df.columns else np.arange(len(plot_df))
        ax_main.plot(plot_x, plot_df[plot_true_col].values, label="Trues", lw=1.7)
        ax_main.plot(plot_x, plot_df[plot_pred_col].values, label="Preds", lw=1.7, ls="-.")
        # 模型分位数带(非 CQR PI):填充 q_low~q_high
        qcols = sorted(c for c in plot_df.columns if str(c).startswith("predict_q"))
        if len(qcols) >= 2:
            ax_main.fill_between(
                plot_x,
                plot_df[qcols[0]].astype(float).values,
                plot_df[qcols[-1]].astype(float).values,
                color="tab:blue", alpha=0.15, label=f"Quantiles [{qcols[0]},{qcols[-1]}]",
            )
        ax_main.legend(loc="upper left")
        ax_main.set_xlabel("Time")
        ax_main.set_ylabel("Value")
        ax_main.set_title("Trues and Preds Timeseries Plot")
        ax_main.grid(True)
        # X 轴日期格式化：避免高频数据刻度标签重叠
        if "time" in plot_df.columns:
            from matplotlib.dates import DateFormatter, DayLocator, AutoDateLocator
            locator = AutoDateLocator()
            ax_main.xaxis.set_major_locator(locator)
            ax_main.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        # 叠加参考序列（次坐标轴，量级与目标差异大时不压扁主曲线）
        if overlay_df is not None and "time" in plot_df.columns:
            ax_overlay = ax_main.twinx()
            overlay_vals = overlay_df.set_index("time")[overlay_col].reindex(plot_df["time"]).to_numpy(dtype=float)
            ax_overlay.plot(
                plot_df["time"], overlay_vals,
                label=overlay_col, lw=1.0, color="gray", alpha=0.7, ls="--",
            )
            ax_overlay.set_ylabel(overlay_col)
            ax_overlay.legend(loc="upper right")
        fig_main.tight_layout()
        fig_main.savefig(args.test_results_dir.joinpath("test_prediction.png"), bbox_inches="tight", dpi=300)
        plt.close(fig_main)
        # plt.show();

        # ---- per-window 绘图 ----
        # 每个滑窗测试窗口单独出一张图，输出到 window_plots/ 子目录
        window_plots_dir = args.test_results_dir.joinpath("window_plots")
        window_plots_dir.mkdir(parents=True, exist_ok=True)
        if "window" in cv_plot_df.columns:
            # 从 test_scores_df 取每个窗口的 time_range 和 MAPE（用于标题）
            window_meta = {}
            if not test_scores_df.empty and "time_range" in test_scores_df.columns:
                for idx, row in test_scores_df.iterrows():
                    if row.get("time_range") == "中位数":
                        continue
                    # test_scores_df 的 index 是窗口号
                    window_meta[idx] = {
                        "time_range": row["time_range"],
                        "mape": row.get("MAPE", None),
                    }
            for win, group in cv_plot_df.groupby("window"):
                group_sorted = group.sort_values("time") if "time" in group.columns else group
                fig_w, ax_w = plt.subplots(figsize=(14, 5))
                x_w = group_sorted["time"] if "time" in group_sorted.columns else np.arange(len(group_sorted))
                ax_w.plot(x_w, group_sorted["Y_trues"].values, label="Trues", lw=1.5)
                ax_w.plot(x_w, group_sorted["Y_preds"].values, label="Preds", lw=1.5, ls="-.")
                # 模型分位数带（非 CQR PI）
                qcols_w = sorted(c for c in group_sorted.columns if str(c).startswith("predict_q"))
                if len(qcols_w) >= 2:
                    ax_w.fill_between(
                        x_w,
                        group_sorted[qcols_w[0]].astype(float).values,
                        group_sorted[qcols_w[-1]].astype(float).values,
                        color="tab:blue", alpha=0.15, label=f"Quantiles [{qcols_w[0]},{qcols_w[-1]}]",
                    )
                meta = window_meta.get(win, {})
                tr = meta.get("time_range", "")
                mape_val = meta.get("mape")
                title = f"Window {int(win)}"
                if tr:
                    title += f": {tr}"
                if mape_val is not None and not np.isnan(mape_val):
                    title += f"  MAPE={mape_val:.2%}"
                ax_w.set_title(title)
                ax_w.set_xlabel("Time")
                ax_w.set_ylabel("Value")
                ax_w.grid(True, alpha=0.3)
                ax_w.legend()
                # X 轴日期格式化：避免高频数据（288 点/天）刻度标签重叠成黑块
                if "time" in group_sorted.columns:
                    from matplotlib.dates import DateFormatter, HourLocator, AutoDateLocator
                    span_hours = (group_sorted["time"].max() - group_sorted["time"].min()).total_seconds() / 3600
                    if span_hours <= 48:
                        ax_w.xaxis.set_major_locator(HourLocator(interval=max(1, int(span_hours / 6))))
                        ax_w.xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
                    else:
                        ax_w.xaxis.set_major_formatter(DateFormatter("%m-%d"))
                # 叠加参考序列（次坐标轴）
                if overlay_df is not None and "time" in group_sorted.columns:
                    ax_ov = ax_w.twinx()
                    ov_vals = overlay_df.set_index("time")[overlay_col].reindex(group_sorted["time"]).to_numpy(dtype=float)
                    ax_ov.plot(
                        group_sorted["time"], ov_vals,
                        label=overlay_col, lw=1.0, color="gray", alpha=0.7, ls="--",
                    )
                    ax_ov.set_ylabel(overlay_col)
                    ax_ov.legend(loc="upper right")
                fig_w.autofmt_xdate(rotation=30)
                fig_w.tight_layout()
                fig_w.savefig(window_plots_dir.joinpath(f"window_{int(win):02d}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig_w)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
