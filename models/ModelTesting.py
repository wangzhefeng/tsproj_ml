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
from data_provider.outlier_handling import (
    empty_train_outlier_report,
    handle_train_outliers,
)
from utils.eval_mask import build_eval_mask
from utils.quantile import monotonize_quantile_columns
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
        target_detrender = payload.get("target_detrender")
        train_outlier_report = empty_train_outlier_report()

        # 滑窗数据分割：先切原始历史，再在窗口内构造训练标签，避免 Direct 标签跨入测试期
        split_result = Tester._evaluate_split(
            payload["df_history"],
            window,
            horizon=horizon,
            window_len=window_len,
            log_prefix=log_prefix,
        )
        if split_result is None:
            return {
                "window": window,
                "test_scores_df": None,
                "cv_plot_df": None,
                "train_outlier_report": train_outlier_report,
            }
        df_history_train, df_history_test = split_result
        df_history_train, train_outlier_report = handle_train_outliers(
            args=args,
            df_history_train=df_history_train,
            target_feature=payload["target_feature"],
            window=window,
            log_prefix=log_prefix,
        )
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
            return {
                "window": window,
                "test_scores_df": None,
                "cv_plot_df": None,
                "train_outlier_report": train_outlier_report,
            }
        X_train, Y_train, target_output_features, categorical_features = build_result
        # 窗口目标特征处理
        Y_train = Y_train.to_frame() if isinstance(Y_train, pd.Series) else Y_train
        y_test_raw = df_history_test[payload["target_feature"]].to_numpy()
        # detrend 开启时 df_history_test 来自 detrended 序列,评分前还原到电平空间
        if target_detrender is not None and target_detrender.is_fitted:
            y_test_raw = target_detrender.restore(y_test_raw, df_history_test["time"])
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
            target_detrender=target_detrender,
            log_prefix=log_prefix,
        )
        y_pred = predictor._predict_by_method()
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
        # 分位数预测(若启用):补入 cv_plot,使回测也体现分位数区间
        if getattr(predictor, "quantile_outputs", None):
            n = len(cv_plot_df_window)
            for q, q_pred in sorted(predictor.quantile_outputs.items(), key=lambda x: float(x[0])):
                q_col = f"predict_q{int(round(float(q) * 100)):02d}"
                cv_plot_df_window[q_col] = np.asarray(q_pred).reshape(-1)[:n]

        # 注入窗口编号，供后续 per-window 绘图使用
        if not cv_plot_df_window.empty:
            cv_plot_df_window["window"] = window

        return {
            "window": window,
            "test_scores_df": eval_scores_window,
            "cv_plot_df": cv_plot_df_window,
            "train_outlier_report": train_outlier_report,
        }

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
        mode: str = "percentile",
        percentile: float = 5.0,
        min_value: float = None,
        max_value: float = None,
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

        test_scores = {
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": root_mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE": mape_value,
            "MAPE Accuracy": mape_accuracy,
            "MAPE Threshold": mape_meta["threshold"],
            "MAPE Upper Threshold": mape_meta["upper_threshold"],
            "MAPE Valid Points": mape_meta["valid_points"],
            "MAPE Excluded Points": mape_meta["excluded_points"],
            "MAPE Excluded Ratio": mape_meta["excluded_ratio"],
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
    def test_results_save(args, log_prefix: str, test_scores_df, cv_plot_df, train_outlier_report=None):
        # 分位数单调化(可选):逐行排序 predict_q* 列(csv 与绘图同步生效)
        cv_plot_df = monotonize_quantile_columns(cv_plot_df, bool(getattr(args, "quantile_monotone", False)))
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
        if cv_plot_df.empty or not required_cols.issubset(set(cv_plot_df.columns)):
            logger.warning(f"{log_prefix} No valid prediction columns found for visualization.")
            return
        if len(cv_plot_df["Y_preds"].values) == 0 or len(cv_plot_df["Y_trues"].values) == 0:
            logger.warning(f"{log_prefix} No data to visualize for test prediction.")
            return
        import matplotlib.pyplot as plt
        plt.figure(figsize=(25, 8))
        # 用未掩码的原始列绘制,保证线条连续不断(eval_mask 掩码仅用于 MAPE 计算,不参与绘图)
        plot_true_col = "Y_trues"
        plot_pred_col = "Y_preds"
        # 按时间排序后再绘制:滑窗 CV 结果默认按 window(最新优先)拼接,直接画会出现
        # 窗口倒序、边界时间倒退造成的"拼接错乱/真值递减"视觉假象(底层预测与指标无误)
        plot_df = cv_plot_df.sort_values("time").reset_index(drop=True) if "time" in cv_plot_df.columns else cv_plot_df
        plot_x = plot_df["time"] if "time" in plot_df.columns else np.arange(len(plot_df))
        plt.plot(plot_x, plot_df[plot_true_col].values, label="Trues", lw=1.7)
        plt.plot(plot_x, plot_df[plot_pred_col].values, label="Preds", lw=1.7, ls="-.")
        # 分位数预测区间带(若回测含分位数列):填充 q_low~q_high
        qcols = sorted(c for c in plot_df.columns if str(c).startswith("predict_q"))
        if len(qcols) >= 2:
            plt.fill_between(
                plot_x,
                plot_df[qcols[0]].astype(float).values,
                plot_df[qcols[-1]].astype(float).values,
                color="tab:blue", alpha=0.15, label=f"PI [{qcols[0]},{qcols[-1]}]",
            )
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Trues and Preds Timeseries Plot")
        plt.grid(True)
        # X 轴日期格式化：避免高频数据刻度标签重叠
        if "time" in plot_df.columns:
            from matplotlib.dates import DateFormatter, DayLocator, AutoDateLocator
            locator = AutoDateLocator()
            plt.gca().xaxis.set_major_locator(locator)
            plt.gca().xaxis.set_major_formatter(DateFormatter("%m-%d %H:%M"))
        plt.tight_layout()
        plt.savefig(args.test_results_dir.joinpath("test_prediction.png"), bbox_inches="tight", dpi=300)
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
                # 分位数区间带
                qcols_w = sorted(c for c in group_sorted.columns if str(c).startswith("predict_q"))
                if len(qcols_w) >= 2:
                    ax_w.fill_between(
                        x_w,
                        group_sorted[qcols_w[0]].astype(float).values,
                        group_sorted[qcols_w[-1]].astype(float).values,
                        color="tab:blue", alpha=0.15, label=f"PI [{qcols_w[0]},{qcols_w[-1]}]",
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
                fig_w.autofmt_xdate(rotation=30)
                fig_w.tight_layout()
                fig_w.savefig(window_plots_dir.joinpath(f"window_{int(win):02d}.png"), dpi=150, bbox_inches="tight")
                plt.close(fig_w)




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
