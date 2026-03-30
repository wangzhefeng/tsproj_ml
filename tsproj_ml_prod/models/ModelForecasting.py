# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelForecasting.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-29
# * Version     : 1.0.032909
# * Description : 生产环境预测模块
# * Link        : link
# * Requirement : pandas, numpy
# ***************************************************

# python libraries
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tsproj_ml_prod.features.FeatureEngineering import FeatureEngineer
from tsproj_ml_prod.utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Forecaster:
    """
    预测辅助类-所有预测方法的公共逻辑
    """
    
    def __init__(self,
                 args: Dict,
                 horizon: int,
                 model: Any,
                 feature_scaler,
                 target_scaler,
                 df_history: pd.DataFrame,
                 df_future: pd.DataFrame,
                 df_date_future: pd.DataFrame,
                 df_weather_future: pd.DataFrame,
                 endogenous_features: List[str],
                 target_feature: str,
                 target_output_features: List[str],
                 categorical_features: List[str],
                 selected_features: List[str] = None,
                 log_prefix: str = "[Forecaster]"):
        self.args = args
        self.horizon = horizon
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.df_history = df_history
        self.df_future = df_future
        self.df_date_future = df_date_future
        self.df_weather_future = df_weather_future
        self.endogenous_features = endogenous_features
        self.target_feature = target_feature
        self.target_output_features = target_output_features
        self.categorical_features = categorical_features
        self.selected_features = selected_features
        self.log_prefix = log_prefix
        logger.info(f"{self.log_prefix} Forecaster params init...")
        logger.info(f"{self.log_prefix} {'-' * 71}")
        # 最大滞后数量
        self.max_lag = max(self.args.lags) if self.args.lags else 1
        logger.info(f"{self.log_prefix} Forecaster max_lag: {self.max_lag}")
        
        # 获取足够的历史数据以构建滞后特征
        self.df_history_for_lags = self.df_history.iloc[-self.max_lag:].copy()
        logger.info(f"{self.log_prefix} Forecaster df_history_for_lags shape: {self.df_history_for_lags.shape}")
        logger.info(f"{self.log_prefix} Forecaster df_history_for_lags columns: {self.df_history_for_lags.columns.tolist()}")
        # 复用特征工程器，避免递归预测中反复实例化
        self.feature_engineer = FeatureEngineer(self.args, self.log_prefix, verbose=False)

    def _apply_selected_feature_subset(self, predictor_features: List[str], categorical_features: List[str]):
        """
        训练阶段做了特征选择时，推理阶段使用同一子集，避免特征空间不一致。
        """
        if not self.selected_features:
            return predictor_features, categorical_features

        selected_set = set(self.selected_features)
        selected_predictor_features = [f for f in predictor_features if f in selected_set]
        if not selected_predictor_features:
            logger.warning(
                f"{self.log_prefix} selected_features does not overlap current predictor_features, fallback to all predictors."
            )
            selected_predictor_features = predictor_features
        selected_categorical_features = [f for f in categorical_features if f in selected_predictor_features]
        return selected_predictor_features, selected_categorical_features

    def _build_direct_forecast_input(self, endogenous_features: List[str]):
        """
        构建 Direct 策略输入：
        - 历史 max_lag 行 + 全部未来外生行
        - 取锚点为最后一个历史行，使 horizon-aware 外生展开可用
        """
        for endo_feat in endogenous_features:
            if endo_feat not in self.df_history_for_lags.columns and endo_feat in self.df_history.columns:
                self.df_history_for_lags[endo_feat] = self.df_history[endo_feat].iloc[-self.max_lag:]

        df_forecast = pd.concat([self.df_history_for_lags, self.df_future.copy()], ignore_index=True, copy=False)
        (df_forecast_featured,
         predictor_features,
         target_output_features,
         categorical_features) = self.feature_engineer.create_features(
            df_series=df_forecast,
            df_date_history=None,
            df_date_future=self.df_date_future,
            df_weather_history=None,
            df_weather_future=self.df_weather_future,
            endogenous_features_with_target=endogenous_features,
            target_feature=self.target_feature,
            horizon=self.horizon,
        )
        predictor_features, categorical_features = self._apply_selected_feature_subset(
            predictor_features, categorical_features
        )
        anchor_idx = max(len(self.df_history_for_lags) - 1, 0)
        X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[anchor_idx:anchor_idx + 1]
        return X_forecast_input, categorical_features
    # ------------------------------
    # 单变量（目标变量滞后特征）预测单变量（目标变量）
    # ------------------------------
    def univariate_single_multi_step_direct_output_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接输出预测(USMDO)
        """
        # 多步预测值收集器
        Y_preds = np.array([])
        # 预测阶段始终使用未来日期/天气进行特征工程，避免被 is_testing 分支误跳过
        (df_future_featured,
         predictor_features,
         target_output_features,
         categorical_features) = self.feature_engineer.create_features(
            df_series=self.df_future,
            df_date_history=None,
            df_date_future=self.df_date_future,
            df_weather_history=None,
            df_weather_future=self.df_weather_future,
            endogenous_features_with_target=self.endogenous_features,
            target_feature=self.target_feature,
            horizon=self.horizon,
        )
        if predictor_features:
            predictor_features, categorical_features = self._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            X_test_future = df_future_featured.reindex(columns=predictor_features).copy()
        else:
            logger.warning(f"{self.log_prefix} predictor_features is empty in USMDO forecast; fallback to raw future frame.")
            X_test_future = self.df_future.copy()
            categorical_features = self.categorical_features
        logger.info(f"{self.log_prefix} after feature engineering df_future_featured shape: {df_future_featured.shape}")
        logger.info(f"{self.log_prefix} after feature engineering X_test_future: \n{X_test_future.head()}")
        logger.info(f"{self.log_prefix} after feature engineering X_test_future shape: {X_test_future.shape}")
        logger.info(f"{self.log_prefix} after feature engineering categorical_features: {categorical_features}")
        # 模型推理
        if len(X_test_future) > 0:
            Y_preds = np.asarray(self.model.predict(X_test_future))
        if Y_preds.size == 0:
            return np.array([])
        if Y_preds.ndim == 2 and Y_preds.shape[1] == 1:
            return Y_preds[:, 0]
        if Y_preds.ndim == 2 and Y_preds.shape[0] == 1:
            return Y_preds[0]

        return Y_preds

    def univariate_single_multi_step_direct_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步直接预测(USMD)
        """
        X_forecast_input, categorical_features = self._build_direct_forecast_input(
            endogenous_features=self.endogenous_features
        )
        # 模型推理
        y_pred_multi_step = np.asarray(self.model.predict(X_forecast_input))
        if y_pred_multi_step.ndim == 2 and y_pred_multi_step.shape[0] == 1:
            y_pred_multi_step = y_pred_multi_step[0]
        else:
            y_pred_multi_step = y_pred_multi_step.reshape(-1)
        if len(y_pred_multi_step) >= len(self.df_future):
            y_preds = y_pred_multi_step[:len(self.df_future)]
        else:
            y_preds = np.pad(
                y_pred_multi_step,
                pad_width=(0, len(self.df_future) - len(y_pred_multi_step)),
                mode="edge",
            )

        return np.asarray(y_preds)

    def univariate_single_multi_step_recursive_forecast(self):
        """
        单变量(内生变量/目标变量)预测单变量(目标变量)多步递归预测(USMR)
        """
        # 多步预测值收集器
        Y_preds = []
        for step in range(self.horizon):
            if step >= len(self.df_future):
                logger.warning(f"Exhausted df_future for step {step}. Stopping recursive forecast.")
                break
            # 1.构建预测特征数据
            df_future_step = self.df_future.iloc[step : step + 1].copy()
            # 2.合并历史数据和当前步数据
            df_forecast = pd.concat([self.df_history_for_lags, df_future_step], ignore_index=True, copy=False)
            # 3.特征工程
            (df_forecast_featured,
             predictor_features,
             target_output_features,
             categorical_features) = self.feature_engineer.create_features(
                df_series=df_forecast,
                df_date_history=None,
                df_date_future=self.df_date_future,
                df_weather_history=None,
                df_weather_future=self.df_weather_future,
                endogenous_features_with_target=self.endogenous_features,
                target_feature=self.target_feature,
                horizon=self.horizon,
            )
            predictor_features, categorical_features = self._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            # 4.提取出当前预测步所需要的特征（最后一行）
            X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[-1:]
            # 5.模型预测
            y_pred_step = float(np.asarray(self.model.predict(X_forecast_input)).reshape(-1)[0])
            Y_preds.append(y_pred_step)
            # 6.将预测值更新回 df_future_step，以便为下一步预测提供滞后特征
            df_future_step_new_row = df_future_step.copy().iloc[-1:]
            df_future_step_new_row[self.target_feature] = y_pred_step
            # 7.将新行添加到历史数据中，进行下一次循环
            self.df_history_for_lags = pd.concat(
                [self.df_history_for_lags, df_future_step_new_row],
                axis=0,
                ignore_index=True,
            )
            self.df_history_for_lags = self.df_history_for_lags.iloc[-self.max_lag:].copy().reset_index(drop=True)

        return np.array(Y_preds)

    def univariate_single_multi_step_direct_recursive_forecast(self):
        """
        单变量多步直接递归预测(USMDR)
        """
        # 严格分块直接：每个块仅调用一次模型，取块长输出
        block_size = int(getattr(self.args, "block_size", 0) or 0)
        if block_size <= 0:
            block_size = min(self.args.lags) if self.args.lags else 1
        logger.info(f"{self.log_prefix} block_size: {block_size}")
        y_preds = []
        while len(y_preds) < len(self.df_future):
            produced = len(y_preds)
            remain = len(self.df_future) - produced
            df_future_remain = self.df_future.iloc[produced:].copy()
            df_forecast = pd.concat([self.df_history_for_lags, df_future_remain], ignore_index=True, copy=False)
            (df_forecast_featured,
             predictor_features,
             target_output_features,
             categorical_features) = self.feature_engineer.create_features(
                df_series=df_forecast,
                df_date_history=None,
                df_date_future=self.df_date_future,
                df_weather_history=None,
                df_weather_future=self.df_weather_future,
                endogenous_features_with_target=self.endogenous_features,
                target_feature=self.target_feature,
                horizon=self.horizon,
            )
            predictor_features, categorical_features = self._apply_selected_feature_subset(
                predictor_features, categorical_features
            )
            anchor_idx = max(len(self.df_history_for_lags) - 1, 0)
            X_forecast_input = df_forecast_featured.reindex(columns=predictor_features).iloc[anchor_idx:anchor_idx + 1]
            pred_vec = np.asarray(self.model.predict(X_forecast_input))
            if pred_vec.ndim == 2 and pred_vec.shape[0] == 1:
                pred_vec = pred_vec[0]
            else:
                pred_vec = pred_vec.reshape(-1)
            take = min(block_size, remain, len(pred_vec))
            block_pred = pred_vec[:take]
            y_preds.extend(block_pred.tolist())
            # 逐点更新历史（块内直接，不再逐点重复调模型）
            for i in range(take):
                df_new = df_future_remain.iloc[i : i + 1].copy()
                df_new[self.target_feature] = float(block_pred[i])
                self.df_history_for_lags = pd.concat(
                    [self.df_history_for_lags, df_new],
                    axis=0,
                    ignore_index=True,
                )
                self.df_history_for_lags = self.df_history_for_lags.iloc[-self.max_lag:].copy().reset_index(drop=True)

        return np.asarray(y_preds[:len(self.df_future)])

    def _predict_by_method(self) -> np.ndarray:
        """
        根据配置分发预测策略并返回一维预测数组
        """
        # ------------------------------
        # 单变量（目标变量滞后特征）预测单变量（目标变量）
        # ------------------------------
        if self.args.pred_method == "univariate-single-multistep-direct-output":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_direct_output_forecast(USMDO)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_direct_output_forecast()
            logger.info(f"{self.log_prefix} USMDO forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "univariate-single-multistep-direct":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_direct_forecast(USMD)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_direct_forecast()
            logger.info(f"{self.log_prefix} USMD forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_recursive_forecast(USMR)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_recursive_forecast()
            logger.info(f"{self.log_prefix} USMR forecast completed, predicted {len(raw_pred)} steps.")
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            logger.info(f"{self.log_prefix} Forecast method: univariate_single_multi_step_direct_recursive_forecast(USMDR)")
            logger.info(f"{self.log_prefix} {'-' * 60}")
            raw_pred = self.univariate_single_multi_step_direct_recursive_forecast()
            logger.info(f"{self.log_prefix} USMDR forecast completed, predicted {len(raw_pred)} steps.")
        else:
            raise ValueError(f"{self.log_prefix} Unsupported pred_method: {self.args.pred_method}")

        pred_arr = np.asarray(raw_pred)
        if pred_arr.ndim == 0:
            return np.asarray([float(pred_arr)])
        if pred_arr.ndim == 1:
            return pred_arr
        if pred_arr.shape[0] == 1:
            return pred_arr[0]
        if pred_arr.shape[1] == 1:
            return pred_arr[:, 0]

        return pred_arr[:, 0]

    def forecast_results_save(self, df_history, df_future, n_per_day):
        """
        输出结果处理
        """
        # 预测结果保存
        df_future = df_future.copy()
        df_future["time"] = pd.to_datetime(df_future["time"])
        df_future = df_future.sort_values(by=["time"]).reset_index(drop=True)
        df_future.to_csv(self.args.pred_results_dir.joinpath("prediction_raw.csv"), encoding="utf_8_sig", index=False)
        # 历史上下文截取：以未来预测起点为边界，取其前最近 2 天历史真值
        y_trues_df_plot = pd.DataFrame()
        if df_history is not None and not df_history.empty and "time" in df_history.columns and "y" in df_history.columns:
            df_history_plot = df_history.copy()
            df_history_plot["time"] = pd.to_datetime(df_history_plot["time"])
            df_history_plot = df_history_plot.sort_values(by=["time"]).dropna(subset=["y"]).reset_index(drop=True)
            if not df_future.empty:
                future_start = df_future["time"].iloc[0]
                history_before_future = df_history_plot[df_history_plot["time"] < future_start]
                if history_before_future.empty:
                    logger.warning(
                        f"{self.log_prefix} No history before forecast start ({future_start}); "
                        "fallback to latest available 2-day history for plotting."
                    )
                    y_trues_df_plot = df_history_plot.tail(2 * n_per_day).copy()
                else:
                    y_trues_df_plot = history_before_future.tail(2 * n_per_day).copy()
            else:
                y_trues_df_plot = df_history_plot.tail(2 * n_per_day).copy()
        # 保留历史上下文，便于生产问题定位
        if df_history is not None and not df_history.empty:
            df_history.copy().to_csv(
                self.args.pred_results_dir.joinpath("history_context.csv"),
                encoding="utf_8_sig",
                index=False,
            )
        # 拼接可视化数据：最近两天历史 + 未来一天预测
        history_part = pd.DataFrame()
        if not y_trues_df_plot.empty:
            history_part = y_trues_df_plot[["time", "y"]].rename(columns={"y": "value"})
            history_part["series_type"] = "history_true"
        future_part = pd.DataFrame()
        if not df_future.empty and "predict_value" in df_future.columns:
            future_part = df_future[["time", "predict_value"]].rename(columns={"predict_value": "value"})
            future_part["series_type"] = "future_pred"
        if not history_part.empty or not future_part.empty:
            plot_concat_df = pd.concat([history_part, future_part], axis=0, ignore_index=True)
            plot_concat_df = plot_concat_df.sort_values(by=["time"]).reset_index(drop=True)
            plot_concat_df.to_csv(
                self.args.pred_results_dir.joinpath("prediction_plot_concat.csv"),
                encoding="utf_8_sig",
                index=False,
            )
        plt.figure(figsize=(25, 8))
        if not y_trues_df_plot.empty and "y" in y_trues_df_plot.columns:
            plt.plot(y_trues_df_plot["time"], y_trues_df_plot["y"], label="Trues", lw=2.0)
        if not df_future.empty and "predict_value" in df_future.columns:
            plt.plot(df_future["time"], df_future["predict_value"], label="Preds", lw=2.0, ls="-.")
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.title(f"模型预测预测--{self.args.pred_method}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=1.0)
        plt.tight_layout()
        # plt.xticks(rotation=45)
        plt.savefig(self.args.pred_results_dir.joinpath("prediction.png"), dpi=300, bbox_inches="tight")
        plt.close()




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
