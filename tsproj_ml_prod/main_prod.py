# -*- coding: utf-8 -*-

# python libraries
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import time
import datetime
import warnings
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from tsproj_ml_prod.data_provider.data_loader import DataLoader
from tsproj_ml_prod.config.model_config_lgbm_usmdo import ModelConfig
# from tsproj_ml_prod.config.model_config_lgbm_usmd import ModelConfig
# from tsproj_ml_prod.config.model_config_lgbm_usmr import ModelConfig
# from tsproj_ml_prod.config.model_config_lgbm_usmdr import ModelConfig
# from tsproj_ml_prod.config.model_config_cab_usmdo import ModelConfig
# from tsproj_ml_prod.config.model_config_cab_usmd import ModelConfig
# from tsproj_ml_prod.config.model_config_cab_usmr import ModelConfig
# from tsproj_ml_prod.config.model_config_cab_usmdr import ModelConfig
from tsproj_ml_prod.features.FeatureEngineering import FeatureEngineer
from tsproj_ml_prod.models.ModelForecasting import Forecaster
from tsproj_ml_prod.models.ModelTesting import Tester
from tsproj_ml_prod.models.ModelTraining import Trainer
from tsproj_ml_prod.utils.frequency import resolve_samples_per_day
from tsproj_ml_prod.utils.log_util import logger
# from model import BaseModelMainClass

warnings.filterwarnings("ignore")


class ModelMainClass:
    """
    生产环境模型主类
    """

    def __init__(self, project, model, node, args) -> None:
        self.project = project
        self.model = model
        self.node = node
        self.args = args
        self.log_prefix = f"project: {project}, model: {model}, node: {node}::"
    
    def _prepare_args(self, model_cfgs: Dict):
        self.args = ModelConfig(model_cfgs = model_cfgs)
        # ------------------------------
        # 数据参数
        # ------------------------------
        # 目标时间序列每天样本数量
        self.n_per_day = resolve_samples_per_day(self.args.freq)
        # 时间序列历史数据开始时刻
        start_time = self.args.start_time
        if start_time is None:
            start_time = self.args.now_time.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=self.args.history_days)
        # 时间序列未来结束时刻
        future_time = self.args.future_time
        if future_time is None:
            future_time = self.args.now_time + datetime.timedelta(days=self.args.predict_days)
        # 时间序列未来结束时刻
        self.train_start_time = start_time
        self.train_end_time = self.args.now_time
        self.forecast_start_time = self.args.now_time
        self.forecast_end_time = future_time
        # ------------------------------
        # 模型测试、预测
        # ------------------------------
        # 预测未来 1 天(24小时)的数据/数据划分长度/预测数据长度
        self.horizon = int(self.args.predict_days * self.n_per_day)
        # 测试窗口数据长度(训练+测试)
        self.window_len = int(self.args.window_days * self.n_per_day)
        # 测试滑动窗口数量, >=1, 1: 单个窗口
        self.n_windows = int(self.args.history_days * self.n_per_day - self.window_len) // self.horizon + 1
        # ------------------------------
        # 结果保存路径
        # ------------------------------
        pred_method_code_map = {
            "univariate-single-multistep-direct-output": "usmdo",
            "univariate-single-multistep-direct": "usmd",
            "univariate-single-multistep-recursive": "usmr",
            "univariate-single-multistep-direct-recursive": "usmdr",
        }
        pred_method_code = pred_method_code_map.get(self.args.pred_method, str(self.args.pred_method).lower())
        self.setting = f"{self.args.model_type}-{pred_method_code}"
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
        date_feature_status = "启用" if self.args.enable_date_features else "禁用"
        weather_feature_status = "启用" if self.args.enable_weather_features else "禁用"
        datetime_feature_status = "启用" if self.args.enable_datetime_features else "禁用"
        lags_feature_status = "启用" if self.args.enable_lags_features else "禁用"
        advanced_feature_status = "启用" if self.args.enable_advanced_features else "禁用"
        testing_status = "启用" if self.args.is_testing else "禁用"
        forecasting_status = "启用" if self.args.is_forecasting else "禁用"
        logger.info(f"{self.log_prefix} 事件(date type features)特征: {date_feature_status}")
        logger.info(f"{self.log_prefix} 气象(weather   features)特征: {weather_feature_status}")
        logger.info(f"{self.log_prefix} 时间(date time features)特征: {datetime_feature_status}")
        logger.info(f"{self.log_prefix} 滞后(lags      features)特征: {lags_feature_status}")
        logger.info(f"{self.log_prefix} 高级(advanced  features)特征: {advanced_feature_status}")
        logger.info(f"{self.log_prefix} 特征缩放: 禁用")
        logger.info(f"{self.log_prefix} 目标缩放: 禁用")
        logger.info(f"{self.log_prefix} 目标逆变换: 禁用")
        logger.info(f"{self.log_prefix} 类别特征编码: 禁用")
        logger.info(f"{self.log_prefix} 模型融合: 禁用")
        logger.info(f"{self.log_prefix} 模型测试: {testing_status}")
        logger.info(f"{self.log_prefix} 模型预测: {forecasting_status}")
        logger.info(f"{self.log_prefix} 窗口并行数: {int(getattr(self.args, 'window_parallel_workers', 1) or 1)}")
        logger.info(f"{self.log_prefix} 多输出并行数: {int(getattr(self.args, 'multi_output_n_jobs', 1) or 1)}")
        logger.info(f"{self.log_prefix} 模型线程数: {int(getattr(self.args, 'model_thread_count', 1) or 1)}")

    def train(self, X_train: pd.DataFrame, Y_train: pd.DataFrame, categorical_features: List, mode: str = "forecast", verbose: bool = False):
        """
        模型训练
        """
        train_start = time.perf_counter()
        # 模型训练类
        model_trainer = Trainer(args=self.args, log_prefix=self.log_prefix)
        # 模型训练
        model, _, _, selected_features = model_trainer.train(
            X_train=X_train,
            Y_train=Y_train,
            feature_scaler=None,
            target_scaler=None,
            categorical_features=categorical_features,
        )
        logger.info(f"{self.log_prefix} Model Training runtime: {time.perf_counter() - train_start:.3f}s")

        return model, None, None, selected_features

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
        test_start = time.perf_counter()
        # ------------------------------
        # 模型滑窗测试结果收集
        # ------------------------------
        test_scores_df = pd.DataFrame()
        cv_plot_df = pd.DataFrame()
        # ------------------------------
        # 判断是否有足够的历史数据保证至少一个完整的测试窗口
        # ------------------------------
        if self.n_windows <= 0:
            logger.warning(
                f"{self.log_prefix} Not enough data for testing with current "
                f"window configuration (Total X points: "
                f"{len(X_train_history)}"
            )
            logger.warning(
                f"{self.log_prefix} Window length: {self.window_len}, "
                f"Horizon: {self.horizon}). No tests will be performed."
            )
            return test_scores_df, cv_plot_df
        # ------------------------------
        # 模型滑窗测试过程
        # ------------------------------
        n_windows = self.n_windows
        payloads = [
            {
                "args": self.args,
                "log_prefix": self.log_prefix,
                "horizon": self.horizon,
                "window_len": self.window_len,
                "window": window,
                "X_train_history": X_train_history,
                "Y_train_history": Y_train_history,
                "df_history": df_history,
                "df_date_history": df_date_history,
                "df_weather_history": df_weather_history,
                "endogenous_features_with_target": endogenous_features_with_target,
                "target_feature": target_feature,
                "target_output_features": target_output_features,
                "categorical_features": categorical_features,
                "train_start_time": self.train_start_time,
                "train_end_time": self.train_end_time,
            }
            for window in range(1, int(n_windows + 1))
        ]
        window_workers = int(getattr(self.args, "window_parallel_workers", 1) or 1)
        window_results = []
        if window_workers > 1 and len(payloads) > 1:
            logger.info(f"{self.log_prefix} Model Testing window parallel workers: {window_workers}")
            with ProcessPoolExecutor(max_workers=window_workers) as executor:
                futures = [executor.submit(Tester._window_test, payload) for payload in payloads]
                for future in as_completed(futures):
                    window_results.append(future.result())
        else:
            for payload in payloads:
                window_results.append(Tester._window_test(payload))
        for result in sorted(window_results, key=lambda x: x["window"]):
            if result["test_scores_df"] is None or result["cv_plot_df"] is None:
                continue
            test_scores_df = pd.concat([test_scores_df, result["test_scores_df"]], axis=0)
            cv_plot_df = pd.concat([cv_plot_df, result["cv_plot_df"]], axis=0)
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
        Tester.test_results_save(self.args, self.log_prefix, test_scores_df, cv_plot_df)
        logger.info(f"{self.log_prefix} Model Testing result saved in: {self.args.test_results_dir}")
        logger.info(f"{self.log_prefix} Model Testing runtime: {time.perf_counter() - test_start:.3f}s")

        return test_scores_df, cv_plot_df

    def forecast(self,
                 model,
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
        # 模型预测
        predictor = Forecaster(
            args=self.args,
            horizon=self.horizon,
            model=model,
            feature_scaler=None,
            target_scaler=None,
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
        df_future_prediction["predict_value"] = np.asarray(Y_pred).reshape(-1)[: len(df_future_prediction)]
        df_future_prediction = df_future_prediction[["time", "predict_value"]]
        predictor.forecast_results_save(df_history, df_future_prediction, self.n_per_day)
        logger.info(f"{self.log_prefix} Model Forecasting result saved in: {self.args.pred_results_dir}")
        logger.info(f"{self.log_prefix} Model Forecasting runtime: {time.perf_counter() - forecast_start:.3f}s")

        return df_future_prediction

    def process_output(self, df_future):
        """
        生产接口输出格式转换
        """
        # 生产环境输出字段适配
        df_future = df_future.copy().reset_index(drop=True)
        for i in range(len(df_future)):
            current_time = df_future.loc[i, "time"]
            df_future.loc[i, "id"] = (f"{self.args.node_id}_{self.args.out_system_id}_{current_time.strftime('%Y%m%d%H%M%S')}")
            df_future.loc[i, "node_id"] = self.args.node_id
            df_future.loc[i, "system_id"] = self.args.out_system_id
            df_future.loc[i, "predict_value"] = str(df_future.loc[i, "predict_value"])
            df_future.loc[i, "count_data_time"] = df_future.loc[i, "time"].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        df_future = df_future[["id", "node_id", "system_id", "predict_value", "count_data_time"]]

        return df_future

    def run(self, input_data: Dict, model_cfgs: Dict = None):
        """
        生产环境统一运行入口
        """
        run_start = time.perf_counter()
        # ------------------------------
        # 参数
        # ------------------------------
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"Model Config...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
        self._prepare_args(model_cfgs)
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
        input_data = dataloader.load_data(input_data=input_data)
        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"{self.log_prefix} {'#' * 90}")
        logger.info(f"{self.log_prefix} Model history data preprocessing...")
        logger.info(f"{self.log_prefix} {'#' * 90}")
        (
            df_history,
            df_date_history,
            df_weather_history,
            endogenous_features_with_target,
            target_feature,
        ) = dataloader.process_history_data(input_data=input_data)
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
        feature_engineer_history = FeatureEngineer(self.args, self.log_prefix, verbose=True)
        (
            df_history_featured,
            predictor_features,
            target_output_features,
            categorical_features,
        ) = feature_engineer_history.create_features(
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
            self.test(
                df_history=df_history,
                X_train_history=X_train_history,
                Y_train_history=Y_train_history,
                df_date_history=df_date_history,
                df_weather_history=df_weather_history,
                endogenous_features_with_target=endogenous_features_with_target,
                target_feature=target_feature,
                predictor_features=predictor_features,
                target_output_features=target_output_features,
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
            (
                df_future,
                df_date_future,
                df_weather_future,
            ) = dataloader.process_future_data(input_data=input_data)

            # 模型训练
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Training start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            model, _, _, selected_features = self.train(
                X_train=X_train_history,
                Y_train=Y_train_history,
                categorical_features=categorical_features,
                mode="forecast",
                verbose=True,
            )

            # 模型预测
            logger.info(f"{self.log_prefix} {'=' * 87}")
            logger.info(f"{self.log_prefix} Model Forecasting start...")
            logger.info(f"{self.log_prefix} {'=' * 87}")
            df_future_predicted = self.forecast(
                model=model,
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
            # 生产输出结果处理与保存
            df_power_future = self.process_output(df_future_predicted)
            df_power_future.to_csv(
                self.args.pred_results_dir.joinpath("prediction.csv"),
                index=False,
                encoding="utf-8",
            )

            logger.info(f"{self.log_prefix} Total runtime: {time.perf_counter() - run_start:.3f}s")

            # 模型输出
            return {"df_future": df_power_future}




# 测试代码 main 函数
def main():
    # ------------------------------
    # model configs
    # ------------------------------ 
    # 项目配置
    node_id = 1
    out_system_id = 1
    # 数据配置
    history_days = 92
    predict_days = 1
    # 数据分割时间
    now_time = datetime.datetime(2026, 1, 1, 0, 0, 0)
    start_time = now_time.replace(hour=0, minute=0, second=0, microsecond=0) - datetime.timedelta(days=history_days)
    future_time = now_time + datetime.timedelta(days=predict_days)
    # 其他 
    # ------------------------------
    # 模型参数
    # ------------------------------
    model_cfgs = {
        # 项目配置
        "nodes": {
            "node": {
                "node_id": node_id,
                "out_system_id": out_system_id,
            }
        },
        # 数据配置、数据分割时间
        "time_range": {
            "start_time": start_time,
            "now_time": now_time,
            "future_time": future_time,
            "before_days": history_days,
            "after_days": predict_days,
        },
        # "model_type": "lightgbm",
        # "pred_method": "USMD",
    }
    # ------------------------------
    # get data
    # ------------------------------
    # data_dir = Path("./model/model_packages/DemandLoad_lingang/dataset/electricity/2026-01-01/lingang/demand_load/lingang_A/")
    data_dir = Path("./tsproj_ml_prod/dataset/electricity/2026-01-01/lingang/demand_load/lingang_A/")
    df_power = pd.read_csv(data_dir.joinpath("df_power.csv"))
    df_date = pd.read_csv(data_dir.joinpath("df_date.csv"))
    df_weather = pd.read_csv(data_dir.joinpath("df_weather.csv"))
    df_date_future = pd.read_csv(data_dir.joinpath("df_date_future.csv"))
    df_weather_future = pd.read_csv(data_dir.joinpath("df_weather_future.csv"))
    df_date_all = pd.concat([df_date.iloc[:-1, ], df_date_future], axis=0)
    df_weather_all = pd.concat([df_weather.iloc[:-1,], df_weather_future], axis=0)
    input_data = {
        "df_power": df_power,
        "df_date": df_date_all,
        "df_weather": df_weather_all,
        "df_date_future": df_date_all,
        "df_weather_future": df_weather_all,
    }
    # ------------------------------
    # 模型测试
    # ------------------------------
    # 创建模型实例
    model = ModelMainClass(
        project = "test",
        model = "test",
        node = "test",
        args = {},
    )
    # 运行模型
    results = model.run(input_data, model_cfgs = model_cfgs)

    logger.info(f"{model.log_prefix} {'#' * 85}")
    logger.info(f"{model.log_prefix} 模型预测流程完成！")
    logger.info(f"{model.log_prefix} {'#' * 85}")

if __name__ == "__main__":
    main()
