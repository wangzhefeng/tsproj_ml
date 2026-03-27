import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import copy
import math
import datetime
import warnings
warnings.filterwarnings("ignore")
from typing import Dict, List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    r2_score,  # R2
    mean_squared_error,  # MSE
    root_mean_squared_error,  # RMSE
    mean_absolute_error,  # MAE
    mean_absolute_percentage_error,  # MAPE
)
from sklearn.preprocessing import StandardScaler

# from model import BaseModelMainClass
from utils.log_util import logger


class ModelMainClass:#(BaseModelMainClass):

    DEFAULT_LAGS = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        96, 97, 98, 99, 100, 101,
        192, 193, 194, 195, 196, 197,
        288, 289, 290, 291, 292, 293,
    ]
    DIRECT_STAT_WINDOWS = [12, 36, 288]

    def __init__(self, project, model, node, args: Dict) -> None:
        self.project = project
        self.model = model
        self.node = node
        self.args = args
        self.log_prefix = f"project: {project}, model: {model}, node: {node}::"
    # ##############################
    # 数据预处理
    # ##############################
    def _preprocess_data(
        self, raw_df: pd.DataFrame, column_name: str, new_column_name: str
    ):
        # copy
        df = copy.deepcopy(raw_df)
        # 转换时间戳类型
        df[new_column_name] = pd.to_datetime(df[column_name])
        # 去除重复时间戳
        df.drop_duplicates(
            subset=new_column_name, keep="last", inplace=True, ignore_index=True
        )

        return df

    def process_history_data(self):
        """
        处理历史数据
        """
        # 数据预处理
        df_power = self._preprocess_data(
            self.input_data["df_power"], "count_data_time", "timeStamp"
        )
        df_date = self._preprocess_data(self.input_data["df_date"], "date", "timeStamp")
        df_weather = self._preprocess_data(
            self.input_data["df_weather"], "ts", "timeStamp"
        )
        # 整理历史功率数据
        df_load = pd.DataFrame(
            {"timeStamp": pd.date_range(self.start_time, self.now_time, freq=self.freq)}
        )
        df_load[self.target] = df_load["timeStamp"].map(
            df_power.set_index("timeStamp")["h_total_use"]
        )  # 将原始数据映射到时间戳完整的 df_load 中
        df_load[self.target] = df_load[self.target].apply(
            lambda x: float(x)
        )  # 功率数据转换为浮点数
        logger.info(f"{self.log_prefix} df_load length after map load: {len(df_load)}")
        df_load.dropna(inplace=True, ignore_index=True)  # 删除含空值的行
        logger.info(f"{self.log_prefix} df_load length after drop NA: {len(df_load)}")
        df_load = df_load[df_load[self.target] > 0]  # 如果需求负荷小于 0，删除
        logger.info(
            f"{self.log_prefix} df_load length after data filter: {len(df_load)}"
        )
        logger.info(
            f"{self.log_prefix} df_load has nan or not: \n{df_load.isna().any()}"
        )  # 缺失值检查
        # 特征工程
        df_load, weather_features = self.extend_weather_feature(df_load, df_weather)
        logger.info(
            f"{self.log_prefix} df_load length after merge weather features: {len(df_load)}"
        )
        df_load, datetime_features = self.extend_datetime_stamp_feature(df_load)
        logger.info(
            f"{self.log_prefix} df_load length after merge datetime features: {len(df_load)}"
        )
        df_load, date_features = self.extend_date_type_feature(df_load, df_date)
        logger.info(
            f"{self.log_prefix} df_load length after merge date features: {len(df_load)}"
        )
        df_load, lag_features = self.extend_lag_feature(df_load, lags=self.lags)
        logger.info(
            f"{self.log_prefix} df_load length after merge lag features: {len(df_load)}"
        )
        # 特征排序
        training_feature_list = (
            lag_features + weather_features + datetime_features + date_features
        )
        df_load = df_load[["timeStamp"] + training_feature_list + [self.target]]
        logger.info(
            f"{self.log_prefix} training_feature_list: \n{training_feature_list}"
        )
        logger.info(
            f"{self.log_prefix} df_load length after feature engineering: {len(df_load)}"
        )
        logger.info(
            f"{self.log_prefix} df_load.head() after feature engineering: \n{df_load.head()}"
        )
        logger.info(
            f"{self.log_prefix} df_load.tail() after feature engineering: \n{df_load.tail()}"
        )

        return (df_load, training_feature_list)

    def process_future_data(self):
        """
        处理未来数据
        """
        # 数据预处理
        df_date_future = self._preprocess_data(
            self.input_data["df_date_future"], "date", "timeStamp"
        )
        df_weather_future = self._preprocess_data(
            self.input_data["df_weather_future"], "ts", "timeStamp"
        )
        # 创建 DataFrame 并添加 timeStamp 列
        df_future = pd.DataFrame(
            {
                "timeStamp": pd.date_range(
                    pd.to_datetime(self.now_time).replace(minute=0, second=0, microsecond=0), 
                    self.future_time, freq=self.freq
                )
            }
        )
        # 特征工程
        df_future, datetime_features = self.extend_datetime_stamp_feature(df_future)
        logger.info(
            f"{self.log_prefix} df_future length after merge datetime features: {len(df_future)}"
        )
        df_future, date_features = self.extend_date_type_feature(
            df_future, df_date_future
        )
        logger.info(
            f"{self.log_prefix} df_future length after merge date features: {len(df_future)}"
        )
        df_future, weather_features = self.extend_future_weather_feature(
            df_future, df_weather_future
        )
        logger.info(
            f"{self.log_prefix} df_future length after merge weather features: {len(df_future)}"
        )
        # 插值填充预测缺失值
        df_future = df_future.interpolate()
        df_future.dropna(inplace=True, ignore_index=True)
        logger.info(
            f"{self.log_prefix} df_future length after interpolate and dropna: {len(df_future)}"
        )
        # 特征列表
        future_feature_list = weather_features + datetime_features + date_features
        logger.info(f"{self.log_prefix} future_feature_list: \n{future_feature_list}")

        return (df_future, future_feature_list)
    # ##############################
    # 特征工程
    # ##############################
    def extend_datetime_stamp_feature(self, df: pd.DataFrame):
        """
        增加时间特征
        """
        df["datetime_minute"] = df["timeStamp"].apply(lambda x: x.minute)
        df["datetime_hour"] = df["timeStamp"].apply(lambda x: x.hour)
        df["datetime_day"] = df["timeStamp"].apply(lambda x: x.day)

        df["datetime_weekday"] = df["timeStamp"].apply(lambda x: x.weekday())
        df["datetime_week"] = df["timeStamp"].apply(lambda x: x.week)
        df["datetime_day_of_week"] = df["timeStamp"].apply(lambda x: x.dayofweek)

        df["datetime_week_of_year"] = df["timeStamp"].apply(lambda x: x.weekofyear)
        df["datetime_month"] = df["timeStamp"].apply(lambda x: x.month)
        df["datetime_days_in_month"] = df["timeStamp"].apply(lambda x: x.daysinmonth)

        df["datetime_quarter"] = df["timeStamp"].apply(lambda x: x.quarter)
        df["datetime_day_of_year"] = df["timeStamp"].apply(lambda x: x.dayofyear)
        df["datetime_year"] = df["timeStamp"].apply(lambda x: x.year)

        datetime_features = [
            "datetime_minute",
            "datetime_hour",
            "datetime_day",
            "datetime_weekday",
            "datetime_week",
            "datetime_day_of_week",
            "datetime_week_of_year",
            "datetime_month",
            "datetime_days_in_month",
            "datetime_quarter",
            "datetime_day_of_year",
            "datetime_year",
        ]

        return df, datetime_features

    def extend_date_type_feature(self, df: pd.DataFrame, df_date: pd.DataFrame):
        """
        增加日期类型特征：
        1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
        """
        # data map
        df["date"] = df["timeStamp"].apply(
            lambda x: x.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        df["date_type"] = df["date"].map(df_date.set_index("timeStamp")["date_type"])
        # date features
        date_features = ["date_type"]

        return df, date_features

    def extend_weather_feature(self, df_load: pd.DataFrame, df_weather: pd.DataFrame):
        """
        处理天气特征
        """
        # 特征筛选
        weather_features_raw = [
            "rt_ssr",
            "rt_ws10",
            "rt_tt2",
            "rt_dt",
            "rt_ps",
            "rt_rain",
        ]
        df_weather = df_weather[["timeStamp"] + weather_features_raw]
        # 删除含空值的行
        df_weather.dropna(inplace=True, ignore_index=True)
        # 将除了timeStamp的列转为float类型
        for col in weather_features_raw:
            df_weather[col] = df_weather[col].apply(lambda x: float(x))
        # 计算相对湿度
        df_weather["cal_rh"] = np.nan
        for i in df_weather.index:
            if (
                df_weather.loc[i, "rt_tt2"] is not np.nan
                and df_weather.loc[i, "rt_dt"] is not np.nan
            ):
                # 通过温度和露点温度计算相对湿度
                temp = (
                    math.exp(
                        17.2693
                        * (df_weather.loc[i, "rt_dt"] - 273.15)
                        / (df_weather.loc[i, "rt_dt"] - 35.86)
                    )
                    / math.exp(
                        17.2693
                        * (df_weather.loc[i, "rt_tt2"] - 273.15)
                        / (df_weather.loc[i, "rt_tt2"] - 35.86)
                    )
                    * 100
                )
                temp = max(min(temp, 100), 0)
                df_weather.loc[i, "cal_rh"] = temp
            else:
                rt_tt2 = df_weather.loc[i, "rt_tt2"]
                rt_dt = df_weather.loc[i, "rt_dt"]
                logger.info(f"{self.log_prefix} rt_tt2 is {rt_tt2}, rt_dt is {rt_dt}")
        # 特征排序
        weather_features = [
            "rt_ssr",  # 太阳总辐射
            "rt_ws10",  # 10m 风速
            "rt_tt2",  # 2M 气温
            "cal_rh",  # 相对湿度
            "rt_ps",  # 气压
            "rt_rain",  # 降雨量
        ]
        df_weather = df_weather[["timeStamp"] + weather_features]

        # 合并功率数据和天气数据
        df_load = pd.merge(df_load, df_weather, on="timeStamp", how="left")
        # 插值填充缺失值
        df_load = df_load.interpolate()
        df_load.dropna(inplace=True, ignore_index=True)

        return df_load, weather_features

    def extend_lag_feature(self, df: pd.DataFrame, lags: List):
        """
        添加滞后特征
        """

        for lag in lags:
            df[f"lag_{lag}"] = df[self.target].shift(lag)
        df.dropna(inplace=True)

        lag_features = [f"lag_{lag}" for lag in lags]

        return df, lag_features

    def extend_future_weather_feature(self, df_future, df_weather_future):
        """
        未来天气数据特征构造

        Args:
            df_future (_type_): _description_
            df_weather_future (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 筛选天气预测数据
        pred_weather_features = [
            "pred_ssrd",
            "pred_ws10",
            "pred_tt2",
            "pred_rh",
            "pred_ps",
            "pred_rain",
        ]
        df_weather_future = df_weather_future[["timeStamp"] + pred_weather_features]
        # 删除含空值的行
        df_weather_future.dropna(inplace=True, ignore_index=True)
        # 数据类型转换
        for col in pred_weather_features:
            df_weather_future[col] = df_weather_future[col].apply(lambda x: float(x))
        # 将预测天气数据整理到预测df中
        df_future["rt_ssr"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_ssrd"]
        )
        df_future["rt_ws10"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_ws10"]
        )
        df_future["rt_tt2"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_tt2"]
        )
        df_future["cal_rh"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_rh"]
        )
        df_future["rt_ps"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_ps"]
        )
        df_future["rt_rain"] = df_future["timeStamp"].map(
            df_weather_future.set_index("timeStamp")["pred_rain"]
        )

        weather_features = ["rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain"]

        return df_future, weather_features
    # ##############################
    # 工具函数
    # ##############################
    def _score_predictions(self, y_true, y_pred):
        """
        模型评价指标
        """
        y_true_series = pd.Series(y_true).astype(float).reset_index(drop=True)
        y_pred_series = pd.Series(y_pred).astype(float).reset_index(drop=True)
        return {
            "R2": r2_score(y_true_series, y_pred_series),
            "mse": mean_squared_error(y_true_series, y_pred_series),
            "rmse": root_mean_squared_error(y_true_series, y_pred_series),
            "mae": mean_absolute_error(y_true_series, y_pred_series),
            "mape": mean_absolute_percentage_error(y_true_series, y_pred_series),
            "accuracy": 1 - mean_absolute_percentage_error(y_true_series, y_pred_series),
        }

    def _log_test_scores(self, test_scores, prefix="model test"):
        """
        输出测试指标日志
        """
        logger.info(f"{self.log_prefix} {prefix} R2: {test_scores['R2']:.4f}")
        logger.info(f"{self.log_prefix} {prefix} mse: {test_scores['mse']:.4f}")
        logger.info(f"{self.log_prefix} {prefix} rmse: {test_scores['rmse']:.4f}")
        logger.info(f"{self.log_prefix} {prefix} mae: {test_scores['mae']:.4f}")
        logger.info(f"{self.log_prefix} {prefix} mape: {test_scores['mape']:.4f}")
        logger.info(f"{self.log_prefix} {prefix} mape accuracy: {test_scores['accuracy']:.4f}")

    def _scale_features(self, train_df, predict_df=None):
        """
        归一化/标准化
        """
        train_scaled = train_df.copy()
        predict_scaled = None if predict_df is None else predict_df.copy()
        scaler = None
        if self.scale:
            scaler = StandardScaler()
            train_scaled.loc[:, train_df.columns] = scaler.fit_transform(train_df)
            if predict_scaled is not None:
                predict_scaled.loc[:, predict_df.columns] = scaler.transform(predict_df)
        return train_scaled, predict_scaled, scaler

    def _split_feature_groups(self, feature_list):
        """
        区分滞后特征和外生特征
        """
        lag_features = [feature for feature in feature_list if feature.startswith("lag_")]
        exogenous_features = [
            feature for feature in feature_list if not feature.startswith("lag_")
        ]
        return lag_features, exogenous_features

    def _build_direct_stat_feature_names(self):
        """
        direct 方法的历史统计特征名称
        """
        feature_names = []
        for window in self.DIRECT_STAT_WINDOWS:
            feature_names.append(f"direct_hist_mean_{window}")
        feature_names.extend(
            [
                "direct_hist_std_12",
                "direct_hist_min_12",
                "direct_hist_max_12",
            ]
        )
        return feature_names

    def _build_direct_stat_features_for_training(self, df: pd.DataFrame):
        """
        构造 direct 方法的历史统计特征：
        仅依赖预测起点之前的真实历史负荷
        """
        df_stats = pd.DataFrame(index=df.index)
        for window in self.DIRECT_STAT_WINDOWS:
            df_stats[f"direct_hist_mean_{window}"] = (
                df[self.target].shift(1).rolling(window=window, min_periods=window).mean()
            )
        df_stats["direct_hist_std_12"] = (
            df[self.target].shift(1).rolling(window=12, min_periods=12).std()
        )
        df_stats["direct_hist_min_12"] = (
            df[self.target].shift(1).rolling(window=12, min_periods=12).min()
        )
        df_stats["direct_hist_max_12"] = (
            df[self.target].shift(1).rolling(window=12, min_periods=12).max()
        )
        return df_stats

    def _build_direct_stat_features_for_forecast(self, history_target: pd.Series):
        """
        根据当前可用历史数据构造 direct 方法预测阶段的统计特征
        """
        history_target = history_target.reset_index(drop=True)
        feature_values = {}
        for window in self.DIRECT_STAT_WINDOWS:
            feature_values[f"direct_hist_mean_{window}"] = (
                history_target.iloc[-window:].mean() if len(history_target) >= window else np.nan
            )
        feature_values["direct_hist_std_12"] = (
            history_target.iloc[-12:].std() if len(history_target) >= 12 else np.nan
        )
        feature_values["direct_hist_min_12"] = (
            history_target.iloc[-12:].min() if len(history_target) >= 12 else np.nan
        )
        feature_values["direct_hist_max_12"] = (
            history_target.iloc[-12:].max() if len(history_target) >= 12 else np.nan
        )
        return feature_values
    # ##############################
    # 模型训练、测试、预测工具
    # ##############################
    # ------------------------------
    # 模型训练、测试
    # ------------------------------
    # 多步直接输出预测、多步递归预测
    def _train_single_model(self, data_X, data_Y, lgbm_params):
        """
        单模型训练：
        - direct_output: 单模型直接预测
        - recursive: 单模型递归预测
        """
        # 特征列表
        feature_list = data_X.columns
        # 训练集、测试集划分
        data_length = len(data_X)
        X_train = data_X.iloc[-data_length : -self.horizon].copy()
        Y_train = data_Y.iloc[-data_length : -self.horizon].copy()
        X_test = data_X.iloc[-self.horizon :].copy()
        Y_test = data_Y.iloc[-self.horizon :].copy()
        # 训练集、测试集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        X_test_df = X_test.copy()
        Y_test_df = Y_test.copy()
        # ------------------------------
        # 模型测试
        # ------------------------------
        # 归一化/标准化
        X_train_scaled, _, scaler_features_test = self._scale_features(X_train)
        # 模型训练
        lgb_model = lgb.LGBMRegressor(**lgbm_params)
        lgb_model.fit(X_train_scaled, Y_train)
        # 模型预测
        if self.pred_method == "recursive":
            Y_predicted = self._recursive_forecast(
                model=lgb_model,
                history=pd.concat([X_train_df, Y_train_df], axis=1),
                future=X_test_df,
                lags=self.lags,
                steps=self.horizon,
                scaler_features=scaler_features_test,
            )
        else:
            _, X_test_scaled, _ = self._scale_features(X_train, X_test)
            Y_predicted = lgb_model.predict(X_test_scaled[feature_list])
        # 模型评价
        test_scores = self._score_predictions(Y_test_df, Y_predicted)
        self._log_test_scores(test_scores)
        # ------------------------------
        # 最终模型训练
        # ------------------------------
        # 所有训练数据
        final_X_train = pd.concat([X_train_df, X_test_df], axis=0)
        final_Y_train = pd.concat([Y_train_df, Y_test_df], axis=0)
        final_X_train_scaled, _, scaler_final = self._scale_features(final_X_train)
        # 模型训练
        final_model = lgb.LGBMRegressor(**lgbm_params)
        final_model.fit(final_X_train_scaled, final_Y_train)

        return final_model, scaler_final#, test_scores
    # 多步直接预测
    def _get_direct_horizon_feature_list(self, feature_list, horizon_step):
        """
        获取 direct horizon 在当前 horizon 下可用的特征列表：
        - lag_j 仅在 j >= horizon_step 时可用
        - 外生特征使用目标时点对应的未来特征
        """
        lag_features, exogenous_features = self._split_feature_groups(feature_list)
        available_lag_features = []
        for lag_feature in lag_features:
            lag_step = int(lag_feature.split("_")[1])
            if lag_step >= horizon_step:
                available_lag_features.append(lag_feature)
        return (
            available_lag_features
            + exogenous_features
            + self._build_direct_stat_feature_names()
        )

    def _build_direct_horizon_samples(self, df, feature_list, horizon_step):
        """
        构造 direct horizon 训练样本：
        当前时刻滞后特征 + 对应 horizon 的未来外生特征 + 对应 horizon 的未来目标
        """
        lag_features, exogenous_features = self._split_feature_groups(feature_list)
        horizon_feature_list = self._get_direct_horizon_feature_list(
            feature_list, horizon_step
        )
        df_horizon = df[["timeStamp"]].copy()
        df_stat_features = self._build_direct_stat_features_for_training(df)
        for lag_feature in lag_features:
            lag_step = int(lag_feature.split("_")[1])
            if lag_step >= horizon_step:
                # 预测 y(t+h) 时，lag_j 对应 y(t+h-j)，仅当 j>=h 时才是已知历史值
                df_horizon[lag_feature] = df[self.target].shift(lag_step - horizon_step)
        for feature in exogenous_features:
            # 外生特征使用目标时点 t+h 对应的未来信息
            df_horizon[feature] = df[feature].shift(-horizon_step)
        for feature in self._build_direct_stat_feature_names():
            df_horizon[feature] = df_stat_features[feature]
        label_col = f"{self.target}_t_plus_{horizon_step}"
        df_horizon[label_col] = df[self.target].shift(-horizon_step)
        df_horizon = df_horizon[["timeStamp"] + horizon_feature_list + [label_col]]
        df_horizon.dropna(inplace=True, ignore_index=True)
        return df_horizon, label_col, horizon_feature_list

    def _train_direct_horizon_models(self, df_train, feature_list, lgbm_params):
        """
        单变量多步直接预测：
        为每个 horizon 训练一个独立模型。
        """
        # 测试 horizon 用于离线评价，训练 horizon 用于最终未来预测
        max_test_horizon = self.horizon
        max_train_horizon = self.horizon
        models = {}
        scalers = {}
        self.direct_feature_map = {}
        horizon_metrics = []
        y_true_all = []
        y_pred_all = []

        # 为每个 horizon 分别训练一个 LightGBM 模型
        for horizon_step in range(1, max_train_horizon + 1):
            df_horizon, label_col, horizon_feature_list = self._build_direct_horizon_samples(
                df_train, feature_list, horizon_step
            )
            if len(df_horizon) <= self.horizon:
                raise ValueError(
                    f"not enough samples for direct at step={horizon_step}"
                )

            X_all = df_horizon[horizon_feature_list].copy()
            y_all = df_horizon[label_col].copy()

            # 训练集、测试集划分
            X_train = X_all.iloc[:-self.horizon].copy()
            y_train = y_all.iloc[:-self.horizon].copy()
            X_test = X_all.iloc[-self.horizon :].copy()
            y_test = y_all.iloc[-self.horizon :].copy()

            # ------------------------------
            # horizon 模型测试
            # ------------------------------
            X_train_scaled, X_test_scaled, _ = self._scale_features(
                X_train, X_test
            )
            test_model = lgb.LGBMRegressor(**lgbm_params)
            test_model.fit(X_train_scaled, y_train)
            y_pred = test_model.predict(X_test_scaled)

            if horizon_step <= max_test_horizon:
                horizon_score = self._score_predictions(y_test, y_pred)
                horizon_score["horizon_step"] = horizon_step
                horizon_metrics.append(horizon_score)
                y_true_all.extend(pd.Series(y_test).tolist())
                y_pred_all.extend(pd.Series(y_pred).tolist())

            # ------------------------------
            # horizon 最终模型
            # ------------------------------
            X_all_scaled, _, scaler_final = self._scale_features(X_all)
            final_model = lgb.LGBMRegressor(**lgbm_params)
            final_model.fit(X_all_scaled, y_all)
            models[horizon_step] = final_model
            scalers[horizon_step] = scaler_final
            self.direct_feature_map[horizon_step] = horizon_feature_list

            if horizon_step in [1, max_test_horizon]:
                horizon_score = self._score_predictions(y_test, y_pred)
                self._log_test_scores(horizon_score, prefix=f"direct step={horizon_step}")

        test_scores = self._score_predictions(y_true_all, y_pred_all)
        test_scores["strategy"] = self.pred_method
        test_scores["horizon_models"] = max_train_horizon
        self.horizon_metrics_df = pd.DataFrame(horizon_metrics)
        logger.info(
            f"{self.log_prefix} direct metrics rows: {len(self.horizon_metrics_df)}"
        )
        self._log_test_scores(test_scores, prefix="direct overall")

        return models, scalers#, test_scores
    # ------------------------------
    # 模型测试
    # ------------------------------
    # TODO
    def _direct_multioutput_forecast(self):
        pass
    
    def _recursive_forecast(
        self, model, history, future, lags, steps, scaler_features=None
    ):
        """
        递归多步预测
        """
        # last 96xday's steps true targets
        pred_history = list(history.iloc[-int(max(lags)) : -1][self.target].values)
        # initial features
        training_feature_list = [
            col for col in history.columns if col not in ["timeStamp", self.target]
        ]
        current_features_df = history[training_feature_list].copy()
        # forecast collection
        predictions = []
        # 预测下一步
        for step in range(steps):
            # 初始预测特征
            if scaler_features is not None:
                current_features = scaler_features.transform(
                    current_features_df.iloc[-1:]
                )
            else:
                current_features = current_features_df.iloc[-1].values
            # 预测
            next_pred = model.predict(current_features.reshape(1, -1))
            # 更新 pred_history
            pred_history.append(next_pred[0])

            # 更新特征: 将预测值作为新的滞后特征
            new_row_df = current_features_df.iloc[-1:].copy()
            # 更新特征: date, weather
            for future_feature in future.columns:
                new_row_df[future_feature] = future.iloc[step][future_feature]
            # 更新特征: lag
            for i in lags:
                if i > len(pred_history):
                    break
                new_row_df[f"lag_{i}"] = pred_history[-i]
            # 更新 current_features_df
            current_features_df = pd.concat(
                [current_features_df, new_row_df],
                axis=0,
                ignore_index=True,
            )

            # 收集预测结果
            predictions.append(next_pred[0])

        return predictions

    def _direct_horizon_forecast(
        self,
        models,
        history,
        future,
        scaler_features=None,
    ):
        """
        direct horizon 预测：
        第 k 步使用第 k 个模型直接输出，不进行递归展开。
        """
        lag_features, exogenous_features = self._split_feature_groups(list(future.columns))
        if lag_features:
            raise ValueError("future dataframe should not include lag features")

        history_target = history[self.target].reset_index(drop=True)
        direct_stat_feature_values = self._build_direct_stat_features_for_forecast(
            history_target
        )
        predictions = []
        max_steps = min(self.horizon, len(future))
        for horizon_step in range(1, max_steps + 1):
            horizon_feature_list = self.direct_feature_map.get(horizon_step)
            if horizon_feature_list is None:
                raise ValueError(
                    f"missing direct horizon feature list for step={horizon_step}"
                )
            # 当前 horizon 的未来外生特征
            future_exogenous_row = future.iloc[horizon_step - 1]
            # 当前 horizon 的预测特征：
            # 滞后特征来自预测起点的历史，外生特征来自对应 horizon 的未来
            feature_row = {}
            horizon_lag_features, horizon_other_features = self._split_feature_groups(
                horizon_feature_list
            )
            direct_stat_feature_names = set(self._build_direct_stat_feature_names())
            horizon_exogenous_features = [
                feature
                for feature in horizon_other_features
                if feature not in direct_stat_feature_names
            ]
            for lag_feature in horizon_lag_features:
                lag_step = int(lag_feature.split("_")[1])
                history_offset = lag_step - horizon_step
                feature_row[lag_feature] = history_target.iloc[-(history_offset + 1)]
            for feature in horizon_exogenous_features:
                feature_row[feature] = future_exogenous_row[feature]
            for feature in self._build_direct_stat_feature_names():
                feature_row[feature] = direct_stat_feature_values[feature]
            feature_df = pd.DataFrame([feature_row])
            scaler = None if scaler_features is None else scaler_features.get(horizon_step)
            if scaler is not None:
                feature_df.loc[:, feature_df.columns] = scaler.transform(feature_df)
            next_pred = models[horizon_step].predict(feature_df)[0]
            predictions.append(next_pred)
        return predictions
    # ##############################
    # 模型训练、预测
    # ##############################
    def train(self, data_X, data_Y, lgbm_params, df_train_full=None):
        """
        模型训练入口
        """
        if self.pred_method == "direct_output":
            return self._train_single_model(data_X, data_Y, lgbm_params)
        elif self.pred_method == "recursive":
            if not self.lags:
                raise ValueError("pred_method='recursive' requires non-empty lags")
            return self._train_single_model(data_X, data_Y, lgbm_params)
        elif self.pred_method == "direct":
            if not self.lags:
                raise ValueError("pred_method='direct' requires non-empty lags")
            if df_train_full is None:
                raise ValueError("df_train_full is required for direct training")
            return self._train_direct_horizon_models(
                df_train=df_train_full,
                feature_list=list(data_X.columns),
                lgbm_params=lgbm_params,
            )
        else:
            raise ValueError(f"unknown pred_method: {self.pred_method}")

    def forecast(self, model, df_train, scaler_features):
        # 未来数据处理
        (df_future, future_feature_list) = self.process_future_data()
        # 预测特征
        df_future = df_future.iloc[-self.horizon:, ]
        X_future = df_future.loc[:, future_feature_list]
        logger.info(f"{self.log_prefix} X_future.head(): \n {X_future.head()} \nX_future length: {len(X_future)} \nX_future.columns: {X_future.columns}")
        # 模型预测
        if len(X_future) > 0:
            if self.pred_method == "direct_output":
                # 单模型直接预测：仅使用未来外生特征
                if scaler_features is not None:
                    X_future_scaled = X_future.copy()
                    X_future_scaled.loc[:, X_future.columns] = (
                        scaler_features.transform(X_future)
                    )
                else:
                    X_future_scaled = X_future
                Y_future = model.predict(X_future_scaled)
            elif self.pred_method == "recursive":
                # 单模型递归预测：用前一步预测值更新下一步滞后特征
                Y_future = self._recursive_forecast(
                    model=model,
                    history=df_train,
                    future=X_future,
                    lags=self.lags,
                    steps=min(self.horizon, len(X_future)),
                    scaler_features=scaler_features,
                )
            elif self.pred_method == "direct":
                # 多模型直接预测：每个 horizon 使用独立模型直接输出
                Y_future = self._direct_horizon_forecast(
                    models=model,
                    history=df_train,
                    future=X_future,
                    scaler_features=scaler_features,
                )
            else:
                raise ValueError(f"unknown pred_method: {self.pred_method}")
            df_future[self.target] = Y_future
            logger.info(f"{self.log_prefix} df_future: \n{df_future.head()} \ndf_future length after forecast: {len(df_future)}")
        # 缺失值删除
        df_future.dropna(inplace=True, ignore_index=True)
        logger.info(f"{self.log_prefix} df_future: \n{df_future.head()}, \ndf_future length after dropna: {len(df_future)}")

        return df_future
    # ##############################
    # 程序流程
    # ##############################
    def process_output(self, df_future):
        for i in range(len(df_future)):
            df_future.loc[i, "id"] = (
                f"{self.node_id}_{self.out_system_id}_{df_future.loc[i, 'timeStamp'].strftime('%Y%m%d%H%M%S')}"
            )
            df_future.loc[i, "node_id"] = self.node_id
            # 区分 in_system_id 和 out_system_id
            df_future.loc[i, "system_id"] = self.out_system_id
            df_future.loc[i, "predict_value"] = str(df_future.loc[i, self.target])
            # df_future.loc[i, "predict_adjustable_amount"] = str(
            #     df_future.loc[i, self.target] * random.uniform(0.05, 0.1)
            # )
            df_future.loc[i, "count_data_time"] = df_future.loc[
                i, "timeStamp"
            ].strftime("%Y-%m-%d %H:%M:%S.%f")[
                :-3
            ]  # 保留毫秒并精确到前3位

        df_future = df_future[
            [
                "id",
                "node_id",
                "system_id",
                "predict_value",
                # "predict_adjustable_amount",
                "count_data_time",
            ]
        ]

        return df_future

    def run(self, input_data: Dict, model_cfgs: Dict):
        """
        实际负荷预测
        """
        # ------------------------------
        # 参数
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model Config...")
        logger.info(f"{80*'='}")
        # 项目配置
        self.node_id = model_cfgs["nodes"]["node"]["node_id"]
        self.out_system_id = model_cfgs["nodes"]["node"]["out_system_id"]
        # 数据配置
        self.target = "load"
        self.freq = "5min"
        self.before_days = model_cfgs["time_range"].get("before_days", 30)
        self.after_days = model_cfgs["time_range"].get("after_days", 1)
        self.input_data = input_data
        self.horizon = int(self.after_days * (24 * 60 / int(self.freq[:-3])) + 1)
        # 预测方法： direct_output, recursive, direct
        self.pred_method = "direct"
        # 数据分割时间
        self.start_time = model_cfgs["time_range"]["start_time"]
        self.now_time = model_cfgs["time_range"]["now_time"]
        self.future_time = model_cfgs["time_range"]["future_time"]
        # 特征工程
        self.lags = [] if self.pred_method == "direct_output" else self.DEFAULT_LAGS.copy()
        self.scale = False
        # 模型超参数
        self.lgbm_params = model_cfgs["lgbm_params"]
        # 其他
        self.horizon_metrics_df = pd.DataFrame()
        self.direct_feature_map = {}
        logger.info(f"{self.log_prefix} pred_method: {self.pred_method}")
        logger.info(f"{self.log_prefix} start_time: {self.start_time}")
        logger.info(f"{self.log_prefix} now_time: {self.now_time}")
        logger.info(f"{self.log_prefix} future_time: {self.future_time}")
        # ------------------------------
        # 历史数据处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Historical Data Processing...")
        logger.info(f"{80*'='}")
        (df_load, training_feature_list) = self.process_history_data()
        # ------------------------------
        # 模型训练
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model Training...")
        logger.info(f"{80*'='}")
        data_X = df_load[training_feature_list]
        data_Y = df_load[self.target]
        (model, 
         scaler_features, 
        #  test_scores
        ) = self.train(
            data_X=data_X,
            data_Y=data_Y,
            lgbm_params=self.lgbm_params,
            df_train_full=df_load,
        )
        # ------------------------------
        # 模型预测
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Model Forecast...")
        logger.info(f"{80*'='}")
        df_future = self.forecast(
            model=model,
            df_train=df_load,
            scaler_features=scaler_features,
        )
        # ------------------------------
        # 输出结果处理
        # ------------------------------
        logger.info(f"{80*'='}")
        logger.info(f"Forecast result processing...")
        logger.info(f"{80*'='}")
        df_power_future = self.process_output(df_future)

        # 模型输出
        return {"df_future": df_power_future}#, test_scores




# 测试代码 main 函数
def main():
    # ------------------------------
    # model configs
    # ------------------------------
    # 项目配置
    node_id = 1
    out_system_id = 1
    # 数据配置
    history_days = 30
    predict_days = 1
    # 数据分割时间
    start_time = None
    now_time = None
    future_time = None
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
            "before_days": -history_days,
            "after_days": predict_days,
        },
        # 模型超参数
        # "lgbm_params_others": {
        #     "boosting_type": "gbdt",
        #     "objective": "regression",
        #     "metric": "rmse",
        #     "max_bin": 31,
        #     "num_leaves": 31,
        #     "learning_rate": 0.05,
        #     "feature_fraction": 0.6,
        #     "bagging_fraction": 0.7,
        #     "bagging_freq": 5,
        #     "lambda_l1": 0.5,
        #     "lambda_l2": 0.5,
        #     "verbose": -1,
        # },
        "lgbm_params": {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "rmse",
            "n_estimators": 300,
            "max_depth": 6,
            "max_bin": 63,
            "num_leaves": 15,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 30,
            "lambda_l1": 1.0,
            "lambda_l2": 1.0,
            "verbose": -1,
        },
    }
    # ------------------------------
    # get data
    # ------------------------------
    data_dir = Path("./model/model_packages/DemandLoad_lingang/dataset/electricity/2026-01-01/lingang/demand_load/lingang_A/")
    df_power = pd.read_csv(data_dir.joinpath(f"df_power.csv"))
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
    test_scores_df = pd.DataFrame()
    # for now in pd.date_range("2025-10-31 00:00:00", "2026-01-01 00:00:00", freq="1d"):
    for now in pd.date_range("2025-10-31 00:00:00", "2025-10-31 00:00:00", freq="1d"):
        logger.info(f"now: {now}")
        # ------------------------------
        # 模型参数更新
        # ------------------------------
        # now = datetime.datetime(2025, 12, 31, 0, 0, 0)                                      # 模型预测的日期时间
        now_time = now.replace(tzinfo=None, minute=0, second=0, microsecond=0)                # 时间序列当前时刻
        start_time = now_time.replace(hour=0) - datetime.timedelta(days=history_days)         # 时间序列历史数据开始时刻
        future_time = now_time + datetime.timedelta(days=predict_days)                        # 时间序列未来结束时刻
        logger.info(f"history data time range: {start_time} ~ {now_time}")
        logger.info(f"predict data time range: {now_time} ~ {future_time}")
        model_cfgs["time_range"]["start_time"] = start_time
        model_cfgs["time_range"]["now_time"] = now_time
        model_cfgs["time_range"]["future_time"] = future_time
        # ------------------------------
        # 模型测试
        # ------------------------------
        model_ins = ModelMainClass(
            project="test",
            model="test",
            node="test",
            args={},
        )
        (result, 
        #  test_scores
        ) = model_ins.run(input_data, model_cfgs)
    """
        # 模型测试结果
        test_scores_df_temp = pd.DataFrame(test_scores, index=[now.date()])
        test_scores_df_temp["pred_method"] = model_ins.pred_method
        test_scores_df = pd.concat([test_scores_df, test_scores_df_temp], axis=0)
    # 模型测试结果保存
    results_dir = Path("./model/model_packages/DemandLoad_lingang/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    test_scores_df.to_csv(results_dir.joinpath(f"test_scores_df-{model_ins.pred_method}2.csv"), encoding="utf-8", index=True)
    """
if __name__ == "__main__":
    main()
