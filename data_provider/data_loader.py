# -*- coding: utf-8 -*-

# ***************************************************
# * File        : data_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-25
# * Version     : 1.0.022513
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import copy
from pathlib import Path
from typing import List, Dict

import pandas as pd
import lightgbm as lgb

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DataLoader:
    
    def __init__(self, 
                 args, 
                 train_start_time,
                 train_end_time,
                 forecast_start_time,
                 forecast_end_time, 
                 log_prefix: str):
        self.args = args
        self.train_start_time = train_start_time
        self.train_end_time = train_end_time
        self.forecast_start_time = forecast_start_time
        self.forecast_end_time = forecast_end_time
        self.log_prefix = log_prefix

    def load_data(self) -> Dict:
        """
        加载所有必要的数据
        
        Returns:
            包含目标序列、日期类型、天气等数据的字典
        """
        logger.info(f"{self.log_prefix} Loading data from {self.args.data_dir}")
        
        input_data = {
            "date_history": None,
            "date_future": None,
            "weather_history": None,
            "weather_future": None,
        }
        
        # 加载目标时间序列数据
        target_data_path = self.args.data_dir / self.args.data_path
        if target_data_path.exists():
            df_target = pd.read_csv(target_data_path)
            # df_target[self.args.target_ts_feat] = pd.to_datetime(df_target[self.args.target_ts_feat])
            input_data["target_series"] = df_target
            logger.info(f"{self.log_prefix} Target series loaded: {df_target.shape}")
        else:
            logger.error(f"{self.log_prefix} Target data not found at {target_data_path}")
            raise FileNotFoundError(f"Target data not found at {target_data_path}")
        
        # 加载历史日期类型数据
        if self.args.date_history_path:
            date_history_path = self.args.data_dir / self.args.date_history_path
            if date_history_path.exists():
                df_date_history = pd.read_csv(date_history_path)
                # df_date_history[self.args.date_ts_feat] = pd.to_datetime(df_date_history[self.args.date_ts_feat])
                input_data["date_history"] = df_date_history
                logger.info(f"{self.log_prefix} Date history loaded: {df_date_history.shape}")
        
        # 加载未来日期类型数据
        if self.args.date_future_path:
            date_future_path = self.args.data_dir / self.args.date_future_path
            if date_future_path.exists():
                df_date_future = pd.read_csv(date_future_path)
                # df_date_future[self.args.date_ts_feat] = pd.to_datetime(df_date_future[self.args.date_ts_feat])
                input_data["date_future"] = df_date_future
                logger.info(f"{self.log_prefix} Date future loaded: {df_date_future.shape}")
        
        # 加载历史天气数据
        if self.args.weather_history_path:
            weather_history_path = self.args.data_dir / self.args.weather_history_path
            if weather_history_path.exists():
                df_weather_history = pd.read_csv(weather_history_path)
                # df_weather_history[self.args.weather_ts_feat] = pd.to_datetime(df_weather_history[self.args.weather_ts_feat])
                input_data["weather_history"] = df_weather_history
                logger.info(f"{self.log_prefix} Weather history loaded: {df_weather_history.shape}")
        
        # 加载未来天气数据
        if self.args.weather_future_path:
            weather_future_path = self.args.data_dir / self.args.weather_future_path
            if weather_future_path.exists():
                df_weather_future = pd.read_csv(weather_future_path)
                # df_weather_future[self.args.weather_ts_feat] = pd.to_datetime(df_weather_future[self.args.weather_ts_feat])
                input_data["weather_future"] = df_weather_future
                logger.info(f"{self.log_prefix} Weather future loaded: {df_weather_future.shape}")
        # ------------------------------
        # 数据合并
        # ------------------------------
        if self.args.date_history_path and self.args.date_future_path:
            df_date_all = pd.concat([df_date_history.iloc[:-1,], df_date_future], axis=0)
        else:
            df_date_all = None
        
        if self.args.weather_history_path and self.args.weather_future_path:
            df_weather_all = pd.concat([df_weather_history.iloc[:-1,], df_weather_future], axis=0)
        else:
            df_weather_all = None
        # 数据收集
        input_data["date_history"] = df_date_all
        input_data["date_future"] = df_date_all
        input_data["weather_history"] = df_weather_all
        input_data["weather_future"] = df_weather_all
        
        return input_data

    def __process_df_timestamp(self, df: pd.DataFrame, col_ts: str):
        """
        时序数据时间特征预处理

        Args:
            df (pd.DataFrame): 时间序列数据
            col_ts (str): 原时间戳列
        """
        if df is not None:
            # 数据拷贝
            df_processed = copy.deepcopy(df)
            # 转换时间戳类型
            df_processed[col_ts] = pd.to_datetime(df_processed[col_ts])
            # del df_processed[ts_col]
            # 去除重复时间戳
            df_processed.drop_duplicates(subset=col_ts, keep="last", inplace=True, ignore_index=True)
            return df_processed
        else:
            return df

    def __process_target_series(self, df_template: pd.DataFrame, df_series: pd.DataFrame, col_ts: str, col_numeric: List, col_categorical: List, col_drop: List):
        """
        目标特征数据预处理
        """
        df_template_copy = df_template.copy()
        if df_series is not None:
            # 目标特征数据转换为浮点数
            series_indexed = df_series.set_index(col_ts)
            if self.args.target in df_series.columns:
                series_indexed[self.args.target] = pd.to_numeric(series_indexed[self.args.target], errors="coerce")
                df_template_copy["y"] = df_template_copy["time"].map(series_indexed[self.args.target])
                target_feature = "y"
            else:
                target_feature = None
                logger.warning(f"{self.log_prefix} Target column '{self.args.target}' does not exist.")
            # 除目标特征外的其他数值类型的内生变量处理
            filtered_col_numeric = [col for col in col_numeric if col not in [col_ts, self.args.target] + col_categorical + col_drop]
            existing_col_numeric = [col for col in filtered_col_numeric if col in df_series.columns]
            if existing_col_numeric:
                for col in filtered_col_numeric:
                    series_indexed[col] = pd.to_numeric(series_indexed[col], errors="coerce")
                    df_template_copy[col] = df_template_copy["time"].map(series_indexed[col])
            # 除目标特征外的其他类别类型的内生变量处理
            filtered_col_categorical = [col for col in col_categorical if col not in [col_ts, self.args.target] + col_numeric + col_drop]
            existing_col_categorical = [col for col in filtered_col_categorical if col in df_series.columns]
            if existing_col_categorical:
                series_indexed[existing_col_categorical] = series_indexed[existing_col_categorical].astype("category")
                df_template_copy[existing_col_categorical] = series_indexed[existing_col_categorical].reindex(df_template_copy["time"]).reset_index(drop=True)
            # 内生变量(Endogenous variable)
            endogenous_features = [col for col in df_template_copy.columns if col not in ["time"]]
            if target_feature and target_feature in endogenous_features:
                 endogenous_features.remove(target_feature)
        else:
            endogenous_features = []
            target_feature = None
        
        return df_template_copy, endogenous_features, target_feature

    def process_history_data(self, input_data: Dict):
        """
        历史数据预处理
        """
        # 历史数据时间戳
        df_history_template = pd.DataFrame({"time": pd.date_range(self.train_start_time, self.train_end_time, freq=self.args.freq, inclusive="left")})
        logger.info(f"{self.log_prefix} template df_history_template: \n{df_history_template}")
        # 数据预处理：目标时间序列特征
        df_history_series = self.__process_df_timestamp(df=input_data["target_series"], col_ts=self.args.target_ts_feat)
        logger.info(f"{self.log_prefix} after __process_df_timestamp df_history_series: \n{df_history_series.head()}")
        df_history, other_endogenous_features, target_feature = self.__process_target_series(
            df_template=df_history_template,
            df_series=df_history_series,
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        logger.info(f"{self.log_prefix} after __process_target_series df_history: \n{df_history.head()}")
        # 所有内生变量(包含目标特征 y)
        endogenous_features_with_target = other_endogenous_features + [target_feature] if target_feature else other_endogenous_features
        logger.info(f"{self.log_prefix} endogenous_features_with_target: {endogenous_features_with_target}")
        logger.info(f"{self.log_prefix} target_feature: {target_feature}")
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_history = self.__process_df_timestamp(df=input_data[f"date_history"], col_ts=self.args.date_ts_feat)
        if df_date_history:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history: \n{df_date_history}")
        else:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history: {df_date_history}")
        # 特征工程：天气特征
        df_weather_history = self.__process_df_timestamp(df=input_data[f"weather_history"], col_ts=self.args.weather_ts_feat)
        if df_weather_history:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history: \n{df_weather_history}")
        else:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history: {df_weather_history}")

        return (df_history, df_date_history, df_weather_history, endogenous_features_with_target, target_feature)

    def process_future_data(self, input_data: Dict):
        """
        处理未来数据
        """
        # 未来数据时间戳
        df_future_template = pd.DataFrame({"time": pd.date_range(self.forecast_start_time, self.forecast_end_time, freq=self.args.freq, inclusive="left")})
        logger.info(f"{self.log_prefix} template df_future_template: \n{df_future_template}")
        """
        # 数据预处理：目标时间序列特征
        df_future_series = self.__process_df_timestamp(df=input_data["df_future_series"], col_ts=self.args.target_ts_feat)
        logger.info(f"{self.log_prefix} after process_df_timestamp df_future_series: \n{df_future_series.head()}")

        df_future, other_endogenous_features, target_feature = self.__process_target_series(
            df_template=df_future_template,
            df_series=df_future_series,
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        logger.info(f"{self.log_prefix} after process_target_series df_future: \n{df_future.head()}")
        # 所有内生变量(没有目标特征 y及其衍生特征)
        endogenous_features_for_lag = other_endogenous_features
        """
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_future = self.__process_df_timestamp(df=input_data[f"date_future"], col_ts=self.args.date_ts_feat)
        if df_date_future:
            logger.info(f"{self.log_prefix} after process_df_timestamp df_date_future: \n{df_date_future}")
        else:
            logger.info(f"{self.log_prefix} after process_df_timestamp df_date_future: {df_date_future}")
        # 特征工程：天气特征
        df_weather_future = self.__process_df_timestamp(df=input_data[f"weather_future"], col_ts=self.args.weather_ts_feat)
        if df_weather_future:
            logger.info(f"{self.log_prefix} after process_df_timestamp df_weather_future: \n{df_weather_future}")
        else:
            logger.info(f"{self.log_prefix} after process_df_timestamp df_weather_future: {df_weather_future}")

        return (df_future_template, df_date_future, df_weather_future)


# TODO 未使用
def get_lgb_train_test_data(train_path, test_path, weight_paths = []):
    """
    读取 LightGBM example demo 数据
    """
    # read data
    df_train = pd.read_csv(train_path, header = None, sep = "\t")
    df_test = pd.read_csv(test_path, header = None, sep = "\t")
    # print(df_train.head())
    # print(df_test.head())

    # split data
    y_train = df_train[0]
    y_test = df_test[0]
    X_train = df_train.drop(0, axis = 1)
    X_test = df_test.drop(0, axis = 1)

    # weight data
    if weight_paths != []:
        W_train = pd.read_csv(weight_paths[0], header = None)[0]
        W_test = pd.read_csv(weight_paths[1], header = None)[0]
        # lightgbm Dataset
        lgb_train = lgb.Dataset(X_train, y_train, weight = W_train, free_raw_data = False)
        lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train, weight = W_test, free_raw_data = False)
        return W_train, W_test, X_train, y_train, X_test, y_test, lgb_train, lgb_eval
    else:
        # lightgbm Dataset
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference = lgb_train)
        return X_train, y_train, X_test, y_test, lgb_train, lgb_eval




# 测试代码 main 函数
def main():
    series = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
        header = 0,
        names = ["Month", "Sales"],
        index_col = None,
        parse_dates = False, 
        date_format = None,
    )
    series["Month"] = series["Month"].apply(lambda x: pd.to_datetime("190" + x, format = "%Y-%m"))
    print(series)
    print(series.info())

if __name__ == "__main__":
    main()
