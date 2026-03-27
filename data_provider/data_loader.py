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
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cab

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
            "target_series": None,
            "date_history": None,
            "date_future": None,
            "weather_history": None,
            "weather_future": None,
        }
        # ------------------------------
        # 加载目标时间序列数据
        # ------------------------------
        target_data_path = self.args.data_dir / self.args.data_path
        if target_data_path.exists():
            df_target = pd.read_csv(target_data_path)
            input_data["target_series"] = df_target
            logger.info(f"{self.log_prefix} Target series loaded: {df_target.shape}")
            logger.info(f"{self.log_prefix} Target series missing values: \n{df_target.isna().sum()}")
        else:
            logger.error(f"{self.log_prefix} Target data not found at {target_data_path}")
            raise FileNotFoundError(f"Target data not found at {target_data_path}")
        # ------------------------------
        # 加载日期类型数据
        # ------------------------------
        # 加载历史日期类型数据
        if self.args.date_history_path:
            date_history_path = self.args.data_dir / self.args.date_history_path
            if date_history_path.exists():
                df_date_history = pd.read_csv(date_history_path)
                input_data["date_history"] = df_date_history
                logger.info(f"{self.log_prefix} Date history loaded: {df_date_history.shape}")
                logger.info(f"{self.log_prefix} Date history missing values: \n{df_date_history.isna().sum()}")
        # 加载未来日期类型数据
        if self.args.date_future_path:
            date_future_path = self.args.data_dir / self.args.date_future_path
            if date_future_path.exists():
                df_date_future = pd.read_csv(date_future_path)
                input_data["date_future"] = df_date_future
                logger.info(f"{self.log_prefix} Date future loaded: {df_date_future.shape}")
                logger.info(f"{self.log_prefix} Date future missing values: \n{df_date_future.isna().sum()}")
        # date 历史和未来数据拼接
        if self.args.date_history_path and self.args.date_future_path:
            df_date_all = pd.concat([df_date_history.iloc[:-1,], df_date_future], axis=0)
        else:
            df_date_all = None
        # 数据收集
        input_data["date_history"] = df_date_all
        input_data["date_future"] = df_date_all
        # ------------------------------
        # 加载气象数据
        # ------------------------------
        # 加载历史气象数据
        if self.args.weather_history_path:
            weather_history_path = self.args.data_dir / self.args.weather_history_path
            if weather_history_path.exists():
                df_weather_history = pd.read_csv(weather_history_path)
                input_data["weather_history"] = df_weather_history
                logger.info(f"{self.log_prefix} Weather history loaded: {df_weather_history.shape}")
                logger.info(f"{self.log_prefix} Weather history missing values: \n{df_weather_history.isna().sum()}")
        # 加载未来气象数据
        if self.args.weather_future_path:
            weather_future_path = self.args.data_dir / self.args.weather_future_path
            if weather_future_path.exists():
                df_weather_future = pd.read_csv(weather_future_path)
                input_data["weather_future"] = df_weather_future
                logger.info(f"{self.log_prefix} Weather future loaded: {df_weather_future.shape}")
                logger.info(f"{self.log_prefix} Weather future missing values: \n{df_weather_future.isna().sum()}")
        # weather 历史和未来数据拼接
        if self.args.weather_history_path and self.args.weather_future_path:
            df_weather_all = pd.concat([df_weather_history.iloc[:-1,], df_weather_future], axis=0)
        else:
            df_weather_all = None
        # 数据收集
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
            # Global 模式下透传 series_id（若存在）
            if getattr(self.args, "enable_global_training", False):
                series_id_col = getattr(self.args, "series_id_feature", "series_id")
                if series_id_col in df_series.columns:
                    df_template_copy[series_id_col] = (
                        series_indexed[series_id_col]
                        .astype("category")
                        .reindex(df_template_copy["time"])
                        .reset_index(drop=True)
                    )
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
        logger.info(f"{self.log_prefix} df_history_template: \n{df_history_template.head()}")
        logger.info(f"{self.log_prefix} df_history_template shape: {df_history_template.shape}")
        # 数据预处理：目标时间序列特征
        df_history_series = self.__process_df_timestamp(df=input_data["target_series"], col_ts=self.args.target_ts_feat)
        logger.info(f"{self.log_prefix} after __process_df_timestamp df_history_series: \n{df_history_series.head()}")
        logger.info(f"{self.log_prefix} after __process_df_timestamp df_history_series shape: {df_history_series.shape}")
        df_history, other_endogenous_features, target_feature = self.__process_target_series(
            df_template=df_history_template,
            df_series=df_history_series,
            col_ts=self.args.target_ts_feat,
            col_numeric=self.args.target_series_numeric_features,
            col_categorical=self.args.target_series_categorical_features,
            col_drop=self.args.target_series_drop_features,
        )
        # 若配置时间窗与目标序列不重叠，导致目标全空，则自动回退到最近可用历史窗口
        if target_feature and target_feature in df_history.columns and df_history[target_feature].notna().sum() == 0:
            logger.warning(
                f"{self.log_prefix} target is empty in configured history window "
                f"[{self.train_start_time}, {self.train_end_time}); fallback to latest available history."
            )
            target_source_col = self.args.target
            ts_source_col = self.args.target_ts_feat
            df_series_valid = df_history_series.copy()
            if target_source_col in df_series_valid.columns:
                df_series_valid[target_source_col] = pd.to_numeric(df_series_valid[target_source_col], errors="coerce")
                df_series_valid = df_series_valid.dropna(subset=[target_source_col]).sort_values(ts_source_col)
                if df_series_valid.empty:
                    raise ValueError(f"{self.log_prefix} No valid target values found in source series.")
                fallback_rows = min(len(df_history_template), len(df_series_valid))
                df_series_valid = df_series_valid.tail(fallback_rows).reset_index(drop=True)
                df_series_valid = df_series_valid.rename(
                    columns={
                        ts_source_col: "time",
                        target_source_col: "y",
                    }
                )
                keep_cols = ["time", "y"] + [col for col in other_endogenous_features if col in df_series_valid.columns]
                df_history = df_series_valid[keep_cols].copy()
                target_feature = "y"
            else:
                raise ValueError(f"{self.log_prefix} Target column '{target_source_col}' does not exist in source series.")
        logger.info(f"{self.log_prefix} after __process_target_series df_history: \n{df_history.head()}")
        logger.info(f"{self.log_prefix} after __process_target_series df_history shape: {df_history.shape}")
        # 所有内生变量(包含目标特征 y)
        endogenous_features_with_target = other_endogenous_features + [target_feature] if target_feature else other_endogenous_features
        logger.info(f"{self.log_prefix} endogenous_features_with_target: {endogenous_features_with_target}")
        logger.info(f"{self.log_prefix}                  target_feature: {target_feature}")
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_history = self.__process_df_timestamp(df=input_data[f"date_history"], col_ts=self.args.date_ts_feat)
        if df_date_history is not None:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history: \n{df_date_history}")
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history shape: {df_date_history.shape}")
        else:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history: {df_date_history}")
        # 特征工程：天气特征
        df_weather_history = self.__process_df_timestamp(df=input_data[f"weather_history"], col_ts=self.args.weather_ts_feat)
        if df_weather_history is not None:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history: \n{df_weather_history}")
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history shape: {df_weather_history.shape}")
        else:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history: {df_weather_history}")

        return (df_history, df_date_history, df_weather_history, endogenous_features_with_target, target_feature)

    def process_future_data(self, input_data: Dict):
        """
        处理未来预测阶段所需的外生数据。

        当前预测链路只需要未来时间模板，以及与时间对齐的日期/天气特征。
        未来目标序列或未来内生变量不在此处加载，递归预测所需的目标值会在
        预测过程中由模型输出逐步回填。
        """
        # 未来数据时间戳
        df_future_template = pd.DataFrame({"time": pd.date_range(self.forecast_start_time, self.forecast_end_time, freq=self.args.freq, inclusive="left")})
        logger.info(f"{self.log_prefix} df_future_template: \n{df_future_template.head()}")
        logger.info(f"{self.log_prefix} df_future_template shape: {df_future_template.shape}")
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_future = self.__process_df_timestamp(df=input_data[f"date_future"], col_ts=self.args.date_ts_feat)
        if df_date_future is not None:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_date_future: \n{df_date_future}")
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_date_future shape: {df_date_future.shape}")
        else:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_date_future: {df_date_future}")
        # 特征工程：天气特征
        df_weather_future = self.__process_df_timestamp(df=input_data[f"weather_future"], col_ts=self.args.weather_ts_feat)
        if df_weather_future is not None:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_weather_future: \n{df_weather_future}")
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_weather_future shape: {df_weather_future.shape}")
        else:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_weather_future: {df_weather_future}")

        return (df_future_template, df_date_future, df_weather_future)


def _to_single_output_label(y: Any) -> Optional[np.ndarray]:
    """
    将标签统一转成单输出 1D numpy 数组。
    多输出场景返回 None，由上层决定跳过原生容器封装。
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            return None
        return y.iloc[:, 0].to_numpy()
    if isinstance(y, pd.Series):
        return y.to_numpy()

    y_arr = np.asarray(y)
    if y_arr.ndim == 1:
        return y_arr
    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        return y_arr.reshape(-1)
    return None


def _catboost_cat_feature_indices(X: pd.DataFrame, categorical_features: Optional[List[str]]) -> List[int]:
    categorical_features = categorical_features or []
    return [X.columns.get_loc(col) for col in categorical_features if col in X.columns]


def _xgboost_feature_types(X: pd.DataFrame, categorical_features: Optional[List[str]]) -> List[str]:
    categorical_set = set(categorical_features or [])
    return ["c" if col in categorical_set else "q" for col in X.columns]


def prepare_native_train_eval_datasets(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: Any,
    X_eval: Optional[pd.DataFrame] = None,
    y_eval: Optional[Any] = None,
    categorical_features: Optional[List[str]] = None,
    sample_weight: Optional[Any] = None,
    eval_sample_weight: Optional[Any] = None,
    free_raw_data: bool = False,
) -> Dict[str, Any]:
    """
    为 LightGBM / XGBoost / CatBoost 构建原生数据容器。

    说明：
    - LightGBM: Dataset
    - XGBoost: DMatrix
    - CatBoost: Pool
    - 当前仅对单输出训练启用；多输出场景保持 sklearn DataFrame/ndarray 主链路
    """
    mt = str(model_type).lower()
    y_train_1d = _to_single_output_label(y_train)
    y_eval_1d = _to_single_output_label(y_eval) if y_eval is not None else None

    if y_train_1d is None:
        return {
            "enabled": False,
            "framework": mt,
            "train_native": None,
            "eval_native": None,
            "reason": "multi_output_not_supported",
        }

    categorical_features = [col for col in (categorical_features or []) if col in X_train.columns]

    try:
        if mt in ["lightgbm", "lgb"]:
            train_native = lgb.Dataset(
                data=X_train,
                label=y_train_1d,
                weight=sample_weight,
                feature_name=list(X_train.columns),
                categorical_feature=categorical_features or "auto",
                free_raw_data=free_raw_data,
            )
            eval_native = None
            if X_eval is not None and y_eval_1d is not None:
                eval_native = lgb.Dataset(
                    data=X_eval,
                    label=y_eval_1d,
                    reference=train_native,
                    weight=eval_sample_weight,
                    feature_name=list(X_eval.columns),
                    categorical_feature=categorical_features or "auto",
                    free_raw_data=free_raw_data,
                )
            return {
                "enabled": True,
                "framework": "lightgbm",
                "train_native": train_native,
                "eval_native": eval_native,
                "reason": None,
            }

        if mt in ["xgboost", "xgb"]:
            feature_names = list(X_train.columns)
            train_native = xgb.DMatrix(
                data=X_train,
                label=y_train_1d,
                weight=sample_weight,
                feature_names=feature_names,
                feature_types=_xgboost_feature_types(X_train, categorical_features),
                enable_categorical=bool(categorical_features),
            )
            eval_native = None
            if X_eval is not None and y_eval_1d is not None:
                eval_native = xgb.DMatrix(
                    data=X_eval,
                    label=y_eval_1d,
                    weight=eval_sample_weight,
                    feature_names=list(X_eval.columns),
                    feature_types=_xgboost_feature_types(X_eval, categorical_features),
                    enable_categorical=bool(categorical_features),
                )
            return {
                "enabled": True,
                "framework": "xgboost",
                "train_native": train_native,
                "eval_native": eval_native,
                "reason": None,
            }

        if mt in ["catboost", "cat"]:
            cat_feature_indices = _catboost_cat_feature_indices(X_train, categorical_features)
            train_native = cab.Pool(
                data=X_train,
                label=y_train_1d,
                weight=sample_weight,
                feature_names=list(X_train.columns),
                cat_features=cat_feature_indices or None,
            )
            eval_native = None
            if X_eval is not None and y_eval_1d is not None:
                eval_native = cab.Pool(
                    data=X_eval,
                    label=y_eval_1d,
                    weight=eval_sample_weight,
                    feature_names=list(X_eval.columns),
                    cat_features=_catboost_cat_feature_indices(X_eval, categorical_features) or None,
                )
            return {
                "enabled": True,
                "framework": "catboost",
                "train_native": train_native,
                "eval_native": eval_native,
                "reason": None,
            }

        return {
            "enabled": False,
            "framework": mt,
            "train_native": None,
            "eval_native": None,
            "reason": "unsupported_model_type",
        }
    except Exception as exc:
        return {
            "enabled": False,
            "framework": mt,
            "train_native": None,
            "eval_native": None,
            "reason": f"native_dataset_build_failed: {exc}",
        }




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
