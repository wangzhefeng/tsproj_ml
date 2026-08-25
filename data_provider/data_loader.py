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

from models.multistep.spec import InputScope, get_strategy_spec
from models.multistep.panel import (
    materialize_panel_future,
    materialize_panel_history,
)
from utils.log_util import logger
from utils.exogenous_contract import (
    select_asof_rows,
    split_role_frames,
    validate_daily_coverage,
)
from utils.weather_contract import (
    validate_weather_availability,
    validate_weather_coverage,
    validate_weather_information_contract,
)

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def materialize_custom_future_sources(
    custom_history: Optional[List[Dict[str, Any]]],
    custom_future: Optional[List[Dict[str, Any]]],
    future_times,
    cutoff,
) -> List[Dict[str, Any]]:
    """按 custom source 的 future_strategy 构造预测期外生帧。

    ``explicit``（默认）保留显式 future 文件；``freeze_last_observation``
    只读取 cutoff 及以前的最后一条历史状态，并重复到全部 future_times。
    """
    history_sources = list(custom_history or [])
    future_sources = list(custom_future or [])
    history_by_name = {str(source.get("name", "custom")): source for source in history_sources}
    future_by_name = {str(source.get("name", "custom")): source for source in future_sources}
    names = list(dict.fromkeys([*history_by_name, *future_by_name]))
    resolved = []
    future_index = pd.DatetimeIndex(pd.to_datetime(future_times))
    cutoff_ts = pd.Timestamp(cutoff)

    for name in names:
        history_source = history_by_name.get(name)
        future_source = future_by_name.get(name)
        strategy_source = history_source or future_source or {}
        strategy = str(strategy_source.get("future_strategy", "explicit") or "explicit").lower()
        if strategy == "explicit":
            if future_source is not None:
                if bool(future_source.get("strict_information_set", False)):
                    ts_col = future_source.get("ts_col")
                    available_at_col = str(
                        future_source.get("available_at_col", "available_at") or "available_at"
                    )
                    selected = select_asof_rows(
                        future_source.get("df"),
                        expected_times=future_index,
                        forecast_origin=cutoff_ts,
                        ts_col=ts_col,
                        available_at_col=available_at_col,
                        label=f"Custom[{name}] future",
                    )
                    columns = list(future_source.get("columns") or [])
                    if selected[columns].isna().to_numpy().any():
                        raise ValueError(f"Custom[{name}] future contains missing feature value(s).")
                    resolved.append({**future_source, "df": selected})
                else:
                    resolved.append(future_source)
            continue
        if strategy != "freeze_last_observation":
            raise ValueError(
                f"Custom source '{name}' has unsupported future_strategy='{strategy}'."
            )
        if history_source is None:
            raise ValueError(
                f"Custom source '{name}' requires history data for freeze_last_observation."
            )

        ts_col = history_source.get("ts_col")
        columns = list(history_source.get("columns") or [])
        history_frame = history_source.get("df")
        if not ts_col or history_frame is None or history_frame.empty:
            raise ValueError(
                f"Custom source '{name}' has no usable history for freeze_last_observation."
            )
        frame = history_frame.copy()
        frame[ts_col] = pd.to_datetime(frame[ts_col])
        frame = frame[frame[ts_col] <= cutoff_ts].sort_values(ts_col)
        if frame.empty:
            raise ValueError(
                f"Custom source '{name}' has no history at or before cutoff {cutoff_ts}."
            )
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Custom source '{name}' missing columns {missing}.")

        last_row = frame.iloc[-1]
        frozen = pd.DataFrame({ts_col: future_index})
        for column in columns:
            frozen[column] = last_row[column]
        resolved.append({**history_source, "df": frozen})

    return resolved


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

    def _is_univariate_method(self) -> bool:
        return (
            get_strategy_spec(getattr(self.args, "pred_method", "")).input_scope
            == InputScope.TARGET_ONLY
        )

    def _load_optional_frame(self, relative_path: Optional[str], label: str) -> Optional[pd.DataFrame]:
        if not relative_path:
            return None
        data_path = self.args.data_dir / relative_path
        if not data_path.exists():
            return None
        df = pd.read_csv(data_path)
        logger.info(f"{self.log_prefix} {label} loaded: {df.shape}")
        logger.info(f"{self.log_prefix} {label} missing values: \n{df.isna().sum()}")
        return df

    def _prepare_exogenous_splits(
        self,
        history_df: Optional[pd.DataFrame],
        future_df: Optional[pd.DataFrame],
        ts_col: Optional[str],
        label: str,
        strict_roles: bool = False,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        if history_df is None and future_df is None:
            return None, None, None
        if ts_col is None:
            raise ValueError(f"{self.log_prefix} {label} timestamp column is required for exogenous slicing.")

        if strict_roles:
            if history_df is None or future_df is None:
                raise ValueError(
                    f"{self.log_prefix} {label} strict role split requires both history and future data."
                )
            history_slice, future_slice = split_role_frames(
                history_df,
                future_df,
                ts_col=ts_col,
                forecast_start=self.forecast_start_time,
                label=label,
            )
            logger.info(f"{self.log_prefix} {label} strict history shape: {history_slice.shape}")
            logger.info(f"{self.log_prefix} {label} strict future shape: {future_slice.shape}")
            return None, history_slice, future_slice

        frames = [df.copy() for df in [history_df, future_df] if df is not None]
        canonical_df = pd.concat(frames, axis=0, ignore_index=True)
        canonical_df[ts_col] = pd.to_datetime(canonical_df[ts_col])
        canonical_df = canonical_df.sort_values(by=[ts_col]).drop_duplicates(
            subset=ts_col,
            keep="last",
        ).reset_index(drop=True)

        forecast_start = pd.Timestamp(self.forecast_start_time)
        history_slice = canonical_df[canonical_df[ts_col] <= forecast_start].copy().reset_index(drop=True)
        future_slice = canonical_df[canonical_df[ts_col] >= forecast_start].copy().reset_index(drop=True)

        logger.info(f"{self.log_prefix} {label} canonical shape after merge/dedup: {canonical_df.shape}")
        logger.info(f"{self.log_prefix} {label} history slice shape: {history_slice.shape}")
        logger.info(f"{self.log_prefix} {label} future slice shape: {future_slice.shape}")

        return canonical_df, history_slice, future_slice

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
            "weather_backtest": None,
            "weather_future": None,
            "custom_history": [],
            "custom_future": [],
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
        df_date_history_raw = self._load_optional_frame(self.args.date_history_path, "Date history")
        df_date_future_raw = self._load_optional_frame(self.args.date_future_path, "Date future")
        (
            _df_date_all,
            df_date_history,
            df_date_future,
        ) = self._prepare_exogenous_splits(
            history_df=df_date_history_raw,
            future_df=df_date_future_raw,
            ts_col=self.args.date_ts_feat,
            label="Date",
            strict_roles=bool(getattr(self.args, "strict_date_information_set", False)),
        )
        input_data["date_history"] = df_date_history
        input_data["date_future"] = df_date_future
        # ------------------------------
        # 加载气象数据
        # ------------------------------
        validate_weather_information_contract(self.args)
        df_weather_history_raw = self._load_optional_frame(self.args.weather_history_path, "Weather history")
        df_weather_backtest_raw = self._load_optional_frame(
            getattr(self.args, "weather_backtest_path", None),
            "Weather backtest",
        )
        df_weather_future_raw = self._load_optional_frame(self.args.weather_future_path, "Weather future")
        if bool(getattr(self.args, "strict_weather_information_set", False)):
            if df_weather_backtest_raw is not None:
                # Backtest天气的发布时间必须相对每个CV fold原点校验；这里仅验证
                # provenance列可解析。日级与月频共用同一as-of规则，不能在全局
                # load阶段套用“早于目标月”这一月频特例。
                validate_weather_availability(
                    df_weather_backtest_raw,
                    ts_col=self.args.weather_ts_feat,
                    label="Backtest weather",
                )
            if df_weather_future_raw is not None:
                validate_weather_availability(
                    df_weather_future_raw,
                    ts_col=self.args.weather_ts_feat,
                    label="Future weather",
                    forecast_origin=self.args.now_time,
                )
        (
            _df_weather_all,
            df_weather_history,
            df_weather_future,
        ) = self._prepare_exogenous_splits(
            history_df=df_weather_history_raw,
            future_df=df_weather_future_raw,
            ts_col=self.args.weather_ts_feat,
            label="Weather",
            strict_roles=bool(getattr(self.args, "strict_weather_information_set", False)),
        )
        input_data["weather_history"] = df_weather_history
        input_data["weather_backtest"] = df_weather_backtest_raw
        input_data["weather_future"] = df_weather_future
        # ------------------------------
        # 加载自定义外生特征（注册表，多来源）
        # ------------------------------
        custom_sources = getattr(self.args, "custom_features", None) or []
        for source in custom_sources:
            name = str(source.get("name") or source.get("history_path") or "custom")
            ts_col = source.get("ts_col")
            if not ts_col:
                raise ValueError(f"{self.log_prefix} Custom source '{name}' missing ts_col.")
            columns = list(source.get("columns") or [])
            if not columns:
                raise ValueError(f"{self.log_prefix} Custom source '{name}' missing columns.")
            categorical_columns = list(source.get("categorical_columns") or [])
            df_history_raw = self._load_optional_frame(source.get("history_path"), f"Custom[{name}] history")
            df_future_raw = self._load_optional_frame(source.get("future_path"), f"Custom[{name}] future")
            _all, df_history, df_future = self._prepare_exogenous_splits(
                history_df=df_history_raw,
                future_df=df_future_raw,
                ts_col=ts_col,
                label=f"Custom[{name}]",
                strict_roles=bool(source.get("strict_information_set", False)),
            )
            base = {
                "name": name,
                "ts_col": ts_col,
                "columns": columns,
                "categorical_columns": categorical_columns,
                "future_strategy": str(source.get("future_strategy", "explicit") or "explicit"),
                # custom 数据在每个时间戳何时可获得：
                # - contemporaneous（默认）：时间戳当期开始即已知；
                # - end_of_period：当期结束后才可得（如当日完整负荷状态）。
                "availability": str(source.get("availability", "contemporaneous") or "contemporaneous"),
                "strict_information_set": bool(source.get("strict_information_set", False)),
                "available_at_col": str(source.get("available_at_col", "available_at") or "available_at"),
            }
            if df_history is not None and not df_history.empty:
                input_data["custom_history"].append({**base, "df": df_history})
            if df_future is not None and not df_future.empty:
                input_data["custom_future"].append({**base, "df": df_future})

        return input_data

    def __process_df_timestamp(self, df: pd.DataFrame, col_ts: str):
        """
        时序数据时间特征预处理

        Args:
            df (pd.DataFrame): 时间序列数据
            col_ts (str): 原时间戳列
        """
        if df is None:
            return df
        if col_ts is None:
            raise ValueError(f"{self.log_prefix} timestamp column is required for time series preprocessing.")
        if col_ts not in df.columns:
            raise ValueError(f"{self.log_prefix} timestamp column '{col_ts}' does not exist in dataframe.")

        # 数据拷贝
        df_processed = copy.deepcopy(df)
        # 转换时间戳类型
        df_processed[col_ts] = pd.to_datetime(df_processed[col_ts])
        # del df_processed[ts_col]
        # 单序列按 time 去重；global panel 必须按复合主键去重。
        duplicate_key = [col_ts]
        if bool(getattr(self.args, "enable_global_training", False)):
            series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
            if series_id_col not in df_processed.columns:
                raise ValueError(
                    f"{self.log_prefix} global panel source missing series ID column "
                    f"'{series_id_col}'."
                )
            duplicate_key = [series_id_col, col_ts]
        df_processed.drop_duplicates(
            subset=duplicate_key,
            keep="last",
            inplace=True,
            ignore_index=True,
        )
        return df_processed

    def __process_target_series(self, df_template: pd.DataFrame, df_series: pd.DataFrame, col_ts: str, col_numeric: List, col_categorical: List, col_drop: List):
        """
        目标特征数据预处理
        """
        df_template_copy = df_template.copy()
        if bool(getattr(self.args, "enable_global_training", False)):
            if df_series is None:
                raise ValueError(f"{self.log_prefix} global panel requires target series data.")
            series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
            filtered_numeric = [
                column
                for column in col_numeric
                if column not in [col_ts, self.args.target, series_id_col, *col_categorical, *col_drop]
            ]
            filtered_categorical = [
                column
                for column in col_categorical
                if column not in [col_ts, self.args.target, series_id_col, *col_numeric, *col_drop]
            ]
            panel = materialize_panel_history(
                df_series,
                df_template_copy["time"],
                series_id_col=series_id_col,
                source_time_col=col_ts,
                target_col=self.args.target,
                numeric_columns=filtered_numeric,
                categorical_columns=filtered_categorical,
                incomplete_policy=str(
                    getattr(self.args, "global_incomplete_series_policy", "raise") or "raise"
                ),
            )
            endogenous_features = [
                column
                for column in panel.columns
                if column not in {"time", series_id_col, "y"}
            ]
            return panel, endogenous_features, "y"
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
        configured_numeric_features = list(getattr(self.args, "target_series_numeric_features", []) or [])
        if configured_numeric_features:
            self.args.target_series_numeric_features = configured_numeric_features
        else:
            self.args.target_series_numeric_features = [
                col
                for col in df_history_series.columns
                if col not in [self.args.target, self.args.target_ts_feat] + \
                self.args.target_series_categorical_features + \
                self.args.target_series_drop_features
            ]
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
        if self._is_univariate_method():
            if not target_feature or target_feature not in df_history.columns:
                raise ValueError(f"{self.log_prefix} univariate prediction requires target feature in history data.")
            keep_columns = ["time", target_feature]
            if bool(getattr(self.args, "enable_global_training", False)):
                series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
                keep_columns.insert(1, series_id_col)
            df_history = df_history[keep_columns].copy()
            if target_feature != "y":
                df_history = df_history.rename(columns={target_feature: "y"})
            target_feature = "y"
            endogenous_features_with_target = ["y"]
            logger.info(
                f"{self.log_prefix} univariate pred_method detected; history data restricted to ['time', 'y']."
            )
        logger.info(f"{self.log_prefix} endogenous_features_with_target: {endogenous_features_with_target}")
        logger.info(f"{self.log_prefix}                  target_feature: {target_feature}")
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_history = self.__process_df_timestamp(df=input_data[f"date_history"], col_ts=self.args.date_ts_feat)
        if df_date_history is not None:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history: \n{df_date_history}")
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history shape: {df_date_history.shape}")
            if bool(getattr(self.args, "strict_date_information_set", False)):
                validate_daily_coverage(
                    df_date_history,
                    expected_times=df_history_template["time"],
                    ts_col=self.args.date_ts_feat,
                    value_columns=self.args.datetype_features,
                    label="Date history",
                )
        else:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_date_history: {df_date_history}")
        # 特征工程：天气特征
        df_weather_history = self.__process_df_timestamp(df=input_data[f"weather_history"], col_ts=self.args.weather_ts_feat)
        if df_weather_history is not None:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history: \n{df_weather_history}")
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history shape: {df_weather_history.shape}")
        else:
            logger.info(f"{self.log_prefix} __process_df_timestamp df_weather_history: {df_weather_history}")

        return (df_history, df_date_history, df_weather_history, endogenous_features_with_target, target_feature,
                input_data["custom_history"])

    def process_weather_backtest_data(self, input_data: Dict) -> Optional[pd.DataFrame]:
        """处理独立的滑窗 ex-ante 气象文件，不与历史实测 canonical merge。"""
        df_weather_backtest = input_data.get("weather_backtest")
        if df_weather_backtest is None:
            return None
        return self.__process_df_timestamp(
            df=df_weather_backtest,
            col_ts=self.args.weather_ts_feat,
        )

    def process_future_data(self, input_data: Dict):
        """
        处理未来预测阶段所需的外生数据。

        当前预测链路只需要未来时间模板，以及与时间对齐的日期/天气特征。
        未来目标序列或未来内生变量不在此处加载，递归预测所需的目标值会在
        预测过程中由模型输出逐步回填。
        """
        # 未来数据时间戳
        df_future_template = pd.DataFrame({"time": pd.date_range(self.forecast_start_time, self.forecast_end_time, freq=self.args.freq, inclusive="left")})
        if bool(getattr(self.args, "enable_global_training", False)):
            series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
            target_source = input_data.get("target_series")
            if target_source is None or series_id_col not in target_source.columns:
                raise ValueError(
                    f"{self.log_prefix} global panel future requires source series IDs in "
                    f"'{series_id_col}'."
                )
            df_future_template = materialize_panel_future(
                tuple(pd.unique(target_source[series_id_col].dropna())),
                df_future_template["time"],
                series_id_col=series_id_col,
            )
        logger.info(f"{self.log_prefix} df_future_template: \n{df_future_template.head()}")
        logger.info(f"{self.log_prefix} df_future_template shape: {df_future_template.shape}")
        # 特征工程：日期类型(节假日、特殊事件)特征
        df_date_future = self.__process_df_timestamp(df=input_data[f"date_future"], col_ts=self.args.date_ts_feat)
        if df_date_future is not None:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_date_future: \n{df_date_future}")
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_date_future shape: {df_date_future.shape}")
            if bool(getattr(self.args, "strict_date_information_set", False)):
                validate_daily_coverage(
                    df_date_future,
                    expected_times=df_future_template["time"],
                    ts_col=self.args.date_ts_feat,
                    value_columns=self.args.datetype_features,
                    label="Date future",
                )
        else:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_date_future: {df_date_future}")
        # 特征工程：天气特征
        df_weather_future = self.__process_df_timestamp(df=input_data[f"weather_future"], col_ts=self.args.weather_ts_feat)
        if df_weather_future is not None:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_weather_future: \n{df_weather_future}")
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_weather_future shape: {df_weather_future.shape}")
        else:
            logger.info(f"{self.log_prefix} after __process_df_timestamp df_weather_future: {df_weather_future}")

        if (
            bool(getattr(self.args, "strict_weather_information_set", False))
            and bool(getattr(self.args, "enable_weather_features", False))
        ):
            validate_weather_coverage(
                df_weather_future,
                df_future_template["time"],
                self.args.weather_ts_feat,
                "Future weather",
            )

        df_custom_future = materialize_custom_future_sources(
            custom_history=input_data.get("custom_history"),
            custom_future=input_data.get("custom_future"),
            future_times=pd.unique(df_future_template["time"]),
            cutoff=self.forecast_start_time - pd.Timedelta(nanoseconds=1),
        )
        return (df_future_template, df_date_future, df_weather_future, df_custom_future)


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
