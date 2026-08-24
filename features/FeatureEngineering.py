# -*- coding: utf-8 -*-

# ***************************************************
# * File        : FeatureEngineering.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : 特征工程: 
# *               - 一、基本特征: 
# *                 - 1.外生变量特征
# *                     - 1.1 日期时间特征(小时、星期、月份、季度等)、周期性编码(sin/cos)
# *                     - 1.2 天气特征(气象数据集成)
# *                     - 1.3 节假日 标记特征(日期类型数据集成)
# *                 - 2.内生变量特征: 
# *                     - 2.1 滞后特征: 单变量(目标变量)滞后特征、多变量(目标变量、其他内生变量)滞后特征
# *               - 二、高级特征: 
# *                 - 1.内生变量特征
# *                     - 1.1 滑动窗口统计特征 (Rolling Window Statistics)
# *                         - load_rolling_mean_3   # 最近3步平均值
# *                         - load_rolling_std_7    # 最近7步标准差
# *                         - load_rolling_min_12   # 最近12步最小值
# *                         - load_rolling_max_12   # 最近12步最大值
# *                     - 1.2 扩展窗口统计特征 (Expanding Window Statistics)
# *                         - load_expanding_mean   # 累积平均值
# *                         - load_expanding_std    # 累积标准差
# *                     - 1.3 差分特征 (Difference Features)
# *                         - load_diff_1          # 一阶差分
# *                         - load_diff_seasonal   # 季节性差分
# *                     - 1.4 百分比变化特征 (Percentage Change Features)
# *                     - 1.5 距离关键事件的时间特征 (Time-based Features)
# *                         - time_since_peak      # 距离峰值的时间
# *                         - time_since_low      # 距离谷值的时间
# *                     - 1.6 周期性特征（正弦余弦编码）(cyclical encoding)
# *                 - 2.交叉(交互)特征 (interaction features)
# *                     - hour_x_load_lag_1 = hour * load_lag_1     # 时间 × 滞后值
# *                     - temp_x_humidity = temperature * humidity  # 温度 × 湿度
# *                     - load_lag_1_squared = load_lag_1 ** 2      # 多项式特征
# *                 - 3.多项式特征 (polynomial features)
# *               - 三、目标编码
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd

from utils.frequency import resolve_freq_step_minutes, resolve_samples_per_day
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


def _resolve_series_group_col(args: Any, df: pd.DataFrame) -> Optional[str]:
    """Global panel 模式返回序列分组列，并校验每行都有实体标识。"""
    if not bool(getattr(args, "enable_global_training", False)):
        return None
    group_col = str(getattr(args, "series_id_feature", "series_id"))
    if group_col not in df.columns:
        raise ValueError(
            f"enable_global_training=true requires series_id column '{group_col}'."
        )
    if bool(df[group_col].isna().any()):
        raise ValueError(
            f"Global panel series_id column '{group_col}' contains missing values."
        )
    return group_col


def _series_shift(
    df: pd.DataFrame,
    column: str,
    periods: int,
    args: Any,
) -> pd.Series:
    """按实体边界 shift；非 global 模式保持原整表语义。"""
    group_col = _resolve_series_group_col(args, df)
    if group_col is None:
        return cast(pd.Series, df[column].shift(periods))
    return cast(
        pd.Series,
        df.groupby(group_col, sort=False, observed=True)[column].shift(periods),
    )


def _filter_supported_lags(lags: List[int], n_samples: int, log_prefix: str) -> List[int]:
    """
    丢弃当前样本数无法支撑的 lag(否则 shift(lag) 产出全 NaN 列,模型无声退化)。

    Args:
        lags: 配置的滞后步数列表。
        n_samples: 当前 df_series 的行数;至少需 lag < n_samples 才能产出 1 个有效值。
        log_prefix: 日志前缀。

    Returns:
        过滤后的 lag 列表(保留原相对顺序)。
    """
    if not lags:
        return []
    usable = [int(l) for l in lags if int(l) < n_samples]
    dropped = [int(l) for l in lags if int(l) >= n_samples]
    if dropped:
        logger.warning(
            f"{log_prefix} 丢弃超出可用样本数的 lag {dropped} "
            f"(当前样本数 {n_samples},需 lag < 样本数);剩余 lags={usable}。"
        )
    return usable


class ExogenousFeatureEngineer:
    """
    exogenous features engineering
    """
    def __init__(self, args, log_prefix="[ExogenousFeaturePreprocessor]", verbose: bool=False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        # 生成的外生特征
        self.exogenous_features = []
        # 收集类别特征
        self.categorical_features = []
        # future_strategy=freeze_last_observation 的 custom 列：属于预测原点冻结状态，
        # Direct horizon 展开时不得 shift 到目标期真实值。
        self.origin_frozen_features = []
    
    def extend_datetime_feature(self, df: pd.DataFrame, step_minutes: float, n_per_day: int):
        """
        日期时间特征
        """
        df_copy = df.copy()
        df_copy["time"] = pd.to_datetime(df_copy["time"])
        time_series = df_copy["time"]

        datetime_features_list = []
        # 时间基础特征（使用 .dt 向量化，避免空表 apply 推断异常）
        feature_map = {
            "minute": lambda s: s.dt.minute,
            "hour": lambda s: s.dt.hour,
            "day": lambda s: s.dt.day,
            "weekday": lambda s: s.dt.weekday,
            "week": lambda s: s.dt.isocalendar().week.astype("int64"),
            "day_of_week": lambda s: s.dt.dayofweek,
            "week_of_year": lambda s: s.dt.isocalendar().week.astype("int64"),
            "month": lambda s: s.dt.month,
            "days_in_month": lambda s: s.dt.daysinmonth,
            "quarter": lambda s: s.dt.quarter,
            "day_of_year": lambda s: s.dt.dayofyear,
            "year": lambda s: s.dt.year,
            # 日历边界标记（bool→0/1）：低频（日频）场景下区分周期边界状态
            "is_month_start": lambda s: s.dt.is_month_start,
            "is_month_end": lambda s: s.dt.is_month_end,
            "is_quarter_start": lambda s: s.dt.is_quarter_start,
            "is_quarter_end": lambda s: s.dt.is_quarter_end,
        }
        for feature_name in self.args.datetime_features:
            if feature_name in feature_map:
                col_name = f"dt_{feature_name}"
                df_copy[col_name] = feature_map[feature_name](time_series)
                datetime_features_list.append(col_name)
        # 周期性特征 (将时间转换为可循环的 sin/cos 形式)
        if 'dt_hour' in df_copy.columns and 'dt_minute' in df_copy.columns:
            minute_of_day = df_copy["dt_hour"] * 60 + df_copy["dt_minute"]
            df_copy["dt_minute_in_day"] = minute_of_day / step_minutes
            df_copy["dt_minute_in_day_sin"] = np.sin(df_copy["dt_minute_in_day"] * (2 * np.pi / n_per_day))
            df_copy["dt_minute_in_day_cos"] = np.cos(df_copy["dt_minute_in_day"] * (2 * np.pi / n_per_day))
            del df_copy["dt_minute_in_day"]
            datetime_features_list.append("dt_minute_in_day_sin")
            datetime_features_list.append("dt_minute_in_day_cos")
        # 日期时间特征收集
        if datetime_features_list:
            self.exogenous_features.extend(datetime_features_list)
            self.categorical_features.extend(self.args.datetime_categorical_features)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} after extend_datetime_feature datetime_features: {datetime_features_list}")

        return df_copy

    def extend_datetype_feature(self, df: pd.DataFrame, df_date: pd.DataFrame, col_ts: str="date", col_type: str="date_type"):
        """
        增加日期类型特征: 
        1-工作日 2-非工作日 3-删除计算日 4-元旦 5-春节 6-清明节 7-劳动节 8-端午节 9-中秋节 10-国庆节
        """
        df_copy = df.copy()
        if df_date is not None and not df_date.empty:
            # data map
            df_copy["date"] = df_copy["time"].dt.normalize() # Use .dt.normalize() to get date part
            df_copy["date_type"] = df_copy["date"].map(df_date.set_index(col_ts)[col_type])
            del df_copy["date"]
            # date features
            date_features_cfg = getattr(self.args, "datetype_features", ["date_type"])
            date_features = [feature for feature in date_features_cfg if feature in df_copy.columns]
        else:
            date_features = []
        # 日期类型特征收集
        if date_features:
            self.exogenous_features.extend(date_features)
            self.categorical_features.extend(self.args.datetype_categorical_features)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} after extend_datetype_feature date_features: {date_features}")

        return df_copy

    def extend_weather_feature(self, df: pd.DataFrame, df_weather: pd.DataFrame, col_ts: str):
        """
        处理天气特征
        """
        df_copy = df.copy()
        if df_weather is not None and not df_weather.empty:
            weather_features_cfg = getattr(
                self.args,
                "weather_features",
                ["rt_ssr", "rt_ws10", "rt_tt2", "cal_rh", "rt_ps", "rt_rain"],
            )
            weather_features_cfg = list(weather_features_cfg)
            # 原生 weather 通路允许配置声明预先派生的数值列；cal_rh 若源文件已
            # 提供则直接保留，否则才读取 rt_tt2/rt_dt 现场计算。
            dependency_cols = []
            if "cal_rh" in weather_features_cfg and "cal_rh" not in df_weather.columns:
                dependency_cols = ["rt_tt2", "rt_dt"]
            weather_source_cols = list(dict.fromkeys(weather_features_cfg + dependency_cols))
            df_weather_filtered = df_weather[
                [col for col in [col_ts] + weather_source_cols if col in df_weather.columns]
            ].copy()
            # 仅删除时间戳缺失，保留其余缺失值，避免时间不对齐时样本被全部清空
            df_weather_filtered = df_weather_filtered.dropna(subset=[col_ts]).reset_index(drop=True)
            if df_weather_filtered.empty:
                logger.warning(f"{self.log_prefix} df_weather became empty after dropping NaNs.")
                return df_copy

            # 将天气数值列统一转成 float 兼容类型。
            for col in weather_source_cols:
                if col in df_weather_filtered.columns:
                    df_weather_filtered[col] = pd.to_numeric(df_weather_filtered[col], errors='coerce')

            if "cal_rh" in weather_features_cfg:
                if "cal_rh" not in df_weather_filtered.columns:
                    df_weather_filtered["cal_rh"] = np.nan
                if {"rt_tt2", "rt_dt"}.issubset(df_weather_filtered.columns):
                    valid_idx = (
                        df_weather_filtered["cal_rh"].isna()
                        & df_weather_filtered["rt_tt2"].notna()
                        & df_weather_filtered["rt_dt"].notna()
                    )
                    if valid_idx.any():
                        t_air_c = df_weather_filtered.loc[valid_idx, "rt_tt2"] - 273.15
                        t_dew_c = df_weather_filtered.loc[valid_idx, "rt_dt"] - 273.15
                        e_s_td = 6.1078 * np.exp((17.2693 * t_dew_c) / (237.29 + t_dew_c))
                        e_s_t = 6.1078 * np.exp((17.2693 * t_air_c) / (237.29 + t_air_c))
                        df_weather_filtered.loc[valid_idx, "cal_rh"] = np.clip(
                            (e_s_td / e_s_t) * 100,
                            0,
                            100,
                        )

            # 特征筛选
            weather_features = weather_features_cfg
            # Keep only features that exist in the dataframe
            weather_features = [f for f in weather_features if f in df_weather_filtered.columns]
            df_weather_filtered = df_weather_filtered[[col_ts] + weather_features]
            
            # 合并目标数据和气象数据
            df_copy = pd.merge(df_copy, df_weather_filtered, left_on="time", right_on=col_ts, how="left")
            # 插值填充天气特征缺失值，保留原始样本行
            weather_existing = [f for f in weather_features if f in df_copy.columns]
            if weather_existing:
                df_copy.loc[:, weather_existing] = (
                    df_copy[weather_existing]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                )
            # 删除无用特征
            if col_ts != "time" and col_ts in df_copy.columns:
                del df_copy[col_ts]
        else:
            weather_features = []
        # 历史天气特征收集
        if weather_features:
            self.exogenous_features.extend(weather_features)
            self.categorical_features.extend(self.args.weather_categorical_features)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} after extend_weather_feature weather_features: {weather_features}")

        return df_copy

    def extend_custom_feature(
        self,
        df: pd.DataFrame,
        custom_sources: Optional[List[Dict[str, Any]]],
    ):
        """合并自定义外生特征（注册表多来源）。

        每个来源 dict: {"name", "ts_col", "columns", "categorical_columns", "df"}。
        按精确时间戳 merge（与 weather 同机制），历史/未来共用——计划类外生特征
        两段列名语义一致，不做 weather 那样的 rt_/pred_ 重命名。
        """
        df_copy = df.copy()
        if not custom_sources:
            return df_copy
        for source in custom_sources:
            name = source.get("name", "custom")
            df_custom = source.get("df")
            col_ts = source.get("ts_col")
            columns = [c for c in (source.get("columns") or [])]
            categorical_columns = [c for c in (source.get("categorical_columns") or [])]
            if df_custom is None or df_custom.empty:
                logger.warning(f"{self.log_prefix} Custom source '{name}' frame is empty; skipped.")
                continue
            keep_cols = [col_ts] + [c for c in columns if c in df_custom.columns]
            missing = [c for c in columns if c not in df_custom.columns]
            if missing:
                logger.warning(f"{self.log_prefix} Custom source '{name}' missing columns {missing}; skipped.")
            if len(keep_cols) <= 1:
                continue
            df_sel = df_custom[keep_cols].copy()
            df_sel[col_ts] = pd.to_datetime(df_sel[col_ts])
            df_sel = df_sel.drop_duplicates(subset=col_ts, keep="last").sort_values(col_ts)
            history_shift_steps = int(source.get("_history_shift_steps", 0) or 0)
            if history_shift_steps:
                offset = pd.tseries.frequencies.to_offset(self.args.freq)
                df_sel[col_ts] = df_sel[col_ts] + history_shift_steps * offset
            for col in keep_cols[1:]:
                if col in categorical_columns:
                    df_sel[col] = df_sel[col].astype("category")
                else:
                    df_sel[col] = pd.to_numeric(df_sel[col], errors="coerce")
            df_copy = pd.merge(df_copy, df_sel, left_on="time", right_on=col_ts, how="left")
            if col_ts != "time" and col_ts in df_copy.columns:
                del df_copy[col_ts]
            added = keep_cols[1:]
            self.exogenous_features.extend(added)
            self.categorical_features.extend([c for c in added if c in categorical_columns])
            if str(source.get("future_strategy", "explicit")).lower() == "freeze_last_observation":
                self.origin_frozen_features.extend(added)
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_custom_feature[{name}] added: {added}")
        return df_copy

    def extend_future_weather_feature(self, df: pd.DataFrame, df_weather: pd.DataFrame, col_ts: str):
        """
        未来气象数据特征构造
        """
        df_copy = df.copy()
        if df_weather is not None and not df_weather.empty:
            # 筛选天气预测数据
            pred_weather_features_map = {
                "pred_ssrd": "rt_ssr",
                "pred_ws10": "rt_ws10",
                "pred_tt2": "rt_tt2",
                "pred_rh": "cal_rh",
                "pred_ps": "rt_ps",
                "pred_rain": "rt_rain"
            }
            weather_features_cfg = getattr(
                self.args,
                "weather_features",
                list(pred_weather_features_map.values()),
            )
            # Filter df_weather for relevant columns and dropna
            # （同时保留 pred_* 源列与白名单目标列——月度统计类数据源直接用白名单名）
            _relevant_cols = (
                [col_ts]
                + list(pred_weather_features_map.keys())
                + list(pred_weather_features_map.values())
                + list(weather_features_cfg)
            )
            _relevant_cols = list(dict.fromkeys(_relevant_cols))
            df_weather_filtered = df_weather[[col for col in _relevant_cols if col in df_weather.columns]].copy()
            df_weather_filtered.dropna(subset=[col_ts], inplace=True, ignore_index=True)
            if df_weather_filtered.empty:
                logger.warning(f"{self.log_prefix} df_weather_future became empty after dropping NaNs.")
                return df_copy

            # 数据类型转换
            for pred_col in pred_weather_features_map.keys():
                if pred_col in df_weather_filtered.columns:
                    # df_weather_filtered[pred_col] = df_weather_filtered[pred_col].apply(lambda x: float(x))
                    df_weather_filtered[pred_col] = pd.to_numeric(df_weather_filtered[pred_col], errors='coerce')
            for weather_col in weather_features_cfg:
                if weather_col in df_weather_filtered.columns:
                    df_weather_filtered[weather_col] = pd.to_numeric(
                        df_weather_filtered[weather_col], errors="coerce"
                    )

            # 将预测气象数据整理到预测df中
            for pred_col, target_col in pred_weather_features_map.items():
                if target_col not in weather_features_cfg:
                    continue
                if pred_col in df_weather_filtered.columns:
                    # Apply specific transformations if defined
                    if pred_col == "pred_ps":
                        df_weather_filtered[pred_col] = df_weather_filtered[pred_col].apply(lambda x: x - 50.0)
                    elif pred_col == "pred_rain":
                        df_weather_filtered[pred_col] = df_weather_filtered[pred_col].apply(lambda x: x - 2.5)
                    df_copy[target_col] = df_copy["time"].map(df_weather_filtered.set_index(col_ts)[pred_col])
                elif target_col in df_weather_filtered.columns:
                    # 数据源已用白名单名（如月度统计文件的 rt_tt2/cal_rh 等），
                    # 无 pred_→rt_ 映射需求，直接按时间戳对齐
                    df_copy[target_col] = df_copy["time"].map(df_weather_filtered.set_index(col_ts)[target_col])
            # 预计算派生天气列不在 pred_* 映射表内，按 canonical 同名列直通。
            weather_indexed = df_weather_filtered.set_index(col_ts)
            for target_col in weather_features_cfg:
                if target_col not in df_copy.columns and target_col in weather_indexed.columns:
                    df_copy[target_col] = df_copy["time"].map(weather_indexed[target_col])
            
            # features to return
            weather_features = weather_features_cfg
            # Ensure to return only features that were actually added
            weather_features = [f for f in weather_features if f in df_copy.columns]
        else:
            weather_features = []
        # 历史天气特征收集
        if weather_features:
            self.exogenous_features.extend(weather_features)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} after extend_future_weather_feature weather_features: {weather_features}")

        return df_copy
    
    def get_generated_features(self) -> List[str]:
        """
        获取所有生成的特征列表
        """
        return self.exogenous_features, self.categorical_features

    def get_origin_frozen_features(self) -> List[str]:
        """返回预测期必须冻结在原点、禁止 horizon shift 的 custom 列。"""
        return list(dict.fromkeys(self.origin_frozen_features))
    
    def reset(self):
        """
        重置生成的特征列表
        """
        self.exogenous_features = []
        self.categorical_features = []
        self.origin_frozen_features = []


class EndogenousFeatureEngineer:
    """
    endogenous features engineering
    """
    def __init__(self, args, log_prefix="[endogenousFeatureProcessor]", verbose: bool=False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        # 生成的内生变量特征
        self.endogenous_features = []
        # 生成的多步预测目标
        self.target_output_features = []
    
    def extend_direct_multi_step_targets(self, df: pd.DataFrame, target: str, horizon: int, start_step: int = 0):
        """
        为多步直接预测创建未来多步目标
        
        Args:
            df: 数据框
            target: 目标变量名
            horizon: 预测horizon
        
        Returns:
            (扩展后的数据框, 目标特征列表)
        """
        df_copy = df.copy()
        if target in df_copy.columns:
            # shift features building
            shift_target_features = []
            for h in range(start_step, start_step + horizon):
                shifted_col_name = f"{target}_shift_{h}"
                df_copy[shifted_col_name] = _series_shift(
                    df_copy,
                    target,
                    -h,
                    self.args,
                ).to_numpy()
                shift_target_features.append(shifted_col_name)
            # 特征收集
            if shift_target_features:
                self.target_output_features.extend(shift_target_features)
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets target_output_features: {self.target_output_features}")
        
        return df_copy

    def extend_lag_feature_univariate(
        self,
        df: pd.DataFrame,
        target: str,
        lags: List[int],
        shift_offset: int = 0,
    ):
        """
        扩展单变量滞后特征(for univariate time series)
        
        Args:
            df: 数据框
            target: 目标变量名
            lags: 滞后期列表
        
        Returns:
            (扩展后的数据框, 新增的滞后特征列表)
        """
        df_lags = df.copy()
        # 将 time 作为索引
        df_lags = df_lags.set_index("time").copy()
        
        lag_features = []
        for lag in lags:
            col_name = f'{target}_lag_{lag}'
            shift_steps = lag + shift_offset
            if shift_steps < 0:
                raise ValueError(
                    f"Lag {lag} with shift_offset {shift_offset} resolves to a future target value."
                )
            df_lags[col_name] = _series_shift(
                df_lags,
                target,
                shift_steps,
                self.args,
            ).to_numpy()
            lag_features.append(col_name)
        # 特征收集
        if lag_features:
            self.endogenous_features.extend(lag_features)
        if self.verbose:
            logger.info(f"{self.log_prefix} after extend_lag_feature_univariate endogenous_features: {self.endogenous_features}")

        return df_lags

    def extend_lag_feature_multivariate(self, df: pd.DataFrame, endogenous_cols: List[str], lags: List[int]):
        """
        扩展多变量滞后特征
        
        Args:
            df: 数据框
            endogenous_cols: 内生变量列表
            lags: 滞后期
        
        Returns:
            (扩展后的数据框, 滞后特征列表, 目标特征列表)
        """
        df_copy = df.copy()
        # 将 time 作为索引
        df_copy = df_copy.set_index("time").copy()
        
        all_lag_features = []
        for col in endogenous_cols:
            if col not in df_copy.columns:
                continue
            
            # 创建滞后特征
            lags_X = [
                _series_shift(df_copy, col, int(lag), self.args)
                for lag in lags
            ]
            lag_col_names_X = [f'{col}_lag_{i}' for i in lags]
            for i, name in enumerate(lag_col_names_X):
                df_copy[name] = lags_X[i].values
                all_lag_features.append(name)
        # 特征收集
        if all_lag_features:
            self.endogenous_features.extend(all_lag_features)
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate endogenous_features: {self.endogenous_features}")
        
        return df_copy

    def get_generated_features(self) -> List[str]:
        """
        获取所有生成的特征列表
        """
        return self.endogenous_features, self.target_output_features
    
    def reset(self):
        """
        重置生成的特征列表
        """
        self.endogenous_features = []
        self.target_output_features = []


class EndogenousAdvancedFeatureEngineer:
    """
    高级特征工程器
    
    新增特征类型:
    1. 滑动窗口统计特征 (rolling window statistics)
    2. 滚动(扩展)窗口统计特征 (expanding window statistics)
    3. 差分特征 (difference features）
    4. 百分比变化特征 (percentage change features)
    5. 距离关键事件的时间特征 (Time-based Features)
    6. 周期性特征 (cyclical encoding)
    7. 交叉(交互)特征 (interaction features)
    8. 多项式特征 (polynomial features)
    """
    
    def __init__(self, log_prefix: str = "[FeatureEngineer]", verbose: bool=False):
        self.log_prefix = log_prefix
        self.verbose = verbose
        # 生成的高级特征工程（内生变量特征）
        self.endogenous_advanced_features = []
    
    @staticmethod
    def _transform_by_group(
        df: pd.DataFrame,
        column: str,
        group_col: Optional[str],
        transform,
    ) -> pd.Series:
        """对单列执行时序变换；global panel 时在每个实体内独立计算。"""
        if group_col is None:
            return cast(pd.Series, transform(df[column]))
        return cast(
            pd.Series,
            df.groupby(group_col, sort=False, observed=True)[column].transform(transform),
        )

    @staticmethod
    def _time_since_event(series: pd.Series, event: str) -> pd.Series:
        """在一条序列内计算距上一个峰/谷的步数，保持既有首行与事件行语义。"""
        if event == "peak":
            event_mask = (series.shift(1) < series) & (series > series.shift(-1))
        elif event == "trough":
            event_mask = (series.shift(1) > series) & (series < series.shift(-1))
        else:
            raise ValueError(f"Unsupported time-since event: {event}")
        event_indices = np.where(event_mask.to_numpy())[0]
        values = []
        for position in range(len(series)):
            if position == 0:
                values.append(0)
                continue
            prior = event_indices[event_indices < position]
            values.append(position - prior[-1] if len(prior) > 0 else position)
        return pd.Series(values, index=series.index, dtype=float)

    def add_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        windows: List[int],
        stats: List[str] = ["mean", "std", "min", "max", "median", "skew", "kurt"],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加滑动窗口统计特征
        
        Args:
            df: 数据框
            columns: 需要计算统计特征的列
            windows: 窗口大小列表，如 [3, 7, 14, 30]
            stats: 统计量列表 ["mean", "std", "min", "max", "median", "skew", "kurt"]
        
        Returns:
            增强后的数据框
        
        Examples:
            >>> fe = EndogenousAdvancedFeatureEngineer()
            >>> df = fe.add_rolling_statistics(df, ['load'], [3, 7], ['mean', 'std'])
            # 生成: load_rolling_mean_3, load_rolling_std_3, load_rolling_mean_7, load_rolling_std_7
        """
        logger.info(f"{self.log_prefix} 添加滑动窗口统计特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for window in windows:
                for stat in ("mean", "std", "min", "max", "median", "skew", "kurt"):
                    if stat not in stats:
                        continue
                    feature_name = f"{col}_rolling_{stat}_{window}"
                    df_enhanced[feature_name] = self._transform_by_group(
                        df,
                        col,
                        group_col,
                        lambda series, window=window, stat=stat: getattr(
                            series.rolling(window=window, min_periods=1, center=False),
                            stat,
                        )(),
                    ).to_numpy()
                    self.endogenous_advanced_features.append(feature_name)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个滑动窗口统计特征。")
        return df_enhanced

    def add_expanding_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        stats: List[str] = ["mean", "std", "min", "max", "median", "skew", "kurt"],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加扩展窗口统计特征
        
        扩展窗口从数据开始到当前位置，窗口大小递增
        
        Args:
            df: 数据框
            columns: 需要计算统计特征的列
            stats: 统计量列表 ["mean", "std", "min", "max", "median", "skew", "kurt"]
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加扩展窗口统计特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for stat in ("mean", "std", "min", "max", "median", "skew", "kurt"):
                if stat not in stats:
                    continue
                feature_name = f"{col}_expanding_{stat}"
                df_enhanced[feature_name] = self._transform_by_group(
                    df,
                    col,
                    group_col,
                    lambda series, stat=stat: getattr(
                        series.expanding(min_periods=1),
                        stat,
                    )(),
                ).to_numpy()
                self.endogenous_advanced_features.append(feature_name)
            
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个扩展窗口统计特征。")
        return df_enhanced
  
    def add_diff_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 7, 24],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加差分特征
        
        差分可以去除趋势，使数据更平稳
        
        Args:
            df: 数据框
            columns: 列名列表
            periods: 差分周期列表
                - 1: 一阶差分（相邻差分）
                - 7: 周差分（去除周周期）
                - 24: 日差分（去除日周期，针对小时数据）
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加差分特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for period in periods:
                feature_name = f'{col}_diff_{period}'
                df_enhanced[feature_name] = self._transform_by_group(
                    df,
                    col,
                    group_col,
                    lambda series, period=period: series.diff(period),
                ).to_numpy()
                self.endogenous_advanced_features.append(feature_name)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个差分特征。")
        return df_enhanced
    
    def add_pct_change_features(
        self,
        df: pd.DataFrame,
        columns: List[str],
        periods: List[int] = [1, 7],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加百分比变化特征
        
        计算相对于前 N 期的百分比变化
        
        Args:
            df: 数据框
            columns: 列名列表
            periods: 周期列表
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加百分比变化特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for period in periods:
                feature_name = f'{col}_pct_change_{period}'
                df_enhanced[feature_name] = self._transform_by_group(
                    df,
                    col,
                    group_col,
                    lambda series, period=period: series.pct_change(
                        periods=period,
                        fill_method=None,
                    ),
                ).to_numpy()
                self.endogenous_advanced_features.append(feature_name)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个百分比变化特征。")
        return df_enhanced
    
    def add_time_since_features(
        self,
        df: pd.DataFrame,
        column: str,
        events: List[str] = ['peak', 'trough'],
        group_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        添加距离关键事件的时间特征
        
        Args:
            df: 数据框
            column: 列名
            events: 事件列表 ['peak', 'trough']
                - 'peak': 距离峰值的时间
                - 'trough': 距离谷值的时间
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加时间距离特征...")
        df_enhanced = df.copy()
        
        if column not in df.columns:
            logger.warning(f"{self.log_prefix} 列 {column} 不存在。")
            return df_enhanced
        
        for event in ("peak", "trough"):
            if event not in events:
                continue
            feature_name = f"{column}_time_since_{event}"
            df_enhanced[feature_name] = self._transform_by_group(
                df,
                column,
                group_col,
                lambda series, event=event: self._time_since_event(series, event),
            ).to_numpy()
            self.endogenous_advanced_features.append(feature_name)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个距离关键事件的时间特征。")
        return df_enhanced
    
    def add_cyclical_features(self, df: pd.DataFrame, column: str="minute", period: int=15) -> pd.DataFrame:
        """
        添加周期性特征（正弦余弦编码）
        
        避免周期性特征的边界问题
        例如: 23 点和 0 点在数值上差距大，但实际很接近

        Args:
            df (pd.DataFrame): 数据框
            column (str): 列名
            period (int): 周期内的样本数量

        Returns:
            增强后的数据框
        """
        actual_column = column
        if actual_column not in df.columns and f"dt_{actual_column}" in df.columns:
            actual_column = f"dt_{actual_column}"
        if actual_column not in df.columns:
            logger.warning(f"{self.log_prefix} 列 {column} 不存在，跳过。")
            return df

        df[f'{actual_column}_sin'] = np.sin(2 * np.pi * df[actual_column] / period)
        df[f'{actual_column}_cos'] = np.cos(2 * np.pi * df[actual_column] / period)
        self.endogenous_advanced_features.append(f"{actual_column}_sin")
        self.endogenous_advanced_features.append(f"{actual_column}_cos")
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个交互(交叉)特征。")
        return df
    
    def add_interaction_features(self, df: pd.DataFrame, column_pairs: List[tuple], operations: List[str] = ["add", "subtract", "multiply", "divide"]) -> pd.DataFrame:
        """
        添加交互(交叉)特征
        
        Args:
            df: 数据框
            column_pairs: 列对列表 [('col1', 'col2'), ...]
            operations: 操作列表 ["add", "subtract", "multiply", "divide"]
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加交互(交叉)特征...")
        df_enhanced = df.copy()
        
        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col1} 或者 {col2} 不存在，跳过。")
                continue

            if 'add' in operations:
                feature_name = f'{col1}_add_{col2}'
                df_enhanced[feature_name] = df[col1] + df[col2]
                self.endogenous_advanced_features.append(feature_name)
            
            if 'subtract' in operations:
                feature_name = f'{col1}_substract_{col2}'
                df_enhanced[feature_name] = df[col1] - df[col2]
                self.endogenous_advanced_features.append(feature_name)
            
            if 'multiply' in operations:
                feature_name = f'{col1}_multiply_{col2}'
                df_enhanced[feature_name] = df[col1] * df[col2]
                self.endogenous_advanced_features.append(feature_name)
            
            if 'divide' in operations:
                feature_name = f'{col1}_divide_{col2}'
                df_enhanced[feature_name] = df[col1] / (df[col2] + 1e-8)  # 避免除零
                self.endogenous_advanced_features.append(feature_name)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个交互(交叉)特征。")
        return df_enhanced
    
    def add_polynomial_features(self, df: pd.DataFrame, columns: List[str], degree: int = 2) -> pd.DataFrame:
        """
        添加多项式特征
        
        Args:
            df: 数据框
            columns: 列名列表
            degree: 多项式阶数
        
        Returns:
            增强后的数据框
        """
        logger.info(f"{self.log_prefix} 添加多项式特征...")
        df_enhanced = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"{self.log_prefix} 列 {col} 不存在，跳过。")
                continue
            
            for d in range(2, degree + 1):
                feature_name = f'{col}_pow_{d}'
                df_enhanced[feature_name] = df[col] ** d
                self.endogenous_advanced_features.append(feature_name)
        
        if self.verbose:
            logger.info(f"{self.log_prefix} 生成 {len(self.endogenous_advanced_features)} 个多项式特征。")
        return df_enhanced
    
    def get_generated_features(self) -> List[str]:
        """
        获取所有生成的特征列表
        """
        return self.endogenous_advanced_features
    
    def reset(self):
        """
        重置生成的特征列表
        """
        self.endogenous_advanced_features = []


class FeatureEngineer:
    """
    特征预处理器
    """
    def __init__(self, args, log_prefix="[FeatureEngineer]", verbose: bool=False):
        self.args = args
        self.log_prefix = log_prefix
        self.verbose = verbose
        # 外生变量特征工程
        self.exogenous_feature_engineer = ExogenousFeatureEngineer(args, log_prefix, verbose=verbose)
        # 内生变量特征工程
        self.endogenous_feature_engineer = EndogenousFeatureEngineer(args, log_prefix, verbose=verbose)
        # 高级特征工程
        self.advanced_feature_engineer = EndogenousAdvancedFeatureEngineer(log_prefix, verbose=verbose)

    def create_exogenouse_features(self, df, df_date_history, df_date_future, df_weather_history, df_weather_future,
                                   df_custom_history=None, df_custom_future=None):
        """
        历史数据特征工程: 外生变量特征
        """
        df_featured = df.copy()
        self.exogenous_feature_engineer.reset()
        
        # 特征工程: 日期类型(节假日、特殊事件)特征
        if getattr(self.args, "enable_date_features", True) and df_date_history is not None:
            df_featured = self.exogenous_feature_engineer.extend_datetype_feature(
                df=df_featured,
                df_date=df_date_history,
                col_ts=self.args.date_ts_feat,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_datetype_feature df_featured: \n{df_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_datetype_feature df_featured shape: {df_featured.shape}")
        elif getattr(self.args, "enable_date_features", True) and df_date_future is not None:
            df_featured = self.exogenous_feature_engineer.extend_datetype_feature(
                df=df_featured,
                df_date=df_date_future,
                col_ts=self.args.date_ts_feat,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_datetype_feature df_featured: \n{df_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_datetype_feature df_featured shape: {df_featured.shape}")
        # 特征工程: 天气特征
        if getattr(self.args, "enable_weather_features", True) and df_weather_history is not None:
            df_featured = self.exogenous_feature_engineer.extend_weather_feature(
                df=df_featured,
                df_weather=df_weather_history,
                col_ts=self.args.weather_ts_feat,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_weather_feature df_featured: \n{df_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_weather_feature df_featured shape: {df_featured.shape}")
        elif getattr(self.args, "enable_weather_features", True) and df_weather_future is not None:
            df_featured = self.exogenous_feature_engineer.extend_future_weather_feature(
                df=df_featured,
                df_weather=df_weather_future,
                col_ts=self.args.weather_ts_feat,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_future_weather_feature df_featured: \n{df_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_future_weather_feature df_featured shape: {df_featured.shape}")
        # 特征工程: 自定义外生特征（注册表多来源；历史段优先，未来段回退）
        custom_sources = df_custom_history if df_custom_history else df_custom_future
        if custom_sources:
            # end_of_period 状态在当期结束后才可用。Recursive/Pointwise 的训练
            # 目标与特征同一时间戳，需把 source 时间戳向后移 1 个 freq 步，使
            # 行 t 合并到 state(t-1)；Direct/DirRec 的行 t 是预测原点，保留
            # state(t) 并在所有 horizon 冻结。
            is_history = bool(df_custom_history)
            shift_history_for_method = str(self.args.pred_method).lower() in {
                "univariate-single-multistep-direct-pointwise",
                "univariate-single-multistep-recursive",
                "multivariate-single-multistep-recursive",
            }
            prepared_sources = []
            for source in custom_sources:
                prepared = dict(source)
                availability = str(prepared.get("availability", "contemporaneous") or "contemporaneous").lower()
                if availability not in {"contemporaneous", "forecast_origin", "end_of_period"}:
                    raise ValueError(
                        f"Custom source '{prepared.get('name', 'custom')}' has unsupported "
                        f"availability='{availability}'."
                    )
                if is_history and availability == "end_of_period" and shift_history_for_method:
                    prepared["_history_shift_steps"] = 1
                prepared_sources.append(prepared)
            df_featured = self.exogenous_feature_engineer.extend_custom_feature(
                df=df_featured,
                custom_sources=prepared_sources,
            )
        # 特征工程: 日期时间特征
        if getattr(self.args, "enable_datetime_features", True):
            df_featured = self.exogenous_feature_engineer.extend_datetime_feature(
                df=df_featured,
                step_minutes=resolve_freq_step_minutes(self.args.freq),
                n_per_day=resolve_samples_per_day(self.args.freq),
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_datetime_feature df_featured: \n{df_featured.head()}")
                # logger.info(f"{self.log_prefix} after extend_datetime_feature df_featured.columns: \n{df_featured.columns}")
                logger.info(f"{self.log_prefix} after extend_datetime_feature df_featured shape: {df_featured.shape}")
        
        # 获取所有生成的特征: 外生变量特征、类别
        exogenous_features, categorical_features = self.exogenous_feature_engineer.get_generated_features()
        categorical_features = sorted(set(categorical_features), key=categorical_features.index)

        # 只允许填充外生列，禁止整帧 interpolate 修改目标 y/内生变量。
        # strict来源已在DataLoader按目标时间轴校验，缺失必须保留为失败信号，
        # 不能用首尾值或0静默伪造。
        custom_cfg = list(getattr(self.args, "custom_features", None) or [])
        strict_information_set = (
            bool(getattr(self.args, "strict_weather_information_set", False))
            or bool(getattr(self.args, "strict_date_information_set", False))
            or any(bool(source.get("strict_information_set", False)) for source in custom_cfg)
        )
        existing_exogenous = [col for col in exogenous_features if col in df_featured.columns]
        if existing_exogenous and not strict_information_set:
            numeric_exogenous = [
                col for col in existing_exogenous
                if col not in categorical_features
            ]
            if numeric_exogenous:
                df_featured.loc[:, numeric_exogenous] = (
                    df_featured[numeric_exogenous]
                    .interpolate(method="linear", limit_direction="both")
                    .ffill()
                    .bfill()
                    .fillna(0.0)
                )
            categorical_existing = [
                col for col in existing_exogenous if col in categorical_features
            ]
            if categorical_existing:
                df_featured.loc[:, categorical_existing] = (
                    df_featured[categorical_existing].ffill().bfill()
                )
        if self.verbose:
            logger.info(f"{self.log_prefix} after exogenous-only fill df_featured shape: {df_featured.shape}")

        return df_featured, exogenous_features, categorical_features

    def _expand_horizon_exogenous_for_direct(
        self,
        df: pd.DataFrame,
        exogenous_features: List[str],
        horizon: int,
        origin_frozen_features: Optional[List[str]] = None,
    ):
        """
        Direct 多步预测场景下，将外生特征扩展为 horizon-aware 形式：
        exog_h1, exog_h2, ..., exog_hH

        行 t 的 col_h(h) = 外生列在目标日 t+h 的值（对齐该 horizon 的实际
        预测目标），帧尾目标日不存在时为 NaN（由训练 dropna 剔除），禁止
        回看原点日取值。h=1 即 shift(-1)，以此类推。
        """
        if horizon <= 1 or not exogenous_features:
            return df, exogenous_features

        df_copy = df.copy()
        expanded_features = []
        expanded_data = {}
        frozen_set = set(origin_frozen_features or [])
        for h in range(1, horizon + 1):
            shift_steps = -h
            for col in exogenous_features:
                # 预测原点状态（freeze_last_observation）只保留基础列，所有 h
                # 共享原点值；生成 state_h 会把训练期真实未来状态泄漏给模型。
                if col in frozen_set:
                    continue
                col_h = f"{col}_h{h}"
                shifted = _series_shift(df_copy, col, shift_steps, self.args)
                # bool 列 shift 后因 NaN 会退化为 object；LightGBM 只接受
                # int/float/bool，故显式转成 0/1/NaN float。其它 dtype 保持。
                if pd.api.types.is_bool_dtype(df_copy[col].dtype):
                    shifted = shifted.astype(float)
                expanded_data[col_h] = shifted
                expanded_features.append(col_h)

        if expanded_data:
            df_copy = pd.concat(
                [df_copy, pd.DataFrame(expanded_data, index=df_copy.index)],
                axis=1,
            )

        if self.verbose:
            logger.info(f"{self.log_prefix} horizon-aware exogenous features generated: {len(expanded_features)}")

        # 保留基础外生列用于推理端统一 schema；训练端 horizon_feature melt
        # 会按 h 从 *_h{h} 折叠回基础列名。普通 multioutput Direct 同时拿到
        # 基础列与 horizon-aware 列，目标日外生可供各输出模型使用。
        return df_copy, list(exogenous_features) + expanded_features

    def create_endogenous_basic_features(self, df_series, target_feature, endogenous_features_with_target, horizon):
        """
        历史数据特征工程: 内生变量特征
        """
        df_series_featured = df_series.copy()
        self.endogenous_feature_engineer.reset()
        align_direct_to_target = bool(
            getattr(self.args, "align_direct_features_to_target", False)
        ) and self.args.pred_method in [
            "univariate-single-multistep-direct",
            "univariate-single-multistep-direct-recursive",
        ]
        configured_lags = self.args.lags if getattr(self.args, "enable_lags_features", True) else []
        group_col = _resolve_series_group_col(self.args, df_series_featured)
        lag_support_samples = len(df_series_featured)
        if group_col is not None:
            lag_support_samples = int(
                df_series_featured.groupby(group_col, sort=False, observed=True).size().min()
            )
        lags = _filter_supported_lags(
            configured_lags,
            n_samples=lag_support_samples,
            log_prefix=self.log_prefix,
        )

        if self.args.pred_method == "univariate-single-multistep-direct-pointwise":
            if bool(getattr(self.args, "align_direct_features_to_target", False)):
                df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_univariate(
                    df=df_series_featured,
                    target=target_feature,
                    lags=lags,
                )
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "univariate-single-multistep-direct":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_univariate(
                df = df_series_featured,
                target = target_feature,
                lags = lags,
                shift_offset=-1 if align_direct_to_target else 0,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured: \n{df_series_featured.head()}")
                # logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured.columns: {df_series_featured.columns}")
                logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured shape: {df_series_featured.shape}")
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = horizon,
                start_step = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "univariate-single-multistep-recursive":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_univariate(
                df = df_series_featured,
                target = target_feature,
                lags = lags,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured shape: {df_series_featured.shape}")
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "univariate-single-multistep-direct-recursive":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_univariate(
                df = df_series_featured,
                target = target_feature,
                lags = lags,
                shift_offset=-1 if align_direct_to_target else 0,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_lag_feature_univariate df_series_featured shape: {df_series_featured.shape}")
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = horizon,
                start_step = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "multivariate-single-multistep-direct":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_multivariate(
                df = df_series_featured,
                endogenous_cols = endogenous_features_with_target,
                lags = lags,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_featured shape: {df_series_featured.shape}")
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = horizon,
                start_step = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "multivariate-single-multistep-recursive":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_multivariate(
                df = df_series_featured,
                endogenous_cols = endogenous_features_with_target,
                lags = lags,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_featured shape: {df_series_featured.shape}")
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "multivariate-single-multistep-direct-recursive":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_multivariate(
                df = df_series_featured,
                endogenous_cols = endogenous_features_with_target,
                lags = lags,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_lag_feature_multivariate df_series_featured shape: {df_series_featured.shape}")
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df = df_series_featured,
                target = target_feature,
                horizon = horizon,
                start_step = 1,
            )
            if self.verbose:
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured: \n{df_series_featured.head()}")
                logger.info(f"{self.log_prefix} after extend_direct_multi_step_targets df_series_featured shape: {df_series_featured.shape}")
        elif self.args.pred_method == "univariate-single-multistep-blend-direct-recursive":
            # Blend = Direct(多步宽表 shift_1..H) + Recursive(1步 shift_0) 融合
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_univariate(
                df=df_series_featured, target=target_feature, lags=lags,
            )
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df=df_series_featured, target=target_feature, horizon=horizon, start_step=1,
            )
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df=df_series_featured, target=target_feature, horizon=1, start_step=0,
            )
        elif self.args.pred_method == "multivariate-single-multistep-blend-direct-recursive":
            df_series_featured = self.endogenous_feature_engineer.extend_lag_feature_multivariate(
                df=df_series_featured, endogenous_cols=endogenous_features_with_target, lags=lags,
            )
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df=df_series_featured, target=target_feature, horizon=horizon, start_step=1,
            )
            df_series_featured = self.endogenous_feature_engineer.extend_direct_multi_step_targets(
                df=df_series_featured, target=target_feature, horizon=1, start_step=0,
            )

        # 获取所有生成的特征: 内生变量特征、多步预测目标特征
        endogenous_features, target_output_features = self.endogenous_feature_engineer.get_generated_features()

        return df_series_featured, endogenous_features, target_output_features

    def create_endogenous_advanced_features(self, df_series):
        """
        历史数据特征工程: 内生生变量高级统计特征
        """
        if self.args.enable_advanced_features:
            # 复制数据
            df_series_featured = df_series.copy()
            group_col = _resolve_series_group_col(self.args, df_series_featured)
            self.advanced_feature_engineer.reset()
            # 添加滞后统计特征
            if getattr(self.args, "enable_rolling_features", True):
                df_series_featured = self.advanced_feature_engineer.add_rolling_statistics(
                    df_series_featured,
                    columns=self.args.rolling_columns,
                    windows=self.args.rolling_windows,
                    stats=self.args.rolling_stats,
                    group_col=group_col,
                )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_rolling_statistics df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_rolling_statistics df_series_featured shape: {df_series_featured.shape}")
            # 添加扩展统计特征
            if getattr(self.args, "enable_expanding_features", True):
                df_series_featured = self.advanced_feature_engineer.add_expanding_statistics(
                    df_series_featured,
                    columns=self.args.expanding_columns,
                    stats=self.args.expanding_stats,
                    group_col=group_col,
                )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_expanding_statistics df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_expanding_statistics df_series_featured shape: {df_series_featured.shape}")
            # 添加差分特征
            if getattr(self.args, "enable_diff_features", True):
                df_series_featured = self.advanced_feature_engineer.add_diff_features(
                    df_series_featured,
                    columns=self.args.diff_columns,
                    periods=self.args.diff_periods,
                    group_col=group_col,
                )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_diff_features df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_diff_features df_series_featured shape: {df_series_featured.shape}")
            # 添加差分特征
            if getattr(self.args, "enable_pct_change_features", True):
                df_series_featured = self.advanced_feature_engineer.add_pct_change_features(
                    df_series_featured,
                    columns=self.args.pct_change_columns,
                    periods=self.args.pct_change_periods,
                    group_col=group_col,
                )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_pct_change_features df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_pct_change_features df_series_featured shape: {df_series_featured.shape}")
            # 添加距离关键事件的时间特征
            if getattr(self.args, "enable_time_since_features", True):
                for time_since_column in self.args.time_since_columns:
                    df_series_featured = self.advanced_feature_engineer.add_time_since_features(
                        df_series_featured,
                        column=time_since_column,
                        events=self.args.time_since_events,
                        group_col=group_col,
                    )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_time_since_features df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_time_since_features df_series_featured shape: {df_series_featured.shape}")
            # 添加周期性特征（正弦余弦编码）
            if getattr(self.args, "enable_cyclical_features", True):
                for cyclical_column in self.args.cyclical_columns:
                    df_series_featured = self.advanced_feature_engineer.add_cyclical_features(
                        df_series_featured,
                        column=cyclical_column,
                        period=self.args.cyclical_period,
                    )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_cyclical_features df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_cyclical_features df_series_featured shape: {df_series_featured.shape}")
            # 添加交互(交叉)特征
            if getattr(self.args, "enable_interaction_features", True):
                df_series_featured = self.advanced_feature_engineer.add_interaction_features(
                    df_series_featured,
                    column_pairs=self.args.interaction_column_pairs,
                    operations=self.args.interaction_operations,
                )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_interaction_features df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_interaction_features df_series_featured shape: {df_series_featured.shape}")
            # 添加多项式特征
            if getattr(self.args, "enable_polynomial_features", True):
                df_series_featured = self.advanced_feature_engineer.add_polynomial_features(
                    df_series_featured,
                    columns=self.args.polynomial_columns,
                    degree=self.args.polynomial_degree,
                )
                if self.verbose:
                    logger.info(f"{self.log_prefix} after add_polynomial_features df_series_featured: \n{df_series_featured.head()}")
                    logger.info(f"{self.log_prefix} after add_polynomial_features df_series_featured shape: {df_series_featured.shape}")

            # 获取所有生成的特征: 内生变量高级特征
            endogenous_advanced_features = self.advanced_feature_engineer.get_generated_features()

            return df_series_featured, endogenous_advanced_features
        else:
            return df_series, []

    def create_features(self,
                        df_series: pd.DataFrame,
                        df_date_history: Optional[pd.DataFrame]=None,
                        df_date_future: Optional[pd.DataFrame]=None,
                        df_weather_history: Optional[pd.DataFrame]=None,
                        df_weather_future: Optional[pd.DataFrame]=None,
                        df_custom_history=None,
                        df_custom_future=None,
                        endogenous_features_with_target: List[str]=["y"],
                        target_feature: str="y",
                        horizon: int=1):
        """
        特征工程: 集成内生变量特征、外生变量特征、内生变量高级特征
        """
        # 复制数据
        df_series_copy = df_series.copy()
        # 用于构建滞后特征的内生变量
        endogenous_features_with_target_copy = endogenous_features_with_target
        # 构建多步直接预测目标变量的目标变量
        target_feature_copy = target_feature
        # ------------------------------
        # Feature engineering
        # ------------------------------
        if self.verbose:
            logger.info(f"{self.log_prefix} 开始数据特征工程...")

        # 历史、未来数据特征工程: 外生变量特征工程
        if self.verbose:
            logger.info(f"{self.log_prefix} 数据特征工程: 外生变量特征...")
        (df_series_featured, exogenous_features, categorical_features) = self.create_exogenouse_features(
            df=df_series_copy, 
            df_date_history=df_date_history, 
            df_date_future=df_date_future,
            df_weather_history=df_weather_history,
            df_weather_future=df_weather_future,
            df_custom_history=df_custom_history,
            df_custom_future=df_custom_future,
        )
        origin_frozen_features = self.exogenous_feature_engineer.get_origin_frozen_features()
        model_horizon = horizon
        if self.args.pred_method in [
            "univariate-single-multistep-direct-recursive",
            "multivariate-single-multistep-direct-recursive",
        ]:
            configured_block = int(getattr(self.args, "block_size", 0) or 0)
            if configured_block > 0:
                model_horizon = min(configured_block, horizon)
        align_direct_to_target = bool(
            getattr(self.args, "align_direct_features_to_target", False)
        ) and self.args.pred_method in [
            "univariate-single-multistep-direct",
            "univariate-single-multistep-direct-recursive",
        ]
        if align_direct_to_target:
            if horizon != 1:
                raise ValueError(
                    "align_direct_features_to_target currently supports horizon=1 only."
                )
            # Direct 的训练目标位于 t+1；把 t+1 的可预知外生量移到特征行 t，
            # 使训练与预测都使用目标月份的气象/日历，而不是预测原点月份。
            for col in exogenous_features:
                if col in df_series_featured.columns:
                    df_series_featured[col] = _series_shift(
                        df_series_featured,
                        col,
                        -1,
                        self.args,
                    ).to_numpy()
        # Direct 系列方法下，按 horizon 展开外生特征
        should_expand_horizon_exogenous = (
            getattr(self.args, "use_horizon_exogenous_for_direct", False)
            or str(getattr(self.args, "direct_strategy", "multioutput")).lower() == "horizon_feature"
        )
        if should_expand_horizon_exogenous and self.args.pred_method in [
            "univariate-single-multistep-direct",
            "multivariate-single-multistep-direct",
            "univariate-single-multistep-direct-recursive",
            "multivariate-single-multistep-direct-recursive",
        ]:
            (df_series_featured, exogenous_features) = self._expand_horizon_exogenous_for_direct(
                df=df_series_featured,
                exogenous_features=exogenous_features,
                horizon=model_horizon,
                origin_frozen_features=origin_frozen_features,
            )
        # Global 模式：保留序列 ID 作为静态外生类别特征
        if getattr(self.args, "enable_global_training", False):
            series_id_col = getattr(self.args, "series_id_feature", "series_id")
            if series_id_col in df_series_featured.columns:
                if series_id_col not in exogenous_features:
                    exogenous_features.append(series_id_col)
                if series_id_col not in categorical_features:
                    categorical_features.append(series_id_col)

        # 历史数据特征工程: 内生变量基本特征工程
        if self.verbose:
            logger.info(f"{self.log_prefix} 数据特征工程: 内生变量基本特征...")
        (df_series_featured, endogenous_features, target_output_features) = self.create_endogenous_basic_features(
            df_series=df_series_featured, 
            endogenous_features_with_target=endogenous_features_with_target_copy,
            target_feature=target_feature_copy, 
            horizon=model_horizon,
        )

        # 历史数据特征工程: 内生变量高级特征工程
        if self.verbose:
            logger.info(f"{self.log_prefix} 数据特征工程: 内生变量高级特征...")
        (df_series_featured, endogenous_advanced_features) = self.create_endogenous_advanced_features(
            df_series=df_series_featured
        )
        if self.verbose:
            logger.info(f"{self.log_prefix} 特征工程结束...")
        if self.verbose:
            logger.info(f"{self.log_prefix} after feature engineering exogenous_features: {exogenous_features}")
            logger.info(f"{self.log_prefix} after feature engineering endogenous_basic_features: {endogenous_features}")
            logger.info(f"{self.log_prefix} after feature engineering endogenous_advanced_features: {endogenous_advanced_features}")
            logger.info(f"{self.log_prefix} after feature engineering target_output_features: {target_output_features}")
            logger.info(f"{self.log_prefix} after feature engineering categorical_features: {categorical_features}")
        # ------------------------------
        # Feature ordering
        # ------------------------------
        # 预测特征 = 外生变量特征 + 内生变量特征（滞后特征） + 内生变量高级特征（滞后统计特征）
        predictor_features = exogenous_features + endogenous_features + endogenous_advanced_features
        if self.verbose:
            logger.info(f"{self.log_prefix} after feature engineering predictor_features: {predictor_features}")
        # 所有特征
        # all_cols_needed = ["time"] + predictor_features + target_output_features
        all_cols_needed = predictor_features + target_output_features
        if self.verbose:
            logger.info(f"{self.log_prefix} after feature engineering all_cols_needed: {all_cols_needed}")
        # 只保留需要的特征
        df_series_featured = df_series_featured[all_cols_needed]
        if self.verbose:
            logger.info(f"{self.log_prefix} after feature engineering df_series_featured: \n{df_series_featured}")
            logger.info(f"{self.log_prefix} after feature engineering df_series_featured shape: {df_series_featured.shape}")

        return df_series_featured, predictor_features, target_output_features, categorical_features

    def predictor_target_split(self, df_series_featured, predictor_features, target_output_features):
        """
        历史数据预测特征、目标特征分离
        """
        X_train_history = df_series_featured[predictor_features]
        Y_train_history = df_series_featured[target_output_features]
        combined_xy = pd.concat([X_train_history, Y_train_history], axis=1)
        # 训练样本仅按目标列过滤，外生缺失在前面已进行兜底填充
        combined_xy = combined_xy.dropna(subset=target_output_features)
        X_train_history = combined_xy[X_train_history.columns]
        Y_train_history = combined_xy[Y_train_history.columns]
        if self.verbose:
            logger.info(f"{self.log_prefix} after predictor_target_split X_train_history: \n{X_train_history.head()}")
            logger.info(f"{self.log_prefix} after predictor_target_split X_train_history.shape: {X_train_history.shape}")
            logger.info(f"{self.log_prefix} after predictor_target_split Y_train_history: \n{Y_train_history.head()}")
            logger.info(f"{self.log_prefix} after predictor_target_split Y_train_history.shape: {Y_train_history.shape}")
        
        return X_train_history, Y_train_history




# 测试代码 main 函数
def main():
    # ------------------------------
    # 创建示例时间序列数据
    # ------------------------------
    np.random.seed(42)
    df = pd.DataFrame({
        'time': pd.date_range('2024-01-01', periods=1000, freq='H'),
        'load': np.random.randn(1000).cumsum() + 100,
        'temperature': 20 + 10 * np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000),
    })
    print(df)

    # ------------------------------
    # 创建特征工程器
    # ------------------------------
    fe = EndogenousAdvancedFeatureEngineer()
    # ------------------------------
    # 添加各种特征
    # ------------------------------
    df = fe.add_rolling_statistics(df, ["load", "temperature"], windows=[3, 7, 24], stats=['mean', 'std', 'min', 'max', 'median', "skew", "kurt"])
    fe.reset()
    df = fe.add_expanding_statistics(df, ["load", "temperature"], stats=['mean', 'std', 'min', 'max', 'median', "skew", "kurt"])
    fe.reset()
    df = fe.add_diff_features(df, ['load'], periods=[1, 24])
    df = fe.add_interaction_features(df, [('load', 'temperature')], operations=['multiply'])
    
    print(f"原始特征数: 3")
    print(f"生成特征数: {len(fe.get_generated_features())}")
    print(f"总特征数: {len(df.columns)}")
    print(f"\n生成的特征列表:")
    for feat in fe.get_generated_features():#[:10]:
        print(f"  - {feat}")

    print(df)

if __name__ == "__main__":
    main()
