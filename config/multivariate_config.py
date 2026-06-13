# -*- coding: utf-8 -*-
import datetime
from dataclasses import dataclass, field

from config.config_sections import BaseModelConfig, default_lags_for_freq


@dataclass
class ModelConfig(BaseModelConfig):
    """ETTm1 多变量示例配置。

    继承的基类保持扁平对外接口，运行代码仍然读取 `cfg.data_path`、
    `cfg.target`、`cfg.pred_method` 等字段。本文件只覆盖 ETTm1
    多变量示例需要从共享配置中单独提出的默认值。
    """

    # 运行窗口：ETTm1 默认锚点沿用历史实验配置。
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2018, 6, 26, 19, 45, 0))

    # 目标序列：OT 为预测目标，其余负荷列作为显式内生数值特征。
    data_dir: str = "./dataset/ETT-small/"
    data_path: str = "ETTm1.csv"
    freq: str = "15min"
    target_ts_feat: str = "date"
    target: str = "OT"
    target_series_numeric_features: list[str] = field(default_factory=lambda: ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"])

    # 外生特征：ETTm1 原始数据不带外生文件，这里使用仿真生成的示例外生数据。
    enable_date_features: bool = True
    date_history_path: str | None = "ETTm1_exogenous/df_date.csv"
    date_future_path: str | None = "ETTm1_exogenous/df_date_future.csv"
    date_ts_feat: str | None = "date"
    datetype_features: list[str] = field(default_factory=lambda: ["date_type"])

    enable_weather_features: bool = True
    weather_history_path: str | None = "ETTm1_exogenous/df_weather.csv"
    weather_future_path: str | None = "ETTm1_exogenous/df_weather_future.csv"
    weather_ts_feat: str | None = "ts"

    # ETTm1 既可以保留数值时间特征，也可将离散时间列作为类别特征。
    datetime_categorical_features: list[str] = field(
        default_factory=lambda: [
            "dt_hour",
            "dt_day",
            "dt_weekday",
            "dt_week",
            "dt_day_of_week",
            "dt_week_of_year",
            "dt_month",
            "dt_days_in_month",
            "dt_quarter",
            "dt_day_of_year",
            "dt_year",
        ]
    )

    # 滞后步数：15min 频率下 1~7 天 = 96~672 步。
    lags: list[int] = field(default_factory=lambda: default_lags_for_freq("15min"))

    # 预测策略：MSMDR = 多变量输入，多步直接递归预测。
    pred_method: str = "multivariate-single-multistep-direct-recursive"
