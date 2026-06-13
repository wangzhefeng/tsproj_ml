# -*- coding: utf-8 -*-
import datetime
from dataclasses import dataclass, field

from config.config_sections import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """electricity_univariate 单变量示例配置。

    继承的基类保持扁平对外接口，运行代码仍然读取 `cfg.data_path`、
    `cfg.target`、`cfg.pred_method` 等字段。本文件只覆盖单变量
    示例数据集需要从共享配置中单独提出的默认值。
    """

    # 运行窗口：以 now_time 为预测锚点，历史/未来窗口由天数参数推导。
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2026, 6, 11, 23, 55, 0))

    # 目标序列：单变量电力负荷示例，默认只使用目标列自身历史。
    data_dir: str = "./dataset/electricity_univariate/"
    data_path: str = "df_power.csv"
    freq: str = "5min"
    target_ts_feat: str = "count_data_time"
    target: str = "h_total_use"

    # 外生特征：示例目录提供日期类型和天气文件。
    enable_date_features: bool = True
    date_history_path: str | None = "df_date.csv"
    date_future_path: str | None = "df_date_future.csv"
    date_ts_feat: str | None = "date"
    datetype_features: list[str] = field(default_factory=lambda: ["date_type"])

    enable_weather_features: bool = True
    weather_history_path: str | None = "df_weather.csv"
    weather_future_path: str | None = "df_weather_future.csv"
    weather_ts_feat: str | None = "ts"
    weather_features: list[str] = field(
        default_factory=lambda: [
            "rt_ssr",
            "rt_ws10",
            "rt_tt2",
            "cal_rh",
            "rt_ps",
            "rt_rain",
        ]
    )
    weather_categorical_features: list[str] = field(default_factory=list)

    # 预测策略：USMD = 单变量输入，多步直接预测。
    pred_method: str = "univariate-single-multistep-direct"
