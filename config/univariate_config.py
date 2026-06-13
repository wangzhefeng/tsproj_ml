# -*- coding: utf-8 -*-
import datetime
from dataclasses import dataclass, field
from typing import Dict

from config.config_sections import BaseModelConfig


@dataclass
class ModelConfig(BaseModelConfig):
    """ETTm1 单变量示例配置。

    继承的基类保持扁平对外接口，运行代码仍然读取 `cfg.data_path`、
    `cfg.target`、`cfg.pred_method` 等字段。本文件只覆盖 ETTm1
    单变量示例需要从共享配置中单独提出的默认值。
    """

    # 运行窗口：ETTm1 默认锚点沿用历史实验配置。
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2018, 6, 26, 19, 45, 0))

    # 目标序列：OT 为预测目标，单变量只使用目标列自身历史。
    data_dir: str = "./dataset/ETT-small/"
    data_path: str = "ETTm1.csv"
    freq: str = "15min"
    target_ts_feat: str = "date"
    target: str = "OT"

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

    # ETTm1 既可将离散时间列作为类别特征传递给树模型，单变量与多变量共用此约定。
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

    # LightGBM 默认参数沿用 ETTm1 实验设置。
    model_params: Dict = field(
        default_factory=lambda: {
            "boosting_type": "gbdt",
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_bin": 63,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "force_col_wise": True,
        }
    )

    # 预测策略：USMD = 单变量输入，多步直接预测。
    pred_method: str = "univariate-single-multistep-direct"
