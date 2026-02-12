# -*- coding: utf-8 -*-

# ***************************************************
# * File        : model_config.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-11
# * Version     : 1.0.021110
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
from typing import List, Dict, Optional
from pathlib import Path
import datetime
from dataclasses import dataclass, field


@dataclass
class ModelConfig_univariate:
    """
    模型配置类
    包含数据路径、特征设置、模型参数等所有配置项
    """
    # 目标时间序列配置
    data_dir: str = "./dataset/electricity_work/demand_load/lingang_A"
    data_path: str = "AIDC_A_dataset.csv"
    data: str = "AIDC_A_dataset"
    freq: str = "5min"
    freq_minutes: int = 5
    target_ts_feat: str = "count_data_time"
    target_series_numeric_features: List[str] = field(default_factory=list)
    target_series_categorical_features: List[str] = field(default_factory=list)
    target_series_drop_features: List[str] = field(default_factory=list)
    target: str = "h_total_use"

    # 日期类型数据配置
    # date_history_path: Optional[str] = "df_date.csv"
    # date_future_path: Optional[str] = "df_date_future.csv"
    # date_ts_feat: Optional[str] = "date"
    date_history_path: Optional[str] = None
    date_future_path: Optional[str] = None
    date_ts_feat: Optional[str] = None
    
    # 天气数据配置
    # weather_history_path: Optional[str] = "df_weather.csv"
    # weather_future_path: Optional[str] = "df_weather_future.csv"
    # weather_ts_feat: Optional[str] = "ts"
    weather_history_path: Optional[str] = None
    weather_future_path: Optional[str] = None
    weather_ts_feat: Optional[str] = None

    # 数据预处理
    scale: bool = False  # 是否进行归一化/标准化
    inverse: bool = False  # 目标变量是否进行归一化/标准化逆变换
    scaler_type: str = "minmax"  # "standard" 或 "minmax"
    use_grouped_scaling: str = False

    # 特征工程配置
    # 特征滞后数列表
    lags: List[int] = field(default_factory=lambda: [
        # 1 * 288,  # Daily lag
        # 2 * 288,
        # 3 * 288,
        # 4 * 288,
        # 5 * 288,
        # 6 * 288,
        # 7 * 288,  # Weekly lag
    ])
    # 日期时间特征
    datetime_features: List[str] = field(default_factory=lambda: [
        'minute', 'hour', 'day', 'weekday', 'week',
        'day_of_week', 'week_of_year', 'month', 'days_in_month',
        'quarter', 'day_of_year', 'year',
    ])
    # 节假日特征
    datetype_features: List[str] = field(fault_factory=lambda: [
        # "date_type"
    ])
    # 天气特征
    weather_features: List[str] = field(default_factory=lambda: [
        # "rt_ssr",   # 太阳总辐射
        # "rt_ws10",  # 10m 风速
        # "rt_tt2",   # 2M 气温
        # "cal_rh",   # 相对湿度
        # "rt_ps",    # 气压
        # "rt_rain",  # 降雨量
    ])
    # 日期时间类别特征
    datetime_categorical_features: List[str] = field(default_factory=lambda: [
        # "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week",
        # "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month",
        # "datetime_quarter", "datetime_day_of_year", "datetime_year",
    ])
    # 节假日特征中的类别特征
    datetype_categorical_features: List[str] = field(default_factory=lambda: [
        # "date_type"
    ])
    # 天气特征中的类别特征
    weather_categorical_features: List[str] = field(default_factory=lambda: [])
    
    # 高级特征工程配置
    enable_advanced_features: bool = False
    enable_rolling_features: bool = False
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 28])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "skew", "kurt"])
    enable_expanding_features: bool = False
    enable_diff_features: bool = True
    diff_period: List[int] = field(default_factory=lambda: [1, 7])
    enable_cyclical_features: bool = True
    enable_interaction_features: bool = True

    # 训练和预测配置
    history_days: int = 31  # 历史数据天数
    predict_days: int = 1  # 预测未来 1 天的数据
    window_days: int = 15  # 滑动窗口天数
    encode_categorical_features: bool = False  # 是否对类别特征进行编码

    # 模型配置
    # 单模型预测
    model_type: str = "lightgbm"
    model_params: Dict = {}
    # 模型融合预测
    enable_ensemble: bool = False
    ensemble_models: List = field(default_factory=lambda: ["lgb", "xgb", "cat"])
    ensemble_method: str = "stacking"  # 'averaging', 'weighted', 'stacking', "blending"
    # 可选预测方法:
    # - 单变量预测单变量
    pred_method: str = "univariate-single-multistep-direct-output"       # USMDO [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接输出预测
    # pred_method: str = "univariate-single-multistep-direct"              # USMD [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接预测
    # pred_method: str = "univariate-single-multistep-recursive"           # USMR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测
    # pred_method: str = "univariate-single-multistep-direct-recursive"    # USMDR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接递归预测
    # - 多变量预测单变量
    # pred_method: str = "multivariate-single-multistep-direct"            # MSMD [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接预测
    # pred_method: str = "multivariate-single-multistep-recursive"         # MSMR [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测
    # pred_method: str = "multivariate-single-multistep-direct-recursive"  # MSMDR [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接递归预测
    objective: str = "regression_l1"  # 训练目标
    loss: str = "mae"  # 训练损失函数
    learning_rate: float = 0.05  # 模型学习率
    patience: int = 100  # 早停步数

    # 模型运行模式
    is_testing: bool = False
    is_forecasting: bool = True
    # 预测推理开始的时间
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2025, 12, 27, 0, 0, 0))

    # 结果保存路径
    checkpoints_dir: str = "./saved_results/pretrained_models/"
    test_results_dir: str = "./saved_results/results_test/"
    pred_results_dir = "./saved_results/results_forecast/"

    # 超参数调优
    perform_tuning: bool = False
    tuning_metric: str = "neg_mean_absolute_error"
    tuning_n_splits: int = 3


@dataclass
class ModelConfig_multivariate:
    """
    模型配置类
    包含数据路径、特征设置、模型参数等所有配置项
    """
    # ------------------------------
    # 数据配置
    # ------------------------------
    # 目标时间序列配置
    # -----------------
    data_dir = Path("./dataset/ETT-small")
    data_path = "ETTm1.csv"
    data = "ETTm1"
    freq = "15min"
    freq_minutes = 15
    target_ts_feat = "date"
    target_series_numeric_features = [
        "HUFL", #"HULL", "MUFL", "MULL", "LUFL", "LULL"
    ]
    target_series_categorical_features = []
    target_series_drop_features = [
        "HULL", "MUFL", "MULL", "LUFL", "LULL"
    ]
    target = "OT"
    # 日期类型数据配置
    # -----------------
    date_history_path = None
    date_future_path = None
    date_ts_feat = None
    # 天气数据配置
    # -----------------
    weather_history_path = None
    weather_future_path = None
    weather_ts_feat = None
    # ------------------------------
    # 数据预处理
    # ------------------------------
    # 是否进行归一化/标准化
    scale = False
    # 目标变量是否进行归一化/标准化逆变换
    inverse = False
    scaler_type = "standard"
    use_grouped_scaling = False
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    # 特征滞后数列表
    lags = [
        1 * 288,  # Daily lag
        2 * 288,
        3 * 288,
        4 * 288,
        5 * 288,
        6 * 288,
        7 * 288,  # Weekly lag
    ]
    # 日期时间特征
    datetime_features = [
        # 'minute', 'hour', 'day', 'weekday', 'week',
        # 'day_of_week', 'week_of_year', 'month', 'days_in_month',
        # 'quarter', 'day_of_year', 'year',
    ]
    # 节假日特征
    datetype_features = []
    # 天气特征
    weather_features = []
    # 日期时间特征中的类别特征
    datetime_categorical_features = [
        # "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week",
        # "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month",
        # "datetime_quarter", "datetime_day_of_year", "datetime_year",
    ]
    # 节假日特征中的类别特征
    datetype_categorical_features = []
    # 天气特征中的类别特征
    weather_categorical_features = []
    # ------------------------------
    # 训练和预测配置
    # ------------------------------
    # 历史数据天数
    history_days = 31
    # 预测未来 1 天的数据
    predict_days = 1
    # 滑动窗口天数
    window_days = 15
    # 是否对类别特征进行编码
    encode_categorical_features = False
    # ------------------------------
    # 模型配置
    # ------------------------------
    model_type = "lightgbm"
    model_params = {}
    ensemble_models = ["lgb", "xgb", "cat"]
    ensemble_method = "stacking"
    # 可选预测方法:
    # - 单变量预测单变量
    # pred_method = "univariate-single-multistep-direct-output"       # USMDO [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接输出预测
    # pred_method = "univariate-single-multistep-direct"              # USMD [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接预测
    # pred_method = "univariate-single-multistep-recursive"             # USMR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测
    # pred_method = "univariate-single-multistep-direct-recursive"    # USMDR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接递归预测
    # - 多变量预测单变量
    pred_method = "multivariate-single-multistep-direct"            # MSMD [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接预测
    # pred_method = "multivariate-single-multistep-recursive"         # MSMR [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测
    # pred_method = "multivariate-single-multistep-direct-recursive"  # MSMDR [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接递归预测
    # 早停步数
    patience = 100
    # ------------------------------
    # 模型运行模式
    # ------------------------------
    is_testing = False
    is_forecasting = False
    # 预测推理开始的时间
    now_time = datetime.datetime(2018, 6, 26, 19, 45, 0)
    # ------------------------------
    # result saved
    # ------------------------------
    checkpoints_dir = "./saved_results/pretrained_models/"
    test_results_dir = "./saved_results/results_test/"
    pred_results_dir = "./saved_results/results_forecast/"
    # ------------------------------
    # hyperparameter tuning
    # ------------------------------
    # Set to True to enable tuning
    perform_tuning = False
    # Metric for hyperparameter tuning
    tuning_metric = "neg_mean_absolute_error"
    # Number of splits for TimeSeriesSplit cross-validation
    tuning_n_splits = 3




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
