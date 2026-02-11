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
import os
import sys
from pathlib import Path
ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)
import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]
os.environ['LOG_NAME'] = LOGGING_LABEL
from utils.log_util import logger


@dataclass
class ModelConfig_univariate:
    """
    模型配置类
    包含数据路径、特征设置、模型参数等所有配置项
    """
    # ------------------------------
    # 数据配置
    # ------------------------------
    # 目标时间序列配置
    # -----------------
    data_dir = "./dataset/electricity_work/demand_load/lingang_A"
    data_path = "AIDC_A_dataset.csv"              # 目标时间序列数据路径
    data = "AIDC_A_dataset"                       # 目标时间序列数据名称
    freq = "5min"                                 # 目标时间序列数据频率
    freq_minutes = 5                              # 目标时间序列数据频率(分钟)
    target_ts_feat = "count_data_time"            # 目标时间序列时间戳特征名称
    target_series_numeric_features = []           # 目标时间序列的数值特征 (其他内生变量)
    target_series_categorical_features = []       # 目标时间序列的类别特征
    target_series_drop_features = []              # 目标时间序列的删除特征
    target = "h_total_use"                        # 目标时间序列预测目标变量名称
    # 日期类型数据配置
    # -----------------
    # date_history_path = "df_date.csv"
    # date_future_path = "df_date_future.csv"
    # date_ts_feat = "date"
    date_history_path = None
    date_future_path = None
    date_ts_feat = None
    
    # 天气数据配置
    # -----------------
    # weather_history_path = "df_weather.csv"
    # weather_future_path = "df_weather_future.csv"
    # weather_ts_feat = "ts"
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
    scaler_type = "minmax"# "standard"
    use_grouped_scaling = False
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    use_advanced_features = True
    rolling_windows = [3, 7, 14, 28]
    use_diff_features = True
    use_cyclical_features = True
    use_interaction_features = True
    # 特征滞后数列表
    lags = [
        # 1 * 288,  # Daily lag
        # 2 * 288,
        # 3 * 288,
        # 4 * 288,
        # 5 * 288,
        # 6 * 288,
        # 7 * 288,  # Weekly lag
    ]
    # 日期时间特征
    datetime_features = [
        'minute', 'hour', 'day', 'weekday', 'week',
        'day_of_week', 'week_of_year', 'month', 'days_in_month',
        'quarter', 'day_of_year', 'year',
    ]
    # 节假日特征
    datetype_features = [
        # "date_type"
    ]
    # 天气特征
    weather_features = [
        # "rt_ssr",   # 太阳总辐射
        # "rt_ws10",  # 10m 风速
        # "rt_tt2",   # 2M 气温
        # "cal_rh",   # 相对湿度
        # "rt_ps",    # 气压
        # "rt_rain",  # 降雨量
    ]
    # 日期时间类别特征
    datetime_categorical_features = [
        # "datetime_hour", "datetime_day", "datetime_weekday", "datetime_week",
        # "datetime_day_of_week", "datetime_week_of_year", "datetime_month", "datetime_days_in_month",
        # "datetime_quarter", "datetime_day_of_year", "datetime_year",
    ]
    # 节假日特征中的类别特征
    datetype_categorical_features = [
        # "date_type"
    ]
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
    model_type = "ensemble"
    ensemble_models = ["lgb", "xgb", "cat"]
    ensemble_method = "stacking"
    model_name = "LightGBM"
    # 可选预测方法:
    # - 单变量预测单变量
    pred_method = "univariate-single-multistep-direct-output"       # USMDO [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接输出预测
    # pred_method = "univariate-single-multistep-direct"              # USMD [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接预测
    # pred_method = "univariate-single-multistep-recursive"             # USMR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测
    # pred_method = "univariate-single-multistep-direct-recursive"    # USMDR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接递归预测
    # - 多变量预测单变量
    # pred_method = "multivariate-single-multistep-direct"            # MSMD [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接预测
    # pred_method = "multivariate-single-multistep-recursive"         # MSMR [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测
    # pred_method = "multivariate-single-multistep-direct-recursive"  # MSMDR [多变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步直接递归预测
    # 训练目标
    objective = "regression_l1"
    # 训练损失函数
    loss = "mae"
    # 模型学习率
    learning_rate = 0.05
    # 早停步数
    patience = 100
    # ------------------------------
    # 模型运行模式
    # ------------------------------
    is_testing = False
    is_forecasting = True
    # 预测推理开始的时间
    now_time = datetime.datetime(2025, 12, 27, 0, 0, 0)
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
    model_name = "LightGBM"
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
    # 训练目标
    objective = "regression_l1"
    # 训练损失函数
    loss = "mae"
    # 模型学习率
    learning_rate = 0.05
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
