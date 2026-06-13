# -*- coding: utf-8 -*-
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """
    模型配置类
    包含数据路径、特征设置、模型参数等所有配置项
    """
    # ------------------------------
    # 模型运行模式
    # ------------------------------
    is_testing: bool = False  # 模型测试
    is_forecasting: bool = True  # 模型预测
    history_days: int = 31  # 历史数据天数
    predict_days: int = 1  # 预测未来天数
    window_days: int = 15  # 滑动窗口天数
    # 预测推理开始的时间
    now_time: datetime.datetime = datetime.datetime(2026, 6, 11, 23, 55, 0)
    # ------------------------------
    # 目标时间序列配置
    # ------------------------------
    data_dir: str = "./dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_IT/"
    data_path: str = "df_power.csv"
    freq: str = "5min"
    target_ts_feat: str = "count_data_time"
    target: str = "h_total_use"
    target_series_numeric_features: List[str] = field(default_factory=lambda: [])
    target_series_categorical_features: List[str] = field(default_factory=lambda: [])
    target_series_drop_features: List[str] = field(default_factory=list)
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    # 日期类型数据配置
    # --------------
    enable_date_features: bool = True
    if enable_date_features:
        date_history_path: Optional[str] = "df_date.csv"
        date_future_path: Optional[str] = "df_date_future.csv"
        date_ts_feat: Optional[str] = "date"
        datetype_features: List[str] = field(default_factory=lambda: ["date_type"])
        datetype_categorical_features: List[str] = field(default_factory=lambda: [])
    else:
        date_history_path: Optional[str] = None
        date_future_path: Optional[str] = None
        date_ts_feat: Optional[str] = None
        datetype_features: List[str] = field(default_factory=lambda: [])
        datetype_categorical_features: List[str] = field(default_factory=lambda: [])
    # 气象数据配置
    # --------------
    enable_weather_features: bool = True
    if enable_weather_features:
        weather_history_path: Optional[str] = "df_weather.csv"
        weather_future_path: Optional[str] = "df_weather_future.csv"
        weather_ts_feat: Optional[str] = "ts"
        weather_features: List[str] = field(default_factory=lambda: [
            "rt_ssr",   # 太阳总辐射
            "rt_ws10",  # 10m 风速
            "rt_tt2",   # 2M 气温
            "cal_rh",   # 相对湿度
            "rt_ps",    # 气压
            "rt_rain",  # 降雨量
        ])
        weather_categorical_features: List[str] = field(default_factory=lambda: [])
    else:
        weather_history_path: Optional[str] = None
        weather_future_path: Optional[str] = None
        weather_ts_feat: Optional[str] = None
        weather_features: List[str] = field(default_factory=lambda: [])
        weather_categorical_features: List[str] = field(default_factory=lambda: [])
    # 日期时间特征
    # --------------
    enable_datetime_features: bool = True
    if enable_datetime_features:
        datetime_features: List[str] = field(default_factory=lambda: [
            'minute', 'hour', 'day', 'weekday', 'week',
            'day_of_week', 'week_of_year', 'month', 'days_in_month',
            'quarter', 'day_of_year', 'year',
        ])
        datetime_categorical_features: List[str] = field(default_factory=lambda: [
            # "dt_hour", "dt_day", "dt_weekday", "dt_week",
            # "dt_day_of_week", "dt_week_of_year", "dt_month", "dt_days_in_month",
            # "dt_quarter", "dt_day_of_year", "dt_year",
        ])
    else:
        datetime_features: List[str] = field(default_factory=lambda: [])
        datetime_categorical_features: List[str] = field(default_factory=lambda: [])
    # 特征滞后数列表
    # --------------
    enable_lags_features: bool = True
    if enable_lags_features:
        lags: List[int] = field(default_factory=lambda: [
            1 * 288,  # Daily lag
            2 * 288,
            3 * 288,
            4 * 288,
            5 * 288,
            6 * 288,
            7 * 288,  # Weekly lag
        ])
    else:
        lags: List[int] = field(default_factory=lambda: [])
    # 高级特征工程配置
    # --------------
    enable_advanced_features: bool = False

    enable_rolling_features: bool = False
    rolling_columns: List[str] = field(default_factory=lambda: ["y"])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 28])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "skew", "kurt"])

    enable_expanding_features: bool = False
    expanding_columns: List[str] = field(default_factory=lambda: ["y"])
    expanding_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "skew", "kurt"])

    enable_diff_features: bool = False
    diff_columns: List[str] = field(default_factory=lambda: ["y"])
    diff_periods: List[int] = field(default_factory=lambda: [1, 7, 24])

    enable_pct_change_features: bool = False
    pct_change_columns: List[str] = field(default_factory=lambda: ["y"])
    pct_change_periods: List[int] = field(default_factory=lambda: [1, 7])

    enable_time_since_features: bool = False
    time_since_columns: List[str] = field(default_factory=lambda: ["y"])
    time_since_events: List[str] = field(default_factory=lambda: ["peak", "trough"])

    enable_cyclical_features: bool = False
    cyclical_columns: List[str] = field(default_factory=lambda: ["minute"])
    cyclical_period: int = field(default_factory=lambda: 15)

    enable_interaction_features: bool = False
    interaction_column_pairs: List[tuple] = field(default_factory=lambda: [("y", "dt_hour")])
    interaction_operations: List[str] = field(default_factory=lambda: ["add", "subtract", "multiply", "divide"])

    enable_polynomial_features: bool = False
    polynomial_columns: List[str] = field(default_factory=lambda: ["y"])
    polynomial_degree: int = field(default_factory=lambda: 2)
    # ------------------------------
    # 数据预处理
    # ------------------------------
    # 预测特征
    scale_features: bool = False  # 是否对预测特征 X 进行归一化/标准化
    feature_scaler_type: str = "minmax"  # 预测特征 X 的缩放方法: "standard" 或 "minmax"
    # 目标特征
    scale_target: bool = False  # 是否对目标变量 Y 进行归一化/标准化
    inverse_target: bool = False  # 预测结果是否对目标变量 Y 进行逆变换
    target_scaler_type: str = "minmax"  # 目标变量 Y 的缩放方法: "none"、"standard"、"minmax"、"log1p"、"robust" 或 "yeo-johnson"
    # 是否对特征使用分组归一化/标准化
    use_grouped_scaling: str = False
    # ------------------------------
    # 模型配置
    # ------------------------------
    # 单模型预测
    model_type: str = "xgboost"
    model_params: Dict = field(default_factory=dict)
    # 模型融合预测
    enable_ensemble: bool = False
    ensemble_models: List = field(default_factory=lambda: ["lgb", "xgb", "cat"])
    ensemble_method: str = "stacking"  # 'averaging', 'weighted', 'stacking', "blending"
    ensemble_val_ratio: float = 0.2

    # 可选预测方法:
    # - 单变量预测单变量
    # pred_method: str = "univariate-single-multistep-direct-output"       # USMDO
    # pred_method: str = "univariate-single-multistep-direct"              # USMD
    # pred_method: str = "univariate-single-multistep-recursive"           # USMR
    # pred_method: str = "univariate-single-multistep-direct-recursive"    # USMDR
    # - 多变量预测单变量
    # pred_method: str = "multivariate-single-multistep-direct"            # MSMD
    # pred_method: str = "multivariate-single-multistep-recursive"         # MSMR
    # pred_method: str = "multivariate-single-multistep-direct-recursive"  # MSMDR
    pred_method: str = "univariate-single-multistep-direct-recursive"
    # 早停步数
    patience: int = 100
    # 是否对类别特征进行编码
    encode_categorical_features: bool = False
    # 多输出策略: multioutput / regressor_chain
    multi_output_strategy: str = "multioutput"
    # 预测类型: point / quantile
    predict_type: str = "point"
    # 分位数预测配置（predict_type=quantile 时生效）
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    # Direct 方法是否使用 horizon-aware 外生特征展开
    use_horizon_exogenous_for_direct: bool = False
    # 全局训练模式（跨序列联合）
    enable_global_training: bool = False
    series_id_feature: str = "series_id"
    # 模型超参数调优
    perform_tuning: bool = False
    tuning_metric: str = "neg_mean_absolute_error"
    tuning_n_splits: int = 3
    # 数据增强（训练集）
    enable_data_augmentation: bool = False
    augmentation_ratio: float = 0.2
    augmentation_feature_noise_std: float = 0.01
    augmentation_target_noise_std: float = 0.005
    augmentation_random_state: int = 42
    # 特征选择（fit on train, reuse on test/forecast）
    enable_feature_selection: bool = False
    feature_selection_method: str = "f_regression"  # f_regression / mutual_info
    feature_selection_max_features: int = 80
    feature_selection_min_features: int = 10
    # 学习率策略
    enable_auto_learning_rate: bool = False
    auto_lr_min: float = 0.005
    auto_lr_max: float = 0.2
    # 鲁棒损失参数（用于 huber scorer）
    huber_delta: float = 1.0
    # ------------------------------
    # 滑窗测试训练集异常处理
    # ------------------------------
    enable_train_outlier_handling: bool = False
    train_outlier_method: str = "local_interpolate"
    high_outlier_threshold: float = 15000.0
    high_outlier_max_run_points: int = 4
    drop_outlier_max_run_points: int = 2
    drop_rebound_min_abs_diff: float = 900.0
    # ------------------------------
    # 性能与并行配置
    # ------------------------------
    window_parallel_workers: int = 1
    multi_output_n_jobs: int = 1
    quantile_parallel_workers: int = 1
    ensemble_parallel_workers: int = 1
    model_thread_count: int = 1
    enable_step_logging: bool = False
    forecast_log_interval: int = 24
    # ------------------------------
    # 结果保存路径
    # ------------------------------
    checkpoints_dir: str = "./saved_results/pretrained_models/"
    test_results_dir: str = "./saved_results/results_test/"
    pred_results_dir: str = "./saved_results/results_forecast/"
