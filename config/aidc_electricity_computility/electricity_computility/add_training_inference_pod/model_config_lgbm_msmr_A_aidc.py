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
    model_cfgs: Optional[Dict] = field(default=None, repr=False)
    # ------------------------------
    # 项目配置
    # ------------------------------
    node_id: Optional[int] = None
    out_system_id: Optional[str] = None
    # ------------------------------
    # 模型运行模式
    # ------------------------------
    is_testing: bool = False
    is_forecasting: bool = True
    history_days: int = 31
    predict_days: int = 1
    window_days: int = 15
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2026, 1, 14, 0, 0, 0))
    start_time: Optional[datetime.datetime] = None
    future_time: Optional[datetime.datetime] = None
    # ------------------------------
    # 目标时间序列配置
    # ------------------------------
    data_dir: str = "./dataset/aidc_electricity_computility/electricity_computility/add_training_inference_pod/"
    data_path: str = "dataset_electricity_with_computility_A.csv"
    freq: str = "5min"
    target_ts_feat: str = "time"
    target: str = "y"
    target_series_numeric_features: List[str] = field(default_factory=lambda: [
        "cpu_util_all_jobs_min",
        "cpu_util_all_jobs_max",
        "cpu_util_all_jobs_mean",
        "cpu_util_all_jobs_std",
        "cpu_util_all_jobs_pod_count",
        "gpu_memory_amount_all_jobs_min",
        "gpu_memory_amount_all_jobs_max",
        "gpu_memory_amount_all_jobs_mean",
        "gpu_memory_amount_all_jobs_std",
        "gpu_memory_amount_all_jobs_sum",
        "gpu_memory_amount_all_jobs_pod_count",
        "gpu_memory_total_all_jobs_min",
        "gpu_memory_total_all_jobs_max",
        "gpu_memory_total_all_jobs_mean",
        "gpu_memory_total_all_jobs_std",
        "gpu_memory_total_all_jobs_sum",
        "gpu_memory_total_all_jobs_pod_count",
        "gpu_memory_util_all_jobs_min",
        "gpu_memory_util_all_jobs_max",
        "gpu_memory_util_all_jobs_mean",
        "gpu_memory_util_all_jobs_std",
        "gpu_memory_util_all_jobs_pod_count",
        "gpu_power_usage_all_jobs_min",
        "gpu_power_usage_all_jobs_max",
        "gpu_power_usage_all_jobs_mean",
        "gpu_power_usage_all_jobs_std",
        "gpu_power_usage_all_jobs_sum",
        "gpu_power_usage_all_jobs_pod_count",
        "gpu_util_all_jobs_min",
        "gpu_util_all_jobs_max",
        "gpu_util_all_jobs_mean",
        "gpu_util_all_jobs_std",
        "gpu_util_all_jobs_pod_count",
        "memory_amount_all_jobs_min",
        "memory_amount_all_jobs_max",
        "memory_amount_all_jobs_mean",
        "memory_amount_all_jobs_std",
        "memory_amount_all_jobs_sum",
        "memory_amount_all_jobs_pod_count",
        "memory_total_all_jobs_min",
        "memory_total_all_jobs_max",
        "memory_total_all_jobs_mean",
        "memory_total_all_jobs_std",
        "memory_total_all_jobs_sum",
        "memory_total_all_jobs_pod_count",
        "memory_util_all_jobs_min",
        "memory_util_all_jobs_max",
        "memory_util_all_jobs_mean",
        "memory_util_all_jobs_std",
        "memory_util_all_jobs_pod_count",
    ])
    target_series_categorical_features: List[str] = field(default_factory=list)
    target_series_drop_features: List[str] = field(default_factory=list)
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    enable_date_features: bool = False
    if enable_date_features:
        date_history_path: Optional[str] = "df_date.csv"
        date_future_path: Optional[str] = "df_date_future.csv"
        date_ts_feat: Optional[str] = "date"
        datetype_features: List[str] = field(default_factory=lambda: ["date_type"])
        datetype_categorical_features: List[str] = field(default_factory=list)
    else:
        date_history_path: Optional[str] = None
        date_future_path: Optional[str] = None
        date_ts_feat: Optional[str] = None
        datetype_features: List[str] = field(default_factory=list)
        datetype_categorical_features: List[str] = field(default_factory=list)

    enable_weather_features: bool = False
    if enable_weather_features:
        weather_history_path: Optional[str] = "df_weather.csv"
        weather_future_path: Optional[str] = "df_weather_future.csv"
        weather_ts_feat: Optional[str] = "ts"
        weather_features: List[str] = field(default_factory=lambda: [
            "rt_ssr",
            "rt_ws10",
            "rt_tt2",
            "cal_rh",
            "rt_ps",
            "rt_rain",
        ])
        weather_categorical_features: List[str] = field(default_factory=list)
    else:
        weather_history_path: Optional[str] = None
        weather_future_path: Optional[str] = None
        weather_ts_feat: Optional[str] = None
        weather_features: List[str] = field(default_factory=list)
        weather_categorical_features: List[str] = field(default_factory=list)

    enable_datetime_features: bool = True
    if enable_datetime_features:
        datetime_features: List[str] = field(default_factory=lambda: [
            "minute", "hour", "day", "weekday", "week",
            "day_of_week", "week_of_year", "month", "days_in_month",
            "quarter", "day_of_year", "year",
        ])
        datetime_categorical_features: List[str] = field(default_factory=list)
    else:
        datetime_features: List[str] = field(default_factory=list)
        datetime_categorical_features: List[str] = field(default_factory=list)

    enable_lags_features: bool = True
    if enable_lags_features:
        lags: List[int] = field(default_factory=lambda: [
            1 * 288,
            2 * 288,
            7 * 288,
        ])
    else:
        lags: List[int] = field(default_factory=list)

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
    time_since_events: List[str] = field(default_factory=lambda: ["peak", "thoughl"])

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
    scale_features: bool = False
    feature_scaler_type: str = "minmax"
    scale_target: bool = False
    inverse_target: bool = False
    target_scaler_type: str = "minmax"
    use_grouped_scaling: str = False
    # ------------------------------
    # 模型配置
    # ------------------------------
    model_type: str = "lightgbm"
    model_params: Dict = field(default_factory=lambda: {
        "boosting_type": "gbdt",
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": 1200,
        "learning_rate": 0.03,
        "max_bin": 63,
        "num_leaves": 31,
        "max_depth": -1,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 64,
        "lambda_l2": 1.0,
        "verbose": -1,
        "random_state": 42,
        "force_col_wise": True,
    })
    enable_ensemble: bool = False
    ensemble_models: List = field(default_factory=lambda: ["lgb", "xgb", "cat"])
    ensemble_method: str = "stacking"
    ensemble_val_ratio: float = 0.2

    pred_method: str = "multivariate-single-multistep-recursive"
    patience: int = 80
    encode_categorical_features: bool = False
    multi_output_strategy: str = "multioutput"
    predict_type: str = "point"
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    use_horizon_exogenous_for_direct: bool = False
    block_size: int = 0
    enable_global_training: bool = False
    series_id_feature: str = "series_id"
    perform_tuning: bool = False
    tuning_metric: str = "neg_mean_absolute_error"
    tuning_n_splits: int = 3
    enable_data_augmentation: bool = False
    augmentation_ratio: float = 0.2
    augmentation_feature_noise_std: float = 0.01
    augmentation_target_noise_std: float = 0.005
    augmentation_random_state: int = 42
    enable_feature_selection: bool = False
    feature_selection_method: str = "f_regression"
    feature_selection_max_features: int = 80
    feature_selection_min_features: int = 10
    enable_auto_learning_rate: bool = False
    auto_lr_min: float = 0.005
    auto_lr_max: float = 0.2
    huber_delta: float = 1.0
    window_parallel_workers: int = 4
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

    def __post_init__(self):
        self._apply_model_cfgs(self.model_cfgs or {})

    def _apply_model_cfgs(self, model_cfgs: Dict):
        if not model_cfgs:
            return

        nodes_cfg = model_cfgs.get("nodes", {})
        if isinstance(nodes_cfg, dict):
            node_cfg = nodes_cfg.get("node", nodes_cfg)
            if isinstance(node_cfg, dict):
                self.node_id = node_cfg.get("node_id", self.node_id)
                self.out_system_id = node_cfg.get("out_system_id", self.out_system_id)

        time_range = model_cfgs.get("time_range", {})
        if isinstance(time_range, dict):
            self.history_days = int(time_range.get("before_days", self.history_days))
            self.predict_days = int(time_range.get("after_days", self.predict_days))
            self.now_time = time_range.get("now_time", self.now_time)
            self.start_time = time_range.get("start_time", self.start_time)
            self.future_time = time_range.get("future_time", self.future_time)
