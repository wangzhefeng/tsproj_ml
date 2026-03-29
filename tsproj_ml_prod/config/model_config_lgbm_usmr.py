# -*- coding: utf-8 -*-
import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """
    模型配置类
    包含生产环境 USMR 所需配置项
    """
    model_cfgs: Optional[Dict] = field(default=None, repr=False)
    # ------------------------------
    # 项目配置
    # ------------------------------
    node_id: Optional[int] = None  # 节点 ID
    out_system_id: Optional[str] = None  # 输出系统 ID
    # ------------------------------
    # 模型运行模式
    # ------------------------------
    is_testing: bool = False  # 模型测试
    is_forecasting: bool = True  # 模型预测
    history_days: int = 92  # 历史数据天数
    predict_days: int = 1  # 预测未来 1 天的数据
    window_days: int = 31  # 滑动窗口总天数(30天训练 + 1天测试)
    # 预测推理开始的时间
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2026, 1, 1, 0, 0, 0))
    start_time: Optional[datetime.datetime] = None
    future_time: Optional[datetime.datetime] = None
    # ------------------------------
    # 目标时间序列配置
    # ------------------------------
    freq: str = "5min"  # 时间序列频率
    target_ts_feat: str = "count_data_time"  # 目标时间戳列
    target: str = "h_total_use"  # 预测目标列
    target_series_numeric_features: List[str] = field(default_factory=list)
    target_series_categorical_features: List[str] = field(default_factory=list)
    target_series_drop_features: List[str] = field(default_factory=list)
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    # 日期类型数据配置
    # --------------
    enable_date_features: bool = True
    if enable_date_features:
        date_ts_feat: Optional[str] = "date"
        datetype_features: List[str] = field(default_factory=lambda: ["date_type"])
        datetype_categorical_features: List[str] = field(default_factory=lambda: [])
    else:
        date_ts_feat: Optional[str] = None
        datetype_features: List[str] = field(default_factory=lambda: [])
        datetype_categorical_features: List[str] = field(default_factory=lambda: [])
    # 气象数据配置
    # --------------
    enable_weather_features: bool = True
    if enable_weather_features:
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
        weather_ts_feat: Optional[str] = None
        weather_features: List[str] = field(default_factory=lambda: [])
        weather_categorical_features: List[str] = field(default_factory=lambda: [])
    # 日期时间特征
    # --------------
    enable_datetime_features: bool = True
    if enable_datetime_features:
        datetime_features: List[str] = field(default_factory=lambda: [
            "minute", "hour", "day", "weekday", "week",
            "day_of_week", "week_of_year", "month", "days_in_month",
            "quarter", "day_of_year", "year",
        ])
        datetime_categorical_features: List[str] = field(default_factory=lambda: [])
    else:
        datetime_features: List[str] = field(default_factory=lambda: [])
        datetime_categorical_features: List[str] = field(default_factory=lambda: [])
    # 特征滞后数列表
    # --------------
    enable_lags_features: bool = True
    if enable_lags_features:
        lags: List[int] = field(default_factory=lambda: [
            1 * 288,
            2 * 288,
            7 * 288,
        ])
    else:
        lags: List[int] = field(default_factory=lambda: [])
    # 高级特征工程配置
    # --------------
    enable_advanced_features: bool = True

    enable_rolling_features: bool = True
    rolling_columns: List[str] = field(default_factory=lambda: ["y"])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 28])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])

    enable_expanding_features: bool = True
    expanding_columns: List[str] = field(default_factory=lambda: ["y"])
    expanding_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max"])

    enable_diff_features: bool = True
    diff_columns: List[str] = field(default_factory=lambda: ["y"])
    diff_periods: List[int] = field(default_factory=lambda: [1, 7, 24])

    enable_pct_change_features: bool = True
    pct_change_columns: List[str] = field(default_factory=lambda: ["y"])
    pct_change_periods: List[int] = field(default_factory=lambda: [1, 7])

    enable_time_since_features: bool = True
    time_since_columns: List[str] = field(default_factory=lambda: ["y"])
    time_since_events: List[str] = field(default_factory=lambda: ["peak", "trough"])

    enable_cyclical_features: bool = True
    cyclical_columns: List[str] = field(default_factory=lambda: ["minute"])
    cyclical_period: int = 15

    enable_interaction_features: bool = True
    interaction_column_pairs: List[tuple] = field(default_factory=lambda: [("y", "dt_hour")])
    interaction_operations: List[str] = field(default_factory=lambda: ["add", "subtract", "multiply", "divide"])

    enable_polynomial_features: bool = True
    polynomial_columns: List[str] = field(default_factory=lambda: ["y"])
    polynomial_degree: int = 2
    # ------------------------------
    # 模型配置
    # ------------------------------
    # 单模型预测
    model_type: str = "lightgbm"
    model_params: Dict = field(default_factory=lambda: {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "n_estimators": 300,
        "max_depth": 6,
        "max_bin": 63,
        "num_leaves": 15,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "lambda_l1": 1.0,
        "lambda_l2": 1.0,
        "verbose": -1,
    })

    # 可选预测方法:
    # - 单变量预测单变量
    pred_method: str = "univariate-single-multistep-recursive"  # USMR [单变量(包含目标变量的所有内生变量)->单变量(目标内生变量)]多步递归预测

    # 多输出策略: multioutput / regressor_chain
    multi_output_strategy: str = "multioutput"

    # Direct 方法是否使用 horizon-aware 外生特征展开
    use_horizon_exogenous_for_direct: bool = False

    # Direct-Recursive 方法的分块大小
    block_size: int = 0
    # ------------------------------
    # 性能与并行配置
    # ------------------------------
    window_parallel_workers: int = 1  # 滑窗并行进程数
    multi_output_n_jobs: int = 1  # 多输出模型并行数
    model_thread_count: int = 1  # 单模型线程数
    enable_step_logging: bool = False
    forecast_log_interval: int = 24
    # ------------------------------
    # 结果保存路径
    # ------------------------------
    test_results_dir: str = "./tsproj_ml_prod/results/results_test/"
    pred_results_dir: str = "./tsproj_ml_prod/results/results_forecast/"

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
