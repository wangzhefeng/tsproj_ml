# -*- coding: utf-8 -*-
"""机器学习时间序列配置的共享分组定义。

本模块按职责拆分配置字段，提升阅读和维护效率；组合后的
`BaseModelConfig` 仍然暴露扁平 dataclass 字段。因此现有运行代码、
YAML 覆盖、CLI 覆盖和 `dataclasses.fields()` 仍然继续使用
`cfg.data_path`、`cfg.pred_method`、`cfg.pred_results_dir` 这类属性。
"""

import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 预测方法元信息 (full_method, short_code, description)
_PRED_METHODS = [
    ("univariate-single-multistep-direct-pointwise", "usmdp", "单变量输入，按未来时点逐点 direct 预测"),
    ("univariate-single-multistep-direct", "usmd", "单变量输入，多步直接预测"),
    ("univariate-single-multistep-recursive", "usmr", "单变量输入，多步递归预测"),
    ("univariate-single-multistep-direct-recursive", "usmdr", "单变量输入，多步直接递归预测"),
    ("multivariate-single-multistep-direct", "msmd", "多变量输入，多步直接预测"),
    ("multivariate-single-multistep-recursive", "msmr", "多变量输入，多步递归预测"),
    ("multivariate-single-multistep-direct-recursive", "msmdr", "多变量输入，多步直接递归预测"),
]

# 派生映射：保持向后兼容
PRED_METHOD_CODE = {full: code for full, code, _ in _PRED_METHODS}  # full name → short code


# 标准频率 → 每天样本数映射，用于生成与频率匹配的滞后步数。
_FREQ_SAMPLES_PER_DAY: Dict[str, int] = {
    "1min": 1440, "5min": 288, "15min": 96, "30min": 48,
    "1h": 24, "2h": 12, "4h": 6, "6h": 4, "8h": 3,
    "12h": 2, "1D": 1, "1W": 1,
}


def default_lags_for_freq(freq: str, days: int = 7) -> List[int]:
    """根据频率返回 1~N 天的滞后步数列表。

    Args:
        freq: pandas 频率字符串，如 ``"5min"``、``"15min"``、``"1h"``。
        days: 滞后天数，默认 7。

    Returns:
        滞后步数列表，例如 ``"15min"`` 返回 ``[96, 192, 288, 384, 480, 576, 672]``。
        无法识别频率时回退到 5min 基准（``n=288``）。
    """
    n = _FREQ_SAMPLES_PER_DAY.get(freq)
    if n is not None:
        return [n * d for d in range(1, days + 1)]

    # 回退：尝试从 "Nmin" 格式解析
    import re
    match = re.match(r"^(\d+)\s*min$", str(freq))
    if match:
        n = 24 * 60 // int(match.group(1))
        return [n * d for d in range(1, days + 1)]

    # 最终回退：5min 基准
    return [288 * d for d in range(1, days + 1)]


@dataclass
class RuntimeConfig:
    """运行模式和时间窗口配置。

    `now_time` 是预测锚点；运行时根据 `history_days`、`predict_steps`
    和 `window_days` 推导历史窗口、预测窗口和滑窗长度。
    """
    is_testing: bool = False  # 模型测试
    is_forecasting: bool = True  # 模型预测
    history_days: int = 31  # 历史数据天数
    predict_steps: int = 288  # 预测步数（以 freq 为单位；默认 288 = 5min 基准下 1 天）
    window_days: int = 15  # 滑动窗口天数
    now_time: datetime.datetime = field(default_factory=lambda: datetime.datetime(2026, 6, 11, 23, 55, 0))  # 预测推理开始时间
    schedule_mode: str = "daily"  # 调度模式: daily=日界对齐(预测下一完整自然日) | intraday=保留调度时刻(从调度时刻起预测)


@dataclass
class TargetSeriesConfig:
    """目标序列和内生变量列配置。

    `target_series_numeric_features` 为空时自动推断数值内生变量；
    非空时作为显式白名单。
    """
    data_dir: str = "./dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_201"
    data_path: str = "df_power.csv"
    freq: str = "5min"
    target_ts_feat: str = "count_data_time"
    target: str = "h_total_use"
    target_series_numeric_features: List[str] = field(default_factory=list)
    target_series_categorical_features: List[str] = field(default_factory=list)
    target_series_drop_features: List[str] = field(default_factory=list)


@dataclass
class ExogenousFeatureConfig:
    """外生特征配置。

    日期类型、气象、日期时间衍生特征都属于不由目标序列自身滞后直接生成的
    外生信息。`enable_*` 字段是运行时开关；路径字段决定开关启用时是否加载
    对应外部文件。
    """
    # 日期类型数据配置
    enable_date_features: bool = True
    date_history_path: Optional[str] = "df_date.csv"
    date_future_path: Optional[str] = "df_date_future.csv"
    date_ts_feat: Optional[str] = "date"
    datetype_features: List[str] = field(default_factory=lambda: ["date_type"])
    datetype_categorical_features: List[str] = field(default_factory=list)

    # 气象数据配置
    enable_weather_features: bool = True
    weather_history_path: Optional[str] = "df_weather.csv"
    weather_future_path: Optional[str] = "df_weather_future.csv"
    weather_ts_feat: Optional[str] = "ts"
    weather_features: List[str] = field(
        default_factory=lambda: [
            "rt_ssr",   # 太阳总辐射
            "rt_ws10",  # 10m 风速
            "rt_tt2",   # 2M 气温
            "cal_rh",   # 相对湿度
            "rt_ps",    # 气压
            "rt_rain",  # 降雨量
        ]
    )
    weather_categorical_features: List[str] = field(default_factory=list)

    # 自定义外生特征注册表（多文件来源，仿 date/weather 通路的通用版）。
    # 每个来源一个 dict：
    #   name: 来源标识（日志/调试用）
    #   history_path: 历史段 CSV（相对 data_dir）
    #   future_path: 未来段 CSV（forecast 用；可为 null——仅回测时）
    #   ts_col: 时间戳列名（精确时间戳 merge，与 weather 同粒度）
    #   columns: 使用的特征列名白名单
    #   categorical_columns: 其中按类别传给 LightGBM 的列名（默认 []）
    # 与 weather 通路的区别：无 rt_/pred_ 列名映射（历史/未来列名一致）、无硬编码列白名单。
    custom_features: List[Dict[str, Any]] = field(default_factory=list)

    # 日期时间特征：从时间戳直接派生，作为可预知外生变量使用。
    enable_datetime_features: bool = True
    datetime_features: List[str] = field(
        default_factory=lambda: [
            "minute",
            "hour",
            "day",
            "weekday",
            "week",
            "day_of_week",
            "week_of_year",
            "month",
            "days_in_month",
            "quarter",
            "day_of_year",
            "year",
        ]
    )
    datetime_categorical_features: List[str] = field(default_factory=list)


@dataclass
class TimeLagFeatureConfig:
    """特征滞后数配置。

    滞后特征由目标序列或内生变量的历史值生成，不再与日期时间外生特征混放。
    """
    # 特征滞后数列表；优先由具体配置根据 freq 显式覆盖，基类默认以 5min 为基准。
    enable_lags_features: bool = True
    lags: List[int] = field(
        default_factory=lambda: default_lags_for_freq("5min")
    )


@dataclass
class AdvancedFeatureConfig:
    """
    可选的内生统计特征、周期特征和交互特征配置。
    """
    # 高级特征总开关：以下子开关仅在此启用时才生效；统计/差分类特征依赖目标列 y，与 USMDP 不兼容
    enable_advanced_features: bool = False
    # 滚动窗口统计特征：在指定窗口上对指定列计算 mean/std/min/max/skew/kurt
    enable_rolling_features: bool = False
    rolling_columns: List[str] = field(default_factory=lambda: ["y"])
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 28])
    rolling_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "skew", "kurt"])
    # 扩展窗口统计特征：从序列起点累计计算 mean/std/min/max/skew/kurt
    enable_expanding_features: bool = False
    expanding_columns: List[str] = field(default_factory=lambda: ["y"])
    expanding_stats: List[str] = field(default_factory=lambda: ["mean", "std", "min", "max", "skew", "kurt"])
    # 差分特征：y_t 与 y_{t-period} 的差值
    enable_diff_features: bool = False
    diff_columns: List[str] = field(default_factory=lambda: ["y"])
    diff_periods: List[int] = field(default_factory=lambda: [1, 7, 24])
    # 百分比变化特征：相对前期的变化率
    enable_pct_change_features: bool = False
    pct_change_columns: List[str] = field(default_factory=lambda: ["y"])
    pct_change_periods: List[int] = field(default_factory=lambda: [1, 7])
    # 距事件时间步特征：统计距最近峰/谷等事件的步数
    enable_time_since_features: bool = False
    time_since_columns: List[str] = field(default_factory=lambda: ["y"])
    time_since_events: List[str] = field(default_factory=lambda: ["peak", "trough"])
    # 周期特征三角编码：对周期性列做 sin/cos 变换（如 minute）
    enable_cyclical_features: bool = False
    cyclical_columns: List[str] = field(default_factory=lambda: ["minute"])
    cyclical_period: int = 15
    # 列对交互特征：对指定列对做加减乘除运算生成新特征
    enable_interaction_features: bool = False
    interaction_column_pairs: List[tuple] = field(default_factory=lambda: [("y", "dt_hour")])
    interaction_operations: List[str] = field(default_factory=lambda: ["add", "subtract", "multiply", "divide"])
    # 多项式特征：对指定列做 n 次幂展开
    enable_polynomial_features: bool = False
    polynomial_columns: List[str] = field(default_factory=lambda: ["y"])
    polynomial_degree: int = 2


@dataclass
class PreprocessingConfig:
    """
    特征缩放、目标缩放和类别编码配置。
    """
    # 预测特征
    scale_features: bool = False  # 是否对预测特征 X 进行归一化/标准化
    feature_scaler_type: str = "minmax"  # 预测特征 X 的缩放方法: "standard" 或 "minmax"
    # 目标特征
    scale_target: bool = False  # 是否对目标变量 Y 进行归一化/标准化
    inverse_target: bool = False  # 预测结果是否对目标变量 Y 进行逆变换
    target_scaler_type: str = "minmax"  # 目标变量 Y 的缩放方法: "none"、"standard"、"minmax"、"log1p"、"robust" 或 "yeo-johnson"
    detrend_target: bool = False  # 是否在特征工程前对原始 y 做线性去趋势(与 scale_target 互斥)
    use_grouped_scaling: bool = False  # 是否对特征使用分组归一化/标准化
    encode_categorical_features: bool = False  # 是否对类别特征进行编码


@dataclass
class ModelStrategyConfig:
    """
    模型类型、融合模式和预测策略配置。
    """
    # 单模型预测
    model_type: str = "lightgbm"
    model_params: Dict = field(default_factory=dict)

    # 模型融合预测
    enable_ensemble: bool = False
    ensemble_models: List[str] = field(default_factory=lambda: ["lgb", "xgb", "cat"])
    ensemble_method: str = "stacking"  # "averaging"、"weighted"、"stacking" 或 "blending"
    ensemble_val_ratio: float = 0.2
    # 融合成员级规格（非空时取代 ensemble_models）：每项 {model, params?, scale?, impute?}
    # - model: 模型类型（ModelFactory 注册名/别名）
    # - params: 该成员独立的参数覆盖（在全局 model_params 之上合并）
    # - scale: 该成员独立特征标准化（线性成员需要，树成员不需要）
    # - impute: 该成员独立中位数填补（训练窗起始行长滞后特征为 NaN，
    #   GBDT 原生容忍，线性/KNN 等成员需要）
    ensemble_model_specs: List[Dict] = field(default_factory=list)

    # 可选预测方法详见 _PRED_METHODS。
    pred_method: str = "univariate-single-multistep-direct"
    multi_output_strategy: str = "multioutput"  # 多输出策略: multioutput / regressor_chain
    predict_type: str = "point"  # 预测类型: point / quantile
    quantiles: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])  # 分位数预测配置，predict_type=quantile 时生效
    quantile_monotone: bool = False  # 分位数单调化开关:逐行排序 predict_q* 消除 quantile crossing(默认关)
    use_horizon_exogenous_for_direct: bool = False  # Direct 方法是否使用 horizon-aware 外生特征展开
    block_size: int = 0  # Direct-Recursive 方法的分块大小

    # 全局训练（面板数据）：跨多条序列联合训练单模型，需配合 series_id_feature 区分序列
    enable_global_training: bool = False  # 全局训练模式，跨序列联合训练
    # 全局训练时标识不同序列的列名（默认 series_id）
    series_id_feature: str = "series_id"


@dataclass
class TrainingEnhancementConfig:
    """
    训练阶段的调参、增强、特征选择和损失函数配置。
    """

    patience: int = 100  # 早停步数

    # 时间衰减样本权重:近期样本权重更高,抑制概念漂移导致的远期噪声污染。
    enable_time_decay_sample_weight: bool = False
    decay_halflife_days: float = 14.0  # 半衰期(天):样本权重衰减到一半所需的样本年龄

    # 模型超参数调优
    perform_tuning: bool = False
    tuning_metric: str = "neg_mean_absolute_error"
    tuning_n_splits: int = 3

    # 数据增强，fit 训练集时生效。
    enable_data_augmentation: bool = False
    augmentation_ratio: float = 0.2
    augmentation_feature_noise_std: float = 0.01
    augmentation_target_noise_std: float = 0.005
    augmentation_random_state: int = 42

    # 特征选择：在训练集上拟合，并复用于测试和预测阶段。
    enable_feature_selection: bool = False
    feature_selection_method: str = "f_regression"  # f_regression / mutual_info
    feature_selection_max_features: int = 80
    feature_selection_min_features: int = 10

    # 学习率策略
    enable_auto_learning_rate: bool = False
    auto_lr_min: float = 0.005
    auto_lr_max: float = 0.2
    huber_delta: float = 1.0  # 鲁棒损失参数，用于 huber scorer


@dataclass
class TrainOutlierConfig:
    """
    滑窗测试阶段训练窗口异常处理配置。
    """
    # 仅在滑窗测试阶段的训练窗口内清洗目标异常值，不影响最终预测的推理阶段。
    enable_train_outlier_handling: bool = False  # 总开关
    train_outlier_method: str = "local_interpolate"  # 清洗方法，目前仅支持 local_interpolate（局部插值）
    high_outlier_threshold: Optional[float] = None  # 高值阈值：y 超过此值视为候选异常；None 表示不启用高值规则（默认关闭）
    high_outlier_max_run_points: int = 4  # 高值连续段长度 ≤ 此值才清洗（短尖峰）；更长的高值段视为真实
    rise_outlier_max_run_points: int = 2  # 骤升-回弹连续段长度上限
    rise_rebound_min_abs_diff: float = 900.0  # 骤升-回弹判定的最小绝对幅度
    low_outlier_threshold: Optional[float] = None  # 低值绝对阈值：y 低于此值视为候选异常；None 表示不启用低值规则（默认关闭）
    low_outlier_max_run_points: int = 4  # 低值连续段长度 ≤ 此值才清洗（短掉零）；更长段视为真实低负荷保留
    drop_outlier_max_run_points: int = 2  # 骤降-回弹连续段长度上限
    drop_rebound_min_abs_diff: float = 900.0  # 骤降-回弹判定的最小绝对幅度


@dataclass
class EvalMaskConfig:
    """
    评估/绘图阶段对低值异常点的掩码配置。

    MAPE / MAPE Accuracy 计算与预测图历史上下文掩码共用同一套阈值，
    避免「相对分位数」在不同数据分布上失效（见 utils/eval_mask.build_eval_mask）。
    """
    mode: str = "percentile"  # percentile | absolute | combined
    percentile: float = 5.0  # percentile/combined 模式下的下分位（%）
    min_value: Optional[float] = None  # 绝对下限：y < 此值视为异常；None 表示不启用绝对过滤（保持默认分位行为）
    max_value: Optional[float] = None  # 绝对上限：y > 此值视为异常（与 mode 正交，仅用于评估/绘图掩码，不清洗数据）


@dataclass
class PerformanceConfig:
    """
    并行度和过程日志配置。
    """
    # 并行度与过程日志：窗口/多输出/分位数/融合各有独立并行维度，按机器核数调整。
    window_parallel_workers: int = 1  # 滑窗测试的窗口级并行进程数
    max_test_windows: Optional[int] = None  # 测试窗口数量上限，None 表示不限
    test_window_stride: int = 1  # 测试窗口索引步长（>1 时跳采窗口）
    multi_output_n_jobs: int = 1  # direct 多输出模型的并行 job 数
    quantile_parallel_workers: int = 1  # 分位数预测的并行进程数
    ensemble_parallel_workers: int = 1  # 模型融合的并行进程数
    model_thread_count: int = 1  # 单模型内部线程数（映射到 LightGBM/XGBoost 等底层库）
    enable_step_logging: bool = False  # 是否打印逐步（特征工程/递归预测）详细日志
    forecast_log_interval: int = 24  # 递归预测时每 N 步打印一次进度


@dataclass
class OutputConfig:
    """
    模型、测试结果和预测结果的输出目录配置。
    """
    checkpoints_dir: str = "./results/pretrained_models/"
    test_results_dir: str = "./results/results_test/"
    pred_results_dir: str = "./results/results_forecast/"
    # 显式场景子路径；空字符串=由 data_dir 自动推导（去掉 dataset/demand_load 段）。
    # 多组配置共用同一 data_dir 时必须显式指定，否则结果会混入同一 scenario 目录。
    scenario_subpath: str = ""
    # 结果目录 setting 后缀（如 "-intraday"）；空=不加。用于同配置不同语义版本的结果隔离。
    setting_suffix: str = ""
    # 测试可视化叠加参考序列：在测试图（test_prediction.png / window_plots）上以次坐标轴叠加一条参考曲线。
    # plot_overlay_path 相对 data_dir 解析（也接受绝对路径）；plot_overlay_col 为叠加列名（文件需含 time + 该列）。
    # 两者任一为空即关闭叠加（默认）。量级与目标差异大时用次坐标轴绘制，不压扁主曲线。
    plot_overlay_path: str = ""
    plot_overlay_col: str = ""


@dataclass
class BaseModelConfig(
    RuntimeConfig,
    TargetSeriesConfig,
    ExogenousFeatureConfig,
    TimeLagFeatureConfig,
    AdvancedFeatureConfig,
    PreprocessingConfig,
    ModelStrategyConfig,
    TrainingEnhancementConfig,
    TrainOutlierConfig,
    EvalMaskConfig,
    PerformanceConfig,
    OutputConfig):
    """
    供公开 `ModelConfig` 继承的扁平组合配置基类。
    """
