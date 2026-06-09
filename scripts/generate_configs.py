#!/usr/bin/env python3
"""
通用配置文件生成工具。

根据 dataset 目录中的数据自动检测特征，生成对应的模型配置文件。

用法:
    # 最简用法（自动检测 freq、now_time、外生文件等）
    uv run python scripts/generate_configs.py \
        --dataset ./dataset/aidc_electricity_computility/electricity/2026-01-01/demand_load/lingang_A \
        --target h_total_use

    # 完整参数
    uv run python scripts/generate_configs.py \
        --dataset ./dataset/aidc_electricity_computility/electricity/2026-01-01/demand_load/lingang_A \
        --target h_total_use \
        --target-ts-feat count_data_time \
        --models lightgbm,xgboost,catboost \
        --strategies usmdo,usmd,usmr,usmdr \
        --now-time 2026-01-01T00:00:00 \
        --freq 5min \
        --variant A \
        --config-dir config/aidc_electricity_computility/electricity/2026-01-01/route_A

    # 预览模式（不实际创建文件）
    uv run python scripts/generate_configs.py \
        --dataset ./dataset/ETT-small/ \
        --data-file ETTm1.csv \
        --target OT \
        --dry-run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# 项目根路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 策略缩写映射
# ---------------------------------------------------------------------------

METHOD_SHORT = {
    "usmdo": "univariate-single-multistep-direct-output",
    "usmd": "univariate-single-multistep-direct",
    "usmr": "univariate-single-multistep-recursive",
    "usmdr": "univariate-single-multistep-direct-recursive",
    "msmd": "multivariate-single-multistep-direct",
    "msmr": "multivariate-single-multistep-recursive",
    "msmdr": "multivariate-single-multistep-direct-recursive",
}

MODEL_SHORT = {
    "lightgbm": "lgbm",
    "lgbm": "lgbm",
    "xgboost": "xgb",
    "xgb": "xgb",
    "catboost": "cab",
    "cab": "cab",
}

MODEL_FULL = {
    "lgbm": "lightgbm",
    "xgb": "xgboost",
    "cab": "catboost",
}

EXOGENOUS_FILE_PATTERNS = [
    "df_date.csv",
    "df_date_future.csv",
    "df_weather.csv",
    "df_weather_future.csv",
]


# ---------------------------------------------------------------------------
# 配置模板
# ---------------------------------------------------------------------------

_BASELINE_CONFIG = '''# -*- coding: utf-8 -*-
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
    history_days: int = {history_days}  # 历史数据天数
    predict_days: int = {predict_days}  # 预测未来天数
    window_days: int = {window_days}  # 滑动窗口天数
    # 预测推理开始的时间
    now_time: datetime.datetime = {now_time}
    # ------------------------------
    # 目标时间序列配置
    # ------------------------------
    data_dir: str = "{data_dir}"
    data_path: str = "{data_path}"
    freq: str = "{freq}"
    target_ts_feat: str = "{target_ts_feat}"
    target: str = "{target}"
    target_series_numeric_features: List[str] = field(default_factory=lambda: {target_series_numeric_features})
    target_series_categorical_features: List[str] = field(default_factory=lambda: {target_series_categorical_features})
    target_series_drop_features: List[str] = field(default_factory=list)
    # ------------------------------
    # 特征工程配置
    # ------------------------------
    # 日期类型数据配置
    # --------------
    enable_date_features: bool = {enable_date_features}
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
    enable_weather_features: bool = {enable_weather_features}
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
    enable_datetime_features: bool = {enable_datetime_features}
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
    enable_lags_features: bool = {enable_lags_features}
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
    model_type: str = "{model_type}"
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
    pred_method: str = "{pred_method}"
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
'''


# ---------------------------------------------------------------------------
# 自动检测
# ---------------------------------------------------------------------------

def detect_data_params(dataset_dir: Path, data_file: Optional[str] = None) -> dict:
    """
    扫描数据目录，自动检测数据特征。

    Returns:
        dict 包含自动检测到的参数键值。
    """
    result = {}

    # --- 检测数据文件 ---
    if data_file:
        data_path = dataset_dir / data_file
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
    else:
        csv_files = sorted([
            f for f in os.listdir(dataset_dir)
            if f.endswith(".csv") and f not in EXOGENOUS_FILE_PATTERNS
        ])
        if not csv_files:
            raise FileNotFoundError(f"在 {dataset_dir} 中未找到 CSV 数据文件")
        data_file = csv_files[0]

    result["data_path"] = data_file

    # --- 读取 CSV ---
    csv_path = dataset_dir / data_file
    # 先读少量行探测格式
    sample = pd.read_csv(csv_path, nrows=5)
    column_names = list(sample.columns)

    if len(column_names) < 2:
        raise ValueError(f"CSV 列数不足 ({len(column_names)}), 至少需要时间列 + 目标列")

    # --- 检测 target_ts_feat (第一列) ---
    result["target_ts_feat"] = column_names[0]

    # --- 检测 freq ---
    time_col = column_names[0]
    full_df = pd.read_csv(csv_path, usecols=[time_col])
    full_df[time_col] = pd.to_datetime(full_df[time_col])
    if len(full_df) >= 2:
        inferred = pd.infer_freq(full_df[time_col].head(100))
        if inferred:
            result["freq"] = inferred
        else:
            # 回退：手动计算时间差
            diff_seconds = (full_df[time_col].iloc[1] - full_df[time_col].iloc[0]).total_seconds()
            if diff_seconds == 60:
                result["freq"] = "1min"
            elif diff_seconds == 300:
                result["freq"] = "5min"
            elif diff_seconds == 900:
                result["freq"] = "15min"
            elif diff_seconds == 1800:
                result["freq"] = "30min"
            elif diff_seconds == 3600:
                result["freq"] = "1h"
            elif diff_seconds == 86400:
                result["freq"] = "1D"
            else:
                result["freq"] = f"{int(diff_seconds / 60)}min"

    # --- 检测 now_time (最后一行时间戳) ---
    last_ts = full_df[time_col].iloc[-1]
    result["now_time_raw"] = last_ts

    # --- 检测外生文件 ---
    files_in_dir = set(os.listdir(dataset_dir))
    result["has_date_exog"] = (
        "df_date.csv" in files_in_dir and "df_date_future.csv" in files_in_dir
    )
    result["has_weather_exog"] = (
        "df_weather.csv" in files_in_dir and "df_weather_future.csv" in files_in_dir
    )

    return result


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="通用配置文件生成工具 — 根据 dataset 数据自动生成 config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 最简用法（自动检测 freq、now_time 等）
  %(prog)s --dataset ./dataset/aidc_electricity_computility/electricity/2026-01-01/demand_load/lingang_A \\
           --target h_total_use

  # 预览模式
  %(prog)s --dataset ./dataset/ETT-small/ --data-file ETTm1.csv --target OT --dry-run

  # 指定模型和策略
  %(prog)s --dataset ./dataset/aidc_electricity_computility/gaoweichao_compare/ \\
           --target y --models catboost --strategies usmd,usmdo --variant A
        """,
    )

    # ---- 必需参数 ----
    parser.add_argument(
        "--dataset", required=True, type=str,
        help="数据目录路径",
    )
    parser.add_argument(
        "--target", required=True, type=str,
        help="目标列名",
    )

    # ---- 数据相关 ----
    parser.add_argument("--data-file", type=str, default=None,
                        help="数据文件名 (默认自动检测目录下第一个非外生 CSV)")
    parser.add_argument("--target-ts-feat", type=str, default=None,
                        help="时间列名 (默认 CSV 第一列)")
    parser.add_argument("--freq", type=str, default=None,
                        help="时间频率, 如 5min/15min/1h/1D (默认自动推断)")
    parser.add_argument("--now-time", type=str, default=None,
                        help="预测基准时间 ISO 格式, 如 2026-01-01T00:00:00 (默认数据最后时间戳)")

    # ---- 特征开关 ----
    parser.add_argument("--no-date", action="store_true",
                        help="禁用日期外生特征 (默认自动检测目录下是否存在 df_date.csv)")
    parser.add_argument("--no-weather", action="store_true",
                        help="禁用气象外生特征 (默认自动检测目录下是否存在 df_weather.csv)")
    parser.add_argument("--no-datetime", action="store_true",
                        help="禁用日期时间特征 (默认开启)")
    parser.add_argument("--no-lags", action="store_true",
                        help="禁用滞后特征 (默认开启)")

    # ---- 模型与策略 ----
    parser.add_argument("--models", type=str, default="lightgbm,xgboost,catboost",
                        help="模型列表, 逗号分隔 (默认: lightgbm,xgboost,catboost)")
    parser.add_argument("--strategies", type=str, default="usmdo,usmd,usmr,usmdr",
                        help="策略列表, 逗号分隔 (默认: usmdo,usmd,usmr,usmdr)")

    # ---- 输出相关 ----
    parser.add_argument("--config-dir", type=str, default=None,
                        help="配置文件输出目录 (默认从 --dataset 路径推导)")
    parser.add_argument("--variant", type=str, default=None,
                        help="文件名后缀, 如 A / B (默认无)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将生成的文件, 不实际创建")

    # ---- 窗口参数 ----
    parser.add_argument("--history-days", type=int, default=31,
                        help="历史数据天数 (默认: 31)")
    parser.add_argument("--predict-days", type=int, default=1,
                        help="预测天数 (默认: 1)")
    parser.add_argument("--window-days", type=int, default=15,
                        help="滑动窗口天数 (默认: 15)")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def generate_config(params: dict, output_path: Path, dry_run: bool = False) -> None:
    """生成单个配置文件。"""
    content = _BASELINE_CONFIG.format(**params)
    if dry_run:
        print(f"  [DRY-RUN] Would create: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        print(f"  Created: {output_path}")


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    # ---- 解析数据目录 ----
    dataset_dir = Path(args.dataset).resolve()
    if not dataset_dir.is_dir():
        print(f"错误: 目录不存在 — {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # ---- 自动检测 ----
    print(f"数据集目录: {dataset_dir}")
    detected = detect_data_params(dataset_dir, data_file=args.data_file)
    data_path = detected["data_path"]
    target_ts_feat = args.target_ts_feat or detected["target_ts_feat"]
    freq = args.freq or detected.get("freq", "5min")
    has_date = args.no_date is False and detected["has_date_exog"]
    has_weather = args.no_weather is False and detected["has_weather_exog"]
    enable_datetime = not args.no_datetime
    enable_lags = not args.no_lags

    # now_time 处理
    if args.now_time:
        now_time_val = pd.Timestamp(args.now_time)
    else:
        now_time_val = detected.get("now_time_raw", pd.Timestamp("2025-01-01"))
    now_time_str = f"datetime.datetime({now_time_val.year}, {now_time_val.month}, {now_time_val.day}, {now_time_val.hour}, {now_time_val.minute}, {now_time_val.second})"

    # ---- 解析模型和策略 ----
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    # 校验
    for m in models:
        if m not in MODEL_SHORT:
            print(f"错误: 不支持的模型 '{m}'，支持: {list(MODEL_SHORT.keys())}", file=sys.stderr)
            sys.exit(1)
    for s in strategies:
        if s not in METHOD_SHORT:
            print(f"错误: 不支持策略 '{s}'，支持: {list(METHOD_SHORT.keys())}", file=sys.stderr)
            sys.exit(1)

    # ---- 输出目录 ----
    if args.config_dir:
        config_base = Path(args.config_dir)
    else:
        # 从 dataset 路径推导: dataset/xxx → config/xxx
        ds_str = str(dataset_dir)
        # 规范化 dataset/ 前缀
        ds_rel = ds_str.replace(str(ROOT) + "/", "").replace("dataset/", "", 1)
        config_base = ROOT / "config" / ds_rel

    # ---- 数据目录路径（写入 config 的相对路径）----
    data_dir_rel = "./" + str(dataset_dir.relative_to(ROOT)) + "/"

    # ---- 打印检测结果 ----
    print(f"数据文件:     {data_path}")
    print(f"时间列:       {target_ts_feat}")
    print(f"目标列:       {args.target}")
    print(f"频率:         {freq}")
    print(f"now_time:     {now_time_val.isoformat()}")
    print(f"日期外生:     {'✅' if has_date else '❌'}")
    print(f"气象外生:     {'✅' if has_weather else '❌'}")
    print(f"日期时间特征: {'✅' if enable_datetime else '❌'}")
    print(f"滞后特征:     {'✅' if enable_lags else '❌'}")
    print(f"模型:         {', '.join(models)}")
    print(f"策略:         {', '.join(strategies)}")
    print(f"输出目录:     {config_base}")
    print(f"模式:         {'DRY-RUN (预览)' if args.dry_run else '实际生成'}")
    print("-" * 50)

    # ---- 生成配置文件 ----
    variant = args.variant
    count = 0

    for model in models:
        model_short = MODEL_SHORT[model]
        model_full = MODEL_FULL.get(model_short, model)
        for strategy in strategies:
            method_full = METHOD_SHORT[strategy]

            # 构建参数字典
            params = {
                "history_days": args.history_days,
                "predict_days": args.predict_days,
                "window_days": args.window_days,
                "now_time": now_time_str,
                "data_dir": data_dir_rel,
                "data_path": data_path,
                "freq": freq,
                "target_ts_feat": target_ts_feat,
                "target": args.target,
                "target_series_numeric_features": "[]",
                "target_series_categorical_features": "[]",
                "enable_date_features": has_date,
                "enable_weather_features": has_weather,
                "enable_datetime_features": enable_datetime,
                "enable_lags_features": enable_lags,
                "model_type": model_full,
                "pred_method": method_full,
            }

            # 构建文件名
            if variant:
                filename = f"model_config_{model_short}_{strategy}_{variant}.py"
            else:
                filename = f"model_config_{model_short}_{strategy}.py"

            output_path = config_base / filename
            generate_config(params, output_path, dry_run=args.dry_run)
            count += 1

    print("-" * 50)
    if args.dry_run:
        print(f"预览完成, 共 {count} 个文件 (未实际创建)")
    else:
        print(f"生成完成, 共 {count} 个配置文件")


if __name__ == "__main__":
    main()
