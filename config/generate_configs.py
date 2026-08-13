#!/usr/bin/env python3
"""
通用配置文件生成工具。

根据 dataset 目录中的数据自动检测特征，生成对应的分组 YAML 配置文件。

用法:
    # 最简用法（自动检测 freq、now_time、外生文件等）
    uv run python config/generate_configs.py \
        --dataset ./dataset/aidc_electricity_computility/electricity/2026-01-01/demand_load/lingang_A \
        --target h_total_use

    # 完整参数
    uv run python config/generate_configs.py \
        --dataset ./dataset/aidc_electricity_computility/electricity/2026-01-01/demand_load/lingang_A \
        --target h_total_use \
        --target-ts-feat count_data_time \
        --models lightgbm,xgboost,catboost \
        --strategies usmdp,usmd,usmr,usmdr \
        --now-time 2026-01-01T00:00:00 \
        --freq 5min \
        --variant A \
        --config-dir config/aidc_electricity_computility/electricity/2026-01-01/route_A

    # 预览模式（不实际创建文件）
    uv run python config/generate_configs.py \
        --dataset ./dataset/ETT-small/ \
        --data-file ETTm1.csv \
        --target OT \
        --dry-run
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from config.config_sections import default_lags_for_freq
from utils.frequency import resolve_samples_per_day

# 项目根路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 策略缩写映射
# ---------------------------------------------------------------------------

METHOD_SHORT = {
    "usmdp": "univariate-single-multistep-direct-pointwise",
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

YAML_OVERRIDE_GROUPS = {
    "runtime": [
        "is_testing",
        "is_forecasting",
        "history_days",
        "predict_steps",
        "window_days",
        "now_time",
    ],
    "target_series": [
        "data_dir",
        "data_path",
        "freq",
        "target_ts_feat",
        "target",
        "target_series_numeric_features",
        "target_series_categorical_features",
        "target_series_drop_features",
    ],
    "exogenous_features": [
        "enable_date_features",
        "date_history_path",
        "date_future_path",
        "date_ts_feat",
        "datetype_features",
        "datetype_categorical_features",
        "enable_weather_features",
        "weather_history_path",
        "weather_future_path",
        "weather_ts_feat",
        "weather_features",
        "weather_categorical_features",
        "enable_datetime_features",
        "datetime_features",
        "datetime_categorical_features",
    ],
    "time_lag_features": [
        "enable_lags_features",
        "lags",
    ],
    "advanced_features": [
        "enable_advanced_features",
        "enable_rolling_features",
        "rolling_columns",
        "rolling_windows",
        "rolling_stats",
        "enable_expanding_features",
        "expanding_columns",
        "expanding_stats",
        "enable_diff_features",
        "diff_columns",
        "diff_periods",
        "enable_pct_change_features",
        "pct_change_columns",
        "pct_change_periods",
        "enable_time_since_features",
        "time_since_columns",
        "time_since_events",
        "enable_cyclical_features",
        "cyclical_columns",
        "cyclical_period",
        "enable_interaction_features",
        "interaction_column_pairs",
        "interaction_operations",
        "enable_polynomial_features",
        "polynomial_columns",
        "polynomial_degree",
    ],
    "preprocessing": [
        "scale_features",
        "feature_scaler_type",
        "scale_target",
        "inverse_target",
        "target_scaler_type",
        "use_grouped_scaling",
        "encode_categorical_features",
    ],
    "model_strategy": [
        "model_type",
        "model_params",
        "enable_ensemble",
        "ensemble_models",
        "ensemble_method",
        "ensemble_val_ratio",
        "ensemble_model_specs",
        "pred_method",
        "multi_output_strategy",
        "predict_type",
        "quantiles",
        "use_horizon_exogenous_for_direct",
        "block_size",
        "enable_global_training",
        "series_id_feature",
    ],
    "training_enhancement": [
        "patience",
        "enable_time_decay_sample_weight",
        "decay_halflife_days",
        "perform_tuning",
        "tuning_metric",
        "tuning_n_splits",
        "enable_data_augmentation",
        "augmentation_ratio",
        "augmentation_feature_noise_std",
        "augmentation_target_noise_std",
        "augmentation_random_state",
        "enable_feature_selection",
        "feature_selection_method",
        "feature_selection_max_features",
        "feature_selection_min_features",
        "enable_auto_learning_rate",
        "auto_lr_min",
        "auto_lr_max",
        "huber_delta",
    ],
    "train_outlier": [
        "enable_train_outlier_handling",
        "train_outlier_method",
        "high_outlier_threshold",
        "high_outlier_max_run_points",
        "drop_outlier_max_run_points",
        "drop_rebound_min_abs_diff",
        "low_outlier_threshold",
        "low_outlier_max_run_points",
        "rise_outlier_max_run_points",
        "rise_rebound_min_abs_diff",
    ],
    "eval_mask": [
        "mode",
        "percentile",
        "min_value",
        "max_value",
    ],
    "performance": [
        "window_parallel_workers",
        "max_test_windows",
        "test_window_stride",
        "multi_output_n_jobs",
        "quantile_parallel_workers",
        "ensemble_parallel_workers",
        "model_thread_count",
        "enable_step_logging",
        "forecast_log_interval",
    ],
    "output": [
        "checkpoints_dir",
        "test_results_dir",
        "pred_results_dir",
    ],
}


# ---------------------------------------------------------------------------
# 配置生成
# ---------------------------------------------------------------------------
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
    result["column_names"] = column_names
    result["sample_dtypes"] = {col: str(dtype) for col, dtype in sample.dtypes.items()}

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
           --target y --models catboost --strategies usmd,usmdp --variant A
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
    parser.add_argument("--strategies", type=str, default="usmdp,usmd,usmr,usmdr",
                        help="策略列表, 逗号分隔 (默认: usmdp,usmd,usmr,usmdr)")

    # ---- 输出相关 ----
    parser.add_argument("--config-dir", type=str, default=None,
                        help="配置文件输出目录 (默认从 --dataset 路径推导)")
    parser.add_argument("--variant", type=str, default=None,
                        help="文件名后缀, 如 A / B (默认无)")
    parser.add_argument("--base-config", type=str, default=None,
                        help="YAML base_config。默认按策略和内生特征自动选择标准配置")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印将生成的文件, 不实际创建")

    # ---- 窗口参数 ----
    parser.add_argument("--history-days", type=int, default=31,
                        help="历史数据天数 (默认: 31)")
    parser.add_argument("--predict-steps", type=int, default=None,
                        help="预测步数, 以 freq 为单位 (默认: 1 天对应的步数)")
    parser.add_argument("--window-days", type=int, default=15,
                        help="滑动窗口天数 (默认: 15)")

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def default_base_config(strategy: str, target_series_numeric_features: Optional[List[str]] = None) -> str:
    """按数据集形态和预测策略选择标准配置入口。"""
    if strategy.startswith("ms") or target_series_numeric_features:
        return "config.multivariate_config"
    return "config.univariate_config"


def _to_yaml_value(value: Any) -> Any:
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    if isinstance(value, tuple):
        return [_to_yaml_value(item) for item in value]
    if isinstance(value, list):
        return [_to_yaml_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_yaml_value(item) for key, item in value.items()}
    return value


def _build_generation_overrides(params: dict) -> dict:
    """只写生成场景显式覆盖的字段，避免 YAML 重复完整基类默认值。"""
    return {
        "history_days": params["history_days"],
        "predict_steps": params["predict_steps"],
        "window_days": params["window_days"],
        "now_time": params["now_time_iso"],
        "data_dir": params["data_dir"],
        "data_path": params["data_path"],
        "freq": params["freq"],
        "target_ts_feat": params["target_ts_feat"],
        "target": params["target"],
        "target_series_numeric_features": params["target_series_numeric_features"],
        "target_series_categorical_features": params["target_series_categorical_features"],
        "target_series_drop_features": params["target_series_drop_features"],
        "enable_date_features": params["enable_date_features"],
        "date_history_path": "df_date.csv" if params["enable_date_features"] else None,
        "date_future_path": "df_date_future.csv" if params["enable_date_features"] else None,
        "date_ts_feat": "date" if params["enable_date_features"] else None,
        "datetype_features": ["date_type"] if params["enable_date_features"] else [],
        "datetype_categorical_features": [],
        "enable_weather_features": params["enable_weather_features"],
        "weather_history_path": "df_weather.csv" if params["enable_weather_features"] else None,
        "weather_future_path": "df_weather_future.csv" if params["enable_weather_features"] else None,
        "weather_ts_feat": "ts" if params["enable_weather_features"] else None,
        "weather_features": [
            "rt_ssr",
            "rt_ws10",
            "rt_tt2",
            "cal_rh",
            "rt_ps",
            "rt_rain",
        ] if params["enable_weather_features"] else [],
        "weather_categorical_features": [],
        "enable_datetime_features": params["enable_datetime_features"],
        "enable_lags_features": params["enable_lags_features"],
        "lags": (
            default_lags_for_freq(params["freq"])
            if params["enable_lags_features"]
            else []
        ),
        "model_type": params["model_type"],
        "pred_method": params["pred_method"],
    }


def build_yaml_config(params: dict) -> dict:
    config_values = _build_generation_overrides(params)
    overrides = {}
    used_fields = set()
    for group_name, field_names in YAML_OVERRIDE_GROUPS.items():
        group_values = {}
        for field_name in field_names:
            if field_name in config_values:
                group_values[field_name] = config_values[field_name]
                used_fields.add(field_name)
        if group_values:
            overrides[group_name] = group_values

    return {
        "base_config": params["base_config"],
        "overrides": overrides,
    }


def generate_yaml_config(params: dict, output_path: Path, dry_run: bool = False) -> None:
    """生成分组 YAML 配置文件。"""
    content = yaml.safe_dump(build_yaml_config(params), sort_keys=False, allow_unicode=True)
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
    sample_dtypes = detected["sample_dtypes"]
    column_names = detected["column_names"]
    target_series_numeric_features = [
        col for col in column_names
        if col not in {target_ts_feat, args.target} and sample_dtypes.get(col, "").startswith(("int", "float"))
    ]
    target_series_categorical_features = [
        col for col in column_names
        if col not in {target_ts_feat, args.target} and col not in target_series_numeric_features
    ]

    # now_time 处理
    if args.now_time:
        now_time_val = pd.Timestamp(args.now_time)
    else:
        now_time_val = detected.get("now_time_raw", pd.Timestamp("2025-01-01"))
    now_time_iso = now_time_val.to_pydatetime().isoformat()

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
    count = 0

    for model in models:
        model_short = MODEL_SHORT[model]
        model_full = MODEL_FULL.get(model_short, model)
        for strategy in strategies:
            method_full = METHOD_SHORT[strategy]

            # 构建参数字典
            params = {
                "history_days": args.history_days,
                "predict_steps": args.predict_steps if args.predict_steps is not None else resolve_samples_per_day(freq),
                "window_days": args.window_days,
                "now_time_iso": now_time_iso,
                "data_dir": data_dir_rel,
                "data_path": data_path,
                "freq": freq,
                "target_ts_feat": target_ts_feat,
                "target": args.target,
                "target_series_numeric_features": target_series_numeric_features,
                "target_series_categorical_features": target_series_categorical_features,
                "target_series_drop_features": [],
                "enable_date_features": has_date,
                "enable_weather_features": has_weather,
                "enable_datetime_features": enable_datetime,
                "enable_lags_features": enable_lags,
                "model_type": model_full,
                "pred_method": method_full,
                "base_config": args.base_config or default_base_config(strategy, target_series_numeric_features),
            }

            # 构建文件名
            filename = f"{model_short}_{strategy}.yaml"

            output_path = config_base / filename
            generate_yaml_config(params, output_path, dry_run=args.dry_run)
            count += 1

    print("-" * 50)
    if args.dry_run:
        print(f"预览完成, 共 {count} 个文件 (未实际创建)")
    else:
        print(f"生成完成, 共 {count} 个配置文件")


if __name__ == "__main__":
    main()
