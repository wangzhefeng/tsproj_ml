#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tsproj_ml 模型配置 dry-run 校验：只加载配置 + 复算 main.py 合法性校验，不训练不预测。

用法（项目根，注意 env -u PYTHONPATH 防 Hermes 注入遮蔽项目 utils/ 包）：
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/<scenario>/route_*/lgbm_*.yaml'
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/aidc_power_15min/**/*.yaml'
    （无参数时默认 glob config/**/*.yaml，会命中聚合/异常配置，按文件名前缀跳过即可）

复算的校验（与 main.py.__init__ 一致）：
  - window_length < history_length
  - window_len - horizon > max(lags)（滞后特征非全 NaN；USMDP 仅在未启用 safe-lag 时除外）
  - USMDP 多步 safe-lag：align_direct_features_to_target=true 时必须启用 lag，且 min(lags) >= horizon
  - 目标分解方法/周期合法，且与 scale_target 互斥
  - n_windows > 0（滑窗数）
  - advanced_features：USMDP 不能直接依赖 y；仅操作历史/未来都存在的列才可用
    （USMDP 仅在 align_direct_features_to_target=true 时生成 y_lag_*）；预测上下文取
    max(lags, 已启用 rolling_windows/diff_periods/pct_change_periods)，且不得超过
    history_length × n_per_day

退出码：0 = 全部通过（可能有提示性警告），1 = 存在硬校验失败。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402
import pandas as pd  # noqa: E402

from config.config_loader import load_yaml_config  # noqa: E402
from models.ModelTesting import Tester  # noqa: E402
from utils.frequency import resolve_samples_per_day  # noqa: E402
from utils.multistep_contract import validate_direct_feature_alignment  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent

# 聚合/异常配置：独立 schema，无 base_config，不能被 load_yaml_config 加载
_NON_MODEL_PREFIXES = ("aggregate", "outlier_detect", "outlier")


def _resolve_runtime_shape(cfg) -> tuple[str, int, int, int, int]:
    """返回 horizon_mode、最终 horizon、训练行数、窗口数、有效训练天数。"""
    horizon_mode = str(getattr(cfg, "horizon_mode", "fixed_steps") or "fixed_steps").lower()
    n_per_day = resolve_samples_per_day(cfg.freq)
    if horizon_mode == "calendar_month":
        if str(cfg.freq) != "1D":
            raise ValueError("horizon_mode=calendar_month currently requires freq=1D")
        train_window_length = getattr(cfg, "train_window_length", None)
        if train_window_length is None or int(train_window_length) <= 0:
            raise ValueError("calendar_month requires train_window_length > 0")
        now_time = cast(pd.Timestamp, pd.Timestamp(cfg.now_time))
        forecast_start = now_time.normalize() + pd.Timedelta(days=1)
        if forecast_start.day != 1:
            raise ValueError("calendar_month requires forecast_start at month start")
        horizon = int(forecast_start.days_in_month)
        train_rows = int(train_window_length) * n_per_day
        history_start = forecast_start - pd.Timedelta(days=int(cfg.history_length))
        theoretical_history = pd.DataFrame(
            {"time": pd.date_range(history_start, forecast_start, freq="1D", inclusive="left")}
        )
        n_windows = len(
            Tester._build_calendar_month_folds(
                theoretical_history,
                train_window_len=train_rows,
            )
        )
        return horizon_mode, horizon, train_rows, n_windows, int(train_window_length)
    if horizon_mode != "fixed_steps":
        raise ValueError(f"unknown horizon_mode={horizon_mode}")
    window_len = int(cfg.window_length) * n_per_day
    horizon = int(cfg.predict_steps)
    train_rows = window_len - horizon
    n_windows = (int(cfg.history_length) * n_per_day - window_len) // horizon + 1
    return horizon_mode, horizon, train_rows, n_windows, int(cfg.window_length)


def check_model_yaml(f: str) -> tuple[Any, list[str]]:
    """加载单个模型 YAML，返回 (cfg, problems)。problems 空 = 通过。"""
    cfg: Any = load_yaml_config(f)
    n_per_day = resolve_samples_per_day(cfg.freq)
    max_lag = max(cfg.lags or [0])
    problems: list[str] = []
    try:
        horizon_mode, horizon, train_rows, n_windows, effective_train_length = _resolve_runtime_shape(cfg)
    except ValueError as exc:
        horizon_mode = str(getattr(cfg, "horizon_mode", "fixed_steps"))
        horizon = int(cfg.predict_steps)
        train_rows = int(cfg.window_length) * n_per_day - horizon
        n_windows = 0
        effective_train_length = int(cfg.window_length)
        problems.append(str(exc))

    if effective_train_length >= cfg.history_length:
        problems.append(
            f"effective_train_length({effective_train_length}) >= history_length({cfg.history_length})"
        )

    try:
        validate_direct_feature_alignment(cfg, horizon)
    except ValueError as exc:
        problems.append(str(exc))

    constructs_lags = (
        cfg.pred_method != "univariate-single-multistep-direct-pointwise"
        or bool(getattr(cfg, "align_direct_features_to_target", False))
    )
    if constructs_lags:
        if train_rows <= max_lag:
            problems.append(
                f"lag 校验失败: train_rows={train_rows} <= max_lag={max_lag} "
                "(Lag features would be all-NaN)"
            )

    # 通过 resolve_decomposition_spec 统一校验（覆盖新写法+旧写法+custom+预留方法）
    from decomposition.spec import resolve_decomposition_spec

    try:
        spec = resolve_decomposition_spec(cfg)
        method = spec.method
        if spec.preset is not None:
            periods = list(spec.preset.periods)
        else:
            periods = []
    except ValueError as exc:
        problems.append(f"decomposition 配置错误: {exc}")
        method = "none"
        periods = []
    decomposition_enabled = method != "none"
    if decomposition_enabled and getattr(cfg, "scale_target", False):
        problems.append("目标分解与 scale_target 同时开启（互斥）")
    if method == "stl" and len(periods) != 1:
        problems.append("stl 需要且只能配置一个周期")
    if method == "mstl" and len(periods) < 2:
        problems.append("mstl 至少需要两个周期")
    if method in {"stl", "mstl"}:
        if any(period < 2 for period in periods):
            problems.append("decomposition_periods 必须全部 >= 2")
        too_long = [period for period in periods if 2 * period > train_rows]
        if too_long:
            problems.append(
                f"decomposition_periods={too_long} 在最短训练窗 {train_rows} 点内不足两个完整周期"
            )

    if n_windows <= 0:
        problems.append(f"n_windows={n_windows} <= 0，测试会被跳过")

    adv = getattr(cfg, "enable_advanced_features", False)
    if adv:
        is_usmdp = cfg.pred_method == "univariate-single-multistep-direct-pointwise"
        target_dependent_features = {
            "rolling": bool(getattr(cfg, "enable_rolling_features", False))
            and "y" in (getattr(cfg, "rolling_columns", []) or []),
            "expanding": bool(getattr(cfg, "enable_expanding_features", False))
            and "y" in (getattr(cfg, "expanding_columns", []) or []),
            "diff": bool(getattr(cfg, "enable_diff_features", False))
            and "y" in (getattr(cfg, "diff_columns", []) or []),
            "pct_change": bool(getattr(cfg, "enable_pct_change_features", False))
            and "y" in (getattr(cfg, "pct_change_columns", []) or []),
            "time_since": bool(getattr(cfg, "enable_time_since_features", False))
            and "y" in (getattr(cfg, "time_since_columns", []) or []),
            "polynomial": bool(getattr(cfg, "enable_polynomial_features", False))
            and "y" in (getattr(cfg, "polynomial_columns", []) or []),
            "interaction": bool(getattr(cfg, "enable_interaction_features", False))
            and any("y" in pair for pair in (getattr(cfg, "interaction_column_pairs", []) or [])),
        }
        enabled_target_ops = [name for name, enabled in target_dependent_features.items() if enabled]
        if is_usmdp and enabled_target_ops:
            problems.append(
                "USMDP advanced_features 不能依赖目标列 y（预测 df_future 无 y）："
                + ", ".join(enabled_target_ops)
            )
        roll_columns = list(getattr(cfg, "rolling_columns", []) or [])
        if (
            is_usmdp
            and not bool(getattr(cfg, "align_direct_features_to_target", False))
            and bool(getattr(cfg, "enable_rolling_features", False))
            and any(column.startswith("y_lag_") for column in roll_columns)
        ):
            problems.append(
                "提示：USMDP 仅在 align_direct_features_to_target=true 时生成 y_lag_*；"
                "当前 rolling_columns 中的 y_lag_* 会被跳过。"
            )
        fixed_lookbacks = [max_lag]
        if getattr(cfg, "enable_rolling_features", False):
            fixed_lookbacks.extend(getattr(cfg, "rolling_windows", []) or [])
        if getattr(cfg, "enable_diff_features", False):
            fixed_lookbacks.extend(getattr(cfg, "diff_periods", []) or [])
        if getattr(cfg, "enable_pct_change_features", False):
            fixed_lookbacks.extend(getattr(cfg, "pct_change_periods", []) or [])
        required_context = max(fixed_lookbacks)
        available_history = cfg.history_length * n_per_day
        if required_context > available_history:
            problems.append(
                f"advanced context {required_context} exceeds available history "
                f"{available_history} (= history_length × n_per_day)"
            )
    cfg._resolved_horizon_mode = horizon_mode
    cfg._resolved_horizon = horizon
    cfg._resolved_train_rows = train_rows
    cfg._resolved_n_windows = n_windows
    return cfg, problems


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "config/**/*.yaml"
    files = sorted(
        f for f in glob.glob(pattern, recursive=True)
        if Path(f).name not in _NON_MODEL_PREFIXES and not Path(f).name.startswith(_NON_MODEL_PREFIXES)
    )
    if not files:
        print(f"no files matched: {pattern}")
        return 1

    hard_failures: list[tuple[str, list[str]]] = []
    warnings: list[tuple[str, list[str]]] = []
    for f in files:
        cfg, problems = check_model_yaml(f)
        method = cfg.pred_method
        n_per_day = resolve_samples_per_day(cfg.freq)
        horizon_mode = cfg._resolved_horizon_mode
        horizon = cfg._resolved_horizon
        train_rows = cfg._resolved_train_rows
        max_lag = max(cfg.lags or [0])
        n_windows = cfg._resolved_n_windows
        adv = getattr(cfg, "enable_advanced_features", False)
        print(f"{f}")
        print(f"    method={method} freq={cfg.freq} horizon_mode={horizon_mode} n_per_day={n_per_day} "
              f"history={cfg.history_length}d train_rows={train_rows} horizon={horizon} "
              f"max_lag={max_lag} n_windows={n_windows} adv={adv} now_time={cfg.now_time}")
        if problems:
            # 硬校验 vs 提示：高级窗口超限和 USMDP 未生成 y_lag_* 都是提示
            soft_prefixes = ("advanced 窗口", "提示：")
            hard = [p for p in problems if not p.startswith(soft_prefixes)]
            soft = [p for p in problems if p.startswith(soft_prefixes)]
            if hard:
                hard_failures.append((f, hard))
            if soft:
                warnings.append((f, soft))

    print("\n=== 汇总 ===")
    if hard_failures:
        print(f"发现 {len(hard_failures)} 个硬校验失败:")
        for f, probs in hard_failures:
            print(f"  {f}:")
            for p in probs:
                print(f"    - {p}")
    if warnings:
        print(f"{len(warnings)} 个提示（不崩溃但建议修）:")
        for f, probs in warnings:
            print(f"  {f}: {probs}")
    if not hard_failures and not warnings:
        print("全部配置通过校验")
    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
