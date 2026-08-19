#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tsproj_ml 模型配置 dry-run 校验：只加载配置 + 复算 main.py 合法性校验，不训练不预测。

用法（项目根，注意 env -u PYTHONPATH 防 Hermes 注入遮蔽项目 utils/ 包）：
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/<scenario>/route_*/lgbm_*.yaml'
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/aidc_power_15min/**/*.yaml'
    （无参数时默认 glob config/**/*.yaml，会命中聚合/异常配置，按文件名前缀跳过即可）

复算的校验（与 main.py.__init__ 一致）：
  - window_length < history_length
  - window_len - horizon > max(lags)（滞后特征非全 NaN；USMDP 逐点法除外）
  - detrend_target 与 scale_target 互斥
  - n_windows > 0（滑窗数）
  - advanced_features：USMDP 不能直接依赖 y；仅操作历史/未来都存在的列才可用
    （USMDP 不自动生成 y_lag_*，引用它们会被跳过并提示）；预测上下文取
    max(lags, 已启用 rolling_windows/diff_periods/pct_change_periods)，且不得超过
    history_length × n_per_day

退出码：0 = 全部通过（可能有提示性警告），1 = 存在硬校验失败。
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml  # noqa: E402

from config.config_loader import load_yaml_config  # noqa: E402
from utils.frequency import resolve_samples_per_day  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent

# 聚合/异常配置：独立 schema，无 base_config，不能被 load_yaml_config 加载
_NON_MODEL_PREFIXES = ("aggregate", "outlier_detect", "outlier")


def check_model_yaml(f: str) -> tuple[object, list[str]]:
    """加载单个模型 YAML，返回 (cfg, problems)。problems 空 = 通过。"""
    cfg = load_yaml_config(f)
    n_per_day = resolve_samples_per_day(cfg.freq)
    window_len = cfg.window_length * n_per_day
    horizon = int(cfg.predict_steps)
    max_lag = max(cfg.lags or [0])
    problems: list[str] = []

    if cfg.window_length >= cfg.history_length:
        problems.append(
            f"window_length({cfg.window_length}) >= history_length({cfg.history_length})"
        )

    if cfg.pred_method != "univariate-single-multistep-direct-pointwise":
        min_train_rows = window_len - horizon
        if min_train_rows <= max_lag:
            problems.append(
                f"lag 校验失败: window_len-horizon={min_train_rows} <= max_lag={max_lag} "
                "(Lag features would be all-NaN)"
            )

    if getattr(cfg, "detrend_target", False) and getattr(cfg, "scale_target", False):
        problems.append("detrend_target 与 scale_target 同时开启（互斥）")

    n_windows = (cfg.history_length * n_per_day - window_len) // horizon + 1
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
        if is_usmdp and bool(getattr(cfg, "enable_rolling_features", False)) and any(
            column.startswith("y_lag_") for column in roll_columns
        ):
            problems.append(
                "提示：USMDP 不会自动生成 y_lag_*；rolling_columns 中的 y_lag_* "
                "会被特征工程跳过，当前配置不会形成该消融特征。"
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
        horizon = int(cfg.predict_steps)
        max_lag = max(cfg.lags or [0])
        n_windows = (cfg.history_length * n_per_day - cfg.window_length * n_per_day) // horizon + 1
        adv = getattr(cfg, "enable_advanced_features", False)
        print(f"{f}")
        print(f"    method={method} freq={cfg.freq} n_per_day={n_per_day} "
              f"history={cfg.history_length}d window={cfg.window_length}d horizon={horizon} "
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
