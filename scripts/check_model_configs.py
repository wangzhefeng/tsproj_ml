#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tsproj_ml 模型配置 dry-run 校验：只加载配置 + 复算 main.py 合法性校验，不训练不预测。

用法（项目根，注意 env -u PYTHONPATH 防 Hermes 注入遮蔽项目 utils/ 包）：
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/<scenario>/route_*/lgbm_*.yaml'
    env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/aidc_power_15min/**/*.yaml'
    （无参数时默认 glob config/**/*.yaml，会命中聚合/异常配置，按文件名前缀跳过即可）

复算的校验（与 main.py.__init__ 一致）：
  - window_days < history_days
  - window_len - horizon > max(lags)（滞后特征非全 NaN；USMDP 逐点法除外）
  - detrend_target 与 scale_target 互斥
  - n_windows > 0（滑窗数）
  - advanced_features：USMDP 不兼容；rolling_windows/diff_periods <= max(lags)
    （预测路径 df_history_for_lags 只有 max_lag 行，超限特征在预测时退化/NaN）

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
    window_len = cfg.window_days * n_per_day
    horizon = int(cfg.predict_steps)
    max_lag = max(cfg.lags or [0])
    problems: list[str] = []

    if cfg.window_days >= cfg.history_days:
        problems.append(
            f"window_days({cfg.window_days}) >= history_days({cfg.history_days})"
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

    n_windows = (cfg.history_days * n_per_day - window_len) // horizon + 1
    if n_windows <= 0:
        problems.append(f"n_windows={n_windows} <= 0，测试会被跳过")

    adv = getattr(cfg, "enable_advanced_features", False)
    if adv:
        if cfg.pred_method == "univariate-single-multistep-direct-pointwise":
            problems.append(
                "USMDP + advanced_features 不兼容（训练有 rolling/diff，预测 df_future 无 y）"
            )
        roll_win = list(getattr(cfg, "rolling_windows", []) or [])
        diff_per = list(getattr(cfg, "diff_periods", []) or [])
        over = [w for w in roll_win if w > max_lag] + [p for p in diff_per if p > max_lag]
        if over:
            problems.append(
                f"advanced 窗口/周期 {over} > max_lag({max_lag})：预测时"
                "(df_history_for_lags 仅 max_lag 行) 滚动退化为累计值、diff 为 NaN"
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
        n_windows = (cfg.history_days * n_per_day - cfg.window_days * n_per_day) // horizon + 1
        adv = getattr(cfg, "enable_advanced_features", False)
        print(f"{f}")
        print(f"    method={method} freq={cfg.freq} n_per_day={n_per_day} "
              f"history={cfg.history_days}d window={cfg.window_days}d horizon={horizon} "
              f"max_lag={max_lag} n_windows={n_windows} adv={adv} now_time={cfg.now_time}")
        if problems:
            # 硬校验 vs 提示：advanced 窗口超限是提示（不崩溃）；其余为硬失败
            hard = [p for p in problems if not p.startswith("advanced 窗口")]
            if hard:
                hard_failures.append((f, hard))
            else:
                warnings.append((f, [p for p in problems if p.startswith("advanced 窗口")]))

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
