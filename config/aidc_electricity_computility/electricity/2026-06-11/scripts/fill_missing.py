# -*- coding: utf-8 -*-
"""缺失值预处理：原始聚合列补 0 + 重算派生列。

设计依据（见对话分析）：
- 目标列 ``h_total_use`` 与时间列 ``count_data_time`` 在所有文件中 100% 完整，绝不填充。
- 真实缺失几乎全部来自 ``inference_*``（推理负载那段时间不存在），语义上等价于 0。
- 派生列（占比/比值/滚动）是按基础列算出来的；若直接对派生列补 0，会把
  ``training_to_inference_power_ratio`` 这类 "undefined" 伪装成 0。因此做法是：
  1) 仅对 162 个原始聚合列补 0；
  2) 丢弃旧的 51 个派生列，用 ``computility_process`` 的 ``add_structural_features``
     / ``add_dynamic_features`` 在补 0 后的基础列上重算 → 内部一致；
  3) 重算后残存的仅有滚动特征开头热身 NaN（diff_1 首行 / diff_3 前 3 行 /
     roll_std_12 首行），统一补 0 得到完全稠密的结果。
- ``*_present`` 标志由生成器按 ``_count`` 列 > 0 计算，补 0 后天然保留 "真实 0" 与
  "原本缺失" 的区分。

输出：在原 ``df.csv`` 旁生成 ``df_filled.csv``，不改动原始文件。
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# 复用算力生成管线里的常量与重算函数，避免逻辑漂移
sys.path.insert(0, str(Path(__file__).resolve().parent))
from computility_process import (  # noqa: E402
    SUM_METRICS,
    UTIL_METRICS,
    POWER_TARGET_COL,
    TIME_COL,
    add_dynamic_features,
    add_structural_features,
)

ROOMS = ("A1_01a", "A1_201", "A3_01e", "A1_IT")


def build_raw_column_names() -> List[str]:
    """生成 162 个原始聚合列名（model_training/inference/pod × 9 指标 × 统计量）。

    统计量集合与 ``computility_process.expand_metric_file`` 完全对齐：
    所有指标都有 min/max/mean/std/count；SUM_METRICS 加 sum；gpu_util 加
    busy_count/busy_ratio；gpu_power_usage 加 high_power_count/high_power_ratio。
    """
    columns: List[str] = []
    all_metrics = UTIL_METRICS | SUM_METRICS
    for prefix in ("training", "inference", "pod"):
        for metric in all_metrics:
            stats = ["min", "max", "mean", "std", "count"]
            if metric in SUM_METRICS:
                stats.append("sum")
            if metric == "gpu_util":
                stats += ["busy_count", "busy_ratio"]
            if metric == "gpu_power_usage":
                stats += ["high_power_count", "high_power_ratio"]
            for stat in stats:
                columns.append(f"{prefix}_{metric}_{stat}")
    return columns


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """对单个房间的 df 应用「补 0 + 重算派生」策略，返回完全稠密的 DataFrame。

    列顺序与原 df 完全一致：在「时间 + 原始列」拷贝上补 0 并重算派生后，按原列名
    把数值回填进 df 的拷贝；时间列与目标列不在 raw/derived 集合内，原样保留。
    """
    if TIME_COL not in df.columns or POWER_TARGET_COL not in df.columns:
        raise ValueError(f"缺少 {TIME_COL} 或 {POWER_TARGET_COL}")

    raw_cols = [c for c in build_raw_column_names() if c in df.columns]

    # 1) 在「时间 + 原始聚合列」拷贝上补 0，顺带丢弃旧派生列
    base = df[[TIME_COL, *raw_cols]].copy()
    base[raw_cols] = base[raw_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    # 2) 重算派生列（structural → dynamic，顺序与生成管线一致）
    base = add_structural_features(base)
    base = add_dynamic_features(base)

    # 3) 残存的只有滚动特征开头热身 NaN，统一补 0
    derived_cols = [c for c in base.columns if c not in (TIME_COL, *raw_cols)]
    base[derived_cols] = base[derived_cols].fillna(0)

    # 4) 按原 df 列序回填数值；时间列/目标列不在 raw/derived 内，原样保留
    result = df.copy()
    result[raw_cols] = base[raw_cols].values
    result[derived_cols] = base[derived_cols].values
    return result


def process_room(room_dir: Path) -> Path:
    src = room_dir / "df.csv"
    if not src.exists():
        raise FileNotFoundError(f"缺少 {src}")

    df = pd.read_csv(src, encoding="utf-8-sig")
    before_missing = int(df.drop(columns=[TIME_COL]).isna().sum().sum())

    filled = fill_missing(df)
    after_missing = int(filled.drop(columns=[TIME_COL]).isna().sum().sum())

    dst = room_dir / "df_filled.csv"
    filled.to_csv(dst, index=False, encoding="utf-8-sig")

    print(
        f"{room_dir.name}: {len(df)} 行, {len(filled.columns)} 列 | "
        f"缺失 {before_missing:,} → {after_missing} (写入 {dst.name})"
    )
    return dst


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对 df.csv 做缺失补 0 + 重算派生，输出 df_filled.csv")
    parser.add_argument(
        "--demand-root",
        type=Path,
        default=Path("dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load"),
        help="demand_load 根目录，包含各机房子目录",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    for room in ROOMS:
        room_dir = args.demand_root / room
        if not room_dir.exists():
            print(f"[跳过] {room}: 目录不存在 {room_dir}")
            continue
        process_room(room_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
