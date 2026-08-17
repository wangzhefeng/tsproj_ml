# -*- coding: utf-8 -*-
"""调度模式驱动的策略特征工程流水线。

数据流:
  1. 加载 merged(站用电+PCS实际) + plan(计划功率)，截断到 >= 2025-11-01（10 月不处理）
  2. 按 [当日22:00 ~ 次日21:55] 窗口提取日级标签（充放电起止/功率/时长/偏差率）
  3. KMeans 聚类分型（silhouette 3~8 自动选 K，只用调度行为特征）
  4. 每类构建站用电典型日曲线（中位数 ± 1σ）
  5. 5min 粒度策略条件化特征（history 直接算，future 用计划+偏差率映射）
  6. 月度典型日（每月第 2 天窗口，不完整则顺延）

输出: dataset/aidc_ess_selfuse_load/forecasting_data/strategy_features/
  - daily_labels_{A,B}.csv                    日级标签 + 聚类 ID
  - cluster_summary_{A,B}.csv                 聚类摘要
  - cluster_month_distribution_{A,B}.csv      类 × 月计数
  - typical_curves_{A,B}.csv                  典型日曲线
  - strategy_features_history_{A,B}.csv       5min 历史特征（custom_features history_path）
  - strategy_features_future_{A,B}.csv        5min 未来特征（custom_features future_path）
  - monthly_typical_{A,B}.csv                 月度典型日
  - typical_curve_plots/、monthly_typical_plots/  可视化

用法（仓库根目录）:
    uv run python config/aidc_ess_selfuse_load/build_strategy_features.py

关键决策（2026-08-17 确认）:
  - 充电判定阈值 -1500 kW（正常充电 -2000~-2500），放电判定 +5000 kW（正常 +8000~+8500）
  - 2025-10 完全不处理，直接从 2025-11-01 起
  - 未来段聚类强制最近邻匹配历史已有模式，不标未知模式
  - 本脚本只产数据文件，不改模型配置
"""
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "dataset/aidc_ess_selfuse_load"
FORECAST_DIR = DATA_DIR / "forecasting_data"
PLAN_DIR = DATA_DIR / "exogenous_strategy_plan/up_sampled"
OUT_DIR = FORECAST_DIR / "strategy_features"

DATA_START = pd.Timestamp("2025-11-01 00:00")  # 10 月不处理
CHARGE_THRESHOLD = -1500.0                     # kW，低于判定为充电段
DISCHARGE_THRESHOLD = 5000.0                   # kW，高于判定为放电段
WINDOW_POINTS = 288                            # 5min × 288 = 24h
K_RANGE = range(3, 9)                          # silhouette 候选 K: 3~8
MIN_CLUSTER_WINDOWS = 10                       # 样本数少于此的类合并到最近类（方案 §7 风险缓解）

# 聚类特征列（只用调度行为，不含站用电）
CLUSTER_FEATURES = [
    "charge_hours", "discharge_hours",
    "charge_power_mean", "discharge_power_mean",
    "plan_charge_hours", "plan_discharge_hours",
    "plan_charge_power_mean", "plan_discharge_power_mean",
]


# ---------------------------------------------------------------------------
# Task 1: 数据加载与日级标签提取
# ---------------------------------------------------------------------------
def load_route(gate: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (hist, future)。hist: 2025-11-01~2026-07-28 的 merged+plan；future: 07-29~07-31 的 plan。"""
    merged = pd.read_csv(
        FORECAST_DIR / f"{gate}_ESS_PCS_merged_5min_20251001_20260728.csv",
        parse_dates=["time"],
    ).set_index("time")
    plan = pd.read_csv(
        PLAN_DIR / f"{gate}_PCS_plan_5min_20251001_20260731.csv",
        parse_dates=["time"],
    ).set_index("time")

    hist = merged.join(plan, how="left")       # 历史段以 merged 为主
    hist = hist[hist.index >= DATA_START]      # 截断 10 月
    future = plan[plan.index > merged.index.max()]  # 未来段
    return hist, future


def get_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """连续 True 区间 [(start, end), ...]，闭区间索引。"""
    padded = np.concatenate([[0], mask.view(np.int8), [0]])
    idx = np.flatnonzero(np.diff(padded))
    return list(zip(idx[::2], idx[1::2] - 1))


def _label_one_window(win: pd.DataFrame, start: pd.Timestamp) -> dict:
    pcs = win["pcs_power"].to_numpy()
    plan = win["pcs_plan"].to_numpy()
    ess = win["ess_power"].to_numpy()

    ch, dch = pcs < CHARGE_THRESHOLD, pcs > DISCHARGE_THRESHOLD
    pch, pdch = plan < CHARGE_THRESHOLD, plan > DISCHARGE_THRESHOLD
    ch_seg, dch_seg = get_segments(ch), get_segments(dch)

    return {
        "window_start": start,
        "window_date": start.strftime("%Y-%m-%d"),
        "month": start.month,
        "is_weekend": int(start.dayofweek >= 5),
        # 实际充电段
        "charge_start_min": ch_seg[0][0] if ch_seg else 0,
        "charge_end_min": ch_seg[-1][1] if ch_seg else 0,
        "charge_power_mean": pcs[ch].mean() if ch.any() else np.nan,
        "charge_hours": ch.sum() * 5 / 60,
        # 实际放电段
        "discharge_start_min": dch_seg[0][0] if dch_seg else 0,
        "discharge_end_min": dch_seg[-1][1] if dch_seg else 0,
        "discharge_power_mean": pcs[dch].mean() if dch.any() else np.nan,
        "discharge_hours": dch.sum() * 5 / 60,
        # 计划侧
        "plan_charge_power_mean": plan[pch].mean() if pch.any() else np.nan,
        "plan_discharge_power_mean": plan[pdch].mean() if pdch.any() else np.nan,
        "plan_charge_hours": pch.sum() * 5 / 60,
        "plan_discharge_hours": pdch.sum() * 5 / 60,
        # 站用电统计
        "ess_power_mean": ess.mean(),
        "ess_power_std": ess.std(),
    }


def extract_window_labels(hist: pd.DataFrame) -> pd.DataFrame:
    """对每个 [22:00 ~ 次日21:55] 完整窗口提取日级标签。"""
    labels = []
    start = hist.index.min().normalize() + pd.Timedelta(hours=22)
    if start < hist.index.min():  # 首日 22:00 已过则从下一天起
        start += pd.Timedelta(days=1)
    last_end = hist.index.max()
    step = pd.Timedelta(minutes=5 * (WINDOW_POINTS - 1))
    while start + step <= last_end:
        win = hist.loc[start:start + step]
        if len(win) == WINDOW_POINTS:
            labels.append(_label_one_window(win, start))
        start += pd.Timedelta(days=1)

    labels_df = pd.DataFrame(labels)
    # 偏差率
    labels_df["charge_power_ratio"] = (
        labels_df["charge_power_mean"] / labels_df["plan_charge_power_mean"]
    ).replace([np.inf, -np.inf], np.nan)
    labels_df["discharge_hours_ratio"] = (
        labels_df["discharge_hours"] / labels_df["plan_discharge_hours"]
    ).replace([np.inf, -np.inf], np.nan)
    return labels_df


# ---------------------------------------------------------------------------
# Task 2: KMeans 聚类分型
# ---------------------------------------------------------------------------
def _merge_small_clusters(labels_arr: np.ndarray, centers: np.ndarray,
                          min_size: int) -> tuple[np.ndarray, dict[int, int]]:
    """样本数 < min_size 的类合并到质心最近的类。

    返回 (重编码标签, full_map)。full_map 把**所有**原始 KMeans 类 ID 映射到
    合并后的新顺序 ID（含被合并掉的类），供 future 段 argmin 结果转换。
    """
    labels = labels_arr.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        small = ids[counts < min_size]
        if len(small) == 0:
            break
        sid = small[np.argmin(counts[counts < min_size])]   # 先合并最小的类
        targets = ids[counts >= min_size]
        if len(targets) == 0:                                # 全是小类则互相合并
            targets = ids[ids != sid]
        d = np.linalg.norm(centers[targets] - centers[sid], axis=1)
        labels[labels == sid] = targets[np.argmin(d)]
    surviving = np.unique(labels)
    surv_map = {int(old): new for new, old in enumerate(surviving)}
    reencoded = np.array([surv_map[int(v)] for v in labels])
    # 全量映射：每个原始类 → 其最终归宿（重编码 ID）
    full_map = {}
    for old_id in range(len(centers)):
        member = labels_arr == old_id
        full_map[int(old_id)] = int(reencoded[member.argmax()]) if member.any() else 0
    return reencoded, full_map


def cluster_patterns(labels: pd.DataFrame) -> tuple[pd.DataFrame, KMeans, StandardScaler, dict, dict]:
    """silhouette 自动选 K（3~8），小类合并；返回 (labels, km, scaler, scores, full_map)。"""
    X = labels[CLUSTER_FEATURES].fillna(0.0)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    scores = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
        scores[k] = silhouette_score(Xs, km.labels_)
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Xs)

    merged_ids, full_map = _merge_small_clusters(km.labels_, km.cluster_centers_,
                                                 MIN_CLUSTER_WINDOWS)
    labels = labels.copy()
    labels["schedule_pattern_id"] = merged_ids
    n_surv = len(set(full_map.values()))
    print(f"  silhouette: {{{', '.join(f'{k}: {v:.3f}' for k, v in scores.items())}}} -> best_k={best_k}"
          f"，合并小类(<{MIN_CLUSTER_WINDOWS})后 {n_surv} 类")
    return labels, km, scaler, scores, full_map


def cluster_summary(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = labels.groupby("schedule_pattern_id").agg(
        n_windows=("window_date", "count"),
        charge_hours=("charge_hours", "mean"),
        discharge_hours=("discharge_hours", "mean"),
        charge_power_mean=("charge_power_mean", "mean"),
        discharge_power_mean=("discharge_power_mean", "mean"),
        ess_power_mean=("ess_power_mean", "mean"),
    )
    month_dist = labels.groupby(["schedule_pattern_id", "month"]).size().unstack(fill_value=0)
    return summary, month_dist


# ---------------------------------------------------------------------------
# Task 3: 典型日曲线构建
# ---------------------------------------------------------------------------
def build_typical_curves(hist: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """每类站用电中位数曲线（288 点）+ 标准差带。"""
    rows = []
    step = pd.Timedelta(minutes=5 * (WINDOW_POINTS - 1))
    for pid, grp in labels.groupby("schedule_pattern_id"):
        curves = []
        for ws in grp["window_start"]:
            win = hist.loc[ws:ws + step, "ess_power"]
            if len(win) == WINDOW_POINTS:
                curves.append(win.to_numpy())
        arr = np.stack(curves)
        rows.append({
            "pattern_id": pid,
            "minute_of_day": np.arange(WINDOW_POINTS),
            "typical_ess_power": np.median(arr, axis=0),
            "std_ess_power": np.std(arr, axis=0),
            "n_windows": len(curves),
        })
    return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)


def plot_typical_curves(curves: pd.DataFrame, out_dir: Path, gate: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = pd.date_range("2025-01-01 22:00", periods=WINDOW_POINTS, freq="5min")
    for pid, sub in curves.groupby("pattern_id"):
        fig, ax = plt.subplots(figsize=(14, 5), dpi=120)
        ax.plot(x, sub["typical_ess_power"], color="#1f77b4", lw=1.2,
                label=f"median (n={sub['n_windows'].iloc[0]})")
        ax.fill_between(x,
                        sub["typical_ess_power"] - sub["std_ess_power"],
                        sub["typical_ess_power"] + sub["std_ess_power"],
                        color="#1f77b4", alpha=0.18, label="±1σ")
        ax.set_title(f"Typical ESS Self-Use Curve - {gate} Pattern {pid}")
        ax.set_ylabel("ESS Power (kW)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{gate}_pattern_{pid}.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Task 4: 策略条件化特征（history + future）
# ---------------------------------------------------------------------------
def window_start_of(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """每个时刻所属的 22:00 窗口起点；00:00~21:55 归属前一日 22:00 窗口。"""
    shifted = idx - pd.Timedelta(hours=22)
    return shifted.normalize() + pd.Timedelta(hours=22)


def minute_of_window(idx: pd.DatetimeIndex) -> np.ndarray:
    """窗口内序号 0~287。"""
    return ((idx - window_start_of(idx)).total_seconds() / 300).astype(int)


def _attach_pattern_and_curve(f: pd.DataFrame,
                              day_map: pd.DataFrame,
                              curves: pd.DataFrame) -> pd.DataFrame:
    """把日级 pattern_id broadcast 到 5min，并挂上典型曲线值。

    边缘兜底：首段（归属前一日 22:00 窗口、无标签）bfill，尾段（不完整窗口）ffill。
    """
    pids = day_map.reindex(window_start_of(f.index)).astype("float")
    f["schedule_pattern_id"] = pids.bfill().ffill().to_numpy().astype(int)
    f["month"] = f.index.month
    f["is_weekend"] = (f.index.dayofweek >= 5).astype(int)

    curve_map = curves.set_index(["pattern_id", "minute_of_day"])["typical_ess_power"]
    keys = pd.MultiIndex.from_arrays([f["schedule_pattern_id"], minute_of_window(f.index)])
    f["typical_ess_power"] = curve_map.reindex(keys).to_numpy()
    return f


def build_history_features(hist: pd.DataFrame, labels: pd.DataFrame,
                           curves: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=hist.index)
    pcs, plan = hist["pcs_power"], hist["pcs_plan"]

    f["actual_charge_binary"] = (pcs < CHARGE_THRESHOLD).astype(int)
    f["actual_discharge_binary"] = (pcs > DISCHARGE_THRESHOLD).astype(int)
    f["plan_charge_binary"] = (plan < CHARGE_THRESHOLD).astype(int)
    f["plan_discharge_binary"] = (plan > DISCHARGE_THRESHOLD).astype(int)
    f["actual_power_abs"] = pcs.abs()
    f["plan_power_abs"] = plan.abs()
    f["charge_power_deviation"] = np.where(f["actual_charge_binary"] == 1, pcs - plan, np.nan)
    f["discharge_power_deviation"] = np.where(f["actual_discharge_binary"] == 1, pcs - plan, np.nan)

    day_map = labels.set_index("window_start")["schedule_pattern_id"]
    f = _attach_pattern_and_curve(f, day_map, curves)
    f.index.name = "time"
    return f.reset_index()


def match_future_patterns(future: pd.DataFrame, km: KMeans,
                          scaler: StandardScaler, full_map: dict[int, int]) -> pd.DataFrame:
    """未来段逐窗口提取计划侧标签 → 最近邻匹配历史聚类中心（强制落到已有模式）。

    argmin 给出的是原始 KMeans 类 ID，经 full_map 转成合并后的顺序 ID。
    """
    step = pd.Timedelta(minutes=5 * (WINDOW_POINTS - 1))
    rows = []
    start = future.index.min().normalize() + pd.Timedelta(hours=22)
    if start < future.index.min():
        start += pd.Timedelta(days=1)
    while start + step <= future.index.max():
        win = future.loc[start:start + step]
        if len(win) == WINDOW_POINTS:
            plan = win["pcs_plan"].to_numpy()
            pch, pdch = plan < CHARGE_THRESHOLD, plan > DISCHARGE_THRESHOLD
            rows.append({
                "window_start": start,
                "charge_hours": 0.0, "discharge_hours": 0.0,        # 实际未知 → 0
                "charge_power_mean": 0.0, "discharge_power_mean": 0.0,
                "plan_charge_hours": pch.sum() * 5 / 60,
                "plan_discharge_hours": pdch.sum() * 5 / 60,
                "plan_charge_power_mean": plan[pch].mean() if pch.any() else 0.0,
                "plan_discharge_power_mean": plan[pdch].mean() if pdch.any() else 0.0,
            })
        start += pd.Timedelta(days=1)

    fut_labels = pd.DataFrame(rows)
    Xs = scaler.transform(fut_labels[CLUSTER_FEATURES].fillna(0.0))
    dist = np.linalg.norm(Xs[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    raw_ids = dist.argmin(axis=1)
    fut_labels["schedule_pattern_id"] = [full_map[int(v)] for v in raw_ids]
    fut_labels["match_distance"] = dist.min(axis=1)
    return fut_labels


def build_future_features(future: pd.DataFrame, fut_labels: pd.DataFrame,
                          curves: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    charge_ratio_med = labels["charge_power_ratio"].median()  # 充电缩水修正
    f = pd.DataFrame(index=future.index)
    plan = future["pcs_plan"]

    f["plan_charge_binary"] = (plan < CHARGE_THRESHOLD).astype(int)
    f["plan_discharge_binary"] = (plan > DISCHARGE_THRESHOLD).astype(int)
    f["plan_power_abs"] = plan.abs()
    f["actual_charge_binary"] = f["plan_charge_binary"]
    f["actual_discharge_binary"] = f["plan_discharge_binary"]
    f["actual_power_abs"] = np.select(
        [f["plan_charge_binary"] == 1, f["plan_discharge_binary"] == 1],
        [plan.abs() * charge_ratio_med, plan.abs()],
        default=0.0,
    )
    f["charge_power_deviation"] = np.nan      # 未来段无实际值
    f["discharge_power_deviation"] = np.nan

    day_map = fut_labels.set_index("window_start")["schedule_pattern_id"]
    f = _attach_pattern_and_curve(f, day_map, curves)
    f.index.name = "time"
    return f.reset_index()


# ---------------------------------------------------------------------------
# Task 5: 月度典型模式
# ---------------------------------------------------------------------------
def build_monthly_typical(hist: pd.DataFrame) -> pd.DataFrame:
    """每月第 2 天的 22:00 窗口；不完整则顺延到该月首个完整窗口。"""
    step = pd.Timedelta(minutes=5 * (WINDOW_POINTS - 1))
    rows = []
    for (year, month), sub in hist.groupby([hist.index.year, hist.index.month]):
        picked = None
        for day in range(2, 29):
            cand = pd.Timestamp(year, month, day, 22, 0)
            if cand + step > sub.index.max():
                break
            win = sub.loc[cand:cand + step]
            if len(win) == WINDOW_POINTS:
                picked = (cand, win)
                break
        if picked is None:
            print(f"  WARNING: {year}-{month:02d} 无完整窗口，跳过")
            continue
        cand, win = picked
        rows.append({
            "month": month,
            "year": year,
            "window_start": cand,
            "minute_of_day": np.arange(WINDOW_POINTS),
            "ess_power": win["ess_power"].to_numpy(),
            "pcs_power": win["pcs_power"].to_numpy(),
            "pcs_plan": win["pcs_plan"].to_numpy(),
        })
    return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)


def plot_monthly_typical(monthly: pd.DataFrame, out_dir: Path, gate: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x = pd.date_range("2025-01-01 22:00", periods=WINDOW_POINTS, freq="5min")
    for (year, month), sub in monthly.groupby(["year", "month"]):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=120)
        ax1.plot(x, sub["ess_power"], color="#1f77b4", lw=1.0, label="ESS self-use")
        ax1.set_ylabel("ESS Power (kW)")
        ax1.legend(fontsize=9)
        ax2.plot(x, sub["pcs_power"], color="#d62728", lw=1.0, label="PCS actual")
        ax2.plot(x, sub["pcs_plan"], color="#2ca02c", lw=1.0, ls="--", label="PCS plan")
        ax2.axhline(0, color="gray", lw=0.5, alpha=0.5)
        ax2.set_ylabel("PCS Power (kW)")
        ax2.legend(fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax1.set_title(f"Monthly Typical Day - {gate} {year}-{month:02d} (window {sub['window_start'].iloc[0].date()})")
        fig.tight_layout()
        fig.savefig(out_dir / f"{gate}_month_{month:02d}.png", bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Task 6: 主流程整合
# ---------------------------------------------------------------------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for gate in ("A", "B"):
        print(f"=== {gate} 路 ===")
        hist, future = load_route(gate)
        print(f"  hist: {len(hist)} 行 ({hist.index.min()} ~ {hist.index.max()})")
        print(f"  future: {len(future)} 行 ({future.index.min()} ~ {future.index.max()})")

        labels = extract_window_labels(hist)
        print(f"  日级标签: {len(labels)} 窗口")

        labels, km, scaler, _, full_map = cluster_patterns(labels)
        summary, month_dist = cluster_summary(labels)
        print(summary.to_string())

        curves = build_typical_curves(hist, labels)
        plot_typical_curves(curves, OUT_DIR / "typical_curve_plots", gate)

        hist_feat = build_history_features(hist, labels, curves)
        fut_labels = match_future_patterns(future, km, scaler, full_map)
        fut_feat = build_future_features(future, fut_labels, curves, labels)

        monthly = build_monthly_typical(hist)
        plot_monthly_typical(monthly, OUT_DIR / "monthly_typical_plots", gate)

        # 落盘
        labels.to_csv(OUT_DIR / f"daily_labels_{gate}.csv", index=False)
        summary.to_csv(OUT_DIR / f"cluster_summary_{gate}.csv")
        month_dist.to_csv(OUT_DIR / f"cluster_month_distribution_{gate}.csv")
        curves.to_csv(OUT_DIR / f"typical_curves_{gate}.csv", index=False)
        hist_feat.to_csv(OUT_DIR / f"strategy_features_history_{gate}.csv", index=False)
        fut_feat.to_csv(OUT_DIR / f"strategy_features_future_{gate}.csv", index=False)
        monthly.to_csv(OUT_DIR / f"monthly_typical_{gate}.csv", index=False)
        print(f"  输出完成 -> {OUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
