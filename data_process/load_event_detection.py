# -*- coding: utf-8 -*-
"""AIDC 负荷事件检测核心（15min 与日频共用的检测算法库）。

事件分类学（label taxonomy，两频率共享同一口径）：
  - shift_up / shift_down    持久水平阶跃：分段水平差超过阈值且后续不回落
                            （对应"设备集中上架/下架"导致的负荷台阶）
  - stress_up / stress_down  临时水平偏移：水平阶跃后在窗口期内回落到原基线
                            （对应"压力测试/临时批量操作/检修迁移"，日级 1~21 天）
  - burst_up / burst_down    日内持续冲击：15min 残差异常连续 5~96 点
                            （1.25h~24h，对应"压测冲击/突发负荷块"）
  - spike_up / spike_down    功率突变：15min 残差异常连续 <=4 点（<=1h），
                            快速回落（对应"瞬时功率突变/切换操作"）

检测由三个独立探测器组成，可单独调用也可组合：
  1. topdown_day_segments + classify_day_events
     自顶向下贪心分段（每轮在所有段内找最强分裂点），再按"是否回落"分类为
     shift（持久阶跃）或 stress（临时偏移）。
  2. detect_short_excursions
     1~2 天的短时偏移（太短无法构成独立分段），对照前后各 7 天侧窗中位数。
  3. detect_intraday_events
     15min 级残差突变：相对 48h 居中滚动中位数的残差，用 7d 居中 MAD 定标，
     连续异常点成段后按长度分为 spike（<=4 点）/ burst（5~96 点）。

注意：检测使用的居中窗口含未来信息，仅用于离线分析/打标/评估，
不应把 lbl_* 列直接当作在线预测特征；feat_* 前缀列均为 trailing 窗口
统计，无未来泄漏，可安全用于预测建模。标签族内唯一例外：
lbl_volatile_day 的阈值（std.shift(1).rolling(60).quantile(0.95)）
严格只用历史，当天 std 当日末可知，因此它是因果标签，经 lag 消费
（如 feat_volatile_prev_day）后可作在线建模特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "EventDetectionConfig",
    "Segment",
    "LoadEvent",
    "topdown_day_segments",
    "classify_day_events",
    "detect_short_excursions",
    "detect_intraday_events",
    "merge_day_events",
    "suppress_boundary_artifacts",
    "events_to_frame",
    "day_intraday_stats",
    "project_events_to_days",
    "project_events_to_points",
]


# ---------------------------------------------------------------------------
# 配置与数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EventDetectionConfig:
    """事件检测阈值（默认值已在 AIDC A/B 路 15min/日频负荷上标定）。"""

    # --- 日级分段 ---
    seg_edge_window: int = 7        # 分裂点前后各取多少天的中位数比较
    seg_min_days: int = 3           # 每个分段最短天数
    seg_min_kw: float = 300.0       # 分裂所需最小水平差（kW）
    seg_min_pct: float = 0.02       # 或最小水平差百分比（与 min_kw 取大）
    seg_max_depth: int = 16         # 自顶向下最大分裂轮数

    # --- 事件分类 ---
    event_min_kw: float = 300.0     # 相邻段水平差低于此值不生成事件
    revert_frac: float = 0.55       # 后段回到前段水平 55% 以内视为"回落"
    temp_max_days: int = 21         # 临时偏移（stress）的最长持续天数

    # --- 短时偏移 ---
    short_side_days: int = 7        # 前后侧窗天数
    short_max_days: int = 2         # 短时偏移窗口最长天数
    short_min_kw: float = 500.0     # 短时偏移最小幅度（kW）
    short_min_pct: float = 0.04     # 或最小幅度百分比（与 min_kw 取大）

    # --- 15min 日内突变 ---
    intraday_baseline_pts: int = 192   # 居中基线窗口（192 点 = 48h）
    intraday_sigma_pts: int = 672      # 残差 MAD 定标窗口（672 点 = 7d）
    intraday_k_sigma: float = 6.0      # 残差显著性倍数
    intraday_min_kw: float = 350.0     # 残差最小绝对幅度（kW）
    intraday_max_gap: int = 2          # 异常点合并允许间隔（点）
    spike_max_pts: int = 4             # spike 最长点数（<=1h）
    burst_max_pts: int = 96            # burst 最长点数（<=24h）


@dataclass(frozen=True)
class Segment:
    """日级水平分段。start/end 为该段首/末日（含）。"""

    start: pd.Timestamp
    end: pd.Timestamp
    level: float          # 段内日水平中位数
    n_days: int


@dataclass(frozen=True)
class LoadEvent:
    """一次负荷事件（跨频率统一结构）。

    kind: shift_up/shift_down/stress_up/stress_down/
          burst_up/burst_down/spike_up/spike_down
    source: day_level（分段/短时偏移）或 intraday（15min 残差）
    amplitude: 日级事件为水平差 kW；日内事件为平均 |残差| kW
    n_days: 日级事件天数；日内事件为点数 / 96（折算天）
    """

    start: pd.Timestamp
    end: pd.Timestamp
    kind: str
    source: str
    base_level: float
    amplitude: float
    n_days: float


# ---------------------------------------------------------------------------
# 1. 日级自顶向下分段
# ---------------------------------------------------------------------------
def topdown_day_segments(
    day_level: pd.Series,
    config: Optional[EventDetectionConfig] = None,
) -> list[Segment]:
    """对日水平序列做自顶向下贪心分段。

    每一轮在所有现有段内扫描候选分裂点 t，比较 t 前后各 edge_window 天
    的中位数差 d，选 |d|/阈值 得分最高且过阈值的分裂点执行分裂，
    直到没有满足显著性的分裂点或达到最大分裂深度。
    """
    cfg = config or EventDetectionConfig()
    if not isinstance(day_level.index, pd.DatetimeIndex):
        raise ValueError("day_level 必须是 DatetimeIndex 索引的日频序列")
    dates = day_level.index
    vals = day_level.to_numpy(dtype=float)
    n = len(vals)
    if n < 2 * cfg.seg_min_days + 1:
        return [Segment(dates[0], dates[-1], float(np.median(vals)), n)]

    def _split(segments: list[tuple[int, int]], depth: int) -> list[tuple[int, int]]:
        if depth > cfg.seg_max_depth:
            return segments
        best: Optional[tuple[float, int, int]] = None
        for si, (s, e) in enumerate(segments):
            if e - s + 1 < 2 * cfg.seg_min_days:
                continue
            for t in range(s + cfg.seg_min_days, e - cfg.seg_min_days + 2):
                left_edge = vals[s:t]
                right_edge = vals[t:e + 1]
                if len(left_edge) > cfg.seg_edge_window:
                    left_edge = left_edge[-cfg.seg_edge_window:]
                if len(right_edge) > cfg.seg_edge_window:
                    right_edge = right_edge[:cfg.seg_edge_window]
                m_l = float(np.median(left_edge))
                m_r = float(np.median(right_edge))
                diff = m_r - m_l
                thr = max(cfg.seg_min_kw, cfg.seg_min_pct * abs(m_l))
                if thr <= 0 or abs(diff) < thr:
                    continue
                score = abs(diff) / thr
                if best is None or score > best[0]:
                    best = (score, si, t)
        if best is None:
            return segments
        _, si, t = best
        s, e = segments[si]
        new_segments = segments[:si] + [(s, t - 1), (t, e)] + segments[si + 1:]
        return _split(new_segments, depth + 1)

    segments_idx = _split([(0, n - 1)], 0)
    return [
        Segment(dates[s], dates[e], float(np.median(vals[s:e + 1])), e - s + 1)
        for s, e in segments_idx
    ]


# ---------------------------------------------------------------------------
# 2. 分段 -> 事件（持久阶跃 vs 临时偏移）
# ---------------------------------------------------------------------------
def classify_day_events(
    segments: list[Segment],
    config: Optional[EventDetectionConfig] = None,
) -> list[LoadEvent]:
    """把相邻段对转成事件：回落到基线的中间段是 stress，否则是 shift。

    遍历相邻段对 (i, i+1)：若存在段 i+2 且其水平回到段 i 的 55% 幅度以内、
    且中间段时长 <= temp_max_days，则段 i+1 整体是一次临时偏移（stress），
    其后的恢复边界不再单独生成事件（i 前进 2）；否则生成持久阶跃（shift）。
    """
    cfg = config or EventDetectionConfig()
    events: list[LoadEvent] = []
    i = 0
    while i < len(segments) - 1:
        seg_l, seg_r = segments[i], segments[i + 1]
        shift = seg_r.level - seg_l.level
        if abs(shift) < cfg.event_min_kw:
            i += 1
            continue
        direction = "up" if shift > 0 else "down"
        nxt = segments[i + 2] if i + 2 < len(segments) else None
        reverted = (
            nxt is not None
            and abs(nxt.level - seg_l.level) < cfg.revert_frac * abs(shift)
            and seg_r.n_days <= cfg.temp_max_days
        )
        if reverted:
            events.append(LoadEvent(
                start=seg_r.start, end=seg_r.end,
                kind=f"stress_{direction}", source="day_level",
                base_level=seg_l.level, amplitude=float(shift),
                n_days=float(seg_r.n_days),
            ))
            # 净残余：临时偏移结束后水平未完全回到基线（如"抬升试验后
            # 部分留存量"），追加一个从基线到新水平的净阶跃事件。
            if nxt is not None:
                net = nxt.level - seg_l.level
                if abs(net) >= cfg.event_min_kw:
                    events.append(LoadEvent(
                        start=nxt.start, end=nxt.end,
                        kind=f"shift_{'up' if net > 0 else 'down'}", source="day_level",
                        base_level=seg_l.level, amplitude=float(net),
                        n_days=float(nxt.n_days),
                    ))
            i += 2  # 跳过恢复段：它不是独立事件
        else:
            events.append(LoadEvent(
                start=seg_r.start, end=seg_r.end,
                kind=f"shift_{direction}", source="day_level",
                base_level=seg_l.level, amplitude=float(shift),
                n_days=float(seg_r.n_days),
            ))
            i += 1
    return events


# ---------------------------------------------------------------------------
# 3. 短时偏移（1~2 天）
# ---------------------------------------------------------------------------
def detect_short_excursions(
    day_level: pd.Series,
    config: Optional[EventDetectionConfig] = None,
) -> list[LoadEvent]:
    """检测 1~short_max_days 天的短时水平偏移（压测/临时操作的日级投影）。

    对每个起点 t、窗口长 m（从长到短优先），比较窗口中位数与前后
    side 天侧窗合并中位数；幅度过阈值即记为 stress 事件并跳过该窗口。
    """
    cfg = config or EventDetectionConfig()
    dates = day_level.index
    vals = day_level.to_numpy(dtype=float)
    n = len(vals)
    events: list[LoadEvent] = []
    t = 0
    while t < n:
        hit = False
        for m in range(cfg.short_max_days, 0, -1):
            if t + m > n:
                continue
            window = vals[t:t + m]
            lo = max(0, t - cfg.short_side_days)
            hi = min(n, t + m + cfg.short_side_days)
            sides = np.concatenate([vals[lo:t], vals[t + m:hi]])
            if len(sides) < cfg.short_side_days:
                continue
            m_win = float(np.median(window))
            m_side = float(np.median(sides))
            amp = m_win - m_side
            if abs(amp) >= max(cfg.short_min_kw, cfg.short_min_pct * abs(m_side)):
                direction = "up" if amp > 0 else "down"
                events.append(LoadEvent(
                    start=dates[t], end=dates[t + m - 1],
                    kind=f"stress_{direction}", source="day_level",
                    base_level=m_side, amplitude=float(amp), n_days=float(m),
                ))
                t = t + m
                hit = True
                break
        if not hit:
            t += 1
    return events


# ---------------------------------------------------------------------------
# 4. 15min 日内突变（spike / burst）
# ---------------------------------------------------------------------------
def detect_intraday_events(
    series: pd.Series,
    config: Optional[EventDetectionConfig] = None,
) -> list[LoadEvent]:
    """15min 序列的日内残差突变检测。

    基线 = 48h 居中滚动中位数；残差的波动率 = 7d 居中 MAD * 1.4826（下限 50）。
    |残差| > max(k*sigma, min_kw) 的点视为异常，间隔 <= max_gap 的异常点
    合并成段：段长 <= spike_max_pts 为 spike（功率突变），5~burst_max_pts
    为 burst（日内冲击）；更长的段属于日级事件，交给分段逻辑处理。
    """
    cfg = config or EventDetectionConfig()
    s = series.astype(float)
    baseline = s.rolling(cfg.intraday_baseline_pts, center=True,
                         min_periods=cfg.intraday_baseline_pts // 2).median()
    resid = s - baseline
    med_resid = resid.rolling(cfg.intraday_sigma_pts, center=True,
                              min_periods=96).median()
    abs_dev = (resid - med_resid).abs()
    mad = abs_dev.rolling(cfg.intraday_sigma_pts, center=True,
                          min_periods=96).median()
    sigma = (1.4826 * mad).clip(lower=50.0)
    anom = (resid.abs() > cfg.intraday_k_sigma * sigma) & (resid.abs() > cfg.intraday_min_kw)

    flags = anom.to_numpy()
    idx = np.flatnonzero(flags)
    if len(idx) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev <= cfg.intraday_max_gap + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))

    index = s.index
    values = s.to_numpy()
    base_arr = baseline.to_numpy()
    events: list[LoadEvent] = []
    for s_i, e_i in runs:
        length = e_i - s_i + 1
        if length > cfg.burst_max_pts:
            continue
        seg_resid = values[s_i:e_i + 1] - base_arr[s_i:e_i + 1]
        mean_resid = float(np.nanmean(seg_resid))
        direction = "up" if mean_resid > 0 else "down"
        kind = "spike" if length <= cfg.spike_max_pts else "burst"
        events.append(LoadEvent(
            start=index[s_i], end=index[e_i],
            kind=f"{kind}_{direction}", source="intraday",
            base_level=float(np.nanmean(base_arr[s_i:e_i + 1])),
            amplitude=float(np.nanmean(np.abs(seg_resid))),
            n_days=round(length / 96.0, 4),
        ))
    return events


# ---------------------------------------------------------------------------
# 事件合并 / 导出
# ---------------------------------------------------------------------------
def suppress_boundary_artifacts(
    intraday_events: list[LoadEvent],
    day_events: list[LoadEvent],
    config: Optional[EventDetectionConfig] = None,
    zone_days: float = 1.0,
    amp_frac: float = 0.8,
) -> list[LoadEvent]:
    """抑制日级事件边界产生的 15min 伪影。

    48h 居中基线跨越水平台阶时，台阶前后数小时的残差可达台阶幅度的一半，
    会被误判成 spike/burst。规则：日内事件若落在某个日级事件边界
    （起点 = 台阶方向，终点次日 = 反方向）前后 zone_days 天内、方向一致、
    且幅度 < amp_frac * 日级幅度，则视为伪影丢弃。
    """
    cfg = config or EventDetectionConfig()
    boundaries: list[tuple[pd.Timestamp, str, float]] = []
    for ev in day_events:
        direction = "up" if ev.kind.endswith("up") else "down"
        boundaries.append((ev.start, direction, abs(ev.amplitude)))
        boundaries.append((ev.end + pd.Timedelta(days=1), "down" if direction == "up" else "up", abs(ev.amplitude)))

    kept: list[LoadEvent] = []
    for ev in intraday_events:
        ev_dir = "up" if ev.kind.endswith("up") else "down"
        is_artifact = False
        for b_time, b_dir, b_amp in boundaries:
            if ev_dir != b_dir:
                continue
            if abs((ev.start - b_time).total_seconds()) <= zone_days * 86400:
                if abs(ev.amplitude) < amp_frac * b_amp:
                    is_artifact = True
                    break
        if not is_artifact:
            kept.append(ev)
    return kept


def merge_day_events(
    segment_events: list[LoadEvent],
    short_events: list[LoadEvent],
) -> list[LoadEvent]:
    """合并日级事件源：短时偏移若与分段事件窗口重叠则丢弃（分段优先）。

    返回按 start 排序的事件列表（未分配 event_id，id 在场景脚本中统一分配）。
    """
    kept: list[LoadEvent] = list(segment_events)
    for ev in short_events:
        ev_dir = "up" if ev.kind.endswith("up") else "down"
        overlap = any(
            ev.start <= seg_ev.end + pd.Timedelta(days=1)
            and seg_ev.start <= ev.end + pd.Timedelta(days=1)
            and seg_ev.kind.endswith(ev_dir)
            for seg_ev in segment_events
        )
        if not overlap:
            kept.append(ev)
    kept.sort(key=lambda e: (e.start, e.end))
    return kept


def events_to_frame(events: list[LoadEvent]) -> pd.DataFrame:
    """事件列表转 DataFrame（事件明细表）。"""
    rows = [{
        "event_start": ev.start,
        "event_end": ev.end,
        "kind": ev.kind,
        "source": ev.source,
        "base_level_kw": round(ev.base_level, 1),
        "amplitude_kw": round(ev.amplitude, 1),
        "amplitude_pct": round(ev.amplitude / ev.base_level * 100, 2) if ev.base_level else np.nan,
        "duration": round(ev.n_days, 2),
    } for ev in events]
    frame = pd.DataFrame(rows, columns=[
        "event_start", "event_end", "kind", "source",
        "base_level_kw", "amplitude_kw", "amplitude_pct", "duration",
    ])
    if not frame.empty:
        frame.insert(0, "event_id", np.arange(1, len(frame) + 1))
    return frame


# ---------------------------------------------------------------------------
# 日内统计（跨频率特征的通用实现）
# ---------------------------------------------------------------------------
def day_intraday_stats(series: pd.Series) -> pd.DataFrame:
    """把 15min 序列聚合成逐日日内统计（供日频脚本做跨频率特征）。

    输出列：intraday_mean/median/std/min/max/range/p95/p5/p95_p5_gap/cv/
            max_abs_step(相邻点最大|Δ|)/peak_time_frac(峰值时刻 0~1)/n_points
    """
    s = series.astype(float)
    grp = s.resample("1D")
    stats = pd.DataFrame({
        "intraday_mean": grp.mean(),
        "intraday_median": grp.median(),
        "intraday_std": grp.std(),
        "intraday_min": grp.min(),
        "intraday_max": grp.max(),
    })
    stats["intraday_range"] = stats["intraday_max"] - stats["intraday_min"]
    stats["intraday_p95"] = grp.quantile(0.95)
    stats["intraday_p5"] = grp.quantile(0.05)
    stats["intraday_p95_p5_gap"] = stats["intraday_p95"] - stats["intraday_p5"]
    stats["intraday_cv"] = stats["intraday_std"] / stats["intraday_mean"]
    stats["intraday_max_abs_step"] = s.diff().abs().resample("1D").max()
    peak_idx = s.resample("1D").apply(lambda x: x.idxmax() if x.notna().any() else pd.NaT)
    stats["intraday_peak_time_frac"] = [
        (t - t.normalize()).total_seconds() / 86400.0
        if pd.notna(t) else np.nan
        for t in peak_idx
    ]
    stats["intraday_n_points"] = grp.count()
    return stats


# ---------------------------------------------------------------------------
# 事件 -> 标签投影
# ---------------------------------------------------------------------------
_KIND_PRIORITY = {
    "spike_up": 0, "spike_down": 0,
    "burst_up": 1, "burst_down": 1,
    "stress_up": 2, "stress_down": 2,
    "shift_up": 3, "shift_down": 3,
}


def project_events_to_days(
    day_index: pd.DatetimeIndex,
    day_events: list[LoadEvent],
    intraday_events: Optional[list[LoadEvent]] = None,
    intraday_stats: Optional[pd.DataFrame] = None,
    volatile_pctl: float = 0.95,
    volatile_lookback: int = 60,
) -> pd.DataFrame:
    """把事件投影成逐日标签表（日频脚本的标签来源）。

    规则：
      - shift 事件：标记事件起始日（新水平首日）为 shift_*_day
      - stress 事件：窗口内每天标记 stress_*_day
      - burst/spike 日内事件：事件发生的当天标记 burst/spike_*_day
      - volatile_day：日内 std 超过滚动 60 天 p95（需提供 intraday_stats）
      - event_day/event_type：以上任一命中（type 按 spike>burst>stress>shift 取优先级）
    """
    days = pd.DatetimeIndex(day_index).normalize()
    labels = pd.DataFrame(index=days)
    for kind in ("shift_up", "shift_down", "stress_up", "stress_down",
                 "burst_up", "burst_down", "spike_up", "spike_down"):
        labels[f"lbl_{kind}_day"] = 0
    labels["lbl_volatile_day"] = 0
    event_of_day: dict[pd.Timestamp, str] = {}

    def _mark(day: pd.Timestamp, kind: str) -> None:
        if day not in labels.index:
            return
        labels.loc[day, f"lbl_{kind}_day"] = 1
        cur = event_of_day.get(day)
        if cur is None or _KIND_PRIORITY[kind] < _KIND_PRIORITY[cur]:
            event_of_day[day] = kind

    for ev in day_events:
        direction_days = pd.date_range(ev.start, ev.end, freq="1D")
        if ev.kind.startswith("shift"):
            _mark(ev.start, ev.kind)  # 阶跃只标起始日
        else:
            for d in direction_days:
                _mark(d, ev.kind)
    for ev in (intraday_events or []):
        _mark(ev.start.normalize(), ev.kind)

    if intraday_stats is not None and "intraday_std" in intraday_stats:
        std = intraday_stats["intraday_std"].reindex(days)
        lookback = volatile_lookback
        base = std.shift(1).rolling(lookback, min_periods=14).quantile(volatile_pctl)
        labels["lbl_volatile_day"] = ((std > base) & base.notna()).astype(int)

    labels["lbl_event_day"] = (
        labels[["lbl_shift_up_day", "lbl_shift_down_day",
                "lbl_stress_up_day", "lbl_stress_down_day",
                "lbl_burst_up_day", "lbl_burst_down_day",
                "lbl_spike_up_day", "lbl_spike_down_day"]].max(axis=1)
    )
    labels["lbl_event_type"] = [event_of_day.get(d, "normal") for d in labels.index]
    labels.loc[labels["lbl_volatile_day"] == 1, "lbl_event_type"] = (
        labels.loc[labels["lbl_volatile_day"] == 1, "lbl_event_type"].replace("normal", "volatile")
    )
    return labels


def project_events_to_points(
    point_index: pd.DatetimeIndex,
    day_events: list[LoadEvent],
    intraday_events: list[LoadEvent],
    shift_zone_days: int = 1,
) -> pd.DataFrame:
    """把事件投影成 15min 逐点标签表（15min 脚本的标签来源）。

    规则：
      - spike/burst 事件：[start, end] 内的每个点标记对应列
      - stress 事件：事件窗口内每天的全部点标记
      - shift 事件：起始日前后各 shift_zone_days 天的"过渡带"内的点标记
      - 逐点 type 按 spike>burst>stress>shift 优先级
    """
    idx = pd.DatetimeIndex(point_index)
    labels = pd.DataFrame(index=idx)
    for kind in ("shift_up", "shift_down", "stress_up", "stress_down",
                 "burst_up", "burst_down", "spike_up", "spike_down"):
        labels[f"lbl_{kind}"] = 0
    point_type = pd.Series("normal", index=idx)

    def _mark_mask(mask: pd.Series, kind: str) -> None:
        labels.loc[mask, f"lbl_{kind}"] = 1
        better = mask & (
            point_type.eq("normal")
            | pd.Series([_KIND_PRIORITY[kind] < _KIND_PRIORITY.get(t, 99)
                         for t in point_type], index=idx)
        )
        point_type.loc[better] = kind

    for ev in intraday_events:
        mask = (idx >= ev.start) & (idx <= ev.end)
        _mark_mask(pd.Series(mask, index=idx), ev.kind)
    for ev in day_events:
        if ev.kind.startswith("shift"):
            zone_start = (ev.start - pd.Timedelta(days=shift_zone_days)).normalize()
            zone_end = (ev.start + pd.Timedelta(days=shift_zone_days) + pd.Timedelta(hours=23, minutes=45))
            mask = (idx >= zone_start) & (idx <= zone_end)
        else:
            zone_start = ev.start.normalize()
            zone_end = ev.end.normalize() + pd.Timedelta(hours=23, minutes=45)
            mask = (idx >= zone_start) & (idx <= zone_end)
        _mark_mask(pd.Series(mask, index=idx), ev.kind)

    labels["lbl_event"] = labels[
        ["lbl_shift_up", "lbl_shift_down", "lbl_stress_up", "lbl_stress_down",
         "lbl_burst_up", "lbl_burst_down", "lbl_spike_up", "lbl_spike_down"]
    ].max(axis=1)
    labels["lbl_event_type"] = point_type
    return labels