# -*- coding: utf-8 -*-
"""data_process.load_event_detection（AIDC 负荷事件检测核心）的回归测试。

用合成序列验证三类探测器与标签投影的行为：
日级临时偏移（stress）、持久阶跃（shift）、15min 日内 spike/burst。
"""

import unittest

import numpy as np
import pandas as pd

from data_process.load_event_detection import (
    EventDetectionConfig,
    classify_day_events,
    day_intraday_stats,
    detect_intraday_events,
    detect_short_excursions,
    merge_day_events,
    project_events_to_days,
    project_events_to_points,
    suppress_boundary_artifacts,
    topdown_day_segments,
)


def _synthetic_15min() -> pd.Series:
    """60 天 15min 合成序列：前 20 天 10000，中间 3 天抬升到 10800（短时偏移），
    之后回到 10000，第 45 天 16:00~17:30 一个 +900 平台（burst），
    第 50 天 10:00 单点 +1200（spike）。"""
    idx = pd.date_range("2025-01-01", periods=60 * 96, freq="15min")
    rng = np.random.default_rng(7)
    base = np.full(len(idx), 10000.0) + rng.normal(0, 40, len(idx))
    s = pd.Series(base, index=idx)
    s.loc["2025-01-21":"2025-01-23"] += 800.0     # 3 天短时抬升
    s.loc["2025-02-14 16:00":"2025-02-14 17:15"] += 900.0  # 6 点 burst
    s.loc["2025-02-19 10:00"] += 1200.0           # 单点 spike
    return s


def _synthetic_day_level() -> pd.Series:
    """120 天日水平：0-59 天 10000；60-79 天 11000（临时 20 天）；
    80-99 天 10000；100-119 天 11300（持久阶跃）。"""
    idx = pd.date_range("2025-01-01", periods=120, freq="1D")
    rng = np.random.default_rng(3)
    vals = np.concatenate([
        10000 + rng.normal(0, 80, 60),
        11000 + rng.normal(0, 80, 20),
        10000 + rng.normal(0, 80, 20),
        11300 + rng.normal(0, 80, 20),
    ])
    return pd.Series(vals, index=idx)


class TestTopdownSegments(unittest.TestCase):
    def test_segments_match_synthetic_levels(self):
        segs = topdown_day_segments(_synthetic_day_level())
        levels = [round(s.level, -2) for s in segs]
        self.assertIn(10000.0, levels)
        self.assertIn(11000.0, levels)
        self.assertIn(11300.0, levels)
        # 边界日应落在合成切换日附近（±2 天）
        starts = [s.start for s in segs]
        self.assertTrue(any(abs((d - pd.Timestamp("2025-03-02")).days) <= 2 for d in starts))
        self.assertTrue(any(abs((d - pd.Timestamp("2025-03-22")).days) <= 2 for d in starts))
        self.assertTrue(any(abs((d - pd.Timestamp("2025-04-11")).days) <= 2 for d in starts))


class TestClassifyEvents(unittest.TestCase):
    def test_temporary_vs_persistent(self):
        segs = topdown_day_segments(_synthetic_day_level())
        events = classify_day_events(segs)
        kinds = [e.kind for e in events]
        self.assertIn("stress_up", kinds)   # 60~79 天抬升后回落
        self.assertIn("shift_up", kinds)    # 100 天起持久阶跃
        stress = next(e for e in events if e.kind == "stress_up")
        self.assertLessEqual(abs(stress.amplitude - 1000), 150)
        self.assertLessEqual(stress.n_days, 21)


class TestNetResidualShift(unittest.TestCase):
    def test_partial_revert_emits_net_shift(self):
        """抬升 1000 后只回落 600（残差 400>=300）：应同时得到
        stress_up 事件和回落水平的净 shift_up 事件。"""
        idx = pd.date_range("2025-01-01", periods=90, freq="1D")
        rng = np.random.default_rng(2)
        vals = np.concatenate([
            10000 + rng.normal(0, 60, 30),
            11000 + rng.normal(0, 60, 10),
            10400 + rng.normal(0, 60, 50),
        ])
        segs = topdown_day_segments(pd.Series(vals, index=idx))
        events = classify_day_events(segs)
        kinds = [e.kind for e in events]
        self.assertIn("stress_up", kinds)
        net_shifts = [e for e in events if e.kind == "shift_up" and e.start >= idx[35]]
        self.assertTrue(net_shifts)
        self.assertLessEqual(abs(net_shifts[0].amplitude - 400), 120)


class TestShortExcursions(unittest.TestCase):
    def test_two_day_dip_detected(self):
        idx = pd.date_range("2025-01-01", periods=40, freq="1D")
        rng = np.random.default_rng(11)
        vals = 10000 + rng.normal(0, 60, 40)
        vals[10:12] -= 900  # 2 天下探
        s = pd.Series(vals, index=idx)
        events = detect_short_excursions(s)
        self.assertTrue(events)
        self.assertTrue(all(e.kind == "stress_down" for e in events))
        first = events[0]
        self.assertEqual(first.start, idx[10])
        self.assertGreater(abs(first.amplitude), 500)

    def test_merge_prefers_segment_events(self):
        seg_ev = detect_short_excursions(_synthetic_day_level())
        short_ev = detect_short_excursions(_synthetic_day_level(),
                                           EventDetectionConfig(short_max_days=1))
        merged = merge_day_events(seg_ev, short_ev)
        self.assertLessEqual(len(merged), len(seg_ev) + len(short_ev))
        # 合并后按时间排序
        starts = [e.start for e in merged]
        self.assertEqual(starts, sorted(starts))


class TestIntradayEvents(unittest.TestCase):
    def test_burst_and_spike_detected(self):
        s15 = _synthetic_15min()
        day_events = detect_short_excursions(s15.resample("1D").median())
        events = suppress_boundary_artifacts(detect_intraday_events(s15), day_events)
        kinds = {e.kind for e in events}
        self.assertIn("burst_up", kinds)
        self.assertIn("spike_up", kinds)
        burst = next(e for e in events if e.kind == "burst_up")
        self.assertEqual(burst.start, pd.Timestamp("2025-02-14 16:00:00"))
        spike = next(e for e in events if e.kind == "spike_up")
        self.assertEqual(spike.start, pd.Timestamp("2025-02-19 10:00:00"))
        self.assertGreater(spike.amplitude, 600)

    def test_flat_series_no_events(self):
        idx = pd.date_range("2025-01-01", periods=10 * 96, freq="15min")
        rng = np.random.default_rng(5)
        s = pd.Series(10000 + rng.normal(0, 30, len(idx)), index=idx)
        self.assertEqual(detect_intraday_events(s), [])


class TestProjections(unittest.TestCase):
    def test_day_labels(self):
        s15 = _synthetic_15min()
        day_level = s15.resample("1D").median()
        day_events = detect_short_excursions(day_level)
        intra = suppress_boundary_artifacts(detect_intraday_events(s15), day_events)
        stats = day_intraday_stats(s15)
        labels = project_events_to_days(day_level.index, day_events, intra, stats)
        self.assertIn("lbl_stress_up_day", labels.columns)
        self.assertEqual(labels.loc["2025-01-21", "lbl_stress_up_day"], 1)
        self.assertEqual(labels.loc["2025-01-21", "lbl_event_type"], "stress_up")
        self.assertEqual(labels.loc["2025-02-14", "lbl_burst_up_day"], 1)
        self.assertEqual(labels.loc["2025-02-19", "lbl_spike_up_day"], 1)
        self.assertEqual(labels.loc["2025-01-05", "lbl_event_type"], "normal")

    def test_point_labels(self):
        s15 = _synthetic_15min()
        day_events = detect_short_excursions(s15.resample("1D").median())
        intra = suppress_boundary_artifacts(detect_intraday_events(s15), day_events)
        labels = project_events_to_points(s15.index, day_events, intra)
        self.assertEqual(labels.loc["2025-02-19 10:00:00", "lbl_spike_up"], 1)
        self.assertEqual(labels.loc["2025-02-19 10:00:00", "lbl_event_type"], "spike_up")
        self.assertEqual(labels.loc["2025-02-14 16:30:00", "lbl_burst_up"], 1)
        self.assertEqual(labels.loc["2025-01-22 08:00:00", "lbl_stress_up"], 1)
        self.assertEqual(labels.loc["2025-02-19 10:00:00", "lbl_event"], 1)
        self.assertEqual(labels.loc["2025-01-05 00:15:00", "lbl_event_type"], "normal")


class TestBoundaryArtifactSuppression(unittest.TestCase):
    def test_step_transition_not_reported_as_spike(self):
        s15 = _synthetic_15min()
        day_events = detect_short_excursions(s15.resample("1D").median())
        raw = detect_intraday_events(s15)
        kept = suppress_boundary_artifacts(raw, day_events)
        # 原始检测会把 01-21 00:00 的台阶起点误报为 spike，抑制后应消失
        self.assertTrue(any(e.start == pd.Timestamp("2025-01-21 00:00:00") for e in raw))
        self.assertFalse(any(e.start == pd.Timestamp("2025-01-21 00:00:00") for e in kept))
        # 真实 burst/spike 不受影响
        self.assertTrue(any(e.kind == "burst_up" and e.start == pd.Timestamp("2025-02-14 16:00:00") for e in kept))
        self.assertTrue(any(e.kind == "spike_up" and e.start == pd.Timestamp("2025-02-19 10:00:00") for e in kept))


class TestIntradayStats(unittest.TestCase):
    def test_stats_columns(self):
        stats = day_intraday_stats(_synthetic_15min())
        for col in ("intraday_mean", "intraday_std", "intraday_range",
                    "intraday_max_abs_step", "intraday_peak_time_frac"):
            self.assertIn(col, stats.columns)
        self.assertAlmostEqual(stats.loc["2025-01-22", "intraday_mean"], 10800, delta=60)
        self.assertAlmostEqual(stats.loc["2025-02-19", "intraday_max_abs_step"], 1200, delta=200)


class TestVolatileDayIsCausal(unittest.TestCase):
    """lbl_volatile_day 的阈值严格 trailing：行 D 的值只依赖 <=D 的数据。

    截断因果性测试——对样本日 D，用截断到 D 的原始序列重算 volatile 标签，
    行 D 的值必须与全量计算一致（防居中窗口回归）。
    """

    def test_volatile_day_truncated_recompute_matches(self):
        s15 = _synthetic_15min()
        day_level = s15.resample("1D").median()
        day_events = detect_short_excursions(day_level)
        intra = suppress_boundary_artifacts(detect_intraday_events(s15), day_events)
        stats_full = day_intraday_stats(s15)
        labels_full = project_events_to_days(day_level.index, day_events, intra, stats_full)

        # 截断到第 40 天末（保留完整天数，避免残缺日统计差异）
        cut = pd.Timestamp("2025-02-09 23:45:00")
        s_cut = s15.loc[:cut]
        labels_cut = project_events_to_days(
            s_cut.resample("1D").median().index, day_events, [],
            day_intraday_stats(s_cut),
        )
        # volatile 阈值滚动窗口 60 天 min_periods=14，截断后 14 天起可比
        common = labels_full.index.intersection(labels_cut.index)[14:]
        vol_full = labels_full.loc[common, "lbl_volatile_day"]
        vol_cut = labels_cut.loc[common, "lbl_volatile_day"]
        mismatch = (vol_full != vol_cut).sum()
        self.assertEqual(mismatch, 0,
                         f"volatile_day 在截断重算下不一致 {mismatch} 天——阈值窗口引入了未来信息")


class TestCrossFreqDayShift(unittest.TestCase):
    """15min 场景的 xf_day_* 统计特征必须是昨日口径（shift(1)）。

    D 日 00:00 的 15min 点拿到的 xf_day_value 必须 = D-1 日的日均值，
    消除「当天全天统计提前到当天 00:00 可见」的日内未来信息。
    """

    def test_xf_day_value_is_prev_day(self):
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from config.aidc_load_15min_daily.load_event_analysis import (  # noqa: E402
            build_cross_freq_day_features,
            load_series_as_daily,
        )

        # 合成 90 天日频数据（含第 50 天 +900 抬升，用于验证「当天不可见」）
        idx = pd.date_range("2025-01-01", periods=90, freq="1D")
        rng = np.random.default_rng(9)
        vals = 10000 + rng.normal(0, 50, 90)
        vals[50] += 900.0
        tmp_csv = Path(__file__).parent / "_tmp_xf_test_daily.csv"
        pd.DataFrame({"time": idx, "value": vals}).to_csv(tmp_csv, index=False)

        try:
            day_stats, ev_flags = build_cross_freq_day_features(tmp_csv, [])
            # D 日行的 xf_day_value 应等于 D-1 日的值（昨日口径）
            d = idx[51]  # 抬升日的次日：只应看到抬升日当天
            self.assertAlmostEqual(day_stats.loc[d, "xf_day_value"], vals[50], delta=1e-6)
            # 抬升日当天 00:00 的点不应看到当天抬升（应等于 D-1 = 基线值）
            d_spike = idx[50]
            self.assertAlmostEqual(day_stats.loc[d_spike, "xf_day_value"], vals[49], delta=1e-6)
            # 首日无昨日数据，应为 NaN
            self.assertTrue(np.isnan(day_stats.loc[idx[0], "xf_day_value"]))
            # 事件标记列存在且默认值正确
            self.assertIn("lbl_prev_day_event_day", ev_flags.columns)
            self.assertEqual(ev_flags["lbl_prev_day_event_day"].sum(), 0)
            self.assertEqual((ev_flags["lbl_prev_day_event_type"] == "none").all(), True)
        finally:
            tmp_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()