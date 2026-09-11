"""因果特征内核黄金测试：同槽与近期状态严格 as-of，短历史 RAISE。

黄金值手算独立于实现（numpy 直算），不委托实现复算。
"""
import unittest

import numpy as np
import pandas as pd


class SeasonalKernelGoldTest(unittest.TestCase):
    def test_history_only_anchor_and_every_slot_visibility(self):
        from feature_engineering.seasonal import same_slot_stats
        times = pd.date_range("2026-08-01", periods=48, freq="1h", tz="Asia/Shanghai")
        history = pd.Series(np.arange(48, dtype=float), index=times)
        anchor = times[-1] + pd.Timedelta(hours=13)
        result = same_slot_stats(history, anchor=anchor, origin=times[-1], period=24, days=(2,))
        self.assertEqual(result[2]["mean"], 24.0)
        extended = pd.Series(np.arange(96, dtype=float), index=pd.date_range(times[0], periods=96, freq="1h"))
        with self.assertRaises(ValueError):
            same_slot_stats(extended, anchor=times[-1] + pd.Timedelta(hours=25),
                            origin=times[-1], period=24, days=(2,))
        with self.assertRaises(ValueError):
            same_slot_stats(history.drop(times[5]), anchor=anchor, origin=times[-1], period=24, days=(2,))

    def test_same_slot_gold_and_causality(self):
        from feature_engineering.seasonal import same_slot_stats

        times = pd.date_range("2026-08-01", periods=96, freq="1h")
        values = pd.Series(np.arange(96, dtype=float), index=times)
        origin = times[47]
        # 锚点 08-03 12:00（index 60）；1 天同槽 = index 36，2 天 = index 12
        stats = same_slot_stats(values, anchor=times[60], origin=origin, period=24, days=(1, 2))
        self.assertEqual(stats[1]["mean"], 36.0)
        self.assertEqual(stats[2]["mean"], float(np.mean([36.0, 12.0])))
        self.assertAlmostEqual(stats[2]["std"], float(np.std([36.0, 12.0], ddof=1)))
        # 未来扰动不影响特征：改 origin 之后的值结果不变
        perturbed = values.copy()
        perturbed.iloc[48:] += 1000.0
        self.assertEqual(
            same_slot_stats(perturbed, anchor=times[60], origin=origin, period=24, days=(1, 2)),
            stats,
        )
        # as-of：槽位本身在 origin 之后（index 60 > 47）时 RAISE
        with self.assertRaises(ValueError):
            same_slot_stats(values, anchor=times[72], origin=origin, period=24, days=(1,))
        # 槽位越出历史起点 RAISE
        with self.assertRaises(ValueError):
            same_slot_stats(values, anchor=times[12], origin=origin, period=24, days=(1,))

    def test_recent_state_gold_and_causality(self):
        from feature_engineering.seasonal import recent_state_stats

        times = pd.date_range("2026-08-01", periods=48, freq="1h")
        values = pd.Series(np.arange(48, dtype=float), index=times)
        stats = recent_state_stats(values, origin=times[30], windows=(6, 12))
        # 窗口 = 含原点的 N 个点 [origin-N+1, origin]：6 → 25..30（5min 下 6 点 = 30min）
        self.assertEqual(stats[6]["level"], 30.0)
        self.assertEqual(stats[6]["mean"], float(np.mean(np.arange(25, 31))))
        self.assertEqual(stats[6]["diff"], 30.0 - 25.0)
        self.assertAlmostEqual(stats[6]["slope"], (30.0 - 25.0) / 5.0)
        self.assertAlmostEqual(stats[6]["std"], float(np.std(np.arange(25, 31), ddof=1)))
        perturbed = values.copy()
        perturbed.iloc[31:] += 500.0
        self.assertEqual(recent_state_stats(perturbed, origin=times[30], windows=(6, 12)), stats)
        with self.assertRaises(ValueError):
            recent_state_stats(values, origin=times[3], windows=(6,))

    def test_seasonal_baseline_gold(self):
        from feature_engineering.seasonal import seasonal_baseline_values

        times = pd.date_range("2026-08-01", periods=96, freq="1h")
        values = pd.Series(np.arange(96, dtype=float), index=times)
        # 原点 08-02 23:00（index 47），标签 48..71；1 天同槽 = 24..47
        baseline = seasonal_baseline_values(values, origin=times[47], horizon=24, period=24, days=1)
        np.testing.assert_allclose(baseline, np.arange(24.0, 48.0))
        # 2 天同槽 = mean(24..47, 0..23) = h+11（h=1..24）
        baseline2 = seasonal_baseline_values(values, origin=times[47], horizon=24, period=24, days=2)
        np.testing.assert_allclose(baseline2, np.arange(1, 25) + 11.0)
        # 标签槽的 k 天历史越过历史起点时 RAISE
        with self.assertRaises(ValueError):
            seasonal_baseline_values(values, origin=times[23], horizon=24, period=24, days=2)

    def test_specs_reject_unknown_fields(self):
        from feature_engineering.seasonal import (
            normalize_seasonal_baseline_spec,
            normalize_same_slot_spec,
            normalize_recent_state_spec,
        )
        for bad in ({"column": "value", "period": 288, "days": [3], "extra": 1},
                    {"column": "value", "period": 0, "days": [3]},
                    {"column": "value", "period": 288, "days": []},
                    {"period": 288, "days": [3]},
                    {"column": "value", "period": 288.5, "days": [3]}):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                normalize_seasonal_baseline_spec(bad)
        for bad in ({"columns": ["value"], "period": 288, "days": [3], "stats": ["mean"], "x": 1},
                    {"columns": [], "period": 288, "days": [3], "stats": ["mean"]},
                    {"columns": ["value"], "period": 288, "days": [3], "stats": ["median"]}):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                normalize_same_slot_spec(bad)
        for bad in ({"columns": ["value"], "windows": [6], "stats": ["level", "x"]},
                    {"columns": ["value"], "windows": [], "stats": ["level"]}):
            with self.subTest(bad=bad), self.assertRaises((ValueError, TypeError)):
                normalize_recent_state_spec(bad)
        self.assertEqual(
            normalize_same_slot_spec({"columns": ["value"], "period": 288, "days": [3, 7], "stats": ["mean", "std"]}),
            {"columns": ("value",), "period": 288, "days": (3, 7), "stats": ("mean", "std")},
        )


if __name__ == "__main__":
    unittest.main()
