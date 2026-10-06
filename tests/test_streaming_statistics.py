"""状态算子的独立 pandas 对照，不放宽既有数值容差。"""
import pickle
import unittest

import numpy as np
import pandas as pd

from feature_engineering.statistics.streaming import EwmState, EventState, StreamingStatistics
from feature_engineering.kernels.history import time_since_event


class StreamingStatisticsTest(unittest.TestCase):
    def test_ewm_exact_prefix_equivalence_and_restore(self):
        rng = np.random.default_rng(991)
        for values in (rng.normal(size=70), np.full(70, 3.0), 1e12 + rng.normal(size=70)):
            for halflife in (0.25, 2.0, 5.7):
                state = EwmState(halflife)
                for i, value in enumerate(values):
                    state.update(float(value))
                    expected = pd.Series(values[:i + 1]).ewm(halflife=halflife, adjust=True)
                    self.assertEqual(state.mean(), expected.mean().iloc[-1])
                    if i:
                        self.assertEqual(state.std(), expected.std().iloc[-1])
                    else:
                        with self.assertRaises(ValueError):
                            state.std()
                    if i == 25:
                        state = pickle.loads(pickle.dumps(state))

    def test_event_confirmation_and_restore(self):
        values = [2.0, 1.0, 2.0, 2.0, 1.0, 5.0, 4.0, 7.0, 5.0]
        state = EventState()
        for i, value in enumerate(values):
            state.update(value)
            for event in ("peak", "trough"):
                self.assertEqual(state.value(event), time_since_event(pd.Series(values[:i + 1]), event))
            self.assertLessEqual(len(state.tail), 2)
            state = pickle.loads(pickle.dumps(state))

    def test_invalid_updates_leave_state_unchanged(self):
        for state in (EwmState(2.0), EventState()):
            state.update(1.0)
            before = pickle.dumps(state)
            for invalid in (float("nan"), float("inf")):
                with self.assertRaises(ValueError):
                    state.update(invalid)
                self.assertEqual(pickle.dumps(state), before)
    def test_restore_rejects_invalid_numeric_state(self):
        advanced = {"ewm": {"columns": ["y"], "halflives": [2.0]},
                    "time_since": {"columns": ["y"]},
                    "expanding": {"columns": ["y"], "stats": ["mean", "min", "max"]}}
        expected = StreamingStatistics(advanced, config_fingerprint="test", time_col="time", freq="h")
        frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=4, freq="h"), "y": [1., 3., 2., 4.]})
        state = expected.updated(frame, origin=frame.time.iloc[-1])
        state.validate_snapshot(expected, state.origin)
        for kind, attribute, value in (("ewm", "weighted_mean", float("nan")),
                                       ("ewm", "covariance", -1.),
                                       ("events", "last_peak", 999),
                                       ("expanding", "minimum", 100.),
                                       ("expanding", "minimum", -123456.)):
            corrupted = pickle.loads(pickle.dumps(state))
            mapping = getattr(corrupted, kind)
            setattr(next(iter(mapping.values())), attribute, value)
            with self.subTest(kind=kind, attribute=attribute, value=value), self.assertRaisesRegex(ValueError, "snapshot"):
                corrupted.validate_snapshot(expected, state.origin)


if __name__ == "__main__":
    unittest.main()
