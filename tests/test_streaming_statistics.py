"""状态算子的独立 pandas 对照，不放宽既有数值容差。"""
import pickle
import unittest

import numpy as np
import pandas as pd

from feature_engineering.streaming_statistics import EwmState, EventState
from feature_engineering.history_statistics import time_since_event


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


if __name__ == "__main__":
    unittest.main()
