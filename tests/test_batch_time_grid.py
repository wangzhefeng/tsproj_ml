"""Pure CSV timestamp geometry, with no model construction or execution."""
import unittest
import pandas as pd
from model_pipeline.batch_artifacts import validate_time_grid


class BatchTimeGridTest(unittest.TestCase):
    def test_exact_grid(self):
        times = pd.date_range("2026-01-01 14:15", periods=2, freq="15min")
        frame = pd.DataFrame({"time": times.repeat(2)})
        validate_time_grid(frame, times, rows_per_time=2)
        with self.assertRaisesRegex(ValueError, "time grid"):
            validate_time_grid(frame.assign(time=frame.time + pd.Timedelta("1h")), times, rows_per_time=2)

    def test_equal_row_count_cannot_hide_wrong_time_multiplicity(self):
        times = pd.date_range("2026-01-01", periods=2, freq="1h")
        frame = pd.DataFrame({"time": [times[0], times[0], times[0], times[1]]})
        with self.assertRaisesRegex(ValueError, "time grid"):
            validate_time_grid(frame, times, rows_per_time=2)


if __name__ == "__main__":
    unittest.main()
