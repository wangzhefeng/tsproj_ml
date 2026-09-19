"""红框点位修正的过去观测合同；不训练模型。"""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / 'config/aidc_hvac_load_5min/scripts'
sys.path.insert(0, str(SCRIPTS))
from clean_hvac_redboxes import detect_candidates, fill_candidates

RECIPE = {'baseline_points': 12, 'minimum_baseline_points': 8,
          'relative_threshold': 0.1, 'absolute_threshold_kw': 1., 'mad_multiplier': 6.}


class RedboxCleaningTest(unittest.TestCase):
    def series(self):
        return pd.Series(100., index=pd.date_range('2026-08-01', periods=6 * 288, freq='5min', name='time'), name='point')

    def test_mask_is_point_specific_past_based_and_envelope_limited(self):
        s = self.series()
        s.iloc[1000:1010] = 300.
        s.iloc[1100] = 400.
        events = [{'start': str(s.index[1000]), 'end': str(s.index[1011])}]
        mask, audit = detect_candidates(s, events, RECIPE)
        self.assertEqual(list(np.flatnonzero(mask)), list(range(1000, 1010)))
        changed = s.copy()
        changed.iloc[1012:] = 10000.
        other, _ = detect_candidates(changed, events, RECIPE)
        pd.testing.assert_series_equal(mask, other)
        result, rows = fill_candidates(s, mask)
        self.assertTrue(result.iloc[1000:1010].eq(100.).all())
        self.assertEqual(result.iloc[1100], 400.)
        self.assertTrue(all(r['status'] == 'filled' for r in rows))
        self.assertTrue(audit)

    def test_fixed_mask_future_observation_changes_neither_value_nor_selection(self):
        s = self.series()
        mask = pd.Series(False, index=s.index)
        mask.iloc[1000:1020] = True
        s.iloc[1000:1020] = 500.
        result, rows = fill_candidates(s, mask)
        changed = s.copy()
        changed.iloc[1000:] = 9000.
        after, changed_rows = fill_candidates(changed, mask)
        np.testing.assert_array_equal(result[mask], after[mask])
        for before, other in zip(rows, changed_rows):
            for key in ('method', 'validation_count', 'validation_start', 'validation_end',
                        'raw_dependency_start', 'raw_dependency_end', 'new_value'):
                self.assertEqual(before[key], other[key])
            self.assertLess(pd.Timestamp(before['raw_dependency_end']), s.index[1000])

    def test_insufficient_history_and_long_segments_are_retained_not_forced(self):
        s = self.series()
        for start, length in ((5, 3), (1000, 73)):
            mask = pd.Series(False, index=s.index)
            mask.iloc[start:start + length] = True
            changed = s.copy()
            changed[mask] = 300.
            result, rows = fill_candidates(changed, mask)
            pd.testing.assert_series_equal(result, changed)
            self.assertTrue(all(r['status'].startswith('unfilled') for r in rows))

    def test_existing_missing_slot_inside_event_recomputed_without_bad_context(self):
        s = self.series()
        s.iloc[1000:1010] = 300.
        s.iloc[1005] = np.nan
        mask = pd.Series(False, index=s.index)
        mask.iloc[1000:1010] = True
        result, rows = fill_candidates(s, mask)
        self.assertTrue(result.iloc[1000:1010].eq(100.).all())
        missing = next(row for row in rows if row['time'] == str(s.index[1005]))
        self.assertTrue(np.isnan(missing['old_raw_value']))
        self.assertEqual(missing['new_value'], 100.)

    def test_prior_selected_anomalies_never_become_reference_observations(self):
        s = self.series()
        s.iloc[1000:1010] = 300.
        s.iloc[1012:1020] = 300.
        events = [{'start': str(s.index[1000]), 'end': str(s.index[1009])},
                  {'start': str(s.index[1012]), 'end': str(s.index[1019])}]
        mask, audit = detect_candidates(s, events, RECIPE)
        # 第二段过去1小时只剩两个可信观测，不借用第一段异常或其补值。
        self.assertTrue(mask.iloc[1000:1010].all())
        self.assertFalse(mask.iloc[1012:1020].any())
        self.assertIn('insufficient_reference', [r['status'] for r in audit])


if __name__ == '__main__':
    unittest.main()
