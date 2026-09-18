"""预测输入异常候选可视化：只标记不清洗。"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'config/aidc_hvac_load_5min/scripts'))
import analyze_forecast_data as visual


class ForecastVisualTest(unittest.TestCase):
    def test_file_visuals_cover_components_and_preserve_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path('hvac_all_devices/route_A/data_with_it.csv')
            source = root / 'forecast_data' / relative
            source.parent.mkdir(parents=True)
            idx = pd.date_range('2026-08-01', periods=288 * 2, freq='5min', name='time')
            frame = pd.DataFrame(index=idx)
            for kind in ['hvac', 'it']:
                for n, b in enumerate(['A1', 'A2', 'A3'], 1):
                    frame[f'{b}_{kind}_total_load'] = n * 100 + np.sin(np.arange(len(idx)))
                frame[kind + '_total_load'] = frame[[f'{b}_{kind}_total_load' for b in ['A1', 'A2', 'A3']]].sum(axis=1)
            frame.to_csv(source)
            before = source.read_bytes()
            for family in ['hvac_all_devices/route_A', 'IT_load']:
                folder = root / 'analysis/imputation/masks' / family
                folder.mkdir(parents=True)
                for name in ['A1_data.csv', 'A2_data.csv', 'A3_data.csv', 'data.csv']:
                    mask = pd.DataFrame({'total_observed': True}, index=idx)
                    if family != 'IT_load' and name in ['A2_data.csv', 'data.csv']:
                        mask.iloc[10, 0] = False
                    mask.to_csv(folder / name)
            dest = root / 'visual'
            summary, quality = visual.analyze_file(root, relative, dest)
            self.assertEqual(len(summary), 8)
            self.assertEqual(quality['component_mismatch_rows'], 0)
            counts = {row['column']: row['n_imputed'] for row in summary}
            self.assertEqual(counts['A1_hvac_total_load'], 0)
            self.assertEqual(counts['A2_hvac_total_load'], 1)
            self.assertEqual(counts['hvac_total_load'], 1)
            self.assertEqual(counts['it_total_load'], 0)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(len(list(dest.glob('*.png'))), 3)
            for path in dest.glob('*.png'):
                self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue((dest / 'candidate_points.csv').exists())
            self.assertTrue((dest / 'candidate_segments.csv').exists())

    def test_invalid_time_grid_rejected_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            relative = Path('hvac_all_devices/route_A/A1_data.csv')
            source = root / 'forecast_data' / relative
            source.parent.mkdir(parents=True)
            source.write_text('time,hvac_total_load\n2026-08-01,1\n2026-08-01,2\n')
            with self.assertRaises(ValueError):
                visual.analyze_file(root, relative, root / 'visual')
            self.assertFalse((root / 'visual').exists())

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'analysis/forecast_data_visual'
            folder.mkdir(parents=True)
            kept = folder / 'keep.txt'
            kept.write_text('untouched')
            with self.assertRaises(FileExistsError):
                visual.run(root)
            self.assertEqual(kept.read_text(), 'untouched')

    def test_spike_low_constant_and_provenance_do_not_change_input(self):
        idx = pd.date_range('2026-08-01', periods=288 * 3, freq='5min', name='time')
        y = pd.Series(100.0, index=idx, name='hvac_total_load')
        y.iloc[100] = 300
        y.iloc[200] = 1
        before = y.copy()
        observed = pd.Series(True, index=idx)
        observed.iloc[100] = False
        detail, summary, segments = visual.analyze_series(y, observed)
        pd.testing.assert_series_equal(y, before)
        self.assertTrue(detail.iloc[100].spike)
        self.assertTrue(detail.iloc[100].jump)
        self.assertFalse(detail.iloc[100].observed)
        self.assertFalse(detail.iloc[101].transition_observed)
        self.assertTrue(detail.iloc[200].low_load)
        self.assertEqual(summary['n_imputed'], 1)
        self.assertGreater(summary['constant_points'], 0)
        self.assertTrue(any(x['kind'] == 'constant' for x in segments))

    def test_small_fluctuations_are_not_spikes_or_jumps(self):
        idx = pd.date_range('2026-08-01', periods=288, freq='5min', name='time')
        y = pd.Series(100 + np.sin(np.arange(288)) * 0.2, index=idx)
        detail, _, _ = visual.analyze_series(y, pd.Series(True, index=idx))
        self.assertFalse(detail.spike.any())
        self.assertFalse(detail.jump.any())


if __name__ == '__main__':
    unittest.main()
