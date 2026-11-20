import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('supplement_raw', ROOT / 'config/aidc_hvac_load_5min/scripts/raw_data/supplement_raw.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RawSupplementTests(unittest.TestCase):
    def test_point_policy(self):
        old = pd.Series([1., np.nan, np.nan])
        got, status, _ = m.supplement_point(old, [pd.Series([1., 2., 3.])], nonnegative=True)
        self.assertEqual(status, 'accepted')
        self.assertEqual(got.tolist(), [1., 2., 3.])
        for candidates, expected in [([pd.Series([9., 2., 3.])], 'overlap_conflict'),
                                     ([pd.Series([1., 2., 3.]), pd.Series([1., 9., 3.])], 'external_conflict'),
                                     ([pd.Series([1., -2., 3.])], 'negative_power')]:
            got, status, _ = m.supplement_point(old, candidates, nonnegative=True)
            pd.testing.assert_series_equal(got, old)
            self.assertEqual(status, expected)

    def test_full_publish_preserves_inputs_and_downstream(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, source, audit = root / 'data', root / 'external', root / 'audit'
            idx = pd.date_range('2025-10-01', periods=2, freq='5min', name='time')
            def put(path, frame):
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(path)
            for family in ['IT_load'] + [f'{d}/{r}' for d in ('hvac_all_devices', 'hvac_remove_devices') for r in ('route_A', 'route_B')]:
                parts = []
                for b in ('A1', 'A2', 'A3'):
                    point = 'it' if family == 'IT_load' else family.split('/')[-1]
                    frame = pd.DataFrame({point: [1., np.nan]}, index=idx)
                    put(data / 'raw_data' / family / f'{b}_data.csv', m.with_total(frame))
                    if family != 'IT_load':
                        put(data / 'raw_data' / family / f'{b}_unit_data.csv', frame)
                    parts.append(frame.add_prefix(b + '_'))
                    put(source / 'route_A' / 'category' / b / f'{point}.csv', pd.DataFrame({'value': [1., 2., 3.]}, index=pd.date_range(idx[0], periods=3, freq='5min', name='time')))
                put(data / 'raw_data' / family / 'data.csv', m.with_total(pd.concat(parts, axis=1)))
            for kind in ('flow', 'temp'):
                put(data / 'raw_data' / 'hvac_flow_temp' / f'data_{kind}.csv', pd.DataFrame({kind: [1., np.nan]}, index=idx))
            for stage in ('imputed_data', 'outlier_remove_data', 'forecast_data'):
                (data / stage).mkdir()
                (data / stage / 'sentinel').write_text('unchanged')
            result = m.build(data, source, audit, end='2025-10-01 00:10', apply=True)
            self.assertEqual(result['status'], 'applied')
            self.assertEqual(result['added_cells'], 18)
            for key, digest in result['after_sha256'].items():
                self.assertEqual(m.sha(data / key), digest)
            for key, digest in result['before_sha256'].items():
                self.assertEqual(m.sha(audit / 'before' / key), digest)
            for key, digest in result['protected_sha256'].items():
                self.assertEqual(m.sha(data / key), digest)
            self.assertEqual(m.read(data / 'raw_data/IT_load/data.csv')['total_load'].tolist(), [3., 6., 9.])
            with self.assertRaises(FileExistsError):
                m.build(data, source, audit)


if __name__ == '__main__':
    unittest.main()
