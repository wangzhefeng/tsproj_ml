"""真实跨楼数据独立回读；不调用生产生成器、不重算预测资产。"""
from pathlib import Path
import json
import hashlib
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1] / 'dataset/aidc_hvac_load_5min'

def read(path):
    return pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class BuildingExtensionAssetsTest(unittest.TestCase):
    def test_sources_scope_masks_totals_and_frozen_forecast(self):
        manifest = json.loads((ROOT / 'analysis/building_extension/manifest.json').read_text())
        corrections = json.loads((ROOT / manifest['event_corrections']).read_text())
        migration = json.loads((ROOT / 'analysis/single_version_migration.json').read_text())
        for mapping in (manifest['source_sha256'], manifest['output_sha256'], corrections['output_sha256'],
                        migration['preserved_forecast_sha256'], migration['preserved_a1_points_sha256']):
            for path, digest in mapping.items():
                self.assertEqual(sha(ROOT / path), digest, path)
        for key, summary in manifest['tables'].items():
            raw = read(ROOT / 'raw_data' / key)
            filled = read(ROOT / 'imputed_data' / key)
            adjusted = read(ROOT / 'outlier_remove_data' / key)
            pd.testing.assert_index_equal(filled.index, raw.index)
            points = raw.columns.drop('total_load')
            observed = raw[points].notna().to_numpy()
            np.testing.assert_array_equal(filled[points].to_numpy()[observed], raw[points].to_numpy()[observed])
            np.testing.assert_allclose(filled.total_load, filled[points].sum(axis=1, min_count=len(points)), equal_nan=True)
            np.testing.assert_allclose(adjusted.total_load, adjusted[points].sum(axis=1, min_count=len(points)), equal_nan=True)
            self.assertEqual(int(filled[points].isna().sum().sum()), summary['unresolved_cells'])
            building = Path(key).stem.split('_')[0]
            scope = np.zeros(len(filled), dtype=bool)
            if not key.startswith('IT_load/'):
                for event in corrections['recipe']['events'][building]:
                    scope |= (filled.index >= event['start']) & (filled.index < event['end_exclusive'])
            pd.testing.assert_frame_equal(adjusted.loc[~scope], filled.loc[~scope])
            if key.startswith('hvac_remove_devices/'):
                whole = read(ROOT / 'outlier_remove_data' / key.replace('hvac_remove_devices', 'hvac_all_devices'))
                pd.testing.assert_frame_equal(adjusted[points], whole[points])
        families = [f'{d}/{r}' for d in ('hvac_all_devices', 'hvac_remove_devices') for r in ('route_A', 'route_B')] + ['IT_load']
        for category in ('imputed_data', 'outlier_remove_data'):
            for family in families:
                total = read(ROOT / category / family / 'data.csv')
                raw = read(ROOT / 'raw_data' / family / 'data.csv')
                pd.testing.assert_index_equal(total.index, raw.index)
                expected = pd.concat([read(ROOT / category / family / f'{b}_data.csv').drop(columns='total_load').add_prefix(b + '_').reindex(raw.index) for b in ('A1', 'A2', 'A3')], axis=1)
                pd.testing.assert_frame_equal(total.drop(columns='total_load'), expected)
                np.testing.assert_allclose(total.total_load, expected.sum(axis=1, min_count=len(expected.columns)), equal_nan=True)

if __name__ == '__main__':
    unittest.main()
