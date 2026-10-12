"""套组分路的独立数值与来源合同，不运行模型。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.raw_data.equipment_mapping import load_mapping, SHEET
from config.aidc_hvac_load_5min.scripts.raw_data.regroup_hvac_equipment import regroup_pair, mask_from_gaps


class EquipmentRegroupTests(unittest.TestCase):
    def test_cross_route_values_and_missingness_preserved(self):
        index = pd.date_range('2026-01-01', periods=2, freq='5min', name='time')
        pair = {'route_A': pd.DataFrame({'chiller': [10., 20.], 'pump': [1., np.nan], 'total_load': [11., 20.]}, index=index),
                'route_B': pd.DataFrame({'tower': [2., 3.], 'total_load': [2., 3.]}, index=index)}
        meta = pd.DataFrame({'spot_id': ['chiller', 'pump', 'tower'], 'group_route': ['B', 'A', 'B']})
        strict = regroup_pair(pair, meta, strict=True)
        self.assertEqual(list(strict['route_B']), ['chiller', 'tower', 'total_load'])
        np.testing.assert_array_equal(strict['route_B'].total_load, [12., 23.])
        self.assertEqual(strict['route_A'].iloc[0].total_load, 1.)
        self.assertTrue(pd.isna(strict['route_A'].iloc[1].total_load))
        np.testing.assert_array_equal(pair['route_A'].chiller, [10., 20.])
        bad = meta.copy()
        bad.loc[0, 'spot_id'] = 'unknown'
        with self.assertRaisesRegex(ValueError, '并集'):
            regroup_pair(pair, bad, strict=True)

    def test_moved_point_changes_only_own_group_mask(self):
        index = pd.date_range('2026-01-01', periods=3, freq='5min', name='time')
        table = pd.DataFrame({'pump': [1., 2., 3.], 'total_load': [1., 2., 3.]}, index=index)
        gaps = pd.DataFrame([{'point': 'pump', 'gap_start': str(index[1]), 'gap_end': str(index[1]),
                              'status': 'filled', 'eligibility_known_at': str(index[2])},
                             {'point': 'other', 'gap_start': str(index[0]), 'gap_end': str(index[2]),
                              'status': 'unfilled_long', 'eligibility_known_at': None}])
        mask = mask_from_gaps(table, gaps)
        self.assertEqual(mask.total_observed.tolist(), [True, False, True])
        self.assertEqual(mask.total_imputed.tolist(), [False, True, False])
        self.assertEqual(mask.eligibility_known_at.tolist(), [index[0], index[2], index[2]])

    def test_real_mapping_has_all_units_and_explicit_exception(self):
        mapping = load_mapping()
        self.assertEqual(mapping.groupby(['building', '东西南北']).ngroups, 18)
        for unit, expected in [('南2', 'B'), ('南3', 'A')]:
            part = mapping[mapping.building.eq('A1') & mapping['东西南北'].eq(unit)]
            self.assertEqual(set(part.group_route), {expected})
            self.assertEqual(set(part.route), {'A', 'B'})
        with TemporaryDirectory() as tmp:
            bad = mapping.copy()
            bad.loc[bad.spot_id.eq('56_11_0_29_1_19_0'), '东西南北'] = '西1'
            path = Path(tmp) / 'bad.xlsx'
            bad.to_excel(path, sheet_name=SHEET, index=False)
            with self.assertRaisesRegex(ValueError, '组成'):
                load_mapping(path)


if __name__ == '__main__':
    unittest.main()
