"""切换特征公开数值边界：确认延迟、缺失隔离与未来扰动不变。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from config.aidc_hvac_load_5min.scripts.v3.forecast_data.switch_features import unit_states, module_events, clean_power
from config.aidc_hvac_load_5min.scripts.v3.forecast_data.prepare_switch import build, verify, target_pairs, sha256


class SwitchFeaturesTest(unittest.TestCase):
    def test_new_outputs_preserve_both_parents_and_response_observations(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = pd.date_range('2026-04-08', periods=9 * 288, freq='5min', name='time')
            units, inputs = [], []
            response = 7 * 288
            for module, prefix, direction in [('north', 'n', '北'), ('south', 's', '南')]:
                for number in (1, 2, 3):
                    code, label = prefix + str(number), direction + str(number)
                    columns = [code + '_' + str(i) for i in range(6)]
                    frame = pd.DataFrame(100., index=index, columns=columns)
                    frame.loc[index[response:response + 12], columns[0]] = .1
                    frame.loc[index[6 * 288:6 * 288 + 3], columns[0]] = np.nan
                    frame.loc[index[8 * 288], columns[1]] = -1.
                    unit = {'code': code, 'label': label, 'module': module, 'route': 'route_A',
                            'points': columns, 'core_points': columns[:5], 'chiller': columns[0]}
                    units.append(unit)
                    for devices, selected in [('hvac_all_devices', columns), ('hvac_remove_devices', columns[:5])]:
                        path = root / f'raw_data/{devices}/route_A/A1_{label}_data.csv'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        frame[selected].to_csv(path)
                        inputs.append(path)
            for source, _ in target_pairs():
                path = root / source
                path.parent.mkdir(parents=True, exist_ok=True)
                value = 9. if 'dr_adjusted' in source.name else 11.
                pd.DataFrame({'target': value, 'hvac_ft_example': 3.25}, index=index).to_csv(path)
                inputs.append(path)
            recipe = {'start': str(index[0]), 'end': str(index[-1]), 'units': units,
                      'state': {'on': 20., 'off': 1., 'confirm': 3, 'max_minutes': 60},
                      'demand_response_events': [{'event_id': 'dr_fixture', 'start': str(index[response]),
                           'rebound_start': str(index[response + 6]), 'end_exclusive': str(index[response + 12])}],
                      'input_sha256': {str(p.relative_to(root)): sha256(p) for p in inputs}}
            report = build(root, recipe)
            self.assertEqual(report['invalid_cells'], 6)
            self.assertEqual(report['filled_cells'], 24)
            self.assertEqual(verify(root)['status'], 'passed')
            for source, destination in target_pairs():
                output = pd.read_csv(root / destination)
                self.assertEqual(output.target.iloc[0], 9. if 'dr_adjusted' in source.name else 11.)
                self.assertFalse(any(c.startswith('lbl_') for c in output))
                self.assertEqual(output.hvac_sw_n1_chiller_kw.iloc[response], .1)
                self.assertEqual(output.hvac_sw_n1_state.iloc[response], 1)
                self.assertEqual(output.hvac_sw_n1_state.iloc[response + 2], 0)
                self.assertEqual(output.hvac_sw_n1_state.iloc[6 * 288], -1)
            with self.assertRaises(FileExistsError):
                build(root, recipe)

    def test_confirmation_missingness_and_prefix_invariance(self):
        index = pd.date_range('2026-01-01', periods=12, freq='5min', name='time')
        power = pd.Series([.1, .1, .1, 100, 100, 100, np.nan, .1, .1, .1, 100, 100], index=index)
        states = unit_states(power, on=20., off=1., confirm=3)
        self.assertEqual(states.state.tolist(), [-1, -1, 0, 0, 0, 1, -1, -1, -1, 0, 0, 0])
        self.assertEqual(states.start.sum(), 1)
        self.assertEqual(states.stop.sum(), 0)
        self.assertTrue(states.loc[index[9], 'left_censored'])
        pd.testing.assert_frame_equal(states.iloc[:8], unit_states(power.iloc[:8], on=20., off=1., confirm=3))

    def test_response_and_real_switch_values_not_cleaned(self):
        index = pd.date_range('2026-01-01', periods=7, freq='5min', name='time')
        raw = pd.DataFrame({'a': [100., 100., .1, 200., 100., -1., np.inf]}, index=index)
        clean, invalid = clean_power(raw)
        np.testing.assert_array_equal(clean.a.iloc[:5], raw.a.iloc[:5])
        self.assertEqual(invalid.a.tolist(), [False, False, False, False, False, True, True])
        self.assertTrue(clean.a.iloc[5:].isna().all())

    def test_replacement_is_confirmed_after_second_edge_not_backdated(self):
        index = pd.date_range('2026-01-01', periods=12, freq='5min', name='time')
        states = pd.DataFrame([[1, 0, 1]] * 3 + [[0, 0, 1]] * 3 + [[0, 1, 1]] * 6,
                              index=index, columns=['n1', 'n2', 'n3'])
        events = module_events(states, pd.Series(True, index=index), max_minutes=60)
        swaps = events[events.kind.eq('replacement')]
        self.assertEqual(len(swaps), 1)
        self.assertEqual(swaps.iloc[0]['confirmed_at'], str(index[6]))
        self.assertEqual(swaps.iloc[0]['from_unit'], 1)
        self.assertEqual(swaps.iloc[0]['to_unit'], 2)
        # 缺口之后不得把恢复数据认成跨缺口切换。
        valid = pd.Series(True, index=index)
        valid.iloc[4] = False
        self.assertFalse(module_events(states, valid, max_minutes=60).kind.eq('replacement').any())


if __name__ == '__main__':
    unittest.main()
