"""inference_columns：训练读 rt_ 实测、推理读 pred_ 预报的阶段列族映射（方案 B，2026-09-07 裁决）。"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_loading import InformationSetRequest, SourceRegistry
from forecasting_core.specs.config import parse_data_spec, parse_model_config


def _payload(**overrides):
    source = {
        'name': 'weather', 'source_type': 'file',
        'columns': [
            {'name': 'rt_tt2', 'role': 'known_future', 'categorical': False},
            {'name': 'pred_tt2', 'role': 'ignored', 'categorical': False},
        ],
        'history_path': 'weather_history.csv', 'future_path': 'weather_future.csv',
        'time_col': 'ts', 'series_id_cols': [], 'availability': 'forecast_origin',
        'inference_columns': {'rt_tt2': 'pred_tt2'},
    }
    source.update(overrides.pop('source_overrides', {}))
    payload = {
        'sources': [
            {'name': 'target_history', 'source_type': 'file',
             'columns': [{'name': 'value', 'role': 'target', 'categorical': False}],
             'history_path': 'target.csv', 'time_col': 'time', 'series_id_cols': [],
             'availability': 'source_time'},
            source,
        ]
    }
    payload.update(overrides)
    return payload


def _write_files(root):
    pd.DataFrame({'time': ['2026-01-01T00:00Z'], 'value': [1.]}).to_csv(root / 'target.csv', index=False)
    pd.DataFrame({'ts': ['2026-01-01T01:00:00Z', '2026-01-01T02:00:00Z'],
                  'rt_tt2': [280.0, 281.0], 'pred_tt2': [275.0, 276.5]}).to_csv(root / 'weather_history.csv', index=False)
    pd.DataFrame({'ts': ['2026-02-01T00:00:00Z'],
                  'rt_tt2': [283.0], 'pred_tt2': [277.0]}).to_csv(root / 'weather_future.csv', index=False)


class InferenceColumnsSpecTest(unittest.TestCase):
    def test_parse_roundtrip_and_fingerprint_semantics(self):
        from forecasting_core.specs.config import ForecastConfigSpec  # noqa: F401
        data = parse_data_spec(_payload()['sources'] and {'sources': _payload()['sources']}, 'fixture')
        source = data.sources[1]
        self.assertEqual(dict(source.inference_columns), {'rt_tt2': 'pred_tt2'})
        self.assertIn('inference_columns', data.canonical_payload()['sources'][1])
        without = _payload()
        del without['sources'][1]['inference_columns']
        self.assertNotEqual(
            data.canonical_payload(),
            parse_data_spec({'sources': without['sources']}, 'fixture').canonical_payload(),
        )

    def test_invalid_mappings_raise(self):
        base = _payload()['sources'][1]
        for bad in (
            {'rt_tt2': 'pred_tt2', 'unknown_kf': 'pred_tt2'},      # key 不是声明的 known_future
            {'rt_tt2': 'rt_undeclared'},                            # value 未声明为 ignored
            {'rt_tt2': 'pred_tt2', 'pred_tt2': 'pred_tt2'},         # key 与 value 冲突
        ):
            with self.subTest(bad=bad):
                source = dict(base)
                source['inference_columns'] = bad
                with self.assertRaises((ValueError, TypeError)):
                    parse_data_spec({'sources': [_payload()['sources'][0], source]}, 'fixture')

    def test_forbidden_on_generated_and_non_known_future(self):
        target_source = dict(_payload()['sources'][0], inference_columns={'value': 'x'})
        with self.assertRaises(ValueError):
            parse_data_spec({'sources': [target_source, _payload()['sources'][1]]}, 'fixture')


class InferenceColumnsRegistryTest(unittest.TestCase):
    def test_training_reads_actual_inference_reads_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_files(root)
            data = parse_data_spec({'sources': _payload()['sources']}, 'fixture')
            registry = SourceRegistry(data, root)
            times = pd.DatetimeIndex(['2026-01-01T01:00Z', '2026-01-01T02:00Z'])
            origin = pd.Timestamp('2026-01-01T00:00Z')
            training = registry.materialize(InformationSetRequest(origin, times, (), target_access='supervised_labels'))
            inference = registry.materialize(InformationSetRequest(origin, times, ()))
            self.assertEqual(training.known_future['weather']['rt_tt2'].tolist(), [280.0, 281.0])
            self.assertEqual(inference.known_future['weather']['rt_tt2'].tolist(), [275.0, 276.5])

    def test_inference_nan_at_requested_times_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_files(root)
            frame = pd.read_csv(root / 'weather_history.csv')
            frame.loc[1, 'pred_tt2'] = None
            frame.to_csv(root / 'weather_history.csv', index=False)
            data = parse_data_spec({'sources': _payload()['sources']}, 'fixture')
            registry = SourceRegistry(data, root)
            times = pd.DatetimeIndex(['2026-01-01T01:00Z', '2026-01-01T02:00Z'])
            origin = pd.Timestamp('2026-01-01T00:00Z')
            with self.assertRaisesRegex(ValueError, 'inference'):
                registry.materialize(InformationSetRequest(origin, times, ()))
            training = registry.materialize(InformationSetRequest(origin, times, (), target_access='supervised_labels'))
            self.assertEqual(training.known_future['weather']['rt_tt2'].tolist(), [280.0, 281.0])


if __name__ == '__main__':
    unittest.main()
