"""六项天气配置、生成器与真实资产的训练/推理列映射回归。"""
from pathlib import Path
import unittest
import tempfile
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from data_loading import InformationSetRequest, SourceRegistry
from forecasting_core.specs.config import parse_data_spec
from scripts import generate_load_15min_matrix as matrix
from scripts import build_scenario_weather as weather_builder

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ('aidc_load_15min_daily', 'aidc_load_15min_rolling',
            'aidc_load_15min_short', 'aidc_ess_selfuse_load')
MAPPING = {'rt_tt2': 'pred_tt2', 'cal_rh': 'pred_rh', 'rt_ssr': 'pred_ssrd',
           'rt_ws10': 'pred_ws10', 'rt_ps': 'pred_ps', 'rt_rain': 'pred_rain'}


class WeatherSixFeaturesTest(unittest.TestCase):
    def test_rebuild_preserves_source_repair_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.csv'
            source.write_text('ts,rt_ps\n2026-01-01,102880\n')
            repair = source.with_suffix('.six_features_repair.json')
            repair.write_text(json.dumps({'source_sha256_after': hashlib.sha256(source.read_bytes()).hexdigest(),
                                          'semantic_status': 'reanalysis substitution'}))
            with patch.object(weather_builder, 'ROOT', root), patch.object(weather_builder, 'SOURCE', source), patch.object(weather_builder, 'SOURCE_REPAIR_REPORT', root / 'absent.json'):
                dest = root / 'history.csv'
                weather_builder.publish(pd.read_csv(source), dest, {'role': 'history'})
                meta = json.loads(dest.with_suffix('.meta.json').read_text())
                self.assertEqual(meta['source_repairs'][0]['report'], repair.name)
                self.assertEqual(meta['source_repairs'][0]['sha256'], hashlib.sha256(repair.read_bytes()).hexdigest())

    def test_matrix_weather_source_keeps_six_and_historical_mapping(self):
        for family in FAMILIES[:3]:
            source = matrix._weather_source(family)
            self.assertEqual({c['name'] for c in source['columns'] if c['role'] == 'known_future'}, set(MAPPING))
            self.assertEqual(source['inference_columns'], MAPPING)
            self.assertEqual(source['availability'], 'forecast_origin')
            self.assertNotIn('future_path', source)
            self.assertNotIn('backtest_path', source)
            self.assertEqual(source['history_path'], f'dataset/{family}/weather_history_15min_20250101_20260831.csv')

    def test_all_weather_configs_and_real_asset_mapping(self):
        sources = {}
        count = 0
        for family in FAMILIES:
            for path in (ROOT / 'config' / family).rglob('*.yaml'):
                payload = yaml.safe_load(path.read_text())
                if not isinstance(payload, dict):
                    continue
                for source in payload.get('data', {}).get('sources', []):
                    if source['name'] != 'weather':
                        continue
                    count += 1
                    with self.subTest(path=str(path)):
                        self.assertEqual(source['inference_columns'], MAPPING)
                        self.assertEqual({c['name'] for c in source['columns'] if c['role'] == 'known_future'}, set(MAPPING))
                        load_yaml_config(path)
                    sources[source['history_path']] = source
        self.assertGreater(count, 0)
        for filename, source in sources.items():
            frame = pd.read_csv(ROOT / filename)
            columns = list(MAPPING) + list(MAPPING.values())
            self.assertTrue(np.isfinite(frame[columns].to_numpy(dtype=float)).all(), filename)
            frame.index = pd.to_datetime(frame[source['time_col']])
            # 覆盖此次 ERA5 修补的三段原生缺口。
            for timestamp in ('2026-01-01 10:00', '2026-02-03 20:00', '2026-08-04 21:00'):
                times = pd.DatetimeIndex([timestamp])
                with tempfile.TemporaryDirectory() as directory:
                    target = Path(directory) / 'target.csv'
                    pd.DataFrame({'ts': [times[0] - pd.Timedelta(hours=1), times[0]], 'value': [1., 1.]}).to_csv(target, index=False)
                    target_source = {'name': 'target', 'source_type': 'file', 'time_col': 'ts',
                                     'history_path': str(target), 'availability': 'source_time',
                                     'columns': [{'name': 'value', 'role': 'target', 'categorical': False}]}
                    registry = SourceRegistry(parse_data_spec({'sources': [target_source, source]}, 'weather-six'), ROOT)
                    for training in (True, False):
                        request = InformationSetRequest(times[0] - pd.Timedelta(hours=1), times, (),
                                                        target_access='supervised_labels' if training else 'history_only')
                        info = registry.materialize(request)
                        expected_cols = list(MAPPING) if training else list(MAPPING.values())
                        np.testing.assert_allclose(info.known_future['weather'][list(MAPPING)].to_numpy(), frame.loc[times, expected_cols].to_numpy())


if __name__ == '__main__':
    unittest.main()
