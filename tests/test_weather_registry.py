"""天气 Registry 与传递输入身份；本地合成 fixture。"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from data_loading import BUILTIN_GENERATORS, InformationSetRequest, SourceRegistry
from data_loading.sources.provenance import source_hashes, generator_hashes
from forecasting_core.specs.config import parse_data_spec
from test_weather_assets import asset_fixture
from test_weather_spec import weather_options, weather_data as weather_payload


def weather_data(root):
    reference, _ = asset_fixture(root)
    options = weather_options()
    options['inputs'] = [reference]
    options['temporal']['timezone'] = 'UTC'
    options['temporal']['freq'] = '1h'
    payload = weather_payload()
    payload['sources'][1]['generator_options'] = options
    payload['sources'][0]['history_path'] = 'target.csv'
    pd.DataFrame({'time':['2026-01-01T00:00Z'],'load':[1.]}).to_csv(root / 'target.csv',index=False)
    return parse_data_spec(payload, 'fixture')


class WeatherRegistryTest(unittest.TestCase):
    def test_registry_base_dir_binding_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = weather_data(root)
            registry = SourceRegistry(data, root, generators=BUILTIN_GENERATORS)
            request = InformationSetRequest('2026-01-01T00:00Z', pd.date_range('2026-01-01T01:00Z',periods=2,freq='1h'), ())
            info = registry.materialize(request)
            self.assertEqual(info.known_future['weather'].temperature.tolist(), [10.,12.])
            proof = json.loads(info.lineage[-1].weather_evidence)[0]
            self.assertEqual(proof['snapshot_id'],'run-1')
            self.assertEqual(len(proof['dependencies']), 4)
            returned = info.known_future['weather']
            returned.loc[0,'temperature'] = 999.
            self.assertEqual(registry.materialize(request).known_future['weather'].temperature.tolist(), [10.,12.])
            hashes = source_hashes(data, root)
            self.assertTrue(any('raw.csv' in key for key in hashes))
            self.assertTrue(generator_hashes(data,registry.generators)['weather'])
            (root / 'raw.csv').write_text('changed')
            with self.assertRaises(ValueError):
                registry.materialize(request)
            with self.assertRaises(ValueError):
                source_hashes(data, root)


if __name__ == '__main__':
    unittest.main()
