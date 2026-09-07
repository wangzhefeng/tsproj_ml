"""天气缓存只按显式输入、原点和地点复用；不触碰正式资产。"""
from dataclasses import replace
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from data_loading import InformationSetRequest, SourceRegistry
from data_loading.information.information_set import WeatherSourceLineage
from data_loading.sources.provenance import source_hashes
from forecasting_core.specs.weather import WeatherGenerationSpec
from forecasting_core.specs.config import parse_data_spec
from forecasting_core.specs import EstimatorSpec, FeatureSpec, ForecastConfigSpec, ForecastProblemSpec, ForecastStrategySpec
from feature_engineering import FeatureCompiler
from test_weather_registry import weather_data
from test_weather_spec import weather_data as weather_payload


class WeatherCacheTest(unittest.TestCase):
    def test_global_locations_and_identity_order_do_not_share_cached_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = weather_data(root)
            manifest = json.loads((root / 'manifest.json').read_text())
            second = copy.deepcopy(manifest['snapshots'][0])
            second.update(location_id='second-site', latitude=31.0)
            frame = pd.read_csv(root / 'normalized.csv')
            frame['temperature_2m'] += 100.
            path = root / 'second.csv'
            frame.to_csv(path, index=False)
            second['normalized'] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            manifest['snapshots'].append(second)
            (root / 'manifest.json').write_text(json.dumps(manifest))
            recipe = data.sources[1].generator_options
            assert isinstance(recipe, WeatherGenerationSpec)
            options = recipe.canonical_payload()
            options['inputs'][0]['sha256'] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
            options['location_map'] = [{'series_id': ['A'], 'location_id': 'fixture-site'}, {'series_id': ['B'], 'location_id': 'second-site'}]
            payload = weather_payload()
            for source in payload['sources']:
                source['series_id_cols'] = ['site']
                source['columns'].append({'name': 'site', 'role': 'key'})
            payload['sources'][0]['history_path'] = 'target.csv'
            payload['sources'][1]['generator_options'] = options
            pd.DataFrame({'time': ['2026-01-01T00:00Z'] * 2, 'site': ['A', 'B'], 'load': [1., 2.]}).to_csv(root / 'target.csv', index=False)
            global_data = parse_data_spec(payload, 'fixture')
            registry = SourceRegistry(global_data, root)
            config = ForecastConfigSpec(
                problem=ForecastProblemSpec(time_col='time', freq='1h', horizon=1, targets=('load',), training_scope='global', series_id_cols=('site',)),
                data=global_data,
                features=FeatureSpec(target_lags={'load': (1,)}, observed_past_lags={}, datetime_features=(), transformations={}),
                strategy=ForecastStrategySpec('direct'), estimator=EstimatorSpec(model_type='ridge', target_adapter='independent'),
                probabilistic={}, validation={'history_steps': 4, 'train_window_steps': 2, 'fold_count': 1, 'stride_steps': 1}, output={},
            )
            compiler = FeatureCompiler(config)
            infos, requests = [], []
            for identities in (('A', 'B'), ('B',), ('B', 'A'), ('A',)):
                request = InformationSetRequest('2026-01-01T00:00Z', pd.DatetimeIndex(['2026-01-01T01:00Z']), identities)
                info = registry.materialize(request)
                result = info.known_future['weather']
                self.assertEqual(dict(zip(result.site, result.temperature)), {key: {'A': 10., 'B': 110.}[key] for key in identities})
                lineage = info.lineage[-1]
                assert isinstance(lineage, WeatherSourceLineage)
                proof = json.loads(lineage.weather_evidence)
                self.assertEqual({item['location_id'] for item in proof}, {'fixture-site' if key == 'A' else 'second-site' for key in identities})
                infos.append(info)
                requests.append(request)
            singles = [compiler.compile(info, request) for info, request in zip(infos, requests)]
            batches = compiler.compile_batch(infos, requests)
            self.assertEqual(len(batches), len(singles))
            for single, batch in zip(singles, batches):
                pd.testing.assert_frame_equal(single.frame, batch.frame)
                self.assertEqual(single.visibility_proof, batch.visibility_proof)
                self.assertEqual(single.source_lineage, batch.source_lineage)

    def test_base_directory_origin_and_updated_inputs_are_isolated(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            roots = [Path(a),Path(b)]
            data = [weather_data(root) for root in roots]
            request = InformationSetRequest('2026-01-01T00:00Z',pd.DatetimeIndex(['2026-01-01T01:00Z']),())
            first = SourceRegistry(data[0],roots[0])
            with patch('socket.create_connection', side_effect=AssertionError('network forbidden')), patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
                before = first.materialize(request)
            self.assertEqual(before.known_future['weather'].temperature.tolist(),[10.])
            early = InformationSetRequest('2025-12-31T23:30Z',request.forecast_times,())
            with self.assertRaises(ValueError):
                first.materialize(early)
            root = roots[1]
            frame = pd.read_csv(root / 'normalized.csv')
            frame['temperature_2m'] += 20.
            frame.to_csv(root / 'normalized.csv',index=False)
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest['snapshots'][0]['normalized']['sha256'] = hashlib.sha256((root / 'normalized.csv').read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            options = data[1].sources[1].generator_options.canonical_payload()
            options['inputs'][0]['sha256'] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
            weather = replace(data[1].sources[1],generator_options=WeatherGenerationSpec.from_mapping(options))
            changed = replace(data[1],sources=(data[1].sources[0],weather))
            second = SourceRegistry(changed,root)
            self.assertEqual(second.materialize(request).known_future['weather'].temperature.tolist(),[30.])
            self.assertNotEqual(source_hashes(data[0],roots[0]),source_hashes(changed,root))
            self.assertEqual(first.materialize(request).known_future['weather'].temperature.tolist(),[10.])
            warmer = replace(weather,generator_options=replace(weather.generator_options,variables=(replace(weather.generator_options.variables[0],unit='K'),)))
            converted = second.generators['weather'](warmer,request)
            self.assertAlmostEqual(converted.temperature.iloc[0],303.15)


if __name__ == '__main__':
    unittest.main()
