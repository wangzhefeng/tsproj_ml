"""代理必须追溯原点前完整依赖，不能目标期 actual 回填。"""
import hashlib
import copy
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from forecasting_core.specs.weather import WeatherGenerationSpec
from test_weather_assets import asset_fixture
from test_weather_spec import weather_options


class WeatherScenarioTest(unittest.TestCase):
    def test_disjoint_historical_partitions_combine_but_overlap_is_rejected(self):
        self.run_partition_case(research=False)

    def test_research_partitions_use_explicit_simulated_not_actual_availability(self):
        self.run_partition_case(research=True)

    def run_partition_case(self, *, research):
        from data_loading.weather_generator.pipeline import generate_weather
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = asset_fixture(root,kind='observation')
            template = manifest['snapshots'][0]
            template['evidence_class'] = 'historical_release_contract'
            snapshots = []
            for hour in (1,2):
                meta = copy.deepcopy(template)
                meta['snapshot_id'] = f'partition-{hour}'
                path = root / f'part-{hour}.csv'
                pd.DataFrame({'time':[f'2025-01-01T0{hour}:00Z'],'temperature_2m':[float(hour)],'available_at':['2026-09-01T00:00Z' if research else f'2025-01-01T0{hour}:30Z']}).to_csv(path,index=False)
                meta['normalized'] = {'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                snapshots.append(meta)
            def run():
                (root / 'manifest.json').write_text(json.dumps({'schema_version':'weather_asset_v1','snapshots':snapshots}))
                options = weather_options()
                options.update(inputs=[{'manifest':'manifest.json','sha256':hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()}],scenario='prior_year_proxy',vintage_policy=None,proxy={'years':1,'leap_day':'reject','data_kind':'observation'})
                options['temporal']['timezone'] = 'UTC'
                if research:
                    options.update(semantics_version='weather_research_v1', research={'release_delay':'7D', 'rationale':'SYNTHETIC assumption'})
                return generate_weather(WeatherGenerationSpec.from_mapping(options),root,pd.Timestamp('2026-01-01',tz='UTC'),pd.date_range('2026-01-01T01:00Z',periods=2,freq='1h'),())
            frame, proof = run()
            self.assertEqual(frame.temperature.tolist(),[1.,2.])
            self.assertEqual(proof['snapshot_ids'],['partition-1','partition-2'])
            snapshots[1]['normalized'] = snapshots[0]['normalized']
            with self.assertRaisesRegex(ValueError,'overlap'):
                run()

    def test_future_actual_perturbation_does_not_change_origin_proxy(self):
        from data_loading.weather_generator.pipeline import generate_weather
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = asset_fixture(root, kind='observation')
            meta = manifest['snapshots'][0]
            meta['evidence_class'] = 'historical_release_contract'
            frame = pd.DataFrame({'time':['2025-01-01T01:00:00Z','2026-01-01T01:00:00Z'], 'temperature_2m':[5.,100.], 'available_at':['2025-01-01T02:00:00Z','2026-01-01T02:00:00Z']})
            def run(values):
                values.to_csv(root / 'normalized.csv', index=False)
                meta['normalized']['sha256'] = hashlib.sha256((root / 'normalized.csv').read_bytes()).hexdigest()
                (root / 'manifest.json').write_text(json.dumps(manifest))
                options = weather_options()
                options.update(inputs=[{'manifest':'manifest.json','sha256':hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()}], scenario='prior_year_proxy', vintage_policy=None, proxy={'years':1,'leap_day':'reject','data_kind':'observation'})
                options['temporal']['timezone'] = 'UTC'
                return generate_weather(WeatherGenerationSpec.from_mapping(options), root, pd.Timestamp('2026-01-01T00:00Z'), pd.DatetimeIndex(['2026-01-01T01:00Z']), ())
            before, proof = run(frame)
            frame.loc[1,'temperature_2m'] = 9999.
            after, _ = run(frame)
            pd.testing.assert_frame_equal(before, after)
            self.assertEqual(before.temperature.tolist(), [5.])
            self.assertEqual(proof['dependency_times'], ['2025-01-01T01:00:00+00:00'])
            frame.loc[0,'available_at'] = '2026-01-01T02:00:00Z'
            with self.assertRaises(ValueError):
                run(frame)

    def test_leap_day_requires_explicit_policy(self):
        from data_loading.weather_generator.scenarios import proxy_time
        time = pd.Timestamp('2024-02-29T12:00Z')
        with self.assertRaises(ValueError):
            proxy_time(time, 'UTC', 'reject', '15min')
        self.assertEqual(proxy_time(time, 'UTC', 'feb28', '15min'), pd.Timestamp('2023-02-28T12:00Z'))
        self.assertEqual(proxy_time(pd.Timestamp('2025-02-28T00:00Z'), 'UTC', 'reject', '1ME'), pd.Timestamp('2024-02-29T00:00Z'))


if __name__ == '__main__':
    unittest.main()
