"""合成天气资产的真实 Ridge 全生命周期；绝非真实来源/场景验收。"""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.specs import ForecastStrategySpec
from forecasting_core.specs.weather import WeatherGenerationSpec
from model_forecasting.deployment import predict_strategy_bundle
from model_pipeline.runner import run_canonical_config
from model_pipeline.supervised_design import SupervisedDesignBuilder
from models.pickle_io import ModelDeployPkl
from test_weather_compiler import config_fixture


class WeatherRuntimeTest(unittest.TestCase):
    def test_real_fit_backtest_forecast_and_bundle_reload_preserve_weather_evidence(self):
        self.run_weather_lifecycle(research=False)

    def test_research_bundle_reload_rejects_production_and_replays_explicitly(self):
        self.run_weather_lifecycle(research=True)

    def run_weather_lifecycle(self, *, research):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            times = pd.date_range('2026-01-01',periods=48,freq='1h')
            pd.DataFrame({'time':times,'load':100.+np.arange(48)*2.}).to_csv(root / 'target.csv',index=False)
            frame = pd.DataFrame({'time':pd.date_range('2026-01-01',periods=52,freq='1h',tz='UTC'),'temperature_2m':10.+np.sin(np.arange(52)), 'available_at':'2025-12-30T00:00Z'})
            frame.to_csv(root / 'normalized.csv',index=False)
            # 合成原始快照与规范值相同，元数据/证据文件仍明确标为 fixture。
            frame.to_csv(root / 'raw.csv',index=False)
            manifest = json.loads((root / 'manifest.json').read_text())
            meta = manifest['snapshots'][0]
            meta.update(init_time='2025-12-29T00:00Z',issued_at='2025-12-29T12:00Z',received_at='2025-12-30T00:00Z')
            for ref in [meta['normalized'],meta['raw'][0]]:
                ref['sha256'] = hashlib.sha256((root / ref['path']).read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(manifest))
            source = config.data.sources[1]
            options = source.generator_options.canonical_payload()
            if research:
                options['research'] = {'release_delay': '8h', 'rationale': 'SYNTHETIC research assumption'}
                options['semantics_version'] = 'weather_research_v1'
            options['inputs'][0]['sha256'] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
            weather = replace(source,generator_options=WeatherGenerationSpec.from_mapping(options))
            config = replace(config,data=replace(config.data,sources=(config.data.sources[0],weather)),strategy=ForecastStrategySpec('recursive'),validation={'forecast_origin':'2026-01-02T23:00:00','history_steps':24,'train_window_steps':8,'fold_count':1,'stride_steps':2})
            original = Path.cwd()
            try:
                os.chdir(root)
                result = run_canonical_config(config,output_root=root / 'results')
            finally:
                os.chdir(original)
            prediction = pd.read_csv(result.forecast_dir / 'prediction.csv')
            scores = pd.read_csv(result.test_dir / 'test_scores_df.csv')
            self.assertEqual(len(prediction),2)
            self.assertFalse(scores.empty)
            self.assertTrue(np.isfinite(prediction.predict_value).all())
            loaded = ModelDeployPkl(str(result.model_dir / 'model.pkl')).load_model()
            if research:
                self.assertEqual(loaded.execution_mode, 'research_replay')
                self.assertEqual(loaded.schema_payload()['execution_mode'], 'research_replay')
            self.assertEqual(loaded.config_fingerprint,config.fingerprint())
            self.assertIn('temperature',loaded.selected_features)
            evidence = [item for item in loaded.source_lineage if item['source']=='weather']
            self.assertTrue(evidence)
            self.assertTrue(all(json.loads(item['weather_evidence']) for item in evidence))
            builder = SupervisedDesignBuilder(config,SourceRegistry(config.data,root))
            origin = pd.Timestamp(config.validation['forecast_origin'])
            designs, provider = builder.forecast_designs(origin,target_transform=loaded.target_transform)
            forecast_times = pd.date_range(origin+pd.Timedelta('1h'),periods=2,freq='1h')
            if research:
                with self.assertRaisesRegex(ValueError, 'research'):
                    predict_strategy_bundle(loaded, designs[0], forecast_times=forecast_times, raw_feature_provider=provider)
            deployed = predict_strategy_bundle(loaded,designs[0],forecast_times=forecast_times,raw_feature_provider=provider,
                                               purpose='research_replay' if research else 'production')
            np.testing.assert_allclose(deployed.values[0,:,0],prediction.predict_value.to_numpy(),rtol=1e-12,atol=1e-8)


if __name__ == '__main__':
    unittest.main()
