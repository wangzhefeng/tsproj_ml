"""合成天气的真实融合 OOF/final/预测/部署闭环，不替代正式场景验收。"""
import hashlib
import json

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from data_loading import SourceRegistry
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.tensors import PointForecastTensor
from model_ensemble.deployment import predict_ensemble_bundle
from model_ensemble.loader import load_ensemble_config
from model_ensemble.runtime import run_ensemble_config
from model_pipeline.supervised_design import SupervisedDesignBuilder
from models.pickle_io import ModelDeployPkl
from test_ensemble_runtime import EnsembleRuntimeTestBase, RUNTIME_SERVICES
from test_weather_registry import weather_data


class WeatherEnsembleRuntimeTest(EnsembleRuntimeTestBase):
    def test_weather_oof_cache_final_fit_and_reloaded_deployment(self):
        self.run_weather_ensemble(research=False)

    def test_research_ensemble_is_not_deployable_in_production(self):
        self.run_weather_ensemble(research=True)

    def run_weather_ensemble(self, *, research):
        data = weather_data(self.root)
        frame = pd.DataFrame({'time':pd.date_range('2026-01-01',periods=76,freq='1h',tz='UTC'), 'temperature_2m':10.+np.sin(np.arange(76)), 'available_at':'2025-12-30T00:00Z'})
        frame.to_csv(self.root/'normalized.csv',index=False)
        frame.to_csv(self.root/'raw.csv',index=False)
        manifest = json.loads((self.root/'manifest.json').read_text())
        meta = manifest['snapshots'][0]
        meta.update(init_time='2025-12-29T00:00Z',issued_at='2025-12-29T12:00Z',received_at='2025-12-30T00:00Z')
        for ref in [meta['normalized'],meta['raw'][0]]:
            ref['sha256'] = hashlib.sha256((self.root/ref['path']).read_bytes()).hexdigest()
        (self.root/'manifest.json').write_text(json.dumps(manifest))
        sources = data.canonical_payload()['sources']
        assert isinstance(sources, list)
        weather = sources[1]
        if research:
            weather['generator_options'].update(semantics_version='weather_research_v1',
                                                research={'release_delay':'8h', 'rationale':'SYNTHETIC assumption'})
        weather['generator_options']['inputs'][0]['sha256'] = hashlib.sha256((self.root/'manifest.json').read_bytes()).hexdigest()
        for name in ('direct','recursive'):
            path = self.root/f'member_{name}.yaml'
            doc = yaml.safe_load(path.read_text())
            doc['data']['sources'].append(weather)
            path.write_text(yaml.safe_dump(doc))
        doc = {key:doc[key] for key in ('schema_version','problem','data','probabilistic','validation')}
        doc['ensemble'] = {'members':[{'name':f'm_{name}','config_ref':f'member_{name}.yaml'} for name in ('direct','recursive')], 'oof':{'train_window_steps':6,'fold_count':2,'stride_steps':1}, 'method':{'name':'averaging'}}
        doc['output'] = {'scenario_subpath':'synthetic-weather-ensemble'}
        path = self.root/'ensemble.yaml'
        path.write_text(yaml.safe_dump(doc))
        config = load_ensemble_config(path)
        result = run_ensemble_config(config,output_root=self.root/'results',base_dir=self.root,services=RUNTIME_SERVICES,use_oof_cache=True)
        self.assertFalse(result['oof_cache_hit'])
        repeated = run_ensemble_config(config,output_root=self.root/'results',base_dir=self.root,services=RUNTIME_SERVICES,use_oof_cache=True)
        self.assertTrue(repeated['oof_cache_hit'])
        scores = pd.read_csv(result['test_dir']/'test_scores_df.csv')
        prediction = pd.read_csv(result['forecast_dir']/'prediction.csv')
        self.assertFalse(scores.empty)
        self.assertEqual(len(prediction),2)
        bundle = ModelDeployPkl(str(result['model_dir']/'model.pkl')).load_model()
        if research:
            self.assertEqual(bundle.execution_mode, 'research_replay')
            with self.assertRaisesRegex(ValueError, 'research'):
                predict_ensemble_bundle(bundle, {}, forecast_times=result['forecast_times'])
        designs, providers = {}, {}
        for name in ('direct','recursive'):
            member = load_yaml_config(self.root/f'member_{name}.yaml')
            assert isinstance(member, ForecastConfigSpec)
            saved = bundle.model['member_bundles'][f'm_{name}']
            self.assertIn('temperature',saved.selected_features)
            builder = SupervisedDesignBuilder(member,SourceRegistry(member.data,self.root))
            origin = pd.Timestamp(doc['validation']['forecast_origin'])
            assert isinstance(origin, pd.Timestamp)
            matrices, provider = builder.forecast_designs(origin,target_transform=saved.target_transform)
            designs[f'm_{name}'] = matrices[0]
            providers[f'm_{name}'] = provider
        deployed = predict_ensemble_bundle(bundle,designs,forecast_times=result['forecast_times'],raw_feature_providers=providers,
                                           purpose='research_replay' if research else 'production')
        assert isinstance(deployed, PointForecastTensor)
        np.testing.assert_allclose(deployed.values,result['combined_values'],rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(deployed.values.ravel(),prediction.predict_value.to_numpy(),rtol=1e-12,atol=1e-8)
