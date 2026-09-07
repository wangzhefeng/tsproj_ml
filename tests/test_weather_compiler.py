"""真实 YAML→Registry→Compiler（合成资产）的单/批一致性。"""
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest

import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from data_loading import BUILTIN_GENERATORS, InformationSetRequest, SourceRegistry
from feature_engineering import FeatureCompiler
from forecasting_core.specs import EstimatorSpec, FeatureSpec, ForecastConfigSpec, ForecastProblemSpec, ForecastStrategySpec
from test_weather_registry import weather_data


def config_fixture(root):
    data = weather_data(root)
    pd.DataFrame({'time':['2025-12-31T23:00Z','2026-01-01T00:00Z'],'load':[0.,1.]}).to_csv(root / 'target.csv',index=False)
    config = ForecastConfigSpec(
        problem=ForecastProblemSpec(time_col='time',freq='1h',horizon=2,targets=('load',),training_scope='local'),
        data=data, features=FeatureSpec(target_lags={'load':(2,)},observed_past_lags={},datetime_features=(),transformations={}),
        strategy=ForecastStrategySpec('direct'), estimator=EstimatorSpec(model_type='ridge',target_adapter='independent'),
        probabilistic={},validation={'history_steps':4,'train_window_steps':2,'fold_count':1,'stride_steps':2},output={},
    )
    path = root / 'model.yaml'
    path.write_text(yaml.safe_dump(config.canonical_payload(), sort_keys=False))
    return load_yaml_config(path)


class WeatherCompilerTest(unittest.TestCase):
    def test_weather_frequency_must_match_model_frequency(self):
        with tempfile.TemporaryDirectory() as directory:
            config = config_fixture(Path(directory))
            source = config.data.sources[1]
            options = source.generator_options
            changed = replace(source,generator_options=replace(options,temporal=replace(options.temporal,freq='5min')))
            with self.assertRaisesRegex(ValueError,'weather frequency'):
                replace(config,data=replace(config.data,sources=(config.data.sources[0],changed)))

    def test_single_batch_yaml_weather_values_visibility_and_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = config_fixture(root)
            registry = SourceRegistry(config.data,root,generators=BUILTIN_GENERATORS)
            compiler = FeatureCompiler(config)
            requests = [InformationSetRequest('2026-01-01T00:30Z',pd.date_range('2026-01-01T01:00Z',periods=2,freq='1h'), (), target_access=access) for access in ['history_only','supervised_labels']]
            infos = [registry.materialize(r) for r in requests]
            pd.testing.assert_frame_equal(infos[0].known_future['weather'],infos[1].known_future['weather'])
            self.assertEqual(infos[0].lineage[-1], infos[1].lineage[-1])
            with self.assertRaisesRegex(ValueError, "history_only"):
                compiler.compile(infos[1],requests[1])
            requests = [requests[0], InformationSetRequest('2026-01-01T00:45Z', requests[0].forecast_times, (), target_access='history_only')]
            infos = [registry.materialize(request) for request in requests]
            singles = [compiler.compile(info,r) for info,r in zip(infos,requests)]
            batches = compiler.compile_batch(infos,requests)
            self.assertEqual(len(batches),2)
            for single,batch in zip(singles,batches):
                pd.testing.assert_frame_equal(single.frame,batch.frame)
                self.assertEqual(single.visibility_proof,batch.visibility_proof)
                self.assertEqual(single.source_lineage,batch.source_lineage)
                cols = [c for c in single.frame if 'temperature' in c]
                self.assertTrue(cols, single.frame.columns)
                self.assertEqual(set(single.frame[cols].to_numpy().ravel()), {10.,12.})
                self.assertFalse(single.source_lineage[-1].includes_target_labels)


if __name__ == '__main__':
    unittest.main()
