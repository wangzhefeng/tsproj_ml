"""显式训练窗融合：OOF独立历史、异构预热、隔离与bundle部署。"""
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from data_loading import SourceRegistry
from forecasting_core.specs.config import parse_model_config
from model_ensemble.deployment import predict_ensemble_bundle
from model_ensemble.loader import load_ensemble_config
from model_ensemble.runtime import run_ensemble_config
from model_ensemble.specs import EnsembleSpecError
from model_pipeline.runner import CanonicalBaseModelRunner
from tests.test_ensemble_runtime import EnsembleRuntimeTestBase, _member_doc, _ensemble_doc, RUNTIME_SERVICES


class EnsembleTemporalWindowTest(EnsembleRuntimeTestBase):
    def bounded_config(self, gap=0):
        for name, strategy in [('direct', 'direct'), ('recursive', 'recursive')]:
            doc = _member_doc(strategy, 'ridge', name)
            if strategy == 'recursive':
                doc['features']['target_lags']['load'] = [2, 4]
            doc['features']['transformations']['advanced'] = {
                'expanding': {'columns': ['load'], 'stats': ['mean', 'std']}}
            doc['validation'].pop('training_window')
            doc['validation']['training_window'] = {'kind': 'rolling', 'history_steps': 24}
            (self.root / f'member_{name}.yaml').write_text(yaml.safe_dump(doc))
        doc = _ensemble_doc('averaging')
        doc['validation'].pop('training_window')
        doc['validation']['training_window'] = {'kind': 'rolling', 'history_steps': 24}
        doc['ensemble']['oof'].update(gap_steps=gap)
        path = self.root / 'ensemble.yaml'
        path.write_text(yaml.safe_dump(doc))
        return load_ensemble_config(path)

    def run_bounded(self, config):
        return run_ensemble_config(config, output_root=self.root, base_dir=self.root,
                                   services=RUNTIME_SERVICES, use_oof_cache=False)

    def test_bounded_members_use_fold_context_and_final_window(self):
        config = self.bounded_config(gap=2)
        fitted = []
        original = CanonicalBaseModelRunner.fit

        def tracked(runner, indices, **kwargs):
            self.assertEqual(runner.builder.history_start, runner.origin - 23 * runner.builder.offset)
            self.assertTrue(all(runner.forecast_times(runner.supervised_origins[i])[-1]
                                <= runner.origin - 2 * runner.builder.offset for i in indices))
            fitted.append((runner.config.strategy.name.value, runner.origin, len(indices)))
            return original(runner, indices, **kwargs)

        with patch.object(CanonicalBaseModelRunner, 'fit', tracked):
            result = self.run_bounded(config)
        self.assertEqual(len(fitted), 4)
        expected = [pd.Timestamp(f['origin']) for f in result['oof'].folds]
        self.assertEqual([r[1] for r in fitted], [x for o in expected for x in (o, o)])
        self.assertNotEqual(fitted[0][2], fitted[1][2])
        self.assertEqual(result['oof'].folds[0]['training_sample_count'], fitted[0][2])
        self.assertEqual(result['oof'].folds[0]['member_training_sample_count'],
                         {'m_direct': fitted[0][2], 'm_recursive': fitted[1][2]})
        self.assertTrue(np.isfinite(result['oof'].values_by_member['m_direct']).all())
        frame = pd.read_csv(result['test_dir'] / 'cv_plot_df.csv', parse_dates=['time'])
        truth = pd.read_csv(self.root / 'data.csv', parse_dates=['time']).set_index('time')['load']
        np.testing.assert_array_equal(frame['actual'].to_numpy() if 'actual' in frame else frame['actual_value'].to_numpy(),
                                      truth.loc[frame.time].to_numpy())
        with (result['model_dir'] / 'model.pkl').open('rb') as file:
            restored = pickle.load(file)
        designs, providers = {}, {}
        for ref in config.members:
            raw = yaml.safe_load((self.root / ref.config_ref).read_text())
            member_config = parse_model_config(raw, source=ref.config_ref)
            runner = CanonicalBaseModelRunner(member_config, SourceRegistry(member_config.data, self.root),
                                              pd.Timestamp(member_config.validation['forecast_origin']))
            member_bundle = restored.model['member_bundles'][ref.name]
            rows, provider = runner.builder.forecast_designs(runner.origin, target_transform=member_bundle.target_transform)
            designs[ref.name], providers[ref.name] = rows[0], provider
            times = runner.forecast_times(runner.origin)
        prediction = predict_ensemble_bundle(restored, designs, forecast_times=times, raw_feature_providers=providers)
        np.testing.assert_allclose(prediction.values, result['combined_values'], rtol=0, atol=1e-12)

    def test_history_before_window_and_future_do_not_change_first_oof_prediction(self):
        config = self.bounded_config()
        before = self.run_bounded(config)
        origin = pd.Timestamp(before['oof'].folds[0]['origin'])
        frame = pd.read_csv(self.root / 'data.csv', parse_dates=['time'])
        frame.loc[(frame.time < origin - pd.Timedelta(hours=23)) | (frame.time > origin), 'load'] += 1e6
        frame.to_csv(self.root / 'data.csv', index=False)
        after = self.run_bounded(config)
        for member in before['oof'].member_order:
            np.testing.assert_array_equal(before['oof'].values_by_member[member][0],
                                          after['oof'].values_by_member[member][0])

    def test_member_window_mismatch_is_rejected_before_fit(self):
        config = self.bounded_config()
        path = self.root / 'member_recursive.yaml'
        doc = yaml.safe_load(path.read_text())
        doc['validation']['training_window']['history_steps'] = 30
        path.write_text(yaml.safe_dump(doc))
        with self.assertRaisesRegex(EnsembleSpecError, 'share the top-level training_window'):
            self.run_bounded(config)
