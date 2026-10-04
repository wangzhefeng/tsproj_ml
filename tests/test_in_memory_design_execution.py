"""业务执行不落盘、轻量预检与延迟设计的真实入口回归。"""
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import unittest
from typing import Any
import weakref
import yaml

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from data_loading import SourceRegistry
from model_pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from model_pipeline.batch_runtime import run_canonical_batch, _load_tasks, _preflight_groups
from model_performance.resource_planner import detect_runtime_budget
import test_batch_runtime as fixtures
from forecasting_core.tensors import PointForecastTensor
from forecasting_core.specs import ForecastConfigSpec
from model_pipeline.lifecycle import CanonicalRuntimeResult


class InMemoryDesignExecutionTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.CanonicalBatchRuntimeTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def test_retired_disk_cache_argument_is_rejected(self):
        f = self.fixture
        config = load_yaml_config(f.paths[0])
        assert isinstance(config, ForecastConfigSpec)
        with self.assertRaisesRegex(TypeError, 'compiled_cache_root'):
            retired_options: dict[str, Any] = {'compiled_cache_root': f.root / 'forbidden'}
            CanonicalBaseModelRunner(config, SourceRegistry(config.data, f.root), f.origin,
                                     **retired_options)
        self.assertFalse((f.root / 'forbidden').exists())

    def test_later_group_compile_failure_preserves_completed_group_and_releases_design(self):
        f = self.fixture
        document = yaml.safe_load(f.paths[1].read_text())
        document['features']['target_lags']['load'] = [2, 3, 5]
        f.paths[1].write_text(yaml.safe_dump(document))
        original = CanonicalBaseModelRunner._compile_supervised_arrays
        released = []
        release_checks = []
        def compile_design(runner):
            if runner.config.estimator.model_type == 'lasso':
                release_checks.append(released[0]() is None)
                raise ValueError('late numerical construction failure')
            original(runner)
            released.append(weakref.ref(runner.Y_all))
        with patch.object(CanonicalBaseModelRunner, '_compile_supervised_arrays', compile_design):
            result = run_canonical_batch(f.paths, output_root=f.root / 'partial')
        self.assertEqual((result.completed_count, result.failed_count), (1, 1))
        self.assertEqual(release_checks, [True])
        self.assertFalse(list(f.root.rglob('_compiled_features')))

    def test_planning_never_builds_full_design_and_concurrent_prepare_runs_once(self):
        f = self.fixture
        config = load_yaml_config(f.paths[0])
        assert isinstance(config, ForecastConfigSpec)
        original = CanonicalBaseModelRunner._compile_supervised_arrays
        with patch.object(CanonicalBaseModelRunner, '_compile_supervised_arrays', autospec=True,
                          side_effect=original) as compile_design:
            runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, f.root), f.origin)
            self.assertEqual(runner.runtime_resources_payload()['design_storage']['retained_array_bytes'], 0)
            compile_design.assert_not_called()
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _: runner.prepare_training(), range(2)))
            self.assertEqual(compile_design.call_count, 1)
            self.assertGreater(runner.runtime_resources_payload()['design_storage']['retained_array_bytes'], 0)
        self.assertFalse(list(f.root.rglob('_compiled_features')))

    def test_preflight_rejects_retired_disk_cache_root(self):
        budget = detect_runtime_budget()
        retired_args: tuple[Any, ...] = ({}, {}, self.fixture.root, budget, 1)
        retired_options: dict[str, Any] = {
            'root': self.fixture.root, 'budget': budget, 'config_workers': 1,
        }
        for args, options in ((retired_args, {}), (({}, {}), retired_options)):
            with self.subTest(options=bool(options)), self.assertRaises(TypeError):
                _preflight_groups(*args, **options)

    def test_global_preflight_is_matrix_free_and_batch_compiles_one_shared_group(self):
        f = self.fixture
        tasks = _load_tasks(f.paths, generators={})
        groups = {tasks[0].raw_fingerprint: list(tasks)}
        state = {'tasks': {task.task_id: {'status': 'pending'} for task in tasks}}
        with patch.object(CanonicalBaseModelRunner, '_compile_supervised_arrays',
                          side_effect=AssertionError('preflight allocated full design')):
            _preflight_groups(groups, state, budget=detect_runtime_budget(), config_workers=1)
        original = CanonicalBaseModelRunner._compile_supervised_arrays
        with patch.object(CanonicalBaseModelRunner, '_compile_supervised_arrays', autospec=True,
                          side_effect=original) as compile_design:
            result = run_canonical_batch(f.paths, output_root=f.root / 'batch')
            self.assertEqual(result.completed_count, 2)
            self.assertEqual(compile_design.call_count, 1)
        self.assertFalse(list(f.root.rglob('_compiled_features')))

    def test_single_lifecycle_avoids_disk_cache_and_matches_explicit_preparation(self):
        f = self.fixture
        config = load_yaml_config(f.paths[0])
        assert isinstance(config, ForecastConfigSpec)
        prepared = CanonicalBaseModelRunner(config, SourceRegistry(config.data, f.root), f.origin)
        inputs = prepared.final_bundle_inputs()
        _, artifact, _ = prepared.fit_final(inputs[2], inputs[3])
        designs, provider = prepared.forecast_designs(f.origin, inputs[0], inputs[1])
        expected = prepared.predict(artifact, designs, provider, prepared.forecast_times(f.origin), inputs[1])
        result = run_canonical_config(config, output_root=f.root / 'single')
        assert isinstance(result, CanonicalRuntimeResult)
        assert isinstance(expected, PointForecastTensor)
        forecast = pd.read_csv(result.forecast_dir / 'prediction.csv')
        np.testing.assert_allclose(forecast['predict_value'], expected.values.reshape(-1), rtol=0, atol=1e-10)
        self.assertFalse(list(f.root.rglob('_compiled_features')))
        # 对照两次独立拟合，输入设计不能被模型训练修改。
        repeated = CanonicalBaseModelRunner(config, SourceRegistry(config.data, f.root), f.origin)
        np.testing.assert_array_equal(prepared.Y_all, repeated.Y_all)
        self.assertTrue(np.isfinite(expected.values).all())
