"""Small completed-fit checkpoint tests; no formal configuration runs."""
import tempfile
import unittest
from pathlib import Path

import numpy as np


class CheckpointStoreTest(unittest.TestCase):
    def test_missing_process_lock_backend_fails_closed(self):
        from unittest.mock import patch
        from forecasting_core.checkpoints import FitCheckpointError
        from model_performance.checkpoints import FileFitCheckpoint
        with tempfile.TemporaryDirectory() as root:
            store = FileFitCheckpoint(root, {"fold": "final"})
            with patch("model_performance.checkpoints.fcntl", None):
                with self.assertRaisesRegex(FitCheckpointError, "POSIX"):
                    store.run(identity={"model": "unit"}, arrays=(), fit=lambda: 42)

    def test_corrupt_schema_hash_and_stale_context(self):
        import json
        from forecasting_core.checkpoints import FitCheckpointError
        from model_performance.checkpoints import FileFitCheckpoint
        for damage in ("hash", "schema", "truncated", "identity"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as root:
                store = FileFitCheckpoint(root, {"config": "cfg", "fold": "fold/2", "implementation": "v1"})
                args = dict(identity={"model": "scalar/0"}, arrays=(np.ones(3),), fit=lambda: 42)
                store.run(**args)
                path = next(Path(root).rglob("*.fit"))
                header, payload = path.read_bytes().split(b"\n", 1)
                meta = json.loads(header)
                if damage == "schema":
                    meta["schema"] = 999
                elif damage == "identity":
                    meta["descriptor"]["context"]["fold"] = "other"
                elif damage == "hash":
                    payload += b"damage"
                elif damage == "truncated":
                    payload = b""
                path.write_bytes(json.dumps(meta).encode() + b"\n" + payload)
                with self.assertRaises(FitCheckpointError) as error:
                    store.run(**args)
                self.assertEqual(error.exception.phase, "load")
                self.assertEqual(error.exception.as_dict()["model"], "scalar/0")
                self.assertEqual(store.child(implementation="v2").run(**args), 42)

    def test_failed_fit_and_failed_atomic_publish_do_not_complete(self):
        from unittest.mock import patch
        from forecasting_core.checkpoints import FitCheckpointError
        from model_performance.checkpoints import FileFitCheckpoint
        with tempfile.TemporaryDirectory() as root:
            store = FileFitCheckpoint(root, {"config": "cfg", "fold": "final"})
            args = dict(identity={"model": "unit"}, arrays=(np.ones(3),))
            def fail():
                raise RuntimeError("not completed")
            with self.assertRaises(FitCheckpointError):
                store.run(**args, fit=fail)
            self.assertFalse(list(Path(root).rglob("*.fit")))
            with patch("model_performance.checkpoints.os.replace", side_effect=OSError("interruption")):
                with self.assertRaises(FitCheckpointError):
                    store.run(**args, fit=lambda: 42)
            self.assertFalse(list(Path(root).rglob("*.fit")))
            self.assertEqual(store.run(**args, fit=lambda: 43), 43)

    def test_concurrent_same_key_fits_once(self):
        from concurrent.futures import ThreadPoolExecutor
        from model_performance.checkpoints import FileFitCheckpoint
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def fit():
                calls.append(1)
                return 42
            def run(_):
                return FileFitCheckpoint(root, {"fold": "final"}).run(
                    identity={"model": "unit"}, arrays=(np.ones(3),), fit=fit)
            with ThreadPoolExecutor(max_workers=3) as pool:
                self.assertEqual(list(pool.map(run, range(3))), [42] * 3)
            self.assertEqual(len(calls), 1)

    def test_completed_unit_replays_and_array_change_misses(self):
        from model_performance.checkpoints import FileFitCheckpoint
        with tempfile.TemporaryDirectory() as root:
            calls = []
            def fit():
                calls.append(1)
                return np.array([42.0])
            args = dict(identity={"model": "scalar/0"}, arrays=(np.ones((3, 2)),))
            store = FileFitCheckpoint(root, {"config": "abc", "fold": "fold/1"})
            np.testing.assert_array_equal(store.run(**args, fit=fit), [42])
            fresh = FileFitCheckpoint(root, {"config": "abc", "fold": "fold/1"})
            np.testing.assert_array_equal(fresh.run(**args, fit=fit), [42])
            self.assertEqual(len(calls), 1)
            fresh.run(identity=args["identity"], arrays=(np.zeros((3, 2)),), fit=fit)
            self.assertEqual(len(calls), 2)


class CountingEstimator:
    calls = 0
    fail_at = None

    def fit(self, X, y, sample_weight=None):
        type(self).calls += 1
        if type(self).calls == type(self).fail_at:
            raise RuntimeError("injected fit interruption")
        self.mean = np.mean(y, axis=0)
        return self

    def predict(self, X):
        return np.broadcast_to(self.mean, (len(X),) + np.shape(self.mean)).copy()


class TrainerCheckpointTest(unittest.TestCase):
    def test_native_and_chain_replay(self):
        from tests import test_canonical_runtime_smoke as fixture
        from dataclasses import replace
        from model_training.trainer import CanonicalTrainer
        from model_training.estimators import EstimatorCapabilities
        from model_performance.checkpoints import FileFitCheckpoint
        for adapter in ("native", "regressor_chain"):
            with self.subTest(adapter=adapter), tempfile.TemporaryDirectory() as root:
                config = fixture.CanonicalRuntimeSmokeTest().build_config(
                    "unused.csv", mode="point", strategy="mimo", horizon=3)
                config = replace(config, estimator=replace(config.estimator, target_adapter=adapter))
                X, Y = (np.ones((6, 2)),), np.arange(18.).reshape(6, 3, 1)
                CountingEstimator.calls, CountingEstimator.fail_at = 0, None
                def train():
                    return CanonicalTrainer(config, estimator_factory=lambda: CountingEstimator(),
                        capabilities=EstimatorCapabilities(True, False, True, False, False, False, False),
                        feature_schema=("a", "b"), checkpoint=FileFitCheckpoint(root, {"fold": "final"})).train(X, Y)
                train()
                count = CountingEstimator.calls
                train()
                self.assertEqual(CountingEstimator.calls, count)

    def test_partial_strategy_fit_resume_without_closure_pickle(self):
        from tests import test_canonical_runtime_smoke as fixture
        from model_training.trainer import CanonicalTrainer
        from model_training.estimators import EstimatorCapabilities
        from model_performance.checkpoints import FileFitCheckpoint
        from forecasting_core.checkpoints import FitCheckpointError
        config = fixture.CanonicalRuntimeSmokeTest().build_config(
            "unused.csv", mode="point", strategy="direct", horizon=3)
        X = tuple(np.arange(12.).reshape(6, 2) + i for i in range(3))
        Y = np.arange(18.).reshape(6, 3, 1)
        CountingEstimator.calls, CountingEstimator.fail_at = 0, 2
        with tempfile.TemporaryDirectory() as root:
            def trainer():
                return CanonicalTrainer(config, estimator_factory=lambda: CountingEstimator(),
                    capabilities=EstimatorCapabilities(True, False, False, False, False, False, False),
                    feature_schema=("a", "b"),
                    checkpoint=FileFitCheckpoint(root, {"config": "test", "fold": "fold/1"}))
            with self.assertRaises(FitCheckpointError) as error:
                trainer().train(X, Y)
            self.assertEqual(error.exception.fold, "fold/1")
            self.assertEqual(error.exception.phase, "fit")
            CountingEstimator.fail_at = None
            artifact = trainer().train(X, Y)
            self.assertEqual(CountingEstimator.calls, 4)
            import pickle
            pickle.loads(pickle.dumps(artifact))
            trainer().train(X, Y)
            self.assertEqual(CountingEstimator.calls, 4)


class RuntimeCheckpointTest(unittest.TestCase):
    def test_config_source_failure_is_structured(self):
        from tests import test_canonical_runtime_smoke as fixture
        from forecasting_core.checkpoints import FitCheckpointError
        from model_pipeline.runner import run_canonical_config
        with tempfile.TemporaryDirectory() as root:
            config = fixture.CanonicalRuntimeSmokeTest().build_config(
                Path(root) / "missing.csv", mode="point")
            from dataclasses import replace
            config = replace(config, validation={key: value for key, value in dict(config.validation).items()
                                                 if key != "forecast_origin"})
            with self.assertRaises(FitCheckpointError) as error:
                run_canonical_config(config, Path(root) / "results", checkpoint_root=Path(root) / "checkpoints")
            self.assertEqual(error.exception.config, config.fingerprint())
            self.assertIsNotNone(error.exception.__cause__)

    def test_global_k2_all_strategy_groups_replay(self):
        from tests import test_canonical_global_runtime as fixture
        from dataclasses import replace
        from unittest.mock import patch
        import pandas as pd
        from data_loading import SourceRegistry
        from model_pipeline.runner import CanonicalBaseModelRunner
        from model_training.estimators.capabilities import _ModelFactoryEstimator
        original_fit = _ModelFactoryEstimator.fit
        original_batch = _ModelFactoryEstimator.fit_multi_output
        case = fixture.CanonicalGlobalRuntimeTest()
        case.setUp()
        self.addCleanup(case.tearDown)
        for mode in ("point", "quantile"):
            for strategy, chunk in fixture.STRATEGIES:
                with self.subTest(mode=mode, strategy=strategy):
                    config = case.build_config(strategy, chunk, mode=mode)
                    config = replace(config, validation={**dict(config.validation),
                        "history_steps": 24, "train_window_steps": 12, "fold_count": 1})
                    calls = []
                    def counted(estimator, *args, **kwargs):
                        calls.append(1)
                        return original_fit(estimator, *args, **kwargs)
                    def batch(estimator, *args, **kwargs):
                        calls.append(1)
                        return original_batch(estimator, *args, **kwargs)
                    def execute():
                        runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, case.root),
                            pd.Timestamp(config.validation.get("forecast_origin")),
                            checkpoint_root=case.root / "checkpoints")
                        scaler, transform, X, Y = runner.final_bundle_inputs()
                        _, artifact, _ = runner.fit_final(X, Y)
                        designs, provider = runner.forecast_designs(runner.origin, scaler, transform)
                        prediction = runner.predict(artifact, designs, provider,
                            runner.forecast_times(runner.origin), transform)
                        return prediction.values if mode == "point" else prediction.quantiles.values
                    with patch.object(_ModelFactoryEstimator, "fit", counted), \
                         patch.object(_ModelFactoryEstimator, "fit_multi_output", batch):
                        first = execute()
                        count = len(calls)
                        self.assertGreater(count, 0)
                        second = execute()
                        self.assertEqual(len(calls), count)
                        np.testing.assert_array_equal(first, second)
                        if mode == "point" and strategy == "direct":
                            with case.target_path.open("a") as handle:
                                handle.write("\n")
                            execute()
                            self.assertGreater(len(calls), count)

    def test_cqr_full_runtime_replay_fixed_and_calendar(self):
        from tests import test_canonical_runtime_smoke as fixture
        from dataclasses import replace
        from unittest.mock import patch
        import pandas as pd
        import pickle
        from model_pipeline.runner import run_canonical_config
        from model_training.estimators.capabilities import _ModelFactoryEstimator
        from probabilistic.calibration import ConformalCalibrationTracker
        original_fit = _ModelFactoryEstimator.fit
        original_apply = ConformalCalibrationTracker.apply_to_frame
        original_collect = ConformalCalibrationTracker.collect_from_frame
        for calendar in (False, True):
            with self.subTest(calendar=calendar), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / "load.csv"
                times = (pd.date_range("2026-01-01", "2026-07-31", freq="1D") if calendar
                         else pd.date_range("2026-01-01", periods=48, freq="1h"))
                pd.DataFrame({"time": times, "load": 10 + np.sin(np.arange(len(times)))}).to_csv(data, index=False)
                config = fixture.CanonicalRuntimeSmokeTest().build_config(data, mode="quantile")
                prob = {**dict(config.probabilistic),
                    "intervals": [{"name": "q10_q90", "lower_quantile": 0.1, "upper_quantile": 0.9}],
                    "calibration": {"method": "cqr", "interval": "q10_q90", "target_coverage": 0.8,
                        "calibration_windows": 2, "min_windows": 1, "min_scores": 1,
                        "label_availability_delay_steps": 0}}
                if calendar:
                    config = replace(config, problem=replace(config.problem, horizon=31, freq="1D"),
                        features=replace(config.features, datetime_features=()),
                        validation={"forecast_origin": str(times[-1]), "horizon_mode": "calendar_month",
                            "train_window_days": 90, "fold_count": 2, "stride_months": 1})
                else:
                    config = replace(config, validation={**dict(config.validation),
                        "history_steps": 30, "train_window_steps": 12, "fold_count": 2})
                config = replace(config, probabilistic=prob)
                calls, events, interrupt = [], [], [not calendar]
                def counted(estimator, *args, **kwargs):
                    calls.append(1)
                    if interrupt[0] and len(calls) == 5:
                        raise RuntimeError("interrupt second fold after completed units")
                    return original_fit(estimator, *args, **kwargs)
                def apply(tracker, *args, **kwargs):
                    events.append(("apply", str(kwargs["forecast_origin"])))
                    return original_apply(tracker, *args, **kwargs)
                def collect(tracker, *args, **kwargs):
                    events.append(("collect", str(kwargs["forecast_origin"])))
                    return original_collect(tracker, *args, **kwargs)
                with patch.object(_ModelFactoryEstimator, "fit", counted), \
                     patch.object(ConformalCalibrationTracker, "apply_to_frame", apply), \
                     patch.object(ConformalCalibrationTracker, "collect_from_frame", collect):
                    if not calendar:
                        from forecasting_core.checkpoints import FitCheckpointError
                        with self.assertRaises(FitCheckpointError):
                            run_canonical_config(config, root / "interrupted", checkpoint_root=root / "checkpoints")
                        interrupt[0] = False
                        events.clear()
                    first = run_canonical_config(config, root / "first", checkpoint_root=root / "checkpoints")
                    if not calendar:
                        self.assertEqual(len(calls), 10)  # 9 completed fits + one interrupted attempt
                    count, first_events = len(calls), list(events)
                    self.assertEqual([event[0] for event in events], ["apply", "collect"] * 2)
                    self.assertLess(events[0][1], events[2][1])
                    events.clear()
                    second = run_canonical_config(config, root / "second", checkpoint_root=root / "checkpoints")
                    self.assertEqual(len(calls), count)
                    self.assertEqual(events, first_events)
                for attr, name in (("test_dir", "cv_plot_df.csv"), ("forecast_dir", "prediction.csv")):
                    pd.testing.assert_frame_equal(pd.read_csv(getattr(first, attr) / name),
                                                  pd.read_csv(getattr(second, attr) / name))
                self.assertEqual(first.bundle.calibration_state, second.bundle.calibration_state)
                self.assertEqual(pickle.loads(pickle.dumps(second.bundle)).calibration_state,
                                 first.bundle.calibration_state)

    def test_shared_quantile_partial_booster_replay(self):
        from tests import test_canonical_runtime_smoke as fixture
        from dataclasses import replace
        from unittest.mock import patch
        import pickle
        from model_pipeline.fold_fit import _fit_quantile
        from model_performance.checkpoints import FileFitCheckpoint
        from forecasting_core.checkpoints import FitCheckpointError
        from model_training.estimators.capabilities import _ModelFactoryEstimator
        original = _ModelFactoryEstimator.fit
        config = fixture.CanonicalRuntimeSmokeTest().build_config(
            "unused.csv", mode="quantile", strategy="direct", horizon=3)
        config = replace(config, estimator=replace(config.estimator, model_type="xgb",
            params={"n_estimators": 2, "max_depth": 2}))
        X = tuple(np.arange(12.).reshape(6, 2) + i for i in range(3))
        Y = np.arange(18.).reshape(6, 3, 1)
        calls, fail = [], [True]
        def counted(estimator, *args, **kwargs):
            calls.append(1)
            if fail[0] and len(calls) == 2:
                raise RuntimeError("shared booster interruption")
            return original(estimator, *args, **kwargs)
        with tempfile.TemporaryDirectory() as root, patch.object(_ModelFactoryEstimator, "fit", counted):
            def fit():
                return _fit_quantile(config, ("a", "b"), X, Y, n_series=1,
                    checkpoint=FileFitCheckpoint(root, {"config": config.fingerprint(), "fold": "final"}))[1]
            with self.assertRaises(FitCheckpointError):
                fit()
            fail[0] = False
            artifact = fit()
            self.assertEqual(len(calls), 4)
            loaded = pickle.loads(pickle.dumps(artifact))
            fit()
            self.assertEqual(len(calls), 4)
            for level in artifact.levels:
                for a, b in zip(artifact.artifacts_by_level[level].model_groups,
                                loaded.artifacts_by_level[level].model_groups):
                    np.testing.assert_array_equal(a.predictor.adapter.predict(X[0]), b.predictor.adapter.predict(X[0]))

    def test_fold_and_final_replay_point_and_quantile(self):
        from tests import test_canonical_runtime_smoke as fixture
        from dataclasses import replace
        from unittest.mock import patch
        import pandas as pd
        import pickle
        from data_loading import SourceRegistry
        from model_pipeline.runner import CanonicalBaseModelRunner
        from model_training.estimators.capabilities import _ModelFactoryEstimator
        original_fit = _ModelFactoryEstimator.fit
        for mode in ("point", "quantile"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / "load.csv"
                pd.DataFrame({"time": pd.date_range("2026-01-01", periods=48, freq="1h"),
                              "load": np.sin(np.arange(48)) + 10}).to_csv(data, index=False)
                config = fixture.CanonicalRuntimeSmokeTest().build_config(data, mode=mode, strategy="direct")
                config = replace(config, validation={**dict(config.validation),
                    "history_steps": 30, "train_window_steps": 12, "fold_count": 2})
                calls = []
                def counted(estimator, *args, **kwargs):
                    calls.append(1)
                    return original_fit(estimator, *args, **kwargs)
                def execute():
                    runner = CanonicalBaseModelRunner(config, SourceRegistry(config.data, root),
                        pd.Timestamp(config.validation.get("forecast_origin")), checkpoint_root=root / "checkpoints")
                    artifacts = [runner.fit(window.train_indices)[-1] for window in runner.backtest_windows()]
                    scaler, transform, X, Y = runner.final_bundle_inputs()
                    trainer, artifact, capabilities = runner.fit_final(X, Y)
                    bundle = runner.build_final_bundle(scaler, transform, trainer, artifact, capabilities)
                    pickle.loads(pickle.dumps(bundle))
                    from forecasting_core.checkpoints import FitCheckpointError
                    bad_Y = Y.copy()
                    bad_Y[0, 0, 0] = np.nan
                    with self.assertRaises(FitCheckpointError) as error:
                        runner.fit_final(X, bad_Y)
                    self.assertEqual(error.exception.fold, "final")
                    self.assertEqual(error.exception.phase, "fit_final")
                    return artifacts + [artifact]
                with patch.object(_ModelFactoryEstimator, "fit", counted):
                    execute()
                    count = len(calls)
                    self.assertGreater(count, 0)
                    execute()
                    self.assertEqual(len(calls), count)


if __name__ == "__main__":
    unittest.main()
