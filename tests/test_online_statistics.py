"""在线全前缀统计：独立完整历史特征、追加原子性与恢复合同。"""
from dataclasses import replace
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from feature_engineering.statistics.streaming import StreamingStatistics
from model_predicting.loops.deployment import predict_strategy_bundle
from pipeline.online import RollingForecastSession
from pipeline.runner import run_canonical_config
from pipeline.lifecycle import CanonicalRuntimeResult
from forecasting_core.tensors.point import PointForecastTensor
from pipeline.supervised_design import SupervisedDesignBuilder
import test_canonical_runtime_smoke as fixtures

ADVANCED = {
    "rolling": {"windows": [5], "stats": ["mean", "std"], "columns": ["load"]},
    "ewm": {"halflives": [2.0, 7.5], "stats": ["mean", "std"], "columns": ["load"]},
    "time_since": {"events": ["peak", "trough"], "columns": ["load"]},
    "expanding": {"stats": ["median", "mean", "std", "min", "max", "min_diff", "max_diff", "skew", "kurt", "entropy"], "columns": ["load"]},
}


def frames(audit):
    # base call 与递归 provider 可对同一步重复编译；比较每次真正消费的列值。
    return pd.concat([item.frame for item in audit]).drop_duplicates("target_time").sort_values("target_time").reset_index(drop=True)


class OnlineStatisticsIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=70, freq="h"),
                                   "load": 100 + np.arange(70) + 5 * np.sin(np.arange(70))})
        self.path = self.root / "data.csv"
        self.frame.to_csv(self.path, index=False)
        self.frame = pd.read_csv(self.path, parse_dates=["time"])

    def fitted(self, strategy="recursive", advanced=ADVANCED):
        config = fixtures.CanonicalRuntimeSmokeTest().build_config(self.path, mode="point", strategy=strategy)
        config = replace(config, features=replace(config.features, transformations={"advanced": advanced}),
                         validation={"forecast_origin": "2026-01-02T23:00:00", "history_steps": 30,
                                     "train_window_steps": 12, "fold_count": 2, "stride_steps": 2})
        result = run_canonical_config(config, output_root=self.root / strategy)
        assert isinstance(result, CanonicalRuntimeResult)
        return config, result.bundle

    def reference(self, config, bundle, history):
        registry = SourceRegistry(config.data, self.root, reader=lambda _path: history.copy(deep=True))
        builder = SupervisedDesignBuilder(config, registry)
        origin = history.time.iloc[-1]
        if config.strategy.name.value == "direct":
            # 在线 provider 使用 single；完整历史也显式走公共 single API。
            # 默认 batch 的 mean/std 有既有独立容差合同，不作为逐位黄金值。
            request = builder.request(origin)
            information_set = registry.materialize(request)
            compiled = [builder.compiler.compile(information_set, request, horizon_steps=(coordinates[0].horizon_step,))
                        for coordinates in builder.plan.call_coordinates]
            rows = [item.frame.loc[:, list(item.schema.feature_names)].to_numpy() for item in compiled]
            prediction = predict_strategy_bundle(bundle, rows[0], forecast_times=request.forecast_times,
                raw_feature_provider=lambda call_index, _coordinates, _dependencies, _predicted: rows[call_index])
            assert isinstance(prediction, PointForecastTensor)
            return prediction, frames(compiled)
        designs, provider = builder.forecast_designs(origin, target_transform=bundle.target_transform)
        prediction = predict_strategy_bundle(bundle, designs[0], forecast_times=builder.request(origin).forecast_times,
                                             raw_feature_provider=provider)
        assert isinstance(prediction, PointForecastTensor)
        return prediction, frames(builder.audit)

    def test_direct_and_recursive_features_match_full_prefix_after_updates_and_restore(self):
        for strategy in ("direct", "recursive"):
            with self.subTest(strategy=strategy):
                config, bundle = self.fitted(strategy)
                session = RollingForecastSession(config, bundle, self.frame.iloc[:48], origin=self.frame.time.iloc[47])
                previous = 48
                for stop in (48, 49, 55, 64):
                    if stop > previous:
                        session.update(self.frame.iloc[previous:stop], origin=self.frame.time.iloc[stop - 1])
                    previous = stop
                    before = pickle.dumps(session.state())
                    prediction = session.predict()
                    assert isinstance(prediction, PointForecastTensor)
                    self.assertEqual(pickle.dumps(session.state()), before, "recursive prediction must not update observation state")
                    reference, features = self.reference(config, bundle, self.frame.iloc[:stop])
                    pd.testing.assert_frame_equal(frames(session.last_audit), features, check_exact=True)
                    np.testing.assert_array_equal(prediction.values, reference.values)
                    self.assertLessEqual(len(session.state()["history"]), session.retention_steps)
                    restored = RollingForecastSession.from_state(config, bundle, pickle.loads(before))
                    restored_prediction = restored.predict()
                    assert isinstance(restored_prediction, PointForecastTensor)
                    np.testing.assert_array_equal(restored_prediction.values, prediction.values)
                    pd.testing.assert_frame_equal(frames(restored.last_audit), features, check_exact=True)

    def test_chunking_prefix_provenance_and_history_rebuild(self):
        config, bundle = self.fitted()
        first = RollingForecastSession(config, bundle, self.frame.iloc[:48], origin=self.frame.time.iloc[47])
        second = RollingForecastSession(config, bundle, self.frame.iloc[:48], origin=self.frame.time.iloc[47])
        first.update(self.frame.iloc[48:60], origin=self.frame.time.iloc[59])
        for stop in range(49, 61):
            second.update(self.frame.iloc[stop - 1:stop], origin=self.frame.time.iloc[stop - 1])
        self.assertEqual(pickle.dumps(first.state()), pickle.dumps(second.state()))
        revised = self.frame.iloc[:60].copy()
        revised.loc[:10, "load"] += 1000
        prior_lineage = tuple(first.last_audit[0].source_lineage)
        prior_frame = frames(first.last_audit)
        first.rebuild(revised, origin=revised.time.iloc[-1])
        prediction = first.predict()
        assert isinstance(prediction, PointForecastTensor)
        reference, expected = self.reference(config, bundle, revised)
        np.testing.assert_array_equal(prediction.values, reference.values)
        pd.testing.assert_frame_equal(frames(first.last_audit), expected, check_exact=True)
        self.assertNotEqual(prior_frame.load_expanding_mean.iloc[0], expected.load_expanding_mean.iloc[0])
        self.assertNotEqual(prior_lineage, tuple(first.last_audit[0].source_lineage))

    def test_direct_default_batch_difference_matches_full_history_single(self):
        config, bundle = self.fitted("direct")
        history = self.frame.iloc[:64]
        origin = history.time.iloc[-1]
        session = RollingForecastSession(config, bundle, history, origin=origin)
        session.predict()
        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.root, reader=lambda _path: history.copy()))
        designs, provider = builder.forecast_designs(origin, target_transform=bundle.target_transform)
        predict_strategy_bundle(bundle, designs[0], forecast_times=builder.request(origin).forecast_times, raw_feature_provider=provider)
        actual, expected = frames(session.last_audit), frames(builder.audit)
        _, single = self.reference(config, bundle, history)
        self.assertEqual(list(actual.columns), list(expected.columns))
        for column in expected:
            # 默认 batch 有既有数值差异；证明新路径没有引入额外差异，
            # 不将 skew/kurt 等列的精确断言改为宽松容差。
            if any(token in column for token in ("_mean", "_std")):
                np.testing.assert_allclose(actual[column], expected[column], rtol=0, atol=1e-8)
            elif pd.api.types.is_numeric_dtype(expected[column]):
                np.testing.assert_array_equal(actual[column] - expected[column], single[column] - expected[column])
            else:
                pd.testing.assert_series_equal(actual[column], expected[column], check_exact=True)

    def test_failure_is_atomic_and_snapshot_does_not_alias_live_statistics(self):
        config, bundle = self.fitted()
        session = RollingForecastSession(config, bundle, self.frame.iloc[:48], origin=self.frame.time.iloc[47])
        saved = pickle.dumps(session.state())
        audit = session.last_audit
        with patch.object(session, "_predict", side_effect=ValueError("injected compile failure")):
            with self.assertRaisesRegex(ValueError, "injected compile failure"):
                session.update(self.frame.iloc[48:50], origin=self.frame.time.iloc[49])
            with self.assertRaisesRegex(ValueError, "injected compile failure"):
                session.rebuild(self.frame.iloc[:50], origin=self.frame.time.iloc[49])
        self.assertEqual(pickle.dumps(session.state()), saved)
        self.assertEqual(session.last_audit, audit)
        copied = session.state()
        copied["statistics"].count = -1
        self.assertEqual(pickle.dumps(session.state()), saved)
        for changes in ({"schema_version": 1}, {"statistics": None}, {"origin": self.frame.time.iloc[46].isoformat()}):
            with self.assertRaises(ValueError):
                RollingForecastSession.from_state(config, bundle, {**pickle.loads(saved), **changes})
        corrupted = pickle.loads(saved)
        corrupted["statistics"].count += 1
        with self.assertRaisesRegex(ValueError, "count|grid"):
            RollingForecastSession.from_state(config, bundle, corrupted)
        session.update(self.frame.iloc[48:50], origin=self.frame.time.iloc[49])
        self.assertEqual(session.state()["statistics"].count, 50)

    def test_statistics_snapshot_must_match_request_origin_and_identity(self):
        config, _bundle = self.fitted("direct")
        builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, self.root))
        statistics = StreamingStatistics(ADVANCED, config_fingerprint=config.fingerprint(), time_col="time", freq="1h")
        statistics = statistics.updated(self.frame.iloc[:49], origin=self.frame.time.iloc[48])
        with self.assertRaisesRegex(ValueError, "identity/origin"):
            builder.forecast_designs(self.frame.time.iloc[47], statistics_provider=statistics)
        statistics.config_fingerprint = "wrong"
        with self.assertRaisesRegex(ValueError, "identity/origin"):
            builder.forecast_designs(self.frame.time.iloc[48], statistics_provider=statistics)

    def test_finite_history_mode_keeps_exact_reference_without_statistics(self):
        config, bundle = self.fitted(advanced={"rolling": ADVANCED["rolling"]})
        session = RollingForecastSession(config, bundle, self.frame.iloc[:48], origin=self.frame.time.iloc[47])
        self.assertIsNone(session.state()["statistics"])
        prediction = session.predict()
        assert isinstance(prediction, PointForecastTensor)
        reference, expected = self.reference(config, bundle, self.frame.iloc[:48])
        np.testing.assert_array_equal(prediction.values, reference.values)
        pd.testing.assert_frame_equal(frames(session.last_audit), expected, check_exact=True)
