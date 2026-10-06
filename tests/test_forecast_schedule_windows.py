"""发报时间、目标区间和训练窗口的小数据端到端契约。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from pipeline.design_identity import compute_raw_design_fingerprint
from pipeline.batch_artifacts import artifact_paths, validate_artifacts
from pipeline.runner import CanonicalBaseModelRunner
from tests.test_raw_history_window import make_config


class ForecastScheduleWindowsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=360, freq="1h")
        self.frame = pd.DataFrame({"time": self.times, "load": 100 + np.sin(np.arange(360) / 12)})
        self.frame.to_csv(self.path, index=False)
        base = make_config(self.path, strategy="direct")
        base = replace(base, features=replace(base.features, transformations={
            **base.features.canonical_payload()["transformations"],
            "direct": {"layout": "single_model_horizon", "align_to_target": False,
                       "horizon_feature": {"enabled": True, "name": "h", "cyclical": False}}}))
        self.base = replace(base, problem=replace(base.problem, horizon=48))

    def config(self, kind="rolling", start="next_day"):
        validation = {k: v for k, v in dict(self.base.validation).items()
                      if k not in {"train_history_steps", "train_window_steps", "seasonal_naive_lag"}}
        validation.update(forecast_origin=self.times[302].isoformat(), history_steps=300,
                          fold_count=2, stride_steps=24,
                          forecast_window={"start": start},
                          training_window=({"kind": "rolling", "history_steps": 120} if kind == "rolling"
                                           else {"kind": "expanding", "start_time": self.times[0].isoformat()}),
                          training={"origin_sampling": {"time_of_day": "14:00"}})
        return replace(self.base, validation=validation)

    def runner(self, config, origin=None):
        return CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root),
                                        self.times[302] if origin is None else origin)

    def test_next_day_multi_day_times_and_safe_sample_labels(self):
        runner = self.runner(self.config())
        times = runner.forecast_times(runner.origin)
        self.assertEqual(times[0], pd.Timestamp("2026-01-14 00:00"))
        self.assertEqual(times[-1], pd.Timestamp("2026-01-15 23:00"))
        self.assertEqual(len(times), 48)
        self.assertTrue(all(o.hour == 14 for o in runner.supervised_origins))
        self.assertTrue(all(runner.forecast_times(o)[-1] <= runner.origin for o in runner.supervised_origins))
        np.testing.assert_array_equal(runner.Y_all[-1, :, 0],
            pd.read_csv(self.path, parse_dates=["time"]).set_index("time").loc[
                runner.forecast_times(runner.supervised_origins[-1]), "load"].to_numpy())

    def test_rolling_and_expanding_refit_overlap_and_final_bundle(self):
        for kind in ("rolling", "expanding"):
            with self.subTest(kind=kind):
                config = self.config(kind)
                runner = self.runner(config)
                windows = runner.backtest_windows()
                starts = [w.metadata["raw_history_start"] for w in windows]
                self.assertEqual(starts[0] == starts[1], kind == "expanding")
                counts = [len(w.train_indices) for w in windows]
                self.assertEqual(counts[0] == counts[1], kind == "rolling")
                fitted = []
                original = CanonicalBaseModelRunner.fit
                def traced(context, indices, **kwargs):
                    result = original(context, indices, **kwargs)
                    fitted.append((context.origin, result[-1]))
                    return result
                with patch.object(CanonicalBaseModelRunner, "fit", traced):
                    result = runner.run(self.root / kind)
                self.assertEqual([x[0] for x in fitted], [w.origin for w in windows])
                self.assertIsNot(fitted[0][1], fitted[1][1])
                frame = pd.read_csv(result.test_dir / "cv_plot_df.csv")
                self.assertEqual(len(frame), 96)
                self.assertTrue(frame.time.duplicated().any())
                self.assertIn("forecast_origin", frame)
                self.assertIn("lead_steps", frame)
                self.assertTrue(list(result.model_dir.rglob("*.pkl")))
                validate_artifacts({"artifacts": artifact_paths(result),
                    "config_fingerprint": config.fingerprint(), "result_identity": result.model_dir.parent.name},
                    require_digests=False)
                # 同一历史时刻独立进行生产式final fit，必须与该折一致。
                standalone = self.runner(config, windows[0].origin)
                inputs = standalone.final_bundle_inputs()
                _, artifact, _ = standalone.fit_final(inputs[2], inputs[3])
                designs, provider = standalone.forecast_designs(standalone.origin, inputs[0], inputs[1])
                prediction = standalone.predict(artifact, designs, provider,
                    standalone.forecast_times(standalone.origin), inputs[1])
                np.testing.assert_allclose(prediction.values.reshape(-1),
                    frame.loc[frame.window == 1, "predict_value"].to_numpy(), rtol=0, atol=1e-12)

    def test_sampling_and_time_windows_have_distinct_raw_cache_keys(self):
        config = self.config()
        variants = [config, self.config("expanding"), self.config(start="after_origin"),
            replace(config, validation={**dict(config.validation),
                "training": {"origin_sampling": {"stride_steps": 12}}})]
        keys = [compute_raw_design_fingerprint(c, base_dir=self.root, origin=self.times[302], generators={})
                for c in variants]
        self.assertEqual(len(set(keys)), len(variants))

    def test_future_perturbation_does_not_change_training_or_forecast(self):
        config = self.config()
        def predict():
            runner = self.runner(config)
            inputs = runner.final_bundle_inputs()
            _, artifact, _ = runner.fit_final(inputs[2], inputs[3])
            designs, provider = runner.forecast_designs(runner.origin, inputs[0], inputs[1])
            return inputs[3], runner.predict(artifact, designs, provider,
                runner.forecast_times(runner.origin), inputs[1]).values
        before = predict()
        self.frame.loc[303:, "load"] += 100000
        self.frame.to_csv(self.path, index=False)
        after = predict()
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_after_origin_gap_and_sparse_stride_are_not_resampled_twice(self):
        config = self.config(start="after_origin")
        config = replace(config, problem=replace(config.problem, horizon=4),
            validation={**dict(config.validation), "forecast_window": {"start": "after_origin", "gap_steps": 2},
                        "training": {"origin_sampling": {"stride_steps": 12, "anchor_time": "2026-01-13T14:00:00"}}})
        runner = self.runner(config)
        self.assertEqual(runner.forecast_times(runner.origin)[0], pd.Timestamp("2026-01-13 17:00"))
        self.assertEqual(runner.forecast_times(runner.origin)[-1], pd.Timestamp("2026-01-13 20:00"))
        fit = runner.fit(tuple(range(len(runner.supervised_origins))))
        self.assertEqual(len(fit[3]), len(runner.supervised_origins))
        self.assertTrue(all(o.hour in (2, 14) for o in runner.supervised_origins))
        self.assertTrue(all(b - a == pd.Timedelta(hours=12)
                            for a, b in zip(runner.supervised_origins, runner.supervised_origins[1:])))
        contiguous = replace(config, validation={**dict(config.validation),
            "forecast_window": {"start": "after_origin"}})
        indexed = self.runner(contiguous)
        indexed.prepare_training()
        designs, labels = indexed.builder.training_row(indexed.supervised_origins[0])
        self.assertEqual(indexed.training_compile["mode"], "indexed")
        for actual, expected in zip(indexed.X_all, designs):
            np.testing.assert_allclose(actual[0:1], expected, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(indexed.Y_all[0], labels)

    def test_invalid_temporal_contracts_fail_before_io(self):
        config = self.config()
        for changes in ({"forecast_window": None}, {"forecast_window": {"start": "wrong"}},
                        {"forecast_window": {"start": "next_day", "gap_steps": 2}},
                        {"training_window": {"kind": "rolling", "history_steps": True}},
                        {"training_window": {"kind": "expanding"}},
                        {"training_window": {"kind": "expanding", "start_time": "bad"}},
                        {"train_window_steps": 3}, {"refit_every": 2}):
            with self.subTest(changes=changes), self.assertRaises((ValueError, TypeError)):
                replace(config, validation={**dict(config.validation), **changes})
