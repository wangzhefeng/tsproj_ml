"""训练原点采样与重训策略：真实小数据链路，不更改回测几何。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import json
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from pipeline.runner import CanonicalBaseModelRunner
from forecasting_core.temporal.sampling import select_training_origins
from tests import test_canonical_runtime_smoke as smoke
from tests.test_raw_history_window import make_config


class TrainingWorkloadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "target.csv"
        self.times = pd.date_range("2026-01-01", periods=52, freq="1h")
        pd.DataFrame({"time": self.times, "load": 100 + np.arange(52, dtype=float) ** 1.5}).to_csv(self.path, index=False)
        self.base = make_config(self.path, strategy="direct")

    def runner(self, config):
        return CanonicalBaseModelRunner(config, SourceRegistry(config.data, self.root), self.times[47])

    def test_clock_sampling_reaches_real_fit(self):
        config = replace(self.base, validation={**dict(self.base.validation),
            "train_history_steps": 44, "train_window_steps": 39,
            "training": {"origin_sampling": {"time_of_day": "12:00"}}})
        runner = self.runner(config)
        fit = runner.fit(tuple(range(len(runner.supervised_origins))))
        self.assertEqual(len(fit[3]), 2)
        audit = runner.execution_evidence(fit[-1], fit[1])["training_workload"]
        self.assertEqual(audit["first_training_origin"], "2026-01-01T12:00:00")
        self.assertEqual(audit["last_training_origin"], "2026-01-02T12:00:00")

    def test_sampling_full_lifecycle_persists_bundle(self):
        validation = {k: v for k, v in dict(self.base.validation).items() if k != "train_history_steps"}
        config = replace(self.base, validation={**validation,
            "training": {"origin_sampling": {"stride_steps": 3}}})
        result = self.runner(config).run(self.root / "full")
        self.assertTrue(result.test_dir.is_dir())
        self.assertTrue(list((self.root / "full").rglob("*.pkl")))

    def test_clock_sampling_and_cap_exact_indices(self):
        origins = tuple(pd.date_range("2026-01-01", periods=96, freq="1h"))
        indices = tuple(range(96))
        self.assertEqual(select_training_origins(origins, indices, None), indices)
        self.assertEqual(select_training_origins(origins, indices, {"time_of_day": "23:00", "max_origins": 2}), (71, 95))
        self.assertEqual(select_training_origins(origins, indices, {"stride_steps": 24}), (23, 47, 71, 95))
        with self.assertRaisesRegex(ValueError, "at least two"):
            select_training_origins(origins, indices, {"time_of_day": "23:55"})

    def test_final_and_fold_use_identical_sampling_and_real_model(self):
        base = smoke.CanonicalRuntimeSmokeTest().build_config(self.path, mode="point", strategy="direct")
        config = replace(base, validation={**dict(base.validation),
            "training": {"origin_sampling": {"stride_steps": 3}}})
        runner = self.runner(config)
        inputs = runner.final_bundle_inputs()
        indices = tuple(range(max(0, len(runner.supervised_origins) - config.validation.backtest.train_window_steps),
                              len(runner.supervised_origins)))
        fitted = runner.fit(indices)
        np.testing.assert_array_equal(inputs[3], fitted[3])
        _, artifact, _ = runner.fit_final(inputs[2], inputs[3])
        self.assertEqual(artifact.training_workload["selected_origins"], len(inputs[3]))
        self.assertEqual(artifact.training_workload["candidate_origins"], len(indices))
        self.assertGreaterEqual(artifact.training_workload["stage_wall_seconds"]["fit_total"], 0)
        designs, provider = runner.forecast_designs(runner.origin, inputs[0], inputs[1])
        prediction = runner.predict(artifact, designs, provider, runner.forecast_times(runner.origin), inputs[1])
        self.assertTrue(np.isfinite(prediction.values).all())

    def test_default_and_explicit_dense_predictions_equal(self):
        plain = self.runner(self.base)
        dense = self.runner(replace(self.base, validation={**dict(self.base.validation), "refit_every": 1,
            "training": {"origin_sampling": {"stride_steps": 1}}}))
        predictions = []
        for runner in (plain, dense):
            fitted = runner.fit(tuple(range(len(runner.supervised_origins))))
            designs, provider = runner.forecast_designs(runner.origin, fitted[0], fitted[1])
            predictions.append(runner.predict(fitted[-1], designs, provider, runner.forecast_times(runner.origin), fitted[1]).values)
        np.testing.assert_array_equal(*predictions)

    def test_sampling_does_not_read_future_labels(self):
        config = replace(self.base, validation={**dict(self.base.validation),
            "training": {"origin_sampling": {"stride_steps": 3}}})
        first = self.runner(config)
        fit1 = first.fit(tuple(range(len(first.supervised_origins))))
        designs, provider = first.forecast_designs(first.origin, fit1[0], fit1[1])
        prediction1 = first.predict(fit1[-1], designs, provider, first.forecast_times(first.origin), fit1[1])
        frame = pd.read_csv(self.path)
        frame.loc[48:, "load"] += 1000000
        frame.to_csv(self.path, index=False)
        second = self.runner(config)
        fit2 = second.fit(tuple(range(len(second.supervised_origins))))
        np.testing.assert_array_equal(fit1[3], fit2[3])
        designs, provider = second.forecast_designs(second.origin, fit2[0], fit2[1])
        prediction2 = second.predict(fit2[-1], designs, provider, second.forecast_times(second.origin), fit2[1])
        np.testing.assert_array_equal(prediction1.values, prediction2.values)

    def test_refit_contract_and_unsupported_modes(self):
        # dev 语义：refit_every 为非负整数（0=只在首折拟合）；refit 复用与
        # target transform 兼容（scaler 随 artifact 冻结），禁 train_history_steps。
        for value in (True, -1, 2.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(self.base, validation={**dict(self.base.validation), "refit_every": value})
        with self.assertRaisesRegex(ValueError, "refit_every"):
            replace(self.base, validation={**dict(self.base.validation), "refit_every": 2})

    def test_training_evidence_has_selected_counts_and_timings(self):
        config = replace(self.base, validation={**dict(self.base.validation),
            "training": {"origin_sampling": {"stride_steps": 3}}})
        runner = self.runner(config)
        fit = runner.fit(tuple(range(len(runner.supervised_origins))))
        audit = runner.execution_evidence(fit[-1], fit[1])["training_workload"]
        self.assertEqual(audit["candidate_origins"], 15)
        self.assertEqual(audit["selected_origins"], 5)
        self.assertEqual(audit["largest_model_rows"], 5)
        for key in ("matrix_materialization", "estimator_fit"):
            self.assertGreaterEqual(audit["stage_wall_seconds"][key], 0)

    def test_sampling_contract_rejects_invalid_values(self):
        for sampling in ({"stride_steps": 0}, {"stride_steps": True}, {"max_origins": 1},
                         {"time_of_day": "24:00"}, {"time_of_day": "8:00"},
                         {"time_of_day": "08:00", "stride_steps": 2}, {"unknown": 1}):
            with self.subTest(sampling=sampling), self.assertRaises(ValueError):
                replace(self.base, validation={**dict(self.base.validation),
                    "training": {"origin_sampling": sampling}})

    def test_stride_selects_training_rows_without_changing_geometry(self):
        config = replace(self.base, validation={**dict(self.base.validation),
            "training": {"origin_sampling": {"stride_steps": 3}}})
        runner = self.runner(config)
        indices = tuple(range(len(runner.supervised_origins)))
        expected = indices[::-3][::-1]
        fit = runner.fit(indices)
        self.assertEqual(len(fit[3]), len(expected))
        np.testing.assert_array_equal(fit[3], runner.Y_all[list(expected)])
        self.assertEqual([w.origin for w in runner.backtest_windows()],
                         [w.origin for w in self.runner(self.base).backtest_windows()])
        self.assertNotEqual(config.fingerprint(), self.base.fingerprint())
