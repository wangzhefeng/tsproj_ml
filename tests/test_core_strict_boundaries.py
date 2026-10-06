"""合同修复回归：真实 YAML 入口拒绝未消费声明，不依赖模型失败兜底。"""
import tempfile
import pickle
import unittest
from dataclasses import replace
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from forecasting_core.probability.spec import probabilistic_spec_from_mapping
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec
from forecasting_core.bundle import ForecastModelBundle
from forecasting_core.probability.distribution import MarginalForecastDistribution
from forecasting_core.tensors.quantile import MarginalQuantileForecastTensor
from forecasting_core.temporal.origin import resolve_origin
from model_predicting.loops.deployment import attach_bundle_prediction_intervals
from model_predicting.artifacts.persistence import persist_model_bundle
from model_training.weights.temporal import temporal_sample_weight
from tests import test_forecast_config_fingerprint as fixture


class StrictTrainingBoundaryTest(unittest.TestCase):
    def test_weight_options_pass_real_yaml_and_invalid_method_fails_early(self):
        payload = fixture.CanonicalConfigFingerprintTest().config().canonical_payload()
        weight = {"method": "exponential", "halflife_days": 2,
                  "anchor": "latest_origin", "normalization": "sum"}
        payload["validation"]["training"] = {"sample_weight": weight}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            config = load_yaml_config(path)
            times = pd.date_range("2026-01-01", periods=3, freq="D")
            actual = temporal_sample_weight(times, times[-1], config.validation["training"]["sample_weight"])
            expected = np.array([0.5, np.sqrt(0.5), 1.0])
            np.testing.assert_allclose(actual, expected / expected.sum(), rtol=1e-14)
            weight["method"] = "time_decay"
            path.write_text(yaml.safe_dump(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exponential"):
                load_yaml_config(path)

    def test_yaml_rejects_unconsumed_training_options(self):
        baseline = fixture.CanonicalConfigFingerprintTest().config().canonical_payload()
        fields = ("early_stopping_patience", "tuning", "augmentation", "feature_selection",
                  "learning_rate", "huber_delta", "blend_weight_windows", "estimator_ensemble")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            for field in fields:
                with self.subTest(field=field):
                    baseline["validation"]["training"] = {field: {}}
                    path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, field):
                        load_yaml_config(path)
            baseline["validation"].pop("training")
            baseline["validation"]["train_outlier"] = {"method": "none"}
            path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train_outlier"):
                load_yaml_config(path)

    def test_null_training_is_a_config_error(self):
        baseline = fixture.CanonicalConfigFingerprintTest().config().canonical_payload()
        baseline["validation"]["training"] = None
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(yaml.safe_dump(baseline), encoding="utf-8")
            with self.assertRaisesRegex((TypeError, ValueError), "training.*mapping"):
                load_yaml_config(path)


class ProbabilityBoundaryTest(unittest.TestCase):
    def test_point_mode_rejects_quantile_only_fields(self):
        for key, value in (("quantiles", [0.1, 0.5, 0.9]), ("crossing", {"method": "none"}),
                           ("point_quantile", 0.5)):
            for parse in (ProbabilisticConfigSpec.from_mapping, probabilistic_spec_from_mapping):
                with self.subTest(key=key, parser=parse):
                    with self.assertRaisesRegex(ValueError, key):
                        parse({"mode": "point", key: value})

    def test_both_boundaries_reject_lossy_calibration_types(self):
        base = {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9]}
        for field, invalid in (("calibration_windows", 5.9), ("min_windows", True),
                               ("min_scores", "2"), ("label_availability_delay_steps", -0.5),
                               ("allow_interval_shrink", "false")):
            for parse in (ProbabilisticConfigSpec.from_mapping, probabilistic_spec_from_mapping):
                with self.subTest(field=field, parser=parse):
                    payload = {**base, "calibration": {
                        "interval": "q10_q90", "target_coverage": 0.8, field: invalid}}
                    with self.assertRaisesRegex((TypeError, ValueError), field):
                        parse(payload)
        for parse in (ProbabilisticConfigSpec.from_mapping, probabilistic_spec_from_mapping):
            with self.assertRaisesRegex((TypeError, ValueError), "report_raw"):
                parse({**base, "crossing": {"report_raw": "false"}})

    def test_single_quantile_has_no_implicit_interval(self):
        config = ProbabilisticConfigSpec.from_mapping({"mode": "quantile", "quantiles": [0.5]})
        spec = probabilistic_spec_from_mapping(config)
        self.assertEqual(spec.quantiles, (0.5,))
        self.assertEqual(spec.intervals, ())

    def test_yaml_and_runtime_reject_inconsistent_interval(self):
        for parse in (ProbabilisticConfigSpec.from_mapping, probabilistic_spec_from_mapping):
            with self.assertRaisesRegex(ValueError, "present in quantiles"):
                parse({"mode": "quantile", "quantiles": [0.1, 0.5, 0.9], "intervals": [
                    {"name": "band", "lower_quantile": 0.2, "upper_quantile": 0.9}]})

    def test_frozen_nested_config_can_be_parsed_without_mutation(self):
        payload = {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                   "crossing": {"method": "none", "report_raw": False}}
        config = ProbabilisticConfigSpec.from_mapping(payload)
        spec = probabilistic_spec_from_mapping(config)
        self.assertFalse(spec.crossing_report_raw)
        self.assertEqual(config.canonical_payload(), payload)


class SavedCalibrationBoundaryTest(unittest.TestCase):
    def test_pickle_revalidates_saved_calibration_and_legacy_version(self):
        valid = self.bundle()
        restored = pickle.loads(pickle.dumps(valid))
        self.assertEqual(restored.schema_payload(), valid.schema_payload())
        for damage, message in (("correction", "finite"), ("origin", "forecast_origin"),
                                ("version", "schema_version")):
            with self.subTest(damage=damage):
                bundle = self.bundle()
                assert bundle.calibration_state is not None
                if damage == "correction":
                    bundle.calibration_state["correction"] = float("nan")
                elif damage == "origin":
                    bundle.calibration_state.pop("forecast_origin")
                else:
                    bundle.schema_version = 1
                encoded = pickle.dumps(bundle)
                with self.assertRaisesRegex(ValueError, message):
                    pickle.loads(encoded)

    def test_save_rejects_mutated_calibration_before_writing(self):
        bundle = self.bundle()
        assert bundle.calibration_state is not None
        bundle.calibration_state["correction"] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model"
            with self.assertRaisesRegex(ValueError, "finite"):
                persist_model_bundle(bundle, destination)
            self.assertFalse(destination.exists())

    def test_legacy_version_and_invalid_origin_are_rejected(self):
        bundle = self.bundle()
        with self.assertRaisesRegex(ValueError, "schema_version"):
            replace(bundle, schema_version=1, pred_method="usmr")
        with self.assertRaisesRegex(ValueError, "finite"):
            resolve_origin(None, "NaT")

    def bundle(self):
        spec = probabilistic_spec_from_mapping({"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
            "calibration": {"interval": "q10_q90", "target_coverage": 0.8}})
        return ForecastModelBundle(schema_version=2, model=object(), feature_scaler=None,
            target_transform=None, selected_features=("x",), input_schema={"columns": ["x"]},
            probabilistic_spec=spec, model_type="qr", pred_method=None,
            canonical_problem={"freq": "1h"}, strategy_spec={"name": "direct"}, estimator_spec={},
            dimensions=(1, 2, 1), series_ids=("s",), target_order=("y",), training_scope="local",
            result_schema_version=2, config_fingerprint="a" * 64,
            calibration_state={"method": "cqr", "interval": "q10_q90", "target_coverage": 0.8,
                "status": "applied", "correction": 0.5, "forecast_origin": "2026-01-01",
                "selected_windows": 3, "selected_scores": 30})

    def test_saved_state_rejects_nonfinite_mismatch_and_missing_origin(self):
        bundle = self.bundle()
        for updates in ({"correction": float("nan")}, {"correction": float("inf")},
                        {"correction": True}, {"status": "typo"}, {"target_coverage": 0.9},
                        {"forecast_origin": "NaT"}, {"interval": "wrong"},
                        {"selected_scores": 0}):
            with self.subTest(updates=updates):
                with self.assertRaises((ValueError, TypeError)):
                    replace(bundle, calibration_state={**bundle.calibration_state, **updates})
        state = dict(bundle.calibration_state)
        state.pop("forecast_origin")
        with self.assertRaisesRegex(ValueError, "forecast_origin"):
            replace(bundle, calibration_state=state)
        with self.assertRaisesRegex(ValueError, "calibration state"):
            replace(bundle, calibration_state=None)

    def test_deployment_rechecks_mutated_state_before_applying(self):
        bundle = self.bundle()
        quantiles = MarginalQuantileForecastTensor(np.array([[[[1., 2., 3.]], [[2., 3., 4.]]]]),
            levels=(0.1, 0.5, 0.9), point_level=0.5, series_ids=("s",),
            forecast_times=pd.date_range("2026-01-02", periods=2, freq="1h"), targets=("y",))
        distribution = MarginalForecastDistribution(quantiles.point(), quantiles)
        attach_bundle_prediction_intervals(bundle, distribution)
        np.testing.assert_array_equal(distribution.metadata["prediction_intervals"]["q10_q90"]["lower"], [[[0.5], [1.5]]])
        bundle.calibration_state["correction"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            attach_bundle_prediction_intervals(bundle, distribution)
