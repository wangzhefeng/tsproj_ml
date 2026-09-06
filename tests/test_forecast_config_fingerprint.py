# -*- coding: utf-8 -*-
"""Canonical config fingerprint and ensemble parsing tests."""

import copy
from dataclasses import replace
import unittest

from forecasting_core.specs import (
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
    parse_model_config,
)


class CanonicalConfigFingerprintTest(unittest.TestCase):
    def test_decomposition_semantic_revision_is_explicit(self):
        config = self.config()
        self.assertNotIn("decomposition_semantics", config.semantic_payload())
        for method in ("linear", "stl", "mstl"):
            periods = {} if method == "linear" else {"periods": [24] if method == "stl" else [24, 72]}
            features = replace(config.features, transformations={"target": {"decomposition": {"method": method, **periods}}})
            enabled = replace(config, features=features)
            self.assertEqual(enabled.semantic_payload().get("decomposition_semantics"), "component_fit_v2")

    @staticmethod
    def components():
        problem = ForecastProblemSpec(
            time_col="time",
            freq="1h",
            horizon=4,
            targets=("load",),
            training_scope="local",
            series_id_cols=(),
        )
        data = DataSpec(
            (
                DataSourceSpec(
                    name="target",
                    source_type="file",
                    columns=(ColumnSpec("load", "target"),),
                    history_path="dataset/load.csv",
                    time_col="time",
                    availability="source_time",
                ),
            )
        )
        features = FeatureSpec(
            target_lags={"load": (1, 2)},
            observed_past_lags={},
            datetime_features=("hour",),
            transformations={},
        )
        estimator = EstimatorSpec(
            model_type="ridge",
            target_adapter="independent",
            params={"alpha": 1.0},
        )
        return problem, data, features, estimator

    def config(self, *, alpha=1.0, validation=None, output=None):
        problem, data, features, estimator = self.components()
        estimator = EstimatorSpec(
            model_type="ridge",
            target_adapter="independent",
            params={"alpha": alpha},
        )
        return ForecastConfigSpec(
            problem=problem,
            data=data,
            features=features,
            strategy=ForecastStrategySpec("direct"),
            estimator=estimator,
            probabilistic={"mode": "point"},
            validation=validation or {
                "history_steps": 48,
                "train_window_steps": 24,
                "fold_count": 3,
                "stride_steps": 4,
            },
            output=output or {},
        )

    def test_output_and_key_order_do_not_change_fingerprint(self):
        first = self.config(
            output={"results_root": "results/a", "scenario_subpath": "case-a"}
        )
        second = self.config(
            output={"scenario_subpath": "case-b", "results_root": "results/b"}
        )

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(len(first.fingerprint()), 64)

    def test_any_forecast_semantic_change_changes_fingerprint(self):
        first = self.config(alpha=1.0)
        second = self.config(alpha=2.0)

        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_training_and_evaluation_identity_changes_fingerprint(self):
        baseline = self.config()
        for field, value in (
            ("history_steps", 49),
            ("train_window_steps", 25),
            ("fold_count", 4),
            ("stride_steps", 5),
        ):
            with self.subTest(field=field):
                validation = dict(baseline.validation)
                validation[field] = value
                self.assertNotEqual(
                    baseline.fingerprint(),
                    self.config(validation=validation).fingerprint(),
                )

    def test_only_nonsemantic_runtime_controls_are_excluded(self):
        first = self.config(
            validation={
                "history_steps": 48,
                "train_window_steps": 24,
                "fold_count": 3,
                "stride_steps": 4,
                "performance": {
                    "window_parallel_workers": 1,
                    "total_thread_limit": 2,
                },
            }
        )
        second = self.config(
            validation={
                "stride_steps": 4,
                "fold_count": 3,
                "train_window_steps": 24,
                "history_steps": 48,
                "performance": {
                    "window_parallel_workers": 8,
                    "total_thread_limit": 8,
                },
            }
        )

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertRegex(first.result_identity(), r"^direct-ridge-local-k1-[0-9a-f]{12}$")

    def test_omitted_and_only_nonsemantic_performance_have_same_fingerprint(self):
        baseline = self.config()
        with_performance = dict(baseline.validation)
        with_performance["performance"] = {
            "window_parallel_workers": 2,
            "model_thread_count": 4,
        }

        configured = self.config(validation=with_performance)

        self.assertEqual(baseline.fingerprint(), configured.fingerprint())
        self.assertEqual(baseline.result_identity(), configured.result_identity())

    def test_data_probabilistic_transform_and_sample_weight_are_semantic(self):
        baseline = self.config()
        mutations = []

        source = baseline.canonical_payload()
        source["data"]["sources"][0]["history_path"] = "dataset/other.csv"
        mutations.append(source)

        probabilistic = baseline.canonical_payload()
        probabilistic["probabilistic"] = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
        mutations.append(probabilistic)

        transform = baseline.canonical_payload()
        transform["features"]["transformations"] = {
            "target": {"decomposition": {"method": "stl", "period": 24}}
        }
        mutations.append(transform)

        sample_weight = baseline.canonical_payload()
        sample_weight["validation"]["training"] = {
            "sample_weight": {
                "method": "time_decay",
                "halflife_days": 24,
            }
        }
        mutations.append(sample_weight)

        for index, payload in enumerate(mutations):
            with self.subTest(index=index):
                changed = parse_model_config(copy.deepcopy(payload), source="changed.yaml")
                self.assertNotEqual(baseline.fingerprint(), changed.fingerprint())

    def test_ensemble_config_uses_members_instead_of_blend_strategy(self):
        """v4: ensemble-shaped YAML parses to EnsembleConfigSpec, not a spec."""
        from model_ensemble.loader import parse_ensemble_document

        payload = {
            "schema_version": 2,
            "problem": {
                "time_col": "time",
                "freq": "1h",
                "horizon": 4,
                "targets": ["load"],
                "training_scope": "local",
                "series_id_cols": [],
            },
            "data": {
                "sources": [
                    {
                        "name": "target_history",
                        "source_type": "file",
                        "columns": [{"name": "load", "role": "target", "categorical": False}],
                        "history_path": "dataset/load.csv",
                        "time_col": "time",
                        "series_id_cols": [],
                        "availability": "source_time",
                    }
                ]
            },
            "probabilistic": {"mode": "point"},
            "ensemble": {
                "members": [
                    {"name": "direct", "config_ref": "direct.yaml"},
                    {"name": "recursive", "config_ref": "recursive.yaml"},
                ],
                "oof": {"train_window_steps": 8, "fold_count": 2, "stride_steps": 1},
                "method": {"name": "averaging"},
            },
            "validation": {},
            "output": {},
        }

        config = parse_ensemble_document(payload, source_path="model_ensemble.yaml")

        self.assertEqual(
            [member.name for member in config.members],
            ["direct", "recursive"],
        )
        self.assertEqual(config.method.name, "averaging")
        self.assertEqual(config.oof.fold_count, 2)

    def test_parser_rejects_ensemble_shape_in_single_model_parser(self):
        """v4 §5.1: parse_model_config rejects ensemble-shaped documents."""
        payload = {
            "schema_version": 2,
            "problem": {
                "time_col": "time",
                "freq": "1h",
                "horizon": 4,
                "targets": ["load"],
                "training_scope": "local",
                "series_id_cols": [],
            },
            "data": {
                "sources": [
                    {
                        "name": "target",
                        "source_type": "file",
                        "columns": [{"name": "load", "role": "target"}],
                        "history_path": "dataset/load.csv",
                        "time_col": "time",
                        "availability": "source_time",
                    }
                ]
            },
            "features": {
                "target_lags": {"load": [1]},
                "observed_past_lags": {},
                "datetime_features": [],
                "transformations": {},
            },
            "strategy": None,
            "ensemble": {
                "members": [
                    {"name": "direct", "strategy": {"name": "direct"}},
                    {"name": "recursive", "strategy": {"name": "recursive"}},
                ],
                "method": "weighted",
                "weights": [0.5, 0.5],
            },
            "estimator": {
                "model_type": "ridge",
                "target_adapter": "independent",
                "params": {},
            },
            "probabilistic": {"mode": "point"},
            "validation": {},
            "output": {},
        }

        with self.assertRaises(ValueError):
            parse_model_config(payload, source="model_ensemble.yaml")


if __name__ == "__main__":
    unittest.main()
