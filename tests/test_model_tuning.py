"""自动调参配方与候选配置的公共合同。"""
from pathlib import Path
import unittest

import optuna

from model_tuning.specs import TuningSpec
import test_canonical_runtime_smoke as fixtures


def search_payload():
    return {
        "trials": 2, "seed": 0, "metric": "RMSE",
        "holdout_origin": "2026-01-03T23:00:00", "holdout_fold_count": 1,
        "parameters": {
            "estimator.params.alpha": {"type": "float", "low": 0.001, "high": 10.0, "log": True},
            "features.target_lags.load": {"type": "categorical", "choices": [[2, 3], [2, 3, 4]]},
        },
    }


class ModelTuningSpecTest(unittest.TestCase):
    def test_sampling_changes_only_declared_fields_and_does_not_mutate_base(self):
        base = fixtures.CanonicalRuntimeSmokeTest().build_config(Path("unused.csv"), mode="point", strategy="direct")
        original = base.canonical_payload()
        spec = TuningSpec.from_mapping(search_payload())
        trial = optuna.trial.FixedTrial({"estimator.params.alpha": 0.1, "features.target_lags.load": 0})
        candidate, parameters = spec.sample_config(base, trial)
        self.assertEqual(candidate.estimator.params["alpha"], 0.1)
        self.assertEqual(candidate.features.target_lags["load"], (2, 3))
        self.assertEqual(parameters["features.target_lags.load"], [2, 3])
        self.assertEqual(base.canonical_payload(), original)
        self.assertEqual(candidate.data, base.data)
        self.assertEqual(candidate.validation, base.validation)
        self.assertNotEqual(candidate.fingerprint(), base.fingerprint())

    def test_recipe_rejects_unknown_fields_and_selection_leakage_paths(self):
        for path in ("data.sources", "validation.forecast_origin", "output.results_root", "estimator.model_type"):
            recipe = search_payload()
            recipe["parameters"] = {path: {"type": "categorical", "choices": [1, 2]}}
            with self.subTest(path=path), self.assertRaises(ValueError):
                TuningSpec.from_mapping(recipe)
        recipe = search_payload()
        recipe["unknown"] = True
        with self.assertRaises(ValueError):
            TuningSpec.from_mapping(recipe)

    def test_recipe_rejects_invalid_distributions(self):
        for distribution in (
            {"type": "float", "low": float("nan"), "high": 1},
            {"type": "float", "low": 0, "high": 1, "log": True},
            {"type": "int", "low": True, "high": 2},
            {"type": "int", "low": 3, "high": 2},
            {"type": "categorical", "choices": []},
            {"type": "categorical", "choices": [1, 1]},
            {"type": "float", "low": 1, "high": 2, "typo": 3},
        ):
            recipe = search_payload()
            recipe["parameters"] = {"estimator.params.alpha": distribution}
            with self.subTest(distribution=distribution), self.assertRaises((ValueError, TypeError)):
                TuningSpec.from_mapping(recipe)

    def test_feature_path_must_exist_in_base(self):
        recipe = search_payload()
        recipe["parameters"] = {"features.transformations.typo": {"type": "categorical", "choices": [1]}}
        spec = TuningSpec.from_mapping(recipe)
        base = fixtures.CanonicalRuntimeSmokeTest().build_config(Path("unused.csv"), mode="point")
        with self.assertRaisesRegex(ValueError, "existing feature"):
            spec.sample_config(base, optuna.trial.FixedTrial({"features.transformations.typo": 0}))


if __name__ == "__main__":
    unittest.main()
