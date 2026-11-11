import unittest
from pathlib import Path

from config.config_loader import load_yaml_config


ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "config/aidc_load_month"
EXPECTED_PARAMS = {
    "dph_stl7_num_leaves15": ("estimator", "params", {"num_leaves": 15}),
    "dph_stl7_num_leaves7": ("estimator", "params", {"num_leaves": 7}),
    "dph_stl7_min_data5": ("estimator", "params", {"min_data_in_leaf": 5}),
    "dph_stl7_min_data3": ("estimator", "params", {"min_data_in_leaf": 3}),
    "dph_stl7_lr005_ne200": (
        "estimator",
        "params",
        {"learning_rate": 0.05, "n_estimators": 200},
    ),
    "dph_stl7_lookback14": ("decomposition", "trend_lookback", 14),
    "dph_stl7_lookback56": ("decomposition", "trend_lookback", 56),
    "dph_stl7_cycles2": ("decomposition", "seasonal_cycles", 2),
    "dph_stl7_cycles8": ("decomposition", "seasonal_cycles", 8),
}


COMMON_EXPERIMENTS = {
    "dph_stl7_num_leaves15",
    "dph_stl7_num_leaves7",
    "dph_stl7_min_data5",
    "dph_stl7_min_data3",
    "dph_stl7_lr005_ne200",
    "dph_stl7_lookback14",
    "dph_stl7_lookback56",
    "dph_stl7_cycles2",
    "dph_stl7_cycles8",
    "dph_stl7_window90",
    "dph_stl7_weight30",
    "dph_stl7_weight60",
    "dph_stl7_no_expanding",
    "dph_stl7_diff37",
    "dph_stl7_diff37_42_49",
}
ROUTE_EXPERIMENTS = {
    "route_A": {
        "dph_stl7_cycles8_lr005_ne200",
        "dph_stl7_cycles8_num_leaves15",
        "dph_stl7_cycles8_window90",
        "dph_stl7_cycles6",
        "dph_stl7_cycles10",
    },
    "route_B": {
        "dph_stl7_diff37_lr005_ne200",
        "dph_stl7_diff37_no_expanding",
        "dph_linear_diff37",
        "dph_stl7_diff37_pointwise",
        "dph_stl7_diff37_49",
    },
}


class LoadMonthParameterExperimentTest(unittest.TestCase):
    def test_matrix_is_complete_and_route_aligned(self):
        paths = sorted(SCENARIO.glob("route_*/experiments/*.yaml"))
        self.assertEqual(len(paths), 40)
        for route in ("route_A", "route_B"):
            route_paths = sorted((SCENARIO / route / "experiments").glob("*.yaml"))
            self.assertEqual(len(route_paths), 20)
            self.assertEqual(
                {path.stem for path in route_paths},
                COMMON_EXPERIMENTS | ROUTE_EXPERIMENTS[route],
            )
            baseline = load_yaml_config(
                SCENARIO / route / "decomposition/lgbm_direct-pointwise-horizon_decomp_stl7.yaml"
            )
            fingerprints = set()
            for path in route_paths:
                config = load_yaml_config(path)
                self.assertEqual(config.probabilistic.get("mode"), "point")
                expected_decomposition = "decomp-linear" if path.stem == "dph_linear_diff37" else "decomp-stl7"
                self.assertEqual(config.result_decomposition_label(), expected_decomposition)
                self.assertEqual(
                    config.output.get("scenario_subpath"),
                    f"aidc_load_month/{route}/experiments/{path.stem}",
                )
                self.assertNotEqual(config.fingerprint(), baseline.fingerprint())
                fingerprints.add(config.fingerprint())
            self.assertEqual(len(fingerprints), 20)

    def test_each_variant_changes_the_requested_semantics(self):
        for path in sorted(SCENARIO.glob("route_*/experiments/*.yaml")):
            config = load_yaml_config(path)
            name = path.stem
            with self.subTest(path=path.relative_to(ROOT)):
                if name in {
                    "dph_stl7_cycles8_lr005_ne200",
                    "dph_stl7_cycles8_num_leaves15",
                    "dph_stl7_cycles8_window90",
                }:
                    decomposition = config.features.transformations["target"]["decomposition"]
                    self.assertEqual(decomposition["seasonal_cycles"], 8)
                    if name.endswith("lr005_ne200"):
                        self.assertEqual(
                            {key: config.estimator.params[key] for key in ("learning_rate", "n_estimators")},
                            {"learning_rate": 0.05, "n_estimators": 200},
                        )
                    elif name.endswith("num_leaves15"):
                        self.assertEqual(config.estimator.params["num_leaves"], 15)
                    else:
                        self.assertEqual(config.validation.get("train_window_days"), 90)
                elif name in {"dph_stl7_cycles6", "dph_stl7_cycles10"}:
                    expected = 6 if name.endswith("cycles6") else 10
                    decomposition = config.features.transformations["target"]["decomposition"]
                    self.assertEqual(decomposition["seasonal_cycles"], expected)
                elif name in {"dph_stl7_diff37_lr005_ne200", "dph_stl7_diff37_no_expanding"}:
                    advanced = config.features.transformations["advanced"]
                    self.assertEqual(advanced["difference"]["periods"], (37,))
                    if name.endswith("lr005_ne200"):
                        self.assertEqual(config.estimator.params["learning_rate"], 0.05)
                        self.assertEqual(config.estimator.params["n_estimators"], 200)
                    else:
                        self.assertNotIn("expanding", advanced)
                elif name == "dph_linear_diff37":
                    self.assertEqual(config.features.transformations["target"]["decomposition"]["method"], "linear")
                    self.assertEqual(config.features.transformations["advanced"]["difference"]["periods"], (37,))
                elif name == "dph_stl7_diff37_pointwise":
                    direct = config.features.transformations["direct"]
                    self.assertFalse(direct["horizon_feature"]["enabled"])
                    self.assertEqual(config.result_method()["method_label"], "direct-pointwise")
                    self.assertEqual(config.features.transformations["advanced"]["difference"]["periods"], (37,))
                elif name == "dph_stl7_diff37_49":
                    self.assertEqual(config.features.transformations["advanced"]["difference"]["periods"], (37, 49))
                elif name in EXPECTED_PARAMS:
                    section, field, expected = EXPECTED_PARAMS[name]
                    if section == "estimator":
                        actual = config.estimator.params
                    else:
                        actual = config.features.transformations["target"]["decomposition"]
                    if isinstance(expected, dict):
                        for key, value in expected.items():
                            self.assertEqual(actual[key], value)
                    else:
                        self.assertEqual(actual[field], expected)
                elif name == "dph_stl7_window90":
                    self.assertEqual(config.validation.get("train_window_days"), 90)
                elif name in {"dph_stl7_weight30", "dph_stl7_weight60"}:
                    half_life = 30 if name.endswith("30") else 60
                    self.assertEqual(
                        config.validation.get("training", {}).get("sample_weight"),
                        {"method": "exponential", "halflife_days": half_life},
                    )
                elif name == "dph_stl7_no_expanding":
                    self.assertNotIn("expanding", config.features.transformations["advanced"])
                elif name == "dph_stl7_diff37":
                    self.assertEqual(
                        config.features.transformations["advanced"]["difference"]["periods"],
                        (37,),
                    )
                elif name == "dph_stl7_diff37_42_49":
                    self.assertEqual(
                        config.features.transformations["advanced"]["difference"]["periods"],
                        (37, 42, 49),
                    )
                else:
                    self.fail(f"unverified experiment variant: {name}")

    def test_route_winners_are_exact_experiment_copies(self):
        for route, name in (
            ("route_A", "dph_stl7_cycles8_lr005_ne200"),
            ("route_B", "dph_stl7_diff37_lr005_ne200"),
        ):
            experiment = SCENARIO / route / "experiments" / f"{name}.yaml"
            selected = SCENARIO / route / f"{name}.yaml"
            self.assertEqual(selected.read_bytes(), experiment.read_bytes())
            self.assertEqual(
                load_yaml_config(selected).result_identity(),
                load_yaml_config(experiment).result_identity(),
            )


if __name__ == "__main__":
    unittest.main()
