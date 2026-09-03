# -*- coding: utf-8 -*-
"""AIDC 15min 全因子 YAML 生成器合同。"""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.config import parse_model_config
from model_ensemble.loader import parse_ensemble_document
from model_ensemble.specs import EnsembleConfigSpec
from scripts import generate_load_15min_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


class Load15minFullFactorialMatrixTest(unittest.TestCase):
    def test_axes_match_approved_full_factorial_design(self):
        self.assertEqual(
            matrix.MODEL_TYPES,
            {
                "st": "st",
                "ridge": "ridge",
                "lasso": "lasso",
                "enet": "enet",
                "lgbm": "lightgbm",
                "xgb": "xgboost",
                "cab": "catboost",
                "rf": "randomforest",
                "histgb": "histgb",
            },
        )
        self.assertEqual(
            tuple(matrix.STRATEGY_VARIANTS),
            (
                "direct-pointwise",
                "direct-pointwise-horizon",
                "direct",
                "recursive",
                "dirrec",
                "dirmo",
                "recmo",
                "dirrecmo",
                "mimo",
            ),
        )
        self.assertEqual(
            matrix.EXOGENOUS_VARIANTS,
            ("holiday", "weather", "holiday-weather"),
        )
        self.assertEqual(
            matrix.DECOMPOSITION_VARIANTS,
            ("linear", "mstl96-672", "stl96"),
        )
        self.assertEqual(
            matrix.ENSEMBLE_METHODS,
            ("averaging", "weighted", "linear_blending", "stacking"),
        )

    def test_state_columns_match_pure_same_route_asset_contract(self):
        self.assertEqual(
            matrix.STATE_COLUMNS,
            (
                "state_roll_1h_mean",
                "state_roll_1h_std",
                "state_roll_4h_mean",
                "state_roll_4h_std",
                "state_roll_24h_range",
                "state_diff_15min",
                "state_diff_1h",
                "state_diff_24h_pct",
                "state_robust_z_7d",
                "state_weekly_base_dev_pct",
            ),
        )
        self.assertNotIn("state_route_diff_pct", matrix.STATE_COLUMNS)

    def test_each_scenario_has_approved_group_counts(self):
        expected = {
            "baseline": 162,
            "add_exogenous": 486,
            "add_endogenous_cross_route": 162,
            "add_endogenous_state": 162,
            "add_decomposition": 486,
            "add_ensemble": 24,
            "add_endogenous_joint": 81,
        }
        for scenario in matrix.SCENARIOS:
            with self.subTest(scenario=scenario):
                configs = matrix.build_expected_configs(scenario)
                counts = Counter(
                    "add_endogenous_joint"
                    if "route_AB" in path.parts
                    else path.parent.name
                    for path in configs
                )
                self.assertEqual(len(configs), 1563)
                self.assertEqual(counts, expected)

    def test_scenario_forecast_origins_match_data_cutoffs(self):
        expected = {
            "aidc_load_15min_daily": ("2026-07-31T23:45:00", "daily"),
            "aidc_load_15min_rolling": ("2026-07-31T14:00:00", "intraday"),
            "aidc_load_15min_short": ("2026-07-31T14:00:00", "intraday"),
        }
        for scenario, (origin, schedule_mode) in expected.items():
            with self.subTest(scenario=scenario):
                payloads = matrix.build_expected_configs(scenario)
                sample = next(iter(payloads.values()))
                self.assertEqual(sample["validation"]["forecast_origin"], origin)
                self.assertEqual(
                    sample["validation"]["schedule_mode"], schedule_mode
                )

    def test_profiled_lgbm_recursive_uses_bounded_window_thread_budget(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        root = ROOT / "config/aidc_load_15min_daily"
        target = configs[root / "route_A/baseline/lgbm_recursive.yaml"]

        self.assertEqual(
            target["validation"]["performance"],
            {
                "window_parallel_workers": 2,
                "model_thread_count": 4,
            },
        )
        self.assertNotIn(
            "performance",
            configs[root / "route_B/baseline/lgbm_recursive.yaml"]["validation"],
        )
        self.assertNotIn(
            "performance",
            configs[root / "route_A/baseline/lgbm_direct.yaml"]["validation"],
        )

    def test_pointwise_variants_have_distinct_horizon_encoding(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        baseline = ROOT / "config/aidc_load_15min_daily/route_A/baseline"
        plain = configs[baseline / "cab_direct-pointwise.yaml"]
        cyclic = configs[baseline / "cab_direct-pointwise-horizon.yaml"]
        plain_direct = plain["features"]["transformations"]["direct"]
        cyclic_direct = cyclic["features"]["transformations"]["direct"]
        self.assertEqual(plain_direct["layout"], "single_model_horizon")
        self.assertTrue(plain_direct["align_to_target"])
        self.assertFalse(plain_direct["horizon_feature"]["cyclical"])
        self.assertTrue(cyclic_direct["horizon_feature"]["cyclical"])

    def test_short_pointwise_uses_safe_target_lags(self):
        configs = matrix.build_expected_configs("aidc_load_15min_short")
        root = ROOT / "config/aidc_load_15min_short"
        for relative in (
            "route_A/baseline/lgbm_direct-pointwise.yaml",
            "route_A/baseline/lgbm_direct-pointwise-horizon.yaml",
            "route_AB/add_endogenous_joint/ridge_direct-pointwise.yaml",
        ):
            with self.subTest(config=relative):
                payload = configs[root / relative]
                for lags in payload["features"]["target_lags"].values():
                    self.assertEqual(lags, [16, 96, 192, 672])

    def test_recmo_and_mo_chunks_use_canonical_names(self):
        daily = matrix.build_expected_configs("aidc_load_15min_daily")
        short = matrix.build_expected_configs("aidc_load_15min_short")
        daily_path = ROOT / "config/aidc_load_15min_daily/route_A/baseline/rf_recmo.yaml"
        short_path = ROOT / "config/aidc_load_15min_short/route_A/baseline/rf_dirmo.yaml"
        self.assertEqual(
            daily[daily_path]["strategy"],
            {"name": "recmo", "output_chunk_length": 24},
        )
        self.assertEqual(
            short[short_path]["strategy"],
            {"name": "dirmo", "output_chunk_length": 4},
        )

    def test_latin_ensembles_reference_baseline_members_and_parse(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        root = ROOT / "config/aidc_load_15min_daily/route_A"
        ensemble_path = (
            root / "add_ensemble/ensemble_latin-a_linear-blending.yaml"
        )
        ensemble = configs[ensemble_path]
        self.assertEqual(
            ensemble["ensemble"]["members"],
            [
                {
                    "name": "st_recursive",
                    "config_ref": "../baseline/st_recursive.yaml",
                },
                {
                    "name": "lgbm_mimo",
                    "config_ref": "../baseline/lgbm_mimo.yaml",
                },
                {
                    "name": "ridge_direct",
                    "config_ref": "../baseline/ridge_direct.yaml",
                },
            ],
        )
        self.assertEqual(
            ensemble["ensemble"]["oof"],
            {
                "train_window_steps": 2784,
                "fold_count": 5,
                "stride_steps": 96,
            },
        )
        self.assertEqual(
            ensemble["ensemble"]["method"],
            {"name": "linear_blending"},
        )
        self.assertTrue(
            {"features", "strategy", "estimator"}.isdisjoint(ensemble)
        )
        baseline = configs[root / "baseline/st_recursive.yaml"]
        for key in ("problem", "data", "probabilistic", "validation"):
            self.assertEqual(ensemble[key], baseline[key])
        parsed = parse_ensemble_document(ensemble, source_path=ensemble_path)
        self.assertIsInstance(parsed, EnsembleConfigSpec)

        short = matrix.build_expected_configs("aidc_load_15min_short")
        short_root = ROOT / "config/aidc_load_15min_short/route_B/add_ensemble"
        short_oof = short[
            short_root / "ensemble_latin-c_stacking.yaml"
        ]["ensemble"]["oof"]
        self.assertEqual(short_oof["train_window_steps"], 1424)
        self.assertEqual(short_oof["fold_count"], 7)
        self.assertEqual(short_oof["stride_steps"], 96)

    def test_representative_group_payloads_pass_typed_parse(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        root = ROOT / "config/aidc_load_15min_daily"
        representatives = (
            root / "route_A/baseline/st_recursive.yaml",
            root / "route_A/add_exogenous/ridge_direct_weather.yaml",
            root / "route_A/add_endogenous_cross_route/lgbm_dirrec.yaml",
            root / "route_A/add_endogenous_state/xgb_mimo.yaml",
            root / "route_A/add_decomposition/enet_recmo_decomp-stl96.yaml",
            root / "route_AB/add_endogenous_joint/histgb_dirmo.yaml",
        )
        for path in representatives:
            with self.subTest(config=path.name):
                parsed = parse_model_config(configs[path], source=path)
                self.assertIsInstance(parsed, ForecastConfigSpec)

    def test_group_specific_sources_and_joint_targets_are_explicit(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        root = ROOT / "config/aidc_load_15min_daily"
        holiday = configs[
            root / "route_A/add_exogenous/lgbm_direct_holiday.yaml"
        ]
        weather = configs[
            root / "route_A/add_exogenous/lgbm_direct_weather.yaml"
        ]
        both = configs[
            root / "route_A/add_exogenous/lgbm_direct_holiday-weather.yaml"
        ]
        self.assertEqual(
            [source["name"] for source in holiday["data"]["sources"]],
            ["target_history", "chinese_holiday"],
        )
        self.assertEqual(
            [source["name"] for source in weather["data"]["sources"]],
            ["target_history", "weather"],
        )
        self.assertEqual(
            [source["name"] for source in both["data"]["sources"]],
            ["target_history", "chinese_holiday", "weather"],
        )

        joint = configs[
            root / "route_AB/add_endogenous_joint/histgb_mimo.yaml"
        ]
        self.assertEqual(joint["problem"]["targets"], ["A_load", "B_load"])
        self.assertEqual(
            set(joint["features"]["target_lags"]),
            {"A_load", "B_load"},
        )
        advanced = joint["features"]["transformations"]["advanced"]
        self.assertEqual(
            advanced["rolling"]["columns"],
            ["A_load", "B_load"],
        )
        self.assertEqual(
            advanced["expanding"]["columns"],
            ["A_load", "B_load"],
        )


if __name__ == "__main__":
    unittest.main()
