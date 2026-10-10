# -*- coding: utf-8 -*-
"""AIDC 15min 全因子 YAML 生成器合同。"""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.config import parse_model_config
from fixtures.runtime_planning import (
    estimator_params as _runtime_estimator_params,
    fit_worker_plan as _runtime_fit_worker_plan,
    scalar_fit_count as _runtime_scalar_fit_count,
)
from model_training.strategies import target_plan_for_config
from scripts import generate_load_15min_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]


class Load15minFullFactorialMatrixTest(unittest.TestCase):
    def test_axes_match_approved_lgbm_only_design(self):
        # 2026-10-09 估计器收敛为 LightGBM；其余估计器轴从 Git 历史溯源。
        self.assertEqual(
            matrix.MODEL_TYPES,
            {
                "lgbm": "lightgbm",
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
        self.assertFalse(hasattr(matrix, "ENSEMBLE_METHODS"))
        self.assertFalse(hasattr(matrix, "LATIN_GROUPS"))

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
            "baseline": 18,
            "add_exogenous": 54,
            "add_endogenous_cross_route": 18,
            "add_endogenous_state": 18,
            "add_decomposition": 54,
            "add_endogenous_joint": 9,
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
                self.assertEqual(len(configs), 171)
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

    def test_stl_and_mstl_use_supported_polynomial_trend_forecast(self):
        for scenario in matrix.SCENARIOS:
            configs = matrix.build_expected_configs(scenario)
            decomposition_payloads = [
                payload["features"]["transformations"]["target"]["decomposition"]
                for payload in configs.values()
                if "target" in payload.get("features", {}).get("transformations", {})
            ]
            seasonal = [
                payload
                for payload in decomposition_payloads
                if payload["method"] in {"stl", "mstl"}
            ]
            with self.subTest(scenario=scenario):
                self.assertEqual(len(seasonal), 36)
                self.assertEqual(
                    {payload["trend_forecast"] for payload in seasonal},
                    {"polynomial"},
                )

    def test_profiled_lgbm_topologies_use_verified_thread_budgets(self):
        window_model = {
            "window_parallel_workers": 4,
            "model_thread_count": 2,
        }
        output_worker = {
            "window_parallel_workers": 1,
            "multi_output_n_jobs": 8,
            "model_thread_count": 1,
        }
        expected_by_scenario = {
            scenario: matrix.build_expected_configs(scenario)
            for scenario in (
                "aidc_load_15min_daily",
                "aidc_load_15min_rolling",
                "aidc_load_15min_short",
            )
        }
        for scenario, configs in expected_by_scenario.items():
            root = ROOT / "config" / scenario
            for route in ("route_A", "route_B"):
                for filename in (
                    "lgbm_recursive.yaml",
                    "lgbm_direct-pointwise.yaml",
                    "lgbm_direct-pointwise-horizon.yaml",
                ):
                    with self.subTest(
                        scenario=scenario,
                        route=route,
                        filename=filename,
                    ):
                        target = configs[root / route / "baseline" / filename]
                        self.assertEqual(
                            target["validation"]["performance"],
                            window_model,
                        )
                for filename in ("lgbm_direct.yaml", "lgbm_mimo.yaml"):
                    with self.subTest(
                        scenario=scenario,
                        route=route,
                        filename=filename,
                    ):
                        target = configs[root / route / "baseline" / filename]
                        self.assertEqual(
                            target["validation"]["performance"],
                            output_worker,
                        )

        daily = expected_by_scenario["aidc_load_15min_daily"]
        daily_root = ROOT / "config/aidc_load_15min_daily"
        for filename in (
            "lgbm_dirrec.yaml",
            "lgbm_dirmo.yaml",
            "lgbm_dirrecmo.yaml",
            "lgbm_recmo.yaml",
        ):
            path = daily_root / "route_A/baseline" / filename
            with self.subTest(profile="p2", filename=filename):
                self.assertEqual(
                    daily[path]["validation"]["performance"],
                    output_worker,
                )
        # route_B 的四策略（及 rolling 两路）只声明多输出并发档。
        for route in ("route_B",):
            for filename in (
                "lgbm_dirrec.yaml",
                "lgbm_dirmo.yaml",
                "lgbm_dirrecmo.yaml",
                "lgbm_recmo.yaml",
            ):
                path = daily_root / f"{route}/baseline" / filename
                with self.subTest(profile="p2-mo-only", route=route, filename=filename):
                    self.assertEqual(
                        daily[path]["validation"]["performance"],
                        {"multi_output_n_jobs": 8},
                    )

        short = expected_by_scenario["aidc_load_15min_short"]
        short_root = ROOT / "config/aidc_load_15min_short"
        for route in ("route_A", "route_B"):
            for filename in (
                "lgbm_dirrec.yaml",
                "lgbm_dirmo.yaml",
                "lgbm_dirrecmo.yaml",
            ):
                path = short_root / route / "baseline" / filename
                with self.subTest(profile="p3", route=route, filename=filename):
                    self.assertEqual(
                        short[path]["validation"]["performance"],
                        output_worker,
                    )
            # short 的 recmo 块输出更小，使用 4 并发档。
            self.assertEqual(
                short[
                    short_root / route / "baseline/lgbm_recmo.yaml"
                ]["validation"]["performance"],
                {"multi_output_n_jobs": 4},
            )

        lgbm_pointwise_path = short_root / "route_A/baseline/lgbm_direct-pointwise.yaml"
        self.assertEqual(
            short[lgbm_pointwise_path]["validation"]["performance"],
            window_model,
        )
        # 物理 YAML 与生成 payload 的 canonical 一致性用零漂移配置核对
        # （baseline 组磁盘侧存在 temporal 迁移遗留的 performance/字段差异）。
        from config.config_loader import load_yaml_config
        parity_path = (
            short_root / "route_A/add_exogenous/lgbm_direct-pointwise_weather.yaml"
        )
        physical = load_yaml_config(parity_path)
        generated = parse_model_config(short[parity_path], parity_path)
        self.assertEqual(physical.canonical_payload(), generated.canonical_payload())
        assert physical.validation.backtest is not None
        self.assertEqual(physical.validation.backtest.fold_count, 31)

        p4_paths = (
            daily_root / "route_A/add_exogenous/lgbm_direct_holiday-weather.yaml",
            daily_root / "route_A/add_endogenous_state/lgbm_direct.yaml",
            daily_root / "route_AB/add_endogenous_joint/lgbm_direct.yaml",
        )
        for path in p4_paths:
            with self.subTest(profile="p4", path=path):
                payload = daily[path]
                self.assertEqual(
                    payload["validation"]["performance"],
                    output_worker,
                )
                config = parse_model_config(payload, path)
                self.assertEqual(_runtime_fit_worker_plan(config), (1, 8))
                self.assertEqual(_runtime_estimator_params(config)["n_jobs"], 1)

        cross_route_path = (
            daily_root / "route_A/add_endogenous_cross_route/lgbm_recursive.yaml"
        )
        self.assertEqual(
            daily[cross_route_path]["validation"]["performance"],
            {
                "window_parallel_workers": 2,
                "model_thread_count": 4,
            },
        )

        p4_negative_controls = (
            daily_root / "route_A/add_exogenous/lgbm_recursive_holiday.yaml",
            daily_root / "route_B/add_endogenous_state/lgbm_recursive.yaml",
            daily_root / "route_B/add_endogenous_cross_route/lgbm_recursive.yaml",
            daily_root
            / "route_A/add_decomposition/lgbm_direct-pointwise-horizon_decomp-linear.yaml",
        )
        for path in p4_negative_controls:
            with self.subTest(profile="p4-negative", path=path):
                self.assertNotIn("performance", daily[path]["validation"])

    def test_pointwise_variants_have_distinct_horizon_encoding(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        baseline = ROOT / "config/aidc_load_15min_daily/route_A/baseline"
        plain = configs[baseline / "lgbm_direct-pointwise.yaml"]
        cyclic = configs[baseline / "lgbm_direct-pointwise-horizon.yaml"]
        plain_direct = plain["features"]["transformations"]["direct"]
        cyclic_direct = cyclic["features"]["transformations"]["direct"]
        self.assertEqual(plain_direct["layout"], "single_model_horizon")
        self.assertTrue(plain_direct["align_to_target"])
        self.assertFalse(plain_direct["horizon_feature"]["cyclical"])
        self.assertTrue(cyclic_direct["horizon_feature"]["cyclical"])

    def test_pointwise_variants_resolve_one_shared_model(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        baseline = ROOT / "config/aidc_load_15min_daily/route_A/baseline"

        for filename, cyclical in (
            ("lgbm_direct-pointwise.yaml", False),
            ("lgbm_direct-pointwise-horizon.yaml", True),
        ):
            with self.subTest(filename=filename):
                path = baseline / filename
                config = parse_model_config(configs[path], path)
                plan = target_plan_for_config(config)
                direct = config.features.transformations["direct"]

                self.assertEqual(plan.model_count, 1)
                self.assertEqual(plan.model_indices, (0,) * config.problem.horizon)
                self.assertEqual(_runtime_scalar_fit_count(config), 1)
                self.assertEqual(
                    direct["horizon_feature"]["cyclical"],
                    cyclical,
                )

        standard_path = baseline / "lgbm_direct.yaml"
        standard = parse_model_config(configs[standard_path], standard_path)
        standard_plan = target_plan_for_config(standard)
        self.assertEqual(standard_plan.model_count, standard.problem.horizon)
        self.assertEqual(_runtime_scalar_fit_count(standard), standard.problem.horizon)

    def test_short_pointwise_uses_safe_target_lags(self):
        configs = matrix.build_expected_configs("aidc_load_15min_short")
        root = ROOT / "config/aidc_load_15min_short"
        for relative in (
            "route_A/baseline/lgbm_direct-pointwise.yaml",
            "route_A/baseline/lgbm_direct-pointwise-horizon.yaml",
            "route_AB/add_endogenous_joint/lgbm_direct-pointwise.yaml",
        ):
            with self.subTest(config=relative):
                payload = configs[root / relative]
                for lags in payload["features"]["target_lags"].values():
                    self.assertEqual(lags, [16, 96, 192, 672])

    def test_recmo_and_mo_chunks_use_canonical_names(self):
        daily = matrix.build_expected_configs("aidc_load_15min_daily")
        short = matrix.build_expected_configs("aidc_load_15min_short")
        daily_path = ROOT / "config/aidc_load_15min_daily/route_A/baseline/lgbm_recmo.yaml"
        short_path = ROOT / "config/aidc_load_15min_short/route_A/baseline/lgbm_dirmo.yaml"
        self.assertEqual(
            daily[daily_path]["strategy"],
            {"name": "recmo", "output_chunk_length": 24},
        )
        self.assertEqual(
            short[short_path]["strategy"],
            {"name": "dirmo", "output_chunk_length": 4},
        )

    def test_representative_group_payloads_pass_typed_parse(self):
        configs = matrix.build_expected_configs("aidc_load_15min_daily")
        root = ROOT / "config/aidc_load_15min_daily"
        representatives = (
            root / "route_A/baseline/lgbm_recursive.yaml",
            root / "route_A/add_exogenous/lgbm_direct_weather.yaml",
            root / "route_A/add_endogenous_cross_route/lgbm_dirrec.yaml",
            root / "route_A/add_endogenous_state/lgbm_mimo.yaml",
            root / "route_A/add_decomposition/lgbm_recmo_decomp-stl96.yaml",
            root / "route_AB/add_endogenous_joint/lgbm_dirmo.yaml",
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
            root / "route_AB/add_endogenous_joint/lgbm_mimo.yaml"
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
