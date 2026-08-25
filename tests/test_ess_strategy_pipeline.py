import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config.aidc_ess_selfuse_load.strategy_features.contracts import (
    FORBIDDEN_FUTURE_PATTERNS,
    JOINT_CLUSTER_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
)
from config.aidc_ess_selfuse_load.strategy_features.pipeline import (
    build_strategy_features,
)

ESS_SAFE_LAGS = [288, 576, 864, 1152, 1440, 1728, 2016]


class EssStrategyPipelineTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name) / "data"
        self.data_root.mkdir(parents=True)
        self.data_start = pd.Timestamp("2026-01-01 00:00:00")
        self.as_of_time = pd.Timestamp("2026-01-18 23:55:00")
        self.future_start = pd.Timestamp("2026-01-19 00:00:00")
        self.history_index = pd.date_range(
            self.data_start, self.as_of_time, freq="5min"
        )
        self.future_index = pd.date_range(
            self.future_start, periods=288, freq="5min"
        )
        self.plan_head_index = pd.date_range(
            self.data_start - pd.Timedelta(hours=2), periods=24, freq="5min"
        )
        self.plan_tail_index = pd.date_range(
            self.future_start + pd.Timedelta(days=1), periods=288, freq="5min"
        )
        self.all_plan_index = (
            self.plan_head_index.append(self.history_index)
            .append(self.future_index)
            .append(self.plan_tail_index)
        )
        self.config_path = Path(self.temp_dir.name) / "strategy_features.yaml"
        self._write_inputs()
        self._write_config()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _target_values(self, index):
        slots = (index.hour * 60 + index.minute) // 5
        days = (index.normalize() - self.data_start.normalize()).days
        return 1000.0 + days * 10.0 + slots.astype(float)

    def _actual_values(self, index):
        slots = (index.hour * 60 + index.minute) // 5
        values = np.where(slots < 72, -2000.0, np.where(slots < 144, 0.0, 6000.0))
        return values + (index.day % 3) * 10.0

    def _plan_values(self, index):
        slots = (index.hour * 60 + index.minute) // 5
        return np.where(slots < 72, -3000.0, np.where(slots < 180, 0.0, 7000.0))

    def _write_inputs(self):
        for route, offset in (("A", 0.0), ("B", 100.0)):
            target = pd.DataFrame(
                {
                    "time": self.history_index,
                    "value": self._target_values(self.history_index) + offset,
                }
            )
            endogenous = pd.DataFrame(
                {
                    "time": self.history_index,
                    "ess_power": self._target_values(self.history_index) + offset,
                    "pcs_power": self._actual_values(self.history_index) + offset,
                }
            )
            plan = pd.DataFrame(
                {
                    "time": self.all_plan_index,
                    "pcs_plan": self._plan_values(self.all_plan_index) + offset,
                }
            )
            target.to_csv(self.data_root / f"{route}_target.csv", index=False)
            endogenous.to_csv(self.data_root / f"{route}_endogenous.csv", index=False)
            plan.to_csv(self.data_root / f"{route}_plan.csv", index=False)

    def _write_config(self):
        config = {
            "logic_version": 2,
            "data_start": self.data_start.isoformat(),
            "as_of_time": self.as_of_time.isoformat(),
            "forecast_steps": 288,
            "freq": "5min",
            "points_per_day": 288,
            "calendar_day": {"start_hour": 0},
            "dispatch_cycle": {"start_hour": 22},
            "states": {
                "actual_charge_threshold_kw": -1500.0,
                "actual_discharge_threshold_kw": 5000.0,
            },
            "similar_day": {
                "lookback_days": 180,
                "k_neighbors": 5,
                "min_history_days": 14,
                "robust_template_days": 7,
                "novelty_low_quantile": 0.75,
                "novelty_high_quantile": 0.95,
                "curve_weight": 0.60,
                "duration_energy_weight": 0.25,
                "transition_weight": 0.15,
                "power_scale_kw": 9000.0,
                "count_scale": 10.0,
                "min_effective_samples": 2.0,
            },
            "joint_clustering": {
                "enabled": True,
                "reference_fit_end": "2026-01-16",
                "pca_variance_ratio": 0.90,
                "candidate_clusters": [2, 3],
                "max_clusters": 5,
                "rare_cluster_min_days": 3,
                "random_state": 42,
                "n_init": 20,
            },
            "routes": {
                route: {
                    "target_path": f"{route}_target.csv",
                    "endogenous_path": f"{route}_endogenous.csv",
                    "plan_path": f"{route}_plan.csv",
                }
                for route in ("A", "B")
            },
        }
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    def _build(self, **kwargs):
        return build_strategy_features(
            self.config_path, data_root=self.data_root, **kwargs
        )

    def test_writes_four_model_files_with_exact_schema_and_audits(self):
        results = self._build(force=True)
        output_dir = self.data_root / "forecasting_data" / "strategy_features"

        for route in ("A", "B"):
            history_path = output_dir / f"model_features_history_{route}.csv"
            future_path = output_dir / f"model_features_future_{route}.csv"
            self.assertTrue(history_path.exists())
            self.assertTrue(future_path.exists())
            history = pd.read_csv(history_path)
            future = pd.read_csv(future_path)
            self.assertEqual(list(history.columns), MODEL_FEATURE_COLUMNS)
            self.assertEqual(list(future.columns), MODEL_FEATURE_COLUMNS)
            self.assertEqual(
                history.drop(columns="time").dtypes.astype(str).tolist(),
                future.drop(columns="time").dtypes.astype(str).tolist(),
            )
            self.assertEqual(len(future), 288)
            self.assertFalse(future.drop(columns="time").isna().any().any())
            forbidden = [
                column
                for column in future.columns
                if any(pattern.search(column) for pattern in FORBIDDEN_FUTURE_PATTERNS)
            ]
            self.assertEqual(forbidden, [])
            self.assertEqual(list(results[route].future.columns), MODEL_FEATURE_COLUMNS)

            audit_dir = output_dir / "audit"
            for name in (
                "calendar_day_quality",
                "dispatch_cycle_summary",
                "similar_day_matches",
                "joint_cluster_assignments_A_fit-20260116" if route == "A" else "joint_cluster_assignments_B_fit-20260116",
            ):
                path = (
                    audit_dir / f"{name}.csv"
                    if name.startswith("joint_cluster_assignments")
                    else audit_dir / f"{name}_{route}.csv"
                )
                self.assertTrue(path.exists())
            artifact_path = (
                output_dir
                / "artifacts"
                / f"joint_cluster_{route}_fit-20260116.joblib"
            )
            self.assertTrue(artifact_path.exists())
            audit_path = audit_dir / f"feature_build_audit_{route}.json"
            self.assertTrue(audit_path.exists())
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(audit["logic_version"], 2)
            self.assertEqual(audit["future_rows"], 288)
            self.assertEqual(audit["coverage_checks"]["future_plan_complete"], True)
            self.assertEqual(
                audit["coverage_checks"]["future_plan_dispatch_cycles_complete"],
                True,
            )
            self.assertEqual(audit["leakage_checks"]["future_sources_at_or_before_as_of"], True)
            self.assertLessEqual(
                pd.Timestamp(audit["leakage_checks"]["future_lag_source_max"]),
                self.as_of_time,
            )
            self.assertEqual(
                sum(audit["fallback_counts"].values()),
                len(pd.date_range(self.data_start.normalize(), self.future_start, freq="1D")),
            )
            self.assertEqual(int(future["plan_is_novel"].sum()), 0)
            self.assertEqual(audit["joint_clustering"]["fit_end"], "2026-01-16T00:00:00")
            self.assertIn(audit["joint_clustering"]["selected_k"], (2, 3))
            self.assertEqual(
                audit["leakage_checks"]["future_joint_source_max"],
                "2026-01-18T00:00:00",
            )
            self.assertEqual(int(future["joint_cluster_feature_ready"].sum()), 288)
            np.testing.assert_allclose(
                future[[f"joint_cluster_lag1_c{i}" for i in range(5)]].sum(axis=1),
                1.0,
            )
            self.assertEqual(
                list(future.columns[-len(JOINT_CLUSTER_FEATURE_COLUMNS) :]),
                JOINT_CLUSTER_FEATURE_COLUMNS,
            )

    def test_plan_and_lags_use_exact_timestamp_alignment(self):
        missing_source = pd.Timestamp("2026-01-10 12:00:00")
        target_path = self.data_root / "A_target.csv"
        target = pd.read_csv(target_path, parse_dates=["time"])
        target = target.loc[target["time"] != missing_source]
        target.to_csv(target_path, index=False)
        actual_missing_source = pd.Timestamp("2026-01-11 13:00:00")
        endogenous_path = self.data_root / "A_endogenous.csv"
        endogenous = pd.read_csv(endogenous_path, parse_dates=["time"])
        endogenous = endogenous.loc[endogenous["time"] != actual_missing_source]
        endogenous.to_csv(endogenous_path, index=False)

        results = self._build(validate_only=True)
        future = results["A"].future.set_index("time")
        plan = pd.read_csv(self.data_root / "A_plan.csv", parse_dates=["time"]).set_index("time")
        pd.testing.assert_series_equal(
            future["pcs_plan"],
            plan.loc[self.future_index, "pcs_plan"],
            check_names=False,
            check_freq=False,
        )

        ready_time = pd.Timestamp("2026-01-19 06:00:00")
        source_time = ready_time - pd.Timedelta(days=1)
        target_full = self._target_values(pd.DatetimeIndex([source_time]))[0]
        actual_full = self._actual_values(pd.DatetimeIndex([source_time]))[0]
        self.assertEqual(future.loc[ready_time, "ess_lag_288"], target_full)
        self.assertEqual(future.loc[ready_time, "pcs_actual_lag_288"], actual_full)
        self.assertEqual(future.loc[ready_time, "lag_feature_ready"], 1)

        missing_lag_time = missing_source + pd.Timedelta(days=1)
        history = results["A"].history.set_index("time")
        self.assertEqual(history.loc[missing_lag_time, "lag_feature_ready"], 0)
        self.assertEqual(history.loc[missing_lag_time, "ess_lag_288"], 0.0)
        self.assertEqual(history.loc[missing_lag_time, "pcs_actual_lag_288"], 0.0)
        self.assertIn(
            actual_missing_source.normalize().isoformat(),
            results["A"].audit["gaps"]["incomplete_actual_days"],
        )

        expected_cycle_start = self.future_start - pd.Timedelta(days=2) + pd.Timedelta(
            hours=22
        )
        summary = results["A"].dispatch_cycle_summary
        expected = summary.loc[summary["cycle_start"] == expected_cycle_start].iloc[0]
        first_future = results["A"].future.iloc[0]
        for state in ("charge", "standby", "discharge"):
            self.assertEqual(
                first_future[f"last_completed_cycle_{state}_hours"],
                expected[f"actual_cycle_{state}_hours"],
            )

    def test_package_exports_phase_one_and_pipeline_apis(self):
        import config.aidc_ess_selfuse_load.strategy_features as package

        for name in (
            "encode_plan_direction",
            "summarize_dispatch_profiles",
            "build_strategy_features",
            "MODEL_FEATURE_COLUMNS",
            "JointClusteringConfig",
        ):
            self.assertIn(name, package.__all__)
            self.assertTrue(hasattr(package, name))

    def test_future_target_and_actual_mutation_do_not_change_features(self):
        baseline = self._build(validate_only=True)
        future_extra = self.future_index
        for route in ("A", "B"):
            target_path = self.data_root / f"{route}_target.csv"
            target = pd.read_csv(target_path, parse_dates=["time"])
            target = pd.concat(
                [
                    target,
                    pd.DataFrame({"time": future_extra, "value": 999999.0}),
                ],
                ignore_index=True,
            )
            target.to_csv(target_path, index=False)

            endogenous_path = self.data_root / f"{route}_endogenous.csv"
            endogenous = pd.read_csv(endogenous_path, parse_dates=["time"])
            endogenous = pd.concat(
                [
                    endogenous,
                    pd.DataFrame(
                        {
                            "time": future_extra,
                            "ess_power": -999999.0,
                            "pcs_power": 888888.0,
                        }
                    ),
                ],
                ignore_index=True,
            )
            endogenous.to_csv(endogenous_path, index=False)

        repeated = self._build(validate_only=True)
        for route in ("A", "B"):
            pd.testing.assert_frame_equal(
                baseline[route].history, repeated[route].history
            )
            pd.testing.assert_frame_equal(
                baseline[route].future, repeated[route].future
            )

    def test_future_plan_missing_raises(self):
        plan_path = self.data_root / "A_plan.csv"
        plan = pd.read_csv(plan_path, parse_dates=["time"])
        plan = plan.loc[plan["time"] != self.future_index[17]]
        plan.to_csv(plan_path, index=False)

        with self.assertRaisesRegex(ValueError, "future plan"):
            self._build(validate_only=True)

    def test_future_dispatch_cycle_missing_tail_raises(self):
        plan_path = self.data_root / "A_plan.csv"
        plan = pd.read_csv(plan_path, parse_dates=["time"])
        plan = plan.loc[plan["time"] != self.plan_tail_index[0]]
        plan.to_csv(plan_path, index=False)

        with self.assertRaisesRegex(ValueError, "future plan dispatch cycle"):
            self._build(validate_only=True)

    def test_config_rejects_partial_natural_day_forecast(self):
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        config["forecast_steps"] = 144
        self.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "forecast_steps"):
            self._build(validate_only=True)

    def test_v2_model_configs_are_pointwise_with_safe_lags(self):
        root = Path(__file__).resolve().parent.parent / "config/aidc_ess_selfuse_load"
        seen = set()
        for route in ("A", "B"):
            for group in ("c0", "c1", "c2", "c3"):
                path = root / f"route_{route}/add_strategy_features/lgbm_usmdp_mean_{group}.yaml"
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
                overrides = loaded["overrides"]
                self.assertEqual(
                    overrides["model_strategy"]["pred_method"],
                    "univariate-single-multistep-direct-pointwise",
                )
                self.assertEqual(overrides["model_strategy"]["predict_type"], "point")
                self.assertTrue(overrides["time_lag_features"]["enable_lags_features"])
                self.assertTrue(overrides["model_strategy"]["align_direct_features_to_target"])
                self.assertEqual(overrides["time_lag_features"]["lags"], ESS_SAFE_LAGS)
                key = (
                    overrides["output"]["scenario_subpath"],
                    overrides["output"]["setting_suffix"],
                )
                self.assertNotIn(key, seen)
                seen.add(key)

    def test_c5_joint_configs_are_testing_only_and_select_joint_columns(self):
        root = Path(__file__).resolve().parent.parent / "config/aidc_ess_selfuse_load"
        for route in ("A", "B"):
            path = (
                root
                / f"route_{route}/add_strategy_features/lgbm_usmdp_mean_c5_joint.yaml"
            )
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            overrides = loaded["overrides"]
            self.assertEqual(
                overrides["output"]["setting_suffix"], "-v2-c5-joint"
            )
            self.assertTrue(overrides["runtime"]["is_testing"])
            self.assertFalse(overrides["runtime"]["is_forecasting"])
            self.assertEqual(
                overrides["model_strategy"]["pred_method"],
                "univariate-single-multistep-direct-pointwise",
            )
            self.assertEqual(overrides["model_strategy"]["predict_type"], "point")
            self.assertTrue(overrides["time_lag_features"]["enable_lags_features"])
            self.assertTrue(overrides["model_strategy"]["align_direct_features_to_target"])
            self.assertEqual(overrides["time_lag_features"]["lags"], ESS_SAFE_LAGS)
            columns = overrides["exogenous_features"]["custom_features"][0][
                "columns"
            ]
            self.assertEqual(
                columns[-len(JOINT_CLUSTER_FEATURE_COLUMNS) :],
                JOINT_CLUSTER_FEATURE_COLUMNS,
            )

    def test_validate_only_writes_nothing_and_overwrite_requires_force(self):
        output_dir = self.data_root / "forecasting_data" / "strategy_features"
        self._build(validate_only=True)
        self.assertFalse(output_dir.exists())

        self._build(force=True)
        with self.assertRaises(FileExistsError):
            self._build()


if __name__ == "__main__":
    unittest.main()
