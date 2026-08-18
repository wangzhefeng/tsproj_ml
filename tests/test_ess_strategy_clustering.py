import unittest
import warnings

import numpy as np
import pandas as pd

from config.aidc_ess_selfuse_load.strategy_features.joint_clustering import (
    JointClusteringConfig,
    build_joint_lag_features,
    fit_joint_cluster_artifact,
    transform_joint_day,
)


class JointClusteringTest(unittest.TestCase):
    def _daily_views(self, days=30):
        start = pd.Timestamp("2026-01-01")
        slots = np.arange(288, dtype=float)
        ess = {}
        actual = {}
        plan = {}
        for offset in range(days):
            day = start + pd.Timedelta(days=offset)
            group = offset % 3
            phase = group * 24
            ess[day] = 40.0 + 15.0 * np.sin(2 * np.pi * (slots - phase) / 288)
            actual[day] = np.where(
                ((slots + phase) % 288) < 96,
                -2200.0 - group * 300.0,
                np.where(((slots + phase) % 288) < 192, 0.0, 7800.0 + group * 300.0),
            )
            plan[day] = np.where(
                ((slots + phase) % 288) < 96,
                -2300.0 - group * 300.0,
                np.where(((slots + phase) % 288) < 192, 0.0, 8000.0 + group * 300.0),
            )
        return ess, actual, plan

    def test_fit_and_transform_three_view_artifact(self):
        ess, actual, plan = self._daily_views()
        config = JointClusteringConfig(
            pca_variance_ratio=0.90,
            candidate_clusters=(2, 3, 4),
            max_clusters=5,
            rare_cluster_min_days=3,
            random_state=42,
            n_init=20,
        )
        fit_end = pd.Timestamp("2026-01-30")

        artifact = fit_joint_cluster_artifact(
            ess,
            actual,
            plan,
            fit_end=fit_end,
            config=config,
        )
        result = transform_joint_day(
            artifact,
            ess[fit_end],
            actual[fit_end],
            plan[fit_end],
        )

        self.assertIn(artifact.selected_k, config.candidate_clusters)
        self.assertEqual(artifact.fit_end, fit_end)
        self.assertEqual(len(artifact.reference_days), 30)
        self.assertEqual(set(artifact.pcas), {"ess", "actual", "plan"})
        for pca in artifact.pcas.values():
            self.assertGreaterEqual(float(pca.explained_variance_ratio_.sum()), 0.90)
        self.assertGreaterEqual(result.cluster_id, 0)
        self.assertLess(result.cluster_id, artifact.selected_k)
        self.assertGreaterEqual(result.distance, 0.0)
        self.assertTrue(np.isfinite(result.distance))

    def test_reference_fit_ignores_days_after_fit_end(self):
        ess, actual, plan = self._daily_views(days=32)
        config = JointClusteringConfig(candidate_clusters=(2, 3), max_clusters=5)
        fit_end = pd.Timestamp("2026-01-30")
        baseline = fit_joint_cluster_artifact(
            ess, actual, plan, fit_end=fit_end, config=config
        )

        for day in (pd.Timestamp("2026-01-31"), pd.Timestamp("2026-02-01")):
            ess[day] = np.full(288, 999999.0)
            actual[day] = np.full(288, -999999.0)
            plan[day] = np.full(288, 777777.0)
        repeated = fit_joint_cluster_artifact(
            ess, actual, plan, fit_end=fit_end, config=config
        )

        np.testing.assert_allclose(
            baseline.kmeans.cluster_centers_, repeated.kmeans.cluster_centers_
        )
        for view in ("ess", "actual", "plan"):
            np.testing.assert_allclose(
                baseline.scalers[view].mean_, repeated.scalers[view].mean_
            )
        self.assertEqual(repeated.reference_days[-1], fit_end)

    def test_joint_lag_features_use_previous_natural_day_and_mark_missing(self):
        ess, actual, plan = self._daily_views(days=32)
        artifact = fit_joint_cluster_artifact(
            ess,
            actual,
            plan,
            fit_end=pd.Timestamp("2026-01-30"),
            config=JointClusteringConfig(candidate_clusters=(2, 3), max_clusters=5),
        )
        del plan[pd.Timestamp("2026-01-30")]
        grid = pd.date_range("2026-01-31", periods=576, freq="5min")

        features, assignments = build_joint_lag_features(
            grid, artifact, ess, actual, plan
        )

        first_day = features.iloc[:288]
        second_day = features.iloc[288:]
        one_hot = [f"joint_cluster_lag1_c{index}" for index in range(5)]
        self.assertEqual(int(first_day["joint_cluster_feature_ready"].sum()), 0)
        self.assertEqual(float(first_day[one_hot].to_numpy().sum()), 0.0)
        self.assertEqual(int(second_day["joint_cluster_feature_ready"].sum()), 288)
        np.testing.assert_allclose(second_day[one_hot].sum(axis=1).to_numpy(), 1.0)
        self.assertEqual(
            assignments.loc[assignments["target_day"] == pd.Timestamp("2026-02-01"), "source_day"].iloc[0],
            pd.Timestamp("2026-01-31"),
        )

    def test_invalid_curve_and_config_raise(self):
        ess, actual, plan = self._daily_views()
        artifact = fit_joint_cluster_artifact(
            ess,
            actual,
            plan,
            fit_end=pd.Timestamp("2026-01-30"),
            config=JointClusteringConfig(candidate_clusters=(2, 3), max_clusters=5),
        )
        with self.assertRaisesRegex(ValueError, "288"):
            transform_joint_day(
                artifact,
                ess[pd.Timestamp("2026-01-30")][:-1],
                actual[pd.Timestamp("2026-01-30")],
                plan[pd.Timestamp("2026-01-30")],
            )
        with self.assertRaisesRegex(ValueError, "candidate_clusters"):
            JointClusteringConfig(candidate_clusters=(3, 2), max_clusters=5)

    def test_constant_view_has_finite_projection_without_runtime_warning(self):
        ess, actual, plan = self._daily_views()
        plan = {day: np.zeros(288) for day in plan}

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            artifact = fit_joint_cluster_artifact(
                ess,
                actual,
                plan,
                fit_end=pd.Timestamp("2026-01-30"),
                config=JointClusteringConfig(
                    candidate_clusters=(2, 3), max_clusters=5
                ),
            )

        self.assertEqual(int(artifact.pcas["plan"].n_components_), 1)
        self.assertEqual(
            float(artifact.pcas["plan"].explained_variance_ratio_.sum()), 1.0
        )


if __name__ == "__main__":
    unittest.main()
