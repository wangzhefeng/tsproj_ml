import unittest

import numpy as np
import pandas as pd

from config.aidc_ess_selfuse_load.strategy_features.similar_day import (
    NaturalDayPlan,
    SimilarDayConfig,
    build_natural_day_plan,
    estimate_similar_day_template,
    plan_distance,
)


SLOTS_PER_DAY = 288


def _day(value):
    return pd.Timestamp(value).normalize()


def _plan(charge_start=0, charge_slots=48, discharge_start=144, discharge_slots=48):
    values = np.zeros(SLOTS_PER_DAY, dtype=float)
    values[charge_start : charge_start + charge_slots] = -4500.0
    values[discharge_start : discharge_start + discharge_slots] = 9000.0
    return values


def _series(value):
    return np.full(SLOTS_PER_DAY, float(value))


class SimilarDayConfigTest(unittest.TestCase):
    def test_defaults_match_approved_plan(self):
        config = SimilarDayConfig()

        self.assertEqual(config.lookback_days, 180)
        self.assertEqual(config.k_neighbors, 5)
        self.assertEqual(config.min_history_days, 14)
        self.assertEqual(config.robust_template_days, 7)
        self.assertEqual((config.q75, config.q95), (0.75, 0.95))
        self.assertEqual(
            (
                config.curve_weight,
                config.duration_energy_weight,
                config.transition_weight,
            ),
            (0.60, 0.25, 0.15),
        )
        self.assertEqual(config.power_scale, 9000.0)
        self.assertEqual(config.count_scale, 10.0)
        self.assertEqual(config.min_effective_samples, 2.0)

    def test_validation_rejects_invalid_values(self):
        invalid_configs = [
            {"lookback_days": 0},
            {"k_neighbors": 0},
            {"min_history_days": 0},
            {"robust_template_days": 0},
            {"q75": 0.95, "q95": 0.75},
            {"curve_weight": 0.5},
            {"power_scale": 0.0},
            {"count_scale": 0.0},
            {"min_effective_samples": 0.0},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    SimilarDayConfig(**kwargs)


class NaturalDayPlanTest(unittest.TestCase):
    def test_representation_contains_scaled_curve_duration_energy_and_transitions(self):
        plan = np.zeros(SLOTS_PER_DAY, dtype=float)
        plan[:72] = -9000.0
        plan[144:216] = 4500.0

        representation = build_natural_day_plan(plan)

        np.testing.assert_allclose(representation.curve[:72], -1.0)
        np.testing.assert_allclose(representation.curve[144:216], 0.5)
        np.testing.assert_allclose(
            representation.duration_energy,
            [6 / 24, 12 / 24, 6 / 24, 6 / 24, 3 / 24],
        )
        self.assertEqual(representation.transition.shape, (10,))
        np.testing.assert_allclose(representation.transition[:4], [0.1, 0.2, 0.1, 0.3])
        np.testing.assert_allclose(
            representation.transition[4:8], [0.0, 1.0, 0.0, -1.0], atol=1e-12
        )
        np.testing.assert_allclose(representation.transition[8:], [1.0, 1.0])

    def test_representation_requires_exactly_288_finite_signed_values(self):
        for values in (
            np.zeros(287),
            np.zeros(289),
            np.concatenate([np.zeros(287), [np.nan]]),
            np.concatenate([np.zeros(287), [np.inf]]),
        ):
            with self.subTest(length=len(values)):
                with self.assertRaises(ValueError):
                    build_natural_day_plan(values)

    def test_identical_plan_distance_is_zero(self):
        representation = build_natural_day_plan(_plan())

        self.assertEqual(plan_distance(representation, representation), 0.0)

    def test_distance_is_weighted_sum_of_block_rms(self):
        left = NaturalDayPlan(
            curve=np.zeros(2),
            duration_energy=np.zeros(2),
            transition=np.zeros(2),
        )
        right = NaturalDayPlan(
            curve=np.ones(2),
            duration_energy=np.full(2, 2.0),
            transition=np.full(2, 3.0),
        )

        distance = plan_distance(left, right)

        self.assertAlmostEqual(distance, 0.60 * 1.0 + 0.25 * 2.0 + 0.15 * 3.0)


class SimilarDayTemplateTest(unittest.TestCase):
    def _history(self, start="2026-01-01", days=20):
        dates = pd.date_range(start, periods=days, freq="1D")
        plans = {
            day: _plan(discharge_start=120 + index)
            for index, day in enumerate(dates)
        }
        ess = {day: _series(index + 1) for index, day in enumerate(dates)}
        return dates, plans, ess

    def test_candidates_are_strictly_prior_complete_within_lookback_and_not_copied(self):
        target_day = _day("2026-08-01")
        config = SimilarDayConfig(
            lookback_days=10,
            k_neighbors=5,
            min_history_days=2,
            robust_template_days=2,
        )
        plan_history = {
            target_day - pd.Timedelta(days=11): _plan(),
            target_day - pd.Timedelta(days=3): _plan(discharge_start=143),
            target_day - pd.Timedelta(days=2): np.zeros(287),
            target_day - pd.Timedelta(days=1): _plan(discharge_start=145),
            target_day: _plan(),
            target_day + pd.Timedelta(days=1): _plan(),
        }
        ess_history = {
            target_day - pd.Timedelta(days=3): _series(3),
            target_day - pd.Timedelta(days=2): _series(2),
            target_day - pd.Timedelta(days=1): _series(1),
            target_day: _series(99),
            target_day + pd.Timedelta(days=1): _series(100),
        }

        result = estimate_similar_day_template(
            target_day,
            _plan(),
            plan_history,
            ess_history,
            config,
        )

        self.assertTrue(result.ready)
        self.assertEqual(result.method, "blended")
        self.assertEqual(
            [match.day for match in result.matches],
            [target_day - pd.Timedelta(days=3), target_day - pd.Timedelta(days=1)],
        )
        self.assertEqual(len(result.matches), 2)
        self.assertTrue(all(match.distance > 0 for match in result.matches))

    def test_weighted_template_std_effective_samples_and_audit_are_finite(self):
        dates, plans, ess = self._history(days=16)
        target_day = dates[-1] + pd.Timedelta(days=1)
        result = estimate_similar_day_template(
            target_day,
            plans[dates[5]],
            plans,
            ess,
            SimilarDayConfig(min_history_days=14, robust_template_days=7),
        )

        self.assertTrue(result.ready)
        self.assertEqual(len(result.matches), 5)
        self.assertEqual(result.matches[0].day, dates[5])
        self.assertEqual(result.matches[0].distance, 0.0)
        self.assertGreater(result.temperature, 0.0)
        self.assertGreaterEqual(result.n_effective, 1.0)
        self.assertEqual(result.nearest_distance, result.matches[0].distance)
        self.assertAlmostEqual(
            result.knn_mean_distance,
            np.mean([match.distance for match in result.matches]),
        )
        for values in (
            result.template,
            result.similar_template,
            result.similar_std,
            result.robust_template,
        ):
            self.assertEqual(values.shape, (SLOTS_PER_DAY,))
            self.assertTrue(np.isfinite(values).all())

    def test_robust_template_is_pointwise_median_of_last_seven_complete_prior_days(self):
        dates, plans, ess = self._history(days=14)
        ess[dates[-3]] = np.zeros(287)
        target_day = dates[-1] + pd.Timedelta(days=1)

        result = estimate_similar_day_template(
            target_day,
            plans[dates[0]],
            plans,
            ess,
            SimilarDayConfig(min_history_days=7, robust_template_days=7),
        )

        expected_dates = [day for day in dates if day != dates[-3]][-7:]
        expected_value = float(np.median([ess[day][0] for day in expected_dates]))
        np.testing.assert_allclose(result.robust_template, expected_value)

    def test_novelty_thresholds_are_causal_and_gate_blends_to_robust_template(self):
        dates, plans, ess = self._history(days=18)
        target_day = dates[-1] + pd.Timedelta(days=1)
        novel_plan = -_plan()
        config = SimilarDayConfig(min_history_days=14, robust_template_days=7)

        baseline = estimate_similar_day_template(
            target_day, novel_plan, plans, ess, config
        )
        changed_future_plans = dict(plans)
        changed_future_plans[target_day + pd.Timedelta(days=1)] = novel_plan
        changed_future_ess = dict(ess)
        changed_future_ess[target_day + pd.Timedelta(days=1)] = _series(9999)
        repeated = estimate_similar_day_template(
            target_day,
            novel_plan,
            changed_future_plans,
            changed_future_ess,
            config,
        )

        self.assertGreaterEqual(baseline.gate, 0.0)
        self.assertLessEqual(baseline.gate, 1.0)
        self.assertIsNotNone(baseline.novelty_q75)
        self.assertIsNotNone(baseline.novelty_q95)
        np.testing.assert_allclose(baseline.template, repeated.template)
        self.assertEqual(baseline.novelty_q75, repeated.novelty_q75)
        self.assertEqual(baseline.novelty_q95, repeated.novelty_q95)
        expected = (
            (1.0 - baseline.gate) * baseline.similar_template
            + baseline.gate * baseline.robust_template
        )
        np.testing.assert_allclose(baseline.template, expected)

    def test_target_and_future_ess_changes_do_not_leak_into_target_template(self):
        dates, plans, ess = self._history(days=16)
        target_day = dates[-1] + pd.Timedelta(days=1)
        future_day = target_day + pd.Timedelta(days=1)
        plans[future_day] = _plan()
        ess[target_day] = _series(1000)
        ess[future_day] = _series(2000)
        config = SimilarDayConfig(min_history_days=14, robust_template_days=7)

        baseline = estimate_similar_day_template(
            target_day, _plan(), plans, ess, config
        )
        ess[target_day] = _series(-1000)
        ess[future_day] = _series(-2000)
        repeated = estimate_similar_day_template(
            target_day, _plan(), plans, ess, config
        )

        np.testing.assert_allclose(baseline.template, repeated.template)
        self.assertEqual(baseline.matches, repeated.matches)

    def test_future_target_uses_explicit_history_cutoff_not_intermediate_future_ess(self):
        dates, plans, ess = self._history(days=16)
        history_cutoff_day = dates[-1]
        target_day = history_cutoff_day + pd.Timedelta(days=2)
        intermediate_future_day = history_cutoff_day + pd.Timedelta(days=1)
        plans[history_cutoff_day] = _plan()
        plans[intermediate_future_day] = _plan()
        ess[intermediate_future_day] = _series(5000)
        config = SimilarDayConfig(min_history_days=14, robust_template_days=7)

        baseline = estimate_similar_day_template(
            target_day,
            _plan(),
            plans,
            ess,
            config,
            history_cutoff_day=history_cutoff_day,
        )
        ess[intermediate_future_day] = _series(-5000)
        repeated = estimate_similar_day_template(
            target_day,
            _plan(),
            plans,
            ess,
            config,
            history_cutoff_day=history_cutoff_day,
        )

        np.testing.assert_allclose(baseline.template, repeated.template)
        self.assertNotIn(
            intermediate_future_day, [match.day for match in baseline.matches]
        )
        self.assertIn(history_cutoff_day, [match.day for match in baseline.matches])

    def test_readiness_and_robust_fallback_are_explicit_without_fake_distance(self):
        dates, plans, ess = self._history(days=14)
        target_day = dates[-1] + pd.Timedelta(days=1)
        no_plan_history = {}

        fallback = estimate_similar_day_template(
            target_day,
            _plan(),
            no_plan_history,
            ess,
            SimilarDayConfig(min_history_days=14, robust_template_days=7),
        )
        not_ready = estimate_similar_day_template(
            dates[5],
            _plan(),
            plans,
            ess,
            SimilarDayConfig(min_history_days=14, robust_template_days=7),
        )

        self.assertTrue(fallback.ready)
        self.assertEqual(fallback.method, "robust_fallback")
        self.assertEqual(fallback.reason, "no_complete_candidates")
        self.assertEqual(fallback.matches, ())
        self.assertIsNone(fallback.novelty_distance)
        self.assertIsNone(fallback.similar_template)
        self.assertTrue(np.isfinite(fallback.template).all())
        self.assertFalse(not_ready.ready)
        self.assertEqual(not_ready.method, "not_ready")
        self.assertEqual(not_ready.reason, "insufficient_complete_history")
        self.assertIsNone(not_ready.template)


if __name__ == "__main__":
    unittest.main()
