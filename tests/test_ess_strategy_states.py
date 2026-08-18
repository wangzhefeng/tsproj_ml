import unittest

import pandas as pd

from config.aidc_ess_selfuse_load.strategy_features.states import (
    OperatingThresholds,
    encode_actual_operating_state,
    encode_plan_direction,
)


class EssStrategyStatesTest(unittest.TestCase):
    def test_plan_direction_is_exact_one_hot(self):
        encoded = encode_plan_direction(pd.Series([-1.0, 0.0, 1.0]))

        self.assertEqual(
            encoded.to_dict("records"),
            [
                {
                    "plan_direction_charge": 1,
                    "plan_direction_standby": 0,
                    "plan_direction_discharge": 0,
                },
                {
                    "plan_direction_charge": 0,
                    "plan_direction_standby": 1,
                    "plan_direction_discharge": 0,
                },
                {
                    "plan_direction_charge": 0,
                    "plan_direction_standby": 0,
                    "plan_direction_discharge": 1,
                },
            ],
        )
        self.assertEqual(
            encoded.columns.tolist(),
            [
                "plan_direction_charge",
                "plan_direction_standby",
                "plan_direction_discharge",
            ],
        )
        self.assertTrue((encoded.sum(axis=1) == 1).all())

    def test_actual_operating_state_keeps_threshold_boundaries_standby(self):
        encoded = encode_actual_operating_state(
            pd.Series([-1500.1, -1500.0, 5000.0, 5000.1])
        )

        self.assertEqual(
            encoded.to_dict("records"),
            [
                {
                    "actual_operating_charge": 1,
                    "actual_operating_standby": 0,
                    "actual_operating_discharge": 0,
                },
                {
                    "actual_operating_charge": 0,
                    "actual_operating_standby": 1,
                    "actual_operating_discharge": 0,
                },
                {
                    "actual_operating_charge": 0,
                    "actual_operating_standby": 1,
                    "actual_operating_discharge": 0,
                },
                {
                    "actual_operating_charge": 0,
                    "actual_operating_standby": 0,
                    "actual_operating_discharge": 1,
                },
            ],
        )
        self.assertEqual(
            encoded.columns.tolist(),
            [
                "actual_operating_charge",
                "actual_operating_standby",
                "actual_operating_discharge",
            ],
        )
        self.assertTrue((encoded.sum(axis=1) == 1).all())

    def test_actual_operating_thresholds_are_configurable(self):
        thresholds = OperatingThresholds(charge_power=-10.0, discharge_power=20.0)
        encoded = encode_actual_operating_state(
            pd.Series([-11.0, -10.0, 20.0, 21.0]),
            thresholds=thresholds,
        )

        self.assertEqual(encoded["actual_operating_charge"].tolist(), [1, 0, 0, 0])
        self.assertEqual(encoded["actual_operating_standby"].tolist(), [0, 1, 1, 0])
        self.assertEqual(encoded["actual_operating_discharge"].tolist(), [0, 0, 0, 1])

    def test_power_encoders_reject_nan_and_non_numeric_values(self):
        cases = {
            "plan_nan": (encode_plan_direction, pd.Series([0.0, float("nan")])),
            "plan_text": (encode_plan_direction, pd.Series([0.0, "invalid"])),
            "plan_inf": (encode_plan_direction, pd.Series([0.0, float("inf")])),
            "actual_nan": (
                encode_actual_operating_state,
                pd.Series([0.0, float("nan")]),
            ),
            "actual_inf": (
                encode_actual_operating_state,
                pd.Series([0.0, float("-inf")]),
            ),
            "actual_text": (
                encode_actual_operating_state,
                pd.Series([0.0, "invalid"]),
            ),
        }
        for name, (encoder, values) in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    encoder(values)


if __name__ == "__main__":
    unittest.main()
