"""分解配置单一解释入口的负向与别名合同。"""
import unittest
from types import SimpleNamespace
from decomposition import resolve_decomposition_spec


class DecompositionValidationTest(unittest.TestCase):
    def test_aliases_are_owned_by_decomposition(self):
        quadratic = resolve_decomposition_spec(SimpleNamespace(decomposition={"method": "quadratic"}))
        damped = resolve_decomposition_spec(SimpleNamespace(decomposition={"method": "damped"}))
        assert quadratic.preset is not None and damped.preset is not None
        self.assertEqual((quadratic.method, quadratic.preset.trend_degree), ("linear", 2))
        self.assertEqual((damped.method, damped.preset.trend_forecast), ("linear", "damped"))

    def test_invalid_types_and_ranges_rejected_before_fit(self):
        for extra in ({"robust": "false"}, {"periods": [24.5]}, {"periods": [True]},
                      {"periods": "24"}, {"trend_degree": 3}, {"trend_degree": True},
                      {"trend_lookback": 0}, {"seasonal_cycles": 0}, {"damping": 0},
                      {"damping": float("nan")}, {"damping": True}):
            with self.subTest(extra=extra):
                with self.assertRaises((ValueError, TypeError)):
                    resolve_decomposition_spec(SimpleNamespace(decomposition={"method": "stl", "periods": [24], **extra}))

    def test_duplicate_periods_rejected(self):
        with self.assertRaises(ValueError):
            resolve_decomposition_spec(SimpleNamespace(decomposition={"method": "mstl", "periods": [24, 24]}))


if __name__ == "__main__":
    unittest.main()
