"""原生 Naive/Theta 基线合同测试；不是业务模型效果证据。"""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


def _seasonal_history(periods=720, period=24, seed=2026):
    times = pd.date_range('2026-01-01', periods=periods, freq='1h')
    rng = np.random.default_rng(seed)
    x = np.arange(periods)
    return pd.Series(
        100 + 20 * np.sin(2 * np.pi * x / period) + 0.01 * x
        + rng.normal(0, .02, periods),
        index=times,
    )


class NativeNaiveTest(unittest.TestCase):
    def test_three_modes_match_closed_form_definitions(self):
        from models.wrappers.naive import NaiveModel
        history = _seasonal_history()
        times = history.index
        # naive：末值平推
        model = NaiveModel({"mode": "naive"})
        model.fit_history(history, as_of=times[-1], freq='1h')
        np.testing.assert_allclose(model.forecast(24), np.full(24, history.iloc[-1]))
        # drift：末值 + h·斜率
        model = NaiveModel({"mode": "drift"})
        model.fit_history(history, as_of=times[-1], freq='1h')
        slope = (history.iloc[-1] - history.iloc[0]) / (len(history) - 1)
        np.testing.assert_allclose(
            model.forecast(24), history.iloc[-1] + np.arange(1, 25) * slope
        )
        # seasonal_naive：h=1..m 映射到 y_{n+h-m}（上季同位置）
        model = NaiveModel({"mode": "seasonal_naive", "seasonal_periods": 24})
        model.fit_history(history, as_of=times[-1], freq='1h')
        np.testing.assert_allclose(model.forecast(24), history.iloc[-24:].to_numpy())
        evidence = model.execution_evidence()
        self.assertEqual(evidence["history_count"], len(history))
        self.assertEqual(evidence["mode"], "seasonal_naive")
        self.assertEqual(evidence["seasonal_periods"], 24)

    def test_catalog_native_contract_and_strict_parameters(self):
        from models.catalog import MODEL_CATALOG
        self.assertIn("naive", MODEL_CATALOG)
        self.assertTrue(MODEL_CATALOG["naive"].native_history)
        cls = importlib.import_module("models.wrappers.naive").NaiveModel
        for params in ({"typo": 1}, {"mode": "theta"}, {"seasonal_periods": True},
                       {"seasonal_periods": 1}):
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                cls(params)

    def test_history_rejection_follows_ets_contract(self):
        cls = importlib.import_module("models.wrappers.naive").NaiveModel
        times = pd.date_range("2026-01-01", periods=60, freq="1h")
        history = pd.Series(100 + np.sin(np.arange(60)), index=times)
        model = cls({"mode": "drift"})
        model.fit_history(history, as_of=times[-1], freq="1h")
        for bad in (history.iloc[::-1], history.iloc[:-1], history.drop(times[5]),
                    history.to_frame(),
                    pd.concat([history, history.iloc[-1:]]), history * np.nan):
            with self.subTest(kind=type(bad).__name__), self.assertRaises((ValueError, TypeError)):
                model.fit_history(bad, as_of=times[-1], freq="1h")
        with self.assertRaisesRegex(ValueError, "not fitted"):
            cls({"mode": "naive"}).forecast(12)
        with self.assertRaisesRegex(ValueError, "seasonal_naive"):
            # 历史长度不足季节周期
            cls({"mode": "seasonal_naive", "seasonal_periods": 288}).fit_history(
                history, as_of=times[-1], freq="1h")

    def test_pickle_roundtrip(self):
        self.assertIsNotNone(importlib.util.find_spec("models.wrappers.naive"),
                             "native Naive implementation is missing")
        cls = importlib.import_module("models.wrappers.naive").NaiveModel
        history = _seasonal_history()
        model = cls({"mode": "seasonal_naive", "seasonal_periods": 24})
        model.fit_history(history, as_of=history.index[-1], freq='1h')
        prediction = model.forecast(24)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(prediction, restored.forecast(24))
        self.assertEqual(model.execution_evidence(), restored.execution_evidence())


class NativeThetaTest(unittest.TestCase):
    def test_fit_forecast_and_evidence_on_seasonal_series(self):
        from models.wrappers.theta import ThetaModel
        history = _seasonal_history()
        times = history.index
        model = ThetaModel({"seasonal_periods": 24})
        model.fit_history(history, as_of=times[-1], freq='1h')
        prediction = model.forecast(24)
        self.assertEqual(prediction.shape, (24,))
        self.assertTrue(np.isfinite(prediction).all())
        evidence = model.execution_evidence()
        self.assertEqual(evidence["history_count"], len(history))
        self.assertEqual(evidence["history_start"], times[0].isoformat())
        self.assertEqual(evidence["history_end"], times[-1].isoformat())
        self.assertEqual(evidence["selected"], "theta")
        self.assertEqual(evidence["candidates"][0]["status"], "converged")
        # 季节信号恢复：预测应跟踪季节幅度而非平推
        seasonal_amplitude = 20.0
        self.assertGreater(prediction.max() - prediction.min(), seasonal_amplitude * 0.5)

    def test_catalog_native_contract_and_strict_parameters(self):
        from models.catalog import MODEL_CATALOG
        self.assertIn("theta", MODEL_CATALOG)
        self.assertTrue(MODEL_CATALOG["theta"].native_history)
        cls = importlib.import_module("models.wrappers.theta").ThetaModel
        for params in ({"typo": 1}, {"seasonal_periods": True},
                       {"method": "arima"}, {"theta": 0.5}, {"theta": True},
                       {"deseasonalize": "yes"}):
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                cls(params)

    def test_history_rejection_and_multiplicative_guard(self):
        cls = importlib.import_module("models.wrappers.theta").ThetaModel
        times = pd.date_range("2026-01-01", periods=60, freq="1h")
        history = pd.Series(100 + np.sin(np.arange(60)), index=times)
        model = cls({"deseasonalize": False})
        model.fit_history(history, as_of=times[-1], freq="1h")
        self.assertEqual(model.forecast(12).shape, (12,))
        for bad in (history.iloc[::-1], history.iloc[:-1], history * np.nan):
            with self.subTest(kind="bad-history"), self.assertRaises((ValueError, TypeError)):
                model.fit_history(bad, as_of=times[-1], freq="1h")
        with self.assertRaisesRegex(ValueError, "not fitted"):
            cls({}).forecast(12)
        # multiplicative 遇非正历史直接拒绝
        shifted = history - history.min() - 1.0
        with self.assertRaisesRegex(ValueError, "positive"):
            cls({"method": "multiplicative", "seasonal_periods": 12}).fit_history(
                shifted, as_of=times[-1], freq="1h")

    def test_pickle_roundtrip(self):
        self.assertIsNotNone(importlib.util.find_spec("models.wrappers.theta"),
                             "native Theta implementation is missing")
        cls = importlib.import_module("models.wrappers.theta").ThetaModel
        history = _seasonal_history()
        model = cls({"seasonal_periods": 24})
        model.fit_history(history, as_of=history.index[-1], freq='1h')
        prediction = model.forecast(24)
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(prediction, restored.forecast(24))
        self.assertEqual(model.execution_evidence(), restored.execution_evidence())


if __name__ == "__main__":
    unittest.main()
