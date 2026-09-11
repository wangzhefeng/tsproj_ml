"""原生 ETS 真实合成序列测试；不是联通业务模型效果证据。"""
import importlib
import importlib.util
import pickle
import unittest

import numpy as np
import pandas as pd


class NativeETSTest(unittest.TestCase):
    def test_real_five_minute_period_and_all_candidate_failures(self):
        from unittest.mock import patch
        from models.wrappers.ets import ETSModel
        times = pd.date_range('2026-08-01', periods=4032, freq='5min')
        rng = np.random.default_rng(2026)
        history = pd.Series(100 + np.sin(np.arange(4032) * 2 * np.pi / 288) + rng.normal(0, .03, 4032), index=times)
        model = ETSModel({'seasonal_periods': 288})
        model.fit_history(history, as_of=times[-1], freq='5min')
        self.assertEqual(model.forecast(288).shape, (288,))
        self.assertEqual(model.execution_evidence()['history_count'], 4032)
        self.assertEqual(model.execution_evidence()['seasonal_periods'], 288)
        with patch('models.wrappers.ets.StatsmodelsETS', side_effect=ValueError('explicit candidate failure')):
            with self.assertRaisesRegex(ValueError, 'all ETS candidates failed'):
                model.fit_history(history, as_of=times[-1], freq='5min')
        self.assertEqual(len(model.execution_evidence()['candidates']), 3)
        self.assertTrue(all(c['status'] == 'failed' for c in model.execution_evidence()['candidates']))
        with self.assertRaisesRegex(ValueError, 'not fitted'):
            model.forecast(288)

    def test_catalog_native_contract_and_strict_parameters(self):
        from models.catalog import MODEL_CATALOG
        self.assertIn("ets", MODEL_CATALOG)
        self.assertTrue(MODEL_CATALOG["ets"].native_history)
        cls = importlib.import_module("models.wrappers.ets").ETSModel
        for params in ({"typo": 1}, {"seasonal_periods": True},
                       {"maxiter": 1001}, {"selection": "auto"},
                       {"candidates": []}, {"candidates": ["ANA", "ANA"]},
                       {"candidates": ["MAM"]}):
            with self.subTest(params=params), self.assertRaises((ValueError, TypeError)):
                cls(params)

    def test_history_rejection_and_failed_refit_does_not_reuse_state(self):
        cls = importlib.import_module("models.wrappers.ets").ETSModel
        times = pd.date_range("2026-01-01", periods=60, freq="1h")
        history = pd.Series(100 + np.sin(np.arange(60)), index=times)
        model = cls({"seasonal_periods": 12, "candidates": ["ANN"]})
        model.fit_history(history, as_of=times[-1], freq="1h")
        for bad in (history.iloc[::-1], history.iloc[:-1], history.drop(times[5]),
                    history.iloc[:5], history.to_frame(),
                    pd.concat([history, history.iloc[-1:]]), history * np.nan):
            with self.subTest(kind=type(bad).__name__), self.assertRaises((ValueError, TypeError)):
                model.fit_history(bad, as_of=times[-1], freq="1h")
        with self.assertRaisesRegex(ValueError, "not fitted"):
            model.forecast(12)

    def test_native_history_fit_and_pickle_roundtrip(self):
        # 动态导入让 RED 明确定位缺失的原生序列实现。
        self.assertIsNotNone(importlib.util.find_spec("models.wrappers.ets"),
                             "native ETS implementation is missing")
        cls = importlib.import_module("models.wrappers.ets").ETSModel
        times = pd.date_range("2026-01-01", periods=120, freq="1h")
        x = np.arange(len(times))
        history = pd.Series(100 + 2 * np.sin(x * 2 * np.pi / 12) + 0.01 * x,
                            index=times)
        model = cls({"seasonal_periods": 12, "candidates": ["ANN", "ANA"],
                     "selection": "bic", "maxiter": 300})
        model.fit_history(history, as_of=times[-1], freq="1h")
        prediction = model.forecast(12)
        self.assertEqual(prediction.shape, (12,))
        self.assertTrue(np.isfinite(prediction).all())
        evidence = model.execution_evidence()
        self.assertEqual(evidence["history_count"], len(history))
        self.assertEqual(evidence["history_start"], times[0].isoformat())
        self.assertEqual(evidence["history_end"], times[-1].isoformat())
        self.assertEqual(len(evidence["candidates"]), 2)
        eligible = [c for c in evidence["candidates"] if c["status"] == "converged"]
        self.assertTrue(eligible)
        self.assertEqual(evidence["selected"], min(eligible, key=lambda c: c["score"])["name"])
        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_array_equal(prediction, restored.forecast(12))
        self.assertEqual(evidence, restored.execution_evidence())


if __name__ == "__main__":
    unittest.main()
