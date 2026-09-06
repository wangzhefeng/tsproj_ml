"""纯时间几何测试：不构造、不训练、不调用预测模型。"""
import unittest
import pandas as pd
from types import SimpleNamespace
from model_ensemble.oof import oof_fold_origins
from model_testing.geometry import TimeGeometry, rolling_origin_folds


class IntradayScheduleContractTest(unittest.TestCase):
    def test_historical_folds_align_to_formal_origin(self):
        origins = tuple(pd.date_range("2026-07-01 00:00", "2026-07-31 10:00", freq="15min"))
        geometry = TimeGeometry(pd.tseries.frequencies.to_offset("15min"), 16)
        folds = rolling_origin_folds(
            origins, geometry, history_steps=None, train_window_steps=100,
            fold_count=3, stride_steps=96,
            schedule_origin=pd.Timestamp("2026-07-31 14:00"),
        )
        self.assertEqual([fold.origin for fold in folds], list(pd.date_range("2026-07-28 14:00", periods=3, freq="1D")))
        for fold in folds:
            self.assertLess(geometry.label_end(origins[fold.train_indices[-1]]), geometry.label_start(fold.origin))

    def test_default_selection_is_unchanged(self):
        origins = tuple(pd.date_range("2026-07-01", periods=30, freq="1h"))
        folds = rolling_origin_folds(
            origins, TimeGeometry(pd.tseries.frequencies.to_offset("1h"), 2),
            history_steps=None, train_window_steps=5, fold_count=2, stride_steps=3,
        )
        self.assertEqual([fold.origin for fold in folds], [origins[-4], origins[-1]])

    def test_oof_uses_same_grid_with_outer_cutoff(self):
        origins = tuple(pd.date_range("2026-07-01", "2026-07-31 10:00", freq="15min"))
        geometry = TimeGeometry(pd.tseries.frequencies.to_offset("15min"), 16)
        runner = SimpleNamespace(supervised_origins=origins, geometry=geometry)
        folds = oof_fold_origins(
            runner, fold_count=2, stride_steps=96, train_window_steps=100,
            schedule_origin=pd.Timestamp("2026-07-31 14:00"),
            outer_cutoff_origin=pd.Timestamp("2026-07-30 14:00"),
        )
        self.assertEqual([fold["origin"] for fold in folds], list(pd.date_range("2026-07-28 14:00", periods=2, freq="1D")))


if __name__ == "__main__":
    unittest.main()
