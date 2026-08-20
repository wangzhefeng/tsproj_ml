# -*- coding: utf-8 -*-
"""Direct/horizon-feature 的目标日外生对齐回归测试。"""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from features.FeatureEngineering import FeatureEngineer


def _engineer(align_direct=False):
    args = SimpleNamespace(
        pred_method="univariate-single-multistep-direct",
        lags=[1, 2],
        enable_lags_features=True,
        align_direct_features_to_target=align_direct,
        enable_date_features=False,
        enable_weather_features=True,
        weather_ts_feat="ts",
        weather_features=["rt_tt2"],
        weather_categorical_features=[],
        enable_datetime_features=True,
        datetime_features=["day_of_week"],
        datetime_categorical_features=[],
        custom_features=[],
        use_horizon_exogenous_for_direct=False,
        enable_global_training=False,
        enable_advanced_features=False,
    )
    return FeatureEngineer(args, "[test]", verbose=False)


class HorizonExogenousExpansionTests(unittest.TestCase):
    def test_col_h_holds_target_day_value(self):
        """行 t 的 col_h(h) 必须等于外生列在 t+h（目标日）的值。"""
        engineer = _engineer()
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=4, freq="1D"),
            "rt_tt2": [10.0, 20.0, 30.0, 40.0],
            "dt_day_of_week": [1.0, 2.0, 3.0, 4.0],
        })
        expanded, features = engineer._expand_horizon_exogenous_for_direct(
            df=df, exogenous_features=["rt_tt2", "dt_day_of_week"], horizon=2,
        )

        # 行 t 的 rt_tt2_h1 应为 rt_tt2(t+1)，rt_tt2_h2 应为 rt_tt2(t+2)
        self.assertEqual(expanded.loc[0, "rt_tt2_h1"], 20.0)
        self.assertEqual(expanded.loc[0, "rt_tt2_h2"], 30.0)
        self.assertEqual(expanded.loc[1, "rt_tt2_h1"], 30.0)
        # datetime 同步对齐目标日
        self.assertEqual(expanded.loc[0, "dt_day_of_week_h1"], 2.0)
        # 帧尾目标日不存在 → NaN（由 dropna 剔除，不得回看原点值）
        self.assertTrue(pd.isna(expanded.loc[3, "rt_tt2_h1"]))
        self.assertEqual(
            features,
            ["rt_tt2", "dt_day_of_week",
             "rt_tt2_h1", "dt_day_of_week_h1",
             "rt_tt2_h2", "dt_day_of_week_h2"],
        )


class HorizonFeatureMeltAlignmentTests(unittest.TestCase):
    def test_melt_aligns_exogenous_to_target_day_and_keeps_lag_at_origin(self):
        """长表行 (i, h) 的外生列取目标日 i+h+1 的值；lag 列保持原点 i 的值。"""
        from models.ModelTraining import Trainer

        args = SimpleNamespace(
            pred_method="univariate-single-multistep-direct",
            direct_strategy="horizon_feature",
            horizon_feature_name="forecast_horizon_idx",
            enable_horizon_cyclical=False,
            lags=[1],
            n_per_day=1,
        )
        trainer = Trainer.__new__(Trainer)
        trainer.args = args
        trainer.log_prefix = "[test]"

        X = pd.DataFrame({
            # 基础外生列保留供推理 schema；训练目标日值由 _h1/_h2 显式携带，
            # 即使 dropna 后 X 只剩原点行，也不会丢失尾部目标日外生。
            "rt_tt2": [10.0, 20.0],
            "rt_tt2_h1": [20.0, 30.0],
            "rt_tt2_h2": [30.0, 40.0],
            "y_lag_1": [1.0, 2.0],
        })
        Y = pd.DataFrame(
            {"y_shift_1": [21.0, 31.0],
             "y_shift_2": [31.0, 41.0]},
        )
        X_long, Y_long, sw_long, h_features = trainer._melt_to_horizon_long(X, Y, None)

        # 长表行 (i=0, h=1)：外生对齐目标日 0+1=1 → rt_tt2=20；lag 保持原点 0 → 1
        row_0_h1 = X_long.iloc[0]
        self.assertEqual(row_0_h1["rt_tt2"], 20.0)
        self.assertEqual(row_0_h1["y_lag_1"], 1.0)
        # 行 (i=0, h=2)：目标日 2 → rt_tt2=30
        self.assertEqual(X_long.iloc[1]["rt_tt2"], 30.0)
        self.assertEqual(X_long.iloc[1]["y_lag_1"], 1.0)
        # 行 (i=1, h=1)：目标日 2 → rt_tt2=30
        self.assertEqual(X_long.iloc[2]["rt_tt2"], 30.0)
        self.assertEqual(X_long.iloc[2]["y_lag_1"], 2.0)
        # dropna 后 X 仅有 2 个原点行，但 h2 显式列仍保留原始帧目标日值 40。
        self.assertEqual(X_long.iloc[3]["rt_tt2"], 40.0)
        # melt 后训练 schema 只保留基础外生列名，与推理端一致。
        self.assertNotIn("rt_tt2_h1", X_long.columns)
        self.assertNotIn("rt_tt2_h2", X_long.columns)
        # Y melt 语义不变
        np.testing.assert_allclose(Y_long["y_shift_1"].to_numpy()[:2], [21.0, 31.0])


if __name__ == "__main__":
    unittest.main()
