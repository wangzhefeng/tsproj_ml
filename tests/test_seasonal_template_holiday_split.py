"""SeasonalTemplate holiday_split 分组合同测试。"""
import unittest

import numpy as np
import pandas as pd

from models.wrappers.seasonal_template import SeasonalTemplateModel


def _frame(n=200, with_holiday=True):
    """合成帧：工作日高负荷、节假日低负荷、日内两个滞后列。"""
    rng = np.random.default_rng(7)
    steps = np.arange(float(n))
    is_holiday = (steps.astype(int) % 11) == 5  # 稀疏节假日标记
    y = 100.0 + 20.0 * (~is_holiday) + 0.05 * steps + rng.normal(0, .5, n)
    data = {
        "load_lag_2": np.roll(y, 2),
        "load_lag_5": np.roll(y, 5),
        "dt_day_of_week": steps.astype(int) % 7,
    }
    if with_holiday:
        data["is_holiday"] = is_holiday.astype(float)
    return pd.DataFrame(data), pd.Series(y)


class SeasonalTemplateHolidaySplitTest(unittest.TestCase):
    def test_holiday_split_groups_learn_distinct_weights(self):
        X, y = _frame()
        X.iloc[:5] = X.iloc[:5]  # no-op 保持 dtype
        model = SeasonalTemplateModel(
            {"holiday_split": True, "day_type_split": True, "min_group_samples": 5},
            log_params=False,
        )
        model.fit(X.iloc[5:], y.iloc[5:])
        self.assertEqual(model.group_split_mode_, "holiday")
        assert model.group_weights_ is not None
        self.assertEqual(sorted(model.group_weights_), ["holiday", "workday"])
        # 节假日组（负荷水平约 100）与工作日组（约 120）的真值水平不同，
        # 两组权重向量不应完全相同
        holiday_w = model.group_weights_["holiday"]
        workday_w = model.group_weights_["workday"]
        self.assertFalse(np.allclose(holiday_w, workday_w))
        prediction = model.predict(X.iloc[5:])
        self.assertTrue(np.isfinite(prediction).all())

    def test_holiday_split_without_column_falls_back_to_day_type(self):
        X, y = _frame(with_holiday=False)
        model = SeasonalTemplateModel({"holiday_split": True}, log_params=False)
        model.fit(X.iloc[5:], y.iloc[5:])
        # 列不存在：回退 day_type_split 分组（默认 True）
        self.assertEqual(model.group_split_mode_, "day_type")

    def test_default_params_keep_day_type_behavior(self):
        X, y = _frame()
        model = SeasonalTemplateModel({}, log_params=False)
        model.fit(X.iloc[5:], y.iloc[5:])
        # 默认 holiday_split=False：维持既有 day_type 行为
        self.assertEqual(model.group_split_mode_, "day_type")

    def test_insufficient_group_samples_fall_back_to_global(self):
        X, y = _frame()
        # 节假日样本只有零星几个：min_group_samples 大于组样本数
        model = SeasonalTemplateModel(
            {"holiday_split": True, "min_group_samples": 10_000}, log_params=False,
        )
        model.fit(X.iloc[5:], y.iloc[5:])
        self.assertIsNone(model.group_weights_)
        self.assertIsNone(model.group_split_mode_)


if __name__ == "__main__":
    unittest.main()
