"""显式缺失编码保持前缀因果性，不把占位误当实测。"""
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data_process.causal_feature_encoding import encode_features


class CausalFeatureEncodingTest(unittest.TestCase):
    def test_invalid_contracts_raise(self):
        base = pd.DataFrame({'time': pd.date_range('2026-01-01', periods=3, freq='5min'),
                             'target': [1., 2., 3.], 'x': [1., np.nan, 3.]})
        cases = [
            (base, ['target']), (base, ['time']), (base, ['absent']),
            (base, ['x', 'x']), (base, []),
            (base.iloc[::-1], ['x']),
            (base.assign(time=[base.time[0]] * 3), ['x']),
            (base.assign(target=[1., np.nan, 3.]), ['x']),
            (base.assign(x=[1., np.inf, 3.]), ['x']),
            (base.assign(x__missing=0), ['x']),
        ]
        for frame, columns in cases:
            with self.subTest(columns=columns, frame=frame.to_dict('list')):
                with self.assertRaises(ValueError):
                    encode_features(frame, time_col='time', targets=['target'], columns=columns)

    def test_missing_and_no_history_are_explicit_and_prefix_invariant(self):
        frame = pd.DataFrame({'time': pd.date_range('2026-01-01', periods=4, freq='5min'),
                              'target': [1., 2., 3., 4.], 'x': [np.nan, 7., np.nan, 20.]})
        original = frame.copy(deep=True)
        result = encode_features(frame, time_col='time', targets=['target'], columns=['x'])
        self.assertEqual(result.x.tolist(), [0., 7., 7., 20.])
        self.assertEqual(result.x__missing.tolist(), [1, 0, 1, 0])
        self.assertEqual(result.x__no_history.tolist(), [1, 0, 0, 0])
        assert_frame_equal(frame, original)
        assert_frame_equal(result.iloc[:3], encode_features(frame.iloc[:3], time_col='time', targets=['target'], columns=['x']))
        frame.loc[3, ['x', 'target']] = -999.
        assert_frame_equal(result.iloc[:3], encode_features(frame, time_col='time', targets=['target'], columns=['x']).iloc[:3])


if __name__ == '__main__':
    unittest.main()
