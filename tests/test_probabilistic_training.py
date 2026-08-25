# -*- coding: utf-8 -*-
"""概率模型 bundle 与 QuantileTrainer 契约测试。"""

import unittest

import pandas as pd

from probabilistic.spec import ProbabilisticSpec
from probabilistic.training import QuantileTrainer
from probabilistic.types import BlendQuantileModel, ProbabilisticModelBundle


class _Backend:
    def __init__(self):
        self.calls = []

    def train_single(self, quantile, X, Y, categorical_features):
        self.calls.append(("single", quantile, tuple(Y.columns)))
        return f"model-{quantile}"

    def train_blend(self, quantile, X, Y_direct, Y_recursive, categorical_features):
        self.calls.append(
            ("blend", quantile, tuple(Y_direct.columns), tuple(Y_recursive.columns))
        )
        return BlendQuantileModel(
            direct=f"direct-{quantile}",
            recursive=f"recursive-{quantile}",
        )


class ProbabilisticTrainingTest(unittest.TestCase):
    @staticmethod
    def _spec():
        return ProbabilisticSpec(
            mode="quantile",
            quantiles=(0.1, 0.5, 0.9),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )

    def test_bundle_requires_exact_model_for_every_quantile(self):
        with self.assertRaisesRegex(ValueError, "models_by_quantile keys"):
            ProbabilisticModelBundle(
                schema_version=1,
                spec=self._spec(),
                model_type="lightgbm",
                pred_method="univariate-single-multistep-recursive",
                models_by_quantile={0.1: object(), 0.5: object()},
                recursive_propagation="median_path",
            )

    def test_single_output_trainer_returns_typed_bundle(self):
        backend = _Backend()
        trainer = QuantileTrainer(
            spec=self._spec(),
            model_type="lightgbm",
            pred_method="univariate-single-multistep-recursive",
            train_single=backend.train_single,
            train_blend=backend.train_blend,
            max_workers=1,
        )
        X = pd.DataFrame({"x": [1.0, 2.0]})
        Y = pd.DataFrame({"y_shift_0": [3.0, 4.0]})

        bundle = trainer.fit(X, Y, categorical_features=[])

        self.assertIsInstance(bundle, ProbabilisticModelBundle)
        self.assertEqual(tuple(bundle.models_by_quantile), (0.1, 0.5, 0.9))
        self.assertEqual([call[1] for call in backend.calls], [0.1, 0.5, 0.9])

    def test_blend_trainer_uses_explicit_blend_model_type(self):
        backend = _Backend()
        trainer = QuantileTrainer(
            spec=self._spec(),
            model_type="lightgbm",
            pred_method="univariate-single-multistep-blend-direct-recursive",
            train_single=backend.train_single,
            train_blend=backend.train_blend,
            max_workers=1,
        )
        X = pd.DataFrame({"x": [1.0, 2.0]})
        Y = pd.DataFrame(
            {
                "y_shift_1": [3.0, 4.0],
                "y_shift_2": [4.0, 5.0],
                "y_shift_0": [2.0, 3.0],
            }
        )

        bundle = trainer.fit(X, Y, categorical_features=[])

        self.assertTrue(bundle.is_blend)
        self.assertTrue(
            all(
                isinstance(model, BlendQuantileModel)
                for model in bundle.models_by_quantile.values()
            )
        )

    def test_blend_short_code_resolves_to_canonical_strategy(self):
        backend = _Backend()
        trainer = QuantileTrainer(
            spec=self._spec(),
            model_type="lightgbm",
            pred_method="usbr",
            train_single=backend.train_single,
            train_blend=backend.train_blend,
        )

        self.assertTrue(trainer.is_blend)
        self.assertEqual(
            trainer.pred_method,
            "univariate-single-multistep-blend-direct-recursive",
        )


if __name__ == "__main__":
    unittest.main()
