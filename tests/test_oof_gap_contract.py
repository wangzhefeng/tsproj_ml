"""OOF 的 gap 隔离标签，而不是扩大训练标签可见范围。"""

import unittest
from dataclasses import dataclass

import pandas as pd

from model_ensemble.oof import oof_fold_origins
from model_testing import validation


@dataclass
class Timeline:
    geometry: validation.TimeGeometry
    supervised_origins: tuple[pd.Timestamp, ...]


class OOFGapContractTest(unittest.TestCase):
    @staticmethod
    def runner(freq="1h", count=30):
        return Timeline(
            geometry=validation.TimeGeometry(
                offset=pd.tseries.frequencies.to_offset(freq), horizon=2,
            ),
            supervised_origins=tuple(pd.date_range("2026-01-01", periods=count, freq=freq)),
        )

    def test_positive_gap_excludes_exact_boundary_and_preserves_holdout(self):
        for freq in ("1h", "15min", "1ME"):
            runner = self.runner(freq)
            origins = runner.supervised_origins
            geometry = runner.geometry
            baseline = oof_fold_origins(
                runner, fold_count=3, stride_steps=1, train_window_steps=6,
            )
            for gap in (1, 3):
                with self.subTest(freq=freq, gap=gap):
                    folds = oof_fold_origins(
                        runner, fold_count=3, stride_steps=1,
                        train_window_steps=6, gap_steps=gap,
                    )
                    self.assertEqual(
                        [fold["origin"] for fold in folds],
                        [fold["origin"] for fold in baseline],
                    )
                    for fold, old in zip(folds, baseline):
                        end = max(geometry.label_end(origins[i]) for i in fold["train_indices"])
                        boundary = geometry.label_start(fold["origin"]) - gap * geometry.offset
                        self.assertLess(end, boundary)
                        self.assertEqual(len(fold["train_indices"]), 6)
                        self.assertEqual(max(fold["train_indices"]), max(old["train_indices"]) - gap)

    def test_zero_gap_keeps_existing_stride_selection(self):
        folds = oof_fold_origins(
            self.runner(), fold_count=3, stride_steps=3, train_window_steps=6,
        )
        self.assertEqual([fold["origin_index"] for fold in folds], [21, 24, 27])
        for fold in folds:
            index = fold["origin_index"]
            self.assertEqual(fold["train_indices"], tuple(range(index - 7, index - 1)))

    def test_positive_gap_and_outer_cutoff_both_hold(self):
        runner = self.runner()
        cutoff = runner.supervised_origins[24]
        folds = oof_fold_origins(
            runner, fold_count=3, stride_steps=1, train_window_steps=6,
            gap_steps=2, outer_cutoff_origin=cutoff,
        )
        for fold in folds:
            geometry = runner.geometry
            self.assertLess(geometry.label_end(fold["origin"]), geometry.label_start(cutoff))
            for index in fold["train_indices"]:
                self.assertLess(
                    geometry.label_end(runner.supervised_origins[index]),
                    geometry.label_start(fold["origin"]) - 2 * geometry.offset,
                )

    def test_no_training_data_after_gap_raises(self):
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            oof_fold_origins(
                self.runner(count=5), fold_count=1, stride_steps=1,
                train_window_steps=2, gap_steps=5,
            )

    def test_shared_predicate_excludes_equal_boundary(self):
        runner = self.runner()
        geometry = runner.geometry
        holdout = runner.supervised_origins[-1]
        for gap in (0, 1, 3):
            safe = holdout - (geometry.horizon + gap) * geometry.offset
            self.assertTrue(validation.is_label_safe(
                safe, geometry.offset, geometry.horizon,
                geometry.label_start(holdout), gap_steps=gap,
            ))
            self.assertFalse(validation.is_label_safe(
                safe + geometry.offset, geometry.offset, geometry.horizon,
                geometry.label_start(holdout), gap_steps=gap,
            ))

    def test_invalid_gap_rejected_even_without_candidates(self):
        runner = self.runner(count=1)
        for gap in (-1, True, 1.5, "1"):
            with self.subTest(gap=gap):
                self.assertRaisesRegex(
                    ValueError, "non-negative integer", oof_fold_origins,
                    runner, fold_count=1, stride_steps=1,
                    train_window_steps=2, gap_steps=gap,
                )


if __name__ == "__main__":
    unittest.main()
