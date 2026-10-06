"""回测模型重训周期：真实拟合、冻结状态、更新输入与 final fit。"""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from pipeline.lifecycle import CanonicalRuntimeResult
from forecasting_core.specs.validation import RuntimeValidationSpec
import test_canonical_runtime_smoke as fixtures


class RefitScheduleTest(unittest.TestCase):
    def test_refit_validation_and_default_identity(self):
        base = fixtures.CanonicalRuntimeSmokeTest().build_config(Path("unused.csv"), mode="point")
        explicit = replace(base, validation={**base.validation.canonical_payload(), "refit_every": 1})
        self.assertEqual(base.fingerprint(), explicit.fingerprint())
        frozen = replace(base, validation={**base.validation.canonical_payload(), "refit_every": 0})
        self.assertNotEqual(base.fingerprint(), frozen.fingerprint())
        for value in (-1, True, 1.5, "2", None):
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                RuntimeValidationSpec.from_mapping({**base.validation.canonical_payload(), "refit_every": value})
        with self.assertRaisesRegex(ValueError, "calendar"):
            RuntimeValidationSpec.from_mapping({"horizon_mode": "calendar_month", "train_window_days": 30,
                "fold_count": 2, "stride_months": 1, "refit_every": 0})

    def test_real_refit_counts_parallel_parity_and_final_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "load.csv"
            pd.DataFrame({"time": pd.date_range("2026-01-01", periods=64, freq="h"),
                "load": 100 + np.arange(64) + 5 * np.sin(np.arange(64) / 3)}).to_csv(data, index=False)
            base = fixtures.CanonicalRuntimeSmokeTest().build_config(data, mode="point", strategy="mimo")
            # 复用要连同 scaler 一起冻结，不能暗中只更新变换。
            base = replace(base, features=replace(base.features, transformations={
                "feature_scaling": {"method": "standard"},
                "target": {"scaling": {"method": "standard"}},
            }))
            fit = CanonicalBaseModelRunner.fit
            reference = None
            for every, workers, expected_count in ((1, 1, 6), (0, 1, 1), (2, 1, 3), (2, 2, 3)):
                config = replace(base, validation={
                    "forecast_origin": "2026-01-02T23:00:00", "history_steps": 35,
                    "train_window_steps": 12, "fold_count": 6, "stride_steps": 2,
                    "refit_every": every,
                    "performance": {"window_parallel_workers": workers, "multi_output_n_jobs": 1},
                })
                with self.subTest(every=every, workers=workers):
                    with patch.object(CanonicalBaseModelRunner, "fit", autospec=True, side_effect=fit) as observed:
                        result = run_canonical_config(config, output_root=root / f"run-{every}-{workers}")
                    assert isinstance(result, CanonicalRuntimeResult)
                    self.assertEqual(observed.call_count, expected_count)
                    self.assertTrue((result.model_dir / "model.pkl").is_file())
                    self.assertTrue((result.forecast_dir / "prediction.csv").is_file())
                    frame = pd.read_csv(result.test_dir / "cv_plot_df.csv")
                    metadata = json.loads((result.test_dir / "result_metadata.json").read_text())
                    evidence = metadata["backtest"]["execution_evidence"]
                    self.assertEqual([item["refitted"] for item in evidence],
                                     [index == 0 or (every > 0 and index % every == 0) for index in range(6)])
                    for item in evidence:
                        self.assertLessEqual(pd.Timestamp(item["fit_origin"]), pd.Timestamp(item["origin"]))
                    if every == 2 and workers == 1:
                        reference = frame
                    if every == 2 and workers == 2:
                        pd.testing.assert_frame_equal(frame, reference)


if __name__ == "__main__":
    unittest.main()
