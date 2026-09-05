"""Calendar batch exercises the real dynamic factory and artifact gate."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from tests import test_canonical_runtime_smoke as fixture
from model_forecasting.batch_runtime import run_canonical_batch, verify_batch_results
from model_training.estimators.capabilities import _ModelFactoryEstimator


class CalendarBatchCheckpointTest(unittest.TestCase):
    def test_calendar_quantile_completion_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range("2026-01-01", "2026-07-31", freq="1D")
            data = root / "load.csv"
            pd.DataFrame({"time": times, "load": 10 + np.sin(np.arange(len(times)))}).to_csv(data, index=False)
            config = fixture.CanonicalRuntimeSmokeTest().build_config(data, mode="quantile")
            config = replace(config, problem=replace(config.problem, horizon=31, freq="1D"),
                features=replace(config.features, datetime_features=()),
                validation={"forecast_origin": str(times[-1]), "horizon_mode": "calendar_month",
                            "train_window_days": 90, "fold_count": 2, "stride_months": 1})
            path = root / "model.yaml"
            path.write_text(yaml.safe_dump(config.canonical_payload(), sort_keys=False))
            report = run_canonical_batch([path], output_root=root / "results")
            state = json.loads(report.state_path.read_text())
            self.assertEqual(report.completed_count, 1, state)
            self.assertEqual(verify_batch_results(report.state_path)["verified_count"], 1)
            self.assertTrue(list(Path(state["checkpoint_root"]).rglob("*.fit")))
            with patch.object(_ModelFactoryEstimator, "fit", side_effect=AssertionError("completed fit replayed")):
                resumed = run_canonical_batch([path], output_root=root / "results")
            self.assertEqual(resumed.completed_count, 1)
