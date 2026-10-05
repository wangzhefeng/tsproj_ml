"""Calendar compilation must be part of each bounded task, not eager setup."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from pipeline.runner import CanonicalBaseModelRunner
from model_testing.loops.calendar_month import run_calendar_month_backtest
from model_testing.loops.scoring import score_holdout_fold
from tests import test_canonical_runtime_smoke as fixture


class CalendarBoundedExecutionTest(unittest.TestCase):
    def test_serial_scores_before_constructing_the_next_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range("2026-01-01", "2026-07-31", freq="1D")
            path = root / "load.csv"
            pd.DataFrame({"time": times, "load": 10 + np.sin(np.arange(len(times)))}).to_csv(path, index=False)
            config = fixture.CanonicalRuntimeSmokeTest().build_config(path, mode="point")
            config = replace(config, problem=replace(config.problem, horizon=31, freq="1D"),
                             features=replace(config.features, datetime_features=()),
                             validation={"forecast_origin": str(times[-1]), "horizon_mode": "calendar_month",
                                         "train_window_days": 90, "fold_count": 3, "stride_months": 1})
            registry = SourceRegistry(config.data, root)
            runner = CanonicalBaseModelRunner(config, registry, times[-1])
            events = []

            def factory(*args, **kwargs):
                events.append("build")
                return CanonicalBaseModelRunner(*args, **kwargs)

            def score(**kwargs):
                events.append("score")
                return score_holdout_fold(**kwargs)

            with patch("model_testing.loops.calendar_month.score_holdout_fold", side_effect=score):
                run_calendar_month_backtest(config, registry, runner, root / "results", runner_factory=factory)
            self.assertEqual(events, ["build", "score"] * 3)
