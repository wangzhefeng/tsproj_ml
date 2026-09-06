"""Family wrappers round-trip at their new pickle paths; no legacy shim."""
import importlib.util
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from models.factory import ModelFactory
from models.pickle_io import ModelDeployPkl

ROOT = Path(__file__).resolve().parents[1]


class ModelWrapperPersistenceTest(unittest.TestCase):
    def test_all_model_families_fit_save_load_predict(self):
        cases = {
            "lgb": ({"n_estimators": 2, "n_jobs": 1}, "lightgbm"),
            "xgb": ({"n_estimators": 2, "n_jobs": 1, "max_depth": 2}, "xgboost"),
            "cat": ({"iterations": 2, "thread_count": 1, "depth": 2}, "catboost"),
            "rf": ({"n_estimators": 2, "n_jobs": 1}, "sklearn_tree"),
            "histgb": ({"max_iter": 2}, "sklearn_tree"),
            "ridge": ({}, "linear"),
            "enet": ({}, "linear"),
            "lasso": ({}, "linear"),
            "qr": ({}, "linear"),
            "st": ({"equal_weight": True}, "seasonal_template"),
        }
        steps = np.arange(32.0)
        X = pd.DataFrame({"load__lag_24": steps + 1.0, "dt_day_of_week": steps % 7})
        y = pd.Series(steps * 2.0 + 3.0)
        with tempfile.TemporaryDirectory() as directory, threadpool_limits(limits=1):
            for name, (params, module) in cases.items():
                with self.subTest(model=name):
                    model = ModelFactory().create_model(name, params, log_params=False)
                    self.assertEqual(type(model).__module__, f"models.wrappers.{module}")
                    model.fit(X, y)
                    expected = model.predict(X)
                    io = ModelDeployPkl(str(Path(directory) / f"{name}.pkl"))
                    io.save_model(model)
                    loaded = io.load_model()
                    self.assertIs(type(loaded), type(model))
                    np.testing.assert_array_equal(loaded.predict(X), expected)

    def test_old_module_paths_are_not_shims(self):
        for module in ("models.ModelFactory", "models.ModelSaveLoad"):
            with self.subTest(module=module):
                self.assertIsNone(importlib.util.find_spec(module))
        # Trusted test payload: protocol-0 GLOBAL referencing the deliberately removed class path.
        with self.assertRaises(ModuleNotFoundError):
            pickle.loads(b"cmodels.ModelFactory\nLightGBMModel\n.")

    def test_pickle_io_import_does_not_append_working_directory(self):
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "import joblib; import utils.log_util; "
            "before = list(sys.path); import models.pickle_io; "
            "assert sys.path == before, (before, sys.path)"
        )
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", code, str(ROOT)], cwd=directory,
                env=environment, capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
