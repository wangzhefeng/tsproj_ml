"""分解产物状态版本与严格合成合同。"""
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from decomposition import DecompositionPipeline, build_pipeline_from_args
from decomposition.composition.additive import AdditiveComposer
from decomposition.contracts.types import ComponentForecast
from models.pickle_io import ModelDeployPkl


class DecompositionPersistenceTest(unittest.TestCase):
    def test_unversioned_state_is_rejected(self):
        legacy = object.__new__(DecompositionPipeline)
        legacy.__dict__.update(method="none", enabled=False, is_fitted=True)
        with self.assertRaisesRegex(ValueError, "incompatible decomposition"):
            pickle.loads(pickle.dumps(legacy))

    def test_removed_component_path_has_actionable_loader_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pkl"
            path.write_bytes(b"cdecomposition.extractors\nSTLExtractor\n.")
            with self.assertRaisesRegex(ValueError, "incompatible decomposition"):
                ModelDeployPkl(str(path)).load_model()

    def test_flat_layout_paths_have_actionable_loader_error(self):
        old_symbols = {
            "pipeline": "DecompositionPipeline", "spec": "DecompositionSpec",
            "presets": "PresetParams", "component_factory": "Components",
            "types": "ComponentFrame", "composers": "AdditiveComposer",
            "base": "ComponentExtractor", "residual_diagnostics": "ResidualDiagnosticsRow",
        }
        with tempfile.TemporaryDirectory() as tmp:
            for module, symbol in old_symbols.items():
                with self.subTest(module=module):
                    path = Path(tmp) / "old.pkl"
                    path.write_bytes(f"cdecomposition.{module}\n{symbol}\n.".encode())
                    with self.assertRaisesRegex(ValueError, "incompatible decomposition"):
                        ModelDeployPkl(str(path)).load_model()

    def test_current_state_roundtrips_all_presets(self):
        x = np.arange(240, dtype=float)
        frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=len(x), freq="h"),
                              "y": 10 + 0.02 * x ** 2 + np.sin(2 * np.pi * x / 24)})
        future = pd.date_range(frame.time.iloc[-1] + pd.Timedelta(hours=1), periods=24, freq="h")
        for raw in ({"method": "none"}, {"method": "linear"}, {"method": "damped"},
                    {"method": "stl", "periods": [24], "trend_degree": 2},
                    {"method": "mstl", "periods": [24, 72], "seasonal_cycles": 1}):
            with self.subTest(raw=raw):
                pipe = build_pipeline_from_args(SimpleNamespace(decomposition=raw)).fit(frame)
                loaded = pickle.loads(pickle.dumps(pipe, protocol=2))
                np.testing.assert_array_equal(pipe.restore(np.zeros(24), future), loaded.restore(np.zeros(24), future))

    def test_composer_rejects_broadcasting_and_truncation(self):
        times = pd.date_range("2026-01-01", periods=3, freq="h")
        composer = AdditiveComposer()
        with self.assertRaises(ValueError):
            composer.compose(np.zeros(3), ComponentForecast(times, {"trend": np.ones(1)}))
        with self.assertRaises(ValueError):
            composer.restore_quantiles({0.5: np.zeros(2)}, ComponentForecast(times, {"trend": np.ones(3)}))


if __name__ == "__main__":
    unittest.main()
