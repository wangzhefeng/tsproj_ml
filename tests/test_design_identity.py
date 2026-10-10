"""原始设计身份隔离：内存共享和执行证据，不含磁盘缓存。"""
from __future__ import annotations
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
import numpy as np
import pandas as pd
from feature_engineering.design_identity import compute_raw_design_fingerprint
from forecasting_core.specs import (ColumnSpec, DataSourceSpec, DataSpec, EstimatorSpec,
    FeatureSpec, ForecastConfigSpec, ForecastProblemSpec, ForecastStrategySpec, parse_model_config)

def _generated_feature_a(*_args, **_kwargs):
    return "a"

def _generated_feature_b(*_args, **_kwargs):
    return "b"


class DesignIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_path = self.root / "load.csv"
        self.times = pd.date_range("2026-01-01", periods=64, freq="1h")
        self._write_values(np.arange(len(self.times), dtype=float))
        self.origin = cast(pd.Timestamp, pd.Timestamp(self.times.to_numpy()[-4]))
        self.config = self._config()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_values(self, values: np.ndarray) -> None:
        pd.DataFrame(
            {
                "time": self.times,
                "load": 100.0 + values,
            }
        ).to_csv(self.data_path, index=False)

    def _config(
        self,
        performance: dict[str, int] | None = None,
    ) -> ForecastConfigSpec:
        validation = {
            "forecast_origin": str(self.origin),
            "history_steps": 32,
            'training_window': {'kind': 'rolling', 'history_steps': 16},
            "fold_count": 1,
            "stride_steps": 2,
        }
        if performance is not None:
            validation["performance"] = performance
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="time",
                freq="1h",
                horizon=2,
                targets=("load",),
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(
                (
                    DataSourceSpec(
                        name="target_history",
                        source_type="file",
                        columns=(ColumnSpec("load", "target"),),
                        history_path=str(self.data_path),
                        time_col="time",
                        availability="source_time",
                    ),
                )
            ),
            features=FeatureSpec(
                target_lags={"load": (2, 3, 4)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations={},
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(
                model_type="ridge",
                target_adapter="independent",
                params={"alpha": 1e-6},
            ),
            probabilistic={"mode": "point"},
            validation=validation,
            output={"scenario_subpath": "design-identity"},
        )

    def test_compiler_source_change_invalidates_raw_design(self) -> None:
        from feature_engineering import design_identity

        def fingerprint():
            return compute_raw_design_fingerprint(
                self.config, base_dir=self.root, origin=self.origin, generators={}
            )

        original = design_identity.file_sha256
        before = fingerprint()

        def changed(path):
            if Path(path).name == "compiler.py":
                return "changed-compiler-implementation"
            return original(path)

        with patch.object(design_identity, "file_sha256", side_effect=changed):
            self.assertNotEqual(before, fingerprint())
        self.assertEqual(before, fingerprint())

    def test_raw_design_inputs_invalidate_fingerprint(self) -> None:
        def fingerprint(
            config: ForecastConfigSpec,
            *,
            origin: pd.Timestamp = self.origin,
            generators: dict[str, Any] | None = None,
        ) -> str:
            return compute_raw_design_fingerprint(
                config,
                base_dir=self.root,
                origin=origin,
                generators=generators or {},
            )

        baseline = fingerprint(self.config)
        payloads: list[dict[str, Any]] = []

        changed_features: dict[str, Any] = copy.deepcopy(
            self.config.canonical_payload()
        )
        changed_features["features"]["target_lags"]["load"] = [2, 3, 5]
        payloads.append(changed_features)

        changed_strategy: dict[str, Any] = copy.deepcopy(
            self.config.canonical_payload()
        )
        changed_strategy["strategy"] = {"name": "mimo"}
        payloads.append(changed_strategy)

        changed_geometry: dict[str, Any] = copy.deepcopy(
            self.config.canonical_payload()
        )
        changed_geometry["validation"]["history_steps"] = 31
        payloads.append(changed_geometry)

        changed_data: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        changed_data["data"]["sources"][0]["name"] = "renamed_target_history"
        payloads.append(changed_data)

        for index, payload in enumerate(payloads):
            with self.subTest(index=index):
                changed = parse_model_config(payload, source=f"raw-change-{index}.yaml")
                self.assertNotEqual(baseline, fingerprint(changed))

        self.assertNotEqual(
            baseline,
            fingerprint(
                self.config,
                origin=cast(pd.Timestamp, self.origin - pd.Timedelta(hours=1)),
            ),
        )

    def test_generator_implementation_invalidates_raw_fingerprint(self) -> None:
        payload: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        payload["data"]["sources"].append(
            {
                "name": "generated_calendar",
                "source_type": "generated",
                "columns": [
                    {
                        "name": "calendar_value",
                        "role": "known_future",
                        "categorical": False,
                    }
                ],
                "time_col": "time",
                "series_id_cols": [],
                "availability": "generator_defined",
                "generator": "test_generator",
            }
        )
        config = parse_model_config(payload, source="generated.yaml")

        first = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={"test_generator": _generated_feature_a},
        )
        second = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={"test_generator": _generated_feature_b},
        )

        self.assertNotEqual(first, second)

    def test_generator_helper_change_invalidates_raw_fingerprint(self) -> None:
        payload: dict[str, Any] = copy.deepcopy(self.config.canonical_payload())
        payload["data"]["sources"].append(
            {
                "name": "generated_calendar",
                "source_type": "generated",
                "columns": [
                    {
                        "name": "calendar_value",
                        "role": "known_future",
                        "categorical": False,
                    }
                ],
                "time_col": "time",
                "series_id_cols": [],
                "availability": "generator_defined",
                "generator": "test_generator",
            }
        )
        config = parse_model_config(payload, source="generated-helper.yaml")

        def load_generator(filename: str, helper_value: str):
            path = self.root / filename
            path.write_text(
                "def _helper():\n"
                f"    return {helper_value!r}\n\n"
                "def generate(*_args, **_kwargs):\n"
                "    return _helper()\n",
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location(path.stem, path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"could not load test generator from {path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.generate

        first = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={
                "test_generator": load_generator("generator_a.py", "a")
            },
        )
        second = compute_raw_design_fingerprint(
            config,
            base_dir=self.root,
            origin=self.origin,
            generators={
                "test_generator": load_generator("generator_b.py", "b")
            },
        )

        self.assertNotEqual(first, second)
