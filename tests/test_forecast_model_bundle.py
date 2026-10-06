# -*- coding: utf-8 -*-
"""ForecastModelBundle 单一持久化测试。"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from model_building.pickle_io import ModelDeployPkl
from forecasting_core.probability.spec import ProbabilisticSpec
from forecasting_core.bundle import ForecastModelBundle
from model_predicting.artifacts.persistence import write_bundle_schema_json


class ForecastModelBundleTest(unittest.TestCase):
    @staticmethod
    def _point_spec():
        return ProbabilisticSpec(
            mode="point",
            quantiles=(),
            point_quantile=0.5,
            recursive_propagation="median_path",
            crossing_method="none",
            crossing_report_raw=True,
            intervals=(),
            calibration=None,
        )

    def test_model_pkl_roundtrip_preserves_single_authoritative_bundle(self):
        bundle = ForecastModelBundle(
            schema_version=2,
            model={"estimator": "fake"},
            feature_scaler={"feature": "scaler"},
            target_transform={"target": "pipeline"},
            selected_features=("x1", "x2"),
            input_schema={"columns": ["x1", "x2"]},
            probabilistic_spec=self._point_spec(),
            model_type="lightgbm",
            pred_method=None, canonical_problem={"freq": "1h"}, strategy_spec={"name": "recursive"},
            estimator_spec={}, dimensions=(1, 2, 1), target_order=("load",),
            training_scope="local", result_schema_version=2, config_fingerprint="a" * 64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pkl"
            with path.open("wb") as handle:
                pickle.dump(bundle, handle)
            loaded = ModelDeployPkl(path).load_model()

        self.assertIsInstance(loaded, ForecastModelBundle)
        self.assertEqual(loaded.selected_features, ("x1", "x2"))
        self.assertEqual(loaded.model, {"estimator": "fake"})

    def test_new_saver_rejects_schema_v1_forecast_bundle(self):
        bundle = ForecastModelBundle(
            schema_version=2,
            model={"estimator": "fake"},
            feature_scaler=None,
            target_transform=None,
            selected_features=(),
            input_schema={},
            probabilistic_spec=self._point_spec(),
            model_type="ridge",
            pred_method=None, canonical_problem={"freq": "1h"}, strategy_spec={"name": "recursive"},
            estimator_spec={}, dimensions=(1, 2, 1), target_order=("load",),
            training_scope="local", result_schema_version=2, config_fingerprint="a" * 64,
        )
        bundle.schema_version = 1
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "schema_version=2"):
                ModelDeployPkl(Path(temp_dir) / "model.pkl").save_model(bundle)

    def test_bundle_roundtrip_records_canonical_identity_without_pred_method(self):
        bundle = ForecastModelBundle(
            schema_version=2,
            model={"strategy": "artifact"},
            feature_scaler=None,
            target_transform=None,
            selected_features=("x",),
            input_schema={"columns": ["x"]},
            probabilistic_spec=self._point_spec(),
            model_type="ridge",
            pred_method=None,
            canonical_problem={
                "time_col": "time",
                "freq": "1h",
                "horizon": 2,
                "targets": ["load", "power"],
                "training_scope": "global",
                "series_id_cols": ["series_id"],
            },
            strategy_spec={"name": "direct"},
            estimator_spec={
                "model_type": "ridge",
                "target_adapter": "independent",
                "capabilities": {"scalar_target": True},
            },
            dimensions=(2, 2, 2),
            target_order=("load", "power"),
            feature_lineage=({"feature": "x", "source": "target"},),
            source_lineage=({"source": "target", "path": "dataset/load.csv"},),
            training_scope="global",
            result_schema_version=2,
            config_fingerprint="a" * 64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pkl"
            schema_path = Path(temp_dir) / "resolved_model.json"
            ModelDeployPkl(path).save_model(bundle)
            loaded = ModelDeployPkl(path).load_model()
            write_bundle_schema_json(bundle, schema_path)
            payload = json.loads(schema_path.read_text(encoding="utf-8"))

        self.assertEqual(loaded.schema_version, 2)
        self.assertEqual(loaded.dimensions, (2, 2, 2))
        self.assertEqual(loaded.target_order, ("load", "power"))
        self.assertNotIn("pred_method", payload)
        self.assertEqual(payload["strategy"], {"name": "direct"})
        self.assertEqual(payload["config_fingerprint"], "a" * 64)

    def test_schema_json_contains_metadata_but_not_model_objects(self):
        bundle = ForecastModelBundle(
            schema_version=2,
            model=object(),
            feature_scaler=object(),
            target_transform=object(),
            selected_features=("x",),
            input_schema={"columns": ["x"]},
            probabilistic_spec=self._point_spec(),
            model_type="ridge",
            pred_method=None, canonical_problem={"freq": "1h"}, strategy_spec={"name": "direct"},
            estimator_spec={}, dimensions=(1, 2, 1), target_order=("load",),
            training_scope="local", result_schema_version=2, config_fingerprint="a" * 64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "probabilistic_schema.json"
            write_bundle_schema_json(bundle, path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["probabilistic"]["mode"], "point")
        self.assertNotIn("model", payload)
        self.assertNotIn("feature_scaler", payload)
        self.assertNotIn("target_transform", payload)



if __name__ == "__main__":
    unittest.main()
