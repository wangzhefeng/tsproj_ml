# -*- coding: utf-8 -*-
"""ForecastModelBundle 单一持久化与 legacy loader 测试。"""

import json
import tempfile
import unittest
from pathlib import Path

from models.ModelSaveLoad import ModelDeployPkl
from models.ModelTraining import Trainer
from probabilistic.spec import ProbabilisticSpec
from probabilistic.types import ForecastModelBundle, ProbabilisticModelBundle


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
            schema_version=1,
            model={"estimator": "fake"},
            feature_scaler={"feature": "scaler"},
            target_transform={"target": "pipeline"},
            selected_features=("x1", "x2"),
            input_schema={"columns": ["x1", "x2"]},
            probabilistic_spec=self._point_spec(),
            model_type="lightgbm",
            pred_method="univariate-single-multistep-recursive",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.pkl"
            ModelDeployPkl(path).save_model(bundle)
            loaded = ModelDeployPkl(path).load_model()

        self.assertIsInstance(loaded, ForecastModelBundle)
        self.assertEqual(loaded.selected_features, ("x1", "x2"))
        self.assertEqual(loaded.model, {"estimator": "fake"})

    def test_schema_json_contains_metadata_but_not_model_objects(self):
        bundle = ForecastModelBundle(
            schema_version=1,
            model=object(),
            feature_scaler=object(),
            target_transform=object(),
            selected_features=("x",),
            input_schema={"columns": ["x"]},
            probabilistic_spec=self._point_spec(),
            model_type="ridge",
            pred_method="univariate-single-multistep-direct",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "probabilistic_schema.json"
            bundle.write_schema_json(path)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["probabilistic"]["mode"], "point")
        self.assertNotIn("model", payload)
        self.assertNotIn("feature_scaler", payload)
        self.assertNotIn("target_transform", payload)

    def test_legacy_quantile_dict_loads_as_runtime_typed_bundle(self):
        legacy = {
            "predict_type": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "median_quantile": 0.5,
            "models": {0.1: "q10", 0.5: "q50", 0.9: "q90"},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "legacy.pkl"
            ModelDeployPkl(path).save_model(legacy)
            loaded = ModelDeployPkl(path).load_model()

        self.assertIsInstance(loaded, ProbabilisticModelBundle)
        self.assertEqual(loaded.models_by_quantile[0.5], "q50")
        self.assertEqual(loaded.spec.quantiles, (0.1, 0.5, 0.9))

    def test_trainer_model_save_writes_only_model_bundle_and_schema_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            trainer = Trainer.__new__(Trainer)
            trainer.args = type(
                "Args",
                (),
                {
                    "checkpoints_dir": checkpoint_dir,
                    "model_type": "lightgbm",
                    "pred_method": "univariate-single-multistep-recursive",
                    "probabilistic_spec": self._point_spec(),
                },
            )()
            trainer.log_prefix = "[test]"

            trainer.model_save(
                {"estimator": "fake"},
                feature_scaler={"feature": "scaler"},
                target_transform={"target": "pipeline"},
                selected_features=["x"],
                input_schema={"columns": ["x"]},
            )

            self.assertTrue((checkpoint_dir / "model.pkl").exists())
            self.assertTrue((checkpoint_dir / "probabilistic_schema.json").exists())
            self.assertFalse((checkpoint_dir / "target_scaler.pkl").exists())
            self.assertFalse((checkpoint_dir / "decomposition_bundle.pkl").exists())
            loaded = ModelDeployPkl(checkpoint_dir / "model.pkl").load_model()
            self.assertIsInstance(loaded, ForecastModelBundle)
            self.assertEqual(loaded.selected_features, ("x",))


if __name__ == "__main__":
    unittest.main()
