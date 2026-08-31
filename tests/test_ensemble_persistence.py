# -*- coding: utf-8 -*-
"""Reference Ensemble 的默认生命周期必须产生可部署 canonical 产物。"""

import json
import pickle
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from data_loading import SourceRegistry
from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.specs.config import parse_model_config
from model_ensemble.deployment import predict_ensemble_bundle
from model_ensemble.loader import load_ensemble_config
from model_ensemble.runtime import run_ensemble_config, run_ensemble_config_file
from model_forecasting.runtime import CanonicalBaseModelRunner
from tests.test_ensemble_runtime import RUNTIME_SERVICES, _ensemble_doc, _member_doc


class EnsemblePersistenceTest(unittest.TestCase):
    def test_runtime_writes_bundle_prediction_and_resolved_metadata(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            times = pd.date_range("2026-01-01", periods=72, freq="1h")
            pd.DataFrame(
                {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
            ).to_csv(root / "data.csv", index=False)
            (root / "member_direct.yaml").write_text(
                yaml.safe_dump(_member_doc("direct", "ridge", "direct")),
                encoding="utf-8",
            )
            (root / "member_recursive.yaml").write_text(
                yaml.safe_dump(_member_doc("recursive", "ridge", "recursive")),
                encoding="utf-8",
            )
            config_path = root / "ensemble.yaml"
            config_path.write_text(
                yaml.safe_dump(_ensemble_doc("averaging")),
                encoding="utf-8",
            )
            config = load_ensemble_config(config_path)

            result = run_ensemble_config(
                config,
                output_root=root / "results",
                base_dir=root,
                services=RUNTIME_SERVICES,
            )

            self.assertIn("run_dir", result)
            self.assertIn("bundle", result)
            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "pretrained_models/model.pkl").is_file())
            self.assertTrue((run_dir / "pretrained_models/resolved_model.json").is_file())
            self.assertTrue((run_dir / "results_forecast/prediction.csv").is_file())
            self.assertTrue((run_dir / "results_forecast/resolved_config.json").is_file())

    def test_config_file_separates_member_ref_root_from_source_asset_root(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            config_dir = root / "configs"
            config_dir.mkdir()
            times = pd.date_range("2026-01-01", periods=72, freq="1h")
            pd.DataFrame(
                {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
            ).to_csv(root / "data.csv", index=False)
            (config_dir / "member_direct.yaml").write_text(
                yaml.safe_dump(_member_doc("direct", "ridge", "direct")),
                encoding="utf-8",
            )
            (config_dir / "member_recursive.yaml").write_text(
                yaml.safe_dump(_member_doc("recursive", "ridge", "recursive")),
                encoding="utf-8",
            )
            config_path = config_dir / "ensemble.yaml"
            config_path.write_text(
                yaml.safe_dump(_ensemble_doc("averaging")),
                encoding="utf-8",
            )

            with patch("model_ensemble.runtime.Path.cwd", return_value=root):
                result = run_ensemble_config_file(
                    config_path,
                    output_root=root / "results",
                    services=RUNTIME_SERVICES,
                )

            run_dir = Path(result["run_dir"])
            self.assertTrue((run_dir / "pretrained_models/model.pkl").is_file())
            with (run_dir / "pretrained_models/model.pkl").open("rb") as file:
                restored = pickle.load(file)
            self.assertEqual(set(restored.model["member_bundles"]), {"m_direct", "m_recursive"})

    def test_reloaded_bundle_predicts_without_yaml_or_oof_cache(self):
        for mode in ("point", "quantile"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                times = pd.date_range("2026-01-01", periods=72, freq="1h")
                pd.DataFrame(
                    {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
                ).to_csv(root / "data.csv", index=False)
                member_paths = []
                for file_name, member_name in (
                    ("member_direct.yaml", "m_direct"),
                    ("member_recursive.yaml", "m_recursive"),
                ):
                    document = _member_doc(
                        "direct",
                        "ridge" if mode == "point" else "qr",
                        member_name,
                    )
                    if mode == "quantile":
                        document["probabilistic"] = {
                            "mode": "quantile",
                            "quantiles": [0.1, 0.5, 0.9],
                            "point_quantile": 0.5,
                        }
                    member_path = root / file_name
                    member_path.write_text(
                        yaml.safe_dump(document),
                        encoding="utf-8",
                    )
                    member_paths.append(member_path)
                ensemble_path = root / f"ensemble_{mode}.yaml"
                ensemble_path.write_text(
                    yaml.safe_dump(_ensemble_doc("linear_blending", mode=mode)),
                    encoding="utf-8",
                )
                config = load_ensemble_config(ensemble_path)
                result = run_ensemble_config(
                    config,
                    output_root=root,
                    base_dir=root,
                    services=RUNTIME_SERVICES,
                )
                bundle = result["bundle"]
                if mode == "quantile":
                    resolved = json.loads(
                        (Path(result["model_dir"]) / "resolved_model.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    method_audit = resolved["ensemble"]["method_artifact"]
                    self.assertEqual(method_audit["quantile_levels"], [0.1, 0.5, 0.9])
                    self.assertEqual(
                        set(method_audit["optimizer_success_by_target"]),
                        {"target_0"},
                    )
                    self.assertEqual(
                        set(method_audit["fallback_reason_by_target"]),
                        {"target_0"},
                    )
                raw_designs = {}
                raw_feature_providers = {}
                forecast_times = None
                for ref, member_path in zip(config.members, member_paths):
                    member_raw = yaml.safe_load(member_path.read_text(encoding="utf-8"))
                    member_config = parse_model_config(member_raw, source=member_path)
                    runner = CanonicalBaseModelRunner(
                        member_config,
                        SourceRegistry(member_config.data, root),
                        pd.Timestamp(member_config.validation["forecast_origin"]),
                    )
                    member_bundle = bundle.model["member_bundles"][ref.name]
                    designs, provider = runner.builder.forecast_designs(
                        runner.origin,
                        target_transform=member_bundle.target_transform,
                    )
                    raw_designs[ref.name] = designs[0]
                    raw_feature_providers[ref.name] = provider
                    forecast_times = runner.forecast_times(runner.origin)

                model_path = Path(result["model_dir"]) / "model.pkl"
                ensemble_path.unlink()
                for member_path in member_paths:
                    member_path.unlink()
                shutil.rmtree(root / "_ensemble_oof")
                with model_path.open("rb") as file:
                    restored = pickle.load(file)

                prediction = predict_ensemble_bundle(
                    restored,
                    raw_designs,
                    forecast_times=forecast_times,
                    raw_feature_providers=raw_feature_providers,
                )
                predicted_values = (
                    prediction.quantiles.values
                    if isinstance(prediction, MarginalForecastDistribution)
                    else prediction.values
                )
                np.testing.assert_allclose(
                    predicted_values,
                    result["combined_values"],
                    rtol=0.0,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
