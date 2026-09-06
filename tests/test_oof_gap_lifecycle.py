"""正 gap 的真实小样本 OOF/final/forecast/cache/bundle 生命周期。"""

import json
import pickle

import numpy as np
import pandas as pd
import yaml

from data_loading import SourceRegistry
from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.specs.config import parse_model_config
from model_ensemble.deployment import predict_ensemble_bundle
from model_ensemble.loader import parse_ensemble_document
from model_ensemble.runtime import run_ensemble_config
from model_pipeline.runner import CanonicalBaseModelRunner
from test_ensemble_runtime import (
    EnsembleRuntimeTestBase, RUNTIME_SERVICES, _ensemble_doc, _member_doc,
)


class OOFGapLifecycleTest(EnsembleRuntimeTestBase):
    def _exercise(self, mode):
        document = _ensemble_doc("averaging", mode=mode)
        document["ensemble"]["oof"]["gap_steps"] = 1
        config = parse_ensemble_document(document)
        member_configs = {}
        for strategy in ("direct", "recursive"):
            member_doc = _member_doc(strategy, "qr" if mode == "quantile" else "ridge", strategy)
            member_doc["probabilistic"] = document["probabilistic"]
            path = self.root / f"member_{strategy}.yaml"
            path.write_text(yaml.safe_dump(member_doc), encoding="utf-8")
            member_configs[f"m_{strategy}"] = parse_model_config(member_doc, source=path)
        first = run_ensemble_config(
            config, base_dir=self.root, output_root=self.root,
            services=RUNTIME_SERVICES,
        )
        self.assertFalse(first["oof_cache_hit"])
        for fold in first["oof"].folds:
            self.assertEqual(fold["gap_steps"], 1)
            self.assertEqual(fold["gap_semantics"], "label_embargo_v1")
            training_end = pd.Timestamp(fold["training_label_end_max"])
            holdout_start = pd.Timestamp(fold["label_start"])
            assert isinstance(training_end, pd.Timestamp)
            assert isinstance(holdout_start, pd.Timestamp)
            self.assertLess(training_end, holdout_start - pd.Timedelta(hours=1))
        cache_metadata = json.loads((
            self.root / "_ensemble_oof" / first["oof_fingerprint"] / "oof_metadata.json"
        ).read_text("utf-8"))
        self.assertEqual(cache_metadata["folds"], list(first["oof"].folds))
        second = run_ensemble_config(
            config, base_dir=self.root, output_root=self.root,
            services=RUNTIME_SERVICES,
        )
        self.assertTrue(second["oof_cache_hit"])
        np.testing.assert_array_equal(first["combined_values"], second["combined_values"])
        self.assertEqual(first["oof"].folds, second["oof"].folds)
        for directory, filename in (
            ("forecast_dir", "prediction.csv"),
            ("test_dir", "cv_plot_df.csv"),
        ):
            frame = pd.read_csv(first[directory] / filename)
            self.assertFalse(frame.empty)
            self.assertFalse(frame.duplicated(
                ["series_id", "time", "target"] + (["window"] if directory == "test_dir" else [])
            ).any())
        with (first["model_dir"] / "model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        self.assertEqual(bundle.config_fingerprint, config.fingerprint())
        raw_designs, providers = {}, {}
        for name, member_config in member_configs.items():
            origin = pd.Timestamp(member_config.validation["forecast_origin"])
            assert isinstance(origin, pd.Timestamp)
            runner = CanonicalBaseModelRunner(
                member_config, SourceRegistry(member_config.data, self.root),
                origin,
            )
            member_bundle = bundle.model["member_bundles"][name]
            designs, provider = runner.builder.forecast_designs(
                runner.origin, target_transform=member_bundle.target_transform,
            )
            raw_designs[name], providers[name] = designs[0], provider
        prediction = predict_ensemble_bundle(
            bundle, raw_designs, forecast_times=first["forecast_times"],
            raw_feature_providers=providers,
        )
        values = prediction.quantiles.values if isinstance(prediction, MarginalForecastDistribution) else prediction.values
        np.testing.assert_allclose(values, first["combined_values"], rtol=0, atol=1e-12)

    def test_point_positive_gap_lifecycle(self):
        self._exercise("point")

    def test_quantile_positive_gap_lifecycle(self):
        self._exercise("quantile")
