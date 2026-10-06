"""异构预热长度与 Global：真实 OOF、外层评分、缓存和部署坐标。"""
import copy
import pickle

import numpy as np
import pandas as pd
import yaml

from data_loading import SourceRegistry
from forecasting_core.artifacts import MarginalForecastDistribution
from forecasting_core.specs.config import parse_model_config
from model_ensemble.inference.deployment import predict_ensemble_bundle
from model_ensemble.configuration.loader import parse_ensemble_document
from model_ensemble.runtime import run_ensemble_config
from pipeline.runner import CanonicalBaseModelRunner
from test_ensemble_runtime import EnsembleRuntimeTestBase, RUNTIME_SERVICES, _ensemble_doc, _member_doc


class EnsemblePanelAlignmentTest(EnsembleRuntimeTestBase):
    def exercise(self, *, panel, mode, method="weighted", params=None):
        data = pd.read_csv(self.root / "data.csv")
        if panel:
            data = pd.concat([data.assign(series_id="A"), data.assign(series_id="B", load=data.load + 20)], ignore_index=True)
            data.to_csv(self.root / "data.csv", index=False)
        document = _ensemble_doc(method, mode=mode)
        if params is not None:
            document["ensemble"]["method"]["params"] = params
        configs = {}
        for index, name in enumerate(("direct", "recursive")):
            raw = _member_doc(name, "qr" if mode == "quantile" else "ridge", name)
            raw["features"]["target_lags"]["load"] = [2, 3] if index == 0 else [1, 4, 5]
            raw["probabilistic"] = copy.deepcopy(document["probabilistic"])
            if panel:
                raw["problem"].update(training_scope="global", series_id_cols=["series_id"])
                source = raw["data"]["sources"][0]
                source["series_id_cols"] = ["series_id"]
                source["columns"].append({"name": "series_id", "role": "key", "categorical": True})
                raw["features"]["transformations"] = {"feature_scaling": {"method": "none", "grouped": False, "encode_categorical": True}}
                raw["validation"]["training_scope"] = {"series_order": ["A", "B"], "incomplete_series_policy": "raise", "unknown_series_policy": "raise"}
            (self.root / f"member_{name}.yaml").write_text(yaml.safe_dump(raw))
            configs[f"m_{name}"] = parse_model_config(raw, source=name)
        document["problem"], document["data"] = raw["problem"], raw["data"]
        config = parse_ensemble_document(document)
        first = run_ensemble_config(config, base_dir=self.root, output_root=self.root, services=RUNTIME_SERVICES)
        second = run_ensemble_config(config, base_dir=self.root, output_root=self.root, services=RUNTIME_SERVICES)
        count = 2 if panel else 1
        self.assertEqual(first["oof"].n_samples, len(first["oof"].folds) * count)
        self.assertEqual(first["bundle"].dimensions, (count, 2, 1))
        self.assertEqual(len(first["backtest"].frame), count * 2)
        self.assertTrue(second["oof_cache_hit"])
        self.assertEqual(first["oof"].series_ids, second["oof"].series_ids)
        for name in configs:
            np.testing.assert_array_equal(first["oof"].values_by_member[name], second["oof"].values_by_member[name])
        with (first["model_dir"] / "model.pkl").open("rb") as handle:
            bundle = pickle.load(handle)
        designs, providers = {}, {}
        for name, member in configs.items():
            origin = pd.Timestamp(member.validation["forecast_origin"])
            runner = CanonicalBaseModelRunner(member, SourceRegistry(member.data, self.root), origin)
            matrices, provider = runner.builder.forecast_designs(origin, target_transform=bundle.model["member_bundles"][name].target_transform)
            designs[name], providers[name] = matrices[0], provider
        prediction = predict_ensemble_bundle(bundle, designs, forecast_times=first["forecast_times"], raw_feature_providers=providers)
        values = prediction.quantiles.values if isinstance(prediction, MarginalForecastDistribution) else prediction.values
        np.testing.assert_allclose(values, first["combined_values"], rtol=0, atol=1e-12)

    def test_local_different_lags(self):
        self.exercise(panel=False, mode="point")

    def test_global_different_lags_point(self):
        self.exercise(panel=True, mode="point")

    def test_global_different_lags_quantile(self):
        self.exercise(panel=True, mode="quantile")

    def test_global_horizon_blending_quantile(self):
        self.exercise(panel=True, mode="quantile", method="linear_blending", params={"weight_scope": "target_horizon"})

    def test_global_dynamic_horizon_point(self):
        self.exercise(panel=True, mode="point", method="adaptive_weighted", params={"weight_scope": "target_horizon", "halflife_days": 0.5})

    def test_global_dynamic_horizon_quantile(self):
        self.exercise(panel=True, mode="quantile", method="adaptive_weighted", params={"weight_scope": "target_horizon", "halflife_days": 0.5})
