"""学习融合器的外层测试必须独立于其元训练标签。"""
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml

from model_ensemble.configuration.loader import parse_ensemble_document
from model_ensemble.runtime import run_ensemble_config
from test_ensemble_runtime import EnsembleRuntimeTestBase, RUNTIME_SERVICES, _ensemble_doc
from pipeline.runner import CanonicalBaseModelRunner


class EnsembleOuterEvaluationTest(EnsembleRuntimeTestBase):
    def test_outer_fit_uses_each_members_final_training_window(self):
        for name, steps in (("direct", 3), ("recursive", 5)):
            path = self.root / f"member_{name}.yaml"
            document = yaml.safe_load(path.read_text())
            document["validation"]["train_window_steps"] = steps
            path.write_text(yaml.safe_dump(document))
        fitted = []
        original = CanonicalBaseModelRunner.fit
        def record(runner, indices, **kwargs):
            fitted.append((runner.config.strategy.name.value, len(indices)))
            return original(runner, indices, **kwargs)
        with patch.object(CanonicalBaseModelRunner, "fit", record):
            self._run("averaging", use_oof_cache=False)
        # 内层 OOF 显式为 6；外层必须复用成员窗口，不能借顶层 9999。
        self.assertEqual([item for item in fitted if item[1] != 6], [("direct", 3), ("recursive", 5)])

    def test_explicit_seasonal_lag_reaches_outer_reference_scores(self):
        document = _ensemble_doc("averaging")
        document["validation"]["seasonal_naive_lag"] = 2
        result = run_ensemble_config(parse_ensemble_document(document), base_dir=self.root, output_root=self.root, services=RUNTIME_SERVICES, use_oof_cache=False)
        scores = result["backtest"].point_scores
        target = scores[scores.scope == "target"].iloc[0]
        self.assertEqual(target["Naive MAE"], 1.0)
        self.assertTrue(np.isfinite(target["MASE"]))

    def test_outer_holdout_is_not_used_to_fit_fusion_weights(self):
        config = parse_ensemble_document(_ensemble_doc("weighted"))
        first = run_ensemble_config(config, base_dir=self.root, output_root=self.root / "first", services=RUNTIME_SERVICES, use_oof_cache=False)
        outer = first["backtest"]
        for fold in outer.metadata["folds"]:
            origin = pd.Timestamp(fold["origin"])
            self.assertTrue(fold["inner_folds"])
            self.assertTrue(all(pd.Timestamp(inner["label_end"]) <= origin for inner in fold["inner_folds"]))
        last_origin = pd.Timestamp(outer.metadata["folds"][-1]["origin"])
        data = pd.read_csv(self.root / "data.csv")
        data.loc[pd.to_datetime(data.time) > last_origin, "load"] += 1000
        data.to_csv(self.root / "data.csv", index=False)
        second = run_ensemble_config(config, base_dir=self.root, output_root=self.root / "second", services=RUNTIME_SERVICES, use_oof_cache=False)
        self.assertEqual(outer.metadata["folds"][-1]["method_artifact"], second["backtest"].metadata["folds"][-1]["method_artifact"])
        np.testing.assert_array_equal(outer.frame.predict_value, second["backtest"].frame.predict_value)
        self.assertFalse(np.array_equal(outer.frame.actual_value, second["backtest"].frame.actual_value))

    def test_outer_score_mask_is_applied_but_plots_keep_actual(self):
        document = _ensemble_doc("averaging")
        document["validation"]["eval_mask"] = {"mode": "absolute", "min_value": 1000000.0}
        result = run_ensemble_config(parse_ensemble_document(document), base_dir=self.root, output_root=self.root, services=RUNTIME_SERVICES, use_oof_cache=False)
        scores = pd.read_csv(result["test_dir"] / "test_scores_df.csv")
        self.assertTrue((scores.n_points == 0).all())
        frame = pd.read_csv(result["test_dir"] / "cv_plot_df.csv")
        self.assertTrue(np.isfinite(frame.actual_value).all())
        metadata = json.loads((result["test_dir"] / "result_metadata.json").read_text())
        self.assertEqual(metadata["evaluation_role"], "outer_holdout")
        self.assertTrue(list((result["test_dir"] / "windows_results").glob("*.png")))
        self.assertTrue((result["test_dir"] / "meta_train_scores.csv").is_file())
