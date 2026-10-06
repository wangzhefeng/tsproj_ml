"""逐 horizon 与按原点时间衰减：独立解析期望，不只检查权重合法。"""
import unittest

import numpy as np
import pandas as pd

from model_ensemble.artifacts import OOFPredictionArtifact, method_artifact_audit_payload
from model_ensemble.configuration.loader import parse_ensemble_document
from model_ensemble.inference.predictor import combine_members
from model_ensemble.training.trainer import fit_fusion_artifact
from test_ensemble_runtime import _ensemble_doc


def config_for(name, params):
    document = _ensemble_doc(name)
    document["ensemble"]["method"]["params"] = params
    return parse_ensemble_document(document)


def oof_for(a, b):
    return OOFPredictionArtifact(
        {"m_direct": a, "m_recursive": b}, ("m_direct", "m_recursive"), ("load",),
        a.shape[1], None, "weights",
        folds=tuple({"fold": index + 1, "origin": timestamp.isoformat(),
                     "label_end": (timestamp + pd.Timedelta(hours=2)).isoformat()}
                    for index, timestamp in enumerate(pd.date_range("2026-01-01", periods=a.shape[0], freq="1D"))),
        series_ids=("local",),
    )


class EnsembleWeightScopeTest(unittest.TestCase):
    def test_horizon_weighted_and_blending_learn_different_horizon_preferences(self):
        actual = np.arange(1.0, 5.0)[:, None, None] * np.ones((1, 2, 1))
        a = actual + np.array([0.0, 2.0])[None, :, None]
        b = actual + np.array([2.0, 0.0])[None, :, None]
        oof = oof_for(a, b)
        for method in ("weighted", "linear_blending"):
            with self.subTest(method=method):
                artifact = fit_fusion_artifact(config_for(method, {"weight_scope": "target_horizon"}), oof, actual, origin=pd.Timestamp("2026-01-05"))
                prediction = combine_members(artifact, oof.values_by_member)
                np.testing.assert_allclose(prediction, actual, rtol=0, atol=1e-7)
                audit = method_artifact_audit_payload(artifact.method_artifact)
                self.assertEqual(audit["weight_scope"], "target_horizon")
                self.assertEqual(len(audit["horizons"]), 2)

    def test_dynamic_weights_favour_recent_skill_and_reject_future_labels(self):
        actual = np.zeros((4, 2, 1))
        a = np.broadcast_to(np.array([1.0, 1.0, 9.0, 9.0])[:, None, None], actual.shape)
        b = np.broadcast_to(np.array([9.0, 9.0, 1.0, 1.0])[:, None, None], actual.shape)
        oof = oof_for(a, b)
        config = config_for("adaptive_weighted", {"metric": "mae", "halflife_days": 1.0})
        artifact = fit_fusion_artifact(config, oof, actual, origin=pd.Timestamp("2026-01-05"))
        decay = np.power(2.0, np.array([-3.0, -2.0, -1.0, 0.0]))
        errors = np.array([np.average([1., 1., 9., 9.], weights=decay), np.average([9., 9., 1., 1.], weights=decay)])
        expected = (1 / errors) / np.sum(1 / errors)
        prediction = combine_members(artifact, oof.values_by_member)
        np.testing.assert_allclose(prediction, expected[0] * a + expected[1] * b, rtol=0, atol=1e-12)
        self.assertGreater(expected[1], expected[0])
        audit = method_artifact_audit_payload(artifact.method_artifact)
        self.assertEqual(audit["forecast_origin"], "2026-01-05T00:00:00")
        self.assertEqual(audit["halflife_days"], 1.0)
        with self.assertRaisesRegex(ValueError, "label.*origin|available"):
            fit_fusion_artifact(config, oof, actual, origin=pd.Timestamp("2026-01-03"))
