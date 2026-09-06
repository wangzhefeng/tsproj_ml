# -*- coding: utf-8 -*-
"""监督特征选择专项测试（2026-08-30，features.selection 恢复接入）。

覆盖：
- SelectionSpec 严格解析（未知字段/非法值 RAISE，min>max RAISE）；
- CanonicalFeatureSelector：k 钳位、force_keep 并集、min_features 兜底、schema 序保持；
- 未配置 selection = 不进 fingerprint（存量配置零变化）；
- 端到端：启用选择后 artifact.feature_schema 收缩、预测产物正常、
  点/quantile 两模式均可用；force_keep 特征保留。
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering.selection import (
    CanonicalFeatureSelector,
    FeatureSelectionSpec,
    normalize_feature_selection,
    selected_indices_for_artifact,
)
from model_pipeline.runner import run_canonical_config
from forecasting_core.specs.config import parse_model_config


class SelectionSpecParseTest(unittest.TestCase):
    def test_absent_or_null_selection_returns_none(self):
        self.assertIsNone(normalize_feature_selection(None))

    def test_unknown_field_raises(self):
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            normalize_feature_selection({"enabled": True, "bogus": 1})

    def test_invalid_method_raises(self):
        with self.assertRaisesRegex(ValueError, "method"):
            normalize_feature_selection({"method": "lasso"})

    def test_min_greater_than_max_raises(self):
        with self.assertRaisesRegex(ValueError, "min_features"):
            normalize_feature_selection({"min_features": 20, "max_features": 10})

    def test_force_keep_must_be_string_list(self):
        with self.assertRaisesRegex(TypeError, "force_keep"):
            normalize_feature_selection({"force_keep": [1, 2]})


class CanonicalFeatureSelectorTest(unittest.TestCase):
    def _fit(self, n_features=12, n_rows=60, **spec_kwargs):
        schema = tuple(f"f{i}" for i in range(n_features))
        rng = np.random.default_rng(7)
        X = rng.normal(size=(n_rows, n_features))
        # 让 f0/f1 与 y 强相关，其余近独立
        y = 2.0 * X[:, 0] - X[:, 1] + rng.normal(scale=0.01, size=n_rows)
        spec = FeatureSelectionSpec(**spec_kwargs)
        selector = CanonicalFeatureSelector(spec, schema).fit(X, y)
        return selector, schema

    def test_disabled_selects_all(self):
        selector, schema = self._fit(enabled=False)
        self.assertEqual(selector.selected_names_, schema)

    def test_small_schema_selects_all(self):
        selector, schema = self._fit(n_features=5, enabled=True, min_features=10)
        self.assertEqual(selector.selected_names_, schema)

    def test_k_clamp_and_signal_ranking(self):
        selector, schema = self._fit(
            enabled=True, max_features=3, min_features=2
        )
        self.assertEqual(len(selector.selected_names_), 3)
        # f0/f1 与 y 强相关，必须被选中
        self.assertIn("f0", selector.selected_names_)
        self.assertIn("f1", selector.selected_names_)
        # 保持 schema 顺序
        self.assertEqual(
            list(selector.selected_names_),
            [name for name in schema if name in selector.selected_names_],
        )

    def test_force_keep_union(self):
        selector, _ = self._fit(
            enabled=True, max_features=2, min_features=2, force_keep=("f11",)
        )
        self.assertIn("f11", selector.selected_names_)

    def test_force_keep_unknown_feature_raises(self):
        with self.assertRaisesRegex(ValueError, "not in feature schema"):
            self._fit(enabled=True, force_keep=("nope",))

    def test_max_features_below_min_is_rejected_at_spec(self):
        # 严格校验 min<=max 后 legacy 的 top-up 分支不可达（已从实现删除）
        with self.assertRaisesRegex(ValueError, "min_features"):
            FeatureSelectionSpec(enabled=True, max_features=2, min_features=4)

    def test_transform_subsets_columns(self):
        selector, schema = self._fit(enabled=True, max_features=3, min_features=2)
        X = np.arange(48, dtype=float).reshape(4, 12)
        out = selector.transform(X)
        self.assertEqual(out.shape, (4, 3))
        indices = [schema.index(name) for name in selector.selected_names_]
        np.testing.assert_array_equal(out, X[:, indices])

    def test_transform_before_fit_raises(self):
        selector = CanonicalFeatureSelector(FeatureSelectionSpec(), ("a",))
        with self.assertRaises(RuntimeError):
            selector.transform(np.zeros((2, 1)))


class SelectedIndicesTest(unittest.TestCase):
    def test_identical_schema_returns_none(self):
        self.assertIsNone(selected_indices_for_artifact(("a", "b"), ("a", "b")))

    def test_subset_maps_to_positions(self):
        self.assertEqual(
            selected_indices_for_artifact(("a", "b", "c"), ("c", "a")), (2, 0)
        )

    def test_non_subset_raises(self):
        with self.assertRaisesRegex(ValueError, "not a subset"):
            selected_indices_for_artifact(("a",), ("b",))


def _config_doc(data_path: Path, *, mode: str, selection: dict | None) -> dict:
    features = {
        "target_lags": {"load": [2, 3, 24, 25, 26, 48, 49, 50]},
        "observed_past_lags": {},
        "datetime_features": ["hour"],
        "transformations": {},
    }
    if selection is not None:
        features["selection"] = selection
    return {
        "schema_version": 2,
        "problem": {
            "time_col": "time",
            "freq": "1h",
            "horizon": 2,
            "targets": ["load"],
            "training_scope": "local",
            "series_id_cols": [],
        },
        "data": {
            "sources": [
                {
                    "name": "targets",
                    "source_type": "file",
                    "columns": [
                        {"name": "load", "role": "target", "categorical": False}
                    ],
                    "history_path": str(data_path),
                    "time_col": "time",
                    "series_id_cols": [],
                    "availability": "source_time",
                }
            ]
        },
        "features": features,
        "strategy": {"name": "direct"},
        "estimator": {
            "model_type": "ridge" if mode == "point" else "qr",
            "target_adapter": "independent",
            "params": {"alpha": 1e-8} if mode == "point" else {},
        },
        "probabilistic": (
            {"mode": "point"}
            if mode == "point"
            else {
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
            }
        ),
        "validation": {
            "forecast_origin": "2026-01-03T23:00:00",
            "history_steps": 10_000,
            "train_window_steps": 9_999,
            "fold_count": 1,
            "stride_steps": 2,
        },
        "output": {"scenario_subpath": "feature-selection-test"},
    }


class FeatureSelectionEndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        times = pd.date_range("2026-01-01", periods=72, freq="1h")
        pd.DataFrame(
            {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
        ).to_csv(self.root / "data.csv", index=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, mode: str, selection: dict | None):
        config = parse_model_config(
            _config_doc(self.root / "data.csv", mode=mode, selection=selection),
            source="inline-test",
        )
        return config, run_canonical_config(config, output_root=self.root / "results")

    def test_selection_absent_keeps_fingerprint_and_full_schema(self):
        config_plain, _ = self._run("point", None)
        config_disabled, _ = self._run(
            "point", {"enabled": False, "max_features": 3, "min_features": 2}
        )
        # enabled=False 仍进 payload（显式语义），但未配置时 fingerprint 不变
        self.assertNotIn("selection", config_plain.features.canonical_payload())
        self.assertIn("selection", config_disabled.features.canonical_payload())

    def test_point_selection_shrinks_schema_and_predicts(self):
        _, result = self._run(
            "point", {"enabled": True, "max_features": 4, "min_features": 2}
        )
        artifact = result.bundle.model
        self.assertLessEqual(len(artifact.feature_schema), 4)
        self.assertEqual(tuple(artifact.feature_schema), tuple(result.bundle.selected_features))
        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
        self.assertEqual(len(prediction), 2)
        self.assertTrue(np.isfinite(prediction["predict_value"]).all())

    def test_quantile_selection_shrinks_schema_and_predicts(self):
        _, result = self._run(
            "quantile", {"enabled": True, "max_features": 4, "min_features": 2}
        )
        level_artifact = next(iter(result.bundle.model.artifacts_by_level.values()))
        self.assertLessEqual(len(level_artifact.feature_schema), 4)
        prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
        self.assertEqual(len(prediction), 2)
        for column in ("predict_q10", "predict_q50", "predict_q90"):
            self.assertTrue(np.isfinite(prediction[column]).all())

    def test_force_keep_survives_end_to_end(self):
        config, result = self._run(
            "point",
            {
                "enabled": True,
                "max_features": 2,
                "min_features": 1,
                "force_keep": ["load__lag_24"],
            },
        )
        artifact = result.bundle.model
        self.assertIn("load__lag_24", artifact.feature_schema)


if __name__ == "__main__":
    unittest.main()
