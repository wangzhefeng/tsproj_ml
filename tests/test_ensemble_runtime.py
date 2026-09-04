# -*- coding: utf-8 -*-
"""E5: reference-based ensemble runtime — full lifecycle on synthetic members.

TDD matrix (v4 §10/E5): strategy/estimator variation, point vs quantile,
learned-method outer-holdout isolation, cache reuse, NNLS parity with the
E0 function-level golden, and bundle-level contracts.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from time import sleep
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from feature_engineering.cache import COMPILED_CACHE_DIR_NAME
from model_ensemble.contracts import EnsembleRuntimeServices
from model_ensemble.loader import load_ensemble_config
from model_ensemble.runtime import run_ensemble_config
from model_forecasting.runtime import CanonicalBaseModelRunner, persist_model_bundle
from model_forecasting.resource_planner import plan_ensemble_resources

from model_ensemble.methods.linear_blending import fit_nonnegative_stacking_weights
from forecasting_core.specs.config import parse_model_config


RUNTIME_SERVICES = EnsembleRuntimeServices(
    runner_factory=CanonicalBaseModelRunner,
    persist_bundle=persist_model_bundle,
    plan_resources=plan_ensemble_resources,
)


def _member_doc(
    strategy: str,
    estimator: str,
    subpath: str,
    data_name: str = "targets",
) -> dict:
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
                    "name": data_name,
                    "source_type": "file",
                    "columns": [{"name": "load", "role": "target", "categorical": False}],
                    "history_path": "data.csv",
                    "time_col": "time",
                    "series_id_cols": [],
                    "availability": "source_time",
                }
            ]
        },
        "features": {
            "target_lags": {"load": [2, 3]},
            "observed_past_lags": {},
            "datetime_features": [],
            "transformations": {},
        },
        "strategy": {"name": strategy},
        "estimator": {
            "model_type": estimator,
            "target_adapter": "independent",
            "params": {"alpha": 1e-8} if estimator == "ridge" else {},
        },
        "probabilistic": {"mode": "point"},
        "validation": {
            "forecast_origin": "2026-01-03T23:00:00",
            "history_steps": 10_000,
            "train_window_steps": 9_999,
            "fold_count": 1,
            "stride_steps": 2,
        },
        "output": {"scenario_subpath": subpath},
    }


def _ensemble_doc(method: str, mode: str = "point") -> dict:
    probabilistic = (
        {"mode": "point"}
        if mode == "point"
        else {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
    )
    estimator_b = "qr" if mode == "quantile" else "ridge"
    return {
        "schema_version": 2,
        "problem": _member_doc("direct", "ridge", "x")["problem"],
        "data": _member_doc("direct", "ridge", "x")["data"],
        "probabilistic": probabilistic,
        "ensemble": {
            "members": [
                {"name": "m_direct", "config_ref": "member_direct.yaml"},
                {"name": "m_recursive", "config_ref": "member_recursive.yaml"},
            ],
            "oof": {"train_window_steps": 6, "fold_count": 2, "stride_steps": 1},
            "method": {"name": method},
        },
        "validation": {
            "forecast_origin": "2026-01-03T23:00:00",
            "history_steps": 10_000,
            "train_window_steps": 9_999,
            "fold_count": 1,
            "stride_steps": 2,
        },
        "output": {"scenario_subpath": f"ens-{method}-{mode}"},
    }


class EnsembleRuntimeTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        times = pd.date_range("2026-01-01", periods=72, freq="1h")
        pd.DataFrame(
            {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
        ).to_csv(self.root / "data.csv", index=False)
        (self.root / "member_direct.yaml").write_text(
            yaml.safe_dump(_member_doc("direct", "ridge", "direct")),
            encoding="utf-8",
        )
        (self.root / "member_recursive.yaml").write_text(
            yaml.safe_dump(_member_doc("recursive", "ridge", "recursive")),
            encoding="utf-8",
        )

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, method: str, mode: str = "point", **kwargs):
        doc = _ensemble_doc(method, mode=mode)
        path = self.root / f"ens_{method}_{mode}.yaml"
        path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        config = load_ensemble_config(path)
        kwargs.setdefault("base_dir", self.root)
        kwargs.setdefault("services", RUNTIME_SERVICES)
        return run_ensemble_config(config, output_root=self.root, **kwargs)


class EnsembleRuntimeMatrixTest(EnsembleRuntimeTestBase):
    def test_averaging_point_end_to_end(self):
        result = self._run("averaging")
        self.assertEqual(result["execution_plan"].selected_axis, "serial")
        resolved = json.loads(
            (result["forecast_dir"] / "resolved_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            resolved["runtime"]["resources"]["execution_plan"]["selected_axis"],
            "serial",
        )
        self.assertEqual(
            resolved["runtime"]["resources"]["workload"]["member_count"],
            2,
        )
        self.assertEqual(
            set(resolved["runtime"]["resources"]["stage_wall_seconds"]),
            {"raw_design", "ensemble_fit", "persist", "total"},
        )
        combined = result["combined_values"]
        self.assertEqual(combined.shape, (1, 2, 1))
        members = result["member_final_values"]
        manual = (members["m_direct"] + members["m_recursive"]) / 2.0
        np.testing.assert_allclose(combined, manual)
        cv_path = result["test_dir"] / "cv_plot_df.csv"
        self.assertTrue(cv_path.exists())
        cv = pd.read_csv(cv_path)
        self.assertTrue(
            {
                "series_id",
                "time",
                "target",
                "window",
                "actual_value",
                "predict_value",
            }
            .issubset(cv.columns)
        )
        self.assertFalse(
            cv.duplicated(["series_id", "time", "target", "window"]).any()
        )

    def test_learned_methods_end_to_end(self):
        for method in ("weighted", "linear_blending", "stacking"):
            with self.subTest(method=method):
                result = self._run(method)
                self.assertTrue(np.isfinite(result["combined_values"]).all())
                self.assertEqual(result["audit"]["method"], method)

    def test_oof_cache_reused_across_fusion_methods(self):
        results = tuple(
            self._run(method)
            for method in ("averaging", "weighted", "linear_blending", "stacking")
        )
        self.assertEqual(len({result["oof_fingerprint"] for result in results}), 1)
        self.assertEqual(
            [result["oof_cache_hit"] for result in results],
            [False, True, True, True],
        )
        self.assertEqual(
            len({result["bundle"].config_fingerprint for result in results}),
            4,
        )
        fingerprint = results[0]["oof_fingerprint"]
        cache_root = self.root / "_ensemble_oof" / fingerprint
        self.assertTrue(cache_root.exists())

    def test_member_config_alias_reuses_semantic_oof_cache(self):
        first = self._run("averaging")
        alias_path = self.root / "member_direct_alias.yaml"
        alias_path.write_bytes((self.root / "member_direct.yaml").read_bytes())
        doc = _ensemble_doc("averaging")
        doc["ensemble"]["members"][0]["config_ref"] = alias_path.name
        config_path = self.root / "ens_averaging_alias.yaml"
        config_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
        second = run_ensemble_config(
            load_ensemble_config(config_path),
            output_root=self.root,
            base_dir=self.root,
            services=RUNTIME_SERVICES,
        )

        self.assertEqual(second["oof_fingerprint"], first["oof_fingerprint"])
        self.assertTrue(second["oof_cache_hit"])
        self.assertNotEqual(
            second["bundle"].config_fingerprint,
            first["bundle"].config_fingerprint,
        )

    def test_members_with_same_design_share_compiled_cache(self):
        (self.root / "member_recursive.yaml").write_text(
            yaml.safe_dump(_member_doc("direct", "lasso", "recursive")),
            encoding="utf-8",
        )
        runners = []

        def runner_factory(config, registry, origin, *, compiled_cache_root):
            runner = CanonicalBaseModelRunner(
                config,
                registry,
                origin,
                compiled_cache_root=compiled_cache_root,
            )
            runners.append(runner)
            return runner

        services = EnsembleRuntimeServices(
            runner_factory=runner_factory,
            persist_bundle=persist_model_bundle,
            plan_resources=plan_ensemble_resources,
        )
        self._run("averaging", services=services)

        self.assertEqual(len(runners), 2)
        self.assertFalse(runners[0].compiled_cache_hit)
        self.assertTrue(runners[1].compiled_cache_hit)
        self.assertEqual(
            runners[0].compiled_cache_fingerprint,
            runners[1].compiled_cache_fingerprint,
        )
        cache_parent = self.root / COMPILED_CACHE_DIR_NAME
        self.assertEqual(len(tuple(cache_parent.iterdir())), 1)

    def test_concurrent_fusion_methods_generate_oof_once(self):
        fit_calls = 0
        count_lock = Lock()
        original_fit = CanonicalBaseModelRunner.fit

        def slow_fit(runner, train_indices):
            nonlocal fit_calls
            with count_lock:
                fit_calls += 1
            sleep(0.05)
            return original_fit(runner, train_indices)

        with patch.object(CanonicalBaseModelRunner, "fit", slow_fit), patch(
            "model_ensemble.runtime.write_forecast_results"
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = tuple(
                    executor.map(self._run, ("averaging", "linear_blending"))
                )

        self.assertEqual(fit_calls, 4)
        self.assertEqual(
            len({result["oof_fingerprint"] for result in results}),
            1,
        )
        self.assertEqual(
            sorted(result["oof_cache_hit"] for result in results),
            [False, True],
        )

    def test_fused_oof_scores_point(self):
        """融合 OOF 评分（2026-08-30）：point 模式产出点指标，无概率分数。"""
        result = self._run("averaging")
        fused = result["fused_oof_scores"]
        point = fused["point"]
        self.assertIsNone(fused["probabilistic"])
        self.assertEqual(
            set(point["scope"]),
            {"target", "aggregate", "horizon", "aggregate_horizon"},
        )
        self.assertTrue(np.isfinite(point["MAE"]).all())
        # 不变量：averaging 的融合 MAE 不超过最差成员的 OOF MAE
        member_maes = [
            scores["load"]["mae"]
            for scores in result["audit"]["member_scores"].values()
        ]
        fused_mae = point[point["scope"] == "aggregate"]["MAE"].iloc[0]
        self.assertLessEqual(fused_mae, max(member_maes) + 1e-9)

    def test_fused_oof_scores_quantile(self):
        """融合 OOF 评分：quantile 模式附带 pinball/central 区间，且
        pinball@0.5 × 2 与点 MAE 交叉一致。"""
        # 成员必须是 quantile 模型（v4 语义：quantile ensemble 融合 quantile 成员）
        for name, strategy in (
            ("member_direct", "direct"),
            ("member_recursive", "recursive"),
        ):
            doc = _member_doc(strategy, "qr", name)
            doc["probabilistic"] = {
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
            }
            (self.root / f"{name}.yaml").write_text(
                yaml.safe_dump(doc), encoding="utf-8"
            )
        result = self._run("averaging", mode="quantile")
        fused = result["fused_oof_scores"]
        prob = fused["probabilistic"]
        self.assertIsNotNone(prob)
        metrics = set(prob["metric"])
        self.assertIn("pinball", metrics)
        self.assertIn("interval_coverage", metrics)
        self.assertEqual(set(prob["interval_name"].dropna()), {"central80"})
        pin50 = prob[
            (prob["metric"] == "pinball")
            & (prob["quantile"] == 0.5)
            & (prob["scope"] == "aggregate")
        ]["value"].iloc[0]
        point_mae = fused["point"][
            fused["point"]["scope"] == "aggregate"
        ]["MAE"].iloc[0]
        self.assertAlmostEqual(pin50 * 2.0, point_mae, places=9)

    def test_stacking_matches_nnls_reference_on_same_inputs(self):
        """New blending shares the NNLS optimizer family with the old function."""
        result = self._run("linear_blending")
        self.assertTrue(np.isfinite(result["combined_values"]).all())
        # identical inputs to the legacy function must give identical weights
        y = np.array([[10.0, 12.0], [20.0, 18.0]])
        e = np.array([[1.0, -1.0], [1.0, -1.0]])
        legacy = fit_nonnegative_stacking_weights(
            [y + e, y - e], y, fallback_weights=(0.5, 0.5)
        )
        from model_ensemble.methods import linear_blending as lb

        artifact = lb.fit_linear_blending(
            {"a": (y + e)[:, :, None], "b": (y - e)[:, :, None]},
            y[:, :, None],
        )
        modern = artifact.weights_by_target["target_0"]
        np.testing.assert_allclose(legacy, modern, atol=1e-12)

    def test_different_estimators_same_strategy(self):
        (self.root / "member_recursive.yaml").write_text(
            yaml.safe_dump(_member_doc("recursive", "ridge", "recursive")),
            encoding="utf-8",
        )
        (self.root / "member_direct.yaml").write_text(
            yaml.safe_dump(_member_doc("direct", "lasso", "direct")),
            encoding="utf-8",
        )
        result = self._run("averaging")
        self.assertTrue(np.isfinite(result["combined_values"]).all())


if __name__ == "__main__":
    unittest.main()
