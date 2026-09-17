"""算力配置与 baseline 的正交性、真实特征值和 as-of 边界；不拟合正式模型。"""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from data_loading import BUILTIN_GENERATORS, SourceRegistry
from model_pipeline.supervised_design import SupervisedDesignBuilder

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"
TRAINING_COLUMNS = ["training_cpu_util_sample_mean", "training_cpu_util_sample_std", "training_gpu_power_usage_sum_kw"]
INFERENCE_COLUMNS = ["inference_gpu_memory_amount_sum_raw", "inference_gpu_memory_util_sample_mean",
                     "inference_memory_amount_sum_raw", "inference_memory_total_sample_mean_raw"]
# 组名 -> 有序算力 source 名 -> 该 source 明确入模列
GROUPS = {
    "add_training_compute": {"training_compute": TRAINING_COLUMNS},
    "add_inference_compute": {"inference_compute": INFERENCE_COLUMNS},
    "add_training_inference_compute": {"training_compute": TRAINING_COLUMNS, "inference_compute": INFERENCE_COLUMNS},
}


class LiantongComputeConfigsTest(unittest.TestCase):
    def paths(self):
        paths = sorted(p for group in GROUPS for p in (SCENARIO / group).glob("*.yaml"))
        self.assertEqual(len(paths), 27)
        return paths

    def test_matrix_changes_only_compute_source_lags_and_output(self):
        names = {p.name for p in (SCENARIO / "baseline").glob("lgbm_*.yaml")}
        self.assertEqual(len(names), 9)
        for group in GROUPS:
            self.assertEqual({p.name for p in (SCENARIO / group).glob("*.yaml")}, names)
        for path in self.paths():
            with self.subTest(path=path):
                config = load_yaml_config(path)
                baseline = load_yaml_config(SCENARIO / "baseline" / path.name)
                self.assertNotEqual(config.fingerprint(), baseline.fingerprint())
                expected_sources = GROUPS[path.parent.name]
                payload = config.canonical_payload()
                compute_sources = [s for s in payload["data"]["sources"] if s["name"] in expected_sources]
                self.assertEqual([s["name"] for s in compute_sources], list(expected_sources))
                for source in compute_sources:
                    columns = expected_sources[source["name"]]
                    self.assertEqual([c["name"] for c in source["columns"]], columns)
                    self.assertEqual({c["role"] for c in source["columns"]}, {"observed_past"})
                    self.assertEqual(source["provider"], "persistence")
                    self.assertEqual(source["availability"], "source_time")
                    self.assertTrue((ROOT / source["history_path"]).is_file())
                payload["data"]["sources"] = [s for s in payload["data"]["sources"] if s["name"] not in expected_sources]
                all_columns = [c for cols in expected_sources.values() for c in cols]
                self.assertEqual(payload["features"]["observed_past_lags"], {c: [288, 576] for c in all_columns})
                payload["features"]["observed_past_lags"] = {}
                # 组合组额外带 baseline_opt 同口径的 recent_state；剥离后须与 baseline 逐字段一致。
                payload["features"]["transformations"]["advanced"].pop("recent_state", None)
                payload["output"] = baseline.canonical_payload()["output"]
                self.assertEqual(payload, baseline.canonical_payload())
                self.assertTrue(config.output["scenario_subpath"].endswith("/" + path.parent.name))

    def test_real_training_forecast_values_and_boundary_isolation(self):
        for path in self.paths():
            config = load_yaml_config(path)
            expected_sources = GROUPS[path.parent.name]
            compute_sources = [s for s in config.data.sources if s.name in expected_sources]
            values = {s.name: pd.read_csv(ROOT / s.history_path, parse_dates=["time"]).set_index("time")
                      for s in compute_sources}
            columns = [c for cols in expected_sources.values() for c in cols]
            for origin_text in ("2026-08-14 23:55", "2026-08-18 23:55", "2026-08-30 23:55"):
                with self.subTest(path=path, origin=origin_text):
                    origin = pd.Timestamp(origin_text)
                    start = origin - pd.Timedelta(minutes=5 * 4031)
                    builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, ROOT, generators=BUILTIN_GENERATORS), history_start=start)
                    steps = (1, 144, 288)
                    request = builder.request(origin)
                    compiled = builder.compiler.compile(builder.registry.materialize(request), request, horizon_steps=steps)
                    added = [c for c in compiled.frame.columns if any(c.startswith(name + "__lag_") for name in columns)]
                    self.assertEqual(len(added), len(columns) * 2)
                    self.assertTrue(np.isfinite(compiled.frame[added].to_numpy()).all())
                    proofs = [p for p in compiled.visibility_proof if p.source_name in expected_sources]
                    self.assertEqual(len(proofs), len(added) * len(steps))
                    frozen = path.stem in {"lgbm_direct", "lgbm_mimo", "lgbm_dirmo"}
                    for proof in proofs:
                        column, lag = proof.feature_name.rsplit("__lag_", 1)
                        anchor = origin if frozen else proof.target_time
                        self.assertEqual(proof.source_time, anchor - pd.Timedelta(minutes=5 * int(lag)))
                        self.assertLessEqual(proof.available_at, origin)
                        self.assertLessEqual(proof.source_time, origin)
                        self.assertIsNone(proof.provider)
                        actual = compiled.frame.iloc[steps.index(proof.horizon_step)][proof.feature_name]
                        self.assertAlmostEqual(actual, values[proof.source_name].loc[proof.source_time, column])
                    if origin_text == "2026-08-14 23:55":
                        designs, labels = builder.training_row(origin - pd.Timedelta(days=1))
                        self.assertTrue(designs)
                        self.assertTrue(all(np.isfinite(d).all() for d in designs))
                        self.assertTrue(np.isfinite(labels).all())
                    if origin_text == "2026-08-30 23:55":
                        with TemporaryDirectory() as temp:
                            changed_sources = []
                            for source in config.data.sources:
                                if source.name not in expected_sources:
                                    changed_sources.append(source)
                                    continue
                                mutated = values[source.name].copy()
                                source_columns = expected_sources[source.name]
                                mutated.loc[(mutated.index < start) | (mutated.index > origin), source_columns] += 1000000
                                changed_path = Path(temp) / f"{source.name}.csv"
                                mutated.reset_index().to_csv(changed_path, index=False)
                                changed_sources.append(replace(source, history_path=str(changed_path)))
                            changed_config = replace(config, data=replace(config.data, sources=tuple(changed_sources)))
                            changed = SupervisedDesignBuilder(changed_config, SourceRegistry(changed_config.data, ROOT, generators=BUILTIN_GENERATORS), history_start=start)
                            result = changed.compiler.compile(changed.registry.materialize(request), request, horizon_steps=steps)
                            np.testing.assert_allclose(compiled.frame[added], result.frame[added], rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
