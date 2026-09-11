"""联通 8 月矩阵与 14+1 天几何；仅解析配置，不训练业务数据。"""
from pathlib import Path
from dataclasses import replace
import tempfile
import unittest
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from config.config_loader import load_yaml_config
from forecasting_core.specs import FixedStepBacktestSpec, ForecastConfigSpec
from model_pipeline.supervised_design import minimum_history_rows
from model_pipeline.supervised_design import SupervisedDesignBuilder, raw_history_backtest_windows
from data_loading import SourceRegistry
from model_pipeline.runner import CanonicalBaseModelRunner, run_canonical_config
from model_testing.geometry import TimeGeometry
from model_training.strategies.base import target_plan_for_config
from scripts.check_model_configs import check_model_yaml

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/add_weather"
VARIANTS = (
    "direct-pointwise", "direct-pointwise-horizon", "direct", "recursive",
    "dirrec", "dirmo", "recmo", "dirrecmo", "mimo",
)


class LiantongAugustConfigsTest(unittest.TestCase):
    def test_nine_variants_on_small_synthetic_sources(self):
        # 明确合成、缩小几何的接线测试，不是联通场景效果或正式回测证据。
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            times = pd.date_range("2026-08-01", periods=96, freq="1h")
            source = root / "target.csv"
            weather_path = root / "weather.csv"
            weather_history_path = root / "weather_history.csv"
            x = np.arange(len(times), dtype=float)
            pd.DataFrame({"time": times, "value": 100 + x / 10 + np.sin(x)}).to_csv(source, index=False)
            weather = pd.DataFrame({"ts": times})
            for column in ("rt_tt2", "cal_rh", "rt_ssr", "rt_ws10", "rt_dt",
                           "pred_tt2", "pred_rh", "pred_ssrd", "pred_ws10", "pred_dt",
                           "rt_ps", "rt_rain", "pred_ps", "pred_rain"):
                weather[column] = 10 + x / 100
            weather.to_csv(weather_history_path, index=False)
            # future 故意不存在：训练和滑窗测试不得触碰它。
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    config = load_yaml_config(DIRECTORY / f"lgbm_{variant}.yaml")
                    assert isinstance(config, ForecastConfigSpec)
                    assert config.strategy is not None
                    transforms = cast(dict[str, Any], config.features.canonical_payload()["transformations"])
                    transforms["advanced"]["rolling"]["windows"] = [4, 8, 16, 28]
                    sources = tuple(
                        replace(s, history_path=str(source)) if s.name == "target_history"
                        else replace(s, history_path=str(weather_history_path), future_path=str(weather_path), inference_columns=dict(s.inference_columns or ())) if s.name == "weather"
                        else s for s in config.data.sources
                    )
                    config = replace(config,
                        problem=replace(config.problem, freq="1h", horizon=4),
                        data=replace(config.data, sources=sources),
                        features=replace(config.features, target_lags={"value": [4, 8, 12, 16, 20, 24, 28]}, transformations=transforms),
                        strategy=replace(config.strategy, output_chunk_length=2) if config.strategy.output_chunk_length else config.strategy,
                        estimator=replace(config.estimator, params={"n_estimators": 5, "num_leaves": 4, "min_child_samples": 2, "verbosity": -1}),
                        validation={"forecast_origin": str(times[-1]), "schedule_mode": "intraday",
                                    "history_steps": 40, "train_window_steps": 13, "fold_count": 2, "stride_steps": 4},
                    )
                    config = replace(config, validation={**dict(config.validation),
                        "train_history_steps": 56,
                        "train_window_steps": 56 - minimum_history_rows(config) - 4 + 1,
                    })
                    with patch.object(CanonicalBaseModelRunner, "final_bundle_inputs", side_effect=AssertionError("final fit forbidden")):
                        result = run_canonical_config(config, output_root=root / variant, backtest_only=True)
                    self.assertFalse(pd.read_csv(result.test_dir / "test_scores_df.csv").empty)
                    self.assertEqual(list((root / variant).rglob("model.pkl")), [])
                    self.assertEqual(list((root / variant).rglob("prediction.csv")), [])
                    baseline = pd.read_csv(result.test_dir / "cv_plot_df.csv")
                    changed = pd.DataFrame({"time": times, "value": 100 + x / 10 + np.sin(x)})
                    # 两个 holdout 分别在索引87/91，最早允许原始历史从32开始。
                    changed.loc[:31, "value"] += 1000000
                    changed.to_csv(source, index=False)
                    rerun = run_canonical_config(config, output_root=root / f"{variant}-perturbed", backtest_only=True)
                    pd.testing.assert_frame_equal(baseline, pd.read_csv(rerun.test_dir / "cv_plot_df.csv"))
                    changed["value"] = 100 + x / 10 + np.sin(x)
                    changed.to_csv(source, index=False)

    def test_exact_matrix_and_source_feature_contract(self):
        paths = sorted(DIRECTORY.glob("*.yaml"))
        self.assertEqual({p.name for p in paths}, {f"lgbm_{name}.yaml" for name in VARIANTS})
        identities = set()
        for path in paths:
            with self.subTest(path=path.name):
                config = load_yaml_config(path)
                assert isinstance(config, ForecastConfigSpec)
                assert config.strategy is not None
                _, errors = check_model_yaml(str(path))
                self.assertEqual(errors, [])
                self.assertEqual(config.problem.time_col, "time")
                self.assertEqual(config.problem.targets, ("value",))
                self.assertEqual(config.problem.freq, "5min")
                self.assertEqual(config.problem.horizon, 288)
                self.assertEqual(config.estimator.model_type, "lightgbm")
                self.assertEqual(config.probabilistic["mode"], "point")
                self.assertNotIn("target", config.features.transformations)
                self.assertEqual(set(config.features.transformations["advanced"]), {"rolling", "expanding"})
                sources = {source.name: source for source in config.data.sources}
                self.assertEqual(set(sources), {"target_history", "chinese_holiday", "weather"})
                self.assertEqual(sources["target_history"].history_path,
                    "dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/target_power_5min_20260801_20260831.csv")
                self.assertEqual(sources["chinese_holiday"].generator, "chinese_holiday")
                assert sources["weather"].inference_columns is not None
                self.assertEqual(len(sources["weather"].inference_columns), 6)
                self.assertEqual(dict(sources["weather"].inference_columns),
                    {"rt_ssr": "pred_ssrd", "rt_tt2": "pred_tt2", "cal_rh": "pred_rh", "rt_ws10": "pred_ws10",
                     "rt_ps": "pred_ps", "rt_rain": "pred_rain"})
                self.assertEqual(sources["weather"].history_path,
                    "dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT/weather_history_5min_20260801_20260831.csv")
                self.assertIsNone(sources["weather"].future_path)
                self.assertIsNone(sources["weather"].backtest_path)
                variant = path.stem.removeprefix("lgbm_")
                expected_strategy = "direct" if variant.startswith("direct-pointwise") else variant
                self.assertEqual(config.strategy.name.value, expected_strategy)
                if variant.startswith("direct-pointwise"):
                    direct = config.features.transformations["direct"]
                    self.assertTrue(direct["align_to_target"])
                    self.assertEqual(direct["horizon_feature"]["enabled"], variant.endswith("-horizon"))
                    self.assertEqual(target_plan_for_config(config).model_count, 1)
                identities.add(config.fingerprint())
        self.assertEqual(len(identities), 9)

    def test_every_fold_has_exactly_fourteen_days_of_raw_history(self):
        # 调度读取生产目标的时间覆盖；不训练正式业务模型。
        times = pd.date_range("2026-08-01", "2026-08-31 23:55", freq="5min")
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                config = load_yaml_config(DIRECTORY / f"lgbm_{variant}.yaml")
                assert isinstance(config, ForecastConfigSpec)
                spec = config.validation.backtest
                assert isinstance(spec, FixedStepBacktestSpec)
                self.assertEqual(spec.train_history_steps, 4032)
                self.assertEqual(spec.train_window_steps, 4032 - minimum_history_rows(config) - 288 + 1)
                self.assertEqual(spec.stride_steps, 288)
                self.assertEqual(spec.fold_count, 17)
                origin = pd.Timestamp(config.validation["forecast_origin"])
                self.assertEqual(origin, times[-1])

                geometry = TimeGeometry(pd.offsets.Minute(5), 288)
                builder = SupervisedDesignBuilder(config, SourceRegistry(config.data, ROOT))
                folds = raw_history_backtest_windows(builder, origin)
                self.assertEqual(len(folds), 17)
                self.assertEqual(geometry.label_start(folds[0].origin), pd.Timestamp("2026-08-15"))
                self.assertEqual(geometry.label_end(folds[-1].origin), times[-1])
                for fold in folds:
                    self.assertEqual(len(fold.train_indices), spec.train_window_steps)
                    train_start = pd.Timestamp(fold.metadata["raw_history_start"])
                    train_end = pd.Timestamp(fold.metadata["raw_history_end"])
                    self.assertEqual(train_start, geometry.label_start(fold.origin) - pd.Timedelta(days=14))
                    self.assertEqual(train_end, fold.origin)
                    self.assertEqual(len(pd.date_range(train_start, train_end, freq="5min")), 4032)


if __name__ == "__main__":
    unittest.main()
