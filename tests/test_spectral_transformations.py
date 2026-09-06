# -*- coding: utf-8 -*-
"""advanced.fourier / advanced.wavelet 频域特征与 rolling entropy 统计测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import InformationSetRequest, SourceRegistry
from feature_engineering import FeatureCompiler
from feature_engineering.spectral import (
    fourier_features,
    signal_entropy,
    wavelet_energy_features,
)
from forecasting_core.specs import (
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
)


def _sine_mix(n):
    # 周期 16（振幅 10）+ 周期 8（振幅 5）+ 常量 500；窗口 64 时恰好落 bin，无泄漏
    k = np.arange(n, dtype=float)
    return 500.0 + 10.0 * np.sin(2.0 * np.pi * k / 16.0) + 5.0 * np.sin(
        2.0 * np.pi * k / 8.0
    )


class SpectralPureFunctionTest(unittest.TestCase):
    def test_signal_entropy_uniform_and_degenerate(self):
        uniform = np.ones(10)
        self.assertAlmostEqual(signal_entropy(uniform), np.log2(10.0), places=9)
        self.assertEqual(signal_entropy(np.zeros(4)), 0.0)

    def test_fourier_features_recovers_known_components(self):
        values = _sine_mix(64)
        features = fourier_features(
            values, top_k=2, band_periods=((2, 12), (12, 32))
        )
        # 主成分：周期 16 -> freq 4/64 = 0.0625，振幅 10；次成分：周期 8
        self.assertAlmostEqual(features["amp_1"], 10.0, places=6)
        self.assertAlmostEqual(features["freq_1"], 4.0 / 64.0, places=9)
        self.assertAlmostEqual(features["amp_2"], 5.0, places=6)
        self.assertAlmostEqual(features["freq_2"], 8.0 / 64.0, places=9)
        # 谱质心：功率加权 (100*0.0625 + 25*0.125) / 125
        self.assertAlmostEqual(features["centroid"], 0.075, places=9)
        # 频带能量比：周期 8 落 [2,12)，周期 16 落 [12,32)
        self.assertAlmostEqual(features["bandenergy_1"], 0.2, places=9)
        self.assertAlmostEqual(features["bandenergy_2"], 0.8, places=9)
        self.assertEqual(
            set(features),
            {
                "amp_1", "amp_2", "freq_1", "freq_2", "phase_1", "phase_2",
                "centroid", "bandenergy_1", "bandenergy_2",
            },
        )

    def test_fourier_features_zero_signal(self):
        features = fourier_features(np.full(32, 7.0), top_k=1)
        self.assertEqual(features["amp_1"], 0.0)
        self.assertEqual(features["centroid"], 0.0)

    def test_fourier_features_rejects_excessive_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            fourier_features(np.ones(8), top_k=10)

    def test_wavelet_energy_ratios_sum_to_one(self):
        values = _sine_mix(64)
        features = wavelet_energy_features(values, wavelet="db4", level=2)
        self.assertEqual(set(features), {"a2", "d2", "d1"})
        total = sum(features.values())
        self.assertAlmostEqual(total, 1.0, places=9)
        self.assertTrue(all(value >= 0.0 for value in features.values()))

    def test_wavelet_rejects_invalid_wavelet_and_level(self):
        values = _sine_mix(64)
        with self.assertRaisesRegex(ValueError, "unknown wavelet"):
            wavelet_energy_features(values, wavelet="not_a_wavelet", level=2)
        with self.assertRaisesRegex(ValueError, "level"):
            wavelet_energy_features(values, wavelet="db4", level=10)
        with self.assertRaisesRegex(ValueError, "level"):
            wavelet_energy_features(values, wavelet="db4", level=0)


class SpectralCompilerTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.times = pd.date_range("2026-01-01", periods=96, freq="1h")
        pd.DataFrame(
            {"ts": self.times, "load": _sine_mix(96)}
        ).to_csv(self.base_dir / "targets.csv", index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def build_config(self, transformations):
        data = DataSpec(
            (
                DataSourceSpec(
                    name="targets",
                    source_type="file",
                    columns=(ColumnSpec("load", "target"),),
                    history_path="targets.csv",
                    time_col="ts",
                    availability="source_time",
                ),
            )
        )
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="ts",
                freq="1h",
                horizon=2,
                targets=("load",),
                training_scope="local",
            ),
            data=data,
            features=FeatureSpec(
                target_lags={"load": (2,)},
                observed_past_lags={},
                datetime_features=("hour",),
                transformations=transformations,
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(model_type="ridge", target_adapter="independent"),
            probabilistic={},
            validation={},
            output={},
        )

    def request(self, origin="2026-01-04 00:00"):
        origin = pd.Timestamp(origin)
        return InformationSetRequest(
            forecast_origin=origin,
            forecast_times=pd.date_range(
                origin + pd.Timedelta(hours=1), periods=2, freq="1h"
            ),
            series_ids=(),
        )

    def compile(self, config, request):
        compiler = FeatureCompiler(config)
        information_set = SourceRegistry(config.data, self.base_dir).materialize(request)
        return compiler.compile(information_set, request)

    def spectral_transformations(self):
        return {
            "advanced": {
                "fourier": {
                    "columns": ["load"],
                    "windows": [64],
                    "top_k": 2,
                    "band_periods": [[2, 12], [12, 32]],
                },
                "wavelet": {
                    "columns": ["load"],
                    "windows": [64],
                    "wavelet": "db4",
                    "level": 2,
                },
            }
        }

    def test_single_compile_fourier_wavelet_match_pure_functions(self):
        config = self.build_config(self.spectral_transformations())
        compiled = self.compile(config, self.request())
        frame = compiled.frame

        # 原点 2026-01-04 00:00 的可见历史为前 73 点，trailing 窗取最后 64 点
        visible = _sine_mix(96)[:73][-64:]
        expected_fourier = fourier_features(
            visible, top_k=2, band_periods=((2, 12), (12, 32))
        )
        expected_wavelet = wavelet_energy_features(visible, wavelet="db4", level=2)
        for key, value in expected_fourier.items():
            column = f"load_fft_{key}_64"
            self.assertIn(column, frame.columns)
            np.testing.assert_allclose(frame[column].to_numpy(), value, rtol=1e-9)
        for key, value in expected_wavelet.items():
            column = f"load_wavelet_energy_{key}_64"
            self.assertIn(column, frame.columns)
            np.testing.assert_allclose(frame[column].to_numpy(), value, rtol=1e-9)

    def test_spectral_features_are_as_of_safe(self):
        # 在预测原点之后追加未来数据，重编译同一原点，特征值必须不变
        config = self.build_config(self.spectral_transformations())
        before = self.compile(config, self.request()).frame

        future_rows = pd.DataFrame(
            {
                "ts": pd.date_range("2026-01-05 01:00", periods=24, freq="1h"),
                "load": 9999.0,
            }
        )
        pd.concat(
            [pd.read_csv(self.base_dir / "targets.csv"), future_rows]
        ).to_csv(self.base_dir / "targets.csv", index=False)
        after = self.compile(config, self.request()).frame

        spectral_columns = [
            column
            for column in before.columns
            if "_fft_" in column or "_wavelet_energy_" in column
        ]
        self.assertTrue(spectral_columns)
        pd.testing.assert_frame_equal(before[spectral_columns], after[spectral_columns])

    def test_batch_compile_matches_single_compile(self):
        config = self.build_config(self.spectral_transformations())
        origins = ["2026-01-04 00:00", "2026-01-04 12:00"]
        requests = [self.request(origin) for origin in origins]
        compiler = FeatureCompiler(config)
        registry = SourceRegistry(config.data, self.base_dir)
        information_sets = [registry.materialize(request) for request in requests]

        eligibility = compiler.batch_eligibility(requests)
        self.assertTrue(eligibility.eligible, eligibility.reason_codes)

        batched = compiler.compile_batch(information_sets, requests)
        singles = [
            compiler.compile(information_set, request)
            for information_set, request in zip(information_sets, requests)
        ]
        for batch_result, single_result in zip(batched, singles):
            pd.testing.assert_frame_equal(batch_result.frame, single_result.frame)

    def test_insufficient_visible_history_raises(self):
        config = self.build_config(self.spectral_transformations())
        # 原点 2026-01-02 12:00 只有 36 个可见点，窗口 64
        with self.assertRaisesRegex(ValueError, "insufficient visible history"):
            self.compile(config, self.request("2026-01-02 12:00"))

    def test_invalid_fourier_spec_raises(self):
        config = self.build_config(
            {"advanced": {"fourier": {"columns": ["load"], "windows": [64], "top_k": 0}}}
        )
        with self.assertRaisesRegex(ValueError, "top_k"):
            self.compile(config, self.request())

    def test_rolling_entropy_stat_matches_manual(self):
        config = self.build_config(
            {
                "advanced": {
                    "rolling": {
                        "columns": ["load"],
                        "windows": [16],
                        "stats": ["entropy"],
                    }
                }
            }
        )
        compiled = self.compile(config, self.request())
        expected = signal_entropy(_sine_mix(96)[:73][-16:])
        np.testing.assert_allclose(
            compiled.frame["load_rolling_entropy_16"].to_numpy(), expected, rtol=1e-9
        )


class PeriodicityAnalysisScriptTest(unittest.TestCase):
    """data_process/periodicity_analysis.py 的冒烟测试（合成数据，已知周期 16）。

    覆盖配置驱动 CLI、FFT top-k 报告与 Engle-Granger 协整诊断。
    """

    def test_report_recovers_known_period(self):
        import subprocess

        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "series.csv"
            times = pd.date_range("2026-01-01", periods=256, freq="1h")
            k = np.arange(256, dtype=float)
            pd.DataFrame(
                {
                    "ts": times,
                    "load": 100.0 + 8.0 * np.sin(2.0 * np.pi * k / 16.0),
                    "temp": 20.0 + 3.0 * np.sin(2.0 * np.pi * k / 16.0),
                }
            ).to_csv(csv_path, index=False)
            config_path = Path(temp_dir) / "periodicity.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"source_path: {csv_path}",
                        "time_col: ts",
                        "target_col: load",
                        "seasonal_period: 16",
                        "fft_top_k: 5",
                        "coint_col: temp",
                        "plot: false",
                        f"output_dir: {temp_dir}",
                    ]
                ),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    str(root / ".venv" / "bin" / "python"),
                    str(root / "data_process" / "periodicity_analysis.py"),
                    str(config_path),
                ],
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = pd.read_csv(Path(temp_dir) / "series_periodicity_report.csv")
        metrics = dict(zip(report["metric"], report["value"]))
        top_periods = json.loads(metrics["fft_top_periods"])
        self.assertAlmostEqual(top_periods[0]["period_samples"], 16.0, places=6)
        self.assertEqual(str(metrics["stl_seasonal_period_used"]), "16")
        self.assertEqual(metrics["coint_col"], "temp")
        self.assertIn(metrics["coint_verdict"], ("协整", "不协整"))


if __name__ == "__main__":
    unittest.main()
