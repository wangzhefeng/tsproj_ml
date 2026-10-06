# -*- coding: utf-8 -*-
"""crossing 配置消费测试（2026-09-01 裂缝修复）。

钉住四条契约：
1. ``repair_marginal_quantile_crossing`` 按 method 分派（none/rearrangement/
   median_preserving_isotonic），默认方法与历史硬编码行为一致；
2. ``resolve_crossing_settings`` 是运行时唯一解析入口，缺省回落默认方法；
3. YAML 层合同拒绝 legacy ``crossing_method``/``conformal`` 键与非法 method；
4. ``report_raw`` 断链修复（2026-10-05）：配置 → 解析 → 组装段 metadata
   （``crossing_report``）→ 回测逐窗 execution_evidence，全链路真实到达。
"""

import unittest

import numpy as np
import pandas as pd

from forecasting_core.probability.spec import resolve_crossing_settings
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
from forecasting_core.specs.probabilistic import ProbabilisticConfigSpec
from forecasting_core.probability.distribution import MarginalForecastDistribution
from forecasting_core.tensors.quantile import MarginalQuantileForecastTensor
from forecasting_core.tensors.point import PointForecastTensor
from model_predicting.loops.predictor import (
    CanonicalMarginalQuantileForecaster,
    build_crossing_report,
    repair_marginal_quantile_crossing,
)
from model_testing.loops.scoring import score_holdout_fold
from model_training.estimators import EstimatorCapabilities
from model_training.quantile import CanonicalMarginalQuantileTrainer


def _crossed_tensor() -> MarginalQuantileForecastTensor:
    # q10 > q50 < q90 的交叉样本：下界高于锚点、上界低于锚点
    values = np.array(
        [
            [
                [[12.0, 10.0, 8.0]],  # 全部倒挂
                [[3.0, 5.0, 4.0]],  # q90 < q50
            ]
        ]
    )
    return MarginalQuantileForecastTensor(
        values=values,
        levels=(0.1, 0.5, 0.9),
        point_level=0.5,
        series_ids=("s0",),
        forecast_times=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        targets=("load",),
    )


class RepairDispatchTest(unittest.TestCase):
    def test_default_matches_legacy_hardcoded_behavior(self):
        tensor = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(tensor)
        # 历史行为：下半轴排序并钳到锚点以下，上半轴排序并钳到锚点以上，锚点不动
        np.testing.assert_allclose(
            repaired.values[0, 0, 0, :], [10.0, 10.0, 10.0]
        )
        np.testing.assert_allclose(
            repaired.values[0, 1, 0, :], [3.0, 5.0, 5.0]
        )

    def test_none_returns_tensor_unchanged(self):
        tensor = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(tensor, method="none")
        self.assertIs(repaired, tensor)

    def test_rearrangement_sorts_without_anchor(self):
        tensor = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(tensor, method="rearrangement")
        np.testing.assert_allclose(
            repaired.values[0, 0, 0, :], [8.0, 10.0, 12.0]
        )
        np.testing.assert_allclose(
            repaired.values[0, 1, 0, :], [3.0, 4.0, 5.0]
        )

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            repair_marginal_quantile_crossing(
                _crossed_tensor(), method="isotonic"
            )


class ResolveCrossingSettingsTest(unittest.TestCase):
    def test_default_without_crossing_block(self):
        method, report_raw = resolve_crossing_settings({"mode": "quantile"})
        self.assertEqual(method, "median_preserving_isotonic")
        self.assertTrue(report_raw)

    def test_explicit_block(self):
        method, report_raw = resolve_crossing_settings(
            {"crossing": {"method": "none", "report_raw": False}}
        )
        self.assertEqual(method, "none")
        self.assertFalse(report_raw)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            resolve_crossing_settings({"crossing": {"method": "isotonic"}})

    def test_unknown_key_raises(self):
        with self.assertRaises(ValueError):
            resolve_crossing_settings(
                {"crossing": {"method": "none", "bogus": 1}}
            )


class YamlContractTest(unittest.TestCase):
    def test_legacy_crossing_method_key_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilisticConfigSpec.from_mapping(
                {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                 "crossing_method": "isotonic"},
                source="test",
            )

    def test_legacy_conformal_key_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilisticConfigSpec.from_mapping(
                {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                 "conformal": {"method": "none"}},
                source="test",
            )

    def test_invalid_crossing_method_rejected(self):
        with self.assertRaises(ValueError):
            ProbabilisticConfigSpec.from_mapping(
                {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
                 "crossing": {"method": "isotonic"}},
                source="test",
            )

    def test_canonical_block_accepted(self):
        spec = ProbabilisticConfigSpec.from_mapping(
            {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9],
             "crossing": {"method": "rearrangement"}},
            source="test",
        )
        self.assertEqual(spec["crossing"]["method"], "rearrangement")


class BuildCrossingReportTest(unittest.TestCase):
    """build_crossing_report 数值合同（手工核算自 _crossed_tensor）。"""

    def test_repaired_report_matches_hand_calculation(self):
        raw = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(raw)
        report = build_crossing_report(raw, repaired)
        assert report is not None
        # 两行均有相邻交叉；相邻对 3/4 倒挂；正向违例幅度 (2+2+0+1)/4
        self.assertEqual(report["row_crossing_rate"], 1.0)
        self.assertAlmostEqual(report["adjacent_crossing_rate"], 0.75)
        self.assertAlmostEqual(report["crossing_magnitude"], 1.25)
        # 修复改动 3/6 个点（row0 q10/q90、row1 q90）；q50 锚点不动
        self.assertAlmostEqual(report["repair_changed_ratio"], 0.5)
        self.assertEqual(report["q50_changed_ratio"], 0.0)

    def test_method_none_reports_raw_only_zero_change(self):
        raw = _crossed_tensor()
        repaired = repair_marginal_quantile_crossing(raw, method="none")
        report = build_crossing_report(raw, repaired)
        assert report is not None
        self.assertEqual(report["row_crossing_rate"], 1.0)
        self.assertEqual(report["repair_changed_ratio"], 0.0)

    def test_single_level_has_no_crossing_semantics(self):
        tensor = MarginalQuantileForecastTensor(
            values=np.array([[[[1.0]]]]),
            levels=(0.5,),
            point_level=0.5,
            series_ids=("s0",),
            forecast_times=pd.DatetimeIndex(["2026-01-01"]),
            targets=("load",),
        )
        self.assertIsNone(build_crossing_report(tensor, tensor))


class _MeanQuantileRegressor:
    def __init__(self, level):
        self.level = float(level)

    def fit(self, X, y):
        self.value = float(np.mean(y) + (self.level - 0.5) * 2.0)
        return self

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


_CAPABILITIES = EstimatorCapabilities(
    scalar_target=True,
    scalar_quantile=True,
    native_multi_target_point=False,
    native_multi_target_quantile=False,
    sample_weight=False,
    categorical=False,
    nan_support=False,
)


def _quantile_config(probabilistic):
    return ForecastConfigSpec(
        problem=ForecastProblemSpec(
            time_col="time",
            freq="1h",
            horizon=2,
            targets=("load",),
            training_scope="local",
            series_id_cols=(),
        ),
        data=DataSpec(
            (
                DataSourceSpec(
                    name="targets",
                    source_type="file",
                    columns=(ColumnSpec("load", "target"),),
                    history_path="unused.csv",
                    time_col="time",
                    availability="source_time",
                ),
            )
        ),
        features=FeatureSpec(
            target_lags={},
            observed_past_lags={},
            datetime_features=(),
            transformations={},
        ),
        strategy=ForecastStrategySpec("direct"),
        estimator=EstimatorSpec(
            model_type="mean_quantile",
            target_adapter="independent",
        ),
        probabilistic=probabilistic,
        validation={},
        output={},
    )


def _predict_distribution(probabilistic) -> MarginalForecastDistribution:
    config = _quantile_config(probabilistic)
    trainer = CanonicalMarginalQuantileTrainer(
        config,
        estimator_factory_for_level=lambda level: lambda: _MeanQuantileRegressor(level),
        capabilities=_CAPABILITIES,
        feature_schema=("x",),
    )
    X_by_call = (np.arange(8.0).reshape(-1, 1), np.arange(8.0).reshape(-1, 1))
    Y = np.arange(16.0).reshape(8, 2, 1)
    artifact = trainer.train(X_by_call, Y, n_series=1)
    forecaster = CanonicalMarginalQuantileForecaster(config, artifact)
    X_predict = np.array([[1.0]])
    return forecaster.predict(
        X_predict,
        series_ids=("__local__",),
        forecast_times=pd.date_range("2026-09-01", periods=2, freq="1h"),
        feature_provider=lambda call_index, *_: (X_predict, X_predict)[call_index],
    )


class ReportRawWiringTest(unittest.TestCase):
    """report_raw 配置 → 组装段 metadata 的真实到达（断链修复验收）。"""

    def test_default_report_raw_attaches_crossing_report(self):
        distribution = _predict_distribution(
            {"mode": "quantile", "quantiles": [0.1, 0.5, 0.9], "point_quantile": 0.5}
        )
        report = distribution.metadata.get("crossing_report")
        self.assertIsNotNone(report)
        assert report is not None
        # 均值回归器输出单调，无交叉
        self.assertEqual(report["row_crossing_rate"], 0.0)
        self.assertEqual(report["repair_changed_ratio"], 0.0)

    def test_report_raw_false_omits_crossing_report(self):
        distribution = _predict_distribution(
            {
                "mode": "quantile",
                "quantiles": [0.1, 0.5, 0.9],
                "point_quantile": 0.5,
                "crossing": {"report_raw": False},
            }
        )
        self.assertNotIn("crossing_report", distribution.metadata)


class _StubScoringRunner:
    """score_holdout_fold 最小协议替身：predict 返回预置分布。"""

    def __init__(self, prediction, actual, times):
        self._prediction = prediction
        self._actual = actual
        self._times = times

    def forecast_designs(self, origin, feature_scaler, target_transform):
        return None, None

    def forecast_times(self, origin):
        return self._times

    def predict(self, artifact, designs, provider, forecast_times, target_transform):
        return self._prediction

    def actual(self, origin_index, forecast_times):
        return self._actual

    def seasonal_naive(self, origin, forecast_times, *, history=None):
        return None

    def target_history(self, origin):
        return self._actual

    def execution_evidence(self, artifact, target_transform):
        return {"stub": True}


class CrossingReportScoringEvidenceTest(unittest.TestCase):
    """crossing_report 经 score_holdout_fold 合入逐窗 execution_evidence。"""

    def _fold_result(self, metadata):
        quantiles = repair_marginal_quantile_crossing(_crossed_tensor())
        times = quantiles.forecast_times
        prediction = MarginalForecastDistribution(
            point=quantiles.point(),
            quantiles=quantiles,
            dependence_model=None,
            metadata=metadata,
        )
        actual = PointForecastTensor(
            values=np.array([[[10.0], [11.0]]]),
            series_ids=("s0",),
            forecast_times=times,
            targets=("load",),
        )
        runner = _StubScoringRunner(prediction, actual, times)
        return score_holdout_fold(
            runner=runner,
            fit_result=(None, None, None, None, object()),
            origin=times[0],
            origin_index=0,
            window=1,
            calibration_tracker=None,
            aggregate_weights=None,
            eval_mask_config=None,
        )

    def test_crossing_report_merged_into_execution_evidence(self):
        result = self._fold_result(
            {"crossing_report": {"row_crossing_rate": 1.0, "repair_changed_ratio": 0.5}}
        )
        self.assertTrue(result.execution_evidence["stub"])
        self.assertEqual(
            result.execution_evidence["crossing_report"]["row_crossing_rate"], 1.0
        )

    def test_absent_crossing_report_keeps_evidence_unchanged(self):
        result = self._fold_result({})
        self.assertNotIn("crossing_report", result.execution_evidence)


if __name__ == "__main__":
    unittest.main()
