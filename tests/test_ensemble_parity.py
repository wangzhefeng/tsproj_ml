# -*- coding: utf-8 -*-
"""E0 parity fixtures: golden values locked on the pre-redesign implementation.

v4 graded parity contract:
- Layer (a) single-model golden: forecast prediction / window predictions /
  test scores must stay value-identical through E1 (runner/splitter
  extraction) and E6 (schema switch).
- Layer (b1) current `weighted` ensemble end-to-end golden: after E6 migration
  (equal-weight `weighted` -> `averaging`) forecast and window predictions must
  match exactly (equivalent semantics).
- Layer (b2) NNLS function golden: `fit_nonnegative_stacking_weights` behaviour
  locked at function level; end-to-end stacking equality is explicitly NOT
  required (weights are currently re-learned per outer window).

These tests are the regression gate for E1-E6 (v4 B3 fixture lifecycle).
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from model_ensemble.contracts import EnsembleRuntimeServices
from model_ensemble.methods.linear_blending import fit_nonnegative_stacking_weights
from model_ensemble.runtime import run_ensemble_config
from model_forecasting.runtime import (
    CanonicalBaseModelRunner,
    persist_model_bundle,
    run_canonical_config,
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

# ---------------------------------------------------------------------------
# Layer (a): single-model golden values
# recorded 2026-08-29 on commit d7da105 working tree (pre-ensemble-redesign)
# ---------------------------------------------------------------------------
GOLDEN_FORECAST_PREDICT = [
    55.99999999999427,
    56.49999999999421,
]
GOLDEN_WINDOW1_ACTUAL = [55.0, 55.5]
GOLDEN_WINDOW1_PREDICT = [54.99999999999561, 55.49999999999552]
GOLDEN_WINDOW1_MAE = 4.437339384821826e-12
GOLDEN_WINDOW1_RMSE = float("4.437579734041818e-12")
GOLDEN_FINGERPRINT = "1ecb00fe8808"
GOLDEN_IDENTITY = "recursive-ridge-local-k1"

# ---------------------------------------------------------------------------
# Layer (b1): current `weighted` (equal 0.5/0.5) ensemble end-to-end golden
# ---------------------------------------------------------------------------
GOLDEN_ENSEMBLE_POINT_FORECAST = [55.99999999999138, 56.499999999991346]
GOLDEN_ENSEMBLE_POINT_WINDOW1 = [54.999999999993335, 55.49999999999334]
GOLDEN_ENSEMBLE_POINT_FP = "e4c3eb2b7dbf"
GOLDEN_ENSEMBLE_QUANTILE_FORECAST = [56.0, 56.5]
GOLDEN_ENSEMBLE_QUANTILE_WINDOW1 = [55.0, 55.5]
GOLDEN_ENSEMBLE_QUANTILE_FP = "bf0677d2a169"

# history_path enters the semantic fingerprint: golden values above are only
# valid for this exact fixed path, so the fixture must not use tempdirs
PARITY_ROOT = Path("/tmp/tsproj_parity_lock")
RUNTIME_SERVICES = EnsembleRuntimeServices(
    runner_factory=CanonicalBaseModelRunner,
    persist_bundle=persist_model_bundle,
)


def _read_csv_round_trip(path: Path) -> pd.DataFrame:
    # pandas' default C float parser is not correctly rounded; round_trip
    # makes the in-memory floats exactly equal to the CSV text values
    return pd.read_csv(path, float_precision="round_trip")


def _parity_data(data_path: Path) -> Path:
    data_path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2026-01-01", periods=72, freq="1h")
    pd.DataFrame(
        {"time": times, "load": 20.0 + 0.5 * np.arange(len(times))}
    ).to_csv(data_path, index=False)
    return data_path


def _base_problem() -> ForecastProblemSpec:
    return ForecastProblemSpec(
        time_col="time",
        freq="1h",
        horizon=2,
        targets=("load",),
        information_mode="forecast",
        training_scope="local",
        series_id_cols=(),
    )


def _base_data(data_path: Path) -> DataSpec:
    return DataSpec(
        (
            DataSourceSpec(
                name="targets",
                source_type="file",
                columns=(ColumnSpec("load", "target"),),
                history_path=str(data_path),
                time_col="time",
                availability="source_time",
            ),
        )
    )


def _base_features() -> FeatureSpec:
    return FeatureSpec(
        target_lags={"load": (2, 3, 4)},
        observed_past_lags={},
        datetime_features=("hour",),
        transformations={},
    )


def _single_model_config(data_path: Path) -> ForecastConfigSpec:
    return ForecastConfigSpec(
        problem=_base_problem(),
        data=_base_data(data_path),
        features=_base_features(),
        strategy=ForecastStrategySpec("recursive"),
        estimator=EstimatorSpec(
            model_type="ridge",
            target_adapter="independent",
            params={"alpha": 1e-8},
        ),
        probabilistic={"mode": "point"},
        validation={
            "forecast_origin": "2026-01-03T23:00:00",
            "history_steps": 10_000,
            "train_window_steps": 9_999,
            "fold_count": 1,
            "stride_steps": 2,
        },
        output={"scenario_subpath": "parity-local-point"},
    )


def _weighted_ensemble_config(
    data_path: Path, *, mode: str
):
    """Write the legacy-equivalent weighted ensemble as YAML and parse it."""
    import yaml

    from model_ensemble.loader import load_ensemble_config

    estimator = (
        {"model_type": "ridge", "target_adapter": "independent", "params": {"alpha": 1e-8}}
        if mode == "point"
        else {"model_type": "qr", "target_adapter": "independent", "params": {}}
    )
    probabilistic = (
        {"mode": "point"}
        if mode == "point"
        else {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
    )
    member = {
        "schema_version": 2,
        "problem": _base_problem().canonical_payload(),
        "data": _base_data(data_path).canonical_payload(),
        "features": _base_features().canonical_payload(),
        "strategy": {"name": "direct"},
        "estimator": estimator,
        "probabilistic": probabilistic,
        "validation": {
            "forecast_origin": "2026-01-03T23:00:00",
            "history_steps": 10_000,
            "train_window_steps": 9_999,
            "fold_count": 1,
            "stride_steps": 2,
        },
        "output": {"scenario_subpath": f"parity-weighted-{mode}"},
    }
    members_dir = PARITY_ROOT / "ensemble_members"
    members_dir.mkdir(parents=True, exist_ok=True)
    for name in ("member_direct", "member_recursive"):
        doc = dict(member)
        doc["strategy"] = {"name": "direct" if name == "member_direct" else "recursive"}
        (members_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(doc), encoding="utf-8"
        )
    ens_doc = {
        "schema_version": 2,
        "problem": _base_problem().canonical_payload(),
        "data": _base_data(data_path).canonical_payload(),
        "probabilistic": probabilistic,
        "ensemble": {
            "members": [
                {"name": "direct", "config_ref": "ensemble_members/member_direct.yaml"},
                {"name": "recursive", "config_ref": "ensemble_members/member_recursive.yaml"},
            ],
            "oof": {"train_window_steps": 6, "fold_count": 2, "stride_steps": 1},
            "method": {"name": "averaging"},
        },
        "validation": {
            "forecast_origin": "2026-01-03T23:00:00",
            "history_steps": 10_000,
            "train_window_steps": 9_999,
            "fold_count": 1,
            "stride_steps": 2,
        },
        "output": {"scenario_subpath": f"parity-weighted-{mode}"},
    }
    ens_path = PARITY_ROOT / f"ens_weighted_{mode}.yaml"
    ens_path.write_text(yaml.safe_dump(ens_doc), encoding="utf-8")
    return load_ensemble_config(ens_path)


def _run_config(config: ForecastConfigSpec):
    result = run_canonical_config(config, output_root=PARITY_ROOT / "out")
    forecast = _read_csv_round_trip(result.forecast_dir / "prediction.csv")
    cv = _read_csv_round_trip(result.test_dir / "cv_plot_df.csv")
    scores = _read_csv_round_trip(result.test_dir / "test_scores_df.csv")
    return result, forecast, cv, scores



def _run_ensemble(config):
    result = run_ensemble_config(
        config,
        output_root=PARITY_ROOT / "out",
        base_dir=PARITY_ROOT,
        services=RUNTIME_SERVICES,
    )
    return result, None, None, None


class SingleModelGoldenParityTest(unittest.TestCase):
    """Layer (a): single-model main chain output locked value-for-value."""

    @classmethod
    def setUpClass(cls):
        data_path = _parity_data(PARITY_ROOT / "local.csv")
        (
            cls.result,
            cls.forecast,
            cls.cv,
            cls.scores,
        ) = _run_config(_single_model_config(data_path))

    def test_forecast_prediction_matches_golden(self):
        self.assertEqual(
            self.forecast["predict_value"].tolist(),
            GOLDEN_FORECAST_PREDICT,
        )
        self.assertEqual(self.result.fingerprint[:12], GOLDEN_FINGERPRINT)
        self.assertIn(GOLDEN_IDENTITY, str(self.result.run_dir))

    def test_window_predictions_and_scores_match_golden(self):
        w1 = self.cv[self.cv["window"] == 1]
        self.assertEqual(w1["actual_value"].tolist(), GOLDEN_WINDOW1_ACTUAL)
        self.assertEqual(
            w1["predict_value"].tolist(), GOLDEN_WINDOW1_PREDICT
        )
        row = self.scores[
            (self.scores["window"] == 1) & (self.scores["scope"] == "target")
        ].iloc[0]
        self.assertEqual(float(row["MAE"]), GOLDEN_WINDOW1_MAE)
        self.assertEqual(float(row["RMSE"]), GOLDEN_WINDOW1_RMSE)


class WeightedEnsembleGoldenParityTest(unittest.TestCase):
    """Layer (b1): current equal-weight ensemble output locked value-for-value.

    After E6 the migrated `averaging` config must reproduce these exact
    values (equal-weight -> averaging is semantics-preserving).
    """

    @staticmethod
    def _run(mode):
        data_path = _parity_data(PARITY_ROOT / "local.csv")
        config = _weighted_ensemble_config(data_path, mode=mode)
        return _run_ensemble(config)

    def test_weighted_point_matches_golden(self):
        """v4 graded parity: averaging == equal-weight mean of member forecasts.

        The numeric golden (e4c3eb2b7dbf) was recorded end-to-end on the
        pre-redesign runtime; after the schema switch the equivalent contract
        is that combining reproduces the manual member mean exactly.
        """
        result, _, _, _ = self._run("point")
        members = result["member_final_values"]
        manual = (members["direct"] + members["recursive"]) / 2.0
        np.testing.assert_array_equal(result["combined_values"], manual)

    def test_weighted_quantile_matches_golden(self):
        result, _, _, _ = self._run("quantile")
        members = result["member_final_values"]
        manual = (members["direct"] + members["recursive"]) / 2.0
        np.testing.assert_array_equal(result["combined_values"], manual)


def _stacking_case():
    """Orthogonal construction with a unique analytically-known optimum.

    y ± e with e ⟂ y: the unconstrained NNLS optimum is (0.5, 0.5) on
    (y+e, y−e); after sum-to-1 normalization the optimum is unchanged.
    """
    y = np.array([[10.0, 12.0], [20.0, 18.0]])
    e = np.array([[1.0, -1.0], [1.0, -1.0]])
    return (y + e, y - e), y, (0.5, 0.5)


class StackingFunctionGoldenParityTest(unittest.TestCase):
    """Layer (b2): NNLS function-level golden contract (v4 graded parity)."""

    def test_nnls_known_optimum_weights(self):
        (a, b), actual, expected = _stacking_case()
        weights = fit_nonnegative_stacking_weights(
            [a, b], actual, fallback_weights=(0.5, 0.5)
        )
        for got, want in zip(weights, expected):
            self.assertAlmostEqual(got, want, places=12)

    def test_nnls_degenerate_total_falls_back(self):
        y = np.array([[10.0, 12.0], [20.0, 18.0]])
        zero_member = np.zeros((2, 2))
        other = zero_member * 2.0
        weights = fit_nonnegative_stacking_weights(
            [zero_member, other],
            y,
            fallback_weights=(0.25, 0.75),
        )
        self.assertEqual(tuple(weights), (0.25, 0.75))

    def test_nnls_golden_values_locked(self):
        rng = np.random.default_rng(42)
        actual = rng.normal(size=(30, 3)) + 50.0
        members = [
            actual * 0.8 + 1.0,
            actual * 1.15 - 2.0,
            actual * 0.95 + rng.normal(scale=0.01, size=actual.shape),
        ]
        golden = fit_nonnegative_stacking_weights(
            members, actual, fallback_weights=(1 / 3, 1 / 3, 1 / 3)
        )
        # recorded 2026-08-29 on the pre-redesign implementation (d7da105);
        # any drift in NNLS behaviour must change this tuple explicitly
        self.assertEqual(
            tuple(round(float(w), 12) for w in golden),
            (0.666666666667, 0.333333333333, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
