import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import SourceRegistry
from forecasting_core.specs.config import parse_model_config
from model_pipeline.production import ProductionForecastRequest, resolve_production_origin
from model_pipeline.runner import run_canonical_config
from tests.test_calendar_month_origin_day import _payload


class ProductionDynamicOriginTest(unittest.TestCase):
    def test_request_validates_dates_and_derives_target_month(self):
        request = ProductionForecastRequest("2026-12-25")
        self.assertEqual(request.target_month, "2027-01")
        self.assertEqual(request.run_scope, "target_month=2027-01")
        for invalid in ("2026-13-01", "2026-02-30", "20260725", 20260725):
            with self.assertRaises(ValueError):
                ProductionForecastRequest(invalid)

    def test_origin_uses_complete_target_grid_and_content_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            times = pd.date_range("2025-10-01", "2026-07-31", freq="1D")
            pd.DataFrame({"time": times, "load": np.arange(len(times)) + 1.0}).to_csv(path, index=False)
            config = parse_model_config(_payload(str(path)), source="<production>")
            identity = config.result_identity()
            registry = SourceRegistry(config.data, Path.cwd())
            request = ProductionForecastRequest("2026-07-25")
            origin, evidence = resolve_production_origin(config, registry, request)
            self.assertEqual(origin, pd.Timestamp("2026-07-31"))
            self.assertEqual(config.result_identity(), identity)
            self.assertEqual(evidence["resolved_origin"], "2026-07-31T00:00:00")
            self.assertEqual(evidence["target_month"], "2026-08")
            self.assertEqual(len(evidence["sources"][0]["sha256"]), 64)
            self.assertTrue(evidence["sources"][0]["series"][0]["grid_complete"])

            pd.DataFrame({"time": times.delete(20), "load": np.arange(len(times) - 1) + 1.0}).to_csv(path, index=False)
            registry = SourceRegistry(config.data, Path.cwd())
            with self.assertRaisesRegex(ValueError, "incomplete time-grid"):
                resolve_production_origin(config, registry, request)

    def test_multiple_series_align_to_common_latest_complete_day(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            times_a = pd.date_range("2026-01-01", "2026-07-31", freq="1D")
            times_b = pd.date_range("2026-01-01", "2026-07-30", freq="1D")
            frame = pd.concat([
                pd.DataFrame({"time": times_a, "load": 1.0, "site": "A"}),
                pd.DataFrame({"time": times_b, "load": 2.0, "site": "B"}),
            ])
            frame.to_csv(path, index=False)
            payload = _payload(str(path))
            payload["problem"]["series_id_cols"] = ["site"]
            payload["data"]["sources"][0]["series_id_cols"] = ["site"]
            payload["data"]["sources"][0]["columns"].append(
                {"name": "site", "role": "key", "categorical": True}
            )
            config = parse_model_config(payload, source="<production-panel>")
            registry = SourceRegistry(config.data, Path.cwd())
            origin, evidence = resolve_production_origin(
                config, registry, ProductionForecastRequest("2026-07-25"),
            )
            self.assertEqual(origin, pd.Timestamp("2026-07-30"))
            self.assertEqual(len(evidence["sources"][0]["series"]), 2)

    def test_origin_inside_requested_month_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "daily.csv"
            times = pd.date_range("2025-10-01", "2026-08-02", freq="1D")
            pd.DataFrame({"time": times, "load": np.arange(len(times)) + 1.0}).to_csv(path, index=False)
            config = parse_model_config(_payload(str(path)), source="<production-late>")
            registry = SourceRegistry(config.data, Path.cwd())
            with self.assertRaisesRegex(ValueError, "entered the requested target month"):
                resolve_production_origin(
                    config, registry, ProductionForecastRequest("2026-07-25"),
                )

    def test_production_runtime_slices_and_scopes_target_month(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "daily.csv"
            times = pd.date_range("2025-10-01", "2026-07-31", freq="1D")
            pd.DataFrame({"time": times, "load": np.arange(len(times)) + 100.0}).to_csv(path, index=False)
            payload = _payload(str(path), origin_day=25)
            payload["strategy"] = {"name": "direct"}
            payload["features"]["target_lags"]["load"] = [37, 42]
            config = parse_model_config(payload, source="<production-runtime>")
            request = ProductionForecastRequest("2026-07-25")
            result = run_canonical_config(
                config, output_root=root / "results", production_request=request,
            )
            prediction = pd.read_csv(result.forecast_dir / "prediction.csv")
            times = pd.to_datetime(prediction["time"])
            self.assertEqual(len(times), 31)
            self.assertEqual(times.min(), pd.Timestamp("2026-08-01"))
            self.assertEqual(times.max(), pd.Timestamp("2026-08-31"))
            self.assertEqual(result.forecast_dir.parent.name, request.run_scope)
            resolved = json.loads((result.forecast_dir / "resolved_config.json").read_text())
            production = resolved["runtime"]["production"]
            self.assertEqual(production["issue_date"], "2026-07-25")
            self.assertEqual(production["resolved_origin"], "2026-07-31T00:00:00")
            self.assertEqual(resolved["runtime"]["forecast_target_month"], "2026-08")


if __name__ == "__main__":
    unittest.main()
