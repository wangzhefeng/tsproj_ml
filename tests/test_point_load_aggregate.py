# -*- coding: utf-8 -*-

import gc
import json
import re
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from config.aidc_load_5min.point_load_aggregate import _build_parser, run_point_load_aggregation


class PointLoadAggregateTest(unittest.TestCase):
    START = "2025-10-01 00:00:00"
    END = "2025-10-01 00:10:00"

    def test_cli_defaults_follow_current_dataset_layout(self):
        args = _build_parser().parse_args([])

        self.assertEqual(
            args.reference_path,
            "dataset/aidc_load_5min/A1_A2_A3_points/all_ids.xlsx",
        )
        self.assertEqual(args.points_root, "dataset/aidc_load_5min/A1_A2_A3_points")
        self.assertEqual(args.output_dir, "dataset/aidc_load_5min/A1_A2_A3_data")

    @staticmethod
    def _write_point(path: Path, values, *, duplicate_first: bool = False) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        times = [
            "2025-09-30 23:55:00",
            "2025-10-01 00:00:00",
            "2025-10-01 00:05:00",
            "2025-10-01 00:10:00",
            "2025-10-01 00:15:00",
        ]
        rows = pd.DataFrame({"time": times, "value": [999.0, *values, 999.0]})
        if duplicate_first:
            rows = pd.concat(
                [rows.iloc[:2], pd.DataFrame({"time": [times[1]], "value": [values[0] + 1]}), rows.iloc[2:]],
                ignore_index=True,
            )
        rows.to_csv(path, index=False)

    def _write_fixture(self, root: Path) -> tuple[Path, Path, Path]:
        points_root = root / "A1_A2_A3_points"
        output_dir = root / "aidc_load_5min"
        reference_path = points_root / "all_ids.xlsx"
        points_root.mkdir(parents=True)
        output_dir.mkdir(parents=True)

        hvac = pd.DataFrame(
            [
                {"data_type": "A1楼暖通电力负荷", "deviceId": "hvac-a1-a", "spot_id": "h1a", "spotName": "输出功率", "备注": "冷水机组", "route": "A"},
                {"data_type": "A1楼暖通电力负荷", "deviceId": "hvac-a1-b", "spot_id": "h1b", "spotName": "输出功率", "备注": "冷却塔", "route": "B"},
                {"data_type": "A1楼暖通电力负荷", "deviceId": "hvac-air", "spot_id": "h_air", "spotName": "机组总功率", "备注": "空调", "route": np.nan},
                {"data_type": "A2楼暖通电力负荷", "deviceId": "hvac-a2-a", "spot_id": "h2a", "spotName": "输出功率", "备注": "冷冻水一次泵", "route": "A"},
                {"data_type": "A2楼暖通电力负荷", "deviceId": "hvac-a2-b", "spot_id": "h2b", "spotName": "输出功率", "备注": "冷冻水二次泵", "route": "B"},
                {"data_type": "A3楼暖通电力负荷", "deviceId": "hvac-a3-a", "spot_id": "h3a", "spotName": "电机功率", "备注": "冷水机组", "route": "A"},
                {"data_type": "A3楼暖通电力负荷", "deviceId": "hvac-a3-b", "spot_id": "h3b_missing", "spotName": "电机功率", "备注": "冷水机组", "route": "B"},
            ]
        )
        ups = pd.DataFrame(
            [
                {"data_type": "A1楼UPS负荷", "deviceId": "ups-a1-a", "spot_id": "u1a_phase_a", "spotName": "A相输出有功功率", "备注": "二楼", "route": "A"},
                {"data_type": "A1楼UPS负荷", "deviceId": "ups-a1-a", "spot_id": "u1a_phase_b", "spotName": "B相输出有功功率", "备注": "二楼", "route": "A"},
                {"data_type": "A1楼UPS负荷", "deviceId": "ups-a1-a", "spot_id": "u1a_phase_c", "spotName": "C相输出有功功率", "备注": "二楼", "route": "A"},
                {"data_type": "A1楼UPS负荷", "deviceId": "ups-a1-a", "spot_id": "u1a_total", "spotName": "总输出有功功率", "unit_spot": "W", "备注": "二楼", "route": "A"},
                {"data_type": "A1楼UPS负荷", "deviceId": "ups-a1-b", "spot_id": "u1b", "spotName": "输入功率", "备注": "二楼", "route": "B"},
                {"data_type": "A1楼UPS负荷", "deviceId": "ups-excluded", "spot_id": "u_excluded", "spotName": "总输出有功功率", "备注": "一楼", "route": "B"},
                {"data_type": "A2楼UPS负荷", "deviceId": "ups-a2-a", "spot_id": "u2a", "spotName": "总有功功率", "备注": "三楼", "route": "A"},
                {"data_type": "A2楼UPS负荷", "deviceId": "ups-a2-b", "spot_id": "u2b", "spotName": "总有功功率", "备注": "三楼", "route": "B"},
                {"data_type": "A3楼UPS负荷", "deviceId": "ups-a3-a", "spot_id": "u3a", "spotName": "总有功功率", "备注": "四楼", "route": "A"},
                {"data_type": "A3楼UPS负荷", "deviceId": "ups-a3-b", "spot_id": "u3b_missing", "spotName": "总有功功率", "备注": "四楼", "route": "B"},
            ]
        )
        cabinet = pd.DataFrame(
            [
                {"data_type": f"{building}楼列头柜负荷", "deviceId": f"cab-{building}-{route}", "spot_id": f"c{building[-1]}{route.lower()}", "spotName": "进线_总有功功率", "备注": "三楼", "route": route}
                for building in ("A1", "A2", "A3")
                for route in ("A", "B")
            ]
        )
        with pd.ExcelWriter(reference_path, engine="openpyxl") as writer:
            hvac.to_excel(writer, sheet_name="暖通负荷", index=False)
            ups.to_excel(writer, sheet_name="UPS负荷", index=False)
            cabinet.to_excel(writer, sheet_name="列头柜负荷", index=False)

        sources = {
            ("A1楼暖通电力负荷", "h1a"): ([1.0, np.nan, 3.0], True),
            ("A1楼暖通电力负荷", "h1b"): ([4.0, 5.0, 6.0], False),
            ("A2楼暖通电力负荷", "h2a"): ([10.0, 20.0, 30.0], False),
            ("A2楼暖通电力负荷", "h2b"): ([40.0, 50.0, 60.0], False),
            ("A3楼暖通电力负荷", "h3a"): ([100.0, 200.0, 300.0], False),
            ("A1楼UPS负荷", "u1a_phase_a"): ([1.0, 2.0, 3.0], False),
            ("A1楼UPS负荷", "u1a_phase_b"): ([4.0, 5.0, 6.0], False),
            ("A1楼UPS负荷", "u1a_phase_c"): ([7.0, 8.0, 9.0], False),
            ("A1楼UPS负荷", "u1a_total"): ([10000.0, 20000.0, 30000.0], False),
            ("A1楼UPS负荷", "u1b"): ([5.0, 6.0, 7.0], False),
            ("A1楼UPS负荷", "u_excluded"): ([1000.0, 1000.0, 1000.0], False),
            ("A2楼UPS负荷", "u2a"): ([10.0, np.nan, 30.0], False),
            ("A2楼UPS负荷", "u2b"): ([20.0, 30.0, 40.0], False),
            ("A3楼UPS负荷", "u3a"): ([100.0, 200.0, 300.0], False),
            ("A1楼列头柜负荷", "c1a"): ([1.0, 2.0, 3.0], False),
            ("A1楼列头柜负荷", "c1b"): ([4.0, 5.0, 6.0], False),
            ("A2楼列头柜负荷", "c2a"): ([10.0, 20.0, 30.0], False),
            ("A2楼列头柜负荷", "c2b"): ([40.0, 50.0, 60.0], False),
            ("A3楼列头柜负荷", "c3a"): ([100.0, 200.0, 300.0], False),
            ("A3楼列头柜负荷", "c3b"): ([400.0, 500.0, 600.0], False),
        }
        for (data_type, spot_id), (values, duplicate_first) in sources.items():
            self._write_point(points_root / data_type / f"{spot_id}.csv", values, duplicate_first=duplicate_first)
        return reference_path, points_root, output_dir

    def test_run_generates_route_outputs_prefers_total_power_and_removes_legacy_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path, points_root, output_dir = self._write_fixture(Path(tmpdir))
            legacy_paths = [
                output_dir / "A1楼暖通电力总负荷.csv",
                output_dir / "A1+A2+A3 UPS总负荷.csv",
                output_dir / "route_A" / "A1楼A路UPS总负荷.csv",
            ]
            for path in legacy_paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("legacy", encoding="utf-8")

            audit = run_point_load_aggregation(
                reference_path=reference_path,
                points_root=points_root,
                output_dir=output_dir,
                start_time=self.START,
                end_time=self.END,
                freq="5min",
            )

            self.assertEqual(len(audit["outputs"]), 24)
            output_paths = sorted(output_dir.glob("route_*/*.csv"))
            self.assertEqual(len(output_paths), 24)
            self.assertTrue(all(re.fullmatch(r"[a-z0-9_]+\.csv", path.name) for path in output_paths))
            self.assertEqual({path.parent.name for path in output_paths}, {"route_A", "route_B"})
            self.assertFalse(any(path.exists() for path in legacy_paths))
            self.assertEqual(set(audit["routes"]), {"A", "B"})
            self.assertEqual(len(audit["removed_legacy_outputs"]), 3)
            self.assertFalse(any("暖通电力总负荷" in item["name"] and "主要设备" not in item["name"] for item in audit["outputs"]))

            combined_a = pd.read_csv(
                output_dir / "route_A" / "a1_a2_a3_route_a_hvac_main_load_5min_20251001_20251001.csv"
            )
            self.assertEqual(list(combined_a.columns), ["time", "h1a", "h2a", "h3a", "value"])
            self.assertEqual(combined_a["value"].tolist(), [112.0, 220.0, 333.0])

            a1_a_ups = pd.read_csv(
                output_dir / "route_A" / "a1_route_a_ups_load_5min_20251001_20251001.csv"
            )
            self.assertEqual(list(a1_a_ups.columns), ["time", "u1a_total", "value"])
            self.assertEqual(a1_a_ups["value"].tolist(), [10.0, 20.0, 30.0])
            self.assertNotIn(
                "u_excluded",
                pd.read_csv(
                    output_dir / "route_B" / "a1_route_b_ups_load_5min_20251001_20251001.csv"
                ).columns,
            )

            a1_a_ups_audit = next(item for item in audit["outputs"] if item["name"] == "A1楼A路UPS总负荷")
            self.assertEqual(a1_a_ups_audit["route"], "A")
            self.assertEqual(a1_a_ups_audit["reference_point_count"], 4)
            self.assertEqual(a1_a_ups_audit["selected_point_count"], 1)
            self.assertEqual(a1_a_ups_audit["excluded_phase_point_count"], 3)
            self.assertEqual(a1_a_ups_audit["unit_converted_point_count"], 1)
            self.assertEqual(a1_a_ups_audit["points"][0]["source_unit"], "W")
            self.assertEqual(a1_a_ups_audit["points"][0]["unit_scale_to_kw"], 0.001)
            self.assertEqual(
                {item["spot_id"] for item in a1_a_ups_audit["excluded_phase_points"]},
                {"u1a_phase_a", "u1a_phase_b", "u1a_phase_c"},
            )

            a3_b_hvac_audit = next(
                item for item in audit["outputs"] if item["name"] == "A3楼B路暖通电力总负荷-主要设备"
            )
            self.assertEqual(a3_b_hvac_audit["included_point_count"], 0)
            self.assertEqual(a3_b_hvac_audit["missing_file_count"], 1)
            self.assertEqual(a3_b_hvac_audit["missing_files"][0]["spot_id"], "h3b_missing")
            self.assertEqual(
                Path(a3_b_hvac_audit["output_path"]),
                output_dir / "route_B" / "a3_route_b_hvac_main_load_5min_20251001_20251001.csv",
            )

            audit_path = output_dir / "A1_A2_A3_points_aggregate_audit.json"
            self.assertEqual(json.loads(audit_path.read_text(encoding="utf-8")), audit)

    def test_run_closes_reference_workbook(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            reference_path, points_root, output_dir = self._write_fixture(Path(tmpdir))

            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ResourceWarning)
                run_point_load_aggregation(
                    reference_path=reference_path,
                    points_root=points_root,
                    output_dir=output_dir,
                    start_time=self.START,
                    end_time=self.END,
                    freq="5min",
                )
                gc.collect()

            resource_warnings = [item for item in captured if issubclass(item.category, ResourceWarning)]
            self.assertEqual(resource_warnings, [])


if __name__ == "__main__":
    unittest.main()
