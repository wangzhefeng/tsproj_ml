"""联通算力：实例不去重、零语义、异常审计与目标拼接。"""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import json
import csv
import io

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"


class ComputilityTest(unittest.TestCase):
    def test_malformed_samples_raise(self):
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            module = importlib.import_module("liantong_computility_process")
        finally:
            sys.path.pop(0)
        grid = pd.date_range("2026-08-01", periods=2, freq="5min", name="time")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metric.csv"
            for raw in ('bad', '[1785513600,"NaN"]', '[1785513600,"inf"]', '[1785513600,"-1"]', '[1785513601,"1"]', '[1785513600.5,"1"]', '[true,"1"]', '[1785513600,null]', '[1785513600,"1",2]'):
                with self.subTest(raw=raw):
                    pd.DataFrame([{"uid": "001", "metric": "lepton__aec2__acn__job__cpu_util", "value": raw}]).to_csv(path, index=False)
                    with self.assertRaises(ValueError):
                        module.aggregate_metric(path, "training", "cpu_util", grid)

    def test_empty_and_duplicate_job(self):
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            module = importlib.import_module("liantong_computility_process")
        finally:
            sys.path.pop(0)
        grid = pd.date_range("2026-08-01", periods=2, freq="5min", name="time")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metric.csv"
            rows = pd.DataFrame([{"uid": "001", "metric": "lepton__aec2__acn__job__cpu_util", "value": "[]"}])
            rows.to_csv(path, index=False)
            frame, quality, audit = module.aggregate_metric(path, "training", "cpu_util", grid)
            self.assertEqual(frame.training_cpu_util_sample_mean.tolist(), [0, 0])
            self.assertEqual(quality.training_cpu_util_no_record.tolist(), [1, 1])
            self.assertEqual(audit["samples"], 0)
            pd.concat([rows, rows]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                module.aggregate_metric(path, "training", "cpu_util", grid)

    def test_cli_complete_outputs_and_protection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for source in ("training", "inference"):
                (root / source).mkdir()
                for metric in ("cpu_util", "gpu_util", "memory_amount", "memory_total", "memory_util", "gpu_memory_amount", "gpu_memory_total", "gpu_memory_util", "gpu_power_usage"):
                    value = "1.2" if metric.endswith("_util") else "100"
                    pd.DataFrame([{"uid": "001", "metric": f"lepton__aec2__acn__job__{metric}", "value": f'[1785513600,"{value}"]'}]).to_csv(root / source / f"lepton__aec2__acn__job__{metric}_merged_20260801_20260831.csv", index=False)
            target = root / "target.csv"
            pd.DataFrame({"time": ["2026-08-01 00:00:00", "2026-08-01 00:05:00"], "value": [123.5, 456.5]}).to_csv(target, index=False)
            command = [sys.executable, str(SCRIPT_DIR / "liantong_computility_process.py"), "--source-dir", str(root), "--target-csv", str(target), "--output-dir", str(root / "out"), "--start", "2026-08-01", "--end", "2026-08-01 00:05:00"]
            result = subprocess.run(command, capture_output=True, text=True, cwd=temp)
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads((root / "out/aidc_comp_liantong_5min/computility_processing_audit.json").read_text())
            self.assertEqual(len(audit["inputs"]), 18)
            self.assertEqual(sum(len(item["repairs"]) for item in audit["inputs"]), 8)
            self.assertEqual(len(audit["outputs"]), 7)
            for source in ("training", "inference"):
                samples = pd.read_csv(root / "out/aidc_comp_liantong_5min" / f"{source}_job.csv", dtype={"job_id": str})
                self.assertEqual(len(samples), 9)
                self.assertEqual(samples.job_id.unique().tolist(), ["001"])
                self.assertEqual(samples.time.unique().tolist(), ["2026-08-01 00:00:00"])
                self.assertEqual(int(samples.is_corrected.sum()), 4)
                self.assertNotIn("instance_id", samples.columns)
                power = samples.loc[samples.metric.eq("gpu_power_usage")].iloc[0]
                self.assertEqual(power.raw_value, 100)
                self.assertEqual(power.processed_value, 100)  # 长表保留 W，不跟聚合列转 kW
                util = samples.loc[samples.metric.eq("cpu_util")].iloc[0]
                self.assertEqual(util.raw_value, 1.2)
                self.assertEqual(util.processed_value, 1.0)
                self.assertEqual(util.source_row, 2)
                self.assertEqual(util.sample_position, 0)
                self.assertEqual(util.epoch_seconds, 1785513600)
            merged = pd.read_csv(root / "out/power_computility_5min_20260801_20260801.csv")
            self.assertEqual(merged.value.tolist(), [123.5, 456.5])
            self.assertEqual(merged.training_cpu_util_sample_mean.tolist(), [1.0, 0.0])
            self.assertFalse(merged.isna().any().any())
            def snapshot():
                return {str(p.relative_to(root / "out")): p.read_bytes() for p in (root / "out").rglob("*") if p.is_file()}
            before = snapshot()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run(command + ["--overwrite"], capture_output=True).returncode, 0)
            self.assertEqual(before, snapshot())
            target.write_text("time,value\n2026-08-01,NaN\n")
            self.assertNotEqual(subprocess.run(command + ["--overwrite"], capture_output=True).returncode, 0)
            self.assertEqual(before, snapshot())
            # 最后一个指标失败时，已写入暂存区的长表不能污染正式结果。
            pd.DataFrame({"time": ["2026-08-01 00:00:00", "2026-08-01 00:05:00"], "value": [123.5, 456.5]}).to_csv(target, index=False)
            broken = root / "inference/lepton__aec2__acn__job__gpu_power_usage_merged_20260801_20260831.csv"
            broken.write_text("uid,metric,value\n001,lepton__aec2__acn__job__gpu_power_usage,bad\n")
            self.assertNotEqual(subprocess.run(command + ["--overwrite"], capture_output=True).returncode, 0)
            self.assertEqual(before, snapshot())

    def test_instance_sum_and_zero_grid(self):
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            module = importlib.import_module("liantong_computility_process")
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metric.csv"
            pd.DataFrame([{"uid": "001", "metric": "lepton__aec2__acn__job__gpu_power_usage", "value": '[1785513600,"100"]|[1785513600,"100"]|[1785513600,"200"]'}]).to_csv(path, index=False)
            grid = pd.date_range("2026-08-01", periods=2, freq="5min", name="time")
            stream = io.StringIO()
            frame, quality, audit = module.aggregate_metric(path, "training", "gpu_power_usage", grid, sample_writer=csv.writer(stream))
            stream.seek(0)
            samples = list(csv.reader(stream))
            self.assertEqual(len(samples), 3)
            self.assertEqual([r[3] for r in samples], ["100", "100", "200"])
            self.assertEqual([r[7] for r in samples], ["0", "1", "2"])
            self.assertEqual(frame.training_gpu_power_usage_sum_kw.tolist(), [0.4, 0.0])
            self.assertEqual(frame.training_gpu_power_usage_job_count.tolist(), [1, 0])
            self.assertEqual(quality.training_gpu_power_usage_sample_count.tolist(), [3, 0])
            self.assertEqual(audit["samples"], 3)

    def test_numeric_parsing_is_not_an_outlier_correction(self):
        sys.path.insert(0, str(SCRIPT_DIR))
        try:
            module = importlib.import_module("liantong_computility_process")
        finally:
            sys.path.pop(0)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metric.csv"
            pd.DataFrame([{"uid": "001", "metric": "lepton__aec2__acn__job__cpu_util", "value": '[1785513600,"0.12345678901234567"]'}]).to_csv(path, index=False)
            stream = io.StringIO()
            module.aggregate_metric(path, "training", "cpu_util", pd.date_range("2026-08-01", periods=1, freq="5min", name="time"), sample_writer=csv.writer(stream))
            stream.seek(0)
            row = next(csv.reader(stream))
            self.assertEqual(row[3], "0.12345678901234567")
            self.assertEqual(row[8], "0")


if __name__ == "__main__":
    unittest.main()
