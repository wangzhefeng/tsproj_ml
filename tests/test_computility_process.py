# -*- coding: utf-8 -*-

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# AIDC 专项脚本目录（路径含日期段，不是合法 Python 包，按脚本自身约定做 sys.path 引导）
_AIDC_SCRIPTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "config" / "aidc_electricity_computility" / "electricity" / "2026-06-11" / "scripts"
)
if str(_AIDC_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AIDC_SCRIPTS_DIR))

from computility_process import (  # noqa: E402
    METRIC_PREFIXES,
    ROOM_MAPPING,
    build_feature_stats,
    build_computility_dataset,
    filter_feature_candidates,
    parse_metric_points,
    process_all_rooms,
    select_features_by_score,
)


class ComputilityProcessTest(unittest.TestCase):
    def test_parse_metric_points_supports_job_and_pod_shapes(self):
        job_points = parse_metric_points('[1778601600,"1.5"]|[1778601900,"2.5"]')
        pod_points = parse_metric_points('[[[1778601600,"0.2"], [1778601900,"0.4"]]]')

        self.assertEqual(job_points, [(1778601600, 1.5), (1778601900, 2.5)])
        self.assertEqual(pod_points, [(1778601600, 0.2), (1778601900, 0.4)])

    def test_build_computility_dataset_outputs_expected_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            room_dir = Path(tmpdir)
            self._write_metric_tree(room_dir)

            result = build_computility_dataset(room_dir)

            self.assertEqual(
                result["count_data_time"].tolist(),
                ["2026-05-13 00:00:00", "2026-05-13 00:05:00"],
            )
            self.assertIn("training_gpu_power_usage_sum", result.columns)
            self.assertIn("inference_gpu_power_usage_sum", result.columns)
            self.assertIn("pod_gpu_power_usage_sum", result.columns)
            self.assertIn("training_cpu_util_count", result.columns)
            self.assertIn("jobs_total_gpu_power_usage_sum", result.columns)
            self.assertIn("jobs_total_gpu_util_mean", result.columns)
            self.assertIn("training_share_gpu_power_usage", result.columns)
            self.assertIn("inference_share_gpu_power_usage", result.columns)
            self.assertIn("pod_minus_jobs_gpu_power_usage_sum", result.columns)
            self.assertIn("training_present", result.columns)
            self.assertIn("inference_present", result.columns)
            self.assertIn("pod_present", result.columns)
            self.assertIn("training_active_jobs", result.columns)
            self.assertIn("inference_active_jobs", result.columns)
            self.assertIn("pod_active_count", result.columns)
            self.assertIn("training_gpu_busy_ratio", result.columns)
            self.assertIn("inference_gpu_busy_ratio", result.columns)
            self.assertIn("pod_gpu_busy_ratio", result.columns)
            self.assertIn("training_high_power_job_ratio", result.columns)
            self.assertIn("training_to_inference_power_ratio", result.columns)
            self.assertIn("training_minus_inference_power_sum", result.columns)
            self.assertIn("pod_overhead_ratio", result.columns)
            self.assertIn("pod_gpu_util_minus_jobs_gpu_util_mean", result.columns)
            self.assertIn("training_gpu_power_usage_sum_diff_1", result.columns)
            self.assertIn("jobs_total_gpu_power_usage_sum_diff_1", result.columns)
            self.assertIn("training_gpu_power_usage_sum_roll_mean_12", result.columns)
            self.assertNotIn("job1_gpu_power_usage_sum", result.columns)
            self.assertNotIn("job2_gpu_power_usage_sum", result.columns)

            row0 = result.iloc[0]
            self.assertAlmostEqual(row0["training_gpu_power_usage_sum"], 10.0)
            self.assertAlmostEqual(row0["inference_gpu_power_usage_sum"], 5.0)
            self.assertAlmostEqual(row0["pod_gpu_power_usage_sum"], 16.0)
            self.assertAlmostEqual(row0["jobs_total_gpu_power_usage_sum"], 15.0)
            self.assertAlmostEqual(row0["training_share_gpu_power_usage"], 10.0 / 15.0)
            self.assertAlmostEqual(row0["inference_share_gpu_power_usage"], 5.0 / 15.0)
            self.assertAlmostEqual(row0["pod_minus_jobs_gpu_power_usage_sum"], 1.0)
            self.assertEqual(int(row0["training_present"]), 1)
            self.assertEqual(int(row0["inference_present"]), 1)
            self.assertEqual(int(row0["pod_present"]), 1)
            self.assertEqual(int(row0["training_active_jobs"]), 1)
            self.assertEqual(int(row0["inference_active_jobs"]), 1)
            self.assertEqual(int(row0["pod_active_count"]), 1)
            self.assertAlmostEqual(row0["training_gpu_busy_ratio"], 1.0)
            self.assertAlmostEqual(row0["inference_gpu_busy_ratio"], 1.0)
            self.assertAlmostEqual(row0["pod_gpu_busy_ratio"], 1.0)
            self.assertAlmostEqual(row0["training_high_power_job_ratio"], 0.0)
            self.assertAlmostEqual(row0["training_to_inference_power_ratio"], 2.0)
            self.assertAlmostEqual(row0["training_minus_inference_power_sum"], 5.0)
            self.assertAlmostEqual(row0["pod_overhead_ratio"], 16.0 / 15.0)
            self.assertAlmostEqual(row0["pod_gpu_util_minus_jobs_gpu_util_mean"], 0.95 - 0.6)
            self.assertTrue(pd.isna(row0["training_gpu_power_usage_sum_diff_1"]))
            self.assertTrue(pd.isna(row0["jobs_total_gpu_power_usage_sum_diff_1"]))
            self.assertAlmostEqual(row0["training_gpu_power_usage_sum_roll_mean_12"], 10.0)

            row1 = result.iloc[1]
            self.assertAlmostEqual(row1["training_gpu_power_usage_sum_diff_1"], 10.0)
            self.assertAlmostEqual(row1["jobs_total_gpu_power_usage_sum_diff_1"], 12.0)
            self.assertAlmostEqual(row1["training_gpu_power_usage_sum_roll_mean_12"], 15.0)

    def test_process_all_rooms_writes_df_computility_and_overlap_df(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            computility_root = root / "算力数据"
            demand_root = root / "2026-06-11" / "demand_load"
            room_src = computility_root / "A1-201"
            room_dst = demand_root / ROOM_MAPPING["A1-201"]
            room_src.mkdir(parents=True)
            room_dst.mkdir(parents=True)
            self._write_metric_tree(room_src)
            pd.DataFrame(
                {
                    "count_data_time": [
                        "2026-05-11 23:55:00",
                        "2026-05-13 00:00:00",
                        "2026-05-13 00:05:00",
                        "2026-05-13 00:10:00",
                    ],
                    "h_total_use": [90.0, 100.0, 110.0, 120.0],
                }
            ).to_csv(room_dst / "df_power.csv", index=False)

            results = process_all_rooms(computility_root, demand_root)

            self.assertEqual([item["room"] for item in results], ["A1-201"])
            df_computility_path = room_dst / "df_computility.csv"
            df_path = room_dst / "df.csv"
            self.assertTrue(df_computility_path.exists())
            self.assertTrue(df_path.exists())
            self.assertTrue((room_dst / "df_selected.csv").exists())
            self.assertTrue((room_dst / "feature_selection_report.csv").exists())

            df_computility = pd.read_csv(df_computility_path)
            merged = pd.read_csv(df_path)
            selected = pd.read_csv(room_dst / "df_selected.csv")
            report = pd.read_csv(room_dst / "feature_selection_report.csv")
            self.assertEqual(df_computility["count_data_time"].tolist(), ["2026-05-13 00:00:00", "2026-05-13 00:05:00"])
            self.assertEqual(merged["count_data_time"].tolist(), ["2026-05-13 00:00:00", "2026-05-13 00:05:00"])
            self.assertEqual(list(merged.columns[:2]), ["count_data_time", "h_total_use"])
            self.assertEqual(list(selected.columns[:2]), ["count_data_time", "h_total_use"])
            self.assertIn("training_gpu_power_usage_sum", merged.columns)
            self.assertIn("feature_name", report.columns)
            self.assertIn("level_pearson", report.columns)
            self.assertIn("vol_spearman", report.columns)
            self.assertFalse((demand_root / "AIDC" / "route_A" / "df.csv").exists())

    def test_feature_selection_filters_noise_and_scores_candidates(self):
        df = pd.DataFrame(
            {
                "count_data_time": pd.date_range("2026-06-01 00:00:00", periods=20, freq="5min").astype(str),
                "h_total_use": list(range(20)),
                "training_gpu_power_usage_sum": list(range(20)),
                "training_gpu_power_usage_mean": list(range(20)),
                "training_gpu_power_usage_min": list(range(20)),
                "training_gpu_util_busy_count": list(range(20)),
                "training_gpu_power_usage_sum_diff_3": list(range(20)),
                "training_gpu_power_usage_sum_roll_std_12": list(range(20)),
                "training_gpu_power_usage_std": list(range(20)),
                "training_gpu_busy_ratio": [v / 20 for v in range(20)],
                "training_present": [1] * 20,
                "inference_gpu_util_mean": [None] * 7 + list(range(13)),
                "training_active_jobs_diff_1": [0] + [1] * 19,
            }
        )

        stats = build_feature_stats(df, "h_total_use")
        filtered = filter_feature_candidates(stats)
        selected = select_features_by_score(filtered)

        drop_reason = dict(zip(filtered["feature_name"], filtered["drop_reason"]))
        self.assertEqual(drop_reason["training_gpu_power_usage_min"], "dropped_by_family_rule")
        self.assertEqual(drop_reason["training_gpu_util_busy_count"], "dropped_by_family_rule")
        self.assertEqual(drop_reason["training_gpu_power_usage_sum_diff_3"], "dropped_by_family_rule")
        self.assertEqual(drop_reason["training_gpu_power_usage_sum_roll_std_12"], "dropped_by_family_rule")
        self.assertEqual(drop_reason["training_present"], "dropped_by_constant")
        self.assertEqual(drop_reason["inference_gpu_util_mean"], "dropped_by_missing_ratio")

        kept = filtered.set_index("feature_name")
        self.assertGreaterEqual(kept.loc["training_gpu_power_usage_sum", "level_pearson"], 0.99)
        self.assertGreaterEqual(kept.loc["training_gpu_power_usage_sum", "level_spearman"], 0.99)
        self.assertGreaterEqual(kept.loc["training_gpu_power_usage_sum", "final_score"], 0.08)
        self.assertGreaterEqual(kept.loc["training_active_jobs_diff_1", "vol_pearson"], 0.0)
        self.assertIn("training_gpu_power_usage_sum", selected)
        self.assertIn("training_gpu_power_usage_std", selected)

    def test_feature_selection_backfills_to_minimum_and_caps_to_maximum(self):
        rows = 40
        df = {
            "count_data_time": pd.date_range("2026-06-01 00:00:00", periods=rows, freq="5min").astype(str),
            "h_total_use": list(range(rows)),
        }
        for idx in range(70):
            df[f"training_gpu_power_usage_{idx:02d}_mean"] = [row * (idx + 1) for row in range(rows)]

        stats = build_feature_stats(pd.DataFrame(df), "h_total_use")
        filtered = filter_feature_candidates(stats)
        selected = select_features_by_score(filtered)

        self.assertEqual(len(selected), 64)

    def _write_metric_tree(self, room_dir: Path) -> None:
        for group in ["training", "inference", "pod"]:
            (room_dir / group).mkdir(parents=True, exist_ok=True)

        job_metric = "lepton__aec2__acn__job__gpu_power_usage_merged_20260512_20260612.csv"
        util_metric = "lepton__aec2__acn__job__gpu_util_merged_20260512_20260612.csv"
        pod_metric = "lepton__aec2__acn__worker__gpu_power_usage_merged_20260512_20260612.csv"
        pod_util_metric = "lepton__aec2__acn__worker__gpu_util_merged_20260512_20260612.csv"
        cpu_metric = "lepton__aec2__acn__job__cpu_util_merged_20260512_20260612.csv"
        pod_cpu_metric = "lepton__aec2__acn__worker__cpu_util_merged_20260512_20260612.csv"

        self._write_job_csv(
            room_dir / "training" / job_metric,
            "uid,metric,value\n"
            'train-1,lepton__aec2__acn__job__gpu_power_usage,"[1778601600,""10.0""]|[1778601900,""20.0""]"\n',
        )
        self._write_job_csv(
            room_dir / "inference" / job_metric,
            "uid,metric,value\n"
            'infer-1,lepton__aec2__acn__job__gpu_power_usage,"[1778601600,""5.0""]|[1778601900,""7.0""]"\n',
        )
        self._write_job_csv(
            room_dir / "training" / util_metric,
            "uid,metric,value\n"
            'train-1,lepton__aec2__acn__job__gpu_util,"[1778601600,""0.8""]|[1778601900,""0.9""]"\n',
        )
        self._write_job_csv(
            room_dir / "inference" / util_metric,
            "uid,metric,value\n"
            'infer-1,lepton__aec2__acn__job__gpu_util,"[1778601600,""0.4""]|[1778601900,""0.5""]"\n',
        )
        self._write_job_csv(
            room_dir / "training" / cpu_metric,
            "uid,metric,value\n"
            'train-1,lepton__aec2__acn__job__cpu_util,"[1778601600,""0.2""]|[1778601900,""0.3""]"\n',
        )
        self._write_job_csv(
            room_dir / "inference" / cpu_metric,
            "uid,metric,value\n"
            'infer-1,lepton__aec2__acn__job__cpu_util,"[1778601600,""0.6""]|[1778601900,""0.7""]"\n',
        )
        self._write_job_csv(
            room_dir / "pod" / pod_metric,
            "uid,metric_name,value\n"
            'pod-1,lepton__aec2__acn__worker__gpu_power_usage,"[[[1778601600,""16.0""], [1778601900,""29.0""]]]"\n',
        )
        self._write_job_csv(
            room_dir / "pod" / pod_util_metric,
            "uid,metric_name,value\n"
            'pod-1,lepton__aec2__acn__worker__gpu_util,"[[[1778601600,""0.95""], [1778601900,""0.85""]]]"\n',
        )
        self._write_job_csv(
            room_dir / "pod" / pod_cpu_metric,
            "uid,metric_name,value\n"
            'pod-1,lepton__aec2__acn__worker__cpu_util,"[[[1778601600,""0.8""], [1778601900,""0.9""]]]"\n',
        )

    def _write_job_csv(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
