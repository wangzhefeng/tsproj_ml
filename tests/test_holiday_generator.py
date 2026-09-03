# -*- coding: utf-8 -*-
"""chinese_holiday generated source 单测（方案 A+B，2026-09-01）。

覆盖：帧合同（覆盖 forecast_times、available_at=origin、值语义）、
DataSpec 合法性（generated + known_future + forecast_origin 组合）、
端到端 registry 物化 + FeatureCompiler 入模 + VisibilityProof、
builtin 注册、导出脚本与在线特征逐点一致（审计兜底合同）。
依赖 chinese-calendar 数据（2004–2026），用 2026 国庆做锚点。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loading import (
    BUILTIN_GENERATORS,
    InformationSetRequest,
    SourceRegistry,
)
from data_loading.holiday_generator import (
    chinese_holiday_frame,
    chinese_holiday_generator,
    generator_name,
)
from feature_engineering import FeatureCompiler
from forecasting_core.specs import (
    AvailabilityPolicy,
    ColumnSpec,
    DataSourceSpec,
    DataSpec,
    EstimatorSpec,
    FeatureSpec,
    ForecastConfigSpec,
    ForecastProblemSpec,
    ForecastStrategySpec,
)


def _holiday_request(start: str, end: str) -> InformationSetRequest:
    times = pd.date_range(start, end, freq="1D", inclusive="left")
    return InformationSetRequest(
        forecast_origin=times[0] - pd.Timedelta(days=1),
        forecast_times=times,
        series_ids=(),
    )


class ChineseHolidayFrameTest(unittest.TestCase):
    def test_frame_semantics_on_national_day_2026(self):
        # 2026-10-01 起国庆假期；前一日为工作日，节前倒计时递减。
        # end 排他：end=10-04 -> 覆盖 9/28..10/3 共 6 天。
        frame = chinese_holiday_frame("2026-09-28", "2026-10-04", freq="1D")
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame["is_holiday"].tolist(), [0, 0, 0, 1, 1, 1])
        names = frame.loc[3:, "holiday_name"].unique().tolist()
        self.assertTrue(all("National Day" in name for name in names))
        # 节前倒计时：9/28 -> 3, 9/29 -> 2, 9/30 -> 1；假日为 0。
        self.assertEqual(frame["next_holiday_days"].tolist(), [3.0, 2.0, 1.0, 0.0, 0.0, 0.0])

    def test_intraday_freq_inherits_day_state(self):
        # end 排他：start 当天 96 个点。
        frame = chinese_holiday_frame("2026-10-01", "2026-10-02", freq="15min")
        self.assertEqual(len(frame), 96)
        self.assertTrue(frame["is_holiday"].all())

    def test_end_exclusive_boundary(self):
        # inclusive="left"：end 日不包含，与仓库时间边界约定一致。
        frame = chinese_holiday_frame("2026-10-01", "2026-10-02", freq="1D")
        self.assertEqual(len(frame), 1)

    def test_out_of_coverage_raises(self):
        with self.assertRaises(Exception):
            chinese_holiday_frame("2027-01-01", "2027-01-03", freq="1D")


class ChineseHolidayGeneratorTest(unittest.TestCase):
    def test_generator_contract_matches_request_grid(self):
        request = _holiday_request("2026-10-01", "2026-10-08")
        frame = chinese_holiday_generator(None, request)
        self.assertEqual(len(frame), request.H)
        self.assertEqual(frame["time"].tolist(), list(request.forecast_times))
        self.assertTrue(frame["available_at"].eq(request.forecast_origin).all())
        self.assertTrue((frame.loc[0:6, "is_holiday"] == 1).all())

    def test_new_columns_prev_adjusted_solar(self):
        """F2+F4：prev_holiday_days 节后恢复、is_adjusted_workday 调休班日、
        solar_term 节令继承（值已于 2026-10 与 2026-08 实测核对）。"""
        frame = chinese_holiday_frame("2026-10-08", "2026-10-12", freq="1D")
        times = pd.to_datetime(frame["time"])
        rows = frame.to_dict("records")
        by_date = {
            str(times.iloc[index].date()): rows[index] for index in range(len(rows))
        }
        # 节后恢复：10/8 距国庆假期尾（10/7）1 天；10/10 距 3 天。
        self.assertEqual(by_date["2026-10-08"]["prev_holiday_days"], 1.0)
        self.assertEqual(by_date["2026-10-10"]["prev_holiday_days"], 3.0)
        # 调休班日：10/10 周六上班=1；10/11 周日补假=0（is_holiday=1）。
        self.assertEqual(by_date["2026-10-10"]["is_adjusted_workday"], 1)
        self.assertEqual(by_date["2026-10-11"]["is_adjusted_workday"], 0)
        self.assertEqual(by_date["2026-10-11"]["is_holiday"], 1)
        # 调休班日不携带节日名（holiday_name 非空 => is_holiday=1）。
        self.assertEqual(by_date["2026-10-10"]["holiday_name"], "")
        # 节令：10 月上旬处寒露；8 月上旬立秋前后。
        self.assertEqual(by_date["2026-10-10"]["solar_term"], "寒露")
        # end 排他：08-06..08-08 两行（08-06 大暑、08-07 立秋）。
        august = chinese_holiday_frame("2026-08-06", "2026-08-08", freq="1D")
        terms = august["solar_term"].tolist()
        self.assertEqual(terms, ["大暑", "立秋"])

    def test_builtin_registration(self):
        self.assertIn(generator_name(), BUILTIN_GENERATORS)
        self.assertIs(BUILTIN_GENERATORS[generator_name()], chinese_holiday_generator)


class HolidaySourceEndToEndTest(unittest.TestCase):
    """DataSpec 合法性 + registry 物化 + compiler 可见性留痕（真实 ForecastConfigSpec）。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        # 目标 history：与预测网格衔接的历史段（9 月），保证 compile 需要
        # 的 target_history 非空。
        history_times = pd.date_range("2026-09-01", "2026-09-30", freq="1D")
        pd.DataFrame({"ts": history_times, "load": 1.0}).to_csv(
            self.base_dir / "targets.csv", index=False
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _config(self) -> ForecastConfigSpec:
        sources = [
            DataSourceSpec(
                name="targets",
                source_type="file",
                columns=(ColumnSpec("load", "target"),),
                history_path="targets.csv",
                time_col="ts",
                availability="source_time",
            ),
            DataSourceSpec(
                name="chinese_holiday",
                source_type="generated",
                generator=generator_name(),
                columns=(
                    ColumnSpec("is_holiday", "known_future", categorical=False),
                    ColumnSpec("holiday_name", "known_future", categorical=True),
                    ColumnSpec("next_holiday_days", "known_future", categorical=False),
                ),
                time_col="time",
                availability="generator_defined",
            ),
        ]
        return ForecastConfigSpec(
            problem=ForecastProblemSpec(
                time_col="ts",
                freq="1D",
                horizon=5,
                targets=("load",),
                training_scope="local",
                series_id_cols=(),
            ),
            data=DataSpec(tuple(sources)),
            features=FeatureSpec(
                target_lags={"load": (1,)},
                observed_past_lags={},
                datetime_features=("day_of_week",),
                # Direct + align_to_target=false：lag 锚点冻结在 origin，
                # 不需要未来 target provider（H=5 > lag=1）。
                transformations={
                    "direct": {
                        "layout": "independent_models",
                        "use_horizon_exogenous": False,
                        "align_to_target": False,
                    }
                },
            ),
            strategy=ForecastStrategySpec("direct"),
            estimator=EstimatorSpec(model_type="ridge", target_adapter="independent"),
            probabilistic={},
            validation={},
            output={},
        )

    def test_dataspec_accepts_generated_known_future_forecast_origin(self):
        config = self._config()  # 构造即校验，非法组合会在 spec 层 RAISE
        holiday_source = next(
            s for s in config.data.sources if s.source_type == "generated"
        )
        # spec 层强约束：generated known_future 必须 generator_defined。
        self.assertEqual(
            holiday_source.availability,
            AvailabilityPolicy.GENERATOR_DEFINED,
        )

    def test_registry_materializes_and_compiler_proofs(self):
        config = self._config()
        registry = SourceRegistry(config.data, self.base_dir, generators=BUILTIN_GENERATORS)
        # 预测 10/1-10/5（国庆假期），origin 为 9/30。
        request = _holiday_request("2026-10-01", "2026-10-06")
        materialized = registry.materialize(request)
        frame = materialized.known_future["chinese_holiday"]
        self.assertEqual(len(frame), request.H)
        self.assertTrue(
            {"time", "is_holiday", "holiday_name", "next_holiday_days"}.issubset(
                frame.columns
            )
        )
        self.assertEqual(frame["is_holiday"].tolist(), [1] * request.H)

        compiled = FeatureCompiler(config).compile(materialized, request)
        self.assertIn("is_holiday", compiled.schema.feature_names)
        self.assertIn("next_holiday_days", compiled.schema.feature_names)
        self.assertEqual(compiled.frame["is_holiday"].tolist(), [1] * request.H)
        proofs = [
            p for p in compiled.visibility_proof if p.feature_name == "is_holiday"
        ]
        self.assertEqual(len(proofs), request.H)
        self.assertTrue(all(p.available_at <= request.forecast_origin for p in proofs))


class ExportScriptConsistencyTest(unittest.TestCase):
    """方案 B 审计兜底合同：导出 CSV 与在线 generated 逐点一致。"""

    def test_export_csv_matches_generator(self):
        import subprocess

        out = Path(self.temp_dir.name) / "holiday_audit.csv"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/export_chinese_holiday_csv.py",
                "--start", "2026-09-28",
                "--end", "2026-10-07",
                "--output", str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        exported = pd.read_csv(out, parse_dates=["time"])
        # CLI --end 为包含语义：2026-10-07 必须在文件中（且为调休班日）。
        last_row = exported.iloc[-1]
        self.assertEqual(str(last_row["time"].date()), "2026-10-07")
        self.assertEqual(int(last_row["is_holiday"]), 1)
        self.assertEqual(last_row["holiday_name"], "National Day")

        request = _holiday_request("2026-09-28", "2026-10-09")
        online = chinese_holiday_generator(None, request).drop(
            columns=["available_at"]
        )
        online["time"] = pd.to_datetime(online["time"])
        # CSV 空串读回是 NaN（工作日无假日名），统一填空串后比较。
        exported["holiday_name"] = exported["holiday_name"].fillna("")
        merged = exported.merge(online, on="time", suffixes=("_csv", "_online"))
        self.assertEqual(len(merged), len(exported))
        for column in ("is_holiday", "holiday_name", "next_holiday_days"):
            self.assertTrue(
                (merged[f"{column}_csv"] == merged[f"{column}_online"]).all(),
                f"export mismatch on {column}",
            )

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
