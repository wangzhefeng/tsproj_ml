"""联通 Excel 离线功率处理合同。"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "config/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"
sys.path.insert(0, str(SCRIPT_DIR))


class LiantongPowerProcessTest(unittest.TestCase):
    def test_excel_to_csv_partial_sum_and_mapping(self):
        import liantong_power_process as processor

        with TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            output = Path(temp) / "output"
            source.mkdir()
            reference = pd.DataFrame({
                "SignalID": ["001", "002", "003", "004"],
                "RoomID": ["room"] * 4,
                "DevAssestID": ["total", "phase", "phase", "phase"],
                "DeviceName": ["总表", "分相表", "分相表", "分相表"],
                "SignalName": ["输入总有功功率"] + [f"A(1)路输入{p}相有功功率" for p in "ABC"],
                "Describe": [None, "KW", "KW", "KW"],
            })
            reference.to_excel(source / "联通IT测点筛选表.xlsx", index=False)
            pd.DataFrame([
                ["2026-08-01 00:01:02", "10", "001", "总表", "输入总有功功率"],
                ["2026-08-01 00:03:35", "2", "002", "分相表", "A(1)路输入A相有功功率"],
                ["2026-08-01 00:01:00", "3", "003", "分相表", "A(1)路输入B相有功功率"],
                ["2026-08-01 00:06:00", "0", "001", "总表", "输入总有功功率"],
                ["1786291200000", None, "unlisted", None, None],
            ], columns=["time", "value", "signalid", "DeviceName", "SignalName"]).to_excel(source / "data-test.xlsx", index=False)
            processor.run(source, output)
            result = pd.read_csv(output / "df_power.csv")
            self.assertEqual(result.shape, (8928, 4))
            self.assertEqual(result.columns.tolist(), ["time", "point_1_value", "point_2_value", "value"])
            self.assertEqual(result.iloc[0].tolist(), ["2026-08-01 00:00:00", 10, 5, 15])
            self.assertEqual(result.loc[1, "value"], 0)
            self.assertTrue(result.loc[1, "point_2_value"] != result.loc[1, "point_2_value"])
            self.assertTrue(result.iloc[2, 1:].isna().all())
            self.assertEqual(result.time.iloc[-1], "2026-08-31 23:55:00")
            mapping = pd.read_csv(output / "point_mapping.csv", dtype=str)
            self.assertEqual(mapping.SignalID.tolist(), ["001", "002", "003", "004"])
            self.assertEqual(mapping.point_column.tolist(), ["point_1_value"] + ["point_2_value"] * 3)


if __name__ == "__main__":
    unittest.main()
