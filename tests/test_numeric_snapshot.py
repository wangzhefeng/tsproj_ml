"""跨折 source-time 数值快照共享底层数组，拒绝运行中数据修订。"""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from data_loading.sources.source_io import SourceFrames
from tests.test_raw_history_window import make_config


class NumericSnapshotTest(unittest.TestCase):
    def test_snapshot_shared_readonly_and_revision_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"time": pd.date_range("2026-01-01", periods=52, freq="h"),
                          "load": np.arange(52.)}).to_csv(path, index=False)
            source = make_config(path, strategy="direct").data.sources[0]
            frames = SourceFrames(Path(directory), pd.read_csv)
            times, values = frames.numeric_history(source, "load")
            _, again = frames.numeric_history(source, "load")
            self.assertIs(values, again)
            self.assertEqual(len(times), 52)
            self.assertFalse(values.flags.writeable)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                frames.numeric_history(source, "load")
