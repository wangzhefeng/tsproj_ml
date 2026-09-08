"""批训练不得复用较晚原点生成的可得性事实；仅编译合成设计。"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from data_loading import BUILTIN_GENERATORS, SourceRegistry
from forecasting_core.specs import ColumnSpec, DataSourceSpec, DataSpec
from model_pipeline.supervised_design import SupervisedDesignBuilder
import test_canonical_runtime_smoke as smoke


class TrainingOriginMaterializationTest(unittest.TestCase):
    def test_generated_calendar_batch_matches_independent_origins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "target.csv"
            times = pd.date_range("2026-08-01", periods=48, freq="1h")
            pd.DataFrame({"time": times, "load": 100 + np.arange(48)}).to_csv(source, index=False)
            config = smoke.CanonicalRuntimeSmokeTest().build_config(source, mode="point", strategy="direct", align_to_target=False)
            calendar = DataSourceSpec(name="holiday", source_type="generated", generator="chinese_holiday",
                columns=(ColumnSpec("is_holiday", "known_future"),), time_col="time", availability="generator_defined")
            config = replace(config, data=DataSpec((*config.data.sources, calendar)))
            batch = SupervisedDesignBuilder(config, SourceRegistry(config.data, root, generators=BUILTIN_GENERATORS))
            single = SupervisedDesignBuilder(config, SourceRegistry(config.data, root, generators=BUILTIN_GENERATORS))
            origins = tuple(times[10:20])
            expected = tuple(single.training_row(origin) for origin in origins)
            actual = batch.training_rows(origins)
            self.assertEqual(len(actual), len(expected))
            for (designs, targets), (expected_designs, expected_targets) in zip(actual, expected):
                np.testing.assert_array_equal(targets, expected_targets)
                for design, expected_design in zip(designs, expected_designs):
                    np.testing.assert_array_equal(design, expected_design)


if __name__ == "__main__":
    unittest.main()
