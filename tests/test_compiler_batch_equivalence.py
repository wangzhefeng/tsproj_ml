# -*- coding: utf-8 -*-
"""compile_batch 向量化路径与逐行 compile 的等价性对照测试。

先写测试后写实现（TDD）：compile_batch 不存在时本文件必须红。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.config_loader import load_yaml_config  # noqa: E402
from data_loading import BUILTIN_GENERATORS  # noqa: E402
from data_loading import InformationSetRequest  # noqa: E402
from data_loading.registry import SourceRegistry  # noqa: E402
from feature_engineering.compiler import FeatureCompiler  # noqa: E402

CONFIG_PATH = (
    "config/aidc_load_15min_daily/route_A/add_exogenous/"
    "lgbm_direct_holiday-weather.yaml"
)


def _build_request(origin: pd.Timestamp) -> InformationSetRequest:
    return InformationSetRequest(
        forecast_origin=origin,
        forecast_times=pd.date_range(
            origin + pd.Timedelta(minutes=15), periods=96, freq="15min"
        ),
        series_ids=(),
    )


class CompilerBatchEquivalenceTest(unittest.TestCase):
    """compile_batch 必须与逐 origin compile 循环输出逐值相等。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml_config(CONFIG_PATH)
        cls.registry = SourceRegistry(
            cls.config.data, ROOT, generators=BUILTIN_GENERATORS
        )
        cls.compiler = FeatureCompiler(cls.config)
        base_origin = pd.Timestamp("2026-07-31 23:45:00")
        cls.origins = tuple(
            base_origin - pd.Timedelta(days=k) for k in range(6)
        )

    def test_frame_values_identical_to_per_origin_loop(self) -> None:
        loop_frames = []
        for origin in self.origins:
            request = _build_request(origin)
            info = self.registry.materialize(request)
            compiled = self.compiler.compile(info, request)
            loop_frames.append(compiled.frame)

        requests = [_build_request(origin) for origin in self.origins]
        information_sets = [
            self.registry.materialize(request) for request in requests
        ]
        batch = self.compiler.compile_batch(information_sets, requests)

        self.assertEqual(len(batch), len(loop_frames))
        for index, (batch_compiled, loop_frame) in enumerate(
            zip(batch, loop_frames)
        ):
            batch_frame = batch_compiled.frame
            self.assertEqual(
                list(batch_frame.columns),
                list(loop_frame.columns),
                f"origin {index}: column mismatch",
            )
            np.testing.assert_allclose(
                batch_frame.to_numpy(dtype=float),
                loop_frame.to_numpy(dtype=float),
                rtol=0,
                atol=0,
                err_msg=f"origin {index}: feature values differ",
            )
            self.assertEqual(
                list(batch_frame.columns),
                list(loop_frame.columns),
                f"origin {index}: column mismatch",
            )
            np.testing.assert_allclose(
                batch_frame.to_numpy(dtype=float),
                loop_frame.to_numpy(dtype=float),
                rtol=0,
                atol=0,
                err_msg=f"origin {index}: feature values differ",
            )

    def test_nan_positions_identical(self) -> None:
        loop_frames = []
        for origin in self.origins:
            request = _build_request(origin)
            info = self.registry.materialize(request)
            loop_frames.append(self.compiler.compile(info, request).frame)

        requests = [_build_request(origin) for origin in self.origins]
        information_sets = [
            self.registry.materialize(request) for request in requests
        ]
        batch = self.compiler.compile_batch(information_sets, requests)
        for index, (batch_compiled, loop_frame) in enumerate(
            zip(batch, loop_frames)
        ):
            batch_frame = batch_compiled.frame
            np.testing.assert_array_equal(
                np.isnan(batch_frame.to_numpy(dtype=float)),
                np.isnan(loop_frame.to_numpy(dtype=float)),
                err_msg=f"origin {index}: NaN mask differs",
            )

    def test_asof_visibility_identical(self) -> None:
        """batch 路径的可见性证明必须与逐行路径逐字段一致。"""
        loop_proofs = []
        for origin in self.origins:
            request = _build_request(origin)
            info = self.registry.materialize(request)
            compiled = self.compiler.compile(info, request)
            loop_proofs.append(compiled.visibility_proof)

        requests = [_build_request(origin) for origin in self.origins]
        information_sets = [
            self.registry.materialize(request) for request in requests
        ]
        batch = self.compiler.compile_batch(information_sets, requests)
        for index, batch_compiled in enumerate(batch):
            self.assertEqual(
                len(batch_compiled.visibility_proof),
                len(loop_proofs[index]),
                f"origin {index}: proof count differs",
            )
            self.assertEqual(
                batch_compiled.visibility_proof,
                loop_proofs[index],
                f"origin {index}: visibility proof fields differ",
            )


if __name__ == "__main__":
    unittest.main()
