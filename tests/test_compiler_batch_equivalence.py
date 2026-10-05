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
    "config/aidc_load_15min_short/route_A/add_exogenous/"
    "lgbm_direct_holiday-weather.yaml"
)


def _build_request(origin: pd.Timestamp) -> InformationSetRequest:
    return InformationSetRequest(
        forecast_origin=origin,
        forecast_times=pd.date_range(
            origin + pd.Timedelta(minutes=15), periods=16, freq="15min"
        ),
        series_ids=(),
    )


class CompilerBatchEquivalenceTest(unittest.TestCase):
    """结构/非统计列逐值相等；仅统计内核允许受量级约束的浮点误差。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_yaml_config(CONFIG_PATH)
        cls.registry = SourceRegistry(
            cls.config.data, ROOT, generators=BUILTIN_GENERATORS
        )
        cls.compiler = FeatureCompiler(cls.config)
        base_origin = pd.Timestamp("2026-07-31 14:00:00")
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
            for column in loop_frame:
                statistical = ("_rolling_mean_" in column or "_rolling_std_" in column
                               or column.endswith(("_expanding_mean", "_expanding_std")))
                if statistical:
                    # 本仓真实窗口观测最大误差 2.68e-9（rolling_std_16，值量级约 39）；
                    # stable 侧 15min 数据观测 1.59e-9。仅 mean/std 使用此绝对误差界，
                    # 不放宽时间、lag、天气或其他列。
                    np.testing.assert_allclose(batch_frame[column], loop_frame[column],
                                               rtol=0, atol=1e-8, err_msg=f"origin {index}: {column}")
                else:
                    pd.testing.assert_series_equal(batch_frame[column], loop_frame[column], check_exact=True)

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
