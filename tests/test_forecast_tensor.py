import pickle
import unittest
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone, tzinfo
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dateutil import tz

import model_forecasting
from model_forecasting import (
    MarginalQuantileForecastTensor,
    PointForecastTensor,
    SampleForecastTensor,
)


class MutableHashableSeriesId:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(tuple(self.value))

    def __eq__(self, other):
        return isinstance(other, MutableHashableSeriesId) and self.value == other.value


class MutableTimezone(tzinfo):
    def __init__(self, offset: timedelta):
        self.offset = offset
        self.vary_per_call = False
        self.call_count = 0

    def utcoffset(self, dt):
        if self.vary_per_call:
            self.call_count += 1
            if self.call_count % 2 == 0:
                return self.offset + timedelta(hours=1)
        return self.offset

    def dst(self, dt):
        return timedelta(0)

    def tzname(self, dt):
        return "MutableTimezone"


class ForecastTensorTestCase(unittest.TestCase):
    def setUp(self):
        self.series_ids = ("series-a", "series-b")
        self.forecast_times = pd.date_range("2026-08-28", periods=3, freq="h")
        self.targets = ("load", "temperature")

    def _make_all_tensor_types(self):
        return (
            PointForecastTensor(
                values=np.ones((2, 3, 2), dtype=float),
                series_ids=self.series_ids,
                forecast_times=self.forecast_times,
                targets=self.targets,
            ),
            MarginalQuantileForecastTensor(
                values=np.ones((2, 3, 2, 2), dtype=float),
                levels=(0.1, 0.9),
                point_level=0.1,
                series_ids=self.series_ids,
                forecast_times=self.forecast_times,
                targets=self.targets,
            ),
            SampleForecastTensor(
                values=np.ones((2, 4, 3, 2), dtype=float),
                series_ids=self.series_ids,
                forecast_times=self.forecast_times,
                targets=self.targets,
            ),
        )

    def assert_tensor_values_are_immutable(self, tensor):
        self.assertFalse(hasattr(tensor, "__dict__"))
        with self.assertRaises(ValueError):
            tensor.values.setflags(write=True)
        with self.assertRaises(ValueError):
            tensor.values.flat[0] = -1.0

    def assert_pickle_round_trip(self, tensor):
        restored = pickle.loads(pickle.dumps(tensor))

        self.assertIs(type(restored), type(tensor))
        np.testing.assert_array_equal(restored.values, tensor.values)
        self.assertEqual(restored.values.dtype, tensor.values.dtype)
        self.assertTrue(restored.forecast_times.equals(tensor.forecast_times))
        self.assertEqual(restored.forecast_times.tz, tensor.forecast_times.tz)
        self.assertEqual(restored.series_ids, tensor.series_ids)
        self.assertEqual(restored.targets, tensor.targets)
        if isinstance(tensor, MarginalQuantileForecastTensor):
            self.assertEqual(restored.levels, tensor.levels)
            self.assertEqual(restored.point_level, tensor.point_level)
        if isinstance(tensor, SampleForecastTensor):
            self.assertEqual(restored.dependence_model, tensor.dependence_model)
        self.assert_tensor_values_are_immutable(restored)

    def test_forecast_times_normalize_all_datetime_units_to_nanoseconds(self):
        expected = pd.DatetimeIndex(
            np.array(
                [
                    "2026-08-28T00:00:00",
                    "2026-08-28T00:00:01",
                    "2026-08-28T00:00:02",
                ],
                dtype="datetime64[ns]",
            )
        )
        expected_ns = tuple(int(value) for value in expected.astype("datetime64[ns]").asi8)

        for unit in ("s", "ms", "us", "ns"):
            forecast_times = pd.DatetimeIndex(expected.to_numpy(dtype=f"datetime64[{unit}]"))
            tensor = PointForecastTensor(
                values=np.ones((2, 3, 2), dtype=float),
                series_ids=self.series_ids,
                forecast_times=forecast_times,
                targets=self.targets,
            )

            with self.subTest(unit=unit):
                self.assertEqual(tensor._forecast_time_ns, expected_ns)
                self.assertTrue(tensor.forecast_times.equals(expected))

    def test_forecast_times_preserve_zoneinfo_and_dateutil_timezone_values(self):
        timezones = (ZoneInfo("Asia/Shanghai"), tz.gettz("America/New_York"))

        for timezone in timezones:
            forecast_times = pd.date_range("2026-08-28", periods=3, freq="h", tz=timezone)
            tensor = PointForecastTensor(
                values=np.ones((2, 3, 2), dtype=float),
                series_ids=self.series_ids,
                forecast_times=forecast_times,
                targets=self.targets,
            )

            with self.subTest(timezone=timezone):
                self.assertIsNot(tensor._forecast_time_tz, timezone)
                self.assertEqual(list(tensor.forecast_times), list(forecast_times))
                self.assertEqual(
                    tensor.forecast_times[0].utcoffset(),
                    forecast_times[0].utcoffset(),
                )

    def test_forecast_times_preserve_canonical_zone_across_dst_transition(self):
        for source_timezone in (
            ZoneInfo("America/New_York"),
            tz.gettz("America/New_York"),
        ):
            forecast_times = pd.date_range(
                "2026-10-31 22:00:00",
                periods=8,
                freq="h",
                tz=source_timezone,
            )
            tensor = PointForecastTensor(
                values=np.ones((1, 8, 2), dtype=float),
                series_ids=("series-a",),
                forecast_times=forecast_times,
                targets=self.targets,
            )

            with self.subTest(source_timezone=source_timezone):
                restored_times = tensor.forecast_times
                self.assertEqual(list(restored_times), list(forecast_times))
                self.assertEqual(
                    [value.utcoffset() for value in restored_times],
                    [value.utcoffset() for value in forecast_times],
                )
                self.assertEqual(getattr(restored_times.tz, "key", None), "America/New_York")
                self.assertEqual(
                    [value.strftime("%Y-%m-%d %H:%M:%S %z") for value in restored_times],
                    [value.strftime("%Y-%m-%d %H:%M:%S %z") for value in forecast_times],
                )

    def test_mutable_timezone_is_frozen_for_forecast_times_and_nested_series_ids(self):
        shared_timezone = MutableTimezone(timedelta(hours=8))
        forecast_times = pd.date_range(
            "2026-08-28",
            periods=3,
            freq="h",
            tz=shared_timezone,
        )
        series_ids = (
            datetime(2026, 8, 28, 0, 0, tzinfo=shared_timezone),
            ("site-b", pd.Timestamp("2026-08-28T01:00:00", tz=shared_timezone)),
        )
        tensor = PointForecastTensor(
            values=np.ones((2, 3, 2), dtype=float),
            series_ids=series_ids,
            forecast_times=forecast_times,
            targets=self.targets,
        )
        fixed_timezone = timezone(timedelta(hours=8))
        expected_forecast_times = pd.date_range(
            "2026-08-28",
            periods=3,
            freq="h",
            tz=fixed_timezone,
        )
        expected_series_ids = (
            datetime(2026, 8, 28, 0, 0, tzinfo=fixed_timezone),
            ("site-b", pd.Timestamp("2026-08-28T01:00:00", tz=fixed_timezone)),
        )

        shared_timezone.offset = timedelta(hours=9)

        self.assertIsNot(tensor._forecast_time_tz, shared_timezone)
        self.assertEqual(list(tensor.forecast_times), list(expected_forecast_times))
        self.assertEqual(tensor.series_ids, expected_series_ids)
        self.assertIsNot(tensor.series_ids[0].tzinfo, shared_timezone)
        self.assertIsNot(tensor.series_ids[1][1].tzinfo, shared_timezone)

    def test_timezone_with_inconsistent_offsets_per_call_is_rejected(self):
        unstable_timezone = MutableTimezone(timedelta(hours=8))
        forecast_times = pd.date_range(
            "2026-08-28",
            periods=3,
            freq="h",
            tz=unstable_timezone,
        )
        unstable_timezone.vary_per_call = True

        with self.assertRaisesRegex(ValueError, "timezone.*fixed"):
            PointForecastTensor(
                values=np.ones((1, 3, 2), dtype=float),
                series_ids=("series-a",),
                forecast_times=forecast_times,
                targets=self.targets,
            )

        unstable_timezone.call_count = 0
        with self.assertRaisesRegex(ValueError, "timezone.*fixed"):
            PointForecastTensor(
                values=np.ones((1, 3, 2), dtype=float),
                series_ids=(datetime(2026, 8, 28, tzinfo=unstable_timezone),),
                forecast_times=self.forecast_times,
                targets=self.targets,
            )

    def test_series_ids_are_canonical_immutable_panel_keys(self):
        timestamp = pd.Timestamp("2026-08-28T00:00:00", tz="UTC")
        series_ids = (
            (
                np.str_("site-a"),
                np.int64(7),
                np.float32(1.5),
                np.bool_(True),
                np.bytes_(b"meter"),
                np.datetime64("2026-08-28T00:00:00", "s"),
                date(2026, 8, 28),
                datetime(2026, 8, 28, 1, 2, 3),
                timestamp,
            ),
        )
        tensor = PointForecastTensor(
            values=np.ones((1, 3, 2), dtype=float),
            series_ids=series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        self.assertEqual(
            tensor.series_ids,
            (("site-a", 7, 1.5, True, b"meter", pd.Timestamp("2026-08-28"), date(2026, 8, 28), datetime(2026, 8, 28, 1, 2, 3), timestamp),),
        )
        self.assertIs(type(tensor.series_ids[0][0]), str)
        self.assertIs(type(tensor.series_ids[0][1]), int)
        self.assertIs(type(tensor.series_ids[0][2]), float)
        self.assertIs(type(tensor.series_ids[0][3]), bool)
        self.assertIs(type(tensor.series_ids[0][4]), bytes)

    def test_series_ids_reject_mutable_custom_hashables_and_nonfinite_floats(self):
        mutable_id = MutableHashableSeriesId(["site-a"])
        invalid_ids = ((mutable_id,), (float("nan"),), (float("inf"),), (("site-a", float("-inf")),))

        for series_ids in invalid_ids:
            with self.subTest(series_ids=series_ids):
                with self.assertRaises((TypeError, ValueError)):
                    PointForecastTensor(
                        values=np.ones((1, 3, 2), dtype=float),
                        series_ids=series_ids,
                        forecast_times=self.forecast_times,
                        targets=self.targets,
                    )

        mutable_id.value.append("mutated")

    def test_series_ids_reject_missing_datetime_like_values(self):
        invalid_ids = (
            (pd.NaT,),
            (pd.Timestamp("NaT"),),
            (np.datetime64("NaT"),),
            (np.datetime64("NaT", "ns"),),
            (("site-a", pd.NaT),),
        )

        for series_ids in invalid_ids:
            with self.subTest(series_ids=series_ids):
                with self.assertRaises(ValueError):
                    PointForecastTensor(
                        values=np.ones((1, 3, 2), dtype=float),
                        series_ids=series_ids,
                        forecast_times=self.forecast_times,
                        targets=self.targets,
                    )

    def test_all_tensor_types_pickle_round_trip_exactly(self):
        timezone = tz.gettz("America/New_York")
        forecast_times = pd.date_range("2026-08-28", periods=3, freq="h", tz=timezone)
        series_ids = (("site-a", np.int64(7)), (pd.Timestamp("2026-08-28", tz="UTC"), b"meter"))
        tensors = (
            PointForecastTensor(
                values=np.arange(12, dtype=np.float32).reshape(2, 3, 2),
                series_ids=series_ids,
                forecast_times=forecast_times,
                targets=self.targets,
            ),
            MarginalQuantileForecastTensor(
                values=np.arange(24, dtype=np.float64).reshape(2, 3, 2, 2),
                levels=(0.1, 0.9),
                point_level=0.1,
                series_ids=series_ids,
                forecast_times=forecast_times,
                targets=self.targets,
            ),
            SampleForecastTensor(
                values=np.arange(48, dtype=np.float32).reshape(2, 4, 3, 2),
                series_ids=series_ids,
                forecast_times=forecast_times,
                targets=self.targets,
            ),
        )

        for tensor in tensors:
            with self.subTest(tensor=type(tensor).__name__):
                self.assert_pickle_round_trip(tensor)

    def test_all_tensor_types_pickle_preserve_exact_dtype_byte_order(self):
        tensor_specs = (
            (PointForecastTensor, (2, 3, 2), {}),
            (
                MarginalQuantileForecastTensor,
                (2, 3, 2, 2),
                {"levels": (0.1, 0.9), "point_level": 0.1},
            ),
            (SampleForecastTensor, (2, 4, 3, 2), {}),
        )

        for tensor_type, shape, extra_kwargs in tensor_specs:
            for dtype in (np.dtype(">f4"), np.dtype("<f8")):
                values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                tensor = tensor_type(
                    values=values,
                    series_ids=self.series_ids,
                    forecast_times=self.forecast_times,
                    targets=self.targets,
                    **extra_kwargs,
                )

                with self.subTest(tensor=tensor_type.__name__, dtype=dtype.str):
                    restored = pickle.loads(pickle.dumps(tensor))
                    self.assertEqual(restored.values.dtype.str, dtype.str)
                    self.assertEqual(restored._value_dtype, dtype.str)
                    self.assertEqual(restored._value_bytes, tensor._value_bytes)
                    self.assertEqual(restored._value_shape, shape)
                    np.testing.assert_array_equal(restored.values, values)
                    self.assert_tensor_values_are_immutable(restored)

    def test_pickle_reconstruction_calls_validated_public_constructors(self):
        for tensor in self._make_all_tensor_types():
            reconstruct, args = tensor.__reduce__()
            with self.subTest(tensor=type(tensor).__name__):
                with patch.object(
                    type(tensor),
                    "__init__",
                    side_effect=RuntimeError("validated constructor called"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "validated constructor called"):
                        reconstruct(*args)

    def test_pickle_reconstruction_rejects_corrupt_or_mismatched_arguments(self):
        for tensor in self._make_all_tensor_types():
            reconstruct, original_args = tensor.__reduce__()
            with self.subTest(tensor=type(tensor).__name__, corruption="shape"):
                args = list(original_args)
                args[2] = (999,) * len(tensor.shape)
                with self.assertRaises((TypeError, ValueError)):
                    reconstruct(*args)

            with self.subTest(tensor=type(tensor).__name__, corruption="targets"):
                args = list(original_args)
                args[-1] = ("load",)
                if isinstance(tensor, SampleForecastTensor):
                    args[-2] = ("load",)
                with self.assertRaises((TypeError, ValueError)):
                    reconstruct(*args)

        quantile = self._make_all_tensor_types()[1]
        reconstruct, original_args = quantile.__reduce__()
        args = list(original_args)
        args[3] = (0.1,)
        with self.assertRaises((TypeError, ValueError)):
            reconstruct(*args)

        sample = self._make_all_tensor_types()[2]
        reconstruct, original_args = sample.__reduce__()
        args = list(original_args)
        args[-1] = object()
        with self.assertRaises((TypeError, ValueError)):
            reconstruct(*args)

    def test_all_public_tensor_types_use_slots_and_immutable_value_backing(self):
        for tensor in self._make_all_tensor_types():
            with self.subTest(tensor=type(tensor).__name__):
                self.assert_tensor_values_are_immutable(tensor)
                with self.assertRaises(FrozenInstanceError):
                    tensor.values = np.zeros_like(tensor.values)

    def test_all_tensor_types_reject_public_and_internal_field_deletion(self):
        internal_fields = (
            "_value_bytes",
            "_value_dtype",
            "_value_shape",
            "_forecast_time_ns",
            "_forecast_time_tz",
        )

        for tensor_index, tensor in enumerate(self._make_all_tensor_types()):
            with self.subTest(tensor=type(tensor).__name__, field="series_ids"):
                with self.assertRaises(FrozenInstanceError):
                    del tensor.series_ids
            for field_name in internal_fields:
                tensor = self._make_all_tensor_types()[tensor_index]
                with self.subTest(tensor=type(tensor).__name__, field=field_name):
                    with self.assertRaises(FrozenInstanceError):
                        delattr(tensor, field_name)

    def test_forecast_times_exposure_cannot_mutate_internal_metadata(self):
        expected = self.forecast_times.copy()
        for tensor in self._make_all_tensor_types():
            with self.subTest(tensor=type(tensor).__name__):
                exposed_times = tensor.forecast_times
                exposed_times.values[0] = np.datetime64("2030-01-01")
                self.assertTrue(tensor.forecast_times.equals(expected))
                self.assertIsNot(tensor.forecast_times, tensor.forecast_times)

    def test_selectors_retain_slots_and_immutable_backing(self):
        for tensor in self._make_all_tensor_types():
            selected_tensors = (
                tensor.select_target("load"),
                tensor.select_series("series-a"),
            )
            if isinstance(tensor, MarginalQuantileForecastTensor):
                selected_tensors += (tensor.point(),)
            for selected in selected_tensors:
                with self.subTest(tensor=type(tensor).__name__, selected=type(selected).__name__):
                    self.assert_tensor_values_are_immutable(selected)

    def test_sample_generation_boundary_is_explicitly_reserved(self):
        with self.assertRaisesRegex(NotImplementedError, "sample generation.*not implemented"):
            SampleForecastTensor.generate()

    def test_point_tensor_properties_and_input_copy(self):
        values = np.arange(12, dtype=float).reshape(2, 3, 2)
        times = self.forecast_times.copy()

        tensor = PointForecastTensor(
            values=values,
            series_ids=self.series_ids,
            forecast_times=times,
            targets=self.targets,
        )

        self.assertEqual(tensor.shape, (2, 3, 2))
        self.assertEqual(tensor.n_series, 2)
        self.assertEqual(tensor.n_steps, 3)
        self.assertEqual(tensor.n_targets, 2)
        self.assertFalse(np.shares_memory(tensor.values, values))
        self.assertIsNot(tensor.forecast_times, times)

        values[0, 0, 0] = -999.0
        self.assertEqual(tensor.values[0, 0, 0], 0.0)
        with self.assertRaises(ValueError):
            tensor.values[0, 0, 0] = -1.0

    def test_point_tensor_supports_single_target(self):
        tensor = PointForecastTensor(
            values=np.ones((2, 3, 1), dtype=float),
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=("load",),
        )

        self.assertEqual(tensor.shape, (2, 3, 1))
        self.assertEqual(tensor.n_targets, 1)

    def test_point_time_major_matrix_has_exact_order_and_inverse(self):
        values = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
            ]
        )
        tensor = PointForecastTensor(
            values=values,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        matrix = tensor.to_time_major_matrix()

        np.testing.assert_array_equal(
            matrix,
            np.array(
                [
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ]
            ),
        )
        self.assertEqual(matrix.shape, (2, 6))
        self.assertFalse(np.shares_memory(matrix, tensor.values))

        restored = PointForecastTensor.from_time_major_matrix(
            matrix,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )
        np.testing.assert_array_equal(restored.values, values)
        matrix[0, 0] = -1.0
        self.assertEqual(restored.values[0, 0, 0], 1.0)
        with self.assertRaises(ValueError):
            restored.values[0, 0, 0] = -1.0

    def test_point_selection_preserves_three_dimensions(self):
        tensor = PointForecastTensor(
            values=np.arange(12, dtype=float).reshape(2, 3, 2),
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        target = tensor.select_target("temperature")
        series = tensor.select_series("series-b")

        self.assertEqual(target.shape, (2, 3, 1))
        self.assertEqual(target.targets, ("temperature",))
        np.testing.assert_array_equal(target.values, tensor.values[:, :, 1:2])
        self.assertEqual(series.shape, (1, 3, 2))
        self.assertEqual(series.series_ids, ("series-b",))
        np.testing.assert_array_equal(series.values, tensor.values[1:2, :, :])
        self.assertFalse(np.shares_memory(target.values, tensor.values))
        self.assertFalse(np.shares_memory(series.values, tensor.values))
        with self.assertRaises(ValueError):
            target.values[0, 0, 0] = -1.0

    def test_select_series_preserves_stored_canonical_id_type(self):
        stored_id = pd.Timestamp("2026-08-28T00:00:00")
        equal_selector = datetime(2026, 8, 28)

        for tensor in self._make_all_tensor_types():
            tensor = type(tensor)(
                values=tensor.values,
                series_ids=(stored_id, "series-b"),
                forecast_times=tensor.forecast_times,
                targets=tensor.targets,
                **(
                    {"levels": tensor.levels, "point_level": tensor.point_level}
                    if isinstance(tensor, MarginalQuantileForecastTensor)
                    else {}
                ),
            )
            selected = tensor.select_series(equal_selector)

            with self.subTest(tensor=type(tensor).__name__):
                self.assertIs(type(selected.series_ids[0]), pd.Timestamp)
                self.assertEqual(selected.series_ids, (stored_id,))

    def test_quantile_point_crossing_and_selection(self):
        values = np.array(
            [
                [
                    [[1.0, 2.0, 3.0], [10.0, 9.0, 8.0]],
                    [[2.0, 3.0, 4.0], [8.0, 9.0, 10.0]],
                    [[3.0, 4.0, 5.0], [7.0, 8.0, 9.0]],
                ],
                [
                    [[4.0, 5.0, 6.0], [6.0, 7.0, 8.0]],
                    [[5.0, 6.0, 7.0], [5.0, 6.0, 7.0]],
                    [[6.0, 7.0, 8.0], [4.0, 5.0, 6.0]],
                ],
            ]
        )
        tensor = MarginalQuantileForecastTensor(
            values=values,
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        point = tensor.point()
        self.assertIsInstance(point, PointForecastTensor)
        self.assertEqual(point.shape, (2, 3, 2))
        np.testing.assert_array_equal(point.values, values[:, :, :, 1])
        crossing_mask = tensor.crossing_mask()
        self.assertEqual(crossing_mask.shape, (2, 3, 2))
        np.testing.assert_array_equal(np.argwhere(crossing_mask), [[0, 0, 1]])
        self.assertIs(tensor.has_crossing(), True)

        target = tensor.select_target("temperature")
        series = tensor.select_series("series-b")
        self.assertEqual(target.shape, (2, 3, 1, 3))
        self.assertEqual(series.shape, (1, 3, 2, 3))
        self.assertFalse(np.shares_memory(point.values, tensor.values))
        self.assertFalse(np.shares_memory(target.values, tensor.values))
        self.assertFalse(np.shares_memory(series.values, tensor.values))

    def test_quantile_point_level_uses_matching_canonical_level(self):
        tensor = MarginalQuantileForecastTensor(
            values=np.ones((2, 3, 2, 3), dtype=float),
            levels=(np.float32(0.1), np.float64(0.5), 0.9),
            point_level=np.float32(0.5),
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        self.assertEqual(tensor.levels, (float(np.float32(0.1)), 0.5, 0.9))
        self.assertIs(tensor.point_level, tensor.levels[1])

        nearby_tensor = MarginalQuantileForecastTensor(
            values=np.ones((2, 3, 2, 2), dtype=float),
            levels=(0.5, 0.500000005),
            point_level=0.500000005,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )
        self.assertIs(nearby_tensor.point_level, nearby_tensor.levels[1])

    def test_quantile_tensor_copies_values(self):
        values = np.ones((2, 3, 2, 3), dtype=float)
        tensor = MarginalQuantileForecastTensor(
            values=values,
            levels=(0.1, 0.5, 0.9),
            point_level=0.5,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        self.assertFalse(np.shares_memory(tensor.values, values))
        values[0, 0, 0, 0] = -1.0
        self.assertEqual(tensor.values[0, 0, 0, 0], 1.0)
        with self.assertRaises(ValueError):
            tensor.values[0, 0, 0, 0] = -1.0

    def test_sample_tensor_contract_and_selection(self):
        values = np.arange(48, dtype=float).reshape(2, 4, 3, 2)
        tensor = SampleForecastTensor(
            values=values,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        self.assertEqual(tensor.shape, (2, 4, 3, 2))
        self.assertEqual(tensor.n_series, 2)
        self.assertEqual(tensor.n_samples, 4)
        self.assertEqual(tensor.n_steps, 3)
        self.assertEqual(tensor.n_targets, 2)
        self.assertIsNone(tensor.dependence_model)
        self.assertFalse(hasattr(tensor, "sample_generator"))
        self.assertFalse(np.shares_memory(tensor.values, values))
        values[0, 0, 0, 0] = -1.0
        self.assertEqual(tensor.values[0, 0, 0, 0], 0.0)
        with self.assertRaises(ValueError):
            tensor.values[0, 0, 0, 0] = -1.0
        with self.assertRaises(FrozenInstanceError):
            tensor.dependence_model = "copula"

        target = tensor.select_target("load")
        series = tensor.select_series("series-a")
        self.assertEqual(target.shape, (2, 4, 3, 1))
        self.assertEqual(series.shape, (1, 4, 3, 2))
        self.assertIsNone(target.dependence_model)
        self.assertIsNone(series.dependence_model)
        self.assertFalse(np.shares_memory(target.values, tensor.values))
        self.assertFalse(np.shares_memory(series.values, tensor.values))

    def test_rejects_invalid_value_arrays(self):
        valid_kwargs = {
            "series_ids": self.series_ids,
            "forecast_times": self.forecast_times,
            "targets": self.targets,
        }
        invalid_values = [
            np.ones((2, 3), dtype=float),
            np.ones((2, 3, 2, 1), dtype=float),
            np.ones((2, 3, 2), dtype=int),
            [[[1.0, 2.0]]],
            np.full((2, 3, 2), np.nan, dtype=float),
            np.full((2, 3, 2), np.inf, dtype=float),
            np.empty((0, 3, 2), dtype=float),
            np.empty((2, 0, 2), dtype=float),
            np.empty((2, 3, 0), dtype=float),
        ]

        for values in invalid_values:
            with self.subTest(values_type=type(values), shape=getattr(values, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    PointForecastTensor(values=values, **valid_kwargs)

    def test_rejects_invalid_identities_and_times(self):
        values = np.ones((2, 3, 2), dtype=float)
        invalid_cases = [
            {"series_ids": (), "forecast_times": self.forecast_times, "targets": self.targets},
            {"series_ids": ("a",), "forecast_times": self.forecast_times, "targets": self.targets},
            {"series_ids": ("a", "a"), "forecast_times": self.forecast_times, "targets": self.targets},
            {"series_ids": self.series_ids, "forecast_times": pd.DatetimeIndex([]), "targets": self.targets},
            {
                "series_ids": self.series_ids,
                "forecast_times": pd.DatetimeIndex(["2026-08-28", "2026-08-28", "2026-08-29"]),
                "targets": self.targets,
            },
            {
                "series_ids": self.series_ids,
                "forecast_times": pd.DatetimeIndex(["2026-08-29", "2026-08-28", "2026-08-30"]),
                "targets": self.targets,
            },
            {
                "series_ids": self.series_ids,
                "forecast_times": pd.DatetimeIndex(["2026-08-28", pd.NaT, "2026-08-30"]),
                "targets": self.targets,
            },
            {"series_ids": self.series_ids, "forecast_times": self.forecast_times[:2], "targets": self.targets},
            {"series_ids": self.series_ids, "forecast_times": self.forecast_times, "targets": ()},
            {"series_ids": self.series_ids, "forecast_times": self.forecast_times, "targets": ("load",)},
            {"series_ids": self.series_ids, "forecast_times": self.forecast_times, "targets": ("load", "load")},
            {"series_ids": self.series_ids, "forecast_times": self.forecast_times, "targets": ("load", "")},
        ]

        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    PointForecastTensor(values=values, **kwargs)

    def test_rejects_unknown_selection(self):
        tensor = PointForecastTensor(
            values=np.ones((2, 3, 2), dtype=float),
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        with self.assertRaises(KeyError):
            tensor.select_target("missing")
        with self.assertRaises(KeyError):
            tensor.select_series("missing")

    def test_rejects_invalid_time_major_matrix(self):
        invalid_matrices = [
            np.ones((2, 3, 2), dtype=float),
            np.ones((2, 5), dtype=float),
            np.ones((1, 6), dtype=float),
            np.ones((2, 6), dtype=int),
            np.full((2, 6), np.nan, dtype=float),
            [[1.0] * 6, [2.0] * 6],
        ]

        for matrix in invalid_matrices:
            with self.subTest(matrix_type=type(matrix), shape=getattr(matrix, "shape", None)):
                with self.assertRaises((TypeError, ValueError)):
                    PointForecastTensor.from_time_major_matrix(
                        matrix,
                        series_ids=self.series_ids,
                        forecast_times=self.forecast_times,
                        targets=self.targets,
                    )

    def test_rejects_invalid_quantile_contract(self):
        valid_kwargs = {
            "series_ids": self.series_ids,
            "forecast_times": self.forecast_times,
            "targets": self.targets,
        }
        invalid_cases = [
            {"values": np.ones((2, 3, 2), dtype=float), "levels": (0.1,), "point_level": 0.1},
            {"values": np.ones((2, 3, 2, 0), dtype=float), "levels": (), "point_level": 0.5},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.1,), "point_level": 0.1},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.1, 0.1), "point_level": 0.1},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.9, 0.1), "point_level": 0.1},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.0, 0.5), "point_level": 0.5},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.5, 1.0), "point_level": 0.5},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.1, np.nan), "point_level": 0.1},
            {"values": np.ones((2, 3, 2, 2), dtype=float), "levels": (0.1, 0.9), "point_level": 0.5},
            {"values": np.full((2, 3, 2, 2), np.inf, dtype=float), "levels": (0.1, 0.9), "point_level": 0.1},
        ]

        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises((TypeError, ValueError)):
                    MarginalQuantileForecastTensor(**case, **valid_kwargs)

    def test_quantile_levels_and_point_level_require_scalar_floats(self):
        values = np.ones((2, 3, 2, 2), dtype=float)
        valid_kwargs = {
            "values": values,
            "series_ids": self.series_ids,
            "forecast_times": self.forecast_times,
            "targets": self.targets,
        }
        invalid_levels = [
            [0.1, 0.9],
            ("0.1", 0.9),
            (1, 0.9),
            (False, 0.9),
            ([0.1], 0.9),
            (np.array(0.1), 0.9),
            (np.array([0.1]), 0.9),
        ]
        for levels in invalid_levels:
            with self.subTest(levels=levels):
                with self.assertRaises(TypeError):
                    MarginalQuantileForecastTensor(levels=levels, point_level=0.9, **valid_kwargs)

        for point_level in ("0.1", 1, True, [0.1], np.array(0.1), np.inf, np.nan):
            with self.subTest(point_level=point_level):
                expected_error = ValueError if isinstance(point_level, (float, np.floating)) else TypeError
                with self.assertRaises(expected_error):
                    MarginalQuantileForecastTensor(
                        levels=(0.1, 0.9),
                        point_level=point_level,
                        **valid_kwargs,
                    )

    def test_quantile_and_sample_reject_metadata_mismatch_and_types(self):
        quantile_values = np.ones((2, 3, 2, 2), dtype=float)
        sample_values = np.ones((2, 4, 3, 2), dtype=float)
        invalid_metadata = [
            ({"series_ids": ["series-a", "series-b"]}, TypeError),
            ({"series_ids": ("series-a",)}, ValueError),
            ({"forecast_times": list(self.forecast_times)}, TypeError),
            ({"forecast_times": self.forecast_times[:2]}, ValueError),
            ({"targets": ["load", "temperature"]}, TypeError),
            ({"targets": ("load",)}, ValueError),
        ]

        for overrides, expected_error in invalid_metadata:
            metadata = {
                "series_ids": self.series_ids,
                "forecast_times": self.forecast_times,
                "targets": self.targets,
            }
            metadata.update(overrides)
            with self.subTest(tensor="quantile", overrides=overrides):
                with self.assertRaises(expected_error):
                    MarginalQuantileForecastTensor(
                        values=quantile_values,
                        levels=(0.1, 0.9),
                        point_level=0.1,
                        **metadata,
                    )
            with self.subTest(tensor="sample", overrides=overrides):
                with self.assertRaises(expected_error):
                    SampleForecastTensor(values=sample_values, **metadata)

    def test_all_tensor_types_reject_blank_or_whitespace_target_names(self):
        invalid_targets = (("load", ""), ("load", "   "), ("load", " temperature"), ("load", "temperature "))
        constructors = [
            lambda targets: PointForecastTensor(
                values=np.ones((2, 3, 2), dtype=float),
                series_ids=self.series_ids,
                forecast_times=self.forecast_times,
                targets=targets,
            ),
            lambda targets: MarginalQuantileForecastTensor(
                values=np.ones((2, 3, 2, 2), dtype=float),
                levels=(0.1, 0.9),
                point_level=0.1,
                series_ids=self.series_ids,
                forecast_times=self.forecast_times,
                targets=targets,
            ),
            lambda targets: SampleForecastTensor(
                values=np.ones((2, 4, 3, 2), dtype=float),
                series_ids=self.series_ids,
                forecast_times=self.forecast_times,
                targets=targets,
            ),
        ]

        for targets in invalid_targets:
            for constructor in constructors:
                with self.subTest(targets=targets, constructor=constructor):
                    with self.assertRaises(ValueError):
                        constructor(targets)

    def test_rejects_invalid_sample_contract(self):
        valid_kwargs = {
            "series_ids": self.series_ids,
            "forecast_times": self.forecast_times,
            "targets": self.targets,
        }
        invalid_values = [
            np.ones((2, 3, 2), dtype=float),
            np.empty((0, 4, 3, 2), dtype=float),
            np.empty((2, 0, 3, 2), dtype=float),
            np.empty((2, 4, 0, 2), dtype=float),
            np.empty((2, 4, 3, 0), dtype=float),
            np.ones((2, 4, 3, 2), dtype=int),
            np.full((2, 4, 3, 2), np.nan, dtype=float),
            np.full((2, 4, 3, 2), np.inf, dtype=float),
            [[[[1.0]]]],
        ]

        for values in invalid_values:
            with self.subTest(values_type=type(values), shape=getattr(values, "shape", None)):
                expected_error = TypeError if not isinstance(values, np.ndarray) or (
                    isinstance(values, np.ndarray) and not np.issubdtype(values.dtype, np.floating)
                ) else ValueError
                with self.assertRaises(expected_error):
                    SampleForecastTensor(values=values, **valid_kwargs)

    def test_quantile_and_sample_reject_unknown_selectors(self):
        quantile = MarginalQuantileForecastTensor(
            values=np.ones((2, 3, 2, 2), dtype=float),
            levels=(0.1, 0.9),
            point_level=0.1,
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )
        sample = SampleForecastTensor(
            values=np.ones((2, 4, 3, 2), dtype=float),
            series_ids=self.series_ids,
            forecast_times=self.forecast_times,
            targets=self.targets,
        )

        for tensor in (quantile, sample):
            with self.subTest(tensor=type(tensor).__name__, selector="target"):
                with self.assertRaises(KeyError):
                    tensor.select_target("missing")
            with self.subTest(tensor=type(tensor).__name__, selector="series"):
                with self.assertRaises(KeyError):
                    tensor.select_series("missing")

    def test_public_exports_use_explicit_marginal_quantile_name(self):
        self.assertEqual(
            model_forecasting.__all__,
            [
                "MarginalQuantileForecastTensor",
                "PointForecastTensor",
                "SampleForecastTensor",
            ],
        )
        self.assertFalse(hasattr(model_forecasting, "QuantileForecastTensor"))


if __name__ == "__main__":
    unittest.main()
