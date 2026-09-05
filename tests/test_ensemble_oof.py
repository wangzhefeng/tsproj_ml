# -*- coding: utf-8 -*-
"""E3: OOF fold contracts, cache invalidation, and calendar_month boundary."""

from __future__ import annotations

import ast
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Lock
from time import sleep

import numpy as np
import pandas as pd

from model_ensemble import cache as oof_cache
from model_ensemble.artifacts import OOFPredictionArtifact
from model_ensemble.oof import oof_fold_origins


from model_testing import validation


class OOFFoldOriginsTest(unittest.TestCase):
    """Fold geometry contracts on synthetic origin timelines."""

    @staticmethod
    def _fake_runner(origins):
        geometry = validation.TimeGeometry(
            offset=pd.tseries.frequencies.to_offset("1h"),
            horizon=2,
        )

        class FakeRunner:
            pass

        runner = FakeRunner()
        runner.geometry = geometry
        runner.supervised_origins = origins
        return runner

    def test_folds_are_chronological_and_non_overlapping(self):
        origins = tuple(pd.date_range("2026-01-01", periods=30, freq="1h"))
        runner = self._fake_runner(origins)
        folds = oof_fold_origins(
            runner,
            fold_count=3,
            stride_steps=1,
            train_window_steps=6,
        )
        self.assertEqual(len(folds), 3)
        origins_seq = [fold["origin"] for fold in folds]
        self.assertEqual(origins_seq, sorted(origins_seq))
        geometry = runner.geometry
        for fold in folds:
            holdout_start = geometry.label_start(fold["origin"])
            for index in fold["train_indices"]:
                self.assertLess(
                    geometry.label_end(origins[index]), holdout_start
                )

    def test_stride_selects_distinct_folds(self):
        origins = tuple(pd.date_range("2026-01-01", periods=40, freq="1h"))
        runner = self._fake_runner(origins)
        folds = oof_fold_origins(
            runner,
            fold_count=2,
            stride_steps=3,
            train_window_steps=6,
        )
        self.assertEqual(len(folds), 2)
        self.assertNotEqual(folds[0]["origin"], folds[1]["origin"])
        gap = (folds[1]["origin"] - folds[0]["origin"]) / pd.Timedelta(hours=1)
        self.assertEqual(int(gap), 3)

    def test_outer_cutoff_excludes_leaking_folds(self):
        origins = tuple(pd.date_range("2026-01-01", periods=30, freq="1h"))
        runner = self._fake_runner(origins)
        cutoff_origin = origins[24]
        folds = oof_fold_origins(
            runner,
            fold_count=3,
            stride_steps=1,
            train_window_steps=6,
            outer_cutoff_origin=cutoff_origin,
        )
        geometry = runner.geometry
        cutoff_label_start = geometry.label_start(cutoff_origin)
        for fold in folds:
            self.assertLess(geometry.label_end(fold["origin"]), cutoff_label_start)

    def test_no_valid_fold_raises(self):
        origins = tuple(pd.date_range("2026-01-01", periods=2, freq="1h"))
        runner = self._fake_runner(origins)
        with self.assertRaises(ValueError):
            oof_fold_origins(
                runner,
                fold_count=1,
                stride_steps=1,
                train_window_steps=4,
            )


def _artifact(fingerprint: str = "f" * 64) -> OOFPredictionArtifact:
    return OOFPredictionArtifact(
        values_by_member={
            "a": np.arange(12.0).reshape(2, 2, 3),
            "b": np.arange(12.0).reshape(2, 2, 3) * -1.0,
        },
        member_order=("a", "b"),
        targets=("load", "power", "temp"),
        horizon=2,
        quantile_levels=None,
        oof_fingerprint=fingerprint,
        folds=({"fold": 1, "origin": "2026-01-01T00:00:00"},),
    )


def _get_or_create_oof_cache_in_process(
    cache_root: str,
    counter,
    attempting,
    entered_factory,
    release_factory,
    result_queue,
) -> None:
    def factory():
        with counter.get_lock():
            counter.value += 1
        entered_factory.set()
        if not release_factory.wait(timeout=10.0):
            raise TimeoutError("test process did not receive factory release")
        return _artifact()

    attempting.set()
    _artifact_result, cache_hit = oof_cache.get_or_create_oof_cache(
        cache_root,
        "f" * 64,
        factory,
    )
    result_queue.put(cache_hit)


class OOFArtifactContractTest(unittest.TestCase):
    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            OOFPredictionArtifact(
                values_by_member={
                    "a": np.zeros((2, 2, 3)),
                    "b": np.zeros((3, 2, 3)),
                },
                member_order=("a", "b"),
                targets=("load", "power", "temp"),
                horizon=2,
                quantile_levels=None,
                oof_fingerprint="x",
            )

    def test_horizon_mismatch_raises(self):
        with self.assertRaises(ValueError):
            OOFPredictionArtifact(
                values_by_member={"a": np.zeros((2, 5, 3))},
                member_order=("a",),
                targets=("load", "power", "temp"),
                horizon=2,
                quantile_levels=None,
                oof_fingerprint="x",
            )


class OOFPositiveDepthTest(unittest.TestCase):
    def test_quantile_depth_enforced(self):
        # Q dimension (5) must match quantile_levels count (3)
        with self.assertRaises(ValueError):
            OOFPredictionArtifact(
                values_by_member={"a": np.zeros((2, 2, 3, 5))},
                member_order=("a",),
                targets=("load", "power", "temp"),
                horizon=2,
                quantile_levels=(0.1, 0.5, 0.9),
                oof_fingerprint="x",
            )


class OOFCacheTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_and_load_round_trip(self):
        artifact = _artifact()
        oof_cache.save_oof_cache(self.root, artifact)
        loaded = oof_cache.load_oof_cache(self.root, artifact.oof_fingerprint)
        self.assertEqual(loaded.member_order, artifact.member_order)
        np.testing.assert_array_equal(
            loaded.values_by_member["a"], artifact.values_by_member["a"]
        )

    def test_same_fingerprint_reuses_cache(self):
        artifact = _artifact()
        first = oof_cache.save_oof_cache(self.root, artifact)
        second = oof_cache.save_oof_cache(self.root, artifact)
        self.assertEqual(first, second)
        self.assertEqual(
            len(list((self.root / "_ensemble_oof").iterdir())), 1
        )

    def test_concurrent_same_key_miss_generates_once(self):
        calls = 0
        count_lock = Lock()

        def factory():
            nonlocal calls
            with count_lock:
                calls += 1
            sleep(0.1)
            return _artifact()

        def get_or_create(_index: int):
            return oof_cache.get_or_create_oof_cache(
                self.root,
                "f" * 64,
                factory,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(get_or_create, range(2)))

        self.assertEqual(calls, 1)
        self.assertEqual(sorted(cache_hit for _, cache_hit in results), [False, True])
        for artifact, _cache_hit in results:
            self.assertEqual(artifact.oof_fingerprint, "f" * 64)

    def test_same_key_miss_generates_once_across_spawn_processes(self):
        context = multiprocessing.get_context("spawn")
        counter = context.Value("i", 0)
        first_attempting = context.Event()
        second_attempting = context.Event()
        entered_factory = context.Event()
        release_factory = context.Event()
        result_queue = context.Queue()
        first = context.Process(
            target=_get_or_create_oof_cache_in_process,
            args=(
                str(self.root),
                counter,
                first_attempting,
                entered_factory,
                release_factory,
                result_queue,
            ),
        )
        second = context.Process(
            target=_get_or_create_oof_cache_in_process,
            args=(
                str(self.root),
                counter,
                second_attempting,
                entered_factory,
                release_factory,
                result_queue,
            ),
        )
        first.start()
        try:
            self.assertTrue(first_attempting.wait(timeout=5.0))
            self.assertTrue(entered_factory.wait(timeout=5.0))
            second.start()
            self.assertTrue(second_attempting.wait(timeout=5.0))
            sleep(0.2)
            self.assertEqual(counter.value, 1)
            release_factory.set()
        finally:
            release_factory.set()
            first.join(timeout=10.0)
            if first.is_alive():
                first.terminate()
                first.join(timeout=5.0)
            if second.pid is not None:
                second.join(timeout=10.0)
                if second.is_alive():
                    second.terminate()
                    second.join(timeout=5.0)

        self.assertEqual(first.exitcode, 0)
        self.assertEqual(second.exitcode, 0)
        self.assertEqual(counter.value, 1)
        self.assertEqual(
            sorted((result_queue.get(timeout=2.0), result_queue.get(timeout=2.0))),
            [False, True],
        )

    def test_missing_cache_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            oof_cache.load_oof_cache(self.root, "0" * 64)

    def test_corrupt_metadata_raises(self):
        artifact = _artifact()
        oof_cache.save_oof_cache(self.root, artifact)
        directory = oof_cache.cache_dir(self.root, artifact.oof_fingerprint)
        (directory / oof_cache.OOF_METADATA_FILE).write_text("{broken", "utf-8")
        with self.assertRaises(ValueError):
            oof_cache.load_oof_cache(self.root, artifact.oof_fingerprint)

    def test_hash_mismatch_raises(self):
        artifact = _artifact()
        oof_cache.save_oof_cache(self.root, artifact)
        directory = oof_cache.cache_dir(self.root, artifact.oof_fingerprint)
        frame = pd.read_csv(
            directory / oof_cache.OOF_PREDICTIONS_FILE, header=[0, 1]
        )
        frame.iloc[0, 2] = 999.0  # corrupt one value
        frame.to_csv(directory / oof_cache.OOF_PREDICTIONS_FILE, index=False)
        with self.assertRaises(ValueError):
            oof_cache.load_oof_cache(self.root, artifact.oof_fingerprint)

    def test_incomplete_cache_raises(self):
        artifact = _artifact()
        directory = oof_cache.cache_dir(self.root, artifact.oof_fingerprint)
        directory.mkdir(parents=True)
        (directory / oof_cache.OOF_PREDICTIONS_FILE).write_text("member,v0\n", "utf-8")
        factory_called = False

        def factory():
            nonlocal factory_called
            factory_called = True
            return artifact

        with self.assertRaises(ValueError):
            oof_cache.get_or_create_oof_cache(
                self.root,
                artifact.oof_fingerprint,
                factory,
            )
        self.assertFalse(factory_called)

    def test_source_hash_changes_fingerprint(self):
        fp1 = oof_cache.compute_oof_fingerprint(
            members={"a": "sha-a"},
            ensemble_payload={"freq": "1h"},
            oof_payload={"fold_count": 2},
            source_hashes={"data.csv": "deadbeef"},
        )
        fp2 = oof_cache.compute_oof_fingerprint(
            members={"a": "sha-a"},
            ensemble_payload={"freq": "1h"},
            oof_payload={"fold_count": 2},
            source_hashes={"data.csv": "deadbeef0"},  # content changed
        )
        self.assertNotEqual(fp1, fp2)

    def test_oof_semantic_inputs_change_fingerprint(self):
        baseline = oof_cache.compute_oof_fingerprint(
            members={"a": "sha-a", "b": "sha-b"},
            ensemble_payload={
                "member_order": ["a", "b"],
                "problem": {"horizon": 2, "targets": ["load"]},
                "probabilistic": {"mode": "point"},
            },
            oof_payload={"fold_count": 2, "stride_steps": 1},
            source_hashes={"a:data:history_path": "deadbeef"},
        )
        variants = (
            {
                "members": {"a": "sha-a-changed", "b": "sha-b"},
            },
            {
                "ensemble_payload": {
                    "member_order": ["b", "a"],
                    "problem": {"horizon": 2, "targets": ["load"]},
                    "probabilistic": {"mode": "point"},
                },
            },
            {
                "ensemble_payload": {
                    "member_order": ["a", "b"],
                    "problem": {"horizon": 3, "targets": ["load"]},
                    "probabilistic": {"mode": "point"},
                },
            },
            {
                "ensemble_payload": {
                    "member_order": ["a", "b"],
                    "problem": {"horizon": 2, "targets": ["load"]},
                    "probabilistic": {
                        "mode": "quantile",
                        "quantiles": [0.1, 0.5, 0.9],
                    },
                },
            },
            {
                "oof_payload": {"fold_count": 3, "stride_steps": 1},
            },
        )
        defaults = {
            "members": {"a": "sha-a", "b": "sha-b"},
            "ensemble_payload": {
                "member_order": ["a", "b"],
                "problem": {"horizon": 2, "targets": ["load"]},
                "probabilistic": {"mode": "point"},
            },
            "oof_payload": {"fold_count": 2, "stride_steps": 1},
            "source_hashes": {"a:data:history_path": "deadbeef"},
        }
        for index, changed in enumerate(variants):
            with self.subTest(index=index):
                self.assertNotEqual(
                    baseline,
                    oof_cache.compute_oof_fingerprint(
                        **{**defaults, **changed},
                    ),
                )

    def test_cache_module_does_not_import_posix_lock_unconditionally(self):
        cache_module = Path(__file__).parents[1] / "model_ensemble/cache.py"
        module = ast.parse(cache_module.read_text(encoding="utf-8"))
        self.assertFalse(
            any(
                (
                    isinstance(statement, ast.ImportFrom)
                    and statement.module == "fcntl"
                )
                or (
                    isinstance(statement, ast.Import)
                    and any(alias.name == "fcntl" for alias in statement.names)
                )
                for statement in module.body
            ),
            "OOF cache module must remain importable on platforms without fcntl",
        )

    def test_file_sha256_stable(self):
        path = self.root / "f.bin"
        path.write_bytes(b"payload")
        self.assertEqual(oof_cache.file_sha256(path), oof_cache.file_sha256(path))


if __name__ == "__main__":
    unittest.main()
