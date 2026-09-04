# -*- coding: utf-8 -*-
"""E2: ensemble loader — reference resolution and shared contract checks."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import yaml

from forecasting_core.specs import (
    DataSpec,
    ForecastProblemSpec,
    OutputSpec,
    ProbabilisticConfigSpec,
    RuntimeValidationSpec,
)
from model_ensemble.loader import (
    load_ensemble_config,
    parse_ensemble_document,
    resolve_members,
    validate_member_sources,
)
from model_ensemble.specs import EnsembleSpecError


SINGLE_MODEL = {
    "schema_version": 2,
    "problem": {
        "time_col": "time",
        "freq": "1h",
        "horizon": 2,
        "targets": ["load"],
        "training_scope": "local",
        "series_id_cols": [],
    },
    "data": {
        "sources": [
            {
                "name": "targets",
                "source_type": "file",
                "columns": [{"name": "load", "role": "target", "categorical": False}],
                "history_path": "data.csv",
                "time_col": "time",
                "series_id_cols": [],
                "availability": "source_time",
            }
        ]
    },
    "features": {
        "target_lags": {"load": [2, 3]},
        "observed_past_lags": {},
        "datetime_features": [],
        "transformations": {},
    },
    "strategy": {"name": "recursive"},
    "estimator": {"model_type": "ridge", "target_adapter": "independent", "params": {}},
    "probabilistic": {"mode": "point"},
    "validation": {"forecast_origin": "2026-01-03T23:00:00"},
    "output": {"scenario_subpath": "member"},
}

ENSEMBLE_DOC = {
    "schema_version": 2,
    "problem": SINGLE_MODEL["problem"],
    "data": SINGLE_MODEL["data"],
    "probabilistic": {"mode": "point"},
    "ensemble": {
        "members": [
            {"name": "direct", "config_ref": "direct.yaml"},
            {"name": "recursive", "config_ref": "recursive.yaml"},
        ],
        "oof": {"train_window_steps": 8, "fold_count": 2, "stride_steps": 1},
        "method": {"name": "averaging"},
    },
    "validation": {"forecast_origin": "2026-01-03T23:00:00"},
    "output": {"scenario_subpath": "ens"},
}


def _variant(strategy: str, subpath: str) -> dict:
    import copy

    doc = copy.deepcopy(SINGLE_MODEL)
    doc["strategy"] = {"name": strategy}
    doc["output"] = {"scenario_subpath": subpath}
    return doc


class EnsembleLoaderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._write("direct.yaml", _variant("direct", "direct"))
        self._write("recursive.yaml", _variant("recursive", "recursive"))
        self._write("ens.yaml", ENSEMBLE_DOC)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, doc: dict):
        (self.root / name).write_text(yaml.safe_dump(doc), encoding="utf-8")

    def test_parse_and_resolve_ok(self):
        config = load_ensemble_config(self.root / "ens.yaml")
        self.assertEqual(config.method.name, "averaging")
        self.assertIsInstance(config.problem, ForecastProblemSpec)
        self.assertIsInstance(config.data, DataSpec)
        self.assertIsInstance(config.probabilistic, ProbabilisticConfigSpec)
        self.assertIsInstance(config.validation, RuntimeValidationSpec)
        self.assertIsInstance(config.output, OutputSpec)
        self.assertEqual(
            [m.name for m in config.members], ["direct", "recursive"]
        )
        members = resolve_members(config, base_dir=self.root)
        self.assertEqual(set(members), {"direct", "recursive"})
        validate_member_sources(config, members)

    def test_missing_ref_raises(self):
        doc = dict(ENSEMBLE_DOC)
        doc["ensemble"] = {
            "members": [
                {"name": "direct", "config_ref": "direct.yaml"},
                {"name": "recursive", "config_ref": "missing.yaml"},
            ],
            "oof": ENSEMBLE_DOC["ensemble"]["oof"],
            "method": ENSEMBLE_DOC["ensemble"]["method"],
        }
        self._write("bad.yaml", doc)
        config = load_ensemble_config(self.root / "bad.yaml")
        with self.assertRaises(EnsembleSpecError):
            resolve_members(config, base_dir=self.root)

    def test_ensemble_of_ensemble_raises(self):
        ens_variant = dict(ENSEMBLE_DOC)
        ens_variant["strategy"] = {"name": "recursive"}  # fake a single-model doc
        self._write("nested_inner.yaml", ens_variant)
        doc = dict(ENSEMBLE_DOC)
        doc["ensemble"] = {
            "members": [
                {"name": "direct", "config_ref": "direct.yaml"},
                {"name": "recursive", "config_ref": "nested_inner.yaml"},
            ],
            "oof": ENSEMBLE_DOC["ensemble"]["oof"],
            "method": ENSEMBLE_DOC["ensemble"]["method"],
        }
        self._write("bad2.yaml", doc)
        config = load_ensemble_config(self.root / "bad2.yaml")
        with self.assertRaises(EnsembleSpecError):
            resolve_members(config, base_dir=self.root)

    def test_problem_mismatch_raises(self):
        doc = _variant("recursive", "recursive")
        doc["problem"] = dict(doc["problem"], horizon=3)
        self._write("other_horizon.yaml", doc)
        members_doc = dict(ENSEMBLE_DOC)
        members_doc["ensemble"] = {
            "members": [
                {"name": "direct", "config_ref": "direct.yaml"},
                {"name": "recursive", "config_ref": "other_horizon.yaml"},
            ],
            "oof": ENSEMBLE_DOC["ensemble"]["oof"],
            "method": ENSEMBLE_DOC["ensemble"]["method"],
        }
        self._write("mismatch.yaml", members_doc)
        config = load_ensemble_config(self.root / "mismatch.yaml")
        with self.assertRaises(EnsembleSpecError):
            resolve_members(config, base_dir=self.root)

    def test_top_level_problem_mismatch_raises(self):
        doc = dict(ENSEMBLE_DOC)
        doc["problem"] = dict(doc["problem"], horizon=99)
        self._write("top_mismatch.yaml", doc)
        config = load_ensemble_config(self.root / "top_mismatch.yaml")
        with self.assertRaises(EnsembleSpecError):
            resolve_members(config, base_dir=self.root)

    def test_public_problem_rejects_information_mode(self):
        import copy

        doc = copy.deepcopy(ENSEMBLE_DOC)
        doc["problem"]["information_mode"] = "forecast"

        with self.assertRaisesRegex(ValueError, "Unknown fields in problem"):
            parse_ensemble_document(doc, source_path="inline.yaml")

    def test_output_changes_do_not_change_ensemble_fingerprint(self):
        import copy

        first_doc = copy.deepcopy(ENSEMBLE_DOC)
        second_doc = copy.deepcopy(ENSEMBLE_DOC)
        second_doc["output"] = {"scenario_subpath": "other-output"}

        first = parse_ensemble_document(first_doc, source_path="first.yaml")
        second = parse_ensemble_document(second_doc, source_path="second.yaml")

        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_performance_changes_do_not_change_ensemble_fingerprint(self):
        import copy

        first_doc = copy.deepcopy(ENSEMBLE_DOC)
        second_doc = copy.deepcopy(ENSEMBLE_DOC)
        first_doc["validation"]["performance"] = {"total_thread_limit": 2}
        second_doc["validation"]["performance"] = {"total_thread_limit": 8}

        first = parse_ensemble_document(first_doc, source_path="first.yaml")
        second = parse_ensemble_document(second_doc, source_path="second.yaml")

        self.assertEqual(first.fingerprint(), second.fingerprint())
        self.assertEqual(first.result_identity(), second.result_identity())

    def test_cycle_self_reference_raises(self):
        doc = dict(ENSEMBLE_DOC)
        doc["ensemble"] = {
            "members": [
                {"name": "direct", "config_ref": "direct.yaml"},
                {"name": "recursive", "config_ref": "selfens.yaml"},
            ],
            "oof": ENSEMBLE_DOC["ensemble"]["oof"],
            "method": ENSEMBLE_DOC["ensemble"]["method"],
        }
        self._write("selfens.yaml", doc)
        config = load_ensemble_config(self.root / "selfens.yaml")
        with self.assertRaises(EnsembleSpecError):
            resolve_members(config, base_dir=self.root)

    def test_quantile_grid_mismatch_raises(self):
        import copy

        rq = _variant("recursive", "recursive")
        rq["probabilistic"] = {
            "mode": "quantile",
            "quantiles": [0.1, 0.5, 0.9],
            "point_quantile": 0.5,
        }
        self._write("rq.yaml", rq)
        doc = copy.deepcopy(ENSEMBLE_DOC)
        doc["probabilistic"] = rq["probabilistic"]
        doc["ensemble"] = {
            "members": [
                {"name": "direct", "config_ref": "direct.yaml"},
                {"name": "recursive", "config_ref": "rq.yaml"},
            ],
            "oof": ENSEMBLE_DOC["ensemble"]["oof"],
            "method": {"name": "averaging"},
        }
        self._write("mixq.yaml", doc)
        config = load_ensemble_config(self.root / "mixq.yaml")
        with self.assertRaises(EnsembleSpecError):
            resolve_members(config, base_dir=self.root)

    def test_unknown_top_level_field_raises(self):
        doc = dict(ENSEMBLE_DOC)
        doc["unexpected"] = 1
        with self.assertRaises(EnsembleSpecError):
            parse_ensemble_document(doc)

    def test_unknown_validation_field_raises_before_return(self):
        import copy

        doc = copy.deepcopy(ENSEMBLE_DOC)
        doc["validation"]["totally_unknown_field"] = 1
        with self.assertRaisesRegex(ValueError, "validation"):
            parse_ensemble_document(doc, source_path="ensemble.yaml")

    def test_unknown_probabilistic_field_raises_before_return(self):
        import copy

        doc = copy.deepcopy(ENSEMBLE_DOC)
        doc["probabilistic"]["totally_unknown_field"] = 1
        with self.assertRaisesRegex(ValueError, "probabilistic"):
            parse_ensemble_document(doc, source_path="ensemble.yaml")

    def test_unknown_output_field_raises_before_return(self):
        import copy

        doc = copy.deepcopy(ENSEMBLE_DOC)
        doc["output"]["totally_unknown_field"] = 1
        with self.assertRaisesRegex(ValueError, "output"):
            parse_ensemble_document(doc, source_path="ensemble.yaml")


if __name__ == "__main__":
    unittest.main()
