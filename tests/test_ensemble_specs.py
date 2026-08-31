# -*- coding: utf-8 -*-
"""E2: ensemble spec parsing, forbidden-field contracts, and loader checks."""

from __future__ import annotations

import unittest

from model_ensemble.specs import (
    ENSEMBLE_FORBIDDEN_TOP_LEVEL,
    EnsembleSpecError,
    MemberRef,
    MethodSpec,
    OOFSpec,
    enforce_forbidden_top_level,
    parse_ensemble_section,
    parse_oof_spec,
)


def _members_payload():
    return [
        {"name": "direct", "config_ref": "a.yaml"},
        {"name": "recursive", "config_ref": "b.yaml"},
    ]


class MemberRefSpecTest(unittest.TestCase):
    def test_rejects_blank_names(self):
        with self.assertRaises(EnsembleSpecError):
            MemberRef(" ", "a.yaml")
        with self.assertRaises(EnsembleSpecError):
            MemberRef("direct", "")

    def test_rejects_unknown_member_fields(self):
        with self.assertRaises(EnsembleSpecError):
            parse_ensemble_section(
                {
                    "members": [
                        {"name": "a", "config_ref": "a.yaml", "extra": 1},
                        {"name": "b", "config_ref": "b.yaml"},
                    ],
                    "oof": {"train_window_steps": 4, "fold_count": 2, "stride_steps": 1},
                    "method": "averaging",
                },
                calendar_month=False,
            )


class OOFSpecTest(unittest.TestCase):
    def test_valid_oof_parses(self):
        oof = OOFSpec(train_window_steps=8, fold_count=3, stride_steps=2, gap_steps=1)
        self.assertEqual(oof.fold_count, 3)
        self.assertEqual(oof.gap_steps, 1)

    def test_unknown_oof_field_raises_at_parse(self):
        with self.assertRaises(EnsembleSpecError):
            parse_oof_spec(
                {
                    "train_window_steps": 8,
                    "fold_count": 3,
                    "stride_steps": 2,
                    "extra": 1,
                },
                calendar_month=False,
            )

    def test_non_positive_values_raise(self):
        with self.assertRaises(EnsembleSpecError):
            OOFSpec(train_window_steps=0, fold_count=2, stride_steps=1)
        with self.assertRaises(EnsembleSpecError):
            OOFSpec(train_window_steps=4, fold_count=0, stride_steps=1)
        with self.assertRaises(EnsembleSpecError):
            OOFSpec(train_window_steps=4, fold_count=2, stride_steps=-1)
        with self.assertRaises(EnsembleSpecError):
            OOFSpec(train_window_steps=4, fold_count=2, stride_steps=1, gap_steps=-1)

    def test_calendar_month_multi_fold_raises(self):
        with self.assertRaises(EnsembleSpecError):
            OOFSpec(
                train_window_steps=8, fold_count=2, stride_steps=1, calendar_month=True
            )

    def test_calendar_month_single_fold_allowed(self):
        oof = OOFSpec(
            train_window_steps=8, fold_count=1, stride_steps=1, calendar_month=True
        )
        self.assertEqual(oof.fold_count, 1)


class MethodSpecTest(unittest.TestCase):
    def test_unknown_method_raises(self):
        with self.assertRaises(EnsembleSpecError):
            MethodSpec("blending", {})

    def test_all_four_methods_accepted(self):
        for name in ("averaging", "weighted", "linear_blending", "stacking"):
            MethodSpec(name, {})

    def test_string_form_parses(self):
        method = parse_ensemble_section(
            {
                "members": _members_payload(),
                "oof": {"train_window_steps": 4, "fold_count": 2, "stride_steps": 1},
                "method": "averaging",
            },
            calendar_month=False,
        )[2]
        self.assertEqual(method.name, "averaging")
        self.assertEqual(method.params, {})


class ForbiddenTopLevelTest(unittest.TestCase):
    def test_each_forbidden_field_raises(self):
        for field in sorted(ENSEMBLE_FORBIDDEN_TOP_LEVEL):
            with self.assertRaises(EnsembleSpecError):
                enforce_forbidden_top_level({field: None})

    def test_strategy_null_raises(self):
        with self.assertRaises(EnsembleSpecError):
            enforce_forbidden_top_level({"strategy": None})

    def test_clean_doc_passes(self):
        enforce_forbidden_top_level(
            {
                "schema_version": 2,
                "problem": {},
                "data": {},
                "probabilistic": {},
                "ensemble": {},
                "validation": {},
                "output": {},
            }
        )


class EnsembleSectionTest(unittest.TestCase):
    def test_duplicate_member_names_raise(self):
        with self.assertRaises(EnsembleSpecError):
            parse_ensemble_section(
                {
                    "members": [
                        {"name": "same", "config_ref": "a.yaml"},
                        {"name": "same", "config_ref": "b.yaml"},
                    ],
                    "oof": {"train_window_steps": 4, "fold_count": 2, "stride_steps": 1},
                    "method": "averaging",
                },
                calendar_month=False,
            )

    def test_duplicate_config_refs_raise(self):
        with self.assertRaises(EnsembleSpecError):
            parse_ensemble_section(
                {
                    "members": [
                        {"name": "a", "config_ref": "same.yaml"},
                        {"name": "b", "config_ref": "same.yaml"},
                    ],
                    "oof": {"train_window_steps": 4, "fold_count": 2, "stride_steps": 1},
                    "method": "averaging",
                },
                calendar_month=False,
            )

    def test_single_member_raises(self):
        with self.assertRaises(EnsembleSpecError):
            parse_ensemble_section(
                {
                    "members": [{"name": "a", "config_ref": "a.yaml"}],
                    "oof": {"train_window_steps": 4, "fold_count": 2, "stride_steps": 1},
                    "method": "averaging",
                },
                calendar_month=False,
            )


if __name__ == "__main__":
    unittest.main()
