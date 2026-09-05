"""正 gap 语义修正产生新实验/缓存身份，零 gap 不扰动历史身份。"""

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy


from model_ensemble import cache
from model_ensemble.loader import parse_ensemble_document
from test_ensemble_loader import ENSEMBLE_DOC
from test_ensemble_oof import _artifact


class OOFGapIdentityTest(unittest.TestCase):
    def test_experiment_identity_changes_only_for_positive_gap(self):
        for gap in (0, 1, 3):
            with self.subTest(gap=gap):
                document = deepcopy(ENSEMBLE_DOC)
                document["ensemble"]["oof"]["gap_steps"] = gap
                config = parse_ensemble_document(document)
                legacy_payload = config.canonical_payload()
                legacy_payload.pop("output")
                legacy_payload["validation"] = config.validation.semantic_payload()
                legacy = hashlib.sha256(json.dumps(
                    legacy_payload, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")).hexdigest()
                if gap:
                    self.assertNotEqual(config.fingerprint(), legacy)
                    self.assertNotEqual(config.result_identity().split("-")[-1], legacy[:12])
                else:
                    self.assertEqual(config.fingerprint(), legacy)

    def test_old_positive_gap_cache_is_missed_without_deletion(self):
        for gap in (0, 1, 3):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as root:
                inputs = {
                    "members": {"a": "a-fp", "b": "b-fp"},
                    "ensemble_payload": {"horizon": 2},
                    "oof_payload": {"train_window_steps": 6, "fold_count": 2, "stride_steps": 1, "gap_steps": gap},
                    "source_hashes": {"a:targets:history": "source-hash"},
                }
                legacy_payload = {
                    "schema_version": cache.OOF_SCHEMA_VERSION,
                    "members": inputs["members"],
                    "ensemble": inputs["ensemble_payload"],
                    "oof": inputs["oof_payload"],
                    "source_hashes": inputs["source_hashes"],
                }
                legacy = hashlib.sha256(json.dumps(
                    legacy_payload, sort_keys=True, ensure_ascii=False,
                ).encode("utf-8")).hexdigest()
                old_path = cache.save_oof_cache(root, _artifact(legacy))
                new_key = cache.compute_oof_fingerprint(**inputs)
                calls = []

                def factory():
                    calls.append(new_key)
                    return _artifact(new_key)

                result, hit = cache.get_or_create_oof_cache(root, new_key, factory)
                self.assertEqual(hit, gap == 0)
                self.assertEqual(calls, [] if gap == 0 else [new_key])
                self.assertEqual(result.oof_fingerprint, new_key)
                self.assertTrue((old_path / cache.OOF_METADATA_FILE).is_file())
                self.assertEqual(cache.load_oof_cache(root, legacy).oof_fingerprint, legacy)
                _, second_hit = cache.get_or_create_oof_cache(root, new_key, factory)
                self.assertTrue(second_hit)


if __name__ == "__main__":
    unittest.main()
