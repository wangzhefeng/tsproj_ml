"""旧训练窗键必须拒绝；OOF只调度折，训练窗由成员合同决定。"""
import unittest

from model_ensemble.specs import EnsembleSpecError, parse_oof_spec
from forecasting_core.specs.config import parse_model_config
from tests.test_ensemble_runtime import _member_doc


class RetiredTemporalContractTest(unittest.TestCase):
    def test_single_model_rejects_legacy_window_without_new_contract(self):
        for raw_history in (False, True):
            with self.subTest(raw_history=raw_history):
                document = _member_doc('direct', 'ridge', 'retired')
                document['validation'].pop('training_window', None)
                document['validation']['train_window_steps'] = 16
                if raw_history:
                    document['validation']['train_history_steps'] = 20
                with self.assertRaisesRegex(ValueError, 'Unknown fields'):
                    parse_model_config(document, source='retired-test')

    def test_oof_uses_member_window_without_independent_sample_cap(self):
        spec = parse_oof_spec({'fold_count': 2, 'stride_steps': 24, 'gap_steps': 1}, calendar_month=False)
        self.assertEqual(spec.fold_count, 2)
        self.assertNotIn('train_window_steps', spec.payload())

    def test_oof_rejects_retired_window_key(self):
        with self.assertRaisesRegex(EnsembleSpecError, 'unknown fields'):
            parse_oof_spec({'fold_count': 2, 'stride_steps': 24, 'train_window_steps': 30}, calendar_month=False)
