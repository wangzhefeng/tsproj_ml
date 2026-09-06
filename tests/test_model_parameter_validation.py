"""估计器构造参数必须生效或显式报错，不能静默过滤。"""

import unittest

from models.factory import ModelFactory


class ModelParameterValidationTest(unittest.TestCase):
    def test_unknown_signature_parameter_is_rejected(self):
        for model_type in ("rf", "histgb", "ridge", "enet", "lasso", "qr"):
            with self.subTest(model_type=model_type):
                with self.assertRaisesRegex(ValueError, "unknown_parameter"):
                    ModelFactory().create_model(
                        model_type, {"unknown_parameter": 1}, log_params=False
                    )

    def test_valid_signature_parameter_is_preserved(self):
        for model_type, params in (
            ("rf", {"n_estimators": 7}),
            ("histgb", {"max_iter": 7}),
            ("ridge", {"alpha": 0.25}),
            ("enet", {"alpha": 0.25}),
            ("lasso", {"alpha": 0.25}),
            ("qr", {"alpha": 0.25}),
        ):
            with self.subTest(model_type=model_type):
                model = ModelFactory().create_model(model_type, params, log_params=False)
                for key, value in params.items():
                    self.assertEqual(model.get_params()[key], value)


if __name__ == "__main__":
    unittest.main()
