"""模型描述及接线的纯静态测试，不构造模型。"""
import ast
from pathlib import Path
import unittest
from models.catalog import MODEL_CATALOG, quantile_parameters

ROOT = Path(__file__).resolve().parents[1]


class ModelCatalogContractTest(unittest.TestCase):
    def test_all_aliases_resolve_to_existing_wrapper_classes(self):
        tree = ast.parse((ROOT / "models/ModelFactory.py").read_text())
        classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
        self.assertEqual(len(MODEL_CATALOG), 18)
        for descriptor in MODEL_CATALOG.values():
            self.assertIn(descriptor.wrapper, classes)
        for path in ("models/ModelFactory.py", "model_training/estimators/capabilities.py", "model_forecasting/resource_planner.py"):
            self.assertIn("from models.catalog import", (ROOT / path).read_text())

    def test_quantile_parameters_preserve_native_conventions(self):
        self.assertEqual(quantile_parameters("lgb", {"alpha": 0.1}, 0.9), {"objective": "quantile", "alpha": 0.9})
        self.assertEqual(quantile_parameters("cat", {}, 0.1), {"loss_function": "Quantile:alpha=0.1"})
        with self.assertRaises(ValueError):
            quantile_parameters("rf", {}, 0.5)


if __name__ == "__main__":
    unittest.main()
