"""融合子包结构及依赖方向；不通过移动持久化类型破坏bundle路径。"""
import ast
from pathlib import Path
import unittest

from model_ensemble import artifacts

ROOT = Path(__file__).resolve().parents[1] / "model_ensemble"
PACKAGES = {
    "configuration": {"specs", "loader", "preflight"},
    "methods": {"averaging", "weighted", "linear_blending", "stacking", "adaptive_weighted", "horizon"},
    "training": {"oof", "trainer", "backtesting", "diagnostics"},
    "inference": {"predictor", "forecast", "deployment"},
    "outputs": {"cache", "persistence", "reporting"},
}
ALLOWED = {
    "configuration": {"configuration", "artifacts", "contracts"},
    "methods": {"methods", "artifacts"},
    "inference": {"inference", "methods", "artifacts", "contracts"},
    "training": {"training", "configuration", "inference", "methods", "artifacts", "contracts"},
    "outputs": {"outputs", "configuration", "inference", "artifacts", "contracts"},
}


def check_direction(source, owner):
    for node in ast.walk(ast.parse(source)):
        modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                   [item.name for item in node.names] if isinstance(node, ast.Import) else [])
        for module in modules:
            if module.startswith("model_ensemble."):
                dependency = module.split(".")[1]
                if dependency not in ALLOWED[owner]:
                    raise AssertionError(f"{owner} must not import {module}")


class EnsemblePackageStructureTest(unittest.TestCase):
    def test_root_and_subpackages_have_explicit_ownership(self):
        self.assertEqual({path.stem for path in ROOT.glob("*.py")}, {"__init__", "runtime", "contracts", "artifacts"})
        for package, modules in PACKAGES.items():
            self.assertEqual({p.stem for p in (ROOT / package).glob("*.py")}, modules | {"__init__"})
            if package != "methods":
                tree = ast.parse((ROOT / package / "__init__.py").read_text())
                self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)))

    def test_subpackage_imports_follow_declared_direction(self):
        for owner in PACKAGES:
            files = list((ROOT / owner).glob("*.py"))
            self.assertTrue(files, owner)
            for path in files:
                with self.subTest(path=path):
                    check_direction(path.read_text(), owner)

    def test_gate_rejects_inference_to_training_and_methods_to_io(self):
        for owner, source in (("inference", "from model_ensemble.training.trainer import fit_ensemble"),
                              ("methods", "import model_ensemble.outputs.cache")):
            with self.assertRaisesRegex(AssertionError, "must not import"):
                check_direction(source, owner)

    def test_persisted_types_keep_original_module(self):
        for name in ("EnsembleArtifact", "OOFPredictionArtifact", "EqualWeightsArtifact", "PerTargetWeightsArtifact",
                     "PerTargetMetaArtifact", "HorizonWeightsArtifact", "TemporalWeightsArtifact"):
            self.assertEqual(getattr(artifacts, name).__module__, "model_ensemble.artifacts")
