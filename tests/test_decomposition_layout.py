"""职责子包布局与依赖方向门禁；不为内部组织引入兼容层。"""
import ast
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1] / "decomposition"
SUBPACKAGES = {
    "configuration", "construction", "contracts", "extraction", "extrapolation",
    "composition", "orchestration", "diagnostics",
}
ALLOWED = {
    "configuration": {"configuration"},
    "contracts": {"contracts"},
    "extraction": {"contracts", "extraction"},
    "extrapolation": {"contracts", "extrapolation"},
    "composition": {"contracts", "composition"},
    "diagnostics": {"contracts", "diagnostics"},
    "orchestration": {"contracts", "configuration", "construction"},
    "construction": {"contracts", "configuration", "extraction", "extrapolation", "composition", "orchestration"},
}


class DecompositionLayoutTest(unittest.TestCase):
    def test_root_has_only_public_entry(self):
        self.assertEqual({p.name for p in ROOT.glob("*.py")}, {"__init__.py"})

    def test_subpackages_have_lightweight_initializers_and_rules(self):
        for name in SUBPACKAGES:
            with self.subTest(name=name):
                directory = ROOT / name
                self.assertTrue((directory / "README.md").is_file())
                tree = ast.parse((directory / "__init__.py").read_text())
                self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)))

    def test_internal_imports_follow_responsibilities(self):
        for path in ROOT.rglob("*.py"):
            relative = path.relative_to(ROOT)
            if len(relative.parts) < 2:
                continue
            owner = relative.parts[0]
            for node in ast.walk(ast.parse(path.read_text())):
                modules = [node.module or ""] if isinstance(node, ast.ImportFrom) else (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [])
                for module in modules:
                    if not module.startswith("decomposition."):
                        continue
                    target = module.split(".")[1]
                    self.assertIn(target, ALLOWED[owner], f"{relative} imports {module}")
                    if relative.as_posix() == "construction/component_factory.py":
                        self.assertNotEqual(target, "orchestration", str(relative))
                    if owner == "orchestration":
                        self.assertNotEqual(module, "decomposition.construction.registry", str(relative))

    def test_public_and_direct_imports_work_in_fresh_process(self):
        subprocess.run([sys.executable, "-c", "from decomposition.construction.registry import build_pipeline; "
                        "from decomposition.orchestration.pipeline import DecompositionPipeline; "
                        "from decomposition import DecompositionSpec; "
                        "assert isinstance(build_pipeline(DecompositionSpec(method='none')), DecompositionPipeline)"],
                       cwd=ROOT.parent, check=True, capture_output=True, text=True)
