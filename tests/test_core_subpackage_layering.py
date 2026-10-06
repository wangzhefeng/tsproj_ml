"""Core 子包依赖方向与真实模块图门禁；含故意违规的负控制。"""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests import test_package_layering as layering

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    "tensors": {"tensors"},
    "execution": {"execution"},
    "temporal": {"temporal"},
    "probability": {"probability", "tensors"},
    "specs": {"specs", "probability", "temporal"},
    "bundle": {"probability"},
    "root": set(),
}


def core_dependencies(root):
    modules = {}
    for path in (root / "forecasting_core").rglob("*.py"):
        name = ".".join(path.relative_to(root).with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name.removesuffix(".__init__")
        modules[name] = path
    edges = {name: set() for name in modules}
    violations = []
    for name, path in modules.items():
        owner = name.split(".")[1] if "." in name else "root"
        if owner not in ALLOWED:
            violations.append(f"unclassified core module: {name}")
            continue
        tree = ast.parse(path.read_text())
        package = ".".join(path.parent.relative_to(root).parts)
        for module, lineno in layering._iter_imports(tree, package):
            if module == "<unresolved_dynamic_import>":
                violations.append(f"{name}:{lineno} unresolved dynamic import")
                continue
            if not module.startswith("forecasting_core"):
                continue  # 项目外依赖由现有包间门禁覆盖。
            target = module.split(".")[1] if "." in module else "root"
            if target not in ALLOWED[owner]:
                violations.append(f"{name}:{lineno} forbidden core dependency {module}")
            if module in modules:
                edges[name].add(module)
        # `from package import child` 也形成真实模块边，不能漏掉环。
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if node.level:
                module = layering.resolve_name("." * node.level + module, package)
            for alias in node.names:
                child = module + "." + alias.name
                if child in modules:
                    edges[name].add(child)
    cycle = layering._find_cycle(edges)
    if cycle:
        violations.append("core module cycle: " + " -> ".join(cycle))
    return violations


class CoreSubpackageLayeringTest(unittest.TestCase):
    def test_core_contracts_follow_direction_and_are_acyclic(self):
        self.assertEqual(core_dependencies(ROOT), [])

    def test_guard_detects_reversed_dependency_and_same_package_cycle(self):
        variants = (
            ({"tensors/probe.py": "from forecasting_core.specs.problem import ForecastProblemSpec\n",
              "specs/problem.py": ""}, "forbidden core dependency"),
            ({"temporal/a.py": "from .b import value\n",
              "temporal/b.py": "from .a import value\n"}, "core module cycle"),
        )
        for files, expected in variants:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, text in files.items():
                    path = root / "forecasting_core" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(text)
                result = unittest.TestResult()
                with patch.object(sys.modules[__name__], "ROOT", root):
                    type(self)("test_core_contracts_follow_direction_and_are_acyclic").run(result)
                self.assertEqual(result.errors, [])
                self.assertEqual(len(result.failures), 1)
                self.assertIn(expected, result.failures[0][1])
