# -*- coding: utf-8 -*-
"""包间分层 DAG 门禁（AGENTS.md「包间分层规则」）。

扫描所有活动包的顶层与函数内 import，强制：

1. 每个包只能依赖显式白名单中的下游包；
2. 包级依赖图不存在强连通分量；
3. 不允许用函数内 project import 绕过循环；
4. ``model_ensemble`` 只能通过 Protocol/注入获取单模型 runner，
   对 ``model_forecasting`` 的依赖限于稳定结果写入接口。

稳定合同位于 ``forecasting_core/``；``model_forecasting/`` 仅负责运行编排。
"""

import ast
from importlib.util import resolve_name
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

PROJECT_PACKAGES = {
    "forecasting_core",
    "model_forecasting",
    "model_ensemble",
    "probabilistic",
    "models",
    "decomposition",
    "data_process",
    "utils",
    "data_loading",
    "feature_engineering",
    "model_training",
    "model_testing",
    "model_evaluation",
}

# 包级允许的项目内依赖（自身包隐含允许）。该表本身即目标 DAG。
ALLOWED_PACKAGES = {
    "utils": set(),
    "forecasting_core": set(),
    "data_process": {"decomposition"},
    "decomposition": set(),
    "models": {"utils"},
    "data_loading": {"forecasting_core"},
    "feature_engineering": {
        "forecasting_core",
        "data_loading",
        "decomposition",
    },
    "model_training": {"forecasting_core", "models"},
    "model_testing": {"forecasting_core"},
    "model_evaluation": {"forecasting_core"},
    "model_forecasting": {
        "data_loading",
        "decomposition",
        "feature_engineering",
        "forecasting_core",
        "model_evaluation",
        "model_testing",
        "model_training",
        "models",
        "probabilistic",
        "utils",
    },
    "probabilistic": {"forecasting_core", "model_training"},
    "model_ensemble": {
        "data_loading",
        "forecasting_core",
        "model_evaluation",
        "model_forecasting",
        "model_testing",
        "utils",
    },
}

# 只对具有架构意义的窄接口做根级约束。
ALLOWED_ROOTS = {
    "model_ensemble": {
        "data_loading",
        "forecasting_core.artifacts",
        "forecasting_core.specs",
        "forecasting_core.tensors",
        "model_evaluation",
        "model_forecasting.deployment",
        "model_forecasting.results",
        "model_testing.backtest",
        "model_testing.validation",  # 共享标签安全合同，不暴露 runner 执行面
        "utils",
    },
}


def _iter_imports(tree, package=""):
    importlib_names = {"importlib"}
    loaders = {"__import__"}
    constants = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            importlib_names.update(alias.asname or alias.name for alias in node.names if alias.name == "importlib")
        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            loaders.update(alias.asname or alias.name for alias in node.names if alias.name == "import_module")
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = node.value.value
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "." * node.level + module
                module = resolve_name(module, package) if package else module
            yield module, node.lineno
        elif isinstance(node, ast.Call):
            function = node.func
            dynamic = (
                isinstance(function, ast.Name) and function.id in loaders
                or isinstance(function, ast.Attribute) and function.attr == "import_module"
                and isinstance(function.value, ast.Name) and function.value.id in importlib_names
            )
            if dynamic:
                argument = node.args[0] if node.args else next((item.value for item in node.keywords if item.arg in {"name", "module"}), None)
                value = argument.value if isinstance(argument, ast.Constant) else constants.get(argument.id) if isinstance(argument, ast.Name) else None
                yield value if isinstance(value, str) else "<unresolved_dynamic_import>", node.lineno


def _package_edges() -> dict[str, set[str]]:
    edges = {package: set() for package in PROJECT_PACKAGES}
    for package in PROJECT_PACKAGES:
        package_dir = ROOT / package
        if not package_dir.is_dir():
            continue
        for path in package_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for module, _ in _iter_imports(tree, ".".join(path.parent.relative_to(ROOT).parts)):
                target = module.split(".")[0].lstrip(".")
                if target in PROJECT_PACKAGES and target != package:
                    edges[package].add(target)
    return edges


def _find_cycle(edges: dict[str, set[str]]) -> list[str]:
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> list[str]:
        if node in visiting:
            start = visiting.index(node)
            return visiting[start:] + [node]
        if node in visited:
            return []
        visiting.append(node)
        for target in sorted(edges[node]):
            cycle = visit(target)
            if cycle:
                return cycle
        visiting.pop()
        visited.add(node)
        return []

    for package in sorted(edges):
        cycle = visit(package)
        if cycle:
            return cycle
    return []


def _function_local_project_imports() -> list[str]:
    found: list[str] = []
    for package in sorted(PROJECT_PACKAGES):
        package_dir = ROOT / package
        if not package_dir.is_dir():
            continue
        for path in sorted(package_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            relative = path.relative_to(ROOT)
            imports = tuple(_iter_imports(tree, ".".join(path.parent.relative_to(ROOT).parts)))
            for function in ast.walk(tree):
                if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for module, lineno in imports:
                    if function.lineno <= lineno <= function.end_lineno and module.split(".")[0] in PROJECT_PACKAGES:
                        found.append(f"{relative}:{lineno} imports {module}")
    return sorted(set(found))


class InterPackageLayeringTest(unittest.TestCase):
    def test_relative_imports_resolve_against_the_owning_package(self):
        self.assertEqual(list(_iter_imports(ast.parse("from ..trainer import Trainer"), "model_training.estimators")), [("model_training.trainer", 1)])

    def test_dynamic_aliases_and_constants_are_visible(self):
        tree = ast.parse('import importlib as il\nfrom importlib import import_module as load\nTARGET = "models.ModelFactory"\nil.import_module(TARGET)\nload("model_forecasting.runtime")\n__import__("forecasting_core.specs")\n')
        modules = {module for module, _ in _iter_imports(tree)}
        self.assertTrue({"models.ModelFactory", "model_forecasting.runtime", "forecasting_core.specs"} <= modules)

    def _violations(self, pkg: str) -> list[str]:
        pkg_dir = ROOT / pkg
        if not pkg_dir.is_dir():
            return []
        found: list[str] = []
        for py in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            rel = str(py.relative_to(ROOT))
            for module, lineno in _iter_imports(tree, ".".join(py.parent.relative_to(ROOT).parts)):
                if module == "<unresolved_dynamic_import>":
                    found.append(f"{rel}:{lineno} unresolved dynamic import")
                    continue
                top = module.split(".")[0].lstrip(".")
                if top == pkg or top not in PROJECT_PACKAGES:
                    continue
                if top not in ALLOWED_PACKAGES.get(pkg, set()):
                    found.append(f"{rel}:{lineno} imports {module} (package not allowed)")
                    continue
                roots = ALLOWED_ROOTS.get(pkg)
                if roots is None:
                    continue
                root = ".".join(module.split(".")[:2]) if module.count(".") else module
                if top in {r.split(".")[0] for r in roots} and root not in roots and module not in roots:
                    # 包级允许但根级未列出
                    if not any(module == r or module.startswith(r + ".") or root == r for r in roots):
                        found.append(f"{rel}:{lineno} imports {module} (root not allowed)")
        return found

    def test_utils_has_no_project_imports(self):
        self.assertEqual(self._violations("utils"), [])

    def test_forecasting_core_has_no_project_imports(self):
        self.assertEqual(self._violations("forecasting_core"), [])

    def test_data_process_stays_infra(self):
        self.assertEqual(self._violations("data_process"), [])

    def test_decomposition_stays_infra(self):
        self.assertEqual(self._violations("decomposition"), [])

    def test_models_stays_infra(self):
        self.assertEqual(self._violations("models"), [])

    def test_data_loading_stage(self):
        self.assertEqual(self._violations("data_loading"), [])

    def test_feature_engineering_stage(self):
        self.assertEqual(self._violations("feature_engineering"), [])

    def test_training_stage(self):
        self.assertEqual(self._violations("model_training"), [])

    def test_testing_stage(self):
        self.assertEqual(self._violations("model_testing"), [])

    def test_evaluation_stage(self):
        self.assertEqual(self._violations("model_evaluation"), [])

    def test_forecasting_orchestration(self):
        self.assertEqual(self._violations("model_forecasting"), [])

    def test_probabilistic_capability(self):
        self.assertEqual(self._violations("probabilistic"), [])

    def test_ensemble_stays_clean(self):
        self.assertEqual(self._violations("model_ensemble"), [])

    def test_ensemble_time_contract_does_not_allow_runtime_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "model_ensemble"
            package.mkdir()
            (package / "probe.py").write_text(
                "from model_testing.validation import is_label_safe\n"
                "from model_forecasting.runtime import CanonicalBaseModelRunner\n",
                encoding="utf-8",
            )
            with patch.object(sys.modules[__name__], "ROOT", root):
                violations = self._violations("model_ensemble")
            self.assertEqual(len(violations), 1)
            self.assertIn("model_forecasting.runtime", violations[0])

    def test_package_graph_is_acyclic(self):
        self.assertEqual(_find_cycle(_package_edges()), [])

    def test_no_function_local_project_imports(self):
        self.assertEqual(_function_local_project_imports(), [])

    def test_ensemble_runner_is_injected(self):
        runtime = ROOT / "model_ensemble/runtime.py"
        tree = ast.parse(runtime.read_text(encoding="utf-8"))
        imports = [module for module, _ in _iter_imports(tree)]
        self.assertNotIn("model_forecasting.runtime", imports)

    def test_forecasting_runtime_delegates_design_fit_and_calendar_backtest(self):
        """C4：runtime 只编排，不再内嵌 design/fit/calendar-backtest 实现。"""
        expected_modules = {
            "design.py",
            "fit_service.py",
            "backtest_runtime.py",
            "persistence.py",
        }
        self.assertTrue(
            expected_modules.issubset(
                {path.name for path in (ROOT / "model_forecasting").glob("*.py")}
            )
        )
        runtime_tree = ast.parse(
            (ROOT / "model_forecasting" / "runtime.py").read_text(encoding="utf-8")
        )
        owned = {
            node.name
            for node in runtime_tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.assertTrue(
            {
                "_RegistryDesignBuilder",
                "_fit_runtime_transforms",
                "_overwrite_calendar_month_backtest",
            }.isdisjoint(owned)
        )


if __name__ == "__main__":
    unittest.main()
