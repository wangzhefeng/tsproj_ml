# -*- coding: utf-8 -*-
"""包间分层规则门禁 v2（AGENTS.md「包间分层规则」；2026-08-30 流水线阶段重排）。

层级（依赖只允许从上往下，ast.walk 全量扫描，函数内延迟 import 同罪）：

    L4  入口      main.py / run.py / config/config_loader.py（不受本门禁约束）
    L3  能力扩展  probabilistic/（概率执行面）、model_ensemble/（融合）
    L2  编排      model_forecasting/runtime.py（唯一编排器，豁免文件）
    L1.5 阶段包   evaluation / testing / training / feature_engineering /
                  data_loading（按流水线阶段单向依赖）
    L1  核心合同  model_forecasting/{specs,tensors,transforms,results}、models/、
                  decomposition/、data_process/
    L0  基础      utils/

白名单制：每个包只允许 import ALLOWED 中列出的项目内包/子模块根；
`model_forecasting/runtime.py` 是编排器，豁免阶段包规则（仍不得 import ensemble）。
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROJECT_PACKAGES = {
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

# 包级允许的项目内依赖（自身包隐含允许）
ALLOWED_PACKAGES = {
    "utils": set(),
    "data_process": {"utils"},
    "decomposition": {"utils", "data_process"},
    "models": {"utils"},
    "data_loading": {"utils", "model_forecasting"},
    "feature_engineering": {"utils", "model_forecasting", "data_loading"},
    "model_training": {"utils", "model_forecasting", "models", "probabilistic"},
    "model_testing": {"utils", "model_forecasting"},
    "model_evaluation": {"utils", "model_forecasting", "probabilistic"},
    "model_forecasting": {"utils", "model_forecasting", "probabilistic", "decomposition"},
    "probabilistic": {"utils", "model_forecasting", "probabilistic", "model_training", "model_evaluation"},
    "model_ensemble": {"utils", "model_forecasting", "probabilistic", "model_ensemble", "model_testing", "data_loading", "model_evaluation"},
}

# 子模块根级精化：ALLOWED_PACKAGES 放行包后，再按「允许的具体根」收敛；
# 未列出的包不做根级限制
ALLOWED_ROOTS = {
    # 数据层只读配置合同，不碰编排/张量
    "data_loading": {"model_forecasting.specs", "utils"},
    # 特征编译可读变换（特征缩放器）与数据层
    "feature_engineering": {"model_forecasting.specs", "model_forecasting.transforms", "model_forecasting.tensors", "data_loading", "utils"},
    # 训练可读模型工厂/序列化与概率合同（quantile 编排的既有交错，仅限 spec/types）
    "model_training": {"model_forecasting.specs", "model_forecasting.tensors", "models", "probabilistic.spec", "probabilistic.types", "utils"},
    "model_testing": {"model_forecasting.tensors", "utils"},
    "model_evaluation": {"model_forecasting.tensors", "probabilistic.types", "utils"},
    # forecasting 核心合同文件（runtime.py 豁免除外）不得触阶段包/model_ensemble/models；
    # results.py 可读写概率类型合同
    "model_forecasting": {"model_forecasting", "probabilistic.types", "decomposition", "utils"},
    # probabilistic 不得触 forecasting 编排面（runtime/results/data/features）
    "probabilistic": {"model_forecasting.specs", "model_forecasting.tensors", "model_forecasting.forecaster", "model_training", "model_evaluation", "probabilistic", "utils"},
}

# 编排器豁免：model_forecasting/runtime.py 可 import 全部阶段包 + models（L2 编排职责），
# 但仍不得 import ensemble
ORCHESTRATOR_FILES = {"model_forecasting/runtime.py", "model_forecasting/forecaster.py"}


def _iter_imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, node.lineno
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = "." * node.level + module
            yield module, node.lineno


class InterPackageLayeringTest(unittest.TestCase):
    def _violations(self, pkg: str) -> list[str]:
        pkg_dir = ROOT / pkg
        if not pkg_dir.is_dir():
            return []
        found: list[str] = []
        for py in sorted(pkg_dir.rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8"))
            rel = str(py.relative_to(ROOT))
            for module, lineno in _iter_imports(tree):
                top = module.split(".")[0].lstrip(".")
                if top == pkg or top not in PROJECT_PACKAGES:
                    continue
                if rel in ORCHESTRATOR_FILES and top != "model_ensemble":
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

    def test_forecasting_core_contracts(self):
        self.assertEqual(self._violations("model_forecasting"), [])

    def test_probabilistic_capability(self):
        self.assertEqual(self._violations("probabilistic"), [])

    def test_ensemble_stays_clean(self):
        self.assertEqual(self._violations("model_ensemble"), [])


if __name__ == "__main__":
    unittest.main()
