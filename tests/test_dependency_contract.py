"""项目依赖清单一致性门禁。"""

from __future__ import annotations

import importlib
import re
import sys
import unittest
from pathlib import Path


tomllib = importlib.import_module("tomllib" if sys.version_info >= (3, 11) else "tomli")


ROOT = Path(__file__).resolve().parents[1]
_NAME_PATTERN = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_names(lines: list[str]) -> set[str]:
    names: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = _NAME_PATTERN.match(stripped)
        if match is not None:
            names.add(_normalize_name(match.group(1)))
    return names


class DependencyContractTests(unittest.TestCase):
    def test_lock_covers_all_direct_dependencies(self) -> None:
        """pyproject 直接依赖必须全部被 uv.lock 锁定。

        requirements.txt 已于 2026-09-01 移除（依赖唯一权威 = pyproject.toml +
        uv.lock）；如需导出冻结清单，用 `uv export` 现场生成，不入库。
        """
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        with (ROOT / "uv.lock").open("rb") as stream:
            lock = tomllib.load(stream)

        direct_names = _requirement_names(list(project["dependencies"]))
        locked_names = {
            _normalize_name(package["name"])
            for package in lock["package"]
        }

        self.assertSetEqual(direct_names - locked_names, set())


if __name__ == "__main__":
    unittest.main()
