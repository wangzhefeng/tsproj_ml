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
    def test_lock_and_export_cover_all_direct_dependencies(self) -> None:
        with (ROOT / "pyproject.toml").open("rb") as stream:
            project = tomllib.load(stream)["project"]
        with (ROOT / "uv.lock").open("rb") as stream:
            lock = tomllib.load(stream)

        direct_names = _requirement_names(list(project["dependencies"]))
        locked_names = {
            _normalize_name(package["name"])
            for package in lock["package"]
        }
        requirement_lines = (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        exported_names = _requirement_names(requirement_lines)

        self.assertIn("uv export", "\n".join(requirement_lines[:3]))
        self.assertSetEqual(direct_names - locked_names, set())
        self.assertSetEqual(direct_names - exported_names, set())


if __name__ == "__main__":
    unittest.main()
