# -*- coding: utf-8 -*-
"""2026-09-01 一次性迁移：probabilistic 段 legacy 键清扫。

- `crossing_method: isotonic` -> `crossing: {method: median_preserving_isotonic, report_raw: true}`
  （isotonic 的历史运行时行为 = 排序+point 锚定钳制，映射后语义零变化）
- `conformal: {method: none, ...}` -> 删除（无操作意图）
- `conformal: {method: cqr, coverage, min_scores, min_windows}` ->
  `intervals:` + `calibration:` canonical 块（激活 CQR，interval 取最外分位对）

只做文本级手术，保持 YAML 其余字节不变。幂等：重复运行零改动。
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

import yaml  # noqa: E402


def _quantile_token(level: float) -> str:
    percent = float(level) * 100.0
    if abs(percent - round(percent)) < 1e-12:
        return str(int(round(percent)))
    return f"{percent:.12f}".rstrip("0").rstrip(".").replace(".", "p")


def migrate_text(path: Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    changed = False
    index = 0
    while index < len(lines):
        line = lines[index]
        if line == "  crossing_method: isotonic\n":
            out.append("  crossing:\n")
            out.append("    method: median_preserving_isotonic\n")
            out.append("    report_raw: true\n")
            changed = True
            index += 1
            continue
        if line == "  conformal:\n":
            # 收集 4 空格子行直到回到 <=2 空格缩进
            block: list[str] = []
            cursor = index + 1
            while cursor < len(lines) and lines[cursor].startswith("    "):
                block.append(lines[cursor])
                cursor += 1
            keys = {}
            for child in block:
                key, _, value = child.strip().partition(":")
                keys[key.strip()] = value.strip()
            method = keys.get("method")
            if method == "none":
                changed = True  # 整块删除
                index = cursor
                continue
            if method == "cqr":
                payload = yaml.safe_load("".join(lines))
                quantiles = sorted(float(q) for q in payload["probabilistic"]["quantiles"])
                lo, hi = quantiles[0], quantiles[-1]
                name = f"q{_quantile_token(lo)}_q{_quantile_token(hi)}"
                coverage = keys["coverage"]
                out.append("  intervals:\n")
                out.append(f"  - name: {name}\n")
                out.append(f"    lower_quantile: {lo}\n")
                out.append(f"    upper_quantile: {hi}\n")
                out.append("  calibration:\n")
                out.append("    method: cqr\n")
                out.append(f"    interval: {name}\n")
                out.append(f"    target_coverage: {coverage}\n")
                out.append("    calibration_windows: 5\n")
                out.append(f"    min_windows: {keys['min_windows']}\n")
                out.append(f"    min_scores: {keys['min_scores']}\n")
                changed = True
                index = cursor
                continue
            raise ValueError(f"{path}: unknown conformal method={method!r}")
        out.append(line)
        index += 1
    if changed:
        path.write_text("".join(out), encoding="utf-8")
    return changed


def main() -> None:
    candidates = sorted(PROJ.glob("config/**/*.yaml"))
    touched = []
    for path in candidates:
        text = path.read_text(encoding="utf-8")
        if "crossing_method:" not in text and "conformal:" not in text:
            continue
        if migrate_text(path):
            touched.append(path)
    print(f"migrated {len(touched)} files")


if __name__ == "__main__":
    main()
