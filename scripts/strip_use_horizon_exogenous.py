# -*- coding: utf-8 -*-
"""一次性卫生清扫：移除全部现役 YAML 中的 `use_horizon_exogenous` 冗余字段。

背景（2026-09-01）：canonical 运行时无该字段任何消费点（known_future 外生
天然按目标时刻取值，历史锚点由 align_to_target 控制），519 个 YAML 中的
声明属 legacy 残留。删除后 fingerprint 语义变化为零（字段从未参与行为），
wangzf 裁决不重跑存量结果、直接修复。

做法（吸取 2026-08-29 批量改写三个坑的教训，见 tsproj-ml-pipeline skill）：
- 逐行 splitlines(keepends=True)，替换正则显式锚定缩进+内容+\\r?\\n，
  回填文件主导行尾；
- 只处理声明了 schema_version: 2 的模型 YAML（数据工具 YAML 不含该字段，
  双重保险跳过）；
- 先 --dry-run 输出对账，无差异再实跑。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# 行首恰好 6 空格缩进的字段行（features.transformations.direct 层级）。
_PATTERN = re.compile(r"^ {6}use_horizon_exogenous: (?:true|false)\r?\n", re.MULTILINE)


def _dominant_eol(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="只统计与预览，不写文件"
    )
    parser.add_argument(
        "--apply", action="store_true", help="实际写入（与 --dry-run 互斥）"
    )
    args = parser.parse_args()
    if args.dry_run == args.apply:
        parser.error("必须且只能指定 --dry-run 或 --apply 之一")

    targets = sorted(
        path
        for path in (_REPO / "config").rglob("*.yaml")
        if "use_horizon_exogenous" in path.read_text(encoding="utf-8")
    )
    changed = 0
    skipped_non_model = 0
    for path in targets:
        text = path.read_text(encoding="utf-8")
        if "schema_version: 2" not in text:
            skipped_non_model += 1
            continue
        new_text, count = _PATTERN.subn("", text)
        if count == 0:
            print(f"WARN 未匹配（形态超出预期）: {path.relative_to(_REPO)}")
            continue
        if count > 1:
            print(f"WARN 多处匹配（{count}）: {path.relative_to(_REPO)}")
        # 防御：确认没有吃掉相邻行的行尾（坑 1 的症状）。
        if _dominant_eol(text) != _dominant_eol(new_text) and count > 0:
            eol = _dominant_eol(text)
            new_text = new_text.replace("\r\n", "\n") if eol == "\n" else new_text
        if args.apply:
            path.write_text(new_text, encoding="utf-8", newline="")
        changed += 1

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] files with field: {len(targets)}, cleaned: {changed}, "
          f"non-model skipped: {skipped_non_model}")
    if args.dry_run:
        print("对账无误后用 --apply 实跑。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
