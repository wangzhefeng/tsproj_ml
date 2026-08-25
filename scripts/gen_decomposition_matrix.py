# -*- coding: utf-8 -*-
"""按 §7.10 方法矩阵派生 add_decomposition 配置（一次性脚本，用后可删）。

规则：
- 模板 = 同目录已有的 add_decomposition 配置（保留模型/数据/特征/性能全部字段）；
- 只改 preprocessing 分解字段、setting_suffix 和首行注释；
- baseline / add_exogenous_weather_date 不生成分解变体（保持 none）。
"""
from pathlib import Path
import re

import yaml

ROOT = Path("config")

# (场景, 目录, 周期列表, 频率标签) → 新方法配置
MSTL_SPECS = {
    ("aidc_ess_selfuse_load", "add_decomposition"): {
        "periods": [288, 2016],  # 日周期 + 周周期（5min）
        "suffix": "decomp-mstl288-2016",
        "label": "mstl288+2016",
    },
    ("aidc_load_15min_daily", "add_decomposition"): {
        "periods": [96, 672],
        "suffix": "decomp-mstl96-672",
        "label": "mstl96+672",
    },
    ("aidc_load_15min_rolling", "add_decomposition"): {
        "periods": [96, 672],
        "suffix": "decomp-mstl96-672",
        "label": "mstl96+672",
    },
    ("aidc_load_15min_short", "add_decomposition"): {
        "periods": [96, 672],
        "suffix": "decomp-mstl96-672",
        "label": "mstl96+672",
    },
}

LINEAR_VARIANTS = {
    "quadratic": {"degree": 2, "forecast": "polynomial"},
    "damped": {"degree": 1, "forecast": "damped"},
}


def rewrite_decomp_fields(text: str, spec: dict) -> str:
    """替换 preprocessing 中分解相关字段。"""
    # 删除旧的分解字段块
    keys = [
        "decomposition_method", "decomposition_periods", "decomposition_robust",
        "decomposition_trend_degree", "decomposition_trend_forecast",
        "decomposition_damping", "decomposition_seasonal_cycles",
    ]
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if any(stripped.startswith(k + ":") for k in keys):
            # 跳过该行；若是 list 值（如 decomposition_periods 后跟 "- x"），跳过后续 list 行
            i += 1
            indent = len(line) - len(line.lstrip())
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.strip()
                if ns.startswith("- ") or nxt.startswith(" " * (indent + 2)) and ns and not any(
                    ns.startswith(k + ":") for k in ["decomposition_", "scale_target", "target_calendar"]
                ) and ns.startswith("-"):
                    i += 1
                else:
                    break
            continue
        out.append(line)
        i += 1
    text = "\n".join(out)

    # 构造新分解块
    block_lines = [
        f"    decomposition_method: {spec['method']}",
    ]
    if spec.get("periods"):
        block_lines.append(f"    decomposition_periods:")
        for p in spec["periods"]:
            block_lines.append(f"    - {p}")
    block_lines += [
        f"    decomposition_robust: {str(spec.get('robust', True)).lower()}",
        f"    decomposition_trend_degree: {spec.get('degree', 1)}",
        f"    decomposition_trend_forecast: {spec.get('forecast', 'polynomial')}",
    ]
    if spec.get("forecast") == "damped":
        block_lines.append(f"    decomposition_damping: {spec.get('damping', 0.98)}")
    block_lines.append(f"    decomposition_seasonal_cycles: {spec.get('cycles', 4)}")

    # 插入到 preprocessing: 之后（保持 target_calendar_normalization 在前）
    m = re.search(r"(?m)^  preprocessing:\n(    target_calendar_normalization:[^\n]*\n)?", text)
    if m:
        insert_at = m.end()
        block = "\n".join(block_lines) + "\n"
        return text[:insert_at] + block + text[insert_at:]
    raise RuntimeError(f"preprocessing section not found")


def derive(src: Path, dst_dir: Path, new_stem: str, spec: dict, label: str):
    text = src.read_text()
    # 替换 suffix（从文本中取旧值）
    m = re.search(r"setting_suffix:\s*([^\n]*)", text)
    old_suffix = m.group(1).strip() if m else ""
    if old_suffix:
        text = text.replace(f"setting_suffix: {old_suffix}", f"setting_suffix: -{spec['suffix']}")
    # 替换首行注释中的分解描述
    text = re.sub(r"^(# .*· 目标分解 ).*$", rf"\g<1>{label}", text, count=1, flags=re.M)
    if "目标分解" not in text.split("\n")[0]:
        # 模板无分解描述时追加
        lines = text.split("\n")
        lines[0] = lines[0].rstrip() + f" · 目标分解 {label}"
        text = "\n".join(lines)
    # 替换分解字段
    text = rewrite_decomp_fields(text, spec)

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{new_stem}.yaml"
    dst.write_text(text)
    return dst


def main():
    created = []
    # 1. 高频 MSTL：以 stl 变体为模板
    for (scene, subdir), mstl in MSTL_SPECS.items():
        for route in sorted((ROOT / scene).glob("route_*")):
            ddir = route / subdir
            templates = sorted(ddir.glob("*_decomp_stl*.yaml"))
            for tpl in templates:
                # 剥离旧分解后缀（_decomp_stl288 / _decomp_stl96），避免 _decomp_decomp- 重复
                stem = re.sub(r"_decomp_stl\d+$", "", tpl.stem)
                new_suffix_short = mstl["suffix"].replace("decomp-", "", 1)
                new_stem = f"{stem}_decomp_{new_suffix_short}"
                dst = ddir / f"{new_stem}.yaml"
                if dst.exists():
                    continue
                spec = {
                    "method": "mstl",
                    "periods": mstl["periods"],
                    "robust": True,
                    "cycles": 4,
                    "suffix": mstl["suffix"],
                }
                created.append(derive(tpl, ddir, new_stem, spec, mstl["label"]))

    # 2. load_month：以现有 decomposition/ 为模板补 stl7 + quadratic/damped 已有则跳过
    lm_specs = {
        "stl7": {"method": "stl", "periods": [7], "robust": True, "cycles": 4,
                 "suffix": "decomp-stl7", "label": "stl7"},
        "quadratic": {"method": "linear", "degree": 2, "forecast": "polynomial",
                      "suffix": "decomp-quadratic", "label": "quadratic"},
        "damped": {"method": "linear", "degree": 1, "forecast": "damped", "damping": 0.98,
                   "suffix": "decomp-damped", "label": "damped"},
    }
    for route in sorted((ROOT / "aidc_load_month").glob("route_*")):
        ddir = route / "decomposition"
        templates = sorted(ddir.glob("*.yaml"))
        if not templates:
            continue
        for key, spec in lm_specs.items():
            for tpl in templates:
                # 剥离方法后缀得到干净基名（lgbm_usmdp_mean_none → lgbm_usmdp_mean）
                base_stem = re.sub(r"_(none|linear|quadratic|damped|decomp_stl7)$", "", tpl.stem)
                new_stem = base_stem
                if key == "stl7":
                    new_stem += "_decomp_stl7"
                elif key == "quadratic":
                    if "_quadratic" in tpl.stem or "_damped" in tpl.stem:
                        continue
                    new_stem = base_stem + "_quadratic"
                elif key == "damped":
                    if "_damped" in tpl.stem or "_quadratic" in tpl.stem:
                        continue
                    new_stem = base_stem + "_damped"
                dst = ddir / f"{new_stem}.yaml"
                if dst.exists() or dst == tpl:
                    continue
                created.append(derive(tpl, ddir, new_stem, spec, spec["label"]))
                break  # 每个变体只从第一个模板派生一次

    # 3. power_month freq_1day：补 quadratic/damped（模板 = linear 变体）
    pm_specs = {
        "quadratic": {"method": "linear", "degree": 2, "forecast": "polynomial",
                      "suffix": "decomp-quadratic", "label": "quadratic"},
        "damped": {"method": "linear", "degree": 1, "forecast": "damped", "damping": 0.98,
                   "suffix": "decomp-damped", "label": "damped"},
    }
    for freq_dir in sorted((ROOT / "aidc_power_month").rglob("freq_1day/add_decomposition")):
        templates = sorted(freq_dir.glob("*_decomp_linear.yaml"))
        for tpl in templates:
            base_stem = tpl.stem.replace("_decomp_linear", "")
            for key, spec in pm_specs.items():
                dst = freq_dir / f"{base_stem}_decomp_{key}.yaml"
                if dst.exists():
                    continue
                created.append(derive(tpl, freq_dir, f"{base_stem}_decomp_{key}", spec, spec["label"]))

    print(f"created {len(created)} configs:")
    for p in created:
        print(f"  {p}")


if __name__ == "__main__":
    main()
