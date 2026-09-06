"""分解诊断产物写入；只接收已计算报告，不依赖分解算法或私有状态。"""
from pathlib import Path
import pandas as pd


def write_diagnostics_report(report: pd.DataFrame | None, output_dir: Path, suffix: str = "") -> Path | None:
    if report is None:
        return None
    if "/" in suffix or "\\" in suffix:
        raise ValueError("Diagnostic suffix must be a filename suffix, not a path.")
    return write_residual_diagnostics(report, output_dir / f"decomposition_diagnostics{suffix}.csv")


def write_residual_diagnostics(report: pd.DataFrame, output_path: Path) -> Path:
    """独占创建，避免同一窗口或重复运行静默覆盖旧报告。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        report.to_csv(stream, index=False)
    return output_path
