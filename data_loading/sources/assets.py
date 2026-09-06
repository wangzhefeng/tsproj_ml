"""单个输入资产的轻量表头预检查，不替代 materialize 的完整校验。"""
import csv
from pathlib import Path

from forecasting_core.specs.data import ColumnRole, DataSourceSpec

_PATH_FIELDS = ("history_path", "backtest_path", "future_path")


def required_columns(source: DataSourceSpec) -> set[str]:
    """维持原审计强度：非 ignored 列、时间和序列键；不检查 available_at。"""
    required = {column.name for column in source.columns if column.role is not ColumnRole.IGNORED}
    if source.time_col:
        required.add(source.time_col)
    required.update(source.series_id_cols)
    return required


def source_paths(source: DataSourceSpec) -> tuple[tuple[str, str], ...]:
    return tuple(
        (role, Path(str(path)).as_posix())
        for role in _PATH_FIELDS
        if (path := getattr(source, role))
    )


def asset_columns(raw_path: str | Path, base_dir: str | Path) -> set[str] | None:
    """缺路径返回 None；空文件/目录/读取错误仍直接传播。"""
    resolved = Path(raw_path)
    if not resolved.is_absolute():
        resolved = Path(base_dir) / resolved
    if not resolved.exists():
        return None
    with resolved.open(newline="", encoding="utf-8-sig") as stream:
        try:
            return set(next(csv.reader(stream)))
        except StopIteration as exc:
            raise ValueError(f"runtime source is empty: {resolved}") from exc
