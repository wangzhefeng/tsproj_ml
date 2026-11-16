"""单版本点位表 I/O 与缺口区间工具。"""
import csv
from pathlib import Path
import numpy as np
import pandas as pd
STEP = pd.Timedelta(minutes=5)

def true_runs(mask):
    """返回半开区间 [start, stop)，不分割超长缺口。"""
    changes = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def read_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle))
    if len(header) != len(set(header)) or header[0] != 'time' or 'total_load' not in header:
        raise ValueError(f'非法列名: {path}')
    frame = pd.read_csv(path, index_col='time', parse_dates=True, float_precision='round_trip')
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.equals(
            pd.date_range(frame.index[0], periods=len(frame), freq=STEP)):
        raise ValueError(f'非法5min时间网格: {path}')
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any():
        raise ValueError(f'负荷中存在非法数值: {path}')
    return frame.astype(float)


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, encoding='utf-8-sig', index_label='time')


def publish_stage(stage, root):
    """先检查全部冲突；只发布新文件，不覆盖已有结果。"""
    paths = sorted(p for p in stage.rglob('*') if p.is_file())
    for path in paths:
        if (root / path.relative_to(stage)).exists():
            raise FileExistsError(root / path.relative_to(stage))
    for path in paths:
        destination = root / path.relative_to(stage)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
