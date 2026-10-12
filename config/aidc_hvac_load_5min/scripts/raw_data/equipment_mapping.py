"""暖通套组映射：保留电气 route，分组 route 按整套冷水机组归属。"""
from pathlib import Path

import pandas as pd

SHEET = '暖通负荷(大设备)'
XLSX = Path(__file__).resolve().parents[4] / 'dataset/aidc_load_5min/A1_A2_A3_points/all_ids.xlsx'
COMPOSITION = {'冷水机组': 1, '冷却塔': 2, '冷却水泵': 1, '冷冻水一次泵': 1, '冷冻水二次泵': 1}


def load_mapping(path=XLSX):
    frame = pd.read_excel(path, sheet_name=SHEET)
    required = ['data_type', 'spot_id', '备注', 'route', '东西南北']
    if not set(required).issubset(frame) or frame[required].isna().to_numpy().any():
        raise ValueError('设备清单缺少必要字段或存在空值')
    if frame.duplicated(['data_type', 'spot_id']).any():
        raise ValueError('同楼点位重复')
    frame['building'] = frame.data_type.str.extract(r'^(A[123])楼暖通电力负荷$', expand=False)
    if frame.building.isna().any() or set(frame.building) != {'A1', 'A2', 'A3'}:
        raise ValueError('楼栋集合不符合合同')
    if not frame.route.isin(['A', 'B']).all():
        raise ValueError('非法路线')
    frame['group_route'] = frame.route
    for building, part in frame.groupby('building'):
        directions = '东西' if building == 'A3' else '南北'
        expected = {f'{d}{n}' for d in directions for n in (1, 2, 3)}
        if set(part['东西南北']) != expected:
            raise ValueError(f'{building}: 模组套组不完整')
        for unit, group in part.groupby('东西南北'):
            if group['备注'].value_counts().to_dict() != COMPOSITION:
                raise ValueError(f'{building}/{unit}: 设备组成不符合1+2+1+1+1')
            route = group.loc[group['备注'].eq('冷水机组'), 'route'].iloc[0]
            if group.route.nunique() != 1 and (building, unit) not in {('A1', '南2'), ('A1', '南3')}:
                raise ValueError(f'{building}/{unit}: 未授权的跨路套组')
            frame.loc[group.index, 'group_route'] = route
    return frame
