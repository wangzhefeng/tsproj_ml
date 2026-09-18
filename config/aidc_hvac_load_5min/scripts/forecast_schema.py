# -*- coding: utf-8 -*-
"""双路预测表的唯一列→填补源映射；不改变原始/填补表字段。"""
from pathlib import Path

from migrate_hvac_data import BUILDINGS, ROUTES, VERSIONS

SCHEMA = 'hvac_dual_route_v1'


def column_sources(version, building, with_it):
    """每列对应一个或多个严格求和来源；返回顺序就是CSV字段顺序。"""
    if version not in VERSIONS or building not in (*BUILDINGS, 'data'):
        raise ValueError(f'未知设备版本或楼栋: {version}/{building}')
    name = building + '_data.csv' if building != 'data' else 'data.csv'
    columns: dict[str, tuple[str, ...]] = {
        f'hvac_total_load_{r}': (f'{version}/route_{r}/{name}',) for r in ('A', 'B')}
    if building == 'data':
        columns['hvac_total_load_AB'] = tuple(f'{version}/route_{r}/data.csv' for r in ('A', 'B'))
    if with_it:
        columns['it_total_load'] = (f'IT_load/{name}',)
    if building == 'data':
        for b in BUILDINGS:
            for r in ('A', 'B'):
                columns[f'{b}_hvac_total_load_{r}'] = (f'{version}/route_{r}/{b}_data.csv',)
        if with_it:
            for b in BUILDINGS:
                columns[f'{b}_it_total_load'] = (f'IT_load/{b}_data.csv',)
    return columns


def target_column(route):
    if route not in ROUTES:
        raise ValueError(f'未知路线: {route}')
    return 'hvac_total_load_' + route.removeprefix('route_')


def file_contract(relative):
    relative = Path(relative)
    if len(relative.parts) != 3:
        raise ValueError(f'未知预测文件路径: {relative}')
    version, route, filename = relative.parts
    with_it = filename.endswith('_with_it.csv')
    base = filename.removesuffix('_with_it.csv') if with_it else filename.removesuffix('.csv')
    names = {**{b + '_data': b for b in BUILDINGS}, 'data': 'data'}
    if not filename.endswith('.csv') or base not in names:
        raise ValueError(f'未知预测文件: {relative}')
    return column_sources(version, names[base], with_it), target_column(route)
