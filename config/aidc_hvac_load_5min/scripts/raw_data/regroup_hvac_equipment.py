"""套组分路与来源掩码数值工具；历史多版本重放入口已退役。"""
import numpy as np
import pandas as pd
ROUTES = ("route_A", "route_B")

def regroup_pair(pair, mapping, *, strict):
    if not pair['route_A'].index.equals(pair['route_B'].index):
        raise ValueError('双路时间轴不同')
    points = pd.concat([pair[r].drop(columns='total_load') for r in ROUTES], axis=1)
    if not points.columns.is_unique or set(points) != set(mapping.spot_id):
        raise ValueError('双路点位并集与清单不一致')
    result = {}
    for route in ROUTES:
        columns = mapping.loc[mapping.group_route.eq(route[-1]), 'spot_id'].tolist()
        table = points[columns].copy()
        table['total_load'] = table.sum(axis=1, min_count=len(columns) if strict else 1)
        result[route] = table
        pd.testing.assert_frame_equal(table[columns], points[columns], check_exact=True)
    return result


def mask_from_gaps(table, gaps, corrections=None):
    """逐点缺口及已应用修正是掩码真值；不搬移旧路线总掩码。"""
    index = table.index
    columns = table.columns.drop('total_load')
    observed = np.ones((len(index), len(columns)), dtype=bool)
    known = index.to_numpy().copy()
    rows = gaps[gaps.point.isin(columns)]
    starts = index.searchsorted(pd.to_datetime(rows.gap_start))
    stops = index.searchsorted(pd.to_datetime(rows.gap_end), side='right')
    positions = columns.get_indexer(rows.point)
    eligibility = pd.to_datetime(rows.eligibility_known_at).to_numpy()
    filled = rows.status.eq('filled').to_numpy() if 'status' in rows else np.ones(len(rows), dtype=bool)
    if pd.isna(eligibility[filled]).any():
        raise ValueError('已填补缺口缺少资格时间')
    for start, stop, col, value, applied in zip(starts, stops, positions, eligibility, filled):
        observed[start:stop, col] = False
        if applied:
            known[start:stop] = np.maximum(known[start:stop], value)
    if corrections is not None and not corrections.empty:
        for row in corrections[corrections.point.isin(columns) & corrections.applied.eq(True)].to_dict('records'):
            stamp = pd.Timestamp(row['time'])
            if stamp in index:
                loc = index.get_loc(stamp)
                observed[loc, columns.get_loc(row['point'])] = False
                known[loc] = max(known[loc], np.datetime64(row['eligibility_known_at']))
    total_observed = observed.all(axis=1) & table.total_load.notna()
    mask = pd.DataFrame({'total_observed': total_observed,
                         'total_imputed': ~total_observed & table.total_load.notna(),
                         'eligibility_known_at': known}, index=index)
    mask.loc[table.total_load.isna(), 'eligibility_known_at'] = pd.NaT
    return mask
