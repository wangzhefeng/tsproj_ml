"""将年度回放交付表接入项目通用张量、评分、回测写盘和绘图通路。"""
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting_core.tensors import PointForecastTensor
from model_evaluation.point import evaluate_point_forecasts
from model_testing.tensor_frames import backtest_tensors_to_long
from model_testing.reporting import write_backtest_results
from data_loading import chinese_holiday_frame


def annual_tensors(frame: pd.DataFrame) -> tuple[PointForecastTensor, PointForecastTensor]:
    times = pd.DatetimeIndex(frame.time)
    actual = PointForecastTensor(frame.y_true.to_numpy().reshape(1, -1, 1), ("__local__",), times, ("value",))
    prediction = PointForecastTensor(frame.y_pred.to_numpy().reshape(1, -1, 1), ("__local__",), times, ("value",))
    return actual, prediction


def write_diagnostics(output: Path, annual: pd.DataFrame) -> None:
    """额外评估表；事后高低负荷分组绝不进入模型特征。"""
    calendar = chinese_holiday_frame(annual.time.min(), annual.time.max().normalize() + pd.Timedelta(days=1), freq='1D')
    frame = annual.copy()
    frame['day'] = frame.time.dt.normalize()
    frame = frame.merge(calendar[['time', 'is_holiday', 'holiday_name']].rename(columns={'time': 'day'}),
                        on='day', validate='many_to_one')
    modeled = frame[frame.time >= '2025-02-01']
    groups = [('coverage', 'all_year_including_january', frame), ('coverage', 'model_period_feb_dec', modeled)]
    groups.extend(('month', str(month), part) for month, part in frame.groupby(frame.time.dt.month))
    labels = np.select([modeled.holiday_name == 'Spring Festival', modeled.holiday_name != '', modeled.is_holiday == 1],
                       ['spring_festival', 'other_named_holiday', 'ordinary_rest_day'], default='calendar_workday')
    groups.extend(('calendar_day_type', label, modeled.loc[labels == label]) for label in sorted(set(labels)))
    lower, upper = modeled.y_true.quantile([.25, .75])
    if lower < upper:
        groups.extend([('ex_post_load_level', 'bottom_quartile', modeled[modeled.y_true <= lower]),
                       ('ex_post_load_level', 'top_quartile', modeled[modeled.y_true >= upper])])
    rows = []
    for dimension, label, part in groups:
        if part.empty:
            continue
        scores = evaluate_point_forecasts(*annual_tensors(part))
        aggregate = scores.loc[scores.scope == 'aggregate'].iloc[0]
        rows.append({'dimension': dimension, 'group': label, 'n_points': int(aggregate.n_points),
                     **{key: float(aggregate[key]) for key in ('MAE', 'RMSE', 'Bias', 'MAPE')}})
    pd.DataFrame(rows).to_csv(output / 'diagnostic_scores_df.csv', index=False)


def write_annual_results(output: Path, annual: pd.DataFrame, freq: str, metadata: dict) -> dict:
    """1月为window 0接入段；其余逐月/逐日编号，年度分数按全年逐点计算。"""
    frame = annual.copy()
    if freq == "1D":
        frame["window"] = frame.time.dt.month - 1
    elif freq == "15min":
        frame["window"] = ((frame.time.dt.floor("1D") - pd.Timestamp("2025-02-01")).dt.days + 1).clip(lower=0)
    else:
        raise ValueError("unsupported annual frequency")
    long_frames, score_frames = [], []
    for window, group in frame.groupby("window", sort=True):
        actual, prediction = annual_tensors(group)
        long_frames.append(backtest_tensors_to_long(actual, prediction, window=int(window)))
        score_frames.append(evaluate_point_forecasts(actual, prediction, window=int(window)))
    write_backtest_results(output, pd.concat(long_frames, ignore_index=True),
                           pd.concat(score_frames, ignore_index=True), aggregate_weighting={"value": 1.0},
                           metadata={**metadata, "january_window": 0,
                                     "january_semantics": "actual passthrough, included in annual scoring"})
    actual, prediction = annual_tensors(frame)
    scores = evaluate_point_forecasts(actual, prediction)
    summary = scores[scores.scope.isin(["target", "aggregate"])]
    summary.to_csv(output / "annual_scores_df.csv", index=False)
    write_diagnostics(output, annual)
    aggregate = summary.loc[summary.scope == "aggregate"].iloc[0]
    result = {"rows": len(frame), "MAE": float(aggregate.MAE), "RMSE": float(aggregate.RMSE),
              "includes_january_actuals": True}
    # 标准long CSV回读，确保绘图所用值与全年交付表逐点一致。
    saved = pd.read_csv(output / "cv_plot_df.csv", parse_dates=["time"])
    if not pd.DatetimeIndex(saved.time).equals(pd.DatetimeIndex(annual.time)):
        raise ValueError("canonical result timestamps differ from annual export")
    np.testing.assert_allclose(saved.actual_value, annual.y_true, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(saved.predict_value, annual.y_pred, rtol=1e-14, atol=1e-14)
    return result
