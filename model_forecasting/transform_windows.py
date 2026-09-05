"""Target-transform fitting windows, independent of fitting implementations."""
from __future__ import annotations

from typing import Any, Sequence
import pandas as pd
from forecasting_core.tensors import PointForecastTensor


def select_transform_history(
    history: PointForecastTensor,
    origins: Sequence[pd.Timestamp],
    *,
    horizon: int,
    freq: str,
    decomposition_history_steps: int | None = None,
) -> tuple[PointForecastTensor, pd.DatetimeIndex, dict[str, Any]]:
    """Deduplicate supervised label times and bound decomposition context."""
    offset = pd.tseries.frequencies.to_offset(freq)
    labels = pd.DatetimeIndex(sorted({
        pd.Timestamp(origin) + step * offset
        for origin in origins for step in range(1, horizon + 1)
    }))
    if labels.empty:
        raise ValueError("transform fitting requires supervised label times")
    positions = history.forecast_times.get_indexer(labels)
    if (positions < 0).any():
        raise ValueError("training label times are missing from visible target history")
    start, end = int(positions.min()), int(positions.max()) + 1
    if decomposition_history_steps is not None:
        if isinstance(decomposition_history_steps, bool) or not isinstance(decomposition_history_steps, int) or decomposition_history_steps < 1:
            raise ValueError("decomposition.fit_history_steps must be a positive integer")
        if decomposition_history_steps < end - start:
            raise ValueError("decomposition context cannot shorten the training label window")
        if decomposition_history_steps > end:
            raise ValueError("insufficient visible history for explicit decomposition context")
        start = end - decomposition_history_steps
    context = PointForecastTensor(
        values=history.values[:, start:end, :], series_ids=history.series_ids,
        forecast_times=history.forecast_times[start:end], targets=history.targets,
    )
    audit = {
        "semantics": "unique_training_labels_v1",
        "scaler_unique_label_count": len(labels),
        "scaler_label_start": labels[0].isoformat(),
        "scaler_label_end": labels[-1].isoformat(),
        "decomposition_history_count": context.n_steps,
        "decomposition_history_start": context.forecast_times[0].isoformat(),
        "decomposition_history_end": context.forecast_times[-1].isoformat(),
    }
    return context, labels, audit
