"""监督标签构造：不参与特征可见性授权。"""
import numpy as np
import pandas as pd


def indexed_training_labels(config, registry, information, origins):
    frames = information.target_history
    targets = []
    for name in config.problem.targets:
        source = next(source for source in config.data.sources if any(column.name == name for column in source.columns))
        times, values = registry.numeric_history(source, name)
        visible = pd.DatetimeIndex(frames[source.name][source.time_col])
        begin = times.get_indexer(visible[:1])[0]
        if begin < 0 or not times[begin:begin + len(visible)].equals(visible):
            raise ValueError("label snapshot does not match visible history")
        values = values[begin:begin + len(visible)]
        positions = visible.get_indexer(origins)
        if (positions < 0).any() or (positions + config.problem.horizon >= len(visible)).any():
            raise ValueError("supervised labels exceed available history")
        windows = np.lib.stride_tricks.sliding_window_view(values, config.problem.horizon)
        if np.array_equal(positions, np.arange(positions[0], positions[-1] + 1)):
            labels = windows[positions[0] + 1:positions[-1] + 2]
        else:
            labels = windows[positions + 1]
        targets.append(labels)
    return targets[0][:, :, None] if len(targets) == 1 else np.stack(targets, axis=-1)
