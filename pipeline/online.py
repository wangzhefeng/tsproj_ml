"""有界部署历史：严格追加/显式重建，复用 canonical 物化、编译和预测。"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import cast
from copy import deepcopy

import numpy as np
import pandas as pd


from data_loading import SourceRegistry
from data_loading.information.information_set import MaterializedInformationSet, SourceLineage
from forecasting_core.bundle import ForecastModelBundle
from forecasting_core.specs import ForecastConfigSpec
from forecasting_core.specs.data import AvailabilityPolicy, ColumnRole
from forecasting_core.specs.validation import FixedStepBacktestSpec
from model_predicting.loops.deployment import predict_strategy_bundle
from pipeline.supervised_design import SupervisedDesignBuilder
from feature_engineering.compilation.requirements import minimum_history_rows
from feature_engineering.statistics.streaming import StreamingStatistics
from feature_engineering.compilation.compiler import CompiledFeatures


class _SnapshotRegistry(SourceRegistry):
    """保留普通 registry 的可见性校验，显式标记内存快照来源。"""

    def __init__(self, config: ForecastConfigSpec, frame: pd.DataFrame):
        digest = hashlib.sha256()
        digest.update(repr(tuple(frame.columns)).encode())
        digest.update(pickle.dumps(frame, protocol=5))
        self.snapshot_version = "online:" + digest.hexdigest()
        super().__init__(config.data, Path.cwd(), reader=lambda _path: frame.copy(deep=True))

    def materialize(self, request, *, source_names=None):
        info = super().materialize(request, source_names=source_names)
        return MaterializedInformationSet(
            target_history=info.target_history, observed_past=info.observed_past,
            known_future=info.known_future, static=info.static,
            observed_future_providers=info.observed_future_providers,
            lineage=tuple(SourceLineage(item.source_name, self.snapshot_version, None,
                                        item.availability_policy, item.includes_target_labels)
                          for item in info.lineage),
        )


class RollingForecastSession:
    """有限历史加精确全前缀统计的追加式预测；不更新训练状态。"""

    def __init__(self, config: ForecastConfigSpec, bundle: ForecastModelBundle,
                 history: pd.DataFrame, *, origin):
        self._configure(config, bundle)
        self.rebuild(history, origin=origin)

    def _configure(self, config: ForecastConfigSpec, bundle: ForecastModelBundle) -> None:
        if (config.problem.training_scope != "local" or config.problem.series_id_cols
                or len(config.data.sources) != 1 or not isinstance(config.validation.backtest, FixedStepBacktestSpec)
                or bundle.execution_mode != "strict"
                or config.validation.get("train_history_steps") is not None):
            raise ValueError("online session requires strict Local fixed-step single-target-source config")
        source = config.data.sources[0]
        if (source.time_col is None or source.source_type != "file" or source.availability != AvailabilityPolicy.SOURCE_TIME
                or source.series_id_cols or source.backtest_path is not None or source.future_path is not None
                or any(column.role not in {ColumnRole.TARGET, ColumnRole.KEY, ColumnRole.IGNORED}
                       for column in source.columns)):
            raise ValueError("online session requires a source_time target file without external inputs")
        if bundle.config_fingerprint != config.fingerprint():
            raise ValueError("online config/bundle identity mismatch")
        fit_origin = config.validation.get("forecast_origin")
        if fit_origin is None:
            raise ValueError("online session requires explicit final forecast_origin")
        self.config = config
        self.bundle = bundle
        self.time_col = source.time_col
        self.offset = pd.tseries.frequencies.to_offset(config.problem.freq)
        self.fit_origin = pd.Timestamp(fit_origin)
        self.retention_steps = minimum_history_rows(config)
        self._last_audit: tuple[CompiledFeatures, ...] = ()

    def _new_statistics(self) -> StreamingStatistics | None:
        advanced = self.config.features.transformations.get("advanced", {})
        if not any(name in advanced for name in ("ewm", "expanding", "time_since")):
            return None
        return StreamingStatistics(advanced, config_fingerprint=self.config.fingerprint(),
                                   time_col=self.time_col, freq=self.config.problem.freq)

    @property
    def last_audit(self):
        return self._last_audit

    def _normalize(self, frame: pd.DataFrame, origin) -> tuple[pd.DataFrame, pd.Timestamp]:
        origin = pd.Timestamp(origin)
        if pd.isna(origin) or origin < self.fit_origin:
            raise ValueError("online origin must not precede the final fit origin")
        if not isinstance(frame, pd.DataFrame) or frame.empty or self.time_col not in frame:
            raise ValueError("online history must be a nonempty timestamped DataFrame")
        frame = frame.copy(deep=True)
        times = pd.DatetimeIndex(pd.to_datetime(frame[self.time_col], errors="raise"))
        if (times.hasnans or not times.is_monotonic_increasing or times.has_duplicates
                or times[-1] != origin
                or not times.equals(pd.date_range(times[0], periods=len(times), freq=self.offset))):
            raise ValueError("online history must be continuous, ordered and end at origin")
        if (origin - self.fit_origin).value % self.offset.nanos:
            raise ValueError("online origin must match the model frequency grid")
        frame[self.time_col] = times
        for target in self.config.problem.targets:
            if target not in frame or not np.isfinite(frame[target].to_numpy(dtype=float)).all():
                raise ValueError("online target history must be finite and complete")
        return frame, cast(pd.Timestamp, origin)

    def _predict(self, frame, origin, statistics):
        builder = SupervisedDesignBuilder(self.config, _SnapshotRegistry(self.config, frame))
        designs, provider = builder.forecast_designs(origin, target_transform=self.bundle.target_transform,
                                                     statistics_provider=statistics)
        prediction = predict_strategy_bundle(self.bundle, designs[0],
            forecast_times=builder.request(origin).forecast_times, raw_feature_provider=provider)
        return prediction, builder.audit

    def rebuild(self, history: pd.DataFrame, *, origin) -> None:
        """显式处理历史修订/补采；所有验证成功之后才替换旧状态。"""
        frame, origin = self._normalize(history, origin)
        statistics = self._new_statistics()
        if statistics is not None:
            statistics = statistics.updated(frame, origin=origin)
        self._install(frame, origin, statistics)

    def _install(self, frame, origin, statistics) -> None:
        """候选状态编译/预测验证成功后，一次发布所有会话字段。"""
        start = origin - (self.retention_steps - 1) * self.offset
        retained = frame.loc[frame[self.time_col] >= start].reset_index(drop=True)
        if len(retained) < self.retention_steps:
            raise ValueError("online history is shorter than the required warm-up")
        _, audit = self._predict(retained, origin, statistics)
        self._history = retained
        self._statistics = statistics
        self.origin = origin
        self._last_audit = audit

    def update(self, observations: pd.DataFrame, *, origin) -> None:
        """仅追加新观测，不 partial_fit；拒绝隐式历史替换。"""
        frame, origin = self._normalize(observations, origin)
        if origin <= self.origin or frame[self.time_col].iloc[0] != self.origin + self.offset:
            raise ValueError("online update must append continuously; use rebuild for revisions")
        if tuple(frame.columns) != tuple(self._history.columns):
            raise ValueError("online update columns must match retained history")
        statistics = (self._statistics.updated(frame, origin=origin)
                      if self._statistics is not None else None)
        self._install(pd.concat([self._history, frame], ignore_index=True), origin, statistics)

    def predict(self):
        prediction, audit = self._predict(self._history, self.origin, self._statistics)
        self._last_audit = audit
        return prediction

    def state(self) -> dict:
        return {"schema_version": 2, "config_fingerprint": self.config.fingerprint(),
                "origin": self.origin.isoformat(), "history": self._history.copy(deep=True),
                "statistics": deepcopy(self._statistics)}

    @classmethod
    def from_state(cls, config, bundle, state):
        if (not isinstance(state, dict) or set(state) != {"schema_version", "config_fingerprint", "origin", "history", "statistics"}
                or type(state["schema_version"]) is not int or state["schema_version"] != 2
                or state["config_fingerprint"] != config.fingerprint()):
            raise ValueError("online state schema/identity mismatch")
        session = cls.__new__(cls)
        session._configure(config, bundle)
        frame, origin = session._normalize(state["history"], state["origin"])
        expected = session._new_statistics()
        statistics = deepcopy(state["statistics"])
        if expected is None:
            if statistics is not None:
                raise ValueError("finite-history config must not carry statistics state")
        else:
            if not isinstance(statistics, StreamingStatistics):
                raise ValueError("online state requires saved statistics; rebuild from full history")
            statistics.validate_snapshot(expected, origin)
            if statistics.count < len(frame) or statistics.history_start > frame[session.time_col].iloc[0]:
                raise ValueError("statistics count/time grid does not cover retained history")
        session._install(frame, origin, statistics)
        return session
