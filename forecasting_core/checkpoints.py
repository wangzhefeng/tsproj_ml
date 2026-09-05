"""Completed-fit persistence contract; training never imports L2 storage."""
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

import numpy as np

T = TypeVar("T")


class FitCheckpoint(Protocol):
    """Persist only the value returned after a complete fit, never its factory."""

    def child(self, **context: Any) -> "FitCheckpoint": ...

    def run(
        self, *, identity: Mapping[str, Any],
        arrays: Sequence[np.ndarray | None], fit: Callable[[], T],
    ) -> T: ...


class FitCheckpointError(RuntimeError):
    """Machine-readable config/fold/model failure, with original exception cause."""

    def __init__(self, *, context: Mapping[str, Any], model: Any, phase: str,
                 key: str, reason: str) -> None:
        self.context = dict(context)
        self.config = context.get("config")
        self.fold = context.get("fold")
        self.model = model
        self.phase = phase
        self.key = key
        self.reason = reason
        super().__init__(f"checkpoint {phase}: config={self.config} fold={self.fold} "
                         f"model={model} key={key}: {reason}")

    def as_dict(self) -> dict[str, Any]:
        return dict(config=self.config, fold=self.fold, model=self.model,
                    phase=self.phase, key=self.key, reason=self.reason,
                    context=self.context)
