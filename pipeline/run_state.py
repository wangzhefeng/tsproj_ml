"""Completion evidence independent of model execution and numerical results."""
from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4


def write_run_state(model_dir: Path, fingerprint: str, status: str) -> None:
    if status not in {"running", "completed", "failed"}:
        raise ValueError("unknown lifecycle status")
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "run_state.json"
    temporary = model_dir / f".run_state-{uuid4().hex}.tmp"
    temporary.write_text(json.dumps({"config_fingerprint": fingerprint, "status": status}), encoding="utf-8")
    os.replace(temporary, path)


def require_completed_state(payload: dict, fingerprint: str) -> None:
    # 严格形状校验即格式守卫：字段集合变化（含旧版 schema_version 残留）一律 RAISE
    if set(payload) != {"config_fingerprint", "status"} or payload["config_fingerprint"] != fingerprint or payload["status"] != "completed":
        raise ValueError("run lifecycle is not completed for this config fingerprint")
