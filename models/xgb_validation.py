"""XGBoost 原生参数预检：子进程隔离告警，无数据拟合。

父进程只传实际 native 参数、特征名与输入维度，不传训练数据。
缓存限于当前进程，且只保存成功预检；锁保证并行 scalar fit 不重复启动子进程。
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import xgboost as xgb


_PREFLIGHT_LOCK = threading.Lock()


@lru_cache(maxsize=128)
def _cached_preflight(request: bytes) -> str:
    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=request, capture_output=True, timeout=30, env=environment,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "XGBoost parameter preflight subprocess failed: "
            + result.stderr.decode("utf-8", errors="replace")
        )
    response = json.loads(result.stdout)
    if response.get("error"):
        raise ValueError("XGBoost parameter validation: " + response["error"])
    return result.stdout.decode("utf-8")


def validate_xgb_parameters(
    params: Mapping[str, Any],
    *,
    num_features: int,
    num_targets: int = 1,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate native configuration without touching parent warning handlers."""
    if num_features < 1 or num_targets < 1:
        raise ValueError("XGBoost preflight requires positive input dimensions")
    request = pickle.dumps({
        "params": dict(sorted(params.items())),
        "num_features": num_features,
        "num_targets": num_targets,
        "feature_names": feature_names,
        "xgboost_version": xgb.__version__,
    }, protocol=pickle.HIGHEST_PROTOCOL)
    with _PREFLIGHT_LOCK:
        # 返回独立字典，防止调用方持久化实际拟合配置时污染另一 fitted unit。
        return json.loads(_cached_preflight(request))


def _worker() -> None:
    import contextlib
    import io
    import re
    import warnings

    # 仅接收父进程自行 pickle 的本地参数，不接收外部缓存或远程文件。
    request = pickle.loads(sys.stdin.buffer.read())
    params = request["params"]
    response: dict[str, Any]
    try:
        for name, value in (("num_feature", request["num_features"]), ("num_target", request["num_targets"])):
            if params.get(name) is not None and int(params[name]) != value:
                raise ValueError(f"{name} conflicts with actual input dimensions")
            params[name] = value
        # 仅强制诊断开关，防止静默模式藏住未知参数；不修改父进程估计器参数。
        params.update(validate_parameters=True, verbosity=1)
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            booster = xgb.Booster()
            if request["feature_names"] is not None:
                booster.feature_names = list(request["feature_names"])
            booster.set_param(params)
            native_config = json.loads(booster.save_config())
        messages = [str(item.message) for item in caught]
        unused = [message for message in messages if re.search(r"Parameters:\s*\{.*?\}\s*are not used", message, re.S)]
        if unused:
            raise ValueError("; ".join(unused))
        response = {
            "status": "validated",
            "xgboost_version": xgb.__version__,
            "num_features": request["num_features"],
            "num_targets": request["num_targets"],
            "native_preflight_config": native_config,
            "warnings": messages,
        }
    except Exception as exc:
        response = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal worker: expected --worker")
    _worker()
