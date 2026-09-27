"""XGBoost 原生参数预检：子进程隔离告警，无数据拟合。

父进程只传实际 native 参数、特征名与输入维度，不传训练数据。
缓存限于当前进程，且只保存成功预检；锁保证并行 scalar fit 不重复启动子进程。

本文件双角色：可导入模块（父进程调用 ``validate_xgb_parameters``）+
子进程 worker（``python xgb_preflight.py --worker``，经 ``Path(__file__)``
直跑，不依赖 PYTHONPATH/cwd）。

文件名约束：worker 直跑时脚本目录位于 ``sys.path[0]``，本文件名不得与
任何第三方包同名（不能叫 xgboost.py，否则 ``import xgboost`` 导入自身）；
父进程侧的特征名提取等重依赖辅助见 ``xgboost_estimator.py``。
"""

import contextlib
import io
import json
import os
import pickle
import re
import subprocess
import sys
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import xgboost as xgb


# 子进程超时（秒）：空 Booster 参数预检为毫秒级，30s 只防挂死
_PREFLIGHT_TIMEOUT_SECONDS = 30
# 成功预检缓存容量：同一参数组合只检一次
_PREFLIGHT_CACHE_SIZE = 128
# worker 侧线程钉扎为 1，避免预检进程与父进程训练线程争抢 CPU
_THREAD_PIN_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

_PREFLIGHT_LOCK = threading.Lock()


@lru_cache(maxsize=_PREFLIGHT_CACHE_SIZE)
def _cached_preflight(request: bytes) -> str:
    environment = dict(os.environ)
    environment.update(_THREAD_PIN_ENV)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=request, capture_output=True, timeout=_PREFLIGHT_TIMEOUT_SECONDS, env=environment,
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
    """在隔离子进程中校验 XGBoost 原生配置，不触碰父进程的 warning 处理器。

    Args:
        params: 估计器实际生效的 native 参数
        num_features: 训练输入特征维度
        num_targets: 目标维度（缺省 1）
        feature_names: 特征名元组（可选，用于维度一致性预检）

    Returns:
        含 status / xgboost_version / native_preflight_config / warnings 的
        独立字典（每次解析新对象）

    Raises:
        ValueError: 维度非法，或子进程判定参数无效/与输入维度冲突
        RuntimeError: 子进程自身失败（超时/解释器错误）
    """
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
    # 内部协议：精确全匹配 "--worker"，拒绝任何多余参数，防误用。
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Internal worker: expected --worker")
    _worker()
