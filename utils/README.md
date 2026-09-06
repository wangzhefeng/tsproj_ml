# utils

`utils/` 只保存无领域语义的基础工具：

- `frequency.py`：频率解析和 samples-per-day。
- `log_util.py`：日志。
- `runtime_env.py`：运行环境准备。

本包不得 import 其他项目包。预测张量、评估、概率、配置和数据领域逻辑均不放入 utils。

## 实际行为与副作用

- `frequency.py` 另提供 `compute_time_decay_weights()`，按样本年龄生成均值归一化的指数衰减权重。月频有独立识别分支；分钟换算中的近似值不是自然月回测几何，不能替代月 offset。
- `log_util.py` 在导入时创建日志目录和 handler，读取 `LOG_NAME`（默认 `main`）及 `SERVICE_LOG_LEVEL`；日志默认落在 `logs/main/`，不是无副作用模块。
- `ensure_runtime_environment()` 只准备临时 Matplotlib 配置目录，并在未设置时补 `MPLCONFIGDIR`；它不创建虚拟环境、不安装依赖，也不管理 uv 缓存。

## 环境与验证

项目运行使用已有 `.venv/bin/python`，依赖仍由根目录 `pyproject.toml` 和 `uv.lock` 管理。测试入口的解释器门禁位于 `tests/run_suite.py`，不在本包；删除 `.uv_cache/` 仅删除下载缓存。

从仓库根执行 `env -u PYTHONPATH`，是为避免外部注入路径遮蔽同名 `utils` 包，而不是切换解释器：

```bash
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_suite_runner
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_package_layering
```
