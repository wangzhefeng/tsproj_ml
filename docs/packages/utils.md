# utils

`utils/` 只保存无领域语义的基础工具：

- `log_util.py`：日志。
- `runtime_env.py`：运行环境准备。

本包不得 import 其他项目包。预测张量、评估、概率、配置和数据领域逻辑均不放入 utils。

## 实际行为与副作用

- 频率规范化由 `forecasting_core/specs/problem.py` 承担，自然月几何使用月 offset；时间衰减训练权重由 `model_training/sample_weight.py` 按真实时间计算。无消费者的旧 `frequency.py` 已退役，不提供兼容导入。
- `log_util.py` 在导入时创建日志目录和 handler，读取 `LOG_NAME`（默认 `main`）及 `SERVICE_LOG_LEVEL`；日志默认落在 `logs/main/`，不是无副作用模块。
- `ensure_runtime_environment()` 只准备临时 Matplotlib 配置目录，并在未设置时补 `MPLCONFIGDIR`；它不创建虚拟环境、不安装依赖，也不管理 uv 缓存。
