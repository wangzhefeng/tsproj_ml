# utils

`utils/` 只保存无领域语义的基础工具：

- `frequency.py`：频率解析和 samples-per-day。
- `log_util.py`：日志。
- `runtime_env.py`：运行环境准备。

本包不得 import 其他项目包。预测张量、评估、概率、配置和数据领域逻辑均不放入 utils。
