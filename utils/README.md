# utils 模块说明

`utils/` 只保存**通用、无领域语义**的项目级工具（L0 基础层）：频率解析、日志、运行环境。无项目内依赖；领域逻辑一律放入其主要消费方所在包（如评估掩码在 `model_forecasting/backtest.py`），不放 utils。

## 文件职责

| 文件 | 职责 | 状态 |
|---|---|---|
| `frequency.py` | pandas 固定频率解析：步长分钟数、每日样本数、月频判定 | 现役（canonical 主链使用） |
| `log_util.py` | 项目日志器：控制台 stderr + `logs/<LOG_NAME>/service*` 按天轮转 | 现役 |
| `runtime_env.py` | 运行期环境变量辅助（`MPLCONFIGDIR` 落到系统临时目录） | 现役（入口/脚本使用） |

## 已迁移出 utils 的内容

| 原位置 | 现位置 | 理由 |
|---|---|---|
| `utils/eval_mask.py` | `model_forecasting/backtest.py::build_eval_mask` | 评估掩码是领域逻辑（D13 rewire），2026-08-30 迁入消费方所在层 |
| `utils/metrics.py` | 已删除 | 指标计算在 `model_forecasting/results.py`（点）与 `probabilistic/metrics.py`（概率） |
| `utils/cv_plot.py` | 已删除 | 绘图逻辑内联在运行层 |
| `utils/exogenous_contract.py` | 已删除（随 `data_provider/` 整包） | as-of 唯一实现在 `model_forecasting/data/registry.py` |
| `utils/weather_contract.py` | 已删除（随 `data_provider/` 整包） | 现役载体为 DataSpec 列角色 + registry as-of |
| `utils/multistep_contract.py` | 已删除 | 曾是策略解析器的转发 facade |
| `utils/conformal.py` | 已删除 | CQR 兼容入口，无生产消费者；直接用 `probabilistic/calibration.py` |
| `utils/quantile.py` | 已删除 | crossing 修复兼容入口，无生产消费者；直接用 `probabilistic/postprocessing.py` |
| `utils/parallel_budget.py` | 已删除 | 旧扁平配置的并行预算，canonical 无消费者 |

## 收录标准

一个模块可以进入 `utils/`，当且仅当：

1. 无预测领域语义（不出现 target/strategy/信息集概念）；
2. 被两个及以上模块消费，或明显是跨模块通用能力；
3. 无副作用边界（`log_util.py` 的导入时建目录是唯一例外，已文档化）。

## frequency.py

主要函数：

- `resolve_freq_step_minutes(freq)`：返回频率对应的分钟步长。
- `resolve_samples_per_day(freq)`：返回一天内样本数（月频特判返回 1）。

约束：频率必须是固定时长；步长为正；一天必须能被该频率整除。

| `freq` | 样本数/天 |
|---|---:|
| `5min` | 288 |
| `15min` | 96 |
| `1h` | 24 |
| `1D` | 1 |

## log_util.py

- `LOG_NAME` 控制日志子目录，默认由入口文件设置；`SERVICE_LOG_LEVEL` 控制级别，默认 `INFO`。
- 导入即创建日志目录；测试或脚本需要隔离时先设置 `LOG_NAME`。
