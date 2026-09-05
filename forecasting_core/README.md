# forecasting_core

`forecasting_core/` 是项目最低层的稳定预测合同包，不执行文件 IO、训练、回测或预测。

## 内容

- `specs/`：problem/data/feature/strategy/estimator/config 严格不可变规格。
- `tensors.py`：point `(N,H,K)`、marginal quantile `(N,H,K,Q)`、sample 类型边界。
- `probabilistic_spec.py`：point/quantile、quantile grid、interval/calibration 配置合同。
- `artifacts.py`：`ForecastModelBundle`、`MarginalForecastDistribution`、`QuantileGrid`。

旧一维 `ForecastDistribution` 与无消费者的 `calibration_runtime_kwargs` / `validate_probabilistic_args` 已退出；使用现役张量分布及概率 spec 解析接口。不提供旧类型 import/pickle 兼容。`PredictionIntervalForecast`、`QuantileGrid` 和明确 unsupported 的 joint-sample 边界保留。

## 依赖规则

本包不得 import 任何其他项目包。`tests/test_package_layering.py` 对所有 Python 文件做 AST 门禁，包括函数内 import。
