# tests 模块说明

`tests/` 是已纳入版本控制的 unittest 回归测试（2026-08-15 起移出 `.gitignore`）。全量入口：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run \
  python -m unittest discover -s tests -p "test_*.py"
```

当前基线：**614 tests，全绿**。最新精确计数以 [`docs/multistep_forecasting_redesign.md`](../docs/multistep_forecasting_redesign.md) 为准。

## 测试分层

| 层 | 代表文件 | 覆盖内容 |
|---|---|---|
| canonical specs | `test_forecast_problem_spec.py`、`test_forecast_data_spec.py`、`test_canonical_strategy_spec.py`、`test_forecast_config_fingerprint.py` | 问题/数据/策略 spec 校验、未知字段与重复 key、fingerprint 语义稳定性 |
| 张量与策略 | `test_forecast_tensor.py`、`test_standard_strategy_executors.py`、`test_multi_target_adapters.py` | `(N,H,K[,Q])` shape 合同、七策略合成语义、independent/chain/native 能力门禁 |
| 数据与特征 | `test_information_set_registry.py`、`test_feature_visibility_compiler.py`、`test_exogenous_information_contract.py`、`test_custom_feature_future_strategy.py`、`test_origin_frozen_custom_features.py` | source registry、as-of 信息集、可见性驱动特征编译、自定义外生注册表 |
| 运行时端到端 | `test_canonical_runtime_smoke.py`、`test_canonical_global_runtime.py`、`test_canonical_ensemble_runtime.py`、`test_config_entrypoints.py` | synthetic 数据真实训练+预测、Global N2K2、Ensemble、入口与 Task27 执行矩阵（46 用例） |
| 结构与边界 | `test_canonical_only_runtime.py` | canonical 单运行时门禁：非 canonical 对象被拒绝、已删模块不存在 |
| 概率预测 | `test_probabilistic_*.py`、`test_multitarget_probabilistic.py`、`test_conformal.py` | quantile grid、crossing 修复、边际分布契约、CQR 数学内核 |
| 分解 | `test_decomposition_spec.py`、`test_decomposition_equivalence.py`、`test_decomposition_bundle.py` | 分解 spec 归一化、新旧语义等价、bundle 兼容 |
| 场景数据链 | `test_computility_process.py`、`test_point_load_aggregate.py`、`test_load_event_detection.py`、`test_load_state_features.py`、`test_power_month_event_features.py`、`test_ess_*.py`、`test_aidc_*.py` | 算力预处理、点负荷聚合、事件标签、负荷状态特征、ESS 策略特征、AIDC 日期窗口 |
| 工具与脚本 | `test_check_model_configs.py`、`test_fuse_aidc_results.py`、`test_select_aidc_leadership_days.py`、`test_adjust_leadership_day_summary_metrics.py`、`test_outlier_*.py`、`test_calendar_month_horizon.py`、`test_horizon_*` | 配置 checker、结果融合/筛选脚本、异常处理、自然月 horizon、目标日外生对齐 |

`fixtures/` 存放测试共享的构造器。

## 约定

- 改动被测模块时必须同步修复测试导入与接口断言，保持全套通过。
- AIDC 专项脚本目录路径含日期段、不是合法 Python 包，相关测试用 `sys.path.insert` 引导后按模块名导入。
- 测试不读写 `results/`；需要产物验证时用临时目录。
- 新能力必须有测试；结构门禁测试（如 `test_canonical_only_runtime.py`）用于防止已删除的 legacy 运行路径复活。
