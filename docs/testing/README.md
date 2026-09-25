# docs/testing — 测试文档中心

2026-09-25 场景收敛：活动预测场景只保留 `aidc_load_15min_short`（仅 LightGBM 配置），其余预测场景（daily/rolling/load_month/power_month/ess_selfuse_load/computility/hongtaiyang）连同其专属测试（`test_liantong_*`、`test_computility_*`、`test_ess_*`、`test_hongtaiyang_*`、`test_aidc_*` 场景守卫、`test_generate_load_15min_matrix`、`test_audit_ensemble_configs` 等）一并退役，历史从 Git 溯源。`test_native_ets`（含 4032 点、5min/288 合成序列、失败透明及可信 pickle）与 `test_seasonal_kernels`（独立黄金值、严格 as-of）为合成数据能力测试，保留在 integration。测试合成拟合不构成正式模型效果证据。

`tests/` 是版本控制内的 unittest 套件，覆盖 core contracts、信息集、特征、七策略、Local/Global、point/quantile、fixed/calendar/monthly runtime、Ensemble、结果 schema、场景数据链和包间结构。

## 章节

| 章节 | 内容 |
|---|---|
| [coverage-mapping.md](coverage-mapping.md) | 定向测试说明、精简与覆盖映射、行为不变重构映射、日历回归 |
| [weather-decomposition.md](weather-decomposition.md) | 天气生成器测试、分解职责收敛、禁止模型执行的定向检查白名单 |

## 执行集合

保留平铺目录及原生 unittest discovery，不移动现有测试或修改 CI。分组在 `run_suite.py` 单点维护，三个集合互斥，其并集等于原生发现全集：

| 集合 | 用途 | 何时执行 |
|---|---|---|
| `fast` | 显式选入的轻量合同、指标、七策略轨迹、结构门禁 | 日常修改；不能替代相关集成验证 |
| `integration` | 真实后端、信息集/编译器、小样本运行链、缓存与部署恢复 | 按改动范围运行；未分类新测试默认进入这里 |
| `audit` | 全仓 YAML、真实资产、场景特征生产链 | 配置/数据/场景模块修改及全量收口 |
| `all` | 三组全集 | 全量收口，串行运行，避免多套测试争抢 CPU |

```bash
# 以下命令均从项目根运行；run_suite 只接受项目 .venv 解释器，其他环境直接拒绝
env -u PYTHONPATH .venv/bin/python tests/run_suite.py fast
env -u PYTHONPATH .venv/bin/python tests/run_suite.py integration --match test_ensemble_runtime
env -u PYTHONPATH .venv/bin/python tests/run_suite.py audit --match test_validation_geometry_manifest
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --list
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --report .hermes/plans/test-suite-report.json

# 原有入口仍发现并执行所有测试，不是快测入口
env -u PYTHONPATH .venv/bin/python -m unittest discover -s tests -p "test_*.py"
```

测试分组入口只接受根目录 `.venv/`（按 `sys.prefix` 校验，错误环境返回 2）；该门禁不拦截绕过入口的原生 unittest 或库调用，直接运行它们时也必须显式使用 `.venv/bin/python`。测试不经过 uv 或 `.uv_cache/`；依赖仍由 `pyproject.toml` + `uv.lock` 管理，必要时用 `uv sync --locked --no-cache` 同步已有 `.venv`，不新建其他环境。

`--match` 是完整测试 ID 的子串过滤；无匹配、重复 ID 或任一模块导入失败返回非零，不允许因为选择 fast 而隐藏其他模块的发现错误。`--list` 仅导入和发现，不执行测试正文；仍会发生项目已有的 import 初始化。`--report` 记录本次实际 testsRun、失败/错误/跳过 ID 和逐测试耗时，不作未经测量的速度承诺。场景模块需要独立选择，不代表其测试可以删除。

## 执行纪律

- 新行为先 RED 再实现；
- 新测试默认纳入 integration，不得通过遗漏发现或 skip 提速；
- runtime 产物使用临时目录；
- `tests/test_package_layering.py` AST 扫描所有 import，包括函数内 import；
- `tests/test_active_config_runtime_contract.py` 保证活动 YAML 与生产 grammar/时间几何一致；
- `tests/test_runtime_asset_audit.py` 要求活动 source 零缺失；
- 删除测试前必须证明生产链已删除，或将具体断言映射到保留测试；高层冒烟不能替代低层负向合同。

最新精确测试数以本次全量命令输出为准，不在多份文档重复维护。
