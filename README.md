# tsproj-ml

基于机器学习回归器的时间序列多步预测项目。项目只接受 canonical schema-2 YAML，支持 Local/Global、单/多目标、七种标准多步策略、point/边际 quantile 和引用式模型融合。

**核心认识**：配置语义以 YAML 为唯一来源；预测输入严格 as-of，缺失/异常直接 RAISE；包依赖为单向 DAG（AST 门禁固化）；结果身份由语义 fingerprint 决定，不自动重跑、不删存量。

## 分支约定

- `stable`：模型测试分支，只通过从 `dev` 合并（fast-forward 优先）推进，不在其上直接开发。
- `dev`：功能开发与重构主分支；阶段性收口完成定向及全量验证后合并进 `stable`，两条分支长期保留。

分支晋级不等于重新训练模型或授予研究配置部署资格；各模型、数据及 bundle 的适用边界仍按文档合同执行。

## 运行与验证

统一入口 `run.py --config-yaml <path>`（必须显式指定配置，无默认值）；测试 `tests/run_suite.py fast|all`。
环境约定、验证命令与纪律的唯一事实源：[`AGENTS.md`](AGENTS.md) §运行与验证。

## 文档目录

| 主题 | 入口 |
|---|---|
| 项目约定（不变量/分支/验证/文档索引） | [`AGENTS.md`](AGENTS.md) |
| 文档中心（全部主题文档索引） | [`docs/README.md`](docs/README.md) |
| 当前能力清单 | [`docs/capabilities.md`](docs/capabilities.md) |
| 架构与目录职责 | [`docs/architecture.md`](docs/architecture.md) |
| 结果合同 | [`docs/results.md`](docs/results.md) |
| 配置文档（schema/几何/数据角色/天气） | [`docs/config/`](docs/config/README.md) |
| 测试文档（执行集合/纪律/覆盖映射） | [`docs/testing/`](docs/testing/README.md) |
| 模型测试场景说明 | [`docs/scenarios/aidc_load_15min_short/模型测试说明.md`](docs/scenarios/aidc_load_15min_short/模型测试说明.md) |
| 各包职责与边界 | [`docs/packages/`](docs/packages/) |

历史方案文档（`docs/redesign/` 等）仅用于 Git/决策溯源，不作为当前实现事实源。
