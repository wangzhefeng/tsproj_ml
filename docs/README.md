# docs — 项目文档中心

本目录按主题组织项目文档，遵循渐进式披露：每个文档只留概要与目录，细节链接到章节文件。

## 主题文档

| 文档 | 内容 | 何时读 |
|---|---|---|
| [capabilities.md](capabilities.md) | 当前能力清单、不支持项、CQR 边界 | 了解项目能做什么 |
| [architecture.md](architecture.md) | 当前架构、包间依赖 DAG、目录职责 | 动包结构/分层 |
| [results.md](results.md) | 结果目录合同、long 表唯一键、bundle 兼容边界 | 动结果产物/identity |
| [config/](config/README.md) | 配置 schema、时间几何、数据角色、天气合同（章节式） | 写/改配置 YAML |
| [testing/](testing/README.md) | 测试执行集合、纪律、覆盖映射（章节式） | 改测试/执行验证 |

## 代码旁与场景文档

- 各包职责与边界：[`packages/`](packages/)（每包一个 md，自原包目录 README 迁入，原位置不再保留文档）；
- 运行/验证/分支/核心不变量：根目录 [`AGENTS.md`](../AGENTS.md)；
- 场景文档：[`scenarios/aidc_load_15min_short/`](scenarios/aidc_load_15min_short/)（模型测试说明 + TODO_AIDC）；联通场景结果溯源 [`scenarios/liantong_IT.md`](scenarios/liantong_IT.md)；
- 脚本与数据目录：[`scripts.md`](scripts.md)、[`dataset/`](dataset/)（均自原目录 README 迁入并纳入版本控制；dataset/ 目录内其余分析报告类 README 属数据产物，保留原位）。

## 历史档案（不作为当前实现事实源）

- [`redesign/`](redesign/)：历史设计文档（architecture/multistep/decomposition/probabilistic 等），仅用于 Git/决策溯源；
- [`TODO_OPTIM.md`](TODO_OPTIM.md)：工程问题台账（OPT-001~022，已收口，OPT-022 天气 as-of 资格待处理）；
- `arch/`、`books/`：架构图与参考资料 PDF。
