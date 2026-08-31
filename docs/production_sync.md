# Production Sync

> 文档状态：当前同步边界（2026-08-31）。历史逐次同步记录通过 Git 溯源，不再作为当前模块清单。

## 1. 边界

`tsproj_ml` 是预测核心实现。生产包只实现平台 adapter、部署入口和生产 I/O，不复制第二套算法或配置 grammar。

同步时必须保持以下合同整体一致：

- schema-2 typed config；
- information set/as-of；
- fixed/calendar/monthly 时间几何；
- feature/target transform；
- 七策略与 estimator capability；
- point/quantile 张量；
- canonical long result 与 schema-2 bundle；
- 引用式 Ensemble 的成员、OOF、cache 和自包含产物。

禁止只复制单个 executor、指标函数或 YAML 字段后继续使用生产侧旧路由。

## 2. 当前可同步模块

| 模块 | 职责 | 同步要求 |
|---|---|---|
| `forecasting_core/` | specs、tensors、artifacts、probabilistic contract | 必须先同步；下游公共合同 |
| `data_loading/` | SourceRegistry、InformationSet、providers | 与 DataSpec、source 资产规则一起同步 |
| `feature_engineering/` | FeatureCompiler、visibility/lineage、selection | 不得绕过可见性证明 |
| `model_training/` | Trainer、七策略、estimators/adapters/capabilities | 不得静默能力降级 |
| `model_testing/` | fixed/calendar folds、actual、seasonal-naive、origin | 与 runtime 时间几何同版本 |
| `model_evaluation/` | point/marginal metrics、eval mask | 保持有效点口径一致 |
| `model_forecasting/` | design、fit、backtest、forecaster、transforms、persistence、results、runtime | 作为单模型生命周期整体同步 |
| `probabilistic/` | quantile training、crossing、独立 calibration 数学内核 | canonical CQR runtime 仍不支持 |
| `model_ensemble/` | loader、OOF、cache、四方法、bundle/runtime | 必须同步 services 注入与成员 bundle |
| `models/` | estimator factory、pickle IO | 只作底层基础设施 |
| `decomposition/` | target decomposition/restore | 与 target transform 一起同步 |
| `utils/frequency.py` | 频率和月步解析 | 月频不得改为固定 Timedelta |

## 3. 不同步内容

- 生产 API 主类、平台父类和鉴权逻辑；
- 生产路径、生产字段映射和部署框架；
- dataset、results、日志和本地场景输出；
- `config/aidc_electricity_computility/.../scripts/` 等场景数据准备脚本；
- `docs/` 下实验性 Python/ipynb；
- 已删除的 legacy 配置、九方法、旧 pkl/宽表 reader、PMML、在线窗口清洗链。

生产侧如必须读取历史产物，应维护独立 legacy adapter；不得把兼容层回灌本仓库。

## 4. 当前能力限制

- 只支持 point 与边际 quantile；不支持联合轨迹样本生成。
- canonical runtime 不执行 CQR，不输出 `predict_pi*`。
- 不支持 ensemble-of-ensemble。
- 缺失/异常默认 RAISE；清洗发生在进入模型之前。
- 不允许伪造未来 date/weather/source 资产以通过审计。

## 5. 同步流程

1. 记录源仓库 commit/工作树版本和同步范围。
2. 先同步 `forecasting_core/`，再按依赖方向同步流水线与 runtime。
3. 用生产 adapter 构造同一 schema-2 config；禁止维护第二套字段白名单。
4. 对 point、quantile、calendar-month、月频、Ensemble 和 Global N2K2 各选最小代表。
5. 同时核对回测、forecast、bundle、resolved config/model 和 long schema。
6. 运行生产侧全量测试；仅 parser/checker 通过不算完成。

## 6. 同步记录模板

```markdown
### YYYY-MM-DD change-id

- tsproj_ml 来源：<commit / reviewed worktree>
- 同步模块：
- 生产 adapter 变更：
- 明确未同步：
- 行为变化与旧结果影响：
- 验证命令与真实结果：
- 未验证风险：
```

当前架构和验收状态以 `docs/multistep_forecasting_redesign.md` 与 `docs/architecture_convergence_plan.md` 为准。
