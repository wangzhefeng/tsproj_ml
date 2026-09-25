# docs/packages — 包文档

每包一个 md，自原包目录 README 迁入（原位置不再保留文档）。内容含各包职责、模块清单与包间边界声明；分层总图见 [`../architecture.md`](../architecture.md)，分层门禁不变量见根目录 `AGENTS.md`。

## 合同与基础层

| 文档 | 包 |
|---|---|
| [forecasting_core.md](forecasting_core.md) | 稳定预测合同层（不依赖任何项目包） |
| [utils.md](utils.md) | L0 基础工具（不依赖任何项目包） |
| [ts_kernels.md](ts_kernels.md) | 低层时序统计算法库 |
| [models.md](models.md) | estimator factory、catalog、wrappers、pickle IO |
| [model_evaluation.md](model_evaluation.md) | 生产评估公式 |
| [decomposition.md](decomposition.md) | 目标分解与分量外推（含 8 个子包职责明细） |

## 数据与特征

| 文档 | 包 |
|---|---|
| [data_loading.md](data_loading.md) | 数据读取与信息集构造 |
| [weather_generator.md](weather_generator.md) | 气象生成（研究回放链，非活动链） |
| [feature_engineering.md](feature_engineering.md) | 特征编译与变换（含 transforms 子包节） |
| [data_process.md](data_process.md) | 离线数据准备工具链 |

## 训练、测试与产物

| 文档 | 包 |
|---|---|
| [model_training.md](model_training.md) | 训练器与七策略 executor |
| [model_testing.md](model_testing.md) | 回测几何、评分与产物 |
| [model_pipeline.md](model_pipeline.md) | 生命周期编排与批调度 |
| [model_forecasting.md](model_forecasting.md) | 预测、部署与 bundle 持久化 |
| [model_performance.md](model_performance.md) | 资源规划、性能档、checkpoint |
| [probabilistic.md](probabilistic.md) | CQR 校准内核 |
| [model_ensemble.md](model_ensemble.md) | 引用式模型融合 |
