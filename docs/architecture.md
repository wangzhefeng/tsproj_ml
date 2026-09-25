# 当前架构与目录职责

代码接线完成不代表真实模型/部署验收完成。

```text
入口/分派
  run.py / batch_run.py / config/config_loader.py
        │
        ├── model_pipeline/          单模型生命周期、监督设计、批调度
        └── model_ensemble/          OOF、融合器、自包含 bundle；执行服务由入口注入
                 │
                 ▼
  data_loading/ → feature_engineering/ → model_training/
                         │ transforms/     │ quantile/七策略
                         ├── model_testing/       回测几何、评分与产物
                         ├── model_forecasting/   预测、部署与 bundle/结果
                         ├── model_performance/   资源、性能档与运行缓存
                         └── probabilistic/       仅 CQR 校准
                 │
                 ▼
  forecasting_core/                  specs/tensors/artifacts/probabilistic contracts
  models/ model_evaluation/ decomposition/ data_process/
                 │
                 ▼
  utils/
```

包间依赖为 DAG；`forecasting_core/` 不依赖任何流水线或运行时包。分层由 `tests/test_package_layering.py` AST 门禁固化。各包职责细节与边界声明见 [`packages/`](packages/) 对应文档。

## 目录职责

| 路径 | 职责 |
|---|---|
| `forecasting_core/` | Forecast/Data/Feature/Strategy/Estimator specs，预测张量，bundle/distribution/probabilistic spec |
| `data_loading/` | SourceRegistry、information set、显式 provider |
| `feature_engineering/` | FeatureCompiler、监督特征选择、transform 配置归一化及 `transforms/` 训练态 |
| `model_training/` | CanonicalTrainer、quantile 训练、七策略 executor、能力探测与多目标 adapter |
| `model_testing/` | fixed-step/calendar-month 几何、actual/seasonal-naive、逐折评分与回测产物 |
| `model_evaluation/` | 点预测与边际 quantile 指标、eval mask |
| `model_pipeline/` | 单模型生命周期、监督设计、fold/final fit 编排、批调度与验收 |
| `model_forecasting/` | point/quantile 预测、crossing、部署、bundle 与预测 long result |
| `model_performance/` | 资源规划、性能档、checkpoint、变换缓存与有界内存缓存 |
| `probabilistic/` | CQR 校准内核与 apply-before-collect 追踪器 |
| `model_ensemble/` | 引用解析、OOF、四种融合方法、缓存、持久化 |
| `models/` | catalog、factory、按 family 分组的 wrappers 与底层 pickle IO |
| `decomposition/` | 趋势/季节/残差分解与恢复 |
| `data_process/` | 进模型前的离线聚合、填补、异常、事件、周期与峰谷分析 |
| `config/` | 活动模型 YAML 唯一场景为 `aidc_load_15min_short`（171 份 LightGBM 单模型）+ 3 份独立数据工具 YAML；2026-09-25 场景收敛前的历史场景配置从 Git 溯源 |
| `scripts/` | 配置、Ensemble 与运行资产审计 |
| `tests/` | unittest、runtime smoke、结构门禁和场景数据链测试 |
