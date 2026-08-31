# tsproj-ml

基于机器学习回归器的时间序列多步预测项目。项目只接受 canonical schema-2 YAML，支持 Local/Global、单/多目标、七种标准多步策略、point/边际 quantile 和引用式模型融合。

## 当前能力

- 七种多步策略：`recursive/direct/mimo/recmo/dirrec/dirmo/dirrecmo`。
- 统一预测张量：point `(N,H,K)`，边际 quantile `(N,H,K,Q)`。
- 显式数据角色：`target/observed_past/known_future/static/key/ignored`。
- 严格 information set：动态 source 执行 `available_at <= forecast_origin`，缺失、重复和越界直接 RAISE。
- 模型：LightGBM、XGBoost、CatBoost、RandomForest、HistGradientBoosting、Ridge、ElasticNet、Lasso、QuantileRegressor、SeasonalTemplate。
- 多目标 adapter：independent、regressor-chain、native capability probe；不支持时 RAISE。
- 目标变换：calendar normalization → decomposition → scaling，按 `(series_id,target)` 隔离并严格逆序恢复。
- 引用式 Ensemble：averaging、weighted、linear blending、stacking；成员 OOF、内容寻址缓存、自包含 bundle。
- 点评估：MAE/RMSE/MAPE/Accuracy、seasonal-naive、eval mask。
- 概率评估：pinball、coverage、width、Winkler、coverage gap。

不支持：联合轨迹样本生成、ensemble-of-ensemble、canonical CQR runtime。现有 `probabilistic.calibration` 只保留数学内核，不代表生产已输出 `predict_pi*`。

## 当前架构

```text
入口/分派
  main.py / run.py / config/config_loader.py
        │
        ├── model_forecasting/       单模型生命周期、推理、结果持久化
        └── model_ensemble/          OOF、融合器、Ensemble bundle/结果
                 │
                 ▼
  probabilistic/                     quantile 训练与独立后处理
                 │
                 ▼
  data_loading/ → feature_engineering/ → model_training/
                         ├── model_testing/
                         └── model_evaluation/
                 │
                 ▼
  forecasting_core/                  specs/tensors/artifacts/probabilistic contracts
  models/ decomposition/ data_process/
                 │
                 ▼
  utils/
```

包间依赖为 DAG；`forecasting_core/` 不依赖任何流水线或运行时包。分层由 `tests/test_package_layering.py` AST 门禁固化。

## 目录职责

| 路径 | 职责 |
|---|---|
| `forecasting_core/` | Forecast/Data/Feature/Strategy/Estimator specs，预测张量，bundle/distribution/probabilistic spec |
| `data_loading/` | SourceRegistry、information set、显式 provider |
| `feature_engineering/` | FeatureCompiler、监督特征选择、transform 配置归一化 |
| `model_training/` | CanonicalTrainer、七策略 executor、能力探测与多目标 adapter |
| `model_testing/` | fixed-step/calendar-month 折、actual、seasonal-naive、origin 原语 |
| `model_evaluation/` | 点预测与边际 quantile 指标、eval mask |
| `model_forecasting/` | 单模型回测、final fit、forecast、bundle 与 long result |
| `probabilistic/` | quantile 训练、crossing/CQR 独立后处理工具 |
| `model_ensemble/` | 引用解析、OOF、四种融合方法、缓存、持久化 |
| `models/` | estimator factory 与底层 pickle IO |
| `decomposition/` | 趋势/季节/残差分解与恢复 |
| `data_process/` | 进模型前的离线聚合、填补、异常、事件、周期与峰谷分析 |
| `config/` | 845 个活动模型 YAML + 41 个独立数据工具 YAML；3 个无有效资产配置归档于 `docs/archived_configs/` |
| `scripts/` | 配置、Ensemble 与运行资产审计 |
| `tests/` | unittest、runtime smoke、结构门禁和场景数据链测试 |

`docs/` 下实验性 Python/ipynb 不属于生产代码。

## 配置与时间几何

模型 YAML 顶层分为：

```text
schema_version / problem / data / features / strategy|ensemble /
estimator / probabilistic / validation / output
```

- 单模型使用 `strategy`；Ensemble 使用成员 `config_ref`，两者互斥。
- `problem.horizon` 以 `freq` 为步长单位。
- fixed-step 配置使用 `validation.history_steps/train_window_steps/fold_count/stride_steps`，四者均按监督 origin steps 保存。
- `horizon_mode=calendar_month` 使用 `train_window_days/fold_count/stride_months`；训练窗按原始日数计，每个历史月和最终目标月动态解析 28/29/30/31 步。
- final fit 与回测使用相同训练窗口合同，不再隐式切换为全部历史。

全部 transformation 使用唯一嵌套 schema：

```yaml
features:
  transformations:
    direct:
      layout: independent_models
    advanced:
      rolling:
        columns: [load]
        windows: [96, 192]
        stats: [mean, std]
    target:
      calendar_normalization: {method: none}
      decomposition: {method: none}
      scaling: {method: none, inverse: false}
```

未知顶层或嵌套字段在 `load_yaml_config()` 阶段 RAISE。

## 运行

安装/同步环境：

```bash
uv sync
```

单配置本地入口：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python main.py
```

CLI：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python run.py --config-yaml config/<scenario>/<model>.yaml
```

模型语义以 YAML 为唯一来源；CLI 不支持模型、数据和训练参数的静默覆盖。

## 结果合同

```text
results/<scenario>/<result_identity>/
├── pretrained_models/
│   ├── model.pkl
│   └── resolved_model.json
├── results_test/
│   ├── cv_plot_df.csv
│   ├── test_scores_df.csv
│   ├── test_scores_probabilistic_df.csv   # quantile 模式
│   └── result_metadata.json
└── results_forecast/
    ├── prediction.csv
    └── resolved_config.json
```

- `prediction.csv` 唯一键：`(series_id,time,target)`。
- `cv_plot_df.csv` 唯一键：`(series_id,time,target,window)`。
- 单模型和 Ensemble 都保存 schema-2 `ForecastModelBundle`。
- Ensemble bundle 自包含成员 bundle 和融合器，部署预测不读取成员 YAML 或 OOF cache。

## 验证

```bash
# 全量测试
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"

# 配置、数据资产与 Ensemble 审计
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_runtime_assets.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_ensemble_configs.py

# 语法与格式
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python -m compileall -q \
  forecasting_core data_loading feature_engineering model_training model_testing \
  model_evaluation model_forecasting probabilistic model_ensemble models \
  decomposition data_process utils config scripts tests main.py run.py
git diff --check
```

当前实现与整改状态见 [`docs/architecture_convergence_plan.md`](docs/architecture_convergence_plan.md)。历史方案文档仅用于 Git/决策溯源，不作为当前实现事实源。
