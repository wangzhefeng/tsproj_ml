# 结果合同

```text
results/
├── pretrained_models/<scenario>/<result_identity>/
│   ├── model.pkl
│   └── resolved_model.json
├── results_test/<scenario>/<result_identity>/
│   ├── cv_plot_df.csv
│   ├── test_scores_df.csv
│   ├── test_scores_probabilistic_df.csv   # quantile 模式
│   └── result_metadata.json
└── results_forecast/<scenario>/<result_identity>/
    ├── prediction.csv
    └── resolved_config.json
```

- `prediction.csv` 唯一键：`(series_id,time,target)`。
- `cv_plot_df.csv` 唯一键：`(series_id,time,target,window)`。
- 单模型和 Ensemble 都保存 schema-2 `ForecastModelBundle`。
- wrappers 模块迁移后，旧 `models.ModelFactory` 路径 bundle 不再兼容，不提供 shim；需重新训练，现有结果不自动删除。见 [`packages/models.md`](packages/models.md)。
- Ensemble bundle 自包含成员 bundle 和融合器，部署预测不读取成员 YAML 或 OOF cache。

identity 规则（`<method_label>-<model_type>-<training_scope>-k<K>-<fingerprint前12位>`）与 long 结果 schema 的唯一入口是 [`packages/model_forecasting.md`](packages/model_forecasting.md)。
