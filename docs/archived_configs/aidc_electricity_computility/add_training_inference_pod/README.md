# Inactive computility configs

这 3 个 schema-2 YAML 于 2026-08-31 经 wangzf 明确授权，从活动 `config/` 集合迁出：

- `lgbm_msmd_a.yaml`
- `lgbm_msmdr_a.yaml`
- `lgbm_msmr_a.yaml`

原因：三者声明的是 `*_all_jobs_*` 聚合特征，但共同引用的
`dataset_electricity_with_computility_A.csv` 使用另一套 pod/资源列族；声明列在物理资产中不存在，无法通过 canonical `SourceRegistry` 合同。

文件按字节原样保留，仅用于历史溯源。没有确认正确数据资产或完成列语义迁移前，不得移回活动 `config/`，也不得通过放松缺列校验使其运行。
