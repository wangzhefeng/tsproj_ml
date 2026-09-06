# scripts

项目级只读审计入口：

| 脚本 | 作用 |
|---|---|
| `check_model_configs.py` | 用生产 parser、FeatureCompiler 和策略合同校验 5,150 个活动模型 YAML |
| `audit_forecast_configs.py` | 用生产 parser 输出 5,150 行 typed catalog（5,072 ForecastConfigSpec + 78 Ensemble），含 identity、fingerprint、概率与带单位几何 |
| `audit_ensemble_configs.py` | 校验 78 个 Ensemble、228 个成员引用、四方法分布、15min 场景无重复 member 与 OOF 孤儿缓存 |
| `audit_runtime_assets.py` | 校验所有活动 source 文件及非 ignored 声明列存在；缺失时退出 1 |
| `audit_aidc_load_15min_designs.py` | 对三个 AIDC 15min 场景全部 4,617 份活动单模型编译一个真实训练设计；成本很高，静态-only 任务禁止运行 |
| `generate_load_15min_matrix.py` | 默认只读校验三个 AIDC 15min 场景的 4,689 份矩阵；`--write` 确定性重建 4,617 份单模型与 72 份 add_ensemble |
| `audit_batch_eligibility.py` | 审计全部活动配置的 supervised compiler 批量资格（batch eligible / fallback 及原因） |
| `export_chinese_holiday_csv.py` | 用 `data_loading/calendar_generator/chinese_holiday.py` 同一实现导出中国节假日特征 CSV（generated source 的审计兜底 / 版本固定 file source） |

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_model_configs.py
env -u PYTHONPATH .venv/bin/python scripts/audit_forecast_configs.py --output /tmp/forecast_catalog.json
env -u PYTHONPATH .venv/bin/python scripts/audit_runtime_assets.py
env -u PYTHONPATH .venv/bin/python scripts/audit_aidc_load_15min_designs.py
env -u PYTHONPATH .venv/bin/python scripts/audit_ensemble_configs.py
```

迁移期一次性脚本不驻留本目录，历史通过 Git 追溯。
