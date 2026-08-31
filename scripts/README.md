# scripts

项目级只读审计入口：

| 脚本 | 作用 |
|---|---|
| `check_model_configs.py` | 用生产 parser、FeatureCompiler 和策略合同校验 845 个活动模型 YAML |
| `audit_forecast_configs.py` | 用生产 parser 输出 845 行 typed catalog（821 单模型/成员 + 24 Ensemble），含 identity、fingerprint、概率与带单位几何 |
| `audit_ensemble_configs.py` | 校验 24 个 Ensemble、48 个成员引用和 OOF 孤儿缓存 |
| `audit_runtime_assets.py` | 校验所有活动 source 文件及非 ignored 声明列存在；缺失时退出 1 |

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_forecast_configs.py --output /tmp/forecast_catalog.json
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_runtime_assets.py
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/audit_ensemble_configs.py
```

迁移期一次性脚本不驻留本目录，历史通过 Git 追溯。
