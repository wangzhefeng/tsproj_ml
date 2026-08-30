# scripts 模块说明

`scripts/` 保存项目级维护工具，全部为只读审计或受控写出，不接入运行时主链。

## 文件职责

| 脚本 | 作用 | 使用时机 |
|---|---|---|
| `check_model_configs.py` | 配置体检：加载全部模型 YAML 并复算 `main.py` 启动时的合法性校验（窗口/lag/策略/概率支持等），不训练不预测 | 改任何配置或解析器后必跑；当前口径 `checked=824 passed=824 hard_failures=0 warnings=0` |
| `audit_forecast_configs.py` | 配置盘点：只读扫描模型 YAML，输出方法分布、canonical identity 分布等确定性 JSON | 需要回答「当前有多少种配置组合」时 |
| `generate_result_invalidation_manifest.py` | 失效清单生成：为每个配置计算旧结果路径、新 fingerprint、新 run path 与失效原因，`--csv-output` 指定输出路径（清单为可再生快照，不随仓库驻留） | 架构语义变更或旧结果清理后需要审计时再生 |

## 用法

```bash
# 全量配置校验（默认 glob config/**/*.yaml，只识别模型 schema）
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py

# 配置审计
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/audit_forecast_configs.py --root config --output /tmp/forecast-config-audit.json

# 从 git 历史复演迁移（只读审计，不写 config/）
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python scripts/migrate_forecast_configs.py --root config --revision HEAD \
  --repository . --report /tmp/migration_report.json
```

## 边界

- 一代一用的临时脚本（如分解矩阵派生/汇总）用后即删，历史从 git 溯源，不留在本目录。
- 场景级数据准备脚本不放这里：数据派生入口在 `data_process/` 与各 `config/<scenario>/` 目录。
