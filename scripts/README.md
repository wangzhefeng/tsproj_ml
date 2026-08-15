# scripts 说明

`scripts/` 保存 `main.py` 本地直跑脚本和少量维护说明。

## 文件职责

| 文件 | 职责 |
|---|---|
| `check_model_configs.py` | 模型配置 dry-run 校验：只加载配置 + 复算 `main.py` 合法性校验（窗口/滞后/互斥组合），不训练不预测 |
| `run_model_test.sh` | 直接触发一次 `main.py` 本地运行 |
| `template.sh` | 可复制修改的 `main.py` 直跑模板 |
| `production_sync.md` | 项目 1 与生产包同步边界记录 |

## 配置 dry-run 校验

批量校验 YAML 配置（如改完一批配置后、或生成新配置后）：

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync python scripts/check_model_configs.py 'config/<scenario>/route_*/lgbm_*.yaml'
```

退出码：0 = 全部通过（可能有提示性警告），1 = 存在硬校验失败。注意用 `env -u PYTHONPATH` 防止外部环境的 PYTHONPATH 遮蔽项目 `utils/` 包。

## 运行模板

`run_model_test.sh` 和 `template.sh` 都不再拼装命令行参数。使用前先在 [main.py](/Users/wangzf/projects/tsproj_ml/main.py) 中修改 `CONFIG_YAML`，以及在对应 YAML 中手动调整模型与运行配置，然后直接执行脚本即可。

## 注意事项

- `run_model_test.sh` 和 `template.sh` 会触发 `main.py`，并产生 `logs/`、`results/` 等运行产物。
