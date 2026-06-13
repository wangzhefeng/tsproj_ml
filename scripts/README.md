# scripts 说明

`scripts/` 保存运行模板和少量维护说明。正式模型运行仍建议通过 `run.py` 完成；配置生成器已移动到 `config/generate_configs.py`。

## 文件职责

| 文件 | 职责 |
|---|---|
| `run_model_test.sh` | 通过环境变量覆盖配置并启动一次测试运行 |
| `template.sh` | 可复制修改的 `run.py` 命令模板 |
| `production_sync.md` | 项目 1 与生产包同步边界记录 |

## 运行模板

`run_model_test.sh` 和 `template.sh` 的默认配置为：

- `CONFIG_YAML=`（为空时使用 `CONFIG_MODULE`）
- `CONFIG_MODULE=config.univariate_config`
- `CONFIG_CLASS=ModelConfig`

可用环境变量覆盖：

```bash
CONFIG_YAML=config/ETT-small/ETTm1/lgbm_msmr.yaml \
CONFIG_CLASS=ModelConfig \
MODEL_TYPE=xgboost \
PRED_METHOD=multivariate-single-multistep-direct \
bash scripts/run_model_test.sh
```

## 注意事项

- `config/generate_configs.py` 会写入配置文件；生成前优先使用 `--dry-run`。
- `run_model_test.sh` 和 `template.sh` 会产生 `logs/`、`saved_results/` 等运行产物。
- `.DS_Store` 和 `__pycache__/` 不是源码或数据契约。
