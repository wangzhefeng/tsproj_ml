# scripts 说明

`scripts/` 保存配置生成、运行模板和维护脚本。正式模型运行仍建议通过 `run.py` 完成。

## 文件职责

| 文件 | 职责 |
|---|---|
| `generate_configs.py` | 从 dataset 叶子目录批量生成分组 YAML 配置 |
| `run_model_test.sh` | 通过环境变量覆盖配置并启动一次测试运行 |
| `template.sh` | 可复制修改的 `run.py` 命令模板 |
| `production_sync.md` | 项目 1 与生产包同步边界记录 |
| `rm_dsstore.sh` | 清理已跟踪的 `.DS_Store` 文件 |
| `update_codes.sh` | 旧式 git 提交/拉取/推送辅助脚本 |

## 配置生成

先用 `--dry-run` 检查将写入的配置路径：

```bash
uv run python scripts/generate_configs.py \
  --dataset dataset/aidc_electricity_computility/electricity/2026-06-11/demand_load/A1_201 \
  --target h_total_use \
  --models lightgbm,xgboost,catboost \
  --strategies usmdo,usmd,usmr,usmdr \
  --variant A1_201 \
  --config-dir config/aidc_electricity_computility/electricity/2026-06-11/A1_201 \
  --dry-run
```

确认后去掉 `--dry-run`。

默认输出为 YAML；需要旧 Python dataclass 配置时显式传 `--format python`。

## 运行模板

`run_model_test.sh` 和 `template.sh` 的默认配置为：

- `CONFIG_YAML=`（为空时使用 `CONFIG_MODULE`）
- `CONFIG_MODULE=config.univariate_config`
- `CONFIG_CLASS=ModelConfig`

可用环境变量覆盖：

```bash
CONFIG_YAML=config/ETT-small/ETTm1/model_config_lgbm_msmr.yaml \
CONFIG_CLASS=ModelConfig \
MODEL_TYPE=xgboost \
PRED_METHOD=multivariate-single-multistep-direct \
bash scripts/run_model_test.sh
```

## 注意事项

- `generate_configs.py` 会写入配置文件；生成前优先使用 `--dry-run`。
- `run_model_test.sh` 和 `template.sh` 会产生 `logs/`、`saved_results/` 等运行产物。
- `update_codes.sh` 涉及 git 操作，使用前必须人工确认当前分支和 diff。
- `.DS_Store` 和 `__pycache__/` 不是源码或数据契约。
