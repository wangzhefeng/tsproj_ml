# scripts 说明

`scripts/` 保存 `main.py` 本地直跑脚本、AIDC 专项处理脚本和少量维护说明。配置生成器已移动到 `config/generate_configs.py`。

## 文件职责

| 文件 | 职责 |
|---|---|
| `aidc/README.md` | AIDC 算力预处理、结果融合和领导汇报脚本说明 |
| `aidc/` | AIDC 电力/算力数据处理与汇报相关脚本目录 |
| `run_model_test.sh` | 直接触发一次 `main.py` 本地运行 |
| `template.sh` | 可复制修改的 `main.py` 直跑模板 |
| `production_sync.md` | 项目 1 与生产包同步边界记录 |

## 运行模板

`run_model_test.sh` 和 `template.sh` 都不再拼装命令行参数。使用前先在 [main.py](/Users/wangzf/projects/tsproj_ml/main.py) 中修改 `CONFIG_YAML`，以及在对应 YAML 中手动调整模型与运行配置，然后直接执行脚本即可。

## 注意事项

- `config/generate_configs.py` 会写入配置文件；生成前优先使用 `--dry-run`。
- `run_model_test.sh` 和 `template.sh` 会触发 `main.py`，并产生 `logs/`、`results/` 等运行产物。
- AIDC 专项脚本已统一收敛到 [scripts/aidc/README.md](/Users/wangzf/projects/tsproj_ml/scripts/aidc/README.md) 对应目录下维护。
- `.DS_Store` 和 `__pycache__/` 不是源码或数据契约。
