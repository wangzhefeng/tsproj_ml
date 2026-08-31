# tests

`tests/` 是版本控制内的 unittest 套件，覆盖 core contracts、信息集、特征、七策略、Local/Global、point/quantile、fixed/calendar/monthly runtime、Ensemble、结果 schema、场景数据链和包间结构。

```bash
env -u PYTHONPATH UV_CACHE_DIR=.uv_cache uv run --no-sync \
  python -m unittest discover -s tests -p "test_*.py"
```

约定：

- 新行为先 RED 再实现；
- runtime 产物使用临时目录；
- `tests/test_package_layering.py` AST 扫描所有 import，包括函数内 import；
- `tests/test_active_config_runtime_contract.py` 保证活动 YAML 与生产 grammar/时间几何一致；
- `tests/test_runtime_asset_audit.py` 要求活动 source 零缺失；
- 删除测试前必须先证明对应生产链已删除或由更高层测试覆盖。

最新精确测试数以本次全量命令输出为准，不在多份文档重复维护。
