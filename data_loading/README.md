# data_loading

`data_loading/` 是 canonical 唯一数据读取与信息集构造层。

- `registry.py`：文件读取、schema 校验、path version、严格 as-of、缓存。
- `information_set.py`：请求、物化信息集、lineage、行位置缓存。
- `providers.py`：observed-past 显式 future provider。

只依赖 `forecasting_core.specs` 与基础库。缺失、重复、时间越界和可得性违规直接 RAISE；离线填补和清洗属于 `data_process/`。
