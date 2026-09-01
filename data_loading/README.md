# data_loading

`data_loading/` 是 canonical 唯一数据读取与信息集构造层。

- `registry.py`：文件读取、schema 校验、path version、严格 as-of、缓存。
- `information_set.py`：请求、物化信息集、lineage、行位置缓存。
- `providers.py`：observed-past 显式 future provider。
- `holiday_generator.py`：中国节假日 builtin generator（`chinese_holiday`，chinese-calendar 后端）。YAML 用法见模块 docstring：`source_type: generated` + `availability: generator_defined`，列 `is_holiday / holiday_name / next_holiday_days`；`next_holiday_days` 对超出已知年历的日期取删失哨兵 400。审计兜底 CSV 导出：`scripts/export_chinese_holiday_csv.py`（同一实现，逐点一致有测试保障）。

只依赖 `forecasting_core.specs` 与基础库。缺失、重复、时间越界和可得性违规直接 RAISE；离线填补和清洗属于 `data_process/`。
