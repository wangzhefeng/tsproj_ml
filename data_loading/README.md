# data_loading

`data_loading/` 是 canonical 唯一数据读取与信息集构造层。

- `registry.py`：文件读取、schema 校验、path version、严格 as-of、缓存。
- `information_set.py`：请求、物化信息集、lineage、行位置缓存。
- `providers.py`：observed-past 显式 future provider。
- `holiday_generator.py`：中国节假日 builtin generator（`chinese_holiday`，chinese-calendar 后端）。YAML 用法见模块 docstring：`source_type: generated` + `availability: generator_defined`，列 `is_holiday / holiday_name / next_holiday_days`；`next_holiday_days` 对超出已知年历的日期取删失哨兵 400。审计兜底 CSV 导出：`scripts/export_chinese_holiday_csv.py`（同一实现，逐点一致有测试保障）。

只依赖 `forecasting_core.specs` 与基础库。缺失、重复、时间越界和可得性违规直接 RAISE；离线填补和清洗属于 `data_process/`。

## 调用入口与输出

上层根据 `DataSpec` 构造 `SourceRegistry`，用 `InformationSetRequest` 表达预测原点、目标时间网格和读取角色，再调用 `materialize()`。输出 `MaterializedInformationSet` 按 target history、observed-past、known-future、static 分区，并携带 `SourceLineage` 与行位置查询缓存；它不是已经拟合或缩放的训练矩阵。

## 信息边界

- `data.sources` 是多文件接入入口。`columns` 是信息投影视图：未声明的物理列不进入模型，声明且非 ignored 的缺失列直接报错。
- `history_path/backtest_path/future_path` 区分历史事实、回测时已发布预报与正式未来输入；不能把回测期实测天气冒充 known-future。
- `target_access=supervised_labels` 仅在训练取标签时放开预测期 target，不放开其他动态 source 的 as-of 边界。
- observed-past 越过预测原点时必须显式选择 `persistence/auxiliary/provided_scenario`；真实历史 lag 无需用 provider 替代。
- `chinese_holiday` 在 `BUILTIN_GENERATORS` 注册；超出后端年历覆盖范围报错，不能生成伪节假日。

## 定向验证

从项目根目录使用已有 `.venv`，不经过 uv 下载缓存：

```bash
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_information_set_registry
env -u PYTHONPATH .venv/bin/python tests/run_suite.py all --match test_holiday_generator
```
