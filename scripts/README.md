# scripts

项目级只读审计入口：

| 脚本 | 作用 |
|---|---|
| `check_model_configs.py` | 用生产 parser、FeatureCompiler 和策略合同校验 5,150 个活动模型 YAML |
| `audit_forecast_configs.py` | 用生产 parser 输出 5,150 行 typed catalog（5,072 ForecastConfigSpec + 78 Ensemble），含 identity、fingerprint、概率与带单位几何 |
| `audit_ensemble_configs.py` | 校验 78 个 Ensemble、228 个成员引用、四方法分布、15min 场景无重复 member 与 OOF 孤儿缓存 |
| `audit_runtime_assets.py` | 校验所有活动 source 文件及非 ignored 声明列存在；缺失时退出 1 |
| `audit_aidc_load_15min_designs.py` | 对三个 AIDC 15min 场景全部 4,617 份活动单模型编译一个真实训练设计；成本很高，静态-only 任务禁止运行 |
| `generate_load_15min_matrix.py` | 默认只读校验三个 AIDC 15min 场景的 4,689 份矩阵；`--write` 确定性重建 4,617 份单模型与 72 份 add_ensemble |
| `audit_batch_eligibility.py` | 审计全部活动配置的 supervised compiler 批量资格（batch eligible / fallback 及原因） |
| `export_chinese_holiday_csv.py` | 用 `data_loading/calendar_generator/chinese_holiday.py` 同一实现导出中国节假日特征 CSV（generated source 的审计兜底 / 版本固定 file source） |

```bash
env -u PYTHONPATH .venv/bin/python scripts/check_model_configs.py
env -u PYTHONPATH .venv/bin/python scripts/audit_forecast_configs.py --output /tmp/forecast_catalog.json
env -u PYTHONPATH .venv/bin/python scripts/audit_runtime_assets.py
env -u PYTHONPATH .venv/bin/python scripts/audit_aidc_load_15min_designs.py
env -u PYTHONPATH .venv/bin/python scripts/audit_ensemble_configs.py
```

迁移期一次性脚本不驻留本目录，历史通过 Git 追溯。

## 天气资产准备

`build_scenario_weather.py` 会传递共享源旁 `.six_features_repair.json` 的内容哈希与补值性质，重建不能丢失再分析替代证据。三个15分钟负荷场景的 `generate_load_15min_matrix.py` 天气 source 同步使用六项及实测/预报映射，避免重建配置退回露点五项或旧分段路径。六项范围与排除场景见 `config/README.md`。

> 本节 generated weather 工具链现属研究回放/取证用途。活动配置走 file + `inference_columns`：`build_scenario_weather.py` 构建截至 2026-08-31 的完整 history，训练用实测列、历史测试预测用预报列。当前无真正未来任务，不再把历史留出区间生成或引用为 future；真正未来预报须另行提供。联通使用独立 `liantong_august_prepare.py`。

`prepare_weather.py --help` 提供本地 `archive` 和 `register` 子命令；默认 dry-run，只有显式 `--write` 才发布。不会联网、猜测发布时间或覆盖旧版本。`archive` 只按 SHA 保全原字节；`register` 需要完整 metadata 与哈希证据，生成 normalized 和不可变 manifest。实际参数以各子命令 `--help` 为准。真实源的单位、地点或证据未齐时只能归档，不能注册为合格模型输入。

供应商 CSV 的行级可得性使用 `--available-at-col`；Open-Meteo UTC ISO hourly JSON 使用 `--availability-csv <path>` 提供严格一对一的 `time,available_at` 证据表，该 CSV 必须作为 metadata 的 raw 哈希依赖。两个适配器的参数不能混用。历史 JSON 不根据抓取日期倒推发布时间；没有证据只能保全 raw。

`audit_runtime_assets.py` 已识别 generated weather 并核验 manifest/raw/normalized 的传递哈希，`weather_errors` 非空时 CLI 退出 1。此静态审计不证明逐 origin 覆盖，也不能替代 P6 模型闭环。

`audit_weather_configs.py --root <repository> --report <json>` 是迁移前置只读审计：清点全部模型中的天气消费者，保留配置身份、列、几何与实验定义，核验 generated 资产，列出 legacy/缺资产/尚未逐窗口验收的阻塞。当前不执行训练或逐窗口 materialize；不能将其报告当作完整 P6 窗口验收。存在阻塞或扫描不到模型均返回非零。

`plan_weather_requests.py --inventory <上述审计JSON> --report <json>` 在配置字节哈希仍匹配时读取真实目标覆盖，规划监督候选原点和最终输出包络；包含calendar-month动态horizon及融合成员引用，按相同几何去重。窗口之外的旧时间缺口不扩大检查范围；正式源校验拒绝的业务缺值保持报错。此工具不读取旧天气值、不训练、不枚举OOF逐折/target-history辅助请求，也未把输出标签映射为proxy年份或原生区间依赖；不能直接当作完整下载清单或as-of验收。存在任何配置规划错误时输出部分结果并返回1，禁止把该部分结果称为全量完成。

## 日历与节假日数据导出

跨场景日历、节假日数据文件统一放在 `dataset/shared/holidays/`；生成逻辑仍位于 `data_loading/calendar_generator/`。导出脚本的 `--output` 保持必填，不强制限制目录，测试可使用临时路径。

```bash
env -u PYTHONPATH .venv/bin/python scripts/export_chinese_holiday_csv.py --start 2025-01-01 --end 2026-12-31 --output dataset/shared/holidays/chinese_holiday_20250101_20261231.csv
```

存入该目录不会自动入模。使用 CSV 的配置需显式声明对应 source 路径、列角色和可得性；在线 generated source 不依赖此目录，无需改为 file source。

推荐完整导出覆盖2025-01-01至2026-12-31（日频730行）；原2025-10-01起的短范围文件保留，不将旧文件名用于承载新的日期范围。
