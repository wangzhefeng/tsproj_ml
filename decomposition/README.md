# decomposition

目标序列分解与结构分量外推算法库。只处理调用方提供的一条可见历史序列，不选择回测折、监督标签、输出目录或模型策略。

## 执行职责

`construction/registry` 构造 Pipeline；Pipeline 调用 Extractor → Forecaster.fit/predict → Composer。真实历史结构和用于历史变换，未来结构和用于预测恢复。残差预测由上层主模型完成。

| 位置 | 职责 |
|---|---|
| `configuration/spec.py` | 唯一配置校验、quadratic/damped 别名归一化、DecompositionSpec |
| `configuration/presets.py` | 已实现 preset 的参数类型与固定组件配方 |
| `construction/component_factory.py` | 按配方构造组件，不引用 Pipeline |
| `construction/registry.py` | 唯一配置到 Pipeline 构造入口 |
| `contracts/types.py` | 原始目标、真实趋势、按周期季节分量、未来分量；残差和结构和为派生量 |
| `contracts/time_axis.py` | 等间隔时间轴与有限目标验证；趋势时间单位为天 |
| `contracts/interfaces.py` | 历史分解与分量外推抽象接口 |
| `extraction/polynomial.py` | 一次/二次历史趋势提取 |
| `extraction/seasonal.py` | STL/MSTL 真实历史分解，不拟合未来模板 |
| `extrapolation/trend.py` | 多项式拟合、damped 近期斜率与未来趋势 |
| `extrapolation/seasonal.py` | 按最近 seasonal_cycles 个周期拟合相位模板并外推 |
| `composition/additive.py` | 唯一加性合成；point/quantile 不允许广播或截断 |
| `orchestration/pipeline.py` | 拟合/变换/预测恢复协调，公开 history_frame，不搬运具体组件私有属性 |
| `diagnostics/components.py` | summarize_components(history_frame) 返回真实分量统计表 |
| `diagnostics/residuals.py` | 单窗残差频谱、跨窗稳定性及显式汇总表，无文件 IO |

## 配置与数值合同

- YAML：`features.transformations.target.decomposition`。算法 preset 为 none/linear/stl/mstl；quadratic、damped 是归一化到 linear 参数组合的别名。custom、乘性及模态分解未实现，直接拒绝。
- 周期为递增不重复的整数样本步；STL 一个周期、MSTL 至少两个。STL 至少两个完整周期，MSTL 必须超过两个完整周期（statsmodels 会移除不满足此条件的周期，运行时明确拒绝）。
- degree 只能为 1/2，lookback/cycles 为正整数，damping 为有限 (0,1] 数值，robust 必须为 bool。未知键、非法类型、缺失/非有限目标直接拒绝。
- linear+damped 的斜率来自近期原始观测；STL/MSTL 的斜率来自近期真实趋势。默认 lookback=28、cycles=4、damping=0.98；均由外推器实际消费。
- 上游按 `(series_id,target)` 隔离，变换固定 calendar normalization → decomposition → scaling，严格逆序恢复。fit_history_steps 只声明上下文长度，监督窗口选择仍由 feature_engineering/transforms/windows.py 负责。
- `ComponentFrame.target` 不混入 components；trend/seasonal 为真实历史分解，不是外推模板，不再出现重复相加的特殊字典键。

## 包间边界

- 直接运行时消费者：feature_engineering/transforms；配置消费者：feature_engineering/transform_specs 与 scripts/check_model_configs。
- FFT/ACF/STL 强度算法位于 `timeseries_analysis/`；离线 DataFrame 排序、时间换算与周期报告位于 data_process/periodicity_analysis.py。
- 报告写入位于 model_testing/decomposition_reports.py，只接受已计算 DataFrame，独占创建、不覆盖。诊断入口不在包根导出；本次不自动给所有训练/回测接入新增报告。
- 主模型训练、多序列张量、监督窗口、缩放、日历归一化、bundle 持久化不属于本包。

## 语义与产物迁移

本次修复 STL/MSTL 的非默认 trend_degree/trend_lookback/seasonal_cycles 未进入实际拟合的缺陷。所有启用分解的配置 semantic payload 增加 `decomposition_semantics: component_fit_v2`；不修改 YAML、不自动重跑、不删除存量 results。

Pipeline 序列化状态版本为 2。旧状态及已移除的 extractors/forecasters 路径拒绝加载，不提供 shim；需要用户显式重新训练。none 若只是未启用的 canonical identity 变换，不会产生分解对象；若旧 bundle 实际嵌套了旧 none Pipeline，其旧对象布局同样不兼容。

实现/验证记录见 `.hermes/plans/2026-09-07_003942-decomposition-boundaries.md`；原设计历史保留于 docs/redesign/decomposition.md。

## 职责子包约定

根目录只保留公共 API 与总体说明。configuration 解释配置；construction 构造 Pipeline/组件；contracts 提供接口、类型、时间轴合同；extraction/extrapolation/composition 实现算法；orchestration 运行编排；diagnostics 只计算诊断数据。

子包 __init__.py 保持轻量，不自动导入 Registry。真实依赖为 construction.registry → orchestration.pipeline → construction.component_factory；组件工厂不导入 Pipeline。根 __init__.py 保留既有公开名称，但内部代码使用精确子模块路径。

本次子包化仅调整组织与导入，不再改变算法、默认值、配置或 component_fit_v2 语义指纹。移动类定义使子包化前的 pickle 路径失效；正式加载器明确拒绝，不保留旧路径 shim、不删除旧结果、不自动覆盖或重训。迁移后产物通过保存/加载/预测往返验证。

迁移记录：`.hermes/plans/2026-09-07-decomposition-subpackages.md`。
