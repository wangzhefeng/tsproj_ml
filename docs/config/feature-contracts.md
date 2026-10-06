# 特征工程严格边界与兼容性

## 当前合同

- 配置构造期检查高级变换字段、操作和严格数值类型；未知字段不忽略，小数整数参数不截断。
- 编译器构造期登记输出名称并校验引用；不允许覆盖既有特征或时间/序列元数据。
- batch 只在同一物化快照和相同历史下界内共享历史；其他请求分组计算。发布时间列历史统计不支持批内共享时明确回退 single。
- DataFrame 缩放输入必须列名唯一且与拟合 schema 相同，允许按名重排；数组按明确的列序合同消费。
- 特征 scaling 仅接受 method/encode_categorical；grouped 已退役，始终逐列拟合。
- 目标 scaling 仅接受 method；inverse 已退役，输出始终按 scaler → decomposition → calendar 恢复原单位。
- 普通交互 divide 不加 epsilon；零分母与非有限结果报错。
- 固定 rolling 窗口不足报错；std/max_diff/min_diff 至少2个样本、skew至少3个、kurt至少4个，未定义统计不填零。
- FFT 偶数窗 Nyquist 不翻倍；单边能量中普通正频权重2、Nyquist权重1。

## 兼容性

这些是显式语义修正，不是无行为卫生改动。活动 YAML 中 inverse 字段移除会改变相应配置身份；存量结果不删除、不自动重训。合法未受影响的配置与数值仍须旧新对照。

公开编译器与转换器的错误输入更早失败；测试应在新 owning 边界验证相同错误原因，不依赖在运行中段才报错。目录重组与类路径迁移另按实施记录验收，未经证据不得声称旧 bundle 兼容。

## 有限窗扩展

在 `features.transformations.advanced` 中声明：

```yaml
rolling_quantile:
  columns: [load]
  windows: [16]
  quantiles: [0.25, 0.75]
lagged_rolling:
  columns: [load]
  windows: [16]
  offsets: [0, 4]
  stats: [mean, std]
```

- 分位窗含原点，使用 pandas 线性插值；quantiles 在 [0,1] 内且唯一。列名如 `load_rolling_quantile_0.25_16`。
- 偏移单位为样本步：offset=4 的窗口在原点前4步结束，需要 window+offset 个历史点；非负整数、完整窗口，不借未来 provider 补齐。列名如 `load_rolling_mean_16_offset_4`。
- 可用既有 `transformations.interactions` 或 advanced interaction 引用这些列组合；引用必须已生成，不能形成环或覆盖同名列。
- single 与 batch 共用有限窗内核；新增窗口在 batch 按信息集/原点计算，indexed 显式不准入，不假装已有向量化快路径。在线历史保留长度沿用同一 requirements 计算。

## 监督选择

每个 call 只对其预测坐标中的 horizon/target 通道评分，先把各通道分数归一到单位最大值，再等权累计，稳定排序选择统一列子集。f_regression 使用裁剪到理论范围的相关系数推导相同排序的 F 比值，避免完全相关时舍入出负分；mutual_info 固定 random_state=0。常数通道不贡献分数，force_keep 在任何规模下均检查名称。不会把不同量纲或相反符号的物理目标先求平均。

## 派生证据与迁移

bundle 的 feature_lineage 包含 derivation（操作、输入引用类别、参数、规则身份）以及 available_at_upper_bound。该时间是编译可见性给出的保守上界，不是重新推算的真实发布时间。外生发布事实仍由原始 source/proof 保留。

模块已分为 compilation/kernels/statistics/transforms；raw-design 全链身份在 pipeline/design_identity.py，监督标签在 pipeline/labels.py。旧模块文件已移除，旧 pickle 路径不提供兼容壳；保留旧产物，使用旧代码读取或显式授权重训。

在线恢复额外校验 EWM 数值范围、事件位置/尾部、expanding 范围及（保存时存在的）完整前缀一致性；这不是任意篡改检测，未保存完整历史的有界状态无法单靠快照证明所有有限值真实。
