# data_process

`data_process/` 是进模型前的离线数据准备工具库：频率聚合、填补方法回测、异常分析、事件检测、峰谷和周期分析。

进入 canonical runtime 的信息集默认缺失/异常即 RAISE；本目录工具只生成新的离线 CSV，不在训练窗口内静默改值。已删除未接生产的 legacy 训练窗口清洗实现。

改动数据链时必须先确认 source、派生产物、配置引用和审计 sidecar，再按依赖顺序重建。
