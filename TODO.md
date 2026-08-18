1. 总用电量月度频率预测
    2. 压力测试标签、上下架识别；
    3. 去趋势化；
    4. A1, A3 服务器功耗，温度、大功率暖通负荷、IT负荷；
       - IT + 暖通（所有暖通相关负荷） + 总负荷 -> 总负荷
    5. 评价指标；
    6. 层级数据
        - 1. 列头柜负荷 -> 服务器功耗
        - 2. UPS 负荷
        - 3. 机房负荷
        - 4. 暖通负荷
        - 5. 气象数据
    7. 缩短历史数据；
2. 站用电
    - 典型曲线方法
      - baseline 测试：无 date, weather, plan_strategy 特征
      - add_exogenous_all: 加 date, weather, plan_strategy 特征
    - 非典型日处理、异常处理；特殊事件识别、捕捉
      - baseline 测试：无 date、weather、plan_strategy 标签、actual_strategy 标签特征
      - add_exogenous_all：加 date、weather、plan_strategy 标签、actual_strategy 标签特征
      - add_endogenous_exogenous：加 date、weather、plan_strategy 标签、actual_strategy 标签特征
    - EMS：两个源数据是否时间戳对齐，是否由于没对齐导致了异常值；站用电的影响因素
    - 非典型日：
      - 2026-07-23
      - 2026-07-17
      - 2026-07-16
      - 2026-07-10
