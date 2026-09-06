# -*- coding: utf-8 -*-

# ***************************************************
# * File        : pickle_io.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-02-28
# * Version     : 0.1.101716
# * Description : 模型与缩放器 pickle 保存/加载
# ***************************************************


# python libraries
import os
from pathlib import Path

import pickle
import joblib

from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class ModelDeployPkl:
    """
    模型离线部署类

    （2026-08-29 架构收敛 D7：ModelDeploy 抽象基类与 ModelDeployPmml 已删除——
    PMML 部署链路全仓零消费；本类为唯一保留的 pickle 保存/加载入口。）
    """
    def __init__(self, save_file_path: str):
        # 模型保存的目标路径，统一转为字符串以兼容 pathlib.Path
        self.save_file_path = os.fspath(save_file_path)

    def save_model(self, model):
        """
        模型保存: 将训练完成的模型保存为pkl文件

        Args:
            model (instance): 模型实例, sklearn机器学习包实例化后训练完毕的模型

        Raises:
            Exception: [description]
        """
        if not self.save_file_path.endswith(".pkl"):
            raise Exception("参数 save_file_path 后缀必须为 'pkl', 请检查.")

        # F8（架构收敛 D1）：低层不 import 高层，schema gate 改 duck-typing。
        # ForecastModelBundle 契约以 schema_version 属性判定，等价于原 isinstance 检查：
        # bundle 对象必有整型 schema_version（构造期强校验）；普通估计器等非
        # schema-2 bundle 对象按原语义放行或拒绝。
        schema_version = getattr(model, "schema_version", None)
        if (
            isinstance(schema_version, int)
            and not isinstance(schema_version, bool)
            and schema_version != 2
        ):
            raise ValueError(
                "new ForecastModelBundle saver requires schema_version=2; "
                "legacy bundles are read-only"
            )

        with open(self.save_file_path, "wb") as f:
            pickle.dump(model, f, protocol = 2)
        # logger.info(f"模型文件已保存至{self.save_file_path}")

    def load_model(self):
        """
        模型加载和使用：载入pkl文件。注意此时预测时列名为['x0', 'x1', ...]

        Raises:
            Exception: [description]

        Returns:
            _type_: sklearn 机器学习包实例类型。预测时用法: model.predict_proba(df[feat_list])[:, 1]
        """
        if not os.path.exists(self.save_file_path):
            raise Exception("参数 save_file_path 指向的文件路径不存在, 请检查.")

        try:
            model = joblib.load(self.save_file_path)
        except ModuleNotFoundError as exc:
            if exc.name in {
                "decomposition.extractors", "decomposition.forecasters",
                "decomposition.pipeline", "decomposition.spec", "decomposition.presets",
                "decomposition.component_factory", "decomposition.types",
                "decomposition.composers", "decomposition.base",
                "decomposition.residual_diagnostics", "decomposition.registry",
                "decomposition.time_axis",
            }:
                raise ValueError(
                    "incompatible decomposition artifact: explicitly refit with component_fit_v2; "
                    "removed component paths are not supported"
                ) from exc
            raise

        # 加载结果原样返回，调用方自行判型；
        # 不做类型检查导入（架构收敛 F8：低层不得反向 import 高层，duck-typing 收口）。
        return model
