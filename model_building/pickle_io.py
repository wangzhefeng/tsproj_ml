# -*- coding: utf-8 -*-

"""模型与缩放器 pickle 保存/加载（离线部署唯一持久化入口）。"""

import os
import pickle
import joblib


class ModelDeployPkl:
    """pickle 部署封装：保存训练产物、按路径重载（唯一持久化入口）。"""
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

        # 低层不依赖 bundle 类型；对象显式携带整数版本时，只接受 schema-2。
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

    def load_model(self):
        """
        从可信 pkl 文件加载对象，特征列和预测接口由保存的对象决定。

        Raises:
            Exception: [description]

        Returns:
            object: 保存的模型或 bundle，不在此层转换对象类型。
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
