# -*- coding: utf-8 -*-

# ***************************************************
# * File        : config_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-06-13
# * Version     : 1.0.061316
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

import importlib


def load_model_config(
    config_module: str = "config.templates.univariate_config",
    config_class: str = "ModelConfig",
    instantiate: bool = False,
):
    module = importlib.import_module(config_module)
    model_config = getattr(module, config_class, None)
    if model_config is None:
        raise ImportError(f"Config class '{config_class}' not found in module: {config_module}")
    if instantiate:
        return model_config()
    return model_config




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
