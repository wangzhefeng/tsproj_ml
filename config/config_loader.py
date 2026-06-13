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

# python libraries
import argparse
import importlib


def _build_arg_parser(config_module: str):
    # args parser
    parser = argparse.ArgumentParser(description="Time series forecasting runner")
    # model config
    parser.add_argument(
        "--config-module",
        default=config_module,
        help="Python module path that exposes ModelConfig",
    )
    # parse args
    parsed_args = parser.parse_args()
    
    return parsed_args


def load_model_config(config_module: str="config.templates.univariate_config"):
    # parsed args
    parsed_args = _build_arg_parser(config_module)
    # config module
    module = importlib.import_module(parsed_args.config_module)
    # config module class: ModelConfig
    model_config = getattr(module, "ModelConfig", None)
    if model_config is None:
        raise ImportError(f"ModelConfig not found in module: {parsed_args.config_module}")
    
    return model_config


ModelConfig = load_model_config(config_module="config.templates.univariate_config")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
