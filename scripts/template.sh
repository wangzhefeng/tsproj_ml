#!/usr/bin/env bash
set -euo pipefail

# main.py 直跑模板。
# 配置切换与模型参数修改请直接编辑：
#   1. main.py 中的 CONFIG_YAML
#   2. 对应 YAML 文件内容

export LOG_NAME="${LOG_NAME:-run-template}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-.uv_cache}"
uv run python main.py
