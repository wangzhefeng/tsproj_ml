#!/usr/bin/env bash
set -euo pipefail

# 参考 config 目录中的配置文件：
#   config/model_config_lgbm_usmd_A.py
#   config/model_config_xgb_usmd_A.py
#   config/model_config_cab_usmd_A.py
#
# 使用方式：
#   bash scripts/run_model_test.sh
# 或覆盖变量：
#   CONFIG_MODULE=config.model_config_xgb_usmd_A MODEL_TYPE=xgboost bash scripts/run_model_test.sh

export LOG_NAME="${LOG_NAME:-run-model-test}"

CONFIG_MODULE="${CONFIG_MODULE:-config.model_config_lgbm_usmd_A}"
CONFIG_CLASS="${CONFIG_CLASS:-ModelConfig_univariate}"

MODEL_TYPE="${MODEL_TYPE:-lightgbm}"
PRED_METHOD="${PRED_METHOD:-univariate-single-multistep-direct}"

IS_TESTING="${IS_TESTING:-1}"
IS_FORECASTING="${IS_FORECASTING:-0}"
NOW_TIME="${NOW_TIME:-2025-12-27T00:00:00}"

SEED="${SEED:-2025}"

python3 -u run.py \
  --config-module "${CONFIG_MODULE}" \
  --config-class "${CONFIG_CLASS}" \
  --seed "${SEED}" \
  --model-type "${MODEL_TYPE}" \
  --pred-method "${PRED_METHOD}" \
  --is-testing "${IS_TESTING}" \
  --is-forecasting "${IS_FORECASTING}" \
  --now-time "${NOW_TIME}"

