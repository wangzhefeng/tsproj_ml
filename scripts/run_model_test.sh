#!/usr/bin/env bash
set -euo pipefail

# 参考 config 目录中的配置文件：
#   config/univariate_config.py
#   config/multivariate_config.py
#
# 使用方式：
#   bash scripts/run_model_test.sh
# 或覆盖变量：
#   CONFIG_YAML=config/aidc_electricity_computility/electricity/2026-06-11/A1_01a/xgb_usmd.yaml bash scripts/run_model_test.sh

export LOG_NAME="${LOG_NAME:-run-model-test}"

CONFIG_YAML="${CONFIG_YAML:-config/ETT-small/ETTm1/lgbm_usmd.yaml}"

MODEL_TYPE="${MODEL_TYPE:-lightgbm}"
PRED_METHOD="${PRED_METHOD:-univariate-single-multistep-direct}"

IS_TESTING="${IS_TESTING:-1}"
IS_FORECASTING="${IS_FORECASTING:-0}"
NOW_TIME="${NOW_TIME:-2025-12-27T00:00:00}"

SEED="${SEED:-2025}"
ENABLE_FEATURE_SELECTION="${ENABLE_FEATURE_SELECTION:-0}"
ENABLE_DATA_AUGMENTATION="${ENABLE_DATA_AUGMENTATION:-0}"
ENABLE_AUTO_LEARNING_RATE="${ENABLE_AUTO_LEARNING_RATE:-0}"
FEATURE_SELECTION_METHOD="${FEATURE_SELECTION_METHOD:-f_regression}"
FEATURE_SELECTION_MAX_FEATURES="${FEATURE_SELECTION_MAX_FEATURES:-80}"
FEATURE_SELECTION_MIN_FEATURES="${FEATURE_SELECTION_MIN_FEATURES:-10}"

cmd=(python3 -u run.py
  --config-yaml "${CONFIG_YAML}" \
  --seed "${SEED}" \
  --model-type "${MODEL_TYPE}" \
  --pred-method "${PRED_METHOD}" \
  --is-testing "${IS_TESTING}" \
  --is-forecasting "${IS_FORECASTING}" \
  --now-time "${NOW_TIME}" \
  --enable-feature-selection "${ENABLE_FEATURE_SELECTION}" \
  --enable-data-augmentation "${ENABLE_DATA_AUGMENTATION}" \
  --enable-auto-learning-rate "${ENABLE_AUTO_LEARNING_RATE}" \
  --feature-selection-method "${FEATURE_SELECTION_METHOD}" \
  --feature-selection-max-features "${FEATURE_SELECTION_MAX_FEATURES}" \
  --feature-selection-min-features "${FEATURE_SELECTION_MIN_FEATURES}"
)

"${cmd[@]}"
