#!/usr/bin/env bash
set -euo pipefail

# run.py 标准模板（可复制后按项目场景调整）
#
# 用法：
#   bash scripts/template.sh
#   MODEL_TYPE=xgboost PRED_METHOD=univariate-single-multistep-recursive bash scripts/template.sh
#   IS_TESTING=0 IS_FORECASTING=1 NOW_TIME=2025-12-27T00:00:00 bash scripts/template.sh

export LOG_NAME="${LOG_NAME:-run-template}"

# 配置模块（YAML 为必填，base_config 从 YAML 内部读取）
CONFIG_YAML="${CONFIG_YAML:-config/ETT-small/ETTm1/model_config_lgbm_usmd.yaml}"

# 基本运行参数
SEED="${SEED:-2025}"
MODEL_TYPE="${MODEL_TYPE:-lightgbm}"
PRED_METHOD="${PRED_METHOD:-univariate-single-multistep-direct}"
IS_TESTING="${IS_TESTING:-1}"
IS_FORECASTING="${IS_FORECASTING:-0}"
NOW_TIME="${NOW_TIME:-2025-12-27T00:00:00}"

# 可选覆盖参数
HISTORY_DAYS="${HISTORY_DAYS:-}"
PREDICT_DAYS="${PREDICT_DAYS:-}"
WINDOW_DAYS="${WINDOW_DAYS:-}"
LAGS="${LAGS:-}"
LEARNING_RATE="${LEARNING_RATE:-}"
MODEL_PARAMS="${MODEL_PARAMS:-}"

# 可选：开启数据增强 / 特征选择 / 自动学习率
ENABLE_DATA_AUGMENTATION="${ENABLE_DATA_AUGMENTATION:-0}"
ENABLE_FEATURE_SELECTION="${ENABLE_FEATURE_SELECTION:-0}"
ENABLE_AUTO_LEARNING_RATE="${ENABLE_AUTO_LEARNING_RATE:-0}"
FEATURE_SELECTION_METHOD="${FEATURE_SELECTION_METHOD:-f_regression}"
FEATURE_SELECTION_MAX_FEATURES="${FEATURE_SELECTION_MAX_FEATURES:-80}"
FEATURE_SELECTION_MIN_FEATURES="${FEATURE_SELECTION_MIN_FEATURES:-10}"
AUTO_LR_MIN="${AUTO_LR_MIN:-0.005}"
AUTO_LR_MAX="${AUTO_LR_MAX:-0.2}"

cmd=(python3 -u run.py
  --config-yaml "${CONFIG_YAML}"
  --seed "${SEED}"
  --model-type "${MODEL_TYPE}"
  --pred-method "${PRED_METHOD}"
  --is-testing "${IS_TESTING}"
  --is-forecasting "${IS_FORECASTING}"
  --now-time "${NOW_TIME}"
  --perform-tuning 0
  --enable-data-augmentation "${ENABLE_DATA_AUGMENTATION}"
  --enable-feature-selection "${ENABLE_FEATURE_SELECTION}"
  --feature-selection-method "${FEATURE_SELECTION_METHOD}"
  --feature-selection-max-features "${FEATURE_SELECTION_MAX_FEATURES}"
  --feature-selection-min-features "${FEATURE_SELECTION_MIN_FEATURES}"
  --enable-auto-learning-rate "${ENABLE_AUTO_LEARNING_RATE}"
  --auto-lr-min "${AUTO_LR_MIN}"
  --auto-lr-max "${AUTO_LR_MAX}"
)

# 仅在传值时附加，避免覆盖 config 默认值
if [[ -n "${HISTORY_DAYS}" ]]; then
  cmd+=(--history-days "${HISTORY_DAYS}")
fi
if [[ -n "${PREDICT_DAYS}" ]]; then
  cmd+=(--predict-days "${PREDICT_DAYS}")
fi
if [[ -n "${WINDOW_DAYS}" ]]; then
  cmd+=(--window-days "${WINDOW_DAYS}")
fi
if [[ -n "${LAGS}" ]]; then
  cmd+=(--lags "${LAGS}")
fi
if [[ -n "${LEARNING_RATE}" ]]; then
  cmd+=(--learning-rate "${LEARNING_RATE}")
fi
if [[ -n "${MODEL_PARAMS}" ]]; then
  cmd+=(--model-params "${MODEL_PARAMS}")
fi

# 新增主流程能力开关
cmd+=(--scale 0)
cmd+=(--encode-categorical-features 0)

echo "Running command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
