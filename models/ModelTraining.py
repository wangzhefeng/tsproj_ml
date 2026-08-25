# -*- coding: utf-8 -*-

# ***************************************************
# * File        : ModelTraining.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-03-01
# * Version     : 1.0.030118
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************

# python libraries
import copy
import os
import re
from pathlib import Path
import math
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    TimeSeriesSplit, 
    GridSearchCV, 
    RandomizedSearchCV
)

from data_provider.data_loader import prepare_native_train_eval_datasets
from features.DataAugment import TimeSeriesAugmenter
from features.FeatureSelection import FeatureSelector
from models.ModelFactory import ModelFactory
from models.ModelSaveLoad import ModelDeployPkl
from models.ModelEnsemble import TimeSeriesEnsembleRegressor, EnsembleConfig
from models.learning_rate import resolve_learning_rate
from models.losses import get_loss_name_from_model_params, get_scorer_by_loss_name
from models.multistep.artifacts import BlendArtifact, MultistepArtifactMetadata
from models.multistep.plans import ExogenousTiming, TrainingLayout
from models.multistep.resolve import resolve_strategy
from models.multistep.spec import InputScope, RolloutFamily
from models.multistep.weights import BlendWeights
from probabilistic.objectives import inject_quantile_objective
from probabilistic.spec import resolve_probabilistic_spec
from probabilistic.training import QuantileTrainer
from probabilistic.types import BlendQuantileModel, ForecastModelBundle
from utils.log_util import logger
from utils.frequency import compute_time_decay_weights

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class DirectMultiOutputRegressor:
    """
    为 Direct 多步预测定制的多输出训练器。

    - 每个 horizon 单独训练一个回归器
    - 支持为每个输出传入独立 eval_set / early stopping
    - 支持按输出维度并行训练
    """

    def __init__(self, estimator_factory, n_jobs: int = 1, log_prefix: str = "[DirectMultiOutputRegressor]"):
        self.estimator_factory = estimator_factory
        self.n_jobs = max(1, int(n_jobs or 1))
        self.log_prefix = log_prefix
        self.estimators_: List[Any] = []

    @staticmethod
    def _to_1d(values: Any) -> np.ndarray:
        return np.asarray(values).reshape(-1)

    def _fit_single_output(self, output_idx: int, X_train, y_train, fit_kwargs: Optional[Dict[str, Any]] = None):
        estimator = self.estimator_factory()
        estimator.fit(X_train, self._to_1d(y_train), **(fit_kwargs or {}))
        return output_idx, estimator

    def fit(self, X_train, Y_train, fit_kwargs_list: Optional[List[Dict[str, Any]]] = None):
        y_frame = Y_train if isinstance(Y_train, pd.DataFrame) else pd.DataFrame(Y_train)
        n_outputs = y_frame.shape[1]
        fit_kwargs_list = fit_kwargs_list or [{} for _ in range(n_outputs)]
        if len(fit_kwargs_list) != n_outputs:
            raise ValueError(
                f"{self.log_prefix} fit_kwargs_list length ({len(fit_kwargs_list)}) "
                f"does not match n_outputs ({n_outputs})."
            )

        estimators = [None] * n_outputs
        if self.n_jobs > 1 and n_outputs > 1:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [
                    executor.submit(
                        self._fit_single_output,
                        output_idx,
                        X_train,
                        y_frame.iloc[:, output_idx],
                        fit_kwargs_list[output_idx],
                    )
                    for output_idx in range(n_outputs)
                ]
                for future in as_completed(futures):
                    output_idx, estimator = future.result()
                    estimators[output_idx] = estimator
        else:
            for output_idx in range(n_outputs):
                _, estimator = self._fit_single_output(
                    output_idx,
                    X_train,
                    y_frame.iloc[:, output_idx],
                    fit_kwargs_list[output_idx],
                )
                estimators[output_idx] = estimator

        self.estimators_ = estimators
        # 训练完成后不再依赖工厂，清空以避免模型保存时因闭包/lambda 无法 pickle。
        self.estimator_factory = None
        return self

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise ValueError(f"{self.log_prefix} multi-output estimators are not fitted yet.")
        preds = [self._to_1d(estimator.predict(X)) for estimator in self.estimators_]
        return np.column_stack(preds)


class HorizonAlignedDirectRegressor(DirectMultiOutputRegressor):
    """每个输出仅消费其对应目标horizon外生列的Direct训练器。

    输入宽表仍使用 ``feature_h1..feature_hH`` 表达目标时刻外生轨迹，
    但第h个估计器只接收 ``feature_h{h}``，并在交给基模型前重命名回
    canonical ``feature``。原点lag/advanced以及未展开的origin-frozen列共享。
    """

    _HORIZON_PATTERN = re.compile(r"^(.+)_h(\d+)$")

    def __init__(self, estimator_factory, n_jobs: int = 1, log_prefix: str = "[HorizonAlignedDirectRegressor]"):
        super().__init__(estimator_factory, n_jobs=n_jobs, log_prefix=log_prefix)
        self.shared_features_: List[str] = []
        self.horizon_bases_: List[str] = []

    def _resolve_feature_layout(self, columns: List[str], n_outputs: int) -> None:
        expanded = {}
        expanded_columns = set()
        for column in columns:
            match = self._HORIZON_PATTERN.match(str(column))
            if match is None:
                continue
            base = match.group(1)
            horizon = int(match.group(2))
            expanded.setdefault(base, {})[horizon] = column
            expanded_columns.add(column)
        if not expanded:
            raise ValueError(f"{self.log_prefix} no horizon-aware exogenous columns found.")
        for base, by_horizon in expanded.items():
            missing = [h for h in range(1, n_outputs + 1) if h not in by_horizon]
            if missing:
                raise ValueError(
                    f"{self.log_prefix} feature '{base}' missing horizon {missing[0]} "
                    f"for {n_outputs} outputs."
                )
        expanded_bases = set(expanded)
        self.shared_features_ = [
            column for column in columns
            if column not in expanded_columns and column not in expanded_bases
        ]
        self.horizon_bases_ = list(expanded)

    def _frame_for_output(self, X: pd.DataFrame, output_idx: int) -> pd.DataFrame:
        horizon = output_idx + 1
        result = X.reindex(columns=self.shared_features_).copy()
        for base in self.horizon_bases_:
            source = f"{base}_h{horizon}"
            if source not in X.columns:
                raise ValueError(f"{self.log_prefix} feature '{base}' missing horizon {horizon}.")
            result[base] = X[source].to_numpy()
        return result

    def _fit_single_output(self, output_idx: int, X_train, y_train, fit_kwargs: Optional[Dict[str, Any]] = None):
        X_output = self._frame_for_output(X_train, output_idx)
        kwargs = dict(fit_kwargs or {})
        if kwargs.get("eval_set"):
            kwargs["eval_set"] = [
                (self._frame_for_output(X_eval, output_idx), y_eval)
                for X_eval, y_eval in kwargs["eval_set"]
            ]
        return super()._fit_single_output(output_idx, X_output, y_train, kwargs)

    def fit(self, X_train, Y_train, fit_kwargs_list: Optional[List[Dict[str, Any]]] = None):
        y_frame = Y_train if isinstance(Y_train, pd.DataFrame) else pd.DataFrame(Y_train)
        self._resolve_feature_layout(list(X_train.columns), y_frame.shape[1])
        return super().fit(X_train, y_frame, fit_kwargs_list=fit_kwargs_list)

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise ValueError(f"{self.log_prefix} multi-output estimators are not fitted yet.")
        preds = [
            self._to_1d(estimator.predict(self._frame_for_output(X, output_idx)))
            for output_idx, estimator in enumerate(self.estimators_)
        ]
        return np.column_stack(preds)


class Trainer:

    def __init__(self, args: Dict, log_prefix: str):
        self.args = args
        self.log_prefix = log_prefix
        self.model_factory = ModelFactory(log_prefix=log_prefix)
        self.model_type = getattr(self.args, "model_type", "lightgbm")
        self.model_param_overrides = copy.deepcopy(getattr(self.args, "model_params", {}) or {})
        self.model_params = self.model_factory.resolve_model_params(
            self.model_type,
            self.model_param_overrides,
        )
        self._apply_model_thread_limits()
        self.augmenter = TimeSeriesAugmenter(
            enabled=bool(getattr(self.args, "enable_data_augmentation", False)),
            augmentation_ratio=float(getattr(self.args, "augmentation_ratio", 0.2)),
            feature_noise_std=float(getattr(self.args, "augmentation_feature_noise_std", 0.01)),
            target_noise_std=float(getattr(self.args, "augmentation_target_noise_std", 0.005)),
            random_state=int(getattr(self.args, "augmentation_random_state", 42)),
            log_prefix=self.log_prefix,
        )
        # 时间衰减样本权重;在 train() 中按启用开关计算,baseline 路径自行计算
        self.sample_weight = None
        self.resolved_strategy = None

    def _get_resolved_strategy(self, horizon: Optional[int] = None):
        if horizon is None:
            horizon = getattr(self.args, "horizon", None)
        if horizon is None:
            horizon = getattr(self.args, "predict_steps", None)
        if horizon is None:
            raise ValueError("Trainer requires args.horizon or args.predict_steps to resolve the training plan.")
        resolved = resolve_strategy(
            self.args,
            int(horizon),
            target_feature=str(getattr(self.args, "target_feature", "y")),
        )
        self.resolved_strategy = resolved
        return resolved

    def _resolve_worker_count(self, attr_name: str, default: int = 1) -> int:
        value = int(getattr(self.args, attr_name, default) or default)
        if value <= 0:
            cpu_count = os.cpu_count() or 1
            return max(1, cpu_count)

        return value

    def _should_use_horizon_aligned_direct(self) -> bool:
        try:
            resolved = self.resolved_strategy or self._get_resolved_strategy()
        except ValueError:
            return False
        return resolved.feature_plan.exogenous_timing == ExogenousTiming.BY_HORIZON

    @staticmethod
    def _thread_limited_params(model_type: str, params: Dict[str, Any], threads: int) -> Dict[str, Any]:
        """
        按模型类型把线程上限注入参数字典（返回新字典，不改入参）。

        HistGB（无 n_jobs 构造参数，底层 OpenMP）与线性模型（单线程/BLAS）
        无线程参数可注，原样返回；HistGB 的窗口并行超额订阅由
        Tester._window_test 的 OMP_NUM_THREADS=1 兜底。
        """
        mt = str(model_type).lower()
        params = copy.deepcopy(params)
        if mt in ["lightgbm", "lgb", "xgboost", "xgb", "randomforest", "rf"]:
            params["n_jobs"] = threads
        elif mt in ["catboost", "cat"]:
            params["thread_count"] = threads
        return params

    def _apply_model_thread_limits(self):
        if not hasattr(self.args, "model_thread_count"):
            return
        raw_value = getattr(self.args, "model_thread_count", None)
        if raw_value is None:
            return
        model_threads = self._resolve_worker_count("model_thread_count", default=1)
        self.model_params = self._thread_limited_params(self.model_type, self.model_params, model_threads)

    def _get_tuning_param_grid(self, model_type: str) -> Dict[str, list]:
        mt = str(model_type).lower()
        if mt in ["lightgbm", "lgb"]:
            return {
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.03, 0.05, 0.08],
                "feature_fraction": [0.7, 0.8, 0.9],
                "bagging_fraction": [0.7, 0.8, 0.9],
                "min_child_samples": [20, 50, 100],
            }
        if mt in ["xgboost", "xgb"]:
            return {
                "max_depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
                "min_child_weight": [1, 3, 5],
            }
        if mt in ["catboost", "cat"]:
            return {
                "depth": [4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08],
                "l2_leaf_reg": [3, 5, 7],
                "bagging_temperature": [0, 0.5, 1.0],
            }
        if mt in ["histgb", "histgradientboosting"]:
            return {
                "max_leaf_nodes": [15, 31, 63],
                "learning_rate": [0.03, 0.05, 0.08],
                "min_samples_leaf": [10, 20, 50],
                "l2_regularization": [0.0, 1.0, 10.0],
            }
        if mt in ["ridge"]:
            return {"alpha": [0.1, 1.0, 10.0, 100.0]}
        if mt in ["elasticnet", "enet"]:
            return {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.2, 0.5, 0.8]}
        if mt in ["lasso"]:
            return {"alpha": [0.001, 0.01, 0.1, 1.0]}
        if mt in ["quantileregressor", "qr"]:
            return {"alpha": [0.0, 0.1, 1.0]}
        if mt in ["seasonaltemplate", "st"]:
            return {"day_type_split": [True, False], "equal_weight": [False, True]}

        return {}

    def _build_multi_output_model(self, base_estimator, n_outputs: int):
        """
        根据配置构建多输出策略模型
        """
        if n_outputs <= 1:
            return base_estimator
        strategy = str(getattr(self.args, "multi_output_strategy", "multioutput")).lower()
        if strategy == "regressor_chain":
            logger.info(f"{self.log_prefix} Multi-output strategy: RegressorChain")
            return RegressorChain(estimator=base_estimator)
        multi_output_n_jobs = self._resolve_worker_count("multi_output_n_jobs", default=1)
        logger.info(f"{self.log_prefix} Multi-output strategy: MultiOutputRegressor")

        return MultiOutputRegressor(estimator=base_estimator, n_jobs=multi_output_n_jobs)

    def _is_fourmethods_univariate_method(self) -> bool:
        try:
            resolved = self.resolved_strategy or self._get_resolved_strategy()
        except ValueError:
            return False
        return (
            resolved.spec.input_scope == InputScope.TARGET_ONLY
            and resolved.spec.rollout != RolloutFamily.BLEND
        )

    def _should_use_fourmethods_baseline_training(self) -> bool:
        """
        四种单变量方法在增强能力全关闭时，自动采用 FourMethods 的训练语义。
        """
        if not self._is_fourmethods_univariate_method():
            return False

        # horizon_feature 模式（USMD/MSMD + direct_strategy=horizon_feature）必须在
        # train() 主路径完成宽表→长表 melt；baseline 路径会跳过 melt，导致训练侧
        # 特征数（不含 h/h_sin/h_cos）与推理侧（含 h 三列）不一致，LightGBM 报
        # feature count mismatch。故强制走主路径。
        if self._should_use_horizon_feature() or self._should_use_horizon_aligned_direct():
            return False

        enhancement_flags = [
            "scale_features",
            "scale_target",
            "enable_feature_selection",
            "enable_data_augmentation",
            "enable_ensemble",
            "perform_tuning",
            "enable_auto_learning_rate",
        ]
        if any(bool(getattr(self.args, flag, False)) for flag in enhancement_flags):
            return False

        return str(getattr(self.args, "predict_type", "point")).lower() == "point"

    @staticmethod
    def _ensure_target_frame(Y_train) -> pd.DataFrame:
        if isinstance(Y_train, pd.DataFrame):
            return Y_train.copy()
        if isinstance(Y_train, pd.Series):
            return Y_train.to_frame()
        y_arr = np.asarray(Y_train)
        if y_arr.ndim == 1:
            return pd.DataFrame({"y": y_arr})
        return pd.DataFrame(y_arr)

    def _train_fourmethods_baseline(self, X_train_df, Y_train_df, categorical_features):
        """
        复刻 FourMethods 的无增强训练路径：原始特征、原始目标、无早停、无调参。
        """
        selected_features = X_train_df.columns.tolist()
        X_train_df_processed = X_train_df.copy()
        Y_train_df_processed = self._ensure_target_frame(Y_train_df)
        lgbm_categorical = list(categorical_features or [])
        model_type = self.model_type
        # baseline 路径不走增强/学习率块,这里独立计算时间衰减权重
        baseline_sample_weight = None
        if bool(getattr(self.args, "enable_time_decay_sample_weight", False)):
            baseline_sample_weight = compute_time_decay_weights(
                n_samples=len(X_train_df_processed),
                n_per_day=int(getattr(self.args, "n_per_day", 1) or 1),
                halflife_days=float(getattr(self.args, "decay_halflife_days", 14.0)),
            )
        estimator_wrapper = self.model_factory.create_model(
            model_type=model_type,
            model_params=self.model_params,
        )

        if Y_train_df_processed.shape[1] == 1:
            logger.info(f"{self.log_prefix} FourMethods baseline training single-output regressor...")
            logger.info(f"{self.log_prefix} {'-' * 71}")
            model = estimator_wrapper
            # fit kwargs 统一组装，各模型封装自行识别/忽略不支持的项
            # （如 XGBoost/RF 忽略 categorical_feature；各封装均支持 sample_weight）
            fit_kwargs = {}
            if lgbm_categorical is not None:
                fit_kwargs["categorical_feature"] = lgbm_categorical
            if baseline_sample_weight is not None:
                fit_kwargs["sample_weight"] = baseline_sample_weight
            model.fit(X_train_df_processed, np.ravel(Y_train_df_processed.values), **fit_kwargs)
        else:
            if baseline_sample_weight is not None:
                logger.warning(
                    f"{self.log_prefix} time-decay sample_weight not supported for "
                    f"multi-output baseline (sklearn wrapper); skipped."
                )
            logger.info(
                f"{self.log_prefix} FourMethods baseline training multi-output regressor "
                f"with {Y_train_df_processed.shape[1]} outputs..."
            )
            logger.info(f"{self.log_prefix} {'-' * 71}")
            model = self._build_multi_output_model(estimator_wrapper.model, n_outputs=Y_train_df_processed.shape[1])
            try:
                model.fit(X_train_df_processed, Y_train_df_processed)
            except PermissionError as e:
                if hasattr(model, "n_jobs") and getattr(model, "n_jobs", 1) != 1:
                    logger.warning(
                        f"{self.log_prefix} Multi-output parallel training failed, fallback to n_jobs=1. error: {e}"
                    )
                    model.set_params(n_jobs=1)
                    model.fit(X_train_df_processed, Y_train_df_processed)
                else:
                    raise

        logger.info(f"{self.log_prefix} FourMethods baseline model training completed!")
        return model, None, None, selected_features

    def _create_model_instance(self, model_type: str, model_params: Dict[str, Any], log_params: bool = True):
        try:
            return self.model_factory.create_model(
                model_type=model_type,
                model_params=model_params,
                log_params=log_params,
            )
        except TypeError:
            return self.model_factory.create_model(
                model_type=model_type,
                model_params=model_params,
            )

    def _train_quantile_single_model(
        self,
        quantile: float,
        model_type: str,
        X_train_df_processed,
        Y_train_df_processed,
        lgbm_categorical,
    ):
        params_q = self._inject_quantile_params(model_type=model_type, params=self.model_params, quantile=quantile)
        estimator_wrapper_q = self._create_model_instance(
            model_type=model_type,
            model_params=params_q,
        )
        if Y_train_df_processed.shape[1] == 1:
            model_q = estimator_wrapper_q
            # fit kwargs 统一组装，各模型封装自行识别/忽略不支持的项
            # （native_train_data 仅 CatBoost 封装使用，其余经 **kwargs 忽略；
            #   CatBoost native 路径下权重已内嵌 Pool，封装会忽略显式 sample_weight）
            fit_kwargs_q = {}
            if lgbm_categorical is not None:
                fit_kwargs_q["categorical_feature"] = lgbm_categorical
            if getattr(self, "native_data_bundle", {}).get("enabled"):
                fit_kwargs_q["native_train_data"] = self.native_data_bundle.get("train_native")
            if getattr(self, "sample_weight", None) is not None:
                fit_kwargs_q["sample_weight"] = self.sample_weight
            model_q.fit(X_train_df_processed, np.ravel(Y_train_df_processed.values), **fit_kwargs_q)
        else:
            strategy = str(getattr(self.args, "multi_output_strategy", "multioutput")).lower()
            if strategy == "regressor_chain":
                if getattr(self, "sample_weight", None) is not None:
                    logger.warning(
                        f"{self.log_prefix} time-decay sample_weight not supported for "
                        "multi-output quantile regressor_chain; skipped."
                    )
                model_q = self._build_multi_output_model(
                    estimator_wrapper_q.model,
                    n_outputs=Y_train_df_processed.shape[1],
                )
                model_q.fit(X_train_df_processed, Y_train_df_processed)
            else:
                X_fit, Y_fit, fit_kwargs_list = self._prepare_multi_output_fit_data(
                    X_train_df_processed,
                    Y_train_df_processed,
                    list(lgbm_categorical or []),
                    getattr(self, "native_data_bundle", {}),
                )
                regressor_cls = (
                    HorizonAlignedDirectRegressor
                    if self._should_use_horizon_aligned_direct()
                    else DirectMultiOutputRegressor
                )
                model_q = regressor_cls(
                    estimator_factory=lambda: self._create_model_instance(
                        model_type=model_type,
                        model_params=params_q,
                        log_params=False,
                    ),
                    n_jobs=self._resolve_worker_count("multi_output_n_jobs", default=1),
                    log_prefix=self.log_prefix,
                )
                model_q.fit(X_fit, Y_fit, fit_kwargs_list=fit_kwargs_list)
        return quantile, model_q

    def _train_blend_quantile_single_model(
        self,
        quantile: float,
        model_type: str,
        X_train_df_processed,
        Y_direct,
        Y_recursive,
        lgbm_categorical,
        actual_categorical,
    ):
        """为 blend（USBR/MSBR）训练单个分位数的 Direct+Recursive 子模型对。

        Direct 子模型走项目 DirectMultiOutputRegressor，逐输出透传同一
        sample_weight；Recursive 子模型走单输出分位数估计器。
        """
        params_q = self._inject_quantile_params(model_type=model_type, params=self.model_params, quantile=quantile)
        # Direct 子模型（多输出 quantile）
        X_fit_d, Y_fit_d, fit_kwargs_list_d = self._prepare_multi_output_fit_data(
            X_train_df_processed,
            Y_direct,
            list(lgbm_categorical or []),
            getattr(self, "native_data_bundle", {}),
        )
        direct_q = DirectMultiOutputRegressor(
            estimator_factory=lambda: self._create_model_instance(
                model_type=model_type,
                model_params=params_q,
                log_params=False,
            ),
            n_jobs=self._resolve_worker_count("multi_output_n_jobs", default=1),
            log_prefix=self.log_prefix,
        )
        direct_q.fit(X_fit_d, Y_fit_d, fit_kwargs_list=fit_kwargs_list_d)
        # Recursive 子模型（单输出 quantile）
        estimator_recursive_q = self._create_model_instance(model_type=model_type, model_params=params_q)
        X_fit_r, y_fit_r, fit_kwargs_r = self._prepare_single_output_fit_data(
            X_train_df_processed, Y_recursive, actual_categorical,
        )
        if lgbm_categorical is not None:
            fit_kwargs_r["categorical_feature"] = lgbm_categorical
        if getattr(self, "native_data_bundle", {}).get("enabled"):
            fit_kwargs_r["native_train_data"] = self.native_data_bundle.get("train_native")
        estimator_recursive_q.fit(X_fit_r, np.ravel(y_fit_r), **fit_kwargs_r)
        return quantile, BlendQuantileModel(
            direct=direct_q,
            recursive=estimator_recursive_q,
        )

    def _inject_quantile_params(self, model_type: str, params: Dict, quantile: float) -> Dict:
        """委托概率模块注入 quantile objective；不支持模型直接失败。"""
        return inject_quantile_objective(model_type, params, quantile)

    def _build_feature_selector(self, categorical_features):
        return FeatureSelector(
            enabled=bool(getattr(self.args, "enable_feature_selection", False)),
            method=str(getattr(self.args, "feature_selection_method", "f_regression")),
            max_features=int(getattr(self.args, "feature_selection_max_features", 80)),
            min_features=int(getattr(self.args, "feature_selection_min_features", 10)),
            force_keep_features=list(categorical_features or []),
            log_prefix=self.log_prefix,
        )

    def _prepare_training_data(self, X_train, Y_train, categorical_features, sample_weight=None):
        X_prepared = X_train.copy()
        Y_prepared = Y_train.copy()
        # 1) 数据增强（仅训练集）；权重随增强同步扩展（增强行继承源行权重）
        X_prepared, Y_prepared, sample_weight = self.augmenter.augment(
            X_prepared, Y_prepared, categorical_features=categorical_features, sample_weight=sample_weight
        )
        # 2) 特征选择（fit on training split）
        selector = self._build_feature_selector(categorical_features=categorical_features)
        X_selected, selected_features = selector.fit_transform(
            X_prepared, Y_prepared, categorical_features=categorical_features
        )
        return X_selected, Y_prepared, selected_features, sample_weight

    def _prepare_native_model_data(self, X_train, Y_train, categorical_features):
        native_bundle = prepare_native_train_eval_datasets(
            model_type=self.model_type,
            X_train=X_train,
            y_train=Y_train,
            categorical_features=categorical_features,
            sample_weight=getattr(self, "sample_weight", None),
        )
        if native_bundle.get("enabled"):
            logger.info(
                f"{self.log_prefix} Prepared native {native_bundle['framework']} training container: "
                f"{type(native_bundle['train_native']).__name__}"
            )
        else:
            logger.info(
                f"{self.log_prefix} Native training container skipped for {self.model_type}: "
                f"{native_bundle.get('reason')}"
            )
        return native_bundle

    # ------------------------------
    # horizon-as-feature（USMD/MSMD 长表训练）
    # ------------------------------
    def _should_use_horizon_feature(self) -> bool:
        """USMD/MSMD 且 direct_strategy=horizon_feature 时启用长表 melt。"""
        try:
            resolved = self.resolved_strategy or self._get_resolved_strategy()
        except ValueError:
            return False
        return resolved.training_plan.layout == TrainingLayout.HORIZON_LONG

    def _is_blend_method(self) -> bool:
        """USBR/MSBR = Direct+Recursive 加权融合。"""
        try:
            resolved = self.resolved_strategy or self._get_resolved_strategy()
        except ValueError:
            return False
        return resolved.training_plan.layout == TrainingLayout.COMPOSITE

    def _resolve_horizon_period(self, horizon: int) -> int:
        """horizon sin/cos 编码周期：子日频用 n_per_day（日周期），日频用 7（周周期）。"""
        n_per_day = int(getattr(self.args, "n_per_day", 1) or 1)
        if n_per_day > 1:
            return n_per_day
        return 7

    _DERIVED_SUFFIX_MARKERS = ("lag_", "rolling_", "diff_")

    @staticmethod
    def _classify_endogenous_derived_for_melt(columns) -> set:
        """识别宽表特征列中由目标序列派生的列（滞后/滚动/差分等）。

        与 Forecaster._classify_endogenous_derived 同一约定：按列名模式
        白名单匹配（y_lag_*、y_rolling_*、y_diff_* 等），其余列按外生
        处理（天气/日历/自定义计划类，预测目标日可知）。
        """
        derived = set()
        for col in columns:
            name = str(col)
            if name == "y" or name.startswith("y_"):
                derived.add(col)
                continue
            for marker in Trainer._DERIVED_SUFFIX_MARKERS:
                if f"_{marker}" in name:
                    derived.add(col)
                    break
        return derived

    def _melt_to_horizon_long(self, X_train_df, Y_train_df, sample_weight):
        """
        Direct 宽表 (N × F, N × H) melt 为 horizon-as-feature 长表。

        输出: X_long (N*H × (F+h_feats)), Y_long (N*H × 1), sw_long (N*H,), h_feature_names
        行对齐: X_long[i*H + h] = lag/rolling 等内生派生列取原点行 i（预测
        原点可知状态）+ 外生列取目标日 i+h+1（与推理端按未来行取外生一致），
        Y_long[i*H+h] = Y_train_df.iloc[i, h]
        """
        H = int(Y_train_df.shape[1])
        N = int(len(X_train_df))

        h_feat_name = str(getattr(self.args, "horizon_feature_name", "forecast_horizon_idx"))
        enable_cyc = bool(getattr(self.args, "enable_horizon_cyclical", True))
        if h_feat_name in X_train_df.columns:
            raise ValueError(
                f"{self.log_prefix} horizon_feature_name '{h_feat_name}' collides with an existing "
                f"training column; rename it via config.horizon_feature_name."
            )

        # X repeat: 每行复制 H 次（row0×H, row1×H, ...）
        # 用 pandas 原生 repeat 保留各列 dtype（np.repeat(.values) 会把混合 dtype 退化为 object）
        X_long_df = X_train_df.loc[X_train_df.index.repeat(H)].reset_index(drop=True)

        # FeatureEngineering 在 dropna 前已生成目标日外生列 base_h1..base_hH；
        # melt 按每个长表行的 h 选择对应列并折叠回 base 名，避免 dropna 后丢失
        # 尾部目标日外生。内生派生列保持原点行 i 的值。
        expanded_map = {}
        for col in X_train_df.columns:
            match = re.match(r"^(.+)_h(\d+)$", str(col))
            if match:
                expanded_map.setdefault(match.group(1), {})[int(match.group(2))] = col
        expanded_cols = {col for by_h in expanded_map.values() for col in by_h.values()}
        if expanded_cols:
            X_long_df = X_long_df.drop(columns=list(expanded_cols), errors="ignore")
            origin_rows = np.repeat(np.arange(N), H)
            h_for_row = np.tile(np.arange(1, H + 1), N)
            for base, by_h in expanded_map.items():
                values = np.full(N * H, np.nan, dtype=float)
                for h, source_col in by_h.items():
                    mask = h_for_row == h
                    values[mask] = X_train_df[source_col].to_numpy(dtype=float)[origin_rows[mask]]
                X_long_df[base] = values

        # horizon feature: h = 1..H, tile N 次
        h_values = np.tile(np.arange(1, H + 1), N)
        X_long_df[h_feat_name] = h_values
        h_feature_names = [h_feat_name]

        if enable_cyc:
            period = self._resolve_horizon_period(H)
            X_long_df[f"{h_feat_name}_sin"] = np.sin(2 * np.pi * h_values / period)
            X_long_df[f"{h_feat_name}_cos"] = np.cos(2 * np.pi * h_values / period)
            h_feature_names += [f"{h_feat_name}_sin", f"{h_feat_name}_cos"]

        # Y melt: C-order flatten → y_long[i*H + h] = Y[i, h]，与 X repeat 对齐
        y_long = Y_train_df.values.flatten()
        y_col_name = Y_train_df.columns[0] if len(Y_train_df.columns) > 0 else "y"
        Y_long_df = pd.DataFrame({y_col_name: y_long})

        # sample_weight repeat（每行权重复制 H 次，保持与 X_long 行对齐）
        sw_long = None
        if sample_weight is not None:
            sw_long = np.repeat(np.asarray(sample_weight), H)

        logger.info(
            f"{self.log_prefix} horizon_feature melt: {N}×{H} → {N * H} rows, "
            f"added features: {h_feature_names}"
        )
        return X_long_df, Y_long_df, sw_long, h_feature_names

    def _prepare_single_output_fit_data(self, X_train_df_processed, Y_train_df_processed, categorical_features):
        """
        为单输出回归器准备训练/验证切分与 fit kwargs。
        目前仅对 LightGBM 点预测接入 early stopping。
        """
        fit_kwargs = {}
        sample_weight = getattr(self, "sample_weight", None)
        y_values = np.ravel(Y_train_df_processed.values)
        model_type = str(self.model_type).lower()
        if model_type not in ["lightgbm", "lgb"]:
            # 非 LightGBM 模型不做验证集切分（无早停）；sample_weight 统一传递，
            # 各模型封装自行决定是否使用（CatBoost native 路径下权重已内嵌 Pool，
            # 封装会忽略显式 sample_weight）
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            return X_train_df_processed, y_values, fit_kwargs

        total_rows = len(X_train_df_processed)
        horizon = int(getattr(self.args, "horizon", 1) or 1)
        val_size = max(horizon, int(math.floor(total_rows * 0.1)))
        if total_rows <= (val_size + 1):
            logger.info(f"{self.log_prefix} Skip early stopping: insufficient rows ({total_rows}) for val_size={val_size}.")
            if sample_weight is not None:
                fit_kwargs["sample_weight"] = sample_weight
            return X_train_df_processed, y_values, fit_kwargs

        X_fit = X_train_df_processed.iloc[:-val_size].copy()
        y_fit = y_values[:-val_size]
        X_val = X_train_df_processed.iloc[-val_size:].copy()
        y_val = y_values[-val_size:]
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["eval_metric"] = "mae"
        fit_kwargs["early_stopping_rounds"] = int(getattr(self.args, "patience", 100) or 100)
        if sample_weight is not None:
            # 权重需与 X_fit 行对齐(切掉验证段)
            fit_kwargs["sample_weight"] = sample_weight[:-val_size]
        logger.info(
            f"{self.log_prefix} Early stopping enabled: "
            f"train_rows={len(X_fit)}, val_rows={len(X_val)}, patience={fit_kwargs['early_stopping_rounds']}"
        )
        return X_fit, y_fit, fit_kwargs

    def _prepare_multi_output_fit_data(
        self,
        X_train_df_processed,
        Y_train_df_processed,
        categorical_features,
        native_data_bundle,
    ):
        """
        为 Direct 多输出模型准备共享训练集和按输出拆分的验证集参数。
        """
        fit_kwargs_list = [{} for _ in range(Y_train_df_processed.shape[1])]
        model_type = str(self.model_type).lower()
        sample_weight = getattr(self, "sample_weight", None)
        if model_type not in ["lightgbm", "lgb", "xgboost", "xgb", "catboost", "cat"]:
            # RF 等无早停机制的模型：不做验证集切分，但统一传递 sample_weight
            if sample_weight is not None:
                for output_idx in range(Y_train_df_processed.shape[1]):
                    fit_kwargs_list[output_idx]["sample_weight"] = sample_weight
            return X_train_df_processed, Y_train_df_processed, fit_kwargs_list

        total_rows = len(X_train_df_processed)
        horizon = int(getattr(self.args, "horizon", 1) or 1)
        val_size = max(horizon, int(math.floor(total_rows * 0.1)))
        if total_rows <= (val_size + 1):
            logger.info(
                f"{self.log_prefix} Skip multi-output early stopping: "
                f"insufficient rows ({total_rows}) for val_size={val_size}."
            )
            # 无验证切分时,各输出直接用全量权重
            if sample_weight is not None:
                for output_idx in range(Y_train_df_processed.shape[1]):
                    fit_kwargs_list[output_idx]["sample_weight"] = sample_weight
            return X_train_df_processed, Y_train_df_processed, fit_kwargs_list

        X_fit = X_train_df_processed.iloc[:-val_size].copy()
        Y_fit = Y_train_df_processed.iloc[:-val_size].copy()
        X_val = X_train_df_processed.iloc[-val_size:].copy()
        Y_val = Y_train_df_processed.iloc[-val_size:].copy()
        patience = int(getattr(self.args, "patience", 100) or 100)
        # 多输出权重按训练切分对齐(各输出共享同一权重序列)
        sw_fit = sample_weight[:-val_size] if sample_weight is not None else None

        for output_idx in range(Y_train_df_processed.shape[1]):
            fit_kwargs = {
                "eval_set": [(X_val, np.ravel(Y_val.iloc[:, output_idx].values))],
                "eval_metric": "mae",
                "early_stopping_rounds": patience,
            }
            if sw_fit is not None:
                fit_kwargs["sample_weight"] = sw_fit
            # fit kwargs 统一组装，各模型封装自行识别/忽略不支持的项
            # （categorical_feature 仅 LightGBM/CatBoost 使用，XGBoost 经 **kwargs 忽略；
            #   native 容器多输出场景不启用，此处仅为接口一致性保留）
            if categorical_features:
                fit_kwargs["categorical_feature"] = categorical_features
            if native_data_bundle.get("enabled"):
                fit_kwargs["native_train_data"] = native_data_bundle.get("train_native")
            fit_kwargs_list[output_idx] = fit_kwargs

        logger.info(
            f"{self.log_prefix} Multi-output early stopping enabled: "
            f"train_rows={len(X_fit)}, val_rows={len(X_val)}, outputs={Y_train_df_processed.shape[1]}, "
            f"patience={patience}"
        )
        return X_fit, Y_fit, fit_kwargs_list

    def _hyperparameters_tuning(self, X_train, Y_train):
        """
        模型超参数调优 (Grid Search / Randomized Search with TimeSeriesSplit)
        """
        logger.info(f"{self.log_prefix} Starting hyperparameter tuning...")

        base_param_grid = self._get_tuning_param_grid(self.model_type)
        if not base_param_grid:
            raise ValueError(f"Unsupported model_type for hyperparameter tuning: {self.model_type}")

        base_estimator = self.model_factory.create_model(
            # 调参阶段保留日志，便于确认真实搜索空间
            model_type=self.model_type,
            model_params=self.model_params
        )

        # Wrap in MultiOutputRegressor if the method is multi-output
        if Y_train.shape[1] == 1:
            model_for_tuning = base_estimator.model
            tuned_param_grid = base_param_grid
        else:
            model_for_tuning = MultiOutputRegressor(
                base_estimator.model,
                n_jobs=self._resolve_worker_count("multi_output_n_jobs", default=1),
            )
            tuned_param_grid = {f"estimator__{k}": v for k, v in base_param_grid.items()}

        # TimeSeriesSplit for cross-validation
        # n_splits determines how many train-test splits to generate.
        # The test set size will be at least self.horizon.
        tscv = TimeSeriesSplit(n_splits=self.args.tuning_n_splits)

        # Use GridSearchCV for exhaustive search or RandomizedSearchCV for faster search
        # RandomizedSearchCV is generally preferred for larger search spaces
        tuning_metric = getattr(self.args, "tuning_metric", None)
        if not tuning_metric:
            loss_name = get_loss_name_from_model_params(self.model_type, self.model_params)
            tuning_metric = get_scorer_by_loss_name(
                loss_name,
                delta=float(getattr(self.args, "huber_delta", 1.0)),
            )
        search = RandomizedSearchCV(
            estimator=model_for_tuning,
            param_distributions=tuned_param_grid,
            n_iter=10, # Number of parameter settings that are sampled
            scoring=tuning_metric,
            cv=tscv,
            verbose=1,
            n_jobs=-1, # Use all available cores
            random_state=42
        )
        search.fit(X_train, Y_train)
        logger.info(f"{self.log_prefix} Best hyperparameters found: {search.best_params_}")
        logger.info(f"{self.log_prefix} Best score: {search.best_score_}")

        # Update model_params with the best ones
        best_params_estimator = {k.replace('estimator__', ''): v for k, v in search.best_params_.items()}
        self.model_params.update(best_params_estimator)
        logger.info(f"{self.log_prefix} Model parameters updated with best tuning results.")
        
        return search.best_estimator_ # Return the best model direct
    
    def train(self, X_train, Y_train, feature_scaler, target_scaler, categorical_features):
        """
        模型训练
        """
        # 训练集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        resolved = self._get_resolved_strategy()
        target_feature = str(getattr(self.args, "target_feature", "y"))
        expected_source_columns = [
            f"{target_feature}_shift_{step}"
            for step in resolved.target_plan.label_steps
        ]
        if list(Y_train_df.columns) != expected_source_columns:
            raise ValueError(
                f"{self.log_prefix} training targets do not match resolved TargetPlan: "
                f"expected {expected_source_columns}, got {list(Y_train_df.columns)}."
            )
        if self._should_use_fourmethods_baseline_training():
            return self._train_fourmethods_baseline(
                X_train_df=X_train_df,
                Y_train_df=Y_train_df,
                categorical_features=categorical_features,
            )
        # ------------------------------
        # 时间衰减样本权重（在增强之前计算：基于原始时序位置定 age；
        # 增强行随后通过 bootstrap 源行索引继承对应权重——
        # 否则增强行 append 在末尾会按位置获得"最新"的最高权重，与设计意图相反）
        # ------------------------------
        base_sample_weight = None
        if bool(getattr(self.args, "enable_time_decay_sample_weight", False)):
            base_sample_weight = compute_time_decay_weights(
                n_samples=len(X_train_df),
                n_per_day=int(getattr(self.args, "n_per_day", 1) or 1),
                halflife_days=float(getattr(self.args, "decay_halflife_days", 14.0)),
            )
        # ------------------------------
        # 数据增强 + 特征选择
        # ------------------------------
        X_train_df, Y_train_df, selected_features, self.sample_weight = self._prepare_training_data(
            X_train_df, Y_train_df, categorical_features, sample_weight=base_sample_weight
        )
        if self.sample_weight is not None and base_sample_weight is not None:
            logger.info(
                f"{self.log_prefix} Time-decay sample_weight enabled: "
                f"n_samples={len(self.sample_weight)}, halflife_days="
                f"{getattr(self.args, 'decay_halflife_days', 14.0)}, "
                f"latest_weight={float(self.sample_weight[len(base_sample_weight) - 1]):.4f}, "
                f"oldest_weight={float(self.sample_weight[0]):.4f}"
            )
        # ------------------------------
        # 学习率配置（固定 or 自动）
        # ------------------------------
        resolved_lr = resolve_learning_rate(
            base_learning_rate=self.model_params.get("learning_rate"),
            n_samples=len(X_train_df),
            auto_enabled=bool(getattr(self.args, "enable_auto_learning_rate", False)),
            min_lr=float(getattr(self.args, "auto_lr_min", 0.005)),
            max_lr=float(getattr(self.args, "auto_lr_max", 0.2)),
        )
        if resolved_lr is not None:
            self.model_params["learning_rate"] = resolved_lr
            logger.info(f"{self.log_prefix} Using learning_rate={resolved_lr:.6f}")
        # ------------------------------
        # horizon_feature melt（USMD/MSMD 专用）：宽表→长表，Y 降为单列，
        # 后续缩放/训练/分位数路径自动走单输出分支（成本 H×→1×）
        # ------------------------------
        if self._should_use_horizon_feature():
            X_train_df, Y_train_df, self.sample_weight, horizon_h_features = self._melt_to_horizon_long(
                X_train_df, Y_train_df, self.sample_weight
            )
            # 特征选择发生在 melt 之前（宽表），h 特征是新追加的，需补入 selected_features
            if selected_features is not None:
                selected_features = list(selected_features) + horizon_h_features
        # ------------------------------
        # 归一化/标准化
        # ------------------------------
        # 特征预处理（训练模式）
        X_train_df_processed, actual_categorical = feature_scaler.fit_transform(X_train_df, categorical_features)
        # 目标变量预处理（训练模式）
        Y_train_df_processed = target_scaler.fit_transform(Y_train_df)
        # feature_scaler.validate_features(X_train_df_processed, stage="training")
        
        # 根据编码策略决定是否传递 categorical_feature
        if self.args.encode_categorical_features:
            lgbm_categorical = None  # 已编码为整数，不传递 categorical_feature
        else:
            lgbm_categorical = actual_categorical  # 未编码，传递 categorical_feature 让 LightGBM 处理
        native_data_bundle = self._prepare_native_model_data(
            X_train_df_processed,
            Y_train_df_processed,
            actual_categorical,
        )
        self.native_data_bundle = native_data_bundle
        # ------------------------------
        # Hyperparameter tuning (if enabled)
        # ------------------------------
        if self.args.perform_tuning:
            best_model = self._hyperparameters_tuning(X_train_df_processed, Y_train_df_processed)
            return best_model, feature_scaler, target_scaler, selected_features
        # ------------------------------
        # 模型训练
        # ------------------------------
        if self.args.enable_ensemble and str(getattr(self.args, "predict_type", "point")).lower() == "point":
            if getattr(self, "sample_weight", None) is not None:
                logger.warning(
                    f"{self.log_prefix} time-decay sample_weight not supported for ensemble path; skipped."
                )
            # 成员规格：ensemble_model_specs 非空时取代 ensemble_models，
            # 每项 {model, params?, scale?}；否则由 ensemble_models 退化为 {model: mt}
            ensemble_specs = list(getattr(self.args, "ensemble_model_specs", None) or []) or [
                {"model": mt} for mt in self.args.ensemble_models
            ]
            member_names = [str(spec.get("model", "")) for spec in ensemble_specs]
            logger.info(f"{self.log_prefix} Ensemble models: {member_names}, Ensemble method: {self.args.ensemble_method}")
            logger.info(f"{self.log_prefix} {'-' * 50}")

            # 成员线程上限：与单模型路径同一 model_thread_count 纪律
            # （窗口并行测试时 payload 已强制为 1）
            ensemble_threads = self._resolve_worker_count("model_thread_count", default=1)
            base_models = []
            for spec in ensemble_specs:
                model_type = str(spec.get("model", "")).strip()
                # 成员参数 = 全局 model_params 覆盖 + 该成员独立 params（成员优先）
                member_params = {**self.model_param_overrides, **(spec.get("params") or {})}
                member_params = self._thread_limited_params(model_type, member_params, ensemble_threads)
                model_wrapper = self.model_factory.create_model(
                    model_type=model_type,
                    model_params=member_params,
                )
                # 多数封装的可训练对象是 wrapper.model（裸 sklearn 估计器）；
                # SeasonalTemplateModel 等无底层估计器（model=None），权重在 wrapper 自身，
                # 此时直接用 wrapper（同样满足 fit/predict 契约）
                estimator = model_wrapper.model if model_wrapper.model is not None else model_wrapper
                if Y_train_df_processed.shape[1] > 1:
                    estimator = MultiOutputRegressor(
                        estimator,
                        n_jobs=self._resolve_worker_count("multi_output_n_jobs", default=1),
                    )
                # 成员独立预处理（线性成员需要；树成员保持原量纲/原生 NaN 处理）：
                # scale -> StandardScaler；impute -> 中位数填补（训练窗起始行长滞后
                # 特征为 NaN，GBDT 原生容忍，线性成员必须填补，predict 端同样生效）
                member_preprocessor = None
                if spec.get("scale") and spec.get("impute"):
                    member_preprocessor = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ])
                elif spec.get("scale"):
                    member_preprocessor = StandardScaler()
                elif spec.get("impute"):
                    member_preprocessor = SimpleImputer(strategy="median")
                base_models.append((model_type, estimator, member_preprocessor))
           
            ensemble = TimeSeriesEnsembleRegressor(
                base_models=base_models,
                config=EnsembleConfig(
                    method=getattr(self.args, "ensemble_method", "averaging"),
                    val_ratio=float(getattr(self.args, "ensemble_val_ratio", 0.2)),
                    random_state=42,
                    parallel_workers=self._resolve_worker_count("ensemble_parallel_workers", default=1),
                ),
            )
            y_train_input = (
                np.ravel(Y_train_df_processed.values)
                if Y_train_df_processed.shape[1] == 1
                else Y_train_df_processed.values
            )
            ensemble.fit(X_train_df_processed, y_train_input)
            
            logger.info(f"{self.log_prefix} Ensemble training completed!")
            return ensemble, feature_scaler, target_scaler, selected_features
        else:
            predict_type = str(getattr(self.args, "predict_type", "point")).lower()
            model_type = self.model_type
            # ------------------------------
            # 单模型 - 点预测
            # ------------------------------
            if predict_type == "point":
                if resolved.training_plan.layout == TrainingLayout.COMPOSITE:
                    # ------------------------------
                    # Blend（Direct+Recursive 融合）：分离 Y，分别训练两个子模型
                    # ------------------------------
                    logger.info(f"{self.log_prefix} Training blend (Direct+Recursive) model...")
                    logger.info(f"{self.log_prefix} {'-' * 71}")
                    dir_cols = list(resolved.target_plan.direct.column_names)
                    rec_cols = list(resolved.target_plan.recursive.column_names)
                    Y_direct = Y_train_df_processed[dir_cols]
                    Y_recursive = Y_train_df_processed[rec_cols]

                    # Direct 子模型（multioutput）
                    logger.info(f"{self.log_prefix} Blend Direct sub-model ({len(dir_cols)} outputs)...")
                    X_fit_d, Y_fit_d, fit_kwargs_list_d = self._prepare_multi_output_fit_data(
                        X_train_df_processed, Y_direct, actual_categorical, native_data_bundle,
                    )
                    model_direct = DirectMultiOutputRegressor(
                        estimator_factory=lambda: self._create_model_instance(
                            model_type=model_type, model_params=self.model_params, log_params=False,
                        ),
                        n_jobs=self._resolve_worker_count("multi_output_n_jobs", default=1),
                        log_prefix=self.log_prefix,
                    )
                    model_direct.fit(X_fit_d, Y_fit_d, fit_kwargs_list=fit_kwargs_list_d)

                    # Recursive 子模型（单输出）
                    logger.info(f"{self.log_prefix} Blend Recursive sub-model (1 output)...")
                    X_fit_r, y_fit_r, fit_kwargs_r = self._prepare_single_output_fit_data(
                        X_train_df_processed, Y_recursive, actual_categorical,
                    )
                    estimator_recursive = self._create_model_instance(
                        model_type=model_type, model_params=self.model_params,
                    )
                    if lgbm_categorical is not None:
                        fit_kwargs_r["categorical_feature"] = lgbm_categorical
                    if native_data_bundle.get("enabled"):
                        fit_kwargs_r["native_train_data"] = native_data_bundle.get("train_native")
                    estimator_recursive.fit(X_fit_r, np.ravel(y_fit_r), **fit_kwargs_r)

                    model = BlendArtifact(
                        direct_model=model_direct,
                        recursive_model=estimator_recursive,
                        weights=BlendWeights.from_args(self.args),
                        metadata=MultistepArtifactMetadata.from_strategy(
                            resolved,
                            selected_features,
                        ),
                    )
                    logger.info(f"{self.log_prefix} Blend training completed!")
                    return model, feature_scaler, target_scaler, selected_features
                elif resolved.training_plan.layout in {
                    TrainingLayout.SINGLE_OUTPUT,
                    TrainingLayout.HORIZON_LONG,
                }:
                    logger.info(f"{self.log_prefix} Training single-output regressor...")
                    logger.info(f"{self.log_prefix} {'-' * 71}")
                    estimator_wrapper = self.model_factory.create_model(
                        model_type=model_type,
                        model_params=self.model_params,
                    )
                    model = estimator_wrapper
                    X_fit, y_fit, fit_kwargs = self._prepare_single_output_fit_data(
                        X_train_df_processed,
                        Y_train_df_processed,
                        actual_categorical,
                    )
                    # fit kwargs 统一组装，各模型封装自行识别/忽略不支持的项
                    # （native_train_data 仅 CatBoost 封装使用，其余经 **kwargs 忽略）
                    if lgbm_categorical is not None:
                        fit_kwargs["categorical_feature"] = lgbm_categorical
                    if native_data_bundle.get("enabled"):
                        fit_kwargs["native_train_data"] = native_data_bundle.get("train_native")
                    model.fit(X_fit, y_fit, **fit_kwargs)
                else:
                    logger.info(f"{self.log_prefix} Training multi-output regressor with {Y_train_df_processed.shape[1]} outputs...")
                    logger.info(f"{self.log_prefix} {'-' * 71}")
                    strategy = str(getattr(self.args, "multi_output_strategy", "multioutput")).lower()
                    if strategy == "regressor_chain":
                        if getattr(self, "sample_weight", None) is not None:
                            logger.warning(
                                f"{self.log_prefix} time-decay sample_weight not supported for "
                                f"RegressorChain/MultiOutputRegressor; skipped."
                            )
                        estimator_wrapper = self.model_factory.create_model(
                            model_type=model_type,
                            model_params=self.model_params,
                        )
                        model = self._build_multi_output_model(
                            estimator_wrapper.model,
                            n_outputs=Y_train_df_processed.shape[1],
                        )
                        model.fit(X_train_df_processed, Y_train_df_processed)
                    else:
                        X_fit, Y_fit, fit_kwargs_list = self._prepare_multi_output_fit_data(
                            X_train_df_processed,
                            Y_train_df_processed,
                            actual_categorical,
                            native_data_bundle,
                        )
                        regressor_cls = (
                            HorizonAlignedDirectRegressor
                            if self._should_use_horizon_aligned_direct()
                            else DirectMultiOutputRegressor
                        )
                        model = regressor_cls(
                            estimator_factory=lambda: self._create_model_instance(
                                model_type=model_type,
                                model_params=self.model_params,
                                log_params=False,
                            ),
                            n_jobs=self._resolve_worker_count("multi_output_n_jobs", default=1),
                            log_prefix=self.log_prefix,
                        )
                        model.fit(X_fit, Y_fit, fit_kwargs_list=fit_kwargs_list)
                logger.info(f"{self.log_prefix} Model training completed!")
                return model, feature_scaler, target_scaler, selected_features
            # ------------------------------
            # 单模型 - 分位数预测
            # ------------------------------
            probabilistic_spec = getattr(self.args, "probabilistic_spec", None)
            if probabilistic_spec is None:
                probabilistic_spec = resolve_probabilistic_spec(self.args)
            logger.info(
                f"{self.log_prefix} Training quantile models for "
                f"quantiles={list(probabilistic_spec.quantiles)}"
            )
            logger.info(f"{self.log_prefix} {'-' * 71}")

            def train_single_backend(q, X, Y, _categorical_features):
                _, model_q = self._train_quantile_single_model(
                    q,
                    model_type,
                    X,
                    Y,
                    lgbm_categorical,
                )
                return model_q

            def train_blend_backend(q, X, Y_direct, Y_recursive, _categorical_features):
                _, model_q = self._train_blend_quantile_single_model(
                    q,
                    model_type,
                    X,
                    Y_direct,
                    Y_recursive,
                    lgbm_categorical,
                    actual_categorical,
                )
                return model_q

            quantile_trainer = QuantileTrainer(
                spec=probabilistic_spec,
                model_type=model_type,
                pred_method=str(self.args.pred_method),
                train_single=train_single_backend,
                train_blend=train_blend_backend,
                max_workers=self._resolve_worker_count(
                    "quantile_parallel_workers",
                    default=1,
                ),
            )
            quantile_bundle = quantile_trainer.fit(
                X_train_df_processed,
                Y_train_df_processed,
                categorical_features=actual_categorical,
            )
            if resolved.spec.rollout == RolloutFamily.BLEND:
                weights = BlendWeights.from_args(self.args)
                quantile_bundle.metadata = {
                    **dict(quantile_bundle.metadata),
                    "blend_weights": weights.metadata(),
                }
            logger.info(f"{self.log_prefix} Quantile model training completed!")
            return quantile_bundle, feature_scaler, target_scaler, selected_features

    def model_save(
        self,
        model,
        *,
        feature_scaler=None,
        target_transform=None,
        selected_features=None,
        input_schema=None,
        target_scaler=None,
        target_decomposer=None,
    ):
        """
        保存唯一权威 ForecastModelBundle；legacy 参数仅用于旧调用方过渡。
        """
        logger.info(f"{self.log_prefix} Model training result saving...")
        logger.info(f"{self.log_prefix} {'-' * 66}")
        probabilistic_spec = getattr(self.args, "probabilistic_spec", None)
        if probabilistic_spec is None:
            probabilistic_spec = resolve_probabilistic_spec(self.args)
        if target_transform is None:
            # 旧调用方没有共享变换栈时保留已有状态，但不再双写 sidecar。
            target_transform = target_decomposer
            if target_transform is None and target_scaler is not None:
                target_transform = target_scaler
        schema = dict(input_schema or {})
        if "columns" not in schema and feature_scaler is not None:
            schema["columns"] = list(
                getattr(feature_scaler, "training_columns", ()) or ()
            )
        resolved = self._get_resolved_strategy()
        schema["multistep"] = MultistepArtifactMetadata.from_strategy(
            resolved,
            schema.get("columns", ()),
        ).payload()
        if bool(getattr(self.args, "enable_global_training", False)):
            series_id_col = str(getattr(self.args, "series_id_feature", "series_id"))
            category_mappings = getattr(feature_scaler, "category_mappings", {}) or {}
            category_info = getattr(feature_scaler, "category_info", {}) or {}
            known_series_ids = list(
                category_mappings.get(series_id_col, {}).get("categories", ())
                or category_info.get(series_id_col, ())
            )
            if not known_series_ids:
                raise ValueError(
                    f"{self.log_prefix} global panel model artifact requires training "
                    f"series IDs for '{series_id_col}'."
                )
            schema["panel"] = {
                "series_id_feature": series_id_col,
                "known_series_ids": known_series_ids,
                "unknown_series_policy": str(
                    getattr(self.args, "global_unknown_series_policy", "raise") or "raise"
                ).lower(),
            }
        bundle = ForecastModelBundle(
            schema_version=1,
            model=model,
            feature_scaler=feature_scaler,
            target_transform=target_transform,
            selected_features=tuple(selected_features or ()),
            input_schema=schema,
            probabilistic_spec=probabilistic_spec,
            model_type=str(getattr(self.args, "model_type", "unknown")),
            pred_method=str(getattr(self.args, "pred_method", "unknown")),
        )
        model_deploy = ModelDeployPkl(save_file_path=self.args.checkpoints_dir.joinpath("model.pkl"))
        model_deploy.save_model(bundle)
        bundle.write_schema_json(
            self.args.checkpoints_dir.joinpath("probabilistic_schema.json")
        )
        logger.info(f"{self.log_prefix} Model saved to {self.args.checkpoints_dir.joinpath('model.pkl')}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
