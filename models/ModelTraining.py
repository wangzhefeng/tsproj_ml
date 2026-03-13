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
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.multioutput import MultiOutputRegressor, RegressorChain
from sklearn.model_selection import (
    TimeSeriesSplit, 
    GridSearchCV, 
    RandomizedSearchCV
)

from features.DataAugment import TimeSeriesAugmenter
from features.FeatureSelection import FeatureSelector
from models.ModelFactory import ModelFactory
from models.ModelSaveLoad import ModelDeployPkl
from models.ModelEnsemble_optim import TimeSeriesEnsembleRegressor, EnsembleConfig
from models.learning_rate import resolve_learning_rate
from models.losses import get_scorer_by_loss_name
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Trainer:
    
    def __init__(self, args: Dict, log_prefix: str):
        self.args = args
        self.log_prefix = log_prefix
        self.model_params = copy.deepcopy(self.args.model_params)
        self.model_factory = ModelFactory(log_prefix=log_prefix)
        self.augmenter = TimeSeriesAugmenter(
            enabled=bool(getattr(self.args, "enable_data_augmentation", False)),
            augmentation_ratio=float(getattr(self.args, "augmentation_ratio", 0.2)),
            feature_noise_std=float(getattr(self.args, "augmentation_feature_noise_std", 0.01)),
            target_noise_std=float(getattr(self.args, "augmentation_target_noise_std", 0.005)),
            random_state=int(getattr(self.args, "augmentation_random_state", 42)),
            log_prefix=self.log_prefix,
        )

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
        logger.info(f"{self.log_prefix} Multi-output strategy: MultiOutputRegressor")
        # 在受限执行环境中避免 loky 多进程信号量权限问题
        return MultiOutputRegressor(estimator=base_estimator, n_jobs=1)

    def _inject_quantile_params(self, model_type: str, params: Dict, quantile: float) -> Dict:
        """
        为不同回归器注入分位数预测参数
        """
        params_q = copy.deepcopy(params)
        mt = str(model_type).lower()
        if mt in ["lightgbm", "lgb"]:
            params_q["objective"] = "quantile"
            params_q["metric"] = "quantile"
            params_q["alpha"] = float(quantile)
        elif mt in ["xgboost", "xgb"]:
            # xgboost >= 2.0 支持 quantileerror
            params_q["objective"] = "reg:quantileerror"
            params_q["quantile_alpha"] = float(quantile)
            params_q["eval_metric"] = "quantile"
        elif mt in ["catboost", "cat"]:
            params_q["loss_function"] = f"Quantile:alpha={float(quantile)}"
        else:
            logger.warning(f"{self.log_prefix} model_type={model_type} has no explicit quantile objective, using default params.")
        return params_q

    def _build_feature_selector(self, categorical_features):
        return FeatureSelector(
            enabled=bool(getattr(self.args, "enable_feature_selection", False)),
            method=str(getattr(self.args, "feature_selection_method", "f_regression")),
            max_features=int(getattr(self.args, "feature_selection_max_features", 80)),
            min_features=int(getattr(self.args, "feature_selection_min_features", 10)),
            force_keep_features=list(categorical_features or []),
            log_prefix=self.log_prefix,
        )

    def _prepare_training_data(self, X_train, Y_train, categorical_features):
        X_prepared = X_train.copy()
        Y_prepared = Y_train.copy()
        # 1) 数据增强（仅训练集）
        X_prepared, Y_prepared = self.augmenter.augment(
            X_prepared, Y_prepared, categorical_features=categorical_features
        )
        # 2) 特征选择（fit on training split）
        selector = self._build_feature_selector(categorical_features=categorical_features)
        X_selected, selected_features = selector.fit_transform(
            X_prepared, Y_prepared, categorical_features=categorical_features
        )
        return X_selected, Y_prepared, selected_features

    def _hyperparameters_tuning(self, X_train, Y_train):
        """
        模型超参数调优 (Grid Search / Randomized Search with TimeSeriesSplit)
        """
        logger.info(f"{self.log_prefix} Starting hyperparameter tuning...")

        # Define parameter grid
        param_grid = {
            'estimator__num_leaves': [15, 31, 63],
            'estimator__learning_rate': [0.01, 0.05, 0.1],
            'estimator__feature_fraction': [0.7, 0.8, 0.9],
            'estimator__lambda_l1': [0.1, 0.5, 1.0],
            'estimator__lambda_l2': [0.1, 0.5, 1.0],
            'estimator__min_child_samples': [20, 50, 100], # Corresponds to min_data_in_leaf
        }

        # Base LightGBM estimator
        lgbm_base = self.model_factory.create_model(
            model_type=self.args.model_type,
            model_params=self.model_params
        )

        # Wrap in MultiOutputRegressor if the method is multi-output
        if Y_train.shape[1] == 1:
            model_for_tuning = lgbm_base.model
            tuned_param_grid = {k.replace("estimator__", ""): v for k, v in param_grid.items()}
        else:
            model_for_tuning = MultiOutputRegressor(lgbm_base.model)
            tuned_param_grid = param_grid

        # TimeSeriesSplit for cross-validation
        # n_splits determines how many train-test splits to generate.
        # The test set size will be at least self.horizon.
        tscv = TimeSeriesSplit(n_splits=self.args.tuning_n_splits)

        # Use GridSearchCV for exhaustive search or RandomizedSearchCV for faster search
        # RandomizedSearchCV is generally preferred for larger search spaces
        tuning_metric = getattr(self.args, "tuning_metric", None)
        if not tuning_metric:
            tuning_metric = get_scorer_by_loss_name(
                getattr(self.args, "loss", "mae"),
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
    
    def train(self, X_train, Y_train, feature_scaler, categorical_features):
        """
        模型训练
        """
        # 训练集
        X_train_df = X_train.copy()
        Y_train_df = Y_train.copy()
        # ------------------------------
        # 数据增强 + 特征选择
        # ------------------------------
        X_train_df, Y_train_df, selected_features = self._prepare_training_data(
            X_train_df, Y_train_df, categorical_features
        )
        # ------------------------------
        # 学习率配置（固定 or 自动）
        # ------------------------------
        resolved_lr = resolve_learning_rate(
            base_learning_rate=getattr(self.args, "learning_rate", None),
            n_samples=len(X_train_df),
            auto_enabled=bool(getattr(self.args, "enable_auto_learning_rate", False)),
            min_lr=float(getattr(self.args, "auto_lr_min", 0.005)),
            max_lr=float(getattr(self.args, "auto_lr_max", 0.2)),
        )
        if resolved_lr is not None:
            self.model_params["learning_rate"] = resolved_lr
            logger.info(f"{self.log_prefix} Using learning_rate={resolved_lr:.6f}")
        # ------------------------------
        # 归一化/标准化
        # ------------------------------
        # 特征预处理（训练模式）
        X_train_df_processed, actual_categorical = feature_scaler.fit_transform(X_train_df, categorical_features)
        # feature_scaler.validate_features(X_train_df_processed, stage="training")
        
        # 根据编码策略决定是否传递 categorical_feature
        if self.args.encode_categorical_features:
            lgbm_categorical = None  # 已编码为整数，不传递 categorical_feature
        else:
            lgbm_categorical = actual_categorical  # 未编码，传递 categorical_feature 让 LightGBM 处理
        # ------------------------------
        # Hyperparameter tuning (if enabled)
        # ------------------------------
        if self.args.perform_tuning:
            best_model = self._hyperparameters_tuning(X_train_df_processed, Y_train_df)
            return best_model, feature_scaler, selected_features
        # ------------------------------
        # 模型训练
        # ------------------------------
        if self.args.enable_ensemble and str(getattr(self.args, "predict_type", "point")).lower() == "point":
            logger.info(f"{self.log_prefix} Ensemble models: {self.args.ensemble_models}, Ensemble method: {self.args.ensemble_method}")
            logger.info(f"{self.log_prefix} {'-' * 50}")
           
            base_models = []
            for model_type in self.args.ensemble_models:
                model_wrapper = self.model_factory.create_model(model_type=model_type, model_params=self.model_params)
                estimator = model_wrapper.model
                if Y_train_df.shape[1] > 1:
                    estimator = MultiOutputRegressor(estimator)
                base_models.append((model_type, estimator))
            
            ensemble = TimeSeriesEnsembleRegressor(
                base_models=base_models,
                config=EnsembleConfig(
                    method=getattr(self.args, "ensemble_method", "averaging"),
                    val_ratio=float(getattr(self.args, "ensemble_val_ratio", 0.2)),
                    random_state=42,
                ),
            )
            y_train_input = np.ravel(Y_train_df.values) if Y_train_df.shape[1] == 1 else Y_train_df.values
            ensemble.fit(X_train_df_processed, y_train_input)
            
            logger.info(f"{self.log_prefix} Ensemble training completed!")
            return ensemble, feature_scaler, selected_features
        else:
            predict_type = str(getattr(self.args, "predict_type", "point")).lower()
            model_type = getattr(self.args, "model_type", "lightgbm")
            # ------------------------------
            # 单模型 - 点预测
            # ------------------------------
            if predict_type == "point":
                estimator_wrapper = self.model_factory.create_model(
                    model_type=model_type,
                    model_params=self.model_params,
                )
                if Y_train_df.shape[1] == 1:
                    logger.info(f"{self.log_prefix} Training single-output regressor...")
                    logger.info(f"{self.log_prefix} {'-' * 71}")
                    model = estimator_wrapper
                    fit_kwargs = {}
                    if str(model_type).lower() in ["lightgbm", "lgb"] and lgbm_categorical is not None:
                        fit_kwargs["categorical_feature"] = lgbm_categorical
                    model.fit(X_train_df_processed, np.ravel(Y_train_df.values), **fit_kwargs)
                else:
                    logger.info(f"{self.log_prefix} Training multi-output regressor with {Y_train_df.shape[1]} outputs...")
                    logger.info(f"{self.log_prefix} {'-' * 71}")
                    model = self._build_multi_output_model(estimator_wrapper.model, n_outputs=Y_train_df.shape[1])
                    model.fit(X_train_df_processed, Y_train_df)
                logger.info(f"{self.log_prefix} Model training completed!")
                return model, feature_scaler, selected_features
            # ------------------------------
            # 单模型 - 分位数预测
            # ------------------------------
            quantiles = [float(q) for q in getattr(self.args, "quantiles", [0.1, 0.5, 0.9])]
            if not quantiles:
                raise ValueError(f"{self.log_prefix} predict_type=quantile but quantiles is empty.")
            quantile_models = {}
            logger.info(f"{self.log_prefix} Training quantile models for quantiles={quantiles}")
            logger.info(f"{self.log_prefix} {'-' * 71}")
            for q in quantiles:
                params_q = self._inject_quantile_params(model_type=model_type, params=self.model_params, quantile=q)
                estimator_wrapper_q = self.model_factory.create_model(
                    model_type=model_type,
                    model_params=params_q,
                )
                if Y_train_df.shape[1] == 1:
                    model_q = estimator_wrapper_q
                    fit_kwargs_q = {}
                    if str(model_type).lower() in ["lightgbm", "lgb"] and lgbm_categorical is not None:
                        fit_kwargs_q["categorical_feature"] = lgbm_categorical
                    model_q.fit(X_train_df_processed, np.ravel(Y_train_df.values), **fit_kwargs_q)
                else:
                    model_q = self._build_multi_output_model(estimator_wrapper_q.model, n_outputs=Y_train_df.shape[1])
                    model_q.fit(X_train_df_processed, Y_train_df)
                quantile_models[q] = model_q
            median_q = min(quantiles, key=lambda x: abs(x - 0.5))
            quantile_bundle = {
                "predict_type": "quantile",
                "quantiles": quantiles,
                "median_quantile": median_q,
                "models": quantile_models,
            }
            logger.info(f"{self.log_prefix} Quantile model training completed!")
            return quantile_bundle, feature_scaler, selected_features

    def model_save(self, model):
        """
        模型保存
        """
        logger.info(f"{self.log_prefix} Model training result saving...")
        logger.info(f"{self.log_prefix} {'-' * 66}")
        model_deploy = ModelDeployPkl(save_file_path=self.args.checkpoints_dir.joinpath("model.pkl"))
        model_deploy.save_model(model)
        logger.info(f"{self.log_prefix} Model saved to {self.args.checkpoints_dir.joinpath('model.pkl')}")




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
