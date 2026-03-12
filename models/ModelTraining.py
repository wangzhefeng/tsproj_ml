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

from models.ModelFactory import ModelFactory
from models.ModelSaveLoad import ModelDeployPkl
from models.ModelEnsemble_optim import TimeSeriesEnsembleRegressor, EnsembleConfig
from utils.log_util import logger

# global variable
LOGGING_LABEL = Path(__file__).name[:-3]


class Trainer:
    
    def __init__(self, args: Dict, log_prefix: str):
        self.args = args
        self.log_prefix = log_prefix
        self.model_params = copy.deepcopy(self.args.model_params)
        self.model_factory = ModelFactory(log_prefix=log_prefix)

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
        search = RandomizedSearchCV(
            estimator=model_for_tuning,
            param_distributions=tuned_param_grid,
            n_iter=10, # Number of parameter settings that are sampled
            scoring=self.args.tuning_metric,
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
            return best_model, feature_scaler
        # ------------------------------
        # 模型训练
        # ------------------------------
        if self.args.enable_ensemble:
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
            return ensemble, feature_scaler
        else:
            # 单模型
            lgbm_estimator = self.model_factory.create_model(
                model_type=getattr(self.args, "model_type", "lightgbm"), 
                model_params=self.model_params,
            )
            # 模型训练
            if Y_train_df.shape[1] == 1:
                logger.info(f"{self.log_prefix} Training single output LGBMRegressor...")
                logger.info(f"{self.log_prefix} {'-' * 71}")
                logger.info(f"{self.log_prefix} Model training...")
                model = lgbm_estimator
                model.fit(X_train_df_processed, np.ravel(Y_train_df.values), categorical_feature=lgbm_categorical)
            elif Y_train_df.shape[1] > 1:
                logger.info(f"{self.log_prefix} Training MultiOutputRegressor with {Y_train.shape[1]} outputs...")
                logger.info(f"{self.log_prefix} {'-' * 71}")
                logger.info(f"{self.log_prefix} Model training...")
                model = MultiOutputRegressor(estimator = lgbm_estimator.model, n_jobs=-1)
                model.fit(X_train_df_processed, Y_train_df)
            
            logger.info(f"{self.log_prefix} Model training completed!")
            return model, feature_scaler

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
