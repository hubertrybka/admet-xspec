from src.predictor.predictor_base import PredictorBase
from typing import List
from sklearn.metrics import roc_auc_score
import logging

import gin
import lightgbm as lgb
import pandas as pd
import numpy as np


@gin.configurable()
class LightGBMClassifier(PredictorBase):
    lightgbm_hyperparameters = [
        "boosting_type",
        "objective",
        "metric",
        "num_leaves",
        "learning_rate",
        "feature_fraction",
        "bagging_fraction",
        "bagging_freq",
        "verbose",
    ]

    def __init__(self, params: dict):
        super().__init__()

        self._validate_hyperparameters(params)
        self.params = params

        assert (
            self.params["objective"] == "binary"
            or self.params["objective"] == "multiclass"
        ), "Wrong 'objective' parameter supplied to LightGBM classifier, did you pass a regression objective?"

        self.model_ready = (
            None  # look to _init_model (may become obsolete in near future!)
        )

    def _validate_hyperparameters(self, hyperparameters_to_validate):
        valid_hyperparameters = [
            h in self.lightgbm_hyperparameters for h in hyperparameters_to_validate
        ]
        if not all(valid_hyperparameters):
            raise ValueError(
                f"Invalid hyperparameter: {hyperparameters_to_validate[valid_hyperparameters.index(False)]}"
            )

    def _get_lgbm_feature_set(self, smiles_list: List[str]) -> pd.DataFrame:
        return pd.DataFrame(self.featurizer.featurize(smiles_list))

    def _get_lgbm_label_set(self, target_list: List[float]) -> pd.DataFrame:
        return pd.DataFrame(np.array(target_list))

    def _get_lgbm_dataset(
        self, smiles_list: List[str], target_list: List[float]
    ) -> lgb.Dataset:
        X, y = (
            self._get_lgbm_feature_set(smiles_list),
            self._get_lgbm_label_set(target_list),
        )

        return lgb.Dataset(X, y)

    @property
    def name(self) -> str:
        """Return the name of the predictor."""
        return "LightGBM_clf"

    @property
    def uses_internal_featurizer(self) -> bool:
        """Return True if the model uses a proprietary featurizer."""
        return False

    def get_hyperparameters(self) -> dict:
        """Return the hyperparameters of the model."""
        return self.params

    def set_hyperparameters(self, hyperparams: dict):
        """Inject hyperparameters into the model."""
        self._validate_hyperparameters(hyperparams)
        self.params = hyperparams

    def _init_model(self):
        """Initialize the model."""
        # TODO
        # Violates the logic present in the SciKit models, note that this has to be implemented
        # due to _init_model being abstract in PredictorBase (and so forming part of the Predictor interface)
        # Perhaps we should reconsider _init_model being part of the general interface for all predictors
        self.model_ready = False

    def predict(self, smiles_list: List[str]) -> List[float]:
        """Predict the target values for the given smiles list."""
        if not self.model_ready:
            raise RuntimeError("Called predict without having trained the model first.")

        X = self._get_lgbm_feature_set(smiles_list)
        y_pred: np.ndarray = self.model.predict(X)

        return list(y_pred)

    def train(self, smiles_list: List[str], target_list: List[float]):
        """Train the model with set hyperparameters."""
        lgb_train = self._get_lgbm_dataset(smiles_list, target_list)

        logging.info(f"Training {self.name} with fixed hyperparameters")

        gbm = lgb.train(
            self.params,
            lgb_train,
            num_boost_round=10,
            valid_sets=[lgb_train],  # eval training data
        )

        logging.info(f"Fitting of {self.name} has converged")

        self.model = gbm
        self.model_ready = True

    def train_optimize(self, smiles_list: List[str], target_list: List[float]):
        """Train the model and optimize hyperparameters"""
        pass

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        """
        Evaluate the model on the given smiles list and target list.
        Returns a dictionary of metrics appropriate for the task.
        """
        X_test = self._get_lgbm_feature_set(smiles_list)
        y_pred = self.model.predict(X_test)

        y_test = self._get_lgbm_label_set(target_list)
        score = roc_auc_score(y_test, y_pred)

        return {"score": score}
