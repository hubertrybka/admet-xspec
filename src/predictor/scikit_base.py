import abc
import logging
from typing import List, Dict, Optional, Any

import numpy as np
from sklearn.utils.validation import check_array
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator

from src.data.featurizer import FeaturizerBase


class ScikitPredictorBase(abc.ABC):
    """
    Mixin providing scikit-learn integration for predictors.

    Implements training, prediction, and hyperparameter optimization using sklearn
    estimators and utilities. Designed for multiple inheritance with task-specific
    predictor bases:
      - class MyRegressor(ScikitPredictorBase, RegressorBase): ...
      - class MyClassifier(ScikitPredictorBase, BinaryClassifierBase): ...

    :param params: Initial hyperparameters for the model
    :type params: Dict[str, Any] or None
    :param target_metric: Scoring metric for optimization (e.g., 'r2', 'roc_auc')
    :type target_metric: str or None
    :param params_distribution: Hyperparameter distributions for randomized search
    :type params_distribution: Dict[str, Any] or None
    :param optimization_iterations: Number of parameter settings sampled during optimization
    :type optimization_iterations: int or None
    :param n_folds: Number of cross-validation folds for optimization
    :type n_folds: int or None
    :param n_jobs: Number of parallel jobs for optimization
    :type n_jobs: int or None
    :param kwargs: Additional arguments passed to parent classes
    :ivar featurizer: Featurizer for molecular representation
    :type featurizer: FeaturizerBase or None
    :ivar model: Trained scikit-learn estimator
    :type model: BaseEstimator or None
    :ivar params: Current hyperparameter configuration
    :type params: Dict[str, Any]
    :ivar target_metric: Metric used for optimization scoring
    :type target_metric: str or None
    :ivar hyper_opt: Hyperparameter optimization configuration
    :type hyper_opt: Dict[str, Any]
    """

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        target_metric: Optional[str] = None,
        params_distribution: Optional[Dict[str, Any]] = None,
        optimization_iterations: Optional[int] = None,
        n_folds: Optional[int] = None,
        n_jobs: Optional[int] = None,
        **kwargs,
    ):
        # Let other bases initialize (RegressorBase/BinaryClassifierBase -> PredictorBase)
        super().__init__(**kwargs)
        self.featurizer: Optional[FeaturizerBase] = getattr(self, "featurizer", None)
        self.model: Optional[BaseEstimator] = None
        self.params: Dict[str, Any] = params or {}
        self.target_metric = target_metric
        self.hyper_opt = {
            "n_iter": optimization_iterations or 10,
            "n_folds": n_folds or 3,
            "n_jobs": n_jobs or 1,
            "params_distribution": params_distribution,
        }

    @abc.abstractmethod
    def _init_model(self) -> BaseEstimator:
        """Return an uninitialized sklearn estimator (subclass must implement)."""
        ...

    def _featurize(self, smiles_list: List[str]) -> np.ndarray:
        if self.featurizer is None:
            raise ValueError(
                "Featurizer is not set. Inject a FeaturizerBase before calling this method."
            )
        X = self.featurizer.featurize(smiles_list)
        arr = np.array(X, dtype=np.float32)
        arr = np.atleast_2d(arr)
        return arr

    def train(self, smiles_list: List[str], target_list: List[float]) -> None:
        """
        Train model with current hyperparameters.

        Initializes model, applies configured hyperparameters, featurizes molecules,
        and fits the estimator.

        :param smiles_list: Training molecule SMILES
        :type smiles_list: List[str]
        :param target_list: Training target values
        :type target_list: List[float]
        :rtype: None
        """

        self.model = self._init_model()
        if self.params:
            self.set_hyperparameters(self.params)

        X = self._featurize(smiles_list)
        y = np.array(target_list, dtype=np.float32)

        logging.info(
            f"Training {self.model.__class__.__name__} with hyperparameters: {self.get_hyperparameters()}"
        )
        self.model.fit(X, y)
        logging.info(f"Training complete for {self.model.__class__.__name__}")

    def optimize(self, smiles_list: List[str], target_list: List[float]) -> None:
        """
        Optimize hyperparameters via randomized search with cross-validation.

        Runs RandomizedSearchCV using configured parameter distributions and
        sets best parameters found. Updates self.model with refitted best estimator.

        :param smiles_list: Training molecule SMILES
        :type smiles_list: List[str]
        :param target_list: Training target values
        :type target_list: List[float]
        :raises ValueError: If params_distribution is not provided
        :rtype: None
        """

        if not self.hyper_opt["params_distribution"]:
            raise ValueError(
                "params_distribution must be provided to run optimization."
            )
        self.model = self._init_model()

        X = self._featurize(smiles_list)
        y = np.array(target_list, dtype=np.float32)

        logging.info(
            f"Starting hyperparameter optimization for {self.model.__class__.__name__}"
        )
        rs = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.hyper_opt["params_distribution"],
            n_iter=self.hyper_opt["n_iter"],
            cv=self.hyper_opt["n_folds"],
            verbose=1,
            n_jobs=self.hyper_opt["n_jobs"],
            refit=True,
            scoring=self.target_metric,
        )
        rs.fit(X, y)

        self.params = rs.best_params_.copy()
        self.model = rs.best_estimator_
        logging.info(f"Optimization complete. Best params: {self.params}")

    def predict(self, smiles_list: List[str]) -> List[float]:
        """
        Generate predictions for molecules.

        For classifiers with predict_proba, returns probability of positive class.
        For regressors, returns predicted values. Handles NaN/inf values in features
        by replacing with 0.

        :param smiles_list: Molecule SMILES to predict
        :type smiles_list: List[str]
        :return: Predicted values or class probabilities
        :rtype: List[float]
        :raises ValueError: If model is not trained
        """

        if self.model is None:
            raise ValueError(
                "Model is not initialized. Call `train` or `optimize` first."
            )
        X = self._featurize(smiles_list)

        if np.isnan(X).any() or np.isinf(X).any():
            num_nan = int(np.sum(np.isnan(X)))
            num_inf = int(np.sum(np.isinf(X)))
            logging.warning(
                f"Input contains {num_nan} NaN(s) and {num_inf} infinite(s). Replacing with 0."
            )
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        X = check_array(X, ensure_all_finite=True, dtype=np.float32)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                preds = proba[:, 1]
            else:
                preds = proba.ravel()
        else:
            preds = self.model.predict(X)
        return list(map(float, np.asarray(preds)))

    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Return current hyperparameters.

        Extracts parameters from fitted model or returns stored params if model
        not initialized. Converts numpy types to Python native types.

        :return: Dictionary of hyperparameter names and values
        :rtype: Dict[str, Any]
        """

        if self.model is None:
            return {
                k: (v.item() if isinstance(v, np.generic) else v)
                for k, v in self.params.items()
            }
        hyperparams = self.model.get_params()
        for k, v in list(hyperparams.items()):
            if isinstance(v, np.generic):
                hyperparams[k] = v.item()
        return hyperparams

    def set_hyperparameters(self, params: Dict[str, Any]) -> None:
        """
        Set hyperparameters on the model.

        Initializes model if needed and validates parameters against model's
        accepted parameters.

        :param params: Hyperparameters to set
        :type params: Dict[str, Any]
        :raises ValueError: If any parameter is not supported by the model
        :rtype: None
        """

        if self.model is None:
            self.model = self._init_model()
        # Validate supported params against model.get_params()
        model_params = self.model.get_params()
        for key in params:
            if key not in model_params:
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not accept hyperparameter `{key}`. Supported hyperparameters: {list(model_params.keys())}"
                )
        self.model.set_params(**params)
