from abc import ABC
from typing import List
import abc
import numpy as np
from sklearn import metrics

from src.data.featurizer import FeaturizerBase


class PredictorBase(abc.ABC):
    """
    Base class for molecular property prediction models.

    Provides unified interface for training, prediction, evaluation, and hyperparameter
    optimization. Supports optional featurization for models that don't use internal
    molecular representations.

    :param random_state: Random seed for reproducibility
    :type random_state: int
    :ivar featurizer: Featurizer instance for models requiring external feature extraction
    :type featurizer: FeaturizerBase or None
    :ivar random_state: Configured random seed
    :type random_state: int
    """

    def __init__(self, random_state: int = 42):
        self.featurizer: FeaturizerBase | None = None  # will be set if applicable
        self.random_state = random_state
        # Set random seed for reproducibility
        np.random.seed(random_state)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Return the name of the predictor.

        :return: Human-readable model name (e.g., 'RandomForest', 'ChemProp')
        :rtype: str
        """
        pass

    @property
    def uses_internal_featurizer(self) -> bool:
        """
        Return True if the model uses proprietary featurization.

        Models with internal featurizers (e.g., graph neural networks) should
        override this to return True.

        :return: Whether model handles featurization internally
        :rtype: bool
        """
        return False

    @abc.abstractmethod
    def get_hyperparameters(self) -> dict:
        """
        Return current hyperparameters of the model.

        :return: Dictionary mapping hyperparameter names to values
        :rtype: dict
        """
        pass

    @abc.abstractmethod
    def set_hyperparameters(self, hyperparams: dict):
        """
        Inject hyperparameters into the model.

        :param hyperparams: Dictionary mapping hyperparameter names to values
        :type hyperparams: dict
        :rtype: None
        """
        pass

    @abc.abstractmethod
    def predict(self, smiles_list: List[str]) -> List[float]:
        """
        Predict target values for given SMILES strings.

        :param smiles_list: List of SMILES strings representing molecules
        :type smiles_list: List[str]
        :return: Predicted values (regression) or class probabilities (classification)
        :rtype: List[float]
        """
        pass

    @abc.abstractmethod
    def train(self, smiles_list: List[str], target_list: List[float]):
        """
        Train the model with currently set hyperparameters.

        :param smiles_list: List of SMILES strings for training molecules
        :type smiles_list: List[str]
        :param target_list: List of target values or labels
        :type target_list: List[float]
        :rtype: None
        """
        pass

    @abc.abstractmethod
    def optimize(self, smiles_list: List[str], target_list: List[float]):
        """
        Optimize hyperparameters and set them internally.

        Performs hyperparameter search (e.g., grid search, Bayesian optimization)
        and updates model configuration with best parameters found.

        :param smiles_list: List of SMILES strings for optimization
        :type smiles_list: List[str]
        :param target_list: List of target values or labels
        :type target_list: List[float]
        :rtype: None
        """
        pass

    @abc.abstractmethod
    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        """
        Evaluate the model on given data.

        Returns task-appropriate metrics (e.g., MSE/R² for regression,
        accuracy/AUC for classification).

        :param smiles_list: List of SMILES strings for evaluation molecules
        :type smiles_list: List[str]
        :param target_list: True target values or labels
        :type target_list: List[float]
        :return: Dictionary mapping metric names to values
        :rtype: dict
        """
        pass

    def get_featurizer(self) -> FeaturizerBase | None:
        """
        Return the configured featurizer.

        :return: Featurizer instance if set, None otherwise
        :rtype: FeaturizerBase or None
        """
        return self.featurizer if self.featurizer else None

    def set_featurizer(self, featurizer: FeaturizerBase):
        """
        Inject featurizer into the model.

        :param featurizer: Featurizer instance for molecular representation
        :type featurizer: FeaturizerBase
        :raises AssertionError: If featurizer is not instance of FeaturizerBase
        :rtype: None
        """
        assert isinstance(
            featurizer, FeaturizerBase
        ), "Featurizer must be an instance of FeaturizerBase"
        self.featurizer = featurizer

    def get_cache_key(self) -> str:
        """
        Generate unique cache key for predictor configuration.

        Hash is based on predictor name and featurizer configuration (if present).
        Does not include hyperparameters, which may vary during optimization.

        :return: Cache key identifying this model configuration
        :rtype: str
        """

        feturizer_key = (
            self.featurizer.get_cache_key() if self.featurizer else "nofeaturizer"
        )
        return f"{self.name}_{feturizer_key}"

    @staticmethod
    def get_metric_callable(metric_name: str):
        """
        Return scikit-learn metric function by name.

        :param metric_name: Name of metric ('accuracy', 'roc_auc', 'mse', 'r2', etc.)
        :type metric_name: str
        :return: Callable metric function from sklearn.metrics
        :rtype: callable
        :raises ValueError: If metric_name is not supported
        """

        metrics_dict = {
            "accuracy": metrics.accuracy_score,
            "roc_auc": metrics.roc_auc_score,
            "f1": metrics.f1_score,
            "precision": metrics.precision_score,
            "recall": metrics.recall_score,
            "mse": metrics.mean_squared_error,
            "mae": metrics.mean_absolute_error,
            "r2": metrics.r2_score,
            "rmse": metrics.root_mean_squared_error,
        }
        if metric_name not in metrics_dict.keys():
            raise ValueError(
                f"Invalid metric name: '{metric_name}'. Supported metrics: {list(metrics_dict.keys())}"
            )
        return metrics_dict[metric_name]


class BinaryClassifierBase(PredictorBase, ABC):
    evaluation_metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]
    """
    Base class for binary classification predictors. Implements common evaluation metrics
    and classification thresholding.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        preds = self.predict(smiles_list)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            if m == "roc_auc":
                # roc_auc needs class probabilities
                metrics_dict[m] = self.get_metric_callable(m)(target_list, preds)
            else:
                binary_preds = self.classify(preds)
                metrics_dict[m] = self.get_metric_callable(m)(target_list, binary_preds)
        return metrics_dict

    def classify(self, preds):
        return np.array(preds) > self.class_threshold

    @property
    def class_threshold(self) -> float:
        """Return the classification threshold value. Default is 0.5."""
        return 0.5


class RegressorBase(PredictorBase, ABC):
    evaluation_metrics = ["mse", "rmse", "mae", "r2"]
    """
    Base class for regression predictors. Implements common evaluation metrics.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, smiles_list: List[str], target_list: List[float]) -> dict:
        preds = self.predict(smiles_list)
        metrics_dict = {}
        for m in self.evaluation_metrics:
            metrics_dict[m] = self.get_metric_callable(m)(target_list, preds)
        return metrics_dict
