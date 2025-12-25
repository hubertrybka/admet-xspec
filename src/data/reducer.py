import abc
import gin

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from src.data.visualizer import ProjectionVisualizer
from typing import Dict, Any


class ReducerBase(abc.ABC):
    """
    Base class for dimensionality reduction algorithms.

    Provides abstract interface for reducing high-dimensional molecular feature spaces
    to lower dimensions for visualization or analysis.

    :param params: Configuration parameters for the reduction algorithm
    :type params: Dict[str, Any]
    :ivar model: The initialized reduction model
    :ivar input_dir: Directory for input data (optional)
    :type input_dir: Path or None
    :ivar output_dir: Directory for output data (optional)
    :type output_dir: Path or None
    """

    def __init__(self, params):
        self.model = self._init_model(params)
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def get_associated_visualizer(self):
        """
        Return the visualizer object for this reducer.

        :return: Visualizer instance compatible with this reducer's output
        :rtype: Visualizer
        """
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        Name of the reducer.

        :return: Human-readable algorithm name (e.g., 'PCA', 'UMAP')
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def get_reduced_df(self, df: pd.DataFrame):
        """
        Apply dimensionality reduction to DataFrame.

        :param df: DataFrame with high-dimensional features
        :type df: pd.DataFrame
        :return: DataFrame with reduced dimensions
        :rtype: pd.DataFrame
        """
        pass

    @abc.abstractmethod
    def _init_model(self, params):
        pass


class ScikitReducerBase(ReducerBase):
    """
    Base class for scikit-learn based dimensionality reducers.

    :param n_dims: Number of dimensions in reduced space
    :type n_dims: int
    :param params: Algorithm-specific parameters
    :type params: Dict[str, Any] or None
    :param plot_title: Title for visualization plots
    :type plot_title: str or None
    :ivar visualizer: ProjectionVisualizer instance for this reducer
    :type visualizer: ProjectionVisualizer
    """

    def __init__(
        self,
        n_dims: int = 2,
        params: Dict[str, Any] = None,
        plot_title: str | None = None,
    ):
        super().__init__(params)
        self.visualizer = ProjectionVisualizer(
            n_dims=n_dims, projection_type=self.name, plot_title=plot_title
        )

    def get_associated_visualizer(self):
        """
        Return the visualizer object for this reducer.

        :return: ProjectionVisualizer configured for this reducer's output
        :rtype: ProjectionVisualizer
        """
        return self.visualizer

    def get_reduced_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit reducer and transform DataFrame to lower dimensional space.

        Creates new DataFrame with columns 'dim_1', 'dim_2', ..., 'dim_n' containing
        the reduced feature vectors.

        :param df: DataFrame with high-dimensional features (each column is a feature)
        :type df: pd.DataFrame
        :return: DataFrame with n_dims columns representing reduced space
        :rtype: pd.DataFrame
        """
        reduced_data_ndarray = self.model.fit_transform(df)
        reduced_df = pd.DataFrame(
            {
                f"dim_{i + 1}": reduced_data_ndarray[:, i]
                for i in range(reduced_data_ndarray.shape[1])
            }
        )

        return reduced_df


@gin.configurable
class PcaReducer(ScikitReducerBase):
    """PCA reducer class."""

    def __init__(
        self, n_dims: int = 2, random_state: int = 42, plot_title: str | None = None
    ):
        params = {
            "n_components": n_dims,
            "random_state": random_state,
        }
        super().__init__(n_dims=n_dims, params=params, plot_title=plot_title)

    def _init_model(self, params):
        """Initialize the model."""
        pca = PCA(**params)
        return pca

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "PCA"


@gin.configurable
class TsneReducer(ScikitReducerBase):
    """T-SNE reducer class."""

    def __init__(
        self,
        n_dims: int = 2,
        perplexity: float = 30.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        params = {
            "perplexity": perplexity,
            "max_iter": max_iter,
            "random_state": random_state,
        }
        super().__init__(n_dims=n_dims, params=params)

    def _init_model(self, params):
        """Initialize the model."""
        tsne_model = TSNE(**params)
        return tsne_model

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "t-SNE"


@gin.configurable
class UmapReducer(ScikitReducerBase):
    """UMAP reducer class."""

    def __init__(
        self,
        n_dims: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        params = {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "random_state": random_state,
        }
        super().__init__(n_dims=n_dims, params=params)

    def _init_model(self, params):
        """Initialize the model."""
        umap_model = UMAP(**params)
        return umap_model

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "UMAP"
