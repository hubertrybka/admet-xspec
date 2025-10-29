import abc
import gin

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from src.data.visualizer import ProjectionVisualizer
from typing import Dict, Any


class ReducerBase(abc.ABC):
    """Dimensionality reduction class."""

    def __init__(self, model_params, visualizer_params):
        self.model = self._init_model(model_params)
        self._init_visualizer(**visualizer_params)
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def _init_model(self, params):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def _init_visualizer(self, **kwargs):
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the reducer."""
        pass

    @abc.abstractmethod
    def get_associated_visualizer(self):
        pass

    @abc.abstractmethod
    def get_unique_output_suffix(self):
        pass

    @abc.abstractmethod
    def get_reduced_df(self, df: pd.DataFrame):
        pass


class ScikitReducerBase(ReducerBase):
    """Base class for scikit-learn based reducers."""

    def __init__(
        self,
        n_dims: int = 2,
        model_params: Dict[str, Any] = None,
        visualizer_params: Dict[str, Any] = None,
    ):
        self.n_dims = n_dims
        self._init_visualizer(**visualizer_params)
        super().__init__(model_params, visualizer_params)

    @property
    def name(self) -> str:
        return "sck_reducer"

    def _init_visualizer(self, **kwargs):
        self.visualizer = ProjectionVisualizer(
            n_dims=self.n_dims, projection_type=self.name, **kwargs
        )

    def get_associated_visualizer(self):
        """Return the visualizer object for this reducer."""
        return self.visualizer

    def get_unique_output_suffix(self) -> str:
        return f"{self.n_dims}_dims"

    def get_reduced_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform PCA on a dataframe, return that dataframe with dim_1, ..., dim_n columns corr. to PCA"""
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
        model_params = {
            "n_components": n_dims,
            "random_state": random_state,
        }
        visualizer_params = {
            "plot_title": plot_title,
        }
        self.n_dims = n_dims
        self.random_state = random_state
        super().__init__(n_dims, model_params, visualizer_params)

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "PCA"

    def _init_model(self, params):
        """Initialize the model."""
        pca = PCA(**params)
        return pca

    def get_unique_output_suffix(self) -> str:
        return f"{self.n_dims}_dims"


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
        self.n_dims = n_dims
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.random_state = random_state
        super().__init__(n_dims=n_dims, params=params)

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "t-SNE"

    def _init_model(self, params):
        """Initialize the model."""
        tsne_model = TSNE(**params)
        return tsne_model

    def get_unique_output_suffix(self) -> str:
        return f"{self.n_dims}_dims_{self.perplexity}_perpl"


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
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.random_state = random_state

    @property
    def name(self) -> str:
        """Name of the reducer."""
        return "UMAP"

    def _init_model(self, params):
        """Initialize the model."""
        umap_model = UMAP(**params)
        return umap_model

    def get_unique_output_suffix(self) -> str:
        return f"{self.n_dims}_{self.n_neighbors}_neigh"
