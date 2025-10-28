import abc
import gin

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from src.data.visualizer import ProjectionVisualizer


class ReducerBase(abc.ABC):
    """Dimensionality reduction class."""

    def __init__(self):
        self.model = self._init_model()
        self.input_dir = None
        self.output_dir = None

    @abc.abstractmethod
    def get_associated_visualizer(self):
        pass

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @abc.abstractmethod
    def get_reduced_df(self, df: pd.DataFrame):
        pass


class ScikitReducerBase(ReducerBase):
    """Base class for scikit-learn based reducers."""

    def __init__(self, n_dims: int = 2):
        super().__init__()
        self.visualizer = ProjectionVisualizer(n_dims=n_dims)
        self.n_dims = n_dims
        self.random_state = 42

    def get_associated_visualizer(self):
        """Return the visualizer object for this reducer."""
        return self.visualizer

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

    def __init__(self, n_dims: int = 2):
        super().__init__(n_dims=n_dims)
        self.model = self._init_model()

    def _init_model(self):
        """Initialize the model."""
        pca = PCA(n_components=self.n_dims, random_state=self.random_state)
        return pca


@gin.configurable
class TsneReducer(ScikitReducerBase):
    """T-SNE reducer class."""

    def __init__(self, n_dims: int = 2, perplexity: float = 30.0, max_iter: int = 1000):
        super().__init__(n_dims=n_dims)
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.model = self._init_model()

    def _init_model(self):
        """Initialize the model."""
        tsne_model = TSNE(
            n_components=self.n_dims,
            random_state=self.random_state,
            perplexity=self.perplexity,
            max_iter=self.max_iter,
        )
        return tsne_model


@gin.configurable
class UmapReducer(ScikitReducerBase):
    """UMAP reducer class."""

    def __init__(self, n_dims: int = 2, n_neighbors: int = 15, min_dist: float = 0.1):
        super().__init__(n_dims=n_dims)
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.model = self._init_model()

    def _init_model(self):
        """Initialize the model."""
        umap_model = UMAP(
            n_components=self.n_dims,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
        )
        return umap_model
