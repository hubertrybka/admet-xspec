import abc
import io
import gin
import matplotlib.pyplot as plt

from PIL import Image
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class ExplorerBase(abc.ABC):
    """
    Base class for dimensionality reduction data explorers. Subclasses are expected to be wrappers for
    unsupervised learning methods like PCA, t-SNE, UMAP with scikit-learn interface (fit_transform).
    """

    def __init__(self, n_dims: int = 2, plot_title: str = "Data projection"):

        if n_dims not in [2, 3]:
            raise ValueError("n_dims must be either 2 or 3")
        self.n_dims = n_dims

        self.model = self._init_model()

        self.input_dir = None
        self.output_dir = None
        self.plot_title = None

    @abc.abstractmethod
    def _init_model(self):
        """Initialize the model."""
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the explorer."""
        pass

    def get_analyzed_form(self, df: pd.DataFrame) -> np.ndarray:
        """Perform dimensionality reduction of a dataframe"""
        reduced_data = self.model.fit_transform(df)
        return reduced_data

    def get_visualizable_form(
        self, dataset_dict: dict[str, np.ndarray]
    ) -> dict[str, pd.DataFrame]:
        """Convert ndarray to pandas dataframe for visualization."""
        out_dict = {}
        for ds, ndarray in dataset_dict.items():
            out_dict[ds] = pd.DataFrame(
                {f"dim_{i + 1}": ndarray[:, i] for i in range(ndarray.shape[1])}
            )

        return out_dict

    def _visualize_2d(
        self, ndarray: np.ndarray, class_series: pd.Series, class_names: list[str]
    ) -> Image.Image:

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            ndarray[:, 0],
            ndarray[:, 1],
            c=class_series,
            s=40,
        )

        ax.set(
            title=self.plot_title,
            xlabel=self.name + " 1",
            ylabel=self.name + " 2",
        )
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        # Add a legend
        legend1 = ax.legend(
            scatter.legend_elements()[0],
            class_names,
            loc="upper right",
            title="Classes",
        )
        ax.add_artist(legend1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)

        return img

    def _visualize_3d(
        self, ndarray: np.ndarray, class_series: pd.Series, class_names: list[str]
    ) -> Image.Image:

        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

        scatter = ax.scatter(
            ndarray[:, 0],
            ndarray[:, 1],
            ndarray[:, 2],
            c=class_series,
            s=40,
        )

        ax.set(
            title=self.plot_title,
            xlabel=self.name + " 1",
            ylabel=self.name + " 2",
            zlabel=self.name + " 3",
        )
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])

        # Add a legend
        legend1 = ax.legend(
            scatter.legend_elements()[0],
            class_names,
            loc="upper right",
            title="Classes",
        )
        ax.add_artist(legend1)

        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)

        return img

    def get_visualization(
        self, reduced_df_dict: dict[str, pd.DataFrame]
    ) -> Image.Image:
        class_series = pd.concat(
            [pd.Series([i] * len(df)) for i, df in enumerate(reduced_df_dict.values())],
            ignore_index=True,
        )

        concatenated_df = pd.concat(
            [df for df in reduced_df_dict.values()], ignore_index=True
        )
        concatenated_ndarray = concatenated_df.to_numpy()

        img = None
        if self.n_dims == 2:
            img = self._visualize_2d(
                concatenated_ndarray, class_series, list(reduced_df_dict.keys())
            )
        elif self.n_dims == 3:
            img = self._visualize_3d(
                concatenated_ndarray, class_series, list(reduced_df_dict.keys())
            )

        return img


@gin.configurable
class PcaExplorer(ExplorerBase):
    def __init__(self, n_dims: int = 2):
        super().__init__(n_dims=n_dims)

    def name(self) -> str:
        return "PCA"

    def _init_model(self):
        """Initialize the model."""
        pca = PCA(n_components=self.n_dims)
        return pca


@gin.configurable
class TsneExplorer(ExplorerBase):
    def __init__(self, n_dims: int = 2):
        super().__init__(n_dims=n_dims)

    def name(self) -> str:
        return "t-SNE"

    def _init_model(self):
        """Initialize the model."""

        tsne = TSNE(n_components=self.n_dims, init="random", random_state=42)
        return tsne


@gin.configurable
class UmapExplorer(ExplorerBase):
    def __init__(self, n_dims: int = 2):
        super().__init__(n_dims=n_dims)

    def _init_model(self):
        """Initialize the model."""
        umap_model = umap.UMAP(n_components=self.n_dims, random_state=42)
        return umap_model

    def name(self) -> str:
        return "UMAP"
