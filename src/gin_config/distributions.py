"""
This module defines various probability distributions used for hyperparameter optimization.
It includes the following distributions, all parametrized by a lower and upper bound (a and b):
- Uniform
- LogUniform
- QUniform (discrete uniform)
- QLogUniform (discrete log uniform)
"""

from scipy.stats import uniform, loguniform
import ray.tune as tune
import gin
import abc


class Distribution(abc.ABC):
    """
    Base class for hyperparameter distributions.

    Provides unified interface for sampling from bounded distributions and
    integration with Ray Tune for hyperparameter optimization.

    :param min: Lower bound of the distribution
    :type min: float
    :param max: Upper bound of the distribution
    :type max: float
    :ivar lower: Configured lower bound
    :type lower: float
    :ivar upper: Configured upper bound
    :type upper: float
    :ivar distribution: Initialized scipy.stats distribution object
    :type distribution: scipy.stats distribution
    """

    def __init__(self, min=0, max=1):
        super().__init__()
        self.lower = min
        self.upper = max
        self.distribution = self._init_distribution()

    def rvs(self, size=1, **kwargs):
        """
        Generate random samples from the distribution.

        :param size: Number of samples to generate
        :type size: int
        :param kwargs: Additional arguments passed to scipy.stats distribution.rvs()
        :return: Single random sample (when size=1) or array of samples
        :rtype: float or np.ndarray
        """

        samples = self.distribution.rvs(size=size, **kwargs)
        return samples[0]

    @abc.abstractmethod
    def _init_distribution(self):
        """
        This method must be implemented by subclasses.
        It should return a scipy.stats distribution object.
        """
        pass


@gin.register
class Uniform(Distribution):
    def _init_distribution(self):
        return uniform(self.lower, self.upper)

    def __str__(self):
        return f"Uniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """
        Get Ray Tune compatible distribution object.

        Used by ChemProp models for hyperparameter search.

        :return: Ray Tune uniform distribution
        :rtype: ray.tune.search.sample.Domain
        """
        return tune.uniform(self.lower, self.upper)


@gin.register
class LogUniform(Distribution):
    def _init_distribution(self):
        return loguniform(self.lower, self.upper)

    def __str__(self):
        return f"LogUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """
        Get Ray Tune compatible distribution object.

        Used by ChemProp models for hyperparameter search.

        :return: Ray Tune uniform distribution
        :rtype: ray.tune.search.sample.Domain
        """
        return tune.loguniform(self.lower, self.upper)


@gin.register
class QUniform(Distribution):
    def _init_distribution(self):
        return uniform(self.lower, self.upper)

    def rvs(self, size=1, **kwargs):
        samples = self.distribution.rvs(size=size, **kwargs)
        return [int(sample) for sample in samples] if len(samples) > 1 else int(samples)

    def __str__(self):
        return f"QUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """
        Get Ray Tune compatible distribution object.

        Used by ChemProp models for hyperparameter search.

        :return: Ray Tune uniform distribution
        :rtype: ray.tune.search.sample.Domain
        """
        return tune.randint(self.lower, self.upper + 1)


@gin.register
class QLogUniform(Distribution):
    def _init_distribution(self):
        return loguniform(self.lower, self.upper)

    def rvs(self, size=1, **kwargs):
        samples = self.distribution.rvs(size=size, **kwargs)
        return [int(sample) for sample in samples] if len(samples) > 1 else int(samples)

    def __str__(self):
        return f"QLogUniform (lower={self.lower}, upper={self.upper})"

    def get_ray_distrib(self):
        """
        Get Ray Tune compatible distribution object.

        Used by ChemProp models for hyperparameter search.

        :return: Ray Tune uniform distribution
        :rtype: ray.tune.search.sample.Domain
        """
        return tune.qloguniform(self.lower, self.upper, q=1, base=10)
