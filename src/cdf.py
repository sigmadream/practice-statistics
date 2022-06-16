import numpy as np

from .distribution import Distribution, underride
from .pmf import Pmf
from .surv import Surv


class Cdf(Distribution):
    """Represents a Cumulative Distribution Function (CDF)."""

    def copy(self, deep=True):
        """Make a copy.
        :return: new Cdf
        """
        return Cdf(self, copy=deep)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a CDF from a sequence of values.
        seq: iterable
        normalize: whether to normalize the Cdf, default True
        sort: whether to sort the Cdf by values, default True
        options: passed to the pd.Series constructor
        :return: CDF object
        """
        # if normalize==True, normalize AFTER making the Cdf
        # so the last element is exactly 1.0
        pmf = Pmf.from_seq(seq, normalize=False, sort=sort, **options)
        return pmf.make_cdf(normalize=normalize)

    def step(self, **options):
        """Plot the Cdf as a step function.
        :param options: passed to pd.Series.plot
        :return:
        """
        underride(options, drawstyle="steps-post")
        self.plot(**options)

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).
        :return: normalizing constant
        """
        total = self.ps[-1]
        self /= total
        return total

    @property
    def forward(self, **kwargs):
        """Compute the forward Cdf
        :param kwargs: keyword arguments passed to interp1d
        :return interpolation function from qs to ps
        """

        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(0, 1),
        )

        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse Cdf
        :param kwargs: keyword arguments passed to interp1d
        :return: interpolation function from ps to qs
        """
        underride(
            kwargs,
            kind="next",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(self.qs[0], np.nan),
        )

        interp = interp1d(self.ps, self.qs, **kwargs)
        return interp

    # calling a Cdf like a function does forward lookup
    __call__ = forward

    # quantile is the same as an inverse lookup
    quantile = inverse

    def median(self):
        """Median (50th percentile).
        :return: float
        """
        return self.quantile(0.5)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Cdf.
        :param normalize: Boolean, whether to normalize the Pmf
        :return: Pmf
        """
        # TODO: check for consistent behavior of copy flag for all make_x
        normalize = kwargs.pop("normalize", False)

        diff = np.diff(self, prepend=0)
        pmf = Pmf(diff, index=self.index.copy(), **kwargs)
        if normalize:
            pmf.normalize()
        return pmf

    def make_surv(self, **kwargs):
        """Make a Surv object from the Cdf.
        :return: Surv object
        """
        normalize = kwargs.pop("normalize", False)
        total = self.ps[-1]
        surv = Surv(total - self, **kwargs)
        surv.total = total
        if normalize:
            self.normalize()
        return surv

    def make_hazard(self, **kwargs):
        """Make a Hazard from the Cdf.
        :return: Hazard
        """
        pmf = self.make_pmf()
        surv = self.make_surv()
        haz = Hazard(pmf / (pmf + surv), **kwargs)
        haz.total = getattr(surv, "total", 1.0)
        return haz

    def make_same(self, dist):
        """Convert the given dist to Cdf
        :param dist:
        :return: Cdf
        """
        return dist.make_cdf()

    def sample(self, n=1):
        """Samples with replacement using probabilities as weights.
        n: number of values
        :return: NumPy array
        """
        ps = np.random.random(n)
        return self.inverse(ps)

    def max_dist(self, n):
        """Distribution of the maximum of `n` values from this distribution.
        n: integer
        :return: Cdf
        """
        ps = self ** n
        return Cdf(ps, self.index.copy())

    def min_dist(self, n):
        """Distribution of the minimum of `n` values from this distribution.
        n: integer
        :return: Cdf
        """
        ps = 1 - (1 - self) ** n
        return Cdf(ps, self.index.copy())
