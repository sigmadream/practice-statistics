import numpy as np
from scipy.interpolate import interp1d

from .cdf import Cdf
from .distribution import Distribution, underride


class Surv(Distribution):
    """Represents a survival function (complementary CDF)."""

    def copy(self, deep=True):
        """Make a copy.
        :return: new Surv
        """
        return Surv(self, copy=deep)

    @staticmethod
    def from_seq(seq, normalize=True, sort=True, **options):
        """Make a Surv from a sequence of values.
        seq: iterable
        normalize: whether to normalize the Surv, default True
        sort: whether to sort the Surv by values, default True
        options: passed to the pd.Series constructor
        :return: Surv object
        """
        cdf = Cdf.from_seq(seq, normalize=normalize, sort=sort, **options)
        return cdf.make_surv()

    def step(self, **options):
        """Plot the Surv as a step function.
        :param options: passed to pd.Series.plot
        :return:
        """
        underride(options, drawstyle="steps-post")
        self.plot(**options)

    def normalize(self):
        """Normalize the survival function (modifies self).
        :return: normalizing constant
        """
        old_total = getattr(self, "total", 1.0)
        self.ps /= old_total
        self.total = 1.0
        return old_total

    @property
    def forward(self, **kwargs):
        """Compute the forward survival function
        :param kwargs: keyword arguments passed to interp1d
        :return array of probabilities
        """
        total = getattr(self, "total", 1.0)
        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(total, 0),
        )
        interp = interp1d(self.qs, self.ps, **kwargs)
        return interp

    @property
    def inverse(self, **kwargs):
        """Compute the inverse survival function
        :param kwargs: keyword arguments passed to interp1d
        :return: interpolation function from ps to qs
        """
        total = getattr(self, "total", 1.0)
        underride(
            kwargs,
            kind="previous",
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=(np.nan, np.nan),
        )
        # sort in descending order
        # I don't remember why
        rev = self.sort_values()

        # If the reversed Surv doesn't get all the way to total
        # add a fake entry at -inf
        if rev.iloc[-1] != total:
            rev[-np.inf] = total

        interp = interp1d(rev, rev.index, **kwargs)
        return interp

    # calling a Surv like a function does forward lookup
    __call__ = forward

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Surv.
        :return: Cdf
        """
        normalize = kwargs.pop("normalize", False)
        total = getattr(self, "total", 1.0)
        cdf = Cdf(total - self, **kwargs)
        if normalize:
            cdf.normalize()
        return cdf

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Surv.
        :return: Pmf
        """
        cdf = self.make_cdf()
        pmf = cdf.make_pmf(**kwargs)
        return pmf

    def make_hazard(self, **kwargs):
        """Make a Hazard from the Surv.
        :return: Hazard
        """
        pmf = self.make_pmf()
        at_risk = self + pmf
        haz = Hazard(pmf / at_risk, **kwargs)
        haz.total = getattr(self, "total", 1.0)
        haz.name = self.name
        return haz

    def make_same(self, dist):
        """Convert the given dist to Surv
        :param dist:
        :return: Surv
        """
        return dist.make_surv()
