import matplotlib.pyplot as plt

from .distribution import Distribution
from .pmf import Pmf
from .surv import Surv


class Hazard(Distribution):
    """Represents a Hazard function."""

    def copy(self, deep=True):
        """Make a copy.
        :return: new Pmf
        """
        return Hazard(self, copy=deep)

    # Hazard inherits __call__ from Distribution

    def normalize(self):
        """Normalize the hazard function (modifies self).
        :return: normalizing constant
        """
        old_total = getattr(self, "total", 1.0)
        self.total = 1.0
        return old_total

    def bar(self, **options):
        """Make a bar plot.
        Note: A previous version of this function use pd.Series.plot.bar,
        but that was a mistake, because that function treats the quantities
        as categorical, even if they are numerical, leading to hilariously
        unexpected results!
        options: passed to plt.bar
        """
        plt.bar(self.qs, self.ps, **options)

    def make_surv(self, **kwargs):
        """Make a Surv from the Hazard.
        :return: Surv
        """
        normalize = kwargs.pop("normalize", False)
        ps = (1 - self).cumprod()
        total = getattr(self, "total", 1.0)
        surv = Surv(ps * total, **kwargs)
        surv.total = total

        if normalize:
            surv.normalize()
        return surv

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Hazard.
        :return: Cdf
        """
        return self.make_surv().make_cdf(**kwargs)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Hazard.
        :return: Pmf
        """
        return self.make_surv().make_cdf().make_pmf(**kwargs)

    def make_same(self, dist):
        """Convert the given dist to Hazard.
        :param dist:
        :return: Hazard
        """
        return dist.make_hazard()

    @staticmethod
    def from_seq(seq, **kwargs):
        """Make a Hazard from a sequence of values.
        seq: iterable
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        kwargs: passed to the pd.Series constructor
        :return: Hazard object
        """
        pmf = Pmf.from_seq(seq, **kwargs)
        return pmf.make_hazard()
