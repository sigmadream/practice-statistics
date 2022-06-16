import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .distribution import Distribution, underride
from .cdf import Cdf
from .hazard import Hazard


class Pmf(Distribution):
    """Represents a probability Mass Function (PMF)."""

    def copy(self, deep=True):
        """Make a copy.
        :return: new Pmf
        """
        return Pmf(self, copy=deep)

    def make_pmf(self, **kwargs):
        """Make a Pmf from the Pmf.
        :return: Pmf
        """
        return self

    # Pmf overrides the arithmetic operations in order
    # to provide fill_value=0 and return a Pmf.

    def add(self, x, **kwargs):
        """Override add to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.add(self, x, **kwargs)
        return Pmf(s)

    __add__ = add
    __radd__ = add

    def sub(self, x, **kwargs):
        """Override the - operator to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.subtract(self, x, **kwargs)
        return Pmf(s)

    __sub__ = sub
    __rsub__ = sub

    def mul(self, x, **kwargs):
        """Override the * operator to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.multiply(self, x, **kwargs)
        return Pmf(s)

    __mul__ = mul
    __rmul__ = mul

    def div(self, x, **kwargs):
        """Override the / operator to default fill_value to 0.
        x: Distribution or sequence
        returns: Pmf
        """
        underride(kwargs, fill_value=0)
        s = pd.Series.divide(self, x, **kwargs)
        return Pmf(s)

    __div__ = div
    __rdiv__ = div
    __truediv__ = div
    __rtruediv__ = div

    def normalize(self):
        """Make the probabilities add up to 1 (modifies self).
        :return: normalizing constant
        """
        total = self.sum()
        self /= total
        return total

    def mean(self):
        """Computes expected value.
        :return: float
        """
        # TODO: error if not normalized
        # TODO: error if the quantities are not numeric
        return np.sum(self.ps * self.qs)

    def mode(self, **kwargs):
        """Most common value.
        If multiple quantities have the maximum probability,
        the first maximal quantity is returned.
        :return: float
        """
        return self.idxmax(**kwargs)

    def var(self):
        """Variance of a PMF.
        :return: float
        """
        m = self.mean()
        d = self.qs - m
        return np.sum(d ** 2 * self.ps)

    def std(self):
        """Standard deviation of a PMF.
        :return: float
        """
        return np.sqrt(self.var())

    def choice(self, *args, **kwargs):
        """Makes a random sample.
        Uses the probabilities as weights unless `p` is provided.
        args: same as np.random.choice
        kwargs: same as np.random.choice
        :return: NumPy array
        """
        underride(kwargs, p=self.ps)
        return np.random.choice(self.qs, *args, **kwargs)

    def add_dist(self, x):
        """Computes the Pmf of the sum of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.add.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs + x)

    def sub_dist(self, x):
        """Computes the Pmf of the diff of values drawn from self and other.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.subtract.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs - x)

    def mul_dist(self, x):
        """Computes the Pmf of the product of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.multiply.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs * x)

    def div_dist(self, x):
        """Computes the Pmf of the ratio of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Pmf
        """
        if isinstance(x, Distribution):
            return self.convolve_dist(x, np.divide.outer)
        else:
            return Pmf(self.ps.copy(), index=self.qs / x)

    def convolve_dist(self, dist, ufunc):
        """Convolve two distributions.
        dist: Distribution
        ufunc: elementwise function for arrays
        :return: new Pmf
        """
        dist = dist.make_pmf()
        qs = ufunc(self.qs, dist.qs).flatten()
        ps = np.multiply.outer(self.ps, dist.ps).flatten()
        series = pd.Series(ps).groupby(qs).sum()

        return Pmf(series)

    def gt_dist(self, x):
        """Probability that a value from pmf1 is greater than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.greater).sum()
        else:
            return self[self.qs > x].sum()

    def lt_dist(self, x):
        """Probability that a value from pmf1 is less than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.less).sum()
        else:
            return self[self.qs < x].sum()

    def ge_dist(self, x):
        """Probability that a value from pmf1 is >= than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.greater_equal).sum()
        else:
            return self[self.qs >= x].sum()

    def le_dist(self, x):
        """Probability that a value from pmf1 is <= than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.less_equal).sum()
        else:
            return self[self.qs <= x].sum()

    def eq_dist(self, x):
        """Probability that a value from pmf1 equals a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.equal).sum()
        else:
            return self[self.qs == x].sum()

    def ne_dist(self, x):
        """Probability that a value from pmf1 is <= than a value from pmf2.
        dist1: Distribution object
        dist2: Distribution object
        :return: float probability
        """
        if isinstance(x, Distribution):
            return self.pmf_outer(x, np.not_equal).sum()
        else:
            return self[self.qs != x].sum()

    def pmf_outer(self, dist, ufunc):
        """Computes the outer product of two PMFs.
        dist: Distribution object
        ufunc: function to apply to the qs
        :return: NumPy array
        """
        dist = dist.make_pmf()
        qs = ufunc.outer(self.qs, dist.qs)
        ps = np.multiply.outer(self.ps, dist.ps)
        return qs * ps

    def bar(self, **options):
        """Make a bar plot.
        Note: A previous version of this function use pd.Series.plot.bar,
        but that was a mistake, because that function treats the quantities
        as categorical, even if they are numerical, leading to hilariously
        unexpected results!
        options: passed to plt.bar
        """
        plt.bar(self.qs, self.ps, **options)

    def make_joint(self, other, **options):
        """Make joint distribution (assuming independence).
        :param self:
        :param other:
        :param options: passed to Pmf constructor
        :return: new Pmf
        """
        qs = pd.MultiIndex.from_product([self.qs, other.qs])
        ps = np.multiply.outer(self.ps, other.ps).flatten()
        return Pmf(ps, index=qs, **options)

    def marginal(self, i, name=None):
        """Gets the marginal distribution of the indicated variable.
        i: index of the variable we want
        name: string
        :return: Pmf
        """
        # The following is deprecated now
        # return Pmf(self.sum(level=i))

        # here's the new version
        return Pmf(self.groupby(level=i).sum())

    def conditional(self, i, val, name=None):
        """Gets the conditional distribution of the indicated variable.
        i: index of the variable we're conditioning on
        val: the value the ith variable has to have
        name: string
        :return: Pmf
        """
        pmf = Pmf(self.xs(key=val, level=i), copy=True)
        pmf.normalize()
        return pmf

    def update(self, likelihood, data):
        """Bayesian update.
        likelihood: function that takes (data, hypo) and returns
                    likelihood of data under hypo, P(data|hypo)
        data: in whatever format likelihood understands
        :return: normalizing constant
        """
        for hypo in self.qs:
            self[hypo] *= likelihood(data, hypo)

        return self.normalize()

    def max_prob(self):
        """Value with the highest probability.
        :return: the value with the highest probability
        """
        return self.idxmax()

    def make_cdf(self, **kwargs):
        """Make a Cdf from the Pmf.
        :return: Cdf
        """
        normalize = kwargs.pop("normalize", False)

        cumulative = np.cumsum(self)
        cdf = Cdf(cumulative, self.index.copy(), **kwargs)

        if normalize:
            cdf.normalize()

        return cdf

    def make_surv(self, **kwargs):
        """Make a Surv from the Pmf.
        :return: Surv
        """
        cdf = self.make_cdf()
        return cdf.make_surv(**kwargs)

    def make_hazard(self, normalize=False, **kwargs):
        """Make a Hazard from the Pmf.
        :return: Hazard
        """
        surv = self.make_surv()
        haz = Hazard(self / (self + surv), **kwargs)
        haz.total = getattr(surv, "total", 1.0)
        if normalize:
            self.normalize()
        return haz

    def make_same(self, dist):
        """Convert the given dist to Pmf
        :param dist:
        :return: Pmf
        """
        return dist.make_pmf()

    @staticmethod
    def from_seq(
            seq,
            normalize=True,
            sort=True,
            ascending=True,
            dropna=True,
            na_position="last",
            **options
    ):
        """Make a PMF from a sequence of values.
        seq: iterable
        normalize: whether to normalize the Pmf, default True
        sort: whether to sort the Pmf by values, default True
        ascending: whether to sort in ascending order, default True
        dropna: whether to drop NaN values, default True
        na_position: If ‘first’ puts NaNs at the beginning,
                        ‘last’ puts NaNs at the end.
        options: passed to the pd.Series constructor
        NOTE: In the current implementation, `from_seq` sorts numerical
           quantities whether you want to or not.  If keeping
           the order of the elements is important, let me know and
           I'll rethink the implementation
        :return: Pmf object
        """
        # compute the value counts
        series = pd.Series(seq).value_counts(normalize=normalize, sort=False, dropna=dropna)

        # make the result a Pmf
        # (since we just made a fresh Series, there is no reason to copy it)
        options["copy"] = False
        pmf = Pmf(series, **options)

        # sort in place, if desired
        if sort:
            pmf.sort_index(inplace=True, ascending=ascending, na_position=na_position)
        return pmf
