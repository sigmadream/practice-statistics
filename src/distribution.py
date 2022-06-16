import numpy as np
import pandas as pd


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.
    d: dictionary
    options: keyword args to add to d
    :return: modified d
    """
    for key, val in options.items():
        d.setdefault(key, val)
    return d


class Distribution(pd.Series):
    def __init__(self, *args, **kwargs):
        """Initialize a Pmf.
        Note: this cleans up a weird Series behavior, which is
        that Series() and Series([]) yield different results.
        See: https://github.com/pandas-dev/pandas/issues/16737
        """
        underride(kwargs, name="")
        if args or ("index" in kwargs):
            super().__init__(*args, **kwargs)
        else:
            underride(kwargs, dtype=np.float64)
            super().__init__([], **kwargs)

    @property
    def qs(self):
        """Get the quantities.
        :return: NumPy array
        """
        return self.index.values

    @property
    def ps(self):
        """Get the probabilities.
        :return: NumPy array
        """
        return self.values

    def head(self, n=3):
        """Override Series.head to return a Distribution.
        n: number of rows
        returns: Distribution
        """
        s = super().head(n)
        return self.__class__(s)

    def tail(self, n=3):
        """Override Series.tail to return a Distribution.
        n: number of rows
        returns: Distribution
        """
        s = super().tail(n)
        return self.__class__(s)

    def transform(self, *args, **kwargs):
        """Override to transform the quantities, not the probabilities."""
        qs = self.index.to_series().transform(*args, **kwargs)
        return self.__class__(self.ps, qs, copy=True)

    def _repr_html_(self):
        """Returns an HTML representation of the series.
        Mostly used for Jupyter notebooks.
        """
        df = pd.DataFrame(dict(probs=self))
        return df._repr_html_()

    def __call__(self, qs):
        """Look up quantities.
        qs: quantity or sequence of quantities
        returns: value or array of values
        """
        string_types = (str, bytes, bytearray)

        # if qs is a sequence type, use reindex;
        # otherwise use get
        if hasattr(qs, "__iter__") and not isinstance(qs, string_types):
            s = self.reindex(qs, fill_value=0)
            return s.to_numpy()
        else:
            return self.get(qs, default=0)

    def mean(self):
        """Expected value.
        :return: float
        """
        return self.make_pmf().mean()

    def mode(self, **kwargs):
        """Most common value.
        If multiple quantities have the maximum probability,
        the first maximal quantity is returned.
        :return: float
        """
        return self.make_pmf().mode(**kwargs)

    def var(self):
        """Variance.
        :return: float
        """
        return self.make_pmf().var()

    def std(self):
        """Standard deviation.
        :return: float
        """
        return self.make_pmf().std()

    def median(self):
        """Median (50th percentile).
        There are several definitions of median;
        the one implemented here is just the 50th percentile.
        :return: float
        """
        return self.make_cdf().median()

    def quantile(self, ps, **kwargs):
        """Quantiles.
        Computes the inverse CDF of ps, that is,
        the values that correspond to the given probabilities.
        :return: float
        """
        return self.make_cdf().quantile(ps, **kwargs)

    def credible_interval(self, p):
        """Credible interval containing the given probability.
        p: float 0-1
        :return: array of two quantities
        """
        tail = (1 - p) / 2
        ps = [tail, 1 - tail]
        return self.quantile(ps)

    def choice(self, *args, **kwargs):
        """Makes a random sample.
        Uses the probabilities as weights unless `p` is provided.
        args: same as np.random.choice
        options: same as np.random.choice
        :return: NumPy array
        """
        pmf = self.make_pmf()
        return pmf.choice(*args, **kwargs)

    def sample(self, *args, **kwargs):
        """Samples with replacement using probabilities as weights.
        Uses the inverse CDF.
        n: number of values
        :return: NumPy array
        """
        cdf = self.make_cdf()
        return cdf.sample(*args, **kwargs)

    def add_dist(self, x):
        """Distribution of the sum of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.add_dist(x)
        return self.make_same(res)

    def sub_dist(self, x):
        """Distribution of the diff of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.sub_dist(x)
        return self.make_same(res)

    def mul_dist(self, x):
        """Distribution of the product of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.mul_dist(x)
        return self.make_same(res)

    def div_dist(self, x):
        """Distribution of the ratio of values drawn from self and x.
        x: Distribution, scalar, or sequence
        :return: new Distribution, same subtype as self
        """
        pmf = self.make_pmf()
        res = pmf.div_dist(x)
        return self.make_same(res)

    def pmf_outer(dist1, dist2, ufunc):
        """Computes the outer product of two PMFs.
        dist1: Distribution object
        dist2: Distribution object
        ufunc: function to apply to the qs
        :return: NumPy array
        """
        # TODO: convert other types to Pmf
        pmf1 = dist1
        pmf2 = dist2

        qs = ufunc.outer(pmf1.qs, pmf2.qs)
        ps = np.multiply.outer(pmf1.ps, pmf2.ps)
        return qs * ps

    def gt_dist(self, x):
        """Probability that a value from self is greater than a value from x.
        x: Distribution, scalar, or sequence
        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.gt_dist(x)

    def lt_dist(self, x):
        """Probability that a value from self is less than a value from x.
        x: Distribution, scalar, or sequence
        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.lt_dist(x)

    def ge_dist(self, x):
        """Probability that a value from self is >= than a value from x.
        x: Distribution, scalar, or sequence
        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.ge_dist(x)

    def le_dist(self, x):
        """Probability that a value from self is <= than a value from x.
        x: Distribution, scalar, or sequence
        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.le_dist(x)

    def eq_dist(self, x):
        """Probability that a value from self equals a value from x.
        x: Distribution, scalar, or sequence
        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.eq_dist(x)

    def ne_dist(self, x):
        """Probability that a value from self is <= than a value from x.
        x: Distribution, scalar, or sequence
        :return: float probability
        """
        pmf = self.make_pmf()
        return pmf.ne_dist(x)

    def max_dist(self, n):
        """Distribution of the maximum of `n` values from this distribution.
        n: integer
        :return: Distribution, same type as self
        """
        cdf = self.make_cdf().max_dist(n)
        return self.make_same(cdf)

    def min_dist(self, n):
        """Distribution of the minimum of `n` values from this distribution.
        n: integer
        :return: Distribution, same type as self
        """
        cdf = self.make_cdf().min_dist(n)
        return self.make_same(cdf)

    prob_gt = gt_dist
    prob_lt = lt_dist
    prob_ge = ge_dist
    prob_le = le_dist
    prob_eq = eq_dist
    prob_ne = ne_dist
