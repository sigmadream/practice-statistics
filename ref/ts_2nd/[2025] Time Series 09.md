
---

<a role="toc_link" id="appA"></a>
235\. 

# AProbability and Statistics Primer

We assume the reader has had a course in calculus based probability and statistics at the level of [Freund and Walpole (1986)](#bibref1#refbib_18). The topics treated here are provided as a reference guide and a quick review of the material used in the text.

## A.1 Distributions and Densities

We work primarily with (absolutely) continuous random variables. If a random variable (rv) _X_ is continuous, its cumulative distribution function (CDF) can be written as

F(x)\=Pr(X≤x)\=∫−∞xf(u) du,x∈R,

where its density function f(x) satisfies

1. f(x)≥0 for all x∈R.
2. ∫−∞∞f(x) dx\=1.

Probabilities can be obtained by integration of the density:

Pr(a≤X≤b)\=F(b)−F(a)\=∫abf(x) dx.

For us, the normal distribution is important. The rv _X_ is said to be normal with mean _μ_ and variance _σ_2, denoted as X∼ N(μ,σ2), if its density function is

f(x)\=1σ2πexp{−12σ2(x−μ)2}forx∈R.

Given a random sample X1,…,Xn, the CDF can be estimated by the _empirical distribution function_ (EDF) given by the proportion of the sample less than or equal to a value _x_,

Fn(x)\=n−1∑i\=1n1{Xi≤x},

where 1{Xi≤x}\=1 if the statement in the brackets is true, and 0 if it is false. The following is an example using a t-distribution with 5 degrees of freedom. [Figure A.1](#appA#figA_1) shows the result comparing the true CDF with the EDF from n\=30 observations.


`set.seed(123)`
`X = rt(30, 5)   _# sample of size 30 from t-dist with 5 df_`
`plot(ecdf(X), col=4, pch=NA, lwd=2, verticals=TRUE)    _# EDF_`
`curve(pt(x,5), -5,5, col=6, lwd=3, add=TRUE)           _# CDF_`

![The empirical distribution function  of a random sample of size 30 from a t population with 5 degrees of freedom compared to its cumulative distribution function](./images/fig9_1.jpg)

Figure A.1: The EDF of a random sample of size 30 from a t population with 5 degrees of freedom. The CDF is superimposed on the EDF. [Return to text.⏎](appA)

236\. 

### Normal QQ Plot

The quantiles _q_ of a standard normal distribution, Z∼N(0,1), are defined by

Pr(Z≤q)\=p,

where _p_ is a probability. For example, the p\=.95 quantile in the standard normal case is the well-known value q\=1.645 because Pr(Z≤1.645)\=.95 (after rounding). Other familiar quantiles of the standard normal are:


`p = c(.025, .05, .10, .25, .5, .75, .90, .95, .975)`
`round(qnorm(p), 3)`
` [1] -1.960 -1.645 -1.282 -0.674 0.000 0.674 1.282    1.645   1.960`

The quantiles associated with a sample x1,…,xn are the quantiles of the EDF. For example, the. 95 quantile of a sample is the smallest data value for which at least 95% of the sample are less or equal to that value. If we arrange the data in ascending order, x(1)≤x(2)≤⋯≤x(n), then x(j) is the _j_\-th quantile of the EDF because j/n is the proportion of the data less than or equal to that value. Typically, a continuity correction is used so that x(j) is considered the (j−12)/n quantile.

If we let q(j) be the (j−12)/n quantile of the standard normal distribution, then a QQ plot is simply a plot of the pairs (q(j),x(j)) for j\=1,…,n. The idea is if the data are a sample from a normal population, the pairs will be approximately linearly related because μx+σxq(j) is an estimate of the expected sample quantile.

For a quick example, [Figure A.2](#appA#figA_2) displays normal QQ plots for two samples. The left side of the graphic is a sample of size 50 from a standard normal 237\. population, and the right side is a sample of size 50 from a t-distribution with four degrees of freedom. The confidence intervals are 95% pointwise intervals.


`set.seed(123)`
`x = rnorm(50); y = rt(50, 4)`
`par(mfrow=1:2)`
`**QQnorm**(x, main=bquote(N(0,1)), ci=95, pch=20, **gg**=TRUE)`
`**QQnorm**(y, main=bquote(t[~4]), ci=95, pch=20, **gg**=TRUE)`

![Normal QQ plots for two samples of size 50, one from a normal and one from a t](./images/fig9_2.jpg)

Figure A.2: Normal QQ plots for two samples of size 50, the left side from a standard normal population and the right side from a t4 population. [Return to text.⏎](appA)

## A.2 Expectation

For a continuous rv _X_ having density function f(x), the expected value of _X_ is defined as

μx\=E(X)\=∫−∞∞x f(x) dx

provided that the integral exists. The expected value of _X_ is typically called the mean of _X_ and is denoted by _μx_ or simply _μ_ when the particular random variable is understood. The mean, or expectation, of X gives a single value that acts as a representative or average of the values of X, and for this reason it is often called a measure of central tendency.

Some properties of expectation are:

1. For any constants _a_ and _b_ we have E(a+bX)\=a+bE(X)\=a+bμx.
2. For two rvs _X_ and _Y_, E(X+Y)\=E(X)+E(Y)\=μx+μy.
3. For two independent rvs _X_ and _Y_, E(XY)\=E(X)E(Y)\=μxμy.
4. E\[g(X)\]\=∫g(x)f(x) dx.

238\. Variance is the average squared deviation from the mean. Assuming it exists, define

σx2\=var(X)\=E(X−μ)2\=∫−∞∞(x−μ)2f(x) dx.

Again, we'll drop the subscript when the particular random variable is understood. The positive square root of the variance, _σ_, is called the standard deviation. Some properties of variance are:

1. For any constants _a_ and _b_, we have var(a+bX)\=b2var(X)\=b2σ2.
2. var(X)\=EX2−μ2.
3. For two independent rvs _X_ and _Y_, var(X+Y)\=var(X)+var(Y).
4. If _X_ has mean _μ_ and variance _σ_2, then the rv  
Z\=X−μσ  
has mean 0 and variance 1\. This transformation is called _standardization_.

The normal distribution is completely specified by its mean and variance, and hence the notation X∼ N(μ,σ2). In addition, the properties above show that if X∼ N(μ,σ2), then Z∼ N(0,1), the _standard normal distribution_,

f(z)\=12πexp{−z22{forz∈R.

Finally, we define the _r_th (central) moment of an rv as

E(X−μ)rr\=1,2,…,

when it exists. If not centered by the mean, the moment E(Xr) is called the raw moment. Also, we may define standardized moments as

κr\=E(X−μσ)r.

Important values are _κ_3, which measures _skewness_, and _κ_4, which measures _kurtosis_.

## A.3 Covariance and Correlation

For two rvs _X_ and _Y_ each with finite variance, the _covariance_ is defined as the expected product,

σxy\=cov(X,Y)\=E\[(X−μx)(Y−μy)\].

239\. Some properties of covariance are:

1. σxy\=cov(X,Y)\=cov(Y,X)\=σyx.
2. |σxy|≤σxσy; see (A.3).
3. var(X)\=cov(X,X).
4. var(X±Y)\=cov(X±Y,X±Y)\=var(X)+var(Y)±2 cov(X,Y).
5. For two independent rvs _X_ and _Y_, cov(X,Y)\=0. However, the other direction is not true; i.e., cov(X,Y)\=0 does not imply _X_ and _Y_ are independent.

_Correlation_ is defined as scaled covariance:

ρ\=corr(X,Y)\=σxyσx σy.

Some properties of correlation are:

1. −1≤ρ≤1.
2. If ρ\=0, we say that _X_ and _Y_ are uncorrelated. This means that _X_ and _Y_ are not _linearly_ related. They may, however, be dependent rvs. For example, if _X_ is symmetric, then _X_ and _X_2 are uncorrelated but highly dependent.
3. If ρ\=±1, then X\=a±bY (almost everywhere), for some numbers _a_ and b\>0.

## A.4 Distributions Related to the Normal

There are three distributions that are essential to inference and we mention these and some of their properties.

### Chi-squared Distribution

This distribution is denoted χν2 where _ν_ is the degrees of freedom. The mean and variance of the distribution are _ν_ and 2ν, respectively. If Z∼N(0,1), then _Z_2 has a χ12 distribution. If _Z_1 and _Z_2 are independent standard normals, then Z12+Z22 has a χ22 distribution, and so on.

The distribution is useful because the sample variance from a normal population is chi-squared. That is, if S2\=1n−1∑i\=1n(Xi− X―)2 is the sample variance from a random sample, X1,…,Xn, of size _n_ from a normal population with variance _σ_2, then

(n−1)S2σ2∼χn−12.

240\. 

### t-Distribution

The t-distribution with _ν_ degrees of freedom arises as follows. Let Z∼N(0,1) and V∼χν2 where _Z_ and _V_ are independent rvs. Then

T\=ZV/ν,

has a t-distribution with _ν_ degrees of freedom. The mean of _T_ is 0 provided ν\>1 and the variance of _T_ is ν/ν−2 provided ν\>2.

The distribution is useful because many estimators can be scaled to this form. For example, in linear regression, tests of hypotheses about regression coefficients can be written in terms of a t-test (see [Section 3.1](#chapter3#sec3_1)). Recall that if X1,…,Xn is a random sample of size _n_ from a normal population with mean _μ_ and variance _σ_2, then

T\= X―−μS/n,

where _S_2 is the sample variance, has a t-distribution with n−1 degrees of freedom.

### F-Distribution

The F-distribution arises from comparing variance components as in (3.5). If _U_ and _V_ are independent chi-squared random variables, U∼χν12 and V∼χν22, then

F\=U/ν1V/ν2

has an F-distribution with (ν1,ν2) degrees of freedom. The mean of the distribution is ν2/(ν2−2) for ν2\>2, and the variance is 2ν22(ν1+ν2−2)/ν1(ν2−2)2(ν2−4) for ν2\>4.

The distribution arose from comparing variances of two normal populations. If we have independent random samples of sizes _n_1 and _n_2 from two normal populations, and S12 and S22 are the respective sample variances, then

F\=S12/σ12S22/σ22

has an F-distribution with n1−1 and n2−1 degrees of freedom. Under the null hypothesis that σ1\=σ2, the test statistic is simply the ratio of the sample variances.

## A.5 241\. Conditional Expectation

Because we deal with dependence, a key tool is conditional expectation, which is typically written as E(X∣Y) where _X_ and _Y_ are rvs of interest. This animal is itself a random variable that takes values E(X∣Y\=y) according to the distribution f(y).

Recall that if the joint density of _X_ and _Y_ is f(x,y), then the conditional density of _X_ given Y\=y is

f(x∣y)\=f(x,y)f(y),

provided f(y)\>0. The conditional expectation of a function g(X) given Y\=y is then

E\[g(X)∣Y\=y\]\=∫g(x) f(x∣y) dx.

This result leads to the law of iterated expectation.

Property A.1 (Law of Iterated Expectation). [Return to text.⏎](#chapter8#b8propA_1) _Assuming all expectations exist,_

E(X)\=E\[E(X∣Y)\].

_Proof._ For the continuous case,

E\[E(X∣Y)\]\=∫yE(X∣Y\=y)f(y)dy\=∫y∫xxf(x∣y)dxf(y)dy\=∫xx\[∫yf(x,y)dy\]dx\=∫xxf(x)dx\=E(X),

where we used the fact that f(x,y)\=f(y) f(x∣y). □

Example A.2 Poison Mixture [Return to text.⏎](#appA#bexam9_2)

Suppose _X_ is the number of accidents during the morning rush hour, and _Y_ is an indicator of whether or not there is precipitation. Let Y\=1 if it is dry and Y\=2 if there is precipitation such that Pr(Y\=1)\=p and Pr(Y\=2)\=q where p+q\=1. Then, suppose X∣Y\=y is Poisson(λy),

Pr(X\=x∣Y\=y)\=λyx e−λy/x!x\=0,1,…; y\=1,2,

where λy\>0. Thus, if it is dry, the number of accidents has a rate of _λ_1 and if it is wet, the rate is λ2\>λ1.

Note that

E(X∣Y\=1)\=∑x\=0∞xλ1xe−λ1x!\=λ1∑x\=1∞λ1(x−1)e−λ1(x−1)!\=λ1

242\. and similarly, E(X∣Y\=2)\=λ2.

Now, E(X∣Y) is a random variable that takes values _λ_1 or _λ_2 with probability _p_ or _q_. For ease, let's write Z\=E(X∣Y) so that

E(X∣Y)\=Z\={λ1 wp p(dry: Y\=1)λ2wp q(wet: Y\=2).

Thus E(Z)\=pλ1+qλ2\=EE(X∣Y) and finally

E(X)\=EE(X∣Y)\=pλ1+qλ2,

as we should have expected.

There is a similar result for the variance.

Property A.3 (Law of Total Variance). [Return to text.⏎](#appA#bpropA_3) _Assuming all expectations exist,_

var(X)\=E\[var(X∣Y)\]+var\[E(X∣Y)\].

Example A.4 Poison Mixture (cont.)

We now use [Property A.3](#appA#propA_3) to calculate the variance of _X_, the number of rush hour accidents from [Example A.2](#appA#exam9_2). We'll do the calculation in two steps.

First, let Z\=var(X∣Y), and because the mean and variance of a Poisson are the same, _Z_ takes values _λ_1 with probability _p_ and _λ_2 with probability _q_. Consequently,

E\[var(X∣Y)\]\=E\[Z\]\=pλ1+qλ2\=E(X).

Now let Z\=E(X∣Y), then

var\[E(X∣Y)\]\=var\[Z\]\=EZ2−E2\[Z\]\=pλ12+qλ22−(pλ1+qλ2)2\=pqλ12+pqλ22−2λ1λ2pq\=pq(λ1−λ2)2.

Finally,

var(X)\=E(X)+pq(λ2−λ1)2\>E(X).

Note that var(X)\>E(X) as opposed to a Poisson where the values are the same. This “overdispersion” is often seen in count data such as the annual counts of major earthquakes listed in **EQcount**:


`c( mean(**EQcount**), var(**EQcount**) )`
` [1]    19.36        51.57`

243\. Some additional facts that are useful are as follows. Let _X_ and _Y_ be random variables, _g_ be a real-valued function, and _a,b_ be constants. Assuming all expectations exist,

1. E\[a∣Y\]\=a
2. E\[aX+bZ∣Y\]\=aE\[X∣Y\]+bE\[Z∣Y\]
3. If _X_ and _Y_ are independent, then E\[X∣Y\]\=E\[X\]
4. E\[Xg(Y)∣Y\]\=g(Y)E\[X∣Y\]
5. Putting X≡1 in item (iv) yields E\[g(Y)∣Y\]\=g(Y)

## A.6 Bivariate Normal Distribution

The bivariate normal distribution is denoted by

(XY)∼N2\[(μxμy), (σx2 ρσxσyρσxσy σy2)\],

where |ρ|<1 is the correlation between _X_ and _Y_. The bivariate normal density is

f(x,y)\=exp{−12(1−ρ2)\[(x−μxσx)2−2ρ(x−μxσx)(y−μyσy)+(y−μyσy)2\]}2πσxσy1−ρ2,

for −∞<x,y<∞.

We note the following:

1. The only case where ρ\=corr(X,Y)\=0 implies _X_ and _Y_ are independent is the case where they are bivariate normal.
2. It is not enough for the marginal distributions to be normal for the joint distribution to be normal. It is easy to construct a situation where _X_ and _Y_ are normal, but (X,Y) is not bivariate normal; e.g., let _X_ and _Z_ be independent normals and let Y\=Z if XZ\>0 and Y\=−Z if XZ≤0. The following code may help in visualizing the result (note that _X_ and _Y_ always have the same sign):  
`x = rnorm(1000); z = rnorm(1000)`  
`y = ifelse(x*z > 0, z, -z)`  
`**scatter.hist**(x, y, hist.col=5, pt.col=6)`
3. If (X,Y) is bivariate normal, then the conditional distribution of _Y_ given X\=x is normal:  
Y∣X\=x∼N(μy+ρσyσx(x−μx), (1−ρ2) σy2).

244\. From the last property,

E(Y∣X\=x)\=μy+ρσyσx(x−μx)var(Y∣X\=x)\=(1−ρ2) σy2,

which is a justification for simple linear regression. If we define β\=ρσyσx and α\=μy+βμx, and have a random sample of _n_ pairs, (xi,Yi) for i\=1,…,n, we fit the regression model

Yi\=α+βxi+ϵi

to the data, where it is assumed that the _ϵi_ are independent normal rvs with mean zero and constant variance σϵ2.

## A.7 Maximum Likelihood Estimation

A common method of parameter estimation is maximum likelihood estimation (MLE). The basic idea is simple and is best explained with examples.

Example A.5 The Earth is Flat

Suppose you randomly ask 10 people if the earth is flat, and 2 say yes. Given that data, what is the most likely proportion of people who think the earth is flat? The obvious answer is 20% because that is the most likely value of the true proportion that results in your data. You would not, for example, say 18% or 30% are more likely based on your data.

If we let _X_ be the number of people who say yes to the flat earth question, then in _n_ repeated trials, the probability distribution of _X_ is Binomial(n,θ),

Pr(X\=x)\=(nx)θx(1−θ)n−x,for x\=0,1,…,n,0<θ<1,

where _θ_ is the true proportion of flat earthers that we wish to estimate. In our example, x\=2 and n\=10. If we now consider _θ_ to be a variable, we can examine the probability in terms of what is called the _likelihood_,

L(θ)\=(102)θ2(1−θ)8,

which is the observed binomial probability for various values of _θ_.

Suppose we now find the value of _θ_ that maximizes L(θ). In this case, we are asking which value of _θ_ most likely gave rise to the data. It is often easier to maximize the log-likelihood with respect to _θ_,

logL(θ)\=2logθ+8log(1−θ),

245\. where we have ignored the term that does not involve _θ_. Taking derivatives and setting the result equal to zero, we have

∂L(θ)∂θ\=2θ−81−θ\=0.

Now multiply through by θ(1−θ) to get

2(1−θ)−8θ\=0,

so that the solution, which is called the MLE of _θ_ is

θ^\=210,

and it agrees with our intuition. The likelihood L(θ) is plotted in [Figure A.3](#appA#figA_3) and shows θ^\=.2 as the most likely value of _θ_.


`th   = 0:100/100`
`like = dbinom(2, 10, th)`
`**tsplot**(th, like, col=4, ylab=bquote(L(theta)), xlab=bquote(theta), **gg**=TRUE)`
`abline(v=.2, col=6, lty=5)`

![Likelihood for the flat earth example binomial distribution example showing the MLE](./images/fig9_3.jpg)

Figure A.3: Likelihood for the flat earth example showingθ^\=.2 as the most likely value of _θ_. [Return to text.⏎](appA)

Example A.6 MLE for a Poisson Rate

Recall that the Poisson is a distribution of counts. For example, _X_ could be the number of accidents at a particular intersection during the morning rush hour that occur at a rate of λ\>0. In this case,

Pr(X\=x)\=λxe−λx!,for x\=0,1,…,

with

E(X)\=λandvar(X)\=λ.

246\. Suppose X1,…,Xn is a random sample of _n_ observations from this distribution. Then the associated joint probability is

Pr(X1\=x1,…,Xn\=xn)\=Pr(X1\=x1)⋯Pr(Xn\=xn)\=λx1e−λx1!…λxne−λxn!\=λ∑i\=1nxie−nλ∏i\=1nxi!.

Now, with the data fixed, we consider _λ_ as the variable and write the likelihood as

L(λ)\=λ∑i\=1nxie−nλ,

where we have dropped the term that does not involve _λ_. And as before, it easier to deal with the log-likelihood,

logL(λ)\=∑i\=1nxilogλ−nλ.

Taking derivatives and setting the result equal to zero, we have

∂L(λ)∂λ\=∑i\=1nxiλ−n\=0,

and solving, the MLE of _λ_ is

λ^\=1n∑i\=1nxi\=x¯,

the sample average. This result makes sense because the rate _λ_ is the mean of the distribution.

For a quick example, we simulated n\=10 observations from a Poisson distribution with rate λ\=2. The log-likelihood is shown in [Figure A.4](#appA#figA_4) along with the MLE x¯\=2.1.


`set.seed(1)`
`logL = function(lam, x) { log(lam)*sum(x) - length(x)*lam }`
`x = rpois(n=10, lambda=2)`
`lam = seq(1, 4, by=.1)`
`**tsplot**(lam, logL(lam, x), col=4, **gg**=TRUE, xlab=bquote(lambda),`
`   ylab=bquote(log~L(lambda)))`
`abline(v=mean(x), col=2, lty=5)`
`c(mean(x), var(x))`
`[1] 2.1      2.1`

![Log likelihood for the Poisson distribution example showing the MLE](./images/fig9_4.jpg)

Figure A.4: Log likelihood for the Poisson example showingλ^\=x¯\=2.1 as the most likely value of _λ_. [Return to text.⏎](appA)

We close this section with a more complicated example.

Example A.7 247\. MLE for a Normal Mean and Variance

Suppose we have a random sample, X1,…,Xn, of size _n_ from a normal population with mean _μ_ and variance _σ_2. Recall that the normal density is given by

f(x)\=1σ2πexp{−12σ2(x−μ)2},−∞<x<∞,

where μ∈R and σ2\>0. Consequently, the joint density of the sample is

f(x1,…,xn)\=f(x1)⋯f(xn)\=∏i\=1n1σ2πexp{−12σ2(xi−μ)2}\=(2π σ2)−n2exp{−12σ2∑i\=1n(xi−μ)2}.

Hence, the log-likelihood in this case, ignoring the constant involving 2π, is

logL(μ,σ2)\=−n2log(σ2)−12σ2∑i\=1n(xi−μ)2.

Taking the partials with respect to _μ_ and _σ_2 and setting the results equal to zero we get,

∂logL(μ,σ2)∂μ\=1σ2∑i\=1n(xi−μ)\=0,(A.1)

∂logL(μ,σ2)∂σ2\=−n2σ2+12σ4∑i\=1n(xi−μ)2\=0.(A.2)

We see from (9.1) that μ^, the MLE of _μ_, must satisfy

0\=∑i\=1n(xi−μ^)\=∑i\=1nxi−nμ^,

248\. or

μ^\=∑i\=1nxin\=x¯,

the sample mean. Multiplying through by 2σ4 in (9.2) at the MLE of _μ_ yields

0\=−nσ2+∑i\=1n(xi−μ^)2,

so that

σ^2\=1n∑i\=1n(xi−x¯)2,

is the MLE of _σ_2.

For a numerical example, we simulated 200 observations from a normal distribution with μ\=100 and σ2\=152 and then found the MLEs. The resulting likelihood in the form of −logL(μ,σ2) is shown in [Figure A.5](#appA#figA_5). For this particular example, μ^\=99.24 and σ^2\=16.072.


`set.seed(90210)`
`N     = 200`
`xdata = rnorm(N, mean=100, sd=15)`
`mean(xdata)   _# µ̂_`
`  [1] 99.24213`
`sd(xdata)*sqrt(1-1/N)   _# σ̂_`
`  [1] 16.06858`

![Likelihood for a normal example showing the MLEs of the mean and of the variance](./images/fig9_5.jpg)

Figure A.5: MLE for normal example. Displayed is the surface of −logL(μ,σ2) for various values of _μ_ and _σ_, and it shows the locations of the MLEs (the minimizers in this case), (μ^,σ^)≈(99,16). [Return to text.⏎](appA)

249\. A contour plot of −logL(μ,σ2) can be obtained as follows ([Figure A.5](#appA#figA_5) is a perspective plot, but the code is a bit too long and convoluted to display here).


`normL = function(x, mu, sigma) {`
`   -sum(dnorm(x, mu, sigma, log=TRUE)) }`
`_# grid of parameter values_`
`mu         = seq(80, 120, length.out=N)`
`sigma      = seq(10, 20, length.out=N)`
`parm.grid = expand.grid(mu=mu, sigma=sigma)`
`_# evaluate -log L over the grid_`
`like       = c()`
`for (i in 1:N^2) {`
`   like[i] = normL(xdata, parm.grid[i,"mu"], parm.grid[i,"sigma"]) }`
`like = matrix(like, nrow=N, ncol=N)`
`contour(mu, sigma, like, xlab="\u03BC", ylab="\u03C3", nlevels=250,`
`    drawlabels=FALSE, col=rainbow(275), lwd=3, main=bquote(-log~L(mu,sigma)))`
`abline(v=mean(xdata), h=sd(xdata)*sqrt(1-1/N), lty=5) _# locate MLEs_`

## A.8 Inequalities

We list some important inequalities. For each item, we assume all expectations exist.

* **Markov:** If _X_ is a non-negative random variable, then for ϵ\>0,  
Pr(X≥ϵ)≤E(X)ϵ.  
_Proof._ Assuming a finite mean, for the continuous case we have  
E(X)\=∫0∞xf(x)dx≥∫ϵ∞xf(x)dx≥ϵ∫ϵ∞f(x)dx\=ϵPr(X≥ϵ),  
and the result follows. □
* **Chebyshev:** For ϵ\>0,  
Pr(|X−E(X)|≥ϵ)≤var(X)ϵ2.  
Chebyshev's inequality is a direct consequence of Markov's inequality by writing the inequality in terms of Y\=(X−EX)2 first. Also, note that if we 250\. write μ\=E(X) and σ2\=var(X), and let ϵ\=δσ for δ\>0, we can write the inequality as  
Pr(μ−δσ≤X≤μ+δσ)≥1−1δ2,  
which gives a lower bound on the probability a random variable is within _δ_ standard deviations of the mean (but it is only useful if δ\>1).
* **Cauchy-Schwarz:** For finite variance random variables _X_ and _Y_,  
|cov(X,Y)|2≤var(X) var(Y).(A.3)  
Since the correlation between _X_ and _Y_ is  
corr(X,Y)\=cov(X,Y)var(X)var(Y),  
it follows that −1≤corr(X,Y)≤1.  
_Proof._ For ease, set EX\=EY\=0. Next, note  
0≤E(X−aY)2\=EX2−2aEXY+a2EY2,  
for any constant _a_. Now plug in a\=EXY/EY2 to get  
0≤EX2−E2XY/EY2.  
Now multiply through by EY2 and simplify,  
E2XY≤EX2 EY2.  
□

## A.9 Central Limit Theorem

A major part of the language of statistical inference includes the notion of large sample distributions of various estimators. Throughout this text, if _Sn_ is a generic statistic based on the data, X1,…,Xn alone, we write

Sn∼⋅N(μn,σn2),

to mean

limn→∞Pr(Sn−μnσn≤z)\=Pr(Z≤z),

where Z∼N(0,1), the standard normal distribution. In this case, we often describe this behavior by writing _Sn is approximately normal_, and interpret ∼⋅ as _is approximately distributed as_, for large sample sizes.

A general result that is useful is the following, which is a consequence of the Lindeberg–Feller Central Limit Theorem (CLT).251\. 

Theorem A.8 _Central Limit Theorem_ [Return to text.⏎](#appA#btheoA_8) _Let_ X1,…,Xn _be independent and identically distributed with mean μ and variance_ σ2. _Suppose_ {aj} _are constants for which_ ∑j\=1naj2/max1≤j≤naj2→∞ _as_ n→∞, _then_

∑j\=1najXj∼⋅N(μ∑j\=1naj, σ2∑j\=1naj2).(A.4)

Note that the classical CLT is [Theorem A.8](#appA#theoA_8) when aj\=1/n. In this case, the result is

X―n∼⋅N(μ,σ2/n),

where X―n\=1n∑j\=1nXj is the sample mean.

Of course, we rarely have independent data, but [Theorem A.8](#appA#theoA_8) can be generalized to stationary data under mild dependence properties. This type of consideration leads to large sample distributions of ARMA parameter estimators such as those given in [Example 4.32](#chapter4#exam4_32). In addition, a generalization of [Theorem A.8](#appA#theoA_8) is used to obtain the approximate distributions of the cosine and sine transforms as given in (7.5) and consequently, the large sample χν2 distribution of the smoothed spectral estimate given in (7.10).

Example A.9 252\. Daniell and the Central Limit Theorem [Return to text.⏎](#chapter7#b7examA_9)

The modified Daniell kernel that is described in [Section 7.2](#chapter7#sec7_2) is a moving average that uses simple averaging, except that the end weights are halved. As an example, let L\=2m+1 be the (odd) number of weights in the moving average and m\=1, then the weights are {ak}\={14,24,14}. If we apply the weights to a sequence of numbers {xt}, the result is

x^t\=14xt−1+12xt+14xt+1.

Applying the same kernel again to x^t yields

^^xt\=14x^t−1+12x^t+14x^t+1,

which simplifies to

^^xt\=116xt−2+416xt−1+616xt+416xt+1+116xt+2.

Note that these kernel weights form a probability distribution. If _X_1 and _X_2 are independent random variables on the integers {−1,0,1} with probabilities {14,12,14}, then the convolution X1+X2 is discrete on the integers {−2,−1,0,1,2} with corresponding probabilities {116,416,616,416,116}. Thus, by the central limit theorem, if we continue to apply the kernel, or equivalently, sum the independent random variables X1+X2+⋯+Xn, the weights (or probabilities) will form a normal distribution. [Figure 7.7](#chapter7#fig7_7) shows a small example, but we do a bigger example here; see [Figure A.6](#appA#figA_6).


`md = function(n){kernel("modified.daniell", m=rep(3,n))}`
`par(mfrow=c(2,3), cex=.8, oma=c(0,0,.5,0))`
`for (i in 1:6){`
` ytop = ifelse(i<4,.2,.12)`
` **tsplot**(md(i), ylab=NA, lwd=2, col=4, ylim=c(0,ytop), xlab=NA, type="h",`
`   **gg**=TRUE)`
`if (i==1) { mtext(bquote(X[1]), side=3, line=-2, adj=.95) } else {`
`   mtext(bquote(sum(X[j], j==1, .(i))), side=3, line=-3, adj=.9) }`
`}`
` title("The CLT in Action", outer=TRUE, adj=.52, line=-.9)`

![A demonstration of how the sum of iid unifroms approaches a normal distribution as the number of values in the sum increases, indicating the property of a central limit theorem](./images/fig9_6.jpg)

Figure A.6: The distribution of the sum (convolutions) of iid random variables that are discrete-valued on the integers −3 to 3 with probabilities based on the modified Daniell kernel (uniform with half-weights at the ends) as shown in the top left picture. [Return to text.⏎](appA)

## A.10 Taylor Expansion

Taylor's Theorem is important in probability and statistics, and it is an essential component of numerical optimization (e.g., in [Section 4.5](#chapter4#sec4_5)) among other things. The theorem is as follows.

Theorem A.10 253\. (Taylor's Theorem). _Let f be a real-valued function on an interval \[a,b\] and let n be a positive integer. If the_ (n−1) _st derivative of_ f(x), _say_ f(n−1)(x), _is continuous on \[a,b\] and the nth derivative_ f(n)(x) _exists on_ (a,b), _then for_ x∈\[a,b\],

f(x)\=f(a)+(x−a)f(1)(a)+(x−a)22!f(2)(a)++⋯+ (x−a)n−1(n−1)! f(n−1)(a)+(x−a)nn!f(n)(ξ),

_for_ a<ξ<x.

The last term is called the remainder

Rn\=(x−a)nn!f(n)(ξ).

If f(x) has derivatives of all orders in a neighborhood of _a_ and Rn→0 as n→∞, then

f(x)\=f(a)+∑n\=1∞(x−a)nn!f(n)(a).

The special case of a\=0 is known as _Maclaurin series_. The following is a list of some series we use in the text.

1. 11−x\=∑n\=0∞xn, for |x|<1.
2. ex\=∑n\=0∞xnn!, for x∈R.
3. cos(x)\=∑n\=0∞(−1)nx2n(2n)!, for x∈R.
4. sin(x)\=∑n\=0∞(−1)nx2n+1(2n+1)!, for x∈R.
5. log(1+x)\=∑n\=1∞(−1)n+1xnn, for x∈(−1,1\].254 is blank.

---

<a role="toc_link" id="appB"></a>
255\. 

# BComplex Number Primer

In this appendix, we give a brief overview of complex numbers and establish some notation and basic operations.

## B.1 Complex Numbers

Most people first encounter complex numbers as solutions to the standard form of a quadratic equation,

ax2+bx+c\=0,

using the quadratic formula giving the two solutions as

x±\=−b±b2−4ac2a.

If b2−4ac≥0, this formula gives two real solutions. However, if b2−4ac<0, there are no real solutions.

For example, the equation x2+1\=0 has no real solutions because for any real number _x_, the square _x_2 is nonnegative. Nevertheless, it is very useful to assume that there is a number _i_ for which

i2\=−1,

so that the two solutions to x2\=−1 are ±i.

Any _complex number_ is an expression of the form z\=a+bi, where a\=ℜ(z) and b\=ℑ(z) are real numbers called the _real part_ of _z_, and the _imaginary part_ of _z_, respectively.

Since any complex number is specified by two real numbers, it can be visualized by plotting a point with coordinates (a,b) in the plane for a complex number z\=a+bi. The plane in which one plots these complex numbers is called the _complex plane_ shown in [Figure B.1](#appB#figB_1).

![The complex plane showing a complex number in terms of its real and imaginary parts, its modulus, and its argument](./images/fig10_1.jpg)

Figure B.1: A complex number z\=a+bi. [Return to text.⏎](appB)

To add (subtract) z\=a+bi and w\=c+di,

z+w\=(a+bi)+(c+di)\=(a+c)+(b+d)i,

z−w\=(a+bi)−(c+di)\=(a−c)+(b−d)i.

256\. To multiply _z_ and _w_,

zw\=(a+bi)(c+di)\=a(c+di)+bi(c+di)\=ac+adi+bci+bdi2\=(ac−bd)+(ad+bc)i

where we have used the defining property i2\=−1. To divide two complex numbers, we can do the following:

zw\=a+bic+di\=a+bic+di⋅c−dic−di\=(a+bi)(c−di)(c+di)(c−di)\=ac+bdc2+d2+bc−adc2+d2 i.

From this formula, it is easy to see that

1i\=−i,

because in the numerator a\=1, b\=0 while in the denominator c\=0, d\=1. The result also makes sense because 1/i should be the inverse of _i_, and indeed,

1i i\=−i⋅i\=−i2\=1.

For any complex number z\=a+bi, the number z¯\=a−bi is called its _complex conjugate._ A frequently used property of the complex conjugate is the following formula

|z|2\=zz¯\=(a+bi)(a−bi)\=a2−(bi)2\=a2+b2.

## B.2 257\. Modulus and Argument

For any given complex number z\=a+bi, the _absolute value_ or _modulus_ is

|z|\=a2+b2

i.e., |z| is the distance from the origin to the point _z_ in the complex plane as displayed in [Figure B.1](#appB#figB_1).

The angle _θ_ in [Figure B.1](#appB#figB_1) is called the _argument_ of the complex number _z_,

argz\=θ,

and it is made unique by defining it on (−π,π\].

From trigonometry, we see from [Figure B.1](#appB#figB_1) that for z\=a+bi,

cos(θ)\=a/|z| and sin(θ)\=b/|z|,

so that

tan(θ)\=sin(θ)cos(θ)\=ba,

and

θ\=arctanba.

For any _θ_, the number

z\=cos(θ)+isin(θ)

lies on the unit circle and consequently has length 1\. Its argument is argz\=θ. Conversely, any complex number on the unit circle is of the form cos(θ)+isin(θ), where _θ_ is its argument.

## B.3 Complex Exponential Function

For a complex number _z_, we now focus on the meaning of ez\=ea+ib. First consider the case a\=0,

Definition B.1. [Return to text.⏎](#appB#bdefiB_1) _For any real number b we set_

eib\=cos(b)+isin(b)

_see [Figure B.2](#appB#figB_2)._

![A unit length complex exponential in terms of the sine and the cosine of its argument](./images/fig10_2.jpg)

Figure B.2: Euler's definition of eib. [Return to text.⏎](appB)

258\. Using [Definition B.1](#appB#defiB_1), we come to the trig identities that we use often,

cos(b)\=eib+e−ib2 and sin(b)\=eib−e−ib2i(B.1)

Note that [Definition B.1](#appB#defiB_1) implies

eiπ\=cos(π)+isin(π)\=−1.

This leads to Euler's famous formula

eiπ+1\=0,

combining the five most basic quantities in mathematics: _e_, _π_, _i_, 1, and 0.

[Definition B.1](#appB#defiB_1) seems reasonable because, if we substitute _bi_ in the Taylor series for _e_ _x_, we get

ebi\=1+bi+(bi)22!+(bi)33!+(bi)44!+⋯\=1+bi−b22!−ib33!+b44!+ib55!−⋯\=1−b2/2!+b4/4!−⋯+i(b−b3/3!+b5/5!−⋯)\=cos(b)+isin(b),

assuming we can replace a real number _x_ by a complex number _ib_. In addition, the formula ex⋅ey\=ex+y still holds when x\=ib and y\=id are complex. That is,

eibeid\=\[cos(b)+isin(b)\]\[cos(d)+isin(d)\]\=cos(b+d)+isin(b+d)\=ei(b+d),

using the trig formulas cos(α±β)\=cos(α)cos(β)∓sin(α)sin(β) and sin(α±β)\=sin(α)cos(β)±cos(α)sin(β).

Requiring ex⋅ey\=ex+y to be true for all complex numbers leads to the definition of ea+bi for arbitrary complex numbers a+bi.

Definition B.2. 259\. _For any complex number_ a+bi _we set_

ea+bi\=ea⋅ebi\=ea\[cos(b)+isin(b)\].

## B.4 Other Useful Properties

### Powers

If we write a complex number in polar coordinates z\=reiθ, then for integer _n_,

zn\=rneinθ.

Putting r\=1 and noting (eiθ)n\=einθ yields de Moivre's formula

(cos(θ)+isin(θ))n\=cos(nθ)+isin(nθ)n\=0,±1,±2,….

### Integrals

Integration with complex exponentials is fairly simple. For example, suppose we must evaluate the complex integral

I\=∫e(3+2i)x dx.

The integral has meaning because e2ix\=cos2x+isin2x, so we may write

I\=∫e3x(cos2x+isin2x)dx\=∫e3xcos2xdx+i∫e3xsin2xdx.

Although breaking the integral down to its real and imaginary parts validates its meaning, it is not the easiest way to evaluate the integral. Rather, keeping the complex exponential intact, we have

I\=∫e(3+2i)xdx\=e(3+2i)x3+2i+C

where we have used that

∫eaxdx\=1aeax+C,

which holds even if _a_ is a complex number.

### Summations

The following result is used in various places throughout the text.260\. 

Property B.3. _For any positive integer n and integers_ j,k\=0,1,…,n−1:

1. _Except for_ j\=0 _or_ j\=n/2,  
∑t\=1ncos2(2πtj/n)\=∑t\=1nsin2(2πtj/n)\=n/2.
2. _When_ j\=0 _or_ j\=n/2,  
∑t\=1ncos2(2πtj/n)\=nbut∑t\=1nsin2(2πtj/n)\=0.
3. _For_ j≠k,  
∑t\=1ncos(2πtj/n)cos(2πtk/n)\=∑t\=1nsin(2πtj/n)sin(2πtk/n)\=0.
4. _Also, for any j and k,_  
∑t\=1ncos(2πtj/n)sin(2πtk/n)\=0.

_Proof._ Most of the results are proved the same way, so we only show the first part of (a). Using (B.1),

∑t\=1ncos2(2πt j/n)\=14∑t\=1n(e2πit j/n+e−2πit j/n)(e2πit j/n+e−2πit j/n)\=14∑t\=1n(e4πit j/n+1+1+e−4πit j/n)\=n2.

□

In the proof (and elsewhere), we used the following result for geometric sums. For any complex number z≠1,

∑t\=1nzt\=z 1−zn1−z.(B.2)

Instead of committing (B.2) to memory, it is much easier to remember how to establish it. Let Sn\=∑t\=1nzt. Then the trick is to write

Sn\=z+z2+⋯+zn,z Sn\=z+ z2+⋯+zn+zn+1.

261\. Now subtract,

(1−z)Sn\=z−zn+1,

which is (B.2). If z\=1, then the sum is of _n_ ones, so Sn\=n.

Consequently, for any frequency of the form ωj\=j/n for j\=0,1,…,n−1,

∑t\=1ne2πiωjt\={0if ωj≠0nif ωj\=0.

When ω\=0, the sum is of _n_ ones, whereas when ω≠0, the numerator of (B.2) is

1−e2πin(j/n)\=1−e2πij\=1−\[cos(2πj)+isin(2πj)\]\=0.

## B.5 Some Trigonometric Identities

We list some identities that are useful to us. These are easily proved using complex exponentials, and some follow directly from others.

(i)cos2(α)+sin2(α)\=1.(ii)sin(α±β)\=sin(α)cos(β)±cos(α)sin(β).(iii)cos(α±β)\=cos(α)cos(β)∓sin(α)sin(β).(iv) 2cos(α)cos(β)\=cos(α+β)+cos(α−β).(v)sin(2α)\=2sin(α)cos(α).(vi)cos(2α)\=cos2(α)−sin2(α)\=2cos2(α)−1.(B.3)262 is blank. 

<a role="toc_link" id="bibref1"></a>
271\. 

# References

* Akaike, H. (1974). A new look at the statistical model identification. IEEE Transactions on Automatic Control, 19(6):716–723.[Return to text.⏎](#chapter3#b3refbib_1)
* Blackman, R. and Tukey, J. (1959). The measurement of power spectra, from the point of view of communications engineering. Dover, pages 185–282.[Return to text.⏎](#chapter7#b7refbib_2)
* Bloomfield, P. (2004). Fourier Analysis of Time Series: An Introduction. John Wiley & Sons.[Return to text.⏎](#chapter7#b7refbib_3)
* Bogert, R., Healy, M., and Tukey, J. (1963). The Quefrency Alanysis of Time Series for Echoes: Cepstrum, Pseudo-Autocovariance, Cross-Cepstrum and Saphe Cracking. In _Proc. Symposium Time Series Analysis, 1963_, pages 209–243.[Return to text.⏎](#chapter7#b7refbib_4)
* Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. J. Econometrics, 31:307–327.[Return to text.⏎](#chapter8#b8refbib_5)
* Bollerslev, T., Engle, R. F., and Nelson, D. B. (1994). ARCH models. Handbook of Econometrics, 4:2959–3038.[Return to text.⏎](#chapter8#b8refbib_6)
* Box, G. and Jenkins, G. (1970). Time Series Analysis, Forecasting, and Control. Holden–Day.[Return to text.⏎](#chapter5#b5refbib_7)
* Brockwell, P. J. and Davis, R. A. (2013). Time Series: Theory and Methods. Springer Science & Business Media.[Return to text.⏎](#chapter7#b7refbib_8)
* CDC (2023). Flu Season. Centers for Disease Control and Prevention. <https://www.cdc.gov/flu/about/season/index.html>.[Return to text.⏎](#chapter8#b8refbib_9)
* Chan, N. H. (2002). Time Series Applications to Finance. John Wiley & Sons, Inc.[Return to text.⏎](#chapter8#b8refbib_10)
* Cleveland, W. S. (1979). Robust locally weighted regression and smoothing scatterplots. Journal of the American Statistical Association, 74(368):829–836.[Return to text.⏎](#chapter3#b3refbib_11)
* Cochrane, D. and Orcutt, G. H. (1949). Application of least squares regression to relationships containing auto-correlated error terms. Journal of the American Statistical Association, 44(245):32–61.272\. [Return to text.⏎](#chapter5#b5refbib_12)
* Cooley, J. W. and Tukey, J. W. (1965). An algorithm for the machine calculation of complex Fourier series. Mathematics of Computation, 19(90):297–301.[Return to text.⏎](#chapter7#b7refbib_13)
* Edelstein-Keshet, L. (2005). Mathematical Models in Biology. Society for Industrial and Applied Mathematics, Philadelphia.[Return to text.⏎](#chapter1#b1refbib_14)
* Efron, B. and Tibshirani, R. J. (1994). An Introduction to the Bootstrap. CRC Press.[Return to text.⏎](#chapter8#b8refbib_15)
* Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50:987–1007.[Return to text.⏎](#chapter8#b8refbib_16)
* Fabio Di Narzo, A., Aznarte, J. L., and Stigler, M. (2009). _tsDyn: Time series analysis based on dynamical systems theory_. <https://CRAN.R-project.org/package=tsDyn>.
* Freund, J. E. and Walpole, R. E. (1986). Mathematical Statistics. Prentice-Hall, 4th edition. <https://archive.org/details/mathematical%5Fstatistics>.[Return to text.⏎](#appA#brefbib_18)
* Gentle, J. E. (2003). Random Number Generation and Monte Carlo Methods. Springer.[Return to text.⏎](#chapter1#b1refbib_19)
* Granger, C. W. and Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. Journal of Time Series Analysis, 1(1):15–29.[Return to text.⏎](#chapter8#b8refbib_20)
* Grenander, U. and Rosenblatt, M. (2008). Statistical Analysis of Stationary Time Series. American Mathematical Soc.[Return to text.⏎](#chapter3#b3refbib_21)
* Hansen, J. and Lebedeff, S. (1987). Global trends of measured surface air temperature. Journal of Geophysical Research: Atmospheres, 92(D11):13345–13372.[Return to text.⏎](#chapter3#b3refbib_22)
* Hansen, J., Sato, M., Ruedy, R., Lo, K., Lea, D. W., and Medina-Elizade, M. (2006). Global temperature change. Proceedings of the National Academy of Sciences, 103(39):14288–14293.[Return to text.⏎](#chapter1#b1refbib_23)
* Hosking, J. R. (1981). Fractional differencing. Biometrika, 68(1):165–176.[Return to text.⏎](#chapter8#b8refbib_24)
* Hurst, H. E. (1951). Long-term storage capacity of reservoirs. Trans. Amer. Soc. Civil Eng., 116:770–799.[Return to text.⏎](#chapter8#b8refbib_25)
* Hurvich, C. M. and Tsai, C.-L. (1989). Regression and time series model selection in small samples. Biometrika, 76(2):297–307.[Return to text.⏎](#chapter3#b3refbib_26)
* Hyndman, R. J. and Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software, 27(3):1–22\. <https://CRAN.R-project.org/package=forecast>.273\.
* IMSL (2020). IMSL Numerical Libraries: Auto Arima. <https://www.imsl.com/blog/auto-arima>.[Return to text.⏎](#chapter4#b4refbib_28)
* Johnson, R. A. and Wichern, D. W. (2002). Applied Multivariate Statistical Analysis. Prentice Hall.[Return to text.⏎](#chapter3#b3refbib_29)
* Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. Journal of Basic Engineering, 82(1):35–45.[Return to text.⏎](#chapter8#b8refbib_30)
* Kalman, R. E. and Bucy, R. S. (1961). New results in linear filtering and prediction theory. Journal of Basic Engineering, 83(1):95–108.[Return to text.⏎](#chapter8#b8refbib_31)
* Kitchin, J. (1923). Cycles and trends in economic factors. The Review of Economic Statistics, pages 10–16.[Return to text.⏎](#chapter3#b3refbib_32)
* McLeod, A. I. and Hipel, K. W. (1978). Preservation of the rescaled adjusted range: 1\. A reassessment of the Hurst phenomenon. Water Resources Research, 14(3):491–508.[Return to text.⏎](#chapter8#b8refbib_33)
* McQuarrie, A. D. and Tsai, C.-L. (1998). Regression and Time Series Model Selection. World Scientific.[Return to text.⏎](#chapter3#b3refbib_34)
* Parzen, E. (1983). Autoregressive Spectral Estimation. Handbook of Statistics, 3:221–247.[Return to text.⏎](#chapter7#b7refbib_35)
* Pozzer, A., Anenberg, S., Dey, S., Haines, A., Lelieveld, J., and Chowdhury, S. (2023). Mortality attributable to ambient air pollution: A review of global estimates. GeoHealth, 7(1):e2022GH000711.[Return to text.⏎](#chapter3#b3refbib_36)
* Press, W. H., Teukolsky, S. A., Vetterling, W. T., and Flannery, B. P. (2007). Numerical Recipes: The Art of Scientific Computing. Cambridge University Press.[Return to text.⏎](#chapter1#b1refbib_37)
* R Core Team (2025). R: A Language and Environment for Statistical Computing. R Foundation for Statistical Computing, Vienna, Austria. <https://www.R-project.org/>.[Return to text.⏎](#preface1#brefbib_38)
* Ryan, J. A. and Ulrich, J. M. (2024). _xts: eXtensible Time Series_. <https://CRAN.R-project.org/package=xts>.
* Schwarz, G. (1978). Estimating the dimension of a model. The Annals of Statistics, 6(2):461–464.[Return to text.⏎](#chapter3#b3refbib_40)
* Shephard, N. (1996). Statistical aspects of arch and stochastic volatility. Monographs on Statistics and Applied Probability, 65:1–68.[Return to text.⏎](#chapter8#b8refbib_41)
* Shewhart, W. A. (1931). Economic Control of Quality of Manufactured Product. ASQ Quality Press.274\. [Return to text.⏎](#chapter5#b5refbib_42)
* Shumway, R., Azari, A., and Pawitan, Y. (1988). Modeling mortality fluctuations in Los Angeles as functions of pollution and weather effects. Environmental Research, 45(2):224–241.[Return to text.⏎](#chapter3#b3refbib_43)
* Shumway, R. and Stoffer, D. (2025). Time Series Analysis and Its Applications: With R Examples. Springer, New York, 5th edition.[Return to text.⏎](#chapter4#b4refbib_44)
* Shumway, R. H. and Verosub, K. L. (1992). State space modeling of paleoclimatic time series. In _Proc. 5th Int. Meeting Stat. Climatol_, pages 22–26.[Return to text.⏎](#chapter3#b3refbib_45)
* Stoffer, D. S. (2026). _astsa: Applied Statistical Time Series Analysis_. <https://CRAN.R-project.org/package=astsa>.
* Sugiura, N. (1978). Further analysts of the data by Akaike's information criterion and the finite corrections: Further analysts of the data by Akaike's. Communications in Statistics-Theory and Methods, 7(1):13–26.[Return to text.⏎](#chapter3#b3refbib_47)
* Tong, H. (1983). Threshold Models in Non-linear Time Series Analysis. Springer-Verlag, New York.[Return to text.⏎](#chapter8#b8refbib_48)
* Trapletti, A. and Hornik, K. (2024). _tseries: Time Series Analysis and Computational Finance_. <https://CRAN.R-project.org/package=tseries>.
* Tsay, R., Chen, R., and Liu, X. (2023). _NTS: Nonlinear Time Series Analysis_. <https://CRAN.R-project.org/package=NTS>.
* Tsay, R. S. (2005). Analysis of Financial Time Series, volume 543. John Wiley & Sons.[Return to text.⏎](#chapter8#b8refbib_51)
* Veenstra, J. Q. (2012). Persistence and Anti-persistence: Theory and Software. PhD thesis, Western University. <https://CRAN.R-project.org/package=arfima>.
* Winters, P. R. (1960). Forecasting sales by exponentially weighted moving averages. Management Science, 6(3):324–342.[Return to text.⏎](#chapter5#b5refbib_53)
* Wold, H. (1954). Causality and econometrics. Econometrica: Journal of the Econometric Society, pages 162–177.[Return to text.⏎](#chapter2#b2refbib_54)
* Wuertz, D., Chalabi, Y., Setz, T., Maechler, M., and Boshnakov, G. N. (2024). _fGarch: Rmetrics - Autoregressive Conditional Heteroskedastic Modelling_. <https://CRAN.R-project.org/package=fGarch>.
* Young, P. C. and Pedregal, D. J. (1999). Macro-economic relativity: government spending, private investment and unemployment in the usa 1948–1998. Structural Change and Economic Dynamics, 10(3-4):359–380.[Return to text.⏎](#chapter3#b3refbib_56)
