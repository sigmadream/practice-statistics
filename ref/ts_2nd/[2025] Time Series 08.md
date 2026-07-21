<a role="toc_link" id="chapter8"></a>
199\. 

# 8Additional Topics\*

In this chapter, we present various advanced topics. The sections may be read in any order. Each topic depends on a basic knowledge of ARMA models, forecasting and estimation, which is the material covered in [Chapter 4](#chapter4) and [Chapter 5](#chapter5).

## 8.1 GARCH Models

Various problems such as option pricing in finance have motivated the study of the _volatility_, or variability, of a time series. ARMA models were used to model the conditional mean (denoted _μt_ here) of a process when the conditional variance (denoted σt2 here) was constant. For example, in the AR(1) model xt\=ϕ0+ϕ1xt−1+wt we have

μt\=E(xt∣xt−1,xt−2,…)\=ϕ0+ϕ1xt−1σt2\=var(xt∣xt−1,xt−2,…)\=var(wt)\=σw2.

In many problems, however, the assumption of a constant conditional variance is violated. Models such as the _autoregressive conditionally heteroscedastic_ or ARCH model, first introduced by [Engle (1982)](#bibref1#refbib_16), were developed to model changes in volatility. These models were later extended to generalized ARCH (GARCH) models by [Bollerslev (1986)](#bibref1#refbib_5).

In these problems, we are concerned with modeling the return or growth rate of a series. Recall if _xt_ is the value of an asset at time _t_, then the return or relative gain, _rt_, of the asset at time _t_ is

rt\=xt−xt−1xt−1≈∇log(xt).

Either value, ∇log(xt) or (xt−xt−1)/xt−1, will be called the _return_ and will be denoted by _rt_. [1](#chapter8#fn8_1)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 1Although it is a misnomer, ∇logxt is often called the _log-return_; but the returns are not being logged. [Return to text.⏎](#chapter8#fn8_18b)

Typically, for financial series, the return _rt_ has a constant conditional mean (typically μt\=0 for assets), but does not have a constant conditional variance, 200\. and highly volatile periods tend to be clustered together. In addition, it is often the case that the autocorrelation structure of _rt_ is that of white noise, while the returns are dependent. This can often be seen by looking at the sample ACF of the squared-returns (or some power transformation of the returns). For example, [Figure 8.1](#chapter8#fig8_1) shows the daily returns of the Dow Jones Industrial Average (DJIA) that we saw in [Chapter 1](#chapter1). In this case, as is typical, the return _rt_ is fairly constant (with μt\=0) and nearly white noise, but there are short-term bursts of high volatility and the squared returns are autocorrelated.


`library(xts)`
`djiar = diff(log(**djia$Close**))`
`layout(matrix(c(1,2,1,3), 2), heights=2:1)`
`plot(djiar, col=4)`
`**acf1**(djiar, ylim=c(-.1,.6))`
`**acf1**(djiar^2, ylim=c(-.1,.6))`

![A graph showing the daily closing returns of the Dow Jones Industrial Average from Aprill 2006 to April 2016,  and the corresponding sample ACF of the returns and of the squared returns, indicating that the returns are white noise, but their squares are not](./images/fig8_1.jpg)

Figure 8.1: DJIA daily closing returns and the sample ACF of the returns and of the squared returns. [Return to text.⏎](chapter8)

The simplest ARCH model, the ARCH(1), models the returns as

rt\=σtεt(8.1)

σt2\=α0+α1rt−12,(8.2)

where _εt_ is standard Gaussian white noise, εt∼ iid N(0,1). The normal assumption may be relaxed; we will discuss this later. As with ARMA models, we must 201\. impose some constraints on the model parameters to obtain desirable properties. An obvious constraint is that α0,α1≥0 because σt2 is a variance.

Notice that the conditional distribution of _rt_ given rt−1 is Gaussian,[2](#chapter8#fn8_2)

rt∣rt−1∼N(0,α0+α1rt−12).(8.3)

which is another way to write (8.1)–(8.2). Also, because E(rt∣rt−1)\=0, we can show that _rt_ is white noise (see [Example 8.4](#chapter8#exam8_4) for details).

It is possible to write the ARCH(1) model as a non-Gaussian AR(1) model in the square of the returns rt2. First, rewrite (8.1)–(8.2) as

rt2\=σt2εt2α0+α1rt−12\=σt2,

by squaring (8.1). Now subtract the two equations to obtain

rt2−(α0+α1rt−12)\=σt2εt2−σt2,

and rearrange it as

rt2\=α0+α1rt−12+vt,(8.4)

where vt\=σt2(εt2−1). Because εt2 is the square of a N(0,1) random variable, εt2−1 is a shifted (to have mean-zero), χ12 random variable. In this case, _vt_ is non-normal white noise (see [Example 8.4](#chapter8#exam8_4) for details).

For the model in (8.4) to be causal, it is clear that 0≤α1<1. However, for rt2 to have a finite variance, we must have α12<1/3 (details are in [Example 8.4](#chapter8#exam8_4)). If this is the case, then the squared process, rt2, follows a causal AR(1) model with ACF given by ρr2(h)\=α1h≥0, for all h\>0. Hence, the model characterizes what we see in [Figure 8.1](#chapter8#fig8_1):

* The returns are white noise.
* The squared returns are autocorrelated.
* The conditional variance of a return depends on previous returns (volatility clustering).

Estimation of the parameters _α_0 and _α_1 of the ARCH(1) model is typically accomplished by conditional MLE based on the normal density specified in (8.3). This leads to finding the values of _α_0 and _α_1 that minimize

l(α0,α1)\=12∑t\=2nln(α0+α1rt−12)+12∑t\=2n(rt2α0+α1rt−12),(8.5)

using numerical methods as described in [Section 4.5](#chapter4#sec4_5).

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 2With rt−1 fixed, σt2\=α0+α1rt−12 is a constant and rt∣rt−1\=σt×εt∼N(0,σt2).[Return to text.⏎](#chapter8#fn8_12b)

202\. The ARCH(1) model can be extended to the general ARCH(_p_) model in an obvious way. That is, (8.1) is retained but (8.2) is extended as follows:

rt\=σtεt,σt2\=α0+α1rt−12+⋯+αprt−p2.

Estimation for ARCH(_p_) also follows in an obvious way from the discussion of estimation for ARCH(1) models.

It is also possible to combine a regression or an ARMA model for the conditional mean,

rt\=μt+σtεt,

where, for example, a simple AR-ARCH model would have

μt\=ϕ0+ϕ1rt−1.

Of course the model can be generalized to have various types of behavior for _μt_.

To fit ARMA-ARCH models, follow these two steps:

1. First, look at the P/ACF of the _returns_, _rt_, and identify an ARMA structure, if any. There is typically either no autocorrelation or very small autocorrelation and often a low order AR or MA will suffice if needed. Estimate _μt_ so that the returns may be centered if necessary.
2. Look at the P/ACF of the _centered squared returns_, (rt−μ^t)2, and decide on an ARCH model. If the P/ACF indicate an AR structure (i.e., ACF tails off, PACF cuts off), then fit an ARCH. If the P/ACF indicate an ARMA structure (i.e., both tail off), use the approach discussed after the next example.

Example 8.1 Analysis of U.S. GNP [Return to text.⏎](chapter8)

In [Example 5.6](#chapter5#exam5_6), we fit an AR(1) model to the U.S. GNP series and we concluded that the residuals appeared to behave like a white noise process. Hence, we would propose that μt\=ϕ0+ϕ1rt−1 where _rt_ is the quarterly growth rate of the GNP.

It has been suggested that the GNP series has ARCH errors, and in this example, we will investigate this claim. If the GNP noise term is ARCH, the squares of the residuals from the fit should behave like a non-Gaussian AR(1) process, as pointed out in (8.4). [Figure 8.2](#chapter8#fig8_2) shows the ACF and PACF of the squared residuals and it appears that there may be some dependence, albeit small, left in the residuals. The figure was generated as follows.


`res = resid( **sarima**(diff(log(**gnp**)), p=1, **details**=FALSE)[[1]] )`
`**acf2**(res^2, 20)`

![Analysis of the squared residuals of the AR(1) fit on the U.S. GNP time series based on the sample ACF and the sample PACF of the  residuals, which indicates a small but significant amount of autocorrelation](./images/fig8_2.jpg)

Figure 8.2: ACF and PACF of the squares of the residuals from the AR(1) fit on U.S. GNP. [Return to text.⏎](chapter8)

203\. We used the package fgarch to fit an AR(1)-ARCH(1) model to the U.S. GNP returns with the following results. A partial output is shown; we note that garch( 1, 0) specifies an ARCH(1) in the code below (details after example).


`library(fGarch)`
`gnpr = diff(log(**gnp**))`
`summary( garchFit(~arma(1,0) + garch(1,0), data = gnpr) )`
`            Estimate   Std.Error t.value Pr(>|t|) <- 2-sided !!!`
` φ0 mu         0.005       0.001    5.867    0.000`
` φ1 ar1        0.367       0.075    4.878    0.000`
` α0 omega      0.000       0.000    8.135    0.000 <- these parameters`
` α1 alpha1     0.194       0.096    2.035    0.042 <- can’t be negative`
` `
`   Standardised Residuals Tests: Statistic p-Value`
`    Jarque-Bera Test   R    Chi^2    9.118   0.010`
`    Shapiro-Wilk Test R     W        0.984   0.014`
`    Ljung-Box Test     R    Q(20)   23.414   0.269`
`    Ljung-Box Test     R^2 Q(20)    37.743   0.010`

Note that the given p-values are two-sided, so they should be halved when considering the ARCH parameters. In this example, we obtain ϕ^0\=.005 (called mu in the output) and ϕ^1\=.367 (called ar1) for the AR(1) parameter estimates; in [Example 5.6](#chapter5#exam5_6), the values were. 008 and. 347, respectively. The ARCH(1) parameter estimates are α^0\=0 (called mega) for the constant and α^1\=.194, which is significant with a p-value of about. 02\. There are a number of tests that are performed on the residuals \[R\] or the squared residuals \[R^2\]. For example, the Jarque–Bera statistic tests the residuals of the fit for normality based on the observed skewness and kurtosis, and it appears that the residuals have some non-normal skewness and kurtosis. The Shapiro–Wilk statistic tests the residuals of the fit for normality based on the empirical order statistics. The other tests, primarily based on the Q-statistic, are used on the residuals and their squares.

204\. The analysis of [Example 8.1](#chapter8#exam8_1) had a few problems. First, it appears that the residuals are not normal (which was the assumption for the _εt_, and there may be some autocorrelation left in the squared residuals; see [Problem 8.2](#chapter8#question8_2)). An extension of ARCH is the _generalized ARCH_ or GARCH model. For example, a GARCH(1,1) model adds an extra component to the volatility equation:

rt\=μt+σtεt(8.6)

σt2\=α0+α1rt−12+β1σt−12.(8.7)

Under the condition that α1+β1<1, using similar manipulations as in (8.4), the GARCH(1,1) model admits a non-Gaussian ARMA(1,1) model for the squared process

rt2\=α0+(α1+β1)rt−12+vt−β1vt−1,(8.8)

where we have set μt\=0 for ease, and where _vt_ is as defined in (8.4). Representation (8.8) follows by writing (8.6)–(8.7) \[with μt\=0 \] as

rt2−σt2\=σt2(εt2−1)β1(rt−12−σt−12)\=β1σt−12(εt−12−1),

subtracting the second equation from the first, and using the fact that, from (8.7), σt2−β1σt−12\=α0+α1rt−12 on the left-hand side of the result. The GARCH(p,q) model retains (8.6) and extends (8.7),

rt\=μt+σtεt,σt2\=α0+∑j\=1pαjrt−j2+∑j\=1qβjσt−j2.

Estimation of the model parameters is similar to the estimation of ARCH parameters. We explore these concepts in the following example.

Example 8.2 GARCH Analysis of the DJIA Returns [Return to text.⏎](chapter8)

As previously mentioned, the daily returns of the DJIA shown in [Figure 8.1](#chapter8#fig8_1) exhibit classic GARCH features. In addition, there is some low-level autocorrelation in the series itself, and to include this behavior, we used the fGarch package to fit an AR(1) \-GARCH(1,1) model to the series using _t_\-errors (rather than normal):


`djiar = diff(log(**djia**[,"Close"]))`
`**acf2**(djiar)      _# exhibits some autocorrelation - ACF in Figure 8.1_`
`res = resid( **sarima**(djiar, 1,0,0, **details**=FALSE)[[1]] )`
`**acf2**(res^2)      _# oozes autocorrelation - see Figure 8.1_`
`_# fit AR-GARCH model_`
`library(fGarch)`
`summary(djia.g <- garchFit( ~arma(1,0)+garch(1,1), data=djiar, cond.dist="std"))`
`             Estimate  Std.Error t.value    Pr(>|t|) <- 2-sided !!!`
`   mu       8.585e-04  1.470e-04     5.842  5.16e-09`
`   ar1    -5.531e-02   2.023e-02    -2.735  0.006239`
`   omega    1.610e-06  4.459e-07     3.611  0.000305`
`   alpha1 1.244e-01    1.660e-02     7.497  6.55e-14`
`   beta1    8.700e-01  1.526e-02    57.022  < 2e-16`
`   shape    5.979e+00  7.917e-01     7.552  4.31e-14`
`   ---`
`   Standardised Residuals Tests:`
`                                  Statistic   p-Value`
`    Ljung-Box Test     R   Q(10) 16.81507 0.0785575`
`    Ljung-Box Test     R^2 Q(10) 15.39137 0.1184312`
`plot(djia.g, which=3) _# similar to Figure 8.3_`

205\. The shape parameter is the degrees of freedom for the _t_ error distribution, which is estimated to be about 6\. Also notice that α^1+β^1 is close to 1; this is often the case. To explore the GARCH predictions of volatility, we calculated and plotted part of the data surrounding the financial crisis of 2008 along with the one-step-ahead predictions of the corresponding volatility, σt2 as a solid line in [Figure 8.3](#chapter8#fig8_3).

![GARCH model one-step-ahead predictions of the DJIA volatility with the estimated volatility superimposed](./images/fig8_3.jpg)

Figure 8.3: GARCH one-step-ahead predictions of the DJIA volatility,_σt_, superimposed on part of the data including the financial crisis of 2008. [Return to text.⏎](chapter8)

Another model that we mention briefly is the _asymmetric power ARCH_ model. The model retains (8.6), rt\=μt+σtεt, but the conditional variance is

σtδ\=α0+∑j\=1pαj(|rt−j|−γjrt−j)δ+∑j\=1qβjσt−jδ.(8.9)

Note that the model is GARCH when δ\=2 and γj\=0, for j∈{1,…,p}. The parameters _γj_ (|γj|≤1) are the _leverage_ parameters, which are a measure of asymmetry, and δ\>0 is the parameter for the power term. A positive \[negative\] value of _γj_'s means that past negative \[positive\] shocks have a deeper impact 206\. on current conditional volatility than past positive \[negative\] shocks. This model couples the flexibility of a varying exponent with the asymmetry coefficient to take the _leverage effect_ into account. Further, to guarantee that σt\>0, we assume that α0\>0, αj≥0 with at least one αj\>0, and βj≥0.

We continue the analysis of the DJIA returns in the following example.

Example 8.3 APARCH Analysis of the DJIA Returns

The package fGarch was used to fit an AR-APARCH model to the DJIA returns discussed in [Example 8.2](#chapter8#exam8_2). As in the previous example, we include an AR(1) in the model to account for the conditional mean. In this case, we may think of the model as rt\=μt+yt where _μt_ is an AR(1), and _yt_ is APARCH noise with conditional variance modeled as (8.9) with _t_\-errors. A partial output of the analysis is given below. We do not include displays, but we show how to obtain them. The predicted volatility is, of course, different than the values shown in [Figure 8.3](#chapter8#fig8_3), but appears similar when graphed.


`library(fGarch)`
`djiar = diff(log(**djia**[,"Close"]))`
`summary(djia.ap <- garchFit( ~arma(1,0)+aparch(1,1), data=djiar,`
`   cond.dist="std"))`
` `
`           Estimate   Std. Error   t value Pr(>|t|) <- still 2-sided !!!`
`  mu      3.270e-04    1.454e-04     2.249   0.0245`
`  ar1    -4.611e-02    1.943e-02    -2.373   0.0177`
`  omega   2.266e-04    4.781e-05     4.740 2.14e-06`
`  alpha1 1.233e-01     1.362e-02     9.053 < 2e-16`
`  gamma1 7.152e-01     1.097e-01     6.518 7.14e-11`
`  beta1   8.834e-01    1.232e-02    71.726 < 2e-16`
`  delta   1.033e+00    1.556e-01     6.638 3.18e-11`
`  shape   5.361e+00    5.513e-01     9.724 < 2e-16`
`  ---`
` `
`  Standardised Residuals Tests:`
`                                  Statistic p-Value`
`   Ljung-Box Test     R    Q(10) 15.89827 0.102582`
`   Ljung-Box Test     R^2 Q(10) 18.24210 0.051015`
`plot(djia.ap)   _# to see all plot options (none shown)_`

In most applications, the distribution of the noise, _εt_ in (8.1), is rarely normal. The package fGarch allows for various distributions to be fit to the data; see the help file for information. Some drawbacks of GARCH and related models are as follows. (i) The GARCH model assumes that positive and negative returns have the same effect because volatility depends on squared returns; the asymmetric models help alleviate this problem. (ii) These models are often restrictive because of the tight constraints on the model parameters. (iii) The likelihood is flat unless _n_ is very large. (iv) The models tend to overpredict volatility because they respond slowly to large isolated returns.

207\. Various extensions to the original model have been proposed to overcome some of the shortcomings we have just mentioned. For example, we have already discussed the fact that fGarch allows for asymmetric return dynamics. In the case of persistence in volatility, the integrated GARCH (IGARCH) model may be used. Recall (8.8) where we showed the GARCH(1,1) model can be written as

rt2\=α0+(α1+β1)rt−12+vt−β1vt−1

and rt2 is stationary if α1+β1<1. The IGARCH model sets α1+β1\=1, in which case the IGARCH(1,1) model is

rt\=σtεtandσt2\=α0+(1−β1)rt−12+β1σt−12.

There are many different extensions to the basic ARCH model that were developed to handle the various situations noticed in practice. Interested readers might find the general discussions in [Bollerslev et al. (1994)](#bibref1#refbib_6) and [Shephard (1996)](#bibref1#refbib_41) worthwhile reading. Two excellent texts on financial time series analysis are [Chan (2002)](#bibref1#refbib_10) and [Tsay (2005)](#bibref1#refbib_51).

Example 8.4 Some ARCH Model Theory \* [Return to text.⏎](chapter8)

Here, we discuss some details regarding ARCH models. Recall that the ARCH(1) model for returns, _rt_, is

rt\=σtεtσt2\=α0+α1rt−12,

where _εt_ is standard Gaussian white noise, εt∼ iid N(0,1).

First, notice that the conditional distribution of _rt_ given rt−1 is Gaussian:

rt|rt−1∼N(0,α0+α1rt−12).(8.10)

In addition, it was shown that the squared returns are a non-Gaussian AR(1) model

rt2\=α0+α1rt−12+vt,(8.11)

where vt\=σt2(εt2−1).

To further explore the properties of ARCH, we define Rs\={rs,rs−1,…}. Then, using [Property A.1](#appA#propA_1) and ([8.10](#appA#propA_1)), we immediately see that _rt_ has a zero mean,

E(rt)\=EE(rt|Rt−1)\=EE(rt|rt−1)\=0.(8.12)

Because E(rt∣Rt−1)\=0, the process _rt_ is said to be a _martingale difference_.

Because _rt_ is a martingale difference, it is also an uncorrelated sequence. For example, with h\>0,

cov(rt,rt−h)\=E(rt−h rt)\=EE(rt−h rt∣Rt−1)\=E{rt−h E(rt∣Rt−1)}\=0.(8.13)

208\. The last line of (8.13) follows because rt−h belongs to the information set Rt−1 for h\>0, and E(rt∣Rt−1)\=0 as determined in (8.12).

An argument similar to (8.12) and (8.13) will establish the fact that the error process _vt_ in (8.4) is also a martingale difference and consequently an uncorrelated sequence. If the variance of _vt_ is finite and constant with respect to time, and 0≤α1<1, then based on [Property 4.38](#chapter4#prop4_38), (8.11) specifies a causal AR(1) process for rt2. Therefore, E(rt2) and var(rt2) must be constant with respect to time _t_. This, implies that

E(rt2)\=var(rt)\=α01−α1

and, after some manipulations,

E(rt4)\=3α02(1−α1)21−α121−3α12,

provided 3α12<1. Note that

var(rt2)\=E(rt4)−\[E(rt2)\]2,

which exists only if 0<α1<1/3≈.58. In addition, these results imply that the kurtosis, _κ_, of _rt_ is

κ\=E(rt4)\[E(rt2)\]2\=31−α121−3α12,

which is never smaller than 3, the kurtosis of the normal distribution. Thus, the marginal distribution of the returns, _rt_, is leptokurtic, or has “fat tails.” Summarizing, if 0≤α1<1, the process _rt_ itself is white noise and its unconditional distribution is symmetrically distributed around zero; this distribution is leptokurtic. If, in addition, 3α12<1, the square of the process, rt2, follows a causal AR(1) model with ACF given by ρy2(h)\=α1h≥0, for all h\>0.

Estimation of the parameters _α_0 and _α_1 of the ARCH(1) model is typically accomplished by conditional MLE. The conditional likelihood of the data r2,…,rn given _r_1, is given by

L(α0,α1|r1)\=∏t\=2nfα0,α1(rt|rt−1),

where the density fα0,α1(rt|rt−1) is the normal density specified in (8.3). Hence, the criterion function to be minimized, l(α0,α1)∝−lnL(α0,α1|r1) is given by

l(α0,α1)\=12∑t\=2nln(α0+α1rt−12)+12∑t\=2n(rt2α0+α1rt−12).

Estimation is accomplished by numerical methods, as described in [Section 4.5](#chapter4#sec4_5). The likelihood of the ARCH model tends to be flat unless _n_ is very large. A discussion of this problem can be found in [Shephard (1996)](#bibref1#refbib_41).

## 8.2 209\. Unit Root Testing

The use of the first difference ∇xt\=(1−B)xt can sometimes be too severe a modification in the sense that an integrated model might represent an overdifferencing of the original process. For example, in [Example 5.9](#chapter5#exam5_9) we fit an ARIMA(1,1,1) model to the logged varve series. The idea of differencing the series was first made in [Example 4.28](#chapter4#exam4_28) because the series appeared to take long 100+ year walks in positive and negative directions.

[Figure 8.4](#chapter8#fig8_4) compares the sample ACF of a generated random walk with that of the logged varve series.[3](#chapter8#fn8_3) In both cases the sample correlations decrease linearly and remain significant for many lags; however, the sample ACF of the random walk has much larger values.


`par(mfrow=2:1)`
`**acf1**(cumsum(rnorm(634)), 100, col=4, main="Series: random walk")`
`**acf1**(log(**varve**), 100, col=4, ylim=c(-.1,1))`

![Comparison of the sample autocorrelations  of a simulated random walk and of the log transformed varve series showing that the ACFs are similar, but the random walk has large values whereas the long memory series does not have large values](./images/fig8_4.jpg)

Figure 8.4: Sample ACFs a random walk and of the log transformed varve series. [Return to text.⏎](chapter8)

First, consider a normal AR(1) process,

xt\=ϕxt−1+wt.

A unit root test provides a way to test whether ϕ\=1 (the null case) as opposed to a causal process (the alternative). That is, it provides a procedure for testing

H0:ϕ\=1versusH1:|ϕ|<1.

To see if the null hypothesis is reasonable, an obvious test statistic would be to consider (ϕ^−1), appropriately normalized, in the hope to develop a t-test, where ϕ^ is one of the optimal estimators discussed in [Section 4.5](#chapter4#sec4_5). The theory for the distribution of ϕ^ does not work here because the process is not stationary under the null hypothesis.

However, the test statistic

T\=n(ϕ^−1)

can be used, and it is known as the unit root or Dickey–Fuller (DF) statistic, although the actual DF test statistic is normalized a little differently. In this case, the distribution of the test statistic does not have a closed form and quantiles of the distribution must be computed by numerical approximation or by simulation. The package tseries provides this test along with more general tests that we mention briefly.

Toward a more general model, we note that the DF test was established by noting that if xt\=ϕxt−1+wt, then

∇xt\=(ϕ−1)xt−1+wt\=γxt−1+wt,

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 3Although a random walk is not stationary, it can be instructive to view a sample ACF in terms of lag alone, even though the true ACF does not depend on lag alone. [Return to text.⏎](#chapter8#fn8_38b)

210\. and one could test H0:γ\=0 by regressing ∇xt on xt−1 and obtaining the regression coefficient estimate γ^. Then, the statistic nγ^ is formed and its large sample distribution derived.

The test was extended to accommodate AR(_p_) models, xt\=∑j\=1pϕjxt−j+wt, in a similar way. For example, write an AR(2) model

xt\=ϕ1xt−1+ϕ2xt−2+wt,

as

xt\=(ϕ1+ϕ2)xt−1−ϕ2(xt−1−xt−2)+wt,

and subtract xt−1 from both sides. This yields

∇xt\=γxt−1+ϕ2∇xt−1+wt,

where γ\=ϕ1+ϕ2−1. To test the hypothesis that the process has a unit root at 1 (i.e., the AR polynomial ϕ(z)\=1−ϕ1z−ϕ2z2\=0 when z\=1), we can test H0:γ\=0 by estimating _γ_ in the regression of ∇xt on xt−1 and ∇xt−1 and forming a test statistic. For AR(_p_) model, one regresses ∇xt on xt−1 and ∇xt−1…,∇xt−p+1, in a similar fashion to the AR(2) case.

This test leads to the so-called augmented Dickey–Fuller test (ADF). While the calculations change for obtaining the large sample null distribution, the basic ideas and machinery remain the same as in the simple case. The choice of _p_ is crucial, and we will discuss some suggestions in the example. For ARMA(_p,q_) models, the ADF test can be used by assuming _p_ is large enough to capture the essential correlation structure; recall ARMA(_p,q_) models are AR(∞) models. An alternative is the Phillips–Perron (PP) test, which differs from the ADF tests mainly in how it deals with serial correlation and heteroscedasticity in the errors.

Example 8.5 211\. Testing Unit Roots in the Glacial Varve Series [Return to text.⏎](chapter8)

In this example we use the package tseries to test the null hypothesis that the log of the glacial varve series has a unit root, versus the alternate hypothesis that the process is stationary. We test the null hypothesis using the available DF, ADF, and PP tests; note that in each case, the general regression equation incorporates a constant and a linear trend. In the ADF test, the default number of AR components included in the model is k≈(n−1)13, which has theoretical justification on how _k_ should grow compared to the sample size _n_. For the PP test, the default value is k≈.04n14.


`library(tseries)`
`adf.test(log(**varve**), k=0)               _# DF test_`
`  Dickey-Fuller = -12.8572, Lag order = 0, p-value < 0.01`
`   alternative hypothesis: stationary`
`adf.test(log(**varve**))                    _# ADF test_`
`  Dickey-Fuller = -3.5166, Lag order = 8, p-value = 0.04071`
`   alternative hypothesis: stationary`
`pp.test(log(**varve**))                     _# PP test_`
`   Dickey-Fuller Z(alpha) = -304.5376,`
`    Truncation lag parameter = 6, p-value < 0.01`
`    alternative hypothesis: stationary`

In each test, we reject the null hypothesis that the logged varve series has a unit root. The conclusion of these tests supports the conclusion of [Example 8.6](#chapter8#exam8_6) in [Section 8.3](#chapter8#sec8_3), where it is postulated that the logged varve series is long memory. Fitting a long memory model to these data would be the natural progression of model fitting once the unit root test hypothesis is rejected.

## 8.3 Long Memory and Fractional Differencing

The conventional ARMA(p,q) process is often referred to as a short memory process because the coefficients in the representation

xt\=∑j\=0∞ψjwt−j,

are dominated by exponential decay where ∑j\=0∞|ψj|<∞ (e.g., recall [Example 4.3](#chapter4#exam4_3)). This result implies the ACF of the short memory process ρ(h)→0 exponentially fast as h→∞. When the sample ACF of a time series decays slowly, the advice given in [Chapter 5](#chapter5) is to difference the series until it seems stationary. Following this advice with the glacial varve series first presented in [Example 4.28](#chapter4#exam4_28) leads to the first difference of the logarithms of the data, xt\= log( varve), being represented as a first-order moving average. In [Example 5.9](#chapter5#exam5_9), further analysis of the residuals led us to fitting an ARIMA(1,1,1) model with 212\. results:

∇x^t\=.23∇x^t−1+w^t−.89w^t−1.

But the use of the first difference ∇xt\=(1−B)xt can sometimes be too severe of a transformation. For example, suppose _xt_ is the causal AR(1) model

xt\=.9xt−1+wt.

If we multiply through by ∇\=(1−B), we have

∇xt\=.9∇xt−1+wt−wt−1.

This means that ∇xt is a problematic ARMA(1,1) because the moving average part is noninvertible. Thus, by overdifferencing in this example, we have gone from a simple causal AR(1) to a noninvertible ARIMA(1,1,1). This is precisely why we gave several warnings about the overuse of differencing in [Chapter 5](#chapter5).

Long memory time series were considered in [Hosking (1981)](#bibref1#refbib_24) and [Granger and Joyeux (1980)](#bibref1#refbib_20) as intermediate compromises between the short memory ARMA type models and the fully integrated nonstationary processes in the ARIMA class. The easiest way to generate a long memory series is to think of using the difference operator (1−B)d for fractional values of _d_, 0<d<.5. A basic long memory series can be generated as

(1−B)dxt\=wt,(8.14)

where _wt_ still denotes white noise with variance σw2. The fractionally differenced series (8.14), for |d|<.5, is often called fractional noise (except when _d_ is zero). Now, _d_ becomes a parameter to be estimated along with σw2. Differencing the original process, as in the Box–Jenkins approach, may be thought of as simply assigning a value of d\=1. This idea has been extended to the class of fractionally integrated ARMA, or ARFIMA models, where −.5<d<.5; when _d_ is negative, the term antipersistent is used. Long memory processes occur in hydrology [(see Hurst, 1951](#bibref1#refbib_25); [McLeod and Hipel, 1978)](#bibref1#refbib_33) and in environmental series, such as the varve data we have previously analyzed, to mention a few examples. Long memory time series data tend to exhibit sample autocorrelations that are not necessarily large (as in the case of d\=1), but persist for a long time. [Figure 8.4](#chapter8#fig8_4) shows the sample ACF, to lag 100, of the log transformed varve series, which exhibits classic long memory behavior.

To investigate its properties, we can use the binomial expansion[4](#chapter8#fn8_4) (d\>−1) to write

wt\=(1−B)dxt\=∑j\=0∞πjBjxt\=∑j\=0∞πjxt−j(8.15)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 4The binomial expansion in this case is the Taylor series about z\=0 for functions of the form (1−z)d. [Return to text.⏎](#chapter8#fn8_48b)

213\. where

πj\=Γ(j−d)Γ(j+1)Γ(−d)(8.16)

with Γ(x+1)\=xΓ(x) being the gamma function. Similarly (d<1), we can write

xt\=(1−B)−dwt\=∑j\=0∞ψjBjwt\=∑j\=0∞ψjwt−j(8.17)

where

ψj\=Γ(j+d)Γ(j+1)Γ(d).(8.18)

When |d|<.5, the processes (8.15) and (8.17) are well-defined stationary processes [(see Brockwell and Davis, 2013, for details)](#bibref1#refbib_8). In the case of fractional differencing, the coefficients satisfy ∑πj2<∞ and ∑ψj2<∞ as opposed to the absolute summability of the coefficients in ARMA processes.

Using the representation (8.17)–(8.18), it can be shown that the ACF of _xt_ is

ρ(h)\=Γ(h+d)Γ(1−d)Γ(h−d+1)Γ(d)∼h2d−1(8.19)

for large _h_. From this we see that for 0<d<.5,

∑h\=−∞∞|ρ(h)|\=∞

and hence the term _long memory_.

To examine a series such as the varve series for a possible long memory pattern, it is convenient to look at ways of estimating _d_. Using (8.16) we obtain the recursions

πj+1(d)\=(j−d)πj(d)(j+1),(8.20)

for j\=0,1,…, with π0(d)\=1. In the normal case, we may estimate _d_ by minimizing the sum of squared errors

Q(d)\=∑wt2(d).

The usual Gauss–Newton method, described in [Section 4.5](#chapter4#sec4_5), leads to the expansion

wt(d)≈wt(d0)+wt′(d0)(d−d0),

where

wt′(d0)\=∂wt∂d|d\=d0

214\. and _d_0 is an initial estimate (guess) at the value of _d_. Setting up the usual regression leads to

d\=d0−∑twt′(d0)wt(d0)∑twt′(d0)2.(8.21)

The derivatives are computed recursively by differentiating (8.20) successively with respect to _d_: πj+1′(d)\=\[(j−d)πj′(d)−πj(d)\]/(j+1), where π0′(d)\=0. The errors are computed from an approximation to (8.15), namely,

wt(d)\=∑j\=0tπj(d)xt−j.(8.22)

It is advisable to omit a number of initial terms from the computation and start the sum, (8.21), at some fairly large value to have a reasonable approximation.

Example 8.6 Long Memory Fitting of the Glacial Varve Series [Return to text.⏎](chapter8)

We consider analyzing the glacial varve series discussed in [Example 3.14](#chapter3#exam3_14) and [Example 4.28](#chapter4#exam4_28). [Figure 3.11](#chapter3#fig3_11) shows the original and log-transformed series (which we denote by _xt_). In [Example 5.9](#chapter5#exam5_9), we noted that _xt_ could be modeled as an ARIMA(1,1,1) process. Here, we fit the fractionally differenced model, (8.14), to _xt_ using the package arfima. Applying the Gauss–Newton iterative procedure previously described leads to a final value of d^≈.373. We can evaluate the performance of the fractional noise fit to the data by examining the innovations as shown in [Figure 8.5](#chapter8#fig8_5).


`library(arfima)   _# assumes arfima package installed_`
`summary(varve.fd <- arfima(log(**varve**), order = c(0,0,0)))`
`  Mode 1 Coefficients:`
`              Estimate Std.Error Th.Std.Err. z-value       Pr(>|z|)`
` d.f         0.3727893 0.0273459     0.0309661 13.6324 < 2.22e-16`
` Fitted mean 3.0814142 0.2646507            NA 11.6433 < 2.22e-16`
` ---`
` sigma^2 estimated as 0.230081`
` Log-likelihood = 466.028; AIC = -926.056; BIC = -912.699`
`_# residual analysis_`
`` innov = resid(varve.fd)[[1]]   _# resid() produces a `list`_ ``
`**sarima**(innov, col=4)           _# arfima residuals (no need to adjust dfs)_`
`**sarima**(log(**varve**),1,1,1,no=TRUE) _# from Example 5.9 (not shown)_`

![Residual analysis of a pure fractional noise fit to varve data](./images/fig8_5.jpg)

Figure 8.5: Residual analysis of the fractional noise fit, w^t\=(1−B).37xt to the logged varve data, _xt_. [Return to text.⏎](chapter8)

In this case, the residuals, w^t\=(1−B).37xt, are comparable with the white noise assumption, and the fractional noise model is much simpler than the ARIMA(1,1,1) model of [Example 5.9](#chapter5#exam5_9).

The set of estimated coefficients πj(d^) are displayed in [Figure 8.6](#chapter8#fig8_6) with π0(d^)\=1 using the code:


`d = coef(varve.fd)[1]`
`p = c(1)`
`for (k in 1:30) { p[k+1] = (k-d)*p[k]/(k+1) }`
`**tsplot**(1:30, p[-1], ylab=bquote(pi(d)), lwd=2, xlab="Index", type="h", col=4)`

![Coefficients for an inverted fitted fractional difference on the varve series](./images/fig8_6.jpg)

Figure 8.6: Coefficients πj(d^) for j\=1,2,…,30 and d^≈.373 in the representation (8.20). [Return to text.⏎](chapter8)

215\. Forecasting long memory processes is similar to forecasting ARIMA models. That is, (8.15) and (8.20) can be used to obtain the truncated forecasts

xn+mn\=−∑j\=1n+m−1πj(d^) xn+m−jn,(8.23)

for m\=1,2,…. Error bounds can be approximated by using

Pn+mn\=σ^w2∑j\=0m−1ψj2(d^)(8.24)

where, as in (8.20),

ψj(d^)\=(j+d^)ψj(d^)(j+1),(8.25)

with ψ0(d^)\=1.

No obvious short memory ARMA-type component can be seen in the ACF of the residuals from the fractionally differenced varve series shown in [Figure 8.5](#chapter8#fig8_5). It is natural, however, that cases will exist in which short memory-type components 216\. will also be present in data that exhibits long memory. Hence, it is natural to define the general ARFIMA(p,d,q),−.5<d<.5 process as

ϕ(B)∇d(xt−μ)\=θ(B)wt,(8.26)

where ϕ(B) and θ(B) are as given in [Chapter 4](#chapter4). Writing the model in the form

ϕ(B)πd(B)(xt−μ)\=θ(B)wt(8.27)

makes it clear how we go about estimating the parameters for the more general model. Forecasting for the ARFIMA(p,d,q) series can be easily done, noting that we may equate coefficients in

ϕ(z)ψ(z)\=(1−z)−dθ(z)(8.28)

and

θ(z)π(z)\=(1−z)dϕ(z)(8.29)

to obtain the representations

xt\=μ+∑j\=0∞ψjwt−j

and

wt\=∑j\=0∞πj(xt−j−μ).

We then can proceed as discussed in (8.23) and (8.24).

Example 8.7 Glacial Varve Series (cont.)

Although there was no indication of a short memory component in [Example 8.6](#chapter8#exam8_6), we demonstrate how to include such terms. As an example, we will 217\. fit an extra MA term and may consider this an overfitting exercise to confirm there are no short memory terms.


`library(arfima)`
`summary(varve1.fd <- arfima(log(**varve**), order=c(0,0,1)))`
`                Estimate Std.Error   Th.Std.Err z-value Pr(>|z|)`
`  theta(1)    0.0705603 0.0648228     0.0670149 1.08851 0.27637`
`  d.f         0.4089730 0.0440908     0.0523832 9.27569 < 2e-16`
`  Fitted mean 3.0775613 0.3541186            NA 8.69076 < 2e-16`
`  ---`
`  sigma^2 estimated as 0.22985; AIC = -925.306; BIC = -907.498`

We see that the short memory MA term is not significant and both information criteria prefer the model without the MA term. To try an AR short memory term, use order= c( 1, 0, 0) instead; doing so leads to the same conclusion.

## 8.4 State Space Models

A very general model that subsumes a whole class of special cases of interest in much the same way that linear regression does is the state space model that was described in [Kalman (1960)](#bibref1#refbib_30) and [Kalman and Bucy (1961)](#bibref1#refbib_31). The model arose in the space tracking setting where the state equation defines the motion equations for the position or state of a spacecraft with location _xt_ and the data _yt_ reflect information that can be observed from a tracking device. We focus on the univariate case here, although it is often applied to multivariate time series. For more details, see [Shumway and Stoffer (2025, ch. 6)](#bibref1#refbib_44) for a general treatment of the model and numerous examples.

The state space model is characterized by two principles. First, there is a hidden or latent process _xt_ called the state process. The unobserved state process is assumed to be an AR(1),

xt\=α+ϕxt−1+wt,(8.30)

where wt∼iid N(0,σw2). In addition, we assume the initial state is x0∼N(μ0,σ02). The second condition is that the observations, _yt_, are given by

yt\=Axt+vt,(8.31)

where _A_ is a constant and the observation noise is vt∼iid N(0,σv2). In addition, _x_0, {wt} and {vt} are uncorrelated. This means that the dependence among the observations is generated by states. The principles are displayed in [Figure 8.7](#chapter8#fig8_7).

![Diagram of a state space model displaying how the data are conditionally independent given the states](./images/fig8_7.jpg)

Figure 8.7: Diagram of a state space model. [Return to text.⏎](chapter8)

A primary aim of any analysis involving the state space model, (8.30)–(8.31), is to produce estimators for the underlying unobserved signal _xt_, given the data y1:s\={y1,…,ys}, to time _s_:

* 218\. When s<t, the problem is called _forecasting_ or _prediction_.
* When s\=t, the problem is called _filtering_.
* When s\>t, the problem is called _smoothing_.

In addition to these estimates, we would also want to measure their precision. The solution to these problems is accomplished via the _Kalman filter_ and _smoother_.

As a simple example, suppose it is necessary to control the internal temperature, _xt_, of a combustion chamber that gets too hot to measure directly. The model for _xt_ is determined by the physics of the system. To determine the current temperature at any given time _t_, a sensor is put near the chamber but on a cooler surface so that the measurement, _yt_, is indirect; e.g., yt≈.01 xt. The measurement error term, _vt_, is based on the calibration or accuracy of the particular sensor. Since it is important that the internal temperature not get too high or low, given measurements y1,y2,…yt−1, it would be important to _predict_ the actual temperature at the next reading, _xt_. If it is predicted to be too high, for example, we could cool the system ahead of time. Next, given a new reading, _yt_, it is important to estimate (_filter_) the current temperature of the chamber. Finally, it may be important to reassess (_smooth_) the internal temperature, _xt_, given the measurements y1,…,yn for n\>t.

First, we present the Kalman filter, which gives the prediction and filtering equations. We use the following notation,

xts\=E(xt∣y1:s)andPts\=E(xt−xts)2.

The advantage of the Kalman filter is that it specifies how to update a prediction when a new observation is obtained without having to reprocess the entire data set.

Property 8.8 (The Kalman Filter). [Return to text.⏎](chapter8)

219\. _For the state space model specified in (8.30) and (8.31), with initial conditions_ x00\=μ0 _and_ P00\=σ02, _for_ t\=1,…,n,

xtt−1\=α+ϕxt−1t−1andPtt−1\=ϕ2Pt−1t−1+σw2.(predict)xtt\=xtt−1+Kt(yt−Axtt−1)andPtt\=\[1−KtA\]Ptt−1,(filter)

_where_

Kt\=Ptt−1A/ΣtandΣt\=A2Ptt−1+σv2.

Important byproducts of the filter are the independent innovations (prediction errors)

εt\=yt−E(yt∣y1:t−1)\=yt−ytt−1\=yt−Axtt−1,(8.32)

with εt∼N(0,Σt).

Derivation of the Kalman filter may be found in many sources such as [Shumway and Stoffer (2025, ch. 6)](#bibref1#refbib_44). The prediction equation follows directly from the model using ideas from [Section 4.6](#chapter4#sec4_6). That is, since xt\=α+ϕxt−1+wt, the prediction is

xtt−1\=α+ϕxt−1t−1+wtt−1\=α+ϕxt−1t−1,

noting wtt−1\=0 because _wt_ is a future system error.

For filtering, we must update the state estimate given a new observation, _yt_. An easy way to think about how the update works is to consider the case where A\=1 so that yt\=xt+vt. In this case, the filter can be written as

xtt\=(1−Kt) xtt−1+Kt yt,

where _Kt_ is called the _gain_ of the new information. Note that Kt\=Ptt−1/(Ptt−1+σv2) so that 0≤Kt≤1 because all terms are variances. We see that the filter is a linear combination of the prediction and the new observation. Moreover, the influence of the new observation depends on the size of the observational noise variance, σv2 and the prediction error, Ptt−1. If _Kt_ is close to zero, the influence of a new observation is small, and if _Kt_ is close to one, that influence is large.

For smoothing, we need estimators for _xt_ based on the entire data sample y1,…,yn, namely, xtn. These estimators are called smoothers because a time plot of xtn for t\=1,…,n is smoother than the forecasts xtt−1 or the filters xtt.

Property 8.9 220\. (The Kalman Smoother). _For the state space model specified in (8.30) and (8.31), with initial conditions_ xnn _and_ Pnn _obtained via [Property 8.8](#chapter8#prop8_8), for_ t\=n,n−1,…,1,

xt−1n\=xt−1t−1+Jt−1(xtn−xtt−1)andPt−1n\=Pt−1t−1+Jt−12(Ptn−Ptt−1)

_where_ Jt−1\=ϕ Pt−1t−1/Ptt−1.

Estimation of the parameters that specify the state space model, (8.30) and (8.31), is similar to estimation for ARIMA models. In fact, R uses the state space form of the ARIMA model for estimation. For ease, we represent the vector of unknown parameters as θ\=(α,ϕ,σw,σv). Unlike the ARIMA model, there is no restriction on the _ϕ_ parameter. The likelihood is computed using the innovation sequence _εt_ given in (8.32). Ignoring a constant, we may write the normal likelihood, LY(θ), as

−2logLY(θ)\=∑t\=1nlogΣt(θ)+∑t\=1nεt2(θ)Σt(θ),(8.33)

where we have emphasized the dependence of the innovations on the parameters _θ_. The numerical optimization procedure combines a Newton-type method for maximizing the likelihood with the Kalman filter for evaluating the innovations given the current value of _θ_.

Example 8.10 Global Temperature [Return to text.⏎](chapter8)

In [Example 1.3](#chapter1#exam1_3) we considered the annual temperature anomalies averaged over the Earth's land area from 1850 to 2023\. In [Example 3.13](#chapter3#exam3_13), we suggested that global temperature behaved as a random walk with drift,

xt\=α+xt−1+wt,

so that ϕ\=1. We may consider the global temperature data as being noisy observations on the _xt_ process,

yt\=xt+vt,

with _vt_ being the measurement error. [Figure 8.8](#chapter8#fig8_8) shows the estimated smoother (with error bounds) superimposed on the observations. The code is as follows.


`fit = **ssm**(**gtemp_land**, A=1, alpha=.01, phi=1, sigw=.01, sigv=.1, **fixphi**=TRUE)`
`          estimate          SE`
`  alpha 0.01428341 0.005139577`
`  sigw 0.06643936 0.013371066`
`  sigv 0.29495355 0.017371490`
`**tsplot**(**gtemp_land**, col=4, type="o", pch=20, ylab="Temperature Deviations")`
`lines(fit$Xs, col=6, lwd=2)`
` xx = c(time(fit$Xs), rev(time(fit$Xs)))`
` yy = c(fit$Xs-2*sqrt(fit$Ps), rev(fit$Xs+2*sqrt(fit$Ps)))`
`polygon(xx, yy, border=8, col=gray(.6, alpha=.25) )`

![The data, and the smoothed annual global land surface temperatures based on a Kalman smoother](./images/fig8_8.jpg)

Figure 8.8: Yearly average global land surface temperature deviations (1850–2023) in ∘C and the estimated Kalman smoother with ±2 error bounds. [Return to text.⏎](chapter8)

221\. Note that we have fixed ϕ\=1 by specifying fixphi=TRUE in the call (the default for this is FALSE). To plot the predictions, change Xs and Ps to Xp and Pp, respectively, in the code above. For the filters, use Xf and Pf.

## 8.5 Cross-Correlation Analysis and Prewhitening

In [Example 2.34](#chapter2#exam2_34) we discussed the fact that to use [Property 2.31](#chapter2#prop2_31), at least one of the series must be white noise. Otherwise, there is no simple way of telling if a cross-correlation estimate is significantly different from zero. For example, in [Example 3.6](#chapter3#exam3_6) and [Problem 3.2](#chapter3#question3_2), we considered the effects of temperature and pollution on cardiovascular mortality. Although it appeared that pollution might lead mortality, it is difficult to discern that relationship without first prewhitening one of the series. In this case, plotting the series as a time plot as in [Figure 3.4](#chapter3#fig3_4) did not help much in determining the lead-lag relationship of the two series. In addition, [Figure 8.9](#chapter8#fig8_9) shows the CCF between the two series, and it is also difficult to extract pertinent information from the graphic.

![Cross-correlation function between cardiovascular mortality and particulate pollution from the LA mortality-pollution study](./images/fig8_9.jpg)

Figure 8.9: CCF between cardiovascular mortality and particulate pollution. [Return to text.⏎](chapter8)

222\. First, consider a simple case where we have two time series _xt_ and _yt_ satisfying

xt\=xt−1+wt,yt\=xt−3+vt,

so that _xt_ leads _yt_ by three time units (_wt_ and _vt_ are independent noise series). To use [Property 2.31](#chapter2#prop2_31), we may whiten _xt_ by simple differencing ∇xt\=wt and to maintain the relationship between _xt_ and _yt_, we should transform the _yt_ in a similar fashion,

∇xt\=wt,∇yt\=∇xt−3+∇vt.

Thus, if the variance of ∇vt is not too large, there will be strong cross-correlation between ∇yt and ∇xt−3 (or ∇yt and ∇xt are correlated at lag 3).

The steps for prewhitening follow the simple case. We have two time series _xt_ and _yt_, and we want to examine the lead-lag relationship between the two. At this point, we have a method to whiten a series using an ARIMA model. That is, if _xt_ is ARIMA, then the residuals from the fit, w^t, should be white noise. We may then use w^t to investigate cross-correlation with a similarly transformed _yt_ series as follows:

1. Fit an ARIMA model to one of the series, _xt_,  
ϕ^(B)(1−B)dxt\=α^+θ^(B)w^t,  
and obtain the residuals by inversion, w^t\=π^(B)xt.  
_An alternative would be to simply fit a large order AR(p) model_ to the (possibly differenced) data, and then use those residuals. In this case, the innovations have a simple form, π^(B)\=ϕ^(B)(1−B)d.
2. 223\. Use the fitted model in the previous step to filter the _yt_ series in the same way,  
y^t\=π^(B)yt.
3. Now perform the cross-correlation analysis on the w^t and y^t processes.

The script **pre.white**() in astsa performs this analysis automatically as demonstrated in the following example.

Example 8.11 Mortality and Pollution [Return to text.⏎](chapter8)

In [Example 3.6](#chapter3#exam3_6), we regressed cardiovascular mortality (**cmort**) on temperature (**tempr**) and particulate pollution (**part**) using only contemporaneous values. In [Problem 3.2](#chapter3#question3_2) we considered fitting an additional component of pollution lagged at four weeks because it appeared that pollution may lead mortality by about a month. However, we did not have all the tools we needed to determine if a lead-lag relationship existed between the two series.

We will concentrate on mortality and pollution and leave the analysis of mortality and temperature for [Problem 8.10](#chapter8#question8_10). [Figure 8.9](#chapter8#fig8_9) shows the sample CCF between mortality and pollution. Notice the resemblance between [Figure 8.9](#chapter8#fig8_9) and [Figure 2.6](#chapter2#fig2_6) prior to prewhitening. The CCF shows that the data have an annual cycle, but it is not easy to determine any lead-lag relationship.

According to the procedure, we first whiten cmort by fitting an appropriate model. The data are shown in [Figure 3.3](#chapter3#fig3_3) where we notice there is trend. An obvious next step would be to examine the behavior of the differenced cardiovascular mortality. [Figure 8.10](#chapter8#fig8_10) shows the sample P/ACF of ∇Mt and an AR(1) is suggested.

![Sample ACF and sample PACF of the differenced cardiovascular mortality time series](./images/fig8_10.jpg)

Figure 8.10: P/ACF of differenced cardiovascular mortality. [Return to text.⏎](chapter8)

224\. Then we used **pre.white**() to perform an automatic cross-correlation procedure on the two difference series. [Figure 8.11](#chapter8#fig8_11) shows the resulting sample CCF, where we note that the zero-lag correlation is predominant. The fact that the two series move at the same time makes sense considering that the data evolve over a week. We do mention that there may be some significant but small correlation when pollution leads by four weeks.


`**ccf2**(**cmort**, **part**, col=4) _# Figure 8.9_`
`**acf2**(diff(**cmort**), col=4) _# Figure 8.10     suggests AR(1)_`
`**pre.white**(**cmort**, **part**, diff=TRUE, col=4)`
`  cmort prewhitened using an AR p = 3`
`  after differencing d = 1`

![Sample cross-correlation function between whitened cardiovascular mortality and filtered particulate pollution](./images/fig8_11.jpg)

Figure 8.11: CCF between whitened cardiovascular mortality and filtered particulate pollution. [Return to text.⏎](chapter8)

In the analysis, we allowed the script to fit an AR model to the differenced mortality data via AIC. We could have restricted that by including order.max= 1 in the call above, but the results are not much different.

## 8.6 Bootstrapping Autoregressive Models

When estimating the parameters of ARMA processes, we rely on large sample results such as [Example 4.32](#chapter4#exam4_32) to develop confidence intervals. For example, for an AR(1), if _n_ is large, an approximate 100(1−α)% confidence interval for _ϕ_ is

ϕ^±zα/21−ϕ^2n.

If _n_ is small or the parameters are close to the boundaries, the large sample approximations can be quite poor. The bootstrap can be helpful in this case. A general treatment of the bootstrap may be found in [Efron and Tibshirani (1994)](#bibref1#refbib_15). 225\. We discuss the case of an AR(1) here, the AR(_p_) case follows directly. For ARMA and more general models, see [Shumway and Stoffer (2025, ch. 6)](#bibref1#refbib_44).

We consider an AR(1) model with a regression coefficient near the boundary of causality and an error process that is symmetric but not normal. Specifically, consider the causal model

xt\=μ+ϕ(xt−1−μ)+wt,(8.34)

where μ\=50, ϕ\=.95, and _wt_ are iid Laplace (double exponential) with location zero, and scale parameter β\=2. The density of _wt_ is given by

g(w)\=12βexp{−|w|/β}−∞<w<∞.

In this example, E(wt)\=0 and var(wt)\=2β2\=8. [Figure 8.12](#chapter8#fig8_12) shows n\=100 simulated observations from this process as well as a comparison between the standard normal and the standard Laplace densities. Notice that the Laplace density has larger tails.

![One hundred observations generated from an AR(1) model with Laplace errors and the standard Laplace distribution compared to a standard normal distribution](./images/fig8_12.jpg)

Figure 8.12: Left: One hundred observations generated from the AR(1) model with Laplace errors, (8.34).Right: Standard Laplace (blue) and normal (red) densities. [Return to text.⏎](chapter8)

To show the advantages of the bootstrap, we will act as if the data are normal. The data in [Figure 8.12](#chapter8#fig8_12) were generated as follows.


`_# data_`
`set.seed(101010)`
`e   = rexp(150, rate=.5); u = runif(150,-1,1); de = e*sign(u)`
`dex = 50 + **sarima.sim**(n=100, ar=.95, innov=de, burnin=50)`
`layout(matrix(1:2, nrow=1), widths=c(5,2))`
`**tsplot**(dex, col=4, ylab=bquote(X[~t]), **gg**=TRUE)`
`_# densities for comparison_`
`g = function(x) {.5*dexp(abs(x), rate = 1/sqrt(2))}`
`w = seq(-6, 6, by=.01)`
`**tsplot**(w, dnorm(w), **gg**=TRUE, col=2, xlab="w", ylab="g(w)")`
`lines(w, g(w), col=4)`

Using these data, we obtained the following Yule–Walker estimates:


`_# estimation_`
`fit = ar.yw(dex, aic=FALSE, order=1)`
`round(estyw <- c(mean=fit$x.mean, ar1=fit$ar, se=sqrt(fit$asy.var.coef),`
`   var=fit$var.pred), 3)`
`    mean    ar1     se    var`
`  44.496 0.966 0.026 6.151`

To assess the finite sample distribution of ϕ^ when n\=100, we simulated 1000 realizations of this AR(1) process and estimated the parameters via Yule–Walker. [5](#chapter8#fn8_5) The finite sampling density of the Yule–Walker estimate of _ϕ_, based on the 1000 repeated simulations, is shown in [Figure 8.13](#chapter8#fig8_13). Based on [Example 4.32](#chapter4#exam4_32), we would say that ϕ^ is approximately normal with mean _ϕ_ (which we will not know) and variance (1−ϕ2)/100, which we would approximate by 226\. (1−.9662)/100\=.0262; this distribution is superimposed on [Figure 8.13](#chapter8#fig8_13). Clearly the sampling distribution is not close to normality for this sample size. The code to perform the simulation is as follows. We use the results at the end of the example.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 5We are using Yule–Walker estimation because for simulation-based methods it is faster than MLE. [Return to text.⏎](#chapter8#fn8_58b)


`_# finite sample distribution_`
`phi.yw = c()`
`for (i in 1:1000){`
`  e = rexp(150, rate=.5); u = runif(150,-1,1); de = e*sign(u)`
`  x = 50 + **sarima.sim**(n=100, ar=.95, innov=de, burnin=50)`
`  phi.yw[i] = ar.yw(x, order=1)$ar`
`}`

![Finite sample density of a Yule--Walker estimate of an AR(1) regression parameter, the corresponding bootstrap distribution and the corresponding normal approximation](./images/fig8_13.jpg)

Figure 8.13: Finite sample density of the Yule–Walker estimate of _ϕ_ (solid line) and the corresponding asymptotic normal density (dashed line). Bootstrap histogram of ϕ^ based on 500 bootstrapped samples. [Return to text.⏎](chapter8)

The preceding simulation required full knowledge of the model, the parameter values, and the noise distribution. In a sampling situation, we would not have the information necessary to do the simulation and consequently would not be able to simulate from the true distribution. The bootstrap, however, gives us a way to attack the problem.

To perform the bootstrap simulation in this case, we replace the parameters with their estimates μ^ and ϕ^ and calculate the errors

w^t\=(xt−μ^)−ϕ^(xt−1−μ^).t\=2,…,100,

conditioning on _x_1.

To obtain one bootstrap sample, first randomly sample with replacement n\=99 values from the set of estimated errors, {w^2,…,w^100} and call the sampled values

{w2∗,…,w100∗}.

Now, generate a bootstrapped data set recursively by setting

xt∗\=μ^+ϕ^(xt−1∗−μ^)+wt∗,t\=2,…,100,

with x1∗ held fixed at _x_1 and μ^\=44.496, ϕ^\=.966.

227\. Next, estimate the parameters as if the data were xt∗. Call these estimates μ^(1), ϕ^(1), and σ^w2(1). Repeat this process a large number, _B_, of times, generating a collection of bootstrapped parameter estimates, {μ^(b),ϕ^(b),σ^w2(b); b\=1,…,B}. We can then approximate the finite sample distribution of an estimator from the bootstrapped parameter values. For example, we can approximate the distribution of ϕ^−ϕ by the empirical distribution of ϕ^(b)−ϕ^, for b\=1,…,B.

[Figure 8.13](#chapter8#fig8_13) shows the bootstrap histogram of 500 bootstrapped estimates of _ϕ_ using the data shown in [Figure 8.12](#chapter8#fig8_12), which is close to the true distribution of ϕ^. The astsa package has a script to perform the bootstrap for AR models; to use it, simply provide the data and the model order:


`boots = **ar.boot**(dex, order=1, plot=FALSE)   _# default is B = 500_`

The script ar.boot will plot the results for the regression parameters by default. But here, we plot all distributions of ϕ^, the true distribution, the bootstrapped distribution, and the normal approximation (see [Figure 8.13](#chapter8#fig8_13)).


`hist(boots[[1]], main=NA, prob=TRUE, ylim=c(0,15), xlim=c(.65,1.05),`
`   col=**astsa.col**(4,.4), xlab=bquote(hat(phi)))`
`lines(density(phi.yw, bw=.02), lwd=2) _# true distribution_`
`u = seq(.75, 1.1, by=.001)            _# normal approximation_`
`lines(u, dnorm(u, mean=estyw[2], sd=estyw[3]), lty=2, lwd=2)`
`legend(.65, 15, bty="n", lty=c(1,0,2), lwd=c(2,0,2), col=1, pch=c(NA,22,NA),`
`   legend=c("true distribution", "bootstrap distribution", "normal`
`   approximation"), pt.bg=c(NA, **astsa.col**(4,.4), NA), pt.cex=2.5)`

If we want a 100(1−α)% confidence interval, we can use the bootstrap distribution of ϕ^ as follows:


`alf = .025   _# 95% CI_`
`quantile(phi.star.yw, probs=c(alf, 1-alf))`
`       2.5%   97.5%`
`     0.7801 0.9689`

228\. This is close to the actual interval based on the simulation:


`quantile(phi.yw, probs=c(alf, 1-alf))`
`       2.5%   97.5%`
`     0.7707 0.9623`

The normal confidence interval is considerably different:


`qnorm(c(alf, 1-alf), mean=estyw[2], sd=estyw[3])`
` [1]   0.9149 1.0172`

## 8.7 Threshold Autoregressive Models

Stationary normal time series have the property that the distribution of the time series forward in time, x1:n\={x1,x2,…,xn} is the same as the distribution backward in time, xn:1\={xn,xn−1,…,x1}. This follows because the autocorrelation functions of each depend only on the time differences, which are the same for x1:n and xn:1. In this case, a time plot of x1:n (the data plotted forward in time) should look similar to a time plot of xn:1 (the data plotted backward in time).

There are, however, many series that do not fit into this category. For example, [Figure 8.14](#chapter8#fig8_14) shows a plot of monthly pneumonia and influenza deaths per 10,000 in the U.S. for 11 years, 1968 to 1978\. Typically, the number of deaths tends to increase faster than it decreases (↑↘), especially during epidemics. Thus, if the data were plotted backward in time, that series would tend to increase slower than it decreases. Also, if monthly pneumonia and influenza deaths followed a linear Gaussian process, we would not expect to see such large bursts of positive and negative changes that occur periodically in this series. In addition, although the number of deaths is typically largest during the winter months, the data are not perfectly seasonal. According to the U.S. Centers for Disease Control and Prevention (CDC), during the 40-year period of 1982–2022, flu activity most often peaked in February (17 seasons), followed by December (7 seasons), January (6 seasons) and March (6 seasons) \[see [CDC, 2023](#bibref1#refbib_9)\]. That is, although the peak of the series occurs in winter, the month in which it peaks varies from year to year. Hence, seasonal ARMA models would not capture this behavior.

![U.S. monthly pneumonia and influenza deaths per 10,000 from 1968 to 1978](./images/fig8_14.jpg)

Figure 8.14: U.S. monthly pneumonia and influenza deaths per 10,000. [Return to text.⏎](chapter8)

In this section we focus on threshold AR models presented in [Tong (1983)](#bibref1#refbib_48). The basic idea is that of fitting several AR models depending on the value of the process, and the appeal is that we can use the intuition from fitting global AR models. For example, a two-regimes _self-exciting threshold AR_ (SETAR) model 229\. has the form

xt\={ϕ0(1)+∑i\=1p1ϕi(1)xt−i+wt(1)if xt−d≤r,ϕ0(2)+∑i\=1p2ϕi(2)xt−i+wt(2)if xt−d\>r,(8.35)

where wt(j)∼iid N(0,σj2) for j\=1,2, the positive integer _d_ is a _delay_, and _r_ is a real number.

These models allow for changes in the AR coefficients over time, and those changes are determined by comparing previous values (back-shifted by a time lag equal to _d_) to fixed threshold values. Each different AR model is referred to as a _regime_. In the definition above, the values (_pj_) of the order of the AR models can differ in each regime, although in many applications, they are equal.

The model can be generalized to include the possibility that the regimes depend on a collection of the past values of the process, or that the regimes depend on an exogenous variable (in which case the model is not self-exciting) such as in predator-prey cases. For example, Canadian lynx discussed in [Example 1.5](#chapter1#exam1_5) have been thoroughly studied and the series is typically used to demonstrate the fitting of threshold models. Recall that the hare is the lynx's overwhelmingly favored prey and its population rises and falls with that of the hare. In this case, it seems reasonable to replace xt−d in (8.35) with yt−d, where _yt_ is the size of the Snowshoe hare population. For the pneumonia and influenza deaths example, however, a self-exciting model seems appropriate given the nature of the spread of the flu.

The popularity of TAR models is due to their being relatively simple to specify, estimate, and interpret as compared to many other nonlinear time series models. In addition, despite its apparent simplicity, the class of TAR models can 230\. reproduce many nonlinear phenomena. In the following example, we use these methods to fit a threshold model to monthly pneumonia and influenza deaths series previously mentioned.

Example 8.12 Threshold Modeling of the Influenza Series

As previously discussed, examination of [Figure 8.14](#chapter8#fig8_14) leads us to believe that the monthly pneumonia and influenza deaths time series, _flut_, is not linear. It is also evident from [Figure 8.14](#chapter8#fig8_14) that there is a slight negative trend in the data. We have found that the most convenient way to fit a threshold model to these data, while removing the trend, is to work with the first differences,

xt\=∇flut,

which are exhibited as points in [Figure 8.16](#chapter8#fig8_16).

The nonlinearity of the data is more pronounced in the plot of the first differences, _xt_. Clearly _xt_ slowly rises for some months, and then sometime in the winter, has a possibility of jumping to a large number once _xt_ reaches a critical value. If the process does make a large jump, then a subsequent significant decrease occurs in _xt_. Another telling graphic is the lag plot of _xt_ versus xt−1 shown in [Figure 8.15](#chapter8#fig8_15), which suggests the possibility of two linear regimes based on whether or not xt−1 exceeds. 04.


`thr    = 0.04`
`culer = **astsa.col**(c(7,3), .5)`
`culers = ifelse(diff(**flu**)<thr, culer[1], culer[2])`
`**tsplot**(lag(diff(**flu**),-1), diff(**flu**), type="p", xlab=bquote(nabla~**flu**[~t-1]),`
`   ylab=bquote(nabla~**flu**[~t]), pch=21, cex=1.25, bg=culers, xy.lines=FALSE,`
`   xy.labels=FALSE)`
`abline(v=thr, lty=2, col=4)`
`U = ts.intersect(lag(diff(**flu**),-1), diff(**flu**))`
`lines(lowess(U[,1], U[,2]), col=6)`

![Scatterplot of the differenced flu series and the differenced flu series lagged by one month](./images/fig8_15.jpg)

Figure 8.15: Scatterplot of ∇flut versus ∇flut−1 with a lowess fit superimposed (line). The vertical dashed line indicates ∇flut−1\=.04. [Return to text.⏎](chapter8)

![First differenced U.S. monthly pneumonia and influenza deaths, and the predicted values based on a threshold model](./images/fig8_16.jpg)

Figure 8.16: First differenced U.S. monthly pneumonia and influenza deaths (points); one-month-ahead predictions (solid line) with ±2 prediction error bounds. The horizontal dashed line is the threshold. [Return to text.⏎](chapter8)

As an initial analysis, we fit the following threshold model

xt\=α(1)+∑j\=1pϕj(1)xt−j+wt(1),xt−1<.04xt\=α(2)+∑j\=1pϕj(2)xt−j+wt(2),xt−1≥.04,(8.36)

with p\=6, assuming this would be larger than necessary.

231\. An order p\=4 was finally selected and the fit was (with rounding)

x^t\=0+.51(.08)xt−1−.20(.06)xt−2+.12(.05)xt−3−.11(.05)xt−4+w^t(1), for xt−1<.04x^t\=.40−.75(.17)xt−1−1.03(.21)xt−2−2.05(1.05)xt−3−6.71(1.25)xt−4+w^t(2),for xt−1≥.04,

where σ^1\=.05 and σ^2\=.07. The threshold of. 04 was exceeded 17 times. Details are provided in the code below.

Using the final model, one-month-ahead predictions can be made, and these are shown in [Figure 8.16](#chapter8#fig8_16) as a line. The model does extremely well at predicting a flu epidemic. Prediction beyond one-month-ahead, however for these models is complicated, but some approximate techniques exist [(see Tong, 1983)](#bibref1#refbib_48).

We note that there are various R packages that can be used to fit these models; for example, tsdyn and nts. The former package is more general but is a bit quirky in its setup. The latter package will fit the model with two regimes only, which works in our case. Using NTS, the code and output for fitting AR(4) models with two regimes is as follows.232\. 


`library(NTS)       _# load package - install it first_`
`flutar = uTAR(diff(**flu**), p1=4, p2=4)`
`  Estimated Threshold: 0.042594`
` `
`  Regime 1:`
`         Estimate Std. Error     t value     Pr(>|t|)`
`  X1 0.004471044 0.004893995 0.9135776 3.630319e-01`
`  X2 0.506649694 0.078318883 6.4690618 3.198875e-09`
`  X3 -0.200086031 0.056573062 -3.5367722 6.043925e-04`
`  X4 0.121047354 0.054462770 2.2225706 2.838883e-02`
`  X5 -0.110938271 0.045979329 -2.4127858 1.756436e-02`
`  nob1 & sigma1: 110 0.04577968`
`  Regime 2:`
`       Estimate Std. Error   t value     Pr(>|t|)`
`  X1 0.4079353 0.04674982 8.725921 1.528671e-06`
`  X2 -0.7483325 0.16643827 -4.496156 7.315328e-04`
`  X3 -1.0323129 0.21136548 -4.884019 3.759579e-04`
`  X4 -2.0450407 1.05000304 -1.947652 7.523490e-02`
`  X5 -6.7117769 1.24538129 -5.389335 1.628721e-04`
`  nob2 & sigma2: 17 0.07209551`
`**sarima**(resid(flutar)) _# residual analysis (not shown)_`
` `
`_##-- graphic --##_`
`innov = resid(flutar)`
`pred = diff(**flu**)[-(1:4)] - innov`
`pred = ts(pred, start=c(1968,6), freq=12)`
`**tsplot**(diff(**flu**), type="p", ylim=c(-.5,.5), pch=20, col=6, **nym**=2,`
`   ylab=bquote(nabla~**flu**[~t]))`
`lines(pred, col=4, lwd=2)`
`abline(h = flutar$thr, lty=6, col=5)`
`_# error bnds_`
`prde1 = sqrt(sum(resid(flutar$model1)^2)/flutar$model1$df)`
`prde2 = sqrt(sum(resid(flutar$model2)^2)/flutar$model2$df)`
`    x = time(diff(**flu**))[-(1:4)]`
`prde = ifelse(lag(x,-1) < flutar$thr, prde1, prde2)`
`   xx = c(x, rev(x))`
`   yy = c(pred - 2*prde, rev(pred + 2*prde))`
`polygon(xx, yy, border=gray(.6,.5), col=gray(.6,.2))`
`legend("bottomright", legend=c("observed", "predicted"), lty=0:1, pch=c(20,NA),`
`   col=c(6,4), lwd=2)`

233\. 

## Problems

* 8.1. Investigate whether the quarterly growth rate of U.S. GDP (gdp) exhibits GARCH behavior. If so, fit an appropriate model to the growth rate.
* 8.2. Investigate if fitting a non-normal GARCH model to the U.S. GNP data set analyzed in [Example 8.1](#chapter8#exam8_1) improves the fit.
* 8.3. Weekly crude oil spot prices in dollars per barrel are in oil. Investigate whether the growth rate of the weekly oil price exhibits GARCH behavior. If so, fit an appropriate model to the growth rate.
* 8.4. The stats package of R contains the daily closing prices of four major European stock indices; type help(EuStockMarkets) for details. Fit a GARCH model to the returns of one of these series and discuss your findings. (Note: The data set contains actual values, and not returns. Hence, the data must be transformed prior to the model fitting.)
* 8.5. Plot the global (ocean only) temperature series, gtemp\_ocean, and then test whether there is a unit root versus the alternative that the process is stationary using the three tests, DF, ADF, and PP, discussed in [Example 8.5](#chapter8#exam8_5). Comment.
* 8.6. Plot the GNP series, gnp, and then test for a unit root against the alternative that the process is explosive. State your conclusion.
* 8.7. The data in climhyd have 454 months of measured values for the climatic variables air temperature, dew point, cloud cover, wind speed, precipitation, and inflow, at Lake Shasta. Plot the data and fit an ARFIMA model to the wind speed series, climhyd$WndSpd, performing all diagnostics. State your conclusion.
* 8.8. The data set arf is 1000 simulated observations from an ARFIMA(1,1,0) model with ϕ\=.75 and d\=.4.  
   1. Plot the data and comment.  
   2. Plot the ACF and PACF of the data and comment.  
   3. Estimate the parameters and test for the significance of the estimates ϕ^ and d^.  
   4. Explain why, using the results of parts (a) and (b), it would seem reasonable to difference the data prior to the analysis. That is, if _xt_ represents the data, explain why we might choose to fit an ARMA model to ∇xt.  
   5. Plot the ACF and PACF of ∇xt and comment.  
   6. Fit an ARMA model to ∇xt and comment.
* 8.9. 234\. Using [Example 8.10](#chapter8#exam8_10) as a guide, fit a state space model to the Johnson & Johnson earnings in jj. Plot the data with (a) the smoothers, (b) the predictors, and (c) the filters, superimposed each with error bounds (three separate graphs). Compare the results of (a), (b), and (c). In addition, what does the estimated value of _ϕ_ tell you about the growth rate in the earnings?
* 8.10.  
   1. Plot the sample CCF between the cardiovascular mortality and temperature series. Compare it to [Figure 8.9](#chapter8#fig8_9) and discuss the results.  
   2. Redo the cross-correlation analysis of [Example 8.11](#chapter8#exam8_11) but for the cardiovascular mortality and temperature series. State your conclusions.
* 8.11. Repeat the bootstrap analysis of [Section 8.6](#chapter8#sec8_6) but with the asymmetric error distribution of a centered standard log-normal (recall _X_ is log-normal if logX is normal; ?rlnorm). To generate _n_ observations from this distribution, use  
`n = 150   _# desired number of obs_`  
`w = rlnorm(n) - exp(.5)`
* 8.12. Fit a threshold AR model to the lynx series from the R datasets package.
* 8.13. The sunspot data (**sunspotz**) are plotted in [Figure 7.14](#chapter7#fig7_14). From a time plot of the data, discuss why it is reasonable to fit a threshold model to the data, and then fit a threshold model.
