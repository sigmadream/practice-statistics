<a role="toc_link" id="chapter4"></a>
77\. 

# 4ARMA Models

Linear regression models are often unsatisfactory for explaining all of the interesting dynamics of a time series. Instead, the introduction of correlation through lagged relationships leads to autoregressive (AR) and moving average (MA) models. These models are often combined to form autoregressive moving average (ARMA) models, which are the focus of this chapter.

## 4.1 Autoregressive Models

Autoregressive models are an obvious extension of linear regression models. An _autoregressive model_ of order _p_, abbreviated AR(_p_), is of the form

xt\=μ+ϕ1(xt−1−μ)+ϕ2(xt−2−μ)+⋯+ϕp(xt−p−μ)+wt,

where _xt_ is stationary and _wt_ is white noise. The mean function of the AR model is E(xt)\=μ, and for convenience the model is written as

xt\=α+ϕ1xt−1+⋯+ϕpxt−p+wt,

where α\=μ(1−ϕ1−⋯−ϕp).

The model is similar to the regression model of [Section 3.1](#chapter3#sec3_1), and hence the term auto- (or self-) regression. Some technical difficulties develop from applying the model because the regressors, xt−1,…,xt−p, are random components, whereas in regression, the regressors are assumed to be fixed. We will also see that restrictions must be put on the AR parameters as opposed to linear regression where there are typically no parameter restrictions.

Example 4.1 The AR(1) Model and Causality [Return to text.⏎](chapter4)

Consider the first-order, zero-mean AR(1) model,

xt\=ϕxt−1+wt.

Because _xt_ must be stationary, we can rule out the case ϕ\=1 because this would make _xt_ a random walk, which we know is not stationary (recall 78\. [Example 2.14](#chapter2#exam2_14)). Similarly, we can rule out ϕ\=−1. In other words, the models

xt\=xt−1+wt,andxt\=−xt−1+wt,

are _not_ AR models because they are not stationary.

As we saw in [Example 2.20](#chapter2#exam2_20), if _xt_ is stationary, then assuming xt−1 and _wt_ are uncorrelated,

var(xt)\=ϕ2var(xt−1)+var(wt),

which, because var(xt−1)\=var(xt), implies

var(xt)\=γ(0)\=σw2 1(1−ϕ2).

Thus, we must have |ϕ|<1 for the process to have a positive (finite) variance. This result coincides with [Example 2.20](#chapter2#exam2_20) where we showed that _ϕ_ is the correlation between _xt_ and xt−1.

To explore the model further, we can iterate the equation backward _k_ times,

xt\=ϕxt−1+wt\=ϕ(ϕxt−2+wt−1⏟xt−1)+wt\=ϕ2xt−2+ϕwt−1+wt\=ϕ2(ϕxt−3+wt−2⏟xt−2)+ϕwt−1+wt\=ϕ3xt−3+ϕ2wt−2+ϕwt−1+wt⋮\=ϕkxt−k+∑j\=0k−1ϕjwt−j.

If we want to continue indefinitely (k→∞), then for that to make sense, we must have |ϕ|<1, so that ϕk→0 exponentially fast.

Thus, provided that |ϕ|<1 we can represent an AR(1) model as a linear process given by

xt\=∑j\=0∞ϕjwt−j.(4.1)

Representation (4.1) is called the _causal solution_ of the model (see [Appendix 4.7](#chapter4#sec4_7) for details). The term causal refers to the fact that _xt_ does not depend on the future. By simple substitution of (4.1) into xt\=ϕxt−1+wt, we see that

∑j\=0∞ϕjwt−j⏟xt\=ϕ (∑k\=0∞ϕkwt−1−k⏟xt−1)+wt.

79\. As a check, the right-hand side is wt+ϕwt−1 \[k\=0\]+ϕ2wt−2 \[k\=1\]+….

Using (4.1), it is easy to see that the AR(1) process is stationary with mean function

E(xt)\=∑j\=0∞ϕjE(wt−j)\=0,

and autocovariance function (h≥0),

![](./images/ufig4_1.jpg)

(4.2) [Return to text.⏎](chapter4)

Recall that γ(h)\=γ(−h), so we will only exhibit the autocovariance function for h≥0. From ([4.2](#chapter4#ufig4_1)), the ACF of an AR(1) is

ρ(h)\=γ(h)γ(0)\=ϕh,h≥0.(4.3)

In addition, from the causal form (4.1) we see that, as required, xt−1 and _wt_ are uncorrelated because xt−1\=∑j\=0∞ϕjwt−1−j is a linear filter of past noise values, wt−1,wt−2,…, which are uncorrelated with _wt_, the present value. Also, the causal form of the model allows us to easily see that if we replace _xt_ by xt−μ, then

xt\=μ+∑j\=0∞ϕjwt−j,

so that the mean function is E(xt)\=μ.

Example 4.2 The Sample Path of an AR(1) Process

[Figure 4.1](#chapter4#fig4_1) shows a time plot of two AR(1) processes, one with ϕ\=.9 and one with ϕ\=−.9; in both cases, σw2\=1. In the first case, ρ(h)\=.9h, for h≥0, so observations close together in time are positively correlated. Thus, observations at contiguous time points will tend to be close in value to each other; this fact shows up in the top of [Figure 4.1](#chapter4#fig4_1) as a very smooth sample path for _xt_.

![Simulated autoregressive models of order one with positive and negative coefficient](./images/fig4_1.jpg)

Figure 4.1: Simulated AR(1) models:ϕ\=.9 (top); ϕ\=−.9 (bottom). [Return to text.⏎](chapter4)

Now, contrast this with the case in which ϕ\=−.9, so that ρ(h)\=(−.9)h, for h≥0. This result means that observations at contiguous time points are negatively correlated, but observations two time points apart are positively correlated, and so on. This fact shows up in the bottom of [Figure 4.1](#chapter4#fig4_1), where, for example, if an observation, _xt_, is positive, the next observation, xt+1, is 80\. typically negative, and the next observation, xt+2, is typically positive. In this case, the sample path is very choppy.


`par(mfrow=c(2,1))`
`**tsplot**(**sarima.sim**(ar= .9, n=100), main=bquote(AR(1)~~~phi==+.9), ylab="x",`
`   col=4, **gg**=TRUE)`
`**tsplot**(**sarima.sim**(ar=-.9, n=100), main=bquote(AR(1)~~~phi==-.9), ylab="x",`
`   col=4, **gg**=TRUE)`

Example 4.3. AR(_p_) and Causality [Return to text.⏎](chapter4)

In [Example 4.1](#chapter4#exam4_1), we saw that an AR(1) has as a causal representation. For example, xt\=.9xt−1+wt can also be written as xt\=∑j\=0∞.9jwt−j. In the general case, it is more complicated to go from one version to another. It is, however, possible to use the command ARMAtoMA to print some of the coefficients.

For example, the AR(2) model

xt\=1.5xt−1−.75xt−2+wt,

can be written in its causal form, xt\=∑j\=0∞ψjwt−j, where ψ0\=1 and

ψj\=2(32)jcos(2π(j−2)12),j\=1,2,….

The _ψ_\-weights were solved for using difference equation theory [(see Shumway and Stoffer, 2025, §3.2)](#bibref1#refbib_44). Notice that the coefficients are cyclic with a period of 12 (like monthly data), but they decrease exponentially fast to zero (because 3/2<1) indicating a short dependence on the past. [Figure 4.2](#chapter4#fig4_2) shows a plot 81\. of the _ψj_ for j\=1,…,50, as well as simulated data from the model. Both show the cyclic-type behavior of this particular model. In this way, the linear process form of the model,

![Simulated autoregressive models of order two with parameters that generate pseudo-cyclic behavior, and the parameters of the causal representation of that model](./images/fig4_2.jpg)

Figure 4.2: Simulated data and _ψ_\-weights of the AR(2),xt\=1.5xt−1−.75xt−2+wt. [Return to text.⏎](chapter4)

xt\=∑j\=1∞2(32)jcos(2π(j−2)12)wt−j+wt,

gives more insight into the model than the regression form of the model,

xt\=1.5xt−1−.75xt−2+wt.

Finally, we note that an AR(_p_) is also an MA(∞).

Details on how to determine when a model is causal are given in [Section 4.7](#chapter4#sec4_7). For an AR(2), the parameter space can be determined to satisfy the conditions given by

ϕ1+ϕ2<1,ϕ2−ϕ1<1,and|ϕ2|<1.

This causality condition specifies a triangular region; see [Example 4.41](#chapter4#exam4_41) and [Figure 4.9](#chapter4#fig4_9) for details. The following code was used for this example.


`set.seed(8675309)`
`x   = **sarima.sim**(ar=c(1.5,-.75), n=144, S=12)`
`psi = ts(c(1, ARMAtoMA(ar=c(1.5, -.75), ma=0, 50)), start=0, freq=12)`
`par(mfrow=c(2,1))`
`**tsplot**(x, main=bquote(AR(2)~~~phi[1]==1.5~~~phi[2]==-.75), col=4, xaxt="n",`
`   **gg**=TRUE)`
` mtext(seq(0,144,by=12), side=1, at=0:12, cex=.8)`
`**tsplot**(psi, col=4, type="o", ylab=bquote(psi-weights), xaxt="n", xlab="Index",`
`   **gg**=TRUE)`
` mtext(seq(0,48,by=12), side=1, at=0:4, cex=.8)`

82\. We now formally define the concept of causality. The importance of this condition is to make sure that a model is not future-dependent. This allows us to be able to predict future values based on only the present and the past.

Definition 4.4. [Return to text.⏎](chapter4) _A time series xt is said to be **causal** if it can be written as_

xt\=μ+∑j\=0∞ψjwt−j

_for constants ψj satisfying_ ∑j\=0∞ψj2<∞.

**Remarks.** As stated in [Property 2.21](#chapter2#prop2_21), any stationary (non-deterministic) time series has a causal representation.

## 4.2 Moving Average Models

The _moving average model_ of order _q_, or MA(_q_), is defined by

xt\=μ+wt+θ1wt−1+θ2wt−2+⋯+θqwt−q,

where _wt_ is white noise.[1](#chapter4#fn4_1) Unlike the autoregressive process, the moving average process is stationary for any values of the parameters θ1,…,θq, and E(xt)\=μ. In addition, the MA(_q_) is already in the causal form of [Definition 4.4](#chapter4#defi4_4) with ψj\=θj and θj\=0 for j\>q.

Example 4.5 The MA(1) Process [Return to text.⏎](chapter4)

Consider the MA(1) model xt\=wt+θwt−1. Think of the noise, _wt_, as a random “shock” to the process at time _t_. One can imagine that what happens today might also be related to the shock from yesterday.

We have E(xt)\=0, and if we replace _xt_ by xt−μ, then E(xt)\=μ. The autocovariance function is

γ(h)\={(1+θ2)σw2h\=0,θσw2|h|\=1,0|h|\>1,

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 1Some texts and software packages write the MA model with negative coefficients; that is, xt\=μ+wt−θ1wt−1−θ2wt−2−⋯−θqwt−q. R uses positive coefficients, but check first when using other software. [Return to text.⏎](#chapter4#fn4_14b)

83\. and the ACF is

ρ(h)\={θ(1+θ2)|h|\=1,0|h|\>1.

Note |ρ(1)|≤1/2 for all values of _θ_ (Problem 4.1). Also, _xt_ is correlated with xt−1, but not with xt−2,xt−3,…. Contrast this with the case of the AR(1) model in which the correlation between _xt_ and xt−k is never zero. When θ\=.9, for example, _xt_ and xt−1 are positively correlated, and ρ(1)\=.497. When θ\=−.9, _xt_ and xt−1 are negatively correlated, ρ(1)\=−.497. [Figure 4.3](#chapter4#fig4_3) shows a time plot of these two processes with σw2\=1. The series for which θ\=.9 is smoother than the series for which θ\=−.9.


`par(mfrow = c(2,1))`
`**tsplot**(**sarima.sim**(ma= .9, n=100), main=bquote(MA(1)~~~theta==+.9), col=4,`
`   ylab="x", **gg**=TRUE)`
`**tsplot**(**sarima.sim**(ma=-.9, n=100), main=bquote(MA(1)~~~theta==-.9), col=4,`
`   ylab="x", **gg**=TRUE)`

![Simulated moving average models of order one with positive and negative coefficient](./images/fig4_3.jpg)

Figure 4.3: Simulated MA(1) models:θ\=.9 (top); θ\=−.9 (bottom). [Return to text.⏎](chapter4)

Example 4.6 Non-uniqueness of MA Models and Invertibility

Using [Example 4.5](#chapter4#exam4_5), we note that for an MA(1) model, the pair σw2\=1 and θ\=5 yield the same autocovariance function as the pair σw2\=25 and θ\=1/5, namely,

γ(h)\={26h\=0,5|h|\=1,0|h|\>1.

84\. Thus, the MA(1) processes

xt\=wt+15wt−1,wt∼iid N(0,25)

and

yt\=vt+5vt−1,vt∼iid N(0,1)

are stochastically the same. We can only observe the time series, _xt_ or _yt_, and not the noise, _wt_ or _vt_, so we cannot distinguish between the models. Hence, we will have to choose only one of them. For convenience, we choose the model with an infinite AR representation. Such a process is called _invertible_.

To discover which model is the invertible model, we can reverse the roles of _xt_ and _wt_ and write the MA(1) model as

wt\=−θwt−1+xt.

As in (4.1), if |θ|<1, then

wt\=∑j\=0∞(−θ)jxt−j,

which is the desired infinite representation of the model. Hence, given a choice, we will choose the model with σw2\=25 and θ\=1/5 because it is invertible.

Henceforth, for uniqueness, we require that a moving average have an _invertible_ representation:

Definition 4.7. [Return to text.⏎](chapter4) _A time series xt is said to be **invertible** if it can be written as_

wt\=∑j\=0∞πjxt−j.

_for constants πj satisfying_ ∑j\=0∞πj2<∞, _where_ π0\=1.

**Remarks.** Aside from the uniqueness problem, invertibility is important because it gives a representation of a present shock, _wt_, in terms of the present and past data. Consequently, the current shock to the system does not depend on the future so that we may estimate it given the data. Also, note that an MA(_q_) is an AR(∞) because we may write the model as

xt\=−∑j\=1∞πjxt−j+wt.

## 4.3 85\. Autoregressive Moving Average Models

We now proceed with the general development of mixed _autoregressive moving average_ (ARMA) models for stationary time series.

Definition 4.8. _A time series_ {xt} _is **ARMA(p,q)** if_

xt\=α+ϕ1xt−1+⋯+ϕpxt−p+wt+θ1wt−1+⋯+θqwt−q,

_where_ wt∼wn(0,σw2), ϕp≠0, θq≠0, _and the model is causal and invertible. If_ E(xt)\=μ, _then_ α\=μ(1−ϕ1−⋯−ϕp).

The ARMA model may be seen as a regression of the present, _xt_, on the past, xt−1,…,xt−p, with correlated errors. That is,

xt\=β0+β1xt−1+⋯+βpxt−p+ϵt,

where

ϵt\=wt+θ1wt−1+⋯+θqwt−q,

although we call the regression parameters _ϕ_ instead of _β_. As opposed to ordinary regression, the _ϕ_ parameters are restricted to certain values to obtain causality, and the _θ_ parameters are restricted to certain values to obtain invertibility.

To better understand the model, we first establish some notation based on the backshift operator defined in [Definition 3.10](#chapter3#defi3_10), Bkxt\=xt−k. Using the backshift operator, we can write the zero-mean AR(_p_) model as

(1−ϕ1B−ϕ2B2−⋯−ϕpBp)xt\=wt.

Thus, it is convenient to define the **autoregressive operator** as

ϕ(B)\=1−ϕ1B−ϕ2B2−⋯−ϕpBp.

so that the AR model is

ϕ(B)xt\=wt.

As in the AR(_p_) case, the zero-mean MA(_q_) model may be written as

xt\=(1+θ1B+θ2B2+⋯+θqBq)wt,

so we define the **moving average operator** as

θ(B)\=1+θ1B+θ2B2+⋯+θqBq

and write an MA(_q_) model as

xt\=θ(B)wt.

86\. Consequently, the general ARMA(p,q) model can be written concisely as

ϕ(B)(xt−μ)\=θ(B)wt,(4.4)

where the orders of ϕ(B) and θ(B) are understood to be _p_ and _q_, respectively.

The form (4.4) points to a problem where we can unnecessarily complicate an ARMA(_p,q_) model by multiplying both sides by another operator,

η(B)ϕ(B)(xt−μ)\=η(B)θ(B)wt,

without changing the dynamics. This is called _parameter redundancy_.

Example 4.9 Parameter Redundancy [Return to text.⏎](chapter4)

Consider a white noise process xt\=wt. Now multiply both sides of the equation by (1−.9B),

(1−.9B)xt\=(1−.9B)wt,

or

xt−.9xt−1\=wt−.9wt−1,

or

xt\=.9xt−1−.9wt−1+wt,(4.5)

which looks like an ARMA(1,1) model. Of course, _xt_ is still white noise; nothing has changed in this regard \[i.e., xt\=wt is the solution to (4.5)\]. But, we have hidden the fact that _xt_ is white noise because of the parameter redundancy or overparameterization.

[Example 4.9](#chapter4#exam4_9) points out the need to be careful when fitting ARMA models to data. Unfortunately, it is easy to fit an overly complex ARMA model to data. For example, if a process is truly white noise, it is possible to fit a significant ARMA(_k,k_) model to the data. Consider the following example.

Example 4.10 Parameter Redundancy and Estimation [Return to text.⏎](chapter4)

Although we have not discussed estimation yet, we present the following demonstration of the problem. We generated 150 iid normals with μ\=5 and σ\=1, and then fit an ARMA(p\=1,q\=1) to the data.


`set.seed(8675309)                    _# Jenny, I got your number_`
`x = rnorm(150, mean=5)               _# generate iid N(5,1)s_`
`**sarima**(x, p=1, q=1, **details**=FALSE)   _# estimation_`
`        Estimate    SE t.value p.value`
`  ar1     -0.960 0.169 -5.685        0`
`  ma1      0.953 0.175   5.444       0`
`  xmean    5.046 0.073 69.391        0`
` sigma^2 estimated as 0.799 on 147 degrees of freedom`

87\. Thus the estimated model looks like

(1+.960B)(x^t−5.046)\=(1+.953B)w^t,

so that ϕ^(B)≈θ^(B). Of course the data are white noise, but the estimation implies a seemingly different result that the data are highly dependent.

This example points out the problem of relying on computational methods without knowing some basic theory: Software will give an answer, but it does not know if you are asking the right question.

Henceforth, we will require an ARMA model to be reduced to its simplest form. A simple way to discover if this problem exists with a model is to write the model with the AR part on the left and the MA part on the right, and then compare each side.

Example 4.11 Checking for Parameter Redundancy [Return to text.⏎](chapter4)

In the previous example, it was easy to see that the left-hand and right-hand sides are nearly the same. For more complicated models, we can use R to compare each side. For example, consider the model

xt\=.3xt−1+.4xt−2+wt+.5wt−1,

which looks like an ARMA(2,1). Now write the model as

(1−.3B−.4B2)xt\=(1+.5B)wt,

or

(1+.5B)(1−.8B)xt\=(1+.5B)wt.

We can cancel (1+.5B) on each side, so the model is really an AR(1),

xt\=.8xt−1+wt.

These situations can be checked easily by looking at the roots (zeros) of the polynomials in _B_ corresponding to each side. If the roots are close, then there may be parameter redundancy:


`AR = c(1, -.3, -.4) _# original AR coefs on the left_`
`polyroot(AR)`
` [1] 1.25-0i -2.00+0i`
`MA = c(1, .5)       _# original MA coefs on the right_`
`polyroot(MA)`
` [1] -2+0i`

This indicates there is one common factor (with root −2), and hence the model is overparameterized and can be reduced.

Example 4.12 88\. Causal and Invertible ARMA [Return to text.⏎](chapter4)

It might be useful at times to write an ARMA model in its causal or invertible forms. For example, consider the model

xt\=.8xt−1+wt−.5wt−1.

We can list some of the causal and invertible coefficients of our ARMA(1,1) model as follows:


`round( ARMAtoMA(ar=.8, ma=-.5, 10),   2)   _# first 10 ψ-weights_`
`  [1] 0.30 0.24 0.19 0.15 0.12 0.10   0.08 0.06 0.05 0.04`
`round( **ARMAtoAR**(ar=.8, ma=-.5, 10),   2) _# first 10 π-weights_`
`  [1] -0.30 -0.15 -0.08 -0.04 -0.02   -0.01 0.00 0.00 0.00 0.00`

Thus, the causal form looks like (ψ0\=1),

xt\=wt+.3wt−1+.24wt−2+.19wt−3+⋯+.05wt−9+.04wt−10+⋯,

whereas the invertible form looks like (π0\=1),

wt\=xt−.3xt−1−.15xt−2−.08xt−3−.04xt−4−.02xt−5−.01xt−6+⋯.

If a model is not causal or invertible, the scripts will work, but the coefficients will not converge to zero. For a random walk, xt\=xt−1+wt, or xt\=∑j\=1twj, for example:


`ARMAtoMA(ar=1, ma=0, 20)`
` [1] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1`

Example 4.13 Causal and Invertible ARMA (cont.)\*

For a way to see how causal and invertible forms are determined, consider the model in [Example 4.12](#chapter4#exam4_12),

(1−.8B)xt\=(1−.5B)wt.(4.6)

Treating _B_ as a number, we can divide both sides by (1−.8B) to obtain the causal form

xt\=(1−.5B)(1−.8B)wt.

Now expand the AR part as a power series 1/(1−.8B)\=∑j\=0∞(.8B)j so that

xt\=(1−.5B)(1+.8B+.82B2+.83B3+…)wt.

Collecting terms in _B_, we see that (4.6) can be written as

xt\=wt+.3∑j\=1∞.8j−1wt−j.

89\. Thus, in the previous example, **ARMAtoMA**(ar=.8, ma=-.5, 10) is simply returning .3\*.8^(0:9). The invertible form is determined similarly as

(1−.8B)(1−.5B)xt\=wt,

or

wt\=xt−.3∑j\=1∞.5j−1xt−j.

Thus, in the previous example, **ARMAtoAR**(ar=.8, ma=-.5, 10) is simply returning \-.3\*.5^(0 :9).

## 4.4 Correlation Functions

In this section we discuss the autocorrelation functions of ARMA models and introduce the concept of partial autocorrelation.

### Autocorrelation Function (acf)

Example 4.14 ACF of an MA(_q_) [Return to text.⏎](chapter4)

Write the model as xt\=∑j\=0qθjwt−j with θ0\=1. Because _xt_ is a finite linear combination of white noise terms, the process is stationary with autocovariance function

γ(h)\=cov(xt+h,xt)\=cov(∑j\=0qθjwt+h−j, ∑k\=0qθkwt−k)\={σw2∑j\=0q−hθjθj+h,0≤h≤q0h\>q,(4.7)

which is similar to the calculation in (2.7). The cutting off of γ(h) after _q_ lags is the signature of the MA(_q_) model. Dividing (4.7) by γ(0) yields the ACF of an MA(_q_):

ρ(h)\={∑j\=0q−hθjθj+h1+θ12+⋯+θq21≤h≤q0h\>q.(4.8)

In addition, we note that ρ(q)≠0 because when h\=q, the numerator in (4.8) is θ0 θq\=θq≠0 (recall θ0\=1).

Example 4.15 90\. ACF of an AR(_p_) and ARMA(_p,q_)

For an AR(_p_) or ARMA(_p,q_) model, write the model in its causal MA(∞) form,

xt\=∑j\=0∞ψjwt−j.

It follows immediately that the autocovariance function of _xt_ can be written as

γ(h)\=cov(xt+h,xt)\=σw2∑j\=0∞ψj+hψj,h≥0,

as was calculated in (2.7). The ACF is given by

ρ(h)\=γ(h)γ(0)\=∑j\=0∞ψj+hψj∑j\=0∞ψj2,h≥0.(4.9)

Unlike the MA(_q_), the ACF of an AR(_p_) or an ARMA(_p,q_) does not cut off at any lag, so using the ACF to help identify the order of an AR or ARMA is difficult.

Result (4.9) is not appealing in that it provides little information about the appearance of the ACF of various models. We can, however, determine what happens for some specific models.

Example 4.16 ACF of an AR(2) [Return to text.⏎](chapter4)

[Figure 4.2](#chapter4#fig4_2) shows n\=144 observations from the AR(2) model

xt\=1.5xt−1−.75xt−2+wt,

with σw2\=1. We examined this model in [Example 4.3](#chapter4#exam4_3) where we noted that the process exhibits pseudo-cyclic behavior at the rate of one cycle every 12 time points. Because the _ψ_\-weights are cyclic, the ACF of the model will also be cyclic with a period of 12\. The code to calculate and plot the first 50 values of the ACF for this model is as follows:


`plot(ARMAacf(ar=c(1.5,-.75), lag.max=50), type="h"); abline(h=0, col=8)`

We discuss this further in [Example 4.19](#chapter4#exam4_19) where we display the ACF on the left side of [Figure 4.4](#chapter4#fig4_4).

![Theoretical PACF and ACF of an autoregressive model of order two](./images/fig4_4.jpg)

Figure 4.4: The P/ACF of an AR(2) model with ϕ1\=1.5 and ϕ2\=−.75. [Return to text.⏎](chapter4)

The general behavior of the ACF of an AR(_p_) or an ARMA(_p,q_) is controlled by the AR part because the MA part has only finite influence.

Example 4.17 91\. ACF of an ARMA(1,1)

Consider the ARMA(1,1) process xt\=ϕxt−1+θwt−1+wt. Using the theory of difference equations, we can show that the ACF is given by

ρ(h)\=(1+θϕ)(ϕ+θ)ϕ(1+2θϕ+θ2) ϕh,h≥1.

Notice that the general pattern of ρ(h) is not different from that of an AR(1) given in (4.3), ρ(h)\=ϕh. Hence, it is unlikely that we will be able to tell the difference between an ARMA(1,1) and an AR(1) based solely on an ACF estimated from a sample (see [Problem 4.4](#chapter4#question4_4)). This consideration will lead us to the partial autocorrelation function.

### Partial Autocorrelation Function (pacf)

In [Example 4.14](#chapter4#exam4_14) we saw that for MA(_q_) models, the ACF will be zero for lags greater than _q_. Moreover, because θq≠0, the ACF will not be zero at lag _q_. Thus, the ACF provides a considerable amount of information about the order of the dependence when the process is a moving average process.

If the process, however, is ARMA or AR, the ACF alone tells us little about the orders of dependence. Hence, it is worthwhile pursuing a function that will behave like the ACF of MA models, but for AR models, namely, the _partial autocorrelation function (PACF)_.

Recall that if _X_, _Y_, and Z\={Z1,…,Zk} are random variables, then the partial correlation between _X_ and _Y_ given _Z_ is obtained by regressing _X_ on _Z_ to obtain the predictor X^, regressing _Y_ on _Z_ to obtain Y^, and then calculating

ρXY|Z\=corr{X−X^, Y−Y^}.

The idea is that ρXY|Z measures the correlation between _X_ and _Y_ with the linear effect of Z1,…,Zk removed (or partialled out). If the variables are multivariate normal, then this definition coincides with ρXY|Z\=corr(X,Y∣Z).

To motivate the idea of partial _auto_correlation, consider a causal AR(1) model, xt\=ϕxt−1+wt. Then,

γx(2)\=cov(xt,xt−2)\=cov(ϕxt−1+wt,xt−2)\=cov(ϕxt−1,xt−2)\=ϕγx(1),

where γx(h) is given in ([4.2](#chapter4#ufig4_1)). Note that cov(wt,xt−2)\=0 from causality because xt−2 involves {wt−2, wt−3,…}, which are all uncorrelated with _wt_. The correlation between _xt_ and xt−2 is not zero as it would be for an MA(1) because _xt_ is dependent on xt−2 through xt−1.

92\. Suppose we break this chain of dependence by removing the linear effect of xt−1 on _xt_ and on xt−2. That is, we find the coefficients _a_ and _b_ that minimize the mean squared errors

E(xt−axt−1)2andE(xt−2−bxt−1)2.

Taking derivatives with respect to _a_ and _b_ and setting the results equal to zero,

E\[(xt−axt−1)xt−1\]\=0andE\[(xt−2−bxt−1)xt−1\]\=0,

or

γx(1)−aγx(0)\=0andγx(1)−bγx(0)\=0,

noting γx(−1)\=γx(1), so that

a\=b\=γx(1)/γx(0)\=ρx(1)\=ϕ

for an AR(1) \[recall (4.3)\].

In this way, we have broken the dependence chain between _xt_ and xt−2,

cov(xt−ϕxt−1,xt−2−ϕxt−1)\=cov(wt,xt−2−ϕxt−1)\=0,

by causality (xt−2 and xt−1 depend only on wt−1,wt−2,…, which are uncorrelated with _wt_). Hence, the tool we need is partial autocorrelation, which is the correlation between _xs_ and _xt_ with the linear effect of everything “in the middle” removed.

Definition 4.18. _The **partial autocorrelation function (PACF)** of a stationary process, xt, denoted_ ϕhh, _for_ h\=1,2,…, _is_

ϕ11\=corr(x1,x0)\=ρ(1)

_and_

ϕhh\=corr(xh−x^h, x0−x^0),h≥2,

_where_ x^0 _and_ x^h _are the regressions of_ x0 _and of xh on_ {x1,x2,…,xh−1}.

Thus, due to the stationarity, for h\>1, the PACF, ϕhh, is the correlation between xt+h and _xt_ with the linear dependence of everything between them, namely {xt+1,…,xt+h−1}, on each, removed.

It is not necessary to actually run regressions to estimate the PACF because the values can be computed recursively using the Durbin–Levinson algorithm [(e.g., see Shumway and Stoffer, 2025, ch. 3)](#bibref1#refbib_44).

Example 4.19 PACF of an AR(_p_) [Return to text.⏎](chapter4)

The PACF of an AR(_p_) model will be zero for all lags larger than _p_, and the PACF at lag _p_ will not be zero because it can be shown that ϕpp\=ϕp (the last parameter in the model).

93\. In [Example 4.16](#chapter4#exam4_16) we looked at the AR(2) model

xt\=1.5xt−1−.75xt−2+wt.

In this case, ϕ11\=ρ(1)\=ϕ1/(1−ϕ2)\=1.5/1.75≈.86, ϕ22\=ϕ2\=−.75, and ϕhh\=0 for h\>2. [Figure 4.4](#chapter4#fig4_4) shows the ACF and the PACF of this AR(2) model.


`ACF = ARMAacf(ar=c(1.5,-.75), ma=0, 24)[-1]`
`PACF = ARMAacf(ar=c(1.5,-.75), ma=0, 24, pacf=TRUE)`
`par(mfrow=1:2)`
`**tsplot**(ACF, type="h", xlab="LAG", ylim=c(-.8,1), col=4, las=1, **gg**=TRUE)`
`abline(h=0, col=8)`
`**tsplot**(PACF, type="h", xlab="LAG", ylim=c(-.8,1), col=4, las=1, **gg**=TRUE)`
`abline(h=0, col=8)`

Example 4.20 PACF of an MA(_q_)

Recall that an invertible MA(_q_) model has an AR(∞) representation,

xt\=−∑j\=1∞πjxt−j+wt.

Moreover, no finite representation exists. From this result, it should be apparent that the PACF will never cut off as is the case for an AR model. For an MA(1), xt\=wt+θwt−1, with |θ|<1, it can be shown that

ϕhh\=−(−θ)h(1−θ2)1−θ2(h+1),h≥1.

We also have the following large sample result for the PACF, which may be compared to the similar result for the ACF given in [Property 2.28](#chapter2#prop2_28).94\. 

Property 4.21 (PACF – Large Sample Distribution). _If a time series is an AR(p) and the sample size n is large, then the sample partial autocorrelations are approximately independent normals,_

ϕ^hh∼⋅N(0,1n)forh\>p.

_This result also holds for_ p\=0, _wherein the process is white noise._

Thus, the sample PACF is typically plotted with ±2/n bounds.

The PACF for MA models behaves much like the ACF for AR models. Also, the PACF for AR models behaves much like the ACF for MA models. Because an invertible ARMA model has an infinite AR representation, the PACF will not cut off. We summarize these results in [Table 4.1](#chapter4#tbl4_1).

__Table 4.1: Behavior of the ACF and PACF for ARMA Models [Return to text.⏎](chapter4)__
| AR(_p_) | MA(_q_)                | ARMA(_p,q_)            |           |
| ------- | ---------------------- | ---------------------- | --------- |
| ACF     | Tails off              | Cuts off after lag _q_ | Tails off |
| PACF    | Cuts off after lag _p_ | Tails off              | Tails off |

Example 4.22 Preliminary Analysis of the Recruitment Series [Return to text.⏎](chapter4)

We consider the problem of modeling the Recruitment series shown in [Figure 1.5](#chapter1#fig1_5). There are 453 months of observations ranging over the years 1950–1987\. The sample ACF and PACF displayed in [Figure 4.5](#chapter4#fig4_5) are consistent with the behavior of an AR(2). The ACF has cycles corresponding roughly to a 12-month period, and the PACF has large values for h\=1,2 and then is essentially zero for higher-order lags. Based on [Table 4.1](#chapter4#tbl4_1), these results suggest that a second-order (p\=2) autoregressive model might provide a good fit. Although we will discuss estimation in detail in [Section 4.5](#chapter4#sec4_5), we ran a regression (OLS) using the data triplets {(x; z1,z2):(x3; x2,x1),(x4; x3,x2),…,(x453; x452,x451)} to fit the model

xt\=ϕ0+ϕ1xt−1+ϕ2xt−2+wt

![Sample ACF and PACF of the fish Recruitment series displaying the behavior of an autoregression of order two](./images/fig4_5.jpg)

Figure 4.5: Sample ACF and PACF of the Recruitment series. Note that the lag axes are in terms of season (12 months in this case). [Return to text.⏎](chapter4)

for t\=3,4,…,453. The values of the estimates were ϕ^0\=6.74(1.11), ϕ^1\=1.35(.04), ϕ^2\=−.46(.04), and σ^w2\=89.72, where the estimated standard errors are in parentheses.

The following code can be used for this analysis. We use the script acf2 from astsa to print and plot the sample ACF and PACF.95\. 


`**acf2**(**rec**, 48, col=4)      _# will produce values and a graphic_`
`(regr = ar.ols(**rec**, order=2, demean=FALSE, intercept=TRUE))`
`   Coefficients:`
`        1        2`
`   1.3541 -0.4632`
` Intercept: 6.737 (1.111)`
` sigma^2 estimated as 89.72`
`regr$asy.se.coef$ar _# standard errors of the estimates_`
`  [1] 0.04178901 0.04187942`

We could have used lm() to do the regression (with some care); however ar.ols() is much easier to use.

## 4.5 Estimation

Throughout this section, we assume we have _n_ observations, x1,…,xn, from a normal ARMA(_p,q_) process in which, initially, the order parameters, _p_ and _q_, are known. Our goal is to estimate the parameters, _μ_, ϕ1,…,ϕp, θ1,…,θq, and σw2.

We begin with _method of moments_ estimators. The idea behind these estimators is that of equating population moments, E(xtk), to sample moments, 1n∑t\=1nxtk, for k\=1,2,…, and then solving for the parameters in terms of the sample moments. We immediately see that if E(xt)\=μ, the method of moments estimator of _μ_ is the sample average, x¯ (k\=1). Thus, while discussing method of moments, we can assume μ\=0 for notational convenience. Although the 96\. method of moments can produce good estimators, they can sometimes lead to suboptimal estimators. We first consider the case in which the method leads to optimal (efficient) estimators, that is, AR(_p_) models,

xt\=ϕ1xt−1+⋯+ϕpxt−p+wt.

If we multiply each side of the AR equation by xt−h for h\=0,1,…,p, take expectations and divide by γ(0) when h\>0, we obtain the following result.

Definition 4.23. [Return to text.⏎](chapter4) _The **Yule–Walker equations** are given by_

ρ(h)\=ϕ1ρ(h−1)+⋯+ϕpρ(h−p),h\=1,2,…,p,σw2\=γ(0) \[1−ϕ1ρ(1)−⋯−ϕpρ(p)\].

The estimators obtained by replacing γ(0) with its estimate, γ^(0) and ρ(h) with its estimate, ρ^(h), are called the _Yule–Walker estimators_. For AR(_p_) models, if the sample size is large, the Yule–Walker estimators are approximately normally distributed, and σ^w2 is close to the true value of σw2.

Example 4.24 Yule–Walker Estimation for an AR(1) [Return to text.⏎](chapter4)

For an AR(1), (xt−μ)\=ϕ(xt−1−μ)+wt, the mean estimate is μ^\=x¯, and the first equation in [Definition 4.23](#chapter4#defi4_23) is

ρ(1)\=ϕρ(0)\=ϕ,

so

ϕ^\=ρ^(1)\=∑t\=1n−1(xt+1−x¯)(xt−x¯)∑t\=1n(xt−x¯)2,

as might be expected. The estimate of the error variance is then

σ^w2\=γ^(0) \[1−ϕ^2\]

recall γ(0)\=σw2/(1−ϕ2) from ([4.2](#chapter4#ufig4_1)) so the estimate makes sense.

Example 4.25 Yule–Walker Estimation of the Recruitment Series [Return to text.⏎](chapter4)

In [Example 4.22](#chapter4#exam4_22) we fit an AR(2) model to the Recruitment series using regression. Now we use Yule–Walker estimation:


`rec.yw = ar.yw(**rec**, order=2)`
`rec.yw$x.mean    _# mean estimate_`
` [1] 62.26278`
`rec.yw$ar        _# phi1 and phi2 estimates_`
` [1] 1.3315874 -0.4445447`
`sqrt(diag(rec.yw$asy.var.coef)) _# their standard errors_`
` [1] 0.04222637 0.04222637`
`rec.yw$var.pred _# error variance estimate_`
` [1] 94.79912`

97\. The estimates are close to the regression values in [Example 4.22](#chapter4#exam4_22) because in general, Yule-Walker estimation is close OLS estimation for AR models.

In the case of AR(_p_) models, the Yule–Walker estimators are optimal estimators, but this is not true for MA(_q_) or ARMA(_p,q_) models. AR(_p_) models are basically linear models, and the Yule–Walker estimators are essentially least squares estimators. MA or ARMA models are nonlinear models, so this technique does not give optimal estimators.

Example 4.26 Method of Moments Estimation for an MA(1) [Return to text.⏎](chapter4)

Consider the MA(1) model, xt\=wt+θwt−1, where |θ|<1. The model can then be written as

xt\=−∑j\=1∞(−θ)jxt−j+wt,(4.10)

which is nonlinear in _θ_. The first two population autocovariances are γ(0)\=σw2(1+θ2) and γ(1)\=σw2θ, so the estimate of _θ_ is found by solving

ρ^(1)\=γ^(1)γ^(0)\=θ^1+θ^2

for θ^.

Two solutions exist, and we would pick the invertible one. If |ρ^(1)|≤12, the solutions are real; otherwise, a real solution does not exist. Even though |ρ(1)|<12 for an invertible MA(1), it may happen that |ρ^(1)|≥12 because it is an estimator. For example, the following simulation with n\=100 produces a value of ρ^(1)\=.55 when the true value is ρ(1)\=.9/(1+.92)\=.497.


`set.seed(1)`
`ma1 = **sarima.sim**(ma = 0.9, n = 100)`
`**acf1**(ma1, plot=FALSE)[1]`
` [1] 0.55`

In fact, at this sample size and value of _θ_, the probability that |ρ^(1)|≥12 is about 38%. Here is a simulation:


`_# generate 10000 MA(1)s and calculate the first sample ACF_`
`r = replicate(10^4, **acf1**(**sarima.sim**(ma=.9, n=100), max.lag=1, plot=FALSE))`
`mean(abs(r) >= .5)   _# .5 exceedance prob_`
` [1] 0.38`

The preferred method of estimation is maximum likelihood estimation (MLE), which determines the values of the parameters that are most _likely_ to have produced the observations. A review of MLE is given in [Section A.7](#appA#secA_7). The case of an AR(1) will be discussed after we present conditional least squares estimation. For normal models and large sample sizes, MLE and conditional least squares are equivalent.

98\. 

### Conditional Least Squares

Recall from [Chapter 3](#chapter3) that in simple linear regression, xt\=β0+β1zt+wt, we minimize

S(β)\=∑t\=1nwt2(β)\=∑t\=1n(xt−\[β0+β1zt\])2

with respect to the _β_s. This is a simple problem because we have all the data pairs, (xt,zt) for t\=1,…,n. For ARMA models, we do not have this luxury.

Consider a simple AR(1) model, xt\=ϕxt−1+wt. In this case, the error sum of squares is

S(ϕ)\=∑t\=1nwt2(ϕ)\=∑t\=1n(xt−ϕxt−1)2.

We have a problem because we do not observe _x_0. We can make life easier by forgetting the problem and conditioning on _x_1. That is, let's perform least squares using the (conditional) sum of squares,

Sc(ϕ)\=∑t\=2nwt2(ϕ)\=∑t\=2n(xt−ϕxt−1)2

because that's easy (it's just OLS) and if _n_ is large, it shouldn't matter much. We know from regression that the solution is

ϕ^\=∑t\=2nxtxt−1∑t\=2nxt−12,

which is nearly the Yule–Walker estimate in [Example 4.24](#chapter4#exam4_24) (replace _xt_ by xt−x¯ if the mean is not zero).

Now we focus on conditional least squares for ARMA(_p,q_) models via _Gauss–Newton_. Write the model parameters as β\=(ϕ1,…,ϕp,θ1,…,θq), and for the ease of discussion, we will put μ\=0. Write the ARMA model in terms of the errors

wt(β)\=xt−∑j\=1pϕjxt−j−∑k\=1qθkwt−k(β),(4.11)

emphasizing the dependence of the errors on the parameters (recall that wt\=∑j\=0∞πjxt−j by invertibility, and the _πj_ are complicated functions of _β_).

Again we have the problem that we don't observe the _xt_ for t≤0, nor the errors _wt_. For conditional least squares, we condition on x1,…,xp (if p\>0) and set wt\=0 for t≤p, in which case, given _β_, we may evaluate (4.11) for t\=p+1,…,n. For example, for an ARMA(1,1),

xt\=ϕxt−1+θwt−1+wt,

99\. we would start at p+1\=2 and set w1\=0 so that

w2\=x2−ϕx1−θw1\=x2−ϕx1w3\=x3−ϕx2−θw2 ⋮wn\=xn−ϕxn−1−θwn−1

Given data, we can evaluate these errors at any values of the parameters, e.g., at ϕ\=θ\=0, the _wt_ are just the data _xt_ for t\>1. Using this conditioning argument, the conditional error sum of squares is

Sc(β)\=∑t\=p+1nwt2(β).(4.12)

Minimizing Sc(β) with respect to _β_ yields the conditional least squares estimates. We could use a brute-force method where we evaluate Sc(β) over a grid of possible values for the parameters and choose the values with the smallest error sum of squares, but this method becomes prohibitive if there are many parameters.

If q\=0, the problem is linear regression as we saw in the case of the AR(1). If q\>0, the problem becomes nonlinear regression, and we will rely on numerical optimization. Gauss–Newton is an iterative method for solving the problem of minimizing (4.12). We demonstrate the method for an MA(1).

Example 4.27 Gauss–Newton for an MA(1) [Return to text.⏎](chapter4)

Consider an MA(1) process, xt\=wt+θwt−1. Write the errors as

wt(θ)\=xt−θwt−1(θ),t\=1,…,n,(4.13)

where we condition on w0(θ)\=0. Our goal is to find the value of _θ_ that minimizes Sc(θ)\=∑t\=1nwt2(θ), which is a nonlinear function of _θ_ \[recall (4.10)\].

Let θ(0) be an initial estimate of _θ_, for example the method of moments estimate. Now use a first-order Taylor approximation (see [Section A.10](#appA#secA_10)) of wt(θ) at θ(0) to get

Sc(θ)\=∑t\=1nwt2(θ)≈∑t\=1n\[wt(θ(0))−(θ−θ(0))zt(θ(0))\]2,(4.14)

where

zt(θ(0))\=−∂wt(θ)∂θ|θ\=θ(0),

(writing the derivative in the negative simplifies the algebra at the end). It turns out that the derivatives have a simple form that makes them easy to evaluate. 100\. Taking derivatives in (4.13),

∂wt(θ)∂θ\=−wt−1(θ)−θ∂wt−1(θ)∂θ,t\=1,…,n,(4.15)

where we set ∂w0(θ)/∂θ\=0. We can also write (4.15) as

zt(θ)\=wt−1(θ)−θzt−1(θ),t\=1,…,n,(4.16)

where z0(θ)\=0. This implies that the derivative sequence is an AR process, which we may easily compute recursively given a value of _θ_.

We will write the right side of (4.14) as

Q(θ)\=∑t\=1n\[wt(θ(0))⏟y−(θ−θ(0))⏟βz(θ(0))⏟x\]2(4.17)

and this is the quantity that we will minimize. The problem is now simple linear regression (y\=βx+ϵ), so that (β^\=∑xy/∑x2)

(θ−θ(0))^\=∑t\=1nzt(θ(0))wt(θ(0))/∑t\=1nzt2(θ(0)),

or

θ^\=θ(0)+∑t\=1nzt(θ(0))wt(θ(0))/∑t\=1nzt2(θ(0)).

Consequently, the Gauss–Newton procedure in this case is, on iteration j+1, set

θ(j+1)\=θ(j)+∑t\=1nzt(θ(j))wt(θ(j))∑t\=1nzt2(θ(j)),j\=0,1,2,…,(4.18)

where the values in (4.18) are calculated recursively using (4.13) and (4.16). The calculations are stopped when |θ(j+1)−θ(j)|, or |Q(θ(j+1))−Q(θ(j))|, are smaller than some preset amount.

Example 4.28 Fitting the Glacial Varve Series [Return to text.⏎](chapter4)

Consider the glacial varve series (_xt_) analyzed in [Example 3.14](#chapter3#exam3_14) and in [Problem 3.6](#chapter3#question3_6), where it was argued that a first-order moving average model might fit the logarithmically transformed and differenced varve series,

∇log(xt)\=log(xt)−log(xt−1).

The transformed series and the sample ACF and PACF are shown in [Figure 4.6](#chapter4#fig4_6) and based on [Table 4.1](#chapter4#tbl4_1), confirm the tendency of ∇log(xt) to behave as a first-order moving average.


`**tsplot**(diff(log(**varve**)), col=4, ylab=bquote(nabla~log~X[~t]), main="Transformed`
`   Glacial Varves")`
`acf2(diff(log(**varve**)), col=4)`

![Difference of the logged glacial varve time series and the corresponding sample PACF and sample ACF](./images/fig4_6.jpg)

Figure 4.6: Transformed glacial varves and corresponding sample ACF and PACF. [Return to text.⏎](chapter4)

101\. We see ρ^(1)≈−.4 and using method of moments for our initial estimate:

θ(0)\=1−1−4ρ^(1)22ρ^(1)≈−.5

based on [Example 4.26](#chapter4#exam4_26) and the quadratic formula. The code to run the Gauss–Newton and the results are:


`x = diff(log(**varve**))                    _#   data_`
`r = **acf1**(x, 1, plot=FALSE)              _#   acf(1)_`
`c(0) -> z -> Sc -> Sz -> Szw -> theta   _#   initialize ..._`
`c(x[1]) -> w                            _#   ... all variables_`
`num = length(x)                         _#   = 633_`
` `
`_## Estimation_`
`theta[1] = (1-sqrt(1-4*(r^2)))/(2*r)     _# MME_`
`niter    = 12`
`for (j in 1:niter){`
` for (t in 2:num){ w[t] = x[t]     - theta[j]*w[t-1]`
`                     z[t] = w[t-1] - theta[j]*z[t-1]`
` }`
` Sc[j]       = sum(w^2)`
` Sz[j]       = sum(z^2)`
` Szw[j]      = sum(z*w)`
` theta[j+1] = theta[j] + Szw[j]/Sz[j]`
`}`
`_## Results (rounded)_`
`cbind(iteration=1:niter-1, thetahat=theta[1:niter], Sc, Sz)`
`   iteration thetahat       Sc       Sz`
`           0   -0.495 158.763 171.305`
`           1   -0.668 150.787 235.245`
`           2   -0.733 149.306 300.405`
`           3   -0.756 149.071 336.646`
`           4   -0.765 149.030 354.019`
`           5   -0.769 149.022 362.039`
`           6   -0.771 149.020 365.693`
`           7   -0.772 149.020 367.349`
`           8   -0.772 149.020 368.098`
`           9   -0.772 149.020 368.436`
`          10   -0.772 149.020 368.589`
`          11   -0.772 149.020 368.658`

102\. The estimate is

θ^\=θ(11)\=−.772,

which results in the conditional sum of squares at convergence being

Sc(−.772)\=149.02.

The final estimate of the error variance is

σ^w2\=149.02632\=.236

with 632 degrees of freedom. The value of the sum of the squared derivatives at convergence is ∑t\=1nzt2(θ(11))\=368.66, and consequently, the estimated standard error of θ^ is

SE(θ^)\=.236/368.66\=.025 

using the standard regression results as an approximation. This leads to an approximate 95% confidence interval for _θ_ being −.772±2(.025)\=(−0.822,−0.722).

[Figure 4.7](#chapter4#fig4_7) displays the conditional sum of squares, Sc(θ) as a function of _θ_, as well as indicating the values of each step of the Gauss–Newton algorithm. Note that the Gauss–Newton procedure takes large steps toward the minimum initially, and then takes very small steps as it gets close to the minimizing value.


`_## Plot conditional SS_`
`c(0) -> w -> cSS`
`th = -seq(.3, .94, .01)`
`for (p in 1:length(th)){`
`  for (t in 2:num){ w[t] = x[t] - th[p]*w[t-1] }`
`  cSS[p] = sum(w^2)      }`
`**tsplot**(th, cSS, ylab=bquote(S[c](#undefined)), xlab=bquote(theta))`
`abline(v=theta[1:12], lty=2, col=4)    _# add previous results to plot_`
`points(theta[1:12], Sc[1:12], pch=16, col=4)`

![Demonstration of Gauss--Newton algorithm for a moving average fit to the difference of the logged glacial varve time series](./images/fig4_7.jpg)

Figure 4.7: Conditional sum of squares versus values of the moving average parameter for the glacial varve example, [Example 4.28](#chapter4#exam4_28). Vertical lines indicate the values of the parameter obtained via Gauss–Newton. [Return to text.⏎](chapter4)

103\. 

### Maximum Likelihood Estimation

A review of maximum likelihood estimation (MLE) for random samples is given in [Section A.7](#appA#secA_7). For time series, the idea is the same, and here we give the specifics for an AR(1) model with zero mean,

xt\=ϕxt−1+wt,

where |ϕ|<1 and wt∼ N(0,σw2). The likelihood is the joint density of the data x1,x2,…,xn, but where the parameters are the variables of interest. We write

L(ϕ,σw)\=fϕ,σw(x1,x2,…,xn)

for the likelihood.

For ease, let θ\=(ϕ,σw). The object of MLE is to find the “most likely” values of _θ_ given the data. This is accomplished by finding the values of _θ_ that maximize the likelihood.

Because an AR(1) is conditionally one-dependent, we may write the likelihood as

L(θ)\=fθ(x1,x2,…,xn)\=fθ(x1)fθ(x2∣x1)fθ(x3∣x2,x1)⋯fθ(xn∣xn−1,…,x1)\=fθ(x1)fθ(x2∣x1)fθ(x3∣x2)⋯fθ(xn∣xn−1).

Now, for t\=2,3,…,n,

xt∣xt−1∼N(ϕxt−1, σw2),

104\. so that

fθ(xt∣xt−1)\=1σw2πexp{−12σw2(xt−ϕxt−1)2}.

To find f(x1), we can use the causal representation as in [Example 4.1](#chapter4#exam4_1) to realize that x1∼ N(0,σw2/(1−ϕ2)), so

fθ(x1)\=1−ϕ2σw2πexp{−1−ϕ22σw2x12}.

Finally, for a zero-mean AR(1), the likelihood is

L(ϕ,σw)\=(2πσw2)−n/2(1−ϕ2)1/2exp\[−S(ϕ)2σw2\],(4.19)

where

S(ϕ)\=∑t\=2n(xt−ϕxt−1)2+(1−ϕ2)x12.(4.20)

Typically S(ϕ) is called the _unconditional sum of squares_. We could have also considered the estimation of _ϕ_ using _unconditional least squares_, that is, estimation by minimizing the unconditional sum of squares, S(ϕ). Using (4.19) and standard normal theory, the maximum likelihood estimate of σw2 is

σ^w2\=n−1S(ϕ^),

where ϕ^ is the MLE of _ϕ_.

If, in (4.19), we take logs, replace σw2 by its MLE, and ignore constants, ϕ^ is the value that minimizes the criterion function[2](#chapter4#fn4_2)

l(ϕ)\=log\[n−1S(ϕ)\]−n−1log(1−ϕ2).(4.21)

Because (4.20) and (4.21) are complicated functions of the parameters, the minimization of l(ϕ) or S(ϕ) is accomplished numerically. In the case of AR models, we have the advantage that, conditional on initial values, they are linear models. That is, we can drop the term in the likelihood that causes the nonlinearity. Conditioning on _x_1, the _conditional likelihood_ becomes

L(ϕ,σw∣x1)\=(2πσw2)−(n−1)/2exp\[−Sc(ϕ)2σw2\],

where the _conditional sum of squares_ is

Sc(ϕ)\=∑t\=2n(xt−ϕxt−1)2.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 2Because l(ϕ)∝−2logL(ϕ,σ^w), it is often called the profile or concentrated log-likelihood. [Return to text.⏎](#chapter4#fn4_24b)

105\. We can now use OLS to see that the conditional MLE of _ϕ_ is

ϕ^^\=∑t\=2nxtxt−1∑t\=2nxt−12,

so that the conditional MLE of σw2 is

σ^^w2\=Sc(ϕ^^)/(n−1).

For large sample sizes, the two methods of estimation are equivalent. The important difference arises when there is a small sample size, in which case unconditional MLE is preferred.

If the mean of _xt_ is not zero, we can include it in the likelihood. Many statistical packages will simply estimate the mean by the sample average, x¯, and then perform MLE on the data xt−x¯. Although the sample mean is not necessarily the MLE of _μ_, it is just as good as the MLE.

Example 4.29 Transformed Glacial Varves (cont.) [Return to text.⏎](#chapter5#b5exam4_29)

In [Example 4.28](#chapter4#exam4_28), we used Gauss–Newton to fit an MA(1) model to the transformed glacial varve series via conditional least squares. To use MLE, we can use the script **sarima** from **astsa** as follows. The transformed data appear to have a zero mean function, so we do not fit a constant.


`**sarima**(diff(log(**varve**)), q=1, **no.constant**=TRUE)`
`_# partial output_`
` initial value -0.551778    _# conditional SS_`
` iter   2 value -0.671626`
`  .     .   .       .`
` iter   9 value -0.723195`
` final value -0.723195`
` converged`
` initial value -0.722700     _# MLE_`
` iter   2 value -0.722702`
` iter   3 value -0.722702`
` final value -0.722702`
` converged`
` `
`Coefficients:`
`       Estimate     SE t.value p.value`
`  ma1   -0.7705 0.0341 -22.6161       0`
`sigma^2 estimated as 0.2353156 on 632 degrees of freedom`
`AIC = 1.398791 AICc = 1.398802 BIC = 1.412853`

The script starts by using the data to pick initial values of the estimates that are within the causal and invertible region of the parameter space. Then, the script uses conditional least squares as in [Example 4.28](#chapter4#exam4_28). Once that process has converged, the next step is to use the conditional estimates to find the MLEs.

The output shows only the iteration number and the value of the sum of squares or the likelihood. It is a good idea to look at the results of the numerical optimization to make sure it converges and that there are no warnings. If there 106\. is trouble converging or there are warnings, it usually means that the proposed model is not even close to reality. The final estimates are θ^\=−.7705(.034) and σ^w2\=.2353. These are nearly the values obtained in [Example 4.28](#chapter4#exam4_28), which were θ^\=−.773(.025) and σ^w2\=.236.

Example 4.30 When Numerical Optimization Fails [Return to text.⏎](chapter4)

In this example, we fit an ARMA(3,3) model to white noise with a large sample size of n\=1000. Recalling [Examples 4.10](#chapter4#exam4_10) and [4.11](#chapter4#exam4_11), we can represent white noise as an ARMA if we allow parameter redundancy. In this example, the parameters are all significant, but the AR and MA sides are nearly identical; i.e., ϕ^(B)≈θ^(B). In addition, note that the numerical routine does not converge and a warning is given (but this will not always happen in this situation).


`set.seed(666)`
`**sarima**(rnorm(1000), p=3, q=3)`
`  initial value -0.017416`
`  iter   2 value -0.017504`
`    .    .           .`
`  iter 78 value -0.022819`
`  iter 79 value -0.022819`
`  final value -0.022819`
`  converged`
`  initial value -0.019482`
`  iter   2 value -0.019495`
`    .    .           .`
`  iter 99 value -0.022822`
`  iter 100 value -0.022828`
`  final value -0.022828`
` stopped after 100 iterations`
`---`
`  Coefficients: _# (-rounded for your pleasure-)_`
`        Estimate     SE t.value p.value`
`  ar1        0.60  0.04     17.1    0.0`
`  ar2        0.48  0.05      8.8    0.0`
`  ar3       -0.93  0.04    -26.1    0.0`
`  ma1       -0.60  0.03    -21.6    0.0`
`  ma2       -0.51  0.04    -12.2    0.0`
`  ma3        0.96  0.03     34.1    0.0`
`  xmean     -0.02  0.03     -0.7    0.5`
` sigma^2 estimated as 0.9540239 on 993 degrees of freedom`
` AIC = 2.808193 AICc = 2.808306 BIC = 2.847455`
`Warning message: ...`
`  possible convergence problem: optim gave code = 1`

Notice that

ϕ^(B)\=1−.60B−.48B2+.93B3θ^(B)\=1−.60B−.51B2+.96B3,

indicating the process is noise, but notice that all the regression parameters are significant.

Example 4.31 107\. Automation: When and Why It Fails\* [Return to text.⏎](chapter4)

The problems discussed in [Examples 4.9](#chapter4#exam4_9), [4.10](#chapter4#exam4_10), [4.11](#chapter4#exam4_11) and [4.30](#chapter4#exam4_30) currently plague automated ARMA fitting routines that seem to be favored in Machine Learning. The main problem is in allowing both AR and MA sides to be automatically fit to the data without checking for parameter redundancy (which can be difficult to check for estimated models).

Other methodological drawbacks are including insignificant parameters and not performing residual analysis to verify a model. Including insignificant parameters in a model can lead to imprecise estimates (details in [Example 4.32](#chapter4#exam4_32)). Two such automated routines are Auto\_Arima in the IMSL Python Numerical Library [(IMSL, 2020)](#bibref1#refbib_28) and auto.arima from the forecast R package.

In the following examples, we use the n\=1000 white noise values generated in [Example 4.30](#chapter4#exam4_30):


`set.seed(666)`
`x = rnorm(1000)`

Using auto.arima, we first try the default method because this is often used. After that, we try a method that looks at all subsets (like all subset regression).


`library(forecast)`
`auto.arima(x)                   _# stepwise_`
`  ARIMA(2,0,1) with zero mean`
`    Coefficients:`
`              ar1       ar2     ma1`
`          -0.9744 -0.0477 0.9509`
`    s.e.   0.0429    0.0321 0.0294`
`   sigma^2 = 0.9657: log likelihood = -1400`
`   AIC=2808.01    AICc=2808.05   BIC=2827.64`
`auto.arima(x, stepwise=FALSE) _# all subsets_`
`  ARIMA(4,0,1) with zero mean`
`    Coefficients:`
`              ar1       ar2      ar3       ar4   ma1`
`          -0.9575 -0.0349 -0.0293 -0.0485 0.9354`
`    s.e.   0.0488    0.0438   0.0438   0.0325 0.0376`
`   sigma^2 = 0.9653: log likelihood = -1398.79`
`   AIC=2809.58    AICc=2809.66   BIC=2839.02`

Both models are overparameterized white noise and include insignificant values (we are not certain why all subsets settles on a worse model than stepwise by any IC). That is, both models are essentially of ARMA(1,1) form with ϕ^1≈−θ^1 just as in [Example 4.10](#chapter4#exam4_10).

Problem 5.10 in [Chapter 5](#chapter5) explores these problems further. For right now, an easy solution to automation for avoiding parameter redundancy is to fit AR(_p_) models of increasing order _p_ (to some limit) and choose the one with the smallest chosen IC. The idea is simple, and its justification is discussed in [Section 7.3](#chapter7#sec7_3). Here is an example on the same white noise series using AIC. In this case, the selected model is white noise (p\=0).


`ar(x) _# uses AIC by default_`
` Order selected 0 sigma^2 estimated as   0.9687`

108\. Most packages use large sample theory to estimate standard errors. We give a few examples in the following.

Example 4.32 Some Large Sample Distributions [Return to text.⏎](chapter4)

For large sample sizes, the MLEs of the ARMA regression parameters are approximately normally distributed. It is worthwhile to examine a few examples along with a corresponding approximate confidence intervals (CIs).

**AR(1):**

ϕ^∼⋅N(ϕ, 1−ϕ2n).

An approximate 100(1−α)% CI for _ϕ_ is

ϕ^±zα/21−ϕ^2n,

where _zq_ is the usual (1−q) quantile of the standard normal distribution.

**AR(2):**

ϕ^1∼⋅N(ϕ1, 1−ϕ22n)andϕ^2∼⋅N(ϕ2, 1−ϕ22n).

Notice that the standard errors of ϕ^1 and ϕ^2 are the same. Thus, approximate 100(1−α)% CIs for _ϕ_1 and _ϕ_2 are

ϕ^1±zα/21−ϕ^22nandϕ^2±zα/21−ϕ^22n.

**MA(1):**

θ^∼⋅N(θ, 1−θ2n).

An approximate 100(1−α)% CI for _θ_ is

θ^±zα/21−θ^2n,

which is similar to the AR(1) case.

**MA(2):**

θ^1∼⋅N(θ1,1−θ22n)andθ^2∼⋅N(θ2,1−θ22n).

Again, notice that the estimated standard errors are the same. Approximate 100(1−α)% CIs for _θ_1 and _θ_2 are

θ^1±zα/21−θ^22nandθ^2±zα/21−θ^22n.

**ARMA(1, 1):**

ϕ^∼⋅N(ϕ,(1−ϕ2) C2(ϕ,θ)n)andθ^∼⋅N(θ,(1−θ2) C2(ϕ,θ)n),

109\. where C(ϕ,θ)\=(1+ϕθϕ+θ). Approximate 100(1−α)% CIs for _ϕ_ and _θ_ are

ϕ^±zα/2(1−ϕ^2) C2(ϕ^,θ^)nandθ^±zα/2(1−θ^2) C2(ϕ^,θ^)n.

Note that if ϕ≈−θ (recall [Example 4.30](#chapter4#exam4_30)), the denominator of C(ϕ,θ) will be close to zero. Consequently, the standard errors will be very large and the CIs can be unduly wide.

The large sample behavior of the parameter estimators displayed in [Example 4.32](#chapter4#exam4_32) gives us an additional insight into the problem of fitting ARMA models to data.

Example 4.33 Overfitting Caveat

In [Examples 4.30](#chapter4#exam4_30) and [4.31](#chapter4#exam4_31), we saw that when fitting ARMA models to data, the AR and MA sides can cancel each other out, leading to overparameterization and overly complex models. For example, simple white noise can be written as an ARMA(k,k) model, for k\=1,2,….

In addition, [Example 4.32](#chapter4#exam4_32) provides some insight into overfitting a model. Suppose a time series follows an AR(1) process, what is the problem if we decide to fit an AR(2) to the data? After all, if the process is truly an AR(1), the estimate of the second AR parameter should be close to zero. The answer is that if we overfit, we obtain less efficient, or less precise parameter estimates. For example, if we fit an AR(1) to an AR(1) process, for large _n_, var(ϕ^1)≈n−1(1−ϕ12). But, if we fit an AR(2) to the AR(1) process, for large _n_, var(ϕ^1)≈n−1(1−ϕ22)\=n−1 because ϕ2\=0. Thus, the variance of _ϕ_1 has been inflated, making the estimator less precise.

For example, we can simulate data from an AR(1) and compare the standard errors of ϕ^1 when fitting the correct model versus overfitting.


`set.seed(1)`
`x = **sarima.sim**(ar=.9, n=100)             _# simulate an AR(1)_`
`**sarima**(x,1,0,0, **no.constant**=TRUE)        _#- fit AR(1)_`
`      Estimate    SE t.value p.value`
`  ar1     0.91 0.04     22.84        0   _# SE is .04_`
`**sarima**(x,2,0,0, **no.constant**=TRUE)        _#- overfit AR(2)_`
`      Estimate    SE t.value p.value`
`  ar1     0.83 0.10      8.43     0.00   _# SE is .10_`
`  ar2     0.09 0.10      0.88     0.38`

Notice that the estimated standard error of ϕ^1 is two and a half times as large in the overfitted model, and the estimate is worse.

We do want to mention, however, that overfitting can be used as a diagnostic tool. For example, if we fit an AR(1) model to the data and are satisfied with that model, then adding one more parameter and fitting an AR(2) should lead to approximately the same model as in the AR(1) fit. We will discuss model diagnostics in more detail in [Section 5.2](#chapter5#sec5_2).

## 4.6 110\. Forecasting

In forecasting, the goal is to predict future values of a time series, xn+m, m\=1,2,…, based on the data, x1,…,xn, collected to the present. Throughout this section, we will assume that the model parameters are known. When the parameters are unknown, we replace them with their estimates.

To understand forecasting normal ARMA processes, it is instructive to first consider a mean-zero AR(1),

xt\=ϕxt−1+wt.

If the mean is not zero, replace _xs_ with xs−μ. For one-step-ahead prediction, given data x1,…,xn, we wish to forecast the value of the time series at the next time point, xn+1. We will call the forecast xn+1n where the notation xtn refers to what we can expect _xt_ to be given the data x1,…,xn.[3](#chapter4#fn4_3) Since

xn+1\=ϕxn+wn+1,

we have

xn+1n\=ϕxnn+wn+1n.

But since we know _xn_ (it is one of our observations), xnn\=xn, and since wn+1 is a future error and independent of x1,…,xn, we have wn+1n\=E(wn+1)\=0. Consequently, the _one-step-ahead forecast_ is

xn+1n\=ϕxn.

The one-step-ahead _mean squared prediction error_ (MSPE) is given by

Pn+1n\=E\[xn+1−xn+1n\]2\=E\[xn+1−ϕxn\]2\=Ewn+12\=σw2.

The two-step-ahead forecast is obtained similarly. Since the model is

xn+2\=ϕxn+1+wn+2,

we predict

xn+2n\=ϕxn+1n+wn+2n.

Again, wn+2 is a future error, so wn+2n\=0. Also, we already know xn+1n\=ϕxn, so the forecast is

xn+2n\=ϕxn+1n\=ϕ2xn.

The two-step-ahead MSPE is given by

Pn+2n\=E\[xn+2−xn+2n\]2\=E\[ϕxn+1+wn+2−ϕ2xn\]2\=E\[wn+2+ϕ(xn+1−ϕxn)\]2\=E\[wn+2+ϕwn+1\]2\=σw2(1+ϕ2).

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 3Formally xtn\=E(xt∣x1,…,xn) is conditional expectation, which is discussed in [Section A.5](#appA#secA_5). [Return to text.⏎](#chapter4#fn4_34b)

111\. Generalizing these results, the _m_\-step-ahead forecast is

xn+mn\=ϕmxn,(4.22)

with MSPE

Pn+mn\=E\[xn+m−xn+mn\]2\=σw2(1+ϕ2+⋯+ϕ2(m−1)).(4.23)

for m\=1,2,….

Because |ϕ|<1, we will have ϕm→0 fast as m→∞. Thus the forecasts in (4.22) will soon go to zero (or the mean). In addition, the MSPE (4.23) will converge to σw2∑j\=0∞ϕ2j\=σw2/(1−ϕ2), which is the variance of the process _xt_; recall ([4.2](#chapter4#ufig4_1)). This means that based on the data x1,…,xn alone, as the forecast horizon _m_ grows, the forecasts will become x¯, the sample mean, with a root mean square prediction error of _S_, the sample standard deviation of the data.

Forecasting an AR(_p_) model is basically the same as forecasting an AR(1) provided the sample size _n_ is larger than the order _p_, which should be the case. Since MA(_q_) and ARMA(_p,q_) models are AR(∞) by invertibility, the same basic techniques can be used. The models are invertible, wt\=xt+∑j\=1∞πjxt−j, and we may write

xn+m\=−∑j\=1∞πjxn+m−j+wn+m.

If we had the infinite history, {xn,xn−1,…,x1,x0,x−1,…}, of the data available, we would predict xn+m by

xn+mn\=−∑j\=1∞πjxn+m−jn

successively for m\=1,2,…. In this case, xtn\=xt for t\=n,n−1,…. We only have the actual data {xn,xn−1,…,x1} available, but a practical solution is to truncate the forecasts as

xn+mn\=−∑j\=1n+m−1πjxn+m−jn,(4.24)

with xtn\=xt for 1≤t≤n. In this case, we are putting xtn\=0 for t≤0. For ARMA models in general, as long as _n_ is large, the approximation works well because the _π_\-weights are going to zero exponentially fast. For large _n_, it can be shown (see [Problem 4.11](#chapter4#question4_11)) that the mean squared prediction error for ARMA(_p,q_) models is approximately (exact if q\=0)

Pn+mn\=σw2∑j\=0m−1ψj2.(4.25)

We saw this result in (4.23) for the AR(1) because in that case, ψj2\=ϕ2j.

Example 4.34 112\. Forecasting the Recruitment Series

In [Examples 4.22](#chapter4#exam4_22) and [4.25](#chapter4#exam4_25) we fit an AR(2) model to the Recruitment series using OLS and Yule–Walker. Here, we use maximum likelihood estimation:


`fish = **sarima**(**rec**, p=2)      _# fit the model_`
`       Estimate     SE t.value p.value`
` ar1     1.3512 0.0416 32.4933        0`
` ar2    -0.4612 0.0417 -11.0687       0`
` xmean 61.8585 4.0039 15.4494         0`

The results are nearly the same as in [Examples 4.22](#chapter4#exam4_22) and [4.25](#chapter4#exam4_25). Using the parameter estimates as the actual parameter values, the forecasts and root MSPEs can be calculated in a similar fashion to the introduction to this section.

[Figure 4.8](#chapter4#fig4_8) shows the result of forecasting the Recruitment series over a 24-month horizon, m\=1,2,…,24, obtained as follows:


`**sarima.for**(**rec**, n.ahead=24, p=2)`
`abline(h=fish[[1]]$coef["xmean"])   _# display estimated mean_`

![Forecasts of the fish Recruitment  time series based on a fitted autoregression of order two](./images/fig4_8.jpg)

Figure 4.8: Twenty-four-month forecasts for the Recruitment series. The actual data shown are from about January 1979 to September 1987, and then the forecasts plus and minus one and two standard error are displayed. The solid horizontal line is the estimated mean function. [Return to text.⏎](chapter4)

![Causal region for an autoregression of order two, indicating when the model has real roots or complex roots](./images/fig4_9.jpg)

Figure 4.9: Causal region for an AR(2) in terms of the parameters. [Return to text.⏎](chapter4)

Note how the forecast levels off to the mean quickly and the prediction intervals are wide and become constant. That is, because of the short memory, the forecasts settle to the estimated mean, 61.86, and the root MSPE becomes quite large (and eventually settles at the standard deviation of all the data).

As a practical matter, truncated prediction can be calculated using simple recursions. We demonstrate the method on an ARMA(1,1) model and note that the method generalizes to any ARMA model.

Example 4.35 113\. Forecasting an ARMA(1,1)\*

Given data x1,…,xn, for forecasting purposes, write the model as

xn+1\=ϕxn+wn+1+θwn.

Then, the one-step-ahead forecast is

xn+1n\=ϕxn+0+θwnn,

noting wn+jn\=0 for j≥1. For m≥2, we have

xn+mn\=ϕxn+m−1n,

which can be calculated recursively, m\=2,3,… .

To calculate wnn, which is needed to initialize the successive forecasts, write

wt\=xt−ϕxt−1−θwt−1

for t\=1,…,n. For truncated forecasting put w0n\=0 and x0\=0, and then iterate the errors forward in time

wtn\=xt−ϕxt−1−θwt−1n,t\=1,…,n.

The MSPEs can be computed iteratively from (4.25) using the _ψ_\-weights. Let rm\=∑j\=0mψj2 (note r0\=1) so that

rm\=rm−1+ψm2

and consequently we may caluclate Pn+mn\=σw2rm−1 recursively for m\=1,2,…. For this example, the _ψ_\-weights satisfy

ψj\=(ϕ+θ)ϕj−1,

for j≥1. This result gives

Pn+mn\=σw2\[1+(ϕ+θ)2∑j\=1m−1ϕ2(j−1)\]\=σw2\[1+(ϕ+θ)2(1−ϕ2(m−1))(1−ϕ2)\].

## 4.7 Causality and Invertibility\*

In this section, we go into more detail about the properties of causality and invertibility. We require ARMA models to meet these requirements for a number of reasons. Causality requires that the present value of the time series does not 114\. depend on the future; otherwise, forecasting would be futile. Invertibility ensures model uniqueness by requiring that the noise is not future-dependent.

The AR operator is

ϕ(B)\=(1−ϕ1B−ϕ2B2−⋯−ϕpBp),(4.26)

and the MA operator is

θ(B)\=(1+θ1B+θ2B2+⋯+θqBq),(4.27)

so that an ARMA model may be written as

ϕ(B)(xt−μ)\=θ(B)wt,

where ϕ(B) and θ(B) have no common factors. In the following, to ease the notation, we assume the mean function is zero, μt\=μ\=0.

Definition 4.36 _(Causality)._ _The **causal form** of the model is given by_

xt\=ϕ(B)−1θ(B)wt\=ψ(B)wt\=∑j\=0∞ψjwt−j,(4.28)

_where_ ψ(B)\=∑j\=0∞ψjBj _(_ψ0\=1 _) and assuming_ ϕ(B)−1 _exists._

Because xt\=ψ(B)wt, we must have

ϕ(B)ψ(B)wt⏟xt\=θ(B)wt,

so the parameters _ψj_ may be obtained by matching coefficients of _B_ in

ϕ(B)ψ(B)\=θ(B).(4.29)

Definition 4.37 (Invertibility). _The **invertible form** of the model is given by_

wt\=θ(B)−1ϕ(B)xt\=π(B)xt\=∑j\=0∞πjxt−j.(4.30)

_where_ π(B)\=∑j\=0∞πjBj _(_π0\=1 _) assuming_ θ(B)−1 _exists._

Likewise, the parameters _πj_ may be obtained by matching coefficients of _B_ in

ϕ(B)\=π(B)θ(B).(4.31)

**Remarks.** ARMA models have stronger causal and invertible properties than specified in [Definition 4.4](#chapter4#defi4_4) and [Definition 4.7](#chapter4#defi4_7) in that the parameters are absolutely 115\. summable, which is stronger than the square summability requirement in the general definitions. That is, for ARMA models,

∑j\=0∞|ψj|<∞and∑j\=0∞|πj|<∞,

because the coefficients converge exponentially fast to zero.

Property 4.38 _Causality and Invertibility (existence)_ [Return to text.⏎](chapter4) _Let_

ϕ(z)\=1−ϕ1z−⋯−ϕpzpandθ(z)\=1+θ1z+⋯+θqzq

_be the AR and MA polynomials obtained by replacing the backshift operator B in (4.26) and (4.27) by a complex number z._

_An ARMA(p,q) model is **causal** if and only if_ ϕ(z)≠0 _for_ |z|≤1. _The coefficients of the linear process given in (4.28) can be determined by solving (_ψ0\=1 _)_

ψ(z)\=∑j\=0∞ψjzj\=θ(z)ϕ(z),|z|≤1.

_An ARMA(p,q) model is **invertible** if and only if_ θ(z)≠0 _for_ |z|≤1. _The coefficients πj of_ π(B) _given in (4.30) can be determined by solving (_π0\=1 _)_

π(z)\=∑j\=0∞πjzj\=ϕ(z)θ(z),|z|≤1.

Another way to describe the conditions of [Property 4.38](#chapter4#prop4_38) is that an ARMA model is causal only if the roots of the AR polynomial are outside the unit circle. Likewise, an ARMA model is invertible only if the roots of the MA polynomial are outside the unit circle. We demonstrate the property in the following examples.

Example 4.39 AR(1) Model

In [Example 4.1](#chapter4#exam4_1) we saw that the AR(1) model xt\=ϕxt−1+wt, or

(1−ϕB)xt\=wt

has the causal representation

xt\=ψ(B)wt\=∑j\=0∞ϕjwt−j,

provided that |ϕ|<1. The AR polynomial is ϕ(z)\=1−ϕz and

1ϕ(z)\=11−ϕ z\=∑j\=0∞ϕjzj,|z|≤1.

116\. We see immediately that ψj\=ϕj. In addition, the root of ϕ(z)\=1−ϕz is z0\=1/ϕ, and |z0|\>1 is equivalent to |ϕ|<1.

Example 4.40 Parameter Redundancy, Causality, Invertibility

Consider the process

xt\=.4xt−1+.45xt−2+wt+wt−1+.25wt−2,

or, in operator form,

(1−.4B−.45B2)xt\=(1+B+.25B2)wt.

At first, _xt_ appears to be an ARMA(2,2) process. But notice that

ϕ(B)\=1−.4B−.45B2\=(1+.5B)(1−.9B)

and

θ(B)\=(1+B+.25B2)\=(1+.5B)2

have a common factor that can be canceled. After cancellation, the operators are ϕ(B)\=(1−.9B) and θ(B)\=(1+.5B), so the model is an ARMA(1,1) model, (1−.9B)xt\=(1+.5B)wt, or

xt\=.9xt−1+.5wt−1+wt.(4.32)

The model is causal because ϕ(z)\=(1−.9z)\=0 when z\=10/9, which is outside the unit circle. The model is also invertible because the root of θ(z)\=(1+.5z) is z\=−2, which is outside the unit circle.

To write the model as a linear process, we can obtain the _ψ_\-weights using [Property 4.38](#chapter4#prop4_38), ϕ(z)ψ(z)\=θ(z), or

(1−.9z)(1+ψ1z+ψ2z2+⋯+ψjzj+⋯)\=1+.5z.

Rearranging, we get

1+(ψ1−.9)z+(ψ2−.9ψ1)z2+⋯+(ψj−.9ψj−1)zj+⋯\=1+.5z.

The coefficients of _z_ on the left and right sides must be the same, so we get ψ1−.9\=.5 or ψ1\=1.4, and ψj−.9ψj−1\=0 for j\>1. Thus, ψj\=1.4(.9)j−1 for j≥1 and (4.32) can be written as

xt\=wt+1.4∑j\=1∞.9j−1wt−j.

The invertible representation using [Property 4.38](#chapter4#prop4_38) is obtained by matching coefficients in θ(z)π(z)\=ϕ(z),

(1+.5z)(1+π1z+π2z2+π3z3+⋯)\=1−.9z.

In this case, the _π_\-weights are given by πj\=(−1)j 1.4 (.5)j−1, for j≥1, and hence, we can also write (4.32) as

xt\=1.4∑j\=1∞(−.5)j−1xt−j+wt.

Example 4.41 117\. Causal Conditions for an AR(2) Process [Return to text.⏎](chapter4)

For an AR(1) model to be causal, we must have ϕ(z)≠0 for |z|≤1. If we solve ϕ(z)\=1−ϕz\=0, we find that the root (or zero) occurs at z0\=1/ϕ, so that |z0|\>1 is equivalent to |ϕ|<1. In this case, it is easy to relate parameter conditions to root conditions.

The AR(2) model is causal when the two roots of ϕ(z)\=1−ϕ1z−ϕ2z2 lie outside of the unit circle. That is, if _z_1 and _z_2 are the roots, then we require |z1|\>1 and |z2|\>1. Using the quadratic formula, this requirement can be written as

|ϕ1±ϕ12+4ϕ2−2ϕ2|\>1.

The roots of ϕ(z) may be real and distinct, real and equal, or a complex conjugate pair. In terms of the coefficients, the equivalent condition is

ϕ1+ϕ2<1,ϕ2−ϕ1<1,and|ϕ2|<1.

This causality condition specifies a triangular region in the parameter space; see [Figure 4.9](#chapter4#fig4_9).

Example 4.42 An AR(2) with Complex Roots

In [Example 4.3](#chapter4#exam4_3) we considered the AR(2) model

xt\=1.5xt−1−.75xt−2+wt,

with σw2\=1. [Figure 4.2](#chapter4#fig4_2) shows the _ψ_\-weights and a simulated sample. This particular model has complex-valued roots and was chosen so the process exhibits pseudo-cyclic behavior at the rate of one cycle every 12 time points.

The autoregressive polynomial for this model is

ϕ(z)\=1−1.5z+.75z2.

118\. The roots, z1,z2, of ϕ(z) are 1±i/3, and arg(z1)\=tan−1(1/3)\=2π/12 radians per unit time (for details, see [Section B.2](#appB#secB_2)). To convert the angle to cycles per unit time, divide by 2π to get 1/12 cycles per unit time. The ACF for this model is shown in [Figure 4.4](#chapter4#fig4_4). The following code calculates the roots of the polynomial and solves for _arg_:


`z = c(1,-1.5,.75)       _# coefficients of the polynomial_`
`                                                 √`
`(z1 = polyroot(z)[1])   _# print one root = 1 + i/ 3_`
` [1] 1+0.57735i`
`arg = Arg(z1)/(2*pi)    _# arg in cycles/pt_`
`1/arg`
` [1] 12`

## Problems

* 4.1. For an MA(1), xt\=wt+θwt−1, show that |ρx(1)|≤1/2 for any number _θ_. For which values of _θ_ does ρx(1) attain its maximum and minimum?
* 4.2. Let {wt; t\=0,1,…} be a white noise process with variance σw2 and let |ϕ|<1 be a constant. Consider the process x0\=w0, and  
xt\=ϕxt−1+wt,t\=1,2,….  
We might use this method to simulate an AR(1) process from simulated white noise.  
   1. Show that xt\=∑j\=0tϕjwt−j for any t\=0,1,….  
   2. Find the E(xt).  
   3. Show that, for t\=0,1,…,  
   var(xt)\=σw21−ϕ2(1−ϕ2(t+1))  
   4. Show that, for h≥0,  
   cov(xt+h,xt)\=ϕhvar(xt).  
   5. Is _xt_ stationary?  
   6. Argue that, as t→∞, the process becomes stationary, so in a sense, _xt_ is “asymptotically stationary.”  
   7. Comment on how you could use these results to simulate _n_ observations of a stationary Gaussian AR(1) model from simulated iid N(0,1) values.  
   8. Now suppose x0\=w0/1−ϕ2. Is this process stationary? _Hint_: Show var(xt) is constant.
* 4.3. 119\. Consider the following two models:  
   1. xt\=.80xt−1−.15xt−2+wt−.30wt−1.  
   2. xt\=xt−1−.50xt−2+wt−wt−1.  
   1. Using [Example 4.10](#chapter4#exam4_10) and [4.11](#chapter4#exam4_11) as guides, check the models for parameter redundancy. If a model has redundancy, find the reduced form of the model.  
   2. A way to tell if an ARMA model is causal is to examine the roots of AR term ϕ(B) to see if there are no roots less than or equal to one in magnitude. Likewise, to determine invertibility of a model, the roots of the MA term θ(B) must not be less than or equal to one in magnitude. Use [Example 4.11](#chapter4#exam4_11) as a guide to determine if the reduced (if appropriate) models (i) and (ii), are causal and/or invertible.  
   3. In [Example 4.3](#chapter4#exam4_3) and [Example 4.12](#chapter4#exam4_12), we used ARMAtoMA and **ARMAtoAR** to exhibit some of the coefficients of the causal \[MA(∞)\] and invertible \[AR(∞)\] representations of a model. If the model is in fact causal or invertible, the coefficients must converge to zero fast. For each of the reduced (if appropriate) models (i) and (ii), find the first 50 coefficients and comment.
* 4.4.  
   1. Compare the _theoretical_ ACF and PACF of an ARMA(1,1), an ARMA(1,0), and an ARMA(0,1) series by plotting the ACFs and PACFs of the three series for ϕ\=.6, θ\=.9. Comment on the capability of the ACF and PACF to determine the order of the models. _Hint:_ See the code for [Example 4.19](#chapter4#exam4_19).  
   2. Use sarima.sim to generate n\=100 observations from each of the three models discussed in (a). Compute the sample ACFs and PACFs for each model and compare it to the theoretical values. How do the results compare with the general results given in [Table 4.1](#chapter4#tbl4_1)?  
   3. Repeat (b) but with n\=500. Comment.
* 4.5. Let _ct_ be the cardiovascular mortality series (**cmort**) discussed in [Example 3.6](#chapter3#exam3_6) and let xt\=∇ct be the differenced data.  
   1. Plot _xt_ and compare it to the actual data plotted in [Figure 3.3](#chapter3#fig3_3). Why does differencing seem reasonable in this case?  
   2. Calculate and plot the sample ACF and PACF of _xt_ and using [Table 4.1](#chapter4#tbl4_1), argue that an AR(1) is appropriate for _xt_.  
   3. Fit an AR(1) to _xt_ using maximum likelihood. The easiest way to do this is to use **sarima** from **astsa**. Comment on the significance of the regression parameter estimates of the model. What is the estimate of the white noise variance?  
   4. 120\. Examine the residuals and comment on whether or not you think the residuals are white.  
   5. Assuming the fitted model is the true model, find the forecasts over a four-week horizon, xn+mn, for m\=1,2,3,4, and the corresponding 95% prediction intervals; n\=508 here. The easiest way to do this is to use **sarima.for**.  
   6. Show how the values obtained in part (e) were calculated.  
   7. What is the one-step-ahead forecast of the actual value of cardiovascular mortality; i.e., what is cn+1n?
* 4.6. Redo the analysis in [Example 4.30](#chapter4#exam4_30) using the same seed, but fit an ARMA(2,2) model to the simulated data. What happens in this case?
* 4.7. For an AR(1) model, determine the general form of the _m_\-step-ahead forecast xn+mn and show  
E\[(xn+m−xn+mn)2\]\=σw21−ϕ2m1−ϕ2.
* 4.8. Repeat the following numerical exercise five times. Generate n\=100 iid N(0,1) observations. Fit an ARMA(1,1) model to the data. Compare the parameter estimates in each case and explain the results.
* 4.9. Generate 10 realizations of length n\=500 each of an ARMA(1,1) process with ϕ\=.9,θ\=.5 and σ2\=1. Find the MLEs of the three parameters in each case and compare the estimators to the true values.
* 4.10. Using [Example 4.27](#chapter4#exam4_27) as your guide, find the Gauss–Newton procedure for estimating the autoregressive parameter, _ϕ_, from the AR(1) model, xt\=ϕxt−1+wt, given data x1,…,xn. Does this procedure produce the unconditional or the conditional estimator?
* 4.11. \* **(Forecast Errors)** In (4.25), we stated without proof that, for large _n_, the mean squared prediction error for ARMA(_p,q_) models is approximately (exact for an AR(_p_) if n\>p) Pn+mn\=σw2∑j\=0m−1ψj2. To establish (4.25), write a future observation in terms of its causal representation, xn+m\=∑j\=0∞ψjwm+n−j. Show that if an infinite history, {xn,xn−1,…,x1,x0,x−1,…}, is available, then  
xn+mn\=∑j\=0∞ψjwm+n−jn\=∑j\=m∞ψjwm+n−j.  
Now, use this result to show that  
E\[xn+m−xn+mn\]2\=E\[∑j\=0m−1ψjwn+m−j\]2\=σw2∑j\=0m−1ψj2.

---

