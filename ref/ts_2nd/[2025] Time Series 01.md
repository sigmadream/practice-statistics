<a role="toc_link" id="chapter1"></a>
1\. 

# 1Time Series Elements

The analysis of data observed at different time points leads to unique problems that are not covered by classical statistics. The dependence introduced by the sampling data over time restricts the applicability of many conventional statistical methods that require random samples. The analysis of such data is commonly referred to as _time series analysis_. In this chapter, we examine some typical time series data sets and explore ways to model the behavior we see in the data.

## 1.1 Introduction

To provide a statistical setting for describing the elements of time series data, the data are represented as a collection of random variables indexed according to the order they are obtained in time. For example, if we observe daily particulate pollution levels at a certain location, we may consider the time series as a sequence of random variables, x1,x2,x3,… , where the random variable _x_1 denotes the level on day one, the variable _x_2 denotes the level on the second day, _x_3 denotes the level on the third day, and so on. In general, a collection of random variables, {xt}, indexed by _t_ is referred to as a _stochastic process_. In this text, _t_ will typically be discrete and vary over the integers t\=0,±1,±2,… or some subset of the integers, or a similar index like months of a year.

The observed values of a stochastic process are referred to as a realization of the stochastic process. Because it will be clear from the context of our discussions, we use the term time series whether we are referring generically to the process or to a particular realization and make no notational distinction between the two concepts.

Historically, time series methods were applied to problems in the physical and environmental sciences. This fact accounts for the engineering nomenclature that permeates the language of time series analysis. The first step in an investigation of time series data involves careful scrutiny of the recorded data plotted over time. Before looking more closely at the particular statistical methods, we mention that two separate, but not mutually exclusive, approaches to time series analysis exist, commonly identified as the _time domain approach_ ([Chapters 4](#chapter4) and [5](#chapter5)) and the _frequency domain approach_ ([Chapters 6](#chapter6) and [7](#chapter7)).

## 1.2 2\. Time Series Data

The primary objective of time series analysis is to develop mathematical models that provide plausible descriptions for sample data. In this section, we examine various types of data sets, and then discuss some models that help explain the behavior of the time series data we observe.

Example 1.1 Johnson & Johnson Quarterly Earnings [Return to text.⏎](chapter1)

[Figure 1.1](#chapter1#fig1_1) shows quarterly earnings per share (QEPS) for Johnson & Johnson and the data logged. There are 84 quarters (21 years) measured from the first quarter of 1960 to the last quarter of 1980\. Modeling such series begins by observing the primary patterns in the time history. In this case, note the increasing underlying trend and variability, and a somewhat regular oscillation superimposed on the trend that seems to repeat over quarters.

![Johnson & Johnson quarterly earnings per share and the same data plotted on a log scale](./images/fig1_1.jpg)

Figure 1.1: Johnson & Johnson quarterly earnings per share, 1960-I to 1980-IV (top). The same data logged (bottom). [Return to text.⏎](chapter1)

If we consider the data as being generated as a small percentage change each quarter, say _rt_ (which can be negative), we might write

xt\=(1+rt)xt−1,

where _xt_ is the earning for quarter _t_. If we log the data, then

log(xt)\=log(1+rt)+log(xt−1),

3\. implying a linear growth rate; i.e., this quarter's logged value is the same as last quarter plus a small amount, log(1+rt). This attribute of the data is displayed by the bottom plot of [Figure 1.1](#chapter1#fig1_1). The code to plot the data for this example is:


`library(**astsa**)   _# astsa should be loaded before each session_`
`**tsplot**(cbind(jj, log(jj)), ylab=c("USD", "log(USD)"), type="o", col=4,`
`   main="Johnson & Johnson QEPS")`

Example 1.2 Dow Jones Industrial Average [Return to text.⏎](#chapter3#b3exam1_2)

As another example of financial time series data, [Figure 1.2](#chapter1#fig1_2) shows the trading day closings and returns (or percent change) of the Dow Jones Industrial Average (DJIA) from 2006 to 2016\. If _xt_ is the value of the DJIA closing on day _t_, then the return is

rt\=(xt−xt−1)/xt−1.

![Dow Jones Industrial Average daily closings and the daily returns](./images/fig1_2.jpg)

Figure 1.2: Dow Jones Industrial Average (DJIA) trading days closings (top) and returns (bottom) from April 20, 2006 to April 20, 2016. [Return to text.⏎](chapter1)

This means that 1+rt\=xt/xt−1 and

log(1+rt)\=log(xt/xt−1)\=log(xt)−log(xt−1),

4\. just as in [Example 1.1](#chapter1#exam1_1). Noting the expansion (see [Section A.10](#appA#secA_10))

log(1+r)\=r−r22+r33−⋯,−1<r≤1,

we see that if _r_ is close to zero, the higher-order terms will be negligible. Consequently, because for financial data, _rt_ is typically a small percentage, we have

log(1+rt)≈rt.

Note the financial crisis of 2008 in [Figure 1.2](#chapter1#fig1_2). The data shown are typical of return data. The mean of the series appears to be stable with an average return of approximately zero, however, the _volatility_ (or variability) of the data exhibits clustering; that is, highly volatile periods tend to be clustered together. A problem in the analysis of these types of financial data is to forecast the volatility of future returns. Various models have been developed to handle these problems, and we discuss some in [Chapter 8](#chapter8). The data set is an xts data file, which can be used to handle observations taken at irregular time intervals such as trading days.


`library(xts)   _# install if necessary_`
`djia_return = diff(log(**djia$Close**))`
`par(mfrow=2:1)`
`plot(**djia$Close**, col=4)`
`plot(djia_return, col=4)`

It is possible to produce a similar graphic using astsa without having to load xts, for example:


`Close = ts(**djia**[,"Close"])               _# make a time series object_`
`x = cbind(Close, Return=diff(log(Close))) _# now columns are aligned_`
`**tsplot**(**timex**(**Djia**), x, col=4, lwd=2, main="DJIA")`

You can see a comparison of _rt_ and log(1+rt) in [Figure 1.3](#chapter1#fig1_3), which shows the seasonally adjusted quarterly growth rate, _rt_, of US GDP compared to the version obtained by calculating the difference of the logged data.5\. 


`**tsplot**(diff(log(**gdp**)), type="o", col=4, ylab="**GDP Growth**") _# diff-log_`
`points(diff(**gdp**)/lag(**gdp**,-1), pch=3, col=6)           _# actual return_`

![US GDP growth rate calculated using the actual returns and using the difference of the log-transformed data showing they are nearly identical](./images/fig1_3.jpg)

Figure 1.3: US GDP growth rate, actual values (+) and by differencing the logs (–∘–). [Return to text.⏎](chapter1)

It turns out that many time series behave like this, so that logging the data and then taking successive differences is a standard data transformation in time series analysis.

Example 1.3 Global Warming and Climate Change [Return to text.⏎](#chapter2#b2exam1_3)

Two global temperature records are shown in [Figure 1.4](#chapter1#fig1_4). The data are annual temperature anomalies averaged over the Earth's land area, and averaged over the surface of the ocean that is free of ice at all times (open ocean). The time period is 1850–2023, and the values are deviations (∘C) from the 1991–2020 average, updated from [Hansen et al. (2006)](#bibref1#refbib_23). The upward trend in both series during the latter part of the twentieth century has been used as an argument for climate change. Most climate scientists agree the main cause of the global warming trend is human expansion of the greenhouse effect. Note that the trend is not linear, with periods of leveling off and then sharp upward trends. It should be obvious that fitting a straight line via simple linear regression of either series (_xt_) on time (_t_), xt\=α+βt+ϵt, would not yield an accurate description of the trend. The code for the graphic is:


`**tsplot**(cbind(**gtemp_land**, **gtemp_ocean**), **spaghetti**=TRUE, lwd=2, pch=20, type="o",`
`   col=**astsa.col**(c(4,2),.5), ylab="Temperature Deviations (\u00B0C)",`
`   main="Global Warming", **addLegend**=TRUE, **location**="topleft", legend=c("Land`
`   Surface", "Sea Surface"))`

![yearly global land  and ocean surface temperature deviations in degrees Celsius](./images/fig1_4.jpg)

Figure 1.4: Yearly average global land surface and ocean surface temperature deviations (1850–2023) in ∘C. [Return to text.⏎](chapter1)

Example 1.4 6\. El Niño – Southern Oscillation (ENSO) [Return to text.⏎](#chapter2#b2exam1_4)

The Southern Oscillation Index (SOI) measures changes in air pressure related to sea surface temperatures in the central Pacific Ocean. The central Pacific warms every two to seven years due to the ENSO effect, which has been blamed for various global extreme weather events. During El Niño, pressure over the eastern and western Pacific reverses, causing the trade winds to diminish and leading to an eastward movement of warm water along the equator. As a result, the surface waters of the central and eastern Pacific warm with far-reaching consequences to weather patterns.

[Figure 1.5](#chapter1#fig1_5) shows monthly values of the Southern Oscillation Index (SOI) and associated Recruitment (an index of the number of young fish entering the population as fishable stock). Both series are for a period of 453 months ranging over the years 1950–1987\. They both exhibit an obvious annual cycle (hot in the summer, cold in the winter), and, though difficult to see, a slower periodic component of roughly four years. The study of the kinds of cycles and their strengths is the subject of [Chapters 6](#chapter6) and [7](#chapter7). The two series are related because fish population size is affected by ocean temperature.

![Comparison of the monthly Southern Oscillation Index and Recruitment (an index of the number of new fish)](./images/fig1_5.jpg)

Figure 1.5: Monthly SOI and Recruitment (estimated new fish), 1950–1987. [Return to text.⏎](chapter1)

The following code will reproduce [Figure 1.5](#chapter1#fig1_5):


`par(mfrow = 2:1)`
`**tsplot**(**soi**, ylab=NA, main="Southern Oscillation Index", col=4)`
` text(1969, .91, "COOL", col=5, font=4)`
` text(1969,-.91, "WARM", col=6, font=4)`
`**tsplot**(**rec**, ylab=NA, main="Recruitment", col=4)`

Example 1.5 7\. Predator–Prey Interactions [Return to text.⏎](#chapter3#b3exam1_5)

While it is clear that predators influence the numbers of their prey, prey affect the number of predators because when prey become scarce, predators may die of starvation or fail to reproduce. Such relationships are often modeled by the Lotka–Volterra equations, which are a pair of simple nonlinear differential equations (e.g., see [Edelstein-Keshet, 2005, ch.6](#bibref1#refbib_14), and [Example 3.8](#chapter3#exam3_8)).

One of the classic studies of predator–prey interactions is the snowshoe hare and lynx pelts purchased by the Hudson's Bay Company of Canada. While this is an indirect measure of predation, the assumption is that there is a direct relationship between the number of pelts collected and the number of hare and lynx in the wild. These predator–prey interactions often lead to cyclical patterns of predator and prey abundance seen in [Figure 1.6](#chapter1#fig1_6). Notice that the lynx and hare population sizes are asymmetric in that they tend to increase slowly and decrease quickly (↗↓).

![Annual Lynx-Hare predator-prey interactions based on the number of pelts purchased by the Hudson's Bay Company of Canada](./images/fig1_6.jpg)

Figure 1.6: Time series of the predator–prey interactions between the snowshoe hare and lynx pelts purchased by the Hudson's Bay Company of Canada. It is assumed there is a direct relationship between the number of pelts collected and the number of hare and lynx in the wild. [Return to text.⏎](chapter1)

The lynx prey varies from small rodents to deer, with the snowshoe hare being its overwhelmingly favored prey. In fact, lynx are so closely tied to the snowshoe hare that its population rises and falls with that of the hare, even though other food sources may be abundant. In this case, it seems reasonable to model the size of the lynx population in terms of the snowshoe population. This idea is explored further in [Example 3.8](#chapter3#exam3_8). [Figure 1.6](#chapter1#fig1_6) may be reproduced as follows.


`**tsplot**(cbind(**Hare**, **Lynx**), col=**astsa.col**(c(2,4),.5), lwd=2, type="o",`
`   pch=c(0,2), ylab="Number", **spaghetti**=TRUE, **addLegend**=TRUE)`
`mtext("(\u00D71000)", side=2, adj=1, line=1.5, cex=.8)`

Example 1.6 8\. Functional MRI Experiment [Return to text.⏎](chapter1)

Often, time series are observed under varying experimental conditions or treatment configurations. Such a set of series is shown in [Figure 1.7](#chapter1#fig1_7), where data are collected from various locations in the brain via functional magnetic resonance imaging (fMRI).

![fMRI data from 6 locations in the brain from an examining  the effects of general anesthesia on pain perception](./images/fig1_7.jpg)

Figure 1.7: fMRI data from two locations in the cortex, the thalamus, and the cerebellum; n\=128 points, one observation taken every 2 seconds. The step line represents the presence or absence of the stimulus. [Return to text.⏎](chapter1)

In fMRI, subjects are put into an MRI scanner and a stimulus is applied for a period of time, and then stopped. This on-off application of a stimulus is repeated and recorded by measuring the blood oxygenation-level dependent (bold) signal intensity, which measures areas of activation in the brain. The bold contrast results from changing regional blood concentrations of oxy- and deoxy-hemoglobin.

The data displayed in [Figure 1.7](#chapter1#fig1_7) are from an experiment that used fMRI to examine the effects of general anesthesia on pain perception by comparing results from anesthetized volunteers while a supramaximal shock stimulus was applied. This stimulus was used to simulate surgical incision without inflicting tissue damage. In this example, the stimulus was applied for 32 seconds and 9\. then stopped for 32 seconds, so that the signal period is 64 seconds. The sampling rate was one observation every 2 seconds for 256 seconds (n\=128).

Notice that the periodicities appear strongly in the motor cortex series but seem to be missing in the thalamus and perhaps in the cerebellum. In this case, it is of interest to statistically determine if the areas in the thalamus and cerebellum are actually responding to the stimulus. Use the following commands for the graphic:


`par(mfrow=c(3,1), cex=.8)`
`x        = ts(**fmri1**[,4:9], start=0, freq=32)`
`names    = c("Cortex","Thalamus","Cerebellum")`
`stimulus = ts(rep(c(rep(.6,16), rep(-.6,16)), 4), start=0, freq=32)`
`for (i in 1:3){`
` j = 2*i-1`
` **tsplot**(x[,j:(j+1)], ylab="BOLD", xlab=NA, main=names[i], col=5:6, lwd=2,`
`   ylim=c(-.6,.6), xaxt="n", **spaghetti**=TRUE)`
` axis(seq(0, 256, 64), side=1, at=0:4)`
` lines(stimulus, type="s", col=gray(.4)) }`
`mtext("seconds", side=1, line=1.75)`

## 1.3 Time Series Models

A primary objective of time series analysis is to develop mathematical models that provide plausible descriptions for sample data like those we have previously seen. The fundamental visual characteristic distinguishing the different series shown in [Example 1.1](#chapter1#exam1_1) – [Example 1.6](#chapter1#exam1_6) is their differing degrees of smoothness. An explanation for this smoothness is that adjacent points in time are correlated, and the value of the series depends in some way on its past. This idea expresses a fundamental way in which we might think about generating realistic-looking time series.

Example 1.7 White Noise [Return to text.⏎](chapter1)

A simple kind of generated series might be a collection of _uncorrelated_ random variables, _wt_, with mean 0 and finite variance σw2. The time series generated from uncorrelated variables is used as a model for noise in engineering applications where it is called _white noise_; we shall sometimes denote this process as wt∼wn(0,σw2). The designation white originates from the analogy with white light, which is the combination of all colors of the visible light spectrum at equal strength (more details can be found in [Chapter 6](#chapter6)). A special version of white noise that we use is when the variables are independent and identically distributed normals, written as wt∼ iid N(0,σw2).

The upper panel of [Figure 1.8](#chapter1#fig1_8) shows a collection of 250 independent standard normal random variables (σw2\=1), plotted in the order they were drawn. 10\. The resulting series bears a resemblance to portions of the DJIA returns in [Figure 1.2](#chapter1#fig1_2).

![Simulated white noise and a moving average showing that averaging decreases the variability (smoothes) the data](./images/fig1_8.jpg)

Figure 1.8: Gaussian white noise series (top) and three-point moving average of the Gaussian white noise series (bottom). [Return to text.⏎](chapter1)

If the stochastic behavior of all time series could be explained in terms of the white noise model, classical statistical methods would suffice. Two ways of introducing serial correlation and more smoothness into time series models are given in [Example 1.8](#chapter1#exam1_8) and [Example 1.9](#chapter1#exam1_9).

Example 1.8 Moving Averages, Smoothing and Filtering [Return to text.⏎](chapter1)

Consider replacing _wt_ in [Example 1.7](#chapter1#exam1_7) with a three-point moving average given by

vt\=13(wt−1+wt+wt+1).

The resulting series is shown in the lower panel of [Figure 1.8](#chapter1#fig1_8). This series is much smoother than the white noise series and has a smaller variance due to averaging. It should also be apparent that averaging removes some of the high frequency (fast oscillations) behavior of the noise. We begin to notice a similarity to some of the noncyclic fMRI series in [Figure 1.7](#chapter1#fig1_7).

A linear combination of values in a time series is referred to, generically, as a filtered series, hence the command filter.[1](#chapter1#fn1_1) To reproduce [Figure 1.8](#chapter1#fig1_8):11\. 


`set.seed(123456789)`
`w = rnorm(250)                    _# 250 N(0,1) variates_`
`v = filter(w, filter=rep(1/3,3)) _# moving average_`
`**tsplot**(cbind(w, v), col=4, ylim=c(-4, 4), las=1, **gg**=TRUE, title=c("white`
`   noise","moving average"))`

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 1If the package dplyr is loaded, see the warning in _Hints for Selected Exercises_. [Return to text.⏎](#chapter1#fn1_11b)

The SOI and Recruitment series in [Figure 1.5](#chapter1#fig1_5), as well as some of the fMRI series in [Figure 1.7](#chapter1#fig1_7), differ from the moving average series because they are dominated by an oscillatory behavior. A number of methods exist for generating series with this quasi-periodic behavior; we illustrate a popular one based on the autoregressive model.

Example 1.9 Autoregressions [Return to text.⏎](chapter1)

Suppose we consider the white noise series _wt_ of [Example 1.7](#chapter1#exam1_7) as input and calculate the output using the second-order equation

xt\=1.5xt−1−.75xt−2+wt(1.1)

successively for t\=1,2,…,250. The resulting output series is shown in [Figure 1.9](#chapter1#fig1_9). Equation (1.1) represents a regression or prediction of the current value _xt_ of a time series as a function of the past two values of the series, and hence the term _autoregression_. A problem with startup values exists here because (1.1) also depends on the initial conditions _x_0 and x−1, but for now we set them to zero. We can then generate data _recursively_ by substituting into (1.1). That is, given w1,w2,…,w250, we could set x−1\=x0\=0 and then start at t\=1:

x1\=1.5x0−.75x−1+w1\= w1x2\=1.5x1−.75x0+w2\= 1.5w1+w2x3\=1.5x2−.75x1+w3x4\=1.5x3−.75x2+w4

![Simulated autoregression showing how the model can capture periodic behavior](./images/fig1_9.jpg)

Figure 1.9: Autoregressive series generated from model (1.1). [Return to text.⏎](chapter1)

and so on. We note the approximate periodic behavior of the series, which 12\. is similar to that displayed by the SOI and Recruitment in [Figure 1.5](#chapter1#fig1_5) and some fMRI series in [Figure 1.7](#chapter1#fig1_7). This particular model is chosen so that the data have pseudo-cyclic behavior of about 1 cycle every 12 points; thus 250 observations should contain about 20 cycles. This autoregressive model and its generalizations can be used as an underlying model for many observed series and will be studied in detail in [Chapter 4](#chapter4).

One way to simulate and plot data from the model (1.1) in R is to use the following commands. The initial conditions are set equal to zero, so note that at least _x_1 and _x_2 do not satisfy (1.1); more details can be found in [Problem 4.2](#chapter4#question4_2). Consequently, we let the filter run an extra 50 values to avoid startup problems.


`set.seed(90210)`
`w = rnorm(250 + 50) _# 50 extra to avoid startup problems_`
`x = filter(w, filter=c(1.5,-.75), method="recursive")[-(1:50)]`
`**tsplot**(x, main="autoregression", col=4, **gg**=TRUE)`

Example 1.10 Random Walk with Drift [Return to text.⏎](chapter1)

A model for analyzing a trend such as seen in the global temperature data in [Figure 1.4](#chapter1#fig1_4) is the random walk with drift model given by

xt\=δ+xt−1+wt(1.2)

for t\=1,2,…, with initial condition x0\=0,[2](#chapter1#fn1_2) and where _wt_ is white noise. The constant _δ_ is called the drift, and when δ\=0, the model is called simply a random walk because the value of the time series at time _t_ is the value of the series at time t−1 plus a completely random movement determined by _wt_. Note that we may rewrite (1.2) as a cumulative sum of white noise variates. That is,

xt\=δ t+∑j\=1twj(1.3)

for t\=1,2,…; either use induction, or plug (1.3) into (1.2) to verify this statement. [Figure 1.10](#chapter1#fig1_10) shows 200 observations generated from the model with δ\=0 and. 3, and with standard normal noise. For comparison, we also superimposed the straight lines δt on the graph. To reproduce [Figure 1.10](#chapter1#fig1_10), use the following code (notice the use of multiple commands per line using a semicolon).


`set.seed(314159265)`
`w = rnorm(200); x = cumsum(w)     _# RW_`
`wd = w +.3;      xd = cumsum(wd) _# RW with drift_`
`**tsplot**(cbind(x, xd), main="random walk", col=2*1:2, **gg**=TRUE, **spaghetti**=TRUE)`
`clip(0,200, 0,80); abline(a=0, b=.3, h=0, lty=2, col=2*2:1)   _# drifts_`

![Simulated random walk with and without drift showing how the variability in the data increases with time](./images/fig1_10.jpg)

Figure 1.10: Random walk, σw\=1, with drift δ\=.3 (upper jagged line), without drift, δ\=0 (lower jagged line), and dashed lines showing the drifts. [Return to text.⏎](chapter1)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ 2Setting x0\=0 is for convenience. It can be any constant or a random variable. [Return to text.⏎](#chapter1#fn1_21b)

Example 1.11 13\. Signal Plus Noise [Return to text.⏎](#chapter2#b2exam1_11)

Many realistic models for generating time series assume an underlying signal with some consistent periodic variation contaminated by noise. For example, it is easy to detect the regular cycle fMRI series displayed on the top of [Figure 1.7](#chapter1#fig1_7). Consider the model

xt\=2cos(2πt+1550)+wt(1.4)

for t\=1,2,…,500, where the first term is regarded as the signal and is shown in the upper panel of [Figure 1.11](#chapter1#fig1_11). We note that a sinusoidal waveform can be written as

Acos(2πωt+ϕ),(1.5)

![Simulated cosines without noise and with varying levels of noise showing how noise attenuates the signal](./images/fig1_11.jpg)

Figure 1.11: Cosine wave with period 50 points (top panel) compared with the cosine wave contaminated with additive white Gaussian noise, σw\=1 (middle panel) and σw\=5 (bottom panel). [Return to text.⏎](chapter1)

where _A_ is the amplitude, _ω_ is the frequency of oscillation, and _ϕ_ is a phase shift. In (1.4), A\=2, ω\=1/50 (one cycle every 50 time points), and ϕ\=.6π.

An additive noise term was taken to be white noise with σw\=1 (middle panel) and σw\=5 (bottom panel) drawn from a normal distribution. Adding the two together obscures the signal as shown in the lower panels of [Figure 1.11](#chapter1#fig1_11). The degree to which the signal is obscured depends on the amplitude of the signal relative to the size of _σw_. The ratio of the amplitude of the signal to _σw_ (or some function of the ratio) is sometimes called the _signal-to-noise ratio (SNR)_; the larger the SNR, the easier it is to detect the signal. Note that the signal is easily discernible in the middle panel, whereas the signal is nearly annihilated in the bottom panel. Typically, we will not observe the signal but the signal obscured by noise.

To reproduce [Figure 1.11](#chapter1#fig1_11), use the following commands:


`cs   = 2*cos(2*pi*(1:500 + 15)/50)       _# signal_`
`w    = rnorm(500)                        _# noise_`
`cos0 = bquote(2*cos(2*pi*(t+15)/50))     _# titles_`
`cos1 = bquote(.(cos0) + N(0,1))`
`cos5 = bquote(.(cos0) + N(0,5^2))`
`**tsplot**(cbind(cs, cs+w, cs+5*w), col=4,   ylab=NA, xlab=c(NA,NA,"Time"), las=1,`
`   **gg**=TRUE, title=c(cos0, cos1, cos5))`

## 1.4 14\. Random Number Generation\*

We do numerical simulations often in this text. For example, we have already used such commands as set.seed and rnorm in examples where we generated data. Truly random numbers are not used in simulations, instead _pseudo-random_ numbers are used for convenience and for reproducibility. The values generated by a random number generator (RNG) are deterministic, but they have the appearance of being random (e.g., they will pass statistical tests of randomness).

Although our focus is on stochastic processes, it is worthwhile discussing random number generation because we use it so often. Most statistical software packages include a number of different RNGs that are more complicated and refined than what we will present here.

15\. For the most part, deterministic generators _G_ are recursive in that given _k_ previous numbers, xn−1,…,xn−k, the next number _xn_ is generated as

xn\=G(xn−1,…,xn−k),

for n\=1,2,…, which is started from a given _seed_ (x0,…,x1−k). A quick-and-dirty generator is the linear congruential generator (LCG) that generates values in the set S\={0,1,…,m−1}, where _m_ is a large integer. Given an initial value (the seed) _x_0, generate numbers {xn} recursively according to

xn\=axn−1+c(modm),

for n\=1,2,…,N, where _N_ is the number of desired values. The choice of these numbers must be done carefully, and various values have been suggested. In [Press et al. (2007)](#bibref1#refbib_37), it is mentioned that the values a\=1664525, c\=1013904223, and m\=232 are about as good as any other 32-bit LCG.

_Sequences generated with the same seed will be identical._ Since the state space is finite, _xn_ must eventually return to a previous value. The smallest integer _p_ such that for some state the sequence returns to that state after _p_ iterations is called the period of the generator. Longer periods are better than shorter periods, but a long period by itself does ensure a good RNG (e.g., an LCG with a\=c\=1 will have a very long period but is a terrible RNG). The period of the LCG discussed above is only 230 (a little more than a billion). The following is another example of a very bad (p\=1) 32-bit LCG:


`x = c(1) _# the bad seed (they are all bad)_`
`for (n in 2:30){ x[n] = (12*x[n-1] + 4) %% 2^32 }`
`x`
`  [1]          1         16        196       2356     28276     339316`
`  [7]    4071796   48861556 586338676 2741096820 2828390772 3875918196`
` [13] 3561345396 4081439092 1732628852 3611677044 390451572 390451572`
` [19] 390451572 390451572 390451572 390451572 390451572 390451572`
` [25] 390451572 390451572 390451572 390451572 390451572 390451572`

With the simulations done today, the period of LCGs can be too short to appear sufficiently random. Most current analytical software systems use more sophisticated generators. For example, the default RNG in R is the _Mersenne Twister_ algorithm that has a long period of 219937−1 (which is a Mersenne prime).

Most samples generated from specified distributions are simulated from standard uniforms, in which case we could use

un\=xn/m.

Many of us were introduced to the method of using uniforms to generate other random variables via the probability integral transform that states if _X_ is continuous with cdf _F_, then U\=F(X) is standard uniform. Then, given a value of _U_, we can obtain X\=F−1(U). The probability integral transform, however, is 16\. generally not a very efficient method of random number generation. For example, a better method for generating standard normals is to use the fact that if _U_1 and _U_2 are independent standard uniforms, U(0,1), then

X1\=−2log(U1)cos(2πU2)andX2\=−2log(U1)sin(2πU2)

are independent standard normals, N(0,1). This technique is called the Box-Muller transform; it does have some problems because computer-generated values can only get so close to zero, which limits the magnitudes of _X_1 and _X_2. The technique, however, can be modified. Further details on the subject of RNG and sampling from nonuniform distributions may be found in [Gentle (2003)](#bibref1#refbib_19) or [Press et al. (2007, ch. 7)](#bibref1#refbib_37).

## Problems

* 1.1.  
   1. Generate n\=100 observations from the autoregression  
   xt\=−.9xt−2+wt  
   with σw\=1, using the method described in [Example 1.9](#chapter1#exam1_9). Next, apply the moving average filter  
   vt\=(xt+xt−1+xt−2+xt−3)/4  
   to _xt_, the data you generated. Now plot _xt_ as a line and superimpose _vt_ as a dashed line.  
   2. Repeat (a) but with  
   xt\=2cos(2πt/4)+wt,  
   where wt∼ iid N(0,1).  
   3. Repeat (a) but where _xt_ is the log of the Johnson & Johnson data discussed in [Example 1.1](#chapter1#exam1_1).  
   4. What is seasonal adjustment (you can do an internet search)?  
   5. State your conclusions (in other words, what did you learn from this exercise).
* 1.2. There are a number of seismic recordings from earthquakes and from mining explosions in **astsa**. All of the data are in the dataframe **eqexp**, but two specific recordings are in **EQ5** and **EXP6**, the fifth earthquake and the sixth explosion, respectively. The data represent two phases or arrivals along the surface, denoted 17\. by P(t\=1,…,1024) and S (t\=1025,…,2048), at a seismic recording station. The recording instruments are in Scandinavia and monitor a Russian nuclear testing site. The general problem of interest is in distinguishing between these waveforms in order to maintain a comprehensive nuclear test ban treaty.  
To compare the earthquake and explosion signals,  
   1. Plot the two series separately in a multifigure plot with two rows and one column.  
   2. Plot the two series on the same graph using different colors or different line types.  
   3. In what way are the earthquake and explosion series different?
* 1.3. In this problem, we explore the difference between random walk and moving average models.  
   1. Generate and plot _nine_ series that are random walks (see [Example 1.10](#chapter1#exam1_10)) of length n\=500 without drift (δ\=0) and σw\=1.  
   2. Generate and plot _nine_ series of length n\=500 that are moving averages of the form _vt_ as defined in [Example 1.8](#chapter1#exam1_8).  
   3. Comment on the differences between the results of part (a) and part (b).
* 1.4. The data in **GDP** are seasonally adjusted quarterly U.S. GDP from 1947-I to 2023-I; note that **GDP** is an updated version of **gdp**. The time period includes the COVID-19 epidemic. The growth rate prior to the epidemic is shown in [Figure 1.3](#chapter1#fig1_3).  
   1. Plot the data and compare it to one of the models discussed in [Section 1.3](#chapter1#sec1_3).  
   2. Reproduce [Figure 1.3](#chapter1#fig1_3) using the **GDP** series. Then, comment on the difference between the two methods of calculating growth rate when the values start to get large in magnitude.  
   3. Which of the models discussed in [Section 1.3](#chapter1#sec1_3) best describe the behavior of the _growth rate_ in U.S. GDP?18 is blank.

---
