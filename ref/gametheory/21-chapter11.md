# 11Auctions with Affiliated Valuations

DOI: [10.1201/b23262-11](./https___doi.org_10.1201_b23262-11.md)

## 11.1 Introduction

The previous chapter assumed that bidder's valuations are independent of each other. This substantially simplified both the game theory and the econometrics. In this chapter we are going to relax this assumption. Now we will allow for bidder valuations to be dependent. These types of auctions are called **affiliated private values auctions**.

The conical example is bidding on the right to drill for oil in the US's outer continental shelf (OCS). The amount of oil that can be pulled out of the ground has nothing to do with the bidder. Similarly, the price of that oil has little or nothing to do with the identity of the bidder.[1](./21-chapter11.md#fn11_1)

The chapter analyzes **common value auctions** and takes the game theory model to OCS auctions. The chapters asks whether bid rings should be allowed in these auctions.

## 11.2 Auctions with Common Values

Here we consider auctions where the bidder's valuations are interdependent. The simplest case of such interdependence is the pure **common value auction**. In this auction, bidder's don't know the exact value of the item they are bidding on. Each bidder draws a signal about the true value of the item such that if all the signals were aggregated, then that would provide a pretty close approximation of the true value of the item.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./21-chapter11.md#fn11_111b)Of course, some bidders may have market power in the oil market.

The classic example is bidders considering how much to bid for the rights to drill for oil in a particular area. The oil field has a particular amount of oil worth a particular value. While unknown, it has nothing to do with who is bidding. The value of the oil field is the same to all bidders. They just don't know what it is. Oil companies, the bidders, will hire different geologists with different models, equipment, and expertise to estimate the value of the oil field. These geologists will come up with different guesses. We call these guesses, signals of the oil field's value.

The section presents the game and the **Bayes Nash equilibrium** of the auction. It discusses an idea called the **winner's curse**, which leads bidders to shade their bid down in equilibrium. The section presents an estimator for **common value auctions** and illustrates the estimator using simulated data.

### 11.2.1 Simple Model

The auction game has _N_ bidders with signals _si_. The signals are drawn from a normal distribution with mean _v_ and variance _σ_2. The true value of the item is _v_. Each bidder's bid is a function of their signal, bi(si). The bidder with the highest bid wins the auction and pays their bid. If the bidder wins, they get the true value of the item, _v_. If the bidder loses, they get nothing.

* Players: _N_ bidders and signals (types) si∈ℜ.
* Strategies: Each bidder _i_ observes a signal (_si_) and chooses a bid bi(si).
* Payoffs:  
   1. bi\>bj∀j≠i: v−bi(si)  
   2. If ∃j s.t. bi<bj: 0
* Beliefs: si∼N(v,σ2), where _v_ is the true value of the item.

If the player _i_ wins, they get _v_ which doesn't have a subscript. This is because it is the same for every bidder. What varies is between bidders, is their signal _si_. Bidders know that their signal is drawn from a normal distribution with mean _v_ but they do not know _v_.

### 11.2.2 Winner's Curse

The winner is the bidder with the highest bid. Consider what happens if all bidders bid their valuations. In that case, the winner bids sN:N (using order statistic notation) which is going to be substantially higher than _v_.


`_>   set.seed(123456789)_`
`_>   N = 10_`
`_>   v = 5_`
`_>   s = rnorm(N, v)_`
` `
`_> max(s)_`
`  [1] 6.415538`
In the example with 10 bidders, the winner bids 6.42, which much higher than the true value 5\. The winner really loses. The bidders should adjust their strategy by shading their bid down in order to account for the winner's curse.

### 11.2.3 Bayes Nash Equilibrium

We often see common value auctions that are also **first price auctions**. Bidders are shading their bids for two different reasons. First, as in the previous chapter, they are shading their bid because it is a first price auction and they need to account for winning and paying what they bid. Second, these bidders have to account for the fact that if they win the auction it is because their signal is higher than everyone else's and thus higher than the true value of the item.

To make things simpler, let's assume that bidders bid their expected value for the item. Assume that bidders use **Bayes rule** and their expected value is conditional on both the signal that they observe and on the fact that they won the auction. Of course, they don't actually know whether they win the auction or not, but their bid is of no importance if they lose. The bidder is thinking through the cases. The only case of interest is the case where the bidder won the auction. They realize that if they won the auction, their bid must, by definition, have been the highest bid.

Again, trading off winning and the amount paid is left out for the moment. So how much information does the bidder have? You may think that they don't have much because all they have is one signal, but it turns out that they know a lot more than that. Because they won the auction, they know that their signal must be higher than everyone else's. So that tells them quite a lot about every body else's signal.

[Chapter 10](./20-chapter10.md) illustrates how order statistics work. The probability that a particular signal _s_ is the highest order statistic (sN:N) is given by the following equation assuming that all the signals are drawn independently from the distribution F(s|μ,σ). The probability that a particular _s_ is higher than all the other signals is then G(s|N,μ,σ)\=F(s|μ,σ)N−1 where _μ_ represents the different possible values of the true value _v_ and there are N−1 other signals. The derivative is then g(s|N,μ,σ)\=(N−1)f(s|μ,σ)F(s|μ,σ)N−2.

We can use this likelihood to determine the bidder's expected value for the item conditional on submitting the winning bid. We just need to use Bayes rule. To determine the expected value, we need the probability that a particular distribution is generating the signals we observe.

γ(μ,σ|s\=sN:N)\=g(s|N,μ,σ)∑μ′,σ′g(s|N,μ′,σ′)(11.1)

If we know all the possible _μ_s and _σ_s and assume a normal distribution and assume that the prior over _μ_s and _σ_s is uniform, then Equation (11.1) gives the probability that the observed signals are generated by a particular _μ_ and _σ_.

Consider a simple version where μ\=v, the true value and σ\=1. In this case g(s|N,v)\=(N−1)ϕ(s−v)Φ(s−v)N−2 and ϕ() and Φ represent the standard normal's density and probability function, respectively.

The expected value of the item given that the bidder has the highest signal, s\=sN:N, is then given by following function.

E(v|N,s)\=∫v′v′γ(v′|s\=sN:N)d(v′)(11.2)

where γ() is defined in Equation (11.1). Given _N_ bidders and a signal _s_ the expected value of the true bid is the integral over the possible true values weighted by the γ() function assuming that _s_ is the highest signal. Remember σ\=1.

Now we have how much the bidder values the item, the next question is how much to bid. Remember this is a sealed bid auction, so it is still the case that the bidder is trading off the value of winning against the probability of winning.

bi(si)\=E(v|N,si)−G(bi|N,si)g(bi|N,si)(11.3)

So we need to determine the probability of winning conditional on the bidder's signal. The simplest assumption to make is that equilibrium bids are monotonically increasing in the signal. This doesn't seem unreasonable.

Given our monotonicity assumption, the probability of winning is just the probability of having the highest signal. We have that G(bi|N,si)\=F(si|v)N−1 and g(bi|N,si)\=(N−1)f(si|v)F(si|v)N−2, which not coincidentally is similar function that we defined previously.

### 11.2.4 Common Value Auction in **R**

Below we create all the probability functions that we need for bidding in first price auctions with common values. It is a little confusing because we are using order statistics for two different things. First, there is the standard method discussed in the previous chapter where the bidder is determining the optimal bid for a first price auction. Second, the bidder is using order statistics to back out their expectation of the value for the item given the signal they observed and conditional upon winning the auction.

The signal distribution is a normal distribution denoted _F_, where _f_ is the density. We use log\_F() and log\_f(). Given N bidders, the distribution of the highest signal is denoted by _G_ and we use log\_G() and log\_g() for the density. The function dnorm() calculates the density of the normal distribution and pnorm() calculates the probability of the normal distribution.


`_>   logf = function(s, v, sigma = 1)_`
`_+     log(dnorm(s, v, sigma))_`
`_>   logF = function(s, v, sigma = 1)_`
`_+     log(pnorm(s, v, sigma))_`
`_>   logG = function(s, v, sigma = 1, N) (N-1)*logF(s, v, sigma)_`
`_>   logg = function(s, v, sigma=1, N) log(N-1) +_`
`_+     logf(s, v, sigma) +_`
`_+     (N-2)*logF(s, v, sigma)_`
Given these probabilities, we can determine the bidder's expected value for the item and their bid. The expectation function E\_fun() takes in two global variables u and sig. These are the possible parameters of the normal distribution determining the signals observed by the bidders. Lastly, there is the bid function b\_fun() based on Equation (11.3).


`_>   Efun = function(s, N) {_`
`_+     gu = matrix(NA,length(u),length(sig))_`
`_+     umat = matrix(NA, length(u), length(sig))_`
`_+     for(j in 1:length(sig)) {_`
`_+       gu[,j] = exp(logg(s, u, sig[j], N))_`
`_+       umat[,j] = u_`
`_+     }_`
`_+     sumgu = sum(gu)_`
`_+     gammau = gu/sumgu_`
`_+     mu = sum(umat*gammau)_`
`_+     sigma = sqrt(sum(umat^2*gammau) - mu^2)_`
`_+     return(list(mu=mu, sigma=sigma))_`
`_+   }_`
`_>   bfun = function(s, N) {_`
`_+     vbar = Efun(s, N)_`
`_+     G = exp(logG(s, vbar$mu, vbar$sigma, N))_`
`_+     g = exp(logg(s, vbar$mu, vbar$sigma, N))_`
`_+     return(vbar$mu - G/g)_`
`_+   }_`
### 11.2.5 Simulation of Common Value Auction using **R**

It helps to work through a simulation. There are 100 auctions and the number of bidders varies. The true value is 0 for each auction, and the signal is distributed standard normal. The function seq() calculates a sequence of numbers between the first and second values with the third value as the step. The function rnorm() generates random numbers from a normal distribution. The function sample() randomly samples from a set of numbers. The function rep() repeats a number a certain number of times.

The code uses sapply() to loop through the signals and calculate the expected value and bid for each signal.


`_>   set.seed(123456789)_`
`_>   M = 100_`
`_>   N = NULL_`
`_>   bids = NULL_`
`_>   ids = NULL_`
`_>   values = NULL_`
`_>   u = seq(-10, 10, 0.15)_`
`_>   sig = seq(0.1, 3, 0.15)_`
`_>   Ns = sample(3:4, M, replace=TRUE)_`
`_>   v = rep(0, M)_`
`_>   sigma = 1_`
`_>   for(i in 1:M) {_`
`_+     ids = c(ids, rep(i, Ns[i]))_`
`_+     N = c(N, rep(Ns[i], Ns[i]))_`
`_+     si = rnorm(Ns[i], v[i], sigma)_`
`_+     values = c(values_,`
`_+                sapply(1:length(si)_,`
`_+                               function(j) Efun(si[j]_,`
`_+                                                 Ns[i])$mu))_`
`_+     bids = c(bids_,`
`_+              sapply(1:length(si)_,`
`_+                           function(j) bfun(si[j]_,`
`_+                                             Ns[i])))_`
`_+   }_`
The code below creates a density plot of the bids and the expected values. The bids are shifted down from the expected values. The expected values are shifted down from the signal distribution in equilibrium.


`_> ggplotsimcvbids = data.frame(_`
`_+   bids = bids_,`
`_+   values = values_`
`_+ ) |>_`
`_+   ggplot(aes(bids)) +_`
`_+   geomdensity(alpha = 0.5) +_`
`_+   geomdensity(aes(values), linetype = 2, alpha = 0.5) +_`
`_+   labs(_`
`_+     x = “values/bids”_,`
`_+     y = “”_,`
`_+     title = “Density of bids and values”_`
`_+   ) +_`
`_+   geomvline(xintercept = 0, linetype = 2_,`
`_+              color = “gray”) +_`
`_+   geomtext(aes(x = -5, y = 0.2, label = “bids”)_,`
`_+             color = “gray”) +_`
`_+   geomtext(aes(x = 2, y = 0.2, label = “values”)_,`
`_+             color = “gray”) +_`
`_+   theme(axis.text.y=elementblank()_,`
`_+         axis.ticks.y=elementblank())_`
[Figure 11.1](./21-chapter11.md#fig11_1) presents the observed bids and the estimated values. As we saw in the IPV case, bids are significantly shaded down from values for first price auctions. Expected values are also significantly shaded down from the original signal observed by the bidder. These expected values are conditional on the signal assuming that signal is highest signal in the Bayes Nash equilibrium. The actual signal is distributed around zero.

![In the graph, the horizontal axis is labeled as values or bids and ranges from negative 6 to 2. The vertical axis shows the density. Two curves are shown. The solid line labeled bids peaks around negative 3 and falls steeply after that. The dashed line labeled values peaks slightly to the right of the bids curve near negative 1 and continues farther right toward positive 2. The values curve is wider and shifted to the right of the bids curve. All data are approximate.](./images/fig11_1.jpg)

[Figure 11.1](chapter11) Plot of the density of the bids and expected values in a first price common values auction. Bids are shifted down from valuations because it is a first price auction.

### 11.2.6 Estimator for Common Values Auctions

The estimator reverse engineers the signal distribution from the bid distribution. We know from Equation (11.3) and the discussion in [Chapter 10](./20-chapter10.md) that we can identify the distribution of the expected values conditional on the signal. Unfortunately, it not generally possible to uniquely determine the signal distribution from the expected value distribution. We will need to rely on parametric restrictions.

### 11.2.7 Common Values Estimator in **R**

To estimate the underlying signal distribution from the observed bids, we will combine maximum likelihood with simulation. The estimator chooses the _μ_ and _σ_ of the signal distribution that maximizes the likelihood of the observed bids given the game theory assumption that bidders bid their expected value conditional on their signal being the highest.

The estimator works by taking a set of parameter values for the distribution of signals, mu and sigma, and simulating the resulting bids, b\_sim. It simulates the signals, then loops through the simulated signals and creates the corresponding simulated bids. It then calculates the log likelihood of observing the observed bids (bids\_temp) given the derived parameters from the bid distribution. It does this for each size of auction in the data.


`_>   fbidml = function(mu, sigma, bidstemp, Ntemp, s) {_`
`_+     Ns = unique(Ntemp)_`
`_+     loglik = rep(NA, length(bidstemp))_`
`_+     for(i in 1:length(Ns)) {_`
`_+       Ni = Ns[i]_`
`_+       index = which(Ntemp==Ns[i])_`
`_+       bsim = sapply(1:length(s), function(i) bfun(s[i], Ni))_`
`_+       mui = mean(bsim, na.rm = TRUE)_`
`_+       sigmai = sd(bsim, na.rm = TRUE)_`
`_+       zi = (bidstemp[index] - mui)/sigmai_`
`_+       loglik[index] = log(dnorm(zi)) - log(sigmai)_`
`_+     }_`
`_+     return(loglik)_`
`_+   }_`
`_>   fbidmlint = function(par, bidstemp, Ntemp) {_`
`_+     set.seed(123456789)_`
`_+     mu = par[1]_`
`_+     sigma = exp(par[2])_`
`_+     s = U*sigma + mu_`
`_+     return(-sum(fbidml(mu, sigma, bidstemp, Ntemp, s)))_`
`_+   }_`
This estimator requires three global variables U, u, and sig.


`_> U = rnorm(1000)_`
`_> a = optim(par = c(0, log(sigma)), fbidmlint_,`
`_+            bidstemp = bids, Ntemp = N_,`
`_+           control = list(trace=0, maxit=100000))_`
The code below estimates the signal distribution from the observed bids.


`_> ggplotsimcvsignals =_`
`_+   data.frame(_`
`_+     signals = rnorm(length(values)_,`
`_+                     a$par[1]_,`
`_+                     exp(a$par[2]))_,`
`_+     values = values_`
`_+   ) |>_`
`_+     ggplot(aes(values)) +_`
`_+     geomdensity(alpha = 0.5) +_`
`_+     geomdensity(aes(signals), linetype = 2, alpha = 0.5) +_`
`_+     labs(_`
`_+       x = “values/signals”_,`
`_+       y = “”_,`
`_+       title = “Density of signals and values”_`
`_+     ) +_`
`_+     geomvline(xintercept = 0, linetype = 2, color = “gray”) +_`
`_+     geomtext(aes(x = -3.5, y = 0.2, label = “values”)_,`
`_+               color = “gray”) +_`
`_+     geomtext(aes(x = 2.5, y = 0.2, label = “signals (est.)”)_,`
`_+               color = “gray”) +_`
`_+     theme(axis.text.y=elementblank()_,`
`_+           axis.ticks.y=elementblank())_`
` `
`_> ggplotsimcvsignals_`
[Figure 11.2](./21-chapter11.md#fig11_2) presents the density of the expected values and the estimated signals in the simulated data. The estimated signals are pretty close to the true distribution, which is a standard normal distribution. The figure shows that the bidders substantially discount their bids from the observed signals. We see that this occurs for two reasons. First, their expected value (conditional on winning) is discounted from their signal. Second, because it is a first price auction, bidders discount their bid from the expected value of the item (see [Figure 11.1](./21-chapter11.md#fig11_1)).

![In the graph, the horizontal axis is labeled as values or signals and ranges from negative 4 to 3. The vertical axis shows the density. Two curves are shown. The solid line labeled values peaks near negative 1 and falls gradually. The dashed line labeled signals estimated peaks near 0 and is wider than the values curve. The values curve lies to the left of the signals curve. Both curves are smooth and bell shaped. All data are approximate.](./images/fig11_2.jpg)

[Figure 11.2](chapter11) Plot of the density of expected values and signals (estimated) in a first price common values auction. The bidder's valuations are shifted down from the estimated signal distribution.

![In the graph, the horizontal axis is labeled as residuals and ranges from negative 4 to 4. The vertical axis shows the density. Two curves are shown. The solid line labeled residuals peaks slightly higher than the dashed line. The dashed line labeled normal is symmetrical and bell shaped. Both curves peak around 0. The residuals curve is slightly sharper and more irregular. All data are approximate.](./images/fig11_3.jpg)

Figure 11.3 Plot of the density of the residual bids against a simulated data set drawn from a normal distribution with the mean and variance equal to the mean and variance for the normalized bids. This indicates that the normal distribution is a reasonable approximation of the bids.

![In the graph, the horizontal axis is labeled as bids or signals and ranges from negative 5 to 7. The vertical axis shows the density. Two curves are shown. The solid curve labeled bids is narrow and tall, peaking at 0. The dashed curve labeled signals estimated is wider and shorter, peaking near 3. The two curves overlap slightly around 1. All data are approximate.](./images/fig11_4.jpg)

[Figure 11.4](chapter11) Plot of the density of normalized bids from all auctions and signals (estimated) in OCS auctions without coalitions that have between 3 and 10 bidders. The bidder's valuations are shifted down from the estimated signal distribution.

## 11.3 Empirical Analysis: Signal Distribution from OCS Auctions using **R**

This section uses data on outer continental shelf (OCS) oil and gas tracts off Texas and Louisiana from 1954 to 1979.[2](./21-chapter11.md#fn11_2)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./21-chapter11.md#fn11_211b)These data are available from Penn State <https://capcp.la.psu.edu/data-and-software/outer-continental-shelf-ocs-auction-data/>.

### 11.3.1 Data

The code brings in the data set, book\_ocs\_ch11.csv. Next we do the trick of creating a residual auction value by regressing bids on observed characteristics of the auctions. The function as.factor() is used to create dummy variables for the block code and date of the auction. The function lm() is used to estimate the linear regression. The residuals are then calculated, and the data set is converted to a data frame.


`_>   file = paste0(dir, “bookocsch11.csv”)_`
`_>   df = read.csv(file) |>_`
`_+     select(_`
`_+       lbid_,`
`_+       lvalue_,`
`_+       lcost_,`
`_+       BlockCode_,`
`_+       Date_,`
`_+       TractNumber_,`
`_+       nCompany_`
`_+     ) |>_`
`_+     na.omit()_`
`_>   lm1 = lm(lbid ~ lvalue + as.factor(BlockCode) +_`
`_+              as.factor(Date) + lcost, data = df)_`
`_>   df$res = lm1$residuals_`
`_>   dt = setDT(df)_`
` `
`_> ggplotocsbids =_`
`_+   df |>_`
`_+   ggplot(aes(res)) +_`
`_+   geomdensity(alpha = 0.5) +_`
`_+   geomdensity(aes(rnorm(length(res)_,`
`_+                           mean(res)_,`
`_+                           sd(res)))_,`
`_+                linetype = 2, alpha = 0.5) +_`
`_+   scalexcontinuous(limits = c(-4,4)) +_`
`_+   labs(_`
`_+     x = “residuals”_,`
`_+     y = “”_,`
`_+     title = “Density of residuals”_`
`_+   ) +_`
`_+   geomvline(xintercept = 0, linetype = 2, color = “gray”) +_`
`_+   geomtext(aes(x = 2, y = 0.2, label = “residuals”)_,`
`_+             color = “gray”) +_`
`_+   geomtext(aes(x = -3, y = 0.2, label = “normal”)_,`
`_+             color = “gray”) +_`
`_+   theme(axis.text.y=elementblank()_,`
`_+         axis.ticks.y=elementblank())_`
` `
`_> ggplotocsbids_`
[Figure 11.5](./21-chapter11.md#fig11_5) presents the normalized bids for the OCS auctions. The figure also shows simulated bids from a normal distribution to suggest that a normal distribution is a reasonable approximation.

![In the graph, the horizontal axis is labeled as normalized bids and ranges from negative 4 to 6. The vertical axis shows the density of bids. Two curves are shown. The solid curve labeled rings peaks around 0.5 and is wider. The dashed curve labeled no rings peaks slightly below 0 and is taller and narrower. The curves overlap around the middle but differ in height and spread. All data are approximate.](./images/fig11_5.jpg)

[Figure 11.5](chapter11) Plot of the density of the amount bid where coalitions are not allowed (“no rings”) and coalitions are allowed (“rings”). The value of the bids have been normalized. The bids are higher when coalitions are allowed in these OCS auctions

### 11.3.2 Estimating the Signal Distribution

We restrict the sample to just those auctions without coalitions in them.[3](./21-chapter11.md#fn11_3) We can use the estimator above to estimate the signal distribution for these auctions. In addition, the code creates an index of observations that have less than 3 bidders, more than 10 bidders and missing residuals. The code then calculates the initial values for the optimization routine and runs the routine. The code creates an object index that determines the auctions with less than 3 bidders and more than 10 bidders. It then uses \-index to drop those auctions.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [3](./21-chapter11.md#fn11_311b)Coalitions are discussed in detail in the next section. They are legal bid rings.


`_>   dt2 = dt[numcoy == N]_`
`_>   index = c(which(dt2$numcoy < 3 | dt2$numcoy > 10)_,`
`_+             which(is.na(dt2$res)))_`
`_>   init = c(mean(dt2$res[-index]), sd(dt2$res[-index]))_`
`_>   b1 = optim(par = init_,`
`_+              fbidmlint_,`
`_+              bidstemp = dt2$res[-index]_,`
`_+              Ntemp = dt2$numcoy[-index]_,`
`_+              control = list(trace = FALSE_,`
`_+                           maxit = 1000000))_`
The code then creates a plot of the estimated signal distribution and the observed bids.


`_> ggplotestcvsignals =_`
`_+   data.frame(_`
`_+     bids = dt$res_,`
`_+     signals = rnorm(length(dt$res)_,`
`_+                     b1$par[1]_,`
`_+                     exp(b1$par[2]))_`
`_+   ) |>_`
`_+     ggplot(aes(bids)) +_`
`_+     geomdensity(alpha = 0.5) +_`
`_+     geomdensity(aes(signals), linetype = 2, alpha = 0.5) +_`
`_+     scalexcontinuous(limits = c(-5,8)) +_`
`_+     labs(_`
`_+       x = “bids/signals”_,`
`_+       y = “”_,`
`_+       title = “Density of signals and bids”_`
`_+     ) +_`
`_+     geomvline(xintercept = 0, linetype = 2_,`
`_+                color = “gray”) +_`
`_+     geomtext(aes(x = -3.5, y = 0.2, label = “bids”)_,`
`_+               color = “gray”) +_`
`_+     geomtext(aes(x = 6.5, y = 0.2, label = “signals (est.)”)_,`
`_+               color = “gray”) +_`
`_+     theme(axis.text.y=elementblank()_,`
`_+           axis.ticks.y=elementblank())_`
` `
`_> ggplotestcvsignals_`
[Figure 11.4](./21-chapter11.md#fig11_4) presents the density of the normalized bids and the estimated signals in the OCS auctions. The figure shows that the bidders substantially discount their bids from the observed signals. We see that this occurs for two reasons. First, their expected value (conditional on winning) is discounted from their signal because of winner's curse. Second, because it is a first price auction, bidders discount their bid from the expected value of the item.

## 11.4 Auctions with Coalitions

The classic data set for considering common values auctions is the US federal government's off-shore drilling auctions. One surprising fact is that the auctions include “coalitions” of bidders. Basically, legal bid rings. Given the discussion about bid rings in the [Chapter 10](./20-chapter10.md), it seems very odd that the government would allow collusion in these auctions.

An obvious policy question is whether the government should in fact allow bid rings in OCS auctions. You are probably thinking that the answer is obviously no. Actually it is not that obvious in the case of common value auctions.

The section works through how bidding in coalitions can be estimated. It then uses the estimated parameters from above and some characteristics of OCS auctions to simulate the policy.

### 11.4.1 The Benefit of Coalitions

The reason for allowing coalitions is that coalitions allow bidders to pool their information about the item's value. By increasing the amount of information available to the bidders, the coalitions may lead to higher bids!

Remember we said that there are two reasons for bidders shading their bids. The first, discussed in [Chapter 10](./20-chapter10.md), states that bidders shade in order to account for the probability of winning and trade off the probability of winning against how much they pay if they win. As the number of bidders increases, the probability of any particular bidder winning decreases, and so the smaller trade off leads to higher bids. As we said previously, a bid ring allows bidders to reduce their bids because their probability of winning is higher. The second reason for shading the bids is because of the information problem. Here the bid ring works in the opposite direction, by pooling information the bidders have a more precise signal of the value of the item which allows them to bid more.

How does a bidder's valuation change with more signals? Assume that the expected valuation for the ring will be the mean of the signals, conditional upon that mean being greater than all the other signals. From statistics, we know that we can approximate the distribution of this sample mean as a normal distribution with a mean equal to the true mean and the variance equal to true variance divided by the sample size.

For the _J_ members of the coalition, the probability of winning the auction with a particular average of signals (s¯) is as follows.

Φ(s¯−μσ)N−J−1(11.4)

where this gives the probability that the _J_ members of the coalition will observe a particular average of their signals multiplied by the probability that the other bidders outside the coalition will have signals below coalition's average.

What about for the bidders outside the coalition?

Φ(s−μσ)N−J−1Φ(s−μσJ)(11.5)

The probability is the probability that a signal _s_ is observed and is greater than all the other bidders outside the coalition and greater than average of the signals that are in the coalition.

### 11.4.2 Estimating Coalitions in **R**

The probabilities with coalitions allowed are the following. For bidders in the coalition, their signal has less noise than for bidders outside the coalition. For all bidders, the number of independent bidders is lower. The code uses \_ in to refer to the bidders in the coalition and \_ out for the bidders outside the coalition.


`_>   logGin = function(s, v, sigma=1, N, J)_`
`_+     (N-J-1)*logF(s, v, sigma/sqrt(J))_`
`_>   loggin = function(s, v, sigma=1, N, J) log(N-J-1) +_`
`_+     logf(s, v, sigma/sqrt(J)) +_`
`_+     (N-J-2)*logF(s, v, sigma/sqrt(J))_`
`_>   logGout = function(s, v, sigma=1, N, J)_`
`_+     (N-J-1)*logF(s, v, sigma)_`
`_>   loggout = function(s, v, sigma=1, N, J) log(N-J-1) +_`
`_+     logf(s, v, sigma) + (N-J-2)*logF(s, v, sigma)_`
The expected values and bids for bidders in and out of the coalition are as you would expect.


`_>   Ein = function(s, Ni, Ji) {_`
`_+     gu = matrix(NA,length(u),length(sig))_`
`_+     umat = matrix(NA, length(u), length(sig))_`
`_+     for(j in 1:length(sig)) {_`
`_+       gu[,j] = exp(loggin(s, u, sig[j], Ni, Ji))_`
`_+       umat[,j] = u_`
`_+     }_`
`_+     sumgu = sum(gu)_`
`_+     gammau = gu/sumgu_`
`_+     mu = sum(umat*gammau)_`
`_+     sigma = sqrt(sum(umat^2*gammau) - mu^2)_`
`_+     return(list(mu=mu, sigma=sigma))_`
`_+   }_`
`_>   Eout = function(s, Ni, Ji) {_`
`_+     gu = matrix(NA,length(u),length(sig))_`
`_+     umat = matrix(NA, length(u), length(sig))_`
`_+     for(j in 1:length(sig)) {_`
`_+       gu[,j] = exp(loggout(s, u, sig[j], Ni, Ji))_`
`_+       umat[,j] = u_`
`_+     }_`
`_+     sumgu = sum(gu)_`
`_+     gammau = gu/sumgu_`
`_+     mu = sum(umat*gammau)_`
`_+     sigma = sqrt(sum(umat^2*gammau) - mu^2)_`
`_+     return(list(mu=mu, sigma=sigma))_`
`_+   }_`
`_>   bin = function(s, N, J) {_`
`_+     vbar = Ein(s, N, J)_`
`_+     G = exp(logG(s, vbar$mu, vbar$sigma, N-J+1))_`
`_+     g = exp(logg(s, vbar$mu, vbar$sigma, N-J+1))_`
`_+     return(vbar$mu - G/g)_`
`_+   }_`
`_>   bout = function(s, N, J) {_`
`_+     vbar = Eout(s, N, J)_`
`_+     G = exp(logG(s, vbar$mu, vbar$sigma, N-J+1))_`
`_+     g = exp(logg(s, vbar$mu, vbar$sigma, N-J+1))_`
`_+     return(vbar$mu - G/g)_`
`_+   }_`
### 11.4.3 Policy Simulation

In the mid-1970s, the Department changed the policy to make illegal for larger bidders to join forces, but that still allowed small bidders to join with big bidders or with other small bidders.

Assume that auctions with coalitions have the same signal distribution as auctions without coalitions. This allows us to use the estimates from the previous section above in the policy simulations.

This analysis is restricted to cases where there is just one coalition and there are more than two bidders. When the number of bids is smaller than the number of bidders, we have coalitions in the auction. This information is then merged back into the main data set. The code uses the estimates of mu and sigma from the previous section to simulate bids in and outside of the coalition. The number of bidders in the auction is from the data. It simulates the auctions allowing for a coalition of bidders and if the coalition is not allowed.


`_> dt1 = dt[, .(num = .N_,`
`_+              numcoy = sum(as.numeric(nCompany)))_,`
`_+          by = TractNumber]_`
`_> dt = merge(dt, dt1, by=“TractNumber”)_`
` `
`_>   dt2 = dt[numcoy - num == 1 & num > 2]_`
`_>   M = length(unique(dt2$TractNumber))_`
`_>   N = dt2$numcoy_`
`_>   mu = rep(b1$par[1], M)_`
`_>   sigma = exp(b1$par[2])_`
`_>   bidssim = NULL_`
`_>   bidscf = NULL_`
`_>   set.seed(123456789)_`
`_>   for(i in 1:M) {_`
`_+     Ni = N[i]_`
`_+     si = rnorm(Ni, mu, sigma)_`
`_+     bidsi = sapply(1:Ni, function(j) bfun(si[j], Ni))_`
`_+     bidsiin = bin(mean(si[1:2]), Ni, 2)_`
`_+     bidsiout = sapply(1:(Ni-1), function(j)_`
`_+       bout(si[j], Ni, 2))_`
`_+     bidssim = c(bidssim, bidsiin, bidsiout)_`
`_+     bidscf = c(bidscf, bidsi)_`
`_+     # print(i)_`
`_+   }_`
The code then creates a plot of the density of the bids in auctions with and without coalitions.


`_> ggplotocsbids =_`
`_+   data.frame(_`
`_+     rings = bidssim_,`
`_+     norings = bidscf_`
`_+   ) |>_`
`_+     filter(_`
`_+       is.finite(rings) & is.finite(norings)_`
`_+     ) |>_`
`_+     ggplot(aes(rings)) +_`
`_+     geomdensity(alpha = 0.5) +_`
`_+     geomdensity(aes(norings), linetype = 2, alpha = 0.5) +_`
`_+     labs(_`
`_+       x = “Normalized bids”_,`
`_+       y = “”_,`
`_+       title = “Density of bids”_`
`_+     ) +_`
`_+     geomtext(aes(x = 3, y = 0.2, label = “rings”)_,`
`_+               color = “gray”) +_`
`_+     geomtext(aes(x = -3, y = 0.2, label = “no rings”)_,`
`_+               color = “gray”) +_`
`_+     theme(axis.text.y=elementblank()_,`
`_+           axis.ticks.y=elementblank())_`
` `
`_> ggplotocsbids_`
[Figure 11.5](./21-chapter11.md#fig11_5) shows that allowing bid rings (coalitions) tends to lead to higher bids! This analysis accounts for the fact that bidders will bid lower because the number of independent bidders is lower. Despite that, the bids are higher showing the advantage of aggregating signals in common values auctions.

## 11.5 Discussion and Further Reading

[Chapter 10](./20-chapter10.md) made a simplifying assumption called independent private values (IPV). This assumption rules out the common values model. [Laffont and Vuong (1996)](./25-refbib.md#ref40) present the main negative result of the common values literature. It states that without strong parametric assumptions it is not possible to identify the exact model generating the data in this setting. Despite this negative result, we could test for whether the auction is a common values auction ([Haile et al., 2006](./25-refbib.md#ref31)).

The analysis in this chapter uses a parametric model. The chapter suggests that allowing cooperation among competitors may lead to better outcomes for the government in the sale of oil drilling rights. See [Paarsch and Hong (2006)](./25-refbib.md#ref49) for more detailed analysis of the econometrics of auctions, including common value auctions. The OCS data used here have been analyzed in a number of papers, for example, [Hendricks et al. (2003)](./25-refbib.md#ref33).

[_OceanofPDF.com_](./https___oceanofpdf.com)
