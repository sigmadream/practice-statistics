309\. 

# Appendices

## A.1: Probability distributions

### Univariate discrete

In the following plots the probability mass functions for several combinations of parameters are denoted with points; lines connect the points for visualization, but the probability is non-zero only at the points.

**Beta-binomial**

![Three probability curves for n equals 25 are plotted against x under Beta–Binomial models with parameter pairs a equals 1, b equals 2, a equals 5, b equals 10, and a equals 10, b equals 20. The first curve decreases steadily from x equals 0 to x equals 25. The other two form bell-shaped distributions, peaking near x equals 8 and x equals 10. The shapes narrow and shift rightward as a and b increase.](./images/ufig10_1.jpg) 

Notation: X∼BetaBinomial(n,a,b)

Support: X∈{0,1,...,n}

Parameters: n∈{1,2,...}, a,b\>0

PMF: Γ(n+1)Γ(x+a)Γ(n−x+b)Γ(a+b)Γ(x+1)Γ(n−x+1)Γ(n+a+b)Γ(a)Γ(b)

Mean: na/(a+b)

Variance: nab(a+b+n)/\[(a+b)2(a+b+1)\]

Notes: If X|θ∼Binomial(n,θ) and θ∼Beta(a,b), then X∼BetaBinomial(n,a,b). If a\=b\=1 then X∼DiscreteUniform(0,n).

**Binomial**

![Three probability mass functions are plotted for discrete X. For n equals 5 and theta equals 0.5, the distribution peaks sharply around X equals 2. For n equals 10 and theta equals 0.5, the distribution shifts rightward and spreads, peaking near X equals 5. For n equals 10 and theta equals 0.1, most mass concentrates at very small X, with the highest probability at X equals 0 and rapidly decreasing thereafter.](./images/ufig10_2.jpg) 

Notation: X∼Binomial(n,θ)

Support: X∈{0,1,...,n}

Parameters: n∈{1,2,...}, θ∈\[0,1\]

PMF: (nx)θx(1−θ)n−x

Mean: nθ

Variance: nθ(1−θ)

Notes: If _X_ is the number of successes in _n_ independent trials each with success probability _θ_, then X∼Binomial(n,θ); if n\=1 then X∼Bernoulli(θ).

310\. **Discrete uniform**

![Two discrete probability mass functions are plotted for X from 0 to 10. The first, with a equal to 1 and b equal to 4, is a flat distribution with probability 0.20 for X from 0 to 4 and 0 elsewhere. The second, with a equal to 2 and b equal to 8, places probability 0.12 on X from 2 to 9 and 0 elsewhere, forming a wider, lower plateau.](./images/ufig10_3.jpg) 

Notation: X∼DiscreteUniform(a,b)

Support: X∈{a,a+1,...,b}

Parameters: a,b∈{...,−1,0,1,...} with a<b

PMF: 1/(b−a+1)

Mean: (a+b)/2

Variance: \[(b−a+1)2−1\]/12

Notes: The discrete uniform can be applied to any finite set. For example, we could say that _X_ is distributed uniformly over the set {1/10,2/10,...,10/10}.

**Negative Binomial**

![The plot compares probability mass functions for three binomial models. With m equals 5 and theta equals 0.5, the distribution is sharply peaked near x equals 3. With m equals 10 and theta equals 0.5, the peak shifts to x equals 5 with a wider spread. With m equals 2 and theta equals 0.05, probabilities remain very small and decrease slowly across x, forming a nearly flat, low curve.](./images/ufig10_4.jpg) 

Notation: X∼NegBinomial(θ,m)

Support: X∈{0,1,2,...}

Parameters: m\>0, θ∈\[0,1\]

PMF: (x+m−1x)θm(1−θ)x

Mean: m(1−θ)/θ

Variance: m(1−θ)/θ2

Notes: In a sequence of independent trials each with success probability _θ_, if _X_ is the number of failures that occur before the mth success (assuming _m_ is an integer), then X∼NegBinomial(θ,m); if m\=1 then X∼Geometric(θ). (The distribution can also be defined with _m_ as the number of failures, but we use the JAGS parameterization.)

**Poisson**

![A discrete probability plot shows three distributions for theta equal to 2, 10, and 20. The theta equals 2 curve is sharply peaked near small x values and rapidly decays. The theta equals 10 curve is wider, peaking around x near 10. The theta equals 20 curve is even broader, peaking near x around 20 with a long right tail. Probability values decrease toward zero as x moves away from each distribution's centre.](./images/ufig10_5.jpg) 

Notation: X∼Poisson(θ)

Support: X∈{0,1,2,...}

Parameters: θ\>0

PMF: θxexp(−θ)x!

Mean: _θ_

Variance: _θ_

Notes: If events occur independently and uniformly over time (space) with the expected number of events in a given time interval (region) equal to _θ_, then the number of events that occur in the interval (region) follows a Poisson(θ) distribution.

311\. 

### Multivariate discrete

**Multinomial**

Notation: X\=(X1,...,Xp)∼Multinomial(n,θ)

Support: Xj∈{0,1,...,n} with ∑j\=1pXj\=n

Parameters: θ\=(θ1,...,θp) with θj∈\[0,1\] and ∑j\=1pθj\=1

PMF: n!∏j\=1pxj!∏j\=1pθjxj

Mean: E(Xj)\=nθj

Variance: Var(Xj)\=nθj(1−θj)

Covariance: Cov(Xj,Xk)\=−nθjθk

Marginal distributions: Xj∼Binomial(n,θj)

Notes: If _n_ independent trials each have _p_ possible outcomes with the probability of outcome _j_ being _θ_ _j_ and _X_ _j_ is the number of the trials that result in outcome _j_, then X\=(X1,...,Xp)∼Multinomial(n,θ).

312\. 

### Univariate continuous

**Asymmetric Laplace distribution**

![A plot displays two Laplace densities. The solid curve has location 0 and mixing probability 0.2, forming a sharp peak at zero with a long right tail. The dashed curve has location 5 and mixing probability 0.8, peaking near five with a long left tail. Both densities decay symmetrically away from their peaks but differ in height and spread, highlighting how changes in location and mixing probability shift and reshape the distribution along the x-axis.](./images/ufig10_6.jpg) 

Notation: X∼ASL(μ,σ,p)

Support: X∈(−∞,∞)

Parameters: location μ∈(−∞,∞), scale σ\>0, shape p∈(0,1)

PDF: 2p(1−p)σexp(−2ρp(x−μ)σ)

where ρp(x)\=px if x≥0 and ρp(x)\=(p−1)x if x<0

Mode: _μ_

Quantile: the _p_ quantile is _μ_

Notes: if p\=0.5, then X∼DE(μ,σ).

**Beta**

![Four Beta densities on zero to one compare shapes: a equals one b equals one is flat, a equals one b equals five declines to the right, a equals twenty b equals twenty forms a sharp peak at one half, and a equals zero point five b equals zero point five is U-shaped with high density near zero and one.](./images/ufig10_7.jpg) 

Notation: X∼Beta(a,b)

Support: X∈\[0,1\]

Parameters: a\>0, b\>0

PDF: Γ(a+b)Γ(a)Γ(b)xa−1(1−x)b−1

Mean: aa+b

Variance: ab(a+b)2(a+b+1)

Notes: If a\=b\=1 then X∼Uniform(0,1); 1−X∼Beta(b,a); X/(1−X)∼BP(a,b).

**Beta prime**

![The plot shows three density curves for different parameter pairs. The solid line for a equals 1, b equals 1 decreases steadily from a high peak near zero. The dashed curve for a equals 2, b equals 1 rises to a low peak around x equals 1 before tapering off. The dotted curve for a equals 1, b equals 2 starts with a very sharp peak at zero and declines more quickly.](./images/ufig10_8.jpg) 

Notation: X∼BP(a,b)

Support: X∈\[0,∞\]

Parameters: shapes a,b\>0

PDF: Γ(a+b)Γ(a)Γ(b)xa−1(1+x)−a−b

Mean: a/(b−1) for b\>1

Variance: a(a+b−1)(b−2)(b−1)2 for b\>2

Notes: X/(1+X)∼Beta(a,b); dX1/c∼GPB(a,b,c,d).

313\. **Gamma**

![Four beta–distribution density curves are shown. When a equals zero point five and b equals zero point five, the curve is U-shaped with high density near zero and three. When a equals one and b equals one, the density is flat across x. When a equals two and b equals two, the curve is symmetric and peaked around x equals one. When a equals ten and b equals ten, the curve is sharply concentrated near x equals one.](./images/ufig10_9.jpg) 

Notation: X∼Gamma(a,b)

Support: X∈\[0,∞\]

Parameters: shape a\>0, scale b\>0

PDF: baΓ(a)xa−1exp(−bx)

Mean: a/b

Variance: a/b2

Notes: cX∼Gamma(a,b/c); if a\=1 then X∼Exponential(b); if a\=ν/2 and b\=1/2 then X∼Chi-squared(ν); 1/X∼InvGamma(a,b).

**Generalized beta prime**

![A plot compares four density curves for a equals one and b equals one under different parameter choices for c and d. When c equals one and d equals one, the curve decreases smoothly from a high peak near zero. Lowering c to zero point eight stretches the curve upward near the origin, while increasing c to one point two lowers the initial peak. Reducing d to zero point six creates a sharper rise near zero followed by faster decline.](./images/ufig10_10.jpg) 

Notation: X∼GBP(a,b,c,d)

Support: X∈(0,∞)

Parameters: shapes a,b,c\>0 and scale d\>0

PDF: cdΓ(a+b)Γ(a)Γ(b)(xd)ac−1{1+(xd)c}−a−b

Mean: d(α+1/c)Γ(b−1/c)Γ(a)Γ(b) if bc\>1

Notes: If a\=b\=1/2 and c\=1 then X∼HC(d).

**Half Cauchy**

![A plot shows two density curves for a positive-valued distribution with scale parameter sigma. The solid curve for sigma equal to one is higher near zero and decreases more sharply. The dashed curve for sigma equal to two starts lower, decreases more slowly, and remains above the solid curve for larger X. The horizontal axis is X from zero to three and the vertical axis is density.](./images/ufig10_11.jpg) 

Notation: X∼HC(σ)

Support: X∈(0,∞)

Parameters: scale σ\>0

PDF: 2πσ(1+x2/σ2)−1

Mode: 0

Notes: cX∼HC(cσ); if Y∼t1(0,σ), then |Y|∼HC(σ).

314\. **Inverse gamma**

![A set of four density curves is presented as a line graph, plotted against x. The curves represent parameter pairs a equals zero point five b equals zero point five a equals one b equals one a equals two b equals two and a equals ten b equals ten.](./images/ufig10_12.jpg) 

Notation: X∼InvGamma(a,b)

Support: X∈\[0,∞\]

Parameters: shape a\>0, scale b\>0

PDF: baΓ(a)x−a−1exp(−b/x)

Mean: ba−1 (if a\>1)

Variance: b2(a−1)2(a−2) (if a\>2)

Notes: cX∼InvGamma(a,cb); 1/X∼Gamma(a,b).

**Laplace/Double exponential**

![A set of three density curves is presented as a line graph, plotted against x. The curves correspond to parameter pairs mu equals zero sigma equals one mu equals zero sigma equals two and mu equals two sigma equals one.](./images/ufig10_13.jpg) 

Notation: X∼DE(μ,σ)

Support: X∈(−∞,∞)

Parameters: location μ∈(−∞,∞), scale σ\>0

PDF: 12σexp(−|x−μ|σ)

Mean: _μ_

Variance: 2σ2

Notes: c+dX∼DE(c+dμ,dσ).

**Logistic**

![A set of three density curves is presented as a line graph, plotted against x. The curves represent parameter pairs mu equals zero sigma equals one mu equals zero sigma equals two and mu equals two sigma equals one.](./images/ufig10_14.jpg) 

Notation: X∼Logistic(μ,σ)

Support: X∈(−∞,∞)

Parameters: location μ∈(−∞,∞), scale σ\>0

PDF: 1σexp\[−(x−μ)/σ\]{1+exp\[−(x−μ)/σ\]}2

Mean: _μ_

Variance: π2σ2/3

Notes: c+dX∼Logistic(c+dμ,dσ); if U∼Uniform(0,1), then μ+σlogit(U)∼Logistic(μ,σ).

315\. **Normal/Gaussian**

![A set of three density curves is presented as a line graph, plotted against x. The curves correspond to parameter pairs mu equals zero sigma equals one mu equals negative two sigma equals one and mu equals zero sigma equals two.](./images/ufig10_15.jpg) 

Notation: X∼Normal(μ,σ2)

Support: X∈(−∞,∞)

Parameters: location μ∈(−∞,∞), scale σ\>0

PDF: 12πσexp\[−(x−μ)22σ2\]

Mean: _μ_

Variance: _σ_2

Notes: c+dX∼Normal(c+dμ,d2σ2); if μ\=0 and σ2\=1 then _X_ follows the standard normal distribution.

**Student's t**

![A set of three density curves presented as a line graph, plotted against x. The curves correspond to parameter pairs mu equals zero sigma equals one nu equals two mu equals zero sigma equals one nu equals five and mu equals two sigma equals two nu equals five.](./images/ufig10_16.jpg) 

Notation: X∼tν(μ,σ2)

Support: X∈(−∞,∞)

Parameters: location μ∈(−∞,∞), scale σ\>0, degrees of freedom ν\>0

PDF: Γ(ν+12)Γ(ν/2)νπσ\[1+(x−μ)2νσ2\]−(ν+1)/2

Mean: _μ_ (if ν\>1)

Variance: σ2νν−2 (if ν\>2)

Notes: c+dX∼tν(c+dμ,d2σ2); if μ\=0 and σ2\=1, then _X_ follows the standard t distribution; if Z∼Normal(0,1) independent of W∼Gamma(ν/2,1/2), then μ+σZ/W/ν∼tν(μ,σ2); if ν\=1, then _X_ follows the Cauchy distribution; _X_ is approximately Normal(μ,σ2) for large _ν_.

**Uniform**

![A pair of density curves is presented as a line graph, plotted against x. One curve represents a equals zero b equals one and the other represents a equals zero point five b equals two point five.](./images/ufig10_17.jpg) 

Notation: X∼Uniform(a,b)

Support: X∈\[a,b\]

Parameters: −∞<a<b<∞

PDF: 1b−a

Mean: (a+b)/2

Variance: (b−a)2/12

Notes: If X1,X2∼iidUniform(0,1), then −2log(X1)cos(2πX2) ∼Normal(0,1); if _X_ ∼ Uniform(0,1) and _F_ is a continuous CDF, then F−1(X) has CDF _F_.

316\. 

### Multivariate continuous

**Dirichlet**

Notation: X\=(X1,...,Xp)∼Dirichlet(θ)

Support: Xj∈\[0,1\] with ∑j\=1pXj\=1

Parameters: θ\=(θ1,...,θp) with θj\>0

PDF: Γ(∑j\=1pθj)∏j\=1pΓ(θj)∏j\=1pxjθj−1

Mean: E(Xj)\=θj/(∑k\=1pθk)

Variance: Var(Xj)\=θj(∑k≠jθk)(∑k\=1pθk)2(1+∑k\=1pθk)

Covariance: Cov(Xj,Xk)\=−θjθk(∑k\=1pθk)2(1+∑k\=1pθk)

Marginal distributions: Xj∼Beta(θj,∑k≠jθk)

Notes: If Wj∼indepGamma(θj,b) and Xj\=Wj/(∑k\=1pWk), then X\=(X1,...,Xp)∼Dirichlet(θ).

**Inverse Wishart**

Notation: X∼InvWishart(ν,Ω)

Support: X\={Xjk} is a p×p symmetric positive definite matrix

Parameters: degrees of freedom ν\>p−1 and p×p symmetric positive definite matrix Ω\={Ωjk}

PDF: |Ω|ν/22pν/2Γp(ν/2)|X|−(ν+p+1)/2exp\[−trace(ΩX−1)/2\]

Mean: E(Xjk)\=1ν−p−1Ωjk (for ν\>p+1)

Variance: Var(Xjk)\=(ν−p+1)Ωkk2+(ν−p−1)ΩjjΩkk(ν−p)(ν−p−1)2(ν−p−3) (for ν\>p+3)

Marginal distributions: Xjj∼InvGamma((ν−p+1)/2,Ωjj/2)

Notes: If Y∼Wishart(ν,Ω−1) then Y−1∼InvWishart(ν,Ω); if ν\=p+1 and Ω is a diagonal matrix then the correlation Xjk/XjjXkk∼Uniform(−1,1).

**Multivariate normal**

Notation: X\=(X1,...,Xp)T∼Normal(μ,Σ)

Support: Xj∈(−∞,∞)

Parameters: mean vector μ\=(μ1,...,μp) with μj∈(−∞,∞) and p×p positive definite covariance matrix Σ

PDF: (2π)−p/2|Σ|−1/2exp\[−12(X −μ)TΣ−1(X −μ)\]

Mean: E(Xj)\=μj

Variance: Var(Xj)\=σj2 where σj2 is the (j,j) element of Σ

Covariance: Cov(Xj,Xk)\=σjk where σjk is the (j,k) element of Σ

Marginal distributions: Xj∼Normal(μj,σj2)

Notes: For _q_\-vector **a** and q×p matrix **b**, a+bX∼Normal(a+b μ,bΣbT).

317\. **Multivariate Student's t**

Notation: X\=(X1,...,Xp)T∼tν(μ,Σ)

Support: Xj∈(−∞,∞)

Parameters: location μ\=(μ1,...,μp) with μj∈(−∞,∞), p×p positive definite matrix Σ and degrees of freedom ν\>0

PDF: Γ(ν/2+p/2)Γ(ν/2)(νπ)p/2|Σ|−1/2\[1+1ν(X−μ)TΣ−1(X−μ)\]−(ν+p)/2

Mean: E(Xj)\=μj (if ν\>1)

Variance: Var(Xj)\=νν−2σj2 where σj2 is the (j,j) element of Σ (if ν\>2)

Covariance: Cov(Xj,Xk)\=νν−2σjk where σjk is the (j,k) element of Σ (if ν\>2)

Marginal distributions: Xj∼tν(μj,σj2)

Notes: For _q_\-vector **a** and q×p matrix **b**, a+bX∼tν(a+b μ,bΣbT); **X** is approximately Normal(μ,Σ) for large _ν_; if X|W∼Normal(μ,Σ/W) and W∼Gamma(ν/2,ν/2), then X∼tν(μ,Σ).

**Wishart**

Notation: X∼Wishart(ν,Ω)

Support: X\={Xjk} is a p×p symmetric positive definite matrix

Parameters: degrees of freedom ν\>p−1 and p×p symmetric positive definite matrix Ω\={Ωjk}

PDF: 12pν/2|Ω|ν/2Γp(ν/2)|X|(ν−p−1)/2exp\[−trace(Ω−1X)/2\] where Γp is the multivariate gamma function

Mean: E(Xjk)\=νΩjk

Variance: Var(Xjk)\=ν(Ωjk2+ΩjjΩkk)

Marginal distributions: Xjj∼Gamma(ν/2,Ωjj/2)

Notes: If _ν_ is an integer and Z1,...,Zν∼iidNormal(0,Ω), then ∑i\=1νZiZiT∼Wishart(ν,Ω).

## A.2: 318\. List of conjugacy pairs

Below is a partial list of conjugacy pairs. In these derivations, all parameters not assigned a prior are assumed to be fixed.

1. **Binomial proportion**  
Likelihood: Y|θ∼Binomial(n,θ)  
Prior: θ∼Beta(a,b)  
Posterior: θ|Y∼Beta(a+Y,b+n−Y)
2. **Negative-binomial proportion**  
Likelihood: Y|θ∼NegBinomial(θ,m)  
Prior: θ∼Beta(a,b)  
Posterior: θ|Y∼Beta(a+m,b+Y)
3. **Multinomial probabilities**  
Likelihood: Y\=(Y1,...,Yp)|θ∼ Multinomial(n,θ)  
Prior: θ∼Dirichlet(α) with α\=(α1,...,αp)  
Posterior: θ|Y∼Dirichlet(α+Y)
4. **Poisson rate**  
Likelihood: Y1,...,Yn|λ∼indepPoisson(Niλ) with _N_ _i_ fixed  
Prior: λ∼Gamma(a,b)  
Posterior: λ|Y∼Gamma(a+∑i\=1nYi,b+∑i\=1nNi)
5. **Mean of a normal distribution**  
Likelihood: Y1,...,Yn|μ∼iid Normal(μ,σ2)  
Prior: μ∼Normal(θ,σ2/m)  
Posterior: μ|Y∼Normal(nY¯+mθn+m,σ2n+m) for Y¯\=∑i\=1nYi/n
6. **Variance of a normal distribution**  
Likelihood: Y1,...,Yn|σ2∼indepNormal(μi,σ2)  
Prior: σ2∼InvGamma(a,b)  
Posterior: σ2|Y∼InvGamma(a+n/2,b+∑i\=1n(Yi−μi)2/2)
7. **Precision of a normal distribution**  
Likelihood: Y1,...,Yn|τ2∼indepNormal(μi,1/τ2)  
Prior: τ2∼Gamma(a,b)  
Posterior: τ2|Y∼Gamma(a+n/2,b+∑i\=1n(Yi−μi)2/2)
8. **Mean vector of a multivariate normal distribution**  
Likelihood: Y1,...,Yn|μ∼indepNormal(Xiμ,Σi)  
Prior: μ∼Normal(θ,Ω)  
Posterior: μ|Y∼Normal(VM,V) with V\=(∑i\=1nXiTΣi−1Xi+Ω−1)−1 and M\=∑i\=1nXiTΣi−1Yi+Ω−1θ  
Special case: If Xi\=I and Σi\=Σ for all _i_, then V\=(nΣ−1+Ω−1)−1 and M\=nΣ−1Y¯+Ω−1θ
9. **Covariance matrix of a multivariate normal distribution**  
Likelihood: Y1,...,Yn|Σ∼indepNormal(μi, Σ)  
Prior: Σ∼InvWishart(ν,R)  
Posterior: Σ|Y∼InvWishart(n+ν,S+R), where S\=∑i\=1n(Yi−μi)(Yi−μi)T
10. 319\. **Precision matrix of a multivariate normal distribution**  
Likelihood: Y1,...,Yn|Ω∼indepNormal(μi, Ω−1)  
Prior: Ω∼Wishart(ν,R)  
Posterior: Ω|Y∼Wishart(n+ν,\[S+R−1\]−1), where S\=∑i\=1n(Yi−μi)(Yi−μi)T
11. **Scale parameter of a gamma distribution**  
Likelihood: Y1,...,Yn|μ∼iid Gamma(ai,wib)  
Prior: b∼Gamma(u,v)  
Posterior: b|Y∼Gamma(∑i\=1nai+u,∑i\=1nwiYi+v)
12. **Arbitrary parameter with discrete prior**  
Likelihood: Y1,...,Yn|θ∼indepfi(Yi|θ)  
Prior: Prob(θ\=θk)\=πk for θ∈{θ1,...,θm}  
Posterior: Prob(θ\=θk|Y)\=Lk/\[∑j\=1mLj\] where Lk\=πk∏i\=1nfi(Yi|θ k)

## A.3: 320\. Derivations

321\. 

### Normal-normal model for a mean

Say Yi|μ∼iidNormal(μ,σ2) for i\=1,...,n with _σ_2 known and prior μ∼Normal(θ,σ2/m). Since the Y1,...,Yn are independent, the likelihood factors as

f(Y|μ)\=∏i\=1nf(Yi|μ)\=∏i\=1n12πσexp\[−(Yi−μ)22σ2\].

Discarding constants that do not depend on _μ_ and expressing the product of exponentials as the exponential of the sum, the likelihood is

f(Y|μ)∝exp\[−∑i\=1n(Yi−μ)22σ2\]∝exp\[−12(−2nY¯σ2μ+nσ2μ2)\]

where Y¯\=∑i\=1nYi/n. The last equality comes from multiplying the quadratic terms, collecting them as a function of their power of _μ_, and discarding terms without a _μ_. Similarly, the prior can be written

π(μ)∝exp\[−m(μ−θ)22σ2\]∝exp\[−12(−2mθσ2μ+mσ2μ2)\].

Because both the likelihood and prior are quadratic in _μ_, they can be combined as

p(μ|Y)∝f(Y|μ)π(μ)∝exp\[−12(−2nY¯+mθσ2μ+n+mσ2μ2)\]∝exp\[−12(−2Mμ+1Vμ2)\],

where M\=(nY¯+mθ)/σ2 and V\=σ2/(n+m). The exponent of the posterior is quadratic in _μ_, and we have seen that a Gaussian PDF is quadratic in the exponent. Therefore, we rearrange the terms in the posterior to reveal its Gaussian PDF form. Completing the square in the exponent (and discarding and/or adding terms that do not depend on _μ_) gives

p(μ|Y)∝exp\[−12(−2Mμ+1Vμ2)\]∝exp\[−(μ−VM)22V\].

Therefore, the posterior is μ|Y∼N(VM,V). Plugging in the above expressions for _M_ and _V_ gives

μ|Y∼N(wY¯+(1−w)θ,σ2n+m)

where w\=n/(n+m).

### Normal-normal model for a mean vector

The model is

Y|β∼Normal(Xβ,Σ) and β∼Normal(μ,Ω).

As with the normal-normal model in [Section 9.4](./17-chapter9.md#sec9_4), we proceed by expressing the exponential of the posterior as a quadratic form in β and then comparing this expression to a multivariate normal to determine the posterior. Using precision matrices U\=Σ−1 and V\=Ω−1, the posterior is

p(β|Y)∝f(Y|β)π(β)∝exp\[−12(Y−Xβ)TU(Y−Xβ)T\]\[−12(β−μ)TV(β−μ)T\]∝exp{−12\[−2(YTUX+μTV)β+β(XTUX+V)β\]}∝exp\[−12(−2WTβ+βTPβ)\]

where W\=XTUY+Vμ and P\=XTUX+V. If β|Y∼Normal(M, S) for some mean vector **M** and covariance matrix **S**, then its PDF can be written

p(β|Y)∝exp\[−12(β− M)TS−1(β−M)\]∝exp\[−12(−2MTS−1β+βTS−1β)\].

To reconcile these two expressions of the posterior, we must have posterior covariance S\=P−1 and S−1M\=W and thus M\=SW\=P−1W. Inserting the expressions for **W** and **P** and replacing precision matrices with covariance matrices gives the posterior

β|Y∼Normal\[Σβ(XTΣ−1Y+Ω−1μ),Σβ\],

where Σβ\=(X′Σ−1X+Ω−1)−1

### Normal-inverse Wishart model for a covariance matrix

The model for the _p_\-vectors Y1,...,Yn given the p×p covariance matrix Σ is

Yi∼indepNormal(μi,Σ) and Σ∼InvWishartp(ν,R).

322\. Using the facts that for arbitrary matrices **A**, **B** and **C**, Trace(A+B)\=Trace(A)+Trace (B) and Trace(ABC)\=Trace(BCA), the likelihood can be written

f(Y|Σ)∝∏i\=1n|Σ|−1/2exp\[−12(Yi−μi)TΣ−1(Yi−μi)\]∝|Σ|−n/2exp\[−12∑i\=1n(Yi−μi)TΣ−1(Yi−μi)\]∝|Σ|−n/2exp{−12∑i\=1nTrace\[(Yi−μi)TΣ−1(Yi−μi)\]}∝|Σ|−n/2exp{−12∑i\=1nTrace\[Σ−1(Yi−μi)(Yi−μi)T\]}∝|Σ|−n/2exp{−12Trace\[∑i\=1nΣ−1(Yi−μi)(Yi−μi)T\]}∝|Σ|−n/2exp{−12Trace\[Σ−1∑i\=1n(Yi−μi)(Yi−μi)T\]}∝|Σ|−n/2exp\[−12Trace(Σ−1W)\]

where W\=∑i\=1n(Yi−μi)(Yi−μi)T. The inverse Wishart prior is

π(Σ)∝|Σ|−(ν+p+1)/2exp\[−12Trace(Σ−1R)\].

Combining the likelihood and prior, the posterior is

p(Σ|Y)∝f(Y|Σ)π(Σ)∝|Σ|−(n+ν+p+1)/2exp{−12Trace\[Σ−1(W +R)\]}.

Therefore, Σ|Y∼InvWishartp(n+ν,∑i\=1n(Yi−μi)(Yi−μi)T+R).

### Jeffreys prior for a normal model

The Gaussian model is Yi∼iidNormal(μ,σ2). Denote τ\=σ2. The log likelihood is

logf(Y|μ,τ)\=−n2log(τ)−12τ∑i\=1(Yi−μ)2.

The information matrix depends on both second derivatives and the cross derivative. The second derivatives are

∂2logf(Y|μ,τ)∂μ2\=∂∂μ1τ∑i\=1(Yi−μ)\=−nτ

and

∂2logf(Y|μ,τ)∂τ2\=∂∂τ−n2τ+12τ2∑i\=1(Yi−μ)2\=n2τ2−1τ3∑i\=1(Yi−μ)2.

323\. The cross derivative is

∂2logf(Y|μ,τ)∂μ∂τ\=∂∂τ1τ∑i\=1(Yi−μ)\=−1τ2∑i\=1(Yi−μ)

Since E(Yi)\=μ and E(Yi−μ)2\=τ, the elements of the information matrix are

−E(∂2logf(Y|μ,τ)∂μ2)\=nτ−E(∂2logf(Y|μ,τ)∂τ2)\=−n2τ2+nττ3\=n2τ2−E(∂2logf(Y|μ,τ)∂μ∂τ)\=0.

324\. The determinant of the 2×2 information matrix is thus

|I(μ,τ)|\=(nτ)(n2τ2)−02\=n22τ3,

and the JP is

π(μ,σ2)∝n22τ3∝1(σ2)3/2.

### Jeffreys prior for multiple linear regression

Assume Y|β,σ2∼Normal(X β,σ2In), and denote τ\=σ2. The log likelihood is

logf(Y|β,τ)\=−n2log(τ)−12τ(Y−Xβ)T(Y−Xβ).

The second derivative with respect to _τ_ is

∂2logf(Y|β,τ)∂τ2\=∂∂τ−n2τ+12τ2(Y−Xβ)T(Y−Xβ)\=n2τ2−1τ3(Y−Xβ)T(Y−Xβ).

Taking derivatives with respect to β requires using matrix calculus identities including the formula for the derivative of a quadratic form,

∂2logf(Y|β,τ)∂β2\=∂∂β1τXT(Y−X β)\=−1τXTX.

The cross derivative is

∂2logf(Y|β,τ)∂β∂τ\=∂∂τ1τXT(Y−Xβ)\=−1τ2XT(Y− Xβ).

Since E(Y)\=Xβ and E(Y−Xβ)T(Y −Xβ)\=nτ, the elements of the information matrix are

−E(∂2logf(Y|μ,τ)∂β2)\=1τXTX−E(∂2logf(Y|μ,τ)∂τ2)\=−n2τ2+nττ3\=n2τ2−E(∂2logf(Y|β,τ)∂μ∂τ)\=0.

The determinant of the (p+1)×(p+1) block-diagonal information matrix is thus

|I(β,τ)|\=|1τXTX|n2τ2\=n2τp+2|XTX|,

and the JP is

π(β,σ2)∝n2τp+2|XTX|∝1(σ2)p/2+1.

### Convergence of the Gibbs sampler

Here we provide: (1) a proof that the Gibbs sampler generates posterior samples after convergence and (2) a discussion of the theory of Markov processes that ensures that Gibbs sampler converges to the posterior distribution.

**Part (1)**: The proof of (1) is equivalent to showing that the posterior distribution is the stationary distribution of this Markov chain. That is, if we make a draw from the posterior distribution and then iterate the Gibbs sampler forward one iteration from this starting point, the next iteration also follows the posterior distribution. To make the derivations tractable, we restrict the proof to the bivariate case with p\=2 and thus θ\=(θ1,θ2) and denote the posterior density as p(θ1,θ2)\=p(θ|Y), the full conditional density as p(θ1|θ2)\=p(θ1|θ2,Y), and the marginal density as p(θ1)\=∫p(θ1,θ2)dθ2\=∫p(θ1|θ2)p(θ2)dθ2. Assume we have reached convergence and so one draw in the chain is a realization from the posterior distribution, say θ∗\=(θ1∗,θ2∗)∼p(θ1,θ2). We would like to show that the subsequent sample also follows the posterior distribution. By recursion, this shows that once the algorithm has converged, all samples follow the posterior.

The next sample, (θ1′,θ2′), drawn from Gibbs sampling has density

q(θ1′,θ2′|θ1∗,θ2∗)\=p(θ1′|θ2∗)p(θ2′|θ1′),

where the two elements of the product represent the updates of the two parameters from their full conditional distribuitons given the current value of the parameters in the chain. We want to show that the marginal distribution of (θ1′,θ2′) integrating over (θ1∗,θ2∗) follows the posterior. The marginal distribution is

g(θ1′,θ2′)\=∫∫q(θ1′,θ2′|θ1∗,θ2∗)f(θ1∗,θ2∗)dθ1∗dθ2∗.

The integral reduces to

g(θ1′,θ2′)\=∫∫p(θ1′|θ2∗)p(θ2′|θ1′)p(θ1∗|θ2∗)p(θ2∗)dθ2∗dθ1∗\=p(θ2′|θ1′)∫p(θ1′|θ2∗)p(θ2∗)\[∫p(θ1∗|θ2∗)dθ1∗\]dθ2∗\=p(θ2′|θ1′)∫p(θ1′|θ2∗)p(θ2∗)dθ2∗\=p(θ2′|θ1′)∫p(θ1′,θ2∗)dθ2∗\=p(θ2′|θ1′)p(θ1′)\=p(θ1′,θ2′),

as desired. The proof for p\>2 similar but involves higher-order integration.

**Part (2)**: Part (1) shows (for a special case) that the stationary distribution of the Gibbs sampler is the posterior distribution. The proof that the Gibbs sampler converges to 325\. its stationary (posterior) distribution draws heavily from Markov chain theory. Given that the posterior distribution is the stationary distribution, \[[156](./19-ref01.md#refbib156)\] proves that a Gibbs sampler converges to the posterior distribution if the chain is aperiodic and _p_\-irreducible. A chain is _aperiodic_ if for any partition of the posterior domain of θ, say {A1,...,Am}, so that each subset has positive posterior probability, then the probability of the chain transitioning from Ai to Aj is positive for any _i_ and _j_. A chain is _p_\-_irreducible_ if for any initial value θ(0) in the support of the posterior distribution and set A with positive posterior probability, i.e., Prob(θ∈A)\>0, there exists an _s_ so that there is a positive probability that the chain will visit A at iteration _s_. Proving convergence then requires showing that the Gibbs sampler is aperiodic and _p_\-irreducible, which is discussed, e.g., in \[[156](./19-ref01.md#refbib156)\] and \[[130](./19-ref01.md#refbib130)\]. A sufficient condition is that for any set A with positive posterior probability and any initial value θ(0) in the support of the posterior distribution, the probability under the Gibbs sampler that θ(1)∈A is positive. This condition is met in all but exotic cases where support of full conditional distributions depends on the values of other parameters, in which case convergence should be studied carefully.

### Marginal distribution of a normal mean under Jeffreys prior

Assume Yi∼iidNormal(μ,σ2) and Jeffreys prior π(μ,σ2)∝(σ2)−3/2. Denoting τ\=σ2, the joint posterior is

p(μ,τ|Y)∝{τ−n/2exp\[−∑i\=1n(Yi−μ)22τ\]}{τ−3/2}∝τ−(n+1)/2−1exp\[−∑i\=1n(Yi−μ)22τ\]∝τ−A−1exp\[−Bτ\],

where A\=(n+1)/2 and B\=∑i\=1n(Yi−μ)2/2. As a function of _τ_, the joint distribution resembles an InvGamma(A,B) PDF. Integrating over _τ_ gives

p(μ|Y)∝∫p(μ,τ|bY)dτ∝∫τ−A−1exp(−B/τ)dτ∝Γ(A)BA∫BAΓ(A)τ−A−1exp(−B/τ)dτ∝Γ(A)BA∝\[∑i\=1n(Yi−μ)2\]−(n+1)/2.

The marginal PDF is a quadratic function of _μ_ raised to the power −(n+1)/2, suggesting that the posterior is a t distribution with _n_ degrees of freedom. Defining σ^2\=∑i\=1n(Yi−Y¯)2/n\=∑i\=1nYi2/n−Y¯2 326\. and completing the square gives

∑i\=1n(Yi−μ)2\=∑i\=1nYi2−2∑i\=1nYiμ+nμ2\=n\[∑i\=1nYi2/n−2Y¯μ+μ2\]\=n\[∑i\=1nYi2/n−Y¯2+Y¯2−2Y¯μ+μ2\]\=n\[∑i\=1nYi2/n−Y¯2+(μ−Y¯)2\]\=n\[σ^2+(μ−Y¯)2\].

Inserting this expression back into the marginal posterior gives

p(μ|Y)∝\[∑i\=1n(Yi−μ)2\]−(n+1)/2∝\[σ^2+(μ−Y¯)2\]−(n+1)/2∝\[1+1n(μ−Y¯σ^/n)2\]−(n+1)/2.

This is Student's t-distribution with location parameter Y¯, scale parameter σ^/n, and _n_ degrees of freedom.

### Marginal posterior of the regression coefficients under Jeffreys prior

Assume Y|β,σ2∼Normal(X β,σ2In) and Jeffreys prior π(β,σ2)∝(σ2)−p/2−1. Denoting τ\=σ2, the joint posterior is

p(β,τ|Y)∝{τ−n/2exp\[−12τ(Y−Xβ)T(Y−Xβ)\]}τ−p/2−1∝τ−A−1exp\[−Bτ\],

where A\=(n+p)/2 and B\=(Y−Xβ)T(Y−Xβ)/2. Marginalizing over _σ_2 gives

p(β|Y)\=∫p(β,τ|Y)dτ∝Γ(A)BA∫BAΓ(A)τ−A−1exp\[−Bτ\]dτ∝B−A∝\[(Y−Xβ)T(Y−Xβ)\]−(n+p)/2.

327\. The quadratic form is factored as

(Y−Xβ)T( Y−Xβ)\=YTY−2YTXβ+βTWβ\=YTY−2β^TWβ+βTWβ\=YTY−β^TWβ^+β^TWβ^−2β^TWβ+βTWβ\=nσ^2+(β−β^)TW(β−β^)

where W\=XTX, β^\=(W)−1XT Y is the usual least squares estimator, and nσ^2\=(Y−Xβ ^)T(Y−Xβ^)\= YTY−β^TWβ^. Therefore,

p(β|Y)∝\[(Y−Xβ)T(Y−Xβ)\]−(n+p)/2∝\[nσ^2+(β−β^)TW(β−β^)\]−(n+p)/2∝\[1+1nσ^2(β−β^)TW(β−β^)\]−(n+p)/2.

The marginal posterior of β is thus the _p_\-dimensional t-distribution with location vector β^, scale matrix σ^2(XTX)−1, and _n_ degrees of freedom.

The two-sample t-test in [Section 4.1.2](./12-chapter4.md#sec4_1_2) is a special case. If we parameterize the means as _μ_ for the first group and μ+δ for the second group, then **X**'s first column has all ones and its second column is _n_1 zeros followed by _n_2 ones, and β\=(μ,δ)T. This gives the least squares estimator as β^\=(Y¯,Y¯2−Y¯1)T and σ^2\=(n1σ^12+n2σ^22)/(n1+n2) and

(XTX)−1\=\[n1+n2n2n2n2\]−1\=1n1n2\[n2−n2−n2n1+n2\]\=\[1n1−1n1−1n11n1+1n2.\].

This gives the components of the joint posterior distribution of β, and since the marginal distributions of multivariate t are univariate, we have δ|Y∼tn\[Y¯2−Y¯1,σ^(1n1+1n2)\]

### A Gamma scale-mixture is half-Cauchy

Say τ|b∼Gamma(0.5,b) and b∼Gamma(0.5,1). Then the marginal distribution of _τ_ is

f(τ)\=∫0∞f(τ|b)f(b)db∝∫{b1/2τ−1/2exp(−bτ)}{b−1/2exp(−b)}db∝τ−1/2∫exp{−b(τ+1)}db∝τ−1/2(τ+1)−1.

Transforming to σ\=τ−1/2 so τ\=σ−2 gives

f(σ)∝(σ−2)−1/2(σ−1+1)−1|dτdσ|∝(σ−2)−1/2(σ−2+1)−1σ−3∝(1+σ2)−1

which is the kernel of the HC(1) PDF.

328\. 

### Variance derivation for the _R_2 prior

Assuming the set-up in (4.18) of [Section 4.2](./12-chapter4.md#sec4_2),

Var(ηi)\=E{Var(∑j\=1pXijβj|Xi)}+Var{E(∑j\=1pXijβj|Xi)}\=E{∑j\=1pXij2Var(βj|Xi)}+Var{∑j\=1pXijE(βj|Xi)}\=∑j\=2pE(Xij2)σ2τ2,

since _β_1 is fixed, and for j\>1 the _β_ _j_ are independent with E(βj|Xi)\=0 and Var(βj|Xi)\=σ2τ2. If we further assume that the covariates are standardized with Xij having mean 0 and variance 1, then E(Xij2)\=1 and Var(ηi)\=(p−1)σ2τ2.

### Proof of posterior consistency

Here we prove posterior consistency in the general but simple case with independent data and parameter with discrete support. Assume that:

* (A1)Yi∼iidf(y|θ) for i\=1,...,n
* (A2)The support is discrete, θ∈{θ1,θ2,...}\=S
* (A3)The prior is proper and the true value θ0∈S has positive prior probability, π(θ0)\>0
* (A4)The Kullback-Leibler divergence  
KL(θ)\=EY|θ0\[log(f(Y|θ0)f(Y|θ))\]  
 satisfies KL(θ)\>0 for all θ≠θ0.

Assumption (A4) ensures that the parameter is identifiable by asserting that on average the likelihood is higher for true value and any other value.

Theorem 1. _Assuming (A1)-(A4),_ Prob(θ\=θ0|Y1,...,Yn)→1 _as_ n→∞.

Proof 1. _For any_ θ∈S,

log\[p(θ|Y)p(θ0|Y)\]\=log\[π(θ)π(θ0)\]+∑i\=1nlog\[f(Yi|θ)f(Yi|θ0)\].

_By the law of large numbers_, 1n∑i\=1nlog\[f(Yi|θ)f(Yi|θ0)\]→−KL(θ), _and thus_

log\[p(θ|Y)p(θ0|Y)\]≈log\[π(θ)π(θ0)\]−nKL(θ).

_Therefore, as_ n→∞ , p(θ|Y)/p(θ0|Y)→0 _for any_ θ≠θ0, _and_ Prob(θ\=θ0|Y) _converges to one._

This proof can be generalized to continuous parameters by discretizng the support and making additional assumptions about the smoothness of the likelihood and prior density functions.

## A.4: 329\. Computational algorithms

### Integrated nested Laplace approximation (INLA)

INLA \[[135](./19-ref01.md#refbib135)\] is a deterministic approximation to the marginal posterior of each parameter that combines many of the ideas discussed in [Section 3.1](./11-chapter3.md#sec3_1). The method is most fitting in the special but common case where the parameter vector θ\=(α,β) can be divided into a low-dimensional α and a high-dimensional β whose posterior is approximately Gaussian. For example, in a random effects model ([Section 4.4](./12-chapter4.md#sec4_4)) α might include the variance components, and β might include all of the Gaussian random effects.

Evoking the Bayesian CLT (i.e., Laplace approximation) in [Section 3.1.3](./11-chapter3.md#sec3_1_3), assume that the conditional posterior of β conditioned on α is approximately

β|α,Y∼Normal(μ(α),Σ(α)),

and denote the corresponding density function as ϕ(β;μ(α),Σ(α)). We first use this approximation for the marginal distribution of the low-dimensional parameter α. Since p(α,β|Y)\=p(β|α,Y)p( α|Y), the marginal posterior of α can be written

p(α|Y)\=p(α,β|Y)p(β|α,Y).

Expanding around the MAP estimate β\=μ(α) and using the Laplace approximation for the denominator gives the approximation

p(α|Y)≈f(Y|α,β)π(α,β|Y)ϕ(β;μ(α),Σ(α))|β\=μ(α).

This low-dimensional distribution and can be evaluated using the methods in [Section 3.1](./11-chapter3.md#sec3_1), e.g., grid approximations or numerical integration.

The Laplace approximation can also be used to approximate the marginal distribution of each element of β. Let β−i be the elements of β excluding _β_ _i_. Following arguments similar to the approximation of the posterior of α,

p(βi|α,Y)∝f(Y |α,β)π(α,β)p(β−i|βi,α,Y).

This can be approximated using a Laplace approximation for p(β−i|βi,α,Y) around its posterior mode (\[[135](./19-ref01.md#refbib135)\] also consider faster approximations). Finally, to obtain p(β−i|Y) requires numerical integration over α, and therefore the Laplace approximation is nested within numerical integration.

### Metropolis–adjusted Langevin algorithm

Metropolis–Hastings sampling ([Section 3.2.1.2](./11-chapter3.md#sec3_2_1_2)) is a flexible algorithm but depends on finding a reasonable candidate distribution. A Gaussian random walk distribution for the candidate θ∗\=(θ1∗,...,θp∗) given the current value at the onset of iteration _s_, θ(s−1), is

θ∗|θ(s−1)∼Normal(θ(s−1),c2Ip),

330\. where c\>0 is a tuning parameter. This candidate is easy to code, very general and surprisingly effective. However, convergence can be improved by tailoring the candidate distribution to the problem at hand. We saw in [Section 3.2.1](./11-chapter3.md#sec3_2_1) that if the candidate distribution is taken to be the full conditional distribution, Metropolis–Hastings sampling becomes Gibbs sampling. While Gibbs sampling is free from tuning parameters and is thus easier to implement, it requires derivation of full conditional distributions which can be tedious and is not always possible.

The Metropolis-adjusted Langevin (MALA) algorithm \[[129](./19-ref01.md#refbib129)\] balances the strengths of random-walk Metropolis and Gibbs sampling. Rather than simply centering the candidate distribution on the current value, MALA uses the gradient of the posterior to push the candidate distribution towards the center of the distribution. This requires computing the gradient of the posterior, and thus the algorithm is more complex than a random walk, but the gradient is typically easier to derive and more generally available than the full conditional distribution required for Gibbs sampling.

Define the gradient vector of the log posterior as ∇(θ)\=\[∇1(θ),...,∇p(θ)\]T where

∇j(θ)\=∂∂θj{log\[f(Y|θ)\]+log\[π(θ)\]}

is the partial derivative with respect to the jth parameter. The candidate is

θ∗|θ(s−1)∼Normal(θ(s−1)+c22∇(θ(s−1)),c2Ip).

Unlike the random-walk candidate distribution, the MALA candidate distribution is asymmetric and requires including the candidate distribution in the acceptance ratio,

R\=f(Y|θ∗)π(θ∗)f(Y|θ(s−1))π(θ(s−1))∏j\=1pϕ{θj(s−1);θj∗+c22∇j(θ∗),c2}∏j\=1pϕ{θj∗;θj(s−1)+c22∇j(θ(s−1)),c2}

where ϕ(y;μ,σ2) is the Gaussian density function with mean _μ_ and variance _σ_2.

As with the standard Metropolis algorithm, the tuning parameter _c_ should be adjusted to give reasonable acceptance probability. Roberts and Rosenthal \[[129](./19-ref01.md#refbib129)\] argue that 0.574 is the optimal acceptance probability, but they claim that acceptance probabilities between 0.4 and 0.8 work well. In this chapter we have assumed that the candidate standard deviation _c_ is the same for all _p_ parameters and that the candidates are independent across parameters. Convergence can often be improved by adapting the candidate covariance to resemble the posterior covariance. Finally, we note that since MALA is simply a special type of MH sampling, it can be used within a larger Gibbs sampling algorithm just like MH sampling steps.

### Hamiltonian Monte Carlo (HMC)

MALA improves on random-walk Metropolis sampling by fitting the candidate distribution to the posterior by incorporating the gradient of the log posterior. However, for highly irregular posterior distributions (e.g., a U-shaped or donut-shaped posterior), one step along the gradient may not be sufficient to traverse the posterior. Hybrid Monte Carlo (HMC; also called Hamiltonian Monte Carlo) sampling \[[113](./19-ref01.md#refbib113)\] generalizes MALA to take multiple random steps guided by the gradient. The simple version in [Algorithm 1](./11-chapter3.md#algo3_1) has two tuning parameters: the step size _c_ and the number of steps _L_. If L\=1, then this algorithm reduces to MALA with _c_ as the candidate standard deviation. Motivation, extensions and tuning of this algorithm are beyond the scope of this text but form the basis for the software STAN \[[28](./19-ref01.md#refbib28)\].331\. 

Algorithm 7 Hamiltonian MCMC


`1: Initialize θ(0)=(θ1(0),...,θp(0))`
`2: **for** s=1,...,S **do**`
`3:    sample z∼Normal(0,Ip)`
`4:    set θ∗=θ(s−1)`
`5:    set z∗=z+c∇(θ∗)/2`
`6:   **for** l=1,...,L **do**`
`7:       set θ∗=θ∗+cz ∗`
`8:       set z∗=z∗+c∇(θ∗)`
`9:       **end for**`
`10:         set z∗=z∗−c∇(θ∗)/2`
`11:         set R=f(Y|θ∗)π(θ∗)f(Y|θ(s−1))π(θ(s−1))⋅exp(−z∗Tz∗/2)exp(−zTz/2)`
`12:         sample U∼Uniform(0,1)`
`13:            **if** U<R **then**`
`14:          θ(s)=θ∗`
`15:          **else**`
`16:                  θ(s)=θ(s−1)`
`17:      **end if**`
`18:  **end for**`

### Delayed rejection and adaptive Metropolis

Delayed Rejection and Adaptive Metropolis (DRAM, \[[69](./19-ref01.md#refbib69)\]) is a combination of two ideas: delayed rejection Metropolis \[[107](./19-ref01.md#refbib107)\] and adaptive Metropolis \[[70](./19-ref01.md#refbib70)\]. Adaptive Metropolis allows the covariance of the candidate distribution to evolve across iterations. The intuition is that if the posterior is irregularly shaped, then a different proposal distribution is needed depending on the current state of the chain. Assuming a Gaussian random-walk candidate distribution, θ∗|θ(s−1)∼Normal(θ(s−1),V(s−1)), the user sets an initial p×p covariance matrix V(0) that is then adapted as

V(s)\=c(V^(s)+δI)

where V^(s) is the sample covariance of the previous samples θ(1),...,θ(s−1), δ\>0 is a small constant to avoid singularities and c\=2.42/p \[[56](./19-ref01.md#refbib56)\].

Delayed rejection Metropolis replaces the standard single proposal in Metropolis–Hastings sampling with multiple proposals considered sequentially. The first stage is a usual Metropolis–Hasting step with candidate θ∗|θ(s−1)∼q(θ∗|θ(s−1)) and acceptance probability

R(θ∗,θ(s−1))\=min{1,p(θ∗|Y)q(θ(s−1)|θ∗)p(θ(s−1)|Y)q(θ∗|θ(s−1))}.

If the first candidate is rejected, a second candidate is proposed as θ′|θ∗,θ(s−1)∼Q(θ′|θ∗,θ(s−1)) and accepted with probability

min{1,p(θ′| Y)q(θ′|θ∗)Q(θ′|θ∗,θ(s−1))\[1−R(θ′,θ∗)\]p(θ(s−1)|Y)q(θ(s−1)|θ∗)Q(θ(s−1)|θ∗,θ′)\[1−R(θ(s−1),θ∗)\]}.

332\. The notation becomes cumbersome, but this can be iterated beyond two candidates. DRAM combines these two ideas by using adaptive Metropolis to tune the Gaussian candidate distributions used for _q_ and _Q_.

### Slice sampling

Slice sampling \[[112](./19-ref01.md#refbib112)\] is a clever way to apply Gibbs sampling when the full conditional distributions do not belong to known parametric families of distributions. Slice sampling introduces an auxiliary variable (i.e., a variable that is not an actual parameter) U\>0 and draws samples from the joint distribution

p∗(θ,U)\=I\[0<U<p(θ|Y)\].

By construction, under _p_\* the marginal distribution of θ is

∫I\[0<U<p(θ|Y)\]dU\=p(θ|Y),

and therefore if samples of (θ,U) are drawn from _p_\*, then the samples of θ follow the posterior distribution. Also, Gibbs sampling can be used to draw samples from _p_\* since the full conditional distributions are both uniform

1. U|θ,Y∼Uniform on \[0,p(θ|Y)\]
2. θ|U,Y∼Uniform on P(U)\={θ;p(θ|Y )\>U}.

Therefore, slice sampling works by drawing from the joint distribution of (θ,U), discarding the samples of _U_ and retaining the samples from θ. The most challenging step is to make a draw from P(U) (see the figure below). For some posteriors P(U) has a simple form, and samples can be drawn directly. In other cases, θ can be drawn from a uniform distribution with a domain that includes P(U) until a sample falls in P(U).

![A single smooth posterior density curve for theta rises from near zero at the left, forms a small bump around zero point five, then reaches a higher peak near zero point seven before tapering toward zero at one. A horizontal line at height two is drawn across the plot and labeled U. Two vertical dashed lines enclose the region where the density exceeds this threshold, marking the interval whose probability mass is labeled P of U beneath the curve.](./images/ufig10_18.jpg) 

**Illustration of slice sampling.** The curve is the posterior density p(θ|Y), the horizontal line represents the auxiliary variable _U_ (i.e., the “slice”), and the bold interval is P(U)\={θ;p(θ|Y)\>U}.
