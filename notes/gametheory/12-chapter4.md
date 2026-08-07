# 4Empirical Entry Games

DOI: [10.1201/b23262-4](./https___doi.org_10.1201_b23262-4.md)

## 4.1 Introduction

This chapter revisits the entry game presented in [Chapter 2](./10-chapter2.md). In this game, we have two firms considering whether or not to enter a market. The issue is that it is costly to enter and both firms would prefer to have the market to themselves. In equilibrium, we will tend to have one firm in the market, but the game theory may not predict which one.

The chapter applies an entry game to the question of which markets the mega bookstores, Borders and Barnes & Noble entered in the 1990s. The 90s saw massive changes in book retailing. Changes in distribution technology made it profitable for big box book stores to enter the market with a huge range of titles as well as other products such as music, games, and even coffee. The chains, Borders and Barnes & Noble led the charge by purchasing smaller chains and entering green fields sites. Which markets did these firms enter? What would have happened if they had been allowed to merge?

The chapter introduces an empirical model of entry based on the decision making model of Dan McFadden. The empirical model is extended to allow the firms to make decisions that are dependent on each other, a game. The chapter analyzes the effect of a merger, if that merger would have occurred in the early 90s.

## 4.2 Empirical Model

The section begins with a description of a standard choice model. This model was originally developed by Dan McFadden in the early 1970s to analyze consumer choice problems. McFadden was interested in who would use the new subway system that had been built in San Francisco, the Bay Area Rapid Transit (BART) system. Here we are adapting the model to analyze the choice of which market the firm will enter. The model is generalized to allow unobserved characteristics of the market to be correlated across firms. It is generalized again to allow firms to make choices that are dependent upon each other.

### 4.2.1 Single Firm Entry Model

By the year 2000, Barnes & Noble had bookstores in 283 counties in the United States, but this is out of over 3,000 counties. Placing these large stores in a location is not cheap, particularly if they involve new construction. So which counties dd Barnes & Noble choose to locate?

Assume that the firm's latent profits from locating in a particular county are as follows.

π1i\=Xi′β1+ξ1i(4.1)

where Xi are market characteristics such as population size, _β_1 determines how these characteristics are mapped into profits for the firm and ξ1i is unobserved characteristics of the market and can be thought of as representing entry costs for the firm. These unobserved characteristics are unobserved by the econometrician (that's you) but they are observed by the firm itself.

If the data include information on the market such as population size (_pop_) and median income (_income_), then the decision to enter can be written as follows.

π1i\=β10+β11popi+β12incomei+ξ1i(4.2)

For Firm 1, their profits in market _i_ depend on popi and incomei as determined by the parameters _β_11 and _β_12, respectively.

In matrix notation we have that the vector of parameters is written as follows.

β1\=\[β10β11β12\](./4.3)

Similarly, the matrix of observed characteristics for _N_ markets as follows.

X\=\[1pop1income11pop2income2⋯1popNincomeN\](./4.4)

The first column is just 1's. In the matrix algebra, this column is multiplied by the _β_10 parameter to give a constant across all the markets. We use the notation Xi′ to emphasize that we are using the _i_th row of the matrix and that is being multiplied by the vector of parameters _β_1.

Barnes & Noble enter the market if and only if the following inequality holds.

Xi′β1+ξ1i\>0 orξ1i\>−Xi′β1(4.5)

So if the unobserved entry costs are low enough, then Barnes & Noble will enter the market.

We expect firm profits to determined by various factors affecting demand for books, costs associated with selling books, and fixed costs associated with the location of the store. If we assume that the unobserved costs of entry are distributed standard normal, ξi∼N(0,1), then the probability of entry is Φ(−Xi′β1), where Φ() is the cumulative distribution of the standard normal. This is the **probit** model introduced in [Chapter 1](./09-chapter1.md). We can estimate the parameters _β_1 using the glm() procedure we used in [Chapter 1](./09-chapter1.md).

maxβ1∑i\=1Ny1ilog(Φ(−Xi′β1))+(1−y1i)log((1−Φ(−Xi′β1))(4.6)

To estimate the model we can find the _β_1 that maximizes the likelihood of the data, where y1i is the observed entry decision for each county.[1](./12-chapter4.md#fn4_1)

### 4.2.2 Multiple Firm-Independent Entry Model

Taking baby steps to our full model, consider a model where we have two firms entering a market but the two firms are making decisions independently of each other. This could be a model of entry of Barnes & Noble and Best Buy. Both are big box stores whose decision to enter a market will be based on similar things, both observed and unobserved by the econometrician. However, with one focused on books and the other focused on electronics, it is unlikely that their decisions to enter a particular market will be dependent on each other.

The two firms will enter market _i_ if and only if the following inequalities hold.

Xi′β1+ξ1i\>0Xi′β2+ξ2i\>0(4.7)

Again Xi represents observed characteristics of the market such as population size. These characteristics are mapped into each firm's profit function by the parameter vectors _β_1 and _β_2.

So far this doesn't seem to be different from the model in the previous section. The difference is that we can allow the unobserved characteristics of the market to be correlated across the two firms. Assume that {ξ1i,ξ2i}∼N(μ,Σ) where μ\={0,0} and

Σ\=\[1ρρ1\](./4.8)

In words, we allow the unobserved characteristics of the markets to be distributed standard bivariate normal. The parameter ρ∈\[−1,1\] represents the correlation across the two firms. This parameter is likely to depend on how similar the two firms are in terms of their customer base and costs of setting up the store. A fast food restaurant and big box book store may have unobserved characteristics that are not vary correlated ρ\=0, while two big box stores may have a highly positive correlation (a high _ρ_).

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./12-chapter4.md#fn4_14b)We log the probabilities so that we don't run into problems because the numbers are too small for the computer to represent.

Let 1 denote entry and 0 denote the choice not to enter. In this model we observe four cases, neither firm enters {0,0}, Firm 1 enters but Firm 2 does not {1,0}, Firm 1 doesn't enter but Firm 2 does {0,1}, and both firms enter {1,1}.

[Figure 4.1](./12-chapter4.md#fig4_1) presents the four cases given the distribution of the unobserved characteristics of the markets for the two firms. The pattern of entry across markets will tell us if the unobserved characteristics are correlated across firms. If we see lots of cases where both firms do the same thing, both enter the market or neither enters the market, then that is consistent with positively correlated unobserved characteristics. If we see lots of cases where there is just one firm in the market but which is pretty evenly distributed, then that is consistent with the unobserved characteristics being uncorrelated or negatively correlated across the two firms.

![A square graph is split into four quadrants by intersecting lines. The horizontal axis is labeled xi subscript 1 with a vertical line from negative x subscript i prime beta subscript 1. The vertical axis is labeled xi subscript 2 with a horizontal line from negative x subscript i prime beta subscript 2. The bottom left quadrant shows 0 comma 0, bottom right shows 1 comma 0, top left shows 0 comma 1, and top right shows 1 comma 1.](./images/fig4_1.jpg)

[Figure 4.1](chapter4) Empirical implications of the two firm independent entry model. Firm 1's entry is denoted in the first position of the brackets. If both firms have unobserved characteristics that are low then neither will enter {0, 0}, if Firm 1's unobserved characteristic is high then Firm 1 will enter {1, 0}, if Firm 2's unobserved characteristic is also high the Firm 2 will also enter {1, 1}.

Again we want to find the _β_1, _β_2 and _ρ_ that maximize the likelihood of the observed data which is represented by _y_1 and _y_2.

### 4.2.3 Entry Game

After all of that set up, we can write down our empirical entry game. Like in the previous section we have two firms considering whether or not they should enter market _i_. They will enter if the following inequalities hold.

Xi′β1−D2iα1+ξ1i\>0Xi′β2−D1iα2+ξ2i\>0(4.9)

where D1i∈{0,1} and D2i∈{0,1}. There are a couple of differences between these inequalities and the inequalities in Equation (4.7). The first is that there are two extra parameters _α_1 and _α_2. These represent the reduced profits associated with competing with the other firm. This only occurs if the other firm enters the market. The profits for Firm 1 are lower if D2i\=1 by the amount _α_1, where D2i\=1 if and only if the second inequality holds. The profits for Firm 2 are lower if D1i\=1 which only occurs if the first inequality holds.

Now you may be able to start seeing the issue. The first inequality depends on the outcome of the second inequality which depends on the outcome of the first inequality! To see what is going on consider a version of the figure presented above.

[Figure 4.2](./12-chapter4.md#fig4_2) represents the game. If the unobserved benefits for both firms are high (fixed cost of entry is low), then both firms will enter {1,1}, if they are low, then neither will enter {0,0}. If they are very high for Firm 1 and very low for Firm 2, then Firm 1 will enter {1,0}. Similarly the other way {0,1}. There is also the intermediate outcome where the model does not make a clear prediction on what will happen. It could be that Firm 1 enters while Firm 2 does not, or it could be the other way around. The model clearly predicts that one firm will enter, it just does not predict which.

![A square grid is divided into nine equal boxes by two vertical and two horizontal lines. Each box shows entry pairs.](./images/fig4_2.jpg) Long Description for Figure 4.2 

The horizontal axis is labeled xi subscript 1, with tick marks at negative x subscript i prime beta subscript 1 and negative x subscript i prime beta subscript 1 plus alpha subscript 1\. The vertical axis is labeled xi subscript 2, with tick marks at negative x subscript i prime beta subscript 2 and negative x subscript i prime beta subscript 2 plus alpha subscript 2\. The vertical and horizontal lines extend from these labeled points. From bottom left to top right, the entries in each box are as follows: 0 comma 0; 1 comma 0; 1 comma 0; 0 comma 0; 0 comma 1 or 1 comma 0; 1 comma 0; 0 comma 1; 0 comma 1; and 1 comma 1.

[Figure 4.2](chapter4) Empirical implications of the entry game. Firm 1's entry is denoted in the first position of the brackets. The region in the middle square has an in determinant outcome. One firm will enter, but it is not clear which.

## 4.3 Empirical Analysis: Bookstore Entry with **R**

The section presents the code for estimating the empirical models presented above. To show the progression from the original entry model to the entry game, the section presents code for all three models. The first model is just a probit, so you could use the glm() function baked into **R**. However, including the code for this case makes it easier to see how the more complicated models work.

### 4.3.1 Single Firm Entry Model in **R**

The estimation algorithm is **maximum likelihood**. In this method, we find the parameter values that maximize the likelihood that the data we observe was generated by a particular model. We will need a function that calculates the probability that the observed data occurred given a particular parameter value. There are a couple of different ways of doing that in **R**. In this section, we will calculate the probabilities **numerically**. That is the section presents a method for approximating the probability by having the computer do a lot of calculations. This method is not as efficient as using a built-in C\-based function, but it is easier to see what is going on, particularly as the problem gets more complicated.


`_> set.seed(123456789)_`
`_> K = 10000_`
`_> Z1 = rnorm(K)_`
This code generates K = 10000 pseudo-random draws from a standard normal distribution. The function rnorm() is the **R** function for drawing from a standard normal. These are **global** variables, meaning that they are available to any function we write. The function set.seed() is used to make sure that the results can be exactly replicated.


`_>   fentry = function(X, beta1) {_`
`_+     X1 = as.matrix(cbind(1, X))_`
`_+     N = dim(X1)[1]_`
`_+     xi = Z1_`
`_+     Xb = X1%*%beta1_`
`_+     p = rep(0, N)_`
`_+     for(k in 1:K) {_`
`_+       pik = Xb + xi[k]_`
`_+       p = p + (pik > 0)_`
`_+     }_`
`_+     return(p/K)_`
`_+   }_`
`_>   floglik = function(X, y, beta) {_`
`_+     epsilon = 1e-5_`
`_+     p = fentry(X, beta)_`
`_+     return(-mean((y == 1)*log(p + epsilon) +_`
`_+                   (y == 0)*log(1 - p + epsilon)))_`
`_+   }_`
The function f\_entry() takes in the matrix of data X (without the column of 1's) and the vector of parameters beta. The matrix X consists of columns of data stating observed characteristics of each market, such as population size and median income.

The probability of entering market _i_ is determined by the probability that profits will be positive given a large number of possible values for the unobserved characteristic xi.

The function f\_loglik calculates the probability that the model is true given the observed data. This code is equivalent to Equation (4.6). This function takes a vector of outcomes, y. The vector y states whether or not the firm entered each of the markets in the data where 1 denotes entry and 0 denotes that the firm did not enter the market. The function transforms the probability into logs so that we don't run into problems where the numbers we are calculating are smaller than the smallest number the computer can handle. Later when we apply this function to an actual problem, we will benefit from the fact that the optimal value for beta is the same for the log-transformed function as it is for the original function. Notice that there is a negative sign in front of the mean() function. This is there because the optimize algorithm used defaults to find the minimum so the minimum of the negative log-likelihood is the maximum log-likelihood. The value epsilon is a small number designed to make sure that the computer doesn't crash if it tries to calculate the log of zero.

### 4.3.2 Multiple Firm-Independent Entry in **R**


`_> Z2 = rnorm(K)_`
This time we need a distribution with two dimensions. So the first step is to create another large set of random numbers drawn from a standard normal function.


`_>   f2entry = function(X, beta1, beta2, rho) {_`
`_+     N = dim(X)[1]_`
`_+     xi1 = Z1_`
`_+     xi2 = Z2*sqrt(1 - rho^2) + rho*Z1_`
`_+     Xb1 = X%*%beta1_`
`_+     Xb2 = X%*%beta2_`
`_+     p00 = p01 = p11 = rep(0, N)_`
`_+     for(k in 1:K) {_`
`_+       pi1k = Xb1 + xi1[k]_`
`_+       pi2k = Xb2 + xi2[k]_`
`_+       p00 = p00 + (pi1k < 0 & pi2k < 0)_`
`_+       p01 = p01 + (pi1k < 0 & pi2k > 0)_`
`_+       p11 = p11 + (pi1k > 0 & pi2k > 0)_`
`_+     }_`
`_+     return(list(p00 = p00/K_,`
`_+                 p01 = p01/K_,`
`_+                 p11 = p11/K))_`
`_+   }_`
`_>   floglik2 = function(X, y, beta1, beta2, rho) {_`
`_+     epsilon = 1e-10_`
`_+     Lik = f2entry(X, beta1, beta2, rho)_`
`_+     return((y[,1] == 0 & y[,2] == 0)*log(Lik$p00 + epsilon) +_`
`_+            (y[,1] == 0 & y[,2] == 1)*log(Lik$p01 + epsilon) +_`
`_+            (y[,1] == 1 & y[,2] == 1)*log(Lik$p11 + epsilon) +_`
`_+            (y[,1] == 1 & y[,2] == 0)*log(1 -_`
`_+                                            Lik$p00 -_`
`_+                                            Lik$p01 -_`
`_+                                            Lik$p11 +_`
`_+                                            epsilon))_`
`_+   }_`
`_>   floglik2int = function(par, X, y) {_`
`_+     X = as.matrix(cbind(1, X))_`
`_+     J = dim(X)[2]_`
`_+     beta1 = par[1:J]_`
`_+     beta2 = par[(J+1):(2*J)]_`
`_+     rho = -1 + 2*exp(par[2*J+1])/(1 + exp(par[2*J+1]))_`
`_+     return(-mean(floglik2(X, y, beta1, beta2, rho)))_`
`_+   }_`
Just by counting lines of code, we see that things are a lot more complicated when we have two firms whose decisions are independent but correlated. The function f\_2entry() has parameters for Firm 1 (beta\_1) and Firm 2 (beta\_2) and the correlation between the unobserved term (rho). The unobserved term xi\_2 is a function of both Z\_1 and Z\_2. The higher rho the more it weights Z\_1. The function determines the probability of observing three cases neither enter, only Firm 1 enters, and both enter. The fourth case is calculated as the residual because probabilities must add to 1.

The code now includes an additional function f\_loglik\_2\_int(). This is an intermediate function designed to be used by the **R**'s optimization algorithm optim(). In this case, the parameter rho is restricted to be between -1 and 1\. It is good coding practice to allow the optimization algorithm to choose what ever values it likes, but then transform those values into the restricted set required by the model. To do this the code uses exp(x)/(1 + exp(x)), also known as the **softmax** function. This function takes any value of x and turns it into a number that lies between 0 and 1.

### 4.3.3 Entry Game in **R**

While the entry game is quite a bit more complicated than the previous model, the estimator is not that different. There are two additional parameters alpha\_1 and alpha\_2 but everything else is pretty much the same. The big difference is that the model cannot distinguish two cases in the data. The model predicts when one firm will enter the market, but it does not predict which firm that will be. Therefore, we need to combine those two cases in order to estimate our model.


`_> fentrygame = function(X, beta1, beta2, alpha1, alpha2_`
`_+                         , rho) {_`
`_+   N = dim(X)[1]_`
`_+   xi1 = Z1_`
`_+   xi2 = Z2*sqrt(1 - rho^2) + rho*Z1_`
`_+   Xb1 = X%*%beta1_`
`_+   Xb2 = X%*%beta2_`
`_+   p00 = p11 = rep(0, N)_`
`_+   for(k in 1:K) {_`
`_+     pi1k = Xb1 + xi1[k]_`
`_+     pi2k = Xb2 + xi2[k]_`
`_+     p00 = p00 + (pi1k < 0 & pi2k < 0)_`
`_+     p11 = p11 +_`
`_+       (pi1k - alpha1 > 0 & pi2k - alpha2 > 0)_`
`_+   }_`
`_+   return(list(p00 = p00/K_,`
`_+               p11 = p11/K))_`
`_+ }_`
The function f\_loglik\_game\_int() is pretty similar to the equivalent function in the previous section. The difference is that there are two additional parameters. We will restrict these two parameters to be positive. We will impose the result that an increase in competition lowers profits (or does nothing).


`_>   floglikgame = function(X, y, beta1, beta2, alpha1_,`
`_+                            alpha2, rho) {_`
`_+     epsilon = 1e-10_`
`_+     Lik = fentrygame(X, beta1, beta2, alpha1, alpha2, rho)_`
`_+     return((y[,1] == 0 & y[,2] == 0)*log(Lik$p00 + epsilon) +_`
`_+            (y[,1] == 1 & y[,2] == 1)*log(Lik$p11 + epsilon) +_`
`_+            ((y[,1] == 1 & y[,2] == 0) +_`
`_+               (y[,1] == 0 & y[,2] == 1))*log(1 -_`
`_+                                                Lik$p00 -_`
`_+                                                Lik$p11 +_`
`_+                                                epsilon))_`
`_+   }_`
`_>   floglikgameint = function(par, X, y) {_`
`_+     X = as.matrix(cbind(1, X))_`
`_+     J = dim(X)[2]_`
`_+     beta1 = par[1:J]_`
`_+     beta2 = par[(J+1):(2*J)]_`
`_+     alpha1 = exp(par[2*J+1])_`
`_+     alpha2 = exp(par[2*J+2])_`
`_+     rho = -1 + 2*exp(par[2*J+3])/(1 + exp(par[2*J+3]))_`
`_+     return(-mean(floglikgame(X, y, beta1, beta2, alpha1_,`
`_+                                alpha2, rho)))_`
`_+   }_`
## 4.4 Empirical Analysis: Bookstore Entry using**R**

Barnes & Noble dates itself to the 1800s, but in the early 1990s, the firm developed the super bookstore. The big box of bookstores. The objective was to carry a huge range of books, music, games, and even food and coffee. It revolutionized book retail in the United States.

Where did Barnes & Noble choose to enter? What determined those locations?

### 4.4.1 Data

The data used here is from [Adams and Basker (2025)](./25-refbib.md#ref2). The authors combine information from publicly available census data and published directories of retail bookstores.

[Table 4.1](./12-chapter4.md#tbl4_1) shows the difference between counties with and without Borders and Barnes & Noble stores. Unsurprisingly, the firms entered counties with larger populations, richer counties, counties with higher educated population, and counties with more bookstores in 1990.

__[Table 4.1](chapter4) Mean County Characteristics by Presence of Barnes & Noble and Borders.__
| College             |         |           |                |    |
| ------------------- | ------- | --------- | -------------- | -- |
| Population          | Income  | Share (%) | Bookstores (#) |    |
| None                | 37,857  | 33,513    | 15             | 2  |
| Only Barnes & Noble | 275,435 | 43,210    | 27             | 14 |
| Only Borders        | 548,218 | 49,118    | 31             | 40 |
| Both                | 272,191 | 46,476    | 26             | 12 |

Notes: Presence of Barnes & Noble and Borders referes to the year 2000\. There are 2,792 counties with neither chain, 155 counties with only Barnes & Noble, 36 counties with only Borders, and 128 counties with both chains. Population, income, and college share use data from 2000\. Income refers to median county-level household income in 2000\. College share is the share of population aged 25 and older with a college degree in 2000\. Bookstores is the total number of bookstores in the county in 1990.

### 4.4.2 Estimation of Single Firm Entry

The model presented in Equation (4.5) can be estimated by combining data on the location of Barnes & Noble stores with various economic and demographic information at the county level. In addition to these data, we have information on the number of book stores in the county in 1990, which is generally earlier than the Barnes & Noble super bookstores came into existence. We can similarly model the entry of Borders.

In this analysis, there is no game. Barnes & Noble and Borders are assumed to make their entry decisions optimally, but these decisions are completely independent of each other. To estimate the parameters of the model, we can either use a maximum-likelihood estimator and the functions f\_loglik() and f\_enter() or we can use the glm() procedure introduced in [Chapter 1](./09-chapter1.md).


`_> file = paste0(dir, “book2000.csv”)_`
`_> dt = fread(file)_`
` `
` `
`_> glm1 = glm(enter ~ logpop2000 + medincome + college +_`
`_+              stores1990_,`
`_+            data = dt_,`
`_+            family = binomial(link = “probit”))_`
This function is used to estimate cases where the outcome is binary, enter or did not enter, and where the unobserved characteristic of the market is distributed as a normal distribution with mean of 0 and standard deviation of 1\. This is a **probit model**.


`_>   init = glm1$coefficients_`
`_>   data = data.frame(y = dt$enter_,`
`_+                     pop = dt$logpop2000_,`
`_+                     income = dt$medincome_,`
`_+                     college = dt$college_,`
`_+                     stores1990 = dt$stores1990)_`
`_>   data = na.omit(data)_`
`_>   res1 = bs(init, floglik_,`
`_+             y = data$y_,`
`_+             X = cbind(data$pop_,`
`_+                       data$income_,`
`_+                       data$college_,`
`_+                       data$stores1990))_`
The glm1 results are used as the initial starting value for the maximum likelihood estimator using f\_loglik(). The code then creates the data to be used by the estimator, including information on which stores Barnes & Noble entered, population size of the county, percentage of college graduates in the county and the number of book stores in the county in 1990\. The object res1 stores the results. This uses a function called bs() which creates pseudo samples from the data set and uses the optim() function to determine the parameter values that maximize the likelihood using the f\_loglik() function.[2](./12-chapter4.md#fn4_2)


`_> dt$enter2 = ifelse(dt$borders > 0, 1, 0)_`
`_> glm2 = glm(enter2 ~ logpop2000 +_`
`_+              medincome +_`
`_+              college + stores1990, data = dt_,`
`_+            family=binomial(link=“probit”))_`
The data set is adjusted to remove observations with missing values. The function na.omit() is used to do this.[3](./12-chapter4.md#fn4_3)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./12-chapter4.md#fn4_24b)The bs() function is available from the github page for the book.

[3](./12-chapter4.md#fn4_34b)**R** uses NA to denote missing values.


`_>   init = glm2$coefficients_`
`_>   data = data.frame(y = dt$enter2_,`
`_+                     pop = dt$logpop2000_,`
`_+                     income = dt$medincome_,`
`_+                     college = dt$college_,`
`_+                     stores1990 = dt$stores1990)_`
`_>   data = na.omit(data)_`
`_>   res2 = bs(init, floglik_,`
`_+             y = data$y_,`
`_+             X = cbind(data$pop_,`
`_+                       data$income_,`
`_+                       data$college_,`
`_+                       data$stores1990))_`
The code above similarly creates a data set for estimating the choice of market to enter for Borders stores. It estimates the probit model using the bs() function and stores the results in the object res2.

### 4.4.3 Estimation of Two Firm Entry Model

[Table 4.2](./12-chapter4.md#tbl4_2) presents the results from the different entry models discussed above. The first model assumes that the two firms make optimal decisions that are independent of each other. In addition, it assumes that the unobserved characteristics faced by the two firms in each market are independent across the two firms.

__[Table 4.2](chapter4) Results from estimates of the three models. The first set of columns labeled “Probit” are estimates assuming that the two firms are making entry decisions that are both strategically independent and statistically independent. The second set of columns labeled “BiProbit” assumes the firm entry decisions are strategically independent. It allows unobserved characteristics of the market to be correlated across firms. The two columns labeled “Game” refer to the case where the entry decisions of the two firms are both strategically and stochastically dependent. The columns labeled “SD” refer to the bootstrap standard errors.__
| Probit          | SD     | BiProbit | SD     | Game | SD     |      |
| --------------- | ------ | -------- | ------ | ---- | ------ | ---- |
| const\_1        | ‒15.06 | 0.22     | ‒15.03 | 0.09 | ‒15.11 | 0.25 |
| Pop\_1          | 1.08   | 0.02     | 1.09   | 0.01 | 1.07   | 0.01 |
| Income\_1       | ‒1.03  | 0.44     | ‒1.04  | 0.25 | ‒0.76  | 0.48 |
| College\_1      | 5.68   | 0.66     | 5.51   | 0.34 | 5.65   | 0.55 |
| Stores\_1990\_1 | 0.28   | 0.09     | 0.28   | 0.07 | 0.37   | 0.11 |
| const\_2        | ‒11.57 | 0.26     | ‒11.54 | 0.11 | ‒11.37 | 0.20 |
| Pop\_2          | 0.66   | 0.03     | 0.66   | 0.01 | 0.65   | 0.02 |
| Income\_2       | 1.75   | 0.51     | 1.74   | 0.25 | 1.31   | 0.80 |
| College\_2      | 2.49   | 0.70     | 2.59   | 0.33 | 2.70   | 0.55 |
| Stores\_1990\_2 | 0.52   | 0.08     | 0.49   | 0.06 | 0.79   | 0.11 |
| alpha\_1        | 0.73   | 0.18     |        |      |        |      |
| alpha\_2        | 0.70   | 0.12     |        |      |        |      |
| rho             | ‒0.08  | 0.10     | 0.47   | 0.10 |        |      |

The two-firm model is similar to single firm entry model. The entry decisions of the two firms are independent of each other but the unobserved characteristics of the markets may be correlated across firms. We say that the firms are strategically independent but the markets are statistically dependent. This is a **biprobit model**. The code again creates the data set for the analysis removes missing values and uses the bs() function calling f\_loglik\_2\_int. It stores the results in res3.


`_>   init = c(glm1$coefficients_,`
`_+            glm2$coefficients_,`
`_+            0)_`
`_>   data = data.frame(y1 = dt$enter_,`
`_+                     y2 = dt$enter2_,`
`_+                     pop = dt$logpop2000_,`
`_+                     income = dt$medincome_,`
`_+                     college = dt$college_,`
`_+                     stores1990 = dt$stores1990)_`
`_>   data = na.omit(data)_`
`_>   res3 = bs(init, floglik2int_,`
`_+             y = cbind(data$y1_,`
`_+                       data$y2)_,`
`_+             X = cbind(data$pop_,`
`_+                       data$income_,`
`_+                       data$college_,`
`_+                       data$stores1990))_`
### 4.4.4 Estimation of the Entry Game Model

For the third model, we have two firms entering the markets and again the unobserved characteristics of the two firms are correlated across markets. This time the decision to enter depends on the choice of the other firm. Decisions are now dependent. This model uses the same data as the previous model. It uses the bs() function with the f\_loglik\_game\_int() likelihood function and saves the results in the object res4.


`_> init = c(glm1$coefficients_,`
`_+          glm2$coefficients_,`
`_+          0_,`
`_+          0_,`
`_+          0)_`
`_> res4 = bs(init, floglikgameint_,`
`_+           y = cbind(data$y1_,`
`_+                     data$y2)_,`
`_+           X = cbind(data$pop_,`
`_+                     data$income_,`
`_+                     data$college_,`
`_+                     data$stores1990))_`
The table of results also presents information on the standard errors of the estimates of the parameters. These values are determined using a **bootstrap method**. This method approximates how estimates may vary when a different sample is used.[4](./12-chapter4.md#fn4_4)

[Table 4.2](./12-chapter4.md#tbl4_2) presents the estimated parameter values for the three models discussed above. The _α_ parameters state that having two firms lowers profits. The store locations are positively associated with population size, education, and the number of existing bookstores. It is unclear if income has any impact. Finally, the unobserved entry costs are highly correlated across the two firms.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [4](./12-chapter4.md#fn4_44b)The bootstrap method approximates estimates from a new sample by resampling the current data and restimating the model using the new pseudo-sample.

### 4.4.5 Model Fit

We can use the estimated parameters to simulate the model and compare the predicted entry decisions to the actual entry decisions. This provides one test of the model's fit. The code, not shown, simulates the model 1,000 times and compares the predicted entry decisions to the actual entry decisions. The results are presented in [Table 4.3](./12-chapter4.md#tbl4_3).

__[Table 4.3](chapter4) Comparison of predictions of the three models to the actual outcomes. The higher percentage on the diagonal the better fit of the model. All models are good at predicting the high probability event, which is no entry by either firm. The non-strategic models are best at predicting when there will be two firms in two-firm markets, while the game theory model is best at predicting the one-firm market.__
| None           | One Firm | Two Firm |      |
| -------------- | -------- | -------- | ---- |
| Probit: None   | 97.2     | 42.3     | 9.4  |
| Probit: One    | 2.5      | 38.8     | 34.9 |
| Probit: Two    | 0.3      | 18.9     | 55.6 |
| BiProbit: None | 97.1     | 41.6     | 9.0  |
| BiProbit: One  | 2.6      | 40.0     | 35.9 |
| BiProbit: Two  | 0.3      | 18.4     | 55.1 |
| Game: None     | 97.1     | 41.0     | 9.0  |
| Game: One      | 2.6      | 41.7     | 36.6 |
| Game: Two      | 0.3      | 17.3     | 54.4 |

[Table 4.3](./12-chapter4.md#tbl4_3) presents model fit results for the three models. There is not much difference between the three estimators, although the game theory model is slightly better at predicting the one-firm market.

## 4.5 Policy Analysis using **R**

What would happen if Barnes & Noble and Borders had merged? Here we are not going to worry about the effect on prices but on which counties the merged firm would enter. Assume that the _α_ parameters remain the same post merger.[5](./12-chapter4.md#fn4_5) The difference the merger brings is that it allows the firms to coordinate entry. This analysis assumes that the merged firm will keep the two distinct brands.

To determine what would happen in this alternate universe, we can simulate thousands of outcomes in the markets for which have data using the parameter values estimated above.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [5](./12-chapter4.md#fn4_54b)Given that the _α_ parameters are accounting for both diversion between stores and prices, we would expect them to be smaller but not zero post merger.

[Table 4.4](./12-chapter4.md#tbl4_4) presents summary of simulations of a merger between Borders and Barnes & Noble. The simulations tend to predict more counties with stores than we actually see, but fewer counties where both stores are present. The model predicts that the merger will lead to fewer markets with a Borders and Barnes & Noble. Not presented is the variation in the estimates, but it is pretty clear that there will be fewer markets with both stores. As some consumers prefer one or the other, people in those markets are worse off. We can think of that as a quality reduction. In addition, the reduced competition in those markets is likely to lead to higher prices.[6](./12-chapter4.md#fn4_6)

__[Table 4.4](chapter4) Comparison of actual entry to simulated entry in 2000 and simulated entry under a merger.__
| Actual        | Sim  | Merge |      |
| ------------- | ---- | ----- | ---- |
| none          | 2919 | 2895  | 2895 |
| BN or Borders | 170  | 191   | 238  |
| both          | 128  | 93    | 46   |

The value in the first row of columns 2 and 3 is the same. The merger changes whether or not the market is will have two firms or one, but not whether or not at least one firm will enter. The reason is that the game explicitly assumes that the two firms coordinate on entry. [Chapters 5](./13-chapter5.md) and [8](./17-chapter8.md) consider games where the firms cannot coordinate. In these cases, the merger may actually lead more firms to enter the market.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [6](./12-chapter4.md#fn4_64b)See the analysis presented in [Chapter 3](./11-chapter3.md).

## 4.6 Discussion and Further Reading

The analysis is based on [Adams and Basker (2025)](./25-refbib.md#ref2). The authors analyze the entry of mega bookstores in the US using publicly available census data and directories of retail bookstores. The entry game analysis is based on [Bresnahan and Reiss (1991a)](./25-refbib.md#ref16) and [Tamer (2003)](./25-refbib.md#ref54). [Chapters 5](./13-chapter5.md), [6](./15-chapter6.md), and [8](./17-chapter8.md) revisit this problem under different modeling assumptions. [Chapter 5](./13-chapter5.md) revisits the problem assuming that the outcome of the game in the data is a mixed strategy Nash equilibrium.

Game theory provides a better explanation of the entry decisions we observe, however it can also makes the analysis more challenging. Entry problems are a type of coordination game and like other coordination games they tend to have multiple equilibria. This means for a set of parameter values, a model makes multiple predictions about what we will see in the data. This makes the models a challenge to estimate. The solution presented here follows [Bresnahan and Reiss (1991a)](./25-refbib.md#ref16) which simply combines all the equilibria into one observed outcome. [Tamer (2003)](./25-refbib.md#ref54) presents a method for estimating a more efficient model. Many papers in empirical industrial organization make alternative assumptions that lead to unique predictions from the game. [Chapter 6](./15-chapter6.md) discusses this option.

Borders and Barnes & Noble did not propose to merger in the 1990s or 2000s. Did they contemplate it? Were they dissuaded by the FTC's challenge of the merger between Blockbuster and Hollywood Video in 1999 and again in 2005?[7](./12-chapter4.md#fn4_7) Borders went into bankruptcy proceedings in 2011 and Barnes & Noble acquired some of its intellectual property.[8](./12-chapter4.md#fn4_8)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [7](./12-chapter4.md#fn4_74b)<https://www.nytimes.com/2005/03/26/business/media/blockbuster-ends-bid-for-rival.html>

[8](./12-chapter4.md#fn4_84b)<https://www.barnesandnobleinc.com/about-bn/history/>. Accessed on 2/20/23.

[_OceanofPDF.com_](./https___oceanofpdf.com)
