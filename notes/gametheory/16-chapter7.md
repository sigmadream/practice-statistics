# 7Repeated Games

DOI: [10.1201/b23262-7](./https___doi.org_10.1201_b23262-7.md)

## 7.1 Introduction

Repeated games are the lens through which we are adapting our thinking on competition and competition policy. In the late 1990s, the economics of competition policy changed pretty dramatically. Game theory and structural econometrics provided antitrust authorities with new tools for modeling mergers. [Chapters 3](./11-chapter3.md) and [4](./12-chapter4.md) introduce methods used to analyze the impact of retail mergers. While these models substantially improved our ability to understand competition and predict the outcomes of mergers, something was not quite right. Our approach to collusion and markets with collusive pricing remained rudimentary. Using the new models to analyze collusion was like pushing a square peg into a round hole.

The analysis presented here on collusive pricing is influenced by two papers. First, economists Nathan Miller, Gloria Sheu, and Matthew Weinberg presented compelling evidence that our new models worked poorly when used to analyze the US beer industry. Their paper, “Oligopolistic Price Leadership and Mergers: The United States Beer Industry,” was published in the _American Economic Review_ in 2021, and suggested a major rethink in the models we need for analyzing competition. Second, Canadian and Australian economists, David Byrne and Nic de Roos, analyzed pricing in the retail gasoline market in Perth Western Australia. Their paper, “Learning to Coordinate: A study in retail gasoline” was published in the _American Economic Review_ in 2019\. The authors use daily pricing data from a large number of retailers to show evidence of price leadership and coordination.

This chapter shows how the lens of repeated games can be used to explain firm behavior and pricing. Repeated interactions change the strategic relationships quite dramatically. In a single shot game, players don't have to account for the consequences of their actions. In repeated games they do.

Using data from Perth retail gasoline stations, the chapter presents two models of competition and pricing based on repeated interactions. The first model is a standard static pricing model that was introduced in [Chapter 3](./11-chapter3.md). The second is a collusive oligopoly pricing model. This model shows how pricing is constrained by the incentives of firms to cheat and choose a lower price. The chapter estimates the parameters using data from gas stations in Perth in a period where margins where low in 2008\. It then compares the profit margins predicted by the collusive model to the actual profit margins of the same firms in 2012\. While well-intentioned, price transparency regulation by the state government seems to cause Perth motorists to pay more for their petrol (gasoline).

The chapter runs a merger simulation using each of the two models of pricing behavior. It suggests that merger analysis should account for these changes when analyzing the likely effect of the merger. The predicted price increase using a repeated game model may be substantially higher than the predicted price increase using the standard static game presented earlier in the book.

## 7.2 Repeated Prisoner's Dilemma

The prisoner's dilemma is the most famous games in game theory. The game shows that the predicted outcome may not be the outcome that is best for both players. It may not be Pareto efficient. How much of that result is due to the set up of the game? What if players had to account for each other's actions? What if the game repeated?

This section presents the prisoner's dilemma game and shows how the Nash equilibrium change when dynamics are added. In particular, it shows when playing Cooperate is supported as a subgame perfect Nash equilibrium.

### 7.2.1 Prisoner's Dilemma

The game set up is as follows. We will relabel the strategies of the game presented in [Chapters 1](./09-chapter1.md) and [2](./10-chapter2.md).

* Players: Player 1 and Player 2
* Strategies: Cooperate or Defect
* Payoffs:  
   1. {Cooperate, Cooperate}: {3, 3}  
   2. {Cooperate, Defect}: {0, 5}  
   3. {Defect, Cooperate}: {5, 0}  
   4. {Defect, Defect}: {2, 2}

In this game, the Nash equilibrium is {Defect, Defect}, even though both would be better off with the {Cooperate, Cooperate} outcome. The question is whether making a slight change to the set up of the game changes the predicted outcome.

### 7.2.2 Normal Form Representation

[Chapter 1](./09-chapter1.md) introduced the normal form representation of a game.

[Table 7.1](./16-chapter7.md#tbl7_1) presents the **normal form** of the prisoner's dilemma game. We can determine the Nash equilibrium by looking at the second column and seeing whether Player 1 is better off choosing the top row or the bottom row. Player 1 gets nothing from choosing the top row and 2 from the bottom. Similarly for Player 2, we can check the Nash equilibrium by looking at the bottom row and seeing if Player 2 wants the first column or the second column. Player 2 gets 0 from the first column and 2 from the second column. {Defect, Defect} is a Nash equilibrium.

__[Table 7.1](chapter7) Normal form representation prisoner's dilemma game, with two players, Player 1 and Player 2\. For Player 1, their choices are the rows and their payoffs are listed first in each cell.__
| P1,P2     | Cooperate | Defect |
| --------- | --------- | ------ |
| Cooperate | 3, 3      | 0, 5   |
| Defect    | 5, 0      | 2, 2   |

### 7.2.3 Finitely Repeated Game

Consider a version of the game where the static prisoner's dilemma (above) is repeated a finite number of times (_T_ times). This could be 20 periods, for example.

The game is different. The players are the same but the strategies are completely different. A **strategy** is a function that maps from the complete history of the game to an action. In period 1, the complete history is null, so the strategy is just the action choices {Cooperate, Defect}. In period 2, the complete history is what ever the outcome was in period 1\. There are 4 possibilities as listed above. From each possibility, the strategy states which action the player will choose. In period 3, things are even more complicated. Now the history includes 16 possibilities. For each of the 4 possible outcomes in period 1, there are 4 possible outcomes in period 2\. Again the strategy states what the player will do in period 3 given each of the 16 possible histories. As you can imagine, for a game with twenty periods, there are an awful lot of possibilities.

What is the Nash equilibrium of this game? It is probably better to ask, what isn't. Consider any set of strategies where one player plays Cooperate in the last period. The game in the last period is essentially the same as a one period game. We know from the analysis of the one period game that the best response must be a strategy where they play Defect in the last period. Now consider sets of strategies where the player always plays Defect in the last period but Cooperate in the second to last period. Again, the best response must be a strategy that states the player plays Defect in the second to last period and Defect in the last period. Using this logic, we can show that the Nash equilibria are associated with playing Defect in every period. That is to say, despite adding a lot of complexity to the game, the prediction doesn't change.

### 7.2.4 Infinitely Repeated Prisoner's Dilemma

What if we make one more change to the game? This time, we repeat the game above for an infinite number of periods. Does this seem like a reasonable change? Infinity is quite a long time. For the strategic behavior to change, we don't literally need an infinite number of periods, we just need the players to be unsure when the game is going to end. It is better to think of a finite period game as one where every player knows exactly when the game is going to end. While an infinitely repeated game is one where players don't know exactly when the game will end.

For this case, we need to add another parameter to the payoffs, r∈\[0,1). This represents a discount rate and is a number between 0 and 1\. The practical reason for adding a discount rate is that with an infinite number of periods we cannot analyze the game. The payoffs are infinite under all possible strategies. As you may remember from calculus, an infinite sum of a non-decreasing sequence is infinite. But with a discount rate, we can create a decreasing sequence that decreases fast enough that our infinite sum sums to a finite number. A player's utility given a particular strategy _s_ is an infinite sum of a discounted sequence of per period payoffs.

U(s)\=∑t\=1∞rtπt(s)(7.1)

where _s_ is the strategy. If we assume that the payoff to the players in each period (πt(s)) is bounded and there is a discount rate so that the payoff is lower in the future than today (r<1), then the infinite sum, U(s)<∞. Now we can analyze the game because different strategies may have different payoffs.

We can also use a very useful trick. If πt(s)\=π(s), that is, if the per-period payoff is constant for a given strategy _s_ and _r_ is strictly between 0 and 1, which it is, we have the following simplification.

U(s)\=π(s)1−r(7.2)

So if the discount rate is 0.9, then the total payoff is 10 times the per-period payoff for the strategy _s_.

Where does this discount rate come from? What does it mean? The most obvious way to think about it is as an interest rate. Actually, one minus the interest rate. When we are thinking about dynamic decisions it makes sense for decision makers to refer to the interest rate when determining the value of future decisions. Here, it may be reasonable to think of the discount rate as representing the probability that the game continues into the next period.

### 7.2.5 Cooperation as a Nash Equilibrium

Can having both players cooperate every period be supported as a Nash equilibrium of the infinitely repeated prisoner's dilemma? Yes.

To construct the supporting strategies, we need to allow players to punish defection. There are many strategies that have this feature but the simplest is called the **grim-trigger strategy**. In this strategy, the player plays Cooperate unless the history includes one of the players playing Defect. In that case, the player plays Defect forever. So “grim” is for the fact that the threat is the worst possible kind and “trigger” is for the fact that the strategy involves a simple event that changes behavior.

A strategy maps from the complete history of the game into the choice of Defect or Cooperate in each period. It is a set of functions for each period _t_, {st}t\=1∞. Each function is st:Ht→{Defect,Cooperate}, where _Ht_ is the complete history of possible actions that could have happened in the previous t−1 periods. In this case, if _Ht_ includes Defect, then st(Ht)\=Defect while if it doesn't, then st(Ht)\=Cooperate.

Is the grim-trigger a subgame perfect Nash equilibrium? Consider the case where both players are playing the grim-trigger. Payoffs are as listed in the previous section.

1. Grim-trigger: 31−r
2. In period _t_ play Defect: 5+r(21−r)

So against the grim-trigger in which the other player is playing Cooperate, if you play Cooperate each period you get 3 each period. If you play Defect in one period, you get 5 for that period but the grim strategy kicks in and you get 2 for every period. For the grim-trigger to be an equilibrium choice (1) must give a higher payoff than choice (2).

The player would prefer to play grim-trigger if the following inequality holds.

31−r\>5+r(21−r)3\>5(1−r)+r23\>5−5r+2r0\>2−3r3r\>2r\>23(7.3)

As long as the players care about the future enough, _r_ is high enough, they will play Cooperate against the grim-trigger. If they are very myopic, _r_ is low, then they will cheat and take the high payoff today.

Just to tie things up we should also check that both players will keep playing Defect after the trigger has been pulled. Looking back at the discussion above we see that this must be the case.

## 7.3 Bertrand Competition

This section revisits the Bertrand pricing game presented in [Chapter 3](./11-chapter3.md). It presents this model assuming logit demand, which is the more common assumption in the literature. It uses this model to estimate parameters of demand using data for retail gasoline from Perth Australia.

### 7.3.1 Two Firm Model

Consider a model where we have two firms that choose prices. The two firms sell similar products but are not exactly the same. For example, the products may be stores that are located in different places. The differentiation between the products is enough to induce some market power for each firm. That is for a small price increase, the firm is not going to lose all its customers. Below we will use this model to analyze retail gasolone. The gasoline itself is identical as it literally comes from the same pipe.[1](./16-chapter7.md#fn7_1) What is different between stations is there location and their brand.

In this model, Firm 1 chooses their price (_p_1) to maximize the margin which is price less cost (_c_1) multiplied by Firm 1's share (s1(p1,p2))

maxp1(p1−c1)s1(p1,p2)(7.4)

The logit model assumes demand has the following form.

s1(p1,p2)\=exp(δ1)exp(δ1)+exp(δ2)(7.5)

where δ1\=−α1p1+ξ1 and δ2\=−α2p2+ξ2. In this model, demand for Firm 1 is determined by firm specific value _ξ_1 and by Firm 1's price _p_1. Sensitivity to price is determined by the parameter _α_1.

### 7.3.2 _J_ Firm Model

Assume that we have _J_ firms and logit demand.

δj\=−αjpj+ξj(7.6)

sj(pj,p−j)\=exp(δj)1+∑j′\=1Jexp(δj′)(7.7)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./16-chapter7.md#fn7_17b)Some stations may add in additives at the end.

where _pj_ is the price charged by firm _j, αj_ represents the price sensitivity of firm _j_'s customers, and _ξj_ is a set of characteristics other than price that determine demand. The notation −j means all the other firms that are not _j_.

### 7.3.3 Nash Equilibrium in **R**

The following are the functions for determining the Nash equilibrium using the logit demand system. The function f\_share() determines the shares from the logit model given the vectors of _ξ_s, _α_s and prices. The function pi\_f() determines the vector of firm profits, while pi\_i() determines the profits for firm _i_ given a vector of prices for the other firms. The function pi\_i\_opt() determines firm _i_'s best response to a set of prices, while pi\_opt() determines the vector of best response. The function ne\_f() determines the Nash equilibrium vector of prices.


`_>   fshare = function(xi, price, alpha) {_`
`_+     delta = xi - alpha*price_`
`_+     expdelta = c(1,exp(delta))_`
`_+     return((expdelta/sum(expdelta))[-1])_`
`_+   }_`
`_>   pif = function(xi, price, alpha, cost) {_`
`_+     return((price - cost)*fshare(xi, price, alpha))_`
`_+   }_`
`_>   pii = function(i, pi, xi, price, alpha, cost) {_`
`_+     price[i] = pi_`
`_+     return(pif(xi, price, alpha, cost)[i])_`
`_+   }_`
`_>   piiopt = function(i, xi, price, alpha, cost) {_`
`_+     ai = optimize(f=pii, interval=c(0, 400)_,`
`_+                    i = i_,`
`_+                    xi = xi_,`
`_+                    price = price_,`
`_+                    cost = cost_,`
`_+                    alpha = alpha_,`
`_+                    maximum = TRUE)_`
`_+     return(ai$maximum)_`
`_+   }_`
`_>   piopt = function(xi, price, alpha, cost) {_`
`_+     N = length(xi)_`
`_+     pricenew = rep(NA, N)_`
`_+     for(i in 1:N) {_`
`_+       pricenew[i] = piiopt(i,xi,price,alpha,cost)_`
`_+     }_`
`_+     return(pricenew)_`
`_+   }_`
Nash equilibrium is determined using a similar iterative approach that we used to determine equilibrium of the Cournot model in [Chapter 3](./11-chapter3.md). It uses a while() to determine when the sequence of price choices converges.


`_> nef = function(xi, price, alpha, cost_,`
`_+                 tol=1e-10, maxiter=10000_,`
`_+                 trace=0) {_`
`_+   price0 = price_`
`_+   diff = sum(abs(price0))_`
`_+   iter = 1_`
`_+   converge = FALSE_`
`_+   while(diff > tol & iter < maxiter) {_`
`_+     price1 = piopt(xi, price0, alpha, cost)_`
`_+     diff = sum(abs(price1 - price0))_`
`_+     price0 = price1_`
`_+     if(trace > 0) {_`
`_+       print(diff)_`
`_+       print(iter)_`
`_+     }_`
`_+     iter = iter + 1_`
`_+   }_`
`_+   if(diff < tol) {converge=TRUE}_`
`_+   return(list(price=price1,converge=converge))_`
`_+ }_`
## 7.4 Empirical Analysis: Retail Gasoline Pricing using **R**

This section estimates the parameters of the model assuming the Perth retail gas oline market is priced competitively, i.e., consistent with static Bertrand Nash equilibrium.

### 7.4.1 Perth Gas Price Data

[Figure 7.1](./16-chapter7.md#fig7_1) presents the average margins by week for the gas stations. While there is a lot of variation in margins in 2008 and 2009, the average stays pretty steady. Things seems to change in 2010, with margins steadily increasing. Why would that be?

![Weekly price margins are plotted with dots from 2008 to 2012 on the horizontal axis and from 0 to 16 on the vertical axis labeled as margin in cents per liter. A smooth line shows the average margin trend. From 2008 to 2010, the line increases from 6 to 7 and then dips slightly. After 2010, the line rises gradually and reaches about 9 by 2012. The dots are scattered around the line at each time point, showing variation in weekly prices. All data are approximate.](./images/fig7_1.jpg)

[Figure 7.1](chapter7) Plot of weekly margins from 2008, 2009, 2010, and 2011\. There is a lot of variation from 2006 to 2010 but prices are moving around the same average. After 2010, the average price starts to increase.

### 7.4.2 Estimating Parameters

In the analysis below, we aggregate up to the brand level and estimate the pricing game between the brands. We assume that margins and shares are the result of Nash equilibrium of the static pricing game. The data provide daily prices, daily wholesale price, the location of the stations and the number of stations for each brand.


`_> file = paste0(dir, “perthgasdata.csv”)_`
`_> dt = fread(file)_`
`_> dt[,.(_`
`_+   margin = mean(margin)_,`
`_+   date = mean(date)_`
`_+   )_,`
`_+          by = c(“week”, “year”)] |>_`
`_+   ggplot(aes(x=date, y=margin)) +_`
`_+   geompoint(color = “gray”) +_`
`_+   geomsmooth(se = FALSE) +_`
`_+   labs(title = “Margin (cents/liter)”_,`
`_+        x = “”_,`
`_+        y = “”)_`
To estimate the firm's marginal costs, we regress prices on the distance between the station and the Kwinana terminal, which is located south of Perth. The assumption is that variation in prices due to distance is determined by the trucking costs of the fuel. We use a quadratic on distance, brand dummies and week dummies to estimate marginal costs for each station. These values are added to the terminal price to get the estimated cost for each station. As we don't have access to quantity information, the brand share is assumed to be equal to the proportion of stations that the brand has.

The model requires estimates of two parameters for each firm. The price sensitivity parameter, _αj_, and the unobserved quality parameter, _ξj_. The first is found from the firm's first order condition when demand is determined by a logit model.

αj\=1(pj−cj)(1−sj)(7.8)

The second value is from inverting the logit demand to get the unobserved characteristic as a function of the observed prices, shares, and the parameter _αj_.

ξj\=αjpj+log(sj)−log(s0)(7.9)

where _s_0 is the outside share.

### 7.4.3 Parameter Estimates

From Equation (7.8) and (7.9), we determine _α_ and _ξ_ for each firm from the observed prices and shares. We are assuming that the observed prices and shares are determined as the outcome of a static Bertrand Nash equilibrium. We are also assuming that the outside good are the independent stations and the smaller brands, Mobil, Wesco and Better Choice. Finally, we are assuming that Caltex and Caltex Woolworths make pricing decisions as if they are the same firm.

To do this analysis we will aggregate up to annual average prices, costs, margins, and shares.

The first step is to select the variables and the year (2008) we will use. Also redefine some of the brand names to make the analysis easier.


`_> dt2 = dt |>_`
`_+   filter(year == 2008) |>_`
`_+   select(_`
`_+     date_,`
`_+     store = TRADINGNAME_,`
`_+     brand = BRANDDESCRIPTION_,`
`_+     price = PRODUCTPRICE_,`
`_+     margin_`
`_+     )_`
`_> dt2$brand[grep(“Caltex”, dt2$brand)] = “Caltex”_`
`_> dt2$brand[which(dt2$brand %in% c(“Independent”_,`
`_+                              “Mobil”_,`
`_+                              “Wesco”_,`
`_+                              “Better Choice”))] = “Independent”_`
The next step is to calculate the shares by determining the number of stations for each brand and then calculating the share for each brand.


`_> stores = dt2[, .N, by = brand]_`
`_> stores$shares = stores$N/sum(stores$N)_`
`_> dt2 = merge(dt2, stores, by = c(“brand”))_`
The next step creates a data set with prices, margins, shares, and costs averaged up to the brand level for 2008\. This also calculates _α_ for each brand.


`_> dt3 = dt2[, .(_`
`_+   price = mean(price, na.rm = TRUE)_,`
`_+   margin = mean(margin, na.rm = TRUE)_,`
`_+   share = mean(shares, na.rm = TRUE)_,`
`_+   cost = mean(-margin + price, na.rm = TRUE)_,`
`_+   alpha = 1/(mean(margin, na.rm = TRUE)*(1 - mean(shares_,`
`_+                                              na.rm = TRUE)))_`
`_+   )_,`
`_+           by = brand]_`
Next, is calculating _ξ_ for each brand. This calculation assumes that the independent stores are the outside option.


`_> indexind = grep(“Independent”, dt3$brand)_`
`_> dt3$xi = NA_`
`_> dt3$xi[-indexind] =_`
`_+   dt3$alpha[-indexind]*dt3$price[-indexind] +_`
`_+   log(dt3$share[-indexind]) -_`
`_+   log(dt3$share[indexind])_`
The final step is to use the function ne\_f() to determine the Nash equilibrium.


`_> af = nef(dt3$xi[-indexind]_,`
`_+            dt3$price[-indexind]_,`
`_+            dt3$alpha[-indexind]_,`
`_+            dt3$cost[-indexind])_`
[Table 7.2](./16-chapter7.md#tbl7_2) presents the estimates for _α_ and _ξ_ for each brand. These values reconcile the observed equilibrium margins and the observed equilibrium shares. BP is able to have higher margins and only slightly lower share because its customers are less price sensitive than Caltex.

__[Table 7.2](chapter7) The table presents the prices, margins, market share, and estimates for _α_ and _ξ_. BP is able to have higher margins and only slightly lower share because its customers are less price sensitive than for Caltex.__
| Brand | Price         | Cost   | Share  | _α_  | _ξ_  |       |
| ----- | ------------- | ------ | ------ | ---- | ---- | ----- |
| 1     | Ampol         | 153.70 | 142.53 | 0.02 | 0.09 | 12.93 |
| 2     | BP            | 148.53 | 140.77 | 0.23 | 0.17 | 26.16 |
| 3     | Caltex        | 146.14 | 139.46 | 0.30 | 0.21 | 32.56 |
| 4     | Coles Express | 144.65 | 139.92 | 0.10 | 0.23 | 34.28 |
| 5     | Eagle         | 152.32 | 147.32 | 0.00 | 0.20 | 26.47 |
| 6     | Gull          | 144.00 | 138.06 | 0.11 | 0.19 | 27.70 |
| 7     | Liberty       | 145.23 | 138.06 | 0.03 | 0.14 | 19.70 |
| 8     | Peak          | 139.98 | 136.33 | 0.04 | 0.29 | 39.42 |
| 9     | Shell         | 152.11 | 142.57 | 0.07 | 0.11 | 17.20 |
| 10    | United        | 138.25 | 135.42 | 0.02 | 0.36 | 48.59 |

Can you compare the parameter estimates using the algebraic approach to the numeric approach to determining the Nash equilibrium? Are they exactly the same? Would you expect them to be?

## 7.5 Repeated Oligopoly

The standard differentiated goods Bertrand model of price competition suggests that the outcome we would expect, while higher than perfect competition, it is not collusion. In the static game, it is always better for the firms to “cheat” and lower their prices in order to increase profits. The question then is whether we would expect to see collusion when firms interact repeatedly.

The section adapts the static Bertrand game to an infinitely repeated setting and determines the optimal pricing under collusion.

### 7.5.1 Collusive Equilibrium

Choosing the static equilibrium Bertrand prices in each period can be supported as a subgame perfect Nash equilibrium of the infinitely repeated game. Like with prisoner's dilemma, it is always fine to play Defect every period.

Given this, we ask whether or under what circumstances can a trigger strategy support collusion. Let _πN_ denote the per-period profits in a Nash equilibrium of the static game, _πC_ denote the collusive profits and _πD_ the profits from defecting and choosing an optimal price when the other firm is offering the collusive price. We have πD\>πC\>πN. This is exactly the same as for a prisoner's dilemma.

Like the prisoner's dilemma, collusion can be supported by a trigger strategy if the following inequality holds.

πC1−r\>πD+r(πN1−r)πC\>(1−r)πD+rπNr(πD−πN)\>πD−πCr\>πD−πCπD−πN(7.10)

The number on the bottom is larger than the number on the top, and so for a large enough _r_ collusion is a Nash equilibrium of the infinitely repeated oligopoly game.

### 7.5.2 Identifying Collusion

Identifying collusion for policy or academic purposes is quite different from identifying collusion for criminal prosecution. A criminal case requires hard evidence, not some cool econometric specification. The best evidence includes credible witnesses, audio recordings, video recordings, and written documents. You may be surprised to learn that there isn't that much economics involved prosecuting a criminal collusion case. There is undercover work, wire tapping, etc, but no economics. Once the criminal case is proven, economists are called in to estimate damages.[2](./16-chapter7.md#fn7_2) Here again, identifying the collusion is not that difficult. At least it is not that difficult given that the FBI has already completed the task. The case record includes the dates when the collusion occurred and (hopefully) dates when the collusion did not occur. The hard part for the econometrician is working out which part of the difference in prices is due to the collusion and which is due to other changes.

Identifying collusion without the assistance of the FBI is difficult. To observe prices and quantities from a market, there is enough exogenous variation such that we can estimate the elasticity of demand. If the products are differentiated, we can use Bertrand Nash equilibrium to identify marginal costs and mark-ups. While assuming a static equilibrium or a dynamic collusive equilibrium gives different estimates of the mark ups and marginal cost, without some other information we can't tell the difference from the data. If we had data on marginal costs, then suddenly things get a lot easier. We can match the implied marginal costs from our proposed behavioral assumptions and see which ones fits better. Alternatively, if the courts tell us that there was a period where the static equilibrium determined prices, then we can use that period to identify marginal costs. We can then compare the implied markups from the static equilibrium to the observed markups to estimate the **super markups** associated with collusion.[3](./16-chapter7.md#fn7_3)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./16-chapter7.md#fn7_27b)This is usually the amount equal to the consumer surplus lost from the collusion.

[3](./16-chapter7.md#fn7_37b)These are markups that are larger than we would expect from firms choosing price consistent with equilibrium of a static Bertrand game.

### 7.5.3 Choosing Super Markups

Choosing the collusive price seems like a simple enough problem for the firms. The collusive price should be equal to the price that a monopoly would charge. Not so fast Sonny Jim! Sure, the monopoly price would be optimal if all the firms behaved as one firm. They are not one firm and they do not behave as such. In each period, each firm may prefer to renege on the deal and choose a lower price, gaining share while everyone else is charging the collusive price.

The collusive price is the solution to the following maximization problem. The super markup (smu) is the difference between the collusive price and the static Nash equilibrium price where the marginal cost is normalized to zero.

maxpcπc(pc)s.t.πd(pc)+rπNE1−r≤πc(pc)1−r(7.11)

where πc(pc) are the per-period profits that one firm makes when everyone chargers _pc_, πd(pc) is the per-period profit when one firm is able to deviate and charge a lower price knowing that everyone else is charging _pc_, and πNE is the per-period profit in a static Bertrand Nash equilibrium. The firm's preferences over profits in future periods is determined by the parameter _r_.

The problem states that firms can charge any collusive price they like as long as all the firms are unwilling to cheat on the agreement. In general, we find that the higher the price, the greater the value in cheating on the agreement.

In general, the colluding firms cannot charge anything they want, in fact they can't even charge the monopoly price. They are constrained both by the demand system and by the incentive of firms to cheat on the agreement.

### 7.5.4 Estimating Collusive Prices in **R**

The following functions are used to determine the collusive price when demand is determined by the logit model. The collusive price is assumed to be equal to the Nash equilibrium price plus a **super markup**. The function f\_smu\_share() determines the vector shares at a particular smu. The function pi\_d() determines the optimal deviation profits given a particular smu. This places a constraint on what smu can be chosen.


`_>   fsmushare = function(smu, xi, price, alpha) {_`
`_+     fshare(xi, price + smu, alpha)_`
`_+   }_`
`_>   pid = function(pd, i, smu, xi, price, alpha, cost) {_`
`_+     price = price + smu_`
`_+     price[i] = pd_`
`_+     share = fshare(xi, price, alpha)_`
`_+     return(share[i]*(pd - cost[i]))_`
`_+   }_`
The function pi\_smu() determines the vector of profits for a particular smu and pi\_smu\_int() is an intermediate function for optim().


`_>   pismu = function(smu, xi, price, alpha, cost_,`
`_+                   r, lambda, pine) {_`
`_+     N = length(xi)_`
`_+     pic = (price + smu - cost)*fsmushare(smu, xi, price_,`
`_+                                             alpha)_`
`_+     PId = matrix(NA, N, 2)_`
`_+     for(i in 1:N) {_`
`_+       ai = optimize(f = pid_,`
`_+                     interval = c(0,2000)_,`
`_+                     i = i_,`
`_+                     smu = smu_,`
`_+                     xi = xi_,`
`_+                     price = price_,`
`_+                     alpha = alpha_,`
`_+                     cost = cost_,`
`_+                     maximum = TRUE)_`
`_+       PId[i,1] = i_`
`_+       PId[i,2] = ai$objective_`
`_+       #print(i)_`
`_+     }_`
`_+     return(pic -_`
`_+              lambda*(((1 - r)*PId[,2] +_`
`_+              r*pine - pic)^2))_`
`_+   }_`
`_>   pismuint = function(par, xi, price, alpha_,`
`_+                         cost, r, pine) {_`
`_+     smu = par[1]_`
`_+     lambda = 1_`
`_+     return(sum(pismu(smu, xi, price_,`
`_+                       alpha, cost, r_,`
`_+                       lambda, pine)))_`
`_+   }_`
## 7.6 Empirical Analysis: Collusion in Perth Gas Stations using **R**

This section estimates the extent of collusion in the the Perth retail gasoline market.

### 7.6.1 Super Markups

With all the parameters of the model estimated from the data under the static Bertrand Nash assumption, we use the parameters to determine the super markup.

As above, to determine the super markup, we need to determine the static Nash profits, the profits from cheating, and we need to make an assumption about discounting. We assume r is equal to 0.9\. What happens at different values?


`_> marginne = (af$price - dt3$cost[-indexind])_`
`_> sharene = fsmushare(0, dt3$xi[-indexind]_,`
`_+                        af$price, dt3$alpha[-indexind])_`
`_> pine = marginne*sharene_`
`_> a1 = optimize(pismuint_,`
`_+               c(0,20)_,`
`_+               xi = dt3$xi[-indexind]_,`
`_+               price = af$price_,`
`_+               alpha = dt3$alpha[-indexind]_,`
`_+               cost = dt3$cost[-indexind]_,`
`_+               r = 0.9_,`
`_+               pine = pine_,`
`_+               maximum = TRUE)_`
`_> a1$maximum_`
`  [1] 4.589335`
`_> a1$maximum/mean(marginne)_`
`  [1] 0.7117553`
In this set up, the super markup is 4.59 cents a liter or 71 percent of the average static Nash margins. Not bad! If we look at [Figure 7.1](./16-chapter7.md#fig7_1) we see that margins increased about 4 cents a liter between 2008 and 2012\. This suggests that the collusive model is a more accurate representation of pricing behavior than the static pricing model.

### 7.6.2 Analyzing Mergers with Collusion

Previously we have discussed the effect of mergers when firm competition and pricing can be modeled as a static Bertrand Nash equilibrium. What if the firms prices are really being determined as a collusive agreement? Will the merger affect prices? You may think that the collusive price is as high as prices could get. What impact could the merger have?

Mergers can cause prices to increase when firms are colluding. With collusion, the merger leads to two changes in the market. First, it has an effect on the static Bertrand Nash equilibrium price and thus the punishment that can be imposed. Second, it changes the set of constraints that are placed on the problem. Post merger, there is one less firm that is trying to cheat on the deal. In theory, these two effects work in different directions. The merger increases profits from the punishment phase, which makes incentivizing firms to collude harder. On the other hand, the number of firms that must be kept in line has been reduced.

Consider a merger between BP and Peak. Peak is very small and so is unlikely to be of concern using standard static Bertrand Nash pricing. To model the effect of the merger, we need to calculate the observed characteristics of the merged firm.


`_> indexm = 9_`
`_> dt3$brand[indexm]_`
`  [1] “Peak”`
`_> bps = dt3$share[2]/(dt3$share[2] + dt3$share[indexm])_`
The variable bp\_s is the share of the merged firm that is BP. The following code calculates the new characteristics of the merged firm. The new firm's costs are the weighted average of each firms average costs. Similarly, the new firm's price sensitivity parameter and unobserved characteristics are weighted averages of the two firms. Do these assumptions make sense? What happens under alternative assumptions?


`_>   brandm = dt3$brand[-c(indexind, indexm)]_`
`_>   brandm[2] = paste0(dt3$brand[indexm], “ and “, dt3$brand[2])_`
`_>   xim = dt3$xi[-c(indexind, indexm)]_`
`_>   alpham = dt3$alpha[-c(indexind, indexm)]_`
`_>   alpham[2] = bps*dt3$alpha[2] + (1 - bps)*dt3$alpha[indexm]_`
`_>   pricem = dt3$price[-c(indexind, indexm)]_`
`_>   pricem[2] = bps*dt3$price[2] + (1 - bps)*dt3$price[indexm]_`
`_>   costm = dt3$cost[-c(indexind, indexm)]_`
`_>   costm[2] = bps*dt3$cost[2] + (1 - bps)*dt3$cost[indexm]_`
Given the change caused by the merger, we can calculate the new Bertrand Nash equilibrium as well as the new equilibrium collusive price.


`_>   afm = nef(xim, pricem, alpham_,`
`_+              costm, maxiter=1000000)_`
`_>   marginm = (afm$price - costm)_`
`_>   sharem = fsmushare(0, xim, afm$price, alpham)_`
`_>   pinem = marginm*sharem_`
`_>   a2 = optimize(pismuint_,`
`_+                 c(0,20)_,`
`_+                 xi = xim_,`
`_+                 price = afm$price_,`
`_+                 alpha = alpham_,`
`_+                 cost = costm_,`
`_+                 r = 0.9_,`
`_+                 pine = pinem_,`
`_+                 maximum = TRUE)_`
If the firms are playing Bertrand static game, then merger has the following impact on prices. Prices will increase 0.4 percent.


`_> (mean(afm$price) - mean(af$price))/mean(af$price)_`
`  [1] 0.004122352`
If the firms are playing a collusive game, then the merger has the following impact on prices. Prices will increase 1.2 percent.


`_> (mean(afm$price+a2$maximum) -_`
`_+     mean(af$price+a1$maximum))/mean(af$price+a1$maximum)_`
`  [1] 0.01238215`
` `
`_> sumtabm = cbind(_`
`_+   as.numeric(af$price)_,`
`_+   as.numeric(c(afm$price[1:7], NA, afm$price[8:9]))_,`
`_+   as.numeric(af$price) + a1$maximum_,`
`_+   as.numeric(c(afm$price[1:7], NA, afm$price[8:9]))_`
`_+ a2$maximum_`
`_+ )_`
`_> rownames(sumtabm) = dt3$brand[-indexind]_`
`_> colnames(sumtabm) = c(“Price”, “Price Merge”_,`
`_+                                “Collusive”, “Collusive Merge”)_`
[Table 7.3](./16-chapter7.md#tbl7_3) presents the impact of the merger on prices under the two different models. Note that while BP's price goes down with the merger, the average price of BP and Peak increases. Remember the new firm is a weighted average of BP and Peak's observed and unobserved characteristics.[4](./16-chapter7.md#fn7_4) The impact of the merger is quite different under the two models of pricing behavior. The average price effect of the merger is small. It is 0.4% increase in prices in the static Bertrand case and 1.2% increase in the collusive case. The difference between the price increase assuming static Bertrand and collusion is over 100%. The assumed method by which firm's determine prices may have a large effect on our prediction of the merger.

__[Table 7.3](chapter7) The table presents impact of the merger under the two pricing models. The first and third column are the pre-merger prices for the two models. The second and fourth columns are the new prices after the simulated merger between BP and Peak. Note that the new prices are given for BP. The impact of the merger is quite different under the two pricing models. For example, Shell's prices don't really change with the merger in the Bertrand static pricing game, while they increase substantially in the repeated game.__
| Price         | Price Merge | Collusive | Collusive Merge |        |
| ------------- | ----------- | --------- | --------------- | ------ |
| Ampol         | 153.70      | 153.81    | 158.29          | 159.67 |
| BP            | 148.53      | 145.77    | 153.12          | 151.62 |
| Caltex        | 146.14      | 146.84    | 150.72          | 152.70 |
| Coles Express | 144.65      | 144.85    | 149.24          | 150.70 |
| Eagle         | 152.32      | 152.33    | 156.91          | 158.18 |
| Gull          | 144.00      | 144.27    | 148.59          | 150.13 |
| Liberty       | 145.23      | 145.31    | 149.82          | 151.17 |
| Peak          | 139.98      | 144.57    |                 |        |
| Shell         | 152.11      | 152.41    | 156.70          | 158.26 |
| United        | 138.25      | 138.28    | 142.84          | 144.14 |

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [4](./16-chapter7.md#fn7_47b)What may be other assumptions that could be made about the new post-merger firm?

## 7.7 Discussion and Further Reading

In the 1990s, industrial organization economists made great gains in understanding markets and competition. But there was something missing. The standard models didn't always capture the pricing behavior. A number of recent papers have shown that we need better models in order to capture the possibility of collusion in markets. Repeated games allow us to use much richer models of pricing behavior and show how collusion can be a more natural feature of how firms determine prices.

The paper by David Byrne and Nic de Roos does not provide us with any model techniques. Rather, the paper takes a very close look at an actual market and shows how firms in that market actually behave ([Byrne and de Roos, 2019](./25-refbib.md#ref18)). Nathan Miller and Matt Weinberg review the joint venture between Miller and Coors. The authors use information on the premerger market to estimate parameters of the model. They show that the observed price increase post merger cannot be explained by our standard static Nash model ([Miller and Weinberg, 2017](./25-refbib.md#ref46)). Nathan Miller, Gloria Sheu, and Matt Weinberg argue that a model that explicitly accounts for collusion is necessary to predict the effect of mergers in some industries ([Miller et al., 2021](./25-refbib.md#ref47)).

A number of papers formally compare the predictions of static pricing models to observed pricing behavior. [Miller and Weinberg (2017)](./25-refbib.md#ref46) use pre-merger pricing to estimate the parameters, then compare simulated pricing to actual pricing after the merger. [Nevo (2001)](./25-refbib.md#ref48) and [Backus et al (2021)](./25-refbib.md#ref11) compare the predictions of collusive pricing models to actual pricing in the ready-to-eat cereal market.

The pricing patterns seen in retail gasoline is very strange ([Lewis, 2012](./25-refbib.md#ref42)). In the Perth data, we see the firms move to a very ordered pattern of pricing on a weekly basis. If you zoom in even closer you see that the brands are using one or two stations to signal which price they will move to for that week. Our analysis in this chapter is based on the work of DOJ Economist, Zhongmin Wang. [Wang (2009)](./25-refbib.md#ref59) uses a mixed strategy dynamic model of to analyze pricing patterns in the Perth data. Nobel prizing winning economists, Eric Maskin and Jean Tirole, show that short term commitment to a price is necessary to get the type of pricing dynamics we see in the data ([Maskin and Tirole, 1988](./25-refbib.md#ref45)). The West Australian state government introduced price regulation that allowed for this type of equilibrium. We generally call it “post and hold” regulation. The post means that the price is publicly displayed and the hold means that the price cannot be changed for some period of time, say 24 hours. By making the price publicly available, it allowed for strategies that are a function of each other's prices. Hold means that the firms can commit to the price which means that the equilibrium cannot devolve into a competitive pricing process. That is, each firm will choose a price to under cut its competitor, like in the original Bertrand game presented in [Chapter 3](./11-chapter3.md).

[_OceanofPDF.com_](./https___oceanofpdf.com)
