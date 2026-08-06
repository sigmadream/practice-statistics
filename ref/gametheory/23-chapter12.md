# 12Moral Hazard

DOI: [10.1201/b23262-12](./https___doi.org_10.1201_b23262-12.md)

## 12.1 Introduction

The book has four parts covering complete information static games, complete information dynamic games, incomplete information static games, and incomplete information dynamic games. This part is the last.

This part considers two types of dynamic games of incomplete information. The first are games in which one player does not observe the other player's action before making their choice. What makes these games dynamic is that the first player's action is associated with a signal that is observed by the second player before making his choice. The second set of games are ones where the action of the first player is observed by the second player, but the second player does not know the first player's payoffs. The second set of games are discussed in the next chapter.

The first set of games are often given the moniker, moral hazard or principal-agent games. Consider a game between an employer and employee, where the principal is the employer and the agent is the employee. The employee takes actions like working hard and thinking carefully, but the employer doesn't get to observe this. The employer may observe signals of these things like the number of hours worked or profits of the firm. In these games, we are generally interested how the employer can get the employee to take actions that the employer prefers even when those actions are not directly observed.

This chapter presents the principal-agent game and applies it to the problem faced by financiers of whaling expeditions in New England in the 1830s. A new, at the time, form of company organization, the corporation, allowed many more people to invest in whaling. Ventures funded through corporations tended to give managers less of a stake in the outcome of the venture and tended to have worse outcomes.

## 12.2 Principal-Agent Game

In the 1830s, whaling was an important industry in the United States, particularly in communities in New England. The industry developed to hunt and kill sperm whales for oil and other products. The ships sailed literally around the world and the voyages would last years. To hunt and kill whales, you need a ship with the necessary equipment, supplies, a captain, and a crew of around 30\. Obviously you need money for all of this. The investment will payoff when you sell the oil and other products rendered from the whales.

How do investors insure that they get a good return on their investment? How do they know that they have a good captain and crew? How do they know that the captain and crew are doing a good job when they are literally on the other side of the world and it is the 1830s?

These operations generally relied on a number of incentive mechanisms. The captain and crew were paid a small share of the profits. The investors hired an agent who was responsible for hiring the captain and crew, determining the voyage's route and communicating with the captain during the voyage. Under the law, the owner of the vessel controlled everything on the vessel including any equipment used. Because of this, investors generally bought an ownership share of the vessel itself.

Two organizational structures were used to finance these hunts. In one case a small number of investors hired an agent. The agent was generally given a large share of the ownership of the vessel and the investors may include family members or members of the local community. An alternative method for raising funds from investors was to create a corporation. These organizations provided some legal recourse for investors and made it clearer who owned what. Corporate whaling enterprises had a much larger set of owners investing smaller amounts of money. Like unincorporated ventures, they still hired an agent who was responsible for overseeing the operation. One big difference between the two organizations seems to be share of ownership given to the agent. For the corporations the evidence suggests that the agent earned a small fraction of the profits.

This section works through a formal game of the investor–agent relationship in 1830s whaling. It then goes through an example with simulated data.

### 12.2.1 Simple No Contract Game

Consider a simplified version of the game faced by whaling ventures. In this game, the agent chooses her effort level and the investor pays her based on what the outcome they observe.

* Players: Investor, Agent
* Strategies:  
   1. Agent: e∈{eL,eH}  
   2. Nature: y∈{yL,yH}, where yH\>yL, as a function of _e_, Pr(yH|eH)\=pH\>Pr(yH|eL)\=pL  
   3. Investor: w(y)∈{wL,wH}
* Payoffs:  
   1. Agent: w(y)−c(e), where c(eH)\=c and c(eL)\=0  
   2. Investor: y−w(y)
* Beliefs: p\=Pr(e\=eH)

In this game, the agent moves first and chooses how much “effort” to put into the venture. If they choose the high effort level _eH_, then it costs them _c_, while the low effort level cost them nothing. Nature observes the effort level and chooses the outcome of the venture. The higher effort level increases the probability of the better outcome _yH_. Lastly, the Investor observes the choice of Nature and pays the Agent _w_. The more she pays the agent, the worse off the investor is.

What is the Bayes Nash equilibrium of the game? Is there an equilibrium where the Agent chooses _eH_? What is the subgame Perfect Nash equilibrium?

The last is a trick question. There is only one subgame, the whole game.[1](./23-chapter12.md#fn12_1)

The Agent's strategy is to choose an effort level _eL_ or _eH_. The Investor's strategy is to choose a payment level given the observed outcome _y_, w(y) and the Investor's beliefs about the choice of the Agent.

### 12.2.2 Bayes Nash Equilibrium

One proposed equilibrium is for the Agent to choose the high effort level _eH_ and for the Investor choose a payment that pays more if the outcome is _yH_ than if the outcome is _yL_. Such payments could make it worthwhile for the Agent to choose the higher cost effort level because that effort level increases the probability of getting a higher payment.

This is not an equilibrium. It is not optimal for the Investor to pay the Agent anything. What ever effort level the Agent chooses, the Investor prefer not to pay the Agent anything.

Assume the equilibrium is for the Agent to choose _eH_ and the Investor pays the Agent w(yH)\=wH and w(yL)\=wL where the following inequality holds.

pHwH+(1−pH)wL−c≥pLwH+(1−pL)wL(12.1)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./23-chapter12.md#fn12_112b)A **perfect Bayesian Nash equilibrium** has a similar flavor to a subgame perfect Nash equilibrium. In a perfect Bayesian Nash equilibrium, the players' strategies must be optimal given their beliefs and the beliefs must be consistent with Bayes' rule at each information set along each equilibrium path that can be reached with positive probability.

The expected payoff from choosing the high effort level is higher than the expected payoff from choose the low effort level. We call this the incentive compatibility (IC) constraint.

If in the proposed equilibrium the payment w(y) is such that the IC holds, then the Agent will choose the high effort level. What if Agent chooses _eH_. What contract should the Investor offer? Let _w_\* denote the Investor's payment if they observe _yH_. In equilibrium, the Investor's expected profits are as follows.

pH(yH−w∗)+(1−pH)(yL−wL)(12.2)

The Investor should offer w∗\=0.

In the proposed equilibrium, the Agent chooses _eH_ and the Investor must believe that is true. Given all of that, why should the Investor pay the Agent anything?

### 12.2.3 Simple Contract Game

The problem in the previous game is that the Investor cannot commit to paying the Agent for their high effort level. The result is that the Agent is not willing to put in the effort. We need a contract. In general, we assume that such a contract can be enforced, for example in a court of law. Alternatively, it could be enforced in the court of public opinion or the court of the back alleys of the New England port town. If the Investor doesn't pay the Agent what is stated in the contract then the Investor gets punished. Making the contract enforceable limits what is contractable. We cannot contract on the effort level of the Agent because that is not observed by anyone but the Agent, certainly not in a court of law. Potentially, we can contract on the outcome (_y_) as that is observed. We will assume that the _y_ is both observable and verifiable by a court of law.

* Players: Investor, Agent
* Strategies:  
   1. Investor: Offer contract w(y)  
   2. Agent: Accept or Reject given offer w(y)  
   3. Agent: Given Accept of w(y) choose e∈{eL,eH}  
   4. Nature: Choose y∈{yL,yH} given _e_, Pr(yH|eH)\=pH\>Pr(yH|eL)\=pL
* Payoffs: If the contract is accepted:  
   1. Investor: E(y(e)−w(y(e)))  
   2. Agent E(w(y(e)))−c(e)  
Assume that both get 0 if rejected.
* Beliefs: p\=Pr(e\=eH)

In the Bayes Nash equilibrium, the Investor offers a contract such that the following equality holds.

pHwH+(1−pH)wH−c\=0(12.3)

The Agent is indifferent between their expected wage if they choose the high effort level and their outside option (which is assumed to be 0). We call this the individual rationality (IR) constraint. In this set up, _wL_ will be negative, but you should just think of this as a value lower than Agent's outside option or alternative if they reject the contract.

Also, the Agent prefers the high effort level to the low effort level. The IC is as follows.

pHwH+(1−pH)wL−c≥pLwH+(1−pL)wL(12.4)

The difference between pay given the good outcome _wH_ and pay given the bad outcome _wL_ is large enough to induce the Agent to choose the high effort level despite the higher cost _c_.

The Investor then chooses _wH_ and _wL_ such that both the IR and IC constraints hold. Moreover, for it to be an equilibrium, it must be that pH(yH−wH)+(1−pH)(yL−wL)≥0. We know from the IR constraint that it must be that the following in equality holds for the contract to be profitable.

pHyH+(1−pH)yL−c≥0(12.5)

So the contract is profitable if the expected output from the project is greater than the cost to the Agent of completing the project.

### 12.2.4 More Complicated Contract Game

Let's model the principal-agent problem associated with investors contracting with an agent to run the whaling venture.

* Players: Investor, Agent
* Strategies:  
   1. Investor: Offer w(y)  
   2. Agent: Accept or Reject offer  
   3. Agent: Choose e≥0.  
   4. Nature: Choose _y_ given _e_, y∼F(e)
* Payoffs  
   1. Agent: E(u(w(y))|e)−c(e)  
   2. Investor: E(y(e)−w(y(e)))
* Beliefs: e∼P.

The Agent chooses an effort level, _e_, which incurs some cost c(e). The contract is based on the outcome, _y_, where this is a random number drawn from a distribution that is determined by Agent's effort choice F(e). Based on this outcome, the Agent is paid w(y). One difference between this model and what we have seen previously is that the Agent's payoff is determined by their utility function u(y). Here we are going to assume that the agent is risk-averse while the principal is risk-neutral. The principal would like to incentivize the agent by giving them a large share of the outcome. This implies that the agent is taking on a lot of risk. The principle will need to compensate the agent by paying them a very large amount that does not vary with the outcome of the venture. We will talk more about this in a bit.

### 12.2.5 Bayes Nash Equilibrium

One of the issues highlighted in this book is the difference between assumptions about the game and assumptions about equilibrium. This issue comes up again here. Look at the payoffs for the Investor. Her expected payoff is not conditional on the effort level of the Agent. In a Bayes Nash equilibrium, we assume that the Investor's beliefs are consistent with equilibrium behavior. In equilibrium, the Agent will choose a particular effort level _e_\*, so _in equilibrium_ the Investor's expected payoff will be E(y(e)−w(y(e))|e\=e∗).

The Agent will choose this effort level to optimize his expected payoff.

maxeE(u(w(y(e))))−c(e)(12.6)

The Agent knows _e_ and so the cost function is not inside the expectation.

The Investor will choose w(y) so as to maximize her profits.

maxw(y)E(y(e)−w(y(e))|e\=e∗)s.t.E(u(w(e∗)))−c(e∗)≥0e∗\=argmaxeE(u(w(y(e)))|e)−c(e)(12.7)

In words, the Investor will choose a payment as a function of the outcome w(y), that maximizes her expected return subject to the Agent choosing to accept the contract and choosing their optimal effort level conditional on the payment offer.

### 12.2.6 Parameterized Model

Assume that the Agent's expected utility is a mean-variance utility. That is, the Agent's utility is increasing in the mean of his payoffs and decreasing in the variance of his payoffs. The agent's dislike of risk is represented by the parameter _r_.

E(u(x))\=μx−rσx22(12.8)

where x∼N(μx,σx2). This utility function is a substantial simplification. It is often justified using a particular utility function and assuming the outcome is normally distributed.

The Agent doesn't like producing effort and the cost of the effort is governed by the parameter _k_ and the cost is increasing in effort at a geometric rate, c(e)\=−ke22. For a particular incentive rate, _b_, the Agent's expected utility is as follows.

b(μx+e)−rb2σx22−ke22(12.9)

The variance of _bx_ is b2σx2. Also the dividing by 2 thing is useful once we get to the first-order condition.

b−ke\=0e\=bk(12.10)

The agent's optimal effort level is increasing in the power of the incentives and decreasing in the cost.

The production function is simply that y(e)∼N(μ+e,σ). The mean of output is increasing 1 to 1 with the effort level _e_.

The incentive contract used by the Investor is a linear function of output, w(y)\=a+by, where _a_ is a constant and _b_ is the fraction of the output received by the Agent. When you look below you see there is no _a_. This is because we have implicitly solve the individual rationality constraint by choosing _a_ such that the Agent is indifferent between accepting the contract and rejecting the contract. We can also substitute in the Agent's optimal effort level.

πI(b)\=μ+bk−rb2(σ2)2−b22k(12.11)

The first-order condition is then given by the following equation.

1k−rbσ2−bk\=01−krbσ2−b\=0b\=11+krσ2(12.12)

The power of the incentive contract is decreasing in the effort cost of the Agent, the risk aversion of the Agent and the variance in the output.

### 12.2.7 Simulation with **R**

In the code, the function in Equation (12.11) is as follows. The parameters mu and sigma are global variables determined below.


`_> PiI = function(b, r, k) {_`
`_+   mu + b/k - r*(b^2)*(sigma^2)/2 - (b^2)/(2*k)_`
`_+ }_`
The parameters of the simulation have been calibrated such that the incentive contract is similar to the average for unincorporated ventures in the data.


`_>   mu = 90000_`
`_>   sigma = 40000_`
`_>   k = 0.000005_`
`_>   r = 0.0002_`
Given the set up we can solve for the optimal share of the output given to the agent and the optimal effort level.


`_> b1 = optimize(PiI, c(0, 1), maximum = TRUE, r=r, k=k)_`
`_> b1$maximum_`
`  [1] 0.3846154`
`_> b1$maximum/k_`
`  [1] 76923.08`
`_> PiI(b1$maximum, r, k)_`
`  [1] 128461.5`
The firm makes $128,000 from the venture with the agent taking 39%. The agent's effort cost is $77,000.

Now consider what happens if the power of the incentive contract is significantly reduced. Let b\=0.05, rather than the optimal level.


`_> 0.05/b1$maximum_`
`  [1] 0.13`
`_> PiI(0.05, r, k)/PiI(b1$maximum, r, k)_`
`  [1] 0.7733832`
The new level of optimal effort for the Agent is 13% of the optimal contract and the Investor's profits also falls but to just 77% of what they would be with the optimal contract.

## 12.3 Empirical Analysis: Whaling Corporations in the 19th Century using **R**

Wellsley College professor, Eric Hilt, documents the surprising failure of the corporate structure in whaling. In the first half of the 19th century, corporations were a relatively new type of institution in the United States. They provided a legal structure for people to raise money from investors where it was clear what rights investors did and did not have. Whaling in New England was generally financed by small groups of investors, many of whom knew each other or were from the same family. Corporations opened up whaling to a much broader range of investors. Given these advantages it is surprising that the corporate structure performed so poorly.

This section using data from whaling venture financial records to understand the contracts with agents used by whaling corporations. Were the contracts responsible for the poor performance of corporations?

### 12.3.1 Whaling Data

Bringing in the data called whaling.csv.

The code below reads in the data and plots average output by year and by how the venture was financed.


`_>   file = paste0(dir, “whaling.csv”)_`
`_>   dt = fread(file)_`
`_>   dt1 = dt[, .(lprodwb = mean(lprodwb_,`
`_+                                  na.rm = TRUE))_,`
`_+              by = .(ayear, corp)]_`
`_>   ggplotlogoutput = setDF(dt1) |>_`
`_+     mutate(_`
`_+       corp = as.factor(corp)_`
`_+     ) |>_`
`_+     ggplot(aes(ayear, lprodwb, corp_,`
`_+                     linetype = corp)) +_`
`_+       geomline() +_`
`_+       labs(_`
`_+         x = “Year”_,`
`_+         y = “”_,`
`_+         title = “Average log output by year”_`
`_+       ) +_`
`_+       theme(axis.text.y=elementblank()_,`
`_+             axis.ticks.y=elementblank())_`
` `
`_> ggplotlogoutput_`
[Figure 12.1](./23-chapter12.md#fig12_1) presents a plot of the productivity of the whaling ventures. It is the ratio of the output of the venture in terms of the value of the oil and other products rendered from the whale to the size of the ship multiplied by the length of the journey.

![In the graph, the horizontal axis is labeled as year and ranges from 30 to 52. The vertical axis is labeled as average log output by year. Two lines are shown. The solid line represents corp 0 and stays higher across the range. The dashed line represents corp 1 and remains lower. Both lines fluctuate year to year, with peaks and dips. The solid line declines early, stabilizes, and then dips again. The dashed line follows a similar but more varied pattern. All data are approximate.](./images/fig12_1.jpg)

[Figure 12.1](chapter12) Line chart of average log of output by year and corporation. It shows output decreasing from 1835 to 1850, but corporate (corp = 1) output is generally lower.

The figure shows two things. First, productivity is falling dramatically over time. It is becoming harder and harder to find whales to kill. Second, the corporate ventures tend be less productive than closely held ventures.

Why are corporations doing so poorly?

### 12.3.2 Regressions

Does the pattern from [Figure 12.1](./23-chapter12.md#fig12_1) hold when we are more careful about accounting for various factors determining the outcome. One of the big variables is the agent themselves. The data include cases where the same agent is used by a closely held firm and by a corporation. Using fixed effects, we can account for differences across agents. We can hold the “agent-effect” fixed.

[Table 12.1](./23-chapter12.md#tbl12_1) presents the results from the **fixed effects analysis**. The empirical model is generally called two-way fixed effects. We have fixed effects for the agent (the first way) and fixed effects for the year (the second way) which are not presented in the table. The idea is that we can account for the agent and the year to isolate the effect of the financial structure. It shows that corporations have lower productivity even accounting for individual agent effects. Specifications (3) and (4) suggest that it may not be the corporate entity itself but due to the larger number of owners associated with the corporate structure. It also shows that having the captain die, is not good for the success of the hunt.

__[Table 12.1](chapter12) OLS regressions of output on corporate form with agent and year fixed effects. Ownership structure is accounted for either through the dummy variable for corporation or by the number of investors.__
| _Dependent variable:_ |                                                   |                                                   |                                                   |       |
| --------------------- | ------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------- | ----- |
| lprod\_wb             |                                                   |                                                   |                                                   |       |
| (1)                   | (2)                                               | (3)                                               | (4)                                               |       |
| corp                  | \-0.455[\*\*](./23-chapter12.md#tblfn12_1_01)   | \-0.411[\*\*](./23-chapter12.md#tblfn12_1_01)   | \-0.279                                           |       |
| |  (0.178)            | (0.179)                                           | (0.442)                                           |                                                   |       |
| I(owners/10)          | \-0.126[\*](./23-chapter12.md#tblfn12_1_01)     | \-0.126[\*](./23-chapter12.md#tblfn12_1_01)     |                                                   |       |
| |  (0.065)            | (0.065)                                           |                                                   |                                                   |       |
| I((owners/10)^ 2)     | 0.014                                             | 0.014                                             |                                                   |       |
| |  (0.012)            | (0.012)                                           |                                                   |                                                   |       |
| atlantic              | 0.048                                             | 0.074                                             | 0.074                                             |       |
| |  (0.053)            | (0.060)                                           | (0.060)                                           |                                                   |       |
| pacific               | \-0.107[\*\*](./23-chapter12.md#tblfn12_1_01)   | \-0.102[\*\*](./23-chapter12.md#tblfn12_1_01)   | \-0.102[\*\*](./23-chapter12.md#tblfn12_1_01)   |       |
| |  (0.044)            | (0.049)                                           | (0.049)                                           |                                                   |       |
| tons                  | \-0.001[\*\*\*](./23-chapter12.md#tblfn12_1_01) | \-0.001[\*\*](./23-chapter12.md#tblfn12_1_01)   | \-0.001[\*\*](./23-chapter12.md#tblfn12_1_01)   |       |
| |  (0.0003)           | (0.0003)                                          | (0.0003)                                          |                                                   |       |
| vesselage             | \-0.003                                           | \-0.004                                           | \-0.004                                           |       |
| |  (0.002)            | (0.002)                                           | (0.002)                                           |                                                   |       |
| capexp                | 0.006                                             | 0.008                                             | 0.008                                             |       |
| |  (0.008)            | (0.010)                                           | (0.010)                                           |                                                   |       |
| capdied               | \-0.338[\*\*\*](./23-chapter12.md#tblfn12_1_01) | \-0.407[\*\*\*](./23-chapter12.md#tblfn12_1_01) | \-0.407[\*\*\*](./23-chapter12.md#tblfn12_1_01) |       |
| |  (0.100)            | (0.121)                                           | (0.121)                                           |                                                   |       |
| Observations          | 831                                               | 809                                               | 671                                               | 671   |
| R2                    | 0.300                                             | 0.339                                             | 0.343                                             | 0.343 |

_Note:_ [\*](./23-chapter12.md#tblfn12_1_1a)p<0.1; [\*\*](./23-chapter12.md#tblfn12_1_1b)p<0.05; [\*\*\*](./23-chapter12.md#tblfn12_1_1c)p<0.01

Why may this be happening? Corporations provided agents with a share of the output, but shares were substantially lower than what we see for the closely held ventures. What are the implications of this difference in compensation to the agent's effort level and profitability of the venture?

### 12.3.3 Calibrating the Model

In order to get some sense of what happens when incentives of the agent are changed, we can use the observed data to calibrate the game presented above.

In order to get apples to apples, we can use the same trick that we used in [Chapter 10](./20-chapter10.md). Regress the output measure on various observed characteristics and then use the residual as the normalized output.


`_>   df1 = setDF(dt) |>_`
`_+     filter(_`
`_+       corp == 0_`
`_+     ) |>_`
`_+     na.omit()_`
`_>   lm5 = lm(outputwb ~ atlantic + pacific + tons +_`
`_+              vesselage + dyear, data = df1)_`
`_>   df1$res = lm5$residuals + lm5$coefficients[1]_`
`_>   mu = mean(df1$res, na.rm = TRUE)_`
`_>   sigma = sd(df1$res, na.rm = TRUE)_`
Given the normalized output, we can use the observed contracts from the closely held firms to back out the parameters on agent's costs (_k_) and risk-preferences (_r_).


`_>   e1 = mean(df1$res, na.rm = TRUE)_`
`_>   sigma1 = sd(df1$res, na.rm=TRUE)_`
`_>   b1 = mean(df1$agtshr, na.rm = TRUE)_`
`_>   k1 = b1/e1_`
`_>   r1 = e1*(1 - b1)/((b1^2)*sigma1^2)_`
` `
`_> PiI(b1, r1, k1)_`
`  [1] 269567.9`
`_> e1*(1 - b1)_`
`  [1] 73007.96`
The closely held firm makes $270,000, and the agent's effort costs are $73,000\. Looking at the corporations, we can again normalize output and estimate the effort level of the agent given the share received by the agent.


`_>   df2 = setDF(dt) |>_`
`_+     filter(_`
`_+       corp == 1_`
`_+     ) |>_`
`_+     select(_`
`_+       outputwb_,`
`_+       atlantic_,`
`_+       pacific_,`
`_+       tons_,`
`_+       vesselage_,`
`_+       dyear_,`
`_+       agtshr_`
`_+     ) |>_`
`_+     na.omit()_`
`_>   df2$res = df2$outputwb -_`
`_+     predict.lm(lm5, df2) + lm5$coefficients[1]_`
`_>   e2 = mean(df2$res, na.rm = TRUE)_`
`_>   b2 = mean(df2$agtshr, na.rm = TRUE)_`
`_>   e2pred = b2/k1_`
` `
`_> PiI(b2, r1, k1)_`
`  [1] 183437.4`
`_> e2pred*(1 - b2)_`
`  [1] 3718.121`
In the case of the profit sharing under the corporate form, the venture's profits fall to $183,000 and the agent's effort would be only $3,700.


`_> e2pred/e1_`
`  [1] 0.02094994`
`_> PiI(b2, r1, k1)/PiI(b1, r1, k1)_`
`  [1] 0.680487`
Given the share offered to agents working for corporations, the predicted effort level falls to just 2% of the unincorporated effort level. That said, the predicted profits only fall to 68% of the unincorporated profits. The reason is that while the output falls, the share of output received by the investors is higher. In addition, the investors don't have to compensate the agent for risk.


`_> e2/e1_`
`  [1] 0.6883835`
While our model predicts that the Agent's effort level will drop precipitously. The effort level does fall by a large amount to 69% of the unincorporated level. The predicted output is much lower than what we actually observe in the corporate ventures. This discrepancy suggests that corporations are providing incentives. They are just not providing incentives in the form of share of output. They must be using other mechanisms like direct supervision and the threat of firing, rather than giving high powered incentives. Those incentives may leading to significantly lower effort levels by the Agent, but the returns to the Investors are not necessarily that bad.

## 12.4 Discussion and Further Reading

Wellsley College professor, Eric Hilt, has created an amazing data set on how whaling firms actually worked. It is very cool to bring modern economic theory to bear on why closely held firms worked so well in the 1830s ([Hilt, 2006](./25-refbib.md#ref34)).

Theoretical work on incentives and the principal-agent problem substantially improved our understanding of contracts and performance pay systems ([Lazear and Oyer, 2013](./25-refbib.md#ref41)). The classic paper on the principal-agent problem is [Holmström (1979)](./25-refbib.md#ref35).

[_OceanofPDF.com_](./https___oceanofpdf.com)
