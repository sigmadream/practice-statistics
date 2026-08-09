# 3Oligopoly

DOI: [10.1201/b23262-3](./https___doi.org_10.1201_b23262-3.md)

## 3.1 Introduction

When most people, even most economists, discuss competition they have in mind the model presented in Econ 101\. There are many firms all adding a small amount to the market. No firm has control over the price that they receive. Each firm can leave or enter the market at will, so any large profits get bid away as more firms enter. In this world, prices are equal to marginal cost and the amount of goods is such that consumers and firms cannot be made better off (without one or the other being made worse off). This is not what industrial organization economists mean when they discuss competition. Economists that work in antitrust and competition policy, distinguish between firms who work together to determine price and firms that work independently of each other.

This chapter presents the standard model of how firms set prices and output when they are doing so independently of each other. [Chapter 8](./17-chapter8.md) will return to oligopoly and allow less independence in price setting. The chapter presents three standard models of competition, Cournot, Bertrand, and Hotelling.[1](./11-chapter3.md#fn3_1) The chapter uses **R** to simulate the Nash equilibrium of a Cournot game with three firms.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./11-chapter3.md#fn3_13b)Confusingly the Hotelling model is often called a Bertrand model. While this chapter presents two examples of a price setting game, many people assign the Bertrand name to any static price setting game where the products are not homogeneous.

The Hotelling model is used to understand pricing of hamburgers in Santa Clara County California in the late 1990s. In particular, the chapter analyzes data on the price of the McDonald's Big Mac and the Burger King Whopper at the outlets throughout Santa Clara County. The model allows us to estimate how prices are affected by competition between firms that are differentiated by location. If one franchisee purchased all the McDonald's outlets in Santa Clara County, what would happen to the price of both the Big Mac and the Whopper in the county?

## 3.2 Cournot's Model

In the 1850s a French applied mathematician made a far fetched claim. Augustine Cournot suggested having competing firms does not necessarily mean that prices will be equal to marginal cost. Tabarnak! How could this be? It is almost by definition that prices equal marginal cost in economics. Some fifty years later, another Frenchman, François Bertrand, argued that Cournot was full of merde. Even with just two firms, prices would equal marginal cost.

The section presents a two-firm version of the Cournot model and then generalizes that to a _N_\-firm version. It uses **R** to numerically simulate the Nash equilibrium in a three-firm version of the model.

### 3.2.1 Two Firm Model

Formally we have the following game where we have two firms choosing quantity (_qi_) and contemplating the impact of their collective choices on profits.

* Players: Firm 1 and Firm 2.
* Strategies:  
   1. Firm 1: q1≥0  
   2. Firm 2: q2≥0
* Payoffs:  
   1. Firm 1: p(q1,q2)×q1−c(q1)  
   2. Firm 2: p(q1,q2)×q2−c(q2)

The first part of Firm 1's payoff is revenue. The price p(q1,q2) is a function of each firm's output in the market. The price is multiplied by the output produced by Firm 1 which is _q_1. The second part is costs which is a function of the amount of output produced. To keep things simple let p(q1,q2)\=a−b×(q1+q2) and cj(qj)\=c×qj. That is, we have linear demand and constant marginal cost. Assume also that a−c\>0. This will become important later.

Cournot's game assumes that both firms make identical goods. This may be a good model of a wheat market or an electricity market. In such markets each firm decides how much to produce and supplies to a centralized exchange that takes in demand and determines the price everyone gets. In the game, each firm does not know how much the other firm produces.

### 3.2.2 Best Response

One issue that people new to game theory find very confusing is the idea of a **best response function**. In the presentation of the game above, it is clearly stated that each firm does not know the other firm's choice when choosing its action. Then five minutes later we claim that firms have a function such that its action is a best response to the other firm's action. How can they both not know what the other firm is doing and have a best response to it? Makes no sense.

Both statements can be true because they refer to different concepts. One is a description of the game and the other is a description of the algorithm used to find the Nash equilibrium of the game. The games discussed in this part of the book assume the players choose actions once and simultaneously. In this sense, each firm does not know how much the other firm is producing. The best response function is an analytical tool used to find the Nash equilibrium. The Nash equilibrium is where each firm chooses the optimal output given the output chosen by the other firm. That is the Nash equilibrium is where each firm's output choice is a best response to the other firm's output choice.

The best response function is the solution to the following optimization problem.

maxq1(a−bq1−bq2)q1−cq1(3.1)

The solution to this optimization problem is given by the first order condition.

a−bq1−bq2−c−bq1\=0orq1\=a−c−bq22b(3.2)

The best response function states that Firm 1's quantity is increasing in the difference a−c, decreasing in the quantity of Firm 2 (_q_2) and decreasing in the willingness of customers to substitute out of the market (_b_). In this model, the actions of the two players are strategic substitutes, in words, when one firm increases its output, the other firm responds by decreasing its output.

### 3.2.3 Nash Equilibrium

The Nash equilibrium is where each firm is playing the best response to the other firm. That is, where the first order condition for Firm 1 (Equation (3.2)) and the equivalent condition for Firm 2 both hold. To solve for the equilibrium we need to solve for two unknowns from a system of two linear equations. Simply counting the number of unknowns and the number of independent linear equations we know that the solution exists and is unique.

In this case, we can use a nice trick to solve it. Because both firms are identical then it must be that in equilibrium they produce the exact same amount. That is q1\=q2\=q. Substituting this into Equation (3.2) allows us to find the solution.

q\=a−c−bq2b2bq\=a−c−bq3bq\=a−cq\=a−c3b(3.3)

This is each firm's output in equilibrium. Equation (3.3) states that output will fall when marginal cost (_c_) increases and when substitutability (_b_) increases. In order for equilibrium output to be positive it must be the case that a−c\>0, which is an assumption made above.

If we substitute this back into demand we can determine price. We have two firms so we need to substitute back both equilibrium quantities.

p\=a−2ba−c3b\=3a−2a+2c3\=a+2c3(3.4)

Equation (3.4) states that price will increase with marginal cost, but are they higher than marginal cost?

p\>c⇔a+2c3\>c⇔a+2c\>3c⇔a−c\>0(3.5)

As long as there is positive output in the market, prices will be greater than marginal cost. Mayhem. Cats and dogs living together!

### 3.2.4 Cournot Model with N Firms

What happens as the number of firms increase? Do we get back to perfect competition?

In this case, Firm 1's best response becomes.

q1\=a−c−b∑j\=2Nqj2b(3.6)

Again to solve for equilibrium with symmetric firms we can do the trick of setting all the output levels to be the same.

q\=a−c−b(N−1)q2b2bq\=a−c−b(N−1)q(N+1)bq\=a−cq\=a−c(N+1)b(3.7)

So in equilibrium, output is decreasing proportionately with the number of firms in the market.

Substituting equilibrium quantities back into demand we can determine prices. Remember we have to multiply by _N_.

p\=a−bNa−c(N+1)b\=(N+1)a−Na+NcN+1\=a+NcN+1\=aN+1+NN+1c(3.8)

Look familiar? It is the equation used to determine price in [Chapter 2](./10-chapter2.md).

We can see that as _N_ gets large, _p_ converges to _c_. That is Cournot's model does give perfect competition but only for a large number of firms in the market.

### 3.2.5 Cournot Model in **R**

In order to better understand how the model works it helps to program it up in **R**. The function price\_cournot() is used to determine the price given market output q. The function br\_cournot() determine the firm's output given the output of the other two firms. This function is based on Equation (3.6). The notation q\[-i\] is used to refer to all the elements of q except the ith element.


`_> pricecournot = function(q) a - b*sum(q)_`
` `
` `
`_> brcournot = function(q, i) {_`
`_+   if (pricecournot(q) > 0) {_`
`_+     return(max(c(0, (a - c[i] - b*sum(q[-i]))/(2*b))))_`
`_+   } else {_`
`_+     return(0)_`
`_+   }_`
`_+ }_`
The best response function checks to make sure prices and quantities are positive. It is based on Equation (3.6).

### 3.2.6 Solve for the Nash Equilibrium with **R**

The algorithm below looks for an equilibrium where the best response's of each firm lead to the same quantity. The algorithm chooses a starting level of output and then calculates the best response for each firm to get a new level of output. It then checks whether the new level of output is the same as the old level of output. If it is, then we have a Nash equilibrium. Remember a Nash equilibrium is where each firm is choosing a level of output that is a best response to all the other firms. If the new level of output is different from the old level of output, the algorithm sets the old level of output to the output just calculated and finds the best response to that amount. The algorithm stops when the new and old amounts become equal or close to equal.

In the code we have a default maximum number of iterations (maxit = 100) and a default level of convergence (epsilon = 1e-5).[2](./11-chapter3.md#fn3_2) The code uses epsilon to refer to a small number. It uses a while() loop to run until one of the conditions fails to hold.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./11-chapter3.md#fn3_23b)1e-5 is a way to write small numbers, in this case 0.00001.

The initial value for the output is determined by Equation (3.7) assuming all firms in the game have the same costs. The algorithm then calculates the best response for each firm and checks whether the new output is the same as the old output. The algorithm sets the old output to the new output and calculates the best response to that output. The algorithm continues until the difference between the new and old output is less than epsilon or the number of iterations exceeds maxit. The algorithm returns the output in equilibrium and whether the algorithm converged.


`_> necournot = function(maxit=100, epsilon=1e-5_,`
`_+                        trace=FALSE, converged=TRUE) {_`
`_+   diff = 10000 # some big number_`
`_+   iter = 1_`
`_+   N = length(c)_`
`_+   qold = rep((a - mean(c))/((N+1)*b), N)_`
`_+   # initial values for qold_`
`_+   while(diff > epsilon & iter < maxit) {_`
`_+     qnew = rep(0, N)_`
`_+     for(i in 1:N) {_`
`_+       qnew[i] = brcournot(qold, i)_`
`_+     }_`
`_+     diff = sum(abs(qnew - qold))_`
`_+     iter = iter + 1_`
`_+     if(trace) {_`
`_+       print(diff)_`
`_+       print(iter)_`
`_+     }_`
`_+     qold = qnew_`
`_+   }_`
`_+   if(iter == maxit) {_`
`_+     converged = FALSE_`
`_+   }_`
`_+   return(list(qstar = qnew, converged = converged))_`
`_+ }_`
This algorithm is not super sophisticated and it takes advantage of uniqueness of the result. The algorithm does allow us to simulate more interesting models than the simple symmetric-firm model presented above.

### 3.2.7 Simulation of Cournot Model in **R**

The following simulation allows the costs to vary between firms and shows how variation in marginal costs leads to differences in market share for firms.


`_>   set.seed(123456789)_`
`_>   N = 3_`
`_>   a = 0.5_`
`_>   b = 0.2_`
`_>   c = a*runif(N) # so costs vary between firms._`
`_>   qstar = necournot()$qstar_`
`_>   pstar = pricecournot(qstar)_`
The results are as follows. The marginal costs (c) are determined randomly. The output (q\_star) and the price (p\_star) are determined by the equilibrium algorithm.


`_> c_`
`  [1] 0.3465879 0.3364405 0.3269508`
`_> qstar_`
`  [1] 0.1545377 0.2052715 0.2527168`
`_> pstar_`
`  [1] 0.3774948`
In this example Firm 1 has the highest costs and the lowest quantity, while Firm 3 has the lowest cost and the highest quantity.

This simulation illustrates a standard result of the Cournot model, the less efficient (higher marginal cost) firms have lower market share and the more efficient (lower marginal cost) firms have higher market share.

It also makes it clear that less efficient firms could be in the market. There is nothing about the basic Cournot game that forces them to leave the market.

## 3.3 Bertrand's Model

Bertrand was having none of it. Perfect competition was not some edge case. In Cournot's game the firms choose quantity, but what happens if the firms choose price?

The section presents a two-firm version of Bertrand's game. Think about two ice-cream stands that are next to each other. Each stand displays the price of a cone of ice-cream cone. If one ice-cream stand charges more than the other, then everyone will buy from the cheaper stand. If both stands charge the same price, then they split the market.

### 3.3.1 Two-Firm Game

This time the two firms choose a price and payoffs are determined by the quantity which is a function of both prices.

* Players: Firm 1 and Firm 2.
* Strategies:  
   1. Firm 1: p1≥0  
   2. Firm 2: p2≥0
* Payoffs:  
   1. Firm 1: p1×q1(p1,p2)−c1×q1(p1,p2)  
   2. Firm 2: p2×q2(p2,p1)−c2×q2(p2,p1)

Again the payoffs for each firm are revenue less costs. This time, the firm chooses the price to charge and the market mechanism determines how much quantity the firm will sell.

q1\={1if q1<q20.5if q1\=q20if q1\>q2(3.9)

Let's assume that the market size is 1\. Equation (3.9) says that if you have the lowest price, then everyone buys from you and if you have the highest price then no one buys from you. If both firms have the same price, they split the market.

This is a market where it is very easy for customers to substitute. One slight reduction in price can cause the whole market to shift.

A real-world example might be a supply contract request for quotes. Which ever firm offers the lowest price gets the full supply contract.

### 3.3.2 Nash Equilibrium

Consider a simple case where both firms have the same marginal cost, c1\=c2\=c. The unique Nash equilibrium is p1\=p2\=c.

To confirm that it is an equilibrium, assume that p1\=c. What happens if p2\>c ? In this case, Firm 2's output is 0 and profits are 0\. What if p2\=c. In this case, Firm 2's output is 0.5, but Firm 2's profits are 0\. Lastly, if p2<c, then Firm 2's output is 1, but Firm 2's profits are negative. So Firm 2 is indifferent between choosing p2\=c or p2\>c. Given that Firm 2 cannot do better than the Nash equilibrium, this confirms p2\=c is optimal for Firm 2\. We can make the exact same argument for Firm 1.

It is not only an equilibrium but a unique equilibrium. A candidate equilibrium is, p1\=p2\>c. Again, keep Firm 1's price at p1\>c. If Firm 2 charges p2\>p1, then Firm 2's profits are 0\. If Firm 2 charges p2\=p1 then Firm 2's profits are positive, 0.5(p1−c). If Firm 2 charges p2\=p1−ϵ where _ϵ_ is some small number, then Firm 2's profits are (p1−c−ϵ). These profits are a lot higher. Sure price is slightly lower, but Firm 2 went from selling to half the market to selling to the whole market. In this game there is a huge incentive to slightly undercut your rival in this market. Because of this incentive, there is no other Nash equilibrium.

Bertrand proved his point. With just two firms, price equals marginal cost, an important characteristic of perfect competition.

What happens in Bertrand's model if the two firms have different marginal costs (_cj_)?

## 3.4 Hotelling's Model

To go from perfect competition being some edge case to it being a constant of the model all we needed to do was assume firms choose price instead of quantity! Nonsense. If you look more closely, the models proposed by the two Frenchmen are very different from each other. In particular, the demand in the Bertrand model is very particular.

Early in the Twentieth Century, the American statistician and economist, Harold Hotelling, suggested a compromise. He suggested a model where firms choose price but where demand was not nearly so particular as Bertrand assumes.

The section presents Hotelling's original model. It then presents a model of differentiated goods with linear demand and determines the Nash equilibrium in prices for that model.

### 3.4.1 Hotelling's Line

[Figure 3.1](./11-chapter3.md#fig3_1) represents Hotelling's game. There are two firms _L_ and _R_. Customers for the two firms “live” along the line. Consider two frozen custard (ice cream) places located at each end of a beach board walk. Your beach chairs may be located closer to one frozen custard place than the other. Customers prefer to go to the closer firm if the products and prices are otherwise the same. Hotelling's key insight is that while firms often compete by selling similar products, these products may not be identical. Moreover, some people may prefer one product to the other. Some people actually prefer Pepsi to Coke. The location on the line represents how much the customer prefers _L_ to _R_.

![A horizontal line is shown with three evenly spaced vertical tick marks. The left tick is labeled L, the middle tick is labeled x subscript L, and the right tick is labeled R.](./images/fig3_1.jpg)

[Figure 3.1](chapter3) Hotelling's line with Firm _L_ located at 0 and Firm _R_ located at 1\. Everyone “living” left of _xL_ purchases from Firm _L_.

The line and the distance between the customer and the firm represent how willing they are to purchase from a particular firm. Importantly, it represents how much the two firm's products are substitutes for each other. In the hamburger example presented below, we use the actual distance between stores to measure substitutability but more generally Hotelling's line is a metaphor for how similar or different two products are from each other. The closer the two firms are, the easier it is to substitute between them and the lower the price is likely to be.

### 3.4.2 Differentiated Goods Game

Now consider a game where there are _N_ firms that all sell a product that is similar but not the same as each other. For example, a set of hamburger restaurants in Santa Clara. While the Big Mac may be the same, the location of each McDonald's outlet is different.

* Players: _N_ firms where each firm has a location {xi,yi}∈ℜ2.
* Strategies: pi≥0 for all i∈{1,...,N}
* Payoffs: (pi−ci)×qi(pi,p−i)

Each of the _N_ firms chooses a price _pi_. The profits are the price less marginal cost (_ci_) multiplied by the quantity sold _qi_. This quantity is determined by both the price the firm chargers _pi_, by all the prices all the other firms charge p−i and by the distances between the firms.[3](./11-chapter3.md#fn3_3)

To make things simpler and more concrete, assume that there are just two firms _i_ and _j_, Firm _i_'s sales are affected by _pi_ and _pj_ in the following way.

qi(pi,pj)\=α+βpi+γpjdij+ϵi(3.10)

It is a linear demand model similar to the model presented earlier in the chapter. The quantity sold by Firm _i_ is a function of the price Firm _i_ charges, the price Firm _j_ charges, and some unobserved term _ϵi_. Assuming, β<0, then the higher the price the lower the demand for _i_'s product. Demand is also a function of the competitor's price _pj_. The extent of this is determined by _γ_, which is positive. The demand for _i_ is higher when the competitor charges a higher price. The extent of the competitor's price matters depends on the distance between the outlets. The larger that distance (dij), the less the two firms compete with each other for customers. Again, dij could represent physical distance or just a measure of the difference between the two products.[4](./11-chapter3.md#fn3_4)

### 3.4.3 Best Response

Given all this, what will be the price in the market? We assume that the price is determined by the Nash equilibrium. The Nash equilibrium is the price such that Firm _i_ is unwilling to change their price given the price charged by Firm _j_, and Firm _j_ is unwilling to change their price given Firm _i_'s price.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [3](./11-chapter3.md#fn3_33b)−i means not _i_.

[4](./11-chapter3.md#fn3_43b)We will Euclidean distance, dij\=((xi−xj)2+(yi−yj)2). This is the as-the-crow-flies distance.

Firm _i_'s problem is as follows.

maxpi(pi−ci)(α+βpi+γpjdij)(3.11)

The solution to the optimization problem is the solution to the first order condition.

(α+βpi+γpjdij)+β(pi−ci)\=0(3.12)

Firm _j_'s problem is similar. The two equations from the first order conditions are the best response functions for each firm.

### 3.4.4 Nash Equilibrium

Given the two equations derived from the first-order conditions, we can write down a system of equations. Each firm's best response function is as follows.

pi\=ci2−α2β−γpj2βdijpj\=cj2−α2β−γpi2βdij(3.13)

The prices are higher when costs are higher. Prices are strategic complements. While there is a negative sign in front of _γ_, we said above that _β_ is negative. When _pj_ increases, then Firm _i_'s best response is to _increase pi_. Again, how much the two firm's interact depends on _γ_ and the distance dij.

## 3.5 Empirical Analysis: Hamburger Competition with **R**

When he was doing his PhD at Stanford, Wash U Marketing professor, Raph Thomadsen, decided to study competition for hamburgers in Santa Clara County. Raph was interested in competition between the two big chains, McDonald's and Burger King as well as competition within the chain brands.

To study competition he got information on each outlet from the health department. Most importantly he got the outlet's location. He then physically visited each outlet determining the price of the hamburgers offered and whether the outlet had other features like a drive through or a playland.

This section models the prices for hamburgers in the late 90s in Santa Clara County using the model presented earlier.

### 3.5.1 Data

The data is provided by Raph Thomadsen and used in his RAND paper ([Thomadsen, 2005](./25-refbib.md#ref56)).[5](./11-chapter3.md#fn3_5) For each outlet there is information about the brand, ownership structure, features of the outlet, age of the outlet, and price of the sandwich.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [5](./11-chapter3.md#fn3_53b)There is a slight descrepancy between the coordinates in the code and the location of the outlets using Google maps.

[Figure 3.2](./11-chapter3.md#fig3_2) is created using the leaflet package in **R** with the coordinates from the code and importing the icons. The figure gives some idea how competition works in Santa Clara county. There is a logo at each brand's location. The size of the logo represents the price charged for the signature sandwich. The higher the price, the bigger the icon. The outlets up near the Stanford campus have fairly high prices. Often a Burger King is paired with a McDonald's, but neither has really low prices. Prices on the west and south side of San Jose seem lower than on the east side and north east side.

![Santa Clara County is shown with McDonald's and Burger King outlets marked using logos. McDonald's is marked with the M logo and Burger King with the crown logo. Outlets appear across cities such as San Jose, Mountain View, Palo Alto, and Cupertino. The size of each logo varies. Larger logos appear in areas like Cupertino and Mountain View. Smaller logos appear around San Jose. Some outlets are closely grouped while others are more spaced out. Roads, highways, and city names are labeled.](./images/fig3_2.jpg)

[Figure 3.2](chapter3) McDonald's and Burger King outlets in Santa Clara County. The size of the logo indicates whether the price of the sandwich is in the top third of prices, the middle third, or the bottom third.

You may be surprised by how much prices for a Big Mac vary across one city. You may also be surprised that the competition of the most interest is competition between McDonald's outlets. How can that be? Doesn't the McDonald's corporate headquarters set prices for the Big Mac? No. It depends. Some outlets in this data are owned by the McDonald's corporation and for those, yes, corporate headquarters would have a lot of say over price. But most of the outlets are owned by franchisees. Under California law, corporate headquarters is restricted in what it can require of them. Headquarters cannot determine the price charged by the outlet for its sandwiches.

### 3.5.2 Estimation ([Part 1](./08-part1.md))

It is traditional when estimating the parameters of a game theory model, to first present estimates from a standard model like a linear regression model. The game theory helps us think about what to put into that regression model but also makes clear that we should be careful in interpreting the results. The raw prices are adjusted to make the regression results look nicer.

The data used in the rest of the analysis is restricted to outlets that are not corporate owned. The issue is that their pricing strategies may look very different from the franchise locations.


`_>   file = paste0(dir,“outletsgtch3.csv”)_`
`_>   data = fread(file) |>_`
`_+     mutate(_`
`_+       LPrice1 = log(100*(Price - 2.49))_,`
`_+       BK = BK == 1_`
`_+     )_`
`_>   lm1 = lm(LPrice1 ~ BK, data)_`
`_>   lm2 = lm(LPrice1 ~ BK + Playland +_`
`_+              DriveThru + Mall, data)_`
`_>   lm3 = lm(LPrice1 ~ BK + Playland +_`
`_+              DriveThru + Mall + Race + Male, data)_`
[Table 3.1](./11-chapter3.md#tbl3_1) presents some regression results of log of sandwich price on characteristics of the outlet. This set up seems to match relatively closely to the table presented in [Thomadsen (2005)](./25-refbib.md#ref56). Prices are lower for The Whopper (on average) and are lower at outlets with various amenities and in Malls as well as in various locations based on demographic characteristics. The coefficients on most of the characteristics are not statistically significantly different from zero. This is probably due to the small sample size. One coefficient that is statistically significantly different from zero is the dummy on being a Burger King outlet. The Whopper is cheaper than the Big Mac.

__[Table 3.1](chapter3) OLS estimates of the equilibrium relationship between price and observed characteristics of the outlet and their location.__
| _Dependent variable:_ |                                               |                                               |                                               |
| --------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------------- |
| LPrice1               |                                               |                                               |                                               |
| (1)                   | (2)                                           | (3)                                           |                                               |
| BK                    | \-0.21[\*\*\*](./11-chapter3.md#tblfn3_3_1) | \-0.23[\*\*\*](./11-chapter3.md#tblfn3_3_1) | \-0.22[\*\*\*](./11-chapter3.md#tblfn3_3_1) |
| |  (0.05)             | (0.05)                                        | (0.05)                                        |                                               |
| Playland              | \-0.08                                        | \-0.07                                        |                                               |
| |  (0.06)             | (0.06)                                        |                                               |                                               |
| DriveThru             | \-0.02                                        | \-0.04                                        |                                               |
| |  (0.06)             | (0.06)                                        |                                               |                                               |
| Mall                  | \-0.13                                        | \-0.12                                        |                                               |
| |  (0.10)             | (0.10)                                        |                                               |                                               |
| Race                  | 0.50                                          |                                               |                                               |
| |  (0.64)             |                                               |                                               |                                               |
| Male                  | 0.16                                          |                                               |                                               |
| |  (0.19)             |                                               |                                               |                                               |
| Constant              | 4.54[\*\*\*](./11-chapter3.md#tblfn3_3_1)   | 4.59[\*\*\*](./11-chapter3.md#tblfn3_3_1)   | 4.50[\*\*\*](./11-chapter3.md#tblfn3_3_1)   |
| |  (0.03)             | (0.06)                                        | (0.11)                                        |                                               |
| Observations          | 79                                            | 79                                            | 79                                            |
| R2                    | 0.19                                          | 0.22                                          | 0.23                                          |

_Note:_ \*p<0.1; \*\*p<0.05; [\*\*\*](./11-chapter3.md#tblfn3_3_1c)p<0.01

### 3.5.3 Empirical Equilibrium

As we did in the previous chapter, we will assume that the prices observed in the data are determined by the Nash equilibrium of the game described above. The implication is that for the whole set of prices we see from all the outlets, the whole set of best responses must hold. In the final data set we have prices and information for 79 outlets in Santa Clara County.

Our estimation problem is to find the _α, β_, and _γ_ such that our 79 best response equations hold given the 79 prices we observe and the distance between each location. Actually, we are going to make our empirical model slightly more complicated by adding parameters for observed characteristics of the outlet.

2βpi−βci+Xi′α+γpjdij+ϵi\=0(3.14)

where Xi is a vector of observed characteristics for outlet _i_, e.g. (1, brandi, drivethrui), where brandi indicates the brand of the outlet and drivethrui is 1 if the outlet has a drive through and 0 otherwise.[6](./11-chapter3.md#fn3_6) The parameter _α_ is now a vector of parameters (α0,α1,α2).[7](./11-chapter3.md#fn3_7)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [6](./11-chapter3.md#fn3_63b)The matrix notation X to represent a block of data. In this case, each row is an outlet and each column is a characteristic of the outlet.

[7](./11-chapter3.md#fn3_73b)We are using matrix notation where two vectors of three elements x′y means (x1y1+x2y2+x3y3). As you can see the matrix notation is a lot more compact.

Writing this down for all the outlets we have 79 equations that need to hold (J\=79). How much they affect each other depends on the parameter _γ_ and the distance between each outlet, dij. The other thing to notice is the assumption that the demand parameters (_α, β, γ_) are the same for every outlet. What differs between the outlets is the observed characteristics Xi, their marginal cost _ci_ and the distance to the other outlets dij and unobserved characteristics of the outlet _ϵi_.

2βp1−βc1+X1′α+γ∑j\=2Jpjd1j+ϵ1\=02βp2−βc2+X2′α+γ(p1d12+∑j\=3Jpjd2j)+ϵ2\=0...2βpJ−βcJ+XJ′α+γ∑j\=1J−1pjdJj)+ϵJ\=0(3.15)

This is quite a mess. It is a lot more compact to write this out using matrix notation.

To see what is going on look at the case where there are just two firms (J\=2). Also let's assume that marginal costs are zero and there are no observed characteristics or unobserved characteristics of the outlets.

2βp1+γp2d12\=02βp2+γp1d21\=0(3.16)

We can write this out with matrices.

2β\[p1p2\]+γ\[01d121d210\]\[p1p2\]\=\[00\](./3.17)

Remembering the matrix multiplication rules, we can write this out to get the equations above. In full matrix notation, we have the following.

2βp+γDp\=0(3.18)

where D is a matrix representing the distances between the stores with zeros on the diagonal and the inverse distance between the stores in each cell, and _p_ is a vector of prices for the sandwiches.

The full empirical equilibrium can be written in matrix notation.

2βp−βc+Xα+γDp+ϵ\=0(3.19)

where _p_ is the vector of prices for the sandwich, _c_ is a vector of marginal costs for each outlet, X is the matrix of observed characteristics for each outlet, _α_ is a vector of representing how customers value those characteristics and D is a full matrix with the distances between all of the outlets in Santa Clara County. You have to admit, it looks a lot nicer.

### 3.5.4 Estimator

Our estimation problem is to find the parameters of the model such that equilibrium prices in the model most closely matche the observed prices. To do this we will use a **method of moments** estimator. It sounds pretty fancy but it is just least squares. The idea is that because there are some unobserved characteristics that are determining the price, represented by _ϵ_ in Equation (3.19) the equation will not precisely hold.

However we will assume that the Equation (3.19) holds on average for each outlet. We are assuming that the unobserved term is zero on average in equilibrium. [Assumption 1](./11-chapter3.md#assu3_1) makes this idea formal.

[Assumption 1.](chapter3) E(ϵi|Xi,p,ci,Di)\=0 _for all_ i∈{1,...,N}, _where_ Di _is the vector of distances from outlet i to every other outlet_.

Now we don't actually know the set of average prices for the outlets in the market. Rather we only observed a set prices for each outlet once. So we are going to assume that the analog of [Assumption 1](./11-chapter3.md#assu3_1) holds in the data we observe. Rather than requiring that this equation is exactly zero, we will look for the parameters that make the average of _ϵi_ squared as small as possible. We are going to minimize the sum of squares, or least squares.

min{α,β,γ}1N∑i\=1N(2βpi−βci+Xiα+γDip)2(3.20)

In the actual estimation we will add one more complication. We don't observe _ci_, the marginal cost at the outlets. We don't think that is a big deal because within a brand, the marginal costs will be very similar across the outlets in the data. One requirement from corporate headquarters is that each outlet use a supplier of similar quality to corporate's preferred supplier. They can't require a particular supplier be used but they can require certain standards be maintained. It is reasonable to think the cost of ingredients into the sandwich is pretty similar across all the outlets within a brand. In addition, all the outlets face similar labor markets and would pay similar wages. All this means that we assume ci\=brandi+ϵc.

### 3.5.5 Estimator in **R**

Now we need to translate all the math above into code so that we run the estimator on the data. The function outlet\_price\_f() maps Equation (3.20) into **R** code. It looks pretty similar but it is not quite the same. There is an extra matrix Omega. We will worry about this matrix in the next section. For the moment it is going to be the **identity matrix**, that is a matrix with all 1's on the diagonal and zeros every where else. Given this, the function is identical to Equation (3.20).[8](./11-chapter3.md#fn3_8) To do matrix multiplication in **R** the operation is A%\*%B where A and B are two matrices where the number of columns of A is equal to the number of rows of B. While the operation A\*B means that each cell of A is multiplied by each cell of B.[9](./11-chapter3.md#fn3_9)


`_> outletpricef = function(price, D, Omega, cost, X_,`
`_+                           alpha, beta, gamma) {_`
`_+     epsilon = beta*price +_`
`_+       gamma*D%*%price +_`
`_+       beta*(price - cost) +_`
`_+       gamma*(Omega*D)%*%(price - cost) +_`
`_+       X%*%alpha_`
`_+     sos = mean(epsilon^2)_`
`_+     return(sos)_`
`_+ }_`
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [8](./11-chapter3.md#fn3_83b)Do some matrix algebra with the identity matrix to convince yourself.

[9](./11-chapter3.md#fn3_93b)A and B should have the same dimensions, but it is not strictly necessary but does lead to weird results if it does not hold.

The first function tries to look as close to the math as possible. The second function is a translation function. It translates from what works best as a function to optimize to what looks nicest. This translation function (outlet\_price\_f\_int()) makes it clear to **R** that X is a matrix.[10](./11-chapter3.md#fn3_10). It then translates a vector par which is used by the optimization function optim(), into our parameters, alpha, beta, and gamma. We force beta to be negative and gamma to be positive. We do this using the exp() function, which takes a number and makes it greater than 0.[11](./11-chapter3.md#fn3_11) As a rule optimization algorithms are not good at subtlety. Better to let it choose what ever parameter values it likes and then translate its choice into what ever restrictions you want to place on the parameter.


`_> outletpricefint = function(par, price, D, Omega, BK, X) {_`
`_+   X = as.matrix(X)_`
`_+   J = dim(X)[2]_`
`_+   alpha = par[1:J]_`
`_+   beta = -exp(par[J+1])_`
`_+   gamma = exp(par[J+2])_`
`_+   cost = cbind(1,BK)%*%par[c(J+3,J+4)]_`
`_+   return(outletpricef(price, D, Omega, cost_,`
`_+                         X, alpha, beta, gamma))_`
`_+ }_`
### 3.5.6 Distances

In order to use the distance between outlets in our estimation we need to calculate distance between outlets. The code uses the dist() function to calculate the Euclidean distance between two points (“as the crow flies distance”). It is used in the loop to calculate the distance between all the outlets in the data set. The code runs a loop in **R** using for(). Whenever running a loop in **R** it is good practice to create an empty object, here it is a matrix dist\_mat, that gets filled in during the loop.[12](./11-chapter3.md#fn3_12)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [10](./11-chapter3.md#fn3_103b)**R** sometimes gets confused about what is a matrix and what isn't, so you need to repeat yourself a bit.

[11](./11-chapter3.md#fn3_113b)It is the exponential function.

[12](./11-chapter3.md#fn3_123b)This little trick speeds up the **R** substantially.


`_>   dist = function(lat0, lon0, lat1, lon1) {_`
`_+     return(sqrt((lat0 - lat1)^2 + (lon0 - lon1)^2))_`
`_+   }_`
`_>   J = dim(data)[1]_`
`_>   distmat = matrix(NA, J, J)_`
`_>   for(i in 1:J) {_`
`_+     for(j in 1:J) {_`
`_+       distmat[i,j] = dist(data$lat[i], data$lon[i]_,`
`_+                            data$lat[j], data$lon[j])_`
`_+     }_`
`_+   }_`
Now we can create our matrix D by finding the inverse of the distance for each outlet combination and setting the values on the diagonal to 0.[13](./11-chapter3.md#fn3_13) Lastly we need to set the stores that have the same location to have a distance that is very close.[14](./11-chapter3.md#fn3_14)


`_> D = 1/distmat_`
`_> diag(D) = 0_`
`_> D[is.infinite(D)] = 1/0.006_`
### 3.5.7 Ownership

Almost there. There is one more thing to discuss before estimating prices and that is ownership. In the pricing model, we made the simplifying assumption that each outlet is priced independently. That is not true in the data. In the data, there are people that own multiple franchises. In this case, the owner is going to price their sandwiches accounting for the fact that they own other outlets.

Assume that one person owns two outlets and must choose the optimal price for each. Remember the demand function for each outlet depends on the prices of both outlets.

max{p1,p2}q1(p1,p2)(p1−c1)+q2(p2,p1)(p2−c2)(3.21)

For this case the first order conditions are as follows.

p1:q1(p1,p2)+dq1dp1(p1−c1)+dq2dp1(p2−c2)\=0p2:q2(p2,p1)+dq1dp2(p1−c1)+dq2dp1(p2−c2)\=0(3.22)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [13](./11-chapter3.md#fn3_133b)In **R** the operation 1/D calculates the inverse at each cell of the matrix, not the matrix inverse. To calculate the matrix inverse use solve().

[14](./11-chapter3.md#fn3_143b)0.006 is the minimum distance of stores that don't have the same location.

So this time if they raise the price of the sandwich in outlet 1, they lose sales (dq1dp1) but they gain back profits in proportion of the people that switch to the other outlet (dq2dp1). Because the cost to the outlet of increasing price is mitigated by the recapturing of customers, prices will be higher.

In our problem, we have the following.

dq1dp1\=βdq1dp2\=γd12(3.23)

The omega notation allows us to keep track of the various ownership possibilities. Each outlet is assumed to own itself (ω11\=ω22\=1), but there is a possibility that both outlets are owned by the same person (ω12\=ω21\=1) or by different people (ω12\=ω21\=0).

βp1+γp2d12+ω11β(p1−c1)+ω12γp2−c2d12\=0βp2+γp1d21+ω22β(p2−c2)+ω21γp1−c1d21\=0(3.24)

We can write it in the following form.

2βp−βc+γDp+γΩ⋅D(p−c)\=0(3.25)

where Ω⋅D means cell by cell multiplication and Ω is the ownership matrix.

Ω\=\[ω11ω12ω21ω22\](./3.26)

Remember that the diagonals of the matrix D are zeros.

### 3.5.8 Ownership of Outlets

In the data, there is an index for the different owners. If the ownership variable is 0 the outlet is independent and individually owned. If the variable is 1, then it is corporate owned (and has been dropped). If the variable is greater than 1 then the number is used to identify which outlets have the same owner. For example, outlets with ownership set to 6 have the same owner. The code below uses which() to find the column where the ownership code is the same. Each outlet is assumed to be owned by itself so the diagonal of the matrix is 1s.


`_>   Omega = matrix(0, J, J)_`
`_>   for(j in 1:J) {_`
`_+     if(data$Ownership[j] > 1) {_`
`_+       Omega[j,which(data$Ownership == data$Ownership[j])] = 1_`
`_+     }_`
`_+   }_`
`_>   diag(Omega) = 1_`
### 3.5.9 Estimation ([Part 2](./14-part2.md))

The final piece of the puzzle is to use the optim() function. This is a basic optimization algorithm used in **R**. It is used because it was made freely available by John C. Nash, no relation to John F. Nash, of Nash equilibrium fame. To use this function we give it a set of starting values (init). Here these are set to zero for alpha and very small numbers for beta and gamma. When the first function is called, it hits these values with exp() and exp(log(a))\=a. That is, _log_ is the inverse of _exp_. The diag() function creates a matrix with values on the diagonal and zeros every where else. The function dim() finds the dimensions of a matrix, where the first element is the number of rows. We give optim() the initial values of the parameters, the function to optimize, then any extra information that the function will need. In this case the values for D, price, Omega, BK, and X. Lastly we can control the maximum number of iterations it will use before stopping. The default is small, so you may want to make this big using maxit. The parameter trace gives you a print out of what the function is doing when it is set to 1.


`_> init = c(rep(0, 5), log(0.001), log(0.001), c(0,0))_`
`_> a1 = optim(par = init_,`
`_+            fn = outletpricefint_,`
`_+            D = D_,`
`_+            price = exp(data$LPrice1)_,`
`_+            Omega = Omega_,`
`_+            BK = data$BK_,`
`_+            X = cbind(data$Playland_,`
`_+                      data$Mall_,`
`_+                      data$DriveThru_,`
`_+                      data$Race_,`
`_+                      data$BK)_,`
`_+            control=list(trace = 0,maxit = 10000000))_`
` `
` `
`_> # alpha_`
`_> a1$par[1:5]_`
`  [1] -0.005985475 -0.017518129   0.005481921 -0.006670482`
`      -0.005309910`
`_> # beta_`
`_> -exp(a1$par[6])_`
`  [1] -2.308111e-05`
`_> # gamma_`
`_> exp(a1$par[7])_`
`  [1] 2.583274e-08`
`_> # cost_`
`_> a1$par[8:9]_`
`  [1] 4.717501 8.238029`
Standard errors have not been calculated for these estimates. How would you do that?

### 3.5.10 Goodness of Fit

It is standard in structural econometrics to present some type of goodness of fit analysis. Here we can use the parameter values to generate the predicted price. We can then compare the prediction to the observed prices.

To do the analysis we need to create a new function. This function finds the new set of prices given the parameter values we found above.


`_> outletpricefint2 = function(par, D, Omega_,`
`_+                                cost, X, alpha, beta, gamma) {_`
`_+   X = as.matrix(X)_`
`_+   price = exp(par)_`
`_+   return(outletpricef(price, D, Omega_,`
`_+                         cost, X, alpha, beta, gamma))_`
`_+ }_`
` `
` `
`_> init = data$LPrice1_`
`_> b1 = optim(par = init_,`
`_+            fn = outletpricefint2_,`
`_+            D = D_,`
`_+            Omega = Omega_,`
`_+            cost = cbind(1,data$BK)%*%a1$par[8:9]_,`
`_+            X = cbind(data$Playland_,`
`_+                      data$Mall_,`
`_+                      data$DriveThru_,`
`_+                      data$Race_,`
`_+                      data$BK)_,`
`_+            alpha = a1$par[1:5]_,`
`_+            beta = -exp(a1$par[6])_,`
`_+            gamma = exp(a1$par[7]))_`
[Figure 3.3](./11-chapter3.md#fig3_3) presents the goodness of fit of the simulation using the estimated parameters. In this exercise, we use the estimated parameters and then simulate the prices that satisfy the Nash equilibrium condition. The fitted curve is a little higher than the actual prices.

![Three curves show density of normalized price from 20 to 160 on the horizontal axis. The solid line shows actual prices. A dashed line shows predicted prices. A dotted line shows merger simulation prices. All three lines rise steeply to a peak around 80, then fall with smaller bumps near 120. The dotted line is slightly wider. The dashed line follows the solid line closely until the peak but diverges after that. The vertical axis is unlabeled.](./images/fig3_3.jpg)

[Figure 3.3](chapter3) Plot of density of actual prices (solid line), predicted prices (dashed line) from the estimating model and simulated prices from the model of the merger of independent McDonald's outlets (dotted line). The dotted line is shifted up from the actual and simulated price distributions.

### 3.5.11 A Merger of McDonald's Outlets

It is not clear if the FTC has ever analyzed the impact of concentration among McDonald's franchisee owners. The economic theory is not different from analyzing the impact of a merger between hospitals or supermarkets.

Our experiment is for the independent McDonald's outlets to be purchased by the same person.


`_>   data$Ownership2 = ifelse(data$BK == 2_,`
`_+                          10_,`
`_+                          data$Ownership)_`
`_>   J = length(data$Ownership2)_`
`_>   Omega2 = matrix(0, J, J)_`
`_>   for(j in 1:J) {_`
`_+     if(data$Ownership2[j] > 1) {_`
`_+       Omega2[j, which(data$Ownership2 == data$Ownership2[j])] = 1_`
`_+     }_`
`_+   }_`
`_>   diag(Omega2) = 1_`
We will assume the merger changes the ownership of the outlets but not the existence of the outlets. In our math and our code, this change is captured using Omega matrix. The code above finds all the McDonald's outlets that are independently owned and sets them to all have the same ownership.

The new firm's pricing decision for any outlet accounts for how that outlet's price affects demand at their other outlets. When one McDonald's increases the price for their Big Mac, some customers will switch to another outlet. After the merger many of these customers switch to outlets that are owned by the same firm. The merger reduces the loss in profits when prices are increased. The merger will lead to higher prices for Big Macs.


`_> init = b1$par_`
`_> c1 = optim(par = init_,`
`_+            fn = outletpricefint2_,`
`_+            D = D_,`
`_+            Omega = Omega2_,`
`_+            cost = cbind(1,data$BK == 1)%*%a1$par[c(8,9)]_,`
`_+            X = cbind(data$Playland_,`
`_+                      data$Mall_,`
`_+                      data$DriveThru_,`
`_+                      data$Race_,`
`_+                      data$BK)_,`
`_+            alpha = a1$par[1:5]_,`
`_+            beta = -exp(a1$par[6])_,`
`_+            gamma = exp(a1$par[7]))_`
The code below generates the ggplot() of the density of prices for the actual prices, the predicted prices from the estimated model and the predicted prices from the model of the merger of independent McDonald's outlets.


`_> ggplotdensoutlets = data.frame(_`
`_+   Price = exp(data$LPrice1)_,`
`_+   sim = exp(b1$par)_,`
`_+   merger = exp(c1$par)_`
`_+ ) |>_`
`_+   ggplot(aes(x = Price)) +_`
`_+   geomdensity(aes(y = ‥scaled‥), alpha = 0.5) +_`
`_+   geomdensity(aes(x = sim, y = ‥scaled‥)_,`
`_+                alpha = 0.5_,`
`_+                linetype = “dashed”) +_`
`_+   geomdensity(aes(x = merger, y = ‥scaled‥)_,`
`_+                alpha = 0.5_,`
`_+                linetype = “dotted”) +_`
`_+   geomtext(aes(x = 50, y = 1, label = “Actual”)) +_`
`_+   geomtext(aes(x = 75, y = 0.3, label = “Sim”)) +_`
`_+   geomtext(aes(x = 140, y = 0.4, label = “Merger Sim”)) +_`
`_+   labs(x = “Price (normalized)”_,`
`_+        y = “”_,`
`_+        title = “”) +_`
`_+   ## no numbers on y axis_`
`_+   scaleycontinuous(breaks = NULL) +_`
`_+   thememinimal()_`
` `
` `
`_> ggplotdensoutlets_`
[Figure 3.3](./11-chapter3.md#fig3_3) shows both the goodness of fit of the model and the simulated impact on price of a merger between all the independent McDonald's outlets. The roll-up of independent McDonald's outlets would lead to a substantial increase in the prices of outlets in the market. Not only the McDonald's outlets but also the Burger King outlets that they compete with.

## 3.6 Discussion and Further Reading

Using game theory to analyze oligopoly models of competition actually predates game theory. In fact, all three models presented in this chapter predate Nash's analysis of game theory, even though all rely on the equilibrium concept.

Hotelling's 1929 paper, _Stability in Competition_, provides much of the intuition for the way many industrial organization economists think about competition for retail products ([Hotelling, 1929](./25-refbib.md#ref37)). In the model, it is the “distance” between products that matters for competition. Not so much the exact number of competitors but how close the competitors are to each other in the minds of consumers.

Somewhat confusingly we refer to a general model of differentiated price competition as a Bertrand model.[15](./11-chapter3.md#fn3_15) The classic paper taking this model to the data is [Berry et al. (1995)](./25-refbib.md#ref14).

The subtlety of the Hotelling model didn't fit well into how US antitrust conducted merger review. The standard merger screen is a measure called the Herfindahl-Hirschman index (HHI) and the change the index caused by the merger. It is calculated by determining which firms are in the market, calculating each firm's share, squaring them and adding them up. While HHI is not a bad approximation of competition in homogeneous goods markets modeled by Cournot, it doesn't make a lot of sense for differentiated goods markets modeled by Hotelling. In the simple Hotelling line, the extent of competition can vary substantially without any change in the HHI.

The 2010 Merger Guidelines from the Department of Justice and the Federal Trade Commission, made an adjustment suggesting that a different screen may be better for differentiated goods mergers. The Upward Price Pressure screen measures how close two firms are by how many customers are diverted for a price increase.[16](./11-chapter3.md#fn3_16)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [15](./11-chapter3.md#fn3_153b)[Chapter 7](./16-chapter7.md) works through this model.

[16](./11-chapter3.md#fn3_163b)<https://www.justice.gov/atr/horizontal-merger-guidelines-08192010> accessed on 11/21/23.

[_OceanofPDF.com_](./https___oceanofpdf.com)
