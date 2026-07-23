# 2Nash Equilibrium

DOI: [10.1201/b23262-2](./https___doi.org_10.1201_b23262-2.md)

## 2.1 Introduction

The modern version of game theory began with the PhD dissertation of the American mathematician, John Nash. You may recognize the name from the hit movie, _A Beautiful Mind_ with Australian Russell Crowe playing West Virginian Nash. Nash showed that for a large set of games, there is at least one outcome that is stable. Nash was interested in which outcome of the game is likely to occur. Stability seems like a prerequisite for any outcome that we are likely to observe. Stability says that if an outcome happens to occur then the outcome of the game is unlikely to change. In contrast, instability says that if an outcome happens to occur then the outcome of the game is likely to change. We are unlikely to observe an unstable outcome. The fact that Nash's solution concept is both a prerequisite for an outcome that is likely to occur and exists for a large set of games, makes it a super valuable idea. Today we call his proposed outcome a **Nash equilibrium**

Nash worked in the sub field of game theory called non-cooperative game theory. As he himself points out in his Nobel essay, this style of game theory was not in style.[1](./10-chapter2.md#fn2_1) It differed substantially from the ideas of Johnny von Neumann who was a senior mathematician at Princeton where Nash did his dissertation. Hungarian-American von Neumann helped develop game theory and was one of the first to suggest its use in economics. His collaboration with German-American economist, Oskar Morgenstern, produced the first game theory text, _Theory of Games and Economic Behavior_ (1944). Luckily for us, Nash was stubborn and his efforts helped turn non-cooperative game theory into what today we just call game theory.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./10-chapter2.md#fn2_12b)<https://www.nobelprize.org/prizes/economic-sciences/1994/ceremony-speech/>. Accessed on 2/23/23.

In this chapter, we will introduce this workhorse equilibrium concept as well as the concept of a **dominant strategy equilibrium**. The chapter illustrates these concepts by looking at the market structure in retail tire stores. How many stores are there going to be in the market? How does the number of stores relate to the price of tires in the market?

## 2.2 What is a Nash Equilibrium?

What is an **equilibrium**? We are looking for an outcome that is a reasonable prediction of the game. We are not looking for the best outcome or the worst outcome, but the outcome with the most empirical content. The one that we are likely to observe in the wild. This outcome is not chosen in some ad hoc manner but based on a set of assumptions. A set of rules.

Definition 4. _An_ equilibrium concept _is a set of rules for determining which outcome(s) of the game will occur_.

There are various assumptions we may want to hold in order to say that a particular outcome is a reasonable prediction of the game. The section presents the idea of **dominance** and how we may come up with a prediction of the game based only on assuming that the players of the game make choices that are rational.

### 2.2.1 Dominance

Before we get to **Nash equilibrium**, consider an alternative equilibrium concept, called **dominant strategy equilibrium**. [Chapter 1](./09-chapter1.md) introduced the concept of a **dominant strategy** and **weakly dominant strategy**.

Definition 5. _A strategy is_ dominant _if it gives the player the highest payoff irrespective of the strategies of the other players. A strategy is_ weakly dominant _if for some strategies of the other players, the payoff is equal highest_.

A dominant strategy is always the best choice. It does not matter what any other player of the game does. Given that it is always the best choice, then presumably it is the one that a rational player is most likely to choose.

Definition 6. _A (weakly)_ dominant strategy equilibrium _is an outcome of the game where all players play strategies that are (weakly) dominant strategies_.

If we assume that players are rational, and there exists an outcome which is a **dominant strategy equilibrium**, then we would expect that outcome to be the one we observe. The issue is that such outcomes may not exist in the game we are studying.

In the discussion above there is a subtle but important assumption. If a player has two choices, _A_ and _B_, where _A_ is part of a proposed equilibrium but _B_ gives the player the exact same payoff, then _A_ is still part of the equilibrium. For _A_ not to be part of an equilibrium, there must be a _B_ that makes the player strictly better off.

### 2.2.2 Prisoner's Dilemma

Consider a version of the prisoner's dilemma game presented in [Chapter 1](./09-chapter1.md). In this version, the actions have generic names, but the payoffs have the same ordering as they did in the version presented in the previous chapter. Remember it is the ordering that matters, just like in Formula 1.

[Table 2.1](./10-chapter2.md#tbl2_1) presents the generic prisoner's dilemma game with two players _P_1 and _P_2 and two actions BLACK and RED. The payoffs are written out in **normal form**, which is a matrix-like representation of the game.

__[Table 2.1](chapter2) Normal form representation of a prisoner's dilemma game with players _P_1 and _P_2 and actions BLACK and RED. _P_1's actions are on the rows. The payoffs are in the cells with _P_1's in the first position.__
| P1,P2 | BLACK | RED  |
| ----- | ----- | ---- |
| BLACK | 3, 3  | 0, 5 |
| RED   | 5, 0  | 2, 2 |

The outcome that has both players playing **dominant strategies** is {RED, RED}.

If you are player 1, you see that you are better off choosing RED irrespective of what player 2 does. In the table, player 1's payoff is the first element in the cell. Player 1's strategies are the rows. If player 2 chooses BLACK, we see that RED has a payoff of 5 which is greater than 3\. If player 2 chooses RED, we that RED has a payoff of 2 which is greater than 0\. RED is the **dominant strategy** for player 1\. Similarly for player 2.

The reason the prisoner's dilemma is so famous is that it predicts an outcome which is bad for the players of the game. Even though two rational players will end up at the dominant strategy equilibrium, both players would be better off choosing {BLACK, BLACK} which has a payoff of {3, 3}. This outcome has a payoff that is strictly better for both players than the predicted outcome which has a payoff of {2, 2}. Our equilibrium concept predicts the outcome that we believe is most likely to occur not necessarily the outcome that is the best for the players.

The prediction of this game drives real world policy. The US Department of Justice (DOJ) has an explicit policy of giving leniency to firms that are the first to provide evidence of collusion in a market.[2](./10-chapter2.md#fn2_2) The policy explicitly aims to create a prisoner's dilemma among colluding firms.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./10-chapter2.md#fn2_22b)<https://www.justice.gov/atr/leniency-program> accessed on November 12 2023.

Does dominant strategy equilibrium predict outcomes in real games? [Chapter 1](./09-chapter1.md) analyzes data from a real TV program where players are playing a prisoner's dilemma game for real money. In _Friend or Foe_, if both players choose Foe then they each leave the show without any of their winnings. While if they both choose Friend then they would share their winnings. In the data, we see that {Foe, Foe} has the highest probability of occurring.[3](./10-chapter2.md#fn2_3)

### 2.2.3 Coordination Game

Now consider a slightly different game. We call the game presented in normal form in [Table 2.2](./10-chapter2.md#tbl2_2), a coordination game. We will soon see why.

__[Table 2.2](chapter2) Normal form representation of a **coordination game** with players _P_1 and _P_2 and actions RED and BLACK. _P_1's action choices are on the rows. Payoffs are in brackets with _P_1 listed first.__
| P1,P2 | BLACK | RED  |
| ----- | ----- | ---- |
| BLACK | 2, 5  | 0, 0 |
| RED   | 0, 0  | 5, 2 |

Assume you are player 1\. Can you work out your best strategy? We can go through the best choice for each choice made by player 2.

* Assume _P_2 chooses BLACK. _P_1's payoffs are:  
   1. BLACK: 2  
   2. RED: 0
* Assume _P_2 chooses RED. _P_1's payoffs are:  
   1. BLACK: 0  
   2. RED: 5

Is there a dominant strategy to this game? No. The best choice for you depends on the choice of player 2\. If player 2 plays BLACK, then you should also play BLACK because 2\>0. If player 2 plays RED, then you should also play RED because 5\>0. See a **coordination game**! Both players should coordinate on the color.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [3](./10-chapter2.md#fn2_32b)The second part of the book considers changes to the prisoner's dilemma game that lead to predicted outcomes where the players have higher payoffs.

What outcome do you predict will happen in this game? While both players want to coordinate they do not agree on which outcome to coordinate. Player 1 prefers that they coordinate on RED and Player 2 prefers that they coordinate on BLACK.

## 2.3 Nash Equilibrium

The section discusses the definition of a **Nash equilibrium** and illustrates how to find one or more Nash equilibria in games that have been presented earlier in the chapter, the prisoner's dilemma and the coordination game.

### 2.3.1 Definition

Definition 7. A Nash equilibrium _is a set of strategies such that each player's strategy has the highest payoff given the strategies of the other players_.

Like with the dominant strategy equilibrium, we assume that each player chooses the strategy that gives the highest payoff. Our base assumption is that the players of the game are rational. The difference here is that choice is dependent on the choice of the other players.

Stability is baked into Nash's concept. Each player is choosing the best option given the choices of all the other players. If all players are choosing the best option, then there is no need for any player to change. The outcome is stable.

### 2.3.2 Algorithm for Finding Nash Equilibrium

While Nash equilibrium has a number of nice properties, finding it is not one of them. We will use the following cumbersome algorithm. One issue new students have with game theory is that it seems like the teacher just picked some outcome at random and voila, it is an equilibrium! Of course, the teacher did not just pick at random. They already knew the answer. In reality, you do not know which one to pick and so unfortunately you just have to try them all. There are no short cuts.

* Step 1: Choose a candidate outcome (set of strategies).
* Step 2: Hold Player 1's strategy fixed. Is Player 2's strategy optimal?  
   1. Yes: Go to Step 3.  
   2. No: Not a Nash equilibrium.
* Step 3: Hold Player 2's strategy fixed. Is Player 1's strategy optimal?  
   1. Yes: Nash equilibrium.  
   2. No: Not a Nash equilibrium.

If the game has more than two players, then the algorithm can be expanded to go through each player in turn.

### 2.3.3 Prisoner's Dilemma Game

Let's test out our algorithm on a game we already know, the prisoner's dilemma. For this game, there exists a more efficient algorithm, but we are illustrating how this more general algorithm works. Remember the first step is just picking an outcome. No magic.

* Step 1: {BLACK,BLACK}
* Step 2: _P_1 plays BLACK. Is BLACK optimal for _P_2?  
   1. BLACK: 3, RED: 5  
   2. No: Not a Nash equilibrium.

Let's pick another one.

* Step 1: {RED,RED}
* Step 2: _P_1 plays RED. Is RED _P_2's optimal strategy?  
   1. BLACK: 0, RED: 2  
   2. Yes. Go to Step 3.
* Step 3: _P_2 plays RED. Is RED _P_1's optimal strategy?  
   1. BLACK: 0, RED: 2  
   2. Yes!  
   3. {RED,RED} is a Nash equilibrium.

OK. There is some magic. You should try the algorithm on the other two outcomes.

### 2.3.4 Coordination Game

Let's try something a little more complicated, the coordination game presented in [Table 2.2](./10-chapter2.md#tbl2_2).

* Step 1: {BLACK,BLACK}
* Step 2: _P_1 plays BLACK. Is BLACK optimal for _P_2?  
   1. BLACK: 5, RED: 0  
   2. Yes. Go to Step 3.
* Step 3: _P_2 plays BLACK. Is BLACK optimal for _P_1?  
   1. BLACK: 2, RED: 0  
   2. Yes.  
   3. {BLACK,BLACK} is a Nash equilibrium.

That was easy/lucky. Are there any other Nash equilibria of this game?

* Step 1: {RED,RED}
* Step 2: _P_1 plays RED. Is RED _P_2's optimal strategy?  
   1. BLACK: 0, RED: 2  
   2. Yes. Go to Step 3.
* Step 3: _P_2 plays RED. Is RED _P_1's optimal strategy?  
   1. BLACK: 0, RED: 5  
   2. Yes.  
   3. {RED,RED} is a Nash equilibrium.

There are two Nash equilibria. Actually, there are even more, but we will come back to that in [Chapter 5](./13-chapter5.md).

One of the valuable things about a Nash equilibrium is that it always exists.[4](./10-chapter2.md#fn2_4) Unfortunately, the cost is that there may be more than one Nash equilibrium in any game. Is one of the Nash equilibria of the coordination game more reasonable than the other? What is the most reasonable prediction of the game? This question of whether there exists more reasonable equilibria is called refinement. In this book, we will consider a number of refinements of Nash equilibrium. [Chapter 4](./12-chapter4.md) analyzes what happens when the game we want to take to the data has multiple equilibria.

## 2.4 Entry Games

The section introduces empirical entry games. It presents the framework for how economists generally think about determinants of the number of firms in the market. You will often hear policy makers complain that prices are high because there is a lack of competition. There are too few firms in the market. Rarely, do policy makers step back and ask why there are too few firms in the market. The section presents a game in which a small number of firms choose to enter the market.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [4](./10-chapter2.md#fn2_42b)A Nash equilibrium exists for any finite game, that is any game where the number of players and the number of strategies is finite.

### 2.4.1 Bresnahan and Reiss

In a series of papers published in the early 1990s, the American economists, Tim Bresnahan and Peter Reiss, analyze the empirical implications of a simple entry game. [Bresnahan and Reiss (1991b)](./25-refbib.md#ref17) analyze average prices for retail tire stores in various U.S. towns. The data shows that there is basically no relationship between observed average prices for tires and the number of tire retailers in the market.[5](./10-chapter2.md#fn2_5) Why do you think that is?

The problem is **endogeneity**.[6](./10-chapter2.md#fn2_6) The raw observations do not account for the fact that these tire retailers choose whether or not to be in the market based on various factors including the cost of selling tires. We are more likely to see more firms in larger markets, but some of these firms are going to have higher costs of selling tires and these higher costs will drive up prices. We have two countervailing forces, more competition driving down prices and less efficient firms driving up prices.

This is not just some academic issue. It was a central concern in the US Federal Trade Commission's (FTC) case against the merger of Staples and Office Depot ([Ashenfelter et al., 2006](./25-refbib.md#ref7)). In the late 1990s, the two office supply super stores wanted to merge. As evidence against the merger, the FTC showed that prices where lower in cities with more stores. Is that a causal statement? Did the increase in the number of stores cause prices to fall? Moreover, would the opposite happen. If the merger took place and the number of independent firms in the market fell, would prices go up?

### 2.4.2 Two Firm Entry Game

Consider a relatively simple version of an entry game where there are just two firms. If only one firm enters then that firm earns monopoly profits and pays a fixed cost of entry, represented by the number 2\. If that firm doesn't enter, nothing happens, which is represented by 0\. If both firms enter there is competition but the firms also have to pay the fixed costs of entry. This payoff is represented by -1, which is bad. So a firm is willing to enter the market, but only if they are a monopolist.

More formally we have two players, the firm's strategies are entered or don't enter and the payoffs depend on whether the other firm enters as well.

* Players: Firm 1, Firm 2
* Strategies:  
   1. Firm 1: Enter, Don't Enter  
   2. Firm 2: Enter, Don't Enter
* Payoffs:  
   1. {Enter, Enter}: {-1, -1}  
   2. {Enter, Don't}: {2, 0}  
   3. {Don't, Enter}: {0, 2}  
   4. {Don't, Don't Enter}: {0, 0}

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [5](./10-chapter2.md#fn2_52b)Surprisingly the authors state that their data shows that “entry lowers margins”. It is unclear how they come to this conclusion.

[6](./10-chapter2.md#fn2_62b)In econometrics we use the term **endogeneity** to mean that different cases observed in the data may not be determined at random. Importantly, case assignment may be directly related to the observed outcome.

We can also represent this game in a normal form payoff matrix.

[Table 2.3](./10-chapter2.md#tbl2_3) presents the normal representation of the game. What is the Nash equilibrium of the game? It seems unlikely that both firms will enter and we see quickly that is not an equilibrium. If both firms enter, then Firm 1 would have been better off not entering as 0 is greater than -1\. See the first column and the payoffs of the first element which are for Firm 1.

__[Table 2.3](chapter2) A normal form representation of a two firm entry game. Firm 1's strategies are the rows and the Firm 2's strategies are the columns. The payoffs are in the cells, with Firm 1's payoff first.__
| Firm 1, Firm 2 | Enter   | Don't |
| -------------- | ------- | ----- |
| Enter          | \-1, -1 | 2, 0  |
| Don't          | 0, 2    | 0, 0  |

How about the case where both firms don't enter? Again we see that Firm 1 is better off entering. Look at the second column and see that 2 is greater than 0.

Let's check the other outcomes more systematically using our algorithm.

* Step 1: {Enter, Don't}
* Step 2: Firm 1 plays Enter. Is Don't optimal for Firm 2?  
   1. Enter: -1, Don't: 0  
   2. Yes. Go to Step 3.
* Step 3\. Firm 2 plays Don't. Is Enter optimal for Firm 1?  
   1. Enter: 2, Don't: 0  
   2. Yes. It is a Nash equilibrium!

Is that the only Nash equilibrium of the game?

No. There is another Nash equilibrium where Firm 2 enters but Firm 1 does not.[7](./10-chapter2.md#fn2_7) This entry game is a coordination-type game.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [7](./10-chapter2.md#fn2_72b)Again there are other equilibria but we will get to them. Be patient!

Game theory makes an interesting prediction. It predicts that the market will be a monopoly, but it doesn't predict which firm will be the monopoly. This turns out to have some implications for the empirical analysis of bookstores analyzed in [Chapter 4](./12-chapter4.md).

### 2.4.3 Many Firm Entry Game

We can make the game more general by having up to N¯ firms choosing whether or not to enter.

* Players: N¯\>0 firms
* Strategies: Enter, Don't Enter
* Payoffs:  
   1. Enter: (a−c)2b(N+1)2−F, where N≤N¯, where _N_ is the number of firms that choose Enter.  
   2. Don't Enter: 0

In this case we have a small number of firms, N¯. Each firm has a entry cost _F_. Entry costs may include finding a retail space, developing the space to sell tires, contracting with wholesalers and manufacturers, etc. Once firms enter (or not), the market mechanism determines the profits each firm will make.

The profit function is on the complicated side. Notice that the profits the firms make in the market are determined by the number of firms that enter (_N_). The more firms that enter, the lower the profits. Competition drives down prices and profits. Moreover, profits could be so low that they are below the fixed cost of entry. In that case, the firms are better off not entering.

To see where the complicated profit function comes from, assume price is determined by the following function.[8](./10-chapter2.md#fn2_8)

p\=aN+1+NcN+1(2.1)

where _c_ is the **marginal cost of production**, _a_ is a demand parameter, and _N_ is the number of firms that enter the market. As the number of firms enter the price falls and it converges to marginal cost as _N_ gets large. The first part is close to zero when N becomes big. The second part is close to c because the fraction of _N_ over N+1 is close to 1 when _N_ is big. The **marginal cost of production** refers to the incremental costs of selling a tire, this may include hourly wages and the wholesale cost of the tire itself.

If a firm enters the market their profits are determined by quantity that they sell (_q_) multiplied by their profit margin (p−c). Demand in the market is assumed to be determined by the following linear function Q(p)\=a−bp. The parameter _a_ determines the level of demand and the parameter _b_ determines the sensitivity of demand to price. Demand falls more dramatically for a particular price increase if _b_ is larger. Each firm in the market is identical and so they just split demand evenly, q\=QN.

q×(p−c)\=(a−c)2b(N+1)2(2.2)

Multiplying demand by margin gives the profit function.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [8](./10-chapter2.md#fn2_82b)This is the solution to a Cournot game with _N_ symmetric firms which is a game discussed in [Chapter 3](./11-chapter3.md).

### 2.4.4 Nash Equilibrium

What is the outcome of this game? A Nash equilibrium requires that each firm is playing its optimal strategy given the actions of all the other firms.

Each firm will enter if and only if the following inequality holds.

(a−c)2b(N+1)2\>F(2.3)

This means that if there are _N_ firms in the market in equilibrium. The inequality above must hold. If it didn't some of the firms would not enter the market. Also, it can't be profitable for another firm to enter. What if one more firm enters? In that case, profits for each firm must fall below the fixed cost of entry.

(a−c)2b(N+2)2<F(2.4)

If there are N<N¯ firms in the market, then it must be profitable for all of those firms to enter, but not profitable for any more firms to enter.

### 2.4.5 Fixed Cost of Entry

Let's add one more complication to the game. Let the fixed costs be a function of the number of firms that enter, F\=θ(N+1)K, where _K_ and _θ_ are some parameters of costs. This assumption states that when more firms are in the market their costs of entry are higher. It may be that land or facility space becomes more expensive when there are more firms looking to use the land or space to sell tires. This is the idea that different firms have different fixed costs of entry. In markets with a small number of firms, those that enter will have low fixed cost of entering. In markets with a large number of firms in equilibrium, the firms will have higher fixed cost of entering.

### 2.4.6 Equilibrium Number of Firms

The number of firms in the market is an integer, a counting number, 1, 2, 85, etc. Unfortunately, while these numbers are easy they are annoying to use for solving equations. To make solving the equation easier we will make the unrealistic assumption that the number of firms in the market is a real number.[9](./10-chapter2.md#fn2_9) Real numbers are useful because they have the property that there exists a solution to our equilibrium equation.

In equilibrium the following equality holds.

(a−c)2b(N+1)2\=θ(N+1)K(2.5)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [9](./10-chapter2.md#fn2_92b)Real numbers include integers and rational numbers (fractions) but also more exotic numbers like _π_ or (2).

If _N_ firms enter then for each firm they get the same profits from entering or not entering. The Nash equilibrium does not predict which _N_ firms enter just that there will be _N_ firms that enter. Solving, the equilibrium number of firms is N\=((a−c)2bθ)1K+2−1. Below we will use logs and it will look nicer.

We see that the equilibrium number of firms is increasing in the profitability of the market and decreasing in the customer's sensitivity to price.

### 2.4.7 Simulation of Entry Game in **R**

To understand how this game works, we can simulate entry into thousands of markets. In the simulation, there are 1,000 markets with a maximum of 10 firms. Demand and cost parameters vary from market to market as do entry costs.

The code uses the function runif(). This generates a set of random numbers that are uniformly distributed between 0 and 1\. The probability of drawing any particular number between 0 and 1 is the same. To be able to replicate the results exactly it uses set.seed().[10](./10-chapter2.md#fn2_10) The different values used are just made up.


`_>   set.seed(123456789)_`
`_>   M = 1000_`
`_>   Nbar = 10_`
`_>   a = 4 + 1*runif(M)_`
`_>   b = 0 + 0.6*runif(M)_`
`_>   c = 1 + 2*runif(M)_`
`_>   theta = 0 + 0.05*runif(M)_`
`_>   K = 0.9_`
We can put the solution to Equation (2.5) in code form. Given the equilibrium entry, we can then plug the numbers back into the pricing equation to determine equilibrium prices in each of the markets. This is an example of where **R** shines. We can write out something that looks pretty similar to the math, but is hiding a lot more complexity. These two lines actually determine the equilibrium number of firms and equilibrium prices for all 1,000 markets.


`_> N = (((a - c)^2)/(b*theta))^(1/(K+2)) - 1_`
`_> p = a/(N+1) + (N*c)/(N+1)_`
We can compare equilibrium prices to the theoretical relationship between prices and the number of firms. To determine the equilibrium prices we calculate the average price at each equilibrium level of the number of firms. To do this, we use the package data.table.[11](./10-chapter2.md#fn2_11) This package is very useful for doing calculations on lots of different subsets of the data. Here we want to do the calculation for each market with the same number of equilibrium entrants. Round the number of firms using round() so that the number of firms is an integer. The resulting data set dt, for data table, has two variables “N” and “p”. The next line takes the average price for each market with _N_ firms and creates a new data set dt1.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [10](./10-chapter2.md#fn2_102b)Computers don't actually generate random numbers. The numbers come from a complicated non-linear function. They look random, but if you know the previous number in the sequence and the function used, then you can exactly determine the next number in the sequence.

[11](./10-chapter2.md#fn2_112b)The syntax used by data.table is different from base **R** and also from tidyverse().


`_> library(data.table)_`
`_> dt = data.table(N = round(N)_,`
`_+                 p = p)_`
`_> dt1 = dt[, .(p = mean(p)), by = N]_`
To calculate the theoretical equivalent, we take the average for the parameter values and calculate the price if the number of firms in the market was determined exogenously rather than in equilibrium. The last line uses the price formula from Equation (2.1).


`_>   am   =   mean(a)_`
`_>   cm   =   mean(c)_`
`_>   Nm   =   1:Nbar_`
`_>   pm   =   am/(Nm + 1) + (Nm*cm)/(Nm + 1)_`
[Figure 2.1](./10-chapter2.md#fig2_1) presents the relationship between the number of firms in the market and the price. This is only an example, but it does show that the theoretical relationship between price and the number of firms can differ from the empirical relationship. The theoretical relationship shows that prices fall with competition. In the simulated data, the relationship between the number of firms and the price in the market is less negative, at least for a small number of firms. The reason is that the empirical relationship is an equilibrium where the firms are taking account of what the price will be when deciding to enter the market.

![In the graph, the vertical axis is labeled as price p or p underscore m and ranges from 0 to 4. The horizontal axis is labeled as N and ranges from 0 to 10. Circles representing theoretical prices decrease steadily from 3 to 2, as N increases from 0 to 10. Triangles representing equilibrium prices also decrease but at a slower rate. They decrease from 3 to 2, starting from N value 2 to 10. All data are approximate.](./images/fig2_1.jpg)

[Figure 2.1](chapter2) Scatter plot of prices from simulated data gives the number of firms (_N_) in the market. The equilibrium prices (dt1$p) (red triangles) fall less quickly than the theoretical prices (p\_m) (black circles).

If we just naively take the number of firms in a market and regress that on price, we are not finding the true relationship between price and competition. We are estimating a relationship mediated by equilibrium decisions of firms. Once we acknowledge this, we have two choices. Throw up our hands and give up or think seriously about these equilibrium decisions and how the data is generated.

## 2.5 Empirical Analysis: Tire Markets using **R**

Bresnahan and Reiss wrote a series of papers where they thought carefully about the empirical implications of entry games. In one of those papers, the authors get data on the number of firms in geographically distinct towns over a wide variety of industries including retail tire stores.

The section discusses the data on retail tire markets and how to take the model presented earlier in the chapter to the data. It estimates the parameters of the game and uses the game to simulate changes to policy such as reducing the costs associated with setting up a new firm.

### 2.5.1 Data

Those data provide information on the number of tire stores in each town, the population of each town, the number of commuters into the town, various economic indicators such as house prices and land prices and various demographic indicators such as age and family income.[12](./10-chapter2.md#fn2_12)

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [12](./10-chapter2.md#fn2_122b)The data used here comes from Jeremy Bejara and his Github repository from a 2019 structural industrial organization course, <https://github.com/jmbejara/comp-econ-sp19/blob/master/lectures/5-14%5FStructural%5FIO%5Fwith%5FMLE/bresnahan-reiss-1991-discussion.ipynb>

The code below also loads the data and plots the data. The data are in a csv file and is read in using read.csv(). The variable dir needs to be defined. This is set equal to the path where the data resides. The data is then plotted using ggplot() from the package ggplot2 (or tidyverse).


`_> file = paste0(dir, “BresnahanAndReiss1991DATA.csv”)_`
`_> read.csv(file) |>_`
`_+   ggplot(mapping = aes(TPOP_,`
`_+                        TIRE)) +_`
`_+   geompoint() +_`
`_+   labs(x = “Total Population (000s log scale)”_,`
`_+        y = “Number of Tire Stores”) +_`
`_+   scalexlog10()_`
Our interest is figuring out what determines the number of stores in a market. [Figure 2.2](./10-chapter2.md#fig2_2) shows that as the market gets larger, the number of firms increases.

![In the graph, the vertical axis is labeled as Number of Tire Stores and ranges from 0 to 12. The horizontal axis is labeled as Total Population in thousands log scale and ranges from 0.1 to 100. Small dots are scattered across the plot. Most dots are below 10 on both axes. The dots increase upward and to the right, showing more tire stores as the population increases. All data are approximate.](./images/fig2_2.jpg)

[Figure 2.2](chapter2) Scatter plot of the empirical relationship between the number of tire stores and the population of the town. It shows the general positive relationship with larger towns having more retail tire stores.

### 2.5.2 Structural Model

We will follow [Bresnahan and Reiss (1991b)](./25-refbib.md#ref17) and build a structural model. The basic idea is that the game theory provides the empirical relationship which we can then match to the data. From that matching, we can back out the parameters of the game theoretic model. Once we have the model parameters we can run policy simulations. At least that is the theory.

We can write Equation (2.5) in logs. We take logs of both sides of the equation and then rearrange to put the log of the number of firms on the left-hand side. This makes the equation look more like a linear regression equation.

(a−c)2\=b(N+1)2θ(N+1)Kor(2+K)log(N+1)\=2log(a−c)−log(b)−log(θ)orlog(N+1)\=22+Klog(a−c)−12+Klog(b)−12+Klog(θ)(2.6)

This transformation is useful for enabling us to use a standard **linear regression** estimator.

The relationship from the structural model states that the number of firms is increasing in average profits of the firm (a−c) and decreasing in the substitution to other products (_b_) and the entry costs (_θ_). The rate of increase is determined by the parameter _K_.

### 2.5.3 Entry Estimator

Equation (2.6) states that in equilibrium the number of firms (log of the number of firms) is determined by factors determining demand size such as total population and per-capita income, factors determining costs such as wages, factors determining the slope of the demand function such as closeness of substitutes and finally factors determining the cost of entry such as property rental costs.

Unfortunately, it is not clear how our data maps into our theoretical parameters. We do have good measures for overall demand, like total population, but we don't have information on wages. We have some information about land prices which may affect rental prices and entry costs. We have information on the number of commuters which may be a measure of both market size and substitution out of the market.

### 2.5.4 Estimation in **R**

To do the estimation we can run a linear regression with the log of the number of retail tire stores against characteristics of the town such as the population, number of commuters, income, and land values. The code below reads in the data and then runs the regression. The data.frame called data removes any observations with missing values using na.omit().[13](./10-chapter2.md#fn2_13) The code then runs two regressions. The first regression includes only the population of the town.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [13](./10-chapter2.md#fn2_132b)**R** uses NA to represent missing values.


`_> data = read.csv(file) |>_`
`_+   mutate(_`
`_+     logstores = log(TIRE + 1)_,`
`_+     population = TPOP_,`
`_+     cummuters = OCTY_,`
`_+     income = PINC_,`
`_+     landvalue = LANDV_`
`_+   ) |>_`
`_+   na.omit()_`
`_> lm1 = lm(logstores ~ population, data)_`
`_> lm2 = lm(logstores ~ population + cummuters +_`
`_+            income + landvalue, data)_`
[Table 2.4](./10-chapter2.md#tbl2_4) shows the empirical relationship between number of stores and various economic factors. It shows that there is strong positive relationship between market size and the number of firms. The empirical relationship between other factors is not obvious, at least in this data.

__[Table 2.4](chapter2) OLS estimates of the relationship between market characteristics and the number of retail tire stores.__
| _Dependent variable:_ |                                             |                                             |
| --------------------- | ------------------------------------------- | ------------------------------------------- |
| log\_stores           |                                             |                                             |
| (1)                   | (2)                                         |                                             |
| population            | 0.08[\*\*\*](./10-chapter2.md#tblfn2_4_1) | 0.07[\*\*\*](./10-chapter2.md#tblfn2_4_1) |
| |  (0.01)             | (0.01)                                      |                                             |
| cummuters             | 0.04                                        |                                             |
| |  (0.07)             |                                             |                                             |
| income                | 0.13[\*\*\*](./10-chapter2.md#tblfn2_4_1) |                                             |
| |  (0.04)             |                                             |                                             |
| land\_value           | 0.07                                        |                                             |
| |  (0.19)             |                                             |                                             |
| Constant              | 0.74[\*\*\*](./10-chapter2.md#tblfn2_4_1) | \-0.03                                      |
| |  (0.05)             | (0.22)                                      |                                             |
| Observations          | 202                                         | 202                                         |
| R2                    | 0.36                                        | 0.40                                        |

_Note:_ \*p<0.1; \*\*p<0.05; [\*\*\*](./10-chapter2.md#tblfn2_4_1c)p<0.01

### 2.5.5 Structural Estimation

The estimates presented in [Table 2.4](./10-chapter2.md#tbl2_4) provide the empirical relationship between the number of firms in the market and observed characteristics of the market. We are interested in mapping those estimates into our game theoretic model. Assume that the observed relationship captured by the second column of [Table 2.4](./10-chapter2.md#tbl2_4) is generated by the game described above (Equation (2.6)). Specifically, it represents a set of Nash equilibria of that game.

We will make the following assumptions.

* a\=a0exp(population)a1exp(commuters)a2ϵa
* c\=ϵc
* b\=b0exp(income)b1ϵb
* θ\=θ0exp(land\_value)θ1ϵθ

That is the demand level is a function of population and commuters and some unobserved characteristic (_ϵa_). You will see in a sec why we wrote this down in such a weird way.[14](./10-chapter2.md#fn2_14) The marginal cost is unobserved. We will be forced to assume that ϵc\=0 for what we do below.[15](./10-chapter2.md#fn2_15) The slope of demand is a function of income and an unobserved characteristic (_ϵb_). Lastly, the cost of entry is a function of land values and an unobserved characteristic (_ϵθ_).

Given these assumptions and the regression results in [Table 2.4](./10-chapter2.md#tbl2_4) we have the following relationships.[16](./10-chapter2.md#fn2_16) The coefficient estimates are on the changes in the values of the observed characteristics.

0.07\=22+Ka10.04\=22+Ka20.13\=−12+Kb10.07\=−12+Kθ1(2.7)

We can't uniquely determine the parameter values from these equations. In order to move forward we will assume that a1\=1. Given this assumption K\=0.86, a2\=0.06, b1\=−0.37 and θ1\=−0.20. The parameter estimates are determined relative to the coefficient on population.

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [14](./10-chapter2.md#fn2_142b)In economics we generally refer to this specification as a Cobb-Douglas function.

[15](./10-chapter2.md#fn2_152b)Given that we don't observe marginal costs our policy estimates will be OK as long as we don't try to model big changes.

[16](./10-chapter2.md#fn2_162b)We are just taking the coefficient values from Model (2) and not worring about how well those coefficients are estimated.

We have one more empirical relationship up our sleeve. The constant in the regression is -0.03\. Given our assumptions, we have that 0.70log(a0)−0.35log(b0)−0.35log(θ0)\=−0.03. Again, we don't have enough information to pin down all the parameters. If we assume that a0\=b0\=1, then θ0\=1.09.

### 2.5.6 Policy Simulation in **R**

What happens if the market size increases or if entry costs fall?

In this case to determine the equilibrium number of tire stores and equilibrium prices we create two functions using function(). These functions take in values for log of total population, log of commuters, log of income, and log of land values. They are based on Equations (2.6) and (2.1) respectively. The p() includes the equilibrium number of firms through the function N().


`_>   N = function() {_`
`_+     exp((2/(2 + K))*(log(a0) + a1*population + a2*commuters) -_`
`_+       (1/(2 + K))*(log(b0) + b1*income) -_`
`_+         (1/(2 + K))*(log(theta0) + theta1*landvalue) - 1)_`
`_+   }_`
`_>   p = function() {_`
`_+     exp((log(a0) + a1*population + a2*commuters))/_`
`_+       (N()+1)_`
`_+   }_`
We can use the estimates of the parameters of the game theoretic model to understand the likely impact of various policy changes.


`_>   K = 0.86_`
`_>   a0 = 1_`
`_>   a1 = 1_`
`_>   a2 = 0.06_`
`_>   b0 = 1_`
`_>   b1 = -0.37_`
`_>   theta0 = 1.09_`
`_>   theta1 = -0.20_`
These are our baseline estimates given the observed data and our baseline estimates for the number of firms and the price in the market, N0 and p0 respectively.[17](./10-chapter2.md#fn2_17)


`_>   population = mean(data$TPOP)_`
`_>   commuters = mean(data$OCTY)_`
`_>   income = mean(data$PINC)_`
`_>   landvalue = mean(data$LANDV)_`
`_>   N0 = N()_`
`_>   p0 = p()_`
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [17](./10-chapter2.md#fn2_172b)Defining variables outside any function means that they are **global** variables that can be accessed by any function. In this case those variables can be accessed by N() and p().

Consider a policy that increases commuters by 10%. This may be a policy that makes it easier to drive to the town or increases railway capacity or reduces parking costs downtown or requires government workers to return to the office. Note the weird way that the change comes in, this is because the measures are in logs.


`_> commuters = mean(data$OCTY) + log(1.1)_`
`_> N()/N0_`
`  [1] 1.004007`
`_> p()/p0_`
`  [1] 1.002058`
The policy has a very small effect, increasing the number of firms in the market by 0.4% and prices by about 0.2%.

Alternatively, consider a policy that increases the population by 10%. This might be something that increases the connectedness of the town to near by towns or allows more housing to be built.


`_> population = mean(data$TPOP) + log(1.1)_`
`_> commuters = mean(data$OCTY)_`
`_> N()/N0_`
`  [1] 1.068922`
`_> p()/p0_`
`  [1] 1.034692`
This policy increases the number of firms by 7% and prices by about 3.5%. Both of these policies increase the number of firms in the market but also increase the retail price of tires. Demand increases, which increases prices, but the impact is mediated by the equilibrium change in the number of firms and competition.

Lastly, consider a policy that decreases entry costs by 10%. This may be a policy that makes it easier for a retail tire store to enter the market, like reducing the permits required to set up the store or construct the store.


`_> population = mean(data$TPOP)_`
`_> theta0 = 0.9*theta0_`
`_> N()/N0_`
`  [1] 1.037526`
`_> p()/p0_`
`  [1] 0.9667752`
This policy increases the number of firms by 4% and reduces prices by 3%.

We may naively expect a policy that reduces entry costs to have a significant impact on both entry and prices. Both effects are mitigated in equilibrium. As it becomes cheaper to enter the market, firms understand that entering may not be that profitable because prices will fall. Similarly, because entry response is modest the price response is modest.

## 2.6 Discussion and Further Reading

In a series of papers, Tim Bresnahan and Peter Reiss showed how game theory could be used to improve empirical analysis of market structures. This chapter uses data from [Bresnahan and Reiss (1991b)](./25-refbib.md#ref17) to illustrate a structural model of firm entry in the retail tire market. In the original paper, the authors examine a number of markets and include information on prices for the retail tire market.

This chapter introduces the idea of a Nash equilibrium. In the entry game, firms will enter as long as it is profitable given all the other firms that will also enter. Because firms can strategically respond to the decisions of other firms, policies that we may naively believe to significantly lower prices, may not.

The chapter introduces the idea of using a structural model to interpret the data. It assumes that the observed relationships in the data are a Nash equilibrium of a particular entry game. While a lot of questionable assumptions are required given the data available, the analysis helps us to understand why various policies may have pretty modest impact on the number of firms in the market and the prices of tires.

In the FTC's case against Staple's acquisition of Office Depot, the FTC presented evidence that there was a positive relationship between prices and the number of retail office supply stores ([Ashenfelter et al., 2006](./25-refbib.md#ref7)). A concern with the analysis is that both the prices and the number of stores may be driven by other factors such as land values ([Manuszak and Moul, 2008](./25-refbib.md#ref44)). The game theory can help us understand potential pitfalls in the empirical analysis and which econometric methods may provide solutions. [Chapter 4](./12-chapter4.md) revisits firm entry games by analyzing the decisions of Barnes & Noble and Borders to enter various markets in the United States.

[_OceanofPDF.com_](./https___oceanofpdf.com)
