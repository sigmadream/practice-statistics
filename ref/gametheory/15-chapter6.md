# 6Dynamic Games

DOI: [10.1201/b23262-6](./https___doi.org_10.1201_b23262-6.md)

## 6.1 Introduction

The first part of the book considers static games of complete information. These games have very simple strategies. Perhaps with the exception of games with mixed strategies, these games are straightforward to analyze. This part of the book considers dynamic games. These games have strategies that can be substantially more complex than those we have seen. This complexity will force us to make important simplifying assumptions.

A **dynamic game** refers to the information available to the player at the time of their choice of action. If the player has no information about what the other players are doing, then it is a static game. If at least one player is able to observe the actions of the other player prior to making their choice, then we have a dynamic game.

To restate whether or not a game of complete information is static or dynamic has to do with information, not time. We can have static games where players move at different times, but they don't get to observe the other player's action prior to their move. We can have dynamic games where players move more or less at the same time, but where at least one player observes the action of the other player prior to their move.

Definition 9. _In a_ dynamic game _at least one player observes information about the other player's actions prior to choosing their action_.

This chapter introduces an alternative representation of a game, the **extensive form** representation. [Chapter 1](./09-chapter1.md) introduced the **normal form** representation of a game. This is the matrix like object with one player on the columns and the other on the rows. The normal form representation is associated with static games while the **extensive form** representation is associated with dynamic games. There is nothing to stop you from using normal form representations for dynamic games or extensive form for static games, but we will see the value of the extensive form representation for dynamic games.

The chapter introduces a new equilibrium concept, **subgame perfection. Perfection** refers to equilibrium refinements. We are interested in reducing the number of predictions for the game that we believe are reasonable. The chapter revisits our analysis of entry game and the choices of Borders and Barnes & Noble about which counties to enter. We can reduce the multiplicity of equilibria in the entry game by assuming that Barnes & Noble moves first and using subgame perfection.

## 6.2 Extensive Form

The section makes an adjustment to the **coordination game** introduced in [Chapter 2](./10-chapter2.md) to illustrate the **extensive form** representation.

### 6.2.1 Coordination Game

Consider a variation on the coordination game. Assume that Player 1 chooses their action first and Player 2 observes that action before choosing their action.

* Players: Player 1 and Player 2
* Strategies:  
   1. Player 1: {BLACK, RED}  
   2. Player 2:  
         1. \* If Player 1 plays BLACK, then BLACK. If Player 1 plays RED, then BLACK.  
         2. \* If Player 1 plays BLACK, then RED. If Player 1 plays RED, then BLACK.  
         3. \* If Player 1 plays BLACK, then BLACK. If Player 1 plays RED, then RED.  
         4. \* If Player 1 plays BLACK, then RED. If Player 1 plays RED, then RED.
* Payoffs  
   1. {BLACK, {{BLACK: BLACK}, {RED: BLACK}}}: {2,5}  
   2. {BLACK, {{BLACK: RED}, {RED: BLACK}}}: {0,0}  
   3. {BLACK, {{BLACK: BLACK}, {RED: RED}}}: {2,5}  
   4. {BLACK, {{BLACK: RED}, {RED: RED}}}: {0,0}  
   5. {RED, {{BLACK: BLACK}, {RED, BLACK}}}: {0,0}  
   6. {RED, {{BLACK: RED}, {RED: BLACK}}}: {0,0}  
   7. {RED, {{BLACK: RED}, {RED: RED}}}: {5,2}  
   8. {RED, {{BLACK: BLACK}, {RED: RED}}}: {5,2}

Wow. Things got complicated right quick! Player 1's strategy is simple. This player does not have any information before they choose their action, so their strategies are just their actions. Things are a lot more complicated for Player 2\. Player 2 gets to observe Player 1's action prior to making their choice. Therefore Player 2's strategy must account for this information. Player 2 has four possible strategies. This is because there are two possible states, Player 1's action choice, and two choices for each state. Two times two is four.

We will use the notation {BLACK: BLACK} to mean if Player 1 plays BLACK, then Player 2 plays BLACK.

### 6.2.2 Strategies Revisited

**Strategies** are one of the three main components of a game. In [Chapter 1](./09-chapter1.md), we stated the definition of a strategy.

Definition 10. _A_ strategy _is a function that maps from the player's information set to the player's actions_.

While in the first part of the book, a strategy was simply an action. Here it is a function. It maps from the information observed to an action. Moreover, it is a function that maps from every possibility to an action. Sure, this is just another way of saying it is a function, but it is really important to remember that it is a complete plan. It states what the player will do in every possible and every conceivable case.

### 6.2.3 Nash Equilibrium of the Coordination Game

Remember the Nash equilibrium algorithm asks us to posit a candidate set of strategies for all the players and then check that each player's strategy is optimal given the posited strategies.

Player 1's strategy is just RED or BLACK, while Player 2's strategy describes what they will do for the two possible cases. The strategy states what Player 2 will do for a situation that never actually occurs. This is what is meant by a complete plan. Player 2's strategy states what they will do under _every_ circumstance, not just what happens in the proposed equilibrium.

Candidate: {RED,{{BLACK:BLACK},{RED:RED}}}

* Assume Player 1 plays RED
* Player 2's payoffs  
   1. {{BLACK:BLACK},{RED:RED}}: 2  
   2. {{BLACK:RED},{RED:RED}}: 2  
   3. {{BLACK:RED},{RED:BLACK}}: 0  
   4. {{BLACK:BLACK},{RED:BLACK}}: 0  
   5. Yes. Player 2's strategy is optimal (not dominated).
* Assume Player 2 plays {{BLACK:BLACK},{RED:RED}}
* Player 1's payoffs  
   1. RED: 5  
   2. BLACK: 2  
   3. Yes. It is a Nash equilibrium.

It is an equilibrium for Player 1 to choose RED and for Player 2 to also choose RED.

Are there any others?

Candidate: {BLACK,{{BLACK:BLACK},{RED:BLACK}}}

* Assume Player 1 plays BLACK
* Player 2's payoffs  
   1. {{BLACK:BLACK},{RED:RED}}: 5  
   2. {{BLACK:RED},{RED:RED}}: 0  
   3. {{BLACK:RED},{RED:BLACK}}: 0  
   4. {{BLACK:BLACK},{RED:BLACK}}: 5  
   5. Yes. Player 2's strategy is optimal (not dominated).
* Assume Player 2 plays {{BLACK:BLACK},{RED:BLACK}
* Player 1's payoffs  
   1. RED: 0  
   2. BLACK: 1  
   3. Yes. It is a Nash equilibrium.

There are at least two Nash equilibrium.

Is {BLACK,{{BLACK:BLACK},{RED:BLACK}}} a likely outcome from the game? The two players coordinate on black, which is fine. What about the idea that Player 2 would choose BLACK when they know Player 1 has chosen RED. Is that reasonable?

### 6.2.4 Game Tree

One way to see the issue with the Nash equilibria is to look at the **game tree**, the extensive form representation presented in [Figure 6.1](./15-chapter6.md#fig6_1).

![A game tree begins with player 1 at the top choosing between black and red. If player 1 chooses black, player 2 chooses between black and red. Choosing black leads to 2 comma 5. Choosing red leads to 0 comma 0. If player 1 chooses red, player 2 again chooses between black and red. Choosing black leads to 0 comma 0. Choosing red leads to 5 comma 2.](./images/fig6_1.jpg)

[Figure 6.1](chapter6) Extensive form representation of the dynamic coordination game tree.

A **game tree** is a directed graph, it consists of **nodes** and **edges**. The nodes represent places where a player makes a decision. An edge is the line and arrow from node to a node further along the game tree. The edge represents the flow of information.

Definition 11. _A_ node _of a game tree is a place where the player makes a choice_.

Consider the Nash equilibrium {BLACK,{BLACK:BLACK,RED:BLACK}}. We checked that it was in fact a Nash equilibrium above. Look at what happens on the tree. If Player 1 plays RED, then Player 2 has a choice of BLACK or RED. In the equilibrium, they choose BLACK, but their payoff would have been higher if they had chosen RED.

Working down the tree, the equilibrium states that Player 1 chooses RED, so we go down the right branch. Now it is Player 2's turn. They can choose RED or BLACK. If they choose RED they get 2, while if they choose BLACK they get 0\. In equilibrium they choose BLACK. Does that make sense? In the next section, we will consider an equilibrium refinement that rules out this case.

## 6.3 Subgame Perfection

The first equilibrium refinement we will consider is **subgame perfection**. An equilibrium refinement is an additional property of the predicted outcome that must be true. **Subgame perfection** requires a set of strategies be a Nash equilibrium but also that the subset of strategies associated with each **subgame** be a Nash equilibrium of the subgame.

The section defines subgame perfection and then works through the implications for the coordination game presented above.

### 6.3.1 Definition

[Definition 12.](chapter6) _A_ subgame _is a game that can be played from any node of the game tree_.

[Definition 4](./15-chapter6.md#defi6_4) introduces the idea of a **subgame**. When you look at the game tree in [Figure 6.1](./15-chapter6.md#fig6_1), there are three distinct nodes. For Player 2, there is the node after which Player 1 plays black and the node after which Player 1 plays red. There is also the initial node where Player 1 makes their choice of black or red. At each of these nodes we describe a separate game. This game is a subgame.

Definition 13. _A_ subgame perfect Nash equilibrium _is a Nash equilibrium where the strategies in each subgame are a Nash equilibrium from that subgame_.

If for a particular Nash equilibrium strategy set we can look at each subgame and associate the strategies with a Nash equilibrium for that subgame, then the strategy set is **subgame perfect**.

### 6.3.2 Coordination Game

In the **coordination game**, there are three subgames. The whole game is a subgame. The other two begin at Player 2's decision node. In these two subgames, there is just one player (Player 2) and their strategies are just the actions {BLACK, RED}.

Is {BLACK, {{BLACK: BLACK}, {RED: BLACK}}} subgame perfect?

Consider the subgame at the node where Player 1 choose RED.

* Players: Player 2
* Strategies: {BLACK, RED}
* Payoffs: BLACK: 0, RED: 2

It is not a Nash equilibrium of the subgame for Player 2 to choose BLACK because they would be better off choosing RED.

Subgame perfection removes outcomes that allow non-credible strategies. Nash equilibrium requires the players to choose strategies that are optimal given the strategies of the other players. Subgame perfection requires players to choose actions that are optimal even if in equilibrium these actions will never be played.

Is it better to be Player 1 or Player 2 in this game? You may think Player 2, as they get to see what Player 1 does and react to it. If we rule out non-credible threats, then Player 1 always gets their way. Player 1 has a **first-mover advantage**. If the first move in the game can commit to a strategy then they have a distinct advantage. They can force the other player to choose actions that the first player prefers.

### 6.3.3 Empirical Entry Game

Consider a different version of entry game analyzed in [Chapter 4](./12-chapter4.md). Instead of having Borders and Barnes & Noble enter at the same time, assume that Barnes & Noble moves first. Borders observed Barnes & Noble's decision to enter or not and then decides to enter.

[Figure 6.2](./15-chapter6.md#fig6_2) represents the entry game. Barnes & Noble moves first and chooses whether or not to enter the market and then Borders chooses. The payoffs state that upon entry the firm earns Xi′β1, but if they face competition then those profits fall by _α_1. The entry costs are captured by ξ1i.

![A game tree starts with B N at the top. The first decision branches into IN and OUT. If IN is chosen, the next node labeled Borders branches into IN and OUT, leading to outcomes a and b. If OUT is chosen at the top, the next node also labeled Borders branches into IN and OUT, leading to outcomes c and d.](./images/fig6_2.jpg)

[Figure 6.2](chapter6) Dynamic entry game tree. a\={Xi′β1−α1+ξ1i,Xi′β2−α2+ξ2i},b\={Xi′β1+ξ1i,0}, c\={0,Xi′β2+ξ2i} and d\={0,0}.

### 6.3.4 Equilibrium in Entry Game

To determine the **subgame perfect Nash equilibrium**, we solve the last subgame first. If Barnes & Noble entered the market, Borders will enter if and only if the following inequality holds.

Xi′β2−α2+ξ2i≥0(6.1)

Borders will only enter if the duopoly profits more than outweigh the entry costs. While if Barnes & Noble has not entered, Borders will enter if the following inequality holds.

Xi′β2+ξ2i≥0(6.2)

This time, Borders enters if monopoly profits are high enough.

Now we go back and consider Barnes & Noble's choices. If they enter, there are two cases. Case 1 is that Borders enters (Xi′β2−α2+ξ2i≥0), they will also enter if the following inequality holds.

Xi′β1−α1+ξ1i≥0(6.3)

Case 2 is that Borders does not enter, and so Barnes & Noble will enter if and only if Xi′β1+ξ1i≥0.

If Barnes & Noble chooses not to enter, then there are also two cases. In Case 1, where Borders enters, it is an equilibrium if and only if Xi′β1−α1+ξ1i≤0. In Case 2, it is an equilibrium if and only if Xi′β1+ξ1i≤0.

### 6.3.5 Empirical Implications

To summarize our four observed outcomes are a subgame perfect Nash equilibrium if the following inequalities hold.

* Both enter: Xi′β1−α1+ξ1i≥0 and Xi′β2−α2+ξ2i≥0.
* BN enters only: Xi′β1+ξ1i≥0 and Xi′β2−α2+ξ2i≤0
* Borders enters only: Xi′β1−α1+ξ1i≤0 and Xi′β2+ξ2i≥0.
* Neither firm enters: Xi′β1+ξ1i≤0 and Xi′β2+ξ2i≤0

How do these compare to the outcomes in [Chapter 4](./12-chapter4.md)? There is no indeterminacy. If you go back and look at [Figure 4.2](./12-chapter4.md#fig4_2) and the middle square it has {0,1} or {1,0}. Under this set up with subgame perfection it becomes {1,0}. Only Barnes & Noble will enter. Where before it there were two equilibria (more if you include mixed strategies), here there is always a unique subgame perfect Nash equilibrium. Of course, the price is that we need to make very strong assumptions about how the game is played.

## 6.4 Empirical Analysis: Bookstore Entry with Subgame Perfection in **R**

The estimator for the subgame perfect Nash equilibrium is basically the same as in [Chapter 4](./12-chapter4.md). The difference is that f\_entry\_spne() can separately estimate the cases where there is just one firm. Where previously there were multiple equilibria, now Barnes & Noble enters, while Borders does not.

### 6.4.1 SPNE Estimator

The code for the estimator is somewhat longer than the estimator used in [Chapter 4](./12-chapter4.md). The reason is that we now assume a unique equilibrium and so we can separately estimate all four possible cases.


`_> fentryspne = function(X, beta1, beta2, alpha1_,`
`_+                         alpha2, rho) {_`
`_+   N = dim(X)[1]_`
`_+   xi1 = Z1_`
`_+   xi2 = Z2*sqrt(1 - rho^2) + rho*Z1_`
`_+   Xb1 = X%*%beta1_`
`_+   Xb2 = X%*%beta2_`
`_+   p00 = p01 = p11 = rep(0, N)_`
`_+   for(k in 1:K) {_`
`_+     pi1k = Xb1 + xi1[k]_`
`_+     pi2k = Xb2 + xi2[k]_`
`_+     p00 = p00 + (pi1k < 0 & pi2k < 0)_`
`_+     p01 = p01 + (pi1k - alpha1 < 0 & pi2k > 0)_`
`_+     p11 = p11 + (pi1k - alpha1 > 0 & pi2k - alpha2 > 0)_`
`_+   }_`
`_+   return(list(p00 = p00/K_,`
`_+               p01 = p01/K_,`
`_+               p11 = p11/K))_`
`_+ }_`
### 6.4.2 SPNE Estimates

[Table 6.1](./15-chapter6.md#tbl6_1) shows that the assumption about simultaneous move gives very similar estimates to the assumption that Barnes & Noble moves first. The big difference is on the estimates of the impact of competition on the two firms. In order to reconcile the observed entry decisions with modeling assumptions, the estimator states Barnes & Noble is not affected much by competition, while Borders is.

__[Table 6.1](chapter6) Results from estimates of the game theory model from [Chapter 4](./12-chapter4.md) and the model assuming Barnes & Noble moves first with a subgame perfect Nash equilibrium. The two columns labeled “Multi” refer to the case where we assume there could between two pure strategy Nash equilibrium. The two columns labeled “SPNE” refer to the model where Barnes & Noble moves first and there is a subgame perfect Nash equilibrium.__
| Multi           | SD      | SPNE | SD      |      |
| --------------- | ------- | ---- | ------- | ---- |
| const\_1        | \-15.11 | 0.25 | \-15.12 | 0.06 |
| Pop\_1          | 1.07    | 0.01 | 1.04    | 0.02 |
| Income\_1       | \-0.76  | 0.48 | \-1.01  | 0.21 |
| College\_1      | 5.65    | 0.55 | 5.61    | 0.19 |
| Stores\_1990\_1 | 0.37    | 0.11 | 0.15    | 0.06 |
| const\_2        | \-11.37 | 0.20 | \-11.15 | 0.15 |
| Pop\_2          | 0.65    | 0.02 | 0.65    | 0.02 |
| Income\_2       | 1.31    | 0.80 | 1.93    | 0.17 |
| College\_2      | 2.70    | 0.55 | 2.85    | 0.17 |
| Stores\_1990\_2 | 0.79    | 0.11 | 0.63    | 0.08 |
| alpha\_1        | 0.73    | 0.18 | 0.50    | 0.16 |
| alpha\_2        | 0.70    | 0.12 | 1.08    | 0.14 |
| rho             | 0.47    | 0.10 | 0.39    | 0.06 |

[Table 6.2](./15-chapter6.md#tbl6_2) compares the model predictions to the actual data for the model presented in [Chapter 4](./12-chapter4.md) and for a model presented here where Barnes & Noble moves first and there is a subgame perfect Nash equilibrium. The subgame perfect Nash equilibrium model does a better job of predicting the case where there is only one firm, but a substantially worse job of predicting the two-firm case.

__[Table 6.2](chapter6) Comparison of predictions of the two models. The subgame perfect Nash equilibrium is somewhat better at fitting the case where there is only one firm, but not when there are two firms.__
| None        | One Firm | Two Firm |      |
| ----------- | -------- | -------- | ---- |
| Multi: None | 97.1     | 41.0     | 9.0  |
| Multi: One  | 2.6      | 41.7     | 36.6 |
| Multi: Two  | 0.3      | 17.3     | 54.4 |
| SPNE: None  | 97.5     | 48.8     | 14.0 |
| SPNE: One   | 2.4      | 43.2     | 53.3 |
| SPNE: Two   | 0.1      | 8.0      | 32.7 |

## 6.5 Discussion and Further Reading

Notice that so far in this part of the book, there has been no discussion of time in these so-called dynamic games. This goes back to the point made earlier, dynamics has to do with information not time. [Chapters 7](./16-chapter7.md) and [8](./17-chapter8.md) introduce time.

The empirical analysis in this chapter is based on work presented in [Adams and Basker (2025)](./25-refbib.md#ref2). The authors use information on the location of Barnes & Nobel and Borders collected from bookstore directories and firm websites to analyze the dynamics of the retail bookstore industry.

[_OceanofPDF.com_](./https___oceanofpdf.com)
