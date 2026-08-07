# 5Mixed Strategies

DOI: [10.1201/b23262-5](./https___doi.org_10.1201_b23262-5.md)

## 5.1 Introduction

[Chapter 2](./10-chapter2.md) stated that John Nash showed that the Nash equilibrium exists in a large set of games. This is a powerful result and is part of the reason why Nash equilibrium is so important. But Nash's proof relies on **mixed strategies**. Mixed strategies are a more complicated method for determining the outcome of a game. Mixed strategies are vital for the analysis of parlor games like chess or cards, and useful for the analysis of games with multiple equilibria like the **coordination game** presented in earlier chapters.

The chapter introduces mixed strategies using the classic childhood game Rock-Paper-Scissors. It analyzes penalty kicks in soccer's English Premier League. The chapter revisits the entry of big box bookstores analyzed in [Chapter 4](./12-chapter4.md). This time it assumes players are playing a mixed strategy when there are multiple equilibria.

## 5.2 Zero-Sum Games

**Zero-sum games** are the oldest types of games studied in game theory. They are a natural way to represent parlor games like chess. In those games, there is a winner and a loser. What the winner wins, the loser loses. This section introduces some important zero-sum games and discusses how to find the equilibrium.

### 5.2.1 Rock-Paper-Scissors

Rock-Paper-Scissors is one of the more famous childhood games. It requires two players with at least one hand. The game provides solutions to many previously unsolvable problems - who gets the last slice of cake, who gets to ride shotgun, who gets to mow the lawn.

* Players: Player 1 and Player 2
* Strategies: Rock, Paper, Scissors
* Payoffs: See [Table 5.1](./13-chapter5.md#tbl5_1)

__[Table 5.1](chapter5) Normal form representation of Rock-Paper-Scissors with two players _P_1 and _P_2, the strategies for _P_1 are on the rows and _P_2's strategies are on the columns. Payoffs are each cell, with _P_1's payoff listed first.__
| P1,P2    | ROCK   | PAPER  | SCISSORS |
| -------- | ------ | ------ | -------- |
| ROCK     | 0, 0   | \-1, 1 | 1, -1    |
| PAPER    | 1, -1  | 0, 0   | \-1, 1   |
| SCISSORS | \-1, 1 | 1, -1  | 0, 0     |

[Table 5.1](./13-chapter5.md#tbl5_1) provides a normal form representation for the game. There are only three outcomes in the game, win, lose, and draw. We can represent these outcomes as payoffs, 1, -1 and 0 respectively. If both players choose the same shape, it is a draw. Then Rock beats Scissors, Paper beats Rock, and Scissors beats Paper. Notice that the numbers in each cell add to 0, this is what we mean by zero-sum.

### 5.2.2 Nash Equilibrium

Does Rock-Paper-Scissors have a Nash equilibrium? To check whether a particular outcome is a Nash equilibrium we can follow the standard algorithm. In the first step, we posit the Nash equilibrium. Then in the second step, assume all but one player plays the posited strategy, determine the optimal strategy for the remaining player(s). If the posited strategy is not optimal, then it is not a Nash equilibrium.

* Step 1: ROCK, ROCK
* Step 2: _P_1 plays ROCK. Is ROCK optimal for _P_2?  
   1. ROCK: 0, PAPER: 1, SCISSORS: -1  
   2. No: Not a Nash equilibrium.

We see that if we posit that _P_1 plays Rock, then it is optimal for _P_2 to play Paper. Remember, Paper beats Rock. So {ROCK,ROCK} is not a Nash equilibrium.

Similarly we can use the algorithm to determine if {PAPER,ROCK} is a Nash equilibrium.

* Step 1: PAPER, ROCK
* Step 2: _P_1 plays PAPER. Is ROCK optimal for _P_2?  
   1. ROCK: -1, PAPER: 0, SCISSORS: 1  
   2. No: Not a Nash equilibrium.

Can you check other outcomes?

[Chapter 2](./10-chapter2.md) states that there exists a Nash equilibrium for this game. Why can't we find it?

### 5.2.3 Mixed Strategies

Theorem 1. _For any finite game, there exists a Nash equilibrium, where that equilibrium may be in mixed strategies_.

Nash found that for any game with a finite number of players and strategies there exists a Nash equilibrium, but only if you allow players to play mixed strategies.

Definition 8. _A_ mixed strategy _is a strategy that puts weights on each action, such that those weights are positive and sum to one_.

An example of a mixed strategy in Rock-Paper-Scissors is {0.2,0.3,0.5}, where that is 20% of the time the player chooses ROCK, 30% of the time they choose PAPER and 50% of the time they choose SCISSORS. This may not be a good mixed strategy but it is a mixed strategy. A strategy that puts 100 percent of the weight on a single action is a **pure strategy**. A **pure strategy Nash equilibrium** is just the Nash equilibrium discussed in [Chapter 2](./10-chapter2.md).

### 5.2.4 Penalty Kicks

Let's consider another famous **zero-sum game**. In soccer there are situations where one player can kick the ball at the goal with only the goal keeper to try to stop it. This may occur when the defending team fouls the team with the ball. The result of the foul is that the team with the ball gets a free shot on goal. In soccer, goal keepers are small relative to the goal and the Kicker is close enough that it is probably the case that the Kicker will score if they kick to one side or the other, unless the Goalie moves that direction at the same time as the kick is made. Because the Kicker is so close, the Goalie can't wait to see where the ball is aimed before diving.

Here is a basic version of the game.

* Players: Kicker, Goalie
* Strategies:  
   1. Kicker: Kick Left, Kick Right  
   2. Goalie: Dive Left, Dive Right
* Payoffs: See [Table 5.2](./13-chapter5.md#tbl5_2)

__[Table 5.2](chapter5) Normal form representation of a penalty kicks game, where the Kicker's actions are in the rows, the payoffs are in the cells and the Kicker's payoff is listed first.__
| Kicker, Goalie | LEFT   | RIGHT  |
| -------------- | ------ | ------ |
| LEFT           | \-1, 1 | 1, -1  |
| RIGHT          | 1, -1  | \-1, 1 |

[Table 5.2](./13-chapter5.md#tbl5_2) presents the normal form representation of the game. Again, the values in each cell sum to zero. If the Kicker scores a goal, they get 1 and the Goalie gets -1\. It is set up assuming that if the Kicker kicks left and the Goalie dives left then the goal is saved. The Goalie gets 1 and the Kicker gets -1\. If the Goalie dives to the left and the kick goes right it is a goal! The Goalie gets -1 and the Kicker gets 1.

Below the chapter looks at real data from penalty kicks. In the real game, a score increases the probability that the scoring team wins (and decreases the probability that the Goalie's team wins).

### 5.2.5 Algorithim for Mixed Strategy Nash Equilibrium

To understand the algorithm for determining a mixed strategy, think about what must be true in equilibrium. If we are in a mixed strategy equilibrium then each player must be indifferent between their strategy choices when the other player's strategy is taken as given. The Goalie must be indifferent between diving to the left or diving to the right, given the Kicker's strategy. If that wasn't the case, the Goalie would prefer to dive left (or dive right) which is a **pure strategy**. So for there to exist a mixed strategy equilibrium both players must be exactly indifferent to each pure strategy. They must be sitting on this knife's edge. Sounds painful.

Let _p_ be the probability that Kicker chooses LEFT. Let _q_ be the probability Goalie chooses LEFT.

* Find _q_ such that for Kicker, the payoff for LEFT equals the payoff for RIGHT.
* Find _p_ such that for Goalie, the payoff for LEFT equals the payoff for RIGHT.

If the Goalie's strategy is _q_, the weight on diving LEFT, then the Kicker's payoffs are:

* LEFT: (q)(−1)+(1−q)(1)\=1−2q
* RIGHT: (q)(1)+(1−q)(−1)\=−1+2q

Reading [Table 5.2](./13-chapter5.md#tbl5_2) and the first row, we see that if the first column is chosen, the Kicker gets -1 and 1 if second column is chosen (the first element).

The algorithm states that we need to find the _q_ such that the Kicker is indifferent between LEFT and RIGHT. What strategy by the Goalie makes the Kicker indifferent between kicking LEFT or RIGHT? What strategy of the Goalie gives the Kicker the same expected payoff? What _q_ is such that 1−2q\=−1+2q ? To repeat. We need to find the strategy of the Goalie that makes the Kicker indifferent between her two choices.

−4q\=−24q\=2q\=24q\=0.5(5.1)

Now let Kicker choose _p_, the weight on kicking LEFT. Goalie's payoffs are:

* LEFT: (p)(1)+(1−p)(−1)\=−1+2p
* RIGHT: (p)(−1)+(1−p)(1)\=1−2p

To see this look at the first column of [Table 5.2](./13-chapter5.md#tbl5_2) and the second element, the Goalie gets 1 if LEFT is chosen and -1 if RIGHT is chosen.

To find the equilibrium level for _p_, determine where the Goalie is indifferent between LEFT and RIGHT. This time, we need to find the strategy of the Kicker that makes the Goalie indifferent between his choices.

−1+2p\=1−2p4p\=2p\=24p\=0.5(5.2)

Nash equilibrium is {p\=0.5,q\=0.5}

One of the greatest penalty scorers was the Argentinian, Diego Maradona. According to Maradona, his secret to penalty kicks was to wait and see what the goalie did and then do the opposite. He cheated! Just like you used to do with your younger sibling when playing Rock-Paper-Scissors. Maradona was playing a different game. A dynamic game.

## 5.3 Rock-Paper-Scissors

We saw above that there is no pure strategy Nash equilibrium for the Rock-Paper-Scissors game. This section finds the mixed strategy Nash equilibrium (MSNE). It then explores how the equilibrium changes when the game becomes more complicated.

### 5.3.1 Mixed Strategy Nash Equilibrium

Now we know how to find a mixed strategy Nash equilibrium, consider a standard version of Rock-Paper-Scissors. Can you guess what the equilibrium is?

Let Player 2 choose {q1,q2}. So _q_1 is the probability Player 2 chooses ROCK, _q_2 is the probability that Player 2 chooses PAPER and 1−q1−q2 is the probability that Player 2 chooses SCISSORS. Find {q1,q2,1−q1−q2} such that Player 1 is indifferent.

Quick algorithm: Guess and confirm.

Guess: {q1\=13,q2\=13} Confirm:

* ROCK: (q1)(0)+(q2)(−1)+(1−q1−q2)(1)\=03+−13+13\=0
* PAPER: (q1)(1)+(q2)(0)+(1−q1−q2)(−1)\=13+03+−13\=0
* SCISSORS: (q1)(−1)+(q2)(1)+(1−q1−q2)(0)\=−13+13+03\=0

Confirmed!

Given that the two players are identical we can guess that the Nash equilibrium choices are the same: {{p1\=13,p2\=13},{q1\=13,q2\=13}}. Can you check that this is in fact the Nash equilibrium?

Are there any others? How would you check?

### 5.3.2 More Complicated Version

Now let's make things a bit more interesting. Assume Player 1 gets a high value of playing ROCK when Player 2 plays SCISSORS. And because this is a zero sum game, Player 2 gets a low value from playing SCISSORS when Player 1 plays ROCK. Probably best not to ask too many more questions. Given this preference for ROCK, what do you think will be the Nash equilibrium?

That's a good guess, but it is wrong.

__Table 5.3 Normal form representation of a more complicated Rock-Paper-Scissors. The payoff changed in the top right cell.__
| P1,P2    | ROCK   | PAPER  | SCISSORS |
| -------- | ------ | ------ | -------- |
| ROCK     | 0, 0   | \-1, 1 | 2, -2    |
| PAPER    | 1, -1  | 0,0    | \-1, 1   |
| SCISSORS | \-1, 1 | 1, -1  | 0, 0     |

What are the payoffs for Player 1? Let Player 2's strategy be {q1,q2,1−q1−q2}.

* ROCK: (q1)(0)+(q2)(−1)+(1−q1−q2)(2)
* PAPER: (q1)(1)+(q2)(0)+(1−q1−q2)(−1)
* SCISSORS: (q1)(−1)+(q2)(1)+(1−q1−q2)(0)

This lets us work out which mixed strategy of Player 2 that makes Player 1 indifferent between their choices. OK. That is, for what values of _q_1 and _q_2 are all three values equal to each other? This looks complicated. Let's use the computer.

### 5.3.3 Using **R** to Determine when Player 1 is Indifferent

We can write the payoffs for Player 1 has functions of Player 2's mixed strategy choice. We can then plot the functions to see if there is a point where Player 1 is indifferent. We can keep adjusting _q_1 and _q_2 until all three lines intersect.


`_>   rock = function(q1, q2) {_`
`_+     q1*0 + q2*(-1) + (1 - q1 - q2)*(2)_`
`_+   }_`
`_>   paper = function(q1, q2) {_`
`_+     q1*1 + q2*0 + (1 - q1 - q2)*(-1)_`
`_+   }_`
`_>   scissors = function(q1, q2) {_`
`_+     q1*(-1) + q2*(1) + (1 - q1 - q2)*(0)_`
`_+   }_`
The code creates the object ggplot\_rps which is a plot of the expected value of playing ROCK, PAPER, and SCISSORS for different 10 different values of _q_1 and q2\=0.42. We are cheating a bit by assuming we already know the value of _q_2.


`_> ggplotrps = data.frame(_`
`_+   q1 = seq(0, 1, by = 0.1)_,`
`_+   rock = rock(seq(0, 1, by = 0.1), rep(0.42, 10))_,`
`_+   paper = paper(seq(0, 1, by = 0.1), rep(0.42, 10))_,`
`_+   scissors = scissors(seq(0, 1, by = 0.1), rep(0.42, 10))_`
`_+ ) |>_`
`_+   ggplot(aes(x = q1)) +_`
`_+     geomline(aes(y = rock)) +_`
`_+     geomline(aes(y = paper), linetype = “dotted”) +_`
`_+     geomline(aes(y = scissors), linetype = “dashed”) +_`
`_+     geomvline(xintercept = 0.33_,`
`_+              linetype = “dashed”_,`
`_+              color = “gray”) +_`
`_+     scalexcontinuous(breaks = seq(0, 1_,`
`_+                                   by = 0.1)) +_`
`_+     scaleycontinuous(breaks = seq(-2, 2_,`
`_+                                   by = 1)) +_`
`_+     labs(title = “Payoff to Player 1”_,`
`_+         x = “Probability P2 chooses ROCK”_,`
`_+         y = “”) +_`
`_+     geomtext(aes(x = 0.5, y = -1, label = “rock”)) +_`
`_+     geomtext(aes(x = 0.5, y = 1, label = “paper”)) +_`
`_+     geomtext(aes(x = 0.8, y = 0, label = “scissors”))_`
` `
` `
`_> ggplotrps_`
[Figure 5.1](./13-chapter5.md#fig5_1) shows where Player 1 is indifferent for different values of _q_. Player 2 tends to play PAPER! If Player 2 chooses ROCK about one-third of the time and PAPER 42% of the time, then Player 1 is indifferent.

![The vertical axis is labeled as payoff to player 1 and ranges from negative 1 to 1. The horizontal axis is labeled as probability player 2 chooses rock and ranges from 0 to 1. Three lines represent expected payoffs for player 1. The solid line labeled rock slopes downward from upper left to lower right. The dotted line labeled paper slopes upward from lower left to upper right. The dashed line labeled scissors curves downward. All three lines intersect at around 0.33 on the horizontal axis.](./images/fig5_1.jpg)

[Figure 5.1](chapter5) The figure plots the expected value of playing ROCK, PAPER and SCISSORS when q2\=0.42 for different values of _q_1. Player 1 is indifferent between all three strategies when q1\=0.33.

### 5.3.4 Solving for MSNE using **R**

What about Player 2? Let Player 1 choose {p1,p2}.

* ROCK: (p1)(0)+(p2)(1)+(1−p1−p2)(−1)
* PAPER: (p1)(1)+(p2)(0)+(1−p1−p2)(−1)
* SCISSORS: (p1)(−2)+(p2)(1)+(1−p1−p2)(0)

For what strategy choice, is Player 2 indifferent between their three choices? For what values of _p_1 and _p_2 are these three values equal to each other?


`_>   rock1 = function(p1, p2) {_`
`_+     p1*0 + p2*(1) + (1 - p1 - p2)*(-1)_`
`_+   }_`
`_>   paper1 = function(p1, p2) {_`
`_+     p1*1 + p2*0 + (1 - p1 - p2)*(-1)_`
`_+   }_`
`_>   scissors1 = function(p1, p2) {_`
`_+     p1*(-2) + p2*(1) + (1 - p1 - p2)*(0)_`
`_+   }_`
For this case, let's use computer muscle to solve the problem. The function f\_rps() determines the values of _p_1 and _p_2 such that Player 2 is indifferent between the three choices. It does this by calculating the Euclidean distance between the choices, or the sum of squared differences. The function f\_rps\_int() is the intermediate function for optim(). This function translates the values chosen by the optimization algorithm into probabilities using the soft-max function.


`_>   frps = function(p1, p2) {_`
`_+     (rock1(p1, p2) - paper1(p1, p2))^2 +_`
`_+              (rock1(p1, p2) - scissors1(p1, p2))^2_`
`_+   }_`
`_>   frpsint = function(par) {_`
`_+     p = exp(par)/(1 + sum(exp(par)))_`
`_+     return(frps(p[1], p[2]))_`
`_+   }_`
`_>   init = c(0,0)_`
`_>   arps = optim(init, frpsint)_`
`_>   exp(arps$par)/(1 + sum(exp(arps$par)))_`
`    [1] 0.2499962 0.2500044`
Player 1 plays ROCK with probability of about one quarter and plays PAPER with probability one quarter and SCISSORS half the time.

What the heck is going on? We said that Player 1 prefers ROCK but ends up playing SCISSORS most of the time. The reason is that Player 2 doesn't like it if Player 1 chooses ROCK so they tend to play PAPER and because Player 2 plays PAPER, Player 1 tends to play SCISSORS. Clear? No?

## 5.4 Empirical Analysis: Penalty Kicks using **R**

In soccer penalty kicks occur for two reasons. The first may occur if a player with the ball is fouled near goal, actually in what is known as the penalty box. This area extends 16 meters either side of the goal and 16 meters forward of the goal. Once fouled in this area, the team with the ball gets a penalty kick. The second is when there is a drawn game and penalty kicks are used to determine the winner. In both cases, the team with the ball gets to kick the ball from 11 meters in front of goal with only the goal keeper between the player and the goal. The Goalie has so little time to react to the kick that we can think of the Goalie and the Kicker choosing their strategies simultaneously.

The section analyzes data from the English Premier League, which is the highest league in English and Welsh soccer.

### 5.4.1 Penalty Kick Game

[Table 5.4](./13-chapter5.md#tbl5_4) presents the normal form representation of the game we will take to the data. Each player has three action choices, LEFT, CENTER, and RIGHT. The payoffs are given by the 9 parameters to be estimated. The parameter pij is the probability that the Kicker scores a goal given that the Kicker chooses i∈{l,c,r} and the Goalie chooses j∈{l,c,r}. The value to the Goalie is just the negative. It is a zero-sum game.

__[Table 5.4](chapter5) Normal form representation of a penalty kicks game, where the Kicker's actions are in the rows, the payoffs are in the cells and the payoffs are the parameters to be estimated. It is a zero-sum game.__
| Kicker, Goalie | LEFT | CENTER | RIGHT |
| -------------- | ---- | ------ | ----- |
| LEFT           | pll  | plc    | plr   |
| CENTER         | pcl  | pcc    | pcr   |
| RIGHT          | prl  | prc    | prr   |

We will allow the two players to use mixed strategies. The vector representing the Kicker's strategy is as follows.

qk\=\[qklqkcqkr\](./5.3)

The weights must some to one, ∑i∈{l,c,r}qki\=1. Similarly, we can represent the Goalie's strategy.

qg\=\[qglqgcqgr\](./5.4)

### 5.4.2 Data

The data used here are from Kaggle and covers penalty kicks in the English Premier League during the 2016/2017 season.[1](./13-chapter5.md#fn5_1)


`_> file = paste0(dir, “penaltydata.csv”)_`
`_> data = fread(file) |>_`
`_+   filter(_`
`_+     KickDirection != “”_`
`_+   )_`
The code brings in the data and removes the cases where the kick direction is missing. The code uses fread() which is part of data.table. The first step is calculating the strategies for both players. The assumption is that every Kicker and Goalie is using the same strategy. Alternatively, you can think of this as the average strategy choice from a distribution of strategies. The code uses table() to count the number of each case and then divides by the total number of observations to get the probability of each strategy choice. The vector _qk_ is then reordered to match the order in [Table 5.4](./13-chapter5.md#tbl5_4). The same is done for the Goalie.


`_> qk = table(data$KickDirection)_`
`_> qk = qk/sum(qk)_`
`_> qk = qk[c(2, 1, 3)]_`
` `
` `
`_> qg = table(data$KickDirection)_`
`_> qg = qg/sum(qg)_`
`_> qg = qg[c(2, 1, 3)]_`
We can also calculate the parameter value from [Table 5.4](./13-chapter5.md#tbl5_4).


`_>   action = c(“L”, “C”, “R”)_`
`_>   resmat = matrix(NA, 3, 3)_`
`_>   for(i in 1:3) {_`
`_+     for(j in 1:3) {_`
`_+       dtij = data[data$KickDirection == action[i] &_`
`_+                    data$KeeperDirection == action[j]]_`
`_+       resmat[i,j] = mean(dtij$Scored == “Scored”)_`
`_+     }_`
`_+   }_`
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./13-chapter5.md#fn5_15b)<https://www.kaggle.com/datasets/mauryashubham/english-premier-league-penalty-dataset-201617> license: CCO: Public Domain, ShubhamMaurya, English Premier League Penalty Dataset, 201617.

[Table 5.5](./13-chapter5.md#tbl5_5) presents the empirical results of the penalty kick game. Does it look like what you expected? We would expect a lower probability of scoring when the Goalie chooses the same direction as the Kicker. These are on the diagonal and we see that in fact the lower probabilities are on the diagonal. Kicking to the center leads to a really low probability of scoring when the Goalie also chooses not to move. You may be surprised how often the Kicker still scores even when the Goalie chooses the same direction. It is 65 percent when kicking LEFT and 56 percent when kicking RIGHT. It is also interesting that it is not symmetric. The probability of scoring when kicking LEFT is higher than the probability of scoring when kicking RIGHT. Kickers are a little more likely to kick LEFT than RIGHT.

__[Table 5.5](chapter5) Strategies and conditional probabilities for penalty kicks in the English Premier League in the 2016/2017 season. The first column is Kicker's strategy and the first row is the Goalie's strategy. The cells the probability that the Kicker scores given the action chosen by the Kicker and the Goalie.__
| q\_k     | G:LEFT | G:CENTER | G:RIGHT |      |
| -------- | ------ | -------- | ------- | ---- |
| q\_g     | 0.46   | 0.17     | 0.38    |      |
| K:LEFT   | 0.46   | 0.65     | 1.00    | 0.88 |
| K:CENTER | 0.17   | 1.00     | 0.00    | 0.89 |
| K:RIGHT  | 0.38   | 0.83     | 1.00    | 0.56 |

Right footed kickers may have a better chance of scoring when kicking to the LEFT. From a right footed kicker, the ball will naturally curve across and away from the Goalie. We can check to see if there are differences in payoffs and strategies between left and right footed kickers.

### 5.4.3 GMM Estimator

One issue with looking at differences by the footedness of the kicker is that the data is limited. A solution is to bring in more information from the game theory.

We know a few things that must be true in the game. In particular, in equilibrium, the strategy for the Kicker must make the Goalie indifferent between the choices. And the strategy for the Goalie must make the Kicker indifferent between the choices.

The Kicker and the Goalie choose _qk_ and _qg_, respectively, such that the following equalities hold. The periods in the subscripts mean that it represents any choice.

qk′p.l\=qk′p.c\=qk′p.rqg′pl.\=qg′pc.\=qg′pr.(5.5)

The vector _qk_ is laid down on its side and we use matrix multiplication rules to multiply it with the conditional probabilities when the Goalie chooses LEFT. These conditional probabilities are the first column of [Table 5.4](./13-chapter5.md#tbl5_4). The elements in _qk_ are multiplied with the corresponding elements in p.l and the three numbers are summed together to give the probability of a goal when the Kicker plays strategy _qk_ and the Goalie plays LEFT.

We will guess the matrix of strategies, _p_, then use equilibrium relationship to solve for _qk_ and _qg_. In addition, we directly observe _qk_, _qg_ and _p_ in the data. Don't we have too many conditions? Yes we do. Won't requiring that there is a mixed strategy Nash equilibrium lead to different estimates? Yes. Yes it will. The data combined with the game theory provide too many conditions and thus too many estimates. The solution is to average over the estimates. We are over identified. The generalized method of moments (GMM) algorithm provides a way to find that average. A method of moments estimator was used in [Chapter 3](./11-chapter3.md). Here we have multiple **moments**.[2](./13-chapter5.md#fn5_2)

### 5.4.4 GMM Estimator in **R**

The first step is to create functions f\_mixed() and f\_mixed\_int(). These functions are used to determine the equilibrium strategy of the Kicker and the Goalie given a set of conditional probabilities of scoring. Given a set of conditional probabilities p, optim() is used to determine the equilibrium strategies q\_k and q\_g. This is done by finding the strategies that minimize the sum of squared differences between payoffs of the three options for the other player.


`_>   fmixed = function(qk, qg, p) {_`
`_+     p = matrix(unlist(p), nrow=3)_`
`_+     pi1 = t(qk)%*%p[,1]_`
`_+     pi2 = t(qk)%*%p[,2]_`
`_+     pi3 = t(qk)%*%p[,3]_`
`_+_`
`_+    pi4 = t(qg)%*%p[1,]_`
`_+    pi5 = t(qg)%*%p[2,]_`
`_+    pi6 = t(qg)%*%p[3,]_`
`_+    return((pi1 - pi2)^2 + (pi1 - pi3)^2 +_`
`_+             (pi4 - pi5)^2 + (pi4 - pi6)^2)_`
`_+   }_`
`_>   fmixedint = function(par, p) {_`
`_+     qk = exp(par[1:3])/sum(exp(par[1:3]))_`
`_+     qg = exp(par[4:6])/sum(exp(par[4:6]))_`
`_+     return(fmixed(qk, qg, p))_`
`_+   }_`
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [2](./13-chapter5.md#fn5_25b)The term refers to a characteristic of the distribution such as the mean (the first moment) or the variance (related to the second moment). But here we use the term more generally and include characteristics of the equilibrium.

The function f\_penalty() takes in the proposed estimate of the conditional probabilities p and the observed actions and outcomes X. It then calculates 7 **moments**. For instance, it determines the difference between the estimate of the Kicker's probability of kicking LEFT against the observed probability that the Kicker kicked LEFT. It also compares the proposed estimate of the conditional probabilities to the actual conditional probabilities in the data. The function f\_penalty\_int() is an intermediate function called by optim(). It turns the parameter values into numbers between 0 and 1\. It then calls f\_gmm(), which is a generic function for calculating the GMM optimization problem.[3](./13-chapter5.md#fn5_3) The standard errors are calculated using the bootstrap.


`_>   fpenalty = function(p, X) {_`
`_+     N = dim(X)[1]_`
`_+     init = rep(0, 6)_`
`_+     p = matrix(p, nrow = 3)_`
`_+     q = optim(par = init, fn = fmixedint, p = p)_`
`_+     qk = exp(q$par[1:3])/sum(exp(q$par[1:3]))_`
`_+     qg = exp(q$par[4:6])/sum(exp(q$par[4:6]))_`
`_+     gk = gg = matrix(0, 3, N)_`
`_+     for(i in 1:3) {_`
`_+       gk[i,] = qk[i] - (X$kicker == action[i])_`
`_+       gg[i,] = qg[i] - (X$goalie == action[i])_`
`_+     }_`
`_+     gs = rep(0, N)_`
`_+     for(i in 1:3) {_`
`_+       for(j in 1:3) {_`
`_+         indexij = which(_`
`_+           X$kicker == action[i] & X$goalie == action[j]_`
`_+         )_`
`_+         gs[indexij] = p[i,j] -_`
`_+           (X$score[indexij] == “TRUE”)_`
`_+       }_`
`_+     }_`
`_+     G = rbind(gk_,`
`_+               gg_,`
`_+               gs)_`
`_+     return(G)_`
`_+   }_`
`_>   fpenaltyint = function(par, X) {_`
`_+     p = exp(par)/(1 + exp(par))_`
`_+     p = matrix(p, nrow = 3)_`
`_+     G = fpenalty(p, X)_`
`_+     return(fgmm(G, K = 7))_`
`_+   }_`
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [3](./13-chapter5.md#fn5_35b)The f\_gmm() function is available from the github site for the book.

### 5.4.5 Difference By Footedness

Does the behavior of the Kicker and the Goalie change depending on the which foot the Kicker generally kicks with? In general a right-footed Kicker will have an easier time scoring to the LEFT. The Kicker will generally strike the ball with their foot slightly to the right of center, that will put a counter-clockwise rotation on the ball which will then curve in the air from right to left.

So kicking LEFT for a right-footed Kicker will generally be better, holding the Goalie's strategy constant. Of course the Goalie gets a say in this. The Goalie may want to choose LEFT more often and thus be more likely to stop the ball when the right-footed Kicker chooses LEFT. For left-footed Kickers, the opposite is true.


`_>   dtR = dt[dt$Foot == “R”]_`
`_>   p = matrix(NA, 3, 3)_`
`_>   for(i in 1:3) {_`
`_+     for(j in 1:3) {_`
`_+       indexij = which(dtR$KickDirection == action[i] &_`
`_+                          dtR$KeeperDirection == action[j])_`
`_+       p[i,j] = mean(dtR$Scored[indexij] == “Scored”)_`
`_+     }_`
`_+   }_`
`_>   p[is.nan(p)] = NA_`
In the code the function is.nan() is used replace infinite values with missing values (NA). The matrix p gives the conditional probability that the Kicker scores given the Kicker is right footed.


`_>   X = data.frame(_`
`_+     “kicker” = dtR$KickDirection_,`
`_+     “goalie” = dtR$KeeperDirection_,`
`_+     “score” = dtR$Scored == “Scored”_`
`_+   )_`
`_>   epsilon = 1e-10_`
`_>   init = log(p + epsilon)_`
`_>   a = optim(par = init_,`
`_+             fn = fpenaltyint_,`
`_+             X = X_,`
`_+             control = list(trace = 0_,`
`_+                            maxit = 100000))_`
Similarly for Left footers.


`_>   dtL = dt[dt$Foot == “L”]_`
`_>   p = matrix(NA, 3, 3)_`
`_>   for(i in 1:3) {_`
`_+     for(j in 1:3) {_`
`_+       indexij = which(dtL$KickDirection==action[i] &_`
`_+                          dtL$KeeperDirection == action[j])_`
`_+       p[i,j] = mean(dtL$Scored[indexij] == “Scored”)_`
`_+     }_`
`_+   }_`
`_>   p[is.nan(p)] = 0_`
[Table 5.6](./13-chapter5.md#tbl5_6) presents the strategies given the footedness of the Kicker. Right-footed kickers will generally more accurate kicking LEFT and we see that they choose LEFT 53% of the time, while left-footed kickers choose RIGHT 52% of the time. Goalie's respond by playing LEFT to a right-footed kicker 54% of the time and RIGHT to a left-footed kickers 80% of the time!

__[Table 5.6](chapter5) Strategy estimates for Right and Left footed players. The first and fourth columns are the averages from the bootstrap for the two cases respectively. The top three rows are the strategies for the Kicker, while the bottom three rows are the strategies for the Goalie.__
| Right      | 0.05 | 0.95 | Left | 0.05 | 0.95 |      |
| ---------- | ---- | ---- | ---- | ---- | ---- | ---- |
| Kicker - L | 0.53 | 0.44 | 0.61 | 0.47 | 0.28 | 0.69 |
| Kicker - C | 0.19 | 0.13 | 0.26 | 0.01 | 0.00 | 0.06 |
| Kicker - R | 0.28 | 0.20 | 0.35 | 0.52 | 0.31 | 0.67 |
| Goalie - L | 0.54 | 0.44 | 0.62 | 0.19 | 0.03 | 0.36 |
| Goalie - C | 0.07 | 0.02 | 0.12 | 0.01 | 0.00 | 0.11 |
| Goalie - R | 0.39 | 0.31 | 0.48 | 0.80 | 0.64 | 0.95 |

The analysis suggests that the Goalie's strategy is completely different depending on the footedness of the kicker.

## 5.5 Multiple Equilibria

Another reason why we may see mixed strategies in a game is because there are multiple equilibria. We saw this in the coordination game. The two players would like to coordinate but it is unclear which choice they should coordinate on. In that situation a mixed strategy Nash equilibrium may be a more reasonable prediction of the outcome. Similarly, there was a coordination problem in our analysis of entry of mega bookstores. This section reconsiders the problem assuming that Borders and Barnes & Noble are playing a mixed strategy Nash equilibrium.

### 5.5.1 Coordination Game

Here is the normal form representation presented in [Chapter 2](./10-chapter2.md).

__Table 5.7 Normal form representation of a coordination game from [Chapter 2](./10-chapter2.md).__
| P1,P2 | BLACK | RED  |
| ----- | ----- | ---- |
| BLACK | 2, 5  | 0, 0 |
| RED   | 0, 0  | 5, 2 |

We use the same algorithm as for zero-sum games to find the MSNE. What is your guess on the equilibrium? Remember that we don't have the zero-sum game weirdness here.

Let _q_1 and _q_2, respectively, be the probability that Player 1 and Player 2 choose BLACK. What is the _q_2 that makes Player 1 indifferent between BLACK and RED?

2q2\=5(1−q2)7q2\=5q2\=57(5.6)

What is the _q_1 that makes Player 2 indifferent between BLACK and RED?

5q1\=2(1−q1)7q1\=2q1\=27(5.7)

Yes. Player 1 prefers RED and so weights their play to RED, while Player 2 prefers BLACK and weights their play to BLACK.

### 5.5.2 Bookstore Entry

[Chapter 4](./12-chapter4.md) presents a game describing entry of two mega bookstores, Barnes & Noble and Borders. In the game Barnes & Noble will enter market _i_ if the following inequality holds.

Xi′β1−D2iα1+ξ1i≥0(5.8)

where D2i∈{0,1} represents the choice to enter market _i_ of Borders. Under the assumptions of the equilibrium the value of D2i is known by Barnes & Noble when they make their entry decision. Both bookstores also know ξ1i, which is the value the econometrician doesn't observe.

In [Chapter 4](./12-chapter4.md) we showed that there is a case where Barnes & Noble will enter but only if Borders does not, and similarly Borders will enter but only if Barnes & Noble does not. At these values of the _ξ_s, we have a coordination game with multiple equilibria.

We can use the same algorithm to determine the mixed strategy Nash equilibrium. Let _q_1 be the probability that Barnes & Noble enters and _q_2 be the probability that Borders enters. What _q_2 makes Barnes & Noble indifferent between entering and not entering?

Xi′β1−q2iα1+ξ1i\=0orq2i\=Xi′β1+ξ1iα1(5.9)

Similarly we can solve for _q_1

q1i\=Xi′β2+ξ2iα1(5.10)

If we assume that Barnes & Noble and Borders are at a mixed strategy Nash equilibrium, then we no longer have a indeterminacy problem with our estimator. For every value of _ξ_1 and _ξ_2 we have a known probability over which outcome will occur.

### 5.5.3 MSNE Estimator in **R**

The estimator is pretty similar to the one used in [Chapter 4](./12-chapter4.md). There are a couple of differences with f\_entry\_mix(). First, the area where there is indeterminacy about which firm will enter, the mixed strategy Nash equilibrium determines what outcome will occur. Second, because there is no indeterminacy we can ask the probability that Firm 2 enters while Firm 1 does not.

To determine the mixed strategy Nash equilibrium we use Equations (5.10) and (5.9). The mixed strategy only occurs when the profits are such that it is only profitable for one of the two firms to enter. Because it is a mixed strategy, there is a possibility that any of the four outcomes occur. If there is an indeterminant outcome the mixed strategy Nash equilibrium determines the probabilities. In the code these are given by q\_1k and q\_2k. The probability of an indeterminant outcome is given by p\_ind.


`_> fentrymix = function(X, beta1, beta2, alpha1, alpha2_,`
`_+                        rho) {_`
`_+   N = dim(X)[1]_`
`_+   xi1 = Z1_`
`_+   xi2 = Z2*sqrt(1 - rho^2) + rho*Z1_`
`_+   Xb1 = X%*%beta1_`
`_+   Xb2 = X%*%beta2_`
`_+   p00 = p01 = p11 = rep(0, N)_`
`_+   for(k in 1:K) {_`
`_+     pi1k = Xb1 + xi1[k]_`
`_+     pi2k = Xb2 + xi2[k]_`
`_+     q1k = max(c(min(c((pi2k)/alpha2,1)),0))_`
`_+     q2k = max(c(min(c((pi1k)/alpha1,1)),0))_`
`_+     pind = (pi1k > 0 &_`
`_+                pi1k - alpha1 < 0 &_`
`_+                pi2k > 0 &_`
`_+                pi2k - alpha2 < 0)_`
`_+     p00 = p00 +_`
`_+       (pi1k < 0 & pi2k < 0) +_`
`_+       pind*(1 - q1k)*(1 - q2k)_`
`_+     p01 = p01 +_`
`_+       (pi1k < 0 & pi2k > 0) +_`
`_+       pind*(1 - q1k)*q2k_`
`_+     p11 = p11 +_`
`_+       (pi1k - alpha1 > 0 & pi2k - alpha2 > 0) +_`
`_+       pind*q1k*q2k_`
`_+   }_`
`_+   return(list(p00 = p00/K_,`
`_+               p01 = p01/K_,`
`_+               p11 = p11/K))_`
`_+ }_`
## 5.6 Empirical Analysis: Bookstore Entry with MSNE using **R**

The analysis presented in this section uses exactly the same data used in [Chapter 4](./12-chapter4.md). It also uses much of same empirical machinery.

[Table 5.8](./13-chapter5.md#tbl5_8) shows that the two different assumptions lead to similar results. Assuming that the outcome is a mixed strategy Nash equilibrium leads to an estimate that Borders is less impacted by competition with Barnes & Noble than the other way around. It also estimates less statistical dependence between the two stores.

__[Table 5.8](chapter5) Results from estimates of the game theory model from [Chapter 4](./12-chapter4.md) and the model assuming a mixed strategy Nash equilibrium. The two columns labeled “Pure” refer to the case where the entry decisions of the two firms are both strategically and statistically dependent, but we assume a pure strategy Nash equilibrium. The two columns labeled “Mix” refer to the same model but assuming the outcome is a mixed strategy Nash equilibrium.__
| Pure            | SD     | Mix  | SD     |      |
| --------------- | ------ | ---- | ------ | ---- |
| const\_1        | ‒15.11 | 0.25 | ‒14.82 | 0.20 |
| Pop\_1          | 1.07   | 0.01 | 1.05   | 0.02 |
| Income\_1       | ‒0.76  | 0.48 | ‒0.97  | 0.23 |
| College\_1      | 5.65   | 0.55 | 5.71   | 0.28 |
| Stores\_1990\_1 | 0.37   | 0.11 | 0.39   | 0.07 |
| const\_2        | ‒11.37 | 0.20 | ‒11.48 | 0.20 |
| Pop\_2          | 0.65   | 0.02 | 0.63   | 0.02 |
| Income\_2       | 1.31   | 0.80 | 1.91   | 0.38 |
| College\_2      | 2.70   | 0.55 | 2.82   | 0.39 |
| Stores\_1990\_2 | 0.79   | 0.11 | 0.77   | 0.10 |
| alpha\_1        | 0.73   | 0.18 | 0.70   | 0.21 |
| alpha\_2        | 0.70   | 0.12 | 0.56   | 0.15 |
| rho             | 0.47   | 0.10 | 0.30   | 0.10 |

[Table 5.9](./13-chapter5.md#tbl5_9) simulates the effect of a merger between Borders and Barnes & Noble on the willingness to have both brands in a market. The welfare impact of the merger is ambiguous. While it is the case that we see a reduction in competition. The merged firm is less like to have both brands in the market. In this way the merger leads to fewer brands in a market which reduces quality and the reduced head to head competition leads to higher prices. However, you see that there is a small reduction in the number of markets without a bookstore. The merger has the benefit of allowing the brands to coordinate their entry decision. While we have a game of complete information, the players cannot coordinate their choice in the mixed strategy Nash equilibrium. Because there is a possibility of having too many firms enter a market, firms reduce their willingness to enter some markets. The merger solves the coordination problem, reduces the possibility of having too many firms enter a market, and increases the willingness for the combined firm to enter some markets.

__[Table 5.9](chapter5) Comparison of actual entry to simulated entry in the year 2000 and simulated entry under a merger. These results are based on the assumption that Borders and Barnes & Noble are playing a mixed strategy Nash equilibrium__
| Actual  | Sim  | Merge |      |
| ------- | ---- | ----- | ---- |
| none    | 2919 | 2895  | 2881 |
| BN      | 155  | 135   | 176  |
| Borders | 15   | 54    | 72   |
| both    | 128  | 95    | 51   |

## 5.7 Discussion and Further Reading

Mixed strategy Nash equilibria are weird. Many people find them unintuitive and the algorithm for finding them is less than obvious. But for some games, they seem like the correct prediction.

While soccer penalty kicks may not represent the types of games you are interested in, it does provide an example of real people making real decisions with real consequences. Sporting contests provide researchers with access to large amounts of data on relatively simple strategic situations. This makes sports a good laboratory for testing game theory's predictions. [Adams (2020)](./25-refbib.md#ref1) presents an analysis of play choice in NFL games.

The chapter introduces one of the most important empirical methodologies in structural estimation, the **generalized method of moments** ([Hansen, 1982](./25-refbib.md#ref32)). Games often provides more moments than parameters to estimate, GMM provides a way to average over the moments.

The chapter also revisits the entry game analyzed in [Chapter 4](./12-chapter4.md). The entry game has multiple equilibria. This chapter assumes that the outcome is the result of a mixed strategy Nash equilibrium. It revisits the merger simulation under the mixed strategy Nash equilibrium assumption. As before the merger reduces the number of markets with both brands, but unlike [Chapter 4](./12-chapter4.md), the merger increases the number of markets with at least one firm.

[_OceanofPDF.com_](./https___oceanofpdf.com)
