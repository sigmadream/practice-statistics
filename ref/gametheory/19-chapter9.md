# 9Bayes Nash Equilibrium

DOI: [10.1201/b23262-9](./https___doi.org_10.1201_b23262-9.md)

## 9.1 Introduction

In the first two parts of the book, the games involved cases where all the players knew everything about what had happened or what was happening. We say that the players have complete information. In [Parts III](./18-part3.md) and [IV](./22-part4.md) of the book, we drop this assumption. The next two parts of the book consider games where players don't necessarily know who they are playing against, what their opponents payoffs are or what actions other players have played. In the first half the book, we could model chess. In this half of the book, we can model Stratego! If you don't know it, Stratego is a board game with pieces that have different attributes and abilities just like chess. The difference is that the pieces are all the exact same shape and the picture denoting the piece is only printed on one side. This means your opponent knows only that you have a piece in a particular square. They don't observe which piece. Stratego can be modeled as a dynamic game of incomplete information.

The problem with modeling games of incomplete information is that it is not clear how to do it. That we can model these situations is really due to some amazing work by an Hungarian mathematician, John Harsayni, who immigrated to Australia to escape the communists and then immigrated to the United States to escape the Australians.

Harsayni's insight was to think of games of incomplete information as games of complete information. Brilliant! What? Harsanyi realized that we could think of the problem of not knowing the other player's payoffs as a problem of not knowing the other player.

This chapter introduces the idea of **beliefs**. If something is unknown by the players, we need a way to quantify what is unknown. The assumption is that each player of the game places probability weights on the unknown events of the game. It is assumed that these probability weights (beliefs) are known to everyone in the game.

This chapter introduces the idea of a Bayesian game and the Nash equilibrium for such a game. It illustrates the idea using entry games. We saw entry games in [Chapters 4](./12-chapter4.md), [5](./13-chapter5.md), and [6](./15-chapter6.md). This time it is assumed that the firms contemplating entering a market do not know about the characteristics of the other firms contemplating entering the market. Interestingly, when players know about the characteristics of the other firms, the equilibrium is a lot more complicated than when players don't know about the characteristics of the other firms. Making the game more complicated makes the game easier to use for empirical analysis. Yale economist, Katja Seim makes this point in her 2006 _RAND Journal of Economics_ paper on the video retail industry ([Seim, 2006](./25-refbib.md#ref55)).

## 9.2 A Bayesian Game

We call a game with uncertainty over the types, a **Bayesian game** because we require players to use **Bayes rule** to update their **beliefs** (the probability weights). Although we are restricting the information available to players, we still require the players to process a lot of information. Newer game theory models relax some of these assumptions and explore the implications.

The section introduces a game based on an actual situation than can happen in undergraduate courses, it then formally defines the game and the equilibrium concept, **Bayes Nash equilibrium**.

### 9.2.1 A Grading Game

Let's analyze a game in which the players are students in a game theory class. Grading is done on a curve. Everyone who gets below the mean score for the class gets a B and everyone above the class mean gets an A. Assume that the students in the class are grade focused. They would prefer to only get As in their courses if they can help it. Also, students can drop the class at any time without penalty. This last bit is unrealistic, but it makes the analysis simpler.

For each student the problem is to determine whether or not to drop the class. Assume that each student observes their own raw score in the class. For example they may know that their grades add up to a grand total of 33 out of 100\. While they don't know raw scores of the other students, they have been told the distribution of raw scores. They can see that the mean is around 50 and scores range from close to 0 up to close to 90.

Assume that each student will stay in the class if they believe that they will get an A but will drop the class if they believe that they will get a B. To simplify the problem, we can assume that each student plays a **cutoff strategy**. A strategy here is a mapping from the player's raw score, to either stay or drop. A **cutoff strategy** states that the player will stay in the course if their score is above some cutoff level and will drop the class if it is below.

### 9.2.2 Grading Game Simulation using **R**

It is easier to look at a simulation. Assume we have 100 students in the class and their raw scores are determined by a normal distribution with a mean of 50 and a standard deviation of 20.


`_>   set.seed(123456789)_`
`_>   N = 100_`
`_>   score = rnorm(N, mean=50, sd=20)_`
`_>   summary(score)_`
`       Min. 1st Qu. Median     Mean 3rd Qu.    Max.`
`       4.41   36.61   52.73   50.50   64.98   89.80`
What will happen in this game? Let's assume that everyone plays a strategy where they never drop. In this case a student in the top half will get an A and a student in the bottom half will get a B.


`_> # Initial grades_`
`_> grade = ifelse(score > mean(score), “A”, “B”)_`
`_> table(grade)_`
`  grade`
`   A B`
`  54 46`
Assume that students play the strategy that if their scores is above 50 they will stay and students will drop if their score is below 50\. That is, if the student thinks they will get an A, they stay in the class. If the student thinks that they will get a B, they will drop the class.

Now, if this is the strategy played by all the players, the grades will be different. Remember the grades are curved based on students who are in the class. The score distribution for those students will be a truncated normal distribution (the top half).


`_> s = score > 50_`
`_> score1 = score[s]_`
`_> summary(score1)_`
`     Min. 1st Qu. Median     Mean 3rd Qu.    Max.`
`    50.18   56.89   64.37   64.95   73.29   89.80`
`_> length(score1)_`
`  [1] 55`
If students play this strategy, then the mean jumps from 50 to 65 and 45 students drop the class. Given the new reality, the grades will be adjusted.


`_> # New grades_`
`_> grade1 = ifelse(score1 > mean(score1), “A”, “B”)_`
`_> table(grade1)_`
`  grade_1`
`   A B`
`  25 30`
`_> mean(score1)_`
`  [1] 64.9546`
The number of students who get an A drops to 25 and they have to have a grade above 65.

Assume that the students adjust to this new reality, so their strategy changes. They will stay if their grade is above 65 as this guarantees them an A for the class.


`_> s1 = score > 65_`
`_> score2 = score[s1]_`
`_> summary(score2)_`
`     Min. 1st Qu. Median     Mean 3rd Qu.    Max.`
`    65.36   71.05   73.49   74.06   78.25   89.80`
Now the remaining students need to update their beliefs about their grade given the strategies of the other students. Given this update, another 30 students drop the class and the mean increases to 74.


`_> grade2 = ifelse(score2 > mean(score2), “A”, “B”)_`
`_> table(grade2)_`
`  grade_2`
`   A B`
`  10 15`
`_> s2 = score > 74_`
`_> score3 = score[s2]_`
`_> summary(score3)_`
`     Min. 1st Qu. Median     Mean 3rd Qu.    Max.`
`    74.39   76.74   78.58   79.56   79.56   89.80`
Updating again, another 15 students drop the class and the average grade increases to 80.

Will the class have any students in it?

Dartmouth College has a provision to stop this unraveling. While Dartmouth's econ department uses a curve, professors are allowed to include students who dropped the class in calculating the grade distribution, where such students are given the lowest score for the purposes of doing the calculation. A policy like this may still lead students to drop based on beliefs of their grade in the class, but it doesn't lead to the unraveling. What is the outcome of the game used by Dartmouth?

### 9.2.3 Definitions

Now we have a taste for how these games work, let's get more formal. What do we mean by a **game of incomplete information**?

Definition 15. _In a_ game of incomplete information, _players don't necessarily observe the actions of other players or know the payoffs of the other players_.

In [Parts I](./08-part1.md) and [II](./14-part2.md) of the book, players know exactly what is happening or has happened at every moment. In [Parts III](./18-part3.md) and [IV](./22-part4.md) of the book, they don't.

[Assumption 2.](chapter9) _There are a set of player types determining the payoffs the player will get in each outcome. This set of types are known to all the players_.

While players don't know exactly what is going on, we will make the assumption that they do know player **types**. [Assumption 2](./19-chapter9.md#assu9_1) is a super important idea, it allows us use all the machinery we have developed for analyzing complete information games to analyze incomplete information games. The player's type captures all the information relevant to the game. If each player's type is observed by every other player of the game, then the game is one of complete information.

[Assumption 3.](chapter9) _Each player knows their own type and the distribution of types (the probability that the other player is a particular type)_.

What makes games of incomplete information hard to think about is the implications of [Assumption 3](./19-chapter9.md#assu9_2). When analyzing the game, we must be careful to remember what exactly the players know and what they do not know.

### 9.2.4 The Game

A Bayesian game has the following form.

* Players: Set of players and a set of types.
* Strategies: A function mapping from type to actions.
* Payoffs: For each type and each outcome a payoff.
* Beliefs: A probability distribution over types that is known by each player.

Our strategies are more complicated than for static games of complete information. Again, a strategy is complete plan. In this case, the complete plan is determined prior to the player knowing their type. That is, the plan states what the player will do for each possible type that they could be.

We have added one more piece to the basic game description. We generally refer to this known probability distribution over types as **beliefs**. Players do not know the types of the other players in the game, but they do know probability that another player is a particular type.

### 9.2.5 Course Grading Game

For the game introduced above, we have the following formal description. In this game, the player's type is the raw score of the player in the class which is denoted _θi_. The raw score is between 0 and 100\. The player's strategy states whether the player will stay or leave based on the raw score they observe.

* Players: N\=100 students where each observes their score in the class, θi∈\[0,100\].
* Strategies: Each player i∈N chooses s(θi)∈{0,1}, where 1 means stay.
* Payoffs:  
   1. Stay and θi\>m: _A_  
   2. Stay and θi≤m: _B_  
Leave: 0, where B<0<A  
where _m_ is the mean grade of the students remaining in the class.
* Beliefs: θi∼F

When we write down the game we don't worry about whether there exists an equilibrium of the game. It is also going to turn out that the initial beliefs don't matter that much, so let's just call it _F_.

### 9.2.6 Equilibrium

Given our new set up, we need a new equilibrium concept.

Definition 16. _A_ Bayes Nash equilibrium _is an outcome where for each type, the outcome cannot be improved upon given the strategies of the other players and beliefs about the distribution of types. Where beliefs are consistent with equilibrium strategies_.

First, we are still assuming Nash equilibrium. Strategies must be optimal given the strategies of the other players. What is new is this idea of beliefs.

Definition 17. _A player's belief is what the player knows about the distribution of types playing each strategy_.

The equilibrium concept requires that the beliefs of the players be consistent with the equilibrium strategies. We are assuming that players are choosing their optimal strategies given expected payoffs. The players may not know exactly what payoff will occur because they don't know exactly which type of player they are playing against.

Here it is important to point out the difference between assumptions about the game and assumptions about the predictions of the game (the equilibrium concept). In the game, the players only know their own type and the distribution of types for the other players. In a Bayes Nash equilibrium, it may be that players know exactly the type of the other players because their beliefs must be updated to be consistent with the equilibrium strategies. Hopefully, this distinction will become clearer as we work through examples.

### 9.2.7 Equilibrium of Course Grade Game

Is there an equilibrium where each student only stays in the class if their grade is an _A_? No.

We are going to show this using a proof by contradiction. Let _c_ be the cutoff such that if θi\>c, then student _i_ stays; otherwise, they drop. If _c_ is the same for every student in the class then _c_ is the minimum grade for the students that stay in the class. Given the letter grades are determined by the mean grade of students that stay in the class, then m\=E(θi|θi\>c). The curve requires that _A_ is given if θi\>m and _B_ is given otherwise.

The proposed equilibrium requires that m≤c. If _c_ and _m_ exists, then m\=E(θi|θi\>c)\>c, a contradiction.

## 9.3 Empirical Entry Game

We first looked at entry games in [Chapter 4](./12-chapter4.md). Those were static games of complete information. We revisited entry games in [Chapter 5](./13-chapter5.md) and again in [Chapter 6](./15-chapter6.md). The last time we modeled them as dynamic games of complete information. Now we are going to revisit them again. This section models entry games as static games of incomplete information. Comparing our analysis in this chapter to the analysis in [Chapters 4](./12-chapter4.md) and [5](./13-chapter5.md), the assumptions in this chapter make the game itself more complicated, but the equilibrium easier to analyze.

It is worth contemplating the difference between the assumption made here and in [Chapter 5](./13-chapter5.md). In both cases in equilibrium the firms don't know exactly if the other firm will enter the market. In both cases, they know the probability that the other firm will enter. These models are different. In the model used in [Chapter 5](./13-chapter5.md), each firm knows the value of the unobserved entry costs, _ξ_. The econometrician does not know this value, but the firms playing the game do. Here, the value of _ξ_ is unknown to both the econometrician and some of the players of the game. This is a subtle distinction but it substantially changes how we estimate the model.

### 9.3.1 The Game

* Players: Barnes & Noble (and ξ1i), Borders (and ξ2i)
* Strategies:  
   1. Given ξ1i (and Xi) Barnes & Noble chooses enter or not enter.  
   2. Given ξ2i (and Xi) Borders chooses to enter or not enter.
* Payoffs:  
   1. Barnes & Noble  
         1. \* Enter: Xi′β1−α1Pr(D2i\=1|ξ1i)+ξ1i  
         2. \* Not Enter: 0  
   2. Borders  
         1. \* Enter: Xi′β2−α2Pr(D1i\=1|ξ2i)+ξ2i  
         2. \* Not Enter: 0  
   3. Beliefs: {ξ1i,ξ2i}∼Φ2(0,Σ), where Σ\=\[1ρρ1\].

where D1i indicates whether or not Barnes & Noble enters, while D2i indicates Borders entry.

This game is similar to the one we used in [Chapter 4](./12-chapter4.md). There are unobserved characteristics of the firms and the market, _ξ_, that are now unobserved by the other player. These are the player types. Entry is determined by observed characteristics Xi of the market _i_ and the parameter _β_ which is the same across markets. The parameter _α_ determines the extent to which competition from the other firm reduces the benefits of entry into a particular market.

### 9.3.2 Equilibrium

The difference between this game and what we saw in [Chapter 4](./12-chapter4.md) is the assumption about what Barnes & Noble knows about _D_2 in equilibrium. In [Chapter 4](./12-chapter4.md) we assumed that in equilibrium, Barnes & Noble knew the exact value of _D_2. They knew whether or not Borders was also entering the market. In this chapter we assume that in equilibrium Barnes & Noble only knows the probability that D2\=1. More precisely, in equilibrium Barnes & Noble accounts for Border's equilibrium strategy in Barnes & Noble's beliefs about _D_2. It's value _D_2 is determined by the following inequality.

Xi′β2−α2D1i+ξ2i\>0(9.1)

Assume that there exists an equilibrium in cutoff strategies, where the cutoff values are {c1i,c2i}, respectively. Barnes & Noble's expectation about Border's entry into market _i_ is as follows.

E(D2i|Xi,ξ1i)\=Pr(ξ2i\>c2i|Xi,ξ1i)\=Pr(ξ2i\>−Xi′β2+αPr(ξ1i\>c1i)|ξ1i)(9.2)

In determining the probability that Borders will enter, Barnes & Noble must account for Border's beliefs that Barnes & Noble will enter. Things are a lot simpler if we assume that the unobserved term, the _ξ_'s are independent of each other. That seems pretty unrealistic. We would expect a lot of things about the market to be correlated. While Barnes & Noble doesn't know Border's costs of entering exactly, it does observe its own costs and can make an inference about Border's costs.[1](./19-chapter9.md#fn9_1)

Even still, there is a distinct advantage of using games of incomplete information to model entry. In [Chapter 4](./12-chapter4.md), there were multiple equilibria at certain values of the _ξ_s. That is not the case here. The equilibrium is unique in the cutoff strategies of the firms. The only issue is determining what that equilibrium is!

### 9.3.3 Estimating Entry Games

Can we back out the distribution of entry costs by looking at the distribution firms in different markets? We can. However, we must solve for the equilibrium in order to do so. It is assumed that the entry costs are distributed bivariate normal. The question is whether we can estimate the correlation coefficient (_ρ_). In a standard bivariate probit model, it is assumed that the actions are correlated but the optimality of the actions only depend on each other through the correlation. Here the actions themselves are interdependent.

The seminal work of [Guerre et al. (2000)](./25-refbib.md#ref29) suggests that it is not necessary to actually solve for the equilibrium. Instead it is simpler to do a “two-step” procedure and the correlation parameter.

In the first step, we use maximum likelihood to estimate the cutoff value. In this step, we are not making any claims about the parameter values we are estimating. We are assuming that the observed data is being generated by some sort of stationary process. As with any discrete choice type data, our identification is heavily reliant on parametric assumptions. The likelihood function is determined by the standard bivariate normal. The probabilities of the four states are as follows. Note the notation. There is a squiggly line over all the parameters to remind us that these are not the structural parameters but the parameter values coming from the **reduced form estimation** in the first step.[2](./19-chapter9.md#fn9_2)

Maximizing the log-likelihood function provides estimates of β\~1, β\~2, and ρ\~. The model for estimating these parameters is presented in [Chapter 4](./12-chapter4.md).

\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ [1](./19-chapter9.md#fn9_19b)To be clear, we assume that the firms each know all the observed characteristics, Xi as well as all the parameter values.

[2](./19-chapter9.md#fn9_29b)We use the term reduced form to refer to standard empirical estimation techniques where we are not relying on assumptions about actors generating the data are behaving.

With this information, we can determine probability of entry for each firm. These probabilities are estimated because they are based on the parameters estimates in the first step.

Pr(D1i\=1|Xi,ξ2i)\=1−Φ(−Xiβ\~^1−ρ\~^ξ2i(1−ρ\~^2))Pr(D2i\=1|Xi,ξ1i)\=1−Φ(−Xiβ\~^2−ρ\~^ξ1i(1−ρ\~^2))(9.3)

Given the estimated probabilities of entry, we can estimate our structural parameters.

The probabilities are now without the squiggly lines. This is a reminder that these probabilities are structural estimates. They come from the game theory model. The probabilities with the squiggly lines are observed in the data and don't depend upon any assumptions about how the data are generated.

We also have two extra parameters, _α_1 and _α_2. Both firms will enter if the following inequality holds. The _ξ_s appears in two different places in the inequalities.

Xiβ1−α1Pr^(D2i\=1|Xi,ξ1i)+ξ1i\>0Xiβ2−α2Pr^(D1i\=1|Xi,ξ2i)+ξ2i\>0(9.4)

where {ξ1i,ξ2i}∼N2(0,Σ) and Σ\=\[1ρρ1\].

### 9.3.4 Estimating Entry Games in **R**

Similarly, we can write out a function to represent the entry inequalities for the second stage of the estimation (Equation (9.4)). In the code, the extra t denotes the parameters estimated from the reduced form model (the squiggly line parameters). [Chapter 4](./12-chapter4.md) estimates these parameters using f\_entry2().


`_> fbiprobit = function(X, beta1, beta2_,`
`_+                        alpha1, alpha2, rho_,`
`_+                        beta1t, beta2t, rhot) {_`
`_+   N = dim(X)[1]_`
`_+   xi1 = Z1_`
`_+   xi2 = Z2*sqrt(1 - rho^2) + rho*Z1_`
`_+   Xb1 = X%*%beta1_`
`_+   Xb2 = X%*%beta2_`
`_+   Xb1t = X%*%beta1t_`
`_+   Xb2t = X%*%beta2t_`
`_+   p00 = p01 = p11 = rep(0, N)_`
`_+   for(k in 1:K) {_`
`_+     D1k = 1 -_`
`_+       pnorm(0, Xb1t + rhot*xi1[k], sqrt(1 - rhot^2))_`
`_+     D2k = 1 -_`
`_+       pnorm(0, Xb2t + rhot*xi2[k], sqrt(1 - rhot^2))_`
`_+     pi1k = Xb1 - alpha1*D2k + xi1[k]_`
`_+     pi2k = Xb2 - alpha2*D1k + xi2[k]_`
`_+     p00 = p00 + (pi1k < 0 & pi2k < 0)_`
`_+     p01 = p01 + (pi1k < 0 & pi2k > 0)_`
`_+     p11 = p11 + (pi1k > 0 & pi2k > 0)_`
`_+   }_`
`_+   return(list(p00 = p00/K_,`
`_+               p01 = p01/K_,`
`_+               p11 = p11/K))_`
`_+ }_`
The second step estimator, f\_biprobit, uses simulation to calculate the probabilities. The code uses pnorm() to calculate the probability of entry given the estimated parameters from the first step.

## 9.4 Empirical Analysis: Mega Bookstore Entry (Again) using **R**

This section re-estimates the entry of Barnes & Noble and Borders under the assumption that each bookstore chain does not know the exact costs of entry of their competitor. This assumption seems more realistic than the assumption made in [Chapter 4](./12-chapter4.md). While the assumption makes the estimator somewhat more complicated, it has the nice property that there aren't multiple equilibria.

### 9.4.1 Estimates

[Table 9.1](./19-chapter9.md#tbl9_1) presents the mean and standard deviation of the coefficient estimates from the first and second stages of the two-step estimator. The estimates show that population, college education and the number of bookstores in 1990 are all important determinants of entry into a county by these two mega bookstore chains. It also shows that Barnes & Noble is much less likely to enter than Borders and that the two firms do not like to compete. It is interesting to compare these results to the results presented in [Chapter 4](./12-chapter4.md), [5](./13-chapter5.md) and [6](./15-chapter6.md).

__[Table 9.1](chapter9) Results from estimates of the first and second stage of the estimates assuming a Bayes Nash equilibrium. The first set of columns labeled “First Stage” are estimates assuming that the two firms are making entry decisions that are strategically independent but statistically independent. These are the same as the results presented in [Chapter 4](./12-chapter4.md) for the same model. The second set of columns labeled “Second Stage” assumes the firm entry decisions are strategically dependent, but each firm does not know exactly if the other firm will enter, but assumes their strategy is consistent with observed entry decisions.__
| First Stage     | SD      | BNE  | SD      |      |
| --------------- | ------- | ---- | ------- | ---- |
| const\_1        | \-15.03 | 0.09 | \-15.03 | 0.23 |
| Pop\_1          | 1.09    | 0.01 | 1.06    | 0.02 |
| Income\_1       | \-1.04  | 0.25 | \-0.75  | 0.36 |
| College\_1      | 5.51    | 0.34 | 5.85    | 0.40 |
| Stores\_1990\_1 | 0.28    | 0.07 | 0.44    | 0.09 |
| const\_2        | \-11.54 | 0.11 | \-11.39 | 0.17 |
| Pop\_2          | 0.66    | 0.01 | 0.63    | 0.02 |
| Income\_2       | 1.74    | 0.25 | 1.80    | 0.39 |
| College\_2      | 2.59    | 0.33 | 2.75    | 0.54 |
| Stores\_1990\_2 | 0.49    | 0.06 | 0.70    | 0.10 |
| alpha\_1        | 0.87    | 0.22 |         |      |
| alpha\_2        | 0.64    | 0.17 |         |      |
| rho             | \-0.08  | 0.10 | \-0.01  | 0.11 |

[Table 9.1](./19-chapter9.md#tbl9_1) presents the model fit excercise comparing the two models with the actual data. The Bayes Nash equilibrium model does a better job of predicting the case where there is only one firm in the market, but does a worse job of predicting the case where there are two firms in the market.

### 9.4.2 Policy

Given our new estimates, we can reconsider the policy question analyzed in [Chapter 4](./12-chapter4.md). In that analysis, the merger resolved a coordination problem for the firms. In general, the two firms prefer to have only one of them in the market, but when the firms are independent they cannot coordinate on which one. In [Chapter 4](./12-chapter4.md), the firms knew whether or not the other firm was going to enter, under these model assumptions they do not. The merger resolves the information problem inherent in the modeling assumptions.

[Table 9.2](./19-chapter9.md#tbl9_2) presents the results of simulating the merger. We see fewer cases where the two firms compete as well as fewer markets with entry. Theoretically this model can have ambiguous predictions about the impact of the merger on consumer welfare, but the empirical estimates suggest that the merger would have lowered consumer welfare.

__[Table 9.2](chapter9) Comparison of actual entry in 2000 compared to simulated entry and simulated entry under a merger assuming a Bayes Nash equilibrium of the entry game.__
| Actual  | Sim  | Merger |      |
| ------- | ---- | ------ | ---- |
| none    | 2919 | 2643   | 2857 |
| BN      | 155  | 284    | 222  |
| Borders | 15   | 148    | 71   |
| both    | 128  | 104    | 29   |

Compare these predictions to the predictions in [Chapter 4](./12-chapter4.md). It is the same policy analysis. What is different is the assumption about the information that the competitors have.

## 9.5 Discussion and Further Reading

Modeling entry games as games of complete information leads to all sorts of weirdness [(Bresnahan and Reiss, 1990](./25-refbib.md#ref15); [Tamer, 2003](./25-refbib.md#ref54)). However, when the game is more realistic by reducing the information available to the players, the game gets a whole lot simpler to analyze. [Seim (2006)](./25-refbib.md#ref55) uses these ideas to analyze entry of retail video stores.

The estimator here is based on [Guerre et al. (2000)](./25-refbib.md#ref29). The argument is that we can assume that the observed data are the result of equilibrium behavior. We can estimate a reduced form model in the first step and then impose equilibrium behavioral assumptions to back out the parameters of the underlying game theory model. We come back to this idea in [Chapter 10](./20-chapter10.md).

This chapter uses data from [Adams and Basker (2025)](./25-refbib.md#ref2) and their analysis of entry of the mega bookstores.

[_OceanofPDF.com_](./https___oceanofpdf.com)
