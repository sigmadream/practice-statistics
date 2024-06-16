# 13Adverse Selection

DOI: [10.1201/b23262-13](./https___doi.org_10.1201_b23262-13.md)

## 13.1 Introduction

This chapter considers **adverse selection** problems. These games are generally split between **signalling** and **screening** games. In a **signalling game**, the player with an unknown type chooses an action and that action choice may provide information to the other player. In a **screening game**, the uninformed player goes first and offers some choices to the player whose type is not known. The observed choice may provide information to the uninformed player.

In either case, the question is whether the observed action provides information about the player's type. We may have equilibria in which the action provides information about the player's type, we call these **separating equilibria**. We also may have equilibria where the action does not provide information about the player's type. We call these **pooling equilibria**.

The chapter presents the most famous game of unknown types, George Akerlof's model a used car market and the lemons problem. It then uses these ideas to think about the market for health insurance. It considers whether government subsidies or taxes can be used to solve the lemons problem in health insurance. The chapter analyzes these policies with a model calibrated using data from the Medical Expenditure Panel Survey (MEPS).

## 13.2 Akerlof's Lemons

Outside economics, there is a perception that economists are gungho believers in free market capitalism. For economists that study markets and how markets work or don't work, it is odd to be called “believers.” George Ackerlof is an economist through and through, he even won the Nobel prize in economics. He is also married to one of the most influential economists in the world, former Treasury Secretary Janet Yellen. Akerlof showed how a simple misallocation of information in a market, would cause that market to fail.

This section introduces Akerlof's lemons market and provides a formal game-theoretic version of the problem.

### 13.2.1 The Lemons Market

Consider a car market. In this market, there are two types of cars, good cars and lemons (bad cars). Unfortunately, consumers cannot observe the type of car prior to purchase. They can, however, resell the car on the used car market after observing the car's type.

Now consider a bunch of new cars become available. The key thing about new cars is that the seller of the cars will sell the car no matter if it is good or a lemon. Consumers buy the car and learn the car's type. After the consumers learn the car's type they can sell the car on the used market or keep it. If prices in the used car market are roughly similar to prices in the new car market, then those customers who purchased lemons will dump them on the used car market and go back to the new car market hoping to get a good car.

Customers on the used car market are going to notice that the proportion of lemons just got a lot higher and so the price in the used car market will fall. Where previously, people that owned good cars may have been willing to sell their car on the used car market, the price dropped because of all the lemons entering the market. So these sellers are going to keep their good cars. With owners of lemons dumping them and owners of good cars keeping them, the used car market becomes increasingly filled with lemons. In response, the price in the used car markets drops. This only exacerbates things. As the price falls more, the only cars that are profitable sell in the used car market are lemons. Finally, the used car market is just lemons.

The market fails.

### 13.2.2 Lemons Game

Akerlof's original argument was not game theory. We can reorient it as a game.

* Players: Seller of car type Bad (b) or Good (g), Buyer
* Strategies:  
   1. Buyer: Offer price, _p_.  
   2. Nature: Determine car type, {b,g}  
   3. Seller: Observe car type and offered price, then Sell or Keep car.
* Payoffs:  
   1. Seller: Sell: E(p|Sell), Keep: _V_  
   2. Buyer: E(V|Sell,p)−p  
   3. V∈{b,g}
* Beliefs: Pr(V\=g)\=π

The Buyer offers a price for the used car. The Seller observe her type, actually the car's type. Given that observation and the observed price offered by the Buyer, the seller chooses to sell or keep the car. Again it is important to separate out what happens in the game and what happens in equilibrium. In equilibrium, the Buyer's price needs to adjust for the strategy of the Seller.

### 13.2.3 Bayes Nash Equilibrium

There is no Bayes Nash equilibrium in which there is a transaction of good cars.

Remember when determining a Bayes Nash equilibrium, we need to update the player's beliefs about the other player's type. Another way to say it, is that beliefs have to be consistent with player strategies in equilibrium.

Consider the case where only _b_ type sellers sell. In this case, the Buyer updates his beliefs on the type of Seller that sells a car on the used car market. Given these updated beliefs, the Buyer will offer a price of _b_. Bad car sellers will sell, but good car sellers won't. This is an equilibrium. A Seller with car of type _b_ is willing to sell at price _b_. A Seller of car type _g_, is not willing to sell at price _b_. The Buyer believes that all cars being sold are type _b_. The buyer offers _b_. We have a **separating equilibrium**.

What if both types of Sellers sell on the market. Both types sell, so the Buyer does not update their beliefs. In this case, the price offered by the buyer will be πg+(1−π)b where _π_ is the probability that the Seller is has a good car. As this price is greater than _b_, sellers of cars of type _b_ will also sell. The price offered is πg+(1−π)b<g. Because this low price is below the value of the good car, the good-type seller will not want to sell. That is to say, this is not an equilibrium. There is no **pooling equilibrium**.

In the Bayes Nash equilibrium of the lemon's market, all cars sold in the used car market are lemons. Is this an accurate representation of the actual used car market? What mechanisms exist that encourage sellers of good cars to sell in the used car market?

## 13.3 Insurance

The most well-known example of **adverse selection** is in the context of insurance. Insurance is the idea that two agents can trade because they have different risk preferences. In actual fact, the insurance company is able to be less risk-averse because it is able to diversify across a large number of bets.

This section presents a model of insurance where buyers of insurance are risk-averse. It then considers what happens when buyers are of different unobserved types. Some buyers tend to be sicker. These buyers will value insurance more. There will be a tendency for the market to behave like Akerlof's lemons market. The sicker types buying insurance and the healthy types not buying insurance. The question is what types of policies could we implement to encourage healthier people to buy insurance.

### 13.3.1 Model

Assume there are two possible states of the world, sick and healthy. In the healthy state, an individual earns a certain income _y_, but in the sick state, the individual earns _y_ and pays _h_ in costs. These may be medical expenses or loss of income, etc. The individual's utility in the two states are u(y) and u(y−h). Importantly, the individual is better off receiving the expected outcome rather than expected utility over the two outcomes. That is the following inequality holds.

u(πy+(1−π)(y−h))\=u(y−(1−π)h)\>πu(y)+(1−π)u(y−h)(13.1)

where (1−π) is the probability that the sick state occurs for this individual.

If the inequality holds, we say that the individual is risk-averse. Mathematically, their utility function is concave.

Because of the higher utility of the expected value than the separate states, the individual is willing to buy a product that pays less in the healthy state but more in the sick state.

Consider a product that pays _h_ in the sick state at a cost (premium) of _p_ that is paid in every state.

πu(y−p)+(1−π)u(y−h+h−p)\=u(y−p)(13.2)

If the premium is low enough, then the individual will prefer to be insured. If p\=(1−π)h+ϵ, then the individual strictly prefers insurance from Equation (13.1) (if _ϵ_ is small enough).

For the insurance company, they would like to offer the product if p−(1−π)h≥0. So there exists a positive _ϵ_ where the insurance company finds it profitable to offer insurance and the individual is willing to purchase the insurance.

### 13.3.2 Game with Unknown Types

So in a world with perfect information (albeit with uncertainty), it is profitable for insurance companies to offer insurance. What if the riskiness of the person is not observed by the insurer? We have two types π∈{πl,πh}, where _πl_ has a much higher probability of being in the sick state. The individual gets to observe their type prior to buying insurance, while the insurance company does not get to observe the individual's type.

In this situation, while both types may like to buy insurance, it is difficult to offer insurance to people with a low probability of being sick. This is basically the same problem as Akerlof's lemons market.

* Players: Buyer of type (Low (_πl_), High (_πh_)), Insurer
* Strategies:  
   1. Insurer: Offer a contract that pays _h_ in the sick state at a premium _p_ (which paid in all states).  
   2. Nature: Choose Buyer type {πl,πh}.  
   3. Buyer: Buy (or not) insurance at the offered premium after observing their type.
* Payoffs:  
   1. Buyer: Buy: u(y−p), Don't Buy: πu(y)+(1−π)u(y−h)  
   2. Insurer: Buy: p−(1−π)h, Don't Buy: 0  
   3. where π∈{πl,πh}
* Beliefs: Pr(π\=πl)\=q

### 13.3.3 Bayes Nash Equilibrium

Consider the case where both types are willing to buy insurance and let _q_ be the proportion of “sick” individuals. In this case, the insurer will want the premium to be such that p−(1−qπl−(1−q)πh)h≥0. For the “sick” individual, they will purchase if the following inequality holds.

u(y−p)≥πlu(y)+(1−πl)u(y−h)(13.3)

Similarly, for the healthy individual.

Is there a Bayes Nash equilibrium of this game? Let use simulation to see what happens.

### 13.3.4 Simulation using **R**

Consider the following simulated game. The buyers are risk averse, with a constant absolute risk-aversion utility function, with the parameter _r_ determining the extent of their aversion.

u(C)\=C1−r1−r(13.4)

This is a relatively simple utility function that is used a lot in the literature.

Healthy individuals only get sick 1% of the time, while sick individuals get sick 10% of the time. The economy has a population where 90% are the healthy type. If an individual gets sick, they lose half of their income.

One difference between the simulation and the set up of the game above is that premiums are assumed to be set at the actuarially fair rate. This is equivalent to saying that the Insurer makes zero profits which is equivalent to saying that there is perfect competition in the insurance market.


`_> uf = function(C, r) {(C^(1 - r))/(1 - r)}_`
`_> Euf = function(r, x, p) sum(p*uf(x, r))_`
`_> premf = function(pi, h) (1 - pi)*h_`
The function u\_f() is the utility function with constant absolute risk aversion. The function Eu\_f() is the Agent's expected utility and prem\_f() calculates the actuarially fair premium.

The parameters for the simulation are as follows. There is no particular rhyme or reason for the choice. The sick type has a 10 percent probability of being in the sick state, while the health type has a 0.1 percent probability of being in the sick state.


`_>   r = 0.7_`
`_>   pil = 0.9_`
`_>   pih = 0.999_`
`_>   q = 0.9_`
`_>   y = 1_`
`_>   h = 0.5_`
Utility for the uninsured for each type of individual.


`_> Euf(r, c(y, y-h), c(pil, 1 - pil))_`
`  [1] 3.270751`
`_> Euf(r, c(y, y-h), c(pih, 1 - pih))_`
`  [1] 3.332708`
Assume we have a **pooling equilibrium**. That is, both types purchase insurance.


`_> prem = premf(q*pil + (1 - q)*pih, h)_`
Given this premium of half of one percent of income, would both types purchase insurance?


`_> Euf(r, c(y - prem, y - prem)_,`
`_+      c(pil, 1 - pil)) >_`
`_+   Euf(r, c(y, y-h), c(pil, 1 - pil))_`
`  [1] TRUE`
`_> Euf(r, c(y - prem, y - prem)_,`
`_+      c(pih, 1 - pih)) >_`
`_+   Euf(r, c(y, y-h), c(pih, 1 - pih))_`
`  [1] FALSE`
No. The healthy type is not willing to purchase insurance at that premium.

Is there an equilibrium where only the sick types are insured? Is there a **separating equilibrium**?


`_> prem = premf(pil, h)_`
`_> Euf(r, c(y - prem, y - prem)_,`
`_+      c(pil, 1 - pil)) >_`
`_+   Euf(r, c(y, y-h), c(pil, 1 - pil))_`
`  [1] TRUE`
`_> prem_`
`  [1] 0.05`
Yes. The sick individual is willing to pay a premium of 5% of her income. But 90% of the population is uninsured.

### 13.3.5 Mandatory Insurance

In the simulation above, only 10% of the population purchases insurance and the premiums are very high.

One solution to this problem is to make insurance mandatory. Or more accurately have some sort of fine or tax for those that don't take up insurance. That is, the payoff to the individual is decreased by the amount of the tax when the individual chooses not to purchase health insurance.


`_> tax = 0.05_`
`_> prem = premf(q*pil + (1 - q)*pih, h)_`
`_> Euf(r, c(y - prem, y - prem)_,`
`_+      c(pil, 1 - pil)) >_`
`_+   Euf(r, c(y - tax, y-h - tax)_,`
`_+        c(pil, 1 - pil))_`
`  [1] TRUE`
`_> Euf(r, c(y - prem, y - prem)_,`
`_+      c(pih, 1 - pih)) >_`
`_+   Euf(r, c(y - tax, y-h - tax)_,`
`_+        c(pih, 1 - pih))_`
`  [1] TRUE`
`_> prem_`
`  [1] 0.04505`
By adding a tax of 5%, we make the non-insurance expected utility lower, so the high type is willing to pay a higher premium to be insured. This allows risks to be pooled and makes it profitable for the insurance company to offer insurance that the healthy individuals are willing to accept.

Under this policy, the sick types do very well. Their premiums drop from 5% of income to 4.5% of income.

## 13.4 Empirical Analysis: Health Insurance using **R**

One of the concerns policy makers have with the health insurance market is that many people do not carry insurance. People who are generally healthy don't carry insurance. This tendency means that Akerlof's lemons problem leads to a market failure. People who tend to be sicker will have insurance, while healthier people will not have insurance.

This section looks at the actual health insurance market in the United States using the Medical Expenditure Panel Survey (MEPS). This data set provides information on how much health costs people actually have, how much income they have, and whether or not the people actually buy insurance.

### 13.4.1 Willingness to Pay

For each subgroup, we can calculate the expected utility in the case where they are uninsured, the case where they are insured in separating equilibrium (the baseline case), and for the case when they are insured under a pooling equilibrium (counterfactual case). To be clear, the analysis assumes that the current observed data are from a separating equilibrium. In each subgroup, there are healthy types that are choosing not to get insurance. The insurance company is assumed to offer a premium that is actuarially fair given the types that purchases insurance.

The utility is a constant absolute risk aversion (CARA) function with risk-parameter r. To derive expected utility, we take the average income for the subgroup, the probability of having medical expenditure, the average expenditure, and the standard deviation of expenditure. In the code, the expected utility is calculated numerically. The code uses a trick of creating a global variable and using transformations of the uniform and the standard normal distributions.


`_>   set.seed(123456789)_`
`_>   K = 1000_`
`_>   U = runif(K) #uniform distribution_`
`_>   Z = qnorm(U) #transform to standard normal_`
`_>   EUexp = function(r, income, exppos, expmean, expsd) {_`
`_+     exp = income - (U < exppos)*(Z*expsd + expmean)_`
`_+     return(Euf(r, ifelse(exp > 0, exp, 0), rep(1/K,K)))_`
`_+   }_`
### 13.4.2 Premiums

There are three premiums we can calculate in the data. First, there are the actual premiums paid by beneficiaries in the data. These are called “out of pocket” premiums. Many Americans have their premiums subsidized. For many working Americans, the premiums are paid in part or full by the firm that they work for. These workers are accepting at least some amount of lower salary in order to get the subsidy on the insurance premium from their employer. There is also a substantial tax benefit to workers who get health insurance, which is paid for by the American tax payer. Second, we can calculate the actuarial fair premium for each subgroup that has insurance. Lastly, we can calculate what the actuarial fair premium would be if the uninsured became insured for each subgroup.

The actuarial fair premium is calculated as the expected expenditure for the subgroup conditioning on having insurance (UNISURD == 2). The out of pocket premium is read from the data.


`_> file = paste0(dir, “mepsfull.csv”)_`
`_> dt1 = fread(file)_`
`_> dtins = dt1[UNINSURD == 2,.(premium = premium_,`
`_+                          premiumalt = exppos*expmean)_,`
`_+            by = c(“agegroup”, “SEX”, “edugroup”)]_`
MEPS data are used to calculate average out of pocket premiums and health expenditures for each subgroup.

The pooled premium is calculated as the average expected expenditure for each subgroup where both insured and unsured individuals are included in the average.


`_> dtpool = dt1[, .(premiumpool = mean(exppos*expmean))_,`
`_+               by = c(“agegroup”, “SEX”, “edugroup”)]_`
The two data sets are merged back into the original data.


`_> dt2 = merge(dt1, dtins_,`
`_+             by = c(“agegroup”, “SEX”, “edugroup”))_`
`_> dt2 = merge(dt2, dtpool_,`
`_+             by = c(“agegroup”, “SEX”, “edugroup”))_`
`_> dt2$premium = dt2$premium.y_`
The code below calculates the premiums by age group and plots the results.


`_> ins = dt2[,.(premium = mean(premium)_,`
`_+             premiumalt = mean(premiumalt)_,`
`_+             premiumpool = mean(premiumpool))_,`
`_+                  by = agegroup]_`
`_> ins = ins[order(agegroup)]_`
`_> lineprems = setDF(ins) |>_`
`_+   ggplot(aes(x = agegroup)) +_`
`_+   geomline(aes(y = premiumalt), linetype = 2) +_`
`_+   geomline(aes(y = premium)) +_`
`_+   geomline(aes(y = premiumpool), linetype = 3) +_`
`_+   labs(x = “Age”_,`
`_+        y = “”_,`
`_+        title = “Premium ($)”) +_`
`_+   scaleycontinuous(limits = c(0,5000)) +_`
`_+   annotate(“text”, x = 50, y = 4000, label = “Seperating”) +_`
`_+   annotate(“text”, x = 62, y = 3000, label = “Pooling”) +_`
`_+   annotate(“text”, x = 60, y = 500, label = “Actual”)_`
` `
`_> lineprems_`
[Figure 13.1](./24-chapter13.md#fig13_1) presents line charts for average premiums across the different age groups. It presents the actual out of pocket premiums paid by insured beneficiaries, as well as the actuarially fair premium for the observed case and for the case where everyone is insured. It shows that the actual premium paid is substantially lower than the actuarially fair premium, particularly for the older beneficiaries. It also shows that premiums would come down in a pooling equilibrium.

![In the graph, the horizontal axis is labeled as age and ranges from 20 to 65. The vertical axis is labeled as premium in dollar and ranges from 0 to 5000. Three lines are shown. The top dashed line labeled separating starts around 1000 and increases steeply after age 40, ending near 5000. The middle dotted line labeled pooling starts near 800 and increases steadily to around 2500. The bottom solid line labeled actual starts near 400, rises slowly, and ends just below 1000. All data are approximate.](./images/fig13_1.jpg)

[Figure 13.1](chapter13) Line graph of average premiums for different age groups in the MEPS data. The lines show that premiums are increasing with age. The actual out of pocket premiums is substantially lower than the actuarial fair premiums for the insured population. The actual premiums increase from about $200 to $1,000 from 25 to 65\. The actuarial fair premiums increase from $1,000 to $4,500 over the age-range. Pooled premiums increase from about $750 to $3,000.

### 13.4.3 Tax Policy

We may have too many people who are choosing to not have health insurance. The uninsured can be a burden on society because when they get sick they may not be able to cover the medical expenses, which moves those expenses to the hospitals or the government. There are a couple of policies that we could use to encourage a higher uptake of health insurance. We could either make choosing to have health insurance cheaper or we can make choosing not to have health insurance more expensive. The US uses the former. Taxes are not paid on the health insurance portion of income, giving a tax subsidy to employed people for purchasing insurance. An alternative policy of taxing those that don't choose to get insurance was originally planned for the insurance exchanges created as part of the Affordable Care Act.

The tax policy charges a tax on people who choose not to buy health insurance. If the tax is high enough, then most people will choose to have health insurance and the insurance premiums will fall.

To determine how behavior changes, calculate the expected value for being insured and being uninsured for the people in the data who are currently insured and for people who are currently uninsured. Below these are denoted dt2\_in and dt2\_un, respectively.


`_> r = 0.7_`
`_> dt2in = dt2[UNINSURD == 2_,`
`_+             .(EUuns = EUexp(r_,`
`_+                               income + premiumalt - premium_,`
`_+                               exppos_,`
`_+                               expmean_,`
`_+                               expsd)_,`
`_+               EUins = EUexp(r_,`
`_+                               income - premium_,`
`_+                               0_,`
`_+                               0_,`
`_+                               0))_,`
`_+             by = “id”]_`
`_> dt2un = dt2[UNINSURD == 1, .(EUuns = EUexp(r_,`
`_+                                               income_,`
`_+                                               exppos_,`
`_+                                               expmean_,`
`_+                                               expsd)_,`
`_+               EUinssep = EUexp(r_,`
`_+                                   income - premiumalt_,`
`_+                                   0_,`
`_+                                   0_,`
`_+                                   0)_,`
`_+               EUinspool = EUexp(r_,`
`_+                                    income - premiumpool_,`
`_+                                    0_,`
`_+                                    0_,`
`_+                                    0))_,`
`_+           by=“id”]_`
The function tax\_policy() calculates the percentage of the population that is insured given a tax rate. The code is a bit ugly. It calculates the expected utility if the uninsured individual remains uninsured with the new tax. It then mergers, the results back into the original data. The function reports the percent of the population that choose to be insured under the policy. It sums the proportion of the population already insured and then for the population that is uninsured it determines whether the expected utility under a pooling equilibrium is greater then the tax.


`_> r = 0.7_`
`_> taxpolicy = function(tax) {_`
`_+   dt21 = dt2[UNINSURD == 1_,`
`_+              .(EUtax = EUexp(r_,`
`_+                                (1 - tax)*income_,`
`_+                                exppos_,`
`_+                                expmean_,`
`_+                                expsd))_,`
`_+             by = “id”]_`
`_+   dt2un1 = merge(dt2un, dt21, by = “id”)_`
`_+   dt2un2 = merge(dt2un1, dt2, by = “id”)_`
`_+   return((sum(dt2$count[dt2$UNINSURD==2], na.rm = TRUE) +_`
`_+             sum((dt2un2$EUtax < dt2un2$EUinspool)*_`
`_+                   dt2un2$count, na.rm = TRUE))/_`
`_+            sum(dt2$count, na.rm = TRUE))_`
`_+ }_`
The code below calculates the percentage of the population that is insured under different tax rates and plots the results.


`_> lineinsprop = data.frame(tax = seq(0, 1, 0.01)_,`
`_+            insprop = sapply(1:101, function(i)_`
`_+              taxpolicy(i/100))) |>_`
`_+   ggplot(aes(x = tax, y = insprop)) +_`
`_+   geomline() +_`
`_+   labs(x = “Tax rate”_,`
`_+        y = “”_,`
`_+        title = “Percent insured”) +_`
`_+   scalexcontinuous(limits = c(0, 0.4)) +_`
`_+   scaleycontinuous(limits = c(0.75, 1)) +_`
`_+   geomhline(yintercept = 1, linetype = 2)_`
` `
` `
`_> lineinsprop_`
[Figure 13.2](./24-chapter13.md#fig13_2) suggests that it is not that easy to encourage people to purchase insurance by imposing a tax penalty on the uninsured. The relationship is non-linear meaning that a small tax can get a large proportion of people onto insurance, but the relative effectiveness falls the higher the tax. The analysis suggests taxes 30 percent rage are needed to get everyone insured.

![In the graph, the horizontal axis is labeled as tax rate and ranges from 0 to 0.4. The vertical axis is labeled as percent insured and ranges from 0.75 to 1.0. A single line rises steadily from 0.85 at tax rate 0 to 1.0 at tax rate around 0.3. The line then remains flat at 1.0 until the end. A horizontal dashed line is at 1.0. All data are approximate.](./images/fig13_2.jpg)

[Figure 13.2](chapter13) Line graph of the percentage of the population insured against the tax rate. Around 85% are insured without any tax, the proportion reaches 100% with a tax rate over 30%.

## 13.5 Discussion and Further Reading

Adverse selection is most famously illustrated by [Akerlof (1970)](./25-refbib.md#ref4) and the idea of a lemons market. Adverse selection and moral hazard have become key to understanding health insurance markets. These ideas formed the basis for policies introduced in the Affordable Care Act that aimed to reduce the number of uninsured. See [Pauly (1974)](./25-refbib.md#ref50). The tax on the uninsured, called the “individual mandate” was subject to lawsuits and was eventually repealed. So did it work? A recent survey by Brookings economist, Matt Fiedler finds that the tax _may_ have reduced the uninsured rate. It turns out to be very tricky to determine what happened given all the changes that occurred with the introduction of the ACA and the complexity of the policy ([Fiedler, 2020](./25-refbib.md#ref23)).

[_OceanofPDF.com_](./https___oceanofpdf.com)
