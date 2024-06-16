## Glossary

**[absolute risk](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-01)**: the proportion of people in a defined group who experience an event of interest within a specified period of time.

**[adjustment/stratification](./14-CHAPTER_4__What_Causes_What_.md#gloss-02)**: inclusion into a regression model of known confounders which are not of direct interest, but are intended to allow a more balanced comparison between groups. The hope is that estimated effects associated with explanatory variables of interest should then be closer to causal effects.

**[aleatory uncertainty](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-03)**: unavoidable unpredictability about the future, also known as chance, randomness, luck and so on.

**[algorithm](./10-INTRODUCTION.md#gloss-04)**: a rule or formula that takes input variables and produces an output, such as a prediction, a classification, or a probability.

**[artificial intelligence (AI)](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-05)**: computer programs intended to perform a task normally associated with human abilities.

**[ascertainment bias](./14-CHAPTER_4__What_Causes_What_.md#gloss-06)**: when the chance of a person being sampled, or a feature being observed, depends on some background factor, for example when people in the treated arm of a randomized trial get closer supervision than the control group.

**[average](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-07)**: a generic term for a single representative value for a set of numbers, for example the mean, median or mode.

**[Bayes factor](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-08)**: the relative support given by a set of data for two alternative hypotheses. For hypotheses _H_0 and _H_1, and data _x_, the ratio is _p_(_x_|_H_0)/_p_(_x_|_H_1).

**[Bayesian](./18-CHAPTER_8__Probability_–_the_Language_of_Uncertainty_and_Variability.md#gloss-09)**: the approach to statistical inference in which probability is used not only for aleatory uncertainty, but also epistemic uncertainty about unknown facts. Bayes’ theorem is then used to revise these beliefs in the light of new evidence.

**[Bayes’ theorem](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-10)**: a rule of probability that shows how evidence _A_ updates prior beliefs of a proposition _B_ to produce posterior beliefs _p_(_B_|_A_), through the formula ![](./images/p382.png). This is easily proved: since _p_(_B_ AND _A_) = _p_(_A_ AND _B_), the multiplication rule of probability means that _p_(_B_|_A_)_p_(_A_) = _p_(_A_|_B_)_p_(_B_), and dividing each side by _p_(_A_) gives the theorem.

**[Bernoulli distribution](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-11)**: if _X_ is a random variable which takes on the value 1 with probability _p_, and 0 with probability 1 − _p_, it is known as a Bernoulli trial with a Bernoulli distribution. _X_ has mean _p_ and variance _p_(1 − _p_).

**[bias/variance trade-off](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-12)**: when fitting a model to be used for prediction, increasing complexity will eventually lead to a model that has less bias, in the sense that it has greater potential to adapt to details of the underlying process, but more variance, since there is not enough data to be confident about the parameters in the model. These elements need to be traded off in order to avoid over-fitting.

**[big data](./10-INTRODUCTION.md#gloss-13)**: an increasingly anachronistic phrase sometimes characterized by four Vs: a huge Volume of data, a Variety of sources such as images, social media accounts or transactions, a high Velocity of acquisition, and possible lack of Veracity due to its routine collection.

**[binary data](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-14)**: variables that can only take on two values, often yes/no responses to a question. Can be mathematically represented by a Bernoulli distribution.

**[binomial distribution](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-15)**: when there are _n_ independent possibilities for an event to occur, each with the same probability, the observed number of events has a binomial distribution. Technically for _n_ independent Bernoulli trials _X_1, _X_2 … _X_ _n_, each with probability _p_ of success, their sum _R_ \= _X_1 \+ _X_2 \+ … + _X_ _n_, has a binomial distribution with mean _np_ and variance _np_(1 − _p_), where ![](./images/p383_a.png). The observed proportion _R/n_ has mean _p_ and variance _p_(1 − _p_)/_n_: _R/n_ can therefore be considered as an estimator of _p_, with standard error ![](./images/p383_b.png).

**[blinding](./14-CHAPTER_4__What_Causes_What_.md#gloss-16)**: when those engaged in a clinical trial do not know what treatment a patient has been given, in order to avoid bias in outcome assessments. Single blinding is when patients do not know what treatment they have been given, double blinding means the people monitoring the patients do not know their treatment, triple blinding is when treatments are labelled say _A_ and _B_, and the statisticians analysing the data and the committee monitoring the results do not know which corresponds to the new treatment.

**[Bonferroni correction](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-17)**: a method for adjusting size (Type I error) or confidence intervals to allow for simultaneous testing of multiple hypotheses. Specifically, when testing _n_ hypotheses, for an overall size (Type I error) of α, each hypothesis is tested with size α_/n_. Equivalently, 100(1 − α_/n_)% confidence intervals are quoted for each estimated quantity. For example, when testing 10 hypotheses with an overall α of 5%, then P-values would be compared to 0.05/10 = 0.005, and 99.5% confidence intervals used.

**[bootstrapping](./17-CHAPTER_7__How_Sure_Can_We_Be_About_What_Is_Going_On__Estimates_and_Intervals.md#gloss-18)**: a way of generating confidence intervals and the distribution of test statistics through resampling the observed data rather than through assuming a probability model for the underlying random variable. A basic bootstrap sample of a data set _x_1, _x_2 … _x_ _n_ is a sample of size _n_ with replacement, so that the bootstrap sample will be drawn from the original set of distinct values, but not generally in the same proportions as the original data set.

**[Brier score](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-19)**: a measure for the accuracy of probabilistic predictions, based on the mean squared prediction error. If _p_1 … _p_ _n_ are the probabilities given to a set of _n_ binary observations _x_1 … _x_ _n_ taking on values 0 and 1, then the Brier score is ![](./images/p384.png). Essentially a mean-squared-error criterion applied to binary data.

**[calibration](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-20)**: the requirement for the observed frequencies of events to match those expected by probabilistic predictions. For example, of the occasions when events are given a probability of 0.7, then the events should actually occur roughly 70% of the time.

**[case-control study](./14-CHAPTER_4__What_Causes_What_.md#gloss-21)**: a retrospective study design in which people with a disease or outcome of interest (the cases) are matched with one or more people who do not have the disease (the controls), and the histories of the two groups are compared to see whether there are exposures which systematically differ between the two groups. This design can only estimate relative risks associated with exposures.

**[categorical variable](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-22)**: a variable that can take on two or more discrete values, which may or may not be ordered.

**[Central Limit Theorem](./17-CHAPTER_7__How_Sure_Can_We_Be_About_What_Is_Going_On__Estimates_and_Intervals.md#gloss-23)**: the tendency for the sample mean of a set of random variables to have a normal sampling distribution, regardless (with certain exceptions) of the shape of the underlying sampling distribution of the random variable. If _n_ independent observations each have mean μ and variance σ2, then under broad assumptions their sample mean is an estimator of μ, and has an approximately normal distribution with mean μ, variance σ2/_n,_ and standard deviati ![](./images/p385_a.png) (also known as the standard error of the estimator).

**[chi-squared test of association](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-24) / [goodness-of-fit test](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-24a)**: a statistical test that indicates the degree of incompatibility of data with an assumed statistical model comprising the null hypothesis, which may be one of lack of association, or some other specfied mathematical form. Specifically, the test compares a set of _m_ observed counts _o_1, _o_2 … _o_ _m_ with a set of expected values _e_1, _e_2 … _e_ _m_ which have been calculated under the null hypothesis. The simplest version of the test statistic is given as

![](./images/p385_b.png)

Under the null hypothesis Χ2 will have an approximate chi-squared sampling distribution, enabling an associated P-value to be calculated.

**[classification tree](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-25)**: a form of classification algorithm in which features are examined in sequence, with the response indicating the next feature to examine, until a classification is made.

**[confidence interval](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-26)**: an estimated interval within which an unknown parameter may plausibly lie. Based on an observed set of data _x_, a 95% confidence interval for μ is an interval whose lower limit _L_(_x_) and upper limit _U_(_x_) has the property that, before observing the data, there is a 95% probability that the random interval (_L_(_X_), _U_(_X_)) contains μ. The Central Limit Theorem, combined with the knowledge that close to 95% of a normal distribution lies between the mean ± 2 standard deviations, means that a common approximation for a 95% confidence interval is the estimate ± 2 standard errors. Suppose we want to find a confidence interval for the difference μ2 − μ1 between two parameters μ2 and μ1. If _T_1 is an estimator of μ1 with standard error _SE_1, and _T_2 is an estimator of μ2 with standard error _SE_2, then _T_2 − _T_1 is an estimator of μ2 − μ1. The variance of the difference between two estimators is the sum of their variances, and so the standard error of _T_2 − _T_1 is given by ![](./images/p386_a.png). From this a 95% confidence interval for the difference μ2 − μ1 can be constructed.

**[confirmatory studies and analyses](./22-CHAPTER_12__How_Things_Go_Wrong.md#gloss-27)**: rigorous studies ideally done to a pre-specified protocol to confirm or negate hypotheses suggested by exploratory studies and analyses.

**[confounder](./14-CHAPTER_4__What_Causes_What_.md#gloss-28)**: a variable which is associated with both a response and a predictor, and which may explain some of their apparent relationship. For example, the height and weight of children are strongly correlated, but much of this association is explained by the age of the child.

**[continuous variable](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-29)**: a random variable _X_ that can, at least in principle, take on any value within a specific range. It has a probability density function _f_ such that ![](./images/p386_b.png), and expectation given by ![](./images/p386_c.png). The probability of _X_ lying in the interval (_A, B_) can be calculated using ![](./images/p386_d.png).

**[control group](./14-CHAPTER_4__What_Causes_What_.md#gloss-30)**: a set of individuals who have not been subject to the exposure of interest, say by randomization.

**[control limits](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-31)**: pre-specified limits for a random variable which are used in quality control to monitor deviation from an intended standard, say displayed on a funnel plot.

**[count variables](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-32)**: variables that can take on integer values 0, 1, 2 and so on.

**[counter-factual](./14-CHAPTER_4__What_Causes_What_.md#gloss-33)**: a ‘what-if’ scenario in which an alternative history of events is considered.

**[cross-sectional study](./14-CHAPTER_4__What_Causes_What_.md#gloss-34)**: when analysis is based solely on the current state of individuals, without any follow-up over time.

**[cross-validation](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-35)**: a way of assessing the quality of an algorithm for prediction or classification by systematically removing some cases to act as a test set.

**cox regression**: See **[hazard ratio](glossary)**.

**[data literacy](./10-INTRODUCTION.md#gloss-37)**: the ability to understand the principles behind learning from data, carry out basic data analyses, and critique the quality of claims made on the basis of data.

**[data science](./10-INTRODUCTION.md#gloss-38)**: the study and application of techniques for deriving insights from data, including constructing algorithms for prediction. Traditional statistical science forms part of data science, which also includes a strong element of coding and data management.

**[deep learning](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-39)**: a machine-learning technique that extends standard artificial neural network models to many layers representing different levels of abstraction, say going from individual pixels of an image through to recognition of objects.

**[dependent events](./18-CHAPTER_8__Probability_–_the_Language_of_Uncertainty_and_Variability.md#gloss-40)**: when the probability of one event depends on the outcome of another event.

**[dependent, response or outcome variable](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-41)**: the variable of primary interest that we wish to predict or explain.

**[epidemiology](./14-CHAPTER_4__What_Causes_What_.md#gloss-42)**: the study of the rates of, and reasons for, the occurrence of disease.

**[epistemic uncertainty](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-43)**: lack of knowledge about facts, numbers or scientific hypotheses.

**[error matrix](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-44)**: a cross-tabulation of correct and incorrect classifications by an algorithm.

**[expectation (mean)](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-45)**: the mean-average of a random variable. It is defined as ∑_xp_(_x_) for a discrete random variable _X_ and ∫_xp_(_x_)_dx_ for a continuous random variable. For example, if _X_ is the result of throwing a fair die, then ![](./images/p388_c.png) for _x_ \= 1, 2, 3, 4, 5, 6, so that ![](./images/p388_d.png)

**[expected frequencies](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-46)**: the numbers of events expected to occur in the future, according to an assumed probability model.

**[exploratory studies and analyses](./22-CHAPTER_12__How_Things_Go_Wrong.md#gloss-47)**: initial flexible studies which allow adaptive changes to design and analyses in order to pursue promising leads, and are intended to generate hypotheses to be tested in confirmatory studies.

**[exposure](./14-CHAPTER_4__What_Causes_What_.md#gloss-48)**: a factor whose impact on a disease, death or other medical outcome is of interest, such as an aspect of the environment or behaviour.

**[external validity](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-49)**: when the conclusions of a study can be generalized to a target group, wider than the immediate population that has been studied. This addresses the relevance of a study.

**[false discovery rate](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-50)**: when testing multiple hypotheses, the proportion of positive claims that turn out to be false-positives.

**[false-positive](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-51)**: an incorrect classification of a ‘negative’ case as a ‘positive’ case.

**[feature engineering](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-52)**: in machine learning, the process of reducing the dimensionality of input variables, creating summary measures intended to encapsulate the information in the whole data.

**[forensic epidemiology](./14-CHAPTER_4__What_Causes_What_.md#gloss-53)**: using knowledge about the causes of disease in populations when making judgements about the causes of disease in individuals.

**[framing](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-54)**: the choice of how to express numbers, which in turn can influence the impression given to audiences.

**[funnel plot](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-55)**: a plot of a set of observations from different units against a measure of their precision, where units might be institutions, areas or studies. Often two ‘funnels’ indicate where we would expect 95% and 99.8% of observations to lie, were there really no underlying differences between the units. When the distribution of the observations is approximately normal, the 95% and 99.8% control limits are essentially the mean ± two and three standard errors.

**[hazard ratio](./22-CHAPTER_12__How_Things_Go_Wrong.md#gloss-56)**: when analysing survival times, the relative risk, associated with an exposure, of suffering an event in a fixed period of time. A Cox regression is a form of multiple regression when the response variable is a survival time, and the coefficients correspond to log(hazard ratios).

**[hierarchical modelling](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-57)**: in Bayesian analysis, when the parameters underlying a number of units, say areas or schools, are themselves assumed to be drawn from a common prior distribution. This results in shrinkage of the parameter estimates for individual units towards an overall mean.

**[hypergeometric distribution](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-58)**: the probability of _k_ successes in _n_ draws, without replacement, from a finite population of size _N_ that contains exactly _K_ objects with that feature, formally given by

![](./images/p389.png)

**[hypothesis testing](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-59)**: a formal procedure for evaluating the support for hypotheses provided by data, generally an amalgam of classic Fisherian tests of a null hypothesis using a P-value, and the Neyman–Pearson structure of null and alternative hypotheses and Type I and Type II errors.

**[icon arrays](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-60)**: a graphic display of frequencies using a set of small images, say of people.

**[independent events](./18-CHAPTER_8__Probability_–_the_Language_of_Uncertainty_and_Variability.md#gloss-61)**: _A_ and _B_ are independent if the occurrence of _A_ does not influence the probability of _B_, so that _p_(_B_|_A_) = _p_(_B_), or equivalently _p_(_B_, _A_) = _p_(_B_)_p_(_A_).

**[independent variable / predictor](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-62)**: a variable that is fixed by design or observation, and whose association with an outcome variable may be of interest.

**[induction / inductive inference](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-63)**: the process of learning about general principles from specific examples.

**[inductive behaviour](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-64)**: a proposal by Jerzy Neyman and Egon Pearson in the 1930s to frame hypothesis testing in terms of decision-making. The ideas of size, power and Type I and Type II errors are remnants.

**[intention to treat](./14-CHAPTER_4__What_Causes_What_.md#gloss-65)**: the principle by which participants in randomized trials are analysed according to whatever intervention they were supposed to get, whether or not they actually received it.

**[interactions](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-66)**: when multiple explanatory variables combine to produce an effect different from that expected from their individual contributions.

**[internal validity](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-67)**: when the conclusions of a study truly apply to the population of a study. This addresses the rigour with which a study has been conducted.

**[inter-quartile range](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-68)**: a measure of the spread of a sample or a population distribution, specifically the distance between the 25th and 75th percentiles. Equivalent to the difference between the 1st and 3rd quartiles.

**[Law of Large Numbers](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-69)**: the process by which the sample mean of a set of random variables tends towards the population mean.

**[least-squares](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-70)**: suppose we have a set of _n_ paired numbers, (_x_1, _y_1), (_x_2, _y_2) … (_x_ _n_, _y_ _n_), and ![](./images/p391_a.png) are the sample mean and standard deviation of the _x_s, and ![](./images/p391_b.png) are the sample mean and standard deviation of the _y_s. Then the least-squares regression line is given by

![](./images/p391_c.png)

where

– _ŷ_ is the predicted value for the dependent variable _y_ for a specified value of the independent variable _x_.

– The gradient is ![](./images/p391_e.png).

– The intercept is _b_0 \= ![](./images/p391_f.png). The least-squares line goes through the centre of gravity ![](./images/p391_i.png).

– The _i_th residual is the difference between the _i_th observation and its predicted value, _y_ _i_ − _ŷ_ _i_.

– The adjusted value of the _i_th observation is the residual added to the intercept, i.e., ![](./images/p391_k.png). It is intended to be the value we would have observed were this an ‘average’ case, that is with _x_ \= ![](./images/p391_g.png) rather than _x_ \= _x_ _i_.

– The residual sum of squares (RSS) is the sum of the squares of the residuals, so that ![](./images/p391_h.png). The least-squares line is defined as the line that minimizes the residual sum of squares.

– The gradient _b_1 and Pearson’s correlation coefficient _r_ are related through the formula _b_1 \= _rs_ _y_ _/s_ _x_. So if the standard deviations of the _x_s and _y_s are the same, then the gradient is exactly equal to the correlation coefficient.

**[likelihood](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-71)**: a measure of the evidential support provided by data for particular parameter values. When a probability distribution for a random variable depends on a parameter, say θ, then after observing data _x_ the likelihood for θ is proportional to _p_(_x_|θ).

**[likelihood ratio](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-72)**: a measure of the relative support that some data provides for two competing hypotheses. For hypotheses _H_0 and _H_1, the likelihood ratio provided by data _x_ is given by _p_(_x_|_H_0)/_p_(_x_|_H_1).

**[logarithmic scale](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-73)**: The logarithm to base 10 of a positive number _x_ is denoted by _y_ \= log10 _x_, or equivalently _x_ \= 10_y_. In statistical analysis, log _x_ generally denotes the natural logarithm _y_\=log_e_ _x_, or equivalently _x_ \= _e_ _y_ where _e_ is the exponential constant 2.718.

**[logistic regression](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-74)**: a form of multiple regression when the response variable is a proportion, and the coefficients correspond to log(odds ratios). Suppose we observe a series of proportions _y_ _i_ \= _r_ _i_ _/n_ _i_, assumed to arise from a binomial variable with underlying probability _p_ _i_ with a corresponding set of predictor variables (_x_ _i_1, _x_ _i_2 … _x_ _ip_). The logarithm of the odds of the estimated probability ![](./images/p392_a.png) is assumed to be a linear regression:

![](./images/p392.png)

Suppose one of the predictor variables, say _x_1, is binary with _x_1 \= 0 corresponding to not being exposed to a potential hazard, and _x_1 \= 1 corresponding to being exposed. Then the coefficient _b_1 is a log (odds ratio).

**[lurking factor](./14-CHAPTER_4__What_Causes_What_.md#gloss-75)**: in epidemiology, an exposure that has not been measured but may be a confounder responsible for some of the observed association: for example, when socioeconomic status has not been measured in a study relating diet with disease.

**[machine learning](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-76)**: procedures for extracting algorithms, say for classification, prediction or clustering, from complex data.

**[margin of error](./17-CHAPTER_7__How_Sure_Can_We_Be_About_What_Is_Going_On__Estimates_and_Intervals.md#gloss-77)**: after a survey, a plausible range in which a true characteristic of a population may lie. These are generally 95% confidence intervals, which are approximately ± 2 standard errors, but sometimes error-bars are used to represent ± 1 standard error.

**[mean (of a population)](./17-CHAPTER_7__How_Sure_Can_We_Be_About_What_Is_Going_On__Estimates_and_Intervals.md#gloss-78)**: _see_ **[expectation](glossary)**

**[mean (of a sample)](./17-CHAPTER_7__How_Sure_Can_We_Be_About_What_Is_Going_On__Estimates_and_Intervals.md#gloss-79)**: suppose we have a set of _n_ data-points, which we label as _x_1, _x_2, …, _x_ _n_ _._ Then their sample mean is given by _m_ \= (_x_1 \+ _x_2 \+ … + _x_ _n_)/_n_, which can be written as ![](./images/p393_a.png). For example, if 3, 2, 1, 0, 1 are the numbers of children reported by 5 people in a sample, then the sample mean is (3 + 2 + 1 + 0 + 1)/5 = 7/5 = 1.4.

**[mean-squared-error (MSE)](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-80)**: a measure of performance when predictions _t_1 … _t_ _n_ are made of observations _x_1 … _x_ _n_, given by ![](./images/p393_b.png).

**[median (of a sample)](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-81)**: the value mid-way along the ordered set of data-points. If the data-points are put in order, we denote the lowest by _x_(1), the second lowest by _x_(2), and so on until the maximum value _x_(_n_). If _n_ is odd, then the sample median is the middle value ![](./images/p393_c.png); if _n_ is even, then the average of the two ‘middle’ points is taken as the median.

**[meta-analysis](./14-CHAPTER_4__What_Causes_What_.md#gloss-82)**: a formal statistical method for combining the results from multiple studies.

**mode (of a population distribution)**: the response with the maximum probability of occurring.

**mode (of a sample)**: the most common value in a set of data.

**[multi-level regression and post-stratification (MRP)](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-85)**: a modern development in survey sampling in which fairly small numbers of responders are obtained from many areas. A regression model is then built relating responses to demographic factors, allowing for additional between-area variability using hierarchical modelling. Knowing the demographics of all areas then allows both local and national predictions to be made, with appropriate uncertainty.

**[multiple linear regression](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-86)**: suppose that for every response _y_ _i_ there are a set of _p_ predictor variables (_x_ _i_1, _x_ _i_2 … _x_ _ip_). Then a least-squares multiple linear regression is given by

![](./images/p394_a.png)

where the coefficients _b_0, _b_1 … _b_ _p_ are chosen to minimize the residual sum of squares ![](./images/p394_b.png). The intercept _b_0 is simply the mean ![](./images/p391_f.png), and the formula for the remaining coefficients is complex but easily computed. Note that _b_0 \= ![](./images/p391_f.png) is the predicted value of an observation _y_ whose predictor variables were the averages (![](./images/p391_g.png)1, ![](./images/p391_g.png)2 … ![](./images/p391_g.png)_p_), and, just as for a linear regression, an adjusted _y_ _i_ is given by the residual plus the intercept, or ![](./images/p391_k.png).

**[multiple testing](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-87)**: when a series of hypothesis tests are carried out, so increasing the chance of at least one false-positive claim (Type 1 error).

**[normal distribution](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-88)**: _X_ has a normal (Gaussian) distribution with mean μ and variance σ2 if it has a probability density function

![](./images/p394_c.png)

Then _E_(_X_) = μ, _V_(_X_) = σ2, _SD_(_X_) = σ. The standardized variable ![](./images/p395_a.png) has mean 0 and variance 1, and is said to have a standard normal distribution. We write Φ for the cumulative probability of a standard normal variable _Z_.

For example, Φ(−1) = 0.16 is the probability of a standard normal variable being less than −1, or equivalently, the probability of a general normal variable being less then one standard deviation below the mean. The 100_p_% percentile of the standard normal distribution is _z_ _p_ where _P_(_Z_ ≤ _z_ _p_) = _p_. Values of Φ are available in standard software or tables, as are percentage points _z_ _p_: for example, the 75th percentile of the standard normal distribution is _z_0.75 \= 0.67.

**[null hypothesis](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-89)**: a default scientific theory, generally representing the absence of an effect or a finding of interest, which is tested using a P-value. Generally denoted _H_0.

**[objective priors](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-90)**: an attempt to remove the subjective element in Bayesian analysis, by pre-specifying prior distributions that are intended to represent ignorance about parameters, and so let the data speak for itself. No overall procedure for setting such priors has been established.

**[odds](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-91), [odds ratios](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-91a)**: if the probability of an event is _p_, the odds of the event is defined by ![](./images/p395_b.png). If the odds of an event in the exposed group is ![](./images/p395_b.png), and the odds in the non-exposed group is ![](./images/p395_c.png): the odds ratio is then given by ![](./images/p395_d.png). If _p_ and _q_ are small, then the odds ratio will be close to the relative risk _p/q,_ but odds ratios and relative risks start to differ when the absolute risks are much more than 20%.

**[one-sided](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-92) and [two-sided tests](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-92a)**: a one-sided hypothesis test is used when a null hypothesis specifies that, say, the effect of a medical treatment is negative. This would only be rejected by large positive values of a test statistic representing an estimated treatment effect. A two-sided test would be appropriate for a null hypothesis that a treatment effect, say, is exactly zero, and so both positive and negative estimates would lead to the null being rejected.

**[one-tailed](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-93) and [two-tailed P-values](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-93a)**: those corresponding to one-sided and two-sided tests.

**[over-fitting](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-94)**: building a statistical model that is over-adapted to training data, so that its predictive ability starts to decline.

**[parameters](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-95)**: the unknown quantities in a statistical model, generally denoted with Greek letters.

**[Pearson correlation coefficient](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-96)**: for a set of _n_ paired numbers, (_x_1, _y_1), (_x_2, _y_2) … (_x_ _n_, _y_ _n_), when ![](./images/p391_g.png), _s_ _x_ are the sample mean and standard deviation of the _x_s, and ![](./images/p391_f.png), _s_ _y_ are the sample mean and standard deviation of the _y_s, the Pearson correlation coefficient is given by

![](./images/p396_a.png)

Suppose _x_s and _y_s have both been standardized to Z-scores given by _u_s and _v_s respectively, so that _u_ _i_ \= (_x_ _i_ – ![](./images/p391_g.png))/_s_ _x_ _,_ and _v_ _i_ \= (_y_ _i_ – ![](./images/p391_f.png))/_s_ _y_. Then the Pearson correlation coefficient can be expressed as ![](./images/p396_b.png), that is the ‘cross-product’ of the Z-scores.

**[percentile (of a population)](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-97)**: there is, for example, a 70% chance of drawing a random observation below the 70th percentile. For a literal population, it is the value below which 70% of the population lie.

**[percentile (of a sample)](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-98)**: the 70th percentile of a sample, for example, is the value that is 70% along the ordered data set: the median is therefore the 50th percentile. Interpolation between points may be necessary.

**[permutation/randomization test](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-99)**: a form of hypothesis test in which the distribution of the test statistic under the null hypothesis is obtained by permuting the labels of the data, rather than through a detailed statistical model for the random variables. Suppose the null hypothesis is that a ‘label’, say being male or female, is not associated with an outcome. Randomization tests examine all possible ways in which labels for individual data-points can be rearranged, each of which are equally likely under the null hypothesis. The test statistic for each of these permutations is calculated, and the P-value is given by the proportion that lead to more extreme test statistics than that actually observed.

**[placebo](./14-CHAPTER_4__What_Causes_What_.md#gloss-100)**: a dummy treatment given to the control arm of a randomized clinical trial, such as a sugar pill disguised to look like the treatment being tested.

**[Poisson distribution](./18-CHAPTER_8__Probability_–_the_Language_of_Uncertainty_and_Variability.md#gloss-101)**: a distribution for a count random variable _X_ for which ![](./images/p397.png) for _x_ \= 0, 1, 2 … Then _E_(_X_) = μ and _V_(_X_) = μ.

**[population](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-102)**: a group from which it is assumed your sample data are drawn, and which provides the probability distribution for a single observation. In a survey this may be a literal population, but when making measurements, or when having all possible data, the population becomes a mathematical idealization.

**[population distribution](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-103)**: when the population literally exists, the pattern of potential observations in the entire population. It also refers to the probability distribution of a generic random variable.

**[posterior distribution](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-104)**: in Bayesian analysis, the probability distribution of unknown parameters after taking into account observed data through Bayes’ theorem.

**[power of a test](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-105)**: the probability of correctly rejecting the null hypothesis, given the alternative hypothesis is true. It is one minus the Type II error rate of a statistical test, and is generally denoted by 1 – β.

**[PPDAC](./10-INTRODUCTION.md#gloss-106)**: a proposed structure for the ‘data cycle’, comprising Problem, Plan, Data collection, Analysis (exploratory or confirmatory) and Conclusions and communication.

**[practical significance](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-107)**: when a finding is of genuine importance. Large studies may give rise to results that are statistically but not practically significant.

**[predictive analytics](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-108)**: using data to create algorithms for making predictions.

**[prior distribution](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-109)**: in Bayesian analysis, the initial probability distribution for the unknown parameters. After observing data, it is revised to the posterior distribution using Bayes’ theorem.

**[probabilistic forecast](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-110)**: a prediction in the form of a probability distribution for a future event, rather than a categorical judgement of what will happen.

**[probability](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-111)**: the formal mathematical expression of uncertainty. Let _P_(_A_) be the probability for an event _A_. Then the rules of probability are:

1. Bounds: 0 ≤ _P_(_A_) ≤ 1, with _P_(_A_) = 0 if _A_ is impossible and _P_(_A_) = 1 if _A_ is certain.
2. Complement: _P_(_A_) = 1 − _P_(NOT _A_).
3. Addition rule: If _A_ and _B_ are mutually exclusive (i.e_.,_ one at most can occur), _P_(_A_ OR _B_) = _P_(_A_) + _P_(_B_).
4. Multiplication rule: For any events _A_ and _B_, _P_(_A_ AND _B_) = _P_(_A_|_B_)_P_(_B_), where _P_(_A_|_B_) represents the probability for _A_ given _B_ has occurred. _A_ and _B_ are independent if and only if _P_(_A_|_B_) = _P_(_A_), i.e., the occurrence of _B_ does not affect the probability for _A_. In this case we have _P_(_A_ AND _B_) = _P_(_A_)_P_(_B_), the multiplication rule for independent events.

**[probability distribution](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-112)**: a generic term for a mathematical expression of the chance of a random variable taking on particular values. A random variable _X_ has a probability distribution function defined by _F_(_x_) = _P_(_X_ ≤ _x_), for all −∞ < _x_ < ∞, i.e., the probability that _X_ is at most _x_.

**[prosecutor’s fallacy](./18-CHAPTER_8__Probability_–_the_Language_of_Uncertainty_and_Variability.md#gloss-113)**: when a small probability of the evidence, given innocence, is mistakenly interpreted as the probability of innocence, given the evidence.

**[prospective cohort study](./14-CHAPTER_4__What_Causes_What_.md#gloss-114)**: when a set of individuals are identified, background factors measured, and then they are followed up and relevant outcomes observed. Such studies are lengthy and expensive, and may not identify many rare events.

**[P-value](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-115)**: a measure of discrepancy between data and a null hypothesis. For a null hypothesis _H_0, let _T_ be a statistic for which large values indicate inconsistency with _H_0. Suppose we observe a value _t_. Then a (one-sided) P-value is the probability of observing such an extreme value, were _H_0 true, that is _P_(_T_ ≥ _t_|_H_0). If both small and large values of _T_ indicate inconsistency with _H_0, then the two-sided P-value is the probability of observing such a large value in either direction. Often the two-sided P-value is simply taken as double the one-sided P-value, while the R software uses the total probability of events which have a lower probability of occurring than that actually observed.

**[quartiles (of a population)](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-116)**: the 25th, 50th and 75th percentiles.

**[randomized controlled trial (RCT)](./14-CHAPTER_4__What_Causes_What_.md#gloss-117)**: an experimental design in which people or other units being tested are randomly allocated to different interventions, thus ensuring, up to the play of chance, that the groups are balanced in both known and unknown background factors. If the groups show subsequent differences in outcome, then either the effect must be due to the intervention or a surprising event has occurred, whose probability can be expressed as a P-value.

**[random match probability](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-118)**: in forensic DNA testing, the probability that a person randomly drawn from a relevant population would match the observed DNA profile that connects a suspect with a crime.

**[random variable](./18-CHAPTER_8__Probability_–_the_Language_of_Uncertainty_and_Variability.md#gloss-119)**: a quantity assumed to have a probability distribution. Before they are observed, random variables are generally given a capital letter such as _X_, while observed values are denoted _x_.

**[range (of a sample)](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-120)**: the maximum minus the minimum, denoted _x_(_n_) – _x_(1).

**[rate ratio](./22-CHAPTER_12__How_Things_Go_Wrong.md#gloss-121)**: the relative increase in the expected number of events in a fixed period of time associated with an exposure. A Poisson regression is a form of multiple regression when the response variable is the observed rate, and the coefficients correspond to log(rate ratios).

**[Receiver Operating Characteristic (ROC) curve](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-122)**: for an algorithm that generates a score, we can choose a particular threshold for the score above which a unit is classified as ‘positive’. As this threshold varies, the ROC curve is formed by plotting the resulting sensitivity (true-positive rate) on the _y_\-axis versus one minus specificity (false-positive rate) on the _x_\-axis.

**[regression coefficient](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-123)**: an estimated parameter in a statistical model, that expresses the strength of relationship between an explanatory variable and an outcome in multiple regression analysis. The coefficient will have a different interpretation depending on whether the outcome variable is a continuous variable (multiple linear regression), a proportion (logistic regression), a count (Poisson regression) or a survival time (Cox regression).

**[regression to the mean](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-124)**: when a high or low observation is followed by one that is less extreme, through the process of natural variation. It occurs because part of the reason for the initial extreme case was chance, and this is unlikely to repeat to the same extent.

**[relative risk](./11-CHAPTER_1__Getting_Things_in_Proportion__Categorical_Data_and_Percentages.md#gloss-125)**: if the absolute risk among people who are exposed to something of interest is _p_, and the absolute risk among people who are not exposed is _q_, then the relative risk is _p/q_.

**[reproducibility crisis](./22-CHAPTER_12__How_Things_Go_Wrong.md#gloss-126)**: the claim that many published scientific findings are based on work of insufficient quality, so that the results fail to be reproduced by other researchers.

**[residual](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-127)**: the difference between an observed value and that predicted by a statistical model.

**[residual error](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-128)**: the generic term for the component of the data that cannot be explained by a statistical model, and so is said to be due to chance variation.

**[retrospective cohort study](./14-CHAPTER_4__What_Causes_What_.md#gloss-129)**: when a set of individuals are identified at a point in the past, and their subsequent outcomes traced up to the present day. Such a study does not require an extended period of follow-up, but is dependent on the appropriate explanatory variables having been measured in the past.

**[reverse causation](./14-CHAPTER_4__What_Causes_What_.md#gloss-130)**: when an association between two variables initially appears to be causal, but could in fact be acting in the opposite direction. For example, people who do not drink alcohol tend to have poorer health outcomes than moderate drinkers, but this is at least partly due to some non-drinkers having given up alcohol due to poor health.

**[sample distribution](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-131)**: the pattern made by a set of numerical or categorical observations. Also known as the empirical or data distribution.

**sample mean**: _see_ **[mean (of a sample)](glossary)**

**[sampling distribution](./17-CHAPTER_7__How_Sure_Can_We_Be_About_What_Is_Going_On__Estimates_and_Intervals.md#gloss-133)**: the probability distribution of a statistic.

**[sensitivity](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-134)**: the proportion of ‘positive’ cases that are correctly identified by a classifier or test, often termed the true-positive rate. One minus sensitivity is also known as the observed Type II error or false-negative rate.

**[sequential testing](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-135)**: when a statistical test is repeatedly carried out on accumulating data, thus inflating the chance of a Type I error occurring at some point. A ‘significant result’ is guaranteed if the process is continued for long enough.

**[shrinkage](./21-CHAPTER_11__Learning_from_Experience_the_Bayesian_Way.md#gloss-136)**: the influence of a prior distribution in Bayesian analysis, in which an estimate tends to be pulled towards either an assumed or an estimated prior mean. This is also known as ‘borrowing strength’, since, say, estimated rates of disease in a specific geographical area are influenced by rates in other areas.

**[signal and the noise](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-137)**: the idea that observed data arises from two components: a deterministic signal which we are really interested in, and random noise that comprises the residual error. The challenge of statistical inference is to appropriately identify the two, and not be misled into thinking that noise is actually a signal.

**[Simpson’s paradox](./14-CHAPTER_4__What_Causes_What_.md#gloss-138)**: when an apparent relationship reverses its sign when a confounding variable is taken into account.

**[size of a test](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-139)**: the Type I error rate of a statistical test, generally denoted by α.

**[skewed distribution](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-140)**: when a sample or population distribution is highly asymmetric, and has a long left- or right-hand tail. This might typically occur for variables such as income and sales of books, when there is extreme inequality. Standard measures (such as means) and standard deviations can be very misleading for such distributions.

**[Spearman’s rank correlation](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-141)**: the rank of an observation is its position in the ordered set, where ‘ties’ are considered to have the same rank. For example, for the data (3, 2, 1, 0, 1) the ranks are (5, 4, 2.5, 1, 2.5). Spearman’s rank correlation is simply the Pearson’s correlation when the _x_s and _y_s are replaced by their respective ranks.

**[specificity](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-142)**: the proportion of ‘negative’ cases that are correctly identified by a classifier or test. One minus specificity is also known as the observed Type I error, or false-positive rate.

**[standard deviation](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-143)**: the square root of the variance of a sample or distribution. For well-behaved, reasonably symmetric data distributions without long tails, we would expect most of the observations to lie within two sample standard deviations from the sample mean.

**[standard error](./19-CHAPTER_9__Putting_Probability_and_Statistics_Together.md#gloss-144)**: the standard deviation of a sample mean, when considered as a random variable. Suppose _X_1, _X_2 … _X_ _n_ are independent and identically distributed random variables drawn from a population distribution with mean μ and standard deviation σ. Then their average _Y_ \= (_X_1 \+ _X_2 \+ … + _X_ _n_ _)/n_ has mean μ and variance σ2/_n_. The standard deviation of _Y_ is σ/√_n_, known as the standard error, and estimated by _s_/√_n_, where _s_ is the sample standard deviation of the observed _X_’s.

**[statistic](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-145)**: a meaningful number derived from a set of data.

**[statistical inference](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-146)**: the process of using sample data to learn about unknown parameters underlying a statistical model.

**[statistical model](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#gloss-147)**: a mathematical representation, containing unknown parameters, of the probability distribution of a set of random variables.

**[statistical science](./10-INTRODUCTION.md#gloss-148)**: the discipline of learning about the world from data, typically involving a problem-solving cycle such as PPDAC.

**[statistical significance](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-149)**: an observed effect is judged to be statistically significant when its P-value corresponding to a null hypothesis is less than some pre-specified level, say 0.05 or 0.001, meaning such an extreme result was unlikely to occur were the null hypothesis, and all other modelling assumptions, to hold.

**[supervised learning](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-150)**: construction of a classification algorithm based on cases with confirmed membership of classes.

**[_t_\-statistic](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-151)**: a test statistic used to test a null hypothesis of a parameter being zero, formed by the ratio of an estimate to its standard error. For large samples, values of above 2 or below −2 correspond to a two-sided P-value of 0.05; exact P-values can be obtained from statistical software.

**[Type I error](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-152)**: when a true null hypothesis is incorrectly rejected in favour of an alternative, so a false-positive claim is made.

**[Type II error](./20-CHAPTER_10__Answering_Questions_and_Claiming_Discoveries.md#gloss-153)**: when an alternative hypothesis is true, but a hypothesis test does not reject the null hypothesis, so the conclusion is a false-negative.

**[unsupervised learning](./16-CHAPTER_6__Algorithms,_Analytics_and_Prediction.md#gloss-154)**: identification of classes based on cases with no identified membership, using some form of clustering procedure.

**[variability](./10-INTRODUCTION.md#gloss-155)**: the inevitable differences that occur between measurements or observations, some of which may be explained by known factors, and the remainder attributed to random noise.

**[variance](./45-footnotes.md#gloss-156)**: for a sample _x_1 … _x_ _n_ with mean ![](./images/p391_g.png), this is generally defined as ![](./images/p405_a.png) (although the denominator can also be _n_ rather than _n_ – 1). For a random variable _X_ with mean μ, the variance is _V_(_X_) = _E_(_X_ – μ)2. The standard deviation is the square root of the variance, so ![](./images/p405_b.png).

**[wisdom of crowds](./12-CHAPTER_2__Summarizing_and_Communicating_Numbers._Lots_of_Numbers.md#gloss-157)**: the idea that a summary derived from a group opinion is closer to the truth than the majority of the individuals.

**[Z-score](./13-CHAPTER_3__Why_Are_We_Looking_at_Data_Anyway__Populations_and_Measurement.md#gloss-158)**: a means of standardizing an observation _x_ _i_ in terms of its distance from the sample mean _m_ expressed in terms of sample standard deviations _s_, so that _z_ _i_ \= (_x_ _i_ – _m_)/_s_. An observation with a Z-score of 3 corresponds to being 3 standard deviations above the mean, which is a fairly extreme outlier. A Z-score can also be defined in terms of a population mean μ and standard deviation σ, in which case _z_ _i_ \= (_x_ _i_ – μ)/σ.
