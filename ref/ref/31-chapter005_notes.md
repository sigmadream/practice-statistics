### CHAPTER 5: MODELLING RELATIONSHIPS USING REGRESSION

[1](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#ch5_end1). M. Friendly _et al_., ‘HistData: Data Sets from the History of Statistics and Data Visualization’ (2018), <https://CRAN.R-project.org/package=HistData>.

[2](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#ch5_end2). J. Pearl and D. Mackenzie, _The Book of Why: The New Science of Cause and Effect_ (Basic Books, 2018), p. 471.

[3](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#ch5_end3). For a fascinating discussion of the risk of modelling, see A. Aggarwal _et al._, ‘Model Risk – Daring to Open Up the Black Box’, _British Actuarial Journal_ 21:2 (2016), 229–96.

[4](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#ch5_end4). Essentially we are saying that changes will be correlated with a baseline measure, even if there is really no real change in the underlying process. We can express this mathematically. Suppose I take an observation at random from a population distribution, call it _X_. Then I take another independent observation from the same distribution, call it _Y_, and look at their difference: _Y_ − _X_. Then it is a rather remarkable fact that the correlation between their difference, _Y_ − _X_, and the first measurement, _X_, is −1/√2 \= −0.71, regardless of the form of the underlying population distribution. For example, if a woman has a child, and then her friend has one, and they see how much heavier the friend’s baby is by taking the weight of the second minus the weight of the first, then this difference has a correlation of −0.71 with the weight of the first baby. This is because, if the first child is light, we expect the second to be heavier just by chance alone, and so the difference would be positive. And if the first child is heavy, then we expect the difference between the weights to be negative.

[5](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#ch5_end5). L. Mountain, ‘Safety Cameras: Stealth Tax or Life-Savers?’, _Significance_ 3 (2006), 111–13.

[6](./15-CHAPTER_5__Modelling_Relationships_Using_Regression.md#ch5_end6). The table below shows the forms of multiple regression used for different types of dependent variable. Each results in a regression coefficient being estimated for each explanatory variable.

| Type of dependent variable | Type of regression | Interpretation of coefficient |
| -------------------------- | ------------------ | ----------------------------- |
| Continuous variables       | Multiple linear    | Gradient                      |
| Events or proportions      | Logistic           | Log(odds ratio)               |
| Counts                     | Poisson            | Log(rate ratio)               |
| Length of survival         | Cox                | Log(hazard ratio)             |
