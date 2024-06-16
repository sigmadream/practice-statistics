# Chapter 20. Ensembles of Models

A model ensemble, where the predictions of multiple single learners are aggregated to make one prediction, can produce a high-performance final model. The most popular methods for creating ensemble models are bagging (Breiman 1996a), random forest (Ho 1995; Breiman 2001a), and boosting (Freund and Schapire 1997). Each of these methods combines the predictions from multiple versions of the same type of model (e.g., classification trees). However, one of the earliest methods for creating ensembles is *model stacking* (Wolpert 1992; Breiman 1996b).

###### Note

Model stacking combines the predictions for multiple models of any type. For example, a logistic regression, a classification tree, and a support vector machine can be included in a stacking ensemble.

This chapter shows how to stack predictive models using the stacks package. We’ll reuse the results from <a href="ch15.xhtml#workflow-sets" data-type="xref">Chapter 15</a> where multiple models were evaluated to predict the compressive strength of concrete mixtures.

The process of building a stacked ensemble is:

1.  Assemble the training set of holdout predictions (produced via resampling).

2.  Create a model to blend these predictions.

3.  For each member of the ensemble, fit the model on the original training set.

In subsequent sections, we’ll describe this process. However, before proceeding, we’ll clarify some nomenclature for the variations of what “the model” can mean. This can quickly become an overloaded term when we are working on a complex modeling analysis! Let’s consider the multilayer perceptron (MLP) model (a.k.a. neural network) created in <a href="ch15.xhtml#workflow-sets" data-type="xref">Chapter 15</a>.

In general, we’ll talk about an MLP model as the *type* of model. Linear regression and support vector machines are other model types.

Tuning parameters are an important aspect of a model. Back in <a href="ch15.xhtml#workflow-sets" data-type="xref">Chapter 15</a>, the MLP model was tuned over 25 tuning parameter values. In the previous chapters, we’ve called these *candidate tuning parameter* values or *model configurations*. In literature on ensembling these have also been called the base models.

###### Note

We’ll use the term *candidate members* to describe the possible model configurations (of all model types) that might be included in the stacking ensemble.

This means that a stacking model can include different types of models (e.g., trees and neural networks) and different configurations of the same model (e.g., trees with different depths).

# Creating the Training Set for Stacking

The first step for building a stacked ensemble relies on the assessment set predictions from a resampling scheme with multiple splits. For each data point in the training set, stacking requires an out-of-sample prediction of some sort. For regression models, this is the predicted outcome. For classification models, the predicted classes or probabilities are available for use, although the latter contains more information than the hard class predictions. For a set of models, a data set is assembled where rows are the training set samples and columns are the out-of-sample predictions from the set of multiple models.

Back in <a href="ch15.xhtml#workflow-sets" data-type="xref">Chapter 15</a>, we used five repeats of 10-fold cross-validation to resample the data. This resampling scheme generates five assessment set predictions for each training set sample. Multiple out-of-sample predictions can occur in several other resampling techniques (e.g., bootstrapping). For the purpose of stacking, any replicate predictions for a data point in the training set are averaged so that there is a single prediction per training set sample per candidate member.

###### Note

Simple validation sets can also be used with stacking since tidymodels considers this to be a single resample.

For the concrete example, the training set used for model stacking has columns for all of the candidate tuning parameter results. <a href="#ensemble-candidate-preds" data-type="xref">Table 20-1</a> presents the first six rows and selected columns.

| Sample \# | Bagged tree | MARS 1 | MARS 2 | Cubist 1 | …   | Cubist 25 | …   |
|-----------|-------------|--------|--------|----------|-----|-----------|-----|
| 1         | 25.18       | 17.92  | 17.21  | 17.79    |     | 17.82     |     |
| 2         | 5.18        | -1.77  | -0.74  | 2.83     |     | 3.87      |     |
| 3         | 9.71        | 7.26   | 5.91   | 6.31     |     | 8.60      |     |
| 4         | 25.21       | 20.93  | 21.52  | 23.72    |     | 21.61     |     |
| 5         | 6.33        | 1.53   | 0.14   | 3.60     |     | 4.57      |     |
| 6         | 7.88        | 4.88   | 1.74   | 7.69     |     | 7.55      |     |

Table 20-1. Predictions from candidate tuning parameter configurations {#ensemble-candidate-preds}

There is a single column for the bagged tree model since it has no tuning parameters. Also, recall that MARS was tuned over a single parameter (the product degree) with two possible configurations, so this model is represented by two columns. Most of the other models have 25 corresponding columns, as shown for Cubist in this example.

###### Warning

For classification models, the candidate prediction columns would be predicted class probabilities. Since these columns add to one for each model, the probabilities for one of the classes can be left out.

To summarize where we are so far, the first step to stacking is to assemble the assessment set predictions for the training set from each candidate model. We can use these assessment set predictions to move forward and build a stacked ensemble.

To start ensembling with the stacks package, create an empty data stack using the `stacks()` function and then add candidate models. Recall that we used workflow sets to fit a wide variety of models to these data. We’ll use the racing results:

```
race_results
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result
#>   <chr>       <list>           <list>    <list>
#> 1 MARS        <tibble [1 × 4]> <opts[3]> <race[+]>
#> 2 CART        <tibble [1 × 4]> <opts[3]> <race[+]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[3]> <rsmp[+]>
#> 4 RF          <tibble [1 × 4]> <opts[3]> <race[+]>
#> 5 boosting    <tibble [1 × 4]> <opts[3]> <race[+]>
#> 6 Cubist      <tibble [1 × 4]> <opts[3]> <race[+]>
#> # … with 6 more rows
```

In this case, our syntax is:

```
library(tidymodels)
library(stacks)
tidymodels_prefer()

concrete_stack <-
  stacks() %>%
  add_candidates(race_results)

concrete_stack
#> # A data stack with 12 model definitions and 21 candidate members:
#> #   MARS: 1 model configuration
#> #   CART: 1 model configuration
#> #   CART_bagged: 1 model configuration
#> #   RF: 1 model configuration
#> #   boosting: 1 model configuration
#> #   Cubist: 1 model configuration
#> #   SVM_radial: 1 model configuration
#> #   SVM_poly: 1 model configuration
#> #   KNN: 3 model configurations
#> #   neural_network: 4 model configurations
#> #   full_quad_linear_reg: 5 model configurations
#> #   full_quad_KNN: 1 model configuration
#> # Outcome: compressive_strength (numeric)
```

Recall that racing methods (introduced in <a href="ch13.xhtml#grid-search" data-type="xref">Chapter 13</a>) are more efficient since they might not evaluate all configurations on all resamples. Stacking requires that all candidate members have the complete set of resamples. `add_candidates()` includes only the model configurations that have complete results.

###### Note

Why use the racing results instead of the full set of candidate models contained in `grid_results`? Either can be used. We found better performance for these data using the racing results. This might be due to the racing method preselecting the best model(s) from the larger grid.

If we had not used the workflowsets package, objects from the tune and finetune could also be passed to `add_candidates()`. This can include both grid and iterative search objects.

# Blend the Predictions

The training set predictions and the corresponding observed outcome data are used to create a *meta-learning model* where the assessment set predictions are the predictors of the observed outcome data. Meta-learning can be accomplished using any model. The most commonly used model is a regularized generalized linear model, which encompasses linear, logistic, and multinomial models. Specifically, regularization via the lasso penalty (Tibshirani 1996), which uses shrinkage to pull points toward a central value, has several advantages:

- Using the lasso penalty can remove candidates (and sometimes whole model types) from the ensemble.

- The correlation between ensemble candidates tends to be very high, and regularization helps alleviate this issue.

Breiman (1996b) also suggested that, when a linear model is used to blend the predictions, it might be helpful to constrain the blending coefficients to be nonnegative. We have generally found this to be good advice, and it is the default for the stacks package (but it can be changed via an optional argument).

Since our outcome is numeric, linear regression is used for the metamodel. Fitting the metamodel is as straightforward as using:

```
set.seed(2001)
ens <- blend_predictions(concrete_stack)
```

This evaluates the meta-learning model over a predefined grid of lasso penalty values and uses an internal resampling method to determine the best value. The `autoplot()` method, shown in <a href="#stacking-autoplot" data-type="xref">Figure 20-1</a>, helps us understand if the default penalization method was sufficient:

```
autoplot(ens)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2001.png" alt="tmwr 2001" />
<h6 id="figure-20-1.-results-of-using-the-autoplot-method-on-the-blended-stacks-object.">Figure 20-1. Results of using the <code>autoplot()</code> method on the blended stacks object.</h6>
</figure>

The top panel of <a href="#stacking-autoplot" data-type="xref">Figure 20-1</a> shows the average number of candidate ensemble members retained by the meta-learning model. We can see that the number of members is fairly constant and, as it increases, the RMSE also increases.

The default range may not have served us well here. To evaluate the meta-learning model with larger penalties, let’s pass an additional option:

```
set.seed(2002)
ens <- blend_predictions(concrete_stack, penalty = 10^seq(-2, -0.5, length = 20))
```

Now, in <a href="#stacking-autoplot-redo" data-type="xref">Figure 20-2</a>, we see a range where the ensemble model becomes worse than with our first blend (but not by much). The *R*<sup>2</sup> values increase with more members and larger penalties:

```
autoplot(ens)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2002.png" alt="tmwr 2002" />
<h6 id="figure-20-2.-the-results-of-using-the-autoplot-method-on-the-updated-blended-stacks-object.">Figure 20-2. The results of using the <code>autoplot()</code> method on the updated blended stacks object.</h6>
</figure>

When blending predictions using a regression model, it is common to constrain the blending parameters to be nonnegative. For these data, this constraint has the effect of eliminating many of the potential ensemble members; even at fairly low penalties, the ensemble is limited to a fraction of the original 18.

The penalty value associated with the smallest RMSE was 0.051. Printing the object shows the details of the meta-learning model:

```
ens
#> ── A stacked ensemble model ─────────────────────────
#>
#> Out of 21 possible candidate members, the ensemble retained 7.
#> Penalty: 0.0513483290743755.
#> Mixture: 1.
#>
#> The 7 highest weighted members are:
#> # A tibble: 7 × 3
#>   member                    type          weight
#>   <chr>                     <chr>          <dbl>
#> 1 boosting_1_04             boost_tree   0.727
#> 2 neural_network_1_17       mlp          0.101
#> 3 Cubist_1_25               cubist_rules 0.0906
#> 4 neural_network_1_04       mlp          0.0820
#> 5 full_quad_linear_reg_1_16 linear_reg   0.0176
#> 6 full_quad_linear_reg_1_17 linear_reg   0.00284
#> # … with 1 more row
#>
#> Members have not yet been fitted with `fit_members()`.
```

The regularized linear regression meta-learning model contained seven blending coefficients across four types of models. The `autoplot()` method can be used again to show the contributions of each model type, to produce <a href="#blending-weights" data-type="xref">Figure 20-3</a>:

```
autoplot(ens, "weights") +
  geom_text(aes(x = weight + 0.01, label = model), hjust = 0) +
  theme(legend.position = "none") +
  lims(x = c(-0.01, 0.8))
```

The boosted tree and neural network models have the largest contributions to the ensemble. For this ensemble, the outcome is predicted with the equation:

``` math
\begin{aligned}
{\text{ensemble}\text{prediction}} & {= - 0.65} \\
 + & {0.77 \times \text{boost}\text{tree}\text{prediction}} \\
 + & {0.16 \times \text{cubist}\text{rules}\text{prediction}} \\
 + & {0.044 \times \text{linear}\text{reg}\text{prediction}} \\
 + & {0.03 \times \text{mlp}\text{prediction}} \\
 + & {0.013 \times \text{mars}\text{prediction}}
\end{aligned}
```

where the predictors in the equation are the predicted compressive strength values from those models.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2003.png" alt="tmwr 2003" />
<h6 id="figure-20-3.-blending-coefficients-for-the-stacking-ensemble.">Figure 20-3. Blending coefficients for the stacking ensemble.</h6>
</figure>

# Fit the Member Models

The ensemble contains seven candidate members, and we now know how their predictions can be blended into a final prediction for the ensemble. However, these individual model fits have not yet been created. To be able to use the stacking model, seven additional model fits are required. These use the entire training set with the original predictors.

The seven models to be fit are:

Boosting  
Number of trees = 1957, minimal node size = 8, tree depth = 7, learning rate = 0.0756, minimum loss reduction = 1.45e–07, and proportion of observations sampled = 0.679

Cubist  
Number of committees = 98 and number of nearest neighbors = 2

Linear regression (quadratic features)  
Amount of regularization = 6.28e–09 and proportion of lasso penalty = 0.636 (config 1)

Linear regression (quadratic features)  
Amount of regularization = 2e–09 and proportion of lasso penalty = 0.668 (config 2)

Neural network  
Number of hidden units = 14, amount of regularization = 0.0345, and number of epochs = 979 (config 1)

Neural network  
Number of hidden units = 22, amount of regularization = 2.08e–10, and number of epochs = 92 (config 2)

Neural network  
Number of hidden units = 26, amount of regularization = 0.0149, and number of epochs = 203 (config 3)

The stacks package has a function, `fit_members()`, that trains and returns these models:

```
ens <- fit_members(ens)
```

This updates the stacking object with the fitted workflow objects for each member. At this point, the stacking model can be used for prediction.

# Test Set Results

Since the blending process used resampling, we can estimate that the ensemble with seven members had an estimated RMSE of 4.12. Recall from <a href="ch15.xhtml#workflow-sets" data-type="xref">Chapter 15</a> that the best boosted tree had a test set RMSE of 3.33. How will the ensemble model compare on the test set? We can `predict()` to find out:

```
reg_metrics <- metric_set(rmse, rsq)
ens_test_pred <-
  predict(ens, concrete_test) %>%
  bind_cols(concrete_test)

ens_test_pred %>%
  reg_metrics(compressive_strength, .pred)
#> # A tibble: 2 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard       3.33
#> 2 rsq     standard       0.957
```

This is moderately better than our best single model. It is fairly common for stacking to produce incremental benefits when compared to the best single model.

# Chapter Summary

This chapter demonstrated how to combine different models into an ensemble for better predictive performance. The process of creating the ensemble can automatically eliminate candidate models to find a small subset that improves performance. The stacks package has a fluent interface for combining resampling and tuning results into a metamodel.
