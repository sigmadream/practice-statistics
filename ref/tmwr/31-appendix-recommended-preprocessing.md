# Appendix. Recommended Preprocessing

The type of preprocessing needed depends on the type of model being fit. For example, models that use distance functions or dot products should have all of their predictors on the same scale so that distance is measured appropriately.

You can learn more about each of these models, and others that might be available, at the [tinymodels website](https://oreil.ly/Eco3u).

This Appendix provides recommendations for baseline levels of preprocessing that are needed for various model functions. In <a href="#preprocessing" data-type="xref">Table A-1</a>, the preprocessing methods are categorized as:

Dummy  
Do qualitative predictors require a numeric encoding (e.g., via dummy variables or other methods)?

ZV  
Should columns with a single unique (i.e., zero variance) value be removed?

Impute  
If some predictors are missing, should they be estimated via imputation?

Decorrelate  
If there are correlated predictors, should this correlation be mitigated? This might mean filtering out predictors, using principal component analysis, or a model-based technique (e.g., regularization).

Normalize  
Should predictors be centered and scaled?

Transform  
Is it helpful to transform predictors to be more symmetric?

The information in <a href="#preprocessing" data-type="xref">Table A-1</a> is not exhaustive and somewhat depends on the implementation. For example, as noted in the table’s footnotes, some models may not require a particular preprocessing operation but the implementation may require it. In the table, ✓ indicates that the method is required for the model and × indicates that it is not. The ◌ symbol means that the model *may* be helped by the technique but it is not required.

<table id="preprocessing" style="width:100%;">
<caption>Table A-1. Preprocessing methods for different models</caption>
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr>
<th>Model</th>
<th>Dummy</th>
<th>ZV</th>
<th>Impute</th>
<th>Decorrelate</th>
<th>Normalize</th>
<th>Transform</th>
</tr>
</thead>
<tbody>
<tr>
<td><code>bag_mars()</code></td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">✓</td>
<td class="center">◌</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>bag_tree()</code></td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">◌<sup><a href="app01.xhtml#fn1" id="fn1-marker" data-type="noteref">a</a></sup></td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>bart()</code></td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">◌<sup><a href="#fn1" class="noteref" data-type="noteref" epub:type="noteref">a</a></sup></td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>boost_tree()</code></td>
<td class="center">×<sup><a href="app01.xhtml#fn2" id="fn2-marker" data-type="noteref">b</a></sup></td>
<td class="center">◌</td>
<td class="center">✓<sup><a href="#fn2" class="noteref" data-type="noteref" epub:type="noteref">b</a></sup></td>
<td class="center">◌<sup><a href="#fn1" class="noteref" data-type="noteref" epub:type="noteref">a</a></sup></td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>C5_rules()</code></td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>cubist_rules()</code></td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>decision_tree()</code></td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">×</td>
<td class="center">◌<sup><a href="#fn1" class="noteref" data-type="noteref" epub:type="noteref">a</a></sup></td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>discrim_flexible()</code></td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>discrim_linear()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>discrim_regularized()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>gen_additive_mod()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>linear_reg()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>logistic_reg()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>mars()</code></td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">✓</td>
<td class="center">◌</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>mlp()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
</tr>
<tr>
<td><code>multinom_reg()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×<sup><a href="#fn2" class="noteref" data-type="noteref" epub:type="noteref">b</a></sup></td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>naive_Bayes()</code></td>
<td class="center">×</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">◌<sup><a href="#fn1" class="noteref" data-type="noteref" epub:type="noteref">a</a></sup></td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>nearest_neighbor()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">◌</td>
<td class="center">✓</td>
<td class="center">✓</td>
</tr>
<tr>
<td><code>pls()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">✓</td>
<td class="center">✓</td>
</tr>
<tr>
<td><code>poisson_reg()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">◌</td>
</tr>
<tr>
<td><code>rand_forest()</code></td>
<td class="center">×</td>
<td class="center">◌</td>
<td class="center">✓<sup><a href="#fn2" class="noteref" data-type="noteref" epub:type="noteref">b</a></sup></td>
<td class="center">◌<sup><a href="#fn1" class="noteref" data-type="noteref" epub:type="noteref">a</a></sup></td>
<td class="center">×</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>rule_fit()</code></td>
<td class="center">✓</td>
<td class="center">×</td>
<td class="center">✓</td>
<td class="center">◌<sup><a href="#fn1" class="noteref" data-type="noteref" epub:type="noteref">a</a></sup></td>
<td class="center">✓</td>
<td class="center">×</td>
</tr>
<tr>
<td><code>svm_*()</code></td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
<td class="center">✓</td>
</tr>
</tbody>
<tbody>
<tr class="footnotes">
<td colspan="7"><p><sup><a href="app01.xhtml#fn1-marker">a</a></sup> Decorrelating predictors may not help improve performance. However, fewer correlated predictors can improve the estimation of variable importance scores; see Fig. 11.4 of <a href="https://oreil.ly/PDIm1">Kuhn and Johnson (2020)</a>. Essentially, the selection of highly correlated predictors is almost random.</p>
<p><sup><a href="app01.xhtml#fn2-marker">b</a></sup> The needed preprocessing for these models depends on the implementation. Specifically: <em>theoretically</em>, any tree-based model does not require imputation. However, many tree ensemble implementations require imputation. While tree-based boosting methods generally do not require the creation of dummy variables, models using the <code>xgboost</code> engine do.</p></td>
</tr>
</tbody>
</table>
