# 부록. 권장 전처리 (Appendix. Recommended Preprocessing)

필요한 전처리의 유형은 피팅되는 모델의 유형에 따라 달라집니다. 예를 들어, 거리 함수(distance functions)나 내적(dot products)을 사용하는 모델은 거리가 적절하게 측정되도록 모든 예측 변수가 동일한 척도(scale)를 가져야 합니다.

[tinymodels 웹사이트](https://oreil.ly/Eco3u)에서 이러한 각 모델과 사용 가능한 다른 모델에 대해 자세히 알아볼 수 있습니다.

이 부록에서는 다양한 모델 함수에 필요한 기본 수준의(baseline levels of) 전처리에 대한 권장 사항을 제공합니다. [표 A-1](#preprocessing)에서 전처리 방법은 다음과 같이 분류됩니다.

더미 (Dummy)  
질적(qualitative) 예측 변수에 숫자 인코딩(더미 변수 또는 기타 방법)이 필요합니까?

ZV  
단일 고유값(즉, 분산이 0인)을 가진 열을 제거해야 합니까?

대체 (Impute)  
일부 예측 변수가 누락된 경우 대체를 통해 추정(estimated via imputation)해야 합니까?

상관관계 제거 (Decorrelate)  
상관관계가 있는 예측 변수가 있는 경우 이 상관관계를 완화(mitigated)해야 합니까? 이는 예측 변수를 필터링하거나, 주성분 분석(principal component analysis)을 사용하거나, 모델 기반 기법(정규화)을 사용하는 것을 의미할 수 있습니다.

정규화 (Normalize)  
예측 변수의 중심을 맞추고 크기를 조정(centered and scaled)해야 합니까?

변환 (Transform)  
예측 변수를 보다 대칭적(symmetric)이도록 변환하는 것이 도움이 됩니까?

[표 A-1](#preprocessing)의 정보가 완전한(exhaustive) 것은 아니며 구현(implementation)에 따라 다소 달라질 수 있습니다. 예를 들어, 표의 각주(footnotes)에 언급된 바와 같이 일부 모델은 특정 전처리 작업이 필요하지 않을 수 있지만 구현에는 필요할 수 있습니다. 표에서 ✓는 해당 모델에 방법이 필요함을 나타내고 ×는 그렇지 않음을 나타냅니다. ◌ 기호는 모델이 이 기법의 도움을 받을 _수도_ 있지만 필수(required)는 아님을 의미합니다.

<table id="preprocessing" style="width:100%;">
<caption>표 A-1. 여러 가지 모델에 대한 전처리 방법</caption>
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
<th>모델</th>
<th>더미</th>
<th>ZV</th>
<th>대체</th>
<th>상관관계 제거</th>
<th>정규화</th>
<th>변환</th>
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
<td colspan="7"><p><sup><a href="app01.xhtml#fn1-marker">a</a></sup> 예측 변수의 상관관계를 제거하는 것은 성능 향상에 도움이 되지 않을 수 있습니다. 그러나 상관관계가 있는 예측 변수가 적을수록 변수 중요도 점수의 추정을 개선할 수 있습니다; <a href="https://oreil.ly/PDIm1">Kuhn and Johnson (2020)</a>의 그림 11.4를 참조하세요. 본질적으로 고도로 상관된 예측 변수의 선택은 거의 무작위(random)입니다.</p>
<p><sup><a href="app01.xhtml#fn2-marker">b</a></sup> 이러한 모델에 필요한 전처리는 구현(implementation)에 따라 다릅니다. 구체적으로: <em>이론적으로(theoretically)</em> 모든 트리 기반 모델은 대체를 요구하지 않습니다. 그러나 많은 트리 앙상블 구현에서는 대체가 필요합니다. 트리 기반 부스팅 방법은 일반적으로 더미 변수 생성을 요구하지 않지만 <code>xgboost</code> 엔진을 사용하는 모델은 더미 변수 생성을 요구합니다.</p></td>
</tr>
</tbody>
</table>
