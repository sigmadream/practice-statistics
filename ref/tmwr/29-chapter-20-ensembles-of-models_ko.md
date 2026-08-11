# 20장. 모델의 앙상블 (Ensembles of Models)

여러 단일 학습자(single learners)의 예측을 집계(aggregated)하여 하나의 예측을 수행하는 모델 앙상블(model ensemble)은 고성능의 최종 모델을 생성할 수 있습니다. 앙상블 모델을 만드는 가장 인기 있는 방법은 배깅(bagging)(Breiman 1996a), 랜덤 포레스트(Ho 1995; Breiman 2001a) 및 부스팅(boosting)(Freund and Schapire 1997)입니다. 이러한 각 방법은 동일한 유형의 모델(분류 트리)의 여러 버전에서 나온 예측을 결합합니다. 그러나 앙상블을 만드는 가장 초기의 방법 중 하나는 *모델 스태킹(model stacking)*입니다(Wolpert 1992; Breiman 1996b).

###### 참고 (Note)

모델 스태킹은 모든 유형의 여러 모델에 대한 예측을 결합합니다. 예를 들어 로지스틱 회귀, 분류 트리 및 서포트 벡터 머신이 스태킹 앙상블에 포함될 수 있습니다.

이 장에서는 stacks 패키지를 사용하여 예측 모델을 스태킹하는 방법을 보여줍니다. 우리는 콘크리트 혼합물(mixtures)의 압축 강도(compressive strength)를 예측하기 위해 여러 모델을 평가했던 [15장](ch15.xhtml#workflow-sets)의 결과를 재사용할 것입니다.

스택형(stacked) 앙상블을 구축하는 과정은 다음과 같습니다.

1. 홀드아웃 예측(holdout predictions, 재표집을 통해 생성됨)의 훈련 세트를 조립(Assemble)합니다.

2. 이러한 예측을 혼합(blend)하는 모델을 만듭니다.

3. 앙상블의 각 멤버(member)에 대해 원래 훈련 세트에 모델을 피팅합니다.

이후(subsequent) 섹션에서는 이 프로세스를 설명합니다. 그러나 계속 진행(proceeding)하기 전에, "모델"이 의미할 수 있는 바의 변형(variations)에 대한 몇 가지 명명법(nomenclature)을 명확히(clarify) 할 것입니다. 이는 복잡한 모델링 분석을 작업할 때 너무 많이 사용되는(overloaded) 용어가 되기 쉽습니다! [15장](ch15.xhtml#workflow-sets)에서 만든 다층 퍼셉트론(MLP) 모델(즉, 신경망)을 고려해 봅시다.

일반적으로, 우리는 MLP 모델을 모델의 *유형(type)*으로 이야기(talk about)할 것입니다. 선형 회귀와 서포트 벡터 머신도 다른 모델 유형입니다.

튜닝 매개변수는 모델의 중요한 측면(aspect)입니다. [15장](ch15.xhtml#workflow-sets)으로 돌아가서, MLP 모델은 25개의 튜닝 매개변수 값에 걸쳐 튜닝되었습니다. 이전 장에서 우리는 이러한 값을 _후보 튜닝 매개변수(candidate tuning parameter)_ 값 또는 *모델 구성(model configurations)*이라고 불렀습니다. 앙상블에 관한 문헌(literature)에서는 이를 기저(base) 모델이라고도 부릅니다.

###### 참고 (Note)

스태킹 앙상블에 포함될 수 있는 가능한 모델 구성(모든 모델 유형 중)을 설명하기 위해 *후보 멤버(candidate members)*라는 용어를 사용할 것입니다.

이는 스태킹 모델에 다양한 유형의 모델(트리 및 신경망)과 동일한 모델의 다양한 구성(깊이가 다른 트리)이 포함될 수 있음을 의미합니다.

# 스태킹을 위한 훈련 세트 생성 (Creating the Training Set for Stacking)

스택형 앙상블을 구축하기 위한 첫 번째 단계는 여러 번 분할(multiple splits)이 있는 재표집 체계(resampling scheme)에서 평가 세트 예측(assessment set predictions)을 사용(relies on)하는 것입니다. 훈련 세트의 각 데이터 포인트에 대해 스태킹은 어떤 종류의 샘플 외(out-of-sample) 예측이 필요합니다. 회귀 모델의 경우 이는 예측된 결과(outcome)입니다. 분류 모델의 경우 예측된 클래스 또는 확률을 사용할 수 있지만, 확률이 확정적인(hard) 클래스 예측보다 더 많은 정보를 포함합니다. 일련의 모델에 대해, 행은 훈련 세트 샘플이고 열은 여러 모델 세트의 샘플 외 예측이 되는 데이터 세트가 조립(assembled)됩니다.

[15장](ch15.xhtml#workflow-sets)으로 돌아가서, 데이터를 재표집하기 위해 10-폴드 교차 검증을 5번 반복(five repeats)하여 사용했습니다. 이 재표집 체계는 각 훈련 세트 샘플에 대해 5개의 평가 세트 예측을 생성합니다. 여러 다른 재표집 기법(부트스트래핑)에서 여러 샘플 외 예측이 발생할 수 있습니다. 스태킹을 위해, 훈련 세트의 데이터 포인트에 대한 어떠한(any) 반복(replicate) 예측이든 후보 멤버당 훈련 세트 샘플당 단일 예측이 있도록 평균(averaged)을 냅니다.

###### 참고 (Note)

단순 검증 세트도 tidymodels에서는 단일 재표집으로 간주(considers)하므로 스태킹에 사용할 수 있습니다.

콘크리트 예제의 경우, 모델 스태킹에 사용되는 훈련 세트에는 모든 후보 튜닝 매개변수 결과에 대한 열이 있습니다. [표 20-1](#ensemble-candidate-preds)은 처음 여섯 행과 선택된 열을 나타냅니다(presents).

| 샘플 \# | 배깅된 트리 | MARS 1 | MARS 2 | Cubist 1 | …   | Cubist 25 | …   |
| ------- | ----------- | ------ | ------ | -------- | --- | --------- | --- |
| 1       | 25.18       | 17.92  | 17.21  | 17.79    |     | 17.82     |     |
| 2       | 5.18        | -1.77  | -0.74  | 2.83     |     | 3.87      |     |
| 3       | 9.71        | 7.26   | 5.91   | 6.31     |     | 8.60      |     |
| 4       | 25.21       | 20.93  | 21.52  | 23.72    |     | 21.61     |     |
| 5       | 6.33        | 1.53   | 0.14   | 3.60     |     | 4.57      |     |
| 6       | 7.88        | 4.88   | 1.74   | 7.69     |     | 7.55      |     |

표 20-1. 후보 튜닝 매개변수 구성의 예측값 {#ensemble-candidate-preds}

배깅된 트리(bagged tree) 모델은 튜닝 매개변수가 없으므로 단일 열만 있습니다. 또한, MARS는 두 가지 가능한 구성으로 단일 매개변수(곱의 차수, product degree)에 걸쳐 튜닝되었으므로 이 모델은 두 개의 열로 나타납니다(represented). 이 예제의 Cubist에서 볼 수 있듯이 대부분의 다른 모델에는 25개의 해당 열이 있습니다.

###### 경고 (Warning)

분류 모델의 경우 후보 예측 열은 예측된 클래스 확률이 됩니다. 이 열들은 각 모델에 대해 합이 1이 되므로 하나의 클래스에 대한 확률은 생략(left out)될 수 있습니다.

지금까지의 위치를 요약(summarize)하자면, 스태킹의 첫 번째 단계는 각 후보 모델의 훈련 세트에 대한 평가 세트 예측을 조립하는 것입니다. 우리는 이러한 평가 세트 예측을 사용하여 스택형 앙상블을 계속해서 구축(move forward and build)할 수 있습니다.

stacks 패키지로 앙상블(ensembling)을 시작하려면 `stacks()` 함수를 사용하여 빈(empty) 데이터 스택을 생성한 다음 후보 모델을 추가합니다(add). 워크플로우 세트(workflow sets)를 사용하여 이러한 데이터에 다양한 모델을 피팅했음을 상기하십시오. 우리는 레이싱(racing) 결과를 사용할 것입니다.

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

이 경우 구문(syntax)은 다음과 같습니다.

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

레이싱 기법([13장](ch13.xhtml#grid-search)에서 소개됨)은 모든 재표집에 대해 모든 구성을 평가하지 않을 수 있기 때문에 더 효율적이라는 것을 상기하십시오(Recall). 스태킹은 모든 후보 멤버가 전체(complete) 재표집 세트를 가질 것을 요구합니다. `add_candidates()`는 완전한(complete) 결과가 있는 모델 구성만 포함합니다.

###### 참고 (Note)

`grid_results`에 포함된 전체 후보 모델 세트 대신 레이싱 결과를 사용하는 이유는 무엇입니까? 어느 쪽이든 사용할 수 있습니다. 레이싱 결과를 사용하여 이 데이터에서 더 나은 성능을 찾았습니다. 이는 레이싱 기법이 더 큰 그리드에서 최적의 모델(들)을 미리 선택하기 때문일 수 있습니다.

workflowsets 패키지를 사용하지 않은 경우, tune 및 finetune의 객체를 `add_candidates()`에 전달할(passed) 수도 있습니다. 여기에는 그리드 검색 객체와 반복 검색(iterative search) 객체가 모두 포함될 수 있습니다.

# 예측 혼합하기 (Blend the Predictions)

훈련 세트 예측과 이에 상응하는(corresponding) 관찰 결과 데이터는 평가 세트 예측이 관찰된 결과 데이터의 예측 변수가 되는 *메타 학습 모델(meta-learning model)*을 생성하는 데 사용됩니다. 메타 학습은 모든 모델을 사용하여 수행할(accomplished) 수 있습니다. 가장 일반적으로 사용되는 모델은 선형, 로지스틱 및 다항(multinomial) 모델을 포괄하는(encompasses) 정규화된 일반화 선형 모델(regularized generalized linear model)입니다. 특히, 축소(shrinkage)를 사용하여 점(points)을 중심 값으로 끌어당기는(pull) 올가미 패널티(lasso penalty)(Tibshirani 1996)를 통한 정규화에는 몇 가지 이점(advantages)이 있습니다.

- 올가미 패널티를 사용하면 앙상블에서 후보(때로는 전체 모델 유형)를 제거할 수 있습니다.

- 앙상블 후보 간의 상관관계(correlation)는 매우 높은 경향이 있으며, 정규화는 이 문제를 완화하는(alleviate) 데 도움이 됩니다.

Breiman (1996b)은 또한 선형 모델을 사용하여 예측을 혼합할 때 혼합 계수(blending coefficients)를 음수가 아니도록(nonnegative) 제한(constrain)하는 것이 유용할 수 있다고 제안했습니다. 우리는 일반적으로 이것이 좋은 조언(advice)임을 알게 되었으며, 이것이 stacks 패키지의 기본값(default)입니다(그러나 선택적 인자를 통해 변경할 수 있습니다).

결과(outcome)가 숫자형(numeric)이므로 메타 모델(metamodel)에는 선형 회귀가 사용됩니다. 메타 모델을 피팅하는 것은 다음을 사용하는 것만큼 간단합니다(straightforward):

```
set.seed(2001)
ens <- blend_predictions(concrete_stack)
```

이것은 사전 정의된 올가미 패널티 값의 그리드에 대해 메타 학습 모델을 평가하고, 내부(internal) 재표집 방법을 사용하여 가장 좋은 값을 결정합니다. [그림 20-1](#stacking-autoplot)에 표시된 `autoplot()` 메서드는 기본 페널티(penalization) 방법이 충분(sufficient)했는지 파악(understand)하는 데 도움이 됩니다.

```
autoplot(ens)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2001.png" alt="tmwr 2001" />
<h6 id="figure-20-1.-results-of-using-the-autoplot-method-on-the-blended-stacks-object.">그림 20-1. 혼합된 stacks 객체에 <code>autoplot()</code> 메서드를 사용한 결과.</h6>
</figure>

[그림 20-1](#stacking-autoplot)의 상단 패널은 메타 학습 모델에 의해 유지된(retained) 앙상블 후보 멤버의 평균 수를 보여줍니다. 멤버의 수가 꽤 일정(constant)하며 이것이 증가함에 따라 RMSE도 증가함을 알 수 있습니다.

여기서는 기본 범위가 그다지 유용하지 않았을(may not have served us well) 수 있습니다. 더 큰 페널티를 주어 메타 학습 모델을 평가하기 위해 추가 옵션을 전달해(pass) 보겠습니다.

```
set.seed(2002)
ens <- blend_predictions(concrete_stack, penalty = 10^seq(-2, -0.5, length = 20))
```

이제 [그림 20-2](#stacking-autoplot-redo)에서 앙상블 모델이 첫 번째 혼합(blend)보다 나빠지는(하지만 그리 크지는 않은) 범위를 확인(see)할 수 있습니다. 멤버가 많아지고 페널티가 커질수록 _R_<sup>2</sup> 값은 증가합니다.

```
autoplot(ens)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2002.png" alt="tmwr 2002" />
<h6 id="figure-20-2.-the-results-of-using-the-autoplot-method-on-the-updated-blended-stacks-object.">그림 20-2. 업데이트된 혼합 stacks 객체에 <code>autoplot()</code> 메서드를 사용한 결과.</h6>
</figure>

회귀 모델을 사용하여 예측을 혼합(blending)할 때, 혼합 매개변수가 음수가 되지 않도록 제한하는 것이 일반적(common)입니다. 이 데이터의 경우, 이러한 제약 조건(constraint)은 여러 잠재적인 앙상블 멤버를 제거하는(eliminating) 효과가 있습니다; 매우 낮은(fairly low) 패널티에서도 앙상블은 원래 18개의 일부(fraction)로 제한(limited)됩니다.

가장 작은 RMSE와 관련된 페널티 값은 0.051이었습니다. 객체를 프린트하면 메타 학습 모델의 세부 정보(details)가 표시됩니다.

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

정규화된 선형 회귀 메타 학습 모델에는 4가지 모델 유형(types of models)에 걸쳐(across) 7개의 혼합 계수(blending coefficients)가 포함(contained)되어 있습니다. `autoplot()` 메서드를 다시 사용하여 [그림 20-3](#blending-weights)을 생성하고 각 모델 유형의 기여도(contributions)를 보여줄 수 있습니다.

```
autoplot(ens, "weights") +
  geom_text(aes(x = weight + 0.01, label = model), hjust = 0) +
  theme(legend.position = "none") +
  lims(x = c(-0.01, 0.8))
```

부스팅된 트리(boosted tree) 모델과 신경망 모델이 앙상블에 가장 크게 기여(contributions)합니다. 이 앙상블의 경우 결과는 다음 방정식(equation)으로 예측됩니다.

```math
\begin{aligned}
{\text{ensemble}\text{prediction}} & {= - 0.65} \\
 + & {0.77 \times \text{boost}\text{tree}\text{prediction}} \\
 + & {0.16 \times \text{cubist}\text{rules}\text{prediction}} \\
 + & {0.044 \times \text{linear}\text{reg}\text{prediction}} \\
 + & {0.03 \times \text{mlp}\text{prediction}} \\
 + & {0.013 \times \text{mars}\text{prediction}}
\end{aligned}
```

방정식의 예측 변수는 해당 모델의 예측된 압축 강도 값입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_2003.png" alt="tmwr 2003" />
<h6 id="figure-20-3.-blending-coefficients-for-the-stacking-ensemble.">그림 20-3. 스태킹 앙상블의 혼합 계수.</h6>
</figure>

# 멤버 모델 피팅 (Fit the Member Models)

앙상블에는 7명의 후보 멤버가 포함되어 있으며 이제 그들의 예측이 앙상블에 대한 최종 예측으로 혼합될 수 있는 방법을 알게 되었습니다(know). 그러나 이러한 개별(individual) 모델 적합(fits)은 아직 생성되지 않았습니다. 스태킹 모델을 사용하려면, 7개의 추가 모델 피팅(model fits)이 필요합니다. 이는 원래(original) 예측 변수와 함께 전체(entire) 훈련 세트를 사용합니다.

피팅(fit)될 7개의 모델은 다음과 같습니다.

부스팅 (Boosting)
트리 수 = 1957, 최소 노드 크기 = 8, 트리 깊이 = 7, 학습률 = 0.0756, 최소 손실 감소 = 1.45e–07, 표집된 관측치 비율 = 0.679

큐비스트 (Cubist)
위원회(committees) 수 = 98 및 최근접 이웃(nearest neighbors) 수 = 2

선형 회귀 (이차 특성) (Linear regression (quadratic features))
정규화 양 = 6.28e–09 및 올가미 패널티 비율 = 0.636 (구성 1)

선형 회귀 (이차 특성) (Linear regression (quadratic features))
정규화 양 = 2e–09 및 올가미 패널티 비율 = 0.668 (구성 2)

신경망 (Neural network)
은닉 유닛(hidden units) 수 = 14, 정규화 양 = 0.0345, 에포크(epochs) 수 = 979 (구성 1)

신경망 (Neural network)
은닉 유닛(hidden 유닛 수 = 22, 정규화 양 = 2.08e–10, 에포크(epochs) 수 = 92 (구성 2)

신경망 (Neural network)
은닉 유닛(hidden units) 수 = 26, 정규화 양 = 0.0149, 에포크(epochs) 수 = 203 (구성 3)

stacks 패키지에는 이러한 모델을 훈련(trains)하고 반환(returns)하는 함수 `fit_members()`가 있습니다.

```
ens <- fit_members(ens)
```

이것은 각 멤버에 대해 피팅된(fitted) 워크플로우 객체로 스태킹 객체를 업데이트합니다. 이 시점(At this point)에서 스태킹 모델을 예측에 사용할 수 있습니다.

# 테스트 세트 결과 (Test Set Results)

혼합 프로세스에서 재표집(resampling)을 사용했기 때문에 7개 멤버가 있는 앙상블의 예상 RMSE가 4.12로 추정(estimate)된다고 할 수 있습니다. [15장](ch15.xhtml#workflow-sets)에서 가장 좋은 부스팅된 트리(boosted tree)의 테스트 세트 RMSE가 3.33이었음을 상기하십시오. 앙상블 모델은 테스트 세트에서 어떻게 비교될까요? `predict()`를 사용하여 알아볼(find out) 수 있습니다.

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

이것은 우리가 가진 가장 좋은 단일 모델보다 적당히(moderately) 더 좋습니다. 최상의 단일 모델과 비교할 때 스태킹이 점진적인(incremental) 이점(benefits)을 생성하는 것은 매우 일반적(fairly common)입니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 더 나은 예측 성능을 위해 여러 모델을 하나의 앙상블로 결합(combine)하는 방법을 보여주었습니다. 앙상블을 만드는 프로세스는 성능을 향상시키는 작은 부분 집합(subset)을 찾기 위해 후보 모델을 자동으로 제거할 수 있습니다. stacks 패키지에는 재표집 및 튜닝 결과를 메타 모델로 결합하기 위한 유창한(fluent) 인터페이스가 있습니다.
