# 14장. 반복 검색 (Iterative Search)

[13장](ch13.xhtml#grid-search)에서는 그리드 검색이 사전에 정의된 후보 값 세트를 가져와서 평가한 다음 최상의 설정을 선택하는 방법을 시연했습니다. 반복 검색(Iterative search) 방법은 다른 전략을 추구합니다(pursue). 검색 프로세스 중에 다음 번에 테스트할 값을 예측합니다.

###### 참고 (Note)

그리드 검색이 불가능(infeasible)하거나 비효율적일 때, 반복 방법은 튜닝 매개변수를 최적화하기 위한 합리적인(sensible) 접근 방식입니다.

이 장에서는 두 가지 검색 방법을 간략히(outlines) 설명합니다. 첫째, 통계적 모델을 사용하여 더 나은 매개변수 설정을 예측하는 *베이지안 최적화(Bayesian optimization)*에 대해 논의합니다. 그 후 이 장에서는 *시뮬레이티드 어닐링(simulated annealing)*이라는 전역(global) 검색 방법을 설명합니다.

설명을 위해 이전 장과 동일한 세포 특성(characteristics)에 대한 데이터를 사용하지만 모델은 변경합니다. 이 장에서는 검색 프로세스의 멋진 2차원 시각화를 제공하기 때문에 서포트 벡터 머신(support vector machine) 모델을 사용합니다.

# 서포트 벡터 머신 모델 (A Support Vector Machine Model)

순차적 튜닝 방법을 시연하기 위한 서포트 벡터 머신(SVM) 모델로 모델링하기 위해, 우리는 [13장](ch13.xhtml#grid-search)에서 설명한 세포 분할(segmentation) 데이터를 다시 한 번 사용합니다. 이 모델에 대한 자세한 내용은 Kuhn and Johnson (2013)을 참조하십시오. 최적화해야 할 두 가지 튜닝 매개변수는 SVM 비용(cost) 값과 방사 기저 함수(radial basis function) 커널 매개변수 $`\sigma`$입니다. 두 매개변수 모두 모델 복잡성과 성능에 심오한(profound) 영향을 미칠 수 있습니다.

SVM 모델은 내적(dot product)을 사용하므로 이러한 이유로 예측 변수를 중심화(center)하고 척도화(scale)해야 합니다. 다층 퍼셉트론 모델과 마찬가지로 이 모델은 PCA 피처 추출을 사용하면 이점을 얻을(benefit from) 수 있습니다. 그러나 이 장에서는 검색 프로세스를 2차원으로 시각화할 수 있도록 이 세 번째 튜닝 매개변수를 사용하지 않을 것입니다.

([13장](ch13.xhtml#grid-search)의 요약에 표시된) 이전에 사용된 객체들과 함께, tidymodels 객체 `svm_rec`, `svm_spec`, `svm_wflow`가 모델 프로세스를 정의합니다:

```
library(tidymodels)
tidymodels_prefer()

svm_rec <-
  recipe(class ~ ., data = cells) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

svm_spec <-
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wflow <-
  workflow() %>%
  add_model(svm_spec) %>%
  add_recipe(svm_rec)
```

두 개의 튜닝 매개변수 `cost`와 `rbf_sigma`에 대한 기본 매개변수 범위는 다음과 같습니다:

```
cost()
#> Cost (quantitative)
#> Transformer: log-2 [1e-100, Inf]
#> Range (transformed scale): [-10, 5]
rbf_sigma()
#> Radial Basis Function sigma (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-10, 0]
```

설명을 위해, 검색의 시각화를 개선하도록 커널 매개변수 범위를 약간 변경해 보겠습니다:

```
svm_param <-
  svm_wflow %>%
  extract_parameter_set_dials() %>%
  update(rbf_sigma = rbf_sigma(c(-7, -1)))
```

반복 검색과 작동 방식에 대한 구체적인 세부 사항을 논의하기 전에, 이 특정 데이터 세트에 대한 두 SVM 튜닝 매개변수와 ROC 곡선 아래 면적 간의 관계를 살펴보겠습니다(explore). 우리는 2,500개의 후보 값으로 구성된 매우 큰 정규 그리드를 구성하고 리샘플링을 사용하여 그리드를 평가했습니다. 이것은 분명히 일반적인 데이터 분석에서는 비현실적(impractical)이고 엄청나게 비효율적입니다. 그러나 검색 프로세스가 취해야 할 경로(path)와 수치적으로 최적의 값이 어디에서 나타나는지(occur)를 명확히 보여줍니다(elucidates).

[그림 14-1](#roc-surface)은 이 그리드를 평가한 결과를 보여주며, 색상이 밝을수록 더 높은(더 나은) 모델 성능에 해당합니다. 매개변수 공간의 대각선 아래쪽(lower diagonal)에 상대적으로 평평하고 성능이 좋지 않은 넓은 띠(swath)가 있습니다. 최상의 성능을 보여주는 능선(ridge)은 공간의 오른쪽 위 부분에 발생합니다. 검은색 점은 최상의 설정을 나타냅니다. 낮은 결과의 고원(plateau)에서 최상의 성능 능선으로의 전환은 매우 날카롭습니다(sharp). 또한 능선 바로 오른쪽에서 ROC 곡선 아래 면적이 급격히(sharp) 떨어집니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1401.png" alt="tmwr 1401" />
<h6 id="figure-14-1.-heatmap-of-the-mean-area-under-the-roc-curve-for-a-high-density-grid-of-tuning-parameter-values.-the-best-point-is-a-solid-dot-in-the-upper-right-corner.">그림 14-1. 튜닝 매개변수 값의 고밀도 그리드에 대한 ROC 곡선 아래 평균 면적의 히트맵. 최상의 지점은 오른쪽 위 모서리에 있는 실선 원(solid dot)입니다.</h6>
</figure>

다음의 검색 절차는 진행하기 전에 최소한 일부 리샘플링된 성능 통계량이 필요합니다. 이 목적을 위해 다음 코드는 매개변수 공간의 평평한(flat) 부분에 존재하는 작은 정규 그리드를 생성합니다. `tune_grid()` 함수는 이 그리드를 리샘플링합니다:

```
set.seed(1401)
start_grid <-
  svm_param %>%
  update(
    cost = cost(c(-6, 1)),
    rbf_sigma = rbf_sigma(c(-6, -4))
  ) %>%
  grid_regular(levels = 2)

set.seed(1402)
svm_initial <-
  svm_wflow %>%
  tune_grid(resamples = cell_folds, grid = start_grid, metrics = roc_res)

collect_metrics(svm_initial)
#> # A tibble: 4 × 8
#>     cost rbf_sigma .metric .estimator  mean     n std_err .config
#>    <dbl>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>
#> 1 0.0156  0.000001 roc_auc binary     0.864    10 0.00864 Prepro...
#> 2 2       0.000001 roc_auc binary     0.863    10 0.00867 Prepro...
#> 3 0.0156  0.0001   roc_auc binary     0.863    10 0.00862 Prepro...
#> 4 2       0.0001   roc_auc binary     0.866    10 0.00855 Prepro...
```

이 초기 그리드는 상당히 동등한(equivalent) 결과를 보여주며, 어느 개별 포인트도 다른 어느 포인트보다 훨씬 낫지 않습니다. 이 결과들은 초기값으로 사용하기 위해 다음 섹션에서 설명하는 반복적인 튜닝 함수에 의해 수집될(ingested) 수 있습니다.

# 베이지안 최적화 (Bayesian Optimization)

베이지안 최적화 기법은 현재 리샘플링 결과를 분석하고 아직 평가되지 않은 튜닝 매개변수 값을 제안하는 예측 모델을 만듭니다. 제안된 매개변수 조합은 그런 다음 리샘플링됩니다. 이러한 결과는 테스트할 더 많은 후보 값을 추천하는 또 다른 예측 모델에 사용되는 식입니다(and so on). 이 프로세스는 정해진 반복 횟수 동안 또는 더 이상의 향상이 발생하지 않을 때까지 계속됩니다(proceeds). Shahriari et al. (2016) 및 Frazier (2018)는 베이지안 최적화에 대한 좋은 소개 자료입니다.

베이지안 최적화를 사용할 때 주된 고려 사항은 모델을 어떻게 생성할 것인지와 해당 모델에서 권장하는 매개변수를 어떻게 선택할 것인지입니다. 먼저 베이지안 최적화에 가장 일반적으로 사용되는 기법인 가우시안 프로세스(Gaussian process) 모델을 살펴보겠습니다.

## 가우시안 프로세스 모델 (A Gaussian Process Model)

가우시안 프로세스(Gaussian process, GP) (Schulz, Speekenbrink, and Krause 2018) 모델은 (*크리깅(kriging) 방법*이라는 이름으로) 공간 통계(spatial statistics)에 역사(history)를 가진 잘 알려진 통계 기법입니다. 이것들은 베이지안 모델을 포함하여 여러 가지 방법으로 파생될(derived) 수 있습니다; 훌륭한 참고 문헌은 Rasmussen and Williams (2006)를 참조하십시오.

수학적으로 GP는 결합 확률 분포(joint probability distribution)가 다변량 가우시안(multivariate Gaussian)인 무작위(random) 변수들의 모음(collection)입니다. 우리 애플리케이션의 컨텍스트에서, 이것은 튜닝 매개변수 후보 값에 대한 성능 지표의 모음입니다. 이전 4개 샘플의 초기 그리드에 대해, 이 4개의 무작위 변수의 실현값(realizations)은 0.8639, 0.8625, 0.8627, 0.8659였습니다. 이들은 다변량 가우시안으로 분포(distributed)한다고 가정합니다(assumed). GP 모델에 대한 독립 변수/예측 변수를 정의하는 입력(inputs)은 대응되는 튜닝 매개변수 값입니다([표 14-1](#initial-gp-data)에 표시됨).

| ROC    | cost    | rbf_sigma |
|--------|---------|-----------|
| 0.8639 | 0.01562 | 0.000001  |
| 0.8625 | 2.00000 | 0.000001  |
| 0.8627 | 0.01562 | 0.000100  |
| 0.8659 | 2.00000 | 0.000100  |

표 14-1. 가우시안 프로세스 모델에 초기 바탕(substrate)으로 사용된 리샘플링 통계량으로, 여기서 `ROC`가 결과(outcome)이고 `cost`와 `rbf_sigma`는 모두 예측 변수입니다. {#initial-gp-data}

가우시안 프로세스 모델은 평균 및 공분산(covariance) 함수로 지정되지만, 후자가 GP 모델의 특성(nature)에 가장 큰 영향을 미칩니다. 공분산 함수는 종종 입력 값($`x`$로 표시됨) 측면에서(in terms of) 매개변수화(parameterized)됩니다. 예로서, 일반적으로 사용되는 공분산 함수는 제곱 지수(squared exponential)<sup><a href="ch14.xhtml#idm45881854319968" id="idm45881854319968-marker" data-type="noteref">1</a></sup> 함수입니다:

``` math
\operatorname{cov}\left( \mathbf{x}_{i},\mathbf{x}_{j} \right) = \exp\left( {- \frac{1}{2}\left| \mathbf{x}_{i} - \mathbf{x}_{j} \right|^{2}} \right) + \sigma_{ij}^{2}
```

여기서 $`\sigma_{ij}^{2}`$는 $`i = j`$일 때 0이 되는 상수 오차 분산(variance) 항입니다. 이 방정식은 다음과 같이 해석됩니다(translates to):

> 두 튜닝 매개변수 조합 사이의 거리가 증가함에 따라(increases), 성능 지표들 사이의 공분산은 기하급수적으로(exponentially) 증가합니다(increase).

방정식의 특성은 또한 이미 관찰된 지점(즉, $`|\mathbf{x}_{i} - \mathbf{x}_{j}|^{2}`$가 0일 때)에서 결과 지표의 변동(variation)이 최소화(minimized)됨을 암시합니다.

이 공분산 함수의 특성을 통해 가우시안 프로세스는 소량의 데이터만 존재하는 경우에도 모델 성능과 튜닝 매개변수 간의 고도로 비선형적인(highly nonlinear) 관계를 나타낼 수 있습니다.

###### 경고 (Warning)

그러나 어떤 경우에는 이러한 모델을 피팅하는 것이 어려울 수 있으며, 튜닝 매개변수 조합의 수가 증가함에 따라 모델의 계산 비용이 더 많이 들게(computationally expensive) 됩니다.

이 모델의 중요한 장점(virtue)은 완전한 확률 모델이 지정(specified)되므로 새로운 입력에 대한 예측이 결과의 전체 분포를 반영할 수 있다는 것입니다. 즉, 평균과 분산 측면에서 모두 새로운 성능 통계량을 예측할 수 있습니다.

고려 중인 두 개의 새로운 튜닝 매개변수가 있다고 가정해 봅시다(under consideration). [표 14-2](#tuning-candidates)에서 후보 A는 후보 B보다 약간 더 나은 평균 ROC 값을 갖습니다(현재 최고는 0.8659). 그러나 분산은 B보다 4배(four-fold) 더 큽니다. 이것은 좋은 것일까요, 나쁜 것일까요? 옵션 A를 선택하는 것이 더 위험하지만(riskier) 잠재적으로 더 높은 수익(return)을 갖습니다. 분산의 증가는 또한 이 새로운 값이 기존 데이터에서 B보다 더 멀리 떨어져 있음(farther)을 반영합니다(reflects). 다음 섹션에서는 베이지안 최적화에 대한 GP 예측의 이러한 측면(aspects)을 더 자세히(in more detail) 살펴봅니다.

| 후보 (Candidate) | 평균 (Mean) | 분산 (Variance) |
|-----------|------|----------|
| A         | 0.90 | 0.000400 |
| B         | 0.89 | 0.000025 |

표 14-2. 추가 샘플링을 위해 고려되는 두 가지 예시 튜닝 매개변수 {#tuning-candidates}

###### 참고 (Note)

베이지안 최적화는 반복적인(iterative) 과정입니다.

4개 결과의 초기 그리드를 기반으로 GP 모델이 피팅되고, 후보들이 예측되며, 5번째 튜닝 매개변수 조합이 선택됩니다. 새로운 구성(configuration)에 대한 성능 추정치를 계산하고, 5개의 기존 결과로 GP를 다시 피팅합니다(refit) (기타 등등).

## 획득 함수 (Acquisition Functions)

가우시안 프로세스가 현재 데이터에 피팅되면(fit to) 어떻게 사용될까요? 우리의 목표는 현재 최고보다 "더 나은 결과"를 가질 가능성이 가장 높은 다음번 튜닝 매개변수 조합을 선택하는 것입니다. 이를 수행하는 한 가지 접근 방식은 (아마도 공간 채우기 설계를 사용하여) 대규모(large) 후보 세트를 만든 다음 각 후보에 대해 평균 및 분산 예측을 수행하는 것입니다. 이 정보를 사용하여 가장 유리한(advantageous) 튜닝 매개변수 값을 선택합니다.

*획득 함수(acquisition functions)*라고 불리는 목적 함수(objective functions)의 한 클래스는 평균과 분산 간의 트레이드오프를 용이하게 합니다. GP 모델의 예측된 분산은 주로 그것들이 기존 데이터에서 얼마나 떨어져 있는지(far away)에 따라 주도된다(driven by)는 점을 상기하십시오. 새로운 후보에 대한 예측된 평균과 분산 간의 트레이드오프는 종종 탐색(exploration)과 활용(exploitation)이라는 렌즈를 통해 보여집니다:

탐색 (Exploration)  
이것은 관찰된 후보 모델이 (만약 있다면) 더 적은 영역 쪽으로(toward) 선택을 편향시킵니다(biases). 이는 더 높은 분산을 가진 후보에게 더 많은 가중치를 부여하는 경향이 있으며 새로운 결과를 찾는 데 중점을 둡니다(focuses on).

활용 (Exploitation)  
이것은 원칙적으로 최상의 (평균) 값을 찾기 위해 평균 예측에 의존합니다(relies on). 이는 기존 결과에 중점을 둡니다.

이를 시연하기 위해(To demonstrate), \[0, 1\] 사이의 값을 가지며 성능 지표가 $`R^2`$인 단일 매개변수를 사용하는 장난감(toy) 예제를 살펴보겠습니다. 실제(true) 함수는 기존 결과를 점(points)으로 갖는 5개의 후보 값과 함께 [그림 14-2](#performance-profile)에 표시되어 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1402.png" alt="tmwr 1402" />
<h6 id="figure-14-2.-hypothetical-true-performance-profile-over-an-arbitrary-tuning-parameter-with-five-estimated-points.">그림 14-2. 5개의 추정된 점(points)이 있는 임의의 튜닝 매개변수에 대한 가상의(Hypothetical) 실제 성능 프로파일.</h6>
</figure>

이 데이터의 경우 GP 모델 피팅이 [그림 14-3](#estimated-profile)에 표시되어 있습니다. 음영(shaded) 영역은 평균 $`\pm`$ 1 표준 오차를 나타냅니다. 두 수직선은 나중에 더 자세히 조사(examined)할 두 개의 후보 점을 나타냅니다.

음영 처리된 신뢰(confidence) 영역은 제곱 지수 분산 함수를 시연합니다; 점 사이에서 매우 커지고 기존 데이터 점에서는 0으로 수렴합니다(converges to zero).

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1403.png" alt="tmwr 1403" />
<h6 id="figure-14-3.-estimated-performance-profile-generated-by-the-gaussian-process-model.-the-shaded-region-shows-one-standard-error-bounds.">그림 14-3. 가우시안 프로세스 모델에 의해 생성된 추정(Estimated) 성능 프로파일. 음영 영역은 1-표준 오차 범위를 보여줍니다.</h6>
</figure>

이 비선형 추세(trend)는 관찰된 각 점을 통과하지만 모델이 완벽하지는 않습니다. 진정 최적인(optimum) 설정 근처에 관찰된 점이 없으며, 이 영역에서는 피팅이 훨씬 더 나을(better) 수 있습니다. 그럼에도 불구하고(Despite this), GP 모델은 우리에게 올바른 방향을 효과적으로 가리킬 수 있습니다.

순수한 활용 관점(standpoint)에서 최선의 선택은 가장 좋은 평균 예측을 갖는 매개변수 값을 선택하는 것입니다. 여기서는 현재 관찰된 최상의 포인트인 0.09 바로 오른쪽에 있는 0.106이 될 것입니다.

탐색을 장려(encourage)하는 방법으로서, 간단한(간단하지만 자주 사용되지는 않는) 접근 방식은 가장 큰 신뢰 구간과 관련된(associated with) 튜닝 매개변수를 찾는 것입니다. 예를 들어, $`R^2`$ 신뢰 경계(bound)에 단일 표준 편차를 사용하면 샘플링할 다음 포인트는 0.236이 될 것입니다. 이것은 관찰된 결과가 없는 영역에 약간 더 들어갑니다. 상한(upper bound)에 사용되는 표준 편차의 수를 늘리면 빈(empty) 영역으로 더 멀리 선택을 밀어 넣을 것입니다.

가장 일반적으로 사용되는 획득 함수 중 하나는 *기대 향상(expected improvement)*입니다. (신뢰 경계 접근 방식과 달리) 향상의 개념은 현재 최상의 결과에 대한 값을 필요로 합니다. GP는 분포를 사용하여 새로운 후보 지점을 설명할 수 있으므로 개선이 발생할 확률(probability)을 사용하여 개선을 보여주는 분포 부분에 가중치를 부여할(weight) 수 있습니다.

예를 들어, 0.10과 0.25 두 개의 후보 매개변수 값([그림 14-3](#estimated-profile)에서 수직선으로 표시됨)을 생각해 보십시오(consider). 피팅된 GP 모델을 사용하여 예측된 $`R^2`$ 분포가 현재 최상의 결과에 대한 참조선(reference line)과 함께 [그림 14-4](#two-candidates)에 표시되어 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1404.png" alt="tmwr 1404" />
<h6 id="figure-14-4.-predicted-performance-distributions-for-two-sampled-tuning-parameter-values.">그림 14-4. 샘플링된 두 튜닝 매개변수 값에 대한 예측(Predicted) 성능 분포.</h6>
</figure>

평균 $`R^2`$ 예측만 고려할 때 0.10의 매개변수 값이 더 나은 선택입니다([표 14-3](#two-exp-improve) 참조). 0.25에 대한 튜닝 매개변수 추천은 평균적으로(on average) 현재 최고보다 더 나쁠(worse) 것으로 예측됩니다. 그러나 분산이 더 높기 때문에 현재 최고보다 더 넓은(more) 전체 확률 면적(probability area)을 갖습니다. 결과적으로 그것은 더 큰(larger) 기대 향상을 갖습니다:

| 매개변수 값 (Parameter value) | 평균 (Mean)   | 표준 편차 (Std dev)   | 기대 향상 (Expected improvement) |
|-----------------|--------|-----------|----------------------|
| 0.10            | 0.8679 | 0.0004317 | 0.000190             |
| 0.25            | 0.8671 | 0.0039301 | 0.001216             |

표 14-3. 두 후보 튜닝 매개변수에 대한 기대 향상 {#two-exp-improve}

튜닝 매개변수의 범위에 걸쳐 기대 향상을 계산할 때, [그림 14-5](#expected-improvement)에 표시된 대로 샘플링 권장 지점은 0.10보다 0.25에 훨씬 가깝습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1405.png" alt="tmwr 1405" />
<h6 id="figure-14-5.-the-estimated-performance-profile-generated-by-the-gaussian-process-model-top-panel-and-the-expected-improvement-bottom-panel.-the-vertical-line-indicates-the-point-of-maximum-improvement.">그림 14-5. 가우시안 프로세스 모델에 의해 생성된 추정 성능 프로파일(상단 패널)과 기대 향상(하단 패널). 수직선은 최대 향상 지점을 나타냅니다.</h6>
</figure>

수많은(Numerous) 획득 함수들이 제안되고 논의(discussed)되어 왔습니다; tidymodels에서는 기대 향상이 기본값(default)입니다.

## `tune_bayes()` 함수

베이지안 최적화를 통한 반복 검색을 구현(implement)하려면 `tune_bayes()` 함수를 사용하세요. 이것의 구문(syntax)은 `tune_grid()`와 매우 유사하지만 몇 가지 추가 인수가 있습니다:

`iter`  
이것은 검색 반복의 최대 횟수입니다.

`initial`  
이것은 정수이거나 `tune_grid()`를 사용하여 생성된 객체, 또는 경주(racing) 함수 중 하나일 수 있습니다. 정수를 사용하면 첫 번째 GP 모델 이전에 샘플링되는 공간 채우기 설계의 크기를 지정합니다.

`objective`  
이것은 어느(which) 획득 함수를 사용해야 하는지에 대한 인수입니다. tune 패키지에는 `exp_improve()` 또는 `conf_bound()`와 같이 여기에 전달할(pass here) 수 있는 함수들이 포함되어 있습니다.

`param_info` 인수  
이 경우 이것은 매개변수의 범위와 사용되는 변환들을 지정합니다. 이들은 검색 공간을 정의하는 데 사용됩니다. 기본 매개변수 객체가 불충분(insufficient)한 상황에서는 `param_info`를 사용하여 기본값을 재정의(override)합니다.

`control` 인수는 이제 `control_bayes()`의 결과를 사용합니다. 다음은 거기에 있는 몇 가지 유용한(helpful) 인수들입니다:

`no_improve`  
이것은 `no_improve` 반복 내에서 개선된 매개변수가 발견(discovered)되지 않으면 검색을 중단(stop)하는 정수입니다.

`uncertain`  
이것 또한 `uncertain` 반복 내에 개선이 없으면 *불확실성 샘플(uncertainty sample)*을 채취(take)하는 정수(또는 `Inf`)입니다. 이것은 분산이 큰(large variation) 다음 후보를 선택합니다. 그것은 평균 예측을 고려하지 않기 때문에 순수한 탐색의 효과를 갖습니다.

`verbose`  
이것은 검색이 진행됨에 따라(proceeds) 로깅 정보를 인쇄하는(print) 논리값(logical)입니다.

이 장의 시작 부분에 있는 첫 번째 SVM 결과를 가우시안 프로세스 모델의 초기 바탕(substrate)으로 사용해 보겠습니다. 이 애플리케이션의 경우 ROC 곡선 아래 면적을 최대화하려고 한다는(want to) 것을 상기하십시오. 우리 코드는 다음과 같습니다:

```
ctrl <- control_bayes(verbose = TRUE)

set.seed(1403)
```
svm_bo <-
  svm_wflow %>%
  tune_bayes(
    resamples = cell_folds,
    metrics = roc_res,
    initial = svm_initial,
    param_info = svm_param,
    iter = 25,
    control = ctrl
  )
```

검색 프로세스는 ROC 곡선 아래 면적에 대해 0.8659라는 초기 최고값(initial best value)으로 시작합니다. 가우시안 프로세스 모델은 이 4개의 통계량을 사용하여 모델을 만듭니다. 대규모 후보 세트가 자동으로 생성되고 기대 향상 획득 함수를 사용하여 점수가 매겨집니다(scored). 첫 번째 반복은 0.86315의 ROC 값으로 결과를 향상시키는 데 실패했습니다. 새로운 결과값으로 또 다른 가우시안 프로세스 모델을 피팅한 후, 두 번째 반복 역시 향상을 가져오지(yield) 못했습니다.

`verbose` 옵션에 의해 생성된 처음 두 번의 반복에 대한 로그는 다음과 같았습니다:

\#\> Optimizing roc_auc using the expected improvement \#\> \#\> ── Iteration 1 ────────────────────────────────────── \#\> \#\> i Current best: roc_auc=0.8659 (@iter 0) \#\> i Gaussian process model \#\> ✓ Gaussian process model \#\> i Generating 5000 candidates \#\> i Predicted candidates \#\> i cost=0.386, rbf_sigma=0.000266 \#\> i Estimating performance \#\> ✓ Estimating performance \#\> ⓧ Newest results: roc_auc=0.8631 (+/-0.00866) \#\> \#\> ── Iteration 2 ────────────────────────────────────── \#\> \#\> i Current best: roc_auc=0.8659 (@iter 0) \#\> i Gaussian process model \#\> ✓ Gaussian process model \#\> i Generating 5000 candidates \#\> i Predicted candidates \#\> i cost=13.8, rbf_sigma=7.83e-07 \#\> i Estimating performance \#\> ✓ Estimating performance \#\> ⓧ Newest results: roc_auc=0.8624 (+/-0.00865)

검색은 계속됩니다(continues). 그 과정(along the way)에서 반복 3, 4, 5, 6, 8, 13, 22, 23, 24에서 결과에 총 9번의 향상이 있었습니다. 최상의 결과는 반복 24에서 ROC 곡선 아래 면적이 0.8986으로 나타났습니다:

\#\> ── Iteration 24 ───────────────────────────────────── \#\> \#\> i Current best: roc_auc=0.8986 (@iter 23) \#\> i Gaussian process model \#\> ✓ Gaussian process model \#\> i Generating 5000 candidates \#\> i Predicted candidates \#\> i cost=31.8, rbf_sigma=0.0016 \#\> i Estimating performance \#\> ✓ Estimating performance \#\> ♥ Newest results: roc_auc=0.8986 (+/-0.00785)

마지막 단계는 다음과 같았습니다:

\#\> ── Iteration 25 ───────────────────────────────────── \#\> \#\> i Current best: roc_auc=0.8986 (@iter 24) \#\> i Gaussian process model \#\> ✓ Gaussian process model \#\> i Generating 5000 candidates \#\> i Predicted candidates \#\> i cost=20, rbf_sigma=0.00188 \#\> i Estimating performance \#\> ✓ Estimating performance \#\> ⓧ Newest results: roc_auc=0.8982 (+/-0.00781)

결과를 조사(interrogate)하는 데 사용되는 함수는 그리드 검색에 사용되는 함수(예: `collect_metrics()` 등)와 동일합니다. 예를 들어:

```
show_best(svm_bo)
#> # A tibble: 5 × 9
#>    cost rbf_sigma .metric .estimator  mean     n std_err .config .iter
#>   <dbl>     <dbl> <chr>   <chr>      <dbl> <int>   <dbl> <chr>   <int>
#> 1  31.8   0.00160 roc_auc binary     0.899    10 0.00785 Iter24     24
#> 2  30.8   0.00191 roc_auc binary     0.899    10 0.00791 Iter23     23
#> 3  31.4   0.00166 roc_auc binary     0.899    10 0.00784 Iter22     22
#> 4  31.8   0.00153 roc_auc binary     0.899    10 0.00783 Iter13     13
#> 5  30.8   0.00163 roc_auc binary     0.899    10 0.00782 Iter15     15
```

`autoplot()` 함수에는 반복 검색 방법을 위한 몇 가지 옵션이 있습니다. [그림 14-6](#progress-plot)은 `autoplot(svm_bo, type = "performance")`를 사용하여 검색에 따라 결과가 어떻게 변했는지 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1406.png" alt="tmwr 1406" />
<h6 id="figure-14-6.-the-progress-of-the-bayesian-optimization-produced-when-the-autoplot-method-is-used-with-type-performance.">그림 14-6. <code>autoplot()</code> 메서드가 <code>type = "performance"</code>와 함께 사용될 때 생성되는 베이지안 최적화의 진행 상황(progress).</h6>
</figure>

추가적인 유형의 플롯은 반복에 따른 매개변수 값을 보여주는 `type = "parameters"`를 사용합니다.

[그림 14-7](#bo-surfaces)은 11번의 반복 후 GP에 의해 추정된 평균, 분산, 기대 향상 표면(surfaces)의 표면을 보여줍니다. 오른쪽 패널은 후보 공간의 오른쪽 측면을 따라(along) 최고로 추정된 향상의 능선(ridge)을 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1407.png" alt="tmwr 1407" />
<h6 id="figure-14-7.-heatmaps-of-the-predicted-mean-rmse-left-variance-of-rmse-middle-and-the-expected-improvement-right-after-11-search-iterations.">그림 14-7. 11번의 검색 반복 후 예측된 평균 RMSE(왼쪽), RMSE의 분산(가운데), 기대 향상(오른쪽)의 히트맵.</h6>
</figure>

[그림 14-8](#bo-search)은 최적화 과정의 세 가지 다른 지점에서의 검색 프로세스를 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1408.png" alt="tmwr 1408" />
<h6 id="figure-14-8.-the-bayesian-optimization-search-path-after-5-11-and-25-iterations.">그림 14-8. 5, 11, 25번의 반복 후 베이지안 최적화 검색 경로.</h6>
</figure>

처음 5번의 반복은 처음에는 좋지 않은 방향으로 이동했지만 빠르게 더 나은 결과에 더 가까워졌습니다. 가운데 패널은 프로세스가 후보 공간의 오른쪽 하단 경계(boundary)로 짧게 진출(foray)하여 진정 최적인 결과 영역을 조사하는(investigates) 처음 11번의 반복을 보여줍니다. 왼쪽 패널에 표시된 나머지 반복은 최상의 결과 영역과 검색 공간의 먼 경계 사이를 전환합니다(switch).

최상의 튜닝 매개변수 조합은 매개변수 공간의 경계에 있지만 베이지안 최적화는 종종 경계의 다른 쪽에 있는 새로운 점을 선택합니다. 탐색과 활용의 비율(ratio)을 조정(adjust)할 수 있지만, 검색은 초기에(early on) 경계 지점을 샘플링하는 경향이 있습니다.

마지막으로 사용자가 `tune_bayes()` 계산을 방해(interrupts)하는 경우 함수는 (오류를 발생시키는(resulting in) 대신) 현재 결과를 반환합니다.

###### 참고 (Note)

초기 그리드로 검색을 시작(seeded)하는 경우 공간 채우기 설계가 정규 설계보다 더 나은 선택일 것입니다. 그것은 매개변수 공간의 더 많은 고유한 값을 샘플링하고 초기 반복에서 표준 편차의 예측을 향상시킬 것입니다.

# 시뮬레이티드 어닐링 (Simulated Annealing)

*시뮬레이티드 어닐링(Simulated annealing)* (SA) (Kirkpatrick, Gelatt, and Vecchi 1983; Van Laarhoven and Aarts 1987)은 금속이 냉각되는 과정에서 영감을 받은 일반적인 비선형 검색 루틴입니다. 이것은 불연속(discontinuous) 함수를 포함하여 다양한 유형의 검색 환경(landscapes)을 효과적으로 탐색할(navigate) 수 있는 전역(global) 검색 방법입니다. 대부분의 그래디언트 기반 최적화 루틴과 달리 시뮬레이티드 어닐링은 이전 솔루션을 재평가(reassess)할 수 있습니다.

## 시뮬레이티드 어닐링 검색 프로세스 (Simulated Annealing Search Process)

시뮬레이티드 어닐링을 사용하는 프로세스는 초기값으로 시작하여 매개변수 공간을 통과하는 통제된 무작위 보행(controlled random walk)에 착수(embarks on)합니다. 각각의 새로운 후보 매개변수 값은 새로운 점을 지역 이웃(local neighborhood) 내에 유지하는 이전 값의 작은 교란(perturbation)입니다.

후보 점은 해당(corresponding) 성능 값을 얻기 위해 리샘플링됩니다. 이것이 이전 매개변수보다 더 나은 결과를 달성하면 새로운 최선(new best)으로 수락(accepted)되고 프로세스가 계속됩니다. 결과가 이전 값보다 나쁘면(worse) 검색 절차는 추가 단계(further steps)를 정의하기 위해 이 매개변수를 계속 사용할 수 있습니다. 이는 두 가지 요인(factors)에 달려 있습니다(depends on). 첫째, 나쁜(bad) 결과를 수락할 가능성(likelihood)은 성능이 나빠질수록 감소합니다. 즉, 성능이 크게 떨어지는 결과보다 약간 더 나쁜 결과가 수락될 가능성이 더 높습니다. 다른 요인은 검색 반복 횟수입니다. 시뮬레이티드 어닐링은 검색이 진행됨에 따라 더 적은 차선의(suboptimal) 값을 수락하기를 원합니다. 이 두 가지 요인으로부터 나쁜 결과에 대한 *수락 확률(acceptance probability)*은 다음과 같이 공식화될(formalized) 수 있습니다:

``` math
\Pr\left\lbrack \text{accept}\text{suboptimal}\text{parameters}\text{at}\text{iteration} i \right\rbrack = \exp\left( c \times D_{i} \times i \right)
```

여기서 $`i`$는 반복 횟수, $`c`$는 사용자 지정(user-specified) 상수, $`D_{i}`$는 이전 값과 새 값 사이의 퍼센트 차이(음수 값은 더 나쁜 결과를 의미함)입니다. 나쁜 결과의 경우 수락 확률을 결정(determine)하고 무작위 균일 숫자(random uniform number)와 비교합니다. 무작위 숫자가 확률 값보다 크면 검색은 현재 매개변수를 버리고(discards) 다음 반복은 이전 값의 인근(neighborhood)에 후보 값을 만듭니다. 그렇지 않으면 다음 반복은 현재(차선의) 값을 기반으로 다음 매개변수 세트를 형성(forms)합니다.

###### 참고 (Note)

시뮬레이티드 어닐링의 수락 확률은 장기적으로(in the long run) 매개변수 공간의 훨씬 더 나은 영역을 찾을 가능성(potential)을 가지고 적어도 단기적으로는(for the short term) 검색이 잘못된 방향으로 진행될 수 있도록 허용합니다.

수락 확률은 어떻게 영향을 받을까요? [그림 14-9](#acceptance-prob)의 히트맵은 반복, 성능 및 사용자 지정 계수(coefficient)에 따라 수락 확률이 어떻게 변할 수 있는지 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1409.png" alt="tmwr 1409" />
<h6 id="figure-14-9.-heatmap-of-the-simulated-annealing-acceptance-probabilities-for-different-coefficient-values.">그림 14-9. 다양한 계수 값에 대한 시뮬레이티드 어닐링 수락 확률의 히트맵.</h6>
</figure>

사용자는 계수를 조정하여 자신의 요구에 맞는(suits their needs) 확률 프로파일을 찾을 수 있습니다. `finetune::control_sim_anneal()`에서 이 `cooling_coef` 인수의 기본값은 0.02입니다. 이 계수를 줄이면(Decreasing) 검색이 낮은(poor) 결과에 대해 더 관대해지도록(forgiving) 장려합니다.

이 프로세스는 정해진 반복 횟수 동안 계속되지만 미리 정해진(predetermined) 반복 횟수 내에 전역 최상의(globally best) 결과가 발생하지 않으면 중단(halt)될 수 있습니다. 그러나 *재시작 임계값(restart threshold)*을 설정하는 것이 매우 도움이 될 수 있습니다. 일련의(a string of) 실패가 있는 경우 이 기능은 마지막 전역 최상의 매개변수 설정을 다시 방문(revisits)하여 새로 시작(starts anew)합니다.

가장 중요한 세부 사항은 반복에서 반복으로 갈 때 튜닝 매개변수를 어떻게 교란(perturb)할지 정의하는 것입니다. 이를 위해 문헌(literature)에는 다양한 방법이 있습니다. 우리는 *일반화된 시뮬레이티드 어닐링(generalized simulated annealing)*이라고 불리는 Bohachevsky, Johnson, and Stein (1986)에 제시된 방법을 따릅니다. 연속적인(continuous) 튜닝 매개변수의 경우 로컬 "인근(neighborhood)"을 지정하기 위해 작은 반경(radius)을 정의합니다. 예를 들어, 두 개의 튜닝 매개변수가 있고 각각이 0과 1로 경계가 정해져(bounded by) 있다고 가정해 봅시다. 시뮬레이티드 어닐링 프로세스는 주변(surrounding) 반경에서 무작위 값을 생성하고 그중 하나를 현재 후보 값으로 무작위로 선택합니다.

우리의 구현에서 이웃은 매개변수 객체의 범위를 기반으로 현재 후보가 0과 1 사이가 되도록 척도화하여(scaling) 결정되므로 0.05와 0.15 사이의 반경 값이 합리적(reasonable)인 것 같습니다. 이러한 값의 경우 검색이 매개변수 공간의 한쪽에서 다른 쪽으로 갈 수 있는 가장 빠른 속도는 약 10회 반복입니다. 반경 크기는 검색이 매개변수 공간을 얼마나 빨리 탐색하는지를 제어합니다. 우리의 구현에서는 서로 다른 크기(magnitudes)의 "지역성(local)"이 새로운 후보 값을 정의하도록 반경 범위를 지정합니다.

설명을 위해, 우리는 두 가지 주요 glmnet 튜닝 매개변수를 사용할 것입니다:

- 전체 정규화의 양 (`penalty`). 이 매개변수의 기본 범위는 $`10^{- 10}`$에서 $`10^{0}`$입니다. 이 매개변수에는 로그(밑 10) 변환을 사용하는 것이 일반적입니다(typical).

- 라쏘 페널티의 비율 (`mixture`). 이것은 변환 없이 0과 1로 경계가 정해져 있습니다.

프로세스는 `penalty = 0.025` 및 `mixture = 0.050`의 초기값으로 시작됩니다. 0.050에서 0.015 사이를 무작위로 변동하는(fluctuates) 반경을 사용하여 데이터가 적절하게 척도화(scaled)되고, 초기 점 주변의 반경에서 무작위 값이 생성된 다음(generated), 그중 하나가 무작위로 후보로 선택됩니다. 설명을 위해 모든 후보 값이 향상(improvements)된다고 가정합니다. 새로운 값을 사용하여 새로운 무작위 인근 세트가 생성되고 하나가 선택되는 식입니다(and so on). [그림 14-10](#iterative-neighborhood)은 검색이 왼쪽 위 모서리(corner)를 향해 진행됨에 따라 6번의 반복을 보여줍니다.

일부 반복 중에 반경을 따라 있는 후보 세트는 매개변수 경계 외부의 점을 제외(exclude)한다는 점에 유의하십시오. 또한, 우리의 구현은 다음 튜닝 매개변수 구성의 선택을 이전 구성과 매우 유사한 새로운 값에서 *멀어지도록(away)* 편향시킵니다(biases).

숫자가 아닌(nonnumeric) 매개변수의 경우, 매개변수 값이 얼마나 자주 변경되는지에 대한 확률을 할당(assign)합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1410.png" alt="tmwr 1410" />
<h6 id="figure-14-10.-an-illustration-of-how-simulated-annealing-determines-what-is-the-local-neighborhood-for-two-numeric-tuning-parameters.-the-clouds-of-points-show-possible-next-values-where-one-would-be-selected-at-random.">그림 14-10. 시뮬레이티드 어닐링이 두 개의 숫자 튜닝 매개변수에 대해 지역 인근이 무엇인지 어떻게 결정하는지에 대한 그림. 점구름(clouds of points)은 무작위로 하나가 선택될 수 있는 다음 번의 가능한 값들을 보여줍니다.</h6>
</figure>

## `tune_sim_anneal()` 함수

시뮬레이티드 어닐링을 통한 반복 검색을 구현하려면 `tune_sim_anneal()` 함수를 사용하세요. 이 함수에 대한 구문(syntax)은 `tune_bayes()`와 거의 동일(identical)합니다. 획득 함수나 불확실성 샘플링에 대한 옵션은 없습니다. `control_sim_anneal()` 함수에는 지역 인근과 냉각 일정(cooling schedule)을 정의하는 몇 가지 세부 사항이 있습니다:

- `no_improve`는 시뮬레이티드 어닐링의 경우 `no_improve` 반복 내에 전역 최상(global best) 또는 향상된 결과가 발견되지 않으면 검색을 중지(stop)하는 정수입니다. 수락된 차선(suboptimal) 또는 버려진(discarded) 매개변수는 "개선 없음(no improvement)"으로 간주(count)됩니다.

- `restart`는 이전 최상의 결과에서 새로 시작하기 전까지 새로운 최상의 결과가 없는 반복 횟수입니다.

- `radius`는 (0, 1) 사이의 수치(numeric) 벡터로, 초기 점 주변의 지역 인근의 최소 및 최대 반경을 정의합니다.

- `flip`은 범주형(categorical) 또는 정수형 매개변수의 값을 변경할(altering) 확률(chances)을 정의하는 확률(probability) 값입니다.

- `cooling_coef`는 $`\exp\left( c \times D_{i} \times i \right)`$의 $`c`$ 계수로, 반복에 따라 수락 확률이 얼마나 빨리 감소하는지(decreases)를 조절(modulates)합니다. `cooling_coef` 값이 크면(Larger) 차선의 매개변수 설정을 수락할 확률이 줄어듭니다.

세포 분할 데이터의 경우 구문은 이전에 사용된 함수들과 매우 일관됩니다(consistent):

```
ctrl_sa <- control_sim_anneal(verbose = TRUE, no_improve = 10L)

set.seed(1404)
svm_sa <-
  svm_wflow %>%
  tune_sim_anneal(
    resamples = cell_folds,
    metrics = roc_res,
    initial = svm_initial,
    param_info = svm_param,
    iter = 50,
    control = ctrl_sa
  )
```

시뮬레이티드 어닐링 프로세스는 4개의 다른 반복에서 새로운 전역 최적값(optimums)을 발견했습니다(discovered). 가장 이른(earliest) 개선은 반복 5에 있었고 최종 최적값은 반복 27에서 발생했습니다(occured). 가장 좋은 전체(overall) 결과는 반복 27에서 0.8985의 ROC 곡선 아래 평균 면적으로 발생했습니다(0.8659의 초기 최상값과 비교하여). 반복 13, 21, 35, 43에서 4번의 재시작(restarts)이 있었고 프로세스 동안 12개의 버려진(discarded) 후보가 있었습니다.

`verbose` 옵션은 검색 프로세스의 세부 정보를 인쇄합니다. 처음 5번의 반복에 대한 출력은 다음과 같습니다:

\#\> Optimizing roc_auc \#\> Initial best: 0.86594 \#\> 1 ◯ accept suboptimal roc_auc=0.86351 (+/-0.008642) \#\> 2 ◯ accept suboptimal roc_auc=0.86233 (+/-0.008657) \#\> 3 + better suboptimal roc_auc=0.86233 (+/-0.008661) \#\> 4 + better suboptimal roc_auc=0.86492 (+/-0.008504) \#\> 5 ♥ new best roc_auc=0.87247 (+/-0.008232)

마지막 10번의 반복에 대한 출력은 다음과 같습니다:

\#\> 40 ◯ accept suboptimal roc_auc=0.89606 (+/-0.008203) \#\> 41 ─ discard suboptimal roc_auc=0.87556 (+/-0.009272) \#\> 42 ─ discard suboptimal roc_auc=0.87198 (+/-0.009301) \#\> 43 ✖ restart from best roc_auc=0.89801 (+/-0.008224) \#\> 44 ◯ accept suboptimal roc_auc=0.89006 (+/-0.008789) \#\> 45 + better suboptimal roc_auc=0.89781 (+/-0.008104) \#\> 46 ◯ accept suboptimal roc_auc=0.89563 (+/-0.008601) \#\> 47 ─ discard suboptimal roc_auc=0.88527 (+/-0.008766) \#\> 48 ◯ accept suboptimal roc_auc=0.8922 (+/-0.008891) \#\> 49 ─ discard suboptimal roc_auc=0.87691 (+/-0.008352) \#\> 50 ◯ accept suboptimal roc_auc=0.88803 (+/-0.008728)

다른 `tune_*()` 함수와 마찬가지로 해당 `autoplot()` 함수는 결과에 대한 시각적 평가(assessments)를 생성합니다. `autoplot(svm_sa, type = "performance")`를 사용하면 반복에 따른 성능([그림 14-11](#sa-iterations))을 보여주고 `autoplot(svm_sa, type = "parameters")`는 특정 튜닝 매개변수 값 대비(versus) 성능([그림 14-12](#sa-parameters))을 플로팅합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1411.png" alt="tmwr 1411" />
<h6 id="figure-14-11.-progress-of-the-simulated-annealing-process-shown-when-the-autoplot-method-is-used-with-type-performance.">그림 14-11. <code>autoplot()</code> 메서드가 <code>type = "performance"</code>와 함께 사용될 때 표시되는 시뮬레이티드 어닐링 프로세스의 진행 상황.</h6>
</figure>

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1412.png" alt="tmwr 1412" />
<h6 id="figure-14-12.-performance-versus-tuning-parameter-values-when-the-autoplot-method-is-used-with-type-parameters.">그림 14-12. <code>autoplot()</code> 메서드가 <code>type = "parameters"</code>와 함께 사용될 때의 튜닝 매개변수 값 대비 성능.</h6>
</figure>

`tune_bayes()`와 마찬가지로 실행을 수동으로(manually) 중지(stopping)하면 완료된(completed) 반복이 반환됩니다.

검색 경로의 시각화는 검색 프로세스가 어디에서 잘 수행되었고 어디에서 길을 잃었는지(went astray) 이해하는 데 도움이 됩니다. [그림 14-13](#sa-plot)은 최적화의 여러 단계를 보여줍니다; 이들은 마지막 최고 결과에서 프로세스를 재시작(restart)하여 구분됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1413.png" alt="tmwr 1413" />
<h6 id="figure-14-13.-a-visualization-of-different-phases-of-the-simulated-annealing-search.">그림 14-13. 시뮬레이티드 어닐링 검색의 여러 단계(phases)에 대한 시각화.</h6>
</figure>

첫 번째 단계에서 검색은 처음에 두 개의 새로운 전역 최적값(실선 원으로 표시됨)을 찾습니다. 이들로부터 일부 설정은 즉시(immediately) 버려지고(밝은 회색 선) 다른 설정은 차선이지만 허용 가능(acceptable)합니다. 정해진 실패 횟수 후 마지막 실선 원에서 다시 시작(restarts)합니다. 다른 단계는 과정에(along the way) 버려진 설정이 많은 상태에서 전역 최적값의 느린 개선(improvement)을 보여줍니다. 프로세스는 허용된 총 반복 횟수를 소진(exhausts)함에 따라 결국 최적의 결과 영역으로의 길을 찾습니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 튜닝 매개변수를 최적화하기 위한 두 가지 반복 검색 방법에 대해 설명했습니다. 베이즈 최적화는 튜닝 매개변수 값을 제안하기 위해 기존 리샘플링 결과에 대해 훈련된 예측 모델을 사용하는 반면, 시뮬레이티드 어닐링은 좋은 값을 찾기 위해 하이퍼파라미터 공간을 보행(walks)합니다. 둘 다 좋은 값을 단독으로 찾거나 초기 그리드 검색 후 성능을 더 미세 조정(finetune)하기 위해 사용되는 후속(follow-up) 방법으로 효과적일 수 있습니다.

<sup>[1](ch14.xhtml#idm45881854319968-marker)</sup> 이 방정식은 현재 사용 중인 SVM 모델과 같은 커널(kernel) 방법에서 사용되는 *방사 기저 함수(radial basis function)*와도 동일합니다. 이것은 우연의 일치(coincidence)입니다; 이 공분산 함수는 우리가 사용하고 있는 SVM 튜닝 매개변수와는 관련이 없습니다(unrelated).
