# 12장. 모델 튜닝 및 과적합의 위험성 (Model Tuning and the Dangers of Overfitting)

모델을 예측에 사용하려면 해당 모델의 매개변수(parameters)를 추정(estimated)해야 합니다. 이러한 매개변수 중 일부는 훈련 데이터에서 직접 추정할 수 있지만 _튜닝 매개변수(tuning parameters)_ 또는 *하이퍼파라미터(hyperparameters)*라고 하는 다른 매개변수는 미리 지정해야 하며 훈련 데이터에서 직접 찾을 수 없습니다. 이들은 알려지지 않은 구조적 값(structural values)이거나, 모델에 큰 영향을 미치지만 데이터에서 직접 추정할 수 없는 다른 종류의 값입니다. 이 장에서는 튜닝 매개변수의 예를 제공하고 tidymodels 함수를 사용하여 튜닝 매개변수를 생성하고 다루는(handle) 방법을 보여줍니다. 또한 이러한 값을 잘못 선택하면 어떻게 과적합(overfitting)으로 이어지는지 시연하고 최적의 튜닝 매개변수 값을 찾기 위한 여러 전략(tactics)을 소개합니다. [13장](ch13.xhtml#grid-search)과 [14장](ch14.xhtml#iterative-search)에서는 튜닝을 위한 특정 최적화 방법에 대해 더 자세히 다룹니다.

# 모델 매개변수 (Model Parameters)

일반적인(ordinary) 선형 회귀에는 모델의 두 매개변수 $`\beta_{0}`$과 $`\beta_{1}`$이 있습니다.

```math
y_{i} = \beta_{0} + \beta_{1}x_{i} + \epsilon_{i}
```

결과(outcome) 변수($`y`$)와 예측(predictor) 변수($`x`$) 데이터가 있을 때 우리는 두 매개변수 $`\beta_{0}`$과 $`\beta_{1}`$을 추정할 수 있습니다.

```math
{\hat{\beta}}_{1} = \frac{\sum_{i}\left( y_{i} - \overline{y} \right)\left( x_{i} - \overline{x} \right)}{\sum_{i}\left( x_{i} - \overline{x} \right)^{2}}
```

그리고

```math
{\hat{\beta}}_{0} = \overline{y} - {\hat{\beta}}_{1}\overline{x}.
```

우리는 이 예제 모델에 대해 데이터로부터 이러한 값을 직접 추정할 수 있습니다. 왜냐하면 분석적으로 다루기 쉽기(analytically tractable) 때문입니다; 즉 데이터가 있다면 이러한 모델 매개변수를 추정할 수 있습니다.

###### 참고 (Note)

모델에 데이터에서 직접 추정할 수 _없는_ 매개변수가 있는 상황이 많이 있습니다.

KNN 모델의 경우 새로운 값 $`x_{0}`$에 대한 예측 방정식은 다음과 같습니다.

```math
\hat{y} = \frac{1}{K}\sum\limits_{\ell = 1}^{K}x_{\ell}^{*}
```

여기서 $`K`$는 이웃 수(number of neighbors)이고 $`x_{\ell}^{*}`$는 훈련 세트에서 $`x_{0}`$에 가까운 $`K`$개의 값입니다. 모델 자체는 모델 방정식으로 정의되지 않습니다; 대신 이전 예측 방정식이 그것을 정의합니다. 이러한 특성은 거리 측정(distance measure)의 다루기 어려울(intractability) 가능성과 함께 (반복적이든 아니든) $`K`$에 대해 풀 수 있는(solved for) 일련의 방정식을 만드는 것을 불가능하게 합니다. 이웃의 수는 모델에 심오한(profound) 영향을 미칩니다; 그것은 클래스 경계(class boundary)의 유연성(flexibility)을 지배합니다. $`K`$ 값이 작으면 경계가 매우 정교(elaborate)한 반면 값이 크면 꽤 매끄러울 수 있습니다.

최근접 이웃(nearest neighbors) 수는 데이터에서 직접 추정할 수 없는 튜닝 매개변수 또는 하이퍼파라미터의 좋은 예입니다.

# 다양한 유형의 모델을 위한 튜닝 매개변수 (Tuning Parameters for Different Types of Models)

다양한 통계 및 기계 학습 모델에는 튜닝 매개변수 또는 하이퍼파라미터의 많은 예가 있습니다.

- 부스팅(Boosting)은 연속적으로 생성되고 이전 모델에 의존하는(depends on) 일련의 기본(base) 모델들을 결합하는 앙상블 방법입니다. 부스팅 반복 횟수(number of boosting iterations)는 대개 최적화가 필요한 중요한 튜닝 매개변수입니다.

- 고전적인 단일 계층 인공 신경망(일명 다층 퍼셉트론(multilayer perceptron))에서 예측 변수들은 둘 이상의 은닉 유닛(hidden units)을 사용하여 결합(combined)됩니다. 은닉 유닛은 (일반적으로 시그모이드와 같은 비선형(nonlinear) 함수인) *활성화 함수(activation function)*로 캡처(captured)된 예측 변수의 선형 결합(linear combinations)입니다. 그런 다음 은닉 유닛은 결과 유닛(outcome units)에 연결됩니다; 회귀 모델에는 하나의 결과 유닛이 사용되며 분류에는 여러 결과 유닛이 필요합니다. 은닉 유닛의 수와 활성화 함수의 유형은 중요한 구조적 튜닝 매개변수입니다.

- 최신 경사 하강법(gradient descent) 방법은 올바른 최적화 매개변수를 찾아 개선됩니다. 이러한 하이퍼파라미터의 예로는 학습률(learning rates), 모멘텀(momentum), 최적화 반복/에포크 수(number of optimization iterations/epochs)가 있습니다 (Goodfellow, Bengio, and Courville 2016). 신경망과 일부 앙상블 모델은 경사 하강법을 사용하여 모델 매개변수를 추정합니다. 경사 하강법과 관련된 튜닝 매개변수는 구조적 매개변수가 아니지만 튜닝이 필요한 경우가 많습니다.

어떤 경우에는 전처리(preprocessing) 기술에 튜닝이 필요합니다.

- 주성분 분석(principal component analysis) 또는 부분 최소 제곱(partial least squares)이라고 하는 그것의 지도 학습(supervised) 사촌(cousin)의 경우, 예측 변수들이 다중공선성(collinearity)과 관련된 더 나은 속성을 갖는 새로운 인공 피처로 대체됩니다. 추출된 성분(components) 수는 튜닝될 수 있습니다.

- 대치(Imputation) 방법은 하나 이상의 예측 변수의 완전한(complete) 값을 사용하여 누락된 예측 변수 값을 추정합니다. 한 가지 효과적인 대치 도구는 완전한 열(columns)의 $`K`$-최근접 이웃을 사용하여 누락된 값을 예측합니다. 이웃의 수는 평균화(averaging) 양을 조절(modulates)하며 튜닝될 수 있습니다.

일부 고전적 통계 모델에도 구조적 매개변수가 있습니다.

- 이진 회귀(binary regression)에서는 보통 로짓(logit) 연결(link)이 사용됩니다(즉, 로지스틱 회귀). 프로빗(probit) 및 보완적(complementary) 로그-로그(cloglog)와 같은 다른 연결 함수도 사용할 수 있습니다 (Dobson 1999). 이 예제는 다음 섹션에서 더 자세히 설명합니다.

- 비베이지안(Non-Bayesian) 종단(longitudinal) 및 반복 측정(repeated measures) 모델은 데이터의 공분산(covariance) 또는 상관관계(correlation) 구조에 대한 지정이 필요합니다. 옵션에는 복합 대칭(compound symmetric, 일명 교환 가능(exchangeable)), 자기회귀(autoregressive), 토플리츠(Toeplitz) 등이 포함됩니다 (Littell, Pendergast, and Natarajan 2000).

매개변수를 튜닝하는 것이 부적절한 반례(counterexample)는 베이지안 분석에 필요한 사전 분포(prior distribution)입니다. 사전 확률(prior)은 증거(evidence)나 데이터가 고려되기 전에 어떤 양(quantity)의 분포에 대한 분석가의 신념(belief)을 캡슐화(encapsulates)합니다. 예를 들어 [11장](ch11.xhtml#compare)에서 우리는 베이지안 ANOVA 모델을 사용했으며 (대칭 분포라는 점을 넘어) 회귀 매개변수의 사전 확률이 무엇이어야 하는지 불분명했습니다. 우리는 꼬리가 두꺼운 1의 자유도를 갖는 _t_-분포를 사전 확률로 선택했습니다; 이는 우리의 추가된 불확실성을 반영합니다. 우리의 사전 믿음(prior beliefs)은 최적화의 대상이 되어서는 안 됩니다. 튜닝 매개변수는 일반적으로 성능을 위해 최적화되지만 "올바른 결과"를 얻기 위해 사전 확률을 조정(tweaked)해서는 안 됩니다.

###### 경고 (Warning)

튜닝할 필요가 _없는_ 매개변수의 또 다른 (아마도 더 논쟁의 여지가 있는(debatable)) 반례는 랜덤 포레스트 또는 배깅(bagging) 모델의 트리 수입니다. 이 값은 대신 결과의 수치적 안정성을 보장할 만큼 충분히 크게 선택되어야 합니다; 즉, 신뢰할 수 있는 결과를 생성할 만큼 충분히 큰 한 이 값을 튜닝해도 성능이 향상되지는 않습니다. 랜덤 포레스트의 경우 이 값은 일반적으로 수천 단위인 반면 배깅에 필요한 트리 수는 대략 50~100개입니다.

# 우리는 무엇을 최적화하는가? (What Do We Optimize?)

튜닝 매개변수를 최적화할 때 어떻게 모델을 평가해야 할까요? 그것은 모델과 모델의 목적(purpose)에 따라 다릅니다.

튜닝 매개변수의 통계적 속성을 다루기 쉬운(tractable) 경우에는 일반적인 통계적 속성을 목적 함수(objective function)로 사용할 수 있습니다. 예를 들어 이진 로지스틱 회귀의 경우 우도(likelihood) 또는 정보 기준(information criteria)을 최대화하여 연결 함수(link function)를 선택할 수 있습니다. 그러나 이러한 통계적 특성은 정확도 지향적(accuracy-oriented) 특성을 사용하여 달성한 결과와 일치(align)하지 않을 수 있습니다. 예로서, Friedman (2001)은 부스팅 트리 앙상블에서 트리의 수를 최적화했으며 우도와 정확도를 최대화할 때 서로 다른 결과를 발견했습니다.

> 과적합을 통해 우도를 저하시키는 것(degrading)이 실제로 오분류(misclassification) 오류율(error rate)을 향상(improves)시킵니다. 직관에 반하는 것(counterintuitive)일 수 있지만 이것은 모순(contradiction)이 아닙니다; 우도와 오류율은 피팅 품질(fit quality)의 다른 측면을 측정합니다.

설명을 위해, [그림 12-1](#two-class-dat)에 표시된 2개의 예측 변수, 2개의 클래스, 그리고 593개의 데이터 포인트를 갖는 훈련 세트로 구성된 분류(classification) 데이터를 고려해 보겠습니다.

이러한 데이터에 선형 클래스 경계를 피팅하는 것으로 시작할 수 있습니다. 이 작업을 수행하는 일반적인 방법은 _로지스틱 회귀_ 형태의 일반화 선형 모델(generalized linear model)을 사용하는 것입니다. 이 모델은 _로짓(logit)_ 변환을 사용하여 샘플이 클래스 1이 될 *로그 오즈(log odds)*를 연관시킵니다(relates):

```math
\log\left( \frac{\pi}{1 - \pi} \right) = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{p}x_{p}
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1201.png" alt="tmwr 1201" />
<h6 id="figure-12-1.-an-example-two-class-classification-data-set-with-two-predictors.">그림 12-1. 두 개의 예측 변수가 있는 2-클래스 분류 데이터 세트의 예.</h6>
</figure>

일반화 선형 모델의 맥락(context)에서, 로짓 함수는 결과($`\pi`$)와 예측 변수 사이의 *연결 함수(link function)*입니다. _프로빗(probit)_ 모델을 포함하는 다른 연결 함수들이 있습니다.

```math
\Phi^{- 1}(\pi) = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{p}x_{p}
```

여기서 $`\Phi`$는 누적(cumulative) 표준 정규 함수입니다. 그리고 _보완적 로그-로그(complementary log-log)_ 모델도 있습니다.

```math
\log\left( - \log(1 - \pi) \right) = \beta_{0} + \beta_{1}x_{1} + ... + \beta_{p}x_{p}
```

이러한 각 모델은 선형 클래스 경계를 생성합니다(results in). 우리는 어떤 것을 사용해야 할까요? 이 데이터에 대해 모델 매개변수 수는 변하지 않기 때문에 통계적 접근 방식은 각 모델에 대해 (로그) 우도를 계산하고 큰 값을 갖는 모델을 결정하는 것입니다. 전통적으로 우도는 [5장](ch05.xhtml#splitting)과 [10장](ch10.xhtml#resampling)의 데이터 분할 또는 리샘플링과 같은 접근 방식을 사용하지 않고 매개변수를 추정하는 데 사용된 것과 동일한 데이터를 사용하여 계산됩니다.

`training_set` 데이터 프레임에 대해, 서로 다른 모델을 계산하고 훈련 세트에 대한 우도 통계를 추출하는 함수를 (`broom::glance()`를 사용하여) 만들어 보겠습니다.

```
library(tidymodels)
tidymodels_prefer()

llhood <- function(...) {
  logistic_reg() %>%
    set_engine("glm", ...) %>%
    fit(Class ~ ., data = training_set) %>%
    glance() %>%
    select(logLik)
}

bind_rows(
  llhood(),
  llhood(family = binomial(link = "probit")),
  llhood(family = binomial(link = "cloglog"))
) %>%
  mutate(link = c("logit", "probit", "c-log-log"))  %>%
  arrange(desc(logLik))
#> # A tibble: 3 × 2
#>   logLik link
#>    <dbl> <chr>
#> 1  -258. logit
#> 2  -262. probit
#> 3  -270. c-log-log
```

이러한 결과에 따르면 로지스틱 모델이 최고의 통계적 특성을 가지고 있습니다.

로그 우도 값의 척도(scale)로 볼 때, 이러한 차이가 중요한지 아니면 무시할 수 있는지(negligible) 이해하기는 어렵습니다. 이 분석을 개선하는 한 가지 방법은 통계를 리샘플링하고 모델링 데이터를 성능 추정에 사용되는 데이터와 분리하는 것입니다. 이 작은 데이터 세트에서는 10-겹 교차 검증 반복이 리샘플링을 위한 좋은 선택입니다. yardstick 패키지에서 `mn_log_loss()` 함수는 음의(negative) 로그 우도를 추정하는 데 사용되며 결과는 [그림 12-2](#resampled-log-lhood)에 나와 있습니다.

```
set.seed(1201)
rs <- vfold_cv(training_set, repeats = 10)

# 리샘플링된 개별 성능 추정치를 반환합니다.
lloss <- function(...) {
  perf_meas <- metric_set(roc_auc, mn_log_loss)

  logistic_reg() %>%
    set_engine("glm", ...) %>%
    fit_resamples(Class ~ A + B, rs, metrics = perf_meas) %>%
    collect_metrics(summarize = FALSE) %>%
    select(id, id2, .metric, .estimate)
}

resampled_res <-
  bind_rows(
    lloss()                                    %>% mutate(model = "logistic"),
    lloss(family = binomial(link = "probit"))  %>% mutate(model = "probit"),
    lloss(family = binomial(link = "cloglog")) %>% mutate(model = "c-log-log")
  ) %>%
  # 로그 손실(log-loss)을 로그 우도(log-likelihood)로 변환:
  mutate(.estimate = ifelse(.metric == "mn_log_loss", -.estimate, .estimate)) %>%
  group_by(model, .metric) %>%
  summarize(
    mean = mean(.estimate, na.rm = TRUE),
    std_err = sd(.estimate, na.rm = TRUE) / sum(!is.na(.estimate)),
    .groups = "drop"
  )

resampled_res %>%
  filter(.metric == "mn_log_loss") %>%
  ggplot(aes(x = mean, y = model)) +
  geom_point() +
  geom_errorbar(aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err),
                width = .1) +
  labs(y = NULL, x = "log-likelihood")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1202.png" alt="tmwr 1202" />
<h6 id="figure-12-2.-means-and-approximate-90-confidence-intervals-for-the-resampled-binomial-log-likelihood-with-three-different-link-functions.">그림 12-2. 세 가지 다른 연결 함수가 있는 리샘플링된 이항 로그 우도에 대한 평균 및 대략적인 90% 신뢰 구간.</h6>
</figure>

###### 참고 (Note)

이 값들의 척도(scale)는 이전 값들과 다릅니다. 왜냐하면 더 작은 데이터 세트에서 계산되었기 때문입니다; `broom::glance()`가 생성한 값은 합계(sum)인 반면 `yardstick::mn_log_loss()`는 평균입니다.

이러한 결과는 연결 함수의 선택이 중요하며 로지스틱 모델이 우수(superior)하다는 상당한 증거가 있음을 보여줍니다.

다른 지표는 어떨까요? 우리는 각 리샘플에 대한 ROC 곡선 아래 면적(ROC AUC)도 계산했습니다. 여러 확률 임계값(thresholds)에 걸쳐 모델의 판별(discriminative) 능력을 반영하는 이러한 결과는 [그림 12-3](#resampled-roc)에서 차이가 부족함을(lack of difference) 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1203.png" alt="tmwr 1203" />
<h6 id="figure-12-3.-means-and-approximate-90-confidence-intervals-for-the-resampled-area-under-the-roc-curve-with-three-different-link-functions.">그림 12-3. 세 가지 다른 연결 함수가 있는 리샘플링된 ROC 곡선 아래 면적에 대한 평균 및 대략적인 90% 신뢰 구간.</h6>
</figure>

구간의 겹침과 x축의 척도를 감안할 때 이러한 옵션 중 어느 것이든 사용할 수 있습니다. [그림 12-4](#three-link-fits)에서 198개 데이터 포인트의 테스트 세트에 세 모델의 클래스 경계를 오버레이(overlaid)할 때 이를 다시 볼 수 있습니다.

###### 경고 (Warning)

이 연습은 지표가 다르면 튜닝 매개변수 값 선택에 대한 결정이 달라질 수 있음을 강조합니다. 이 경우 하나의 지표는 모델을 명확하게 분류(sort)하는 것으로 보이지만 다른 지표는 아무런 차이를 나타내지 않습니다.

지표 최적화(Metric optimization)는 지표의 악용(gaming of metrics)을 포함한 여러 문제를 탐구하는 Thomas and Uminsky (2020)에 의해 철저히 논의되었습니다. 그들은 다음과 같이 경고합니다.

> 현재 AI 접근 방식에서 지표 최적화의 불합리한(unreasonable) 효과는 현장에 대한 근본적인 도전이며, 본질적인 모순을 낳습니다. 오로지(solely) 지표만 최적화하는 것은 최적의 결과와는 거리가 먼 결과를 낳습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1204.png" alt="tmwr 1204" />
<h6 id="figure-12-4.-the-linear-class-boundary-fits-for-three-link-functions.">그림 12-4. 세 가지 연결 함수에 대한 선형 클래스 경계 피팅.</h6>
</figure>

# 잘못된 매개변수 추정치의 결과 (The Consequences of Poor Parameter Estimates)

많은 튜닝 매개변수가 모델 복잡성의 양을 조절(modulate)합니다. 복잡성이 증가할수록 종종 모델이 모방(emulate)할 수 있는 패턴의 유연성(malleability)이 증가함을 의미합니다. 예를 들어 [8장](ch08.xhtml#recipes)에 표시된 대로 스플라인 함수의 자유도를 추가하면 예측 방정식의 얽힘(intricacy)이 증가합니다. 이는 데이터의 근본적인(underlying) 모티프(motifs)가 복잡할 때 장점이지만 새로운 데이터에서 재현되지 않을 우연한(chance) 패턴을 과대 해석(overinterpretation)하는 결과를 낳을 수도 있습니다. *과적합(Overfitting)*은 모델이 훈련 데이터에 너무 많이 적응(adapts)하는 상황입니다; 모델을 구축하는 데 사용된 데이터에 대해서는 성능이 뛰어나지만 새로운 데이터에 대해서는 성능이 떨어집니다.

###### 경고 (Warning)

모델 매개변수를 튜닝하면 모델 복잡성이 증가할 수 있으므로 잘못 선택하면 과적합이 발생할 수 있습니다.

이 장의 첫 번째 섹션에서 설명한 단일 계층 신경망(single-layer neural network) 모델을 상기해 보십시오. 단일 은닉 유닛과 시그모이드 활성화 함수가 있는 분류용 신경망은 사실상(for all intents and purposes) 단순한 로지스틱 회귀일 뿐입니다. 그러나 은닉 유닛의 수가 증가함에 따라 모델의 복잡성도 증가합니다. 사실, Cybenko (1989)는 네트워크 모델이 시그모이드 활성화 유닛을 사용할 때 은닉 유닛이 충분히 많기만 하다면 해당 모델이 보편적 근사기(universal function approximator)임을 보여주었습니다.

우리는 이전 섹션의 동일한 2-클래스 데이터에 신경망 분류 모델을 피팅하며 은닉 유닛의 수를 변경(varying)했습니다. 성능 지표로 ROC 곡선 아래 면적을 사용하면 은닉 유닛이 더 많이 추가됨에 따라 훈련 세트에 대한 모델의 유효성이 증가합니다. 네트워크 모델은 철저하고(thoroughly) 세심하게(meticulously) 훈련 세트를 학습합니다. 모델이 훈련 세트 ROC 값을 기준으로 자신을 판단한다면 오류를 거의 제거할 수 있도록 많은 수의 은닉 유닛을 선호합니다.

[5장](ch05.xhtml#splitting)과 [10장](ch10.xhtml#resampling)에서는 단순히 훈련 세트를 재예측(repredicting)하는 것이 모델 평가를 위한 좋지 않은 접근 방식임을 시연했습니다. 여기서 신경망은 훈련 세트에서 보는 패턴을 매우 빠르게 과대 해석(overinterpret)하기 시작합니다. [그림 12-5](#two-class-boundaries)에서 훈련 및 테스트 세트에 겹쳐진 (훈련 세트로 개발된) 세 가지 예제 클래스 경계를 비교해 보십시오.

단일 유닛 모델은 (선형이 되도록 제약되어 있기 때문에) 데이터에 매우 유연하게 적응하지 못합니다. 4개의 은닉 유닛이 있는 모델은 데이터 주류(mainstream)에서 멀리 떨어진 값에 대한 비현실적인 경계를 만들어 과적합의 징후를 보이기 시작합니다. 이는 데이터 오른쪽 위(upper-right) 모서리에 있는 첫 번째 클래스의 단일 데이터 포인트 때문에 발생합니다. 은닉 유닛이 20개 정도 되면 모델은 훈련 세트를 암기(memorizing)하기 시작하여 재대입(resubstitution) 오류율을 최소화하기 위해 해당 데이터 주변에 작은 섬(islands)을 만듭니다. 이러한 패턴은 테스트 세트에서 반복되지 않습니다. 이 마지막 패널은 모델이 효과적이도록 복잡성을 제어하는 튜닝 매개변수를 조절해야(modulated) 하는 방법을 보여주는 좋은 예입니다. 20-유닛 모델의 경우 훈련 세트 ROC 곡선 아래 면적(ROC AUC)은 0.944이지만 테스트 세트 값은 0.855입니다.

이러한 과적합 발생은 우리가 플롯(plot)할 수 있는 두 개의 예측 변수에서 분명히 나타납니다. 그러나 일반적으로 모델 과적합을 감지하려면 정량적(quantitative) 접근 방식을 사용해야 합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1205.png" alt="tmwr 1205" />
<h6 id="figure-12-5.-class-boundaries-for-three-models-with-increasing-numbers-of-hidden-units.-the-boundaries-are-fit-on-the-training-set-and-shown-for-the-training-and-test-sets.">그림 12-5. 증가하는 은닉 유닛 수가 있는 세 모델에 대한 클래스 경계. 경계는 훈련 세트에 피팅되었으며 훈련 및 테스트 세트에 대해 표시됩니다.</h6>
</figure>

###### 참고 (Note)

모델이 훈련 세트를 과도하게 강조하는(overemphasizing) 때를 감지하는 해결책은 표본 외(out-of-sample) 데이터를 사용하는 것입니다.

테스트 세트를 사용하는 것보다 어떤 형태의 리샘플링이 필요합니다. 이는 반복적인 접근 방식(10-겹 교차 검증) 또는 단일 데이터 소스(검증 세트)를 의미할 수 있습니다.

# 최적화를 위한 두 가지 일반적인 전략 (Two General Strategies for Optimization)

튜닝 매개변수 최적화는 일반적으로 그리드 검색(grid search)과 반복 검색(iterative search)이라는 두 가지 범주 중 하나에 속합니다.

*그리드 검색(Grid search)*은 평가할 매개변수 값 세트를 미리 정의하는 경우입니다. 그리드 검색과 관련된 주요 선택(choices)은 그리드를 만드는 방법과 평가할 매개변수 조합의 수입니다. 그리드 검색은 매개변수 공간(parameter space)을 포괄하는 데 필요한 그리드 포인트의 수가 차원의 저주(curse of dimensionality)로 인해 관리할 수 없게 될(unmanageable) 수 있으므로 종종 비효율적이라고 판단됩니다. 이 우려에는 진실이 담겨 있지만 프로세스가 최적화되지 않았을 때 사실(most true)입니다. 이에 대해서는 [13장](ch13.xhtml#grid-search)에서 자세히 설명합니다.

_반복 검색(Iterative search)_ 또는 순차(sequential) 검색은 이전 결과를 기반으로 순차적으로 새로운 매개변수 조합을 발견할(discover) 때입니다. 거의 모든 비선형 최적화 방법이 적절하지만 일부는 다른 것보다 더 효율적입니다. 어떤 경우에는 최적화 프로세스를 시작하기 위해 하나 이상의 매개변수 조합에 대한 초기 결과 세트가 필요합니다. 반복 검색에 대해서는 [14장](ch14.xhtml#iterative-search)에서 자세히 설명합니다.

[그림 12-6](#tuning-strategies)은 0과 1 사이의 범위를 갖는 두 튜닝 매개변수가 있는 상황에 대해 이 두 가지 접근 방식을 보여주는 두 패널을 보여줍니다. 각각에서 등고선(contours) 세트는 매개변수와 결과 간의 실제(시뮬레이션된) 관계를 보여줍니다. 최적의 결과는 오른쪽 위 모서리에 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1206.png" alt="tmwr 1206" />
<h6 id="figure-12-6.-examples-of-predefined-grid-tuning-and-an-iterative-search-method.-the-lines-represent-contours-of-a-performance-metric-it-is-best-in-the-upper-right-side-of-the-plot.">그림 12-6. 미리 정의된(predefined) 그리드 튜닝 및 반복 검색 방법의 예. 선은 성능 지표의 등고선을 나타냅니다; 플롯의 오른쪽 윗부분이 좋습니다.</h6>
</figure>

[그림 12-6](#tuning-strategies)의 왼쪽 패널은 공간 채우기 설계(space-filling design)라고 하는 그리드 유형을 보여줍니다. 이것은 튜닝 매개변수 조합들이 서로 가깝지 않도록 매개변수 공간을 포괄하기 위해 고안된 실험 설계(experimental design)의 한 유형입니다. 이 설계를 위한 결과는 어떤 지점(points)도 진정 최적인 위치에 정확히 배치하지 않습니다. 그러나 하나의 지점이 일반적인 부근(vicinity)에 있으며, 아마도 최적 값의 노이즈 범위 내에 있는 성능 지표 결과를 가질 것입니다.

[그림 12-6](#tuning-strategies)의 오른쪽 패널은 전역(global) 검색 방법인 Nelder-Mead 심플렉스 방법(simplex method) (Olsson and Nelson 1975)의 결과를 보여줍니다. 시작점(starting point)은 매개변수 공간의 왼쪽 아래 부분입니다. 검색은 최적의 위치에 도달할 때까지 공간을 가로질러 굽이쳐(meanders) 이동하며, 수치적으로 최상의 값에 최대한 가까워지기 위해 노력합니다(strives). 이 특정 검색 방법은 효과적이긴 하지만 효율성(efficiency) 측면에서는 알려져 있지(not known for) 않습니다; 특히 최적 값 근처에서 많은 함수 평가가 필요합니다. [14장](ch14.xhtml#iterative-search)에서는 더 효율적인 검색 알고리즘에 대해 설명합니다.

###### 참고 (Note)

하이브리드 전략도 하나의 옵션이며 잘 작동할 수 있습니다. 초기 그리드 검색 후, 최상의 그리드 조합에서 순차적 최적화(sequential optimization)를 시작할 수 있습니다.

이러한 전략의 예는 다음 두 장에서 자세히 논의합니다. 계속 진행하기 전에, dials 패키지를 사용하여 tidymodels에서 튜닝 매개변수 객체로 작업하는 방법을 알아보겠습니다.

# tidymodels의 튜닝 매개변수 (Tuning Parameters in tidymodels)

우리는 이미 이전 장들에서 레시피 및 모델 지정(specifications)에 대한 튜닝 매개변수에 해당하는 상당히 많은 인수를 다루었습니다. 다음을 튜닝하는 것이 가능합니다.

- [8장](ch08.xhtml#recipes)에서 논의한 인근 지역(neighborhoods)을 "기타(other)" 범주로 결합하기 위한 임계값(threshold) (인수명 `threshold`)

- 자연 스플라인(natural spline)의 자유도 수 (`deg_free`, [8장](ch08.xhtml#recipes))

- 트리 기반 모델에서 분할을 실행하는 데 필요한 데이터 포인트 수 (`min_n`, [6장](ch06.xhtml#models))

- 페널티 모델에서의 정규화(regularization) 양 (`penalty`, [6장](ch06.xhtml#models))

parsnip 모델 지정의 경우, 두 가지 종류의 매개변수 인수가 있습니다. *기본 인수(Main arguments)*는 자주 성능 최적화의 대상이 되며 여러 엔진에서 사용할 수 있는 인수입니다. 기본 튜닝 매개변수는 모델 지정(specification) 함수의 최상위 인수입니다. 예를 들어 `rand_forest()` 함수에는 자주 지정되거나 최적화되는 `trees`, `min_n`, `mtry`와 같은 기본 인수가 있습니다.

보조 튜닝 매개변수 세트는 _엔진에 고유한(engine specific)_ 매개변수입니다. 이들은 자주 최적화되지 않거나 특정 엔진에만 고유합니다. 다시 랜덤 포레스트를 예로 들면, ranger 패키지에는 다른 패키지에서 사용되지 않는 몇 가지 인수가 포함되어 있습니다. 한 가지 예는 게인 페널티화(gain penalization)로, 트리 귀납(induction) 프로세스에서 예측 변수 선택을 정규화합니다. 이 매개변수는 앙상블에 사용되는 예측 변수 수와 성능 간의 트레이드오프(trade-off)를 조절하는 데 도움이 될 수 있습니다 (Wundervald, Parnell, and Domijan 2020). `ranger()`에서 이 인수의 이름은 `regularization.factor`입니다. parsnip 모델 지정을 통해 값을 지정하기 위해 `set_engine()`에 보충 인수로 추가됩니다.

```
rand_forest(trees = 2000, min_n = 10) %>%                   # <- 기본 인수
  set_engine("ranger", regularization.factor = 0.5)         # <- 엔진 고유 인수
```

###### 경고 (Warning)

기본 인수는 엔진 간의 불일치(inconsistencies)를 제거하기 위해 조화된(harmonized) 명명 체계를 사용하는 반면, 엔진 고유 인수는 그렇지 않습니다.

어떤 인수를 최적화해야 하는지 tidymodels 함수에 어떻게 신호를 보낼 수 있을까요? 매개변수는 `tune()` 값을 할당함으로써 튜닝용으로 표시(marked)됩니다. 이 장의 앞부분에서 사용된 단일 계층 신경망의 경우 다음을 사용하여 은닉 유닛의 수가 튜닝용으로 지정됩니다(designated):

```
neural_net_spec <-
  mlp(hidden_units = tune()) %>%
  set_engine("keras")
```

`tune()` 함수는 특정 매개변수 값을 실행하지 않습니다; 단지 표현식(expression)을 반환할 뿐입니다.

```
tune()
#> tune()
```

이 `tune()` 값을 인수에 내장(Embedding)하면 최적화를 위해 해당 매개변수에 태그(tag)를 지정합니다. 다음 두 장에 나오는 모델 튜닝 함수는 모델 지정 및/또는 레시피를 파싱(parse)하여 태그가 지정된 매개변수를 발견합니다. 이러한 함수는 매개변수의 특성(가능한 값의 범위 등)을 이해하므로 자동으로 매개변수를 구성하고 처리할 수 있습니다.

객체에 대한 튜닝 매개변수를 열거(enumerate)하려면 `extract_parameter_set_dials()` 함수를 사용하세요.

```
extract_parameter_set_dials(neural_net_spec)
#> Collection of 1 parameters for tuning
#>
#>    identifier         type    object
#>  hidden_units hidden_units nparam[+]
```

결과에는 `nparam[+]` 값이 표시되어 은닉 유닛 수가 수치(numeric) 매개변수임을 나타냅니다.

매개변수에 이름을 연관시키는 선택적인 식별(identification) 인수가 있습니다. 이는 다른 곳에서 동일한 종류의 매개변수가 튜닝될 때 유용할(come in handy) 수 있습니다. 예를 들어, [10장](ch10.xhtml#resampling) 끝부분의 Ames 주택 데이터 예시에서 레시피는 경도와 위도를 모두 스플라인 함수로 인코딩했습니다. 잠재적으로 서로 다른 수준의 매끄러움(smoothness)을 갖도록 두 개의 스플라인 함수를 튜닝하려면 각 예측 변수에 대해 한 번씩 `step_ns()`를 두 번 호출합니다. 매개변수를 식별할 수 있도록(identifiable) 식별 인수는 임의의 문자열을 사용할 수 있습니다.

```
ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train)  %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = tune()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Longitude, deg_free = tune("longitude df")) %>%
  step_ns(Latitude,  deg_free = tune("latitude df"))

recipes_param <- extract_parameter_set_dials(ames_rec)
recipes_param
#> Collection of 3 parameters for tuning
#>
#>    identifier      type    object
#>     threshold threshold nparam[+]
#>  longitude df  deg_free nparam[+]
#>   latitude df  deg_free nparam[+]
```

두 스플라인 매개변수 모두 `identifier`와 `type` 열이 동일하지 않다는 점에 유의하십시오.

워크플로를 사용하여 레시피와 모델 지정이 결합되면 두 매개변수 세트가 모두 표시됩니다.

```
wflow_param <-
  workflow() %>%
  add_recipe(ames_rec) %>%
  add_model(neural_net_spec) %>%
  extract_parameter_set_dials()
wflow_param
#> Collection of 4 parameters for tuning
#>
#>    identifier         type    object
#>  hidden_units hidden_units nparam[+]
#>     threshold    threshold nparam[+]
#>  longitude df     deg_free nparam[+]
#>   latitude df     deg_free nparam[+]
```

###### 경고 (Warning)

신경망은 비선형 패턴을 모방하는 능력이 매우 뛰어납니다. 이 유형의 모델에 스플라인 항을 추가하는 것은 불필요(unnecessary)합니다; 설명을 위해서만 이 모델과 레시피를 결합했습니다.

각 튜닝 매개변수 인수에는 dials 패키지에 해당하는 함수가 있습니다. 대다수(vast majority)의 경우 함수는 매개변수 인수와 동일한 이름을 갖습니다.

```
hidden_units()
#> # Hidden Units (quantitative)
#> Range: [1, 10]
threshold()
#> Threshold (quantitative)
#> Range: [0, 1]
```

`deg_free` 매개변수는 반례(counterexample)입니다; 자유도(degrees of freedom)의 개념은 다양한 컨텍스트에서 나타납니다. 스플라인과 함께 사용될 때 스플라인에 대해 기본적으로(by default) 호출되는(invoked) `spline_degree()`라는 특수화된(specialized) dials 함수가 있습니다.

```
spline_degree()
#> Piecewise Polynomial Degree (quantitative)
#> Range: [1, 10]
```

dials 패키지에는 특정 매개변수 객체를 추출하기 위한 편의 함수도 있습니다.

```
# id 값을 사용하여 매개변수 식별:
wflow_param %>% extract_parameter_dials("threshold")
#> Threshold (quantitative)
#> Range: [0, 0.1]
```

매개변수 세트 내부에서 매개변수의 범위(range)는 제자리(in place)에서 업데이트될 수도 있습니다.

```
extract_parameter_set_dials(ames_rec) %>%
  update(threshold = threshold(c(0.8, 1.0)))
#> Collection of 3 parameters for tuning
#>
#>    identifier      type    object
#>     threshold threshold nparam[+]
#>  longitude df  deg_free nparam[+]
#>   latitude df  deg_free nparam[+]
```

`extract_parameter_set_dials()`에 의해 생성된 *매개변수 세트(parameter sets)*는 (필요한 경우) tidymodels 튜닝 함수에서 소비(consumed)됩니다. 튜닝 매개변수 객체에 대한 기본값을 수정(modification)해야 하는 경우, 수정된 매개변수 세트가 적절한 튜닝 함수로 전달됩니다.

###### 참고 (Note)

일부 튜닝 매개변수는 데이터 차원에 의존합니다(depend on). 예를 들어, 최근접 이웃의 수는 1에서 데이터의 행 수 사이여야 합니다.

어떤 경우에는 가능한 값의 범위에 대한 합리적인 기본값을 갖는 것이 쉽습니다. 다른 경우에는 매개변수 범위가 중요하며 (함부로) 가정할 수 없습니다. 랜덤 포레스트 모델의 주요 튜닝 매개변수는 트리의 각 분할에 대해 무작위로 샘플링되는 예측 변수 열의 수이며, 일반적으로 `mtry()`로 표시(denoted)됩니다. 예측 변수의 수를 모르면 이 매개변수 범위를 미리 구성할 수 없으며 마무리(finalization)가 필요합니다.

```
rf_spec <-
  rand_forest(mtry = tune()) %>%
  set_engine("ranger", regularization.factor = tune("regularization"))

rf_param <- extract_parameter_set_dials(rf_spec)
rf_param
#> Collection of 2 parameters for tuning
#>
#>      identifier                  type    object
#>            mtry                  mtry nparam[?]
#>  regularization regularization.factor nparam[+]
#>
#> Model parameters needing finalization:
#>    # Randomly Selected Predictors ('mtry')
#>
#> See `?dials::finalize` or `?dials::update.parameters` for more information.
```

완전한(Complete) 매개변수 객체는 해당 요약에 `[+]`를 갖습니다; `[?]` 값은 가능한 범위의 적어도 한쪽 끝이 누락(missing)되었음을 나타냅니다. 이를 처리하는 두 가지 방법이 있습니다. 첫 번째는 데이터 차원에 대해 알고 있는 것을 기반으로 범위를 추가하기 위해 `update()`를 사용하는 것입니다.

```
rf_param %>%
  update(mtry = mtry(c(1, 70)))
#> Collection of 2 parameters for tuning
#>
#>      identifier                  type    object
#>            mtry                  mtry nparam[+]
#>  regularization regularization.factor nparam[+]
```

그러나 이 접근 방식은 열을 추가하거나 빼는(subtract) 단계를 사용하는 레시피가 워크플로에 연결된(attached) 경우에는 작동하지 않을 수 있습니다. 해당 단계들이 튜닝용으로 예정되어(slated for) 있지 않은 경우, `finalize()` 함수는 레시피를 한 번 실행하여 차원(dimensions)을 얻을 수 있습니다.

```
pca_rec <-
  recipe(Sale_Price ~ ., data = ames_train) %>%
  # 면적(square-footage) 예측 변수를 선택하고 이들의 PCA 성분을 추출합니다.
  step_normalize(contains("SF")) %>%
  # 예측 변수 분산의 95%를 포착하는 데 필요한
  # 성분 수를 선택합니다.
  step_pca(contains("SF"), threshold = .95)

updated_param <-
  workflow() %>%
  add_model(rf_spec) %>%
  add_recipe(pca_rec) %>%
  extract_parameter_set_dials() %>%
  finalize(ames_train)
updated_param
#> Collection of 2 parameters for tuning
#>
#>      identifier                  type    object
#>            mtry                  mtry nparam[+]
#>  regularization regularization.factor nparam[+]
updated_param %>% extract_parameter_dials("mtry")
#> # Randomly Selected Predictors (quantitative)
#> Range: [1, 74]
```

레시피가 준비되면 `finalize()` 함수는 `mtry`의 위쪽(upper) 범위를 74개의 예측 변수로 설정하는 것을 학습합니다.

게다가(Additionally), `extract_parameter_set_dials()`의 결과에는 엔진 고유 매개변수(있는 경우)도 포함됩니다. 이것들은 기본 인수와 동일한 방식으로 발견(discovered)되어 매개변수 세트에 포함됩니다. dials 패키지에는 잠재적으로 튜닝 가능한 모든 엔진 고유 매개변수에 대한 매개변수 함수가 포함되어 있습니다.

```
rf_param
#> Collection of 2 parameters for tuning
#>
#>      identifier                  type    object
#>            mtry                  mtry nparam[?]
#>  regularization regularization.factor nparam[+]
#>
#> Model parameters needing finalization:
#>    # Randomly Selected Predictors ('mtry')
#>
#> See `?dials::finalize` or `?dials::update.parameters` for more information.
regularization_factor()
#> Gain Penalization (quantitative)
#> Range: [0, 1]
```

마지막으로, 일부 튜닝 매개변수는 변환(transformations)과 연관시키는 것이 좋습니다. 이에 대한 좋은 예는 많은 정규화된(regularized) 회귀 모델과 관련된 페널티 매개변수입니다. 이 매개변수는 음수(nonnegative)가 아니며(nonnegative) 로그 단위로 값을 변경하는 것이 일반적입니다. 기본 dials 매개변수 객체는 기본적으로 변환이 사용됨을 나타냅니다.

```
penalty()
#> Amount of Regularization (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-10, 0]
```

이것은 특히 범위를 변경할 때 아는 것이 중요합니다. 새로운 범위 값은 변환된 단위(transformed units)여야 합니다.

```
# 페널티 값을 0.1에서 1.0 사이로 설정하는 올바른(correct) 방법
penalty(c(-1, 0)) %>% value_sample(1000) %>% summary()
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#>   0.101   0.181   0.327   0.400   0.589   0.999

# 잘못된(incorrect) 방법:
penalty(c(0.1, 1.0)) %>% value_sample(1000) %>% summary()
#>    Min. 1st Qu.  Median    Mean 3rd Qu.    Max.
#>    1.26    2.21    3.68    4.26    5.89   10.00
```

원하는 경우 `trans` 인수를 사용하여 스케일(scale)을 변경할 수 있습니다. 자연 단위(natural units)를 사용할 수 있지만 범위는 동일합니다.

```
penalty(trans = NULL, range = 10^c(-10, 0))
#> Amount of Regularization (quantitative)
#> Range: [1e-10, 1]
```

# 이 장의 요약 (Chapter Summary)

이 장에서는 데이터에서 직접 추정할 수 없는 모델 하이퍼파라미터를 튜닝하는 프로세스를 소개했습니다. 이러한 매개변수를 튜닝하면 종종 모델이 지나치게 복잡해지도록 허용함으로써 과적합을 초래할 수 있으므로, 적절한 평가 지표와 함께 리샘플링된 데이터 세트를 사용하는 것이 중요합니다. 올바른 값을 결정하기 위한 두 가지 일반적인 전략인 그리드 검색과 반복 검색이 있으며, 이는 다음 두 장에서 자세히 살펴볼 것입니다. tidymodels에서는 최적화할 매개변수를 식별하는 데 `tune()` 함수가 사용되며, dials 패키지의 함수는 튜닝 매개변수 객체를 추출하고 이들과 상호작용할 수 있습니다.
