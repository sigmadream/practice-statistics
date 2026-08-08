# 19장. 언제 예측을 신뢰해야 합니까? (When Should You Trust Your Predictions?)

예측 모델은 입력 데이터가 주어지면 거의 항상 예측을 생성할 수 있습니다. 그러나 많은 상황에서 그러한 예측을 생성하는 것은 부적절합니다(inappropriate). 새로운 데이터 포인트가 모델을 생성하는 데 사용된 데이터 범위를 크게 벗어나는(well outside) 경우, 예측을 하는 것은 부적절한 *외삽(extrapolation)*일 수 있습니다. 부적절한 예측의 더 정성적인(qualitative) 예는 모델이 완전히 다른 맥락에서 사용되는 경우입니다. [14장](ch14.xhtml#iterative-search)에서 사용된 세포 분할 데이터는 인간 유방암 세포를 이미지 내에서 정확하게 분리(isolated)할 수 있는지 여부를 플래그(flags)로 표시합니다. 이 데이터로 구축된 모델을 동일한 목적으로 위 세포(stomach cells)에 부적절하게 적용할 수 있습니다. 예측을 생성할 수는 있지만, 다른 세포 유형에 적용할 수 있을 가능성은 낮습니다(unlikely to be applicable).

이 장에서는 잠재적인 예측 품질을 정량화(quantifying)하기 위한 두 가지 방법에 대해 논의합니다:

모호한 영역 (Equivocal zones)
이 방법은 예측값을 사용하여 결과가 의심스러울(suspect) 수 있음을 사용자에게 알립니다(alert).

적용 가능성 (Applicability)
이 방법은 예측 변수를 사용하여 새로운 샘플에 대한 외삽의 양(있는 경우)을 측정합니다.

# 모호한 결과 (Equivocal Results)

모델 결과에 코로나19(COVID-19)에 감염되었을 확률이 51%라고 나타난다면, 약간의 회의감(skepticism)을 가지고 진단을 바라보는 것이 자연스러울 것입니다. 실제로 규제 기관에서는 종종 많은 의료 진단 기기에 *모호한 영역(equivocal zone)*을 두도록 요구합니다. 이 영역은 환자에게 예측을 보고해서는 안 되는 결과 범위로, 예를 들어 환자에게 보고하기에는 너무 불확실한 일부 코로나19 검사 결과 범위입니다. 예시는 Danowski et al. (1970) 및 Kerleguer et al. (2003)을 참조하십시오. 의료 진단 외의 분야에서 생성된 모델에도 동일한 개념을 적용할 수 있습니다.

###### 경고 (Warning)

어떤 경우에는 예측과 관련된(associated with) 불확실성의 양이 너무 높아서 신뢰할 수 없습니다.

두 개의 클래스와 두 개의 예측 변수(*x* 및 *y*)로 분류 데이터를 시뮬레이션할 수 있는 함수를 사용해 보겠습니다. 참(true) 모델은 다음 방정식이 있는 로지스틱 회귀 모델입니다:

``` math
{logit}(p) = - 1 - 2x - \frac{x^{2}}{5} + 2y^{2}
```

두 예측 변수는 0.70의 상관관계를 갖는 이변량 정규 분포(bivariate normal distribution)를 따릅니다. 200개 샘플의 훈련 세트와 50개 샘플의 테스트 세트를 생성해 보겠습니다:

```
library(tidymodels)
tidymodels_prefer()

simulate_two_classes <-
  function (n, error = 0.1, eqn = quote(-1 - 2 * x - 0.2 * x^2 + 2 * y^2))  {
    # 약간 상관관계가 있는 예측 변수
    sigma <- matrix(c(1, 0.7, 0.7, 1), nrow = 2, ncol = 2)
    dat <- MASS::mvrnorm(n = n, mu = c(0, 0), Sigma = sigma)
    colnames(dat) <- c("x", "y")
    cls <- paste0("class_", 1:2)
    dat <-
      as_tibble(dat) %>%
      mutate(
        linear_pred = !!eqn,
        # 약간의 오분류(misclassification) 노이즈 추가
        linear_pred = linear_pred + rnorm(n, sd = error),
        prob = binomial()$linkinv(linear_pred),
        class = ifelse(prob > runif(n), cls[1], cls[2]),
        class = factor(class, levels = cls)
      )
    dplyr::select(dat, x, y, class)
  }

set.seed(1901)
training_set <- simulate_two_classes(200)
testing_set  <- simulate_two_classes(50)
```

베이지안 방법을 사용하여 로지스틱 회귀 모델을 추정(estimate)합니다(매개변수에 기본 가우시안 사전 분포(prior distributions) 사용):

```
two_class_mod <-
  logistic_reg() %>%
  set_engine("stan", seed = 1902) %>%
  fit(class ~ . + I(x^2)+ I(y^2), data = training_set)
print(two_class_mod, digits = 3)
#> parsnip model object
#>
#> stan_glm
#>  family:       binomial [logit]
#>  formula:      class ~ . + I(x^2) + I(y^2)
#>  observations: 200
#>  predictors:   5
#> ------
#>             Median MAD_SD
#> (Intercept)  1.092  0.287
#> x            2.290  0.423
#> y            0.314  0.354
#> I(x^2)       0.077  0.307
#> I(y^2)      -2.465  0.424
#>
#> ------
#> * For help interpreting the printed output see ?print.stanreg
#> * For info on the priors used see ?prior_summary.stanreg
```

피팅된(fitted) 클래스 경계(boundary)가 [그림 19-1](#glm-boundaries)의 테스트 세트에 겹쳐져(overlaid) 있습니다. 클래스 경계에 가장 가까운 데이터 포인트가 가장 불확실합니다. 값이 약간만 변경되어도 예측된 클래스가 변경될 수 있습니다. 일부 결과를 실격(disqualifying) 처리하는 간단한 방법 중 하나는 값이 50%(또는 특정 상황에 적절한 확률 컷오프)를 중심으로 일부 범위 내에 있으면 "모호함(equivocal)"이라고 부르는 것입니다. 모델이 적용되는 문제에 따라, 이는 신뢰할 수 있는 예측이 가능하기 전에 또 다른 측정값을 수집(collect)해야 하거나 더 많은 정보가 필요함을 나타낼 수 있습니다.

불확실한 결과를 제거했을 때 성능이 얼마나 향상되는지에 기초하여 컷오프 주변 대역(band)의 너비를 결정할 수 있습니다. 그러나 보고 가능한(reportable) 비율(사용 가능한 결과의 예상 비율)도 추정해야 합니다. 예를 들어, 실제(real-world) 상황에서 완벽한 성능을 내지만 모델에 전달된 샘플의 2%에 대해서만 예측을 발표(release)하는 것은 유용하지 않을 것입니다.

테스트 세트를 사용하여 성능 향상과 충분히 보고 가능한(reportable) 결과 확보 사이의 균형을 결정해 보겠습니다. 다음을 사용하여 예측이 생성됩니다:

```
test_pred <- augment(two_class_mod, testing_set)
test_pred %>% head()
#> # A tibble: 6 × 6
#>        x      y class   .pred_class .pred_class_1 .pred_class_2
#>    <dbl>  <dbl> <fct>   <fct>               <dbl>         <dbl>
#> 1  1.12  -0.176 class_2 class_2           0.0256          0.974
#> 2 -0.126 -0.582 class_2 class_1           0.555           0.445
#> 3  1.92   0.615 class_2 class_2           0.00620         0.994
#> 4 -0.400  0.252 class_2 class_2           0.472           0.528
#> 5  1.30   1.09  class_1 class_2           0.163           0.837
#> 6  2.59   1.36  class_2 class_2           0.0317          0.968
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1901.png" alt="tmwr 1901" />
<h6 id="figure-19-1.-simulated-two-class-data-set-with-a-logistic-regression-fit-and-decision-boundary.">그림 19-1. 로지스틱 회귀 적합(fit) 및 결정 경계가 있는 시뮬레이션된 2-클래스 데이터 세트.</h6>
</figure>

tidymodels에서 probably 패키지에는 모호한 영역에 대한 함수가 포함되어 있습니다. 두 개의 클래스가 있는 경우 `make_two_class_pred()` 함수는 모호한 영역이 있는 예측 클래스를 갖는 팩터 유사(factor-like) 열을 만듭니다:

```
library(probably)

lvls <- levels(training_set$class)

test_pred <-
  test_pred %>%
  mutate(.pred_with_eqz = make_two_class_pred(.pred_class_1, lvls, buffer = 0.15))

test_pred %>% count(.pred_with_eqz)
#> # A tibble: 3 × 2
#>   .pred_with_eqz     n
#>       <clss_prd> <int>
#> 1           [EQ]     9
#> 2        class_1    20
#> 3        class_2    21
```

0.50 ± 0.15 이내에 있는 행에는 `[EQ]` 값이 지정됩니다.

###### 참고 (Note)

중요: 이 예에서 `[EQ]`는 팩터 수준(factor level)이 아니라 해당 열의 어트리뷰트(attribute)입니다.

팩터 수준이 원래 데이터와 동일하므로 오차 행렬(confusion matrices) 및 기타 통계를 오류 없이 계산할 수 있습니다. yardstick 패키지의 표준 함수를 사용할 때 모호한 결과는 `NA`로 변환되며 확정적인(hard) 클래스 예측을 사용하는 계산에는 사용되지 않습니다. 다음 오차 행렬들의 차이점을 확인하십시오(notice):

```
# 모든 데이터
test_pred %>% conf_mat(class, .pred_class)
#>           Truth
#> Prediction class_1 class_2
#>    class_1      20       6
#>    class_2       5      19

# 보고 가능한(Reportable) 결과만:
test_pred %>% conf_mat(class, .pred_with_eqz)
#>           Truth
#> Prediction class_1 class_2
#>    class_1      17       3
#>    class_2       5      16
```

데이터에서 이러한 행을 필터링하기 위한 `is_equivocal()` 함수도 제공(available)됩니다.

모호한 영역이 정확도를 높이는 데 도움이 됩니까? [그림 19-2](#equivocal-zone-results)와 같이 다양한 버퍼 크기를 살펴보겠습니다:

```
# 버퍼를 변경한 다음 성능을 계산하는 함수입니다.
eq_zone_results <- function(buffer) {
  test_pred <-
    test_pred %>%
    mutate(.pred_with_eqz = make_two_class_pred(.pred_class_1, lvls, buffer = buffer))
  acc <- test_pred %>% accuracy(class, .pred_with_eqz)
  rep_rate <- reportable_rate(test_pred$.pred_with_eqz)
  tibble(accuracy = acc$.estimate, reportable = rep_rate, buffer = buffer)
}

# 일련의 버퍼를 평가(Evaluate)하고 결과를 플로팅합니다.
map_dfr(seq(0, .1, length.out = 40), eq_zone_results) %>%
  pivot_longer(c(-buffer), names_to = "statistic", values_to = "value") %>%
  ggplot(aes(x = buffer, y = value, lty = statistic)) +
  geom_step(size = 1.2, alpha = 0.8) +
  labs(y = NULL, lty = NULL)
```

[그림 19-2](#equivocal-zone-results)는 예측의 약 10%를 사용할 수 없게 되는(unusable) 대가(cost)로 정확도가 몇 퍼센트 포인트 향상된다는 것을 보여줍니다! 이러한 타협점(compromise)의 가치(value)는 모델 예측이 어떻게 사용될지에 따라 다릅니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1902.png" alt="tmwr 1902" />
<h6 id="figure-19-2.-the-effect-of-equivocal-zones-on-model-performance.">그림 19-2. 모호한 영역이 모델 성능에 미치는 영향.</h6>
</figure>

이 분석은 예측 클래스 확률이 분류 모델 불확실성의 근본적인(fundamental) 측정값이기 때문에 예측 클래스 확률을 사용하여 데이터 포인트를 실격(disqualify) 처리하는 데 중점을 두었습니다(focused on). 조금 더 나은 접근 방식은 클래스 확률의 표준 오차를 사용하는 것입니다. 베이지안 모델을 사용했기 때문에 우리가 찾은 확률 추정치는 사실 사후 예측 분포(posterior predictive distribution)의 평균입니다. 즉(In other words), 베이지안 모델은 클래스 확률에 대한 분포를 제공합니다. 이 분포의 표준 편차를 측정하면 확률의 *예측 표준 오차(standard error of prediction)*를 얻을 수 있습니다. 대부분의 경우(In most cases) 이 값은 평균 클래스 확률과 직접적으로 관련(directly related)이 있습니다. 확률이 $`p`$인 베르누이(Bernoulli) 확률 변수의 경우 분산(variance)이 $`p(1 - p)`$임을 기억할 것입니다(might recall). 이러한 관계 때문에 표준 오차는 확률이 50%일 때 가장 큽니다. 클래스 확률을 사용하여 모호한 결과를 할당하는 대신, 예측의 표준 오차에 컷오프를 사용할 수도 있습니다.

예측 표준 오차의 한 가지 중요한 측면(aspect)은 단순한 클래스 확률 이상을 고려(takes into account)한다는 것입니다. 상당한 외삽(extrapolation)이나 비정상적인(aberrant) 예측 변수 값이 있는 경우에는 표준 오차가 증가할 수 있습니다. 예측의 표준 오차를 사용하는 것의 이점(benefit)은 그것이 문제의 소지가 있는 예측(단순히 불확실한 것과는 반대되는)에 플래그(flag)를 지정할 수도 있다는 것입니다. 베이지안 모델을 사용한 한 가지 이유는 이 모델이 자연스럽게 예측 표준 오차를 추정하기 때문입니다; 이를 계산할 수 있는 모델은 많지 않습니다. 테스트 세트의 경우 `type = "pred_int"`를 사용하면 상한(upper limits)과 하한(lower limits)이 생성되고 `std_error`는 해당 수량(quantity)에 대한 열을 추가합니다. 80% 구간(intervals)의 경우:

```
test_pred <-
  test_pred %>%
  bind_cols(
    predict(two_class_mod, testing_set, type = "pred_int", std_error = TRUE)
  )
```

모델과 데이터가 잘 작동(well behaved)하는 예제의 경우, [그림 19-3](#std-errors)은 공간(space) 전체에 걸친 예측 표준 오차를 보여줍니다:

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1903.png" alt="tmwr 1903" />
<h6 id="figure-19-3.-the-effect-of-the-standard-error-of-prediction-overlaid-with-the-test-set-data.">그림 19-3. 테스트 세트 데이터와 중첩된(overlaid) 예측 표준 오차의 효과.</h6>
</figure>

샘플 예측을 배제(preclude)하기 위한 척도(measure)로 표준 오차를 사용하는 것은 숫자형 결과가 있는 모델에도 적용될 수 있습니다. 그러나 다음 섹션에 표시된 것처럼 이 방법이 항상 효과가 있는(work) 것은 아닐 수 있습니다.

# 모델 적용 가능성 판단 (Determining Model Applicability)

모호한 영역은 모델 출력을 기반으로 예측의 신뢰성을 측정(measure)하려고 시도합니다. 예측의 표준 오차와 같은 모델 통계(statistics)가 외삽의 영향을 측정할 수 없으므로, 예측을 신뢰할지 여부를 평가하고(assess) "특정 데이터 포인트를 예측하는 데 우리 모델이 적용 가능한가(applicable)?"라는 질문에 대답하기 위해 다른 방법이 필요할 수 있습니다. [Kuhn and Johnson (2020)](https://oreil.ly/rsZqK)에서 광범위하게 사용되고 [2장](ch02.xhtml#tidyverse)에서 처음 살펴본 시카고 기차 데이터를 살펴보겠습니다(Let's take). 목표는 매일 Clark and Lake 기차역(train station)에 들어오는 고객 수를 예측하는 것입니다.

modeldata 패키지(예제 데이터 세트가 있는 tidymodels 패키지)의 데이터 세트에는 2001년 1월 22일부터 2016년 8월 28일 사이의 일일 값이 있습니다. 데이터의 마지막 2주를 사용하여 작은 테스트 세트를 만들어 보겠습니다:

```
## `Chicago` 데이터 세트와 `stations` 모두 로드
data(Chicago)

Chicago <- Chicago %>% select(ridership, date, one_of(stations))

n <- nrow(Chicago)

Chicago_train <- Chicago %>% slice(1:(n - 14))
Chicago_test  <- Chicago %>% slice((n - 13):n)
```

주요(main) 예측 변수는 날짜뿐만 아니라 Clark and Lake를 포함한 여러 기차역에서 지연된(lagged) 승객(ridership) 데이터입니다. 승객 예측 변수는 서로 높은 상관관계가 있습니다(highly correlated). 다음 레시피에서 날짜 열은 몇 개의 새로운 피처로 확장(expanded)되고, 승객 예측 변수는 부분 최소 제곱(PLS) 성분을 사용하여 표현됩니다. [16장](ch16.xhtml#dimensionality)에서 논의한 바와 같이 PLS(Geladi and Kowalski 1986)는 새로운 피처가 서로 연관되지 않도록(decorrelated) 처리되었지만(but are) 결과 데이터를 예측하는 역할은 하는, 주성분 분석의 지도 버전입니다.

전처리된 데이터를 사용하여 표준 선형 모델을 피팅합니다:

```
base_recipe <-
  recipe(ridership ~ ., data = Chicago_train) %>%
  # 날짜 피처 만들기
  step_date(date) %>%
  step_holiday(date, keep_original_cols = FALSE) %>%
  # 팩터 열에서 가변수 만들기
  step_dummy(all_nominal()) %>%
  # 고유값이 하나인 열 제거
  step_zv(all_predictors()) %>%
  step_normalize(!!!stations)%>%
  step_pls(!!!stations, num_comp = 10, outcome = vars(ridership))

lm_spec <-
  linear_reg() %>%
  set_engine("lm")

lm_wflow <-
  workflow() %>%
  add_recipe(base_recipe) %>%
  add_model(lm_spec)

set.seed(1902)
lm_fit <- fit(lm_wflow, data = Chicago_train)
```

데이터가 테스트 세트에 얼마나 잘 맞습니까(fit)? 테스트 세트에 대해 `predict()`를 수행하여 예측과 예측 구간(prediction intervals)을 모두 찾을 수 있습니다:

```
res_test <-
  predict(lm_fit, Chicago_test) %>%
  bind_cols(
    predict(lm_fit, Chicago_test, type = "pred_int"),
    Chicago_test
  )

res_test %>% select(date, ridership, starts_with(".pred"))
#> # A tibble: 14 × 5
#>   date       ridership .pred .pred_lower .pred_upper
#>   <date>         <dbl> <dbl>       <dbl>       <dbl>
#> 1 2016-08-15     20.6  20.3        16.2         24.5
#> 2 2016-08-16     21.0  21.3        17.1         25.4
#> 3 2016-08-17     21.0  21.4        17.3         25.6
#> 4 2016-08-18     21.3  21.4        17.3         25.5
#> 5 2016-08-19     20.4  20.9        16.7         25.0
#> 6 2016-08-20      6.22  7.52        3.34        11.7
#> # … with 8 more rows
res_test %>% rmse(ridership, .pred)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard       0.865
```

이 정도면 꽤 좋은 결과입니다. [그림 19-4](#chicago-2016)는 예측값을 95% 예측 구간과 함께 시각화한 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1904.png" alt="tmwr 1904" />
<h6 id="figure-19-4.-two-weeks-of-2016-predictions-for-the-chicago-data-along-with-95-prediction-intervals.">그림 19-4. 95% 예측 구간이 포함된 시카고 데이터의 2016년 2주 예측.</h6>
</figure>

승객 수의 규모(scale)를 고려할 때, 이렇게 단순한 모델치고는 결과가 특히 좋아 보입니다. 이 모델이 배포(deployed)되었다면, 몇 년 후인 2020년 6월에는 얼마나 잘 예측(done)했을까요? 예측 모델이 입력 데이터가 주어지면 거의 항상 그러하듯이(as a predictive model almost always will), 모델은 예측을 성공적으로 수행합니다:

```
res_2020 <-
  predict(lm_fit, Chicago_2020) %>%
  bind_cols(
    predict(lm_fit, Chicago_2020, type = "pred_int"),
    Chicago_2020
  )

res_2020 %>% select(date, contains(".pred"))
#> # A tibble: 14 × 4
#>   date       .pred .pred_lower .pred_upper
#>   <date>     <dbl>       <dbl>       <dbl>
#> 1 2020-06-01 20.1        15.9         24.3
#> 2 2020-06-02 21.4        17.2         25.6
#> 3 2020-06-03 21.5        17.3         25.6
#> 4 2020-06-04 21.3        17.1         25.4
#> 5 2020-06-05 20.7        16.6         24.9
#> 6 2020-06-06  9.04        4.88        13.2
#> # … with 8 more rows
```

비록 이 데이터가 원래 훈련 세트의 기간(time period)을 훨씬(well beyond) 지났지만, 예측 구간의 너비는 대략(about) 같습니다. 하지만 2020년 전 세계적(global) 대유행(pandemic)을 고려할 때, 이 데이터에 대한 모델의 성능은 형편(abysmal)없습니다:

```
res_2020 %>% select(date, ridership, starts_with(".pred"))
#> # A tibble: 14 × 5
#>   date       ridership .pred .pred_lower .pred_upper
#>   <date>         <dbl> <dbl>       <dbl>       <dbl>
#> 1 2020-06-01     0.002 20.1        15.9         24.3
#> 2 2020-06-02     0.005 21.4        17.2         25.6
#> 3 2020-06-03     0.566 21.5        17.3         25.6
#> 4 2020-06-04     1.66  21.3        17.1         25.4
#> 5 2020-06-05     1.95  20.7        16.6         24.9
#> 6 2020-06-06     1.08   9.04        4.88        13.2
#> # … with 8 more rows
res_2020 %>% rmse(ridership, .pred)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard        17.2
```

[그림 19-5](#chicago-2020)에서 이 끔찍한(terrible) 모델 성능을 시각적으로 확인할 수 있습니다.

선형 회귀의 신뢰 구간과 예측 구간은 데이터가 훈련 세트의 중심(center)에서 점점 더 벗어날(removed)수록(as) 확장(expand)됩니다. 하지만 그 효과는 이러한 예측이 형편없다(poor)고 플래그를 지정할 만큼 극적(dramatic)이지는 않습니다.

###### 경고 (Warning)

때때로 모델에 의해 생성된 통계(statistics)가 예측 품질을 잘 측정하지 못할 수 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1905.png" alt="tmwr 1905" />
<h6 id="figure-19-5.-two-weeks-of-2020-predictions-for-the-chicago-data-along-with-95-prediction-intervals.">그림 19-5. 95% 예측 구간이 포함된 시카고 데이터의 2020년 2주 예측.</h6>
</figure>

이 상황은 새로운 예측(즉, 모델의 *적용 영역(applicability domain)*)에 대해 모델이 얼마나 적용 가능한지를 정량화(quantify)할 수 있는 두 번째 방법론(secondary methodology)을 가짐으로써 피할 수(avoided) 있습니다. Jaworska, Nikolova-Jeliazkova, and Aldenberg (2005) 또는 Netzeva et al. (2005)과 같이 적용 영역(applicability domain) 모델을 계산하는 다양한 방법이 있습니다. 이 장에서 사용된 접근 방식은 새로운 데이터 포인트가 훈련 데이터를 얼마나 (벗어났다면) 벗어났는지(how much (if any) a new data point is beyond) 측정하려고 시도하는 비교적(fairly) 단순한 비지도 방법입니다.<sup><a href="ch19.xhtml#idm45881845223120" id="idm45881845223120-marker" data-type="noteref">1</a></sup>

###### 참고 (Note)

아이디어는 새로운 포인트가 훈련 세트와 얼마나 유사한지(similar)를 측정하는 점수를 예측에 수반(accompany)하는 것입니다.

잘 작동하는 한 가지 방법은 숫자형 예측 변수 값에 주성분 분석(PCA)을 사용하는 것입니다. 다른 기차역(California 및 Austin 역)의 승객 수(ridership)에 해당하는 두 개의 예측 변수만 사용하여 과정을 설명해 보겠습니다. 훈련 세트는 [그림 19-6](#pca-reference-dist)의 패널 (a)에 나와 있습니다. 이 역들의 승객 데이터는 높은 상관관계(highly correlated)가 있으며, 산점도(scatter plot)에 표시된 두 개의 분포는 평일(weekdays)과 주말(weekends)의 승객 수에 해당합니다.

첫 번째 단계는 훈련 데이터에 대해 PCA를 수행(conduct)하는 것입니다. 훈련 세트의 PCA 점수는 [그림 19-6](#pca-reference-dist)의 패널 (b)에 나와 있습니다. 다음으로 이 결과를 사용하여 각 훈련 세트 포인트에서 PCA 데이터의 중심까지의 거리(distance)를 측정합니다([그림 19-6](#pca-reference-dist)의 패널 (c)). 그런 다음 이 *참조 분포(reference distribution)*([그림 19-6](#pca-reference-dist)의 패널 (d))를 사용하여 데이터 포인트가 훈련 데이터의 주류(mainstream)에서 얼마나 떨어져 있는지(how far) 추정할 수 있습니다.

<figure class="width-90">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1906.png" alt="tmwr 1906" />
<h6 id="figure-19-6.-the-pca-reference-distribution-based-on-the-training-set.">그림 19-6. 훈련 세트를 기반으로 한 PCA 참조 분포.</h6>
</figure>

새로운 샘플의 경우 훈련 세트의 중심까지의 거리와 함께 PCA 점수가 계산됩니다.

그러나 새로운 샘플의 거리가 *X*일 때 그것은 무엇을 의미할까요? PCA 성분은 데이터 세트마다 범위가 다를 수 있기 때문에 거리가 너무 멀다(too large)고 할 명백한(obvious) 한계(limit)가 없습니다.

한 가지 접근 방식은 훈련 세트 데이터와의 거리를 "정상(normal)"으로 취급(treat)하는 것입니다. 새로운 샘플의 경우, 새로운 거리가 참조 분포(훈련 세트에서 파생된)의 범위와 어떻게 비교(compares to)되는지 확인할 수 있습니다. 훈련 세트 중 얼마나 많은(how much) 데이터가 새로운 샘플보다 덜 극단적(less extreme)인지 반영(reflect)하는 새로운 샘플에 대한 백분위수(percentile)를 계산할 수 있습니다.

###### 참고 (Note)

백분위수가 90%라는 것은 대부분의(most of the) 훈련 세트 데이터가 새로운 샘플보다 데이터 중심에 더 가깝다는(closer) 것을 의미합니다.

[그림 19-7](#two-new-points)의 플롯은 테스트 세트 샘플(삼각형과 파선)과 2020년 샘플(원형과 실선)을 훈련 세트의 PCA 거리와 함께 겹쳐서 표시(overlays)합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1907.png" alt="tmwr 1907" />
<h6 id="figure-19-7.-the-reference-distribution-with-two-new-points-one-using-the-test-set-and-one-from-the-2020-data.">그림 19-7. 두 개의 새로운 포인트(하나는 테스트 세트를 사용하고 다른 하나는 2020년 데이터에서 사용)가 있는 참조 분포.</h6>
</figure>

테스트 세트 포인트의 거리는 1.28입니다. 이 포인트는 훈련 세트 분포의 51.8% 백분위수에 위치하며(in), 이는 훈련 세트의 주류 내에 아늑하게(snugly) 위치(within)함을 나타냅니다.

2020년 샘플은 어떠한(any of the) 훈련 세트 샘플보다 중심에서 멀리 떨어져 있습니다(백분위수는 100%). 이는 샘플이 매우 극단적(extreme)이며, 해당 예측은 심각한 외삽(severe extrapolation)이 될 것(아마 보고되지 않아야 할(should not be reported))임을 나타냅니다.

applicable 패키지는 PCA를 사용하여 적용 영역 모델(applicability domain model)을 개발할 수 있습니다. 우리는 20개의 지연된(lagged) 역별(station) 승객 예측 변수를 PCA 분석의 입력(inputs)으로 사용할 것입니다. 거리 계산에 사용될 성분(components) 수를 결정하는 `threshold`라는 추가 인자(argument)가 있습니다. 우리의 예에서는, 우리는 승객 예측 변수의 분산(variation) 중 99%를 설명(account for)할 수 있을 만큼 충분한 성분을 사용해야 함을 나타내는 큰 값을 사용할 것입니다:

```
library(applicable)
pca_stat <- apd_pca(~ ., data = Chicago_train %>% select(one_of(stations)),
                    threshold = 0.99)
pca_stat
#> # Predictors:
#>    20
#> # Principal Components:
#>    9 components were needed
#>    to capture at least 99% of the
#>    total variation in the predictors.
```

`autoplot()` 메서드는 참조 분포를 플로팅합니다. 플로팅할 데이터를 선택하기 위한 선택적(optional) 인자가 있습니다. 훈련 세트의 거리 분포만 플로팅하려면 `distance` 값을 추가합니다. 이 코드는 [그림 19-8](#ap-autoplot)에 플롯을 생성합니다:

```
autoplot(pca_stat, distance) + labs(x = "distance")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1908.png" alt="tmwr 1908" />
<h6 id="figure-19-8.-the-results-of-using-the-autoplot-method-on-an-applicable-object.">그림 19-8. applicable 객체에서 <code>autoplot()</code> 메서드를 사용한 결과.</h6>
</figure>

x축은 거리의 값을 나타내고 y축은 분포의 백분위수(percentiles)를 표시합니다. 예를 들어, 훈련 세트 샘플 중 절반은(half of the) 거리가 3.7 미만이었습니다.

새로운 데이터에 대한 백분위수를 계산하기 위해 `score()` 함수는 `predict()`와 같은 방식(same way)으로 작동합니다:

```
score(pca_stat, Chicago_test) %>% select(starts_with("distance"))
#> # A tibble: 14 × 2
#>   distance distance_pctl
#>      <dbl>         <dbl>
#> 1     4.88          66.7
#> 2     5.21          71.4
#> 3     5.19          71.1
#> 4     5.00          68.5
#> 5     4.36          59.3
#> 6     4.10          55.2
#> # … with 8 more rows
```

이 값들은 상당히(fairly) 합리적인 것 같습니다. 2020년 데이터의 경우:

```
score(pca_stat, Chicago_2020) %>% select(starts_with("distance"))
#> # A tibble: 14 × 2
#>   distance distance_pctl
#>      <dbl>         <dbl>
#> 1     9.39          99.8
#> 2     9.40          99.8
#> 3     9.30          99.7
#> 4     9.30          99.7
#> 5     9.29          99.7
#> 6    10.1            1
#> # … with 8 more rows
```

2020년 거리 값은 이러한 예측 변수 값들이 훈련 시 모델이 본 대다수의(vast majority) 데이터의 외부에(outside of) 있음을 나타냅니다. 이러한 포인트는 예측이 아예(at all) 보고되지 않거나 회의감(skepticism)을 가지고 보여지도록(viewed) 플래그를 지정(flagged)해야 합니다.

###### 참고 (Note)

이 분석의 중요한 측면 중 하나는 적용 영역(applicability domain) 모델을 개발(develop)하는 데 사용되는 예측 변수가 무엇인지와 관련(concerns)이 있습니다. 우리의 분석에서는 원시(raw) 예측 변수 열을 사용했습니다. 그러나 모델을 구축할 때는 그 자리에(in their place) PLS 점수 피처가 사용되었습니다. `apd_pca()`는 이 중 어느 것을 사용해야 할까요? `apd_pca()` 함수는 거리가 개별(individual) 예측 변수 열 대신 PLS 점수를 반영(reflect)하도록 공식(formula) 대신 레시피를 입력으로 사용할 수도 있습니다. 두 가지 방법을 모두 평가하여 어느 방법이 더 관련성 있는(more relevant) 결과를 도출(gives)하는지 파악(understand)할 수 있습니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 모델 소비자(consumers of models)에게 예측을 보고해야 하는지 여부를 평가(evaluating)하기 위한 두 가지 방법을 보여주었습니다. 모호한 영역(Equivocal zones)은 결과/예측을 처리(deal with)하며 예측의 불확실성 정도가 너무 큰 경우 유용할 수 있습니다.

적용 영역(Applicability domain) 모델은 특성/예측 변수를 처리하며 예측을 수행할 때 발생하는 외삽의 양(있는 경우)을 정량화(quantify)합니다. 이 장에서는 적용 가능성(applicability)을 측정하는 다른 많은 방법이 있지만 주성분 분석을 사용하는 기본적(basic) 방법을 보여주었습니다. applicable 패키지에는 모든 예측 변수가 이진(binary)인 데이터 세트를 위한 특수(specialized) 방법도 포함되어 있습니다. 이 방법은 참조 분포(reference distribution)를 정의하기 위해 훈련 세트 데이터 포인트 간의 유사도 점수(similarity scores)를 계산합니다.

<sup>[1](ch19.xhtml#idm45881845223120-marker)</sup> Bartley et al. (2019)는 생태학적 연구에 적용한 또 다른 방법을 보여줍니다.
