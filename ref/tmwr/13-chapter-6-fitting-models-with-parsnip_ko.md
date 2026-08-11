# 6장. parsnip을 이용한 모델 피팅 (Fitting Models with parsnip)

parsnip 패키지는 tidymodels 메타패키지의 일부인 R 패키지 중 하나입니다. 이 패키지는 다양하고 다른 모델들에 대해 유창(fluent)하고 표준화된 인터페이스를 제공합니다. 이 장에서는 실무에서 모델을 이해하고 구축하는 데 왜 공통 인터페이스가 유익한지에 대한 동기를 부여하고 parsnip 패키지를 사용하는 방법을 보여줍니다.

특히, 우리는 몇 가지 간단한 모델링 문제에 적합할 수 있는, parsnip 객체를 사용하여 직접 `fit()` 및 `predict()`를 수행하는 방법에 초점을 맞출 것입니다. [7장](ch07.xhtml#workflows)에서는 모델과 전처리기(preprocessors)를 결합하여 `workflow` 객체라는 것으로 만드는, 많은 모델링 작업에 있어 더 나은 접근 방식을 설명합니다.

# 모델 생성 (Create a Model)

데이터가 모델링 알고리즘을 위해 숫자 행렬과 같이 준비된 형식으로 인코딩되면 모델 구축 프로세스에 사용될 수 있습니다.

선형 회귀 모델이 우리의 초기 선택이었다고 가정해 봅시다. 이는 결과 변수 데이터가 숫자이고 예측 변수가 단순한 기울기(slopes)와 절편(intercepts)의 관점에서 결과 변수와 관련되어 있음을 지정하는 것과 같습니다.

```math
y_{i} = \beta_{0} + \beta_{1}x_{1i} + ... + \beta_{p}x_{pi}
```

모델 파라미터를 추정하는 데 다양한 방법을 사용할 수 있습니다.

- *일반 선형 회귀(Ordinary linear regression)*는 모델 파라미터를 구하기 위해 전통적인 최소제곱법(method of least squares)을 사용합니다.

- *정규화된 선형 회귀(Regularized linear regression)*는 최소제곱법에 페널티(penalty)를 추가하여 예측 변수를 제거하거나 그들의 계수를 0으로 축소함으로써(shrinking) 단순성을 장려합니다. 이는 베이지안(Bayesian) 또는 비 베이지안(non-Bayesian) 기술을 사용하여 실행할 수 있습니다.

R에서 첫 번째 경우에 대해 stats 패키지를 사용할 수 있습니다. `lm()` 함수를 사용한 선형 회귀의 구문은 다음과 같습니다.

```
model <- lm(formula, data, ...)
```

여기서 `...`는 `lm()`에 전달할 다른 옵션들을 상징합니다. 이 함수에는 결과 변수를 `y`로, 예측 변수를 `x`로 전달할 수 있는 `x`/`y` 인터페이스가 _없습니다_.

두 번째 경우인 정규화를 사용한 추정을 위해 rstanarm 패키지를 사용하여 베이지안 모델을 피팅할 수 있습니다.

```
model <- stan_glm(formula, data, family = "gaussian", ...)
```

이 경우 `...`를 통해 전달되는 다른 옵션에는 파라미터의 사전 분포(prior distributions)에 대한 인수와 모델의 수치적 측면에 대한 세부 정보가 포함됩니다. `lm()`과 마찬가지로 공식(formula) 인터페이스만 사용할 수 있습니다.

정규화된 회귀에 널리 사용되는 비 베이지안 접근법은 glmnet 모델 (Friedman, Hastie, 및 Tibshirani 2010)입니다. 그것의 구문은 다음과 같습니다.

```
model <- glmnet(x = matrix, y = vector, family = "gaussian", ...)
```

이 경우 예측 변수 데이터는 이미 숫자 행렬로 포맷되어 있어야 합니다. `x`/`y` 메소드만 있고 공식 메소드는 없습니다.

이러한 인터페이스들은 데이터가 모델 함수에 전달되는 방식이나 인수의 측면에서 이질적(heterogeneous)이라는 점에 유의하세요. 첫 번째 문제는 여러 다른 패키지에 걸쳐 모델을 피팅하기 위해서는 데이터를 여러 다른 방식으로 포맷해야 한다는 점입니다. `lm()` 및 `stan_glm()`에는 공식 인터페이스만 있는 반면 `glmnet()`에는 없습니다. 다른 유형의 모델의 경우 인터페이스가 더욱 이질적일 수 있습니다. 데이터 분석을 하려는 사람에게 이러한 차이점은 각 패키지의 구문을 암기해야 하므로 매우 실망스러울 수 있습니다.

tidymodels의 경우 모델을 지정하는 접근 방식이 더 통합되도록 의도되었습니다.

수학적 구조를 바탕으로 모델 유형 지정  
선형 회귀, 랜덤 포레스트, KNN 등. 대부분 이것은 Stan이나 glmnet과 같이 사용되어야 하는 소프트웨어 패키지를 반영합니다. 이것들은 그 자체로(in their own right) 모델이며, parsnip은 이것들을 모델링을 위한 엔진(engines)으로 사용함으로써 일관된 인터페이스를 제공합니다.

필요한 경우 모델의 모드(mode) 선언  
모드는 예측 결과의 유형을 반영합니다. 숫자 결과의 경우 모드는 회귀(regression)이고 질적 결과의 경우 분류(classification)입니다.<sup>[1](ch06.xhtml#idm45881873685760-marker)</sup> 모델 알고리즘이 선형 회귀와 같이 한 가지 유형의 예측 결과만 처리할 수 있는 경우 모드는 이미 설정된 것입니다.

이러한 사양은 데이터를 참조하지 않고 구축됩니다. 예를 들어 우리가 설명한 세 가지 경우의 경우:

```
library(tidymodels)
tidymodels_prefer()

linear_reg() %>% set_engine("lm")
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm

linear_reg() %>% set_engine("glmnet")
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: glmnet

linear_reg() %>% set_engine("stan")
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: stan
```

모델의 세부 사항이 지정되면 `fit()` 함수(공식을 사용) 또는 `fit_xy()` 함수(데이터가 이미 전처리된 경우)를 사용하여 모델 추정을 수행할 수 있습니다. parsnip 패키지는 사용자가 기본 모델의 인터페이스에 무관심할 수 있게 해 줍니다. 모델링 패키지의 함수에 `x`/`y` 인터페이스만 있더라도 항상 공식을 사용할 수 있습니다.

`translate()` 함수는 parsnip이 사용자의 코드를 패키지의 구문으로 어떻게 변환하는지에 대한 세부 정보를 제공할 수 있습니다.

```
linear_reg() %>% set_engine("lm") %>% translate()
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
#>
#> Model fit template:
#> stats::lm(formula = missing_arg(), data = missing_arg(), weights = missing_arg())

linear_reg(penalty = 1) %>% set_engine("glmnet") %>% translate()
#> Linear Regression Model Specification (regression)
#>
#> Main Arguments:
#>   penalty = 1
#>
#> Computational engine: glmnet
#>
#> Model fit template:
#> glmnet::glmnet(x = missing_arg(), y = missing_arg(), weights = missing_arg(),
#>     family = "gaussian")

linear_reg() %>% set_engine("stan") %>% translate()
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: stan
#>
#> Model fit template:
#> rstanarm::stan_glm(formula = missing_arg(), data = missing_arg(),
#>     weights = missing_arg(), family = stats::gaussian, refresh = 0)
```

`missing_arg()`는 아직 제공되지 않은 데이터의 자리 표시자(placeholder)일 뿐이라는 점에 유의하세요.

###### 참고 (Note)

우리는 glmnet 엔진을 위한 필수적인 `penalty` 인수를 제공했습니다. 또한 Stan과 glmnet 엔진의 경우 `family` 인수가 기본값으로 자동 추가되었습니다. 이 섹션의 뒷부분에서 보여지듯이 이 옵션은 변경할 수 있습니다.\

경도 및 위도의 함수로만 Ames 데이터의 주택 판매 가격을 예측하는 방법을 살펴보겠습니다.<sup>[2](ch06.xhtml#idm45881873576064-marker)</sup>

```
lm_model <-
  linear_reg() %>%
  set_engine("lm")

lm_form_fit <-
  lm_model %>%
  # Sale_Price가 이미 로그 변환되었다는 것을 기억하세요
  fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

lm_xy_fit <-
  lm_model %>%
  fit_xy(
    x = ames_train %>% select(Longitude, Latitude),
    y = ames_train %>% pull(Sale_Price)
  )

lm_form_fit
#> parsnip model object
#>
#>
#> Call:
#> stats::lm(formula = Sale_Price ~ Longitude + Latitude, data = data)
#>
#> Coefficients:
#> (Intercept)    Longitude     Latitude
#>     -302.97        -2.07         2.71
lm_xy_fit
#> parsnip model object
#>
#>
#> Call:
#> stats::lm(formula = ..y ~ ., data = data)
#>
#> Coefficients:
#> (Intercept)    Longitude     Latitude
#>     -302.97        -2.07         2.71
```

parsnip은 서로 다른 패키지에 대해 일관된 모델 인터페이스를 가능하게 할 뿐만 아니라 모델 인수의 일관성도 제공합니다. 동일한 모델을 피팅하는 서로 다른 함수가 서로 다른 인수 이름을 가지는 것은 흔한 일입니다. 랜덤 포레스트(Random forest) 모델 함수가 좋은 예입니다. 일반적으로 사용되는 세 가지 인수는 앙상블(ensemble)의 트리 수, 트리 내의 각 분할(split)에서 무작위로 샘플링할 예측 변수의 수, 그리고 분할을 수행하는 데 필요한 데이터 포인트의 수입니다. 이 알고리즘을 구현하는 3개의 서로 다른 R 패키지에 대해 해당 인수는 [표 6-1](#rand-forest-args)에 표시되어 있습니다.

| 인수 유형 (Argument type) | ranger          | randomForest | sparklyr                  |
| ------------------------- | --------------- | ------------ | ------------------------- |
| 샘플링된 예측 변수의 수   | `mtry`          | `mtry`       | `feature_subset_strategy` |
| 트리의 수                 | `num.trees`     | `ntree`      | `num_trees`               |
| 분할할 데이터 포인트의 수 | `min.node.size` | `nodesize`   | `min_instances_per_node`  |

<span id="rand-forest-args">표 6-1. 다양한 랜덤 포레스트 함수에 대한 인수 이름 예시</span>

인수 지정을 덜 고통스럽게 만들기 위해, parsnip은 패키지 내부 및 패키지 간에 공통 인수 이름을 사용합니다. [표 6-2](#parsnip-args)는 랜덤 포레스트의 경우 parsnip 모델이 어떤 이름을 사용하는지 보여줍니다.

| 인수 유형 (Argument type) | parsnip |
| ------------------------- | ------- |
| 샘플링된 예측 변수의 수   | `mtry`  |
| 트리의 수                 | `trees` |
| 분할할 데이터 포인트의 수 | `min_n` |

<span id="parsnip-args">표 6-2. parsnip에서 사용하는 랜덤 포레스트 인수 이름</span>

인정하건대 이것은 기억해야 할 또 하나의 인수 집합입니다. 그러나 다른 유형의 모델이 동일한 인수 유형을 가질 때 이러한 이름은 여전히 적용됩니다. 예를 들어 부스팅 트리 앙상블(boosted tree ensembles)도 많은 수의 트리 기반 모델을 생성하므로 `trees`가 여기서도 사용되고 `min_n` 등도 마찬가지로 사용됩니다.

원래의 인수 이름 중 일부는 상당히 전문 용어(jargony)일 수 있습니다. 예를 들어 glmnet 모델에서 사용할 정규화 양을 지정하기 위해 그리스 문자 `lambda`가 사용됩니다. 이 수학적 표기법은 통계 문헌에서 널리 사용되지만 `lambda`가 무엇을 나타내는지(특히 모델 결과를 소비하는 사람들에게) 명확하지 않은 사람도 많습니다. 이것이 정규화에 사용되는 페널티이므로 parsnip은 인수 이름 `penalty`로 표준화합니다. 비슷하게 KNN 모델의 이웃 수(neighbors)는 `k` 대신 `neighbors`로 불립니다. 인수 이름을 표준화할 때 우리의 경험 법칙은 다음과 같습니다.

> 실무자가 도식이나 표에 이 이름들을 포함시킨다면, 그 결과를 보는 사람들이 이 이름을 이해할 수 있을까?

parsnip 인수 이름이 원래 이름과 어떻게 매핑되는지 이해하려면 모델에 대한 도움말 파일(`?rand_forest`를 통해 사용 가능)과 `translate()` 함수를 사용하십시오:

```
rand_forest(trees = 1000, min_n = 5) %>%
  set_engine("ranger") %>%
  set_mode("regression") %>%
  translate()
#> Random Forest Model Specification (regression)
#>
#> Main Arguments:
#>   trees = 1000
#>   min_n = 5
#>
#> Computational engine: ranger
#>
#> Model fit template:
#> ranger::ranger(x = missing_arg(), y = missing_arg(), case.weights = missing_arg(),
#>     num.trees = 1000, min.node.size = min_rows(~5, x), num.threads = 1,
#>     verbose = FALSE, seed = sample.int(10^5, 1))
```

parsnip의 모델링 함수는 모델 인수를 두 가지 범주로 분리합니다.

주요 인수 (Main arguments)  
더 일반적으로 사용되며 여러 엔진에 걸쳐 사용할 수 있는 경향이 있음

엔진 인수 (Engine arguments)  
특정 엔진에 국한되거나 덜 자주 사용됨

예를 들어 이전 랜덤 포레스트 코드의 변환에서는 `num.threads`, `verbose` 및 `seed` 인수가 기본적으로 추가되었습니다. 이러한 인수는 랜덤 포레스트 모델의 ranger 구현에 국한되며 주요 인수로서는 의미가 없습니다. 엔진별(engine-specific) 인수는 `set_engine()`에서 지정할 수 있습니다. 예를 들어 `ranger::ranger()` 함수가 피팅에 대한 자세한 정보를 인쇄하도록 하려면:

```
rand_forest(trees = 1000, min_n = 5) %>%
  set_engine("ranger", verbose = TRUE) %>%
  set_mode("regression")
#> Random Forest Model Specification (regression)
#>
#> Main Arguments:
#>   trees = 1000
#>   min_n = 5
#>
#> Engine-Specific Arguments:
#>   verbose = TRUE
#>
#> Computational engine: ranger
```

# 모델 결과 사용하기 (Use the Model Results)

모델이 생성되고 피팅되면 다양한 방식으로 결과를 사용할 수 있습니다. 결과를 도식화(plot)하거나 출력(print)하거나 달리 모델 출력을 검사할 수 있습니다. 피팅된 모델을 포함하여 몇 가지 수량이 parsnip 모델 객체에 저장됩니다. 이는 `fit`이라는 요소(element)에서 찾을 수 있으며, `extract_fit_engine()` 함수를 사용하여 반환할 수 있습니다.

```
lm_form_fit %>% extract_fit_engine()
#>
#> Call:
#> stats::lm(formula = Sale_Price ~ Longitude + Latitude, data = data)
#>
#> Coefficients:
#> (Intercept)    Longitude     Latitude
#>     -302.97        -2.07         2.71
```

인쇄 및 도식화와 같은 일반적인 메소드(methods)를 이 객체에 적용할 수 있습니다.

```

```

lm_form_fit %>% extract_fit_engine() %>% vcov()
#> (Intercept) Longitude Latitude
#> (Intercept) 207.311 1.57466 -1.42397
#> Longitude 1.575 0.01655 -0.00060
#> Latitude -1.424 -0.00060 0.03254

```

###### 경고 (Warning)

parsnip 모델의 `fit` 요소를 모델 예측 함수에 절대 전달하지 마십시오. 즉, `predict(lm_form_fit)`를 사용하되 `predict(lm_form_fit$fit)`는 *사용하지 마십시오*. 데이터가 어떤 방식이로든 전처리된 경우, 잘못된 예측이 생성됩니다(때로는 오류 없이). 기본 모델의 예측 함수는 모델을 실행하기 전에 데이터에 어떤 변환이 이루어졌는지 알지 못합니다. 예측에 대한 자세한 내용은 ["예측하기 (Make Predictions)"](#parsnip-predictions)를 참조하십시오.

base R의 일부 기존 메소드에 대한 한 가지 문제는 결과가 가장 유용하지 않을 수 있는 방식으로 저장된다는 것입니다. 예를 들어 `lm` 객체에 대한 `summary()` 메소드는 파라미터 값, 불확실성 추정치 및 p-값이 포함된 표를 포함하여 모델 피팅 결과를 인쇄하는 데 사용할 수 있습니다. 이러한 특정한 결과들을 저장할 수도 있습니다.

```

model_res <-
lm_form_fit %>%
extract_fit_engine() %>%
summary()

# 모델 계수 표는 `coef` 메소드를 통해 액세스할 수 있습니다.

param_est <- coef(model_res)
class(param_est)
#> [1] "matrix" "array"
param_est
#> Estimate Std. Error t value Pr(>|t|)
#> (Intercept) -302.974 14.3983 -21.04 3.640e-90
#> Longitude -2.075 0.1286 -16.13 1.395e-55
#> Latitude 2.710 0.1804 15.02 9.289e-49

```

이 결과에 대해 주목해야 할 몇 가지 사항이 있습니다. 첫째, 객체는 숫자 행렬(numeric matrix)입니다. 이 데이터 구조는 계산된 모든 결과가 숫자이고 행렬 객체가 데이터 프레임보다 더 효율적으로 저장되기 때문에 선택되었을 가능성이 큽니다. 이러한 선택은 컴퓨팅 효율성이 매우 중요했던 1970년대 후반에 이루어졌을 것입니다. 둘째, 숫자가 아닌 데이터(계수 레이블)는 행 이름(row names)에 포함됩니다. 파라미터 레이블을 행 이름으로 유지하는 것은 원래 S 언어의 규약과 매우 일치합니다.

합리적인 다음 단계는 파라미터 값을 시각화하는 것일 수 있습니다. 이렇게 하려면 파라미터 행렬을 데이터 프레임으로 변환하는 것이 합리적입니다. 행 이름을 도식에서 사용할 수 있도록 열(column)로 추가할 수 있습니다. 그러나 기존 행렬의 열 이름 중 일부는 일반 데이터 프레임에 유효한 R 열 이름이 아닐 수 있음을 알 수 있습니다(`"Pr(>|t|)"`). 또 다른 복잡한 문제는 열 이름의 일관성입니다. `lm` 객체의 경우 검정 통계량(test statistic)에 대한 열은 `"Pr(>|t|)"`이지만 다른 모델의 경우 다른 검정이 사용될 수 있으며, 그 결과 열 이름이 달라지고 (`"Pr(>|z|)"`) 검정의 유형이 열 이름에 인코딩됩니다.

이러한 추가적인 데이터 형식화(formatting) 단계를 극복하는 것이 불가능하지는 않지만, 특히 다양한 유형의 모델에 대해 다를 수 있기 때문에 장애물이 됩니다. 행렬은 데이터를 단일 유형(숫자)으로 제한하기 때문에 재사용성이 높은 데이터 구조가 아닙니다. 또한 차원 이름(dimension names)에 일부 데이터를 유지하는 것 역시 문제가 되는데, 이러한 데이터가 일반적으로 사용되려면 추출되어야 하기 때문입니다.

해결책으로 broom 패키지는 많은 유형의 모델 객체를 깔끔한(tidy) 구조로 변환할 수 있습니다. 예를 들어 선형 모델에 `tidy()` 메소드를 사용하면 다음과 같이 생성됩니다.

```

tidy(lm_form_fit)
#> # A tibble: 3 × 5
#> term estimate std.error statistic p.value
#> <chr> <dbl> <dbl> <dbl> <dbl>
#> 1 (Intercept) -303. 14.4 -21.0 3.64e-90
#> 2 Longitude -2.07 0.129 -16.1 1.40e-55
#> 3 Latitude 2.71 0.180 15.0 9.29e-49

```

열 이름은 모델 전체에 걸쳐 표준화되어 있으며 어떤 추가 데이터(통계적 검정 유형)도 포함하지 않습니다. 이전에 행 이름에 포함되었던 데이터는 이제 `term`이라는 열에 있습니다. tidymodels 생태계의 한 가지 중요한 원칙은 함수가 *예측 가능하고, 일관되며, 놀랍지 않은(unsurprising)* 값을 반환해야 한다는 것입니다.

<span id="parsnip-predictions"></span>
# 예측하기 (Make Predictions)

parsnip이 기존 R 모델링 함수와 다른 또 다른 영역은 `predict()`에서 반환되는 값의 형식입니다. 예측의 경우 parsnip은 항상 다음 규칙을 준수합니다.

1.  결과는 항상 tibble입니다.

2.  tibble의 열 이름은 항상 예측 가능합니다.

3.  tibble에는 항상 입력 데이터 세트와 동일한 수의 행이 있습니다.

예를 들어 숫자 데이터가 예측될 때:

```

ames_test_small <- ames_test %>% slice(1:5)
predict(lm_form_fit, new_data = ames_test_small)
#> # A tibble: 5 × 1
#> .pred
#> <dbl>
#> 1 5.22
#> 2 5.21
#> 3 5.28
#> 4 5.27
#> 5 5.28

```

예측값의 행 순서는 항상 원본 데이터와 같습니다.

###### 참고 (Note)

일부 열 이름에 마침표(.)가 접두사로 붙는 이유는 무엇입니까? 일부 tidyverse 및 tidymodels 인수 및 반환 값에는 마침표가 포함됩니다. 이는 중복된 이름을 가진 데이터를 병합하는 것을 방지하기 위함입니다. `pred`라는 이름의 예측 변수가 포함된 데이터 세트도 있습니다!

이 세 가지 규칙을 사용하면 예측값을 원본 데이터와 병합하기가 더 쉬워집니다.

```

ames_test_small %>%
select(Sale_Price) %>%
bind_cols(predict(lm_form_fit, ames_test_small)) %>%

# 결과에 95% 예측 구간(prediction intervals)을 추가합니다.

bind_cols(predict(lm_form_fit, ames_test_small, type = "pred_int"))
#> # A tibble: 5 × 4
#> Sale_Price .pred .pred_lower .pred_upper
#> <dbl> <dbl> <dbl> <dbl>
#> 1 5.02 5.22 4.91 5.54
#> 2 5.39 5.21 4.90 5.53
#> 3 5.28 5.28 4.97 5.60
#> 4 5.28 5.27 4.96 5.59
#> 5 5.28 5.28 4.97 5.60

```

첫 번째 규칙에 대한 동기는 예측 함수에서 상이한(dissimilar) 데이터 유형을 생성하는 일부 R 패키지에서 비롯되었습니다. 예를 들어 ranger 패키지는 랜덤 포레스트 모델을 계산하는 데 훌륭한 도구입니다. 그러나 출력으로 데이터 프레임이나 벡터를 반환하는 대신, (예측값을 포함하여) 여러 값이 내장된 특수한 객체를 반환합니다. 이는 데이터 분석가가 스크립트에서 우회(work around)해야 할 또 하나의 단계일 뿐입니다. 또 다른 예로, 네이티브 glmnet 모델은 모델 세부 정보 및 데이터 특성에 따라 예측을 위해 최소 4가지 다른 출력 유형을 반환할 수 있습니다. 이는 [표 6-3](#predict-types)에 나와 있습니다.

| 예측 유형 (Type of prediction) | 반환 값 (Returns a:)            |
|--------------------------|---------------------------------|
| 숫자 (Numeric)                  | 숫자 행렬 (Numeric matrix)                  |
| 클래스 (Class)                    | 문자 행렬 (character matrix)                |
| 확률 (2개 클래스) (Probability (2 classes))  | 숫자 행렬 (두 번째 레벨만) (Numeric matrix (2nd level only)) |
| 확률 (3개 이상 클래스) (Probability (3+ classes)) | 3D 숫자 배열 (모든 레벨) (3D numeric array (all levels))   |

<span id="predict-types">표 6-3. glmnet 예측 유형에 대한 서로 다른 반환 값들</span>

또한 결과의 열 이름은 glmnet 모델 객체 내에서 `lambda`라고 불리는 벡터에 매핑되는 코딩된 값을 포함합니다. 이 훌륭한 통계 방법은 분석가가 우용하게 쓰기 위해 추가 코드를 요구하는 특별한 경우를 모두 마주쳐야 하기 때문에 실무에서 사용하는 데 실망감을 줄 수 있습니다.

두 번째 tidymodels 예측 규칙에 대한, 여러 유형의 예측에 대해 예측 가능한 열 이름은 [표 6-4](#predictable-column-names)에 나와 있습니다.

| 유형 값 (Type value) | 열 이름 (Column name(s))             |
|------------|----------------------------|
| `numeric`  | `.pred`                    |
| `class`    | `.pred_class`              |
| `prob`     | `.pred_{class levels}`     |
| `conf_int` | `.pred_lower, .pred_upper` |
| `pred_int` | `.pred_lower, .pred_upper` |

<span id="predictable-column-names">표 6-4. 예측 유형과 열 이름의 tidymodels 매핑</span>

출력의 행 수에 관한 세 번째 규칙은 매우 중요합니다. 예를 들어 새 데이터의 어떤 행에든 결측값이 포함되어 있으면 해당 행의 출력은 결측 결과(missing results)로 채워집니다(padded). parsnip에서 모델 인터페이스와 예측 유형을 표준화하는 것의 주요 이점은 다른 모델을 사용할 때 구문이 동일하다는 것입니다. Ames 데이터를 모델링하기 위해 의사결정나무(decision tree)를 사용했다고 가정해 보겠습니다. 모델 사양 외에는 코드 파이프라인에 유의미한 차이가 없습니다.

```

tree_model <-
decision_tree(min_n = 2) %>%
set_engine("rpart") %>%
set_mode("regression")

tree_fit <-
tree_model %>%
fit(Sale_Price ~ Longitude + Latitude, data = ames_train)

ames_test_small %>%
select(Sale_Price) %>%
bind_cols(predict(tree_fit, ames_test_small))
#> # A tibble: 5 × 2
#> Sale_Price .pred
#> <dbl> <dbl>
#> 1 5.02 5.15
#> 2 5.39 5.15
#> 3 5.28 5.32
#> 4 5.28 5.32
#> 5 5.28 5.32

```

이는 서로 다른 모델에 걸쳐 데이터 분석 프로세스와 구문을 동질화(homogenizing)할 때 얻을 수 있는 이점을 보여줍니다. 이를 통해 사용자는 R 패키지 간의 구문 차이에 집중해야 하는 대신 결과와 해석에 시간을 할애할 수 있습니다.

# parsnip 확장 패키지 (parsnip-Extension Packages)

parsnip 패키지 자체에는 여러 모델에 대한 인터페이스가 포함되어 있습니다. 그러나 패키지 설치 및 유지 관리를 용이하게 하기 위해 다른 모델 세트에 대한 parsnip 모델 정의를 가진 다른 tidymodels 패키지가 있습니다. discrim 패키지에는 판별 분석 방법(discriminant analysis methods) (선형 또는 2차 판별 분석)이라고 불리는 분류 기법 세트에 대한 모델 정의가 있습니다. 이런 방식으로 parsnip 설치에 필요한 패키지 종속성이 줄어듭니다. parsnip과 함께 사용할 수 있는 모든 모델 목록(CRAN에 있는 여러 패키지에 걸쳐)은 [tidymodels 웹사이트](https://oreil.ly/FB0BM)에서 찾을 수 있습니다.

# 모델 사양 생성 (Creating Model Specifications)

수많은 모델 사양(specifications)을 작성하거나 모델을 생성하기 위한 코드를 작성하는 방법을 기억하는 것은 지루해질 수 있습니다. parsnip 패키지에는 도움이 될 수 있는 [RStudio 애드인(addin)](https://oreil.ly/8qhDY)이 포함되어 있습니다. *Addins* 툴바 메뉴에서 이 애드인을 선택하거나 다음 코드를 실행하면:

```

parsnip_addin()

```

RStudio IDE의 Viewer 패널에 각 모델 모드에 대해 가능한 모델 목록이 포함된 창이 열립니다. 이들은 소스 코드 패널에 작성될 수 있습니다.

모델 목록에는 CRAN에 있는 parsnip 및 parsnip 인접(parsnip-adjacent) 패키지의 모델이 포함됩니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 표준 구문을 사용하여 R 패키지 전체에 걸쳐 모델에 대한 공통 인터페이스를 제공하는 parsnip 패키지를 소개했습니다. 인터페이스와 결과 객체는 예측 가능한 구조를 가지고 있습니다.

앞으로 우리가 사용할 Ames 데이터를 모델링하기 위한 코드는 다음과 같습니다.

```

library(tidymodels)
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

lm_model <- linear_reg() %>% set_engine("lm")

```

<sup>[1](ch06.xhtml#idm45881873685760-marker)</sup> parsnip은 분류 모델의 결과 열이 *팩터(factor)*로 인코딩되도록 제약합니다. 이진 숫자(binary numeric) 값을 사용하면 오류가 발생합니다.

<sup>[2](ch06.xhtml#idm45881873576064-marker)</sup> `fit()`과 `fit_xy()`의 차이점은 무엇입니까? `fit_xy()` 함수는 항상 데이터를 있는 그대로 기본 모델 함수에 전달합니다. 데이터를 전달하기 전에 더미/지시 변수를 생성하지 않습니다. 모델 사양과 함께 `fit()`을 사용하면, 이것은 거의 항상 질적 예측 변수에서 가변수(dummy variables)가 생성됨을 의미합니다. 기본 함수가 행렬(glmnet처럼)을 요구한다면 `fit()`이 가변수를 만들 것입니다. 그러나 기본 함수가 공식을 사용한다면 `fit()`은 단지 해당 함수에 공식을 전달할 뿐입니다. 우리는 공식을 사용하는 모델링 함수의 99%가 가변수를 생성한다고 추정합니다. 나머지 1%에는 순수 숫자 예측 변수만을 요구하지 않는 트리 기반(tree-based) 방법이 포함됩니다. tidymodels에서 공식을 사용하는 것에 대한 자세한 내용은 ["workflow()는 공식을 어떻게 사용하는가? (How Does a workflow() Use the Formula?)"](ch07.xhtml#workflow-encoding)를 참조하십시오.
```
