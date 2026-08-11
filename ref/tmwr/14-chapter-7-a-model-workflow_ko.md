# 7장. 모델 워크플로 (A Model Workflow)

[6장](ch06.xhtml#models)에서는 모델을 정의하고 피팅하는 데 사용할 수 있는 parsnip 패키지에 대해 논의했습니다. 이 장에서는 *모델 워크플로(model workflow)*라는 새로운 개념을 소개합니다. 이 개념(및 해당 tidymodels `workflow()` 객체)의 목적은 ([1장](ch01.xhtml#software-modeling)에서 논의한) 모델링 프로세스의 주요 부분들을 캡슐화(encapsulate)하는 것입니다. 워크플로는 두 가지 면에서 중요합니다. 첫째, 데이터 분석의 추정 구성 요소에 대한 단일 진입점(single point of entry)이므로 워크플로 개념을 사용하면 좋은 방법론이 장려됩니다. 둘째, 사용자가 프로젝트를 더 잘 구성할 수 있게 해 줍니다. 이 두 가지 점은 다음 섹션에서 논의됩니다.

# 모델은 어디서 시작되고 어디서 끝나는가? (Where Does the Model Begin and End?)

지금까지 우리가 "모델"이라는 용어를 사용할 때, 그것은 몇 가지 예측 변수와 하나 이상의 결과 변수를 연관시키는 구조 방정식(structural equation)을 의미했습니다. 선형 회귀(linear regression)를 다시 예로 들어 생각해 봅시다. 결과 변수 데이터는 $y_i$로 표시되며 훈련 세트에는 $`i = 1...n`$개의 샘플이 있습니다. 모델에 사용되는 $`p`$개의 예측 변수 $`x_{i1},...,x_{ip}`$가 있다고 가정해 봅시다. 선형 회귀는 다음과 같은 모델 방정식을 생성합니다.

```math
{\hat{y}}_{i} = {\hat{\beta}}_{0} + {\hat{\beta}}_{1}x_{i1} + ... + {\hat{\beta}}_{p}x_{ip}
```

이것은 선형 모델이지만 파라미터에서만 선형입니다. 예측 변수는 비선형 항(nonlinear terms) ($`\log\left( x_{i} \right)`$)이 될 수 있습니다.

###### 경고 (Warning)

모델링 프로세스에 대해 생각하는 전통적인 방식은 모델 피팅(model fit)만 포함한다고 생각하는 것입니다.

일부 간단한 데이터 세트의 경우 모델 자체를 피팅하는 것이 전체 프로세스일 수 있습니다. 그러나 모델을 피팅하기 전에 다음과 같은 다양한 선택 사항과 추가 단계가 종종 발생합니다.

- 예제 모델에 $`p`$개의 예측 변수가 있지만 종종 $`p`$개 이상의 후보 예측 변수로 시작합니다. 탐색적 데이터 분석이나 도메인 지식을 사용하여 일부 예측 변수를 분석에서 제외할 수 있습니다. 다른 경우에는 모델을 위한 최소 예측 변수 집합에 대해 데이터 기반의 선택을 하기 위해 특성 선택 알고리즘(feature selection algorithm)을 사용할 수 있습니다.

- 중요한 예측 변수의 값이 결측(missing)된 경우가 있습니다. 데이터 세트에서 이 샘플을 제거하는 대신 데이터의 다른 값을 사용하여 결측값을 대체(impute)할 수 있습니다. 예를 들어 $`x_{1}`$이 결측되었지만 예측 변수 $`x_{2}`$ 및 $`x_{3}`$과 상관관계가 있는 경우, 대체 방법은 $`x_{2}`$ 및 $`x_{3}`$의 값으로부터 결측된 $`x_{1}`$ 관측치를 추정할 수 있습니다.

- 예측 변수의 척도(scale)를 변환하는 것이 유익할 수 있습니다. 새로운 척도가 어떠해야 하는지에 대한 사전 지식이 없다면 통계적 변환 기법, 기존 데이터 및 일부 최적화 기준을 사용하여 적절한 척도를 추정할 수 있습니다. PCA와 같은 다른 변환은 예측 변수 그룹을 취하여 예측 변수로 사용될 새로운 기능(features)으로 변환합니다.

이러한 예는 모델 피팅 전에 발생하는 단계와 관련이 있지만 모델이 생성된 후에 발생하는 작업(operations)도 있을 수 있습니다. 결과가 이진(binary)(`event` 및 `non-event`)인 분류 모델이 생성된 경우 50%의 확률 차단점(probability cutoff)을 사용하여 이산 클래스 예측(discrete class prediction), 즉 엄격한 예측(hard prediction)을 생성하는 것이 관례입니다. 예를 들어 분류 모델이 사건(event)의 확률을 62%로 추정할 수 있습니다. 일반적인 기본값을 사용하면 엄격한 예측은 `event`가 됩니다. 그러나 모델이 오탐(false positive) 결과(즉, 실제 비사건(nonevents)이 사건(events)으로 분류되는 곳)를 줄이는 데 더 집중해야 할 수도 있습니다. 이를 수행하는 한 가지 방법은 컷오프를 50%에서 더 큰 값으로 높이는 것입니다. 이는 새로운 샘플을 사건(event)이라고 부르는 데 필요한 증거의 수준을 높입니다. 이는 참 긍정률(true positive rate)(나쁜 것)을 감소시키지만 오탐(false positives)을 줄이는 데는 더 극적인 영향을 미칠 수 있습니다. 컷오프 값의 선택은 데이터를 사용하여 최적화되어야 합니다. 이는 모델 피팅 단계에 포함되어 있지 않더라도 모델이 얼마나 잘 작동하는지에 중대한 영향을 미치는 후처리(postprocessing) 단계의 한 예입니다.

파라미터를 추정하는 데 사용되는 특정 모델을 피팅하는 데만 초점을 맞추는 대신 더 광범위한 *모델링 프로세스*에 초점을 맞추는 것이 중요합니다. 이 더 광범위한 프로세스에는 모든 전처리 단계, 모델 피팅 자체 및 잠재적인 후처리 활동이 포함됩니다. 이 책에서는 이러한 보다 포괄적인 개념을 *모델 워크플로(model workflow)*라고 지칭하고 최종 모델 방정식을 생성하기 위해 그 모든 구성 요소를 처리하는 방법을 강조할 것입니다.

###### 참고 (Note)

Python이나 Spark와 같은 소프트웨어에서는 이와 유사한 단계의 모음을 *파이프라인(pipelines)*이라고 합니다. tidymodels에서 *파이프라인*이라는 용어는 파이프 연산자(magrittr의 `%>%` 또는 더 새로운 기본 연산자 `|>`)로 서로 연결된 일련의 작업을 이미 내포하고 있습니다. 이러한 상황에서 모호한 용어를 사용하는 대신 모델링과 관련된 계산 작업의 시퀀스를 *워크플로(workflows)*라고 부릅니다.

데이터 분석의 분석적 구성 요소를 하나로 묶는 것은 또 다른 이유로 중요합니다. 향후 장에서는 구조적 매개변수를 최적화(즉, 모델 튜닝)하는 방법뿐만 아니라 성능을 정확하게 측정하는 방법을 보여줍니다. 훈련 세트에서 모델 성능을 올바르게 정량화(quantify)하기 위해 [10장](ch10.xhtml#resampling)에서는 리샘플링 방법 사용을 권장합니다. 이 작업을 제대로 수행하려면 분석의 데이터 기반(data-driven) 부분을 검증에서 제외해서는 안 됩니다. 이를 위해 워크플로에는 모든 중요한 추정 단계가 포함되어야 합니다.

이를 설명하기 위해 주성분 분석(PCA) 신호 추출을 고려해 보십시오. 이에 대해서는 [8장](ch08.xhtml#recipes) 및 [16장](ch16.xhtml#dimensionality)에서 자세히 설명하겠습니다. PCA는 상관관계가 있는 예측 변수를 원본 세트의 대부분의 정보를 포착하고 상관관계가 없는 새로운 인공적인 특성으로 대체하는 방법입니다. 새로운 특성은 예측 변수로 사용될 수 있고 모델 파라미터를 추정하는 데 최소제곱 회귀를 사용할 수 있습니다.

모델 워크플로에 대해 생각하는 두 가지 방식이 있습니다. [그림 7-1](#figure-7-1.-incorrect-mental-model-of-where-model-estimation-occurs-in-the-data-analysis-process.)은 _잘못된_ 방법을 보여줍니다. 즉, PCA 전처리 단계를 _모델링 워크플로의 일부가 아닌_ 것으로 간주하는 것입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0701.png" alt="tmwr 0701" />
<h6 id="figure-7-1.-incorrect-mental-model-of-where-model-estimation-occurs-in-the-data-analysis-process.">그림 7-1. 데이터 분석 프로세스에서 모델 추정이 일어나는 위치에 대한 잘못된 멘탈 모델.</h6>
</figure>

여기서 오류는 PCA가 주성분을 생성하기 위해 상당한 계산을 수행하지만, 그 작업과 관련된 불확실성은 없다고 가정한다는 것입니다. PCA 구성 요소는 _알려진(known)_ 것으로 취급되며 워크플로에 포함되지 않으면 PCA의 효과를 적절하게 측정할 수 없습니다.

[그림 7-2](#figure-7-2.-correct-mental-model-of-where-model-estimation-occurs-in-the-data-analysis-process.)는 _적절한_ 접근 방식을 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0702.png" alt="tmwr 0702" />
<h6 id="figure-7-2.-correct-mental-model-of-where-model-estimation-occurs-in-the-data-analysis-process.">그림 7-2. 데이터 분석 프로세스에서 모델 추정이 일어나는 위치에 대한 올바른 멘탈 모델.</h6>
</figure>

이런 식으로 PCA 전처리는 모델링 프로세스의 일부로 간주됩니다.

# 워크플로의 기본 (Workflow Basics)

workflows 패키지를 통해 사용자는 모델링과 전처리 객체를 결합할 수 있습니다. Ames 데이터와 단순한 선형 모델로 다시 시작해 보겠습니다.

```
library(tidymodels)  # workflows 패키지를 포함합니다
tidymodels_prefer()

lm_model <-
  linear_reg() %>%
  set_engine("lm")
```

워크플로는 항상 parsnip 모델 객체를 필요로 합니다.

```
lm_wflow <-
  workflow() %>%
  add_model(lm_model)

lm_wflow
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: None
#> Model: linear_reg()
#>
#> ── Model ────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
```

이 워크플로가 데이터를 전처리하는 방법을 아직 지정하지 않았다는 것에 주목하세요. `Preprocessor: None`.

모델이 매우 간단한 경우 표준 R 공식을 전처리기로 사용할 수 있습니다.

```
lm_wflow <-
  lm_wflow %>%
  add_formula(Sale_Price ~ Longitude + Latitude)

lm_wflow
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Formula
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Sale_Price ~ Longitude + Latitude
#>
#> ── Model ────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
```

워크플로에는 모델을 생성하는 데 사용할 수 있는 `fit()` 메소드가 있습니다. [6장](ch06.xhtml#models) 끝부분의 요약에서 생성된 객체를 사용하면 다음을 확인할 수 있습니다.

```
lm_fit <- fit(lm_wflow, ames_train)
lm_fit
#> ══ Workflow [trained] ═══════════════════════════════
#> Preprocessor: Formula
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Sale_Price ~ Longitude + Latitude
#>
#> ── Model  ───────────────────────────────────────────
#>
#> Call:
#> stats::lm(formula = ..y ~ ., data = data)
#>
#> Coefficients:
#> (Intercept)    Longitude     Latitude
#>     -302.97        -2.07         2.71
```

피팅된 워크플로에 대해 예측(`predict()`)을 수행할 수도 있습니다.

```
predict(lm_fit, ames_test %>% slice(1:3))
#> # A tibble: 3 × 1
#>   .pred
#>   <dbl>
#> 1  5.22
#> 2  5.21
#> 3  5.28
```

`predict()` 메서드는 [6장](ch06.xhtml#models)에서 parsnip 패키지에 대해 설명한 것과 동일한 규칙 및 명명 규칙을 따릅니다.

모델과 전처리기 모두 제거하거나 업데이트할 수 있습니다.

```
lm_fit %>% update_formula(Sale_Price ~ Longitude)
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Formula
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Sale_Price ~ Longitude
#>
#> ── Model ────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
```

이 새 객체에서 새 공식이 이전 모델 피팅과 일치하지 않기 때문에 이전에 피팅된 모델이 제거되었음이 출력에 표시됩니다.

# 원시 변수를 workflow()에 추가하기 (Adding Raw Variables to the workflow())

모델에 데이터를 전달하는 또 다른 인터페이스인 `add_variables()` 함수가 있습니다. 이 함수는 변수 선택을 위해 dplyr과 유사한 구문을 사용합니다. 이 함수에는 `outcomes`와 `predictors`라는 두 가지 기본 인수가 있습니다. 이는 `c()`를 사용하여 여러 선택기를 캡처하는 tidyverse 패키지의 tidyselect 백엔드와 유사한 선택 방식을 사용합니다.

```
lm_wflow <-
  lm_wflow %>%
  remove_formula() %>%
  add_variables(outcome = Sale_Price, predictors = c(Longitude, Latitude))
lm_wflow
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Variables
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Outcomes: Sale_Price
#> Predictors: c(Longitude, Latitude)
#>
#> ── Model ────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
```

예측 변수는 다음과 같이 보다 일반적인 선택기를 사용하여 지정될 수도 있습니다.

```
predictors = c(ends_with("tude"))
```

한 가지 좋은 점은 예측 변수 인수에 잘못 지정된 모든 결과 변수 열이 조용히 제거된다는 것입니다. 이는 다음을 사용하는 것을 용이하게 합니다.

```
predictors = everything()
```

모델이 피팅될 때 이 사양은 이러한 데이터를 변경 없이 데이터 프레임으로 조합하여 기본 함수에 전달합니다.

```
fit(lm_wflow, ames_train)
#> ══ Workflow [trained] ═══════════════════════════════
#> Preprocessor: Variables
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Outcomes: Sale_Price
#> Predictors: c(Longitude, Latitude)
#>
#> ── Model ────────────────────────────────────────────
#>
#> Call:
#> stats::lm(formula = ..y ~ ., data = data)
#>
#> Coefficients:
#> (Intercept)    Longitude     Latitude
#>     -302.97        -2.07         2.71
```

기본 모델링 방법이 데이터로 일반적으로 수행하는 작업을 수행하도록 하려면 `add_variables()`가 유용한 인터페이스가 될 수 있습니다. 이 장의 다음 섹션에서 볼 수 있듯이 이는 더 복잡한 모델링 사양도 용이하게 합니다. 그러나 다음 섹션에서 언급하듯이 `glmnet` 및 `xgboost`와 같은 모델은 사용자가 팩터 예측 변수로부터 지시 변수(indicator variables)를 만들 것으로 기대합니다. 이러한 경우 레시피 또는 공식 인터페이스가 일반적으로 더 나은 선택이 될 것입니다.

다음 장에서는 워크플로에 추가할 수 있는 더 강력한 전처리기(*레시피(recipe)*라고 함)를 살펴보겠습니다.

# workflow()는 공식을 어떻게 사용하는가? (How Does a workflow() Use the Formula?)

[3장](ch03.xhtml#base-r)에서 R의 공식 메소드에는 여러 가지 목적이 있다는 것을 상기해 보세요(자세한 내용은 [8장](ch08.xhtml#recipes)에서 논의할 것입니다). 그 중 하나는 원본 데이터를 분석에 적합한 형식으로 적절하게 인코딩하는 것입니다. 여기에는 인라인(inline) 변환(`log(x)`) 실행, 가변수 열 생성, 상호 작용(interactions) 또는 기타 열 확장 생성 등이 포함될 수 있습니다. 그러나 많은 통계 방법에는 다양한 유형의 인코딩이 필요합니다.

- 트리 기반 모델을 위한 대부분의 패키지는 공식 인터페이스를 사용하지만 범주형 예측 변수를 가변수(dummy variables)로 인코딩하지 _않습니다_.

- 패키지는 모델 함수에게 분석에서 예측 변수를 처리하는 방법을 알려주는 특별한 인라인 함수를 사용할 수 있습니다. 예를 들어 생존 분석(survival analysis) 모델에서 `strata(site)`와 같은 공식 항은 `site` 열이 계층화 변수(stratification variable)임을 나타냅니다. 즉, 일반 예측 변수로 취급되어서는 안 되며 모델에 대응하는 위치 매개변수 추정치가 없음을 의미합니다.

- 일부 R 패키지는 기본 R 함수가 구문 분석하거나 실행할 수 없는 방식으로 공식을 확장했습니다. 다수준 모델(혼합 모델(mixed models) 또는 계층적 베이지안(hierarchical Bayesian) 모델)에서 `(week | subject)`와 같은 모델 항은 `week` 열이 `subject` 열의 각 값에 대해 다른 기울기 매개변수 추정치를 갖는 임의 효과(random effect)임을 나타냅니다.

워크플로는 범용 인터페이스입니다. `add_formula()`를 사용할 때 워크플로는 데이터를 어떻게 전처리해야 할까요? 전처리는 모델에 종속되므로 workflows 패키지는 가능할 때마다 기본 모델이 수행하는 작업을 에뮬레이트하려고 시도합니다. 불가능한 경우 공식 처리는 공식에 사용된 열에 아무것도 수행하지 않아야 합니다. 이에 대해 좀 더 자세히 살펴보겠습니다.

## 트리 기반 모델 (Tree-Based Models)

우리가 데이터에 트리를 피팅할 때 parsnip 패키지는 모델링 함수가 무엇을 할 것인지 이해합니다. 예를 들어 ranger 또는 randomForest 패키지를 사용하여 랜덤 포레스트 모델이 피팅되는 경우, 워크플로는 팩터(factors)인 예측 변수 열을 그대로 둬야 한다는 것을 알고 있습니다.

반례로, xgboost 패키지로 생성된 부스팅 트리(boosted tree)의 경우(`xgboost::xgb.train()`은 수행하지 않으므로) 사용자가 팩터 예측 변수로부터 가변수를 생성해야 합니다. 이 요구 사항은 모델 사양 객체에 내장되어 있으며 xgboost를 사용하는 워크플로가 이 엔진을 위한 지시 변수 열(indicator columns)을 생성합니다. 또한 부스팅 트리를 위한 또 다른 엔진인 C5.0은 가변수를 요구하지 않으므로 워크플로에서 생성되지 않는다는 점에 유의하세요.

이러한 결정은 각 모델 및 엔진 조합에 대해 내려집니다.

## 특수 공식 및 인라인 함수 (Special Formulas and Inline Functions)

다수의 다수준(multilevel) 모델은 lme4 패키지에서 고안된 공식 사양으로 표준화되었습니다. 예를 들어 주제(subjects)에 대한 임의 효과가 있는 회귀 모델을 피팅하려면 다음 공식을 사용할 것입니다.

```
library(lme4)
lmer(distance ~ Sex + (age | Subject), data = Orthodont)
```

이 결과로 각 주제(subject)는 `age`에 대해 추정된 절편과 기울기 매개변수를 갖게 됩니다.

문제는 표준 R 메서드가 이 공식을 제대로 처리할 수 없다는 것입니다.

```
model.matrix(distance ~ Sex + (age | Subject), data = Orthodont)
#> Warning in Ops.ordered(age, Subject): '|' is not meaningful for ordered factors
#>      (Intercept) SexFemale age | SubjectTRUE
#> attr(,"assign")
#> [1] 0 1 2
#> attr(,"contrasts")
#> attr(,"contrasts")$Sex
#> [1] "contr.treatment"
#>
#> attr(,"contrasts")$`age | Subject`
#> [1] "contr.treatment"
```

그 결과는 행이 0개인 데이터 프레임입니다.

###### 경고 (Warning)

문제는 이 특별한 공식을 표준 `model.matrix()` 방식이 아닌 기본 패키지 코드에서 처리해야 한다는 것입니다.

이 공식을 `model.matrix()`와 함께 사용할 수 있다고 하더라도 공식이 모델의 통계적 속성도 지정하기 때문에 여전히 문제가 될 수 있습니다.

workflows 패키지의 해결책은 `add_model()`에 전달할 수 있는 선택적인(optional) 보충 모델 공식입니다. `add_variables()` 사양은 기본 열(bare column) 이름만 제공하고 그런 다음 모델에 주어지는 실제 공식은 `add_model()` 내에서 설정됩니다.

```
library(multilevelmod)

multilevel_spec <- linear_reg() %>% set_engine("lmer")

multilevel_workflow <-
  workflow() %>%
  # 데이터를 있는 그대로 전달합니다.
  add_variables(outcome = distance, predictors = c(Sex, age, Subject)) %>%
  add_model(multilevel_spec,
            # 이 공식이 모델에 주어집니다
            formula = distance ~ Sex + (age | Subject))

multilevel_fit <- fit(multilevel_workflow, data = Orthodont)
multilevel_fit
#> ══ Workflow [trained] ═══════════════════════════════
#> Preprocessor: Variables
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Outcomes: distance
#> Predictors: c(Sex, age, Subject)
#>
#> ── Model ────────────────────────────────────────────
#> Linear mixed model fit by REML ['lmerMod']
#> Formula: distance ~ Sex + (age | Subject)
#>    Data: data
#> REML criterion at convergence: 471.2
#> Random effects:
#>  Groups   Name        Std.Dev. Corr
#>  Subject  (Intercept) 7.391
#>           age         0.694    -0.97
#>  Residual             1.310
#> Number of obs: 108, groups:  Subject, 27
#> Fixed Effects:
#> (Intercept)    SexFemale
#>       24.52        -2.15
```

생존 분석(survival analysis)을 위해 이전에 언급된 survival 패키지의 `strata()` 함수를 사용할 수도 있습니다.

```
library(censored)

parametric_spec <- survival_reg()

parametric_workflow <-
  workflow() %>%
  add_variables(outcome = c(fustat, futime), predictors = c(age, rx)) %>%
  add_model(parametric_spec,
            formula = Surv(futime, fustat) ~ age + strata(rx))

parametric_fit <- fit(parametric_workflow, data = ovarian)
parametric_fit
#> ══ Workflow [trained] ═══════════════════════════════
#> Preprocessor: Variables
#> Model: survival_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Outcomes: c(fustat, futime)
#> Predictors: c(age, rx)
#>
#> ── Model ────────────────────────────────────────────
#> Call:
#> survival::survreg(formula = Surv(futime, fustat) ~ age + strata(rx),
#>     data = data, model = TRUE)
#>
#> Coefficients:
#> (Intercept)         age
#>     12.8734     -0.1034
#>
#> Scale:
#>   rx=1   rx=2
#> 0.7696 0.4704
#>
#> Loglik(model)= -89.4   Loglik(intercept only)= -97.1
#>  Chisq= 15.36 on 1 degrees of freedom, p= 9e-05
#> n= 26
```

이 두 번의 호출에서 모두 모델별 공식이 어떻게 사용되었는지 확인하십시오.

# 여러 워크플로를 한 번에 생성하기 (Creating Multiple Workflows at Once)

어떤 상황에서는 데이터에 적절한 모델을 찾기 위해 수많은 시도가 필요합니다. 예를 들어:

- 예측 모델의 경우 다양한 모델 유형을 평가하는 것이 좋습니다. 이를 위해서는 사용자가 여러 모델 사양을 만들어야 합니다.

- 모델의 순차적 테스트는 일반적으로 확장된 예측 변수 집합에서 시작합니다. 이 "전체 모델(full model)"을 각 예측 변수를 차례로 제거하는 동일한 모델의 시퀀스와 비교합니다. 기본적인 가설 검정 방법이나 경험적 검증을 사용하여 각 예측 변수의 효과를 격리(isolated)하고 평가할 수 있습니다.

이러한 상황뿐만 아니라 다른 상황에서도 여러 다른 전처리기 및/또는 모델 사양 세트에서 많은 워크플로를 생성하는 것은 지루하거나 번거로울(onerous) 수 있습니다. 이 문제를 해결하기 위해 workflowset 패키지는 워크플로 구성 요소들의 조합을 생성합니다. 전처리기 목록(공식, dplyr 선택기 또는 [8장](ch08.xhtml#recipes)에서 설명한 특징 공학(feature engineering) 레시피 객체)을 모델 사양 목록과 결합하여 워크플로 세트(workflow set)를 결과로 만들 수 있습니다.

예를 들어, Ames 데이터에서 주택 위치가 나타나는 다양한 방식에 초점을 맞추고 싶다고 가정해 봅시다. 이러한 예측 변수를 포착하는 일련의 공식을 생성할 수 있습니다.

```
location <- list(
  longitude = Sale_Price ~ Longitude,
  latitude = Sale_Price ~ Latitude,
  coords = Sale_Price ~ Longitude + Latitude,
  neighborhood = Sale_Price ~ Neighborhood
)
```

이러한 표현(representations)은 `workflow_set()` 함수를 사용하여 하나 이상의 모델과 교차(crossed)될 수 있습니다. 이를 시연하기 위해 이전의 선형 모델 사양을 사용해 보겠습니다.

```
library(workflowsets)
location_models <- workflow_set(preproc = location, models = list(lm = lm_model))
location_models
#> # A workflow set/tibble: 4 × 4
#>   wflow_id        info             option    result
#>   <chr>           <list>           <list>    <list>
#> 1 longitude_lm    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 latitude_lm     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 coords_lm       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 neighborhood_lm <tibble [1 × 4]> <opts[0]> <list [0]>
location_models$info[[1]]
#> # A tibble: 1 × 4
#>   workflow   preproc model      comment
#>   <list>     <chr>   <chr>      <chr>
#> 1 <workflow> formula linear_reg ""
extract_workflow(location_models, id = "coords_lm")
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Formula
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Sale_Price ~ Longitude + Latitude
#>
#> ── Model ────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
```

워크플로 세트는 주로 리샘플링(resampling)과 함께 작동하도록 설계되었으며 이에 대해서는 [10장](ch10.xhtml#resampling)에서 설명합니다. `option` 및 `result` 열은 리샘플링의 결과인 특정 유형의 객체들로 채워져야 합니다. 이에 대한 자세한 내용은 [11장](ch11.xhtml#compare) 및 [15장](ch15.xhtml#workflow-sets)에서 설명합니다.

그동안 각 공식에 대한 모델 피팅을 생성하고 이를 `fit`이라는 새 열에 저장해 보겠습니다. 기본적인 dplyr 및 purrr 작업을 사용하겠습니다.

```
location_models <-
   location_models %>%
   mutate(fit = map(info, ~ fit(.x$workflow[[1]], ames_train)))
location_models
#> # A workflow set/tibble: 4 × 5
#>   wflow_id        info             option    result     fit
#>   <chr>           <list>           <list>    <list>     <list>
#> 1 longitude_lm    <tibble [1 × 4]> <opts[0]> <list [0]> <workflow>
#> 2 latitude_lm     <tibble [1 × 4]> <opts[0]> <list [0]> <workflow>
#> 3 coords_lm       <tibble [1 × 4]> <opts[0]> <list [0]> <workflow>
#> 4 neighborhood_lm <tibble [1 × 4]> <opts[0]> <list [0]> <workflow>
location_models$fit[[1]]
#> ══ Workflow [trained] ═══════════════════════════════
#> Preprocessor: Formula
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> Sale_Price ~ Longitude
#>
#> ── Model ────────────────────────────────────────────
#>
#> Call:
#> stats::lm(formula = ..y ~ ., data = data)
#>
#> Coefficients:
#> (Intercept)    Longitude
#>     -184.40        -2.02
```

여기서는 모델들을 매핑하기 위해 purrr 함수를 사용하지만 [11장](ch11.xhtml#compare)에서는 워크플로 세트를 피팅하는 더 쉽고 더 나은 접근 방식이 소개될 것입니다.

###### 참고 (Note)

일반적으로 워크플로 세트에는 훨씬 더 많은 기능이 있습니다! 여기에서는 기본 사항을 다루었지만 워크플로 세트의 뉘앙스와 이점은 [15장](ch15.xhtml#workflow-sets)에서 비로소 설명될 것입니다.

# 테스트 세트 평가하기 (Evaluating the Test Set)

모델 개발을 마무리하고 최종 모델을 결정했다고 가정해 보겠습니다. 전체 훈련 세트에 모델을 *피팅(fit)*하고 테스트 세트로 모델을 *평가(evaluate)*하는 `last_fit()`이라는 편리한 함수가 있습니다.

`lm_wflow`를 예로 들면, 모델과 초기 훈련/테스트 분할을 함수에 전달할 수 있습니다.

```
final_lm_res <- last_fit(lm_wflow, ames_split)
final_lm_res
#> # Resampling results
#> # Manual resampling
#> # A tibble: 1 × 6
#>   splits             id               .metrics .notes   .predictions .workflow
#>   <list>             <chr>            <list>   <list>   <list>       <list>
#> 1 <split [2342/588]> train/test split <tibble> <tibble> <tibble>     <workflow>
```

###### 참고 (Note)

`last_fit()`은 데이터 프레임이 아닌 데이터 분할(data split)을 입력으로 취한다는 점에 유의하세요. 이 함수는 분할을 사용하여 최종 피팅 및 평가를 위한 훈련 및 테스트 세트를 생성합니다.

`.workflow` 열에는 피팅된 워크플로가 포함되어 있으며 다음을 사용하여 결과에서 추출(pulled out)할 수 있습니다.

```
fitted_lm_wflow <- extract_workflow(final_lm_res)
```

마찬가지로 `collect_metrics()` 및 `collect_predictions()`는 각각 성능 지표와 예측값에 대한 접근(access)을 제공합니다.

```
collect_metrics(final_lm_res)
collect_predictions(final_lm_res) %>% slice(1:5)
```

실제 동작하는 `last_fit()`과 이를 다시 사용하는 방법에 대해서는 [16장](ch16.xhtml#dimensionality)에서 자세히 설명하겠습니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 예측 변수를 결과에 연결하는 알고리즘의 파라미터를 추정하는 것 이상을 포괄하는 모델링 프로세스에 대해 배웠습니다. 이 프로세스에는 전처리 단계와 모델이 피팅된 후 수행되는 작업(operations)도 포함됩니다. 우리는 모델링 프로세스의 중요한 구성 요소를 캡처할 수 있는 *모델 워크플로(model workflow)*라는 개념을 도입했습니다. _워크플로 세트(workflow set)_ 내에 여러 워크플로를 생성할 수도 있습니다. `last_fit()` 함수는 훈련 세트에 최종 모델을 피팅하고 테스트 세트로 평가하는 데 편리합니다.

Ames 데이터의 경우, 앞으로 다시 사용될 관련 코드는 다음과 같습니다.

```
library(tidymodels)
data(ames)

ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <-
  workflow() %>%
  add_model(lm_model) %>%
  add_variables(outcome = Sale_Price, predictors = c(Longitude, Latitude))

lm_fit <- fit(lm_wflow, ames_train)
```
