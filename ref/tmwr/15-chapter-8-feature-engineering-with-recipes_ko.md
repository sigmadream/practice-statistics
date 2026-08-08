# 8장. 레시피를 활용한 특징 공학 (Feature Engineering with Recipes)

특징 공학(Feature engineering)은 모델이 효과적으로 사용할 수 있도록 예측 변수 값을 다시 형식화(reformatting)하는 작업을 수반합니다. 여기에는 데이터의 중요한 특성을 가장 잘 표현하기 위한 데이터의 변환(transformations) 및 인코딩이 포함됩니다. 데이터 세트에 비율로 모델에 더 효과적으로 표현될 수 있는 두 개의 예측 변수가 있다고 상상해 보십시오. 기존의 두 변수의 비율로부터 새로운 예측 변수를 만드는 것은 특징 공학의 간단한 예입니다.

에임스(Ames) 주택의 위치를 좀 더 관련된 예로 들어보겠습니다. 이 공간 정보를 모델에 노출하는 방법에는 동네(정성적 척도), 경도/위도, 가장 가까운 학교까지의 거리 등 다양한 방법이 있습니다. 모델링에서 이 데이터를 인코딩하는 방법을 선택할 때 결과 변수와 가장 관련이 있다고 생각되는 옵션을 선택할 수 있습니다. 예를 들어 숫자형(예: 거리) 대 범주형(예: 동네)과 같은 데이터의 원래 형식도 특징 공학을 선택하는 주요 요인입니다.

모델링을 위한 더 나은 특성(features)을 구축하기 위한 전처리(preprocessing)의 다른 예는 다음과 같습니다:

- 특성 추출(feature extraction)이나 일부 예측 변수의 제거를 통해 예측 변수 간의 상관관계(Correlation)를 줄일 수 있습니다.

- 일부 예측 변수에 결측값(missing values)이 있는 경우 하위 모델(sub-model)을 사용하여 대치(impute)할 수 있습니다.

- 분산(variance) 유형 측정을 사용하는 모델은 변환을 추정하여 일부 치우친(skewed) 예측 변수의 분포가 대칭이 되도록 강제(coercing)함으로써 이점을 얻을 수 있습니다.

특징 공학과 데이터 전처리에는 모델에서 요구할 수 있는 재포맷팅(reformatting)도 포함될 수 있습니다. 일부 모델은 기하학적 거리 측정 항목(geometric distance metrics)을 사용하므로, 숫자 예측 변수는 모두 동일한 단위가 되도록 중심화(centered) 및 척도화(scaled)되어야 합니다. 그렇지 않으면 거리 값은 각 열의 척도에 의해 편향(biased)될 것입니다.

###### 참고 (Note)

모델마다 전처리 요구 사항이 다르며 트리 기반(tree-based) 모델과 같은 일부 모델은 전처리가 거의 필요하지 않습니다. [부록](app01.xhtml#pre-proc-table)에는 다양한 모델에 권장되는 전처리 기법의 작은 표가 포함되어 있습니다.

이 장에서는 서로 다른 특징 공학 및 전처리 작업을 단일 객체로 결합한 다음 이러한 변환을 서로 다른 데이터 세트에 적용하는 데 사용할 수 있는 [recipes 패키지](https://oreil.ly/b34bX)를 소개합니다. recipes 패키지는 모델을 위한 parsnip과 마찬가지로 핵심 tidymodels 패키지 중 하나입니다.

이 장에서는 [7장](ch07.xhtml#workflows)의 끝부분에 요약된 대로 Ames 주택 데이터와 지금까지 이 책에서 만들어진 R 객체를 사용합니다.

# Ames 주택 데이터를 위한 간단한 recipe() (A Simple recipe() for the Ames Housing Data)

이 섹션에서는 Ames 주택 데이터에서 사용할 수 있는 예측 변수의 작은 하위 집합에 중점을 둘 것입니다:

- 동네 (`Neighborhood`) (정성적(qualitative), 훈련 세트에 29개 동네 있음)

- 지상 총 생활 면적 (연속형, `Gr_Liv_Area`로 명명됨)

- 건축 연도 (`Year_Built`)

- 건물 유형 (`Bldg_Type`, 값은 `OneFam`($`n = 1,936`$), `TwoFmCon`($`n = 50`$), `Duplex`($`n = 88`$), `Twnhs`($`n = 77`$) 및 `TwnhsE`($`n = 191`$))

초기의 일반 선형 회귀 모델이 이 데이터에 피팅되었다고 가정해 보겠습니다. [4장](ch04.xhtml#ames)에서 판매 가격이 사전 로그 변환(prelogged)되었음을 상기하면 `lm()`에 대한 표준 호출은 다음과 같을 수 있습니다:

```
lm(Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Year_Built + Bldg_Type, data = ames)
```

이 함수가 실행되면 데이터는 데이터 프레임에서 숫자 *설계 행렬(design matrix)* (또는 *모델 행렬(model matrix)*이라고도 함)로 변환된 다음 최소제곱법을 사용하여 파라미터를 추정합니다. [3장](ch03.xhtml#base-r)에서 우리는 R 모델 공식의 여러 목적을 나열했습니다. 지금은 데이터 조작(data manipulation) 측면에만 집중해 보겠습니다. 이전 공식이 수행하는 작업은 일련의 단계로 분해(decomposed)될 수 있습니다:

1.  판매 가격은 결과 변수로 정의되는 반면 동네, 총 생활 면적, 건축 연도 및 건물 유형 변수는 모두 예측 변수로 정의됩니다.

2.  총 생활 면적 예측 변수에 로그 변환이 적용됩니다.

3.  동네 및 건물 유형 열은 숫자가 아닌 형식에서 숫자 형식으로 변환됩니다(최소제곱법은 숫자 예측 변수를 요구하기 때문입니다).

[3장](ch03.xhtml#base-r)에서 언급했듯이 공식 메소드는 `predict()` 함수에 전달되는 새 데이터를 포함하여 모든 데이터에 이러한 데이터 조작을 적용합니다.

레시피(recipe)도 데이터 처리를 위한 일련의 단계를 정의하는 객체입니다. 모델링 함수 내부의 공식 메소드와 달리, 레시피는 `step_*()` 함수를 통해 즉시 실행하지 않고 단계를 정의합니다. 그것은 무엇을 해야 하는지에 대한 사양일 뿐입니다. 다음은 [5장](ch05.xhtml#splitting) 끝부분의 코드 요약을 기반으로 하는 이전 공식과 동등한(equivalent) 레시피입니다.

```
library(tidymodels) # recipes 패키지를 포함합니다
tidymodels_prefer()

simple_ames <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_dummy(all_nominal_predictors())
simple_ames
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor          4
#>
#> Operations:
#>
#> Log transformation on Gr_Liv_Area
#> Dummy variables from all_nominal_predictors()
```

이것을 분해해 보겠습니다:

1.  공식과 함께 `recipe()`를 호출하면 레시피에 "성분(ingredients)" 또는 변수(예: 예측 변수, 결과 변수)의 *역할(roles)*을 알려줍니다. 열의 데이터 유형을 결정하기 위해 `ames_train` 데이터만 사용합니다.

2.  `step_log()`는 `Gr_Liv_Area`가 로그 변환되어야 함을 선언합니다.

3.  `step_dummy()`는 어떤 변수가 정성적 형식에서 정량적 형식으로 변환되어야 하는지를 지정하는데, 이 경우 가변수(dummy variables) 또는 지시 변수(indicator variables)를 사용합니다. 지시 변수 또는 가변수는 정성적 정보를 인코딩하는 이진 숫자 변수(1과 0으로 된 열)입니다. 이 장의 뒷부분에서 이러한 종류의 변수에 대해 더 깊이 파고들 것입니다.

`all_nominal_predictors()` 함수는 사실상(in nature) 현재 팩터(factor) 또는 문자형인 예측 변수 열의 이름을 캡처합니다. 이것은 `starts_with()` 또는 `matches()`와 유사하지만 레시피 내부에서만 사용할 수 있는 dplyr과 유사한 선택기(selector) 함수입니다.

###### 참고 (Note)

recipes 패키지에 특화된 다른 선택기들은 `all_numeric_predictors()`, `all_numeric()`, `all_predictors()`, `all_outcomes()`입니다. dplyr에서처럼 쉼표로 구분된 하나 이상의 인용 부호 없는 표현식을 사용하여 각 단계의 영향을 받는 열을 선택할 수 있습니다.

공식이나 원시 예측 변수(raw predictors)보다 레시피를 사용하면 어떤 장점이 있습니까? 다음을 포함하여 몇 가지가 있습니다:

- 이러한 계산은 모델링 함수와 밀접하게 결합되어 있지 않기 때문에 여러 모델에 걸쳐 재활용될 수 있습니다.

- 레시피는 공식이 제공할 수 있는 것보다 더 광범위한 데이터 처리 선택을 가능하게 합니다.

- 구문이 매우 간결할 수 있습니다. 예를 들어 `all_nominal_predictors()`는 특정 유형의 처리를 위해 많은 변수를 캡처하는 데 사용할 수 있는 반면, 공식은 각각을 명시적으로 나열해야 합니다.

- 모든 데이터 처리는 반복되거나 심지어 다른 파일에 분산되어 있는 스크립트 대신 단일 R 객체에서 캡처될 수 있습니다.

# 레시피 사용하기 (Using Recipes)

[7장](ch07.xhtml#workflows)에서 논의했듯이 전처리 선택 및 특징 공학은 일반적으로 별도의 작업이 아니라 모델링 워크플로의 일부로 고려되어야 합니다. workflows 패키지에는 다양한 유형의 전처리기를 처리하기 위한 상위 수준(high-level) 함수가 포함되어 있습니다. 우리의 이전 워크플로(`lm_wflow`)는 단순한 dplyr 선택기 세트를 사용했습니다. 더 복잡한 특징 공학을 사용하여 그 접근 방식을 개선하기 위해 모델링을 위한 데이터를 전처리하는 데 `simple_ames` 레시피를 사용해 보겠습니다.

이 객체를 워크플로에 첨부할 수 있습니다:

```
lm_wflow %>%
  add_recipe(simple_ames)
#> Error in `add_recipe()`:
#> ! A recipe cannot be added when variables already exist.
```

그것은 작동하지 않았습니다! 한 번에 하나의 전처리 방법만 가질 수 있으므로 레시피를 추가하기 전에 기존 전처리기를 제거해야 합니다.

```
lm_wflow <-
  lm_wflow %>%
  remove_variables() %>%
  add_recipe(simple_ames)
lm_wflow
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: linear_reg()
#>
#> ── Preprocessor ─────────────────────────────────────
#> 2 Recipe Steps
#>
#> • step_log()
#> • step_dummy()
#>
#> ── Model ────────────────────────────────────────────
#> Linear Regression Model Specification (regression)
#>
#> Computational engine: lm
```

간단한 `fit()` 호출을 사용하여 레시피와 모델을 모두 추정해 보겠습니다:

```
lm_fit <- fit(lm_wflow, ames_train)
```

`predict()` 메소드는 훈련 세트에 사용된 것과 동일한 전처리를 모델의 `predict()` 메소드에 전달하기 전에 새로운 데이터에 적용합니다:

```
predict(lm_fit, ames_test %>% slice(1:3))
#> # A tibble: 3 × 1
#>   .pred
#>   <dbl>
#> 1  5.08
#> 2  5.32
#> 3  5.28
```

기본(bare) 모델 객체나 레시피가 필요한 경우 이를 검색할 수 있는 `extract_*` 함수가 있습니다:

```
# 추정된 후에 레시피 가져오기:
lm_fit %>%
  extract_recipe(estimated = TRUE)
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor          4
#>
#> Training data contained 2342 data points and no missing data.
#>
#> Operations:
#>
#> Log transformation on Gr_Liv_Area [trained]
#> Dummy variables from Neighborhood, Bldg_Type [trained]

# 모델 피팅을 깔끔하게(tidy) 하기 위해:
lm_fit %>%
  # 이것은 parsnip 객체를 반환합니다:
  extract_fit_parsnip() %>%
  # 이제 선형 모델 객체를 깔끔하게 만듭니다:
  tidy() %>%
  slice(1:5)
#> # A tibble: 5 × 5
#>   term                       estimate std.error statistic   p.value
#>   <chr>                         <dbl>     <dbl>     <dbl>     <dbl>
#> 1 (Intercept)                -0.669    0.231        -2.90 3.80e-  3
#> 2 Gr_Liv_Area                 0.620    0.0143       43.2  2.63e-299
#> 3 Year_Built                  0.00200  0.000117     17.1  6.16e- 62
#> 4 Neighborhood_College_Creek  0.0178   0.00819       2.17 3.02e-  2
#> 5 Neighborhood_Old_Town      -0.0330   0.00838      -3.93 8.66e-  5
```

###### 참고 (Note)

워크플로 객체 외부에서 레시피를 사용하는 (그리고 디버깅하는) 도구는 [16장](ch16.xhtml#dimensionality)에 설명되어 있습니다.

# recipe()에서 데이터가 어떻게 사용되는가 (How Data Are Used by the recipe())

데이터는 다양한 단계에서 레시피로 전달됩니다.

첫째, `recipe(..., data)`를 호출할 때 데이터 세트는 각 열의 데이터 유형을 결정하는 데 사용되어 `all_numeric()` 또는 `all_numeric_predictors()`와 같은 선택기를 사용할 수 있게 합니다.

둘째, `fit(workflow, data)`를 사용하여 데이터를 준비할 때 훈련 데이터는 팩터 수준(levels)을 결정하는 것부터 PCA 구성 요소를 계산하는 것, 그리고 그 사이의 모든 것에 이르기까지 `workflow`의 일부일 수 있는 레시피를 포함한 모든 추정 작업에 사용됩니다.

###### 경고 (Warning)

모든 전처리 및 특징 공학 단계는 *오직* 훈련 데이터만 사용합니다. 그렇지 않으면 새 데이터와 함께 사용할 때 정보 유출(information leakage)이 모델의 성능에 부정적인 영향을 미칠 수 있습니다.

마지막으로 `predict(workflow, new_data)`를 사용할 때, 레시피의 모델이나 전처리기 파라미터는 `new_data`의 값을 사용하여 재추정되지 않습니다. `step_normalize()`를 사용한 중심화(centering) 및 척도화(scaling)를 예로 들어보겠습니다. 이 단계를 사용하면 훈련 세트에서 적절한 열의 평균과 표준 편차가 결정됩니다. 예측 시점의 새로운 샘플들은 `predict()`가 호출될 때 훈련으로부터의 이 값들을 사용하여 표준화(standardized)됩니다.

# 단계의 예 (Examples of Steps)

계속 진행하기 전에, recipes 패키지의 기능에 대해 더 자세히 살펴보고 가장 중요한 `step_*()` 함수 중 일부를 탐색해 보겠습니다. 이러한 레시피 단계 함수는 각각 특징 공학 프로세스에서 특정한 가능한 단계를 지정하며 서로 다른 레시피 단계는 데이터의 열에 다른 영향을 미칠 수 있습니다.

## 정성적 데이터를 숫자 형식으로 인코딩하기 (Encoding Qualitative Data in a Numeric Format)

가장 일반적인 특징 공학 작업 중 하나는 명목형 또는 정성적 데이터(팩터 또는 문자)가 숫자로 인코딩되거나 표현될 수 있도록 변환하는 것입니다. 때로는 이러한 변환 이전에 유용한 방식으로 정성적 열의 팩터 수준을 변경할 수 있습니다. 예를 들어 `step_unknown()`을 사용하여 결측값을 전용 팩터 수준으로 변경할 수 있습니다. 유사하게 미래 데이터에서 새로운 팩터 수준이 나타날 것으로 예상되는 경우 `step_novel()`이 이 목적을 위해 새로운 수준을 할당할 수 있습니다.

또한 `step_other()`를 사용하면 훈련 세트에서 팩터 수준의 빈도(frequencies)를 분석하여 자주 발생하지 않는 값을 지정할 수 있는 임곗값(threshold)과 함께 포괄적인(catch-all) 수준인 "기타(other)"로 변환할 수 있습니다. 좋은 예가 우리 데이터의 `Neighborhood` 예측 변수이며, [그림 8-1](#figure-8-1.-frequencies-of-neighborhoods-in-the-ames-training-set.)에 나타나 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0801.png" alt="tmwr 0801" />
<h6 id="figure-8-1.-frequencies-of-neighborhoods-in-the-ames-training-set.">그림 8-1. Ames 훈련 세트의 동네 빈도.</h6>
</figure>

여기서 우리는 두 동네가 훈련 데이터에 5개 미만의 부동산을 가지고 있음을 알 수 있습니다(Landmark 및 Green Hills). 이 경우 Landmark 동네의 주택은 훈련 세트에 전혀 포함되지 않았습니다. 일부 모델의 경우 열에 단일 0이 아닌 항목이 있는 가변수를 갖는 것이 문제가 될 수 있습니다. 최소한 이러한 특징(features)이 모델에 중요할 가능성은 매우 낮습니다. 레시피에 `step_other(Neighborhood, threshold = 0.01)`를 추가하면 동네의 하위 1%가 "기타(other)"라는 새로운 수준으로 하나로 묶일(lumped) 것입니다. 이 훈련 세트에서는 7개 동네가 여기에 포함될 것입니다.

Ames 데이터의 경우 다음을 사용하도록 레시피를 수정할 수 있습니다:

```
simple_ames <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())
```

###### 참고 (Note)

전부는 아니지만 많은 기반(underlying) 모델 계산에서 예측 변수 값이 숫자로 인코딩되어야 합니다. 눈에 띄는 예외로는 트리 기반 모델, 규칙 기반(rule-based) 모델 및 나이브 베이즈(naive Bayes) 모델이 있습니다.

팩터 예측 변수를 숫자 형식으로 변환하는 가장 일반적인 방법은 가변수(dummy variables) 또는 지시 변수(indicator variables)를 만드는 것입니다. Ames 데이터에서 5개의 수준(levels)을 가진 팩터 변수인 건물 유형(building type) 예측 변수를 예로 들어보겠습니다([표 8-1](#dummy-vars) 참조). 가변수의 경우 단일 `Bldg_Type` 열이 값이 0 또는 1인 4개의 숫자 열로 대체됩니다. 이러한 이진 변수(binary variables)는 특정 팩터 수준 값을 나타냅니다. R에서는 첫 번째 팩터 수준(이 경우 `OneFam`)에 대한 열을 제외하는 것이 관례입니다. `Bldg_Type` 열은 행이 해당 값을 가질 때 1이고 그렇지 않으면 0인 `TwoFmCon`이라는 열로 바뀔 것입니다. 다른 세 개의 열도 비슷하게 생성됩니다:

| 원시 데이터 (Raw data) | TwoFmCon | Duplex | Twnhs | TwnhsE |
|----------|----------|--------|-------|--------|
| OneFam   | 0        | 0      | 0     | 0      |
| TwoFmCon | 1        | 0      | 0     | 0      |
| Duplex   | 0        | 1      | 0     | 0      |
| Twnhs    | 0        | 0      | 1     | 0      |
| TwnhsE   | 0        | 0      | 0     | 1      |

표 8-1. 정성적 예측 변수에 대한 이진 인코딩(즉, 가변수)의 예 {#dummy-vars}

왜 5개 모두가 아닐까요? 가장 근본적인 이유는 단순성 때문입니다. 이 네 열의 값을 알면 이것들이 상호 배타적인(mutually exclusive) 범주이기 때문에 마지막 값을 알아낼 수 있습니다. 더 기술적으로 고전적인 정당화는 일반 선형 회귀를 포함한 여러 모델에서 열 사이에 선형 종속성(linear dependencies)이 있을 때 수치적 문제가 발생한다는 것입니다. 5개의 건물 유형 지시자 열이 모두 포함된다면 (만약 있다면) 절편(intercept) 열에 더해질 것입니다. 이것은 기반(underlying) 행렬 대수에서 문제 또는 아마도 명백한 오류를 유발할 것입니다.

일부 모델에서는 전체 인코딩 세트를 사용할 수 있습니다. 이를 전통적으로 *원-핫 인코딩(one-hot encoding)*이라고 하며 `step_dummy()`의 `one_hot` 인수를 사용하여 달성할 수 있습니다.

`step_dummy()`의 한 가지 유용한 특징은 결과적으로 나오는 가변수의 이름이 어떻게 지정될지에 대해 더 많은 제어를 할 수 있다는 것입니다. 기본 R에서 가변수 이름은 변수 이름과 수준(level)을 섞어 `NeighborhoodVeenker`와 같은 이름을 만듭니다. 기본적으로 레시피는 이름과 수준 사이의 구분 기호로 밑줄을 사용하며(예: `Neighborhood_Veenker`), 이름에 사용자 지정 형식을 사용하는 옵션이 있습니다. 레시피의 기본 명명 규칙(naming convention)은 `starts_with("Neighborhood_")`와 같은 선택기를 사용하여 향후 단계에서 이러한 새 열을 캡처하기 쉽게 만듭니다.

전통적인 가변수는 전체 숫자 특성 세트를 생성하기 위해 가능한 모든 범주를 알아야 합니다. 숫자 형식으로 이 변환을 수행하는 다른 방법이 있습니다. *특징 해싱(Feature hashing)* 방법은 범주의 값만 고려하여 미리 정의된 가변수 풀에 할당합니다. *효과(Effect)* 또는 *우도 인코딩(likelihood encodings)*은 원본 데이터를 해당 데이터의 *효과*를 측정하는 단일 숫자 열로 바꿉니다. 특징 해싱 및 효과 인코딩 모두 데이터에서 새로운 팩터 수준이 나타나는 상황을 원활하게 처리할 수 있습니다. [17장](ch17.xhtml#categorical)에서는 단순한 가변수나 지시 변수 이외에도 범주형 데이터를 인코딩하기 위한 이러한 방법 및 기타 방법을 탐구합니다.

###### 참고 (Note)

서로 다른 레시피 단계는 데이터의 변수에 적용될 때 다르게 동작합니다. 예를 들어 `step_log()`는 이름을 바꾸지 않고 제자리(in place)에서 열을 수정합니다. `step_dummy()`와 같은 다른 단계는 원래 데이터 열을 제거하고 이름이 다른 하나 이상의 열로 바꿉니다. 레시피 단계의 효과는 수행되는 특징 공학 변환의 유형에 따라 달라집니다.

## 상호작용 항 (Interaction Terms)

상호작용(Interaction) 효과는 두 개 이상의 예측 변수를 포함합니다. 이러한 효과는 한 예측 변수가 결과에 미치는 영향이 하나 이상의 다른 예측 변수에 따라 달라질 때 발생합니다. 예를 들어 출퇴근 시간에 교통량이 얼마나 될지 예측하려고 한다면, 두 가지 잠재적인 예측 변수는 출퇴근하는 특정 시간대와 날씨일 수 있습니다. 그러나 교통량과 악천후 사이의 관계는 시간대에 따라 다릅니다. 이 경우 원래 두 개의 예측 변수(주효과(main effects)라고 함)와 함께 모델에 두 예측 변수 간의 상호작용 항을 추가할 수 있습니다. 수치상으로 예측 변수 간의 상호작용 항은 이들의 곱으로 인코딩됩니다. 상호작용은 결과에 미치는 영향 측면에서 정의되며 다양한 유형의 데이터(예: 숫자형, 범주형 등)의 조합이 될 수 있습니다. [Kuhn and Johnson (2020)의 7장](https://oreil.ly/WCpCP)에서는 상호작용과 이를 탐지하는 방법에 대해 더 자세히 논의합니다.

Ames 훈련 세트를 탐색한 후 [그림 8-2](#building-type-interactions)에 나타난 것처럼 총 생활 면적에 대한 회귀 기울기가 건물 유형에 따라 다름을 발견할 수 있습니다.

```
ggplot(ames_train, aes(x = Gr_Liv_Area, y = 10^Sale_Price)) +
  geom_point(alpha = .2) +
  facet_wrap(~ Bldg_Type) +
  geom_smooth(method = lm, formula = y ~ x, se = FALSE, color = "lightblue") +
  scale_x_log10() +
  scale_y_log10() +
  labs(x = "Gross Living Area", y = "Sale Price (USD)")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0802.png" alt="tmwr 0802" />
<h6 id="figure-8-2.-gross-living-area-in-log-10-units-versus-sale-price-also-in-log-10-units-for-five-different-building-types.">그림 8-2. 5가지 건물 유형에 대한 총 생활 면적(로그 10 단위) 대비 판매 가격(또한 로그 10 단위).</h6>
</figure>

레시피에서 상호작용은 어떻게 지정될까요? 기본 R 공식은 `:`을 사용하여 상호작용을 나타내므로 다음과 같이 사용합니다:

```
Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) + Bldg_Type +
  log10(Gr_Liv_Area):Bldg_Type
# 또는
Sale_Price ~ Neighborhood + log10(Gr_Liv_Area) * Bldg_Type
```

여기서 `*`는 해당 열들을 주효과와 상호작용 항으로 확장합니다. 다시 말하지만 공식 메서드는 많은 작업을 동시에 수행하며 팩터 변수(`Bldg_Type` 등)가 먼저 가변수로 확장되어야 하고 상호작용에 결과적으로 생성된 이진 열이 모두 포함되어야 함을 이해합니다.

레시피는 더 명시적이고 순차적이며 더 많은 제어권을 제공합니다. 현재 레시피에서 `step_dummy()`는 이미 가변수를 생성했습니다. 상호작용을 위해 이들을 어떻게 결합할까요? 추가 단계는 물결표(tilde) 오른쪽 항이 상호작용인 `step_interact(~ *interaction terms*)`처럼 보일 것입니다. 여기에는 선택기(selectors)가 포함될 수 있으므로 다음을 사용하는 것이 적절합니다:

```
simple_ames <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  # Gr_Liv_Area는 이전 단계에서 로그 척도로 변환되었습니다
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") )
```

이 공식에서 상호작용을 `+`로 구분하여 추가적인 상호작용을 지정할 수 있습니다. 또한 레시피는 서로 다른 변수들 간의 상호작용만 사용합니다. 공식에서 `var_1:var_1`을 사용하는 경우 이 항은 무시됩니다.

레시피에서 건물 유형에 대한 가변수를 아직 만들지 않았다고 가정해 보십시오. 다음과 같이 이 단계에 팩터 열을 포함하는 것은 부적절합니다:

`step_interact` `(` `~` `Gr_Liv_Area` `:` `Bldg_Type` `)`

이것은 `step_interact()`가 사용하는 기반(기본 R) 코드에 가변수를 만든 다음 상호작용을 형성하도록 지시하는 것입니다. 실제로 이 오류가 발생하면 이로 인해 예기치 않은 결과가 발생할 수 있다는 경고가 나타납니다.

###### 경고 (Warning)

이 동작은 더 많은 제어권을 제공하지만 R의 표준 모델 공식과는 다릅니다.

가변수 이름 지정과 마찬가지로 레시피는 상호작용 항에 대해 더 일관된 이름을 제공합니다. 이 경우 상호작용의 이름은 `Gr_Liv_Area:Bldg_TypeDuplex`(데이터 프레임에 유효한 열 이름이 아님) 대신 `Gr_Liv_Area_x_Bldg_Type_Duplex`로 지정됩니다.

###### 참고 (Note)

*순서가 중요함을 기억하십시오*. 총 생활 면적은 상호작용 항 이전에 로그 변환됩니다. 이후 이 변수와의 상호작용 역시 로그 척도를 사용할 것입니다.

## 스플라인 함수 (Spline Functions)

예측 변수가 결과 변수와 비선형(nonlinear) 관계를 가질 때, 일부 유형의 예측 모델은 훈련 중에 이 관계를 적응적으로 근사화(approximate)할 수 있습니다. 그러나 일반적으로 더 단순한 것이 낫고 선형 피팅(linear fit)과 같은 단순한 모델을 사용하고 Ames 주택 데이터의 경도 및 위도처럼 이를 필요로 할 수 있는 예측 변수에 특정 비선형 특성을 추가하려고 하는 것은 드문 일이 아닙니다. 이를 수행하는 일반적인 방법 중 하나는 *스플라인(spline)* 함수를 사용하여 데이터를 표현하는 것입니다. 스플라인은 기존 숫자 예측 변수를 모델이 유연하고 비선형적인 관계를 모방할 수 있게 하는 열의 집합으로 바꿉니다. 데이터에 스플라인 항(spline terms)이 더 많이 추가될수록 관계를 비선형적으로 표현하는 능력이 증가합니다. 안타깝게도 우연히 발생하는 데이터 추세를 포착할 가능성(즉, 과적합(overfitting))도 증가할 수 있습니다.

`ggplot` 내에서 `geom_smooth()`를 사용해 본 적이 있다면 데이터의 스플라인 표현을 사용해 본 적이 있을 것입니다. 예를 들어 [그림 8-3](#ames-latitude-splines)의 각 패널은 위도 예측 변수에 대해 각기 다른 수의 평활 스플라인(smooth splines)을 사용합니다:

```
library(patchwork)
library(splines)

plot_smoother <- function(deg_free) {
  ggplot(ames_train, aes(x = Latitude, y = 10^Sale_Price)) +
    geom_point(alpha = .2) +
    scale_y_log10() +
    geom_smooth(
      method = lm,
      formula = y ~ ns(x, df = deg_free),
      color = "lightblue",
      se = FALSE
    ) +
    labs(title = paste(deg_free, "Spline Terms"),
         y = "Sale Price (USD)")
}

( plot_smoother(2) + plot_smoother(5) ) / ( plot_smoother(20) + plot_smoother(100) )
```

splines 패키지의 `ns()` 함수는 *자연 스플라인(natural splines)*이라는 함수를 사용하여 특성(feature) 열을 생성합니다.

[그림 8-3](#ames-latitude-splines)의 일부 패널은 피팅이 좋지 않습니다. 2개의 항은 데이터를 *과소적합(underfit)*하는 반면 100개의 항은 *과적합(overfit)*합니다. 5개와 20개의 항이 있는 패널은 데이터의 주요 패턴을 포착하는 합리적으로 평활한(reasonably smooth) 피팅으로 보입니다. 이는 적절한 양의 "비선형성(nonlinearness)"이 중요함을 나타냅니다. 그러면 스플라인 항의 수는 이 모델의 *튜닝 매개변수(tuning parameter)*로 간주될 수 있습니다. 이러한 유형의 매개변수는 [12장](ch12.xhtml#tuning)에서 살펴봅니다.

레시피에서는 여러 단계로 이러한 유형의 항을 만들 수 있습니다. 이 예측 변수에 대한 자연 스플라인 표현을 추가하려면:

```
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type + Latitude,
         data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, deg_free = 20)
```

동네(neighborhood)와 위도(latitude) 모두 동일한 기반 데이터를 다른 방식으로 나타내기 때문에 사용자는 둘 다 모델에 있어야 하는지 여부를 결정해야 합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0803.png" alt="tmwr 0803" />
<h6 id="figure-8-3.-sale-price-versus-latitude-with-trend-lines-using-natural-splines-with-different-degrees-of-freedom.">그림 8-3. 서로 다른 자유도(degrees of freedom)의 자연 스플라인을 사용한 추세선(trend lines)이 있는 위도 대비 판매 가격.</h6>
</figure>

## 특성 추출 (Feature Extraction)

여러 특성을 한 번에 표현하는 또 다른 일반적인 방법을 *특성 추출(feature extraction)*이라고 합니다. 이러한 기술의 대부분은 광범위한 집합 전체의 정보를 포착하는 예측 변수로부터 새로운 특성을 생성합니다. 예를 들어 주성분 분석(PCA)은 더 적은 수의 특성을 사용하여 예측 변수 세트의 원래 정보를 가능한 한 많이 추출하려고 시도합니다. PCA는 선형 추출 방법입니다. 즉, 각 새로운 특성은 원래 예측 변수의 선형 조합(linear combination)입니다. PCA의 좋은 점 중 하나는 주성분 또는 PCA 점수(scores)라고 불리는 각각의 새로운 특성들이 서로 상관관계가 없다는 것입니다. 이 때문에 PCA는 예측 변수 간의 상관관계를 줄이는 데 매우 효과적일 수 있습니다. PCA는 오직 예측 변수만 인식(aware of)한다는 점에 유의하세요. 새로운 PCA 특성은 결과 변수와 연관이 없을 수 있습니다.

Ames 데이터에서 지하실의 총 면적(`Total_Bsmt_SF`), 1층의 면적(`First_Flr_SF`), 총 생활 면적(`Gr_Liv_Area`) 등과 같은 여러 예측 변수가 부동산의 크기를 측정합니다. 잠재적으로 중복(redundant)되는 이러한 변수들을 더 작은 특성 세트로 표현하기 위한 옵션으로 PCA가 될 수 있습니다. 총 생활 면적을 제외하고 이러한 예측 변수의 이름에는 (제곱피트를 나타내는) `SF`라는 접미사가 있으므로 PCA에 대한 레시피 단계는 다음과 같을 수 있습니다:

`# 주택 크기 예측 변수를 포착하기 위해 정규 표현식을 사용합니다:` `step_pca``(``matches``(``"(SF$)|(Gr_Liv)"``))`

이 모든 열은 제곱피트로 측정된다는 점에 유의하세요. PCA는 모든 예측 변수가 동일한 척도에 있다고 가정합니다. 이 경우에는 사실이지만 각 열을 중심화(center)하고 척도화(scale)하는 `step_normalize()`를 이 단계보다 선행하는 경우가 많습니다.

독립 성분 분석(ICA), 비음수 행렬 분해(NNMF), 다차원 척도법(MDS), 균일 다양체 근사 및 투영(UMAP) 등과 같은 다른 추출 방법을 위한 기존 레시피 단계들이 있습니다.

## 행 샘플링 단계 (Row Sampling Steps)

레시피 단계는 데이터 세트의 행에도 영향을 미칠 수 있습니다. 예를 들어, 클래스 불균형(class imbalances)에 대한 *서브샘플링(subsampling)* 기법은 모델에 제공되는 데이터의 클래스 비율(proportions)을 변경합니다. 이러한 기법들은 종종 전반적인 성능을 향상시키지는 않지만 예측된 클래스 확률의 분포가 더 잘 동작(behaved)하도록 만들 수 있습니다. 클래스 불균형이 있는 데이터를 서브샘플링할 때 시도할 수 있는 접근 방식은 다음과 같습니다.

다운샘플링 (Downsampling)  
데이터를 다운샘플링하면 소수 클래스(minority class)를 유지하고 클래스 빈도가 균형을 이루도록 다수 클래스(majority class)의 무작위 샘플을 추출합니다.

업샘플링 (Upsampling)  
업샘플링은 소수 클래스의 샘플을 복제하여 클래스의 균형을 맞춥니다. 일부 기술은 소수 클래스 데이터와 유사한 새로운 샘플을 합성하여 이 작업을 수행하는 반면, 다른 방법은 단순히 동일한 소수 샘플을 반복적으로 추가합니다.

하이브리드 방법 (Hybrid methods)  
다운샘플링과 업샘플링을 결합하여 수행합니다.

[themis](https://oreil.ly/MWdh6) 패키지에는 서브샘플링을 통해 클래스 불균형을 해결하는 데 사용할 수 있는 레시피 단계가 있습니다. 단순한 다운샘플링의 경우 다음을 사용합니다:

`step_downsample``(``outcome_column_name``)`

###### 경고 (Warning)

오직 훈련 세트만 이러한 기술의 영향을 받아야 합니다. 테스트 세트 또는 다른 홀드아웃(holdout) 샘플은 레시피를 사용하여 처리될 때 있는 그대로 남겨두어야 합니다. 이러한 이유로 모든 서브샘플링 단계는 `skip` 인수의 기본값이 `TRUE`로 설정되어 있습니다.

다른 단계 함수들도 행 기반(row-based)입니다: `step_filter()`, `step_sample()`, `step_slice()`, `step_arrange()`. 이러한 단계들의 거의 모든 사용에서 `skip` 인수는 `TRUE`로 설정되어야 합니다.

## 일반 변환 (General Transformations)

원래 dplyr 작업을 반영(Mirroring)하여, 데이터에 대한 다양한 기본 작업을 수행하는 데 `step_mutate()`를 사용할 수 있습니다. `Bedroom_AbvGr / Full_Bath`, 즉 Ames 주택 데이터에 대한 욕실 대비 침실의 비율과 같이 두 변수의 비율을 계산하는 것과 같은 간단한 변환에 사용하는 것이 가장 좋습니다.

###### 경고 (Warning)

이 유연한 단계를 사용할 때, 전처리 과정에서 데이터 유출(data leakage)을 피하기 위해 각별히 주의하세요. 예를 들어, 변환 `x = w > mean(w)`를 생각해 보십시오. 새로운 데이터나 테스트 데이터에 적용될 때 이 변환은 훈련 데이터에서 `w`의 평균이 아니라 *새로운* 데이터에서 `w`의 평균을 사용할 것입니다.

## 자연어 처리 (Natural Language Processing)

레시피는 열이 특성(features)인 전통적인 구조에 있지 않은 데이터도 처리할 수 있습니다. 예를 들어 [textrecipes](https://oreil.ly/iwP9x) 패키지는 자연어 처리 방법을 데이터에 적용할 수 있습니다. 입력 열은 일반적으로 텍스트 문자열이며, 데이터를 토큰화(tokenize)(예: 텍스트를 개별 단어로 분할)하고, 토큰을 필터링(filter out)하고, 모델링에 적합한 새 특성을 생성하기 위해 여러 단계를 사용할 수 있습니다.

# 새 데이터를 위해 단계 건너뛰기 (Skipping Steps for New Data)

판매 가격 데이터는 이미 `ames` 데이터 프레임에서 로그 변환되었습니다. 다음과 같이 사용하는 것은 어떨까요?

`step_log``(``Sale_Price``,` `base` `=` `10``)`

이렇게 하면 알려지지 않은 판매 가격을 가진 새로운 주택(properties)에 레시피가 적용될 때 실패(failure)를 일으킬 것입니다. 가격은 우리가 예측하려는 대상이기 때문에, 데이터에 이 변수에 대한 열이 없을 가능성이 큽니다. 사실 정보 유출을 방지하기 위해 많은 tidymodels 패키지는 예측할 때 사용되는 데이터를 분리(isolate)합니다. 이는 예측 시간에 훈련 세트와 결과 변수 열을 사용할 수 없음을 의미합니다.

###### 참고 (Note)

결과 변수 열에 대한 간단한 변환의 경우, 이러한 작업은 *레시피 외부에서 수행(conducted outside of the recipe)*하는 것을 강력히 권장합니다.

그러나 이것이 적절한 해결책이 아닌 다른 상황들이 있습니다. 예를 들어 심각한 클래스 불균형이 있는 분류 모델의 경우, 모델링 함수에 제공되는 데이터를 *서브샘플링(subsampling)*하는 것이 일반적입니다. 예를 들어, 두 개의 클래스가 있고 10%의 이벤트 발생률(event rate)이 있다고 가정해 봅시다. 단순하지만 논란의 여지가 있는 접근 방식은 모델에 모든 이벤트(events)와 무작위로 추출한 10%의 비이벤트(nonevent) 샘플이 제공되도록 데이터를 *다운샘플링*하는 것입니다.

문제는 예측되는 데이터에 동일한 서브샘플링 프로세스가 적용되어서는 안 된다는 것입니다. 결과적으로 레시피를 사용할 때 일부 작업은 모델에 주어지는 데이터에만 적용되도록 보장하는 메커니즘이 필요합니다. 각 단계 함수에는 `TRUE`로 설정하면 `predict()` 함수에 의해 무시될 수 있는 `skip`이라는 옵션이 있습니다. 이러한 방식으로 새 샘플에 적용할 때 오류를 일으키지 않고 모델링 데이터에 영향을 미치는 단계를 격리(isolate)할 수 있습니다. 그러나 `fit()`을 사용할 때는 모든 단계가 적용됩니다.

작성 시점을 기준으로, 오직 훈련 데이터에만 적용되는 recipes 및 themis 패키지의 단계 함수는 다음과 같습니다:

- `step_adasyn()`

- `step_bsmote()`

- `step_downsample()`

- `step_filter()`

- `step_nearmiss()`

- `step_rose()`

- `step_sample()`

- `step_slice()`

- `step_smote()`

- `step_smotenc()`

- `step_tomek()`

- `step_upsample()`

# recipe() 깔끔하게 정리하기 (Tidy a recipe())

[3장](ch03.xhtml#base-r)에서 우리는 통계적 객체를 위한 `tidy()` 동사를 소개했습니다. 개별 레시피 단계뿐만 아니라 레시피를 위한 `tidy()` 메소드도 있습니다. 계속 진행하기 전에 이 장에서 논의한 몇 가지 새로운 단계를 사용하여 Ames 데이터에 대한 확장된 레시피를 만들어 보겠습니다:

```
ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)
```

레시피 객체와 함께 호출될 때 `tidy()` 메소드는 레시피 단계의 요약을 제공합니다:

```
tidy(ames_rec)
#> # A tibble: 5 × 6
#>   number operation type     trained skip  id
#>    <int> <chr>     <chr>    <lgl>   <lgl> <chr>
#> 1      1 step      log      FALSE   FALSE log_66JTU
#> 2      2 step      other    FALSE   FALSE other_ePfcw
#> 3      3 step      dummy    FALSE   FALSE dummy_Z18Cl
#> 4      4 step      interact FALSE   FALSE interact_JLU36
#> 5      5 step      ns       FALSE   FALSE ns_rvsqQ
```

이 결과는 개별 단계를 식별(identifying)하는 데 도움이 될 수 있으며, 아마도 특정 단계에 대해 `tidy()` 메소드를 실행할 수 있게 해 줄 것입니다.

우리는 모든 단계 함수 호출에서 `id` 인수를 지정할 수 있습니다; 그렇지 않으면 무작위 접미사를 사용하여 생성됩니다. 동일한 유형의 단계가 레시피에 두 번 이상 추가된 경우 이 값을 설정하는 것이 도움이 될 수 있습니다. 우리는 `step_other()`에 대해 `tidy()`를 원할 것이므로 미리 `id`를 지정해 보겠습니다:

```
ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01, id = "my_id") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)
```

이 새로운 레시피로 워크플로를 다시 피팅하겠습니다:

```
lm_wflow <-
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)
```

우리가 지정한 식별자 `id`와 함께 `tidy()` 메소드를 다시 호출하면 `step_other()` 적용에 대한 결과를 얻을 수 있습니다:

```
estimated_recipe <-
  lm_fit %>%
  extract_recipe(estimated = TRUE)

tidy(estimated_recipe, id = "my_id")
#> # A tibble: 22 × 3
#>   terms        retained           id
#>   <chr>        <chr>              <chr>
#> 1 Neighborhood North_Ames         my_id
#> 2 Neighborhood College_Creek      my_id
#> 3 Neighborhood Old_Town           my_id
#> 4 Neighborhood Edwards            my_id
#> 5 Neighborhood Somerset           my_id
#> 6 Neighborhood Northridge_Heights my_id
#> # … 16개 행이 더 있습니다
```

여기서 `step_other()`를 사용하기 위해 본 `tidy()` 결과는 어떤 팩터 수준이 유지되었는지 보여줍니다(다시 말해서 새로운 "기타(other)" 범주에 추가되지 않았습니다).

우리가 필요로 하는 레시피의 단계를 안다면 식별자인 `number`와 함께 `tidy()` 메소드를 호출할 수도 있습니다:

```
tidy(estimated_recipe, number = 2)
#> # A tibble: 22 × 3
#>   terms        retained           id
#>   <chr>        <chr>              <chr>
#> 1 Neighborhood North_Ames         my_id
#> 2 Neighborhood College_Creek      my_id
#> 3 Neighborhood Old_Town           my_id
#> 4 Neighborhood Edwards            my_id
#> 5 Neighborhood Somerset           my_id
#> 6 Neighborhood Northridge_Heights my_id
#> # … 16개 행이 더 있습니다
```

각 `tidy()` 메서드는 해당 단계에 대한 관련 정보를 반환합니다. 예를 들어 `step_dummy()`에 대한 `tidy()` 메서드는 가변수로 변환된 변수들이 있는 열과 각 열에 대해 알려진 모든 수준(levels)이 포함된 또 다른 열을 반환합니다.

# 열의 역할 (Column Roles)

처음 `recipe()`를 호출할 때 공식을 사용하면, 물결표(tilde)의 어느 쪽에 있는지에 따라 각 열에 *역할(roles)*이 할당됩니다. 이러한 역할은 `"predictor"`(예측 변수) 또는 `"outcome"`(결과 변수)입니다. 그러나 필요에 따라 다른 역할을 할당할 수 있습니다.

예를 들어, 우리의 Ames 데이터 세트에서 원시(raw) 데이터에는 주소(address)에 대한 열이 포함되어 있었습니다.<sup><a href="ch08.xhtml#idm45881864636560" id="idm45881864636560-marker" data-type="noteref">1</a></sup> 예측이 이루어진 후 문제가 있는 결과를 자세히 조사할 수 있도록 해당 열을 데이터에 유지하는 것이 유용할 수 있습니다. 다시 말해서 열은 예측 변수나 결과 변수가 아닐 때에도 중요할 수 있습니다.

이 문제를 해결하기 위해 `add_role()`, `remove_role()`, `update_role()` 함수가 도움이 될 수 있습니다. 예를 들어, 주택 가격 데이터의 경우 다음을 사용하여 주소(street address) 열의 역할을 수정할 수 있습니다:

```
ames_rec %>% update_role(address, new_role = "street address")
```

이 변경 후 데이터 프레임의 `address` 열은 더 이상 예측 변수가 아니며 레시피에 따라 `"street address"`가 될 것입니다. 임의의 문자열(character string)을 역할로 사용할 수 있습니다. 또한 열은 여러 역할을 가질 수 있으므로(`add_role()`을 통해 추가 역할이 추가됨) 둘 이상의 컨텍스트(context)에서 선택할 수 있습니다.

데이터가 *리샘플링(resampled)*될 때 이것이 도움이 될 수 있습니다. 모델 피팅과 관련 없는 열을 (외부 벡터에 두기보다는) 동일한 데이터 프레임에 유지하는 데 도움이 됩니다. [10장](ch10.xhtml#resampling)에 설명된 리샘플링은 주로 행 서브샘플링(row subsampling)에 의해 데이터의 대체 버전을 만듭니다. 주소(street address)가 다른 열에 있다면 추가적인 서브샘플링이 필요할 것이며 더 복잡한 코드와 오류 발생 가능성이 높아질 수 있습니다.

마지막으로 모든 단계 함수에는 단계 결과에 역할을 할당할 수 있는 `role` 필드가 있습니다. 많은 경우 단계의 영향을 받는 열은 기존 역할을 유지합니다. 예를 들어 우리 `ames_rec` 객체에 대한 `step_log()` 호출은 `Gr_Liv_Area` 열에 영향을 미쳤습니다. 이 단계의 기본 동작(default behavior)은 새 열이 생성되지 않기 때문에 이 열에 대한 기존 역할을 유지하는 것입니다. 반례로 스플라인을 생성하는 단계는 스플라인 열이 모델에서 일반적으로 사용되는 방식이므로 새 열의 기본 역할을 `"predictor"`로 지정합니다. 대부분의 단계에는 합리적인 기본값이 있지만 기본값이 다를 수 있으므로 설명서 페이지를 확인하여 할당될 역할을 이해하십시오.

# 이 장의 요약 (Chapter Summary)

이 장에서는 가변수(dummy variables)를 생성하는 것부터 클래스 불균형 등을 다루는 것까지 유연한 특징 공학과 데이터 전처리를 위해 레시피를 사용하는 방법을 배웠습니다. 특징 공학은 정보 유출(information leakage)이 쉽게 발생할 수 있고 모범 사례(good practices)가 채택되어야 하는 모델링 프로세스의 중요한 부분입니다. recipes 패키지와 recipes를 확장하는 다른 패키지들 사이에 사용 가능한 단계가 100개가 넘습니다. 가능한 모든 레시피 단계는 [tidymodels 웹사이트](https://oreil.ly/FB0BM)에 나열되어 있습니다. recipes 프레임워크는 모델링에 앞서 데이터를 전처리하고 변환하기 위한 풍부한 데이터 조작 환경을 제공합니다. 추가로 [사용자 정의(custom) 단계 생성 방법](https://oreil.ly/0JPFP)을 볼 수 있습니다.

여기서 우리의 작업은 전적으로 워크플로 객체 내부에서만 레시피를 사용했습니다. 특징 공학은 모델과 함께 추정되어야 하기 때문에 모델링의 경우 그렇게 사용하는 것이 권장됩니다. 그러나 시각화(visualization) 및 다른 활동의 경우 워크플로가 적절하지 않을 수 있습니다. 레시피 전용 함수가 더 많이 필요할 수 있습니다. [16장](ch16.xhtml#dimensionality)에서는 레시피 피팅, 사용, 문제 해결(troubleshooting)을 위한 더 낮은 수준(lower-level)의 API에 대해 논의합니다.

이후 장에서 사용할 코드는 다음과 같습니다:

```
library(tidymodels)
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

ames_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <-
  workflow() %>%
  add_model(lm_model) %>%
  add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)
```

<sup>[1](ch08.xhtml#idm45881864636560-marker)</sup> 우리 버전의 이 데이터에는 해당 열이 포함되어 있지 않습니다.
