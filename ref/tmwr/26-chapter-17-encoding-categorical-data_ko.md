# 17장. 범주형 데이터 인코딩 (Encoding Categorical Data)

R의 통계 모델링에서 범주형(categorical) 또는 명목형(nominal) 데이터의 선호되는 표현은 제한된 수의 다른 값을 취할 수 있는 변수인 *요인(factor)*입니다; 내부적으로 요인은 텍스트 레이블 세트와 함께 정수 값의 벡터로 저장됩니다.<sup><a href="ch17.xhtml#idm45881849589504" id="idm45881849589504-marker" data-type="noteref">1</a></sup> [8장](ch08.xhtml#recipes)에서는 정성적(qualitative) 또는 명목형 데이터를 대부분의 모델 알고리즘에 더 적합한 표현으로 인코딩하거나 변환하는 접근 방식을 포함하여 피처 엔지니어링 접근 방식을 소개했습니다. Ames 주택 데이터의 `Bldg_Type`(`OneFam`, `TwoFmCon`, `Duplex`, `Twnhs` 및 `TwnhsE` 수준(levels)을 가짐)과 같은 범주형 변수를 [표 17-1](#encoding-dummies)에 표시된 것과 같은 가변수(dummy variables) 또는 지시 변수(indicator variables) 세트로 변환하는 방법을 논의했습니다.

| Raw data | TwoFmCon | Duplex | Twnhs | TwnhsE |
| -------- | -------- | ------ | ----- | ------ |
| OneFam   | 0        | 0      | 0     | 0      |
| TwoFmCon | 1        | 0      | 0     | 0      |
| Duplex   | 0        | 1      | 0     | 0      |
| Twnhs    | 0        | 0      | 1     | 0      |
| TwnhsE   | 0        | 0      | 0     | 1      |

표 17-1. 정성적 예측 변수에 대한 이진 인코딩(즉, 가변수)의 그림 {#encoding-dummies}

많은 모델 구현은 범주형 데이터에 대해 이러한 숫자 표현으로의 변환이 필요합니다.

###### 참고 (Note)

[부록](app01.xhtml#pre-proc-table)에는 다양한 모델에 대해 권장되는 전처리 기법 표가 제시되어 있습니다; 표의 얼마나 많은 모델이 모든 예측 변수에 대해 숫자형 인코딩을 필요로 하는지 확인하십시오(notice).

그러나 일부 현실적인(realistic) 데이터 세트의 경우, 직관적인(straightforward) 가변수가 잘 맞지 않습니다(not a good fit). 이는 범주가 _너무 많거나_ 예측 시점에 _새로운_ 범주가 있기 때문에 종종 발생합니다. 이 장에서는 이러한 문제를 해결(address)하는 범주형 예측 변수를 인코딩하기 위한 더 정교한(sophisticated) 옵션에 대해 논의합니다. 이러한 옵션은 [embed](https://oreil.ly/lQfup) 및 [textrecipes](https://oreil.ly/PHwTv) 패키지에서 tidymodels 레시피 단계(steps)로 사용할 수 있습니다.

# 인코딩이 필요한가요? (Is an Encoding Necessary?)

트리 또는 규칙을 기반으로 하는 모델과 같은 소수의 모델은 범주형 데이터를 기본적으로(natively) 처리할 수 있으며 이러한 종류의 피처를 인코딩하거나 변환할 필요가 없습니다. 트리 기반 모델은 `Bldg_Type`과 같은 변수를 기본적으로(natively) 요인 수준 그룹으로 분할(partition)할 수 있습니다(아마도 한 그룹에는 `OneFam`만 있고 다른 그룹에는 `Duplex`와 `Twnhs`가 함께 있는 식으로). 나이브 베이즈 모델은 모델 구조가 범주형 변수를 기본적으로 다룰 수 있는 또 다른 예입니다. 데이터 세트의 모든 다양한 종류의 `Bldg_Type`과 같이 각 수준(level) 내에서 분포가 계산됩니다.

범주형 피처를 기본적으로 처리할 수 있는 모델은 숫자형의 연속형 피처*도(also)* 다룰 수 있으므로 이러한 변수의 변환 또는 인코딩은 선택 사항이 됩니다. 이것이 모델 성능이나 모델 훈련 시간에 어떤 식으로든 도움이 될까요? [Kuhn and Johnson (2020)의 5.7절](https://oreil.ly/0ImIU)에서 보여주는 바와 같이, 변환되지 않은 요인 변수를 가진 벤치마크 데이터 세트를 동일한 피처에 대해 변환된 가변수와 비교해 볼 때 일반적으로 그렇지 않습니다(Typically no). 더미 인코딩(dummy encodings)을 사용한다고 해서 일반적으로 더 나은 모델 성능이 도출(result in)되지는 않았지만 종종 모델을 훈련하는 데 더 많은 시간이 필요했습니다.

###### 참고 (Note)

모델이 허용하는 경우 변환되지 않은 범주형 변수로 시작하는 것이 좋습니다. 더 복잡한 인코딩이 종종 그러한 모델에 대해 더 나은 성능을 도출하지 않는다는 점에 유의하십시오.

# 순서형 예측 변수 인코딩 (Encoding Ordinal Predictors)

때때로 정성적 열은 "낮음(low)", "중간(medium)", "높음(high)"과 같이 *순서가 지정(ordered)*될 수 있습니다. 기본 R(base R)에서 기본 인코딩 전략은 데이터의 다항식 확장(polynomial expansions)인 새로운 숫자형 열을 만드는 것입니다. [표 17-2](#encoding-ordered-table)에 표시된 예시와 같이 5개의 서수(ordinal) 값을 가진 열의 경우, 요인 열은 1차(linear), 2차(quadratic), 3차(cubic) 및 4차(quartic) 항에 대한 열로 대체됩니다.

| Raw data        | Linear | Quadratic | Cubic | Quartic |
| --------------- | ------ | --------- | ----- | ------- |
| None            | –0.63  | 0.53      | –0.32 | 0.12    |
| A little        | –0.32  | –0.27     | 0.63  | –0.48   |
| Some            | 0.00   | –0.53     | 0.00  | 0.72    |
| A bunch         | 0.32   | –0.27     | –0.63 | –0.48   |
| Copious amounts | 0.63   | 0.53      | 0.32  | 0.12    |

표 17-2. 순서가 있는 변수를 인코딩하기 위한 다항식 확장. {#encoding-ordered-table}

이것이 불합리한(unreasonable) 것은 아니지만, 사람들이 유용하다고 느끼는 경향이 있는 접근 방식은 아닙니다. 예를 들어, 11차 다항식은 아마도 1년의 달(months)에 대한 서수 요인을 인코딩하는 효과적인 방법이 아닐 것입니다. 대신 정규(regular) 요인으로 변환하는 `step_unorder()`와 같이 순서가 있는 요인과 관련된 레시피 단계를 시도하거나, 각 요인 수준에 특정 숫자 값을 매핑하는 `step_ordinalscore()`를 고려해 보십시오.

# 예측 변수 인코딩을 위해 결과 사용하기 (Using the Outcome for Encoding Predictors)

가변수 또는 지시 변수보다 더 복잡한 인코딩을 위한 여러 옵션이 있습니다. _효과(effect)_ 또는 *우도 인코딩(likelihood encodings)*이라고 하는 한 방법은 원본 범주형 변수를 해당 데이터의 효과를 측정하는 단일 숫자형 열로 대체합니다(Micci-Barreca 2001; Zumel and Mount 2019). 예를 들어, Ames 주택 데이터의 이웃(neighborhood) 예측 변수에 대해 [그림 17-1](#encoding-mean-price)과 같이 각 이웃에 대한 평균 또는 중앙값 판매 가격을 계산하고 원래 데이터 값 대신 이 평균을 대입(substitute)할 수 있습니다.

```
ames_train %>%
  group_by(Neighborhood) %>%
  summarize(mean = mean(Sale_Price),
            std_err = sd(Sale_Price) / sqrt(length(Sale_Price))) %>%
  ggplot(aes(y = reorder(Neighborhood, mean), x = mean)) +
  geom_point() +
  geom_errorbar(aes(xmin = mean - 1.64 * std_err, xmax = mean + 1.64 * std_err)) +
  labs(y = NULL, x = "Price (mean, log scale)")
```

이러한 종류의 효과 인코딩은 범주형 변수의 수준이 많은 경우에 잘 작동합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1701.png" alt="tmwr 1701" />
<h6 id="figure-17-1.-mean-home-price-for-neighborhoods-in-the-ames-training-set-which-can-be-used-as-an-effect-encoding-for-this-categorical-variable.">그림 17-1. 이 범주형 변수에 대한 효과 인코딩으로 사용할 수 있는 Ames 훈련 세트의 이웃에 대한 평균 주택 가격.</h6>
</figure>

## tidymodels의 효과 인코딩 (Effect Encodings in tidymodels)

tidymodels에서 embed 패키지에는 `step_lencode_glm()`, `step_lencode_mixed()` 및 `step_lencode_bayes()`와 같은 다양한 종류의 효과 인코딩을 위한 여러 레시피 단계 함수가 포함되어 있습니다. 이러한 단계는 일반화 선형 모델(GLM)을 사용하여 범주형 예측 변수의 각 수준이 결과(outcome)에 미치는 효과를 추정합니다. `step_lencode_glm()`과 같은 레시피 단계를 사용할 때는 인코딩할 변수를 먼저 지정한 다음 `vars()`를 사용하여 결과를 지정합니다.

```
library(embed)

ames_glm <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_glm(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

ames_glm
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor          6
#>
#> Operations:
#>
#> Log transformation on Gr_Liv_Area
#> Linear embedding for factors via GLM for Neighborhood
#> Dummy variables from all_nominal_predictors()
#> Interactions with Gr_Liv_Area:starts_with("Bldg_Type_")
#> Natural splines on Latitude, Longitude
```

[16장](ch16.xhtml#dimensionality)에서 자세히 설명했듯이 훈련 데이터를 사용하여 전처리 변환을 위한 매개변수를 피팅(fit)하거나 추정하기 위해 레시피를 `prep()`할 수 있습니다. 그런 다음 준비된 이 레시피를 `tidy()`하여 결과를 볼 수 있습니다.

```
glm_estimates <-
  prep(ames_glm) %>%
  tidy(number = 2)

glm_estimates
#> # A tibble: 29 × 4
#>   level              value terms        id
#>   <chr>              <dbl> <chr>        <chr>
#> 1 North_Ames          5.15 Neighborhood lencode_glm_ZsXdy
#> 2 College_Creek       5.29 Neighborhood lencode_glm_ZsXdy
#> 3 Old_Town            5.07 Neighborhood lencode_glm_ZsXdy
#> 4 Edwards             5.09 Neighborhood lencode_glm_ZsXdy
#> 5 Somerset            5.35 Neighborhood lencode_glm_ZsXdy
#> 6 Northridge_Heights  5.49 Neighborhood lencode_glm_ZsXdy
#> # … with 23 more rows
```

이 방법을 통해 생성된 새로 인코딩된 `Neighborhood` 숫자형 변수를 사용할 때 원래 수준(`"North_Ames"`와 같은)을 GLM의 `Sale_Price`에 대한 추정치로 대체합니다.

이와 같은 효과 인코딩 방법은 데이터에서 새로운(novel) 요인 수준에 직면(encountered)하는 상황도 원활하게(seamlessly) 처리할 수 있습니다. 이 `value`는 구체적인 이웃 정보가 없을 때 GLM에서 예측한 가격입니다.

```
glm_estimates %>%
  filter(level == "..new")
#> # A tibble: 1 × 4
#>   level value terms        id
#>   <chr> <dbl> <chr>        <chr>
#> 1 ..new  5.23 Neighborhood lencode_glm_ZsXdy
```

###### 경고 (Warning)

효과 인코딩은 강력할 수 있지만 주의해서 사용해야 합니다(used with care). 효과는 데이터 분할 후 훈련 세트에서 계산되어야 합니다. 이러한 유형의 지도 전처리는 과적합을 피하기 위해 엄격하게(rigorously) 리샘플링되어야 합니다([10장](ch10.xhtml#resampling) 참조).

범주형 변수에 대한 효과 인코딩을 생성할 때, 실제로는(effectively) 실제 모델 내부에 미니 모델(mini-model)을 계층화(layering)하는 것입니다. 효과 인코딩에서 과적합 가능성은 [7장](ch07.xhtml#workflows)에 설명된 대로 피처 엔지니어링이 모델 프로세스의 일부로 간주되어야 _하는_ 이유와 리샘플링 내부에서 모델 매개변수와 함께 피처 엔지니어링이 추정되어야 하는 이유에 대한 대표적인 예입니다.

## 부분 풀링(Partial Pooling)을 사용한 효과 인코딩

`step_lencode_glm()`으로 효과 인코딩을 만들면 각 요인 수준(이 예에서는 이웃)에 대한 효과를 개별적으로 추정합니다. 그러나 이러한 이웃 중 일부에는 주택이 많고(many houses) 일부에는 소수(few)만 있습니다. North Ames의 354개 훈련 세트 주택보다 Landmark 이웃의 단일(single) 훈련 세트 주택에 대한 가격 측정에서 훨씬 더 많은 불확실성(uncertainty)이 있습니다. 표본 크기가 작은 수준이 전체 평균을 향해 축소(shrunken)되도록 *부분 풀링(partial pooling)*을 사용하여 이러한 추정치를 조정(adjust)할 수 있습니다. 각 수준에 대한 효과는 혼합형(mixed) 또는 계층적(hierarchical) 일반화 선형 모델을 사용하여 한 번에(all at once) 모델링됩니다.

```
ames_mixed <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_lencode_mixed(Neighborhood, outcome = vars(Sale_Price)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

ames_mixed
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor          6
#>
#> Operations:
#>
#> Log transformation on Gr_Liv_Area
#> Linear embedding for factors via mixed effects for Neighborhood
#> Dummy variables from all_nominal_predictors()
#> Interactions with Gr_Liv_Area:starts_with("Bldg_Type_")
#> Natural splines on Latitude, Longitude
```

결과를 보기 위해 이 레시피를 `prep()`하고 `tidy()`해 보겠습니다.

```
mixed_estimates <-
  prep(ames_mixed) %>%
  tidy(number = 2)

mixed_estimates
#> # A tibble: 29 × 4
#>   level              value terms        id
#>   <chr>              <dbl> <chr>        <chr>
#> 1 North_Ames          5.15 Neighborhood lencode_mixed_SC9hi
#> 2 College_Creek       5.29 Neighborhood lencode_mixed_SC9hi
#> 3 Old_Town            5.07 Neighborhood lencode_mixed_SC9hi
#> 4 Edwards             5.10 Neighborhood lencode_mixed_SC9hi
#> 5 Somerset            5.35 Neighborhood lencode_mixed_SC9hi
#> 6 Northridge_Heights  5.49 Neighborhood lencode_mixed_SC9hi
#> # … with 23 more rows
```

그런 다음 새로운(New) 수준은 GLM의 경우와 거의 동일한 값으로 인코딩됩니다.

```
mixed_estimates %>%
  filter(level == "..new")
#> # A tibble: 1 × 4
#>   level value terms        id
#>   <chr> <dbl> <chr>        <chr>
#> 1 ..new  5.23 Neighborhood lencode_mixed_SC9hi
```

###### 참고 (Note)

`step_lencode_bayes()`를 사용하여 효과에 대한 완전한(fully) 베이지안 계층적 모델을 동일한 방식으로(in the same way) 사용할 수 있습니다.

[그림 17-2](#encoding-compare-pooling)에서 부분 풀링(partial pooling) 대(versus) 풀링 안 함(no pooling)을 사용하여 효과를 시각적으로 비교해 보겠습니다.

```
glm_estimates %>%
  rename(`no pooling` = value) %>%
  left_join(
    mixed_estimates %>%
      rename(`partial pooling` = value), by = "level"
  ) %>%
  left_join(
    ames_train %>%
      count(Neighborhood) %>%
      mutate(level = as.character(Neighborhood))
  ) %>%
  ggplot(aes(`no pooling`, `partial pooling`, size = sqrt(n))) +
  geom_abline(color = "gray50", lty = 2) +
  geom_point(alpha = 0.7) +
  coord_fixed()
#> Warning: Removed 1 rows containing missing values (geom_point).
```

[그림 17-2](#encoding-compare-pooling)에서 풀링을 안 하는 것과 비교할 때 이웃 효과에 대한 대부분의 추정치가 거의 같다는 것을 확인하십시오. 그러나 주택 수가 적은 이웃은 평균 효과를 향해(위 또는 아래로) 당겨졌습니다(pulled). 풀링을 사용하면 해당 이웃의 가격에 대한 증거(evidence)가 많지 않기 때문에 효과 추정치를 평균을 향해 축소합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1702.png" alt="tmwr 1702" />
<h6 id="figure-17-2.-comparing-the-effect-encodings-for-neighborhoods-estimated-without-pooling-to-those-with-partial-pooling.">그림 17-2. 풀링 없이 추정된 이웃에 대한 효과 인코딩과 부분 풀링이 있는 효과 인코딩 비교.</h6>
</figure>

# 피처 해싱 (Feature Hashing)

[8장](ch08.xhtml#recipes)에서 설명한 전통적인 가변수는 모든 숫자형 피처 세트를 생성하기 위해 가능한 모든 범주를 알아야 합니다. _피처 해싱(Feature hashing)_ 방법(Weinberger et al. 2009)은 가변수도 생성하지만 범주의 값만 고려하여 미리 정의된(predefined) 가변수 풀(pool)에 할당(assign)합니다. Ames의 `Neighborhood` 값을 다시 살펴보고 `rlang::hash()` 함수를 사용하여 자세히 알아보겠습니다(understand more):

```
library(rlang)

ames_hashed <-
  ames_train %>%
  mutate(Hash = map_chr(Neighborhood, hash))

ames_hashed %>%
  select(Neighborhood, Hash)
#> # A tibble: 2,342 × 2
#>   Neighborhood    Hash
#>   <fct>           <chr>
#> 1 North_Ames      076543f71313e522efe157944169d919
#> 2 North_Ames      076543f71313e522efe157944169d919
#> 3 Briardale       b598bec306983e3e68a3118952df8cf0
#> 4 Briardale       b598bec306983e3e68a3118952df8cf0
#> 5 Northpark_Villa 6af95b5db968bf393e78188a81e0e1e4
#> 6 Northpark_Villa 6af95b5db968bf393e78188a81e0e1e4
#> # … with 2,336 more rows
```

이 해싱 함수에 Briardale을 입력(input)하면 항상 동일한 출력을 얻습니다. 이 경우 이웃을 *키(keys)*라고 하고 출력을 *해시(hashes)*라고 합니다.

###### 참고 (Note)

해싱 함수는 가변(variable) 크기의 입력을 받아 고정(fixed) 크기의 출력에 매핑합니다. 해싱 함수는 암호학(cryptography)과 데이터베이스에서 일반적으로 사용됩니다.

`rlang::hash()` 함수는 128비트 해시를 생성하므로 가능한 해시 값이 `2^128`개 있음을 의미합니다. 이것은 일부 애플리케이션에서는 좋지만 _카디널리티가 높은(high-cardinality)_ 변수(수준이 많은 변수)의 피처 해싱에는 도움이 되지 않습니다. 피처 해싱에서 가능한 해시 수는 하이퍼파라미터이며 정수 해시의 모듈로(modulo)를 계산하여 모델 개발자가 설정합니다. `Hash %% 16`을 사용하면 16개의 가능한 해시 값을 얻을 수 있습니다.

```
ames_hashed %>%
  ## 먼저 R이 처리할 수 있는 정수에 대해 더 작은 해시 만들기
  mutate(Hash = strtoi(substr(Hash, 26, 32), base = 16L),
         ## 이제 모듈로 취하기
         Hash = Hash %% 16) %>%
  select(Neighborhood, Hash)
#> # A tibble: 2,342 × 2
#>   Neighborhood     Hash
#>   <fct>           <dbl>
#> 1 North_Ames          9
#> 2 North_Ames          9
#> 3 Briardale           0
#> 4 Briardale           0
#> 5 Northpark_Villa     4
#> 6 Northpark_Villa     4
#> # … with 2,336 more rows
```

이제 원래 데이터의 28개 이웃이나 엄청나게 방대한 수의 원래 해시 대신 16개의 해시 값이 생겼습니다. 이 방법은 매우 빠르고 메모리 효율적(memory efficient)이며 가능한 범주가 많은 경우 좋은 전략이 될 수 있습니다.

###### 참고 (Note)

피처 해싱은 카디널리티가 높은 범주형 데이터뿐만 아니라 텍스트 데이터에도 유용합니다. 텍스트 예측 변수를 사용한 사례 연구 데모는 [Hvitfeldt and Silge (2021)의 6.7절](https://oreil.ly/mN7fo)을 참조하십시오.

textrecipes 패키지의 tidymodels 레시피 단계를 사용하여 피처 해싱을 구현(implement)할 수 있습니다.

```
library(textrecipes)
ames_hash <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_dummy_hash(Neighborhood, signed = FALSE, num_terms = 16L) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") ) %>%
  step_ns(Latitude, Longitude, deg_free = 20)

ames_hash
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor          6
#>
#> Operations:
#>
#> Log transformation on Gr_Liv_Area
#> Feature hashing with Neighborhood
#> Dummy variables from all_nominal_predictors()
#> Interactions with Gr_Liv_Area:starts_with("Bldg_Type_")
#> Natural splines on Latitude, Longitude
```

피처 해싱은 빠르고 효율적(efficient)이지만 몇 가지 단점(downsides)이 있습니다. 예를 들어, 다른 카테고리 값이 종종 같은 해시 값에 매핑됩니다. 이것을 _충돌(collision)_ 또는 *앨리어싱(aliasing)*이라고 합니다. Ames의 이웃에서 이것이 얼마나 자주 일어났습니까? [표 17-3](#encoding-hash)은 해시 값당 이웃 수의 분포(distribution)를 제시합니다.

| Number of neighborhoods within a hash feature | Number of occurrences |
| --------------------------------------------- | --------------------- |
| 0                                             | 1                     |
| 1                                             | 7                     |
| 2                                             | 4                     |
| 4                                             | 1                     |

표 17-3. 각 이웃 수에서의 해시 피처 수 {#encoding-hash}

각 해시 값에 매핑된 이웃의 수는 0에서 4까지 다양합니다(varies). 1보다 큰 모든 해시 값은 해시 충돌의 예입니다.

피처 해싱을 사용할 때 고려해야 할(consider) 몇 가지 사항은 무엇일까요?

- 해시 함수는 되돌릴(reversed) 수 없기 때문에 피처 해싱은 직접적으로 해석(interpretable)할 수 없습니다. 해시 값에서 입력 카테고리 수준이 무엇이었는지 또는 충돌이 발생했는지 확인할(determine) 수 없습니다.

- 해시 값의 수는 이 전처리 기법의 *튜닝 매개변수(tuning parameter)*이며, 여러분의 특정(particular) 모델링 접근 방식에 적합한 것을 결정하기 위해 여러 값을 시도해야 합니다. 해시 값이 낮으면 더 많은 충돌이 도출(results in)되지만 값이 높다고 해서 원래의 고-카디널리티 변수보다 더 개선(improvement)되는 것은 아닙니다.

- 피처 해싱은 미리 결정된(predetermined) 가변수에 의존하지 않으므로 예측 시 새로운(new) 카테고리 수준을 처리할 수 있습니다.

- `signed = TRUE`를 사용하여 _부호 있는(signed)_ 해시로 해시 충돌을 줄일 수 있습니다. 이것은 해시의 부호(sign)에 따라 값을 1에서 +1 또는 –1로 확장(expands)합니다.

###### 경고 (Warning)

이 예제에서 볼 수 있듯이 일부 해시 열에는 0만 포함될 가능성(likely)이 있습니다. `step_zv()`를 통한 0-분산(zero-variance) 필터를 권장하여 이러한 열을 필터링해 내는 것을 추천합니다.

# 더 많은 인코딩 옵션 (More Encoding Options)

요인을 숫자 표현으로 변환하기 위한 더 많은 옵션이 제공(available)됩니다.

수준이 많은 범주형 변수를 낮은 차원의 벡터 세트로 변환하기 위해 전체 _엔티티 임베딩(entity embeddings)_ 세트를 구축(build)할 수 있습니다(Guo and Berkhahn 2016). 이 접근 방식은 Ames의 이웃과 함께 사용한 예제보다 훨씬 더 많은 카테고리 수준을 가진 명목형(nominal) 변수에 적합(best suited)합니다.

###### 참고 (Note)

엔티티 임베딩의 아이디어는 텍스트 데이터에서 단어 임베딩(word embeddings)을 만드는 데 사용되는 방법에서 비롯됩니다(comes from). 단어 임베딩에 대한 자세한 내용은 [Hvitfeldt and Silge (2021)의 5장](https://oreil.ly/k3yCZ)을 참조하십시오.

범주형 변수에 대한 임베딩은 embed 패키지의 `step_embed()` 함수를 사용하여 텐서플로(TensorFlow) 신경망을 통해 학습(learned)될 수 있습니다. 결과만 단독으로(alone) 사용하거나 선택적으로(optionally) 결과와 일련의 추가 예측 변수를 함께 사용할 수 있습니다. 피처 해싱과 마찬가지로, 생성할 새로운 인코딩 열의 수는 피처 엔지니어링의 하이퍼파라미터입니다. 신경망 구조(은닉 유닛의 수) 및 신경망을 적합하는 방법(훈련할 에포크 수, 지표를 측정할 때 검증에 사용할 데이터의 양)에 대해서도 결정을 내려야 합니다(must make decisions).

이진(binary) 결과를 다루는 데 사용할 수 있는 또 다른(Yet one more) 옵션은 이진 결과와의 연관성(association)을 기반으로 카테고리 수준의 세트를 변환하는 것입니다. 이 _증거 가중치(weight of evidence, WoE)_ 변환(Good 1985)은 "베이즈 요인(Bayes factor)"(사전 승산(prior odds)에 대한 사후 승산(posterior odds)의 비율)의 로그를 사용하고 각 범주 수준을 WoE 값에 매핑하는 사전(dictionary)을 만듭니다. WoE 인코딩은 embed 패키지의 `step_woe()` 함수로 결정할 수 있습니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 범주형 예측 변수를 인코딩하기 위해 전처리 레시피를 사용하는 방법을 배웠습니다. 범주형 변수를 숫자형 표현으로 변환하기 위한 직관적인(straightforward) 옵션은 수준(levels)에서 가변수(dummy variables)를 만드는 것이지만, 이 옵션은 카디널리티가 높은(수준이 너무 많은) 변수가 있거나 예측 시점에 참신한(novel) 값(새로운 수준)을 볼 수 있는 경우에는 잘 작동하지 않습니다. 이러한 상황에서 한 가지 옵션은 결과를 사용하는 지도 인코딩 방법인 *효과 인코딩(effect encodings)*을 만드는 것입니다. 효과 인코딩은 범주를 풀링(pooling)하거나 풀링하지 않고 학습할 수 있습니다. 또 다른 옵션은 _해싱(hashing)_ 함수를 사용하여 카테고리 수준을 더 작고 새로운 가변수 세트에 매핑하는 것입니다. 피처 해싱은 빠르고 메모리 사용 공간(footprint)이 적습니다. 다른 옵션에는 엔티티 임베딩(신경망을 통해 학습) 및 증거 가중치(weight of evidence) 변환이 포함됩니다.

대부분의 모델 알고리즘은 범주형 변수에 대해 이러한 유형의 변환 또는 인코딩이 필요합니다. 트리 및 규칙을 기반으로 하는 모델을 포함한 소수의 모델은 범주형 변수를 기본적으로(natively) 처리할 수 있으며 이러한 인코딩이 필요하지 않습니다.

<sup>[1](ch17.xhtml#idm45881849589504-marker)</sup> 이는 범주형 변수가 종종 빨간색, 파란색 및 초록색을 나타내는 `0, 1, 2`와 같이 정수만으로 직접 표현되는 Python의 통계 모델링과 대조(contrast)됩니다.
