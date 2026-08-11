# 11장. 리샘플링을 사용한 모델 비교 (Comparing Models with Resampling)

우리가 두 개 이상의 모델을 만들고 나면 다음 단계는 어느 것이 가장 좋은지 이해하기 위해 이들을 비교하는 것입니다. 어떤 경우 비교는 동일한 모델을 다른 피처(features)나 전처리 방법으로 평가하는 _모델 내(within-model)_ 비교일 수 있습니다. 대안적으로 [10장](ch10.xhtml#resampling)에서 선형 회귀와 랜덤 포레스트 모델을 비교했을 때와 같은 _모델 간(between-model)_ 비교가 더 일반적인 시나리오입니다.

어느 경우든 결과는 각 모델에 대해 리샘플링된 요약 통계(RMSE, 정확도 등)의 모음(collection)입니다. 이 장에서는 먼저 워크플로 세트(workflow sets)를 사용하여 여러 모델을 피팅하는 방법을 시연하겠습니다. 그런 다음 리샘플링 통계의 중요한 측면에 대해 논의하겠습니다. 마지막으로 (가설 검정 또는 베이지안 접근 방식을 사용하여) 모델을 공식적으로 비교하는 방법을 살펴보겠습니다.

# 워크플로 세트를 사용하여 다중 모델 생성하기 (Creating Multiple Models with Workflow Sets)

[7장](ch07.xhtml#workflows)에서는 다른 전처리기(preprocessors) 및/또는 모델을 조합하여(combinatorially) 생성할 수 있는 워크플로 세트의 개념을 설명했습니다. [10장](ch10.xhtml#resampling)에서는 상호작용 항(interaction term)과 경도(longitude) 및 위도(latitude)에 대한 스플라인 함수(spline functions)가 포함된 Ames 데이터에 대한 레시피를 사용했습니다. 워크플로 세트로 더 많은 것을 시연하기 위해 이러한 전처리 단계를 점진적으로(incrementally) 추가하는 세 가지 다른 선형 모델을 만들어 보겠습니다. 우리는 이러한 추가 항(terms)이 모델 결과를 향상시키는지 테스트할 수 있습니다. 세 가지 레시피를 만든 다음 워크플로 세트로 결합하겠습니다.

```
library(tidymodels)
tidymodels_prefer()

basic_rec <-
  recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
           Latitude + Longitude, data = ames_train) %>%
  step_log(Gr_Liv_Area, base = 10) %>%
  step_other(Neighborhood, threshold = 0.01) %>%
  step_dummy(all_nominal_predictors())

interaction_rec <-
  basic_rec %>%
  step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type_") )

spline_rec <-
  interaction_rec %>%
  step_ns(Latitude, Longitude, deg_free = 50)

preproc <-
  list(basic = basic_rec,
       interact = interaction_rec,
       splines = spline_rec
  )

lm_models <- workflow_set(preproc, list(lm = linear_reg()), cross = FALSE)
lm_models
#> # A workflow set/tibble: 3 × 4
#>   wflow_id    info             option    result
#>   <chr>       <list>           <list>    <list>
#> 1 basic_lm    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 interact_lm <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 splines_lm  <tibble [1 × 4]> <opts[0]> <list [0]>
```

이러한 각 모델을 차례로 리샘플링하고 싶습니다. 그렇게 하기 위해 `workflow_map()`이라는 purrr와 유사한 함수를 사용할 것입니다. 이 함수는 워크플로에 적용할 함수의 초기 인수를 사용하고 이어서 해당 함수에 대한 옵션을 사용합니다. 또한 진행률(progress)을 인쇄하는 `verbose` 인수와 각 모델이 다른 모델과 동일한 난수 스트림을 사용하도록 하는 `seed` 인수를 설정합니다.

```
lm_models <-
  lm_models %>%
  workflow_map("fit_resamples",
               # `workflow_map()`에 대한 옵션:
               seed = 1101, verbose = TRUE,
               # `fit_resamples()`에 대한 옵션:
               resamples = ames_folds, control = keep_pred)
#> i 1 of 3 resampling: basic_lm
#> ✓ 1 of 3 resampling: basic_lm (766ms)
#> i 2 of 3 resampling: interact_lm
#> ✓ 2 of 3 resampling: interact_lm (825ms)
#> i 3 of 3 resampling: splines_lm
#> ✓ 3 of 3 resampling: splines_lm (920ms)
lm_models
#> # A workflow set/tibble: 3 × 4
#>   wflow_id    info             option    result
#>   <chr>       <list>           <list>    <list>
#> 1 basic_lm    <tibble [1 × 4]> <opts[2]> <rsmp[+]>
#> 2 interact_lm <tibble [1 × 4]> <opts[2]> <rsmp[+]>
#> 3 splines_lm  <tibble [1 × 4]> <opts[2]> <rsmp[+]>
```

이제 `option` 및 `result` 열이 채워졌음에 주목하십시오. 전자(former)는 주어진(재현성을 위해) `fit_resamples()`에 대한 옵션을 포함하고 후자(latter) 열은 `fit_resamples()`에 의해 생성된 결과를 포함합니다.

워크플로 세트에는 성능 통계를 대조(collate)하는 `collect_metrics()`를 포함하여 몇 가지 편의 함수가 있습니다. 관심 있는 특정 지표로 `filter()`할 수 있습니다.

```
collect_metrics(lm_models) %>%
  filter(.metric == "rmse")
#> # A tibble: 3 × 9
#>   wflow_id    .config          preproc model .metric .estimator   mean     n std_err
#>   <chr>       <chr>            <chr>   <chr> <chr>   <chr>       <dbl> <int>   <dbl>
#> 1 basic_lm    Preprocessor1_M… recipe  line… rmse    standard   0.0803    10 0.00264
#> 2 interact_lm Preprocessor1_M… recipe  line… rmse    standard   0.0799    10 0.00272
#> 3 splines_lm  Preprocessor1_M… recipe  line… rmse    standard   0.0785    10 0.00282
```

이전 장의 랜덤 포레스트 모델은 어떨까요? 먼저 그것을 자체 워크플로 세트로 변환한 다음 행을 결합하여(binding rows) 세트에 추가할 수 있습니다. 이를 위해서는 모델을 리샘플링할 때 제어(control) 함수에서 `save_workflow = TRUE` 옵션이 설정되어 있어야 합니다.

```
four_models <-
  as_workflow_set(random_forest = rf_res) %>%
  bind_rows(lm_models)
four_models
#> # A workflow set/tibble: 4 × 4
#>   wflow_id      info             option    result
#>   <chr>         <list>           <list>    <list>
#> 1 random_forest <tibble [1 × 4]> <opts[0]> <rsmp[+]>
#> 2 basic_lm      <tibble [1 × 4]> <opts[2]> <rsmp[+]>
#> 3 interact_lm   <tibble [1 × 4]> <opts[2]> <rsmp[+]>
#> 4 splines_lm    <tibble [1 × 4]> <opts[2]> <rsmp[+]>
```

[그림 11-1](#workflow-set-r-squared)에 출력된 `autoplot()` 메서드는 최고에서 최악의 순서로 각 모델에 대한 신뢰 구간(confidence intervals)을 보여줍니다. 이 장에서는 결정 계수(coefficient of determination, 일명 _R_<sup>2</sup>)에 중점을 두고 플롯을 설정하기 위해 호출에서 `metric = "rsq"`를 사용할 것입니다.

```
library(ggrepel)
autoplot(four_models, metric = "rsq") +
  geom_text_repel(aes(label = wflow_id), nudge_x = 1/8, nudge_y = 1/100) +
  theme(legend.position = "none")
```

이 _R_<sup>2</sup> 신뢰 구간 플롯에서 랜덤 포레스트 방법이 가장 잘 수행하고 있으며 레시피 단계를 더 많이 추가함에 따라 선형 모델이 약간 개선(minor improvements)됨을 알 수 있습니다.

이제 네 가지 모델 각각에 대해 10개의 리샘플링된 성능 추정치가 있으므로 이러한 요약 통계를 사용하여 모델 간(between-model) 비교를 수행할 수 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1101.png" alt="tmwr 1101" />
<h6 id="figure-11-1.-confidence-intervals-for-the-coefficient-of-determination-using-four-different-models.">그림 11-1. 네 가지 다른 모델을 사용한 결정 계수의 신뢰 구간.</h6>
</figure>

# 리샘플링된 성능 통계 비교 (Comparing Resampled Performance Statistics)

세 가지 선형 모델에 대한 앞의 결과들을 고려할 때 추가 항(terms)이 선형 모델의 평균 RMSE 또는 _R_<sup>2</sup> 통계를 근본적으로(profoundly) 향상시키지 않는 것으로 보입니다. 차이는 작지만 시스템의 실험적 노이즈보다 클 수 있습니다, 즉 통계적으로 유의미한(statistically significant) 것으로 간주될 수 있습니다. 우리는 추가 항이 _R_<sup>2</sup>를 증가시킨다는 가설(hypothesis)을 공식적으로 테스트할 수 있습니다.

###### 참고 (Note)

모델 간(between-model) 비교를 수행하기 전에 리샘플링 통계에 대한 리샘플 내 상관관계(within-resample correlation)를 논의하는 것이 중요합니다. 각 모델은 동일한 교차 검증 폴드로 측정되었으며, 동일한 리샘플에 대한 결과는 비슷한 경향이 있습니다.

즉, 모델 전반에 걸쳐 성능이 낮은 경향이 있는 리샘플이 있고 성능이 높은 경향이 있는 리샘플이 있습니다. 통계에서 이것을 변동(variation)의 _리샘플 간(resample-to-resample)_ 구성 요소라고 합니다.

설명을 위해 선형 모델과 랜덤 포레스트에 대한 개별 리샘플링 통계를 수집해 보겠습니다. 각 집에 대해 관측된 판매 가격과 예측된 판매 가격 간의 상관관계를 측정하는 각 모델의 _R_<sup>2</sup> 통계에 중점을 둘 것입니다. _R_<sup>2</sup> 지표만 유지하도록 `filter()`하고 결과를 재구성(reshape)한 다음 지표가 서로 어떻게 상관되어 있는지(correlated) 계산해 보겠습니다.

```
rsq_indiv_estimates <-
  collect_metrics(four_models, summarize = FALSE) %>%
  filter(.metric == "rsq")

rsq_wider <-
  rsq_indiv_estimates %>%
  select(wflow_id, .estimate, id) %>%
  pivot_wider(id_cols = "id", names_from = "wflow_id", values_from = ".estimate")

corrr::correlate(rsq_wider %>% select(-id), quiet = TRUE)
#> # A tibble: 4 × 5
#>   term          random_forest basic_lm interact_lm splines_lm
#>   <chr>                 <dbl>    <dbl>       <dbl>      <dbl>
#> 1 random_forest        NA        0.887       0.888      0.889
#> 2 basic_lm              0.887   NA           0.993      0.997
#> 3 interact_lm           0.888    0.993      NA          0.987
#> 4 splines_lm            0.889    0.997       0.987     NA
```

이러한 상관관계는 높으며(high) 모델 전체에 걸쳐 큰 리샘플 내 상관관계(within-resample correlations)가 있음을 나타냅니다. [그림 11-2](#rsquared-resamples)에서 이를 시각적으로 확인하기 위해 리샘플을 연결하는 선과 함께 각 모델의 _R_<sup>2</sup> 통계가 표시되어 있습니다.

```
rsq_indiv_estimates %>%
  mutate(wflow_id = reorder(wflow_id, .estimate)) %>%
  ggplot(aes(x = wflow_id, y = .estimate, group = id, color = id, lty = id)) +
  geom_line(alpha = .8, lwd = 1.25) +
  theme(legend.position = "none")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1102.png" alt="tmwr 1102" />
<h6 id="figure-11-2.-resample-statistics-across-models.">그림 11-2. 모델 전반에 걸친 리샘플 통계.</h6>
</figure>

리샘플 간 효과(resample-to-resample effect)가 실제가 아니라면 평행선이 없을 것입니다. 상관관계에 대한 통계적 테스트는 이러한 상관관계의 크기(magnitudes)가 단순히 노이즈가 아닌지를 평가합니다. 선형 모델의 경우:

```
rsq_wider %>%
  with( cor.test(basic_lm, splines_lm) ) %>%
  tidy() %>%
  select(estimate, starts_with("conf"))
#> # A tibble: 1 × 3
#>   estimate conf.low conf.high
#>      <dbl>    <dbl>     <dbl>
#> 1    0.997    0.987     0.999
```

상관관계 테스트 결과(상관관계 `estimate` 및 신뢰 구간)는 리샘플 내 상관관계(within-resample correlation)가 실제인 것으로 보임을 보여줍니다.

추가적인 상관관계는 분석에 어떤 영향을 미칠까요? 두 변수 차이의 분산(variance)을 고려해 보겠습니다.

```math
\operatorname{Var}\lbrack X - Y\rbrack = \operatorname{Var}\lbrack X\rbrack + \operatorname{Var}\lbrack Y\rbrack - 2\operatorname{Cov}\lbrack X,Y\rbrack
```

마지막 항은 두 항목 간의 공분산(covariance)입니다. 유의미한(significant) 양의 공분산이 있는 경우 이 차이에 대한 통계적 테스트는 두 모델의 차이를 비교할 때 결정적으로 힘이 부족할(critically under-powered) 수 있습니다. 즉 리샘플 간 효과를 무시하면 모델 비교 시 모델 간 차이가 없다고 편향되게 판단할(bias our model comparisons toward finding no differences) 것입니다.

###### 경고 (Warning)

리샘플링 통계의 이러한 특성은 다음 두 섹션에서 작용할 것입니다(come into play).

모델을 비교하거나 리샘플링 결과를 살펴보기 전에 관련된 *실질적 효과 크기(practical effect size)*를 정의하는 것이 도움이 될 수 있습니다. 이러한 분석은 _R_<sup>2</sup> 통계에 중점을 두기 때문에 실질적 효과 크기는 중요하게 생각되는(matters) 현실적인(realistic) 차이로 간주하는 _R_<sup>2</sup>의 변화량(change)입니다. 예를 들어 두 모델의 _R_<sup>2</sup> 값이 $`\pm 2`$% 이내이면 두 모델이 실질적으로(practically) 다르지 않다고 생각할 수 있습니다. 이 경우 2%보다 작은 차이는 통계적으로 유의하더라도 중요하지 않은 것으로 간주(deemed)됩니다.

실질적 유의성(Practical significance)은 주관적(subjective)입니다; 중요성에 대한 임계값(threshold)에 대해 두 사람은 매우 다른 생각을 가질 수 있습니다. 그러나 나중에 모델을 결정할 때 이러한 고려 사항이 매우 도움이 될 수 있음을 보여드리겠습니다.

# 단순 가설 검정 방법 (Simple Hypothesis Testing Methods)

단순 가설 검정(simple hypothesis testing)을 사용하여 모델을 공식적으로(formally) 비교할 수 있습니다. 친숙한 선형 통계 모델을 생각해 봅시다.

```math
y_{ij} = \beta_{0} + \beta_{1}x_{i1} + ... + \beta_{p}x_{ip} + \epsilon_{ij}
```

이 다목적(versatile) 모델은 회귀 모델을 생성하는 단 데 사용될 뿐만 아니라 그룹을 비교하기 위해 널리 사용되는 분산 분석(analysis of variance, ANOVA) 기법의 기반이 되기도 합니다. ANOVA 모델에서 예측 변수($`x_{ij}`$)는 서로 다른 그룹에 대한 이진(binary) 더미 변수(dummy variables)입니다. 이로부터 $`\beta`$ 매개변수(parameters)는 가설 검정 기술을 사용하여 두 개 이상의 그룹이 서로 다른지 여부를 추정(estimate)합니다.

우리 특정 상황에서는 ANOVA가 모델 비교도 할 수 있습니다. 리샘플링된 개별 _R_<sup>2</sup> 통계가 ANOVA 모델에서 _결과 데이터(outcome data)_ (즉, $`y_{ij}`$) 역할을 하고 모델들이 _예측 변수(predictors)_ 역할을 한다고 가정해 보겠습니다. 이 데이터 구조의 표본 추출(sampling)이 [표 11-1](#model-anova-data)에 나와 있습니다.

| Y = rsq | model         | X1  | X2  | X3  | id     |
| ------- | ------------- | --- | --- | --- | ------ |
| 0.8108  | basic_lm      | 0   | 0   | 0   | Fold01 |
| 0.8134  | interact_lm   | 1   | 0   | 0   | Fold01 |
| 0.8615  | random_forest | 0   | 1   | 0   | Fold01 |
| 0.8217  | splines_lm    | 0   | 0   | 1   | Fold01 |
| 0.8045  | basic_lm      | 0   | 0   | 0   | Fold02 |
| 0.8103  | interact_lm   | 1   | 0   | 0   | Fold02 |

표 11-1. 분석을 위한 데이터 세트로서의 모델 성능 통계. {#model-anova-data}

테이블의 `X1`, `X2`, `X3` 열은 `model` 열의 값에 대한 지표(indicators)입니다. 이들의 순서는 R이 이를 정의하는 방식과 동일하게, `model`별로 알파벳 순서(alphabetically)로 정의되었습니다.

우리의 모델 비교를 위한 특정 ANOVA 모델은 다음과 같습니다.

```math
y_{ij} = \beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2} + \beta_{3}x_{i3} + \epsilon_{ij}
```

여기서

- $`\beta_{0}`$은 (즉, 스플라인이나 상호 작용이 없는) 기본 선형 모델의 평균 _R_<sup>2</sup> 통계 추정치입니다.

- $`\beta_{1}`$은 상호 작용이 기본 선형 모델에 추가될 때 평균 _R_<sup>2</sup>의 변화(change)입니다.

- $`\beta_{2}`$는 기본 선형 모델과 랜덤 포레스트 모델 간의 평균 _R_<sup>2</sup>의 변화입니다.

- $`\beta_{3}`$은 기본 선형 모델과 상호 작용 및 스플라인이 있는 모델 간의 평균 _R_<sup>2</sup>의 변화입니다.

이러한 모델 매개변수로부터 통계적으로 모델을 비교하기 위한 가설 검정과 p-값이 생성되지만 우리는 리샘플 간 효과(resample-to-resample effect)를 처리하는 방법과 씨름해야 합니다(contend with). 역사적으로 리샘플 그룹은 *블록 효과(block effect)*로 간주되었으며 모델에 적절한 항이 추가되었습니다. 또는 이 특정 리샘플들이 가능한 더 큰 리샘플 모집단에서 무작위로 추출되었다는 *무작위 효과(random effect)*로 간주될 수도 있습니다. 그러나 우리는 이러한 효과(effects)에 실제로 관심이 있는 것이 아닙니다; 관심 있는 차이의 분산이 제대로 추정될 수 있도록 모델에서 이를 조정하기만(adjust for) 원할 뿐입니다.

리샘플을 무작위 효과로 취급하는 것은 이론적으로 매력적입니다. 이런 유형의 무작위 효과가 포함된 ANOVA 모델을 피팅하는 방법에는 선형 혼합 모델(linear mixed model) (Faraway 2016)이나 (다음 섹션에 나오는) 베이지안 계층 모델(Bayesian hierarchical model)이 포함될 수 있습니다.

한 번에 두 모델을 비교하는 간단하고 빠른 방법은 ANOVA 모델에서 _R_<sup>2</sup> 값의 차이를 결과 데이터로 사용하는 것입니다. 결과(outcomes)는 리샘플별로 짝지어지기 때문에(matched) 차이에는 리샘플 간 효과가 포함되지 않으며 이러한 이유로 표준 ANOVA 모델이 적절합니다. 설명을 위해 `lm()`에 대한 이 호출(call)은 두 선형 회귀 모델 간의 차이를 테스트합니다.

```
compare_lm <-
  rsq_wider %>%
  mutate(difference = splines_lm - basic_lm)

lm(difference ~ 1, data = compare_lm) %>%
```

tidy(conf.int = TRUE) %>%
select(estimate, p.value, starts_with("conf"))
#> # A tibble: 1 × 4
#> estimate p.value conf.low conf.high
#> <dbl> <dbl> <dbl> <dbl>
#> 1 0.00913 0.0000256 0.00650 0.0118

# 또는 대응 표본(paired) t-검정을 사용할 수도 있습니다.

rsq_wider %>%
with( t.test(splines_lm, basic_lm, paired = TRUE) ) %>%
tidy() %>%
select(estimate, p.value, starts_with("conf"))
#> # A tibble: 1 × 4
#> estimate p.value conf.low conf.high
#> <dbl> <dbl> <dbl> <dbl>
#> 1 0.00913 0.0000256 0.00650 0.0118

````

이런 방식으로 각 쌍별(pair-wise) 차이를 평가할 수 있습니다. p-값이 *통계적으로 유의미한(statistically significant)* 신호를 나타낸다는 점에 유의하십시오; 경도와 위도에 대한 스플라인 항 모음이 효과가 있는 것으로 나타납니다. 그러나 *R*<sup>2</sup>의 차이는 0.91%로 추정됩니다. 만약 우리의 실질적 효과 크기(practical effect size)가 2%라면, 우리는 이 항들을 모델에 포함할 가치가 없다고 생각할 수 있습니다.

###### 참고 (Note)

앞서 p-값에 대해 간략히 언급했지만 실제로 p-값은 무엇일까요? Wasserstein and Lazar (2016)에 따르면: "비공식적으로, p-값은 지정된 통계 모델 하에서 데이터의 통계적 요약(비교된 두 그룹 간의 표본 평균 차이)이 관측된 값과 같거나 더 극단적(extreme)일 확률입니다."

다시 말해, 만약 이 분석이 차이가 없다는 귀무 가설(null hypothesis) 하에서 많이 반복된다면, p-값은 우리의 관찰된 결과가 이에 비해 얼마나 극단적일지를 반영합니다.

# 베이지안 방법 (Bayesian Methods)

우리는 방금 가설 검정을 사용하여 공식적으로 모델을 비교했지만 무작위 효과 및 베이지안 통계(McElreath 2020)를 사용하여 이러한 공식 비교를 수행하는 보다 일반적인 접근 방식을 취할 수도 있습니다. 모델이 ANOVA 방법보다 더 복잡하긴 하지만 해석은 p-값 접근 방식보다 더 간단하고 이해하기 쉽습니다(straightforward). 이전 ANOVA 모델의 형태는 다음과 같습니다.

``` math
y_{ij} = \beta_{0} + \beta_{1}x_{i1} + \beta_{2}x_{i2} + \beta_{3}x_{i3} + \epsilon_{ij}
````

여기서 잔차(residuals) $`\epsilon_{ij}`$는 독립적이며 평균이 0이고 일정한 표준 편차 $`\sigma`$를 갖는 가우스 분포(Gaussian distribution, 정규 분포)를 따른다고 가정합니다. 이러한 가정으로부터, 통계 이론은 추정된 회귀 매개변수(regression parameters)가 다변량 가우스 분포를 따름을 보여주며 이를 통해 p-값과 신뢰 구간이 도출(derived)됩니다.

베이지안 선형 모델은 추가적인 가정을 합니다. 잔차에 대한 분포를 지정하는 것 외에도 모델 매개변수($`\beta_{j}`$ 및 $`\sigma`$)에 대한 _사전 분포(prior distribution)_ 지정이 필요합니다. 이것들은 모델이 관측 데이터에 노출되기 전에 가정하는 매개변수에 대한 분포입니다. 예를 들어, 우리 모델에 대한 사전 분포의 간단한 세트는 다음과 같을 수 있습니다.

```math
\begin{aligned}
\epsilon_{ij} & {\sim N(0,\sigma)} \\
\beta_{j} & {\sim N(0,10)} \\
\sigma & {\sim \text{exponential}(1)}
\end{aligned}
```

이러한 사전 확률(priors)은 모델 매개변수의 가능하거나 유력한 범위를 설정하며 알 수 없는(unknown) 매개변수는 갖지 않습니다. 예를 들어, $`\sigma`$에 대한 사전 확률은 값이 0보다 커야 하고 매우 오른쪽으로 꼬리가 긴(right-skewed) 형태이며 일반적으로 3 또는 4 미만임을 나타냅니다.

회귀 매개변수는 10의 표준 편차로 다소 넓은 사전 분포를 갖는다는 점에 유의하십시오. 많은 경우, 대칭이고 종 모양(bell shaped)이라는 것 이상으로 사전 확률에 대한 강한 의견(opinion)을 가지고 있지 않을 수 있습니다. 큰 표준 편차는 상당히 무정보적인(uninformative) 사전 확률을 의미합니다; 매개변수가 취할 수 있는 가능한 값 측면에서 지나치게 제한적이지 않습니다. 이것은 데이터가 매개변수 추정 중에 더 많은 영향을 미치도록 허용합니다.

관측된 데이터와 사전 분포 지정이 주어지면, 모델 매개변수를 추정할 수 있습니다. 모델 매개변수의 최종 분포는 사전 확률과 우도 추정치(likelihood estimates)의 조합입니다. 이러한 매개변수의 *사후 분포(posterior distributions)*가 주요 관심 분포입니다. 이들은 모델의 추정된 매개변수에 대한 완전한 확률적(probabilistic) 설명(description)입니다.

## 무작위 절편 모델 (A Random Intercept Model)

리샘플이 적절하게 모델링되도록 베이지안 ANOVA 모델을 조정하기 위해 우리는 *무작위 절편 모델(random intercept model)*을 고려합니다. 여기서 우리는 리샘플이 절편(intercept)만을 변경함으로써 모델에 영향을 미친다고 가정합니다. 이는 리샘플이 회귀 매개변수 $`\beta_{j}`$에 차등적(differential) 영향을 미치지 못하도록 제한합니다; 이들은 리샘플 전반에 걸쳐 동일한 관계를 갖는 것으로 가정됩니다. 이 모델 방정식은 다음과 같습니다.

```math
y_{ij} = \left( \beta_{0} + b_{i} \right) + \beta_{1}x_{i1} + \beta_{2}x_{i2} + \beta_{3}x_{i3} + \epsilon_{ij}
```

이는 리샘플링된 통계에 대해 무리한(unreasonable) 모델이 아니며, [그림 11-2](#rsquared-resamples)처럼 모델들에 걸쳐 그려볼 때 모델 전반에 걸쳐 상당히 평행한 효과를 갖는 경향이 있습니다(즉, 선의 교차(crossover)가 거의 없음).

이 모델 구성을 위해 무작위 효과의 사전 분포에 대한 추가 가정이 이루어집니다. 이 분포에 대한 합리적인 가정은 또 다른 종 모양 곡선과 같은 대칭 분포입니다. 요약 통계 데이터에서 10이라는 유효 표본 크기(effective sample size)를 감안할 때 표준 정규 분포보다 넓은 사전 확률을 사용해 보겠습니다. 우리는 자유도(degree of freedom)가 하나인 _t_-분포(즉, $`b_{i} \sim t(1)`$)를 사용할 것이며, 이는 유사한 가우스 분포보다 꼬리가 더 두껍습니다(heavier tails).

tidyposterior 패키지에는 리샘플링된 모델을 비교할 목적으로 이러한 베이지안 모델을 피팅하는 함수가 있습니다. 기본 함수는 `perf_mod()`라고 하며 다음과 같은 다양한 유형의 객체에 대해 "그냥 작동(just work)"하도록 구성되어 있습니다.

- 워크플로 세트의 경우, 그룹이 워크플로에 대응하는 ANOVA 모델을 생성합니다. 기존 모델에서는 튜닝 매개변수를 최적화하지 않았습니다(다음 세 장 참조). 세트의 워크플로 중 하나에 튜닝 매개변수에 대한 데이터가 있는 경우 각 워크플로에 대해 가장 좋은 튜닝 매개변수 세트가 베이지안 분석에 사용됩니다. 즉, 튜닝 매개변수가 있음에도 불구하고 `perf_mod()`는 *워크플로 간(between-workflow) 비교*를 수행하는 데 초점을 맞춥니다.

- 리샘플링을 사용하여 튜닝된 단일 모델이 포함된 객체의 경우 `perf_mod()`는 *모델 내(within-model) 비교*를 수행합니다. 이 상황에서 베이지안 ANOVA 모델에서 테스트된 그룹화(grouping) 변수는 튜닝 매개변수로 정의된 하위 모델(submodels)입니다.

- `perf_mod()` 함수는 둘 이상의 모델/워크플로 결과와 연관된 성능 지표 열이 있는 rsample로 생성된 데이터 프레임도 사용할 수 있습니다. 이들은 비표준 수단에 의해 생성되었을 수도 있습니다.

이러한 유형의 모든 객체에서 `perf_mod()` 함수는 적절한 베이지안 모델을 결정하고 리샘플링 통계로 그 모델을 피팅합니다. 우리 예제의 경우 워크플로와 연관된 네 세트의 _R_<sup>2</sup> 통계를 모델링합니다.

tidyposterior 패키지는 rstanarm 패키지를 통해 모델을 지정하고 피팅하는 데 [Stan 소프트웨어](https://mc-stan.org)를 사용합니다. 해당 패키지 내의 함수에는 기본 사전 확률이 있습니다(자세한 내용은 `?priors` 참조). 다음 모델은 (_t_-분포를 따르는) 무작위 절편을 제외한 모든 매개변수에 기본 사전 확률을 사용합니다. 추정 프로세스에서는 난수를 사용하므로 함수 호출 내에서 시드(seed)가 설정됩니다. 추정 프로세스는 반복적이며 *체인(chains)*이라고 하는 모음(collections)에서 여러 번 복제(replicated)됩니다. `iter` 매개변수는 함수에게 각 체인에서 추정 프로세스를 얼마나 오래 실행할지 알려줍니다. 여러 체인이 사용되는 경우 그 결과가 결합됩니다(진단 평가(diagnostic assessments)에 의해 검증(validated)되었다고 가정):

```
library(tidyposterior)
library(rstanarm)

# rstanarm 패키지는 엄청난(copious) 양의 출력을 생성합니다; 그러한 결과들은
# 여기에 표시되지 않지만 잠재적인 문제에 대해 검사(inspecting)할 가치가 있습니다.
# 로깅을 제거하기 위해 `refresh = 0` 옵션을 사용할 수 있습니다.
rsq_anova <-
  perf_mod(
    four_models,
    metric = "rsq",
    prior_intercept = rstanarm::student_t(df = 1),
    chains = 4,
    iter = 5000,
    seed = 1102
  )
```

생성된 객체에는 내부(element 내부에 `stan`이라는 이름으로 있음)에 포함된 Stan 객체뿐만 아니라 리샘플링 프로세스에 대한 정보도 포함되어 있습니다. 우리는 회귀 매개변수의 사후 분포에 가장 관심이 있습니다. tidyposterior 패키지에는 이러한 사후 분포를 티블로 추출하는 `tidy()` 메서드가 있습니다.

```
model_post <-
  rsq_anova %>%
  # 사후 분포에서 무작위 샘플을 가져옵니다.
  # 재현이 가능하도록 시드를 다시 설정하십시오.
  tidy(seed = 1103)

glimpse(model_post)
#> Rows: 40,000
#> Columns: 2
#> $ model     <chr> "random_forest", "random_forest", "random_forest", …
#> $ posterior <dbl> 0.8293, 0.8238, 0.8276, 0.8209, 0.8213, 0.8132, 0.8241, …
```

네 가지 사후 분포가 [그림 11-3](#four-posteriors)에 시각화되어 있습니다.

```
model_post %>%
  mutate(model = forcats::fct_inorder(model)) %>%
  ggplot(aes(x = posterior)) +
  geom_histogram(bins = 50, color = "white", fill = "blue", alpha = 0.4) +
  facet_wrap(~ model, ncol = 1)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1103.png" alt="tmwr 1103" />
<h6 id="figure-11-3.-posterior-distributions-for-the-coefficient-of-determination-using-four-different-models.">그림 11-3. 네 가지 다른 모델을 사용한 결정 계수의 사후 분포.</h6>
</figure>

이 히스토그램은 각 모델에 대한 평균 _R_<sup>2</sup> 값의 추정된 확률 분포를 설명합니다. 특히 세 가지 선형 모델의 경우 약간의 겹침(overlap)이 있습니다.

[그림 11-4](#credible-intervals)에 표시된 모델 결과를 위한 기본 `autoplot()` 메서드와 함께, 오버레이된 밀도 플롯(overlaid density plots)을 보여주는 깔끔하게 정리된(tidied) 객체도 있습니다.

```
autoplot(rsq_anova) +
  geom_text_repel(aes(label = workflow), nudge_x = 1/8, nudge_y = 1/100) +
  theme(legend.position = "none")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1104.png" alt="tmwr 1104" />
<h6 id="figure-11-4.-credible-intervals-derived-from-the-model-posterior-distributions.">그림 11-4. 모델 사후 분포에서 파생된 신뢰 구간(credible intervals).</h6>
</figure>

베이지안 모델에서 리샘플링을 사용할 때의 놀라운(wonderful) 측면 중 하나는, 일단 매개변수에 대한 사후 확률이 있으면 매개변수 조합에 대한 사후 분포를 얻는 것이 아주 쉽다는(trivial) 것입니다. 예를 들어 두 선형 회귀 모델을 비교하기 위해 우리는 평균의 차이에 관심이 있습니다. 이 차이의 사후 확률은 개별 사후 확률에서 샘플링을 하고 그 차이를 취(taking the differences)함으로써 계산됩니다. `contrast_models()` 함수가 이를 수행할 수 있습니다. 비교 대상을 지정하기 위해 `list_1` 및 `list_2` 매개변수는 문자열 벡터(character vectors)를 취하고 해당 리스트의 모델 간의 차이를 계산합니다(`list_1 - list_2`로 매개변수화(parameterized)됨).

두 선형 모델을 비교하고 [그림 11-5](#posterior-difference)에서 그 결과를 시각화할 수 있습니다.

```
rqs_diff <-
  contrast_models(rsq_anova,
                  list_1 = "splines_lm",
                  list_2 = "basic_lm",
                  seed = 1104)

rqs_diff %>%
  as_tibble() %>%
  ggplot(aes(x = difference)) +
  geom_vline(xintercept = 0, lty = 2) +
  geom_histogram(bins = 50, color = "white", fill = "red", alpha = 0.4)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1105.png" alt="tmwr 1105" />
<h6 id="figure-11-5.-posterior-distribution-for-the-difference-in-the-coefficient-of-determination.">그림 11-5. 결정 계수 차이에 대한 사후 분포.</h6>
</figure>

사후 확률은 분포의 중심이 0보다 크다는 것(스플라인이 있는 모델이 일반적으로 더 큰 값을 가짐을 나타냄)을 보여주지만 어느 정도(to a degree) 0과 겹치기도 합니다. 이 객체에 대한 `summary()` 메서드는 분포의 평균뿐만 아니라 베이지안에서의 신뢰 구간(confidence intervals)과 유사한(analog) 신용 구간(credible intervals)도 계산합니다.

```
summary(rqs_diff) %>%
  select(-starts_with("pract"))
#> # A tibble: 1 × 6
#>   contrast               probability    mean   lower  upper  size
#>   <chr>                        <dbl>   <dbl>   <dbl>  <dbl> <dbl>
#> 1 splines_lm vs basic_lm           1 0.00913 0.00507 0.0131     0
```

`probability` 열은 0보다 큰 사후 확률의 비율을 반영합니다. 이것은 양의(positive) 차이가 실제일 확률입니다. 그 값은 0에 가깝지 않으므로 통계적 유의성, 즉 통계적으로 실제 차이가 0이 아니라는 개념(idea)에 대한 강력한 근거를 제공합니다.

그러나 평균 차이의 추정치는 0에 상당히 가깝습니다. 우리가 앞서 제안한 실질적 효과 크기(practical effect size)가 2%임을 상기하십시오. 사후 분포를 사용하여, 실질적으로 유의미할 확률을 계산할 수도 있습니다. 베이지안 분석에서 이는 _ROPE 추정치(ROPE estimate)_, 즉 실질적 동등성 영역(Region Of Practical Equivalence) (Kruschke and Liddell 2018)입니다. 이를 추정하기 위해 summary 함수에 대한 `size` 옵션이 사용됩니다.

```
summary(rqs_diff, size = 0.02) %>%
  select(contrast, starts_with("pract"))
#> # A tibble: 1 × 4
#>   contrast               pract_neg pract_equiv pract_pos
#>   <chr>                      <dbl>       <dbl>     <dbl>
#> 1 splines_lm vs basic_lm         0           1         0
```

`pract_equiv` 열은 `[-size, size]` 내에 있는 사후 확률의 비율입니다(`pract_neg` 및 `pract_pos` 열은 이 구간의 아래와 위 비율입니다). 이 큰 값은 우리 효과 크기에 대해 두 모델이 실질적으로 동일할 압도적 확률(overwhelming probability)이 있음을 나타냅니다. 비록 이전 플롯에서 차이가 0이 아닐 가능성이 높음을 보여주었음에도 동등성 테스트(equivalence test)는 실질적으로 유의미하지 않을 만큼 그 차이가 충분히 작음을 시사합니다.

동일한 프로세스를 사용하여 랜덤 포레스트 모델을 리샘플링된 선형 회귀 분석 중 하나 또는 두 개 모두와 비교할 수 있습니다. 사실 워크플로 세트와 함께 `perf_mod()`가 사용될 때 `autoplot()` 메서드는 각 워크플로를 현재 최고(이 경우 랜덤 포레스트 모델)와 비교하는 `pract_equiv` 결과를 보여줄 수 있습니다.

```
autoplot(rsq_anova, type = "ROPE", size = 0.02) +
  geom_text_repel(aes(label = workflow)) +
  theme(legend.position = "none")
```

[그림 11-6](#practical-equivalence)은 2% 실질적 효과 크기를 사용할 때 선형 모델 중 어느 것도 랜덤 포레스트 모델에 근접하지 않음을 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1106.png" alt="tmwr 1106" />
<h6 id="figure-11-6.-probability-of-practical-equivalence-for-an-effect-size-of-2.">그림 11-6. 2%의 효과 크기에 대한 실질적 동등성 확률.</h6>
</figure>

## 리샘플링 양의 효과 (The Effect of the Amount of Resampling)

리샘플 수는 이러한 유형의 공식적인 베이지안 비교에 어떤 영향을 미칠까요? 리샘플 수가 많을수록 전체 리샘플링 추정치의 정밀도(precision)가 높아집니다; 그 정밀도는 이 유형의 분석에 전파됩니다(propagates). 설명을 위해 반복 교차 검증을 사용하여 더 많은 리샘플을 추가했습니다. 사후 분포가 어떻게 변했을까요? [그림 11-7](#intervals-over-replicates)은 (10-겹 교차 검증의 10번 반복에서 생성된) 최대 100개의 리샘플을 사용한 90% 신용 구간을 보여줍니다.<sup><a href="ch11.xhtml#idm45881858984368" id="idm45881858984368-marker" data-type="noteref">1</a></sup>

```
ggplot(intervals,
       aes(x = resamples, y = mean)) +
  geom_path() +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill = "red", alpha = .1) +
  labs(x = "Number of Resamples (repeated 10-fold cross-validation)")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1107.png" alt="tmwr 1107" />
<h6 id="figure-11-7.-probability-of-practical-equivalence-to-the-random-forest-model.">그림 11-7. 랜덤 포레스트 모델과의 실질적 동등성 확률.</h6>
</figure>

리샘플을 더 추가함에 따라 구간의 너비(width)가 줄어듭니다. 분명히 10개의 리샘플에서 30개로 이동하는 것이 80개에서 100개로 이동하는 것보다 영향이 더 큽니다. "다수(large)"의 리샘플을 사용할 경우 수확 체감(diminishing returns)이 있습니다("다수"라는 것은 데이터 세트마다 다를 것입니다).

# 이 장의 요약 (Chapter Summary)

이 장에서는 모델 간 성능의 차이를 테스트하기 위한 공식적인 통계 방법에 대해 설명했습니다. 우리는 동일한 리샘플에 대한 결과가 유사한 경향이 있는 리샘플 내 효과(within-resample effect)를 시연했습니다; 유효한(valid) 모델 비교를 위해 리샘플링된 요약 통계의 이러한 측면에 대해 적절한 분석이 필요합니다. 또한 통계적 유의성과 실질적 유의성 모두 모델 비교를 위한 중요한 개념이지만 두 가지는 서로 다릅니다.

<sup>[1](ch11.xhtml#idm45881858984368-marker)</sup> `intervals`를 생성하는 코드는 [GitHub에서](https://oreil.ly/CmvNU) 사용할 수 있습니다.
