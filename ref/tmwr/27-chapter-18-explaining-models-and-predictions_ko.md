# 18장. 모델 및 예측 설명 (Explaining Models and Predictions)

[1장](ch01.xhtml#software-modeling)에서는 모델의 분류(taxonomy)를 간략하게 설명(outlined)하고 모델이 일반적으로 설명적(descriptive), 추론적(inferential) 또는 예측적(predictive) 중 하나 이상으로 구축(built)된다고 제안했습니다. 우리는 적절한 메트릭(회귀의 경우 RMSE 또는 분류의 경우 ROC 곡선 아래 면적과 같은)으로 측정된 모델 성능이 모든 모델링 애플리케이션에서 중요할 수 있다고 제안했습니다. 이와 유사하게, 모델이 왜 그런 예측을 하는지에 대한 답을 제시하는(answering) 모델 설명(explanations)은 모델의 목적이 주로 설명적이든, 가설을 테스트하려는 것이든, 아니면 예측을 하기 위한 것이든 중요할 수 있습니다. "왜?"라는 질문에 답하면 모델링 실무자(practitioners)는 어떤 피처가 예측에서 중요했는지, 심지어 피처의 다양한 값에서 모델 예측이 어떻게 변하는지 이해할 수 있습니다. 이 장에서는 모델에게 왜 그러한 예측을 하는지 질문하는(ask) 방법을 다룹니다(covers).

선형 회귀와 같은 일부 모델의 경우 모델이 예측을 수행하는 이유를 설명하는 방법이 대개(usually) 명확(clear)합니다. 선형 모델의 구조에는 해석하기 직관적인(straightforward) 각 예측 변수에 대한 계수(coefficients)가 포함되어 있습니다. 설계상(by design) 비선형 동작(behavior)을 포착(capture)할 수 있는 랜덤 포레스트와 같은 다른 모델의 경우 모델 자체의 구조만으로 모델의 예측을 설명하는 방법이 덜 투명합니다(less transparent). 대신, 모델 설명자(explainer) 알고리즘을 적용하여 예측에 대한 이해를 생성(generate)할 수 있습니다.

###### 참고 (Note)

모델 설명에는 *전역(global)*과 *지역(local)*의 두 가지 유형이 있습니다. 전역 모델 설명은 전체 관측치 세트에 대해 집계된(aggregated) 전반적인 이해를 제공합니다; 지역 모델 설명은 단일 관측치에 대한 예측에 대한 정보를 제공합니다.

# 모델 설명을 위한 소프트웨어 (Software for Model Explanations)

tidymodels 프레임워크 자체에는 모델 설명을 위한 소프트웨어가 포함되어 있지 않습니다. 대신, tidymodels로 훈련되고 평가된 모델은 [lime](https://oreil.ly/bzCAq), [vip](https://oreil.ly/UpoQf) 및 [DALEX](https://oreil.ly/KPZLQ)와 같은 R 패키지의 다른 보충(supplementary) 소프트웨어로 설명될 수 있습니다. 우리는 종종 다음을 선택합니다.

- 모델 구조를 활용하고(take advantage of)(흔히 더 빠른) _모델 기반(model-based)_ 방법을 사용하고자 할 때 vip 함수

- 모든 모델에 적용될 수 있는 _모델 불가지론적(model-agnostic)_ 방법을 사용하고자 할 때 DALEX 함수

[10장](ch10.xhtml#resampling)과 [11장](ch11.xhtml#compare)에서는 교호작용이 있는 선형 모델과 랜덤 포레스트 모델을 포함하여 아이오와 주 Ames의 주택 가격을 예측하기 위해 여러 모델을 훈련하고 비교했으며, 결과는 [그림 18-1](#explain-obs-pred)에 나와 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1801.png" alt="tmwr 1801" />
<h6 id="figure-18-1.-comparing-predicted-prices-for-a-linear-model-with-interactions-and-a-random-forest-model.">그림 18-1. 교호작용이 있는 선형 모델과 랜덤 포레스트 모델의 예측 가격 비교.</h6>
</figure>

이 두 모델 모두에 대해 모델 불가지론적 설명자(explainers)를 구축하여 이러한 예측을 하는 이유를 알아보겠습니다. DALEX를 위한 DALEXtra 추가 기능(add-on) 패키지를 사용할 수 있으며, 이는 tidymodels에 대한 지원을 제공합니다. Biecek and Burzykowski (2021)는 모델 설명에 DALEX를 사용하는 방법에 대한 철저한(thorough) 탐구를 제공합니다; 이 장에서는 tidymodels에 특화된 일부 중요한 접근 방식만 요약합니다. DALEX를 사용하여 전역적(global)이든 지역적(local)이든 어떤 종류의 모델 설명을 계산하려면, 먼저 적절한 데이터를 준비한 다음 각 모델에 대한 *설명자(explainer)*를 만듭니다.

```
library(DALEXtra)
vip_features <- c("Neighborhood", "Gr_Liv_Area", "Year_Built",
                  "Bldg_Type", "Latitude", "Longitude")

vip_train <-
  ames_train %>%
  select(all_of(vip_features))

explainer_lm <-
  explain_tidymodels(
    lm_fit,
    data = vip_train,
    y = ames_train$Sale_Price,
    label = "lm + interactions",
    verbose = FALSE
  )

explainer_rf <-
  explain_tidymodels(
    rf_fit,
    data = vip_train,
    y = ames_train$Sale_Price,
    label = "random forest",
    verbose = FALSE
  )
```

###### 경고 (Warning)

선형 모델은 일반적으로 해석하고 설명하기가 직관적(straightforward)입니다; 선형 모델을 위해 별도의 모델 설명 알고리즘을 사용하는 자신을 자주 발견하지는 못할(may not often find yourself using) 것입니다. 그러나 선형 모델이라도 스플라인(splines)과 교호작용 항이 있으면 예측을 이해하거나 설명하기 어려울 수 있습니다!

모델 설명 가능성(explainability) 동안 상당한(significant) 피처 엔지니어링 변환을 다루는 것은 우리가 가진 몇 가지 옵션을 강조(highlights)합니다(또는 때로는 그러한 분석의 모호성(ambiguity)을 강조합니다). 전역 또는 지역 모델 설명은 다음 측면(in terms of)에서 정량화할 수 있습니다.

- 의미 있는 피처 엔지니어링 변환 없이 존재하는 _원래의 기본적인 예측 변수_, 또는

- 이 예제에서처럼 차원 축소([16장](ch16.xhtml#dimensionality)) 또는 교호작용 및 스플라인 항을 통해 생성된 _파생된(Derived) 피처_

# 지역 설명 (Local Explanations)

지역 모델 설명은 단일 관측치에 대한 예측에 관한 정보를 제공합니다. 예를 들어, North Ames 이웃에 있는 오래된 듀플렉스(duplex)([4장](ch04.xhtml#ames))를 고려해 보겠습니다.

```
duplex <- vip_train[120,]
duplex
#> # A tibble: 1 × 6
#>   Neighborhood Gr_Liv_Area Year_Built Bldg_Type Latitude Longitude
#>   <fct>              <dbl>      <dbl> <fct>        <dbl>     <dbl>
#> 1 North_Ames          1040       1949 Duplex        42.0     -93.6
```

모델이 이 듀플렉스에 대해 주어진 가격을 예측하는 이유를 이해하는 데 가능한 접근 방식에는 여러 가지가 있습니다. 하나는 DALEX 함수 `predict_parts()`로 구현되는 분석(break-down) 설명입니다; 개별(individual) 피처에 기인한 기여도(contributions attributed to)가 우리의 듀플렉스와 같은 특정 관측치에 대한 평균 모델의 예측을 어떻게 변화(change)시키는지 계산합니다. 선형 모델의 경우 듀플렉스 상태(`Bldg_Type = 3`),<sup><a href="ch18.xhtml#idm45881848206144" id="idm45881848206144-marker" data-type="noteref">1</a></sup> 크기, 경도 및 연식이 모두 절편(intercept)에서 가격을 떨어뜨리는(driven down) 데 가장 많이 기여합니다(contribute):

```
lm_breakdown <- predict_parts(explainer = explainer_lm, new_observation = duplex)
lm_breakdown
#>                                           contribution
#> lm + interactions: intercept                     5.221
#> lm + interactions: Gr_Liv_Area = 1040           -0.082
#> lm + interactions: Bldg_Type = 3                -0.049
#> lm + interactions: Longitude = -93.608903       -0.043
#> lm + interactions: Year_Built = 1949            -0.039
#> lm + interactions: Latitude = 42.035841         -0.007
#> lm + interactions: Neighborhood = 1              0.001
#> lm + interactions: prediction                    5.002
```

이 선형 모델은 위도와 경도에 대한 스플라인 항을 사용하여 훈련되었으므로 여기에 표시된 `Longitude`의 가격 기여도는 모든 개별 스플라인 항의 효과를 결합합니다. 기여도는 파생된 스플라인 피처가 아니라 원래의 `Longitude` 피처 측면(in terms of)에서입니다.

랜덤 포레스트 모델의 경우 가장 중요한 피처가 약간 다르며, 크기, 연식 및 듀플렉스 상태가 가장 중요합니다.

```
rf_breakdown <- predict_parts(explainer = explainer_rf, new_observation = duplex)
rf_breakdown
#>                                       contribution
#> random forest: intercept                     5.221
#> random forest: Year_Built = 1949            -0.076
#> random forest: Gr_Liv_Area = 1040           -0.075
#> random forest: Bldg_Type = 3                -0.027
#> random forest: Longitude = -93.608903       -0.043
#> random forest: Latitude = 42.035841         -0.028
#> random forest: Neighborhood = 1             -0.003
#> random forest: prediction                    4.969
```

###### 경고 (Warning)

이와 같은 모델 분석 설명은 피처의 *순서(order)*에 의존합니다(depend on).

휴리스틱을 통해 선택된(chosen via a heuristic) 선형 모델에 대한 기본값과 동일하도록(same as the default) 랜덤 포레스트 모델 설명의 `order`를 선택하면 피처의 상대적(relative) 중요도를 변경할 수 있습니다.

```
predict_parts(
  explainer = explainer_rf,
  new_observation = duplex,
  order = lm_breakdown$variable_name
)
#>                                       contribution
#> random forest: intercept                     5.221
#> random forest: Gr_Liv_Area = 1040           -0.075
#> random forest: Bldg_Type = 3                -0.019
#> random forest: Longitude = -93.608903       -0.023
#> random forest: Year_Built = 1949            -0.104
#> random forest: Latitude = 42.035841         -0.028
#> random forest: Neighborhood = 1             -0.003
#> random forest: prediction                    4.969
```

이러한 분석 설명이 순서를 기반으로 변경된다는 사실을 사용하여 모든(또는 많은) 가능한 순서(orderings)에 대해 가장 중요한 피처를 계산할 수 있습니다. 이것이 섀플리 덧셈 설명(Shapley additive explanations, SHAP)(Lundberg and Lee 2017)의 배후에 있는(behind) 아이디어로, 피처의 평균 기여도가 피처 순서 지정의 다양한 조합 또는 "연합(coalitions)" 하에서(under) 계산됩니다. 무작위 순서 지정을 `B = 20`으로 사용하여 듀플렉스에 대한 SHAP 속성(attributions)을 계산해 보겠습니다.

```
set.seed(1801)
shap_duplex <-
  predict_parts(
    explainer = explainer_rf,
    new_observation = duplex,
    type = "shap",
    B = 20
  )
```

`plot(shap_duplex)`를 호출하여 DALEX의 기본 플롯 메서드를 사용하거나 기반이 되는(underlying) 데이터에 액세스하고 커스텀 플롯을 만들 수 있습니다. [그림 18-2](#duplex-rf-shap)의 상자 플롯은 우리가 시도한 모든 순서 지정(orderings)에서 기여도의 분포를 표시하고, 막대(bars)는 각 피처에 대한 평균 속성을 표시합니다.

```
library(forcats)
shap_duplex %>%
  group_by(variable) %>%
  mutate(mean_val = mean(contribution)) %>%
  ungroup() %>%
  mutate(variable = fct_reorder(variable, abs(mean_val))) %>%
  ggplot(aes(contribution, variable, fill = mean_val > 0)) +
  geom_col(data = ~distinct(., variable, mean_val),
           aes(mean_val, variable),
           alpha = 0.5) +
  geom_boxplot(width = 0.5) +
  theme(legend.position = "none") +
  scale_fill_viridis_d() +
  labs(y = NULL)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1802.png" alt="tmwr 1802" />
<h6 id="figure-18-2.-shapley-additive-explanations-from-the-random-forest-model-for-a-duplex-property.">그림 18-2. 듀플렉스 자산에 대한 랜덤 포레스트 모델의 섀플리 덧셈 설명.</h6>
</figure>

데이터 세트의 다른 관측치는 어떨까요? Gilbert 이웃에 있는 더 크고 새로운 1가구 주택(one-family home)을 살펴보겠습니다.

```
big_house <- vip_train[1269,]
big_house
#> # A tibble: 1 × 6
#>   Neighborhood Gr_Liv_Area Year_Built Bldg_Type Latitude Longitude
#>   <fct>              <dbl>      <dbl> <fct>        <dbl>     <dbl>
#> 1 Gilbert             2267       2002 OneFam        42.1     -93.6
```

동일한 방식으로 이 주택에 대한 SHAP 평균 속성을 계산할 수 있습니다.

```
set.seed(1802)
shap_house <-
  predict_parts(
    explainer = explainer_rf,
    new_observation = big_house,
    type = "shap",
    B = 20
  )
```

결과는 [그림 18-3](#gilbert-shap)에 나와 있습니다; 듀플렉스와 달리(unlike) 이 집의 크기와 연식은 집값이 비싸지는(higher price) 데 기여합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1803.png" alt="tmwr 1803" />
<h6 id="figure-18-3.-shapley-additive-explanations-from-the-random-forest-model-for-a-one-family-home-in-gilbert.">그림 18-3. Gilbert의 1가구 주택에 대한 랜덤 포레스트 모델의 섀플리 덧셈 설명.</h6>
</figure>

# 전역 설명 (Global Explanations)

전역 피처 중요도 또는 변수 중요도라고도 하는 전역 모델 설명은 전체 훈련 세트에 대해 집계(aggregated)되어 전체적인 선형 및 랜덤 포레스트 모델의 예측을 주도하는 데 어떤 피처가 가장 중요한지 이해하는 데 도움이 됩니다. 이전 섹션에서 개별 주택의 판매 가격을 예측하는 데 어떤 변수나 피처가 가장 중요한지 다루었다면(addressed), 전역 피처 중요도는 모델 전체에(in aggregate) 대해 가장 중요한 변수를 다룹니다.

###### 참고 (Note)

변수 중요도를 계산하는 한 가지 방법은 피처를 *순열(permute)*하는 것입니다(Breiman 2001a). 우리는 피처의 값을 섞거나(permute or shuffle) 모델에서 예측한 다음, 섞기(shuffling) 전과 비교하여 모델이 데이터에 얼마나 더 잘 맞지 않는지 측정할 수 있습니다.

열을 섞었을 때 모델 성능이 크게 저하(degradation)되면 중요한(important) 것이고, 열의 값을 섞어도 모델 성능에 큰 차이가 없다면 중요한 변수가 아닙니다. 이 접근 방식은 모든 종류의 모델에 적용될 수 있으며(*모델 불가지론적(model agnostic)*임), 결과를 이해하기가 직관적(straightforward)입니다.

DALEX를 사용하여 `model_parts()` 함수를 통해 이러한 종류의 변수 중요도를 계산합니다.

```
set.seed(1803)
vip_lm <- model_parts(explainer_lm, loss_function = loss_root_mean_square)
set.seed(1804)
vip_rf <- model_parts(explainer_rf, loss_function = loss_root_mean_square)
```

다시 말하지만(Again), `plot(vip_lm, vip_rf)`를 호출하여 DALEX의 기본 플롯 메서드를 사용할 수 있지만 기저가 되는(underlying) 데이터를 탐색(exploration), 분석 및 플로팅에 사용할 수 있습니다. 플로팅을 위한 함수를 만들어 보겠습니다.

```
ggplot_imp <- function(...) {
  obj <- list(...)
  metric_name <- attr(obj[[1]], "loss_name")
  metric_lab <- paste(metric_name,
                      "after permutations\n(higher indicates more important)")

  full_vip <- bind_rows(obj) %>%
    filter(variable != "_baseline_")

  perm_vals <- full_vip %>%
    filter(variable == "_full_model_") %>%
    group_by(label) %>%
    summarise(dropout_loss = mean(dropout_loss))

  p <- full_vip %>%
    filter(variable != "_full_model_") %>%
    mutate(variable = fct_reorder(variable, dropout_loss)) %>%
    ggplot(aes(dropout_loss, variable))
  if(length(obj) > 1) {
    p <- p +
      facet_wrap(vars(label)) +
      geom_vline(data = perm_vals, aes(xintercept = dropout_loss, color = label),
                 size = 1.4, lty = 2, alpha = 0.7) +
      geom_boxplot(aes(color = label, fill = label), alpha = 0.2)
  } else {
    p <- p +
      geom_vline(data = perm_vals, aes(xintercept = dropout_loss),
                 size = 1.4, lty = 2, alpha = 0.7) +
      geom_boxplot(fill = "#91CBD765", alpha = 0.4)

  }
  p +
    theme(legend.position = "none") +
    labs(x = metric_lab,
         y = NULL,  fill = NULL,  color = NULL)
}
```

`ggplot_imp(vip_lm, vip_rf)`를 사용하면 [그림 18-4](#global-rf)가 생성됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1804.png" alt="tmwr 1804" />
<h6 id="figure-18-4.-global-explainer-for-the-random-forest-and-linear-regression-models.">그림 18-4. 랜덤 포레스트 및 선형 회귀 모델을 위한 전역 설명자.</h6>
</figure>

[그림 18-4](#global-rf)의 각 패널에 있는 파선(dashed line)은 선형 모델이나 랜덤 포레스트 모델 중 하나인 전체 모델의 RMSE를 보여줍니다. 더 오른쪽에 있는 피처는 순열하면(permuting) 더 높은 RMSE가 도출(results in)되므로 더 중요합니다. 이 플롯에서 배울 수 있는 흥미로운 정보가 꽤 많습니다; 예를 들어, 교호작용/스플라인이 있는 선형 모델에서는 이웃(neighborhood)이 꽤 중요하지만 랜덤 포레스트 모델에서는 두 번째로 덜 중요한 피처입니다.

# 지역 설명에서 전역 설명 구축 (Building Global Explanations from Local Explanations)

이 장의 지금까지 단일 관측치에 대한 지역 모델 설명(섀플리 덧셈 설명을 통해)과 전체 데이터 세트에 대한 전역 모델 설명(피처를 순열하여)에 초점을 맞췄습니다(focused on). *부분 의존성 프로파일(partial dependence profiles)*과 같이 지역 모델 설명을 집계하여 전역 모델 설명을 구축하는 것도 가능합니다.

###### 참고 (Note)

부분 의존성 프로파일은 Ames의 예상 주택 가격과 같은 모델 예측의 기대값이 연식이나 총 생활 면적과 같은 피처의 함수로서 어떻게 변하는지 보여줍니다.

이러한 프로파일을 구축하는 한 가지 방법은 개별 관측치에 대한 프로파일을 집계하거나 평균을 내는 것입니다(aggregating or averaging). 개별 관측치의 예측이 주어진 피처의 함수로서 어떻게 변하는지 보여주는 프로파일을 ICE(individual conditional expectation, 개별 조건부 기대) 프로파일 또는 CP(ceteris paribus, 다른 모든 조건이 동일할 때) 프로파일이라고 합니다. 이러한 개별 프로파일(훈련 세트의 관측치 500개에 대해)을 계산한 다음 DALEX 함수 `model_profile()`을 사용하여 집계(aggregate)할 수 있습니다.

```
set.seed(1805)
pdp_age <- model_profile(explainer_rf, N = 500, variables = "Year_Built")
```

이 객체의 기저가 되는(underlying) 데이터를 플로팅하기 위한 또 다른 함수를 만들어 보겠습니다.

```
ggplot_pdp <- function(obj, x) {

  p <-
    as_tibble(obj$agr_profiles) %>%
    mutate(`_label_` = stringr::str_remove(`_label_`, "^[^_]*_")) %>%
    ggplot(aes(`_x_`, `_yhat_`)) +
    geom_line(data = as_tibble(obj$cp_profiles),
              aes(x = {{ x }}, group = `_ids_`),
              size = 0.5, alpha = 0.05, color = "gray50")

  num_colors <- n_distinct(obj$agr_profiles$`_label_`)

  if (num_colors > 1) {
    p <- p + geom_line(aes(color = `_label_`, lty = `_label_`), size = 1.2)
  } else {
    p <- p + geom_line(color = "midnightblue", size = 1.2, alpha = 0.8)
  }

  p
}
```

이 함수를 사용하면 랜덤 포레스트 모델의 비선형(nonlinear) 동작을 볼 수 있는 [그림 18-5](#year-built)가 생성됩니다.

```
ggplot_pdp(pdp_age, Year_Built)  +
  labs(x = "Year built",
       y = "Sale Price (log)",
       color = NULL)
```

다른 연도에 지어진 주택의 판매 가격은 대개 평탄(mostly flat)하며 약 1960년 이후 약간 상승합니다(modest rise). 부분 의존성 프로파일은 모델의 다른 특성에 대해서도 계산할 수 있으며 `Bldg_Type`과 같은 데이터의 그룹에 대해서도 계산할 수 있습니다. 이러한 프로파일에 1,000개의 관측치를 사용해 보겠습니다.

```
set.seed(1806)
pdp_liv <- model_profile(explainer_rf, N = 1000,
                         variables = "Gr_Liv_Area",
                         groups = "Bldg_Type")

ggplot_pdp(pdp_liv, Gr_Liv_Area) +
  scale_x_log10() +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "Gross living area",
       y = "Sale Price (log)",
       color = NULL, lty = NULL)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1805.png" alt="tmwr 1805" />
<h6 id="figure-18-5.-partial-dependence-profiles-for-the-random-forest-model-focusing-on-the-year-built-predictor.">그림 18-5. 건축 연도 예측 변수에 초점을 맞춘 랜덤 포레스트 모델의 부분 의존성 프로파일.</h6>
</figure>

이 코드는 [그림 18-6](#building-type-profiles)을 생성하는데, 여기서 우리는 약 1,000에서 3,000 제곱피트의 생활 면적 사이에서 판매 가격이 가장 많이 증가(increases the most)하고, 주택 유형이 다르면(1가구 주택이나 다른 유형의 타운하우스 등) 대개 생활 공간이 많아질수록 유사하게 증가하는 가격 추세를 보인다는 것을 확인할 수 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1806.png" alt="tmwr 1806" />
<h6 id="figure-18-6.-partial-dependence-profiles-for-the-random-forest-model-focusing-on-building-types-and-gross-living-area.">그림 18-6. 건물 유형 및 총 생활 면적에 초점을 맞춘 랜덤 포레스트 모델의 부분 의존성 프로파일.</h6>
</figure>

기본 DALEX 플롯에 `plot(pdp_liv)`를 사용하는 옵션이 있지만, 여기서는 기저가 되는(underlying) 데이터로 플롯을 만들고 있기 때문에 예측이 다르게 변경되는지 시각화하고 이러한 하위 그룹 간의 불균형을 강조하기 위해 피처 중 하나로 패싯(facet)을 설정할 수도 있습니다([그림 18-7](#building-type-facets) 참조):

```
as_tibble(pdp_liv$agr_profiles) %>%
  mutate(Bldg_Type = stringr::str_remove(`_label_`, "random forest_")) %>%
  ggplot(aes(`_x_`, `_yhat_`, color = Bldg_Type)) +
  geom_line(data = as_tibble(pdp_liv$cp_profiles),
            aes(x = Gr_Liv_Area, group = `_ids_`),
            size = 0.5, alpha = 0.1, color = "gray50") +
  geom_line(size = 1.2, alpha = 0.8, show.legend = FALSE) +
  scale_x_log10() +
  facet_wrap(~Bldg_Type) +
  scale_color_brewer(palette = "Dark2") +
  labs(x = "Gross living area",
       y = "Sale Price (log)",
       color = NULL)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1807.png" alt="tmwr 1807" />
<h6 id="figure-18-7.-partial-dependence-profiles-for-the-random-forest-model-focusing-on-building-types-and-gross-living-area-using-facets.">그림 18-7. 패싯을 사용하여 건물 유형 및 총 생활 면적에 초점을 맞춘 랜덤 포레스트 모델의 부분 의존성 프로파일.</h6>
</figure>

모델 설명을 구축하는 데 올바른(correct) 접근 방식은 단 하나만 있는 것이 아니며, 이 장에서 간략히 설명한(outlined) 옵션이 완전한(exhaustive) 것은 아닙니다. 개별 수준과 전역 수준 모두에서 설명을 위한 좋은 옵션과 하나에서 다른 하나로 연결하는 방법(bridge)을 강조(highlighted)했으며, 추가적인 독서(further reading)를 위해 Biecek and Burzykowski (2021)와 [Molnar (2020)](https://oreil.ly/P5Vlg)을 추천합니다(point you to).

# 다시 콩으로! (Back to Beans!)

[16장](ch16.xhtml#dimensionality)에서는 고차원 데이터를 모델링할 때 차원 축소를 피처 엔지니어링이나 전처리 단계로 사용하는 방법에 대해 논의했습니다. 콩 유형을 예측하는 마른 콩(dry bean) 형태(morphology) 측정값의 예제 데이터 세트에 대해 부분 최소 제곱(PLS) 차원 축소와 정규화 판별 분석 모델을 결합(combined with)하여 훌륭한(great) 결과를 확인했습니다. 이러한 형태적 특성 중 어떤 것이 콩 유형 예측에서 _가장_ 중요했습니까? 이 장 전반에 걸쳐(throughout) 간략히 설명된 것과 동일한 접근 방식을 사용하여 모델 불가지론적 설명자(model-agnostic explainer)를 만들고 `model_parts()`를 통해, 이를테면 전역 모델 설명을 계산할 수 있습니다.

```
set.seed(1807)
vip_beans <-
  explain_tidymodels(
    rda_wflow_fit,
    data = bean_train %>% select(-class),
    y = bean_train$class,
    label = "RDA",
    verbose = FALSE
  ) %>%
  model_parts()
```

이전에 정의한 중요도 플로팅 함수인 `ggplot_imp(vip_beans)`를 사용하면 [그림 18-8](#bean-explainer)이 생성됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1808.png" alt="tmwr 1808" />
<h6 id="figure-18-8.-global-explainer-for-the-regularized-discriminant-analysis-model-on-the-beans-data.">그림 18-8. 콩 데이터의 정규화 판별 분석 모델에 대한 전역 설명자.</h6>
</figure>

###### 경고 (Warning)

[그림 18-8](#bean-explainer)에서 볼 수 있는 전역 피처 중요도 측정값은 모든 PLS 성분의 효과를 통합(incorporate)하지만 원래(original) 변수의 측면(in terms of)에서 나타냅니다.

[그림 18-8](#bean-explainer)은 모양(shape) 계수가 콩 유형을 예측하는 데 가장 중요한 특성 중 하나이며, 특히 면적 $`A`$, 장축 $`L`$, 단축 $`l`$을 고려(takes into account)하는 견고성(solidity) 척도(measure)인 모양 계수 4가 중요함을 보여줍니다.

```math
\text{SF4} = \frac{A}{\pi(L/2)(l/2)}
```

[그림 18-8](#bean-explainer)에서 형태 계수 1(장축과 면적의 비율), 단축 길이 및 둥글기(roundness)가 콩 품종을 예측하는 데 두 번째로 중요한 콩 특성임을 알 수 있습니다.

# 이 장의 요약 (Chapter Summary)

일부 유형의 모델에서는 모델이 특정 예측을 수행한 이유에 대한 대답이 직관적(straightforward)이지만, 다른 유형의 모델에서는 예측에 어떤 특성이 상대적으로 가장 중요한지 이해하기 위해 별도의 설명자 알고리즘을 사용해야 합니다. 훈련된 모델에서 두 가지 주요(main) 종류의 모델 설명을 생성할 수 있습니다. 전역 설명은 전체 데이터 세트에 대해 집계된 정보를 제공하는 반면 지역 설명은 단일 관측치에 대한 모델 예측에 대한 이해를 제공합니다.

DALEX 및 해당 지원 패키지 DALEXtra, vip, lime과 같은 패키지를 tidymodels 분석에 통합하여 이러한 모델 설명자를 제공할 수 있습니다. 모델 설명은 모델 성능의 추정치와 함께 모델이 적절(appropriate)하고 효과적인지(effective) 이해하는 한 부분에 불과합니다; [19장](ch19.xhtml#trust)에서는 예측의 품질과 신뢰성(trustworthiness)을 추가로(further) 탐구합니다.

<sup>[1](ch18.xhtml#idm45881848206144-marker)</sup> 모델 설명을 위한 이 패키지는 이러한 유형의 출력에서 범주형 예측 변수의 *수준(level)*에 중점을 둡니다(focuses on). 듀플렉스의 경우 `Bldg_Type = 3`이고 North Ames의 경우 `Neighborhood = 1`인 것과 같습니다.
