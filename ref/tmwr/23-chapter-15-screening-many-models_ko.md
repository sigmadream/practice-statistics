# 15장. 많은 모델 선별 (Screening Many Models)

우리는 [7장](ch07.xhtml#workflows)에서 워크플로 세트(workflow sets)를 소개하고 [11장](ch11.xhtml#compare)에서 리샘플링된 데이터 세트와 함께 이를 사용하는 방법을 시연했습니다. 이 장에서는 여러 모델링 워크플로의 이러한 세트에 대해 더 자세히(in more detail) 논의하고 이것이 도움이 될 수 있는 사용 사례(use case)를 설명(describe)합니다.

아직 잘 이해되지 않은 새로운 데이터 세트가 있는 프로젝트의 경우, 데이터 실무자(practitioner)는 많은 모델 및 전처리기 조합을 선별(screen)해야 할 수 있습니다. 새로운(novel) 데이터 세트에 어떤 방법이 가장 잘 작동할지에 대한 사전(a priori) 지식이 거의 없거나 전혀 없는 것이 일반적입니다.

###### 참고 (Note)

좋은 전략은 다양한 모델링 접근 방식을 시도하는 데 초기 노력을 약간 기울여 무엇이 가장 잘 작동하는지 파악한(determine) 다음, 작은 모델 세트를 미세 조정/최적화하는 데 추가 시간을 투자하는(invest) 것입니다.

워크플로 세트는 이 프로세스를 만들고 관리하기 위한 사용자 인터페이스를 제공합니다. 또한 이 장 뒷부분에서 논의될 경주 방법(racing methods)을 사용하여 이러한 모델을 효율적으로 평가하는 방법도 시연할 것입니다.

# 콘크리트 혼합 강도 모델링 (Modeling Concrete Mixture Strength)

여러 모델 워크플로를 선별하는 방법을 시연하기 위해 *Applied Predictive Modeling* (Kuhn and Johnson 2013)에 나온 콘크리트 혼합물(mixture) 데이터를 예제로 사용하겠습니다. 해당 책의 10장에서는 재료를 예측 변수로 사용하여 콘크리트 혼합물의 압축(compressive) 강도를 예측하는 모델을 시연했습니다. 다양한 예측 변수 세트와 전처리 요구 사항에 따라(with) 매우 다양한 모델이 평가되었습니다. 워크플로 세트가 어떻게 모델에 대한 이러한 대규모 테스트 프로세스를 더 쉽게 만들 수 있을까요?

먼저 데이터 분할 및 리샘플링 구성(schemes)을 정의해 보겠습니다:

```
library(tidymodels)
tidymodels_prefer()
data(concrete, package = "modeldata")
glimpse(concrete)
#> Rows: 1,030
#> Columns: 9
#> $ cement               <dbl> 540.0, 540.0, 332.5, 332.5, 198.6, 266.0, …
#> $ blast_furnace_slag   <dbl> 0.0, 0.0, 142.5, 142.5, 132.4, 114.0, 95.0, …
#> $ fly_ash              <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, …
#> $ water                <dbl> 162, 162, 228, 228, 192, 228, 228, 228, 228, …
#> $ superplasticizer     <dbl> 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, …
#> $ coarse_aggregate     <dbl> 1040.0, 1055.0, 932.0, 932.0, 978.4, 932.0, …
#> $ fine_aggregate       <dbl> 676.0, 676.0, 594.0, 594.0, 825.5, 670.0, …
#> $ age                  <int> 28, 28, 270, 365, 360, 90, 365, 28, 28, 28, …
#> $ compressive_strength <dbl> 79.99, 61.89, 40.27, 41.05, 44.30, 47.03, …
```

`compressive_strength` 열이 결과(outcome)입니다. `age` 예측 변수는 테스트 시점의 콘크리트 샘플 연령을 일(days) 단위로 알려주며(콘크리트는 시간이 지남에 따라 강해집니다), `cement` 및 `water`와 같은 나머지 예측 변수는 입방 미터당 킬로그램 단위의 콘크리트 구성 성분(components)입니다.

###### 경고 (Warning)

이 데이터 세트의 일부 사례(cases)에서는 동일한 콘크리트 공식이 여러 번 테스트되었습니다. 이러한 반복적인(replicate) 혼합물은 훈련 세트와 테스트 세트 모두에 분산(distributed)될 수 있으므로 개별 데이터 포인트로 포함하지 않는 편이 좋습니다(We'd rather not). 그렇게 하면 우리의 성능 추정치가 인위적으로 부풀려질(artificially inflate) 수 있습니다.

이 문제를 해결하기 위해 모델링에 콘크리트 혼합물당 평균 압축 강도를 사용할 것입니다:

```
concrete <-
   concrete %>%
   group_by(across(-compressive_strength)) %>%
   summarize(compressive_strength = mean(compressive_strength),
             .groups = "drop")
nrow(concrete)
#> [1] 992
```

기본적인(default) 3:1 훈련 대 테스트 비율을 사용하여 데이터를 분할(split)하고 5회 반복하는 10-겹 교차 검증을 사용하여 훈련 세트를 리샘플링해 보겠습니다:

```
set.seed(1501)
concrete_split <- initial_split(concrete, strata = compressive_strength)
concrete_train <- training(concrete_split)
concrete_test  <- testing(concrete_split)

set.seed(1502)
concrete_folds <-
   vfold_cv(concrete_train, strata = compressive_strength, repeats = 5)
```

일부 모델(특히 신경망, *K*-최근접 이웃 및 서포트 벡터 머신)은 중심화(centered) 및 척도화(scaled)된 예측 변수를 필요로 하므로 일부 모델 워크플로에는 이러한 전처리 단계가 있는 레시피가 필요합니다. 다른 모델의 경우 전통적인 반응 표면(response surface) 설계 모델 확장(즉, 2차 및 2원(two-way) 교호작용)이 좋은 생각입니다. 이러한 목적으로 두 가지 레시피를 만듭니다:

```
normalized_rec <-
   recipe(compressive_strength ~ ., data = concrete_train) %>%
   step_normalize(all_predictors())

poly_recipe <-
   normalized_rec %>%
   step_poly(all_predictors()) %>%
   step_interact(~ all_predictors():all_predictors())
```

모델의 경우 parsnip 추가 기능(add-in)을 사용하여 모델 사양 세트를 만듭니다:

```
library(rules)
library(baguette)

linear_reg_spec <-
   linear_reg(penalty = tune(), mixture = tune()) %>%
   set_engine("glmnet")

nnet_spec <-
   mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
   set_engine("nnet", MaxNWts = 2600) %>%
   set_mode("regression")

mars_spec <-
   mars(prod_degree = tune()) %>%  #<- GCV를 사용하여 항(terms) 선택
   set_engine("earth") %>%
   set_mode("regression")

svm_r_spec <-
   svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
   set_engine("kernlab") %>%
   set_mode("regression")

svm_p_spec <-
   svm_poly(cost = tune(), degree = tune()) %>%
   set_engine("kernlab") %>%
   set_mode("regression")

knn_spec <-
   nearest_neighbor(neighbors = tune(), dist_power = tune(), weight_func = tune()) %>%
   set_engine("kknn") %>%
   set_mode("regression")

cart_spec <-
   decision_tree(cost_complexity = tune(), min_n = tune()) %>%
   set_engine("rpart") %>%
   set_mode("regression")

bag_cart_spec <-
   bag_tree() %>%
   set_engine("rpart", times = 50L) %>%
   set_mode("regression")

rf_spec <-
   rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
   set_engine("ranger") %>%
   set_mode("regression")

xgb_spec <-
   boost_tree(tree_depth = tune(), learn_rate = tune(), loss_reduction = tune(),
              min_n = tune(), sample_size = tune(), trees = tune()) %>%
   set_engine("xgboost") %>%
   set_mode("regression")

cubist_spec <-
   cubist_rules(committees = tune(), neighbors = tune()) %>%
   set_engine("Cubist")
```

Kuhn and Johnson (2013)의 분석은 신경망이 레이어에 최대 27개의 은닉 유닛(hidden units)을 가져야 한다고 지정(specifies)합니다. `extract_parameter_set_dials()` 함수는 매개변수 세트를 추출하며, 올바른 매개변수 범위를 갖도록 수정합니다:

```
nnet_param <-
   nnet_spec %>%
   extract_parameter_set_dials() %>%
   update(hidden_units = hidden_units(c(1, 27)))
```

이러한 모델을 레시피에 어떻게 일치(match)시키고, 튜닝한 다음(tune them), 성능을 효율적으로 평가할 수 있을까요? 워크플로 세트가 솔루션을 제공합니다.

# 워크플로 세트 생성 (Creating the Workflow Set)

워크플로 세트는 전처리기와 모델 사양의 이름이 지정된(named) 리스트를 가져와서 여러 워크플로가 포함된 객체로 결합합니다. 가능한 전처리기의 종류(kinds)는 세 가지입니다:

- 표준 R 공식 (formula)

- 레시피 객체 (추정/준비(prepping) 이전)

- 결과와 예측 변수를 선택하기 위한 dplyr 스타일 선택기(selector)

첫 번째 워크플로 세트 예로서, 예측 변수를 동일한 단위로 만들어야 하는 비선형 모델들에 예측 변수를 단순히 표준화하기만 하는 레시피를 결합해 보겠습니다:

```
normalized <-
   workflow_set(
      preproc = list(normalized = normalized_rec),
      models = list(SVM_radial = svm_r_spec, SVM_poly = svm_p_spec,
                    KNN = knn_spec, neural_network = nnet_spec)
   )
normalized
#> # A workflow set/tibble: 4 × 4
#>   wflow_id                  info             option    result
#>   <chr>                     <list>           <list>    <list>
#> 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 normalized_neural_network <tibble [1 × 4]> <opts[0]> <list [0]>
```

전처리기가 하나만 있기 때문에 이 함수는 이 값을 사용하여 워크플로 세트를 생성합니다. 전처리기에 둘 이상의 항목이 포함된 경우 함수는 전처리기와 모델의 모든 조합을 생성합니다.

`wflow_id` 열은 자동으로 생성되지만 `mutate()`를 호출하여 수정할 수 있습니다. `info` 열에는 일부 식별자(identifiers)와 워크플로 객체가 포함된 티블(tibble)이 포함되어 있습니다. 워크플로는 추출할 수 있습니다:

```
normalized %>% extract_workflow(id = "normalized_KNN")
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: nearest_neighbor()
#>
#> ── Preprocessor ─────────────────────────────────────
#> 1 Recipe Step
#>
#> • step_normalize()
#>
#> ── Model ────────────────────────────────────────────
#> K-Nearest Neighbor Model Specification (regression)
#>
#> Main Arguments:
#>   neighbors = tune()
#>   weight_func = tune()
#>   dist_power = tune()
#>
#> Computational engine: kknn
```

`option` 열은 워크플로를 평가할 때 사용할 인수에 대한 자리 표시자(placeholder)입니다. 예를 들어 신경망 매개변수 객체를 추가하려면:

```
normalized <-
   normalized %>%
   option_add(param_info = nnet_param, id = "normalized_neural_network")
normalized
#> # A workflow set/tibble: 4 × 4
#>   wflow_id                  info             option    result
#>   <chr>                     <list>           <list>    <list>
#> 1 normalized_SVM_radial     <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 normalized_SVM_poly       <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 normalized_KNN            <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 normalized_neural_network <tibble [1 × 4]> <opts[1]> <list [0]>
```

tune 또는 finetune 패키지의 함수를 사용하여 워크플로를 튜닝(또는 리샘플링)할 때 이 인수가 사용됩니다.

`result` 열은 튜닝 또는 리샘플링 함수의 출력을 위한 자리 표시자입니다.

다른 비선형 모델의 경우 결과와 예측 변수에 dplyr 선택기를 사용하는 또 다른 워크플로 세트를 만들어 보겠습니다:

```
model_vars <-
   workflow_variables(outcomes = compressive_strength,
                      predictors = everything())

no_pre_proc <-
   workflow_set(
      preproc = list(simple = model_vars),
      models = list(MARS = mars_spec,
                    CART = cart_spec,
                    CART_bagged = bag_cart_spec,
                    RF = rf_spec,
                    boosting = xgb_spec,
                    Cubist = cubist_spec)
   )
no_pre_proc
#> # A workflow set/tibble: 6 × 4
#>   wflow_id           info             option    result
#>   <chr>              <list>           <list>    <list>
#> 1 simple_MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 simple_CART        <tibble [1 × 4]> <opts[0]> <list [0]>
```
#> 3 simple_CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 simple_RF          <tibble [1 × 4]> <opts[0]> <list [0]>
#> 5 simple_boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 6 simple_Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>
```

마지막으로, 적절한 모델과 함께 비선형 항(terms) 및 교호작용(interactions)을 사용하는 세트를 조립(assemble)합니다:

```
with_features <-
   workflow_set(
      preproc = list(full_quad = poly_recipe),
      models = list(linear_reg = linear_reg_spec, KNN = knn_spec)
   )
```

이러한 객체는 `workflow_set`이라는 추가 클래스가 있는 티블(tibbles)입니다. 행(row) 결합은 세트의 상태(state)에 영향을 주지 않으며 결과 자체도 워크플로 세트입니다:

```
all_workflows <-
   bind_rows(no_pre_proc, normalized, with_features) %>%
   # 워크플로 ID를 좀 더 단순하게 만듭니다:
   mutate(wflow_id = gsub("(simple_)|(normalized_)", "", wflow_id))
all_workflows
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result
#>   <chr>       <list>           <list>    <list>
#> 1 MARS        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 2 CART        <tibble [1 × 4]> <opts[0]> <list [0]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[0]> <list [0]>
#> 4 RF          <tibble [1 × 4]> <opts[0]> <list [0]>
#> 5 boosting    <tibble [1 × 4]> <opts[0]> <list [0]>
#> 6 Cubist      <tibble [1 × 4]> <opts[0]> <list [0]>
#> # … with 6 more rows
```

# 모델 튜닝 및 평가 (Tuning and Evaluating the Models)

`all_workflows`의 거의 모든 멤버(members)에는 튜닝 매개변수가 포함되어 있습니다. 성능을 평가하기 위해 표준 튜닝 또는 리샘플링 함수(예: `tune_grid()` 등)를 사용할 수 있습니다. `workflow_map()` 함수는 세트의 모든 워크플로에 동일한 함수를 적용합니다; 기본값은 `tune_grid()`입니다.

이 예에서는 최대 25개의 다른 매개변수 후보를 사용하여 각 워크플로에 그리드 검색이 적용됩니다(applied). `tune_grid()`를 실행(execution)할 때마다 함께 사용할 공통 옵션 세트가 있습니다. 예를 들어, 다음 코드에서는 그리드 크기 25와 함께 각 워크플로에 대해 동일한 리샘플링 및 제어(control) 객체를 사용할 것입니다. `workflow_map()` 함수에는 `seed`라는 추가 인수가 있으며, 이는 `tune_grid()`의 각 실행이 동일한 난수를 소비(consumes)하도록 보장(ensure)하는 데 사용됩니다:

```
grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
   )

grid_results <-
   all_workflows %>%
   workflow_map(
      seed = 1503,
      resamples = concrete_folds,
      grid = 25,
      control = grid_ctrl
   )
```

결과는 `option`과 `result` 열이 업데이트되었음을 보여줍니다:

```
grid_ctrl <-
   control_grid(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
   )

full_results_time <-
   system.time(
      grid_results <-
         all_workflows %>%
         workflow_map(seed = 1503, resamples = concrete_folds, grid = 25,
                      control = grid_ctrl, verbose = TRUE)
   )
#> i  1 of 12 tuning:     MARS
#> ✓  1 of 12 tuning:     MARS (2.7s)
#> i  2 of 12 tuning:     CART
#> ✓  2 of 12 tuning:     CART (27.6s)
#> i    No tuning parameters. `fit_resamples()` will be attempted
#> i  3 of 12 resampling: CART_bagged
#> ✓  3 of 12 resampling: CART_bagged (18.5s)
#> i  4 of 12 tuning:     RF
#> i Creating preprocessing data to finalize unknown parameter: mtry
#> ✓  4 of 12 tuning:     RF (1m 9.2s)
#> i  5 of 12 tuning:     boosting
#> ✓  5 of 12 tuning:     boosting (2m 4.1s)
#> i  6 of 12 tuning:     Cubist
#> ✓  6 of 12 tuning:     Cubist (2m 0.7s)
#> i  7 of 12 tuning:     SVM_radial
#> ✓  7 of 12 tuning:     SVM_radial (40.2s)
#> i  8 of 12 tuning:     SVM_poly
#> ✓  8 of 12 tuning:     SVM_poly (7m 46.4s)
#> i  9 of 12 tuning:     KNN
#> ✓  9 of 12 tuning:     KNN (43.2s)
#> i 10 of 12 tuning:     neural_network
#> ✓ 10 of 12 tuning:     neural_network (1m 22s)
#> i 11 of 12 tuning:     full_quad_linear_reg
#> ✓ 11 of 12 tuning:     full_quad_linear_reg (57.9s)
#> i 12 of 12 tuning:     full_quad_KNN
#> ✓ 12 of 12 tuning:     full_quad_KNN (2m 59.8s)

num_grid_models <- nrow(collect_metrics(grid_results, summarize = FALSE))
```

우리의 `grid_results`는 어떻게 보일까요?

```
grid_results
#> # A workflow set/tibble: 12 × 4
#>   wflow_id    info             option    result
#>   <chr>       <list>           <list>    <list>
#> 1 MARS        <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 2 CART        <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 3 CART_bagged <tibble [1 × 4]> <opts[3]> <rsmp[+]>
#> 4 RF          <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 5 boosting    <tibble [1 × 4]> <opts[3]> <tune[+]>
#> 6 Cubist      <tibble [1 × 4]> <opts[3]> <tune[+]>
#> # … with 6 more rows
```

`option` 열에는 이제 우리가 `workflow_map()` 호출에서 사용한 모든 옵션이 포함되어 있습니다. 이는 우리의 결과를 재현 가능(reproducible)하게 만듭니다. `result` 열에서 `"tune[+]"` 및 `"rsmp[+]"` 표기법(notations)은 객체에 문제가 없음을 의미합니다. 어떤 이유로든 모든 모델이 실패한 경우 `"tune[x]"`와 같은 값이 나타납니다.

`grid_results`와 같은 결과를 조사(examining)하기 위한 몇 가지 편리한(convenience) 함수가 있습니다. `rank_results()` 함수는 어떤 성능 지표를 기준으로 모델을 정렬합니다. 기본적으로 메트릭 세트의 첫 번째 메트릭(이 경우 RMSE)을 사용합니다. RMSE만 살펴보도록 `filter()`를 적용해 보겠습니다:

```
grid_results %>%
   rank_results() %>%
   filter(.metric == "rmse") %>%
   select(model, .config, rmse = mean, rank)
#> # A tibble: 252 × 4
#>   model      .config                rmse  rank
#>   <chr>      <chr>                 <dbl> <int>
#> 1 boost_tree Preprocessor1_Model04  4.25     1
#> 2 boost_tree Preprocessor1_Model06  4.29     2
#> 3 boost_tree Preprocessor1_Model13  4.31     3
#> 4 boost_tree Preprocessor1_Model14  4.39     4
#> 5 boost_tree Preprocessor1_Model16  4.46     5
#> 6 boost_tree Preprocessor1_Model03  4.47     6
#> # … with 246 more rows
```

또한 기본적으로 이 함수는 모든 후보 세트의 순위를 매깁니다; 그렇기 때문에 동일한 모델이 출력에 여러 번 나타날 수 있습니다. `select_best`라는 옵션을 사용하여 최상의 튜닝 매개변수 조합을 사용하여 모델의 순위를 매길 수 있습니다.

`autoplot()` 메서드는 순위를 플로팅(plots)합니다; 또한 `select_best` 인수도 있습니다. [그림 15-1](#workflow-set-ranks)의 플롯은 각 모델에 대한 최상의 결과를 시각화하며 다음을 사용하여 생성됩니다:

```
autoplot(
   grid_results,
   rank_metric = "rmse",  # <- 모델을 정렬하는 방법
   metric = "rmse",       # <- 시각화할 메트릭
   select_best = TRUE     # <- 워크플로당 하나의 포인트
) +
   geom_text(aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
   lims(y = c(3.5, 9.5)) +
   theme(legend.position = "none")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1501.png" alt="tmwr 1501" />
<h6 id="figure-15-1.-estimated-rmse-and-approximate-confidence-intervals-for-the-best-model-configuration-in-each-workflow.">그림 15-1. 각 워크플로에서 최상의 모델 구성에 대한 추정 RMSE(및 대략적인 신뢰 구간).</h6>
</figure>

[그림 15-2](#workflow-sets-autoplot)와 같이 특정 모델의 튜닝 매개변수 결과를 보려는 경우(In case you want to see), 플로팅할 모델을 나타내기 위해 `id` 인수는 `wflow_id` 열에서 단일 값을 취할 수 있습니다:

```
autoplot(grid_results, id = "Cubist", metric = "rmse")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1502.png" alt="tmwr 1502" />
<h6 id="figure-15-2.-the-autoplot-results-for-the-cubist-model-contained-in-the-workflow-set.">그림 15-2. 워크플로 세트에 포함된 Cubist 모델에 대한 <code>autoplot()</code> 결과.</h6>
</figure>

`collect_predictions()` 및 `collect_metrics()`를 위한 메서드들도 있습니다.

콘크리트 혼합 데이터를 사용한 예제 모델 선별은 총 25,200개의 모델을 피팅합니다. 2개의 워커(workers)를 병렬로 사용했을 때, 추정 프로세스를 완료하는 데 1.9시간이 걸렸습니다.

# 효율적인 모델 선별 (Efficiently Screening Models)

대규모 모델 세트를 효율적으로 선별하기 위한 효과적인 방법 중 하나는 [13장](ch13.xhtml#grid-search)에서 설명한 경주(racing) 접근 방식을 사용하는 것입니다. 워크플로 세트를 사용하면 이 경주 접근 방식에 `workflow_map()` 함수를 사용할 수 있습니다. 워크플로 세트를 파이프로 연결한 후, 사용하는 인수는 워크플로에 적용할 함수라는 점을 상기하십시오; 이 경우 `"tune_race_anova"` 값을 사용할 수 있습니다. 적절한(appropriate) 제어(control) 객체도 전달합니다; 그렇지 않으면(otherwise) 옵션은 이전 섹션의 코드와 동일합니다:

```
library(finetune)

race_ctrl <-
   control_race(
      save_pred = TRUE,
      parallel_over = "everything",
      save_workflow = TRUE
   )

race_results <-
   all_workflows %>%
   workflow_map(
      "tune_race_anova",
      seed = 1503,
      resamples = concrete_folds,
      grid = 25,
      control = race_ctrl
   )
```

새로운 객체는 매우 비슷해 보이지만 `result` 열의 요소(elements)에 `"race[+]"`라는 값이 표시되어 다른 유형의 객체임을 나타냅니다:

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

결과를 조사(interrogate)하기 위해 이 객체에 대해 동일하게 유용한(helpful) 함수들을 사용할 수 있으며, 사실 [그림 15-3](#workflow-set-racing-ranks)<sup><a href="ch15.xhtml#idm45881852122544" id="idm45881852122544-marker" data-type="noteref">1</a></sup>에 표시된 기본적인 `autoplot()` 메서드는 [그림 15-2](#workflow-sets-autoplot)와 유사한 추세(trends)를 생성합니다. 이것은 다음 코드에 의해 생성됩니다:

```
autoplot(
   race_results,
   rank_metric = "rmse",
   metric = "rmse",
   select_best = TRUE
) +
   geom_text(aes(y = mean - 1/2, label = wflow_id), angle = 90, hjust = 1) +
   lims(y = c(3.0, 9.5)) +
   theme(legend.position = "none")
```

전반적으로 경주 접근 방식은 전체 그리드의 25,200개 모델로 구성된 전체 세트 중 18.46%인 총 4,652개의 모델을 추정했습니다. 그 결과 경주 방식은 4.5배(4.5-fold) 더 빨랐습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1503.png" alt="tmwr 1503" />
<h6 id="figure-15-3.-estimated-rmse-and-approximate-confidence-intervals-for-the-best-model-configuration-in-each-workflow-in-the-racing-results.">그림 15-3. 경주 결과 내의 각 워크플로에서 최상의 모델 구성에 대한 추정 RMSE(및 대략적인 신뢰 구간).</h6>
</figure>

비슷한 결과를 얻었을까요? 두 객체에 대해 결과의 순위를 매기고 병합한(merge) 다음 [그림 15-4](#racing-concordance)에서 서로에 대해(against one another) 플로팅(plot)합니다:

```
matched_results <-
   rank_results(race_results, select_best = TRUE) %>%
   select(wflow_id, .metric, race = mean, config_race = .config) %>%
   inner_join(
      rank_results(grid_results, select_best = TRUE) %>%
         select(wflow_id, .metric, complete = mean,
                config_complete = .config, model),
      by = c("wflow_id", ".metric"),
   ) %>%
   filter(.metric == "rmse")

library(ggrepel)

matched_results %>%
   ggplot(aes(x = complete, y = race)) +
   geom_abline(lty = 3) +
   geom_point() +
   geom_text_repel(aes(label = model)) +
   coord_obs_pred() +
   labs(x = "Complete Grid RMSE", y = "Racing RMSE")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1504.png" alt="tmwr 1504" />
<h6 id="figure-15-4.-estimated-rmse-for-the-full-grid-and-racing-results.">그림 15-4. 전체 그리드와 경주 결과에 대한 추정 RMSE.</h6>
</figure>

경주(racing) 접근 방식은 모델의 41.67%에 대해서만 전체 그리드와 동일한 후보 매개변수를 선택했지만, 경주에 의해 선택된 모델의 성능 지표는 거의 동일했습니다. RMSE 값의 상관관계(correlation)는 0.968이었고 순위(rank) 상관관계는 0.951이었습니다. 이는 한 모델 내에 거의 동일한 결과를 갖는 여러 튜닝 매개변수 조합이 있음을 나타냅니다.

# 모델 마무리하기 (Finalizing a Model)

이전 장에서 보여준 것과 마찬가지로 최종 모델을 선택하고 훈련 세트에 피팅하는 과정은 간단(straightforward)합니다. 첫 번째 단계는 마무리할 워크플로를 선택하는 것입니다. 부스팅된 트리 모델이 잘 작동했으므로 세트에서 해당 모델을 추출(extract)하고, 매개변수를 수치상 가장 좋은(numerically best) 설정으로 업데이트한 후 훈련 세트에 피팅하겠습니다:

```
best_results <-
   race_results %>%
   extract_workflow_set_result("boosting") %>%
   select_best(metric = "rmse")
best_results
#> # A tibble: 1 × 7
#>   trees min_n tree_depth learn_rate loss_reduction sample_size .config
#>   <int> <int>      <int>      <dbl>          <dbl>       <dbl> <chr>
#> 1  1957     8          7     0.0756    0.000000145       0.679 Preprocessor1_Model04

boosting_test_results <-
   race_results %>%
   extract_workflow("boosting") %>%
   finalize_workflow(best_results) %>%
   last_fit(split = concrete_split)
```

[그림 15-5](#concrete-test-results)에서 테스트 세트 메트릭 결과를 보고 예측을 시각화할 수 있습니다:

```
collect_metrics(boosting_test_results)
#> # A tibble: 2 × 4
#>   .metric .estimator .estimate .config
#>   <chr>   <chr>          <dbl> <chr>
#> 1 rmse    standard       3.33  Preprocessor1_Model1
#> 2 rsq     standard       0.956 Preprocessor1_Model1
```

```
boosting_test_results %>%
   collect_predictions() %>%
   ggplot(aes(x = compressive_strength, y = .pred)) +
   geom_abline(color = "gray50", lty = 2) +
   geom_point(alpha = 0.5) +
   coord_obs_pred() +
   labs(x = "observed", y = "predicted")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1505.png" alt="tmwr 1505" />
<h6 id="figure-15-5.-observed-versus-predicted-values-for-the-test-set.">그림 15-5. 테스트 세트에 대한 관찰(observed) 값 대 예측(predicted) 값.</h6>
</figure>

이 콘크리트 혼합물에 대해 관찰된 압축 강도와 예측된 압축 강도가 얼마나 잘 정렬(align)되는지 여기에서 볼 수 있습니다.

# 이 장의 요약 (Chapter Summary)

종종 데이터 실무자는 당면한(at hand) 작업을 위해 가능한 다수의 모델링 접근 방식을 고려해야 합니다. 특히 새로운 데이터 세트의 경우 및/또는 어떤 모델링 전략이 가장 잘 작동할지에 대한 지식이 거의 없을 때 그렇습니다. 이 장에서는 이러한 상황에서 여러 모델 또는 피처 엔지니어링 전략을 조사(investigate)하기 위해 워크플로 세트를 사용하는 방법을 설명(illustrated)했습니다. 경주 방법은 고려 중인 모든 후보 모델을 피팅하는 것보다 모델 순위를 더 효율적으로 매길 수 있습니다.

<sup>[1](ch15.xhtml#idm45881852122544-marker)</sup> 2022년 2월 현재(As of), Intel 아키텍처와 비교하여 ARM 아키텍처(Apple M1 칩)의 macOS를 사용하여 훈련했을 때 신경망에 대해 약간 다른 성능 지표를 볼 수 있습니다.
