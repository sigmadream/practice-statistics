# 13장. 그리드 검색 (Grid Search)

[12장](ch12.xhtml#tuning)에서는 `tune()` 함수를 사용하여 최적화를 위해 전처리 레시피 및/또는 모델 지정(specifications)에 인수를 표시(mark)하거나 태그를 지정하는 방법을 시연했습니다. 무엇을 최적화할지 알고 나면 매개변수를 어떻게 최적화할지 묻는 질문을 다룰(address) 차례입니다. 이 장에서는 매개변수의 가능한 값을 선험적으로(a priori) 지정하는 _그리드 검색(grid search)_ 방법에 대해 설명합니다. ([14장](ch14.xhtml#iterative-search)에서는 반복(iterative) 검색 방법을 설명하여 논의를 이어갈 것입니다.)

그리드를 조립(assembling)하기 위한 두 가지 주요 접근 방식을 살펴보는 것으로 시작해 보겠습니다.

# 정규 및 비정규 그리드 (Regular and Nonregular Grids)

그리드에는 두 가지 주요 유형이 있습니다. 정규 그리드(regular grid)는 각 매개변수(해당 가능한 값의 세트와 함께)를 요인별로(factorially), 즉 세트의 모든 조합을 사용하여 결합합니다. 대안적으로, 비정규 그리드(nonregular grid)는 매개변수 조합이 적은(small) 포인트 세트에서 형성되지 않는 그리드입니다.

각 유형을 더 자세히 살펴보기 전에, 다층 퍼셉트론(multilayer perceptron) 모델(일명 단일 계층 인공 신경망)이라는 예제 모델을 고려해 보겠습니다. 튜닝용으로 표시된 매개변수는 다음과 같습니다.

- 은닉 유닛(hidden units)의 수

- 모델 훈련의 피팅(fitting) 에포크(epochs)/반복(iterations) 수

- 가중치 감소(weight decay) 페널티화(penalization)의 양

parsnip을 사용하여 nnet 패키지를 사용하는 분류 모델 피팅에 대한 지정(specification)은 다음과 같습니다.

```
library(tidymodels)
tidymodels_prefer()

mlp_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_engine("nnet", trace = 0) %>%
  set_mode("classification")
```

`trace = 0` 인수는 훈련 프로세스의 추가 로깅을 방지합니다. [12장](ch12.xhtml#tuning)에 표시된 대로 `extract_parameter_set_dials()` 함수는 알 수 없는 값을 가진 인수 세트를 추출하고 그 dials 객체를 설정할 수 있습니다.

```
mlp_param <- extract_parameter_set_dials(mlp_spec)
mlp_param %>% extract_parameter_dials("hidden_units")
#> # Hidden Units (quantitative)
#> Range: [1, 10]
mlp_param %>% extract_parameter_dials("penalty")
#> Amount of Regularization (quantitative)
#> Transformer: log-10 [1e-100, Inf]
#> Range (transformed scale): [-10, 0]
mlp_param %>% extract_parameter_dials("epochs")
#> # Epochs (quantitative)
#> Range: [10, 1000]
```

이 출력은 매개변수 객체가 완전함(complete)을 나타내며 기본 범위를 인쇄합니다. 이 값들은 다양한 유형의 매개변수 그리드를 생성하는 방법을 보여주는 데 사용됩니다.

###### 참고 (Note)

역사적으로 에포크 수는 조기 종료(early stopping)에 의해 결정되었습니다; 훈련 세트를 재예측(repredicting)하면 과적합이 발생하므로 별도의 검증 세트가 오류율에 따라 훈련 길이를 결정했습니다. 우리의 경우에는 가중치 감소 페널티를 사용하면 과적합이 금지(prohibit)될 것이며, 페널티와 에포크 수를 튜닝하는 데 해로울 것은 거의 없습니다.

## 정규 그리드 (Regular Grids)

정규 그리드는 개별 매개변수 값 세트의 조합입니다. 먼저, 사용자는 각 매개변수에 대해 고유한(distinct) 값 세트를 생성합니다. 가능한 값의 수가 각 매개변수마다 동일할 필요는 없습니다. tidyr 함수 `crossing()`은 정규 그리드를 생성하는 한 가지 방법입니다.

```
crossing(
  hidden_units = 1:3,
  penalty = c(0.0, 0.1),
  epochs = c(100, 200)
)
#> # A tibble: 12 × 3
#>   hidden_units penalty epochs
#>          <int>   <dbl>  <dbl>
#> 1            1     0      100
#> 2            1     0      200
#> 3            1     0.1    100
#> 4            1     0.1    200
#> 5            2     0      100
#> 6            2     0      200
#> # … with 6 more rows
```

매개변수 객체는 매개변수의 범위를 알고 있습니다. dials 패키지에는 다양한 유형의 그리드를 생성하기 위해 매개변수 객체를 입력으로 사용하는 일련의 `grid_*()` 함수들이 포함되어 있습니다. 예를 들어:

```
grid_regular(mlp_param, levels = 2)
#> # A tibble: 8 × 3
#>   hidden_units      penalty epochs
#>          <int>        <dbl>  <int>
#> 1            1 0.0000000001     10
#> 2           10 0.0000000001     10
#> 3            1 1                10
#> 4           10 1                10
#> 5            1 0.0000000001   1000
#> 6           10 0.0000000001   1000
#> # … with 2 more rows
```

`levels` 인수는 생성할 매개변수당 레벨 수입니다. 명명된 값 벡터(named vector of values)를 사용할 수도 있습니다.

```
mlp_param %>%
  grid_regular(levels = c(hidden_units = 3, penalty = 2, epochs = 2))
#> # A tibble: 12 × 3
#>   hidden_units      penalty epochs
#>          <int>        <dbl>  <int>
#> 1            1 0.0000000001     10
#> 2            5 0.0000000001     10
#> 3           10 0.0000000001     10
#> 4            1 1                10
#> 5            5 1                10
#> 6           10 1                10
#> # … with 6 more rows
```

각 매개변수 세트의 가능한 모든 값을 사용하지 않는 정규 그리드를 생성하는 기법(techniques)이 있습니다. 이러한 _일부 실시 요인 설계(fractional factorial designs)_ (Box, Hunter, and Hunter 2005)도 사용될 수 있습니다. 자세한 내용은 [실험 설계에 대한 CRAN Task View](https://oreil.ly/PvLCj)를 참조하십시오.

###### 경고 (Warning)

정규 그리드는 특히 튜닝 매개변수가 중간 내지 큰 수로 존재할 때 계산 비용이 많이 들 수 있습니다. 이것은 많은 모델에 해당되지만 모든 모델에 해당되지는 않습니다. 이 장에서 더 논의하겠지만, 정규 그리드를 사용할 때 튜닝 시간이 _감소하는_ 많은 모델이 있습니다!

정규 그리드 사용의 한 가지 이점은 튜닝 매개변수와 모델 지표(metrics) 간의 관계(relationships) 및 패턴을 쉽게 이해할 수 있다는 것입니다. 이러한 설계의 팩토리얼 특성을 통해 매개변수 간의 교란(confounding)을 거의 일으키지 않고 각 매개변수를 개별적으로 검사할 수 있습니다.

## 비정규 그리드 (Nonregular Grids)

비정규 그리드를 생성하기 위한 몇 가지 옵션이 있습니다. 첫 번째는 매개변수 범위 전반에 걸쳐 무작위 샘플링을 사용하는 것입니다. `grid_random()` 함수는 매개변수 범위 전반에 걸쳐 독립적인 균일 난수(independent uniform random numbers)를 생성합니다. (우리가 `penalty`에 대해 가진 것처럼) 매개변수 객체에 연관된 변환(transformation)이 있는 경우 변환된 스케일에서 난수가 생성됩니다. 예제 신경망의 매개변수에 대한 무작위 그리드를 만들어 보겠습니다.

```
set.seed(1301)
mlp_param %>%
  grid_random(size = 1000) %>% # 'size'는 조합의 수입니다
  summary()
#>   hidden_units      penalty           epochs
#>  Min.   : 1.00   Min.   :0.0000   Min.   : 10
#>  1st Qu.: 3.00   1st Qu.:0.0000   1st Qu.:266
#>  Median : 5.00   Median :0.0000   Median :497
#>  Mean   : 5.38   Mean   :0.0437   Mean   :510
#>  3rd Qu.: 8.00   3rd Qu.:0.0027   3rd Qu.:761
#>  Max.   :10.00   Max.   :0.9814   Max.   :999
```

`penalty`의 경우 난수는 로그(밑 10) 척도(scale)에서 균일(uniform)하지만 그리드의 값은 자연 단위(natural units)입니다.

무작위 그리드의 문제점은, 중소 규모(small-to-medium) 그리드의 경우 무작위 값으로 인해 매개변수 조합이 겹칠 수 있다는 것입니다. 또한 무작위 그리드는 전체 매개변수 공간을 포괄해야 하지만 좋은 포괄 범위(coverage)의 가능성은 그리드 값의 수에 따라 증가합니다. 15개의 후보 포인트 샘플에 대해서도 [그림 13-1](#random-grid)은 예시인 다층 퍼셉트론에 대한 포인트 간의 일부 겹침(overlap)을 보여줍니다.

```
library(ggforce)
set.seed(1302)
mlp_param %>%
  # 'original = FALSE' 옵션은 페널티를 log10 단위로 유지합니다
  grid_random(size = 20, original = FALSE) %>%
  ggplot(aes(x = .panel_x, y = .panel_y)) +
  geom_point() +
  geom_blank() +
  facet_matrix(vars(hidden_units, penalty, epochs), layer.diag = 2) +
  labs(title = "Random design with 20 candidates")
```

훨씬 더 나은 접근 방식은 *공간 채우기 설계(space-filling designs)*라고 하는 실험 설계 세트를 사용하는 것입니다. 서로 다른 설계 방법은 약간 다른 목표를 가지고 있지만 일반적으로 겹치거나(overlapping) 중복되는(redundant) 값의 가능성이 가장 작은 상태로 매개변수 공간을 덮는(cover) 점 구성을 찾습니다. 이러한 설계의 예로는 라틴 하이퍼큐브(Latin hypercubes) (McKay, Beckman, and Conover 1979), 최대 엔트로피 설계(maximum entropy designs) (Shewry and Wynn 1987), 최대 투영 설계(maximum projection designs) (Joseph, Gul, and Ba 2015) 등이 있습니다. 개요는 Santner et al. (2003)을 참조하십시오.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1301.png" alt="tmwr 1301" />
<h6 id="figure-13-1.-three-tuning-parameters-with-20-points-generated-at-random.">그림 13-1. 20개 포인트가 무작위로 생성된 세 가지 튜닝 매개변수.</h6>
</figure>

dials 패키지에는 라틴 하이퍼큐브 및 최대 엔트로피 설계를 위한 함수가 포함되어 있습니다. `grid_random()`과 마찬가지로 주요 입력은 매개변수 조합의 수와 매개변수 객체입니다. [그림 13-2](#space-filling-design)에서 15개 후보 매개변수 값에 대해 무작위 설계(random design)와 라틴 하이퍼큐브 설계를 비교해 보겠습니다.

```
set.seed(1303)
mlp_param %>%
  grid_latin_hypercube(size = 20, original = FALSE) %>%
  ggplot(aes(x = .panel_x, y = .panel_y)) +
  geom_point() +
  geom_blank() +
  facet_matrix(vars(hidden_units, penalty, epochs), layer.diag = 2) +
  labs(title = "Latin Hypercube design with 20 candidates")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1302.png" alt="tmwr 1302" />
<h6 id="figure-13-2.-three-tuning-parameters-with-20-points-generated-using-a-space-filling-design.">그림 13-2. 공간 채우기 설계를 사용하여 20개 포인트가 생성된 세 가지 튜닝 매개변수.</h6>
</figure>

완벽하지는 않지만 이 라틴 하이퍼큐브 설계는 점들을 서로 더 멀리 떨어뜨려 놓고(spaces the points farther away) 하이퍼파라미터 공간에 대한 더 나은 탐색을 가능하게 합니다.

공간 채우기 설계는 매개변수 공간을 나타내는 데 매우 효과적일 수 있습니다. tune 패키지에서 사용하는 기본 설계는 최대 엔트로피 설계입니다. 이들은 후보 공간을 잘 덮고 좋은 결과를 찾을 가능성을 크게(drastically) 높이는 그리드를 생성하는 경향이 있습니다.

# 그리드 평가 (Evaluating the Grid)

최상의 튜닝 매개변수 조합을 선택하기 위해 해당 모델을 훈련하는 데 사용되지 않은 데이터를 사용하여 각 후보 세트를 평가(assessed)합니다. 리샘플링 방법이나 단일 검증 세트가 이 목적에 잘 작동합니다. 프로세스(및 구문(syntax))는 tune 패키지의 `fit_resamples()` 함수를 사용한 [10장](ch10.xhtml#resampling)의 접근 방식과 매우 유사(resembles)합니다.

리샘플링 후, 사용자는 가장 적절한 후보 매개변수 세트를 선택합니다. 경험적으로(empirically) 최상의 매개변수 조합을 선택하거나 단순성과 같이 모델 피팅의 다른 측면(aspects)으로 선택을 편향시키는(bias) 것이 합리적일 수 있습니다.

우리는 이 장과 다음 장에서 모델 튜닝을 시연하기 위해 분류 데이터 세트를 사용합니다. 이 데이터는 암 연구를 위한 자동화된 현미경 검사 실험실 도구를 개발한 Hill et al. (2007)에서 가져온 것입니다. 이 데이터는 2,019개의 인간 유방암 세포에 대한 56개의 이미징 측정(imaging measurements)으로 구성됩니다. 이러한 예측 변수는 세포의 다양한 부분(세포 핵(nucleus), 세포 경계(cell boundary) 등)의 모양 및 강도 특성을 나타냅니다. 예측 변수 간에는 높은 정도의 상관관계가 있습니다. 예를 들어, 세포 핵 및 세포 경계의 크기와 모양을 측정하는 몇 가지 다른 예측 변수들이 있습니다. 또한 개별적으로 많은 예측 변수가 비대칭 분포(skewed distributions)를 갖습니다.

각 세포는 두 클래스 중 하나에 속합니다. 이는 자동화된 실험실 테스트의 일부이기 때문에 추론(inference)보다는 예측 능력에 초점을 맞추었습니다.

데이터는 modeldata 패키지에 포함되어 있습니다. 분석에 필요하지 않은 열 하나(`case`)를 제거해 보겠습니다.

```
library(tidymodels)
data(cells)
cells <- cells %>% select(-case)
```

데이터의 차원을 감안할 때, 10-겹 교차 검증을 사용하여 성능 지표를 계산할 수 있습니다.

```
set.seed(1304)
cell_folds <- vfold_cv(cells)
```

예측 변수 간의 높은 상관관계를 고려할 때, 예측 변수 간의 상관관계를 제거(decorrelate)하기 위해 PCA 피처 추출을 사용하는 것이 타당(makes sense)합니다. 다음 레시피에는 대칭성을 높이기 위해 예측 변수를 변환하고(transform), 동일한 척도가 되도록 정규화하고(normalize), 피처 추출을 수행하는 단계가 포함되어 있습니다. 유지(retain)할 PCA 성분 수도 모델 매개변수와 함께 튜닝됩니다.

###### 경고 (Warning)

결과 PCA 성분은 기술적으로 동일한 척도에 있지만 더 낮은 순위(lower-rank) 성분이 높은 순위(higher-rank) 성분보다 더 넓은 범위를 갖는 경향이 있습니다. 이러한 이유로 우리는 예측 변수들이 동일한 평균과 분산을 갖도록 강제(coerce)하기 위해 다시 정규화합니다.

많은 예측 변수가 비대칭 분포를 보입니다. PCA는 분산을 기반으로 하므로 극단값(extreme values)이 이러한 계산에 해로운(detrimental) 영향을 미칠 수 있습니다. 이에 대응하기(counter) 위해, 각 예측 변수에 대한 여-존슨(Yeo-Johnson) 변환(Yeo and Johnson 2000)을 추정하는 레시피 단계를 추가해 보겠습니다. 원래 결과(outcome)의 변환으로 의도(intended)되었지만 더 대칭적인 분포를 장려하는(encourage) 변환을 추정하는 데 사용할 수도 있습니다. 이 단계인 `step_YeoJohnson()`은 `step_normalize()`를 통한 초기 정규화 직전에 레시피에서 발생합니다. 그런 다음 이 피처 엔지니어링 레시피를 신경망 모델 지정 `mlp_spec`과 결합해 보겠습니다.

```
mlp_rec <-
  recipe(class ~ ., data = cells) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_pca(all_numeric_predictors(), num_comp = tune()) %>%
  step_normalize(all_numeric_predictors())

mlp_wflow <-
  workflow() %>%
  add_model(mlp_spec) %>%
  add_recipe(mlp_rec)
```

기본 범위 중 몇 가지를 조정하기 위해 매개변수 객체 `mlp_param`을 만들어 보겠습니다. 에포크 수를 더 작은 범위(50에서 200 에포크)로 변경할 수 있습니다. 또한 `num_comp()`의 기본 범위는 매우 좁은 범위(1에서 4개 성분)를 기본값으로 사용합니다; 우리는 범위를 40개 성분으로 늘리고 최솟값을 0으로 설정할 수 있습니다.

```
mlp_param <-
  mlp_wflow %>%
  extract_parameter_set_dials() %>%
  update(
    epochs = epochs(c(50, 200)),
    num_comp = num_comp(c(0, 40))
  )
```

###### 참고 (Note)

`step_pca()`에서 0개의 PCA 성분을 사용하는 것은 피처 추출을 건너뛰는(skip) 숏컷(shortcut)입니다. 이런 방식으로 원본 예측 변수를 PCA 성분이 포함된 결과와 직접 비교할 수 있습니다.

`tune_grid()` 함수는 그리드 검색을 수행하는 기본 함수입니다. 비록 그리드와 관련된 추가 인수가 있긴 하지만 그 기능은 `fit_resamples()`와 매우 유사합니다.

`grid`  
정수(integer) 또는 데이터 프레임입니다. 정수가 사용될 때 함수는 후보 매개변수 조합의 `grid` 수만큼 공간 채우기 설계를 생성합니다. 특정 매개변수 조합이 존재하는 경우 `grid` 매개변수를 사용하여 해당 조합을 함수에 전달합니다.

`param_info`  
매개변수 범위를 정의하기 위한 선택적(optional) 인수입니다. 이 인수는 `grid`가 정수일 때 가장 유용합니다.
그렇지 않으면 `tune_grid()`에 대한 인터페이스는 `fit_resamples()`와 동일합니다. 첫 번째 인수는 모델 지정 또는 워크플로 중 하나입니다. 모델이 주어질 때 두 번째 인수는 레시피 또는 공식(formula)이 될 수 있습니다. 필요한 다른 인수는 rsample 리샘플링 객체(`cell_folds`)입니다. 다음 호출은 리샘플링 중에 ROC 곡선 아래 면적을 측정하도록 지표(metric) 세트도 전달합니다.

시작하기 위해 리샘플 전반에 걸쳐 3개의 레벨이 있는 정규 그리드를 평가해 보겠습니다.

```
roc_res <- metric_set(roc_auc)
set.seed(1305)
mlp_reg_tune <-
  mlp_wflow %>%
  tune_grid(
    cell_folds,
    grid = mlp_param %>% grid_regular(levels = 3),
    metrics = roc_res
  )
mlp_reg_tune
#> # Tuning results
#> # 10-fold cross-validation
#> # A tibble: 10 × 4
#>   splits             id     .metrics          .notes
#>   <list>             <chr>  <list>            <list>
#> 1 <split [1817/202]> Fold01 <tibble [81 × 8]> <tibble [0 × 3]>
#> 2 <split [1817/202]> Fold02 <tibble [81 × 8]> <tibble [0 × 3]>
#> 3 <split [1817/202]> Fold03 <tibble [81 × 8]> <tibble [0 × 3]>
#> 4 <split [1817/202]> Fold04 <tibble [81 × 8]> <tibble [0 × 3]>
#> 5 <split [1817/202]> Fold05 <tibble [81 × 8]> <tibble [0 × 3]>
#> 6 <split [1817/202]> Fold06 <tibble [81 × 8]> <tibble [0 × 3]>
#> # … with 4 more rows
```

결과를 이해하는 데 사용할 수 있는 고급(high-level) 편의(convenience) 함수가 있습니다. 첫째, 정규 그리드에 대한 `autoplot()` 메서드는 [그림 13-3](#regular-grid-plot)에서 튜닝 매개변수 전반에 걸친 성능 프로파일(profiles)을 보여줍니다.

```
autoplot(mlp_reg_tune) +
  scale_color_viridis_d(direction = -1) +
  theme(legend.position = "top")
```

이 데이터의 경우 페널티 양이 ROC 곡선 아래 면적에 가장 큰 영향을 미칩니다. 에포크 수는 성능에 뚜렷한(pronounced) 영향을 미치지 않는 것으로 보입니다. 은닉 유닛 수의 변화는 정규화 양이 적을 때(low) (그리고 성능에 해를 끼칠 때) 가장 중요한 것으로 보입니다. `show_best()` 함수를 사용하여 볼 수 있듯이 대략적으로 동등한(roughly equivalent) 성능을 갖는 여러 매개변수 구성(configurations)이 있습니다.

```
show_best(mlp_reg_tune) %>% select(-.estimator)
#> # A tibble: 5 × 9
#>   hidden_units penalty epochs num_comp .metric  mean     n std_err .config
#>          <int>   <dbl>  <int>    <int> <chr>   <dbl> <int>   <dbl> <chr>
#> 1            5       1     50        0 roc_auc 0.897    10 0.00857 Prepro…
#> 2           10       1    125        0 roc_auc 0.895    10 0.00898 Prepro…
#> 3           10       1     50        0 roc_auc 0.894    10 0.00960 Prepro…
#> 4            5       1    200        0 roc_auc 0.894    10 0.00784 Prepro…
#> 5            5       1    125        0 roc_auc 0.892    10 0.00822 Prepro…
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1303.png" alt="tmwr 1303" />
<h6 id="figure-13-3.-the-regular-grid-results.">그림 13-3. 정규 그리드 결과.</h6>
</figure>

이러한 결과를 바탕으로 더 큰 가중치 감소 페널티 값으로 그리드 검색을 한 번 더 수행하는 것이 합리적일 것입니다.

공간 채우기 설계를 사용하려면 `grid` 인수에 정수를 제공하거나 `grid_*()` 함수 중 하나가 데이터 프레임을 생성할 수 있습니다. 20개의 후보 값을 가진 최대 엔트로피 설계를 사용하여 동일한 범위를 평가하려면:

```
set.seed(1306)
mlp_sfd_tune <-
  mlp_wflow %>%
  tune_grid(
    cell_folds,
    grid = 20,
    # 적절한 범위를 사용하기 위해 매개변수 객체를 전달합니다.
    param_info = mlp_param,
    metrics = roc_res
  )
mlp_sfd_tune
#> # Tuning results
#> # 10-fold cross-validation
#> # A tibble: 10 × 4
#>   splits             id     .metrics          .notes
#>   <list>             <chr>  <list>            <list>
#> 1 <split [1817/202]> Fold01 <tibble [20 × 8]> <tibble [0 × 3]>
#> 2 <split [1817/202]> Fold02 <tibble [20 × 8]> <tibble [0 × 3]>
#> 3 <split [1817/202]> Fold03 <tibble [20 × 8]> <tibble [0 × 3]>
#> 4 <split [1817/202]> Fold04 <tibble [20 × 8]> <tibble [0 × 3]>
#> 5 <split [1817/202]> Fold05 <tibble [20 × 8]> <tibble [0 × 3]>
#> 6 <split [1817/202]> Fold06 <tibble [20 × 8]> <tibble [0 × 3]>
#> # … with 4 more rows
```

`autoplot()` 메서드도 이러한 설계와 함께 작동하지만 결과의 형식은 다를 것입니다. [그림 13-4](#sfd-plot)는 `autoplot(mlp_sfd_tune)`을 사용하여 생성되었습니다.

이 한계 효과(marginal effects) 플롯([그림 13-4](#sfd-plot))은 각 매개변수와 성능 지표의 관계를 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1304.png" alt="tmwr 1304" />
<h6 id="figure-13-4.-the-autoplot-method-results-when-used-with-a-space-filling-design.">그림 13-4. 공간 채우기 설계와 함께 사용된 경우 <code>autoplot()</code> 메서드 결과.</h6>
</figure>

###### 경고 (Warning)

이 플롯을 검사할 때 주의(Take care)하십시오; 정규 그리드가 사용되지 않았으므로 다른 튜닝 매개변수의 값이 각 패널에 영향을 미칠 수 있습니다.

페널티 매개변수는 가중치 감소량이 작을수록 더 나은 성능을 내는 것으로 보입니다. 이는 정규 그리드의 결과와 정반대입니다. 각 패널의 각 지점은 나머지 3개의 튜닝 매개변수와 공유되므로 한 패널의 추세가 다른 매개변수들의 영향을 받을 수 있습니다. 정규 그리드를 사용하면 각 패널의 각 지점이 다른 매개변수들에 대해 균등하게(equally) 평균화됩니다. 이러한 이유로 정규 그리드에서는 각 매개변수의 효과가 더 잘 격리(isolated)됩니다.

정규 그리드와 마찬가지로 `show_best()`는 수치상 최상의 결과를 보고할 수 있습니다.

```
show_best(mlp_sfd_tune) %>% select(-.estimator)
#> # A tibble: 5 × 9
#>   hidden_units       penalty epochs num_comp .metric  mean     n std_err .config
#>          <int>         <dbl>  <int>    <int> <chr>   <dbl> <int>   <dbl> <chr>
#> 1            8 0.594             97       22 roc_auc 0.880    10 0.00998 Preprocess…
#> 2            3 0.00000000649    135        8 roc_auc 0.878    10 0.00956 Preprocess…
#> 3            9 0.141            177       11 roc_auc 0.873    10 0.0104  Preprocess…
#> 4            8 0.0000000103      74        9 roc_auc 0.869    10 0.00761 Preprocess…
#> 5            6 0.00581          129       15 roc_auc 0.865    10 0.00658 Preprocess…
```

일반적으로 모델 피팅의 다양한 측면(aspects)이 고려(taken into account)되도록 여러 지표(metrics)에 걸쳐 모델을 평가하는 것이 좋은 생각(good idea)입니다. 또한, 더 단순한 모델과 연관된 약간(slightly) 차선의(suboptimal) 매개변수 조합을 선택하는 것이 종종 합리적일 수 있습니다. 이 모델의 경우, 단순성(simplicity)은 더 큰 페널티 값 및/또는 더 적은 은닉 유닛에 해당합니다(corresponds to).

`fit_resamples()`의 결과와 마찬가지로, 리샘플 및 튜닝 매개변수 전반에 걸쳐 중간(intermediary) 모델 피팅을 유지(retaining)하는 것은 대개 가치가 없습니다. 그러나 이전과 마찬가지로 `control_grid()`에 대한 `extract` 옵션은 피팅된(fitted) 모델 및/또는 레시피의 보존(retention)을 허용합니다. 또한 `save_pred` 옵션을 `TRUE`로 설정하면 평가 세트 예측(predictions)이 유지(retains)되며 `collect_predictions()`를 사용하여 이에 접근할 수 있습니다.

# 모델 마무리하기 (Finalizing the Model)

`show_best()`를 통해 발견된 가능한 모델 매개변수 세트 중 하나가 이 데이터에 대한 매력적인 최종 옵션이라면, 우리는 그것이 테스트 세트에서 얼마나 잘 수행되는지 평가하고 싶을 수 있습니다. 그러나 `tune_grid()`의 결과는 적절한 튜닝 매개변수를 선택하기 위한 바탕(substrate)만을 제공합니다. 이 함수는 최종 모델을 _피팅하지 않습니다(does not fit)_.

최종 모델을 피팅하려면 매개변수 값의 최종 세트를 결정해야 합니다. 이를 수행하는 두 가지 방법이 있습니다.

- 적절해 보이는 값을 수동으로 고르거나(pick)

- `select_*()` 함수를 사용합니다.

예를 들어, `select_best()`는 수치상 최상의 결과를 가진 매개변수를 선택합니다. 정규 그리드 결과로 돌아가서 어느 것이 가장 좋은지 살펴보겠습니다.

```
select_best(mlp_reg_tune, metric = "roc_auc")
#> # A tibble: 1 × 5
#>   hidden_units penalty epochs num_comp .config
#>          <int>   <dbl>  <int>    <int> <chr>
#> 1            5       1     50        0 Preprocessor1_Model08
```

[그림 13-3](#regular-grid-plot)을 되돌아보면, 많은 양의 정규화(penalization)를 사용하여 원본 예측 변수에 대해 125 에포크 동안 훈련된 단일 은닉 유닛을 가진 모델이 이 옵션과 경쟁력 있게 수행되며 더 단순하다는 것을 알 수 있습니다. 이것은 기본적으로 페널티화된 로지스틱 회귀(penalized logistic regression)입니다! 이 매개변수들을 수동으로 지정하기 위해 이러한 값을 사용하여 티블을 생성한 다음, 값을 워크플로에 다시 접합(splice)하기 위해 _마무리(finalization)_ 함수를 사용할 수 있습니다.

```
logistic_param <-
  tibble(
    num_comp = 0,
    epochs = 125,
    hidden_units = 1,
    penalty = 1
  )

final_mlp_wflow <-
  mlp_wflow %>%
  finalize_workflow(logistic_param)
final_mlp_wflow
#> ══ Workflow ═════════════════════════════════════════
#> Preprocessor: Recipe
#> Model: mlp()
#>
#> ── Preprocessor ─────────────────────────────────────
#> 4 Recipe Steps
#>
#> • step_YeoJohnson()
#> • step_normalize()
#> • step_pca()
#> • step_normalize()
#>
#> ── Model ────────────────────────────────────────────
#> Single Layer Neural Network Specification (classification)
#>
#> Main Arguments:
#>   hidden_units = 1
#>   penalty = 1
#>   epochs = 125
#>
#> Engine-Specific Arguments:
#>   trace = 0
#>
#> Computational engine: nnet
```

이 마무리된 워크플로에는 더 이상 `tune()` 값이 포함되어 있지 않습니다. 이제 모델을 전체 훈련 세트에 피팅할 수 있습니다.

```
final_mlp_fit <-
  final_mlp_wflow %>%
  fit(cells)
```

이제 이 객체를 사용하여 새로운 데이터에 대한 미래 예측을 할 수 있습니다.

워크플로를 사용하지 않은 경우 모델 및/또는 레시피의 마무리는 `finalize_model()` 및 `finalize_recipe()`를 사용하여 수행됩니다.

# 튜닝 지정 생성을 위한 도구 (Tools for Creating Tuning Specifications)

usemodels 패키지는 데이터 프레임과 모델 공식을 가져온 다음 모델 튜닝을 위한 R 코드를 작성할(write out) 수 있습니다. 이 코드는 요청된 모델과 예측 변수 데이터에 따라(depend on) 단계가 달라지는 적절한 레시피도 생성합니다. 예를 들어, Ames 주택 데이터의 경우 `xgboost` 모델링 코드는 다음을 통해 생성될 수 있습니다.

```
library(usemodels)

use_xgboost(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
              Latitude + Longitude,
            data = ames_train,
            # 일부 코드를 설명하는 주석(comments) 추가:
            verbose = TRUE)
```

결과 코드는 다음과 같습니다.

```
xgboost_recipe <-
  recipe(formula = Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
    Latitude + Longitude, data = ames_train) %>%
  step_novel(all_nominal_predictors()) %>%
  ## 이 모델은 예측 변수가 수치형일 것을 요구합니다. 정성적(qualitative) 예측 변수를 수치형으로 변환하는
  ## 가장 일반적인 방법은 이러한 예측 변수들로부터
  ## 이진(binary) 지시(indicator) 변수(일명 더미 변수)를 생성하는 것입니다. 그러나 이 모델의 경우
  ## 인자(factors)의 각 레벨에 대해 이진 지시 변수를
  ## 만들 수 있습니다('원-핫 인코딩(one-hot encoding)'으로 알려져 있음).
  step_dummy(all_nominal_predictors(), one_hot = TRUE) %>%
  step_zv(all_predictors())

xgboost_spec <-
  boost_tree(trees = tune(), min_n = tune(), tree_depth = tune(), learn_rate = tune(),
    loss_reduction = tune(), sample_size = tune()) %>%
  set_mode("regression") %>%
  set_engine("xgboost")

xgboost_workflow <-
  workflow() %>%
  add_recipe(xgboost_recipe) %>%
  add_model(xgboost_spec)

set.seed(69305)
xgboost_tune <-
  tune_grid(xgboost_workflow,
            resamples = stop("add your rsample object"),
            grid = stop("add number of candidate points"))
```

usemodels가 데이터에 대해 이해하는 바를 바탕으로 이 코드는 필요한 최소한의 전처리입니다. 다른 모델의 경우 모델의 기본 요구 사항(basic needs)을 충족하기(fulfill) 위해 `step_normalize()`와 같은 연산이 추가됩니다. 어떤 리샘플(`resamples`)을 튜닝에 사용할지, 그리고 어떤 종류의 그리드(`grid`)를 사용할지 선택하는 것은 모델링 실무자(practitioner)로서의 우리 책임(responsibility)이라는 점에 유의하십시오.

###### 참고 (Note)

usemodels 패키지는 `tune = FALSE` 인수를 설정하여 튜닝이 없는 모델 피팅 코드를 생성하는 데도 사용될 수 있습니다.

# 효율적인 그리드 검색을 위한 도구 (Tools for Efficient Grid Search)

몇 가지 다른 트릭과 최적화를 적용하여 그리드 검색의 계산 효율성을 높이는 것이 가능합니다. 이 섹션에서는 여러 가지 기법을 설명합니다.

## 하위 모델 최적화 (Submodel Optimization)

단일 모델 피팅에서 재피팅(refitting) 없이 여러 튜닝 매개변수를 평가할 수 있는 모델 유형이 있습니다.

예를 들어, 부분 최소 제곱(partial least squares, PLS)은 주성분 분석(PCA)의 지도(supervised) 버전입니다(Geladi and Kowalski 1986). 이것은 (PCA처럼) 예측 변수의 변동을 극대화하는 성분을 생성하지만, 동시에 이러한 예측 변수와 결과(outcome) 간의 상관관계를 극대화하려고 합니다. PLS에 대해서는 [16장](ch16.xhtml#dimensionality)에서 더 자세히 살펴보겠습니다. 튜닝 매개변수 중 하나는 유지할 PLS 성분의 수입니다. 100개의 예측 변수가 있는 데이터 세트를 PLS를 사용하여 피팅한다고 가정해 봅시다. 유지할 가능한 성분의 수는 1에서 50까지 다양할 수 있습니다. 그러나 많은 구현(implementations)에서 단일 모델 피팅은 `num_comp`의 여러 값에 걸쳐 예측값을 계산할 수 있습니다. 결과적으로 100개의 성분으로 생성된 PLS 모델은 `num_comp <= 100`인 모든 수에 대해서도 예측을 수행할 수 있습니다. 이렇게 하면 중복되는(redundant) 모델 피팅을 생성하는 대신 단일 피팅을 사용하여 많은 하위 모델(submodels)을 평가할 수 있으므로 시간이 절약됩니다.

모든 모델이 이 기능을 활용할(exploit) 수 있는 것은 아니지만, 널리 사용되는 많은 모델이 이 기능을 활용합니다.

- 부스팅 모델은 일반적으로 부스팅 반복 횟수에 대한 여러 값에 걸쳐 예측을 수행할 수 있습니다.

- glmnet 모델과 같은 정규화 방법은 모델을 피팅하는 데 사용되는 정규화 양에 걸쳐 동시(simultaneous) 예측을 수행할 수 있습니다.

- 다변량 적응 회귀 스플라인(Multivariate adaptive regression splines, MARS)은 선형 회귀 모델에 비선형 피처 세트를 추가합니다(Friedman 1991). 유지할 항(terms)의 수는 튜닝 매개변수이며, 단일 모델 피팅에서 이 매개변수의 여러 값에 걸쳐 예측을 수행하는 것은 계산적으로 빠릅니다.

tune 패키지는 적용 가능한(applicable) 모델이 튜닝될 때마다 이 유형의 최적화를 자동으로 적용합니다.

예를 들어, 부스팅된 C5.0 분류 모델(Kuhn and Johnson 2013)을 세포 데이터에 피팅한 경우, 부스팅 반복 횟수(`trees`)를 튜닝할 수 있습니다. 다른 모든 매개변수를 기본값으로 설정하고 이전에 사용한 것과 동일한 리샘플에서 1에서 100까지의 반복을 평가할 수 있습니다.

```
c5_spec <-
  boost_tree(trees = tune()) %>%
  set_engine("C5.0") %>%
  set_mode("classification")

set.seed(1307)
c5_spec %>%
  tune_grid(
    class ~ .,
    resamples = cell_folds,
    grid = data.frame(trees = 1:100),
    metrics = roc_res
  )
```

하위 모델 최적화가 없으면 `tune_grid()` 호출은 100개의 하위 모델을 리샘플링하는 데 62.2분이 소요되었습니다. 최적화를 사용하면 동일한 호출이 100*초*가 걸렸습니다(37배 속도 향상). 단축된(reduced) 시간은 `tune_grid()`가 1,000개의 모델을 피팅하는 것과 10개의 모델을 피팅하는 것의 차이입니다.

###### 참고 (Note)

하위 모델 예측 트릭을 사용하거나 사용하지 않고 모델을 피팅했음에도 불구하고, 이 최적화는 parsnip에 의해 자동으로 적용됩니다.

## 병렬 처리 (Parallel Processing)

[10장](ch10.xhtml#resampling)에서 이전에 언급했듯이 병렬 처리는 모델을 리샘플링할 때 실행 시간을 줄이는 효과적인 방법입니다. 이 이점은 그리드 검색을 통한 모델 튜닝에도 전달되지만(conveys to) 추가적인 고려 사항이 있습니다.

두 가지 다른 병렬 처리 구성(schemes)을 고려해 보겠습니다.

그리드 검색을 통해 모델을 튜닝할 때 두 개의 고유한 루프(distinct loops)가 있습니다. 하나는 리샘플에 대한 것이고 다른 하나는 고유한 튜닝 매개변수 조합에 대한 것입니다. 의사 코드(pseudocode)에서 이 프로세스는 다음과 같습니다.

```
for (rs in resamples) {
  # 분석 및 평가 세트 생성
  # 데이터 전처리 (공식 또는 레시피)
  for (mod in configurations) {
    # {rs} 분석 세트에 모델 {mod} 피팅
    # {rs} 평가 세트 예측
  }
}
```

기본적으로 tune 패키지는 외부(outer) 루프와 내부(inner) 루프 모두가 아니라 리샘플(외부 루프)에 대해서만 병렬화합니다.

이것은 전처리 방법 비용이 많이 들 때(expensive) 최적의 시나리오입니다. 그러나 이 접근 방식에는 두 가지 잠재적인 단점이 있습니다.

- 전처리 비용이 많이 들지 않을 때 달성할 수 있는(achievable) 속도 향상을 제한합니다.

- 병렬 워커(parallel workers)의 수는 리샘플의 수로 제한됩니다. 예를 들어, 10-겹 교차 검증을 사용하면 컴퓨터에 코어가 10개 이상 있더라도 10개의 병렬 워커만 사용할 수 있습니다.

병렬 처리가 작동하는 방식을 설명하기 위해, 5-겹 교차 검증을 사용하여 7개의 모델 튜닝 매개변수 값이 있는 경우를 사용하겠습니다. [그림 13-5](#one-resample-per-worker)는 작업(tasks)이 워커 프로세스에 할당되는 방식을 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1305.png" alt="tmwr 1305" />
<h6 id="figure-13-5.-worker-processes-when-parallel-processing-matches-resamples-to-a-specific-worker-process.">그림 13-5. 병렬 처리가 리샘플을 특정 워커 프로세스와 일치시킬 때의 워커 프로세스.</h6>
</figure>

각 폴드(fold)는 자체 워커 프로세스에 할당되며 모델 매개변수만 튜닝되므로 전처리는 폴드/워커당 한 번씩 수행됩니다. 워커 프로세스가 5개 미만으로 사용된 경우 일부 워커는 여러 개의 폴드를 받게 됩니다.

`tune_*()` 함수에 대한 제어(control) 함수에서 `parallel_over` 인수는 프로세스가 실행되는 방식을 제어합니다. 이전 병렬화 전략을 사용하려면 인수는 `parallel_over = "resamples"`입니다.

리샘플을 병렬 처리하는 대신 대체(alternate) 구성은 리샘플 및 모델에 대한 루프를 단일 루프로 결합합니다. 의사 코드에서 이 프로세스는 다음과 같습니다.

```
all_tasks <- crossing(resamples, configurations)

for (iter in all_tasks) {
  # {iter}에 대한 분석 및 평가 세트 생성
  # 데이터 전처리 (공식 또는 레시피)
  # {iter} 분석 세트에 모델 {iter} 피팅
  # {iter} 평가 세트 예측
}
```

이 경우 이제 병렬화가 단일 루프에 대해 발생합니다. 예를 들어, $`M`$ 개의 튜닝 매개변수 값에 대해 5-겹 교차 검증을 사용하면 루프는 $`5 \times M`$ 번 반복(iterations)하여 실행됩니다. 이것은 사용할 수 있는 잠재적인 워커의 수를 증가시킵니다. 그러나 데이터 전처리와 관련된 작업이 여러 번 반복됩니다. 이러한 단계에 비용이 많이 든다면 이 접근 방식은 비효율적일 것입니다.

tidymodels에서 검증 세트는 단일 리샘플로 취급됩니다. 이 경우에는 이 병렬화 구성이 가장 좋습니다.

[그림 13-6](#distributed-tasks)은 이 구성에서 워커에 대한 작업의 위임(delegation)을 보여줍니다; 동일한 예제이지만 10개의 워커를 사용합니다.

여기서 각 워커 프로세스는 여러 개의 폴드를 처리하며 전처리가 불필요하게(needlessly) 반복됩니다. 예를 들어, 첫 번째 폴드의 경우 전처리가 한 번이 아니라 일곱 번 계산되었습니다.

이 구성에 대해 제어 함수 인수는 `parallel_over = "everything"`입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1306.png" alt="tmwr 1306" />
<h6 id="figure-13-6.-worker-processes-when-preprocessing-and-modeling-tasks-are-distributed-to-many-workers.">그림 13-6. 전처리 및 모델링 작업이 여러 워커에 분산(distributed)될 때의 워커 프로세스.</h6>
</figure>

## 부스팅된 트리 벤치마킹 (Benchmarking Boosted Trees)

서로 다른 가능한 병렬화 구성을 비교하기 위해, 우리는 5-겹 교차 검증 및 10개의 후보 모델을 사용하여 4,000개 샘플의 데이터 세트를 사용하여 xgboost 엔진으로 부스팅된 트리를 튜닝했습니다. 이 데이터는 어떤 추정(estimation)도 필요하지 않은 기본(baseline) 전처리가 필요했습니다. 전처리는 세 가지 다른 방식으로 처리되었습니다.

1. dplyr 파이프라인을 사용하여 모델링 전에 데이터를 전처리합니다(나중 플롯에서는 "none"으로 표시됨).

2. 레시피를 통해 동일한 전처리를 수행합니다("light" 전처리로 표시됨).

3. 레시피와 함께 계산 비용이 많이 드는 추가 단계를 추가합니다("expensive"로 표시됨).

첫 번째와 두 번째 전처리 옵션은 비교를 위해, 즉 두 번째 옵션에서 레시피의 계산 비용을 측정하기 위해 고안되었습니다. 세 번째 옵션은 `parallel_over = "everything"`을 사용하여 중복 계산을 수행하는 비용을 측정합니다.

우리는 10개의 물리적 코어와 20개의 가상 코어(하이퍼스레딩(hyperthreading)을 통해)가 있는 컴퓨터에서 다양한 수의 워커 프로세스와 두 가지 `parallel_over` 옵션을 사용하여 이 프로세스를 평가했습니다.

먼저 [그림 13-7](#parallel-times)에서 원시(raw) 실행 시간을 고려해 보겠습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1307.png" alt="tmwr 1307" />
<h6 id="figure-13-7.-execution-times-for-model-tuning-versus-the-number-of-workers-using-different-delegation-schemes.-the-diagonal-black-line-indicates-a-linear-speedup-where-the-addition-of-a-new-worker-process-has-maximal-effect.">그림 13-7. 서로 다른 위임(delegation) 구성을 사용한 경우의 워커 수 대비 모델 튜닝 실행 시간. 대각선 검은색 선은 새로운 워커 프로세스의 추가가 최대 효과를 갖는 선형 속도 향상을 나타냅니다.</h6>
</figure>

리샘플이 5개밖에 없었기 때문에 `parallel_over = "resamples"`일 때 사용되는 코어 수는 5개로 제한됩니다.

"none"과 "light"에 대한 처음 두 패널의 곡선을 비교해 보면:

- 패널 간 실행 시간의 차이가 거의 없습니다. 이는 이 데이터의 경우 레시피에서 전처리 단계를 수행하는 데 대한 실제적인 계산상의 불이익(penalty)이 없음을 나타냅니다.

- 많은 코어에서 `parallel_over = "everything"`을 사용하면 몇 가지 이점이 있습니다. 그러나 그림에서 볼 수 있듯이 병렬 처리의 이점 대부분은 처음 5명의 워커에서 발생합니다.

비용이 많이 드는 전처리 단계를 사용하면 실행 시간에 상당한 차이가 있습니다. `parallel_over = "everything"`을 사용하는 것은 모든 코어를 사용하더라도 5개의 코어만 사용하는 `parallel_over = "resamples"`가 도달하는 실행 시간을 달성하지 못하기 때문에 문제가 됩니다. 이는 계산 구성에서 비용이 많이 드는 전처리 단계가 불필요하게 반복되기 때문입니다.

[그림 13-8](#parallel-speedups)에서 속도 향상 측면에서 이 데이터를 볼 수도 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1308.png" alt="tmwr 1308" />
<h6 id="figure-13-8.-speed-ups-for-model-tuning-versus-the-number-of-workers-using-different-delegation-schemes.">그림 13-8. 서로 다른 위임 구성을 사용한 경우의 워커 수 대비 모델 튜닝 속도 향상.</h6>
</figure>

이 데이터의 경우 가장 좋은 속도 향상은 `parallel_over = "resamples"`이고 계산 비용이 많이 들 때 발생합니다. 그러나 후자의 경우 이전 분석에서 전반적인 모델 피팅이 더 느리다는 것을 나타냅니다.

병렬 처리와 함께(in conjunction with) 하위 모델 최적화 방법을 사용하면 어떤 이점이 있을까요? 이 장의 앞부분에 표시된 C5.0 분류 모델도 10명의 워커를 사용하여 병렬로 실행되었습니다. 병렬 계산은 7.5배 속도 향상을 위해 13.3초가 걸렸습니다(두 실행 모두 하위 모델 최적화 트릭을 사용함). 하위 모델 최적화 트릭과 병렬 처리 사이에는 가장 기본적인 그리드 검색 코드에 비해 총 282배의 속도 향상이 있었습니다.

###### 경고 (Warning)

전반적으로 증가된 계산 절감(savings)은 모델마다 다르며 그리드 크기, 리샘플 수 등의 영향도 받습니다. 계산적으로 매우 효율적인 모델은 병렬 처리에서 그만큼의 이점을 얻지 못할 수도 있습니다.

## 전역 변수(Global Variables)에 대한 접근

tidymodels를 사용할 때 로컬 환경(일반적으로 전역(global) 환경)의 값을 모델 객체에 사용하는 것이 가능합니다.

###### 참고 (Note)

여기서 "환경(environment)"은 무엇을 의미할까요? R의 환경을 작업할 수 있는 변수를 저장하는 장소(place)로 생각하십시오. 자세한 내용은 Wickham (2019)의 "Environments" 장을 참조하십시오.

모델 매개변수로 사용할 변수를 정의한 다음 `linear_reg()`와 같은 함수에 전달하면 변수는 일반적으로 전역 환경에 정의됩니다.

```
coef_penalty <- 0.1
spec <- linear_reg(penalty = coef_penalty) %>% set_engine("glmnet")
spec
#> Linear Regression Model Specification (regression)
#>
#> Main Arguments:
#>   penalty = coef_penalty
#>
#> Computational engine: glmnet
```

parsnip 패키지로 생성된 모델은 이와 같은 인수를 *quosures*로 저장합니다; 이것들은 객체의 이름과 그것이 위치한 환경을 모두 추적하는 객체입니다.

```
spec$args$penalty
#> <quosure>
#> expr: ^coef_penalty
#> env:  global
```

이 변수가 전역 환경에서 생성되었으므로 `env: global`을 갖는다는 점에 유의하십시오. `spec`에 의해 정의된 모델 지정은 해당 세션도 전역 환경을 사용하기 때문에 사용자의 정규(regular) 세션에서 실행될 때 올바르게 작동합니다; R은 `coef_penalty` 객체를 쉽게 찾을 수 있습니다.

###### 경고 (Warning)

이러한 모델이 병렬 워커를 사용하여 평가될 경우 실패할 수 있습니다. 병렬 처리에 사용되는 특정 기술에 따라 워커가 전역 환경에 접근하지 못할 수 있습니다.

병렬로 실행될 코드를 작성할 때는 객체에 대한 참조(reference)보다는 실제 데이터를 객체에 삽입하는 것이 좋습니다. 이를 위해 rlang 및 dplyr 패키지가 매우 유용할 수 있습니다. 예를 들어 `!!` 연산자는 객체에 단일 값을 접합(splice)할 수 있습니다.

```
spec <- linear_reg(penalty = !!coef_penalty) %>% set_engine("glmnet")
spec$args$penalty
#> <quosure>
#> expr: ^0.1
#> env:  empty
```

이제 출력은 `^0.1`이며, 객체에 대한 참조 대신 값이 거기에 있음을 나타냅니다. 여러 개의 외부 값을 객체에 삽입해야 하는 경우 `!!!` 연산자가 도움을 줄 수 있습니다.

```
mcmc_args <- list(chains = 3, iter = 1000, cores = 3)

linear_reg() %>% set_engine("stan", !!!mcmc_args)
#> Linear Regression Model Specification (regression)
#>
#> Engine-Specific Arguments:
#>   chains = 3
#>   iter = 1000
#>   cores = 3
#>
#> Computational engine: stan
```

레시피 선택기(selectors)는 전역 변수에 대한 접근을 원할 수 있는 또 다른 장소입니다. 두 번째 광학 채널(optical channel)을 사용하여 측정된 세포 데이터의 모든 예측 변수를 사용해야 하는 레시피 단계가 있다고 가정해 봅시다. 이러한 열 이름의 벡터를 생성할 수 있습니다.

```
library(stringr)
ch_2_vars <- str_subset(names(cells), "ch_2")
ch_2_vars
#> [1] "avg_inten_ch_2"   "total_inten_ch_2"
```

우리는 이들을 레시피 단계에 하드코딩(hard-code)할 수 있지만, 데이터가 변경될 경우에 대비해(in case) 프로그래밍 방식으로(programmatically) 참조하는 것이 더 나을 것입니다. 이를 수행하는 두 가지 방법은 다음과 같습니다.

```
# 여전히 전역 데이터에 대한 참조를 사용합니다 (~_~;)
recipe(class ~ ., data = cells) %>%
  step_spatialsign(all_of(ch_2_vars))
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor         56
#>
#> Operations:
#>
#> Spatial sign on  all_of(ch_2_vars)

# 값을 단계에 삽입합니다 ヽ(•‿•)ノ
recipe(class ~ ., data = cells) %>%
  step_spatialsign(!!!ch_2_vars)
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor         56
#>
#> Operations:
#>
#> Spatial sign on  "avg_inten_ch_2", "total_inten_ch_2"
```

필요한 모든 정보가 레시피 객체에 내장되어(embedded) 있기 때문에 후자가 병렬 처리에 더 좋습니다.

## 경주 방법 (Racing Methods)

그리드 검색의 한 가지 문제는 튜닝 매개변수를 평가하기 전에 모든 리샘플에 걸쳐 모든 모델을 피팅해야 한다는 것입니다. 대신, 튜닝 과정 중 어느 시점에 중간 분석(interim analysis)을 수행하여 정말 끔찍한 매개변수 후보를 제거(eliminate)할 수 있다면 유용할 것입니다. 이는 임상 시험에서의 *무익성 분석(futility analysis)*과 유사할(akin to) 것입니다. 만약 신약이 지나치게 낮은(또는 좋은) 성과를 보인다면, 결정을 내리기 위해 시험이 끝날 때까지 기다리는 것은 잠재적으로 비윤리적일 수 있습니다.

머신 러닝에서는 *경주 방법(racing methods)*이라는 기법이 유사한 기능을 제공합니다(Maron and Moore 1993). 여기서 튜닝 프로세스는 초기 리샘플 하위 집합(subset)에서 모든 모델을 평가합니다. 현재 성능 지표를 기반으로 일부 매개변수 세트는 후속(subsequent) 리샘플에서 고려되지 않습니다.

예를 들어, 이 장에서 살펴본 정규 그리드를 사용한 다층 퍼셉트론 튜닝 프로세스에서 처음 3개의 폴드만 수행한 후 결과는 어떻게 보일까요? [11장](ch11.xhtml#compare)에 표시된 것과 유사한 기법을 사용하여 리샘플링된 ROC 곡선 아래 면적이 결과(outcome)이고 예측 변수가 매개변수 조합에 대한 지시자(indicator)인 모델을 피팅할 수 있습니다. 이 모델은 리샘플 간(resample-to-resample) 효과를 고려(takes into account)하고 각 매개변수 설정(setting)에 대한 점(point) 및 구간(interval) 추정치를 생성합니다. 모델의 결과는 현재 가장 성능이 좋은 매개변수에 상대적인 ROC 값의 손실을 측정하는 일측(one-sided) 95% 신뢰 구간입니다.

[그림 13-9](#racing-process)는 프로세스의 여러 반복에서 결과를 보여줍니다. 첫 번째 반복 패널에 표시된 점은 단일 ROC AUC 값을 보여줍니다. 반복이 진행됨에 따라 점들은 리샘플링된 ROC 통계량의 평균입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1309.png" alt="tmwr 1309" />
<h6 id="figure-13-9.-the-racing-process-for-20-tuning-parameters-and-10-resamples.">그림 13-9. 20개의 튜닝 매개변수와 10개의 리샘플에 대한 경주 프로세스.</h6>
</figure>

세 번째 반복에서 선두(leading) 모델 구성이 변경되었고 알고리즘은 일측 신뢰 구간을 계산합니다. 신뢰 구간이 0을 포함하는 매개변수 세트는 그 성능이 최상의 결과와 통계적으로 다르지 않다는 증거가 부족할(lack evidence) 것입니다. 우리는 14개의 설정을 유지(retain)합니다; 이들은 더 리샘플링됩니다. 나머지 6개의 하위 모델은 더 이상 고려되지 않습니다.

이 프로세스는 남은 구성을 계속 리샘플링하고 현재 결과에 따라 통계 분석이 반복됩니다. 고려 대상에서 더 많은 하위 모델이 제거될 수 있습니다. 최종 리샘플 이전에 거의 모든 하위 모델이 제거되며(eliminated), 마지막 반복에서는 두 개만 남습니다.<sup><a href="ch13.xhtml#idm45881854865424" id="idm45881854865424-marker" data-type="noteref">1</a></sup>

###### 경고 (Warning)

경주 방법은 중간 분석이 빠르고 일부 매개변수 설정이 낮은 성능을 가지는 한 기본적인 그리드 검색보다 더 효율적일 수 있습니다. 또한 모델이 하위 모델 예측을 활용(exploit)할 능력이 _없을 때_ 가장 유용합니다.

finetune 패키지에는 경주용 함수가 포함되어 있습니다. `tune_race_anova()` 함수는 분산 분석(ANOVA) 모델을 수행하여 다른 모델 구성의 통계적 유의성을 테스트합니다. 이전에 표시된 필터링을 재현하는(reproduce) 구문(syntax)은 다음과 같습니다.

```
library(finetune)

set.seed(1308)
mlp_sfd_race <-
  mlp_wflow %>%
  tune_race_anova(
    cell_folds,
    grid = 20,
    param_info = mlp_param,
    metrics = roc_res,
    control = control_race(verbose_elim = TRUE)
  )
```

인수는 `tune_grid()`의 인수를 반영(mirror)합니다. `control_race()` 함수에는 제거(elimination) 절차를 위한 옵션이 있습니다.

[그림 13-9](#racing-process)에 표시된 것처럼 전체 리샘플 세트가 평가된 후 고려 대상인 튜닝 매개변수 조합은 두 개였습니다. `show_best()`는 최상의 모델(성능 순으로 순위가 매겨짐)을 반환하지만 한 번도 제거되지 편집된 구성만 반환합니다.

```
show_best(mlp_sfd_race, n = 10)
#> # A tibble: 2 × 10
#>   hidden_units penalty epochs num_comp .metric .estimator  mean     n std_err
#>          <int>   <dbl>  <int>    <int> <chr>   <chr>      <dbl> <int>   <dbl>
#> 1            8  0.814     177       15 roc_auc binary     0.887    10 0.0103
#> 2            3  0.0402    151       10 roc_auc binary     0.885    10 0.00810
#> # … with 1 more variable: .config <chr>
```

설정을 버리기(discarding) 위한 다른 중간 분석 기법도 있습니다. 예를 들어, Krueger, Panknin, and Braun (2015)은 전통적인 순차(sequential) 분석 방법을 사용하는 반면, Max Kuhn (2014)은 데이터를 스포츠 대회(competition)로 취급하고 브래들리-테리(Bradley-Terry) 모델(Bradley and Terry 1952)을 사용하여 매개변수 설정의 승리 능력(winning ability)을 측정합니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 모델 튜닝에 사용될 수 있는 두 가지 주요 그리드 검색 클래스(정규 및 비정규)에 대해 논의하고, 수동으로 또는 `grid_*()` 함수 제품군을 사용하여 이러한 그리드를 구성하는 방법을 시연했습니다. `tune_grid()` 함수는 리샘플링을 사용하여 이러한 모델 매개변수 후보 세트를 평가할 수 있습니다. 이 장에서는 또한 최종 피팅에 대한 매개변수 값을 업데이트하기 위해 모델, 레시피 또는 워크플로를 마무리하는(finalize) 방법을 보여주었습니다. 그리드 검색은 계산 비용이 많이 들 수 있지만 이러한 검색의 실험 설계에서 신중한 선택을 통해 다루기 쉽게(tractable) 만들 수 있습니다.

다음 장에서 재사용될 데이터 분석 코드는 다음과 같습니다.

```
library(tidymodels)

data(cells)
cells <- cells %>% select(-case)

set.seed(1304)
cell_folds <- vfold_cv(cells)

roc_res <- metric_set(roc_auc)
```

<sup>[1](ch13.xhtml#idm45881854865424-marker)</sup> 이 접근 방식의 계산적 측면에 대한 자세한 내용은 Max Kuhn (2014)을 참조하십시오.
