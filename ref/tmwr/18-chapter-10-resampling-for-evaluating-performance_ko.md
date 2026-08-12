# 10장. 성능 평가를 위한 리샘플링 (Resampling for Evaluating Performance)

우리는 모델의 성능을 평가하기 위해 조합해야 하는 여러 부분을 이미 다루었습니다. [9장](ch09.xhtml#performance)에서는 모델 성능을 측정하기 위한 통계를 설명했습니다. [5장](ch05.xhtml#splitting)에서는 데이터 소비(data spending)의 개념을 소개했으며, 편향되지 않은(unbiased) 성능 추정치를 얻기 위해 테스트 세트를 사용할 것을 권장했습니다. 그러나 우리는 대개 _테스트 세트를 사용하기 전에_ 단일 모델 또는 여러 모델의 성능을 이해해야 합니다.

###### 경고 (Warning)

일반적으로 우리는 먼저 모델 성능을 평가하기 전에는 테스트 세트와 함께 사용할 최종 모델을 결정할 수 없습니다. 성능을 신뢰할 수 있게 측정해야 할 필요성과 우리가 사용할 수 있는 데이터 분할(훈련 및 테스트) 사이에는 간극(gap)이 있습니다.

이 장에서는 이러한 간극을 메울 수 있는 리샘플링(resampling)이라는 접근 방식을 설명합니다. 리샘플링 성능 추정치는 테스트 세트의 추정치와 유사한 방식으로 새로운 데이터에 일반화(generalize)될 수 있습니다. [11장](ch11.xhtml#compare)은 리샘플링 결과를 비교하는 통계적 방법을 시연하여 이 장을 보완합니다.

리샘플링의 가치를 완전히 이해하기 위해, 먼저 자주 실패할 수 있는 재대입(resubstitution) 접근 방식을 살펴보겠습니다.

# 재대입 접근 방식 (The Resubstitution Approach)

(새로운 데이터나 테스트 데이터가 아닌) 훈련에 사용했던 것과 동일한 데이터로 성능을 측정할 때, 우리는 데이터를 *재대입(resubstituted)*했다고 말합니다. 이 개념을 설명하기 위해 Ames 데이터를 다시 사용해 보겠습니다. [8장](ch08.xhtml#recipes)의 끝부분에는 현재 Ames 분석 상태가 요약되어 있습니다. 여기에는 `ames_rec`라는 이름의 레시피 객체, 선형 모델, 그리고 해당 레시피와 모델을 사용하는 `lm_wflow`라는 워크플로가 포함됩니다. 이 워크플로는 훈련 세트에 피팅되어 `lm_fit`이라는 결과를 낳았습니다.

이 선형 모델과 비교하기 위해 다른 유형의 모델을 피팅할 수도 있습니다. *랜덤 포레스트(Random forests)*는 훈련 세트의 약간씩 다른 버전들로부터 다수의 의사 결정 나무(decision trees)를 생성하여 작동하는 트리 앙상블(tree ensemble) 방법입니다(Breiman 2001a). 이러한 나무들의 집합이 앙상블을 구성합니다. 새로운 샘플을 예측할 때 앙상블의 각 멤버는 별도의 예측을 수행합니다. 이들을 평균화하여 새로운 데이터 포인트에 대한 최종 앙상블 예측을 만듭니다.

랜덤 포레스트 모델은 매우 강력하며, 기반 데이터 패턴을 매우 밀접하게 모방(emulate)할 수 있습니다. 이 모델은 계산 집약적일 수 있지만 유지 관리가 매우 적습니다; 즉 (부록([Appendix](app01.xhtml#pre-proc-table))에 문서화된 대로) 전처리가 거의 필요하지 않습니다.

(추가 전처리 단계 없이) 선형 모델과 동일한 예측 변수 집합을 사용하여, `"ranger"` 엔진(계산에 ranger R 패키지를 사용함)을 통해 훈련 세트에 랜덤 포레스트 모델을 피팅할 수 있습니다. 이 모델은 전처리가 필요하지 않으므로 다음과 같이 간단한 공식을 사용할 수 있습니다.

```
rf_model <-
  rand_forest(trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

rf_wflow <-
  workflow() %>%
  add_formula(
    Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
      Latitude + Longitude) %>%
  add_model(rf_model)

rf_fit <- rf_wflow %>% fit(data = ames_train)
```

선형 모델과 랜덤 포레스트 모델을 어떻게 비교해야 할까요? 시연을 위해 우리는 훈련 세트를 예측하여 이른바 _외견상 지표(apparent metric)_ 또는 *재대입 지표(resubstitution metric)*를 생성할 것입니다. 이 함수는 예측을 생성하고 결과를 형식화합니다.

```
estimate_perf <- function(model, dat) {
  # `model` 및 `dat` 객체의 이름을 캡처합니다
  cl <- match.call()
  obj_name <- as.character(cl$model)
  data_name <- as.character(cl$dat)
  data_name <- gsub("ames_", "", data_name)

  # 다음 지표들을 추정합니다.
  reg_metrics <- metric_set(rmse, rsq)

  model %>%
    predict(dat) %>%
    bind_cols(dat %>% select(Sale_Price)) %>%
    reg_metrics(Sale_Price, .pred) %>%
    select(-.estimator) %>%
    mutate(object = obj_name, data = data_name)
}
```

RMSE와 _R_<sup>2</sup>가 모두 계산됩니다. 재대입 통계는 다음과 같습니다.

```
estimate_perf(rf_fit, ames_train)
#> # A tibble: 2 × 4
#>   .metric .estimate object data
#>   <chr>       <dbl> <chr>  <chr>
#> 1 rmse       0.0365 rf_fit train
#> 2 rsq        0.960  rf_fit train
estimate_perf(lm_fit, ames_train)
#> # A tibble: 2 × 4
#>   .metric .estimate object data
#>   <chr>       <dbl> <chr>  <chr>
#> 1 rmse       0.0754 lm_fit train
#> 2 rsq        0.816  lm_fit train
```

이 결과를 바탕으로 보면 랜덤 포레스트가 판매 가격 예측 능력이 훨씬 뛰어납니다; RMSE 추정치가 선형 회귀보다 두 배 더 좋습니다. 이 가격 예측 문제를 위해 이 두 모델 중 하나를 선택해야 한다면, 우리가 사용하는 로그 척도에서 RMSE가 약 절반 크기이기 때문에 아마도 랜덤 포레스트를 선택할 파티션입니다. 다음 단계는 최종 검증을 위해 랜덤 포레스트 모델을 테스트 세트에 적용하는 것입니다.

```
estimate_perf(rf_fit, ames_test)
#> # A tibble: 2 × 4
#>   .metric .estimate object data
#>   <chr>       <dbl> <chr>  <chr>
#> 1 rmse       0.0704 rf_fit test
#> 2 rsq        0.852  rf_fit test
```

테스트 세트의 RMSE 추정치 0.0704는 훈련 세트 값인 0.0365보다 _훨씬 더 나쁩니다_! 왜 이런 일이 발생했을까요?

많은 예측 모델은 데이터로부터 복잡한 추세를 학습할 수 있습니다. 통계학에서 이들은 보통 *편향이 낮은 모델(low bias models)*로 불립니다.

###### 참고 (Note)

이 맥락에서 *편향(bias)*은 데이터의 진정한(true) 패턴이나 관계와 모델이 모방할 수 있는 패턴의 유형 간의 차이입니다. 많은 블랙박스 머신러닝 모델은 편향이 낮아, 이는 그들이 복잡한 관계를 재현(reproduce)할 수 있음을 의미합니다. 다른 모델들(선형/로지스틱 회귀, 판별 분석(discriminant analysis) 및 기타)은 그만큼 적응성이 높지 않으며 _편향이 높은(high bias)_ 모델로 간주됩니다.<sup><a href="ch10.xhtml#idm45881862747584" id="idm45881862747584-marker" data-type="noteref">1</a></sup>

편향이 낮은 모델의 경우 높은 예측 능력으로 인해 때때로 모델이 훈련 세트 데이터를 거의 암기(memorizing)하는 결과를 낳을 수 있습니다. 확실한 예로 이웃이 하나뿐인 KNN 모델을 생각해 보십시오. 이것은 다른 데이터 세트에 대해 실제로 얼마나 잘 작동하는지에 관계없이 훈련 세트에 대해 항상 완벽한 예측을 제공할 것입니다. 랜덤 포레스트 모델도 이와 유사합니다; 훈련 세트를 다시 예측하는 것은 항상 인위적으로 낙관적인 성능 추정치를 초래할 것입니다.

두 모델 모두에 대해 [표 10-1](#rmse-results)은 훈련 및 테스트 세트에 대한 RMSE 추정치를 요약합니다.

| object | Train  | Test   |
| ------ | ------ | ------ |
| lm_fit | 0.0754 | 0.0736 |
| rf_fit | 0.0365 | 0.0704 |

표 10-1. 훈련 및 테스트 세트에 대한 성능 통계. {#rmse-results}

선형 회귀 모델은 복잡성이 제한되어 있기 때문에 훈련과 테스트 간에 일관성이 있음에 주목하십시오.<sup><a href="ch10.xhtml#idm45881862678160" id="idm45881862678160-marker" data-type="noteref">2</a></sup>

###### 경고 (Warning)

이 예제의 주요 시사점(takeaway)은 훈련 세트를 다시 예측하면 성능에 대해 인위적으로 낙관적인 추정치가 나온다는 것입니다. 대부분의 모델에서 이것은 좋은 생각이 아닙니다.

테스트 세트를 즉시 사용해서는 안 되고 훈련 세트를 다시 예측하는 것이 나쁜 생각이라면 어떻게 해야 할까요? 교차 검증(cross-validation)이나 검증 세트(validation sets)와 같은 리샘플링 방법이 해결책입니다.

# 리샘플링 방법 (Resampling Methods)

리샘플링 방법은 모델링을 위해 일부 데이터를 사용하고 평가를 위해 다른 데이터를 사용하는 프로세스를 모방하는 경험적 시뮬레이션 시스템입니다. 대부분의 리샘플링 방법은 반복적(iterative)이며, 이는 이 프로세스가 여러 번 반복됨을 의미합니다. [그림 10-1](#resampling-scheme)의 다이어그램은 리샘플링 방법이 일반적으로 어떻게 작동하는지 보여줍니다.

[그림 10-1](#resampling-scheme)에서 볼 수 있듯이 리샘플링은 훈련 세트에서만 수행됩니다. 테스트 세트는 관여하지 않습니다. 리샘플링의 각 반복(iteration)마다 데이터는 두 개의 하위 표본(subsamples)으로 분할(partitioned)됩니다.

- 모델은 *분석 세트(analysis set)*로 피팅됩니다.

- 모델은 *평가 세트(assessment set)*로 평가됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1001.png" alt="tmwr 1001" />
<h6 id="figure-10-1.-data-splitting-scheme-from-the-initial-data-split-to-resampling.">그림 10-1. 초기 데이터 분할에서 리샘플링까지의 데이터 분할 체계.</h6>
</figure>

이 두 하위 표본은 훈련 및 테스트 세트와 다소 유사합니다. 우리는 초기 데이터 분할과의 혼동을 피하기 위해 _분석_ 및 *평가*라는 용어를 사용합니다. 이러한 데이터 세트는 상호 배타적(mutually exclusive)입니다. 분석 및 평가 세트를 생성하는 데 사용되는 분할 방식(partitioning scheme)이 대개 해당 방법을 정의하는 특성입니다.

리샘플링을 20번 반복한다고 가정해 봅시다. 이것은 분석 세트들에 대해 20개의 별도 모델이 피팅되고, 대응하는 평가 세트들이 20개의 성능 통계 세트를 생성함을 의미합니다. 모델에 대한 최종 성능 추정치는 이 통계의 20개 반복(replicates)의 평균입니다. 이 평균은 매우 좋은 일반화(generalization) 특성을 가지며 재대입 추정치보다 훨씬 뛰어납니다.

다음 섹션에서는 일반적으로 사용되는 여러 리샘플링 방법을 정의하고 각각의 장단점을 논의합니다.

## 교차 검증 (Cross-Validation)

교차 검증은 잘 확립된(well established) 리샘플링 방법입니다. 여러 가지 변형(variations)이 있지만 일반적인 교차 검증 방법은 _V_-겹 교차 검증(_V_-fold cross-validation)입니다. 데이터는 대략 동일한 크기의 *V*개 세트(이를 *폴드(folds)*라고 함)로 무작위 분할됩니다. 설명을 위해 무작위 폴드 할당이 있는 30개의 훈련 세트 포인트의 데이터 세트에 대해 _V_ = 3인 경우가 [그림 10-2](#cross-validation-allocation)에 나타나 있습니다. 기호 안의 숫자는 샘플 번호입니다.

[그림 10-2](#cross-validation-allocation)에서 기호의 음영(shade)은 무작위로 할당된 폴드를 나타냅니다. ([5장](ch05.xhtml#splitting)에서 이미 논의한) 층화 샘플링(Stratified sampling) 역시 폴드 할당을 위한 옵션입니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1002.png" alt="tmwr 1002" />
<h6 id="figure-10-2.-v-fold-cross-validation-randomly-assigns-data-to-folds.">그림 10-2. V-겹 교차 검증은 데이터를 폴드에 무작위로 할당합니다.</h6>
</figure>

3-겹(three-fold) 교차 검증의 경우 리샘플링의 세 가지 반복이 [그림 10-3](#cross-validation)에 설명되어 있습니다. 각 반복마다 한 폴드가 평가 통계를 위해 제외되고(held out) 나머지 폴드가 모델을 위한 기반이 됩니다. 이 과정이 각 폴드에 대해 계속되어 세 개의 모델이 세 세트의 성능 통계를 생성합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1003.png" alt="tmwr 1003" />
<h6 id="figure-10-3.-v-fold-cross-validation-data-usage.">그림 10-3. V-겹 교차 검증 데이터 사용.</h6>
</figure>

_V_ = 3일 때, 분석 세트는 훈련 세트의 2/3이고 각 평가 세트는 구분되는(distinct) 1/3입니다. 성능에 대한 최종 리샘플링 추정치는 *V*개의 각 반복(replicates)을 평균냅니다.

_V_ = 3을 사용하는 것은 교차 검증을 설명하기 위한 좋은 선택이지만 신뢰할 수 있는 추정치를 생성하기에는 너무 낮기 때문에 실제로는 잘못된 선택입니다. 실제 환경에서 _V_ 값은 흔히 5 또는 10입니다; 우리는 10-겹 교차 검증이 대부분의 상황에서 좋은 결과를 얻기에 충분히 크기 때문에 일반적으로 기본값(default)으로 선호합니다.

###### 참고 (Note)

*V*를 변경하면 어떤 효과가 있을까요? 값이 크면 편향이 작은 리샘플링 추정치가 나오지만 분산(variance)이 커집니다. 더 작은 _V_ 값은 편향이 크지만 분산이 작습니다. 우리는 반복을 통해 잡음(noise)은 줄어들지만 편향은 줄어들지 않기 때문에 10-겹을 선호합니다.<sup><a href="ch10.xhtml#idm45881862631264" id="idm45881862631264-marker" data-type="noteref">3</a></sup>

주요 입력(input)은 폴드 수(기본값 10)와 함께 훈련 세트 데이터 프레임입니다.

```
set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 10)
ames_folds
#> #  10-fold cross-validation
#> # A tibble: 10 × 2
#>   splits             id
#>   <list>             <chr>
#> 1 <split [2107/235]> Fold01
#> 2 <split [2107/235]> Fold02
#> 3 <split [2108/234]> Fold03
#> 4 <split [2108/234]> Fold04
#> 5 <split [2108/234]> Fold05
#> 6 <split [2108/234]> Fold06
#> # … 4개 행이 더 있습니다
```

`splits`라는 이름의 열에는 데이터를 분할하는 방법에 대한 정보가 포함되어 있습니다(초기 훈련/테스트 파티션을 만드는 데 사용된 객체와 유사함). `splits`의 각 행에는 전체 훈련 세트의 내장된 사본(embedded copy)이 있지만, R은 메모리에 데이터의 복사본을 만들지 않을 만큼 똑똑합니다.<sup><a href="ch10.xhtml#idm45881862579184" id="idm45881862579184-marker" data-type="noteref">4</a></sup> 티블 내부의 print 메서드는 각 빈도(frequency)를 표시합니다. `[2107/235]`는 분석 세트에 대략 2000개의 샘플이 있고 해당 특정 평가 세트에는 235개가 있음을 나타냅니다.

이러한 객체에는 항상 파티션에 레이블을 지정하는 `id`라는 문자형(character) 열이 포함되어 있습니다.<sup><a href="ch10.xhtml#idm45881862621792" id="idm45881862621792-marker" data-type="noteref">5</a></sup>

분할된 데이터를 수동으로 검색하기 위해, `analysis()` 및 `assessment()` 함수가 해당 데이터 프레임을 반환합니다.

```
# 첫 번째 폴드에 대해:
ames_folds$splits[[1]] %>% analysis() %>% dim()
#> [1] 2107   74
```

[tune](https://oreil.ly/WdI3T)과 같은 tidymodels 패키지에는 고수준(high-level) 사용자 인터페이스가 포함되어 있어 일상적인 작업에 `analysis()`와 같은 함수가 일반적으로 필요하지 않습니다. [10장](#resampling)은 이러한 리샘플에 대해 모델을 피팅하는 함수를 시연합니다.

다양한 교차 검증의 변형이 있습니다; 우리는 중요한 것들을 살펴볼 것입니다.

## 반복 교차 검증 (Repeated Cross-Validation)

교차 검증의 중요한 변형은 반복(repeated) _V_-겹 교차 검증입니다. 데이터 크기 또는 기타 특성에 따라 _V_-겹 교차 검증으로 생성된 리샘플링 추정치는 과도하게 잡음이 많을 수 있습니다(excessively noisy).<sup><a href="ch10.xhtml#idm45881862534224" id="idm45881862534224-marker" data-type="noteref">6</a></sup> 많은 통계 문제와 마찬가지로 잡음을 줄이는 한 가지 방법은 데이터를 더 수집하는 것입니다. 교차 검증의 경우 이는 *V*개 이상의 통계를 평균화하는 것을 의미합니다.

_V_-겹 교차 검증의 *R*회 반복(repeats)을 만들기 위해, 동일한 폴드 생성 프로세스를 *R*번 수행하여 *V*개 파티션의 *R*개 모음(collections)을 생성합니다. 이제 *V*개의 통계를 평균화하는 대신 $`V \times R`$개의 통계가 최종 리샘플링 추정치를 생성합니다. 중심극한정리(Central Limit Theorem)에 의해, $`V \times R`$에 비해 데이터가 많기만 하다면 각 모델의 요약 통계는 정규 분포를 향해 가는 경향이 있습니다.

Ames 데이터를 고려해 봅시다. 평균적으로 10-겹 교차 검증은 약 234개의 속성을 포함하는 평가 세트를 사용합니다. 만약 선택한 통계가 RMSE라면 우리는 그 추정치의 표준 편차를 $`\sigma`$로 표시할 수 있습니다. 단순 10-겹 교차 검증을 사용하면 평균 RMSE의 표준 오차는 $`\sigma/\sqrt{10}`$입니다. 이것이 너무 잡음이 많다면 반복은 표준 오차를 $`\sigma/\sqrt{10R}`$로 줄입니다. $`R`$개의 반복이 있는 10-겹 교차 검증의 경우, [그림 10-4](#variance-reduction)의 플롯은 반복에 따라 표준 오차<sup><a href="ch10.xhtml#idm45881862495648" id="idm45881862495648-marker" data-type="noteref">7</a></sup>가 얼마나 빨리 감소하는지 보여줍니다.

반복 횟수가 많아질수록 표준 오차에 미치는 영향은 덜한 경향이 있습니다. 그러나 기본 $`\sigma`$ 값이 비현실적으로 크다면(impractically large), 수확 체감(diminishing returns)에도 불구하고 추가적인 계산 비용을 들일 가치가 있을 수 있습니다.

반복(repeats)을 만들려면 추가적인 `repeats` 인수와 함께 `vfold_cv()`를 호출합니다.

```
vfold_cv(ames_train, v = 10, repeats = 5)
#> #  10-fold cross-validation repeated 5 times
#> # A tibble: 50 × 3
#>   splits             id      id2
#>   <list>             <chr>   <chr>
#> 1 <split [2107/235]> Repeat1 Fold01
#> 2 <split [2107/235]> Repeat1 Fold02
#> 3 <split [2108/234]> Repeat1 Fold03
#> 4 <split [2108/234]> Repeat1 Fold04
#> 5 <split [2108/234]> Repeat1 Fold05
#> 6 <split [2108/234]> Repeat1 Fold06
#> # … 44개 행이 더 있습니다
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1004.png" alt="tmwr 1004" />
<h6 id="figure-10-4.-relationship-between-the-relative-variance-in-performance-estimates-versus-the-number-of-cross-validation-repeats.">그림 10-4. 성능 추정치의 상대적 분산과 교차 검증 반복 횟수 간의 관계.</h6>
</figure>

## 하나 남기기 교차 검증 (Leave-One-Out Cross-Validation)

교차 검증의 한 가지 변형은 하나 남기기(leave-one-out, LOO) 교차 검증으로, 여기서 *V*는 훈련 세트의 데이터 포인트 수입니다. 훈련 세트 샘플이 $`n`$개인 경우, 훈련 세트의 $`n - 1`$개 행을 사용하여 $`n`$개의 모델이 피팅됩니다. 각 모델은 제외된 단일 데이터 포인트를 예측합니다. 리샘플링이 끝나면 $`n`$개의 예측이 풀링(pooled)되어 단일 성능 통계를 생성합니다.

하나 남기기(Leave-one-out) 방법은 거의 다른 모든 방법과 비교할 때 부족함이 있습니다. 병적으로(pathologically) 적은 표본을 제외한 모든 경우에 LOO는 계산적으로 과도하며 우수한 통계적 특성을 가지지 않을 수 있습니다. rsample 패키지에 `loo_cv()` 함수가 포함되어 있긴 하지만, 이러한 객체들은 일반적으로 더 넓은 tidymodels 프레임워크에 통합되어 있지 않습니다.

## 몬테카를로 교차 검증 (Monte Carlo Cross-Validation)

_V_-겹 교차 검증의 또 다른 변형은 몬테카를로 교차 검증(MCCV, Xu and Liang, 2001)입니다. _V_-겹 교차 검증과 마찬가지로 이것은 데이터의 고정된 비율을 평가 세트에 할당합니다. MCCV와 일반 교차 검증의 차이점은 MCCV의 경우 이러한 데이터 비율이 매번 무작위로 선택된다는 것입니다. 이는 상호 배타적이지 않은 평가 세트들을 낳습니다. 이러한 리샘플링 객체를 생성하려면:

```
mc_cv(ames_train, prop = 9/10, times = 20)
#> # Monte Carlo cross-validation (0.9/0.1) with 20 resamples
#> # A tibble: 20 × 2
#>   splits             id
#>   <list>             <chr>
#> 1 <split [2107/235]> Resample01
#> 2 <split [2107/235]> Resample02
#> 3 <split [2107/235]> Resample03
#> 4 <split [2107/235]> Resample04
#> 5 <split [2107/235]> Resample05
```

#> 6 <split [2107/235]> Resample06
#> # … 14개 행이 더 있습니다

```

## 검증 세트 (Validation Sets)

[5장](ch05.xhtml#splitting)에서 테스트 세트와 분리하여 성능을 추정하기 위해 따로 떼어놓는(set aside) 단일 파티션인 검증 세트의 사용에 대해 간략하게 논의했습니다. 검증 세트를 사용할 때 초기 사용 가능한 데이터 세트는 훈련 세트, 검증 세트 및 테스트 세트로 분할됩니다([그림 10-5](#three-way-split) 참조).

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1005.png" alt="tmwr 1005" />
<h6 id="figure-10-5.-a-three-way-initial-split-into-training-testing-and-validation-sets.">그림 10-5. 훈련, 테스트, 그리고 검증 세트로의 초기 3분할(three-way split).</h6>
</figure>

검증 세트는 원본 데이터 풀(pool)이 매우 클 때 종종 사용됩니다. 이 경우 여러 리샘플링 반복을 수행할 필요 없이 단일 대규모 파티션으로 모델 성능의 특성을 규명하기에 충분할 수 있습니다.

rsample 패키지를 사용하면 검증 세트는 다른 리샘플링 객체와 같습니다; 이 유형은 단일 반복만 있다는 점만 다릅니다.<sup><a href="ch10.xhtml#idm45881862367296" id="idm45881862367296-marker" data-type="noteref">8</a></sup> [그림 10-6](#validation-split)은 이 방식을 보여줍니다.

모델 피팅에 데이터의 3/4을 사용하는 검증 세트 객체를 만들려면:

```

set.seed(1002)
val_set <- validation_split(ames_train, prop = 3/4)
val_set
#> # Validation Set Split (0.75/0.25)
#> # A tibble: 1 × 2
#> splits id
#> <list> <chr>
#> 1 <split [1756/586]> validation

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1006.png" alt="tmwr 1006" />
<h6 id="figure-10-6.-a-two-way-initial-split-into-training-and-testing-with-an-additional-validation-set-split-on-the-training-set.">그림 10-6. 훈련 세트에 추가적인 검증 세트 분할이 포함된 훈련 및 테스트로의 초기 2분할(two-way split).</h6>
</figure>

## 부트스트래핑 (Bootstrapping)

부트스트랩(Bootstrap) 리샘플링은 이론적 특성이 다루기 어려운(intractable) 통계량의 표집 분포(sampling distribution)를 근사화(approximating)하기 위한 방법으로 원래 고안되었습니다(Davison and Hinkley 1997). 이를 사용하여 모델 성능을 추정하는 것은 이 방법의 이차적인(secondary) 응용(application)입니다.

훈련 세트의 부트스트랩 샘플은 훈련 세트와 크기가 같지만 *복원(with replacement)* 추출된 샘플입니다. 즉, 일부 훈련 세트 데이터 포인트가 분석 세트에 여러 번 선택됩니다. 각 데이터 포인트가 적어도 한 번 훈련 세트에 포함될 확률은 63.2%입니다. 평가 세트에는 분석 세트로 선택되지 않은 모든 훈련 세트 샘플이 포함됩니다(평균적으로 훈련 세트의 36.8%). 부트스트래핑을 할 때 평가 세트를 종종 *아웃오브백(out-of-bag, OOB)* 샘플이라고 부릅니다.

30개 샘플의 훈련 세트에 대한 세 개의 부트스트랩 샘플의 개략도가 [그림 10-7](#bootstrapping)에 나와 있습니다. 평가 세트의 크기가 다름에 유의하십시오. rsample 패키지를 사용하여 다음과 같은 부트스트랩 리샘플을 만들 수 있습니다.

```

bootstraps(ames_train, times = 5)
#> # Bootstrap sampling
#> # A tibble: 5 × 2
#> splits id
#> <list> <chr>
#> 1 <split [2342/858]> Bootstrap1
#> 2 <split [2342/855]> Bootstrap2
#> 3 <split [2342/852]> Bootstrap3
#> 4 <split [2342/851]> Bootstrap4
#> 5 <split [2342/867]> Bootstrap5

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1007.png" alt="tmwr 1007" />
<h6 id="figure-10-7.-bootstrapping-data-usage.">그림 10-7. 부트스트래핑 데이터 사용.</h6>
</figure>

부트스트랩 샘플은 교차 검증과 달리 분산(variance)이 매우 낮은 성능 추정치를 생성하지만 상당한 비관적 편향(pessimistic bias)이 있습니다. 이는 모델의 실제(true) 정확도가 90%인 경우 부트스트랩은 그 값을 90% 미만으로 추정하는 경향이 있음을 의미합니다. 편향의 양은 충분한 정확도로 경험적으로 결정할 수 없습니다. 또한 편향의 양은 성능 지표의 척도에 따라 달라집니다. 예를 들어, 정확도가 90%일 때와 70%일 때 편향이 다를 가능성이 높습니다.

부트스트랩은 또한 많은 모델 내부에서 사용됩니다. 예를 들어 앞서 언급한 랜덤 포레스트 모델에는 1,000개의 개별 의사 결정 나무(decision trees)가 포함되어 있었습니다. 각 트리는 훈련 세트의 다른 부트스트랩 샘플의 산물(product)이었습니다.

## 롤링 예측 출처 리샘플링 (Rolling Forecasting Origin Resampling)

데이터에 강력한 시간 구성 요소(time component)가 있는 경우 리샘플링 방법은 데이터 내의 계절적(seasonal) 및 기타 시간적(temporal) 추세를 추정하기 위한 모델링을 지원해야 합니다. 훈련 세트에서 값을 무작위로 추출하는(samples) 기술은 모델이 이러한 패턴을 추정하는 능력을 저해할(disrupt) 수 있습니다.

롤링 예측 출처(Rolling forecast origin) 리샘플링(Hyndman and Athanasopoulos 2018)은 실제 시계열 데이터가 종종 파티션되는 방식을 모방하여 과거(historical) 데이터로 모델을 추정하고 최신 데이터로 평가하는 방법을 제공합니다. 이 유형의 리샘플링에서는 초기 분석 및 평가 세트의 크기가 지정됩니다. 리샘플링의 첫 번째 반복은 시리즈의 처음부터 시작하여 이 크기를 사용합니다. 두 번째 반복은 동일한 데이터 크기를 사용하지만 정해진 수의 샘플만큼 이동(shifts over)합니다.

예를 들어 15개의 샘플로 구성된 훈련 세트를 분석 크기 8개 샘플, 평가 세트 크기 3개 샘플로 리샘플링했습니다. 두 번째 반복에서는 첫 번째 훈련 세트 샘플을 버리고 두 데이터 세트 모두 하나씩 앞으로 이동합니다. 이 구성은 [그림 10-8](#rolling)에 표시된 대로 5개의 리샘플을 낳습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1008.png" alt="tmwr 1008" />
<h6 id="figure-10-8.-data-usage-for-rolling-forecasting-origin-resampling.">그림 10-8. 롤링 예측 출처 리샘플링을 위한 데이터 사용.</h6>
</figure>

이 방법에는 두 가지 다른 구성(configurations)이 있습니다.

- 분석 세트는 (크기가 동일하게 유지되는 대신) 누적(cumulatively)해서 커질 수 있습니다. 첫 번째 초기 분석 세트 이후 이전 데이터를 버리지 않고(discarding) 새 샘플이 누적(accrue)될 수 있습니다.

- 리샘플이 1씩 증가할 필요는 없습니다. 예를 들어 대규모 데이터 세트의 경우 증가 블록(incremental block)은 하루가 아닌 일주일이나 한 달이 될 수 있습니다.

1년 분량의 데이터에서 30일 블록의 세트 6개가 분석 세트를 정의한다고 가정해 보겠습니다. 29일을 건너뛰는(skip) 30일 평가 세트의 경우, rsample 패키지를 사용하여 다음과 같이 지정할 수 있습니다.

```

time_slices <-
tibble(x = 1:365) %>%
rolling_origin(initial = 6 \* 30, assess = 30, skip = 29, cumulative = FALSE)

data_range <- function(x) {
summarize(x, first = min(x), last = max(x))
}

map_dfr(time_slices$splits, ~   analysis(.x) %>% data_range())
#> # A tibble: 6 × 2
#>   first  last
#>   <int> <int>
#> 1     1   180
#> 2    31   210
#> 3    61   240
#> 4    91   270
#> 5   121   300
#> 6   151   330
map_dfr(time_slices$splits, ~ assessment(.x) %>% data_range())
#> # A tibble: 6 × 2
#> first last
#> <int> <int>
#> 1 181 210
#> 2 211 240
#> 3 241 270
#> 4 271 300
#> 5 301 330
#> 6 331 360

```

# 성능 추정 (Estimating Performance)

이 장에서 논의된 리샘플링 방법 중 어느 것이든 (전처리, 모델 피팅 등을 포함하는) 모델링 프로세스를 평가하는 데 사용할 수 있습니다. 이러한 방법들은 다른 그룹의 데이터가 모델을 훈련하고 모델을 평가하는 데 사용되기 때문에 효과적입니다. 다시 강조하지만, 리샘플링을 사용하는 과정은 다음과 같습니다.

1.  리샘플링 중에 분석 세트를 사용하여 데이터를 전처리하고 자체에 그 전처리를 적용한 후 이렇게 처리된 데이터를 사용하여 모델을 피팅합니다.

2.  분석 세트에서 생성된 전처리 통계가 평가 세트에 적용됩니다. 평가 세트의 예측값이 새 데이터에 대한 성능을 추정합니다.

이 과정(sequence)은 모든 리샘플에 대해 반복됩니다. *B*개의 리샘플이 있는 경우 각 성능 지표에 대해 *B*개의 반복(replicates)이 있습니다. 최종 리샘플링 추정치는 이러한 *B*개 통계의 평균입니다. 검증 세트와 같이 *B* = 1인 경우 개별 통계가 전체 성능을 나타냅니다.

`rf_wflow` 객체에 포함된 이전 랜덤 포레스트 모델을 다시 고려해 보겠습니다. `fit_resamples()` 함수는 `fit()`과 유사하지만, `data` 인수를 갖는 대신, 이 장에서 설명한 것과 같은 `rset` 객체를 필요로 하는 `resamples`를 갖습니다. 이 함수에 대해 가능한 인터페이스들은 다음과 같습니다.

```

model_spec %>% fit_resamples(formula, resamples, ...)
model_spec %>% fit_resamples(recipe, resamples, ...)
workflow %>% fit_resamples( resamples, ...)

```

다음과 같은 다양한 추가 선택(optional) 인수들이 있습니다.

`metrics`
계산할 성능 통계 지표 세트(metric set)입니다. 기본적으로 회귀 모델은 RMSE 및 *R*<sup>2</sup>를 사용하는 반면 분류 모델은 ROC 곡선 아래 면적과 전체 정확도를 계산합니다. 이 선택은 모델 평가 중에 어떤 예측이 생성되는지도 정의한다는 점에 유의하십시오. 분류의 경우 정확도만 요청하면 (필요하지 않으므로) 평가 세트에 대한 클래스 확률 추정치가 생성되지 않습니다.

`control`
`control_resamples()`에 의해 생성된 다양한 옵션이 포함된 목록(list)입니다. control 인수에는 다음이 포함됩니다.

`verbose`
로깅 인쇄(printing)를 위한 논리값(logical)입니다.

`extract`
(이 장 후반부에서 논의할) 각 모델 반복(iteration)에서 객체를 유지하기(retaining) 위한 함수입니다.

`save_pred`
평가 세트 예측을 저장하기 위한 논리값입니다.

우리 예제의 경우, 모델 피팅과 잔차(residuals)를 시각화하기 위해 예측을 저장해 보겠습니다.

```

keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

set.seed(1003)
rf_res <-
rf_wflow %>%
fit_resamples(resamples = ames_folds, control = keep_pred)
rf_res
#> # Resampling results
#> # 10-fold cross-validation
#> # A tibble: 10 × 5
#> splits id .metrics .notes .predictions
#> <list> <chr> <list> <list> <list>
#> 1 <split [2107/235]> Fold01 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [235 × 4]>
#> 2 <split [2107/235]> Fold02 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [235 × 4]>
#> 3 <split [2108/234]> Fold03 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [234 × 4]>
#> 4 <split [2108/234]> Fold04 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [234 × 4]>
#> 5 <split [2108/234]> Fold05 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [234 × 4]>
#> 6 <split [2108/234]> Fold06 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [234 × 4]>
#> # … 4개 행이 더 있습니다

```

반환 값(return value)은 입력된 리샘플과 유사한 티블과 몇 가지 추가 열(columns)입니다.

`.metrics`
이것은 평가 세트 성능 통계가 포함된 티블의 리스트(list) 열입니다.

`.notes`
이것은 리샘플링 중 생성된 경고(warnings) 또는 오류를 목록화(cataloging)하는 또 다른 티블 리스트 열입니다. 이러한 오류는 이후 리샘플링 실행을 멈추게 하지 않습니다.

`.predictions`
`save_pred = TRUE`일 때 존재하며, 이 리스트 열에는 표본 외(out-of-sample) 예측을 포함하는 티블이 포함되어 있습니다.

이 리스트 열들은 위압적(daunting)으로 보일 수 있지만 tidyr를 사용하거나 tidymodels가 제공하는 편의(convenience) 함수를 사용하여 쉽게 재구성할 수 있습니다. 예를 들어 더 유용한 형식으로 성능 지표를 반환하려면:

```

collect_metrics(rf_res)
#> # A tibble: 2 × 6
#> .metric .estimator mean n std_err .config
#> <chr> <chr> <dbl> <int> <dbl> <chr>
#> 1 rmse standard 0.0721 10 0.00305 Preprocessor1_Model1
#> 2 rsq standard 0.831 10 0.0108 Preprocessor1_Model1

```

이것들은 개별 반복들에 대해 평균화된 리샘플링 추정치입니다. 각 리샘플에 대한 지표를 얻으려면 `summarize = FALSE` 옵션을 사용하세요.

장 앞부분의 재대입 추정치보다 성능 추정치가 얼마나 훨씬 더 현실적인지 주목하십시오!

평가 세트 예측을 얻으려면:

```

assess_res <- collect_predictions(rf_res)
assess_res
#> # A tibble: 2,342 × 5
#> id .pred .row Sale_Price .config
#> <chr> <dbl> <int> <dbl> <chr>
#> 1 Fold01 5.10 10 5.09 Preprocessor1_Model1
#> 2 Fold01 4.92 27 4.90 Preprocessor1_Model1
#> 3 Fold01 5.21 47 5.08 Preprocessor1_Model1
#> 4 Fold01 5.13 52 5.10 Preprocessor1_Model1
#> 5 Fold01 5.13 59 5.10 Preprocessor1_Model1
#> 6 Fold01 5.13 63 5.11 Preprocessor1_Model1
#> # … 2,336개 행이 더 있습니다

```

일관성 및 사용 편의성을 위해 예측 열 이름들은 [6장](ch06.xhtml#models)에서 parsnip 모델에 대해 논의한 규칙(conventions)을 따릅니다. 관측된 결과 변수 열은 항상 원본 데이터의 원래 열 이름을 사용합니다. `.row` 열은 원본 훈련 세트의 행과 일치하는 정수(integer)이므로 이러한 결과를 원본 데이터와 적절히 정렬하고(arranged) 조인(joined)할 수 있습니다.

###### 참고 (Note)

부트스트랩이나 반복 교차 검증과 같은 일부 리샘플링 방법의 경우 원본 훈련 세트의 각 행마다 여러 예측값이 있을 수 있습니다. 요약된 값(반복 예측의 평균)을 얻으려면 `collect_predictions(object,` `summarize = TRUE)`를 사용하세요.

이 분석은 10-겹 교차 검증을 사용했기 때문에 각 훈련 세트 샘플에 대해 하나의 고유한 예측이 있습니다. 이 데이터는 잠재적으로 모델이 어디에서 실패했는지 이해하기 위해 도움이 되는 모델의 플롯을 생성할 수 있습니다. 예를 들어 [그림 10-9](#ames-resampled-performance)는 관측된 값과 제외된(held-out) 상태에서 예측된 값을 비교합니다([그림 9-2](ch09.xhtml#ames-performance-plot)와 유사함):

```

assess_res %>%
ggplot(aes(x = Sale_Price, y = .pred)) +
geom_point(alpha = .15) +
geom_abline(color = "red") +
coord_obs_pred() +
ylab("Predicted")

```

훈련 세트에서 관측된 판매 가격이 낮지만 모델에 의해 심하게 과대예측된 집이 두 채 있습니다. 이 집들은 어떤 집들일까요? `assess_res` 결과에서 알아보겠습니다.

```

over_predicted <-
assess_res %>%
mutate(residual = Sale_Price - .pred) %>%
arrange(desc(abs(residual))) %>%
slice(1:2)
over_predicted
#> # A tibble: 2 × 6
#> id .pred .row Sale_Price .config residual
#> <chr> <dbl> <int> <dbl> <chr> <dbl>
#> 1 Fold09 4.97 32 4.11 Preprocessor1_Model1 -0.858
#> 2 Fold08 4.93 317 4.12 Preprocessor1_Model1 -0.815

```

```

ames_train %>%
slice(over_predicted$.row) %>%
select(Gr_Liv_Area, Neighborhood, Year_Built, Bedroom_AbvGr, Full_Bath)
#> # A tibble: 2 × 5
#> Gr_Liv_Area Neighborhood Year_Built Bedroom_AbvGr Full_Bath
#> <int> <fct> <int> <int> <int>
#> 1 832 Old_Town 1923 2 1
#> 2 733 Iowa_DOT_and_Rail_Road 1952 2 1

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1009.png" alt="tmwr 1009" />
<h6 id="figure-10-9.-out-of-sample-observed-versus-predicted-values-for-an-ames-regression-model-using-log-10-units-on-both-axes.">그림 10-9. 양쪽 축에 로그 10 단위를 사용한 Ames 회귀 모델에 대한 표본 외 관측값 대비 예측값.</h6>
</figure>

특히 성능이 떨어지는 이러한 예들을 식별하면 이 특정 예측이 왜 그렇게 떨어지는지 추적하고(follow up) 조사하는 데 도움이 될 수 있습니다.

다시 전체 집들로 돌아가 봅시다. 교차 검증 대신 검증 세트를 어떻게 사용할 수 있을까요? 우리의 이전 rsample 객체로부터:

```

val_res <- rf_wflow %>% fit_resamples(resamples = val_set)
val_res
#> # Resampling results
#> # Validation Set Split (0.75/0.25)
#> # A tibble: 1 × 4
#> splits id .metrics .notes
#> <list> <chr> <list> <list>
#> 1 <split [1756/586]> validation <tibble [2 × 4]> <tibble [0 × 3]>

collect_metrics(val_res)
#> # A tibble: 2 × 6
#> .metric .estimator mean n std_err .config
#> <chr> <chr> <dbl> <int> <dbl> <chr>
#> 1 rmse standard 0.0695 1 NA Preprocessor1_Model1
#> 2 rsq standard 0.843 1 NA Preprocessor1_Model1

```

이러한 결과들 또한 재대입 성능 추정치보다 테스트 세트 결과에 훨씬 가깝습니다.

###### 참고 (Note)

이 분석들에서 리샘플링 결과는 테스트 세트 결과와 매우 가깝습니다. 두 종류의 추정치는 서로 밀접한 상관관계를 가지는 경향이 있습니다. 그러나 이것은 우연(random chance)일 수 있습니다. `55`라는 시드 값은 리샘플을 생성하기 전에 난수(random numbers)를 고정했습니다. 이 값을 변경하고 분석을 다시 실행하여 리샘플링된 추정치가 테스트 세트 결과와 마찬가지로 일치하는지 조사해 보십시오.

# 병렬 처리 (Parallel Processing)

리샘플링 중에 생성된 모델들은 서로 독립적입니다. 이런 종류의 계산을 종종 "엄청나게 병렬적(embarrassingly parallel)"이라고 합니다; 각 모델은 문제없이 동시에 피팅될 수 있습니다.<sup><a href="ch10.xhtml#idm45881861599600" id="idm45881861599600-marker" data-type="noteref">9</a></sup> tune 패키지는 병렬 계산을 용이하게 하기 위해 [foreach](https://oreil.ly/o821g) 패키지를 사용합니다. 선택한 기술에 따라 이러한 계산은 동일한 컴퓨터의 프로세서들에 분할되거나 다른 컴퓨터들에 분할될 수 있습니다.

단일 컴퓨터에서 수행되는 계산의 경우, 가능한 워커(worker) 프로세스의 수는 parallel 패키지에 의해 결정됩니다.

```

# 하드웨어의 물리적 코어 수:

parallel::detectCores(logical = FALSE)
#> [1] 10

# 동시에 사용할 수 있는 가능한 독립 프로세스 수:

parallel::detectCores(logical = TRUE)
#> [1] 20

```

이 두 값의 차이는 컴퓨터의 프로세서와 관련이 있습니다. 예를 들어 대부분의 인텔 프로세서는 각 물리적 코어에 대해 두 개의 가상 코어를 생성하는 하이퍼스레딩(hyperthreading)을 사용합니다. 이러한 추가 리소스가 성능을 향상시킬 수는 있지만, 병렬 처리로 인해 생성되는 대부분의 속도 향상은 처리가 물리적 코어 수보다 적게 사용할 때 발생합니다.

`fit_resamples()` 및 tune의 기타 함수의 경우, 사용자가 병렬 백엔드(backend) 패키지를 등록할 때 병렬 처리가 발생합니다. 이러한 R 패키지는 병렬 처리를 실행하는 방법을 정의합니다. Unix 및 macOS 운영 체제에서 계산을 분할하는 한 가지 방법은 스레드를 포크(forking)하는 것입니다. 이를 활성화하려면 doMC 패키지를 로드하고 foreach로 병렬 코어 수를 등록합니다.

```

# Unix 및 macOS 전용

library(doMC)
registerDoMC(cores = 2)

# 이제 fit_resamples()를 실행합니다...

```

이것은 `fit_resamples()`가 두 코어 각각에서 계산의 절반을 실행하도록 지시합니다. 계산을 순차 처리(sequential processing)로 재설정하려면:

```

registerDoSEQ()

```

대안으로, 계산을 병렬화하는 다른 접근 방식은 네트워크 소켓(network sockets)을 사용합니다. doParallel 패키지는 이 방법(모든 운영 체제에서 사용 가능)을 가능하게 합니다.

```

# 모든 운영 체제

library(doParallel)

# 클러스터(cluster) 객체를 생성하고 등록합니다.

cl <- makePSOCKcluster(2)
registerDoParallel(cl)

# 이제 fit_resamples()`를 실행합니다...

stopCluster(cl)

```

병렬 처리를 용이하게 하는 또 다른 R 패키지는 [future](https://oreil.ly/8LLjC) 패키지입니다. foreach와 마찬가지로 이 패키지는 병렬성을 위한 프레임워크를 제공합니다. future 패키지는 doFuture 패키지를 통해 foreach와 함께(in conjunction with) 사용됩니다.

###### 참고 (Note)

foreach를 위한 병렬 백엔드가 있는 R 패키지는 `"do"` 접두사로 시작합니다.

tune 패키지를 사용한 병렬 처리는 처음 몇 개의 코어에 대해 선형적인(linear) 속도 향상을 제공하는 경향이 있습니다. 즉, 두 개의 코어를 사용하면 계산이 두 배 더 빠릅니다. 데이터 및 모델 유형에 따라 선형적 속도 향상은 4~5개 코어 이후에 저하됩니다(deteriorates). 더 많은 코어를 사용하면 작업을 완료하는 데 걸리는 시간은 여전히 단축되겠지만, 추가 코어에 대해서는 수확 체감(diminishing returns)만 있을 뿐입니다.

병렬성에 대한 마지막 참고 사항 하나로 마무리하겠습니다. 이러한 각 기술에 대해 사용되는 각 추가 코어에 따라 메모리 요구 사항이 배가(multiply)됩니다. 예를 들어 현재 데이터 세트의 메모리 용량이 2GB이고 코어 3개를 사용하는 경우 총 메모리 요구량은 8GB(각 워커 프로세스당 2GB + 원본)입니다. 너무 많은 코어를 사용하면 계산(및 컴퓨터)이 상당히 느려질 수 있습니다.

# 리샘플링된 객체 저장하기 (Saving the Resampled Objects)

리샘플링 중에 생성된 모델은 유지(retained)되지 않습니다. 이러한 모델은 성능을 평가하기 위한 목적으로 훈련되며 일반적으로 성능 통계를 계산한 후에는 필요하지 않습니다. 특정 모델링 접근 방식이 우리 데이터 세트에 적합한 옵션으로 판명되면, 모델 파라미터를 더 많은 데이터로 추정할 수 있도록 전체 훈련 세트에 다시 피팅하는 것이 좋은 선택입니다.

리샘플링 중에 생성된 이러한 모델은 보존(preserved)되지 않지만, 해당 모델이나 그 구성 요소 중 일부를 보관(keeping)하는 방법이 있습니다. `control_resamples()`의 `extract` 옵션은 단일 인수를 사용하는 함수를 지정합니다; 우리는 `x`를 사용할 것입니다. 실행 시 `fit_resamples()`에 워크플로를 제공했는지 여부에 관계없이 `x`는 피팅된 워크플로 객체를 낳습니다. workflows 패키지에는 객체의 여러 구성 요소(모델, 레시피 등)를 가져올(pull) 수 있는 함수가 있음을 상기하십시오.

[8장](ch08.xhtml#recipes)에서 개발한 레시피를 사용하여 선형 회귀 모델을 피팅해 보겠습니다.

```

ames*rec <-
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
Latitude + Longitude, data = ames_train) %>%
step_other(Neighborhood, threshold = 0.01) %>%
step_dummy(all_nominal_predictors()) %>%
step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type*") ) %>%
step_ns(Latitude, Longitude, deg_free = 20)

lm_wflow <-
workflow() %>%
add_recipe(ames_rec) %>%
add_model(linear_reg() %>% set_engine("lm"))

lm_fit <- lm_wflow %>% fit(data = ames_train)

# 레시피를 선택합니다.

extract_recipe(lm_fit, estimated = TRUE)
#> Recipe
#>
#> Inputs:
#>
#> role #variables
#> outcome 1
#> predictor 6
#>
#> Training data contained 2342 data points and no missing data.
#>
#> Operations:
#>
#> Collapsing factor levels for Neighborhood [trained]
#> Dummy variables from Neighborhood, Bldg_Type [trained]
#> Interactions with Gr_Liv_Area:(Bldg_Type_TwoFmCon + Bldg_Type_Duplex + B... [trained]
#> Natural splines on Latitude, Longitude [trained]

```

워크플로에서 피팅된 모델 객체에 대한 선형 모델 계수를 저장할 수 있습니다.

```

get_model <- function(x) {
extract_fit_parsnip(x) %>% tidy()
}

# 다음을 사용하여 테스트합니다.

# get_model(lm_fit)

```

이제 이 함수를 열 개의 리샘플된 피팅들에 적용해 보겠습니다. 추출 함수(extraction function)의 결과는 리스트 객체에 래핑되어 티블로 반환됩니다.

```

ctrl <- control_resamples(extract = get_model)

lm_res <- lm_wflow %>% fit_resamples(resamples = ames_folds, control = ctrl)
lm_res
#> # Resampling results
#> # 10-fold cross-validation
#> # A tibble: 10 × 5
#> splits id .metrics .notes .extracts
#> <list> <chr> <list> <list> <list>
#> 1 <split [2107/235]> Fold01 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [1 × 2]>
#> 2 <split [2107/235]> Fold02 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [1 × 2]>
#> 3 <split [2108/234]> Fold03 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [1 × 2]>
#> 4 <split [2108/234]> Fold04 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [1 × 2]>
#> 5 <split [2108/234]> Fold05 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [1 × 2]>
#> 6 <split [2108/234]> Fold06 <tibble [2 × 4]> <tibble [0 × 3]> <tibble [1 × 2]>
#> # … 4개 행이 더 있습니다

```

이제 중첩된(nested) 티블이 있는 `.extracts` 열이 있습니다. 여기에는 무엇이 들어 있을까요? 서브셋(subsetting)을 통해 알아보겠습니다.

```

lm_res$.extracts[[1]]
#> # A tibble: 1 × 2
#> .extracts .config
#> <list> <chr>
#> 1 <tibble [73 × 5]> Preprocessor1_Model1

# 결과를 얻기 위해

lm_res$.extracts[[1]][[1]]
#> [[1]]
#> # A tibble: 73 × 5
#> term estimate std.error statistic p.value
#> <chr> <dbl> <dbl> <dbl> <dbl>
#> 1 (Intercept) 1.48 0.320 4.62 4.11e- 6
#> 2 Gr_Liv_Area 0.000158 0.00000476 33.2 9.72e-194
#> 3 Year_Built 0.00180 0.000149 12.1 1.57e- 32
#> 4 Neighborhood_College_Creek -0.00163 0.0373 -0.0438 9.65e- 1
#> 5 Neighborhood_Old_Town -0.0757 0.0138 -5.47 4.92e- 8
#> 6 Neighborhood_Edwards -0.109 0.0310 -3.53 4.21e- 4
#> # … 67개 행이 더 있습니다

```

이것이 모델 결과를 저장하는 복잡한(convoluted) 방법인 것처럼 보일 수 있습니다. 그러나 `extract`는 유연하며 사용자가 리샘플당 단일 티블만 저장한다고 가정하지 않습니다. 예를 들어 `tidy()` 메서드는 모델뿐만 아니라 레시피에서도 실행될 수 있습니다. 이 경우 두 개의 티블로 이루어진 리스트가 반환됩니다.

우리의 더 간단한 예제의 경우 다음을 사용하여 모든 결과를 평면화(flattened)하고 수집할 수 있습니다.

```

all_coef <- map_dfr(lm_res$.extracts, ~ .x[[1]][[1]])

# 단일 예측 변수에 대한 반복(replicates)을 표시합니다.

filter(all_coef, term == "Year_Built")
#> # A tibble: 10 × 5
#> term estimate std.error statistic p.value
#> <chr> <dbl> <dbl> <dbl> <dbl>
#> 1 Year_Built 0.00180 0.000149 12.1 1.57e-32
#> 2 Year_Built 0.00180 0.000151 12.0 6.45e-32
#> 3 Year_Built 0.00185 0.000150 12.3 1.00e-33
#> 4 Year_Built 0.00183 0.000147 12.5 1.90e-34
#> 5 Year_Built 0.00184 0.000150 12.2 2.47e-33
#> 6 Year_Built 0.00180 0.000150 12.0 3.35e-32
#> # … 4개 행이 더 있습니다

```

[13장](ch13.xhtml#grid-search)과 [14장](ch14.xhtml#iterative-search)에서는 모델 튜닝을 위한 여러 함수(suite of functions)를 다룹니다. 이들의 인터페이스는 `fit_resamples()`와 유사하며 여기에 설명된 많은 기능들이 이러한 함수에도 적용됩니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 데이터 분석의 기본 도구 중 하나인 모델 결과의 성능과 변동성(variation)을 측정하는 기능에 대해 설명합니다. 리샘플링을 사용하면 테스트 세트를 사용하지 않고도 모델이 얼마나 잘 작동하는지 확인할 수 있습니다.

`fit_resamples()`라는 tune 패키지의 중요한 함수가 소개되었습니다. 이 함수에 대한 인터페이스는 모델 튜닝 도구를 설명하는 향후 장에서도 사용됩니다.

지금까지 Ames 데이터에 대한 데이터 분석 코드는 다음과 같습니다.

```

library(tidymodels)
data(ames)
ames <- mutate(ames, Sale_Price = log10(Sale_Price))

set.seed(502)
ames_split <- initial_split(ames, prop = 0.80, strata = Sale_Price)
ames_train <- training(ames_split)
ames_test <- testing(ames_split)

ames*rec <-
recipe(Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
Latitude + Longitude, data = ames_train) %>%
step_log(Gr_Liv_Area, base = 10) %>%
step_other(Neighborhood, threshold = 0.01) %>%
step_dummy(all_nominal_predictors()) %>%
step_interact( ~ Gr_Liv_Area:starts_with("Bldg_Type*") ) %>%
step_ns(Latitude, Longitude, deg_free = 20)

lm_model <- linear_reg() %>% set_engine("lm")

lm_wflow <-
workflow() %>%
add_model(lm_model) %>%
add_recipe(ames_rec)

lm_fit <- fit(lm_wflow, ames_train)

rf_model <-
rand_forest(trees = 1000) %>%
set_engine("ranger") %>%
set_mode("regression")

rf_wflow <-
workflow() %>%
add_formula(
Sale_Price ~ Neighborhood + Gr_Liv_Area + Year_Built + Bldg_Type +
Latitude + Longitude) %>%
add_model(rf_model)

set.seed(1001)
ames_folds <- vfold_cv(ames_train, v = 10)

keep_pred <- control_resamples(save_pred = TRUE, save_workflow = TRUE)

set.seed(1003)
rf_res <- rf_wflow %>% fit_resamples(resamples = ames_folds, control = keep_pred)

```

<sup>[1](ch10.xhtml#idm45881862747584-marker)</sup> 이에 대한 논의는 [Kuhn and Johnson (2020)의 1.2.5절](https://oreil.ly/pfcLQ)을 참조하십시오.

<sup>[2](ch10.xhtml#idm45881862678160-marker)</sup> 랜덤 포레스트 모델처럼 선형 모델이 훈련 세트를 거의 암기하는 것이 가능합니다. `ames_rec` 객체에서 `longitude` 및 `latitude`에 대한 스플라인 항의 개수를 큰 숫자(가령 1,000)로 변경하십시오. 이렇게 하면 재대입 RMSE는 매우 작고 테스트 세트 RMSE는 훨씬 큰 모델 피팅이 생성됩니다.

<sup>[3](ch10.xhtml#idm45881862631264-marker)</sup> *V* 변경 결과에 대한 더 긴 설명은 [Kuhn and Johnson (2020)의 3.4절](https://oreil.ly/Mvv6Y)을 참조하십시오.

<sup>[4](ch10.xhtml#idm45881862579184-marker)</sup> 이를 직접 확인하려면 `lobstr::obj_size(ames_folds)` 및 `lobstr::obj_size(ames_train)`을 실행해 보십시오. 리샘플 객체의 크기는 원본 데이터 크기의 10배보다 훨씬 작습니다.

<sup>[5](ch10.xhtml#idm45881862621792-marker)</sup> 일부 리샘플링 방법에는 여러 `id` 필드가 필요합니다.

<sup>[6](ch10.xhtml#idm45881862534224-marker)</sup> 더 자세한 내용은 [Kuhn and Johnson (2020)의 3.4.6절](https://oreil.ly/nt1SS)을 참조하십시오.

<sup>[7](ch10.xhtml#idm45881862495648-marker)</sup> 이것들은 *근사(approximate)* 표준 오차입니다. 다음 장에서 논의하겠지만, 리샘플된 결과의 전형적인 특성인 반복 간(within-replicate) 상관관계가 있습니다. 이 플롯에 표시된 간단한 계산들은 변동(variation)의 이러한 추가 구성 요소를 무시하기 때문에 표준 오차의 잡음 감소를 과대평가(overestimates)합니다.

<sup>[8](ch10.xhtml#idm45881862367296-marker)</sup> 본질적으로 검증 세트는 몬테카를로 교차 검증의 단일 반복으로 간주될 수 있습니다.

<sup>[9](ch10.xhtml#idm45881861599600-marker)</sup> Schmidberger et al. (2009)는 이러한 기술에 대한 기술적(technical) 개요를 제공합니다.
```
