# 9장. 모델의 효과성 판단하기 (Judging Model Effectiveness)

모델이 완성되면 얼마나 잘 작동하는지 알아야 합니다. 효과성(effectiveness)을 추정하기 위한 정량적(quantitative) 접근 방식은 우리가 모델을 이해하거나, 여러 모델을 비교하거나, 성능을 향상시키기 위해 모델을 조정(tweak)할 수 있게 해줍니다. tidymodels에서 우리가 초점을 맞추는 것은 경험적 검증(empirical validation)입니다. 이것은 일반적으로 모델 생성에 사용되지 않은 데이터를 효과성을 측정하기 위한 기질(substrate, 기반)로 사용하는 것을 의미합니다.

###### 경고 (Warning)

경험적 검증에 대한 최선의 접근 방식은 [10장](ch10.xhtml#resampling)에서 소개될 _리샘플링(resampling)_ 방법을 사용하는 것입니다. 이 장에서는 테스트 세트(test set)를 사용하여 경험적 검증의 필요성에 대한 동기를 부여할 것입니다. [5장](ch05.xhtml#splitting)에서 설명한 대로 테스트 세트는 단 한 번만 사용할 수 있다는 점을 명심하십시오.

모델의 효과성을 판단할 때 어떤 지표(metrics)를 검사할지에 대한 여러분의 결정이 매우 중요할 수 있습니다. 이후 장에서는 특정 모델 파라미터가 경험적으로 최적화될 것이며 기본 성능 지표를 사용하여 가장 좋은 하위 모델(submodel)을 선택할 것입니다. 잘못된 지표를 선택하면 의도하지 않은 결과가 쉽게 발생할 수 있습니다. 예를 들어 회귀(regression) 모델의 두 가지 일반적인 지표는 평균 제곱근 오차(root mean squared error, RMSE)와 결정계수(coefficient of determination, 즉 *R*²)입니다. 전자는 *정확도(accuracy)*를 측정하고 후자는 *상관관계(correlation)*를 측정합니다. 이들은 반드시 같은 것은 아닙니다. [그림 9-1](#figure-9-1.-observed-versus-predicted-values-for-models-that-are-optimized-using-the-rmse-compared-to-the-coefficient-of-determination.)은 이 둘의 차이를 보여줍니다.

RMSE에 최적화된 모델은 변동성(variability)이 더 크지만 결과 변수의 범위에 걸쳐 상대적으로 균일한 정확도를 갖습니다. 오른쪽 패널은 관측값과 예측값 사이에 더 밀접한 상관관계가 있음을 보여주지만 이 모델은 꼬리(tails) 부분에서 성능이 떨어집니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0901.png" alt="tmwr 0901" />
<h6 id="figure-9-1.-observed-versus-predicted-values-for-models-that-are-optimized-using-the-rmse-compared-to-the-coefficient-of-determination.">그림 9-1. 결정계수와 비교하여 RMSE를 사용하여 최적화된 모델들의 관측값 대비 예측값.</h6>
</figure>

이 장에서는 모델 성능 측정에 초점을 맞춘 핵심 tidymodels 패키지인 yardstick 패키지를 시연할 것입니다. 구문을 설명하기 전에 모델이 예측보다는 추론(inference)에 초점을 맞출 때 성능 지표를 사용한 경험적 검증이 가치 있는지(worthwhile) 알아보겠습니다.

# 성능 지표와 추론 (Performance Metrics and Inference)

주어진 모델의 효과성은 모델이 어떻게 사용될지에 따라 달라집니다. 추론(inferential) 모델은 주로 관계를 이해하는 데 사용되며 일반적으로 모델을 정의하는 확률적 분포(probabilistic distributions) 및 기타 생성적(generative) 특성의 선택(및 유효성)을 강조합니다. 이와 대조적으로 주로 예측을 위해 사용되는 모델의 경우 예측 강도(predictive strength)가 가장 중요하며, 근본적인 통계적 특성에 대한 기타 우려는 덜 중요할 수 있습니다. 예측 강도는 예측이 관측 데이터에 얼마나 가까운지, 즉 실제 결과에 대한 모델 예측의 충실도(fidelity)에 의해 결정되는 경우가 많습니다. 이 장은 예측 강도를 측정하는 데 사용할 수 있는 함수들에 초점을 맞춥니다. 그러나 추론 모델을 개발하는 사람들을 위한 우리의 조언은 예측을 주된 목표로 모델을 사용하지 않을 때에도 이러한 기술을 사용하는 것입니다.

추론 통계 관행의 오랜 문제점은 추론에만 집중하면 모델의 신뢰성(credibility)을 평가하기 어렵다는 것입니다. 예를 들어 333명의 환자를 연구하여 인지 장애(cognitive impairment)에 영향을 미치는 요인을 확인한 Craig-Schapiro et al. (2011)의 알츠하이머병 데이터를 생각해 보십시오. 분석은 알려진 위험 인자를 사용하여 결과가 이진형(장애/비장애)인 로지스틱 회귀 모델을 구축할 수 있습니다. 나이, 성별 및 아폴로지단백 E 유전자형(apolipoprotein E genotype)에 대한 예측 변수를 고려해 보겠습니다. 후자는 이 유전자의 세 가지 주요 변이(variants)의 6가지 가능한 조합을 가진 범주형 변수입니다. 아폴로지단백 E는 치매와 관련이 있는 것으로 알려져 있습니다(Jungsu, Basak, and Holtzman 2009).

이 분석에 대한 피상적이지만(superficial) 드물지 않은 접근 방식은 주효과(main effects)와 상호작용(interactions)이 있는 대형 모델을 피팅한 다음, 통계적 검정(statistical tests)을 사용하여 사전에 정의된 일부 수준에서 통계적으로 유의미한 모델 항들의 최소 집합을 찾는 것입니다. 3개의 요인과 이들의 2원(two-way) 및 3원(three-way) 상호작용이 있는 전체 모델(full model)이 사용된 경우, 초기 단계는 순차적 우도비 검정(sequential likelihood ratio tests)(Hosmer and Lemeshow 2000)을 사용하여 상호작용을 검정하는 것입니다. 예제 알츠하이머병 데이터에 대한 이러한 종류의 접근 방식을 단계별로 살펴보겠습니다.

- 모든 2원 상호작용이 있는 모델을 추가적인 3원 상호작용이 있는 모델과 비교할 때 우도비 검정은 0.888의 p-값을 생성합니다. 이는 3원 상호작용과 관련된 4개의 추가 모델 항이 이들을 모델에 유지할 만큼 데이터의 변동을 충분히 설명한다는 증거가 없음을 의미합니다.

- 다음으로 2원 상호작용도 유사하게 상호작용이 없는 모델에 대해 평가됩니다. 여기서 p-값은 0.0382입니다. 이는 다소 경계선에 있지만(borderline), 작은 샘플 크기를 고려할 때 10개의 가능한 2원 상호작용 중 일부가 모델에 중요하다는 증거가 있다고 결론짓는 것이 신중(prudent)할 것입니다.

- 여기서부터 우리는 결과에 대한 몇 가지 설명을 만들 것입니다. 상호작용은 더 깊이 탐구할 흥미로운 생리적 또는 신경학적 가설을 촉발(spark)할 수 있기 때문에 논의하는 것이 특히 중요합니다.

얕기는(shallow) 하지만 이 분석 전략은 문헌은 물론 실제 환경에서도 일반적입니다. 특히 실무자(practitioner)가 데이터 분석에 대한 공식 교육을 제한적으로 받은 경우 더욱 그렇습니다.

이 접근 방식에서 누락된 정보 중 하나는 이 모델이 실제 데이터에 얼마나 잘 들어맞는지입니다. [10장](ch10.xhtml#resampling)에서 논의된 리샘플링 방법을 사용하여 우리는 이 모델의 정확도를 약 73.3%로 추정할 수 있습니다. 정확도는 종종 모델 성능의 좋지 않은 척도(measure)입니다. 우리는 일반적으로 이해되기 때문에 여기서 정확도를 사용합니다. 모델이 데이터에 대해 73.3%의 충실도를 가지고 있다면 이 모델이 내놓는 결론을 신뢰해야 할까요? 데이터에서 장애가 없는 환자의 기준 비율(baseline rate)이 72.7%라는 것을 깨닫기 전까지는 그렇게 생각할 수 있습니다. 즉, 우리의 통계 분석에도 불구하고 이 2-요인 모델은 관측 데이터와 관계없이 항상 환자가 장애가 없다고 예측하는 단순한 휴리스틱보다 겨우 0.6% 더 나은 것처럼 보입니다.

이 장의 나머지 부분에서는 경험적 검증을 통해 모델을 평가하는 일반적인 접근 방식에 대해 논의할 것입니다. 이러한 접근 방식은 결과 데이터의 특성 즉, 순수 숫자형, 이진 클래스, 3개 이상의 클래스 수준으로 그룹화됩니다.

###### 참고 (Note)

이 분석의 요점은 모델의 통계적 특성을 최적화한다고 해서 모델이 데이터에 잘 들어맞는다는 것을 의미하지는 않는다는 생각을 입증하는 것입니다. 순전히 추론적인 모델의 경우에도 데이터 충실도에 대한 어떤 측정치(measure)가 추론 결과와 함께 동반되어야 합니다. 이를 사용하여 분석 소비자들은 결과에 대한 기대치를 보정(calibrate)할 수 있습니다.

# 회귀 지표 (Regression Metrics)

[6장](ch06.xhtml#models)에서 tidymodels의 예측 함수가 예측값을 위한 열을 가진 티블(tibbles)을 생성한다는 것을 상기하십시오. 이 열들은 일관된 이름을 가지며, 성능 지표를 생성하는 yardstick 패키지의 함수들은 일관된 인터페이스를 가지고 있습니다. 이 함수들은 벡터(vector) 기반이 아닌 데이터 프레임(data frame) 기반이며 다음과 같은 일반적인 구문을 갖습니다.

```
function(data, truth, ...)
```

여기서 `data`는 데이터 프레임 또는 티블이고 `truth`는 관측된 결과 변수 값이 있는 열입니다. 줄임표(ellipses) 또는 기타 인수는 예측값이 포함된 열(들)을 지정하는 데 사용됩니다.

이를 설명하기 위해 [8장](ch08.xhtml#recipes) 맨 끝에 있는 모델을 살펴보겠습니다. 이 모델 `lm_wflow_fit`은 선형 회귀 모델과 경도 및 위도에 대한 상호작용 및 스플라인 함수로 보강된(supplemented) 예측 변수 집합을 결합합니다. 이것은 훈련 세트(`ames_train`으로 명명됨)에서 생성되었습니다. 모델링 프로세스의 현시점에서는 테스트 세트를 사용하는 것을 권장하지 않지만 기능과 구문을 설명하기 위해 여기에서 사용하겠습니다. 데이터 프레임 `ames_test`는 588개의 부동산(properties)으로 구성됩니다. 시작하기 위해 예측을 생성해 보겠습니다.

```
ames_test_res <- predict(lm_fit, new_data = ames_test %>% select(-Sale_Price))
ames_test_res
#> # A tibble: 588 × 1
#>   .pred
#>   <dbl>
#> 1  5.07
#> 2  5.31
#> 3  5.28
#> 4  5.33
#> 5  5.30
#> 6  5.24
#> # … 582개 행이 더 있습니다
```

회귀 모델에서 예측된 숫자 결과의 이름은 `.pred`입니다. 예측값을 대응하는 관측된 결과값과 맞춰(match) 보겠습니다.

```
ames_test_res <- bind_cols(ames_test_res, ames_test %>% select(Sale_Price))
ames_test_res
#> # A tibble: 588 × 2
#>   .pred Sale_Price
#>   <dbl>      <dbl>
#> 1  5.07       5.02
#> 2  5.31       5.39
#> 3  5.28       5.28
#> 4  5.33       5.28
#> 5  5.30       5.28
#> 6  5.24       5.26
#> # … 582개 행이 더 있습니다
```

이 값들은 대체로 가까워(close) 보이지만 성능 지표를 계산하지 않았기 때문에 모델이 어떻게 작동하고 있는지 아직 정량적으로 이해하지 못합니다. 예측된 결과와 관측된 결과 모두 로그 10 단위라는 점에 유의하세요. 예측값이 원래 단위를 사용하여 보고되더라도 (변환 척도가 사용된 경우) 변환된 척도에서 예측값을 분석하는 것이 가장 좋은 방법(best practice)입니다.

지표를 계산하기 전에 [그림 9-2](#ames-performance-plot)에 데이터를 플로팅해 보겠습니다.

```
ggplot(ames_test_res, aes(x = Sale_Price, y = .pred)) +
  # 대각선을 만듭니다.
  geom_abline(lty = 2) +
  geom_point(alpha = 0.5) +
  labs(y = "Predicted Sale Price (log10)", x = "Sale Price (log10)") +
  # x축과 y축의 비율을 동일하게 맞춥니다.
  coord_obs_pred()
```

상당히 과대예측된(overpredicted), 즉 점선(dashed line) 위로 꽤 높이 있는 낮은 가격의 부동산이 하나 있습니다.

`rmse()` 함수를 사용하여 이 모델의 평균 제곱근 오차(root mean squared error)를 계산해 보겠습니다.

```
rmse(ames_test_res, truth = Sale_Price, estimate = .pred)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.0736
```

이것은 yardstick 함수 출력의 표준 형식을 보여줍니다. 숫자 결과에 대한 지표는 대개 `.estimator` 열의 값이 "standard"입니다. 이 열에 대해 다른 값을 가진 예가 다음 섹션에 나와 있습니다.

여러 지표를 한 번에 계산하기 위해 *지표 세트(metric set)*를 만들 수 있습니다. *R*²와 평균 절대 오차(mean absolute error)를 추가해 보겠습니다.

```
ames_metrics <- metric_set(rmse, rsq, mae)
ames_metrics(ames_test_res, truth = Sale_Price, estimate = .pred)
#> # A tibble: 3 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 rmse    standard      0.0736
#> 2 rsq     standard      0.836
#> 3 mae     standard      0.0549
```

이 깔끔한 데이터 형식은 지표를 수직으로 쌓습니다(stacks). 평균 제곱근 오차 및 평균 절대 오차 지표는 모두 결과 변수의 척도(따라서 우리 예의 경우 `log10(Sale_Price)`)에 있으며 예측값과 관측값의 차이를 측정합니다. *R*² 값은 예측값과 관측값 사이의 제곱된 상관관계를 측정하므로 1에 가까울수록 좋습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0902.png" alt="tmwr 0902" />
<h6 id="figure-9-2.-observed-versus-predicted-values-for-an-ames-regression-model-with-log-10-units-on-both-axes.">그림 9-2. 양쪽 축이 로그 10 단위인 Ames 회귀 모델에 대한 관측값 대비 예측값.</h6>
</figure>

###### 경고 (Warning)

yardstick 패키지에는 조정된 *R*²(adjusted *R*²)에 대한 함수가 포함되어 있지 _않습니다_. 결정계수의 이러한 수정은 모델을 피팅하는 데 사용된 동일한 데이터가 모델을 평가하는 데 사용될 때 일반적으로 사용됩니다. 이 지표는 tidymodels에서 완전히 지원되지 않는데 그 이유는 모델 피팅에 사용된 데이터 세트와는 별도의 데이터 세트에서 성능을 계산하는 것이 항상 더 나은 접근 방식이기 때문입니다.

# 이진 분류 지표 (Binary Classification Metrics)

모델 성능을 측정하는 다른 방법을 설명하기 위해 다른 예제로 넘어가겠습니다. modeldata 패키지(tidymodels 패키지 중 또 다른 하나)에는 두 개의 클래스("Class1" 및 "Class2")가 있는 테스트 데이터 세트의 예제 예측값이 포함되어 있습니다.

```
data(two_class_example)
tibble(two_class_example)
#> # A tibble: 500 × 4
#>   truth   Class1   Class2 predicted
#>   <fct>    <dbl>    <dbl> <fct>
#> 1 Class2 0.00359 0.996    Class2
#> 2 Class1 0.679   0.321    Class1
#> 3 Class2 0.111   0.889    Class2
#> 4 Class1 0.735   0.265    Class1
#> 5 Class2 0.0162  0.984    Class2
#> 6 Class1 0.999   0.000725 Class1
#> # … 494개 행이 더 있습니다
```

두 번째와 세 번째 열은 테스트 세트에 대한 예측된 클래스 확률인 반면 `predicted`는 이산형 예측(discrete predictions)입니다.

엄격한(hard) 클래스 예측의 경우 다양한 yardstick 함수가 도움이 됩니다.

```
# 혼동 행렬(confusion matrix):
conf_mat(two_class_example, truth = truth, estimate = predicted)
#>           Truth
#> Prediction Class1 Class2
#>     Class1    227     50
#>     Class2     31    192

# 정확도(Accuracy):
accuracy(two_class_example, truth, predicted)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.838

# 매튜스 상관계수(Matthews correlation coefficient):
mcc(two_class_example, truth, predicted)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 mcc     binary         0.677

# F1 지표:
f_meas(two_class_example, truth, predicted)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 f_meas  binary         0.849

# 이 세 가지 분류 지표를 함께 결합하기
classification_metrics <- metric_set(accuracy, mcc, f_meas)
classification_metrics(two_class_example, truth = truth, estimate = predicted)
#> # A tibble: 3 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy binary         0.838
#> 2 mcc      binary         0.677
#> 3 f_meas   binary         0.849
```

매튜스 상관계수와 F1 점수 모두 혼동 행렬을 요약하지만 긍정 및 부정 예제 모두의 질(quality)을 측정하는 `mcc()`에 비해 `f_meas()` 지표는 긍정 클래스, 즉 관심 사건(event of interest)을 강조합니다. 이 예제와 같은 이진 분류 데이터 세트의 경우 yardstick 함수에는 긍정(positive) 및 부정(negative) 수준(levels)을 구별하기 위한 `event_level`이라는 표준 인수가 있습니다. 기본값(이 코드에서 사용한 것)은 결과 팩터의 _첫 번째_ 수준이 관심 사건이라는 것입니다.

###### 참고 (Note)

이와 관련하여 R 함수에는 약간의 이질성(heterogeneity)이 있습니다; 일부는 첫 번째 수준을 사용하고 다른 일부는 두 번째 수준을 관심 사건을 나타내는 데 사용합니다. 우리는 첫 번째 수준이 가장 중요하다는 것이 더 직관적이라고 생각합니다. 두 번째 수준 논리는 결과를 0/1로 인코딩하는 데서 비롯되었으며(이 경우 두 번째 값이 이벤트임), 불행히도 일부 패키지에 남아 있습니다. 그러나 tidymodels(및 기타 많은 R 패키지)는 범주형 결과를 팩터로 인코딩하도록 요구하며 이러한 이유로 사건을 두 번째 수준으로 취급하는 과거의(legacy) 정당성은 무의미해집니다.

두 번째 수준이 이벤트인 경우의 예:

```
f_meas(two_class_example, truth, predicted, event_level = "second")
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 f_meas  binary         0.826
```

이 출력에서 "binary"라는 `.estimator` 값은 이진 클래스에 대한 표준 공식을 사용할 것임을 나타냅니다.

엄격한(hard) 클래스 예측보다는 예측 확률을 입력으로 사용하는 수많은 분류 지표들이 있습니다. 예를 들어, 수신자 조작 특성(receiver operating characteristic, ROC) 곡선은 다양한 이벤트 임곗값(thresholds) 연속체(continuum)에 대한 민감도(sensitivity)와 특이도(specificity)를 계산합니다. 예측된 클래스 열은 사용되지 않습니다. 이 방법에 대한 두 가지 yardstick 함수가 있습니다. `roc_curve()`는 ROC 곡선을 구성하는 데이터 포인트를 계산하고 `roc_auc()`는 곡선 아래 면적(area under the curve)을 계산합니다.

이러한 유형의 지표 함수에 대한 인터페이스는 `...` 인수 자리표시자(placeholder)를 사용하여 적절한 클래스 확률 열을 전달합니다. 두 클래스(two-class) 문제의 경우 관심 사건에 대한 확률 열이 함수에 전달됩니다.

```
two_class_curve <- roc_curve(two_class_example, truth, Class1)
two_class_curve
#> # A tibble: 502 × 3
#>   .threshold specificity sensitivity
#>        <dbl>       <dbl>       <dbl>
#> 1 -Inf           0                 1
#> 2    1.79e-7     0                 1
#> 3    4.50e-6     0.00413           1
#> 4    5.81e-6     0.00826           1
#> 5    5.92e-6     0.0124            1
#> 6    1.22e-5     0.0165            1
#> # … 496개 행이 더 있습니다

roc_auc(two_class_example, truth, Class1)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc binary         0.939
```

`two_class_curve` 객체는 [그림 9-3](#example-roc-curve)에 표시된 것처럼 곡선을 시각화하기 위해 `ggplot` 호출에 사용할 수 있습니다. 세부 사항을 처리해 주는 `autoplot()` 메서드가 있습니다.

```
autoplot(two_class_curve)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0903.png" alt="tmwr 0903" />
<h6 id="figure-9-3.-example-roc-curve.">그림 9-3. ROC 곡선 예시.</h6>
</figure>

곡선이 대각선에 가깝다면 모델의 예측이 무작위 추측(random guessing)보다 나을 것이 없을 것입니다. 곡선이 왼쪽 상단 모서리에 있으므로 모델이 다양한 임곗값에서 성능이 좋다는 것을 알 수 있습니다.

`gain_curve()`, `lift_curve()`, `pr_curve()` 등 확률 추정치를 사용하는 여러 가지 다른 함수가 있습니다.

# 다중 클래스 분류 지표 (Multiclass Classification Metrics)

세 개 이상의 클래스가 있는 데이터는 어떨까요? 이를 입증하기 위해 4개의 클래스가 있는 다른 예제 데이터 세트를 살펴보겠습니다.

```
data(hpc_cv)
tibble(hpc_cv)
#> # A tibble: 3,467 × 7
#>   obs   pred     VF      F       M          L Resample
#>   <fct> <fct> <dbl>  <dbl>   <dbl>      <dbl> <chr>
#> 1 VF    VF    0.914 0.0779 0.00848 0.0000199  Fold01
#> 2 VF    VF    0.938 0.0571 0.00482 0.0000101  Fold01
#> 3 VF    VF    0.947 0.0495 0.00316 0.00000500 Fold01
#> 4 VF    VF    0.929 0.0653 0.00579 0.0000156  Fold01
#> 5 VF    VF    0.942 0.0543 0.00381 0.00000729 Fold01
#> 6 VF    VF    0.951 0.0462 0.00272 0.00000384 Fold01
#> # … 3,461개 행이 더 있습니다
```

이전과 마찬가지로 각 클래스에 대한 예측 확률의 4가지 다른 열과 함께 관측 및 예측 결과에 대한 팩터가 있습니다. (이 데이터에는 `Resample` 열도 포함되어 있습니다. 이러한 `hpc_cv` 결과는 10-겹 교차 검증(10-fold cross-validation)과 연관된 표본 외(out-of-sample) 예측에 대한 것입니다. 당분간 이 열은 무시될 것이며 [10장](ch10.xhtml#resampling)에서 리샘플링에 대해 심도 있게 논의할 것입니다.)

이산형(discrete) 클래스 예측을 사용하는 지표에 대한 함수는 이진 클래스의 대응 항목(counterparts)과 동일합니다.

```
accuracy(hpc_cv, obs, pred)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy multiclass     0.709

mcc(hpc_cv, obs, pred)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 mcc     multiclass     0.515
```

이 결과에서 "multiclass" `.estimator`가 나열되어 있음에 유의하세요. "binary"와 마찬가지로, 이것은 3개 이상의 클래스 수준이 있는 결과에 대한 공식이 사용되었음을 나타냅니다. 매튜스 상관계수는 원래 두 클래스를 위해 설계되었지만 클래스 수준이 더 많은 경우로 확장되었습니다.

두 개의 클래스만 있는 결과를 다루도록 설계된 지표를 취하여 두 개 이상의 클래스가 있는 결과로 확장하는 방법이 있습니다. 예를 들어, 민감도(sensitivity)와 같은 지표는 정의상 두 개의 클래스(즉, "이벤트" 및 "비이벤트")에 특화된 참 긍정률(true positive rate)을 측정합니다. 우리 예제 데이터에서 이 지표를 어떻게 사용할 수 있을까요?

우리의 4-클래스 결과에 민감도를 적용하는 데 사용할 수 있는 래퍼(wrapper) 방법이 있습니다. 이러한 옵션은 매크로 평균화(macro-averaging), 매크로 가중 평균화(macro-weighted averaging) 및 마이크로 평균화(micro-averaging)입니다.

- 매크로 평균화는 표준 2-클래스 통계를 사용하여 일대다(one-versus-all) 지표 집합을 계산합니다. 이것들이 평균화됩니다.

- 매크로 가중 평균화도 동일한 작업을 수행하지만 평균은 각 클래스의 샘플 수에 의해 가중치가 부여(weighted)됩니다.

- 마이크로 평균화는 각 클래스에 대한 기여도(contribution)를 계산하고 이들을 집계(aggregates)한 다음 그 집계로부터 단일 지표를 계산합니다.

분류 지표를 3개 이상의 클래스가 있는 결과로 확장하는 방법에 대한 자세한 내용은 Wu and Zhou (2017) 및 Opitz and Burst (2019)를 참조하세요.

민감도를 예로 들면 일반적인 2-클래스 계산은 올바르게 예측된 이벤트 수를 실제 이벤트 수로 나눈 비율입니다. 이러한 평균화 방법에 대한 수동 계산은 다음과 같습니다.

```
class_totals <-
  count(hpc_cv, obs, name = "totals") %>%
  mutate(class_wts = totals / sum(totals))
class_totals
#>   obs totals class_wts
#> 1  VF   1769   0.51024
#> 2   F   1078   0.31093
#> 3   M    412   0.11883
#> 4   L    208   0.05999

cell_counts <-
  hpc_cv %>%
  group_by(obs, pred) %>%
  count() %>%
  ungroup()

# 일대다(1-vs-all)를 사용하여 4가지 민감도를 계산합니다
one_versus_all <-
  cell_counts %>%
  filter(obs == pred) %>%
  full_join(class_totals, by = "obs") %>%
  mutate(sens = n / totals)
one_versus_all
#> # A tibble: 4 × 6
#>   obs   pred      n totals class_wts  sens
#>   <fct> <fct> <int>  <int>     <dbl> <dbl>
#> 1 VF    VF     1620   1769    0.510  0.916
#> 2 F     F       647   1078    0.311  0.600
#> 3 M     M        79    412    0.119  0.192
#> 4 L     L       111    208    0.0600 0.534

# 3가지 다른 추정치:
one_versus_all %>%
  summarize(
    macro = mean(sens),
    macro_wts = weighted.mean(sens, class_wts),
    micro = sum(n) / sum(totals)
  )
#> # A tibble: 1 × 3
#>   macro macro_wts micro
#>   <dbl>     <dbl> <dbl>
#> 1 0.560     0.709 0.709
```

다행히도 이러한 평균화 방법을 수동으로 구현할 필요는 없습니다. 대신 yardstick 함수는 `estimator` 인수를 통해 이러한 방법을 자동으로 적용할 수 있습니다.

```
sensitivity(hpc_cv, obs, pred, estimator = "macro")
#> # A tibble: 1 × 3
#>   .metric     .estimator .estimate
#>   <chr>       <chr>          <dbl>
#> 1 sensitivity macro          0.560
sensitivity(hpc_cv, obs, pred, estimator = "macro_weighted")
#> # A tibble: 1 × 3
#>   .metric     .estimator     .estimate
#>   <chr>       <chr>              <dbl>
#> 1 sensitivity macro_weighted     0.709
sensitivity(hpc_cv, obs, pred, estimator = "micro")
#> # A tibble: 1 × 3
#>   .metric     .estimator .estimate
#>   <chr>       <chr>          <dbl>
#> 1 sensitivity micro          0.709
```

확률 추정치를 다룰 때 다중 클래스 유사어(analogs)가 있는 일부 지표가 있습니다. 예를 들어, Hand and Till(2001)은 ROC 곡선에 대한 다중 클래스 기법을 결정했습니다. 이 경우 _모든_ 클래스 확률 열이 함수에 제공되어야 합니다.

```
roc_auc(hpc_cv, obs, VF, F, M, L)
#> # A tibble: 1 × 3
#>   .metric .estimator .estimate
#>   <chr>   <chr>          <dbl>
#> 1 roc_auc hand_till      0.829
```

다중 클래스 결과에 이 지표를 적용하기 위한 옵션으로 매크로 가중 평균화도 사용할 수 있습니다.

```
roc_auc(hpc_cv, obs, VF, F, M, L, estimator = "macro_weighted")
#> # A tibble: 1 × 3
#>   .metric .estimator     .estimate
#>   <chr>   <chr>              <dbl>
#> 1 roc_auc macro_weighted     0.868
```

마지막으로 이러한 모든 성능 지표는 dplyr 그룹화(groupings)를 사용하여 계산할 수 있습니다. 이 데이터에는 리샘플링 그룹에 대한 열이 있다는 것을 상기하십시오. 리샘플링에 대해 아직 자세히 논의하지는 않았지만 그룹화된 데이터 프레임을 지표 함수에 전달하여 각 그룹에 대한 지표를 계산할 수 있는 방법에 유의하십시오:

```
hpc_cv %>%
  group_by(Resample) %>%
  accuracy(obs, pred)
#> # A tibble: 10 × 4
#>   Resample .metric  .estimator .estimate
#>   <chr>    <chr>    <chr>          <dbl>
#> 1 Fold01   accuracy multiclass     0.726
#> 2 Fold02   accuracy multiclass     0.712
#> 3 Fold03   accuracy multiclass     0.758
#> 4 Fold04   accuracy multiclass     0.712
#> 5 Fold05   accuracy multiclass     0.712
#> 6 Fold06   accuracy multiclass     0.697
#> # … 4개 행이 더 있습니다
```

그룹화는 또한 `autoplot()` 메서드로 변환되며 그 결과는 [그림 9-4](#grouped-roc-curves)에 나와 있습니다.

```
# 각 폴드(fold)에 대한 4개의 일대다(1-vs-all) ROC 곡선
hpc_cv %>%
  group_by(Resample) %>%
  roc_curve(obs, VF, F, M, L) %>%
  autoplot() +
  theme(legend.position = "none")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0904.png" alt="tmwr 0904" />
<h6 id="figure-9-4.-resampled-roc-curves-for-each-of-the-four-outcome-classes.">그림 9-4. 4가지 결과 클래스 각각에 대한 리샘플링된 ROC 곡선.</h6>
</figure>

이 시각화는 서로 다른 그룹들이 모두 거의 동일하게 작동하지만, `VF` ROC 곡선이 왼쪽 상단 모서리에 더 많이 있기 때문에 `VF` 클래스가 `F` 또는 `M` 클래스보다 더 잘 예측된다는 것을 보여줍니다. 이 예제는 리샘플을 그룹으로 사용하지만 데이터의 모든 그룹화를 사용할 수 있습니다. 이 `autoplot()` 메소드는 결과 클래스 및/또는 그룹에 걸쳐 모델의 효과성을 보여주는 빠른 시각화 방법이 될 수 있습니다.

# 이 장의 요약 (Chapter Summary)

서로 다른 지표는 모델 피팅의 서로 다른 측면을 측정합니다. 예를 들어 RMSE는 정확도를 측정하는 반면 _R_<sup>2</sup>는 상관관계를 측정합니다. 주어진 모델이 주로 예측에 사용되지 않더라도 모델 성능을 측정하는 것은 중요합니다. 예측력(predictive power)은 추론적 또는 설명적 모델에서도 중요합니다. yardstick 패키지의 함수는 데이터를 사용하여 모델의 효과성을 측정합니다. 주요 tidymodels 인터페이스는 (벡터 인수를 갖는 것과는 반대로) tidyverse 원칙과 데이터 프레임을 사용합니다. 회귀 및 분류 지표에 적합한 지표는 각기 다르며, 그 안에서도 다중 클래스 결과와 같이 통계를 추정하는 방법이 때때로 다릅니다.
