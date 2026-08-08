# 16장. 차원 축소 (Dimensionality Reduction)

차원 축소(Dimensionality reduction)는 데이터 세트를 고차원 공간에서 저차원 공간으로 변환하며, 변수가 "너무 많다"고 의심될 때 좋은 선택이 될 수 있습니다. 변수, 일반적으로 예측 변수의 과잉(excess)은 고차원에서는 데이터를 이해하거나 시각화하기 어렵기 때문에 문제가 될 수 있습니다.

# 차원 축소는 어떤 문제를 해결할 수 있나요? (What Problems Can Dimensionality Reduction Solve?)

차원 축소는 피처 엔지니어링이나 탐색적 데이터 분석(exploratory data analysis)에 사용될 수 있습니다. 예를 들어, 고차원 생물학 실험에서 모델링 이전의 첫 번째 작업 중 하나는 데이터에 원치 않는 추세(trends)가 있는지(예: 실험실 간의 차이와 같이 관심 있는 질문과 관련 없는 효과) 확인하는 것입니다. 수십만 개의 차원이 있을 때 데이터를 디버깅하는 것은 어려우며, 차원 축소는 탐색적 데이터 분석에 도움이 될 수 있습니다(aid).

다수의 예측 변수를 가질 때 발생할 수 있는 또 다른 결과는 모델에 해(harm)가 될 수 있다는 것입니다. 가장 간단한 예는 모델을 피팅하는 데 사용되는 데이터 포인트의 수보다 예측 변수의 수가 적어야 하는 일반 선형 회귀와 같은 방법입니다. 또 다른 문제는 예측 변수 간의 상관관계가 모델을 추정하는 데 사용되는 수학적 연산에 부정적인 영향을 미칠 수 있는 다중공선성(multicollinearity)입니다. 예측 변수의 수가 극도로 많으면 실제 기저 효과(real underlying effects)가 동일한 수로 존재할 가능성은 매우 희박(fairly unlikely)합니다. 예측 변수는 동일한 잠재 효과(latent effect(s))를 측정하고 있을 수 있으며, 따라서 그러한 예측 변수는 높은 상관관계를 갖게 됩니다. 많은 차원 축소 기법은 이러한 상황에서 성공적(thrive)입니다. 사실, 대부분은 활용(exploited)할 수 있는 예측 변수 간의 이러한 관계가 있을 때만 효과적일 수 있습니다.

주성분 분석(Principal component analysis, PCA)은 선형 방법에 의존하고 비지도(unsupervised) 방식(즉, 결과(outcome) 데이터를 고려하지 않음)이므로 데이터 세트의 열 수를 줄이는 가장 직관적인(straightforward) 방법 중 하나입니다. 고차원 분류 문제의 경우 주요 PCA 성분의 초기 플롯이 클래스 간의 명확한 분리(separation)를 보여줄 수 있습니다. 이 경우 선형 분류기가 잘 작동할 것이라고 가정하는 것이 상당히 안전합니다. 그러나 역(converse)은 참이 아닙니다; 분리가 부족하다고 해서 문제가 극복할 수 없는(insurmountable) 것은 아닙니다.

###### 참고 (Note)

새로운 모델링 프로젝트를 시작할 때 데이터의 차원을 줄이면 모델링 문제가 얼마나 어려울 수 있는지에 대한 직관(intuition)을 얻을 수 있습니다.

이 장에서 논의되는 차원 축소 방법은 일반적으로 피처 선택 방법이 *아닙니다*. PCA와 같은 방법은 더 작은 새 피처의 하위 집합(subset)을 사용하여 원래 예측 변수를 나타냅니다(represent). 이러한 새 피처를 계산하려면 원래 예측 변수가 모두 필요합니다. 이에 대한 예외는 새 피처를 만들 때 예측 변수의 영향을 완전히 제거할 수 있는 능력(ability)을 가진 희소(sparse) 방법입니다.

###### 참고 (Note)

이 장에는 두 가지 목표가 있습니다:

- 레시피를 사용하여 원래 예측 변수 세트의 주요 측면(aspects)을 포착(capture)하는 작은 피처 세트를 만드는 방법을 시연합니다.

- 레시피가 어떻게 단독으로 사용될 수 있는지 설명합니다([8장](ch08.xhtml#recipes)에서와 같이 워크플로 객체 내에서 사용되는 것과는 대조적으로).

후자(latter)는 레시피를 테스트하거나 디버깅할 때 도움이 됩니다. 그러나 [8장](ch08.xhtml#recipes)에서 설명한 것처럼 모델링에 레시피를 사용하는 가장 좋은 방법은 워크플로 객체 내에서 사용하는 것입니다.

tidymodels 패키지 외에도 이 장에서는 다음 패키지를 사용합니다: baguette, beans, bestNormalize, corrplot, discrim, embed, ggforce, klaR, [learntidymodels](https://oreil.ly/lyJtX), [mixOmics](https://oreil.ly/DaXYl), uwot.

# 백문이 불여일견… 콩 (A Picture Is Worth a Thousand…Beans)

예제 데이터 세트에 대해 레시피와 함께 차원 축소를 사용하는 방법을 단계별로 살펴보겠습니다(walk through). Koklu and Ozkan (2020)은 말린 콩의 시각적 특성에 대한 데이터 세트를 발표하고 이미지에서 말린 콩의 품종(varieties)을 결정하는 방법을 설명했습니다. 이 데이터의 차원은 많은 실제(real-world) 모델링 문제에 비해 그리 크지 않지만 피처 수를 줄이는 방법을 시연하는 훌륭한 작업 예제를 제공합니다. 그들의 논문(manuscript)에서 발췌:

> 이 연구의 주요 목적(primary objective)은 인구(population) 형태의 작물 생산으로부터 균일한(uniform) 종자 품종을 얻는 방법을 제공하는 것으로, 종자가 단일 품종으로 인증되지(certified as a sole variety) 않도록 하는 것입니다. 따라서 균일한 종자 분류를 얻기 위해 유사한 특징을 가진 7가지의 등록된 말린 콩 품종을 구별하기(distinguish) 위한 컴퓨터 비전 시스템이 개발되었습니다. 분류 모델을 위해 7개의 다른 등록된 말린 콩 곡물 13,611개의 이미지를 고해상도 카메라로 촬영했습니다.

각 이미지에는 여러 개의 콩이 포함되어 있습니다. 어떤 픽셀이 특정 콩에 해당하는지 결정하는 과정을 *이미지 분할(image segmentation)*이라고 합니다. 이러한 픽셀을 분석하여 색상 및 형태(morphology)(즉, 모양)와 같이 각 콩에 대한 특징을 만들 수 있습니다. 다른 콩 품종은 다르게 보이기 때문에 이러한 특징을 사용하여 결과(콩 품종)를 모델링합니다. 훈련 데이터는 수동으로 레이블이 지정된 이미지 세트에서 가져오며, 이 데이터 세트는 Cali, Horoz, Dermason, Seker, Bombay, Barbunya 및 Sira의 7가지 콩 품종을 구별할 수 있는 예측 모델을 만드는 데 사용됩니다. 효과적인 모델을 만들면 제조업체가 한 묶음(batch) 콩의 동질성(homogeneity)을 정량화(quantify)하는 데 도움이 될 수 있습니다.

객체의 모양을 정량화하는 수많은 방법이 있습니다(Yang, Kpalma, and Ronsin 2008). 많은 방법이 관심 객체의 경계(boundaries) 또는 영역(regions)과 관련되어 있습니다. 특징의 예는 다음과 같습니다:

- *면적(area)* (또는 크기)은 객체의 픽셀 수 또는 객체 주변의 볼록 껍질(convex hull)의 크기를 사용하여 추정할 수 있습니다.

- 경계의 픽셀 수와 경계 상자(bounding box)(객체를 둘러싸는 가장 작은 직사각형)의 면적을 사용하여 *둘레(perimeter)*를 측정할 수 있습니다.

- *장축(major axis)*은 객체의 가장 극단적인(extreme) 부분을 연결하는 가장 긴 선을 정량화합니다. *단축(minor axis)*은 장축에 수직(perpendicular)입니다.

- 동일한 둘레를 가진 원의 면적에 대한 객체 면적의 비율을 사용하여 객체의 *조밀도(compactness)*를 측정할 수 있습니다. 예를 들어 기호 "•"와 "×"는 조밀도가 매우 다릅니다.

- 객체가 얼마나 *길쭉한지(elongated)* 또는 직사각형 모양(oblong)인지에 대한 다양한 측정값도 있습니다. 예를 들어, *이심률(eccentricity)* 통계량은 장축과 단축의 비율입니다. 둥글기(roundness)와 볼록함(convexity)에 대한 관련 추정치도 있습니다.

[그림 16-1](#eccentricity)에서 여러 모양에 대한 이심률을 주목하십시오(Notice).

원이나 정사각형과 같은 모양은 이심률이 낮은 반면 길쭉한 모양은 값이 높습니다. 또한 이 메트릭은 객체의 회전에 영향을 받지 않습니다(unaffected).

이러한 많은 이미지 특징은 높은 상관관계를 가집니다; 면적이 큰 객체는 둘레도 클 가능성이 더 높습니다. 종종 동일한 근본적인(underlying) 특성(예: 크기)을 정량화하는 데는 여러 가지 방법이 있습니다.

콩 데이터에서는 면적, 둘레, 장축 길이, 단축 길이, 종횡비(aspect ratio), 이심률, 볼록 면적, 등가 직경(equivalent diameter), 범위(extent), 견고성(solidity), 둥글기, 조밀도, 형상(shape) 계수 1, 형상 계수 2, 형상 계수 3, 형상 계수 4의 16가지 형태적(morphology) 특징이 계산되었습니다. 후자의 네 가지는 Symons and Fulcher (1988)에 설명되어 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1601.png" alt="tmwr 1601" />
<h6 id="figure-16-1.-some-example-shapes-and-their-eccentricity-statistics.">그림 16-1. 몇 가지 모양 예시와 그 이심률 통계량.</h6>
</figure>

데이터를 로드하는 것으로 시작할 수 있습니다:

```
library(tidymodels)
tidymodels_prefer()
library(beans)
```

###### 경고 (Warning)

차원 축소 기법을 평가할 때, 특히 모델 내에서 사용할 경우 좋은 데이터 규율(data discipline)을 유지하는 것이 중요합니다.

분석을 위해 먼저 `initial_split()`으로 테스트 세트를 보류(holding back)하는 것부터 시작합니다. 나머지 데이터는 훈련 및 검증 세트로 분할됩니다:

```
set.seed(1601)
bean_split <- initial_split(beans, strata = class, prop = 3/4)

bean_train <- training(bean_split)
bean_test  <- testing(bean_split)

set.seed(1602)
bean_val <- validation_split(bean_train, strata = class, prop = 4/5)
bean_val$splits[[1]]
#> <Training/Validation/Total>
#> <8163/2043/10206>
```

다른 방법들이 얼마나 잘 수행되는지 시각적으로 평가하기 위해 훈련 세트(*n* = 8,163개의 콩)에서 방법을 추정하고 검증 세트(*n* = 2,043)를 사용하여 결과를 표시(display)할 수 있습니다.

차원 축소를 시작하기 전에 데이터를 조사(investigating)하는 데 시간을 할애(spend some time)할 수 있습니다. 이러한 여러 모양 특징이 아마도(probably) 유사한 개념을 측정하고 있다는 것을 알고 있으므로, 다음 코드를 사용하여 [그림 16-2](#beans-corr-plot)에 있는 데이터의 상관관계 구조(correlation structure)를 살펴보겠습니다:

```
library(corrplot)
tmwr_cols <- colorRampPalette(c("#91CBD765", "#CA225E"))
bean_train %>%
  select(-class) %>%
  cor() %>%
  corrplot(col = tmwr_cols(200), tl.col = "black", method = "ellipse")
```

<figure class="width-90">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1602.png" alt="tmwr 1602" />
<h6 id="figure-16-2.-correlation-matrix-of-the-predictors-with-variables-ordered-via-clustering.">그림 16-2. 클러스터링을 통해 정렬된 변수들이 있는 예측 변수의 상관행렬.</h6>
</figure>

면적과 둘레 또는 형상 계수 2와 3과 같이 이러한 예측 변수의 대부분은 높은 상관관계를 보입니다. 여기서 그것을 할(do it) 시간은 갖지 않지만, 결과(outcome) 범주(categories) 전체에 걸쳐 이 상관 구조가 크게(significantly) 변하는지 확인하는 것도 중요합니다. 이는 더 나은 모델을 만드는 데 도움이 될 수 있습니다.

# 스타터 레시피 (A Starter Recipe)

이제 더 작은 공간에서 콩 데이터를 살펴볼(look at) 시간입니다. 차원 축소 단계 이전에 데이터를 전처리하기 위한 기본 레시피로 시작할 수 있습니다. 몇 가지 예측 변수는 비율(ratios)이므로 비대칭 분포(skewed distributions)를 가질 가능성이 높습니다. 이러한 분포는 (PCA에서 사용되는 것과 같은) 분산 계산을 망칠(wreak havoc on) 수 있습니다. [bestNormalize 패키지](https://oreil.ly/v26pw)에는 예측 변수에 대해 대칭 분포를 강제(enforce)할 수 있는 단계(step)가 있습니다. 이를 사용하여 비대칭 분포 문제를 완화(mitigate)하겠습니다:

```
library(bestNormalize)
bean_rec <-
  # bean_val 분할 객체에서 훈련 데이터 사용
  recipe(class ~ ., data = analysis(bean_val$splits[[1]])) %>%
  step_zv(all_numeric_predictors()) %>%
  step_orderNorm(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())
```

###### 참고 (Note)

`recipe()` 함수를 호출할(invoking) 때 단계(steps)가 어떤 방식으로든 추정(estimated)되거나 실행(executed)되지 않는다는 점을 기억하십시오.

이 레시피는 차원 축소 분석을 위한 추가 단계로 확장(extended)될 것입니다. 그전에, 레시피가 워크플로 외부에서(outside of) 어떻게 사용될 수 있는지 살펴보겠습니다(go over).

# 야생의 레시피 (Recipes in the Wild)

[8장](ch08.xhtml#recipes)에서 언급했듯이, 레시피가 포함된 워크플로는 `fit()`을 사용하여 레시피와 모델을 추정하고, `predict()`를 사용하여 데이터를 처리하고 모델 예측을 수행합니다. recipes 패키지에는 같은 목적으로 사용할 수 있는 유사한(analogous) 함수가 있습니다:

- `prep(recipe, training)`은 훈련 세트에 레시피를 피팅합니다.

- `bake(recipe, new_data)`는 `new_data`에 레시피 연산을 적용합니다.

[그림 16-3](#recipe-process)은 이를 요약합니다. 이러한 각 기능을 더 자세히(in more detail) 살펴보겠습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1603.png" alt="tmwr 1603" />
<h6 id="figure-16-3.-summary-of-recipe-related-functions.">그림 16-3. 레시피 관련 함수 요약.</h6>
</figure>

## 레시피 준비하기 (Preparing a Recipe)

`prep(bean_rec)`을 사용하여 훈련 세트 데이터로 `bean_rec`을 추정해 보겠습니다:

```
bean_rec_trained <- prep(bean_rec)
bean_rec_trained
#> Recipe
#>
#> Inputs:
#>
#>       role #variables
#>    outcome          1
#>  predictor         16
#>
#> Training data contained 8163 data points and no missing data.
#>
#> Operations:
#>
#> Zero variance filter removed <none> [trained]
#> orderNorm transformation on area, perimeter, major_axis_length, minor_axis... [trained]
#> Centering and scaling for area, perimeter, major_axis_length, minor_axis_... [trained]
```

###### 참고 (Note)

레시피에 대한 `prep()`은 모델에 대한 `fit()`과 같다는 것을 기억하십시오.

출력에서 단계들이 훈련되었고(trained) 선택기(selectors)가 더 이상 일반적(general)이지 않다는(즉, `all_numeric_predictors()`) 점에 주목하십시오. 이제 선택된 실제 열을 보여줍니다. 또한 `prep(bean_rec)`은 `training` 인수를 필요로 하지 않습니다(does not require). 해당 인수에 어떤 데이터든 전달(pass)할 수 있지만, 생략(omitting)하면 `recipe()` 호출의 원래 `data`가 사용됩니다. 우리의 경우 이것은 훈련 세트 데이터였습니다.

`prep()`의 한 가지 중요한 인수는 `retain`입니다. `retain = TRUE`(기본값)인 경우 훈련 세트의 추정된 버전이 레시피 내에 보관(kept)됩니다. 이 데이터 세트는 레시피에 나열된 모든 단계를 사용하여 전처리되었습니다. `prep()`은 진행됨에 따라(proceeds) 레시피를 실행해야 하므로(execute), 나중에 해당 데이터 세트를 사용할 경우 중복(redundant) 계산을 피할(avoided) 수 있도록 이 버전의 훈련 세트를 유지하는 것이 유리(advantageous)할 수 있습니다. 그러나 훈련 세트가 큰 경우 그렇게 많은 양의 데이터를 메모리에 보관하는 것은 문제(problematic)가 될 수 있습니다. 이를 피하려면 `retain = FALSE`를 사용하십시오.

이 추정된 레시피에 새로운 단계가 추가되면, 다시 적용(reapplying)된 `prep()`은 훈련되지 않은 단계만 추정합니다. 이것은 우리가 다른 피처 추출 방법을 시도할 때 유용하게 쓰일(come in handy) 것입니다.

###### 경고 (Warning)

레시피로 작업할 때 오류가 발생(encounter)하면, 문제 해결(troubleshoot)을 위해 `prep()`을 `verbose` 옵션과 함께 사용할 수 있습니다:

```
bean_rec_trained %>%
  step_dummy(cornbread) %>%  # <- 실제 예측 변수가 아님
  prep(verbose = TRUE)
#> oper 1 step zv [pre-trained]
#> oper 2 step orderNorm [pre-trained]
#> oper 3 step normalize [pre-trained]
#> oper 4 step dummy [training]
#> Error in `chr_as_locations()`:
#> ! Can't subset columns that don't exist.
#> ✖ Column `cornbread` doesn't exist.
```

분석에서 일어나는 일을 이해하는 데 도움이 될 수 있는 또 다른 옵션은 `log_changes`입니다:

```
show_variables <-
  bean_rec %>%
  prep(log_changes = TRUE)
#> step_zv (zv_6JtxV): same number of columns
#>
#> step_orderNorm (orderNorm_4r8al): same number of columns
#>
#> step_normalize (normalize_x6oqH): same number of columns
```

## 레시피 굽기 (Baking the Recipe)

###### 참고 (Note)

레시피와 함께 `bake()`를 사용하는 것은 모델과 함께 `predict()`를 사용하는 것과 매우(much) 비슷합니다; 훈련 세트에서 추정된 연산(operations)은 테스트 데이터 또는 예측 시점의(at prediction time) 새로운 데이터와 같은 모든 데이터에 적용됩니다.

예를 들어, 검증 세트 샘플을 처리할 수 있습니다:

```
bean_validation <- bean_val$splits %>% pluck(1) %>% assessment()
bean_val_processed <- bake(bean_rec_trained, new_data = bean_validation)
```

[그림 16-4](#bean-area)는 레시피가 준비되기 전후의 `area` 예측 변수 히스토그램을 보여줍니다:

```
library(patchwork)
p1 <-
  bean_validation %>%
  ggplot(aes(x = area)) +
  geom_histogram(bins = 30, color = "white", fill = "blue", alpha = 1/3) +
  ggtitle("Original validation set data")

p2 <-
  bean_val_processed %>%
  ggplot(aes(x = area)) +
  geom_histogram(bins = 30, color = "white", fill = "red", alpha = 1/3) +
  ggtitle("Processed validation set data")

p1 + p2
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1604.png" alt="tmwr 1604" />
<h6 id="figure-16-4.-the-area-predictor-before-and-after-preprocessing.">그림 16-4. 전처리 전후의 <code>area</code> 예측 변수.</h6>
</figure>

`bake()`의 두 가지 중요한 측면(aspects)을 여기서 주목할 가치(worth noting)가 있습니다.

첫째, 앞에서 언급했듯이 `prep(recipe, retain = TRUE)`를 사용하면 기존의(existing) 처리된 버전의 훈련 세트가 레시피에 유지(keeps)됩니다. 이를 통해 사용자는 `bake(recipe, new_data = NULL)`을 사용할 수 있으며, 이는 추가 계산 없이 해당 데이터 세트를 반환합니다. 예를 들어:
```
bake(bean_rec_trained, new_data = NULL) %>% nrow()
#> [1] 8163
bean_val$splits %>% pluck(1) %>% analysis() %>% nrow()
#> [1] 8163
```

훈련 세트가 병적으로(pathologically) 크지 않은 경우 이 `retain` 값을 사용하면 계산 시간을 많이 절약(save)할 수 있습니다.

둘째, 어떤 열을 반환할지 지정하기 위해 호출 시 추가 선택기를 사용할 수 있습니다. 기본 선택기는 `everything()`이지만 더 구체적인 지시자(directives)를 사용할 수 있습니다.

이러한 옵션 중 일부를 설명(illustrate)하기 위해 다음 섹션에서 `prep()` 및 `bake()`를 사용할 것입니다.

# 피처 추출 기법 (Feature Extraction Techniques)

레시피는 tidymodels에서 차원 축소를 위한 주요 옵션이므로, 변환을 추정하고 결과 데이터를 플로팅(plot)하는 함수를 작성해 보겠습니다:

```
plot_validation_results <- function(recipe, dat = assessment(bean_val$splits[[1]])) {
  set.seed(1)
  plot_data <-
    recipe %>%
    # 추가 단계 추정
    prep() %>%
    # 데이터 처리 (기본적으로 검증 세트)
    bake(new_data = dat, all_predictors(), all_outcomes()) %>%
    # 데이터를 더 읽기 쉽도록 샘플링
    sample_n(250)

  # 준인용(quasiquotation)과 함께 사용할 수 있도록 피처 이름을 기호로 변환
  nms <- names(plot_data)
  x_name <- sym(nms[1])
  y_name <- sym(nms[2])

  plot_data %>%
    ggplot(aes(x = !!x_name, y = !!y_name, col = class,
               fill = class, pch = class)) +
    geom_point(alpha = 0.9) +
    scale_shape_manual(values = 1:7) +
    # 동일한 크기의 축 만들기
    coord_obs_pred() +
    theme_bw()
}
```

이 장에서 이 함수를 여러 번 재사용할 것입니다.

일련의 여러 피처 추출 방법론이 여기서 탐색됩니다(explored). 대부분의 개요는 [Kuhn and Johnson (2020)의 6.3.1절](https://oreil.ly/xllmg)과 그 안의 참고문헌에서 찾을 수 있습니다. UMAP 방법은 McInnes, Healy, and Melville (2020)에 설명되어 있습니다.

## 주성분 분석 (Principal Component Analysis)

이 책에서 이미 PCA를 여러 번 언급했으며, 이제 더 자세히 알아볼(go into more detail) 시간입니다. PCA는 예측 변수의 선형 조합을 사용하여 새로운 피처를 정의하는 비지도 방법입니다. 이러한 피처는 원본 데이터에서 가능한 한 많은 분산(variation)을 설명(account for)하려고 시도합니다. 원래 레시피에 `step_pca()`를 추가하고 함수를 사용하여 [그림 16-5](#bean-pca)에서 검증 세트의 결과를 시각화합니다:

```
bean_rec_trained %>%
  step_pca(all_numeric_predictors(), num_comp = 4) %>%
  plot_validation_results() +
  ggtitle("Principal Component Analysis")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1605.png" alt="tmwr 1605" />
<h6 id="figure-16-5.-first-two-principal-component-scores-for-the-bean-validation-set-by-class.">그림 16-5. 콩 검증 세트에 대한 클래스별 처음 두 주성분 점수.</h6>
</figure>

처음 두 성분인 `PC1`과 `PC2`가, 특히 함께 사용될 때, 클래스 간 구별(distinguishing)이나 분리(separating)에 효과적인 역할을 하는 것을 볼 수 있습니다. 이로 인해 이러한 콩을 분류하는 전반적인 문제가 특별히 어렵지는 않을 것이라고 예상(expect)할 수 있습니다.

PCA가 비지도 방식이라는 것을 상기하십시오. 이 데이터에 대해 예측 변수의 분산을 가장 많이 설명하는 PCA 성분은 클래스를 예측하는(predictive of) 역할도 우연히(happen to) 하는 것으로 나타났습니다. 어떤 피처가 성능을 주도(driving)하고 있을까요? learntidymodels 패키지에는 각 성분에 대한 상위 피처를 시각화하는 데 도움이 되는 함수가 있습니다. 준비된 레시피가 필요합니다; 다음 코드에서는 `prep()` 호출과 함께 PCA 단계가 추가됩니다:

```
library(learntidymodels)
bean_rec_trained %>%
  step_pca(all_numeric_predictors(), num_comp = 4) %>%
  prep() %>%
  plot_top_loadings(component_number <= 4, n = 5) +
  scale_fill_brewer(palette = "Paired") +
  ggtitle("Principal Component Analysis")
```

이렇게 하면 [그림 16-6](#pca-loadings)이 생성됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1606.png" alt="tmwr 1606" />
<h6 id="figure-16-6.-predictor-loadings-for-the-pca-transformation.">그림 16-6. PCA 변환을 위한 예측 변수 로딩.</h6>
</figure>

상위 로딩은 대부분(mostly) 이전 상관관계 플롯의 왼쪽 상단 부분에 표시된 상관 예측 변수(correlated predictors) 클러스터와 관련(related)되어 있습니다: 둘레, 면적, 장축 길이 및 볼록 면적. 이들은 모두 콩 크기와 관련이 있습니다. Symons and Fulcher (1988)의 형상 계수 2는 장축 길이의 세제곱에 대한 면적(area over the cube of the major axis length)이므로 역시 콩 크기와 관련이 있습니다. 신율(elongation) 측정값이 두 번째 PCA 성분을 지배(dominate)하는 것으로 보입니다.

## 부분 최소 제곱 (Partial Least Squares)

[“하위 모델 최적화 (Submodel Optimization)”](ch13.xhtml#submodel-trick)에서 소개한 PLS는 PCA의 지도(supervised) 버전입니다. 이 방법은 예측 변수의 분산을 최대화하는 동시에 해당 성분과 결과(outcome) 간의 관계도 최대화하는 성분을 찾으려고 시도합니다. [그림 16-7](#bean-pls)은 약간 수정된 버전의 PCA 코드 결과를 보여줍니다:

```
bean_rec_trained %>%
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = 4) %>%
  plot_validation_results() +
  ggtitle("Partial Least Squares")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1607.png" alt="tmwr 1607" />
<h6 id="figure-16-7.-first-two-pls-component-scores-for-the-bean-validation-set-by-class.">그림 16-7. 콩 검증 세트에 대한 클래스별 처음 두 PLS 성분 점수.</h6>
</figure>

[그림 16-7](#bean-pls)에 플로팅된 처음 두 PLS 성분은 처음 두 PCA 성분과 거의(nearly) 동일(identical)합니다! 이러한 PCA 성분이 콩의 품종을 분리하는 데 매우 효과적이기 때문에 이러한 결과를 찾은 것입니다. 나머지 성분은 다릅니다. [그림 16-8](#pls-loadings)은 각 성분의 상위 피처인 로딩을 시각화합니다:

```
bean_rec_trained %>%
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = 4) %>%
  prep() %>%
  plot_top_loadings(component_number <= 4, n = 5, type = "pls") +
  scale_fill_brewer(palette = "Paired") +
  ggtitle("Partial Least Squares")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1608.png" alt="tmwr 1608" />
<h6 id="figure-16-8.-predictor-loadings-for-the-pls-transformation.">그림 16-8. PLS 변환을 위한 예측 변수 로딩.</h6>
</figure>

견고성(즉, 콩의 밀도(density))은 둥글기(roundness)와 함께 세 번째 PLS 성분을 주도합니다. 견고성은 콩 경계의 불규칙성을 측정할 수 있기 때문에 콩 표면의 "울퉁불퉁함(bumpiness)"과 관련된 콩 특징을 포착(capturing)하고 있을 수 있습니다.

## 독립 성분 분석 (Independent Component Analysis)

ICA는 (상관관계가 없는 것과는 대조적으로) 통계적으로 서로 가능한 한 독립적인 성분을 찾는다는 점에서 PCA와 약간 다릅니다. ICA 성분의 "비-가우시안성(non-Gaussianity)"을 최대화하거나 PCA처럼 정보를 압축하는 대신 정보를 분리(separating)하는 것으로 생각(thought of)할 수 있습니다. [그림 16-9](#bean-ica)를 생성하기 위해 `step_ica()`를 사용해 보겠습니다:

```
bean_rec_trained %>%
  step_ica(all_numeric_predictors(), num_comp = 4) %>%
  plot_validation_results() +
  ggtitle("Independent Component Analysis")
```

이 플롯을 조사(Inspecting)해 보면, ICA를 사용할 때 처음 몇 개의 성분에서 클래스 간의 분리가 많이 나타나지 않습니다. 이러한 독립적인(또는 가능한 한 독립적인) 성분은 콩 유형을 분리하지 못합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1609.png" alt="tmwr 1609" />
<h6 id="figure-16-9.-first-two-ica-component-scores-for-the-bean-validation-set-by-class.">그림 16-9. 콩 검증 세트에 대한 클래스별 처음 두 ICA 성분 점수.</h6>
</figure>

## 균일 매니폴드 근사 및 투영 (Uniform Manifold Approximation and Projection)

UMAP은 비선형 차원 축소를 위해 널리 사용되는(popular) t-SNE 방법과 유사합니다. 원본 고차원 공간에서 UMAP은 거리 기반 최근접 이웃(distance-based nearest neighbor) 방법을 사용하여 데이터 포인트가 연관될 가능성이 더 높은 데이터의 국소적(local) 영역을 찾습니다. 데이터 포인트 간의 관계는 대부분의 포인트가 연결되지 않은 유향 그래프(directed graph) 모델로 저장(saved)됩니다.

거기서부터(From there), UMAP은 그래프의 포인트를 축소된 차원 공간으로 변환(translates)합니다. 이를 수행(To do this)하기 위해 알고리즘에는 교차 엔트로피(cross-entropy)를 사용하여 그래프가 잘 근사(approximated)되도록 더 작은 피처 세트에 데이터 포인트를 매핑하는 최적화 프로세스가 있습니다.

매핑을 생성(create)하기 위해 embed 패키지에는 이 방법을 위한 단계 함수가 포함되어 있으며 [그림 16-10](#bean-umap)에 시각화되어 있습니다:

```
library(embed)
bean_rec_trained %>%
  step_umap(all_numeric_predictors(), num_comp = 4) %>%
  plot_validation_results() +
  ggtitle("UMAP")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1610.png" alt="tmwr 1610" />
<h6 id="figure-16-10.-the-first-two-umap-component-scores-for-the-bean-validation-set-by-class.-results-are-shown-for-unsupervised-and-supervised-versions.">그림 16-10. 콩 검증 세트에 대한 클래스별 처음 두 UMAP 성분 점수. 결과는 비지도 및 지도 버전에 대해 표시됩니다.</h6>
</figure>

결과 플롯은 [그림 16-10](#bean-umap)의 왼쪽에 표시됩니다. 클러스터 간 공간(between-cluster space)이 뚜렷하지만(pronounced) 클러스터에는 서로 다른 클래스가 이질적으로 혼합(heterogeneous mixture)되어 포함될 수 있습니다.

UMAP의 지도 버전도 있습니다:

```
bean_rec_trained %>%
  step_umap(all_numeric_predictors(), outcome = "class", num_comp = 4) %>%
  plot_validation_results() +
  ggtitle("UMAP (supervised)")
```

[그림 16-10](#bean-umap)에 표시된 지도 방법은 데이터를 모델링하는 데 유망(promising)해 보입니다.

UMAP은 피처 공간을 축소하는 강력한 방법입니다. 그러나 튜닝 매개변수(예: 이웃의 수 등)에 매우 민감할(sensitive) 수 있습니다. 이러한 이유(For this reason)로 이러한 데이터에 대해 결과가 얼마나 견고한지(robust) 평가하기 위해 몇 가지 매개변수를 실험해 보는 것이 도움이 될 것입니다(would help).

# 모델링 (Modeling)

PLS 및 UMAP 방법 모두 다양한 모델과 함께 조사(investigating)할 가치가 있습니다. 변환을 전혀 수행하지 않는 것과 함께 이러한 차원 축소 기법을 적용한 단일 레이어 신경망, 배깅된(bagged) 트리, 유연한 판별 분석(FDA), 나이브 베이즈 및 정규화 판별 분석(RDA) 등 다양하고 다른 모델을 살펴보겠습니다(explore).

이제 "모델링 모드"로 돌아왔으므로(Now that we are back in "modeling mode"), 다음 코드에서 일련의(a series of) 모델 사양을 만든 다음 워크플로 세트를 사용하여 모델을 튜닝(tune)하겠습니다. 모델 매개변수는 레시피 매개변수(예: 축소된 차원의 크기, UMAP 매개변수)와 함께(in conjunction with) 튜닝(tuned)됩니다:

```
library(baguette)
library(discrim)

mlp_spec <-
  mlp(hidden_units = tune(), penalty = tune(), epochs = tune()) %>%
  set_engine('nnet') %>%
  set_mode('classification')

bagging_spec <-
  bag_tree() %>%
  set_engine('rpart') %>%
  set_mode('classification')

fda_spec <-
  discrim_flexible(
    prod_degree = tune()
  ) %>%
  set_engine('earth')

rda_spec <-
  discrim_regularized(frac_common_cov = tune(), frac_identity = tune()) %>%
  set_engine('klaR')

bayes_spec <-
  naive_Bayes() %>%
  set_engine('klaR')
```

시도해 볼(we'll try) 차원 축소 방법을 위한 레시피도 필요합니다. 기본 레시피인 `bean_rec`로 시작하여 다양한 차원 축소 단계로 확장(extend)해 보겠습니다:

```
bean_rec <-
  recipe(class ~ ., data = bean_train) %>%
  step_zv(all_numeric_predictors()) %>%
  step_orderNorm(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())

pls_rec <-
  bean_rec %>%
  step_pls(all_numeric_predictors(), outcome = "class", num_comp = tune())

umap_rec <-
  bean_rec %>%
  step_umap(
    all_numeric_predictors(),
    outcome = "class",
    num_comp = tune(),
    neighbors = tune(),
    min_dist = tune()
  )
```

다시 한번 말하지만(Once again), workflowsets 패키지는 전처리기와 모델을 가져와 서로 교차(crosses)시킵니다. 튜닝 매개변수 조합 전체에 걸쳐(across) 병렬 처리가 동시에(simultaneously) 작동(work)할 수 있도록 `control` 옵션인 `parallel_over`가 설정됩니다. `workflow_map()` 함수는 그리드 검색을 적용하여 10개의 매개변수 조합에 걸쳐 모델/전처리 매개변수(있는 경우)를 최적화합니다. 다중 클래스(multiclass) ROC 곡선 아래 면적이 검증 세트에서 추정됩니다:

```
ctrl <- control_grid(parallel_over = "everything")
bean_res <-
  workflow_set(
    preproc = list(basic = class ~., pls = pls_rec, umap = umap_rec),
    models = list(bayes = bayes_spec, fda = fda_spec,
                  rda = rda_spec, bag = bagging_spec,
                  mlp = mlp_spec)
  ) %>%
  workflow_map(
    verbose = TRUE,
    seed = 1603,
    resamples = bean_val,
    grid = 10,
    metrics = metric_set(roc_auc),
    control = ctrl
  )
```

ROC 곡선 아래 면적의 검증 세트 추정치를 기준으로 모델의 순위를 매길 수 있습니다:

```
rankings <-
  rank_results(bean_res, select_best = TRUE) %>%
  mutate(method = map_chr(wflow_id, ~ str_split(.x, "_", simplify = TRUE)[1]))

tidymodels_prefer()
filter(rankings, rank <= 5) %>% dplyr::select(rank, mean, model, method)
#> # A tibble: 5 × 4
#>    rank  mean model               method
#>   <int> <dbl> <chr>               <chr>
#> 1     1 0.995 mlp                 basic
#> 2     2 0.995 discrim_regularized pls
#> 3     3 0.994 mlp                 pls
#> 4     4 0.994 naive_Bayes         pls
#> 5     5 0.994 discrim_flexible    basic
```

[그림 16-11](#dimensionality-rankings)은 이 순위를 설명(illustrates)합니다.

이러한 결과에서 대부분의 모델이 매우 좋은 성능을 제공(give)한다는 것은 분명합니다; 나쁜 선택은 거의 없습니다. 시연을 위해 우리는 PLS 피처가 있는 RDA 모델을 최종 모델로 사용할 것입니다. 수치상 가장 좋은 매개변수로 워크플로를 마무리(finalize)하고 훈련 세트에 피팅한 다음 테스트 세트로 평가(evaluate)합니다:

```
rda_res <-
  bean_res %>%
  extract_workflow("pls_rda") %>%
  finalize_workflow(
    bean_res %>%
      extract_workflow_set_result("pls_rda") %>%
      select_best(metric = "roc_auc")
  ) %>%
  last_fit(split = bean_split, metrics = metric_set(roc_auc))

rda_wflow_fit <- rda_res$.workflow[[1]]
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_1611.png" alt="tmwr 1611" />
<h6 id="figure-16-11.-area-under-the-roc-curve-from-the-validation-set.">그림 16-11. 검증 세트의 ROC 곡선 아래 면적.</h6>
</figure>

테스트 세트에서 메트릭(다중 클래스 ROC AUC) 결과는 어떻습니까?

```
collect_metrics(rda_res)
#> # A tibble: 1 × 4
#>   .metric .estimator .estimate .config
#>   <chr>   <chr>          <dbl> <chr>
#> 1 roc_auc hand_till      0.995 Preprocessor1_Model1
```

꽤 좋네요! 변수 중요도(variable importance) 방법을 시연하기 위해 [18장](ch18.xhtml#explain)에서 이 모델을 사용할 것입니다.

# 이 장의 요약 (Chapter Summary)

차원 축소는 모델링뿐만 아니라 탐색적 데이터 분석에 도움이 되는 방법이 될 수 있습니다. recipes 및 embed 패키지에는 다양하고 다른 방법들과 워크플로 세트를 위한 단계들이 포함되어 있어 데이터 세트에 적합한 방법을 더 쉽게(facilitates) 선택할 수 있습니다. 이 장에서는 또한 레시피의 문제 디버깅을 위해서나, 탐색적 데이터 분석 및 데이터 시각화를 위해 레시피 단독으로(on their own) 사용하는 방법에 대해서도 논의했습니다.
