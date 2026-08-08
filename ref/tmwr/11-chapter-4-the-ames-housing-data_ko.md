# 4장. Ames 주택 데이터 (The Ames Housing Data)

이 장에서는 이 책 전반에 걸쳐 모델링 예제에서 사용할 Ames 주택 데이터 세트 (De Cock 2011)를 소개합니다. 이 장에서 다루는 것과 같은 탐색적 데이터 분석(Exploratory data analysis)은 신뢰할 수 있는 모델을 구축하는 데 중요한 첫 번째 단계입니다. 이 데이터 세트에는 아이오와주 에임스(Ames)에 있는 2,930개 부동산에 대한 정보가 포함되어 있으며 다음과 관련된 열이 있습니다:

- 주택 특성 (침실, 차고, 벽난로, 수영장, 현관 등)

- 위치 (동네)

- 부지 정보 (구역(zoning), 모양, 크기 등)

- 상태(condition) 및 품질(quality) 평가

- 판매 가격(Sale price)

원시 주택 데이터는 De Cock (2011)에 제공되어 있지만, 이 책의 분석에서는 modeldata 패키지에서 사용할 수 있는 변환된 버전을 사용합니다. 이 버전에는 데이터에 대한 몇 가지 [변경 및 개선 사항](https://oreil.ly/OSIQ0)이 있습니다. 예를 들어, 각 부동산에 대해 경도와 위도 값이 결정되었습니다. 또한 일부 열은 더 분석에 적합하도록 수정되었습니다. 예를 들어:

- 원시 데이터에서 주택에 특정 특성이 없으면 암묵적으로 결측값으로 인코딩되었습니다. 예를 들어 2,732개의 부동산에는 골목(alleyway)이 없었습니다. 이들을 결측치로 두는 대신 변환된 버전에서는 골목이 없음을 나타내도록 레이블이 다시 지정되었습니다.

- 범주형 예측 변수(categorical predictors)는 R의 팩터(factor) 데이터 유형으로 변환되었습니다. tidyverse와 base R 모두 기본적으로 데이터를 팩터로 가져오는 방식에서 멀어졌지만, 모델링을 위해 질적 데이터(qualitative data)를 저장하는 데에는 이 데이터 유형이 단순한 문자열보다 더 나은 접근 방식입니다.\

- 각 주택에 대한 일련의 품질 설명자(quality descriptors)는 예측 변수라기보다 결과 변수(outcomes)에 가깝기 때문에 제거했습니다.

데이터를 로드하려면:

```
library(modeldata) # 이것은 tidymodels 패키지에 의해서도 로드됩니다
data(ames)

# 또는 한 줄로:
data(ames, package = "modeldata")

dim(ames)
#> [1] 2930   74
```

[그림 4-1](#figure-4-1.-property-locations-in-ames-iowa.)은 에임스의 부동산 위치를 보여줍니다. 위치는 다음 섹션에서 다시 다루게 됩니다.

<figure class="width-80">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0401.png" alt="tmwr 0401" />
<h6 id="figure-4-1.-property-locations-in-ames-iowa.">그림 4-1. 아이오와주 에임스의 부동산 위치.</h6>
</figure>

에임스 중심부의 빈 데이터 포인트는 아이오와 주립대학교(Iowa State University)에 해당합니다.

###### 참고 (Note)

우리의 모델링 목표는 특성 및 위치와 같이 우리가 가진 다른 정보를 바탕으로 주택의 판매 가격을 예측하는 것입니다.

# Ames의 주택 특징 탐색 (Exploring Features of Homes in Ames)

우리가 예측하려는 결과 변수인 주택의 마지막 판매 가격(USD)에 초점을 맞추어 탐색적 데이터 분석을 시작하겠습니다. [그림 4-2](#figure-4-2.-sale-prices-of-houses-in-ames-iowa.)에서 판매 가격의 분포를 보기 위해 히스토그램을 생성할 수 있습니다:

```
library(tidymodels)
tidymodels_prefer()

ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram(bins = 50, col= "white")
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0402.png" alt="tmwr 0402" />
<h6 id="figure-4-2.-sale-prices-of-houses-in-ames-iowa.">그림 4-2. 아이오와주 에임스의 주택 판매 가격.</h6>
</figure>

이 플롯은 데이터가 오른쪽으로 치우쳐(right-skewed) 있음을 보여줍니다. 비싼 주택보다 저렴한 주택이 더 많습니다. 판매 가격의 중앙값(median)은 $160,000이었고 가장 비싼 주택은 $755,000였습니다. 이 결과를 모델링할 때, 가격을 로그 변환(log-transformed)해야 한다는 강력한 주장이 제기될 수 있습니다. 이러한 유형의 변환이 갖는 이점은 음수 판매 가격으로 예측되는 주택이 없다는 점과, 비싼 주택을 예측할 때 발생하는 오차가 모델에 과도한 영향을 미치지 않는다는 점입니다. 또한 통계적인 관점에서 로그 변환은 추론을 더 합법적으로 만드는 방식으로 분산(variance)을 안정화시킬 수도 있습니다. 이제 유사한 단계를 사용하여 [그림 4-3](#figure-4-3.-sale-prices-of-houses-in-ames-iowa-after-a-log-base-10-transformation.)에 표시된 대로 변환된 데이터를 시각화할 수 있습니다:

```
ggplot(ames, aes(x = Sale_Price)) +
  geom_histogram(bins = 50, col= "white") +
  scale_x_log10()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0403.png" alt="tmwr 0403" />
<h6 id="figure-4-3.-sale-prices-of-houses-in-ames-iowa-after-a-log-base-10-transformation.">그림 4-3. 밑이 10인 로그 변환을 거친 후의 아이오와주 에임스 주택 판매 가격.</h6>
</figure>

완벽하지는 않지만 방금 설명한 이유로 인해 변환되지 않은 데이터를 사용하는 것보다 더 나은 모델을 만들 수 있을 것입니다.

###### 경고 (Warning)

결과 변수를 변환할 때의 단점은 대부분 모델 결과의 해석과 관련이 있습니다.

성능 측정과 마찬가지로 모델 계수의 단위를 해석하기 더 어려울 수 있습니다. 예를 들어 *평균 제곱근 오차(root mean squared error)* (RMSE)는 회귀 모델에서 사용되는 일반적인 성능 지표입니다. 이것은 관측값과 예측값의 차이를 계산에 사용합니다. 판매 가격이 로그 척도(log scale)에 있는 경우 이 차이(즉, 잔차)도 로그 척도에 있습니다. 이러한 로그 척도에서 RMSE가 0.15인 모델의 품질을 이해하기 어려울 수 있습니다.

이러한 단점에도 불구하고 이 책에서 사용되는 모델은 이 결과 변수에 대해 로그 변환을 사용합니다. *이 시점부터*, 결과 변수 열은 `ames` 데이터 프레임에 사전 로그 변환(prelogged)됩니다:

```
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
```

우리 모델링을 위한 이 데이터의 또 다른 중요한 측면은 지리적 위치입니다. 이 공간 정보는 두 가지 방식으로 데이터에 포함되어 있습니다: 질적인 `Neighborhood` 레이블과 양적인 경도 및 위도 데이터입니다. 공간 정보를 시각화하기 위해 [그림 4-4](#figure-4-4.-neighborhoods-in-ames-represented-using-a-convex-hull.)는 각 동네의 데이터 주변에 볼록 껍질(convex hulls)을 사용하여 [그림 4-1](#figure-4-1.-property-locations-in-ames-iowa.)의 데이터를 복제합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0404.png" alt="tmwr 0404" />
<h6 id="figure-4-4.-neighborhoods-in-ames-represented-using-a-convex-hull.">그림 4-4. 볼록 껍질(convex hull)을 사용하여 나타낸 에임스의 동네들.</h6>
</figure>

몇 가지 눈에 띄는 패턴을 볼 수 있습니다. 첫째, 에임스 중앙에는 데이터 포인트가 비어 있습니다. 이는 주거용 주택이 없는 아이오와 주립대학교 캠퍼스에 해당합니다. 둘째, 인접한 동네가 많지만 지리적으로 고립된 동네도 있습니다. 예를 들어 [그림 4-5](#figure-4-5.-locations-of-homes-in-timberland.)에서 볼 수 있듯이 Timberland는 거의 모든 다른 동네와 떨어져 있습니다.

[그림 4-6](#figure-4-6.-locations-of-homes-in-meadow-village-and-mitchell.)은 에임스 남서쪽에 있는 Meadow Village 동네가 Mitchell 동네를 구성하는 부동산들의 바다 안에서 마치 부동산의 섬과 같은 모습임을 시각화합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0405.png" alt="tmwr 0405" />
<h6 id="figure-4-5.-locations-of-homes-in-timberland.">그림 4-5. Timberland의 주택 위치.</h6>
</figure>

<figure class="width-80">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0406.png" alt="tmwr 0406" />
<h6 id="figure-4-6.-locations-of-homes-in-meadow-village-and-mitchell.">그림 4-6. Meadow Village와 Mitchell의 주택 위치.</h6>
</figure>

지도의 세부적인 검사는 동네 레이블이 완전히 신뢰할 수 있는 것은 아님을 보여주기도 합니다. 예를 들어 [그림 4-7](#figure-4-7.-locations-of-homes-in-somerset-and-northridge.)은 인접한 Somerset 동네의 주택들로 둘러싸여 있지만 Northridge에 있는 것으로 분류된 일부 부동산을 보여줍니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0407.png" alt="tmwr 0407" />
<h6 id="figure-4-7.-locations-of-homes-in-somerset-and-northridge.">그림 4-7. Somerset과 Northridge의 주택 위치.</h6>
</figure>

또한 Crawford에 있다고 표시된 10개의 고립된 주택이 있는데, [그림 4-8](#figure-4-8.-locations-of-homes-in-crawford.)에서 볼 수 있듯이 이 주택들은 해당 동네에 있는 다른 주택 대다수와 가깝지 않습니다.

<figure class="width-80">
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0408.png" alt="tmwr 0408" />
<h6 id="figure-4-8.-locations-of-homes-in-crawford.">그림 4-8. Crawford의 주택 위치.</h6>
</figure>

에임스 동쪽의 주요 도로에 인접한 "Iowa Department of Transportation (DOT) and Rail Road" 동네 역시 눈에 띕니다 ([그림 4-9](#figure-4-9.-homes-labeled-as-iowa-department-of-transportation-dot-and-rail-road.) 참조). 이 동네에는 여러 주택 클러스터뿐만 아니라 세로 방향의 이상치(longitudinal outliers)도 있습니다. 가장 동쪽에 있는 두 집은 다른 위치에서 고립되어 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2022] Tidy Modeling with R/assets/tmwr_0409.png" alt="tmwr 0409" />
<h6 id="figure-4-9.-homes-labeled-as-iowa-department-of-transportation-dot-and-rail-road.">그림 4-9. Iowa Department of Transportation (DOT) and Rail Road로 레이블이 지정된 주택들.</h6>
</figure>

[1장](ch01.xhtml#software-modeling)에서 설명했듯이 모델링을 시작하기 전에 탐색적 데이터 분석을 수행하는 것은 매우 중요합니다. 이러한 주택 데이터에는 데이터를 어떻게 처리하고 모델링해야 하는지에 대해 흥미로운 도전 과제를 제시하는 특징들이 있습니다. [17장](ch17.xhtml#categorical)과 같은 이후 장에서 이 중 많은 부분을 설명합니다. 이 탐색 단계에서 살펴볼 수 있는 몇 가지 기본적인 질문은 다음과 같습니다:

- 개별 예측 변수의 분포에 이상하거나 눈에 띄는 것이 있는가? 치우침(skewness)이 심하거나 병적인 분포(pathological distributions)가 있는가?

- 예측 변수 간에 높은 상관관계(correlations)가 있는가? 예를 들어 주택 크기와 관련된 여러 예측 변수가 있습니다. 일부는 중복(redundant)되는가?

- 예측 변수와 결과 변수 간에 연관성(associations)이 있는가?

이러한 질문 중 상당수는 다가오는 예제에서 이 데이터가 사용됨에 따라 다시 검토될 것입니다.

# 이 장의 요약 (Chapter Summary)

이 장에서는 Ames 주택 데이터 세트를 소개하고 그 특징 중 일부를 조사했습니다. 이 데이터 세트는 이후 장에서 tidymodels 구문을 보여주기 위해 사용될 것입니다. 이와 같은 탐색적 데이터 분석은 모든 모델링 프로젝트의 필수적인 구성 요소입니다. EDA는 더 나은 모델링 실습에 기여하는 정보를 밝혀냅니다.

이후 장으로 가져갈 Ames 데이터 세트를 준비하기 위한 중요한 코드는 다음과 같습니다:

```
library(tidymodels)
data(ames)
ames <- ames %>% mutate(Sale_Price = log10(Sale_Price))
```
