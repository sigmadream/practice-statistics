# 제10장. 탐색적 데이터 분석 (Exploratory Data Analysis)

# 소개 (Introduction)

이 장에서는 데이터를 체계적으로 탐색하기 위해 시각화(visualization)와 변환(transformation)을 사용하는 방법을 보여줍니다. 통계학자들은 이 작업을 *탐색적 데이터 분석(exploratory data analysis)*, 줄여서 EDA라고 부릅니다. EDA는 반복적인 주기입니다. 여러분은:

1. 데이터에 대한 질문을 생성합니다.
2. 데이터를 시각화, 변환, 모델링하여 답을 찾습니다.
3. 배운 내용을 사용하여 질문을 구체화하거나 새로운 질문을 생성합니다.

EDA는 엄격한 규칙이 있는 공식적인 프로세스가 아닙니다. 무엇보다도 EDA는 마음가짐(state of mind)입니다. EDA의 초기 단계에서는 떠오르는 모든 아이디어를 자유롭게 조사해야 합니다. 이러한 아이디어 중 일부는 성공할 것이고, 일부는 막다른 골목에 다다를 것입니다. 탐색이 계속됨에 따라, 여러분은 결국 글로 작성하고 다른 사람들과 소통할 몇 가지 특히 생산적인 통찰력(insights)에 집중하게 될 것입니다.

주요 연구 질문이 여러분에게 완벽하게 주어지더라도, 항상 데이터의 품질을 조사해야 하기 때문에 EDA는 모든 데이터 분석에서 중요한 부분입니다. 데이터 정리(data cleaning)는 EDA의 한 가지 적용 사례일 뿐입니다: 여러분은 데이터가 기대치를 충족하는지 여부에 대해 질문합니다. 데이터 정리를 하려면 시각화, 변환, 모델링과 같은 EDA의 모든 도구를 배포해야 합니다.

## 사전 준비 (Prerequisites)

이 장에서는 dplyr과 ggplot2에 대해 배운 내용을 결합하여 대화형으로 질문을 던지고, 데이터로 답을 찾고, 다시 새로운 질문을 던질 것입니다.

```
library(tidyverse)
```

# 질문 (Questions)

> “일상적인 통계 질문은 없으며, 의심스러운 통계적 일상만 있을 뿐입니다(There are no routine statistical questions, only questionable statistical routines).” — 데이비드 콕스 경 (Sir David Cox)

> “틀린 질문에 대한 정확한 답보다, 종종 모호하더라도 올바른 질문에 대한 대략적인 답이 훨씬 낫습니다. 틀린 질문에 대한 정확한 답은 항상 정밀하게 만들어질 수 있기 때문입니다.” — 존 튜키 (John Tukey)

EDA 동안 여러분의 목표는 데이터에 대한 이해를 높이는 것입니다. 이를 수행하는 가장 쉬운 방법은 질문을 조사를 이끄는 도구로 사용하는 것입니다. 질문을 할 때, 그 질문은 데이터세트의 특정 부분에 주의를 집중시키고 어떤 그래프, 모델 또는 변환을 만들어야 할지 결정하는 데 도움을 줍니다.

EDA는 근본적으로 창조적인 과정입니다. 대부분의 창조적인 과정과 마찬가지로, *품질 좋은(quality)* 질문을 던지는 핵심은 *많은(quantity)* 질문을 생성하는 것입니다. 분석을 시작할 때 데이터세트에서 어떤 통찰력을 얻을 수 있는지 알 수 없기 때문에 처음부터 의미 있는 질문을 던지기는 어렵습니다. 반면에, 새로운 질문을 던질 때마다 데이터의 새로운 측면에 노출되고 발견을 할 가능성이 높아집니다. 발견한 내용을 바탕으로 각 질문에 새로운 질문을 덧붙인다면, 데이터의 가장 흥미로운 부분에 빠르게 접근하고 생각할 거리를 던져주는 질문 세트를 개발할 수 있습니다.

연구를 안내하기 위해 어떤 질문을 해야 하는지에 대한 규칙은 없습니다. 하지만 데이터 내에서 발견을 하는 데 항상 유용한 두 가지 유형의 질문이 있습니다. 이 질문들을 대략적으로 표현하면 다음과 같습니다:

1. 내 변수(variables) 내에서는 어떤 유형의 변동(variation)이 발생하는가?
2. 내 변수들 사이에는 어떤 유형의 공변동(covariation)이 발생하는가?

이 장의 나머지 부분에서는 이 두 가지 질문을 살펴볼 것입니다. 변동과 공변동이 무엇인지 설명하고, 각 질문에 답하는 여러 가지 방법을 보여드리겠습니다.

# 변동 (Variation)

*변동(Variation)* 은 변수의 값이 측정마다 변하는 경향을 말합니다. 실생활에서 변동은 쉽게 볼 수 있습니다. 연속형 변수를 두 번 측정하면 두 개의 다른 결과를 얻게 됩니다. 이는 빛의 속도와 같이 일정한 양을 측정하더라도 마찬가지입니다. 여러분의 각 측정에는 측정마다 변하는 소량의 오차가 포함됩니다. 서로 다른 대상(예: 사람들의 눈 색깔)을 가로질러 측정하거나 다른 시간(예: 다른 순간에 있는 전자의 에너지 수준)에 측정하는 경우에도 변수는 변할 수 있습니다. 모든 변수는 고유한 변동 패턴을 가지고 있으며, 이는 동일한 관측치에 대한 측정 간에, 그리고 관측치를 가로질러 어떻게 변하는지에 대한 흥미로운 정보를 나타낼 수 있습니다. 이 패턴을 이해하는 가장 좋은 방법은 변수 값의 분포(distribution)를 시각화하는 것이며, 이는 <a href="ch01.html#chp-data-visualize" data-type="xref">제1장</a>에서 배웠습니다.

`diamonds` 데이터세트에 있는 약 54,000개 다이아몬드의 무게(`carat`) 분포를 시각화하는 것으로 탐색을 시작하겠습니다. `carat`은 수치형 변수(numerical variable)이므로 히스토그램을 사용할 수 있습니다:

```
ggplot(diamonds, aes(x = carat)) +
  geom_histogram(binwidth = 0.5)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in01.png" alt="A histogram of carats of diamonds, with the x-axis ranging from 0 to 4.5 and the y-axis ranging from 0 to 30000. The distribution is right skewed with very few diamonds in the bin centered at 0, almost 30000 diamonds in the bin centered at 0.5, approximately 15000 diamonds in the bin centered at 1, and much fewer, approximately 5000 diamonds in the bin centered at 1.5. Beyond this, there&#39;s a trailing tail." />
</figure>

이제 변동을 시각화할 수 있으므로 그래프에서 무엇을 찾아야 할까요? 그리고 어떤 유형의 후속 질문을 던져야 할까요? 그래프에서 찾을 수 있는 가장 유용한 정보 유형과 각 정보 유형에 대한 몇 가지 후속 질문 목록을 다음 섹션에 정리해 두었습니다. 좋은 후속 질문을 던지는 핵심은 호기심(무엇에 대해 더 알고 싶은가?)과 회의주의(이것이 어떻게 오해를 불러일으킬 수 있는가?)에 의존하는 것입니다.

## 전형적인 값 (Typical Values)

막대 차트(bar charts)와 히스토그램 모두에서 높은 막대는 변수의 흔한 값을 보여주고 짧은 막대는 덜 흔한 값을 보여줍니다. 막대가 없는 곳은 데이터에서 관찰되지 않은 값을 나타냅니다. 이 정보를 유용한 질문으로 바꾸려면 예상치 못한 것을 찾으십시오:

- 어떤 값이 가장 흔한가요? 그 이유는 무엇인가요?
- 어떤 값이 드문가요? 그 이유는 무엇인가요? 그것이 여러분의 예상과 일치하나요?
- 특이한 패턴을 볼 수 있나요? 그것을 어떻게 설명할 수 있을까요?

작은 다이아몬드의 `carat` 분포를 살펴보겠습니다:

```
smaller <- diamonds |> 
  filter(carat < 3)

ggplot(smaller, aes(x = carat)) +
  geom_histogram(binwidth = 0.01)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in02.png" alt="A histogram of carats of diamonds, with the x-axis ranging from 0 to 3 and the y-axis ranging from 0 to roughly 2500. The binwidth is quite narrow (0.01), resulting in a very large number of skinny bars. The distribution is right skewed, with many peaks followed by bars in decreasing heights, until a sharp increase at the next peak." />
</figure>

이 히스토그램은 몇 가지 흥미로운 질문을 제안합니다:

- 왜 정수 캐럿과 흔한 분수 캐럿의 다이아몬드가 더 많은가요?
- 왜 각 봉우리의 바로 왼쪽보다 각 봉우리의 바로 오른쪽에 있는 다이아몬드가 더 많은가요?

시각화는 데이터에 하위 그룹이 존재함을 시사하는 군집(clusters)을 드러낼 수도 있습니다. 하위 그룹을 이해하려면 다음과 같이 질문하십시오:

- 각 하위 그룹 내의 관측치들은 서로 어떻게 비슷한가요?
- 서로 다른 군집에 있는 관측치들은 서로 어떻게 다른가요?
- 군집을 어떻게 설명하거나 묘사할 수 있나요?
- 군집의 모습이 오해를 불러일으킬 수 있는 이유는 무엇인가요?

이러한 질문 중 일부는 데이터로 대답할 수 있지만 일부는 데이터에 대한 도메인 전문 지식(domain expertise)을 요구합니다. 많은 질문들이 변수 *사이의* 관계를 탐색하도록 유도할 것입니다. 예를 들어 한 변수의 값이 다른 변수의 동작을 설명할 수 있는지 확인하는 것입니다. 곧 이 부분에 대해 다룰 것입니다.

## 특이값 (Unusual Values)

이상치(Outliers)는 이례적인 관측치, 다시 말해 패턴에 맞지 않는 것처럼 보이는 데이터 포인트입니다. 때로는 이상치가 데이터 입력 오류이기도 하고, 때로는 데이터 수집 시 우연히 관찰된 극단값일 뿐이며, 때로는 중요한 새로운 발견을 암시하기도 합니다. 데이터가 많을 때 이상치는 히스토그램에서 보기 어려울 때가 있습니다. 예를 들어, `diamonds` 데이터세트에서 `y` 변수의 분포를 생각해 보세요. 이상치의 유일한 증거는 x축의 유난히 넓은 한계선입니다.

```
ggplot(diamonds, aes(x = y)) + 
  geom_histogram(binwidth = 0.5)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in03.png" alt="A histogram of lengths of diamonds. The x-axis ranges from 0 to 60 and the y-axis ranges from 0 to 12000. There is a peak around 5, and the data appear to be completely clustered around the peak." />
</figure>

가장 흔한 구간(bins)에 관측치가 너무 많아서 드문 구간은 매우 짧아져 보기가 어렵습니다 (0을 뚫어지게 쳐다보면 무언가 발견할지도 모르지만요). 특이값을 쉽게 보려면 <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>을 사용하여 y축을 작은 값으로 확대해야 합니다:

```
ggplot(diamonds, aes(x = y)) + 
  geom_histogram(binwidth = 0.5) +
  coord_cartesian(ylim = c(0, 50))
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in04.png" alt="A histogram of lengths of diamonds. The x-axis ranges from 0 to 60 and the y-axis ranges from 0 to 50. There is a peak around 5, and the data appear to be completely clustered around the peak. Other than those data, there is one bin at 0 with a height of about 8, one a little over 30 with a height of 1 and another one a little below 60 with a height of 1." />
</figure>

<a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>에는 x축을 확대해야 할 때를 대비한 <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>xlim()</code></a> 인자(argument)도 있습니다. ggplot2에는 조금 다르게 작동하는 <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>xlim()</code></a> 및 <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>ylim()</code></a> 함수도 있습니다. 이들은 한계(limits) 밖의 데이터를 버립니다.

이를 통해 0, ~30, ~60이라는 세 가지 특이값이 있음을 알 수 있습니다. 우리는 dplyr로 그것들을 뽑아냅니다:

```
unusual <- diamonds |> 
  filter(y < 3 | y > 20) |> 
  select(price, x, y, z) |>
  arrange(y)
unusual
#> # A tibble: 9 × 4
#>   price     x     y     z
#>   <int> <dbl> <dbl> <dbl>
#> 1  5139  0      0    0   
#> 2  6381  0      0    0   
#> 3 12800  0      0    0   
#> 4 15686  0      0    0   
#> 5 18034  0      0    0   
#> 6  2130  0      0    0   
#> 7  2130  0      0    0   
#> 8  2075  5.15  31.8  5.12
#> 9 12210  8.09  58.9  8.06
```

`y` 변수는 이러한 다이아몬드의 세 가지 차원(dimensions) 중 하나를 mm 단위로 측정합니다. 다이아몬드는 너비가 0mm일 수 없으므로 이 값은 틀림없이 오류입니다. EDA를 수행함으로써 결측값(missing data)이 0으로 코딩된 것을 발견했는데, 단순히 `NA`를 검색했다면 절대 찾지 못했을 것입니다. 앞으로 우리는 잘못된 계산을 방지하기 위해 이 값들을 `NA`로 다시 코딩하기로 결정할 수 있습니다. 우리는 또한 32mm와 59mm의 측정이 그럴듯하지 않다고 의심할 수 있습니다: 그 다이아몬드들은 1인치가 넘는 길이지만 수십만 달러가 넘지는 않습니다!

이상치를 포함해서 그리고 이상치를 제외하고 분석을 반복하는 것이 좋은 관행입니다. 만약 이상치가 결과에 미치는 영향이 최소화되고 왜 그런 값이 있는지 알아낼 수 없다면, 생략하고 넘어가는 것이 타당합니다. 하지만 이상치가 결과에 실질적인 영향을 미친다면, 정당한 이유 없이 제거해서는 안 됩니다. 원인(예: 데이터 입력 오류)을 파악하고 결과물(write-up)에서 이를 제거했음을 밝혀야 합니다.

## 연습문제 (Exercises)

1. `diamonds`의 `x`, `y`, `z` 각 변수의 분포를 탐색해 보세요. 무엇을 알게 되나요? 다이아몬드를 떠올려보고 어떤 차원이 길이, 너비, 깊이인지 어떻게 결정할 수 있을지 생각해 보세요.
2. `price`의 분포를 탐색해 보세요. 특이하거나 놀라운 점을 발견했나요? (힌트: `binwidth`에 대해 주의 깊게 생각하고 다양한 값을 시도해 보세요.)
3. 0.99 캐럿인 다이아몬드는 몇 개인가요? 1 캐럿인 다이아몬드는 몇 개인가요? 차이의 원인은 무엇이라고 생각하나요?
4. 히스토그램을 확대할 때 <a href="https://ggplot2.tidyverse.org/reference/coord_cartesian.html" class="orm:hideurl"><code>coord_cartesian()</code></a>과 <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>xlim()</code></a> 또는 <a href="https://ggplot2.tidyverse.org/reference/lims.html" class="orm:hideurl"><code>ylim()</code></a>을 비교하고 대조해 보세요. `binwidth`를 설정하지 않은 채로 두면 어떻게 되나요? 막대의 절반만 보이도록 확대하려고 하면 어떻게 되나요?

# 특이값 (Unusual Values)

데이터세트에서 특이값을 발견하고 분석의 나머지 부분으로 넘어가고 싶다면, 두 가지 선택지가 있습니다:

1. 이상한 값이 포함된 행 전체를 버립니다:

    ```
    diamonds2 <- diamonds |> 
      filter(between(y, 3, 20))
    ```

    하나의 유효하지 않은 값이 해당 관측치의 다른 모든 값들도 유효하지 않다는 것을 의미하지는 않기 때문에 이 옵션을 추천하지 않습니다. 게다가, 데이터의 품질이 낮다면 모든 변수에 이 접근 방식을 적용했을 때 남는 데이터가 없을 수도 있습니다!

2. 대신, 특이값을 결측값으로 대체하는 것을 추천합니다. 이를 수행하는 가장 쉬운 방법은 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>를 사용하여 변수를 수정된 복사본으로 바꾸는 것입니다. <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a> 함수를 사용하여 특이값을 `NA`로 바꿀 수 있습니다:

    ```
    diamonds2 <- diamonds |> 
      mutate(y = if_else(y < 3 | y > 20, NA, y))
    ```

결측값을 어디에 표시해야 할지 명확하지 않기 때문에 ggplot2는 그것들을 플롯에 포함하지 않지만, 제거되었다는 경고는 표시합니다:

```
ggplot(diamonds2, aes(x = x, y = y)) + 
  geom_point()
#> Warning: Removed 9 rows containing missing values (`geom_point()`).
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in05.png" alt="A scatterplot of widths vs. lengths of diamonds. There is a strong, linear association between the two variables. All but one of the diamonds has length greater than 3. The one outlier has a length of 0 and a width of about 6.5." />
</figure>

해당 경고를 억제하려면 `na.rm = TRUE`를 설정하세요:

```
ggplot(diamonds2, aes(x = x, y = y)) + 
  geom_point(na.rm = TRUE)
```

어떤 때는 결측값이 있는 관측치와 기록된 값이 있는 관측치를 무엇이 다르게 만드는지 이해하고 싶을 때가 있습니다. 예를 들어, <a href="https://rdrr.io/pkg/nycflights13/man/flights.html" class="orm:hideurl"><code>nycflights13::flights</code></a><sup><a href="ch10.html#idm44771307799104" id="idm44771307799104-marker" data-type="noteref">1</a></sup>에서 `dep_time` 변수의 결측값은 항공편이 취소되었음을 나타냅니다. 따라서 취소된 항공편과 취소되지 않은 항공편의 예정된 출발 시간을 비교하고 싶을 수 있습니다. <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>를 사용하여 `dep_time`이 누락되었는지 확인하는 새로운 변수를 만들어 이를 수행할 수 있습니다.

```
nycflights13::flights |> 
  mutate(
    cancelled = is.na(dep_time),
    sched_hour = sched_dep_time %/% 100,
    sched_min = sched_dep_time %% 100,
    sched_dep_time = sched_hour + (sched_min / 60)
  ) |> 
  ggplot(aes(x = sched_dep_time)) + 
  geom_freqpoly(aes(color = cancelled), binwidth = 1/4)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in06.png" alt="A frequency polygon of scheduled departure times of flights. Two lines represent flights that are cancelled and not cancelled. The x-axis ranges from 0 to 25 minutes and the y-axis ranges from 0 to 10000. The number of flights not cancelled are much higher than those cancelled." />
</figure>

하지만 취소된 항공편보다 취소되지 않은 항공편이 훨씬 많기 때문에 이 플롯은 훌륭하지 않습니다. 다음 섹션에서는 이러한 비교를 개선하기 위한 몇 가지 기술을 탐색할 것입니다.

## 연습문제 (Exercises)

1. 히스토그램에서 결측값은 어떻게 되나요? 막대 차트에서 결측값은 어떻게 되나요? 히스토그램과 막대 차트에서 결측값이 다르게 처리되는 이유는 무엇인가요?
2. <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>과 <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>에서 `na.rm = TRUE`는 어떤 역할을 하나요?
3. 항공편이 취소되었는지 여부로 색상이 지정된 `scheduled_dep_time`의 도수 다각형(frequency plot)을 다시 만들어 보세요. 또한 `cancelled` 변수로 패싯(facet)을 나누어 보세요. 취소된 항공편보다 취소되지 않은 항공편이 더 많은 효과를 완화하기 위해 패싯 분할 함수에서 `scales` 변수의 다른 값을 실험해 보세요.

# 공변동 (Covariation)

변동이 하나의 변수 *내의(within)* 행동을 설명한다면, 공변동은 변수들 *사이의(between)* 행동을 설명합니다. *공변동(Covariation)* 은 둘 이상의 변수 값이 연관된 방식으로 함께 변하는 경향입니다. 공변동을 발견하는 가장 좋은 방법은 두 개 이상의 변수 사이의 관계를 시각화하는 것입니다.

## 범주형 변수와 수치형 변수 (A Categorical and a Numerical Variable)

예를 들어, <a href="https://ggplot2.tidyverse.org/reference/geom_histogram.html" class="orm:hideurl"><code>geom_freqpoly()</code></a>를 사용하여 다이아몬드의 가격이 그 품질(`cut`으로 측정됨)에 따라 어떻게 달라지는지 탐색해 보겠습니다:

```
ggplot(diamonds, aes(x = price)) + 
  geom_freqpoly(aes(color = cut), binwidth = 500, linewidth = 0.75)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in07.png" alt="A frequency polygon of prices of diamonds where each cut of carat (Fair, Good, Very Good, Premium, and Ideal) is represented with a different color line. The x-axis ranges from 0 to 30000 and the y-axis ranges from 0 to 5000. The lines overlap a great deal, suggesting similar frequency distributions of prices of diamonds. One notable feature is that Ideal diamonds have the highest peak around 1500." />
</figure>

`cut`이 데이터에서 순서형 요인 변수(ordered factor variable)로 정의되어 있기 때문에 ggplot2가 `cut`에 대해 순서가 있는 색상 척도(ordered color scale)를 사용한다는 점에 유의하세요. 이에 대해서는 <a href="ch16.html#sec-ordered-factors" data-type="xref">“순서형 팩터(Ordered Factors)”</a>에서 자세히 배울 것입니다.

입력에 출력의 한 셀에 해당하는 여러 행이 있는 경우 어떻게 되는지 궁금할 수도 있습니다. 다음 예에는 `id A` 및 `measurement bp1`에 해당하는 두 개의 행이 있습니다:

`df` `<-` `tribble``(` `~``id``,` `~``measurement``,` `~``value``,` `"A"``,` `"bp1"``,` `100``,` `"A"``,` `"bp1"``,` `102``,` `"A"``,` `"bp2"``,` `120``,` `"B"``,` `"bp1"``,` `140``,` `"B"``,` `"bp2"``,` `115` `)`

이 데이터를 피벗하려고 하면 리스트 열(list-columns)이 포함된 출력을 얻게 됩니다. 이에 대해서는 <a href="ch23.html#chp-rectangling" data-type="xref">23장</a>에서 자세히 알아볼 것입니다:

`df` `|>` `pivot_wider``(` `names_from` `=` `measurement``,` `values_from` `=` `value` `)` `` #> Warning: Values from `value` are not uniquely identified; output will contain `` `#> list-cols.` `` #> • Use `values_fn = list` to suppress this warning. `` `` #> • Use `values_fn = {summary_fun}` to summarise duplicates. `` `#> • Use the following dplyr code to identify duplicates.` `#> {data} %>%` `#> dplyr::group_by(id, measurement) %>%` `#> dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%` `#> dplyr::filter(n > 1L)` `#> # A tibble: 2 × 3` `#> id bp1 bp2 ` `#> <chr> <list> <list> ` `#> 1 A <dbl [2]> <dbl [1]>` `#> 2 B <dbl [1]> <dbl [1]>`

아직 이런 종류의 데이터로 작업하는 방법을 모르기 때문에, 어디에 문제가 있는지 파악하기 위해 경고 메시지의 힌트를 따르고 싶을 것입니다:

`df` `|>` `group_by``(``id``,` `measurement``)` `|>` `summarize``(``n` `=` `n``(),` `.groups` `=` `"drop"``)` `|>` `filter``(``n` `>` `1``)` `#> # A tibble: 1 × 3` `#> id measurement n` `#> <chr> <chr> <int>` `#> 1 A bp1 2`

그런 다음 데이터에 무엇이 잘못되었는지 파악하고 근본적인 손상을 복구하거나 그룹화 및 요약 기술을 사용하여 행과 열 값의 각 조합이 단일 행만 갖도록 보장하는 것은 여러분의 몫입니다.

# 요약 (Summary)

이 장에서는 변수가 열에 있고 관측치가 행에 있는 데이터인 정돈된 데이터(tidy data)에 대해 배웠습니다. 정돈된 데이터는 대부분의 함수에서 이해되는 일관된 구조이기 때문에 tidyverse에서의 작업을 더 쉽게 만듭니다. 주요 과제는 전달받은 어떤 구조의 데이터든 정돈된 형식으로 변환하는 것입니다. 이를 위해 많은 정돈되지 않은 데이터셋을 정돈할 수 있게 해주는 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>에 대해 배웠습니다. 여기서 제시한 예제는 <a href="https://tidyr.tidyverse.org/articles/pivot.html" class="orm:hideurl"><code>vignette("pivot", package = "tidyr")</code></a>에서 선택한 것이므로, 이 장에서 도움을 받지 못한 문제에 직면한다면 다음에 시도해 볼 좋은 자료가 바로 그 비네트(vignette)입니다.

또 다른 과제는, 주어진 데이터셋에 대해 더 길거나 넓은 버전을 "정돈된(tidy)" 것이라고 명명하는 것이 불가능할 수 있다는 점입니다. 이는 정돈된 데이터가 각 열에 하나의 변수를 가진다고 말했지만 실제로 변수가 무엇인지는 정의하지 않았던(그리고 정의하기가 놀랍도록 어렵습니다) 우리의 정돈된 데이터 정의를 일부 반영하는 것입니다. 분석을 가장 쉽게 만들어주는 것이라면 무엇이든 변수라고 말하는 등 실용적으로 접근해도 전혀 문제가 없습니다. 따라서 어떤 계산을 어떻게 수행할지 알아내는 데 막혔다면 데이터 구성을 전환하는 것을 고려해 보세요. 필요에 따라 정돈을 해제하고, 변환하고, 다시 정돈하는 것을 두려워하지 마세요!

이 장을 즐겁게 읽었고 기본 이론에 대해 더 알고 싶다면, *Journal of Statistical Software*에 게시된 [“Tidy Data” 논문](https://oreil.ly/86uxw)에서 그 역사와 이론적 토대에 대해 자세히 알아볼 수 있습니다.

이제 상당한 양의 R 코드를 작성하고 있으므로 코드를 파일 및 디렉터리로 구성하는 방법에 대해 더 자세히 알아볼 때입니다. 다음 장에서는 스크립트와 프로젝트의 이점과 여러분의 삶을 편하게 만들어 줄 많은 도구에 대해 모두 알아볼 것입니다.

<sup>[1](ch05.html#idm44771326722336-marker)</sup> 2000년 어느 시점에 탑 100에 포함되었고, 차트에 등장한 후 최대 72주까지 추적된 노래라면 포함됩니다.

<sup>[2](ch05.html#idm44771328141216-marker)</sup> 이 아이디어에 대해서는 <a href="ch18.html#chp-missing-values" data-type="xref">18장</a>에서 다시 다루겠습니다.
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in19.png" alt="A scatter plot of residuals vs. carat of diamonds. The x-axis ranges from 0 to 5, the y-axis ranges from 0 to almost 4. Much of the data are clustered around low values of carat and residuals. There is a clear, curved pattern showing decrease in residuals as carat increases." />
</figure>

carat과 price 사이의 강한 관계를 제거하고 나면, cut과 price 사이의 관계에서 예상했던 것을 볼 수 있습니다: 크기에 비해 품질이 좋은 다이아몬드가 더 비쌉니다.

```
ggplot(diamonds_aug, aes(x = cut, y = .resid)) + 
  geom_boxplot()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_10in20.png" alt="Side-by-side box plots of residuals by cut. The x-axis displays the various cuts (Fair to Ideal), the y-axis ranges from 0 to almost 5. The medians are quite similar, between roughly 0.75 to 1.25. Each of the distributions of residuals is right skewed, with many outliers on the higher end." />
</figure>

이 책에서는 모델링(modeling)에 대해 논의하지 않습니다. 왜냐하면 모델이 무엇이고 어떻게 작동하는지 이해하는 것은 데이터 랭글링(data wrangling)과 프로그래밍을 위한 도구를 손에 쥐었을 때 가장 쉽기 때문입니다.

# 요약 (Summary)

이 장에서는 데이터 내의 변동(variation)을 이해하는 데 도움이 되는 다양한 도구를 배웠습니다. 한 번에 하나의 변수를 다루는 기술과 변수 쌍을 다루는 기술을 보았습니다. 데이터에 수십 또는 수백 개의 변수가 있다면 이것이 고통스러울 정도로 제한적으로 보일 수 있지만, 이러한 기술들은 다른 모든 기술이 구축되는 기초(foundation)입니다.

다음 장에서는 결과를 소통(communicate)하는 데 사용할 수 있는 도구에 집중할 것입니다.

<sup>[1](ch10.html#idm44771307799104-marker)</sup> 함수(또는 데이터세트)가 어디서 왔는지 명시해야 할 때, `package::function()` 또는 `package::dataset`이라는 특별한 형태를 사용한다는 점을 기억하세요.
