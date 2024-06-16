---
title: "시각화: EDA"
---

```{r}
#| echo: false
source("_common.R")
```

> 시각화(visualization)와 변환(transformation)을 사용해 데이터를 체계적으로 탐색하는 방법을 배웁니다. 통계학자들은 이 작업을 탐색적 데이터 분석(exploratory data analysis), 줄여서 EDA라고 부릅니다.

EDA는 반복적인 주기(iterative cycle)입니다.

1. 데이터에 관한 질문을 만듭니다.
2. 데이터를 시각화, 변환, 모델링해 답을 찾습니다.
3. 배운 내용으로 질문을 다듬고(refine) 새로운 질문을 만듭니다.

EDA는 엄격한 규칙을 따르는 형식적인 과정이 아닙니다. 무엇보다 EDA는 마음가짐(state of mind)입니다. 초기에는 떠오르는 아이디어를 자유롭게 조사해야 합니다. 일부는 성공하고 일부는 막다른 골목(dead ends)에 이릅니다. 탐색을 이어가면 결국 문서화하고 다른 사람과 공유할 생산적인 통찰력(productive insights) 몇 가지로 범위가 좁혀집니다(home in on).

주요 연구 질문(primary research questions)이 쟁반에 담겨 제공되더라도(handed to you on a platter) 분석에서는 늘 데이터 품질을 조사해야 합니다. 그래서 EDA는 모든 데이터 분석에서 중요합니다. 데이터 정리(Data cleaning)는 EDA의 한 가지 적용일 뿐입니다. 데이터가 기대치를 충족하는지 묻고 시각화, 변환, 모델링 같은 EDA 도구를 모두 활용(deploy)합니다.

```{r}
#| label: setup
#| message: false
library(tidyverse)
```

## 질문

> "일상적인 통계적 질문(routine statistical questions)은 없다. 의심스러운 통계적 일상(questionable statistical routines)만 있을 뿐이다." - 데이비드 콕스 경(Sir David Cox)

> "항상 정밀해질 수 있는 잘못된 질문에 대한 정확한 답보다는, 종종 모호하더라도 올바른 질문에 대한 근사적인 답이 훨씬 낫다." - 존 튜키(John Tukey)

EDA의 목표는 데이터를 더 깊이 이해하는 것입니다. 질문을 조사의 안내 도구로 삼으면 이 목표에 쉽게 다가갑니다. 질문은 데이터셋의 특정 부분에 집중하게 하고 어떤 그래프, 모델, 변환을 만들지 결정하도록 돕습니다.

EDA는 본질적으로 창조적인 과정(creative process)입니다. 다른 창조적 작업과 마찬가지로 양질의(quality) 질문을 얻으려면 많은 양의(quantity) 질문을 던져봐야 합니다. 분석을 시작할 때는 데이터셋에서 어떤 통찰력을 얻을지 모르기 때문에 핵심을 찌르는(revealing) 질문이 바로 나오기 어렵습니다. 하지만 새로운 질문은 데이터의 새로운 측면을 드러내고 발견 가능성을 높입니다. 발견한 내용을 바탕으로 후속 질문(follow up)을 이어가면 흥미로운 데이터 부분을 빠르게 파고들어(drill down) 생각할 거리를 던져주는(thought-provoking) 질문 모음을 개발합니다.

연구를 안내할 질문에 정해진 규칙은 없습니다. 다만 다음 두 유형은 데이터에서 무언가를 발견할 때 늘 유용합니다.

1. 내 변수 내에서 어떤 유형의 변동(variation)이 발생합니까?
2. 내 변수들 사이에서 어떤 유형의 공변동(covariation)이 발생합니까?

이 장의 나머지 부분에서는 이 두 질문을 살펴봅니다. 변동과 공변동을 설명하고 각 질문에 답하는 여러 방법을 알아보겠습니다.

## 변동

변동(Variation)은 측정할 때마다(measurement to measurement) 변수의 값이 달라지는 경향입니다. 실생활에서도 쉽게 볼 수 있습니다. 연속형(continuous) 변수를 두 번 측정하면 서로 다른 결과가 나옵니다. 빛의 속도처럼 일정한 수치(quantities)를 측정해도 마찬가지입니다.

각 측정에는 매번 달라지는 작은 오차(error)가 들어 있습니다. 다른 대상(다른 사람들의 눈 색깔)이나 다른 시간(다른 순간의 전자 에너지 수준)을 측정할 때도 변수는 달라집니다. 모든 변수에는 고유한 변동 패턴이 있습니다. 이 패턴은 같은 관측치를 반복 측정할 때와 여러 관측치 전체에서 변수가 어떻게 달라지는지를 알려줍니다. 이를 이해하는 좋은 방법은 변수 값의 분포(distribution)를 시각화하는 것입니다.

`diamonds` 데이터셋에 있는 약 54,000개 다이아몬드의 무게(`carat`) 분포를 시각화하며 탐색을 시작하겠습니다. `carat`은 숫자형(numerical) 변수이므로 히스토그램을 사용합니다.

```{r}
#| fig-alt: |
#|   다이아몬드의 캐럿 히스토그램으로, x축의 범위는 0에서 4.5이고 
#|   y축의 범위는 0에서 30000입니다. 분포는 오른쪽으로 꼬리가 길며(right skewed) 
#|   중심이 0인 구간(bin)에는 다이아몬드가 거의 없고 중심이 0.5인 구간에는 
#|   거의 30000개의 다이아몬드가 있으며 중심이 1인 구간에는 약 
#|   15000개의 다이아몬드가 있고 중심이 1.5인 구간에는 훨씬 적은 약 5000개의 
#|   다이아몬드가 있습니다. 이를 넘어서면 늘어지는 꼬리(trailing tail)가 있습니다.
ggplot(diamonds, aes(x = carat)) +
  geom_histogram(binwidth = 0.5)
```

이제 변동을 시각화했습니다. 플롯에서 무엇을 찾고 어떤 후속 질문을 해야 할까요? 그래프에서 얻는 유용한 정보와 정보 유형별 후속 질문을 아래에 정리했습니다. 좋은 후속 질문은 호기심(무엇을 더 알고 싶은가요?)과 회의주의(이것이 어떻게 오해를 불러일으킬까요?)에서 나옵니다.

### 일반적인 값

막대 차트와 히스토그램에서 높은 막대는 변수의 일반적인(common) 값을, 짧은 막대는 덜 일반적인 값을 나타냅니다. 막대가 없는 곳은 데이터에서 관찰되지 않은 값입니다. 이 정보를 유용한 질문으로 바꾸려면 예상 밖의 값을 찾으세요.

- 어떤 값이 일반적입니까? 왜 그럴까요?
- 어떤 값이 드문가요(rare)? 왜 그럴까요? 그것이 여러분의 기대치와 일치합니까?
- 비정상적인 패턴이 보이나요? 무엇으로 설명하겠습니까?

더 작은 다이아몬드에 대한 `carat`의 분포를 살펴보겠습니다.

```{r}
#| fig-alt: |
#|   다이아몬드의 캐럿 히스토그램으로, x축의 범위는 0에서 3이고 
#|   y축의 범위는 0에서 대략 2500입니다. 구간 너비(binwidth)가 상당히 좁아 
#|   (0.01) 매우 많은 수의 얇은 막대가 생깁니다. 분포는 
#|   오른쪽으로 꼬리가 길며 여러 정점(peaks)이 있고 이어서 높이가 감소하는 막대들이 
#|   나타나다가 다음 정점에서 급격히 증가합니다.
smaller <- diamonds |> 
  filter(carat < 3)

ggplot(smaller, aes(x = carat)) +
  geom_histogram(binwidth = 0.01)
```

이 히스토그램에서 몇 가지 흥미로운 질문이 나옵니다.

- 왜 정수 캐럿과 캐럿의 일반적인 분수(fractions)에 더 많은 다이아몬드가 몰려 있을까요?
- 왜 각 정점의 약간 왼쪽보다 각 정점의 약간 오른쪽에 더 많은 다이아몬드가 있을까요?

시각화에서 군집(clusters)이 드러난다면 데이터에 하위 그룹(subgroups)이 있을 수 있습니다. 하위 그룹을 이해하려면 다음처럼 질문하세요.

- 각 하위 그룹 내의 관측치는 어떻게 서로 비슷합니까?
- 별도의 군집에 있는 관측치는 서로 어떻게 다릅니까?
- 군집을 어떻게 설명하거나 묘사하겠습니까?
- 군집의 모양이 오해를 불러일으킬 수 있는 이유는 무엇입니까?

데이터로 답하는 질문도 있지만 도메인 전문 지식(domain expertise)이 필요한 질문도 있습니다. 또 많은 질문은 한 변수의 값이 다른 변수의 움직임을 설명하는지 확인하는 관계 탐색으로 이어집니다. 이 내용은 곧 다루겠습니다.

### 비정상적인 값

이상치(Outliers)는 패턴에 맞지 않는 듯한 비정상적인 관측치입니다. 데이터 입력 오류일 때도 있고 수집 과정에서 우연히 관찰된 극한의(extremes) 값일 때도 있습니다. 때로는 중요한 새 발견의 단서가 됩니다.

데이터가 많으면 히스토그램에서 이상치를 찾기 어렵습니다. 다이아몬드 데이터셋의 `y` 변수 분포를 예로 보겠습니다. 이상치의 유일한 흔적은 x축의 비정상적으로 넓은 한계(limits)입니다.

```{r}
#| fig-alt: |
#|   다이아몬드의 길이(lengths) 히스토그램. x축의 범위는 0에서 60이고 
#|   y축의 범위는 0에서 12000입니다. 약 5 부근에 정점이 있으며 
#|   데이터는 정점 주위에 완전히 군집해 있는 것처럼 보입니다.
ggplot(diamonds, aes(x = y)) + 
  geom_histogram(binwidth = 0.5)
```

일반적인 구간에는 관측치가 너무 많아서 드문 구간이 매우 짧게 보입니다(물론 0을 뚫어지게 쳐다보면 무언가를 발견할 수도 있겠지만요). 비정상적인 값은 `coord_cartesian()`으로 y축의 작은 값을 확대(zoom)하면 잘 보입니다.

```{r}
#| fig-alt: |
#|   다이아몬드의 길이 히스토그램. x축의 범위는 0에서 60이고 
#|   y축의 범위는 0에서 50입니다. 약 5 부근에 정점이 있으며 데이터는 
#|   정점 주위에 완전히 군집해 있는 것처럼 보입니다. 이러한 데이터 외에 
#|   높이가 약 8인 0의 구간이 하나 있고 높이가 1인 30을 조금 넘는 구간이 하나 있으며 
#|   높이가 1인 60을 조금 밑도는 구간이 하나 있습니다.
ggplot(diamonds, aes(x = y)) + 
  geom_histogram(binwidth = 0.5) +
  coord_cartesian(ylim = c(0, 50))
```

`coord_cartesian()`에는 x축을 확대하는 `xlim()` 인수도 있습니다. ggplot2에는 조금 다르게 작동하는 `xlim()`과 `ylim()` 함수도 있으며 한계 밖의 데이터를 버립니다(throw away).

비정상적인 값은 세 가지(0, ~30, ~60)입니다. dplyr로 이 값들을 뽑아냅니다(pluck them out).

```{r}
#| include: false
old <- options(tibble.print_max = 10, tibble.print_min = 10)
```

```{r}
unusual <- diamonds |> 
  filter(y < 3 | y > 20) |> 
  select(price, x, y, z) |>
  arrange(y)
unusual
```

```{r}
#| include: false
options(old)
```

`y` 변수는 다이아몬드의 세 가지 차원 중 하나를 mm 단위로 측정합니다. 다이아몬드의 폭은 0mm일 수 없으므로 이 값들은 분명 잘못되었습니다. EDA 덕분에 0으로 코딩된 결측 데이터(missing data)를 발견했습니다. 단순히 `NA`만 검색했다면 찾지 못했을 것입니다. 앞으로 오해의 소지가 있는 계산을 막으려면 이 값을 `NA`로 다시 코딩해야 합니다. 32mm와 59mm의 측정치도 믿기 어렵다고(implausible) 의심할 만합니다. 그 다이아몬드는 길이가 1인치가 넘지만 수십만 달러가 들지 않습니다!

이상치를 포함한 분석과 제외한 분석을 각각 해보는 것이 좋습니다. 결과에 미치는 영향이 작고 이상치가 생긴 이유도 알 수 없다면 빼고 진행해도 합리적입니다. 하지만 결과에 상당한 영향을 미친다면 정당한 이유(justification) 없이 제외해서는 안 됩니다. 원인(데이터 입력 오류)을 파악하고 보고서(write-up)에 이상치를 제거했다고 밝혀야 합니다.

### 연습 문제

1. `diamonds` 데이터셋에서 `x`, `y`, `z` 변수 각각의 분포를 탐색하세요. 무엇을 알 수 있나요? 다이아몬드의 형태를 고려해 어떤 치수가 길이, 너비, 깊이인지 판단해 보세요.

2. `price`의 분포를 탐색하세요. 특이하거나 놀라운 점을 발견하셨나요? (힌트: `binwidth`를 신중하게 정하고 넓은 범위의 값을 시도해 보세요.)

3. 0.99 캐럿의 다이아몬드는 몇 개인가요? 1 캐럿은 몇 개인가요? 차이의 원인이 무엇이라고 생각하시나요?

4. 히스토그램을 확대할 때 `coord_cartesian()`과 `xlim()` 또는 `ylim()`을 비교하세요. `binwidth`를 설정하지 않으면 어떻게 되나요? 막대의 절반만 보이도록 확대(zoom)하면 어떻게 되나요?

## 비정상적인 값

데이터셋에서 비정상적인 값을 발견한 뒤에도 분석을 계속하는 방법은 두 가지입니다.

1. 이상한 값이 포함된 행 전체를 삭제합니다(Drop the entire row). 값 하나가 잘못됐다고 해당 관측치의 다른 값까지 모두 유효하지 않은 것은 아니므로 권장하지 않는 방법입니다. 게다가 저품질 데이터(low quality data)의 모든 변수에 이 방법을 적용하면 데이터가 하나도 남지 않을 수 있습니다!

```{r}
#| eval: false
diamonds2 <- diamonds |> 
  filter(between(y, 3, 20))
```

2. 비정상적인 값을 결측값으로 바꾸는 방법을 권장합니다. `mutate()`로 변수를 수정한 복사본으로 대체하면 간단합니다. `if_else()` 함수는 비정상적인 값을 `NA`로 바꿉니다. 결측값을 플롯할 위치가 명확하지 않으므로 ggplot2는 이를 제외하고 제거되었다는 경고를 표시합니다.

```{r}
diamonds2 <- diamonds |> 
  mutate(y = if_else(y < 3 | y > 20, NA, y))
```

```{r}
#| dev: "png"
#| fig-alt: |
#|   다이아몬드의 너비(widths) 대 길이(lengths)의 산점도. 두 변수 사이에 
#|   강한 선형(linear) 연관성이 있습니다. 다이아몬드 중 하나를 제외한 모든 
#|   다이아몬드의 길이는 3보다 큽니다. 하나의 이상치는 길이가 0이고 너비가 
#|   약 6.5입니다. 
ggplot(diamonds2, aes(x = x, y = y)) + 
  geom_point()
```

이 경고를 억제하려면 `na.rm = TRUE`를 설정합니다.

```{r}
#| eval: false
ggplot(diamonds2, aes(x = x, y = y)) + 
  geom_point(na.rm = TRUE)
```

때로는 결측값이 있는 관측치와 기록된 값(recorded values)이 있는 관측치가 어떻게 다른지 궁금할 때가 있습니다. 예를 들어 `nycflights13::flights`[^eda-1]에서 `dep_time`의 결측값은 항공편이 취소되었음을 나타냅니다.

취소된 항공편과 취소되지 않은 항공편의 예정 출발 시간(scheduled departure times)을 비교해 보겠습니다. `is.na()`로 `dep_time`의 누락 여부를 나타내는 새 변수를 만들면 됩니다.

[^eda-1]: 함수(또는 데이터셋)의 출처를 명시해야 할 때는 `package::function()` 또는 `package::dataset`이라는 특수한 형태를 사용한다는 것을 기억하세요.

```{r}
#| fig-alt: |
#|   항공편의 예정 출발 시간 빈도 다각형. 두 선은 
#|   취소된 항공편과 취소되지 않은 항공편을 나타냅니다. x축의 범위는 
#|   0에서 25분이고 y축의 범위는 0에서 10000입니다. 취소되지 않은 항공편의 수가 
#|   취소된 항공편보다 훨씬 많습니다.
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

하지만 취소되지 않은 항공편이 취소된 항공편보다 훨씬 많아서 이 플롯은 그다지 좋지 않습니다. 다음 섹션에서는 비교를 개선하는 몇 가지 방법을 살펴봅니다.

### 연습 문제 (Exercises)

1.  히스토그램에서 결측값은 어떻게 되나요? 막대 차트에서 결측값은 어떻게 되나요? 히스토그램과 막대 차트에서 결측값이 처리되는 방식에 차이가 있는 이유는 무엇인가요?

2.  `mean()`과 `sum()`에서 `na.rm = TRUE`는 무슨 역할을 하나요?

3.  항공편이 취소되었는지 여부에 따라 색상이 지정된 `scheduled_dep_time`의 빈도 플롯을 다시 만드세요. 또한 `cancelled` 변수로 패싯(facet)을 만드세요. 패싯 함수의 `scales` 변수에 여러 가지 값을 실험하여 취소되지 않은 항공편이 취소된 항공편보다 많은 경우의 효과를 완화해(mitigate) 보세요.

## 공변동

변동(variation)이 한 변수 안에서 값이 달라지는 양상을 묘사한다면 공변동(covariation)은 둘 이상의 변수가 함께 달라지는 양상을 묘사합니다. 공변동은 여러 변수의 값이 서로 연관되어 함께 변하는 경향입니다. 변수 간 관계를 시각화하면 이 경향이 드러납니다.

### 범주형 변수와 숫자형 변수

예를 들어 `geom_freqpoly()`를 사용하여 다이아몬드의 가격이 다이아몬드의 품질(`cut`으로 측정됨)에 따라 어떻게 달라지는지 탐색해 보겠습니다.

```{r}
#| fig-alt: |
#|   다이아몬드 가격의 빈도 다각형으로, 캐럿의 각 컷(Fair, 
#|   Good, Very Good, Premium, Ideal)이 서로 다른 색상의 
#|   선으로 표현됩니다. x축의 범위는 0에서 30000이고 y축의 범위는 0에서 
#|   5000입니다. 선들이 많이 겹쳐 있어 다이아몬드 가격의
#|   빈도 분포가 비슷합니다. Ideal 다이아몬드가 약 1500에서
#|   높은 정점을 이룹니다.
ggplot(diamonds, aes(x = price)) + 
  geom_freqpoly(aes(color = cut), binwidth = 500, linewidth = 0.75)
```

`cut`은 데이터에서 순서형 팩터 변수(ordered factor variable)로 정의되어 있어 ggplot2는 순서가 있는 색상 척도(ordered color scale)를 사용합니다.

전체 개수가 결정하는 높이가 `cut`마다 너무 달라 분포 모양의 차이를 보기 어렵습니다. 따라서 `geom_freqpoly()`의 기본 모양은 여기서 그다지 유용하지 않습니다.

비교하기 쉽도록 y축 표시값을 바꾸겠습니다. 개수 대신 밀도(density)를 표시합니다. 밀도는 각 빈도 다각형 아래의 면적이 1이 되도록 표준화한(standardized) 개수입니다.

```{r}
#| fig-alt: |
#|   다이아몬드 가격의 밀도 빈도 다각형으로, 캐럿의 각 컷(Fair, 
#|   Good, Very Good, Premium, Ideal)이 서로 다른 색상의 
#|   선으로 표현됩니다. x축의 범위는 0에서 20000입니다. 선들이 많이 겹쳐 
#|   있어 다이아몬드 가격의 밀도 분포가 비슷합니다.
#|   Fair 다이아몬드를 제외한 모든 다이아몬드가 약 1500의 가격에서
#|   높은 정점을 이루며 Fair 다이아몬드의 평균이 다른 다이아몬드보다 높습니다.
ggplot(diamonds, aes(x = price, y = after_stat(density))) + 
  geom_freqpoly(aes(color = cut), binwidth = 500, linewidth = 0.75)
```

`density`는 `diamonds` 데이터셋의 변수가 아니므로 `y`에 매핑하기 전에 계산해야 합니다. 이때 `after_stat()` 함수를 사용합니다.

이 플롯에는 다소 놀라운 점이 있습니다. fair 다이아몬드(최저 품질)의 평균 가격이 높아 보입니다! 다만 빈도 다각형은 해석하기가 조금 어렵고 이 플롯에는 많은 정보가 겹쳐 있습니다.

이 관계를 시각적으로 더 단순한 나란히 놓인(side-by-side) 상자 그림(boxplots)으로 탐색해 보겠습니다.

```{r}
#| fig-alt: |
#|   컷별 다이아몬드 가격의 나란히 놓인 상자 그림. 가격의 
#|   분포는 각 컷(Fair, Good, Very Good, Premium, 
#|   Ideal) 모두 오른쪽으로 꼬리가 깁니다. 중앙값(medians)은 서로 가까우며 Ideal
#|   다이아몬드의 중앙값이 낮고 Fair의 중앙값이 높습니다.
ggplot(diamonds, aes(x = cut, y = price)) +
  geom_boxplot()
```

분포 정보는 훨씬 적지만 상자 그림이 더 간결(compact)해서 비교하기 쉽습니다(한 플롯에 더 많이 넣을 수도 있습니다). 이 플롯은 품질이 좋은 다이아몬드가 일반적으로 더 저렴하다는 직관에 반하는(counter-intuitive) 발견을 뒷받침합니다! 연습 문제에서는 그 이유를 알아내는 과제(challenge)를 다룹니다.

`cut`은 순서형 팩터(ordered factor)입니다. fair는 good보다 나쁘고 good은 very good보다 나쁜 식으로 순서가 이어집니다. 많은 범주형 변수에는 이런 내재적(intrinsic) 순서가 없습니다. 더 유용한(informative) 디스플레이를 만들려면 순서를 재정렬(reorder)해야 합니다. 한 가지 방법은 `fct_reorder()`입니다. 여기서는 `mpg` 데이터셋의 `class` 변수를 예로 간단히 살펴보겠습니다. 고속도로 주행 거리가 class에 따라 어떻게 다른지 알아봅시다.

```{r}
#| fig-alt: |
#|   class별 자동차 고속도로 주행 거리의 나란히 놓인 상자 그림. class는 
#|   x축에 있습니다(2seaters, compact, midsize, minivan, pickup, subcompact, 
#|   suv).
ggplot(mpg, aes(x = class, y = hwy)) +
  geom_boxplot()
```

추세(trend)가 잘 보이도록 `hwy`의 중앙값을 기준으로 `class`를 재정렬합니다.

```{r}
#| fig-alt: |
#|   class별 자동차 고속도로 주행 거리의 나란히 놓인 상자 그림. class는 
#|   x축에 있으며 고속도로 주행 거리 중앙값이 증가하는 순서(pickup, 
#|   suv, minivan, 2seater, subcompact, compact, midsize)로 정렬됩니다.
ggplot(mpg, aes(x = fct_reorder(class, hwy, median), y = hwy)) +
  geom_boxplot()
```

변수 이름이 길면 `geom_boxplot()`을 90도 뒤집는(flip) 편이 낫습니다. `x`와 `y` 심미성 매핑을 교환(exchanging)하면 됩니다.

```{r}
#| fig-alt: |
#|   class별 자동차 고속도로 주행 거리의 나란히 놓인 상자 그림. class는 
#|   y축에 있으며 고속도로 주행 거리 중앙값이 증가하는 순서로 정렬됩니다.
ggplot(mpg, aes(x = hwy, y = fct_reorder(class, hwy, median))) +
  geom_boxplot()
```

#### 연습 문제

1. 지금까지 배운 내용으로 취소된 항공편과 취소되지 않은 항공편의 출발 시간 시각화를 개선하세요.

2. EDA를 기반으로 다이아몬드 데이터셋에서 다이아몬드 가격을 예측하는 데 중요해 보이는 변수는 무엇입니까? 해당 변수는 cut과 어떤 상관관계가 있습니까? 이 두 관계의 조합으로 인해 왜 낮은 품질의 다이아몬드가 더 비싼 결과를 낳게 될까요?

3. `x`와 `y` 변수를 교환하는 대신 세로형 상자 그림에 새로운 레이어로 `coord_flip()`을 추가하여 가로형 상자 그림을 만드세요. 변수를 교환하는 것과 어떻게 비교되나요?

4. 상자 그림은 훨씬 작은 데이터셋이 일반적이던 시대(era)에 개발되어 엄청나게 많은 수의 "외부 값(outlying values)"을 표시하는 경향이 있습니다. 이 문제를 해결하는 한 가지 접근법이 문자 값 플롯(letter value plot)입니다. lvplot 패키지를 설치하고 `geom_lv()`로 price 대 cut의 분포를 표시해 보세요. 무엇을 알 수 있나요? 플롯을 어떻게 해석하나요?

5. `geom_violin()`을 사용하여 `diamonds` 데이터셋의 범주형 변수 대 다이아몬드 가격의 시각화를 생성한 다음 패싯 처리된 `geom_histogram()`, 색상이 지정된 `geom_freqpoly()`, 색상이 지정된 `geom_density()`를 생성하세요. 네 가지 플롯을 비교하고 대조하세요. 범주형 변수의 수준(levels)에 따라 숫자형 변수의 분포를 시각화하는 각 방법의 장단점은 무엇입니까?

6. 데이터셋이 작다면 오버플로팅(overplotting)을 방지하는 `geom_jitter()`가 연속형 변수와 범주형 변수의 관계를 파악하는 데 유용합니다. ggbeeswarm 패키지에는 `geom_jitter()`와 유사한 여러 방법이 있습니다. 목록을 작성하고 각각의 기능을 간략하게 설명하세요.

### 두 개의 범주형 변수

범주형 변수 사이의 공변동을 시각화하려면 각 수준 조합(combination of levels)의 관측치 수를 세어야 합니다. 내장 함수인 `geom_count()`를 이용하는 것이 한 가지 방법입니다.

```{r}
#| fig-alt: |
#|   다이아몬드의 color 대 cut 산점도. cut(Fair, Good, Very Good, Premium 
#|   Ideal) 및 color(D, E, F, G, G, I, J) 수준의 각 조합마다
#|   하나의 점이 있습니다. 점의 크기는 해당 조합에 대한 관측치 수를 
#|   나타냅니다. 범례는 이러한 크기가 1000에서 4000 
#|   사이임을 나타냅니다.
ggplot(diamonds, aes(x = cut, y = color)) +
  geom_count()
```

플롯에 있는 원의 크기는 각 값의 조합에서 관측치가 얼마나 많이 발생했는지를 나타냅니다. 특정 x 값과 y 값 사이의 강한 상관관계가 공변동으로 나타납니다.

변수 간의 관계를 탐색하는 또 다른 방법은 dplyr로 개수를 계산하는 것입니다.

```{r}
diamonds |> 
  count(color, cut)
```

계산한 결과는 `geom_tile()`과 fill 심미성으로 시각화합니다.

```{r}
#| fig-alt: |
#|   다이아몬드의 cut 대 color 타일 플롯(tile plot). 각 타일은 
#|   cut/color 조합을 나타내며 각 타일의 관측치 수에 따라 타일 
#|   색상이 지정됩니다. 다른 컷보다 Ideal 다이아몬드가 더 많으며 
#|   높은 수는 color G인 Ideal 다이아몬드입니다. Fair 다이아몬드와
#|   color I인 다이아몬드는 빈도가 낮습니다.
diamonds |> 
  count(color, cut) |>  
  ggplot(aes(x = color, y = cut)) +
  geom_tile(aes(fill = n))
```

범주형 변수에 순서가 없다면(unordered) seriation 패키지로 행과 열을 동시에 재정렬해 흥미로운 패턴을 더 분명하게 드러낼 수 있습니다. 플롯이 크다면 heatmaply 패키지로 대화형 플롯(interactive plots)을 만들어 볼 수 있습니다.

#### 연습 문제

1. color 내의 cut 분포 또는 cut 내의 color 분포를 더 명확하게 표시하려면 위의 카운트 데이터셋의 스케일을 어떻게 조정(rescale)해야 할까요?

2. color를 `x` 심미성에 매핑하고 `cut`을 `fill` 심미성에 매핑한 분할 막대 차트를 사용하면 어떤 다른 데이터 통찰을 얻을 수 있나요? 각 세그먼트에 속하는 개수를 계산하세요.

3. `geom_tile()`을 dplyr과 함께 사용하여 평균 항공편 출발 지연이 목적지 및 연중 월(month of year)에 따라 어떻게 달라지는지 탐색하세요. 이 플롯을 읽기 어렵게 만드는 이유는 무엇입니까? 어떻게 개선하겠습니까?

### 두 개의 숫자형 변수

두 숫자형 변수 사이의 공변동을 시각화하는 좋은 방법은 이미 살펴봤습니다. `geom_point()`로 산점도를 그리면 점의 패턴에서 공변동이 드러납니다.
예를 들어 다이아몬드의 캐럿 크기와 가격 사이에는 양의 관계(positive relationship)가 있습니다. 캐럿이 높은 다이아몬드일수록 가격도 높습니다. 이 관계는 지수적(exponential)입니다.

```{r}
#| dev: "png"
#| fig-alt: |
#|   price 대 carat 산점도. 관계는 양의 관계이고 
#|   다소 강하며 지수적(exponential)입니다.
ggplot(smaller, aes(x = carat, y = price)) +
  geom_point()
```

데이터셋이 커지면 점들이 오버플로팅(overplot)되어 짙은 검은색 영역에 쌓입니다. 그러면 2차원 공간에서 데이터 밀도의 차이나 추세를 파악하기 어려워 산점도의 유용성이 떨어집니다. 앞에서 살펴본 해결책 중 하나는 `alpha` 심미성으로 투명도를 추가하는 것입니다.

```{r}
#| dev: "png"
#| fig-alt: |
#|   price 대 carat 산점도. 관계는 양의 관계이고 다소 강하며 
#|   지수적입니다. 점들이 투명하여 다른 영역보다 
#|   점의 수가 더 많은 군집이 드러납니다. 뚜렷한 군집은
#|   1, 1.5, 2캐럿 다이아몬드에 대한 군집입니다.
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_point(alpha = 1 / 100)
```

하지만 데이터셋이 매우 크면 투명도만으로 해결하기 어렵습니다. 이때는 구간(bin)이 대안입니다. 앞에서는 `geom_histogram()`과 `geom_freqpoly()`로 1차원 구간을 나눴습니다. 이제 `geom_bin2d()`와 `geom_hex()`로 2차원 구간을 나누는 방법을 살펴보겠습니다.

`geom_bin2d()`와 `geom_hex()`는 좌표 평면을 2차원 구간(2d bins)으로 나누고 채우기 색상(fill color)으로 각 구간의 점 개수를 표시합니다. `geom_bin2d()`는 직사각형 구간을, `geom_hex()`는 육각형 구간(hexagonal bins)을 만듭니다. `geom_hex()`를 쓰려면 hexbin 패키지를 설치해야 합니다.

```{r}
#| layout-ncol: 2
#| fig-width: 3
#| fig-alt: |
#|   플롯 1: price 대 carat의 구간 밀도 플롯(binned density plot). 플롯 2: price 
#|   대 carat의 육각형 구간 플롯. 두 플롯 모두 다이아몬드의 
#|   낮은 캐럿과 낮은 가격에 다이아몬드가 밀집해 있습니다.
ggplot(smaller, aes(x = carat, y = price)) +
  geom_bin2d()

# install.packages("hexbin")
ggplot(smaller, aes(x = carat, y = price)) +
  geom_hex()
```

또 다른 방법은 연속형 변수 하나를 구간으로 나눠 범주형 변수처럼 다루는 것입니다. 그러면 앞에서 배운 범주형 변수와 연속형 변수의 조합을 시각화하는 기법을 적용하면 됩니다. 예를 들어 `carat`을 구간으로 나누고 각 그룹의 상자 그림을 그립니다.

```{r}
#| fig-alt: |
#|   carat별 price의 나란히 놓인 상자 그림. 각 상자 그림은 
#|   무게가 0.1캐럿씩 차이 나는 다이아몬드를 나타냅니다. 상자 그림은 캐럿이
#|   증가함에 따라 중앙값 가격도 높아집니다. 또한 1.5캐럿
#|   이하의 다이아몬드는 오른쪽으로 꼬리가 긴(right skewed) 가격 분포를 가지고 1.5에서 2캐럿은
#|   대략 대칭적인 가격 분포를 가지며 그 이상 나가는 다이아몬드는
#|   왼쪽으로 꼬리가 긴(left skewed) 분포를 가집니다. 더 저렴하고 작은 다이아몬드는
#|   위쪽에 이상치가 있고 더 비싸고 큰 다이아몬드는 아래쪽에 이상치가 있습니다.
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_boxplot(aes(group = cut_width(carat, 0.1)))
```

위에서 사용한 `cut_width(x, width)`는 `x`를 `width` 너비의 구간으로 나눕니다.
상자 그림은 기본적으로 관측치 수와 관계없이 (이상치의 수를 제외하고) 크기가 거의 같습니다. 그래서 각 상자 그림이 서로 다른 수의 점을 요약한다는 사실을 알아보기 어렵습니다.

이 차이는 `varwidth = TRUE`를 사용해 상자 그림의 너비를 점의 수에 비례하도록(proportional) 만들면 드러납니다.

#### 연습 문제

1. 상자 그림으로 조건부 분포(conditional distribution)를 요약하는 대신 빈도 다각형(frequency polygon)을 사용해도 됩니다. `cut_width()` 대 `cut_number()`를 사용할 때 무엇을 고려해야 하나요? 그것이 `carat`과 `price`의 2차원 분포 시각화에 어떤 영향을 미치나요?

2. `price`로 분할된 `carat`의 분포를 시각화하세요.

3. 매우 큰 다이아몬드의 가격 분포는 작은 다이아몬드와 어떻게 비교되나요? 예상한 대로인가요, 아니면 놀라운가요?

4. 배운 기술 중 두 가지를 결합하여 cut, carat, price의 결합된 분포를 시각화하세요.

5. 2차원 플롯은 1차원 플롯에서 보이지 않는 이상치를 드러냅니다. 예를 들어 다음 플롯의 일부 점은 `x`와 `y`를 따로 검사하면 정상으로 보이지만 두 값의 조합이 비정상적이어서 이상치가 됩니다. 이 경우 구간 플롯(binned plot)보다 산점도가 더 나은 디스플레이인 이유는 무엇입니까?

```{r}
#| eval: false
diamonds |> 
  filter(x >= 4) |> 
  ggplot(aes(x = x, y = y)) +
  geom_point() +
  coord_cartesian(xlim = c(4, 11), ylim = c(4, 11))
```

6. `cut_width()`로 너비가 같은 상자를 만드는 대신 `cut_number()`로 점의 수가 거의 같은 상자를 만들 수도 있습니다. 이 접근법의 장점과 단점은 무엇입니까?

```{r}
#| eval: false
ggplot(smaller, aes(x = carat, y = price)) + 
  geom_boxplot(aes(group = cut_number(carat, 20)))
```

## 패턴과 모델

두 변수 사이에 체계적인(systematic) 관계가 있다면 데이터에서 패턴으로 나타납니다. 패턴을 발견하면 다음과 같이 자문해 보세요.

1. 이 패턴이 우연(coincidence) (즉, 무작위 확률(random chance))에 의한 것일 수 있을까?

2. 패턴이 암시하는(implied by) 관계를 어떻게 설명할까?

3. 패턴이 암시하는 관계는 얼마나 강할까?

4. 관계에 영향을 미칠 수 있는 다른 변수들은 무엇이 있을까?

5. 데이터의 개별 하위 그룹(subgroups)을 살펴보면 관계가 바뀌는가?

데이터의 패턴은 관계를 파악할 단서(clues), 곧 공변동을 드러냅니다.
변동(variation)이 불확실성을 만드는 현상이라면 공변동(covariation)은 불확실성을 줄이는 현상입니다. 두 변수가 공변동하면 한 변수의 값으로 다른 변수의 값을 더 정확하게 예측합니다. 공변동이 인과 관계(causal relationship)(특수한 경우)에서 비롯됐다면 한 변수의 값으로 다른 변수의 값을 통제(control)할 수도 있습니다.

모델은 데이터에서 패턴을 추출하는 도구입니다. diamonds 데이터를 예로 들어 보겠습니다. cut과 carat, carat과 price가 긴밀하게(tightly) 관련되어 있어 cut과 price의 관계를 이해하기가 어렵습니다. 모델로 price와 carat 사이의 매우 강한 관계를 제거하고 남은 미묘한 차이(subtleties)를 탐색합니다. 다음 코드는 `carat`으로 `price`를 예측하는 모델을 적합(fits)한 뒤 잔차(residuals)(예측값과 실제값의 차이)를 계산합니다. 잔차에는 carat의 효과를 제거한 다이아몬드 가격이 나타납니다. 여기서는 `price`와 `carat`의 원시 값(raw values) 대신 먼저 로그(log) 변환을 하고 변환한 값에 모델을 적합한다는 점에 유의하세요. 마지막으로 잔차를 지수화(exponentiate)해 원래 가격 척도로 되돌립니다.

```{r}
#| message: false
#| dev: "png"
#| fig-alt: |
#|   다이아몬드의 잔차(residuals) 대 캐럿 산점도. x축의 범위는 0에서
#|   5이고 y축의 범위는 0에서 거의 4입니다. 많은 데이터가 낮은 캐럿 및 잔차
#|   값 주위에 모여 있습니다. 캐럿이 증가함에 따라 잔차가 감소하는 것을
#|   보여주는 명확한 곡선 패턴(curved pattern)이 있습니다.
library(tidymodels)

diamonds <- diamonds |>
  mutate(
    log_price = log(price),
    log_carat = log(carat)
  )

diamonds_fit <- linear_reg() |>
  fit(log_price ~ log_carat, data = diamonds)

diamonds_aug <- augment(diamonds_fit, new_data = diamonds) |>
  mutate(.resid = exp(.resid))

ggplot(diamonds_aug, aes(x = carat, y = .resid)) + 
  geom_point()
```

carat과 price 사이의 강한 관계를 제거하면 cut과 price 사이에서 예상했던 관계가 드러납니다. 크기가 같다면 품질이 더 좋은 다이아몬드가 더 비쌉니다.

```{r}
#| fig-alt: |
#|   cut별 잔차의 나란히 놓인 상자 그림. x축에는 다양한 
#|   컷(Fair에서 Ideal까지)이 표시되고 y축의 범위는 0에서 거의 5까지입니다.
#|   중앙값(medians)은 대략 0.75에서 1.25 사이로 매우 유사합니다.
#|   잔차의 각 분포는 오른쪽으로 꼬리가 길며 높은 쪽에 많은
#|   이상치가 있습니다.
ggplot(diamonds_aug, aes(x = cut, y = .resid)) + 
  geom_boxplot()
```

이 책에서는 모델링을 다루지 않습니다. 데이터 랭글링(data wrangling)과 프로그래밍 도구를 익힌 뒤라야 모델이 무엇이고 어떻게 작동하는지 이해하기 쉽기 때문입니다.

<!-- HUMANIZE-SUMMARY
원본 글자수: 21,184자
윤문본 글자수: 19,649자
변경률: 10.5% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 9 → 0
- A-7 가지다 직역: 2 → 0
- A-10 가능 표현 남발: 29 → 0
- A-11 목적절 남발: 8 → 0
- A-15 본문 추상 주어·만능 동사: 13 → 0
- C-11 연결어미 뒤 쉼표: 10 → 0

자체검증: 6/6 통과
1. 고유명사·수치·날짜·인용·내용 앵커 보존: 통과
2. 변경률 30% 이하: 통과
3. 장르 유지: 통과
4. register 유지: 통과
5. 잔존 S1 패턴 0건: 통과
6. 새 비유·수사·상투구 없음: 통과

등급: A
사유: S1·선별 S2 패턴이 남지 않았고 본문 변경률이 10~25% 범위임.

주요 변경 하이라이트:
- "엄격한 규칙 세트가 있는 형식적인 프로세스" → "엄격한 규칙을 따르는 형식적인 과정"
- "변수 내부의 행동 / 변수 사이의 행동" → "한 변수 안에서 값이 달라지는 양상 / 둘 이상의 변수가 함께 달라지는 양상"
- "투명도를 추가하기 위해 alpha 심미성을 사용" → "alpha 심미성으로 투명도를 추가"
- "모델을 사용하는 것이 가능합니다" → "모델로 관계를 제거하고 남은 차이를 탐색"
-->
