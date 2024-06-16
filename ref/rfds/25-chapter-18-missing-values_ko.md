# 18장. 결측값(Missing Values)

# 소개

이 책의 앞부분에서 결측값의 기본 사항에 대해 이미 배웠습니다. <a href="ch01.html#chp-data-visualize" data-type="xref">1장</a>에서 플롯을 만들 때 경고를 유발하는 것으로 처음 보았고, <a href="ch03.html#sec-summarize" data-type="xref">"summarize()"</a>에서는 요약 통계 계산을 방해하는 것으로 보았으며, <a href="ch12.html#sec-na-comparison" data-type="xref">"결측값(Missing Values)"</a>에서는 그것들의 전염성(infectious nature)과 존재 여부를 확인하는 방법에 대해 배웠습니다. 이제 더 깊이 있게 다루어 자세한 내용을 배울 수 있도록 돌아오겠습니다.

`NA`로 기록된 결측값 작업을 위한 몇 가지 일반적인 도구를 논의하는 것으로 시작하겠습니다. 그런 다음 단순히 데이터에 없는 값인 암시적 결측값(implicitly missing values)의 개념을 탐색하고, 이를 명시적으로 만드는 데 사용할 수 있는 도구들을 보여드리겠습니다. 데이터에 나타나지 않는 요인(factor) 수준으로 인해 발생하는 빈 그룹(empty groups)에 대한 관련된 논의로 마무리하겠습니다.

## 사전 준비

결측 데이터를 다루는 함수들은 주로 tidyverse의 핵심 멤버인 dplyr과 tidyr에서 제공됩니다.

```
library(tidyverse)
```

# 명시적 결측값(Explicit Missing Values)

먼저, 명시적 결측값, 즉 `NA`로 보이는 셀을 생성하거나 제거하는 데 편리한 몇 가지 도구를 탐색해 봅시다.

## 마지막 관측치 이월(Last Observation Carried Forward)

결측값이 흔히 사용되는 용도 중 하나는 데이터 입력의 편의성입니다. 데이터를 수동으로 입력할 때, 결측값은 종종 이전 행의 값이 반복됨(또는 이월됨)을 나타냅니다.

```
treatment <- tribble(
  ~person,           ~treatment, ~response,
  "Derrick Whitmore", 1,         7,
  NA,                 2,         10,
  NA,                 3,         NA,
  "Katherine Burke",  1,         4
)
```

이러한 결측값을 <a href="https://tidyr.tidyverse.org/reference/fill.html" class="orm:hideurl"><code>tidyr::fill()</code></a>을 사용하여 채울 수 있습니다. 이것은 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>처럼 작동하여, 열 집합을 취합니다.

```
treatment |>
  fill(everything())
#> # A tibble: 4 × 3
#>   person           treatment response
#>   <chr>                <dbl>    <dbl>
#> 1 Derrick Whitmore         1        7
#> 2 Derrick Whitmore         2       10
#> 3 Derrick Whitmore         3       10
#> 4 Katherine Burke          1        4
```

이러한 처리를 흔히 "마지막 관측치 이월(last observation carried forward)", 줄여서 *locf*라고 부릅니다. 더 특이한 방식으로 생성된 결측값을 채우기 위해 `.direction` 인수를 사용할 수 있습니다.

## 고정값(Fixed Values)

때로는 결측값이 알려진 고정된 값, 흔하게는 0을 나타냅니다. <a href="https://dplyr.tidyverse.org/reference/coalesce.html" class="orm:hideurl"><code>dplyr::coalesce()</code></a>를 사용하여 이를 대체할 수 있습니다.

```
x <- c(1, 4, 5, 7, NA)
coalesce(x, 0)
#> [1] 1 4 5 7 0
```

때로는 특정 구체적인 값이 실제로 결측값을 나타내는 정반대의 문제에 부딪힐 것입니다. 이는 결측값을 적절히 표현할 방법이 없는 구형 소프트웨어에서 생성된 데이터에서 흔히 발생하며, 그 대신 99나 -999 같은 특별한 값을 사용해야 합니다.

가능하다면, 예를 들어 <a href="https://readr.tidyverse.org/reference/read_delim.html" class="orm:hideurl"><code>readr::read_csv()</code></a>의 `na` 인수를 사용(`read_csv(path, na = "99")`)하여 데이터를 읽어 들일 때 이 문제를 처리하세요. 나중에 문제를 발견했거나 데이터 소스에서 읽을 때 이를 처리할 방법을 제공하지 않는 경우, <a href="https://dplyr.tidyverse.org/reference/na_if.html" class="orm:hideurl"><code>dplyr::na_if()</code></a>를 사용할 수 있습니다.

```
x <- c(1, 4, 5, 7, -99)
na_if(x, -99)
#> [1]  1  4  5  7 NA
```

## NaN

계속하기 전에 가끔 마주치게 될 특별한 유형의 결측값이 하나 있습니다. `NaN`("난"으로 발음됨), 즉 숫자가 아님(not a number)을 의미합니다. 일반적으로 `NA`와 똑같이 동작하기 때문에 이것에 대해 아는 것은 그다지 중요하지 않습니다.

```
x <- c(NA, NaN)
x * 10
#> [1]  NA NaN
x == 1
#> [1] NA NA
is.na(x)
#> [1] TRUE TRUE
```

드물게 `NA`와 `NaN`을 구별해야 하는 경우 `is.nan(x)`를 사용할 수 있습니다.

불확정된 결과를 갖는 수학적 연산을 수행할 때 일반적으로 `NaN`을 접하게 됩니다.

```m
0 / 0
#> [1] NaN
0 * Inf
#> [1] NaN
Inf - Inf
#> [1] NaN
sqrt(-1)
#> Warning in sqrt(-1): NaNs produced
#> [1] NaN
```

# 암시적 결측값(Implicit Missing Values)

지금까지 우리는 _명시적으로(explicitly)_ 누락된 결측값, 즉 데이터에서 `NA`를 볼 수 있는 값에 대해 이야기했습니다. 하지만 전체 데이터 행이 데이터에 단순히 없는 경우, 결측값은 _암시적으로(implicitly)_ 누락될 수도 있습니다. 각 분기별로 특정 주식의 가격을 기록하는 간단한 데이터 세트로 그 차이를 설명해 보겠습니다.

```
stocks <- tibble(
  year  = c(2020, 2020, 2020, 2020, 2021, 2021, 2021),
  qtr   = c(   1,    2,    3,    4,    2,    3,    4),
  price = c(1.88, 0.59, 0.35,   NA, 0.92, 0.17, 2.66)
)
```

이 데이터 세트에는 두 개의 누락된 관측치가 있습니다.

- 2020년 4분기의 `price`는 값이 `NA`이기 때문에 명시적으로 누락되었습니다.
- 2021년 1분기의 `price`는 데이터 세트에 단순히 나타나지 않기 때문에 암시적으로 누락되었습니다.

이 차이에 대해 생각하는 한 가지 방법은 다음과 같은 선문답(Zen-like koan)을 이용하는 것입니다.

> 명시적 결측값은 부재(absence)의 존재(presence)이다.\
>
> 암시적 결측값은 존재(presence)의 부재(absence)이다.

때로는 작업할 물리적인 무언가를 갖기 위해 암시적 결측을 명시적으로 만들고 싶을 때가 있습니다. 다른 경우에는 데이터의 구조에 의해 명시적 결측이 강제되며, 여러분은 그것들을 없애고 싶어 합니다. 다음 섹션들에서는 암시적 결측과 명시적 결측 사이를 이동하기 위한 몇 가지 도구에 대해 논의합니다.

## 피벗(Pivoting)

여러분은 이미 암시적 결측을 명시적으로, 혹은 그 반대로 만들 수 있는 한 가지 도구인 피벗팅을 보았습니다. 행과 새 열의 모든 조합에는 어떤 값이 있어야 하기 때문에, 데이터를 넓게(wider) 만들면 암시적 결측값이 명시적으로 만들어질 수 있습니다. 예를 들어, `quarter`를 열에 배치하기 위해 `stocks`를 피벗하면 두 결측값이 모두 명시적이 됩니다.

```
stocks |>
  pivot_wider(
    names_from = qtr,
    values_from = price
  )
#> # A tibble: 2 × 5
#>    year   `1`   `2`   `3`   `4`
#>   <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1  2020  1.88  0.59  0.35 NA
#> 2  2021 NA     0.92  0.17  2.66
```

기본적으로 데이터를 길게(longer) 만들면 명시적 결측값이 보존되지만, 데이터가 타이디(tidy)하지 않아서 존재하는 구조적인 결측값인 경우 `values_drop_na = TRUE`를 설정하여 이를 삭제(암시적으로 만듦)할 수 있습니다. 더 자세한 내용은 <a href="ch05.html#sec-tidy-data" data-type="xref">"타이디 데이터(Tidy Data)"</a>의 예제를 참조하세요.

## 완성(Complete)

<a href="https://tidyr.tidyverse.org/reference/complete.html" class="orm:hideurl"><code>tidyr::complete()</code></a>는 존재해야 하는 행의 조합을 정의하는 변수 집합을 제공하여 명시적 결측값을 생성할 수 있게 해줍니다. 예를 들어, `stocks` 데이터에는 `year`와 `qtr`의 모든 조합이 존재해야 한다는 것을 알고 있습니다.

```
stocks |>
  complete(year, qtr)
#> # A tibble: 8 × 3
#>    year   qtr price
#>   <dbl> <dbl> <dbl>
#> 1  2020     1  1.88
#> 2  2020     2  0.59
#> 3  2020     3  0.35
#> 4  2020     4 NA
#> 5  2021     1 NA
#> 6  2021     2  0.92
#> # … with 2 more rows
```

일반적으로 여러분은 기존 변수의 이름으로 <a href="https://tidyr.tidyverse.org/reference/complete.html" class="orm:hideurl"><code>complete()</code></a>를 호출하여 누락된 조합을 채웁니다. 그러나 때로는 개별 변수 자체가 불완전할 수 있으므로, 대신 자체 데이터를 제공할 수 있습니다. 예를 들어, `stocks` 데이터 세트가 2019년부터 2021년까지 이어져야 한다는 것을 안다면, `year`에 해당 값을 명시적으로 제공할 수 있습니다.

```
stocks |>
  complete(year = 2019:2021, qtr)
#> # A tibble: 12 × 3
#>    year   qtr price
#>   <dbl> <dbl> <dbl>
#> 1  2019     1 NA
#> 2  2019     2 NA
#> 3  2019     3 NA
#> 4  2019     4 NA
#> 5  2020     1  1.88
#> 6  2020     2  0.59
#> # … with 6 more rows
```

변수의 범위는 맞지만 모든 값이 존재하지 않는 경우, `full_seq(x, 1)`을 사용하여 `min(x)`부터 `max(x)`까지 간격이 1인 모든 값을 생성할 수 있습니다.

경우에 따라서는 변수들의 단순한 조합으로 전체 관측치 집합을 생성할 수 없습니다. 이 경우 <a href="https://tidyr.tidyverse.org/reference/complete.html" class="orm:hideurl"><code>complete()</code></a>가 해주는 작업을 수동으로 수행할 수 있습니다. 존재해야 하는 모든 행이 포함된 데이터 프레임을 생성(필요한 모든 기술의 조합을 사용하여)한 다음, 이를 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>dplyr::full_join()</code></a>을 사용해 원본 데이터 세트와 결합하는 것입니다.

## 조인(Joins)

여기서 암시적으로 누락된 관측치를 드러내는 또 다른 중요한 방법인 조인(joins)이 나옵니다. <a href="ch19.html#chp-joins" data-type="xref">19장</a>에서 조인에 대해 더 자세히 배울 것이지만, 한 데이터 세트의 값 누락을 다른 데이터 세트와 비교할 때만 종종 알 수 있기 때문에 여기서 간단히 언급하고자 합니다.

`dplyr::anti_join(x, y)`는 `y`와 일치하지 않는 `x`의 행만 선택하기 때문에 여기에서 유용한 도구입니다. 예를 들어, 두 번의 <a href="https://dplyr.tidyverse.org/reference/filter-joins.html" class="orm:hideurl"><code>anti_join()</code></a>을 사용하여 `flights`에 언급된 4개의 공항과 722대의 비행기에 대한 정보가 누락되었음을 드러낼 수 있습니다.

```
library(nycflights13)

flights |>
  distinct(faa = dest) |>
  anti_join(airports)
#> Joining with `by = join_by(faa)`
#> # A tibble: 4 × 1
#>   faa
#>   <chr>
#> 1 BQN
#> 2 SJU
#> 3 STT
#> 4 PSE

flights |>
  distinct(tailnum) |>
  anti_join(planes)
#> Joining with `by = join_by(tailnum)`
#> # A tibble: 722 × 1
#>   tailnum
#>   <chr>
#> 1 N3ALAA
#> 2 N3DUAA
#> 3 N542MQ
#> 4 N730MQ
#> 5 N9EAMQ
#> 6 N532UA
#> # … with 716 more rows
```

## 연습문제

1. 항공사(carrier)와 `planes`에서 누락된 것으로 보이는 행들 사이에 어떤 관계를 찾을 수 있나요?

# 요인(Factors)과 빈 그룹(Empty Groups)

마지막 누락 유형은 요인(factors)으로 작업할 때 발생할 수 있는, 어떤 관측치도 포함하지 않는 빈 그룹(empty group)입니다. 예를 들어, 사람들에 대한 일부 건강 정보가 포함된 데이터 세트가 있다고 상상해 보세요.

```
health <- tibble(
  name   = c("Ikaia", "Oletta", "Leriah", "Dashay", "Tresaun"),
  smoker = factor(c("no", "no", "no", "no", "no"), levels = c("yes", "no")),
  age    = c(34, 88, 75, 47, 56),
)
```

그리고 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>dplyr::count()</code></a>를 사용하여 흡연자의 수를 세고 싶다고 가정해 봅시다.

```
health |> count(smoker)
#> # A tibble: 1 × 2
#>   smoker     n
#>   <fct>  <int>
#> 1 no         5
```

이 데이터 세트에는 비흡연자만 포함되어 있지만 흡연자가 존재한다는 것을 우리는 알고 있습니다. 흡연자(원문은 nonsmoker라고 했으나 문맥상 smoker의 그룹이 비어있음을 의미) 그룹은 비어 있습니다. `.drop = FALSE`를 사용하면 데이터에서 보이지 않는 그룹을 포함하여 모든 그룹을 유지하도록 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>에 요청할 수 있습니다.

```
health |> count(smoker, .drop = FALSE)
#> # A tibble: 2 × 2
#>   smoker     n
#>   <fct>  <int>
#> 1 yes        0
#> 2 no         5
```

동일한 원리가 ggplot2의 이산형 축(discrete axes)에도 적용되며, 값이 없는 수준(levels)은 삭제됩니다. 적절한 이산형 축에 `drop = FALSE`를 제공하여 강제로 표시하도록 할 수 있습니다.

```
ggplot(health, aes(x = smoker)) +
  geom_bar() +
  scale_x_discrete()

ggplot(health, aes(x = smoker)) +
  geom_bar() +
  scale_x_discrete(drop = FALSE)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_18in01.png" alt="x축에 단일 값 'no'가 있는 막대 차트. 이전 플롯과 동일한 막대 차트이지만 이제 x축에 'yes'와 'no'라는 두 개의 값이 있습니다. 'yes' 범주에 대한 막대는 없습니다." />
</figure>

동일한 문제가 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>dplyr::group_by()</code></a>에서도 더 일반적으로 발생합니다. 여기서도 모든 요인 수준을 보존하기 위해 `.drop = FALSE`를 사용할 수 있습니다.

```
health |>
  group_by(smoker, .drop = FALSE) |>
  summarize(
    n = n(),
    mean_age = mean(age),
    min_age = min(age),
    max_age = max(age),
    sd_age = sd(age)
  )
#> # A tibble: 2 × 6
#>   smoker     n mean_age min_age max_age sd_age
#>   <fct>  <int>    <dbl>   <dbl>   <dbl>  <dbl>
#> 1 yes        0      NaN     Inf    -Inf   NA
#> 2 no         5       60      34      88   21.6
```

빈 그룹을 요약할 때 요약 함수가 길이가 0인 벡터에 적용되기 때문에 여기서 몇 가지 흥미로운 결과를 얻습니다. 길이가 0인 빈 벡터와 각각의 길이가 1인 결측값들 사이에는 중요한 차이가 있습니다.

```
# 두 개의 결측값을 포함하는 벡터
x1 <- c(NA, NA)
length(x1)
#> [1] 2

# 아무것도 포함하지 않는 벡터
x2 <- numeric()
length(x2)
#> [1] 0
```

모든 요약 함수는 길이가 0인 벡터와 함께 작동하지만 처음 보기에는 놀라운 결과를 반환할 수 있습니다. 여기서 `mean(age)`가 `NaN`을 반환하는 것을 볼 수 있는데, 그 이유는 `mean(age)` = `sum(age)/length(age)`이고, 여기서는 0/0이기 때문입니다. <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>max()</code></a>와 <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>min()</code></a>은 빈 벡터에 대해 -Inf와 Inf를 반환합니다. 따라서 그 결과를 새로운 데이터의 비어 있지 않은 벡터와 결합하여 다시 계산하면 새로운 데이터의 최솟값 또는 최댓값을 얻게 됩니다.<sup><a href="ch18.html#idm44771285285168" id="idm44771285285168-marker" data-type="noteref">1</a></sup>

때로는 요약을 수행한 다음 <a href="https://tidyr.tidyverse.org/reference/complete.html" class="orm:hideurl"><code>complete()</code></a>를 사용하여 암시적 결측을 명시적으로 만드는 것이 더 간단한 접근법이 될 수 있습니다.

```
health |>
  group_by(smoker) |>
  summarize(
    n = n(),
    mean_age = mean(age),
    min_age = min(age),
    max_age = max(age),
    sd_age = sd(age)
  ) |>
  complete(smoker)
#> # A tibble: 2 × 6
#>   smoker     n mean_age min_age max_age sd_age
#>   <fct>  <int>    <dbl>   <dbl>   <dbl>  <dbl>
#> 1 yes       NA       NA      NA      NA   NA
#> 2 no         5       60      34      88   21.6
```

이 접근법의 주요 단점은 개수가 0이어야 한다는 것을 알고 있음에도 불구하고 개수에 대해 `NA`를 얻는다는 것입니다.

# 요약

결측값은 이상합니다! 때로는 명시적인 `NA`로 기록되지만, 다른 때에는 그것들이 없다는 사실만으로 알아차릴 수 있습니다. 이 장에서는 명시적 결측값으로 작업하기 위한 몇 가지 도구와 암시적 결측값을 찾아내기 위한 몇 가지 도구를 제공했으며, 암시적인 것이 어떻게 명시적이 될 수 있는지 그리고 그 반대의 경우에 대한 몇 가지 방법에 대해 논의했습니다.

다음 장에서는 이 책의 이 부분의 마지막 장인 조인(joins)을 다룹니다. 이것은 지금까지의 장들과는 약간의 변화가 있는데, 데이터 프레임 안에 무언가를 넣는 것이 아니라 데이터 프레임 전체를 다루는 도구에 대해 논의할 것이기 때문입니다.

<sup>[1](ch18.html#idm44771285285168-marker)</sup> 다시 말해, `min(c(x, y))`는 항상 `min(min(x), min(y))`와 같습니다.
