# 26장. 반복 (Iteration)

# 소개

이 장에서는 다른 객체에 대해 동일한 작업을 반복적으로 수행하는 반복(iteration)을 위한 도구를 배웁니다. R의 반복은 일반적으로 대부분 암묵적(implicit)이며 공짜로 얻어지기 때문에 다른 프로그래밍 언어와는 다소 다르게 보이는 경향이 있습니다. 예를 들어, R에서 숫자형 벡터 `x`를 두 배로 만들려면 그냥 `2 * x`를 작성하면 됩니다. 대부분의 다른 언어에서는 일종의 for 루프를 사용하여 `x`의 각 요소를 명시적으로 두 배로 만들어야 합니다.

이 책에서는 이미 여러 "것들(things)"에 대해 동일한 작업을 수행하는 작지만 강력한 도구들을 몇 가지 제공했습니다.

- <a href="https://ggplot2.tidyverse.org/reference/facet_wrap.html" class="orm:hideurl"><code>facet_wrap()</code></a> 및 <a href="https://ggplot2.tidyverse.org/reference/facet_grid.html" class="orm:hideurl"><code>facet_grid()</code></a>는 각 하위 집합(subset)에 대한 플롯을 그립니다.
- <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>는 각 하위 집합에 대한 요약 통계를 계산합니다.
- <a href="https://tidyr.tidyverse.org/reference/unnest_wider.html" class="orm:hideurl"><code>unnest_wider()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/unnest_longer.html" class="orm:hideurl"><code>unnest_longer()</code></a>는 리스트 열의 각 요소에 대해 새로운 행과 열을 생성합니다.

이제 다른 함수를 입력으로 취하는 함수를 중심으로 구축되었기 때문에 종종 _함수형 프로그래밍(functional programming)_ 도구라고 불리는 좀 더 일반적인 도구에 대해 배울 시간입니다. 함수형 프로그래밍을 배우면 추상적으로 빠지기 쉽지만, 이 장에서는 여러 열 수정, 여러 파일 읽기, 여러 객체 저장이라는 세 가지 일반적인 작업에 초점을 맞춰 구체적으로 설명하겠습니다.

## 사전 준비

이 장에서는 tidyverse의 두 핵심 멤버인 dplyr과 purrr에서 제공하는 도구에 중점을 둘 것입니다. dplyr은 전에 본 적이 있지만, [purrr](https://oreil.ly/f0HWP)는 처음일 것입니다. 이 장에서는 몇 가지 purrr 함수만 사용할 것이지만, 프로그래밍 기술을 향상시키면서 탐구해 보기 좋은 패키지입니다.

```
library(tidyverse)
```

# 다중 열 수정하기 (Modifying Multiple Columns)

이 단순한 티블(tibble)이 있고 관측치 수를 세고 모든 열의 중앙값을 계산하고 싶다고 가정해 보겠습니다.

```
df <- tibble(
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10)
)
```

복사 및 붙여넣기로 수행할 수 있습니다.

```
df |> summarize(
  n = n(),
  a = median(a),
  b = median(b),
  c = median(c),
  d = median(d),
)
#> # A tibble: 1 × 5
#>       n      a      b       c     d
#>   <int>  <dbl>  <dbl>   <dbl> <dbl>
#> 1    10 -0.246 -0.287 -0.0567 0.144
```

이것은 두 번 이상 복사하여 붙여넣지 말라는 경험 법칙을 위반하며, 수십 또는 수백 개의 열이 있는 경우 이 작업이 지루해질 것임을 상상할 수 있습니다. 대신 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>를 사용할 수 있습니다.

```
df |> summarize(
  n = n(),
  across(a:d, median),
)
#> # A tibble: 1 × 5
#>       n      a      b       c     d
#>   <int>  <dbl>  <dbl>   <dbl> <dbl>
#> 1    10 -0.246 -0.287 -0.0567 0.144
```

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>에는 특히 중요한 세 가지 인자가 있으며, 다음 섹션에서 자세히 설명하겠습니다. <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>를 사용할 때마다 처음 두 가지 인자를 사용할 것입니다. 첫 번째 인자 `.cols`는 반복할 열을 지정하고 두 번째 인자 `.fns`는 각 열로 수행할 작업을 지정합니다. 출력 열의 이름에 대한 추가 제어가 필요할 때 `.names` 인자를 사용할 수 있으며, 이는 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 함께 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>를 사용할 때 특히 중요합니다. 또한 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 함께 작동하는 중요한 두 가지 변형인 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_any()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_all()</code></a>에 대해서도 논의할 것입니다.

## .cols로 열 선택하기 (Selecting Columns with .cols)

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>의 첫 번째 인자인 `.cols`는 변환할 열을 선택합니다. 이는 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="ch03.html#sec-select" data-type="xref">“select()”</a>와 동일한 사양을 사용하므로 <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>starts_with()</code></a> 및 <a href="https://tidyselect.r-lib.org/reference/starts_with.html" class="orm:hideurl"><code>ends_with()</code></a>와 같은 함수를 사용하여 이름을 기반으로 열을 선택할 수 있습니다.

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>에 특히 유용한 두 가지 추가 선택 기술인 <a href="https://tidyselect.r-lib.org/reference/everything.html" class="orm:hideurl"><code>everything()</code></a> 및 <a href="https://tidyselect.r-lib.org/reference/where.html" class="orm:hideurl"><code>where()</code></a>가 있습니다. <a href="https://tidyselect.r-lib.org/reference/everything.html" class="orm:hideurl"><code>everything()</code></a>은 간단합니다. 모든(그룹화되지 않은) 열을 선택합니다.

```
df <- tibble(
  grp = sample(2, 10, replace = TRUE),
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10)
)

df |>
  group_by(grp) |>
  summarize(across(everything(), median))
#> # A tibble: 2 × 5
#>     grp       a       b     c     d
#>   <int>   <dbl>   <dbl> <dbl> <dbl>
#> 1     1 -0.0935 -0.0163 0.363 0.364
#> 2     2  0.312  -0.0576 0.208 0.565
```

그룹화 열(여기서는 `grp`)은 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>에 의해 자동으로 보존되므로 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>에 포함되지 않습니다.

<a href="https://tidyselect.r-lib.org/reference/where.html" class="orm:hideurl"><code>where()</code></a>를 사용하면 데이터 유형에 따라 열을 선택할 수 있습니다.

`where(is.numeric)`  
모든 숫자형 열을 선택합니다.

`where(is.character)`  
모든 문자열 열을 선택합니다.

`where(is.Date)`  
모든 날짜 열을 선택합니다.

`where(is.POSIXct)`  
모든 날짜-시간 열을 선택합니다.

`where(is.logical)`  
모든 논리형 열을 선택합니다.

다른 선택자와 마찬가지로 이것들을 부울 대수(Boolean algebra)와 결합할 수 있습니다. 예를 들어, `!where(is.numeric)`은 모든 비숫자형 열을 선택하고 `starts_with("a") & where(is.logical)`은 이름이 "a"로 시작하는 모든 논리형 열을 선택합니다.

## 단일 함수 호출하기 (Calling a Single Function)

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>의 두 번째 인자는 각 열이 어떻게 변환될지 정의합니다. 그림과 같이 간단한 경우에는 단일 기존 함수가 됩니다. 이것은 R의 아주 특별한 기능입니다. 한 함수(`median`, `mean`, `str_flatten`, …)를 다른 함수(`across`)에 전달하는 것입니다. 이것이 R을 함수형 프로그래밍 언어로 만드는 기능 중 하나입니다.

이 함수를 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>에 전달하여 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>가 이를 호출할 수 있도록 한다는 점을 유의하는 것이 중요합니다. 우리가 직접 호출하는 것이 아닙니다. 즉, 함수 이름 뒤에 절대 `()`가 오면 안 됩니다. 잊어버리면 오류가 발생합니다.

```
df |>
  group_by(grp) |>
  summarize(across(everything(), median()))
#> Error in `summarize()`:
#> ℹ In argument: `across(everything(), median())`.
#> Caused by error in `is.factor()`:
#> ! argument "x" is missing, with no default
```

이 오류는 입력 없이 함수를 호출했기 때문에 발생합니다. 예:

```
median()
#> Error in is.factor(x): argument "x" is missing, with no default
```

## 다중 함수 호출하기 (Calling Multiple Functions)

더 복잡한 경우 추가 인자를 제공하거나 여러 변환을 수행하고 싶을 수 있습니다. 간단한 예제로 이 문제를 설명해 보겠습니다. 데이터에 결측값이 있으면 어떻게 될까요? <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>은 이러한 결측값을 전파하여 최적이 아닌 결과를 제공합니다.

```
rnorm_na <- function(n, n_na, mean = 0, sd = 1) {
  sample(c(rnorm(n - n_na, mean = mean, sd = sd), rep(NA, n_na)))
}

df_miss <- tibble(
  a = rnorm_na(5, 1),
  b = rnorm_na(5, 1),
  c = rnorm_na(5, 2),
  d = rnorm(5)
)
df_miss |>
  summarize(
    across(a:d, median),
    n = n()
  )
#> # A tibble: 1 × 5
#>       a     b     c     d     n
#>   <dbl> <dbl> <dbl> <dbl> <int>
#> 1    NA    NA    NA  1.15     5
```

이러한 결측값을 제거하기 위해 `na.rm = TRUE`를 <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>에 전달할 수 있다면 좋을 것입니다. 그렇게 하려면 <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>을 직접 호출하는 대신 원하는 인자로 <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>을 호출하는 새로운 함수를 만들어야 합니다.

```
df_miss |>
  summarize(
    across(a:d, function(x) median(x, na.rm = TRUE)),
    n = n()
  )
#> # A tibble: 1 × 5
#>       a     b      c     d     n
#>   <dbl> <dbl>  <dbl> <dbl> <int>
#> 1 0.139 -1.11 -0.387  1.15     5
```

이것은 약간 장황하므로 R에는 편리한 단축키가 포함되어 있습니다. 이러한 종류의 일회용(또는 _익명_)<sup><a href="ch26.html#idm44771267612512" id="idm44771267612512-marker" data-type="noteref">1</a></sup> 함수의 경우 `function`을 `\`로 대체할 수 있습니다.<sup><a href="ch26.html#idm44771267610256" id="idm44771267610256-marker" data-type="noteref">2</a></sup>

```
df_miss |>
  summarize(
    across(a:d, \(x) median(x, na.rm = TRUE)),
    n = n()
  )
```

어느 경우든 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>는 효과적으로 다음 코드로 확장됩니다.

```
df_miss |>
  summarize(
    a = median(a, na.rm = TRUE),
    b = median(b, na.rm = TRUE),
    c = median(c, na.rm = TRUE),
    d = median(d, na.rm = TRUE),
    n = n()
  )
```

<a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>에서 결측값을 제거할 때 얼마나 많은 값이 제거되었는지 알 수 있다면 좋을 것입니다. <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>에 두 개의 함수, 즉 중앙값을 계산하는 함수와 결측값을 세는 함수를 제공하여 알아낼 수 있습니다. `.fns`에 명명된 목록을 사용하여 여러 함수를 제공합니다.

```
df_miss |>
  summarize(
    across(a:d, list(
      median = \(x) median(x, na.rm = TRUE),
      n_miss = \(x) sum(is.na(x))
    )),
    n = n()
  )
#> # A tibble: 1 × 9
#>   a_median a_n_miss b_median b_n_miss c_median c_n_miss d_median d_n_miss
#>      <dbl>    <int>    <dbl>    <int>    <dbl>    <int>    <dbl>    <int>
#> 1    0.139        1    -1.11        1   -0.387        2     1.15        0
#> # … with 1 more variable: n <int>
```

주의 깊게 살펴보면, 열 이름이 원본 열의 이름인 `.col`과 함수의 이름인 `.fn`을 사용하여 `{.col}_{.fn}`과 같은 접착 사양(glue specification, <a href="ch14.html#sec-glue" data-type="xref">“str_glue()”</a>)을 사용하여 명명되었음을 직감할 수 있습니다. 우연이 아닙니다! 다음 섹션에서 배우겠지만 `.names` 인자를 사용하여 자신만의 접착 사양을 제공할 수 있습니다.

## 열 이름 (Column Names)

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>의 결과는 `.names` 인자에 제공된 사양에 따라 이름이 지정됩니다. 함수의 이름이 먼저 오기를 원하는 경우 자체 사양을 지정할 수 있습니다.<sup><a href="ch26.html#idm44771267377264" id="idm44771267377264-marker" data-type="noteref">3</a></sup>

```
df_miss |>
  summarize(
    across(
      a:d,
      list(
        median = \(x) median(x, na.rm = TRUE),
        n_miss = \(x) sum(is.na(x))
      ),
      .names = "{.fn}_{.col}"
    ),
    n = n(),
  )
#> # A tibble: 1 × 9
#>   median_a n_miss_a median_b n_miss_b median_c n_miss_c median_d n_miss_d
#>      <dbl>    <int>    <dbl>    <int>    <dbl>    <int>    <dbl>    <int>
#> 1    0.139        1    -1.11        1   -0.387        2     1.15        0
#> # … with 1 more variable: n <int>
```

`.names` 인자는 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 함께 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>를 사용할 때 특히 중요합니다. 기본적으로 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>의 출력은 입력과 동일한 이름을 갖습니다. 즉, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 내의 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>는 기존 열을 대체합니다. 예를 들어, 여기서는 <a href="https://dplyr.tidyverse.org/reference/coalesce.html" class="orm:hideurl"><code>coalesce()</code></a>를 사용하여 `NA`를 `0`으로 바꿉니다.

```
df_miss |>
  mutate(
    across(a:d, \(x) coalesce(x, 0))
  )
#> # A tibble: 5 × 4
```

#> a b c d
#> <dbl> <dbl> <dbl> <dbl>
#> 1 0.434 -1.25 0 1.60
#> 2 0 -1.43 -0.297 0.776
#> 3 -0.156 -0.980 0 1.15
#> 4 -2.61 -0.683 -0.785 2.13
#> 5 1.11 0 -0.387 0.704

```

대신 새 열을 만들려면 `.names` 인자를 사용하여 출력에 새 이름을 지정할 수 있습니다.

```

df_miss |>
mutate(
across(a:d, \(x) abs(x), .names = "{.col}\_abs")
)
#> # A tibble: 5 × 8
#> a b c d a_abs b_abs c_abs d_abs
#> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1 0.434 -1.25 NA 1.60 0.434 1.25 NA 1.60
#> 2 NA -1.43 -0.297 0.776 NA 1.43 0.297 0.776
#> 3 -0.156 -0.980 NA 1.15 0.156 0.980 NA 1.15
#> 4 -2.61 -0.683 -0.785 2.13 2.61 0.683 0.785 2.13
#> 5 1.11 NA -0.387 0.704 1.11 NA 0.387 0.704

```

## 필터링 (Filtering)

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>는 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 잘 어울리지만, 보통 여러 조건을 `|` 또는 `&`와 결합하기 때문에 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 함께 사용하기에는 더 어색합니다. <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>가 여러 논리 열을 만드는 데 도움이 될 수 있다는 것은 분명하지만, 그 다음엔 어떻게 해야 할까요? 따라서 dplyr은 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_any()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>if_all()</code></a>이라는 두 가지 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>의 변형을 제공합니다.

```

# same as df_miss |> filter(is.na(a) | is.na(b) | is.na(c) | is.na(d))

df_miss |> filter(if_any(a:d, is.na))
#> # A tibble: 4 × 4
#> a b c d
#> <dbl> <dbl> <dbl> <dbl>
#> 1 0.434 -1.25 NA 1.60
#> 2 NA -1.43 -0.297 0.776
#> 3 -0.156 -0.980 NA 1.15
#> 4 1.11 NA -0.387 0.704

# same as df_miss |> filter(is.na(a) & is.na(b) & is.na(c) & is.na(d))

df_miss |> filter(if_all(a:d, is.na))
#> # A tibble: 0 × 4
#> # … with 4 variables: a <dbl>, b <dbl>, c <dbl>, d <dbl>

```

## 함수에서의 across() (across() in Functions)

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>는 여러 열에서 작업할 수 있도록 해주기 때문에 프로그래밍하는 데 특히 유용합니다. 예를 들어, [Jacob Scott](https://oreil.ly/6vVc4)은 여러 lubridate 함수를 래핑하여 모든 날짜 열을 연, 월, 일 열로 확장하는 이 작은 헬퍼를 사용합니다.

```

expand_dates <- function(df) {
df |>
mutate(
across(where(is.Date), list(year = year, month = month, day = mday))
)
}

df_date <- tibble(
name = c("Amy", "Bob"),
date = ymd(c("2009-08-03", "2010-01-16"))
)

df_date |>
expand_dates()
#> # A tibble: 2 × 5
#> name date date_year date_month date_day
#> <chr> <date> <dbl> <dbl> <int>
#> 1 Amy 2009-08-03 2009 8 3
#> 2 Bob 2010-01-16 2010 1 16

```

<a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>는 첫 번째 인자가 tidy-select를 사용하기 때문에 단일 인자에 여러 열을 쉽게 제공할 수 있도록 해줍니다. <a href="ch25.html#sec-embracing" data-type="xref">“언제 포용(Embrace)해야 하는가?”</a>에서 논의한 대로 해당 인자를 포용해야 한다는 것만 기억하면 됩니다. 예를 들어 이 함수는 기본적으로 숫자형 열의 평균을 계산합니다. 그러나 두 번째 인자를 제공하면 선택한 열만 요약하도록 선택할 수 있습니다.

```

summarize_means <- function(df, summary_vars = where(is.numeric)) {
df |>
summarize(
across({{ summary_vars }}, \(x) mean(x, na.rm = TRUE)),
n = n()
)
}
diamonds |>
group_by(cut) |>
summarize_means()
#> # A tibble: 5 × 9
#> cut carat depth table price x y z n
#> <ord> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <int>
#> 1 Fair 1.05 64.0 59.1 4359. 6.25 6.18 3.98 1610
#> 2 Good 0.849 62.4 58.7 3929. 5.84 5.85 3.64 4906
#> 3 Very Good 0.806 61.8 58.0 3982. 5.74 5.77 3.56 12082
#> 4 Premium 0.892 61.3 58.7 4584. 5.97 5.94 3.65 13791
#> 5 Ideal 0.703 61.7 56.0 3458. 5.51 5.52 3.40 21551

diamonds |>
group_by(cut) |>
summarize_means(c(carat, x:z))
#> # A tibble: 5 × 6
#> cut carat x y z n
#> <ord> <dbl> <dbl> <dbl> <dbl> <int>
#> 1 Fair 1.05 6.25 6.18 3.98 1610
#> 2 Good 0.849 5.84 5.85 3.64 4906
#> 3 Very Good 0.806 5.74 5.77 3.56 12082
#> 4 Premium 0.892 5.97 5.94 3.65 13791
#> 5 Ideal 0.703 5.51 5.52 3.40 21551

```

## pivot_longer()와의 비교 (Versus pivot_longer())

계속하기 전에 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>(<a href="ch05.html#sec-pivoting" data-type="xref">“데이터 길게 만들기”</a>) 사이의 흥미로운 연관성을 지적할 가치가 있습니다. 많은 경우, 먼저 데이터를 피벗(pivot)한 다음 열 단위가 아닌 그룹 단위로 연산을 수행하여 동일한 계산을 수행합니다. 예를 들어 이 다기능 요약을 살펴보겠습니다.

```

df |>
summarize(across(a:d, list(median = median, mean = mean)))
#> # A tibble: 1 × 8
#> a_median a_mean b_median b_mean c_median c_mean d_median d_mean
#> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1 0.0380 0.205 -0.0163 0.0910 0.260 0.0716 0.540 0.508

```

데이터를 더 길게 피벗한 다음 요약하여 동일한 값을 계산할 수 있습니다.

```

long <- df |>
pivot_longer(a:d) |>
group_by(name) |>
summarize(
median = median(value),
mean = mean(value)
)
long
#> # A tibble: 4 × 3
#> name median mean
#> <chr> <dbl> <dbl>
#> 1 a 0.0380 0.205
#> 2 b -0.0163 0.0910
#> 3 c 0.260 0.0716
#> 4 d 0.540 0.508

```

그리고 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>와 동일한 구조를 원한다면 다시 피벗할 수 있습니다.

```

long |>
pivot*wider(
names_from = name,
values_from = c(median, mean),
names_vary = "slowest",
names_glue = "{name}*{.value}"
)
#> # A tibble: 1 × 8
#> a_median a_mean b_median b_mean c_median c_mean d_median d_mean
#> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
#> 1 0.0380 0.205 -0.0163 0.0910 0.260 0.0716 0.540 0.508

```

동시에 계산하고자 하는 열 그룹이 있을 때, 현재 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>로는 해결할 수 없는 문제에 부딪힐 때가 있기 때문에 이것은 알아두면 유용한 기술입니다. 예를 들어 데이터 프레임에 값과 가중치가 모두 포함되어 있고 가중 평균을 계산하고 싶다고 가정해 보겠습니다.

```

df_paired <- tibble(
a_val = rnorm(10),
a_wts = runif(10),
b_val = rnorm(10),
b_wts = runif(10),
c_val = rnorm(10),
c_wts = runif(10),
d_val = rnorm(10),
d_wts = runif(10)
)

```

현재 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>로 이 작업을 수행할 방법은 없지만,<sup><a href="ch26.html#idm44771266547392" id="idm44771266547392-marker" data-type="noteref">4</a></sup> <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>를 사용하면 비교적 간단합니다.

```

df*long <- df_paired |>
pivot_longer(
everything(),
names_to = c("group", ".value"),
names_sep = "*"
)
df_long
#> # A tibble: 40 × 3
#> group val wts
#> <chr> <dbl> <dbl>
#> 1 a 0.715 0.518
#> 2 b -0.709 0.691
#> 3 c 0.718 0.216
#> 4 d -0.217 0.733
#> 5 a -1.09 0.979
#> 6 b -0.209 0.675
#> # … with 34 more rows

df_long |>
group_by(group) |>
summarize(mean = weighted.mean(val, wts))
#> # A tibble: 4 × 2
#> group mean
#> <chr> <dbl>
#> 1 a 0.126
#> 2 b -0.0704
#> 3 c -0.360
#> 4 d -0.248

````

필요한 경우 이를 원래 형태로 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>할 수 있습니다.

## 연습문제 (Exercises)

1.  다음을 통해 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a> 기술을 연습하세요.

    1.  <a href="https://allisonhorst.github.io/palmerpenguins/reference/penguins.html" class="orm:hideurl"><code>palmerpenguins::penguins</code></a>의 각 열에 있는 고유값의 수를 계산합니다.

    2.  `mtcars`의 모든 열의 평균을 계산합니다.

    3.  `cut`, `clarity` 및 `color`로 `diamonds`를 그룹화한 다음 관측치 수를 세고 각 숫자형 열의 평균을 계산합니다.

2.  <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>에서 함수 목록을 사용하지만 이름을 지정하지 않으면 어떻게 됩니까? 출력 이름은 어떻게 지정됩니까?

3.  날짜 열이 확장된 후 자동으로 제거되도록 `expand_dates()`를 조정하세요. 인자를 포용해야 합니까?

4.  이 함수의 파이프라인의 각 단계가 무엇을 하는지 설명하세요. <a href="https://tidyselect.r-lib.org/reference/where.html" class="orm:hideurl"><code>where()</code></a>의 어떤 특수 기능을 활용하고 있습니까?

    ```
    show_missing <- function(df, group_vars, summary_vars = everything()) {
      df |>
        group_by(pick({{ group_vars }})) |>
        summarize(
          across({{ summary_vars }}, \(x) sum(is.na(x))),
          .groups = "drop"
        ) |>
        select(where(\(x) any(x > 0)))
    }
    nycflights13::flights |> show_missing(c(year, month, day))
    ```

# 다중 파일 읽기 (Reading Multiple Files)

이전 섹션에서는 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>dplyr::across()</code></a>를 사용하여 여러 열에서 변환을 반복하는 방법을 배웠습니다. 이 섹션에서는 디렉터리의 모든 파일에 대해 무언가를 수행하기 위해 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a>을 사용하는 방법을 배웁니다. 약간의 동기로 시작해 보겠습니다. 읽고 싶은 Excel 스프레드시트<sup><a href="ch26.html#idm44771266252816" id="idm44771266252816-marker" data-type="noteref">5</a></sup>로 가득 찬 디렉터리가 있다고 가정해 보겠습니다. 복사 및 붙여넣기로 수행할 수 있습니다.

````

data2019 <- readxl::read_excel("data/y2019.xlsx")
data2020 <- readxl::read_excel("data/y2020.xlsx")
data2021 <- readxl::read_excel("data/y2021.xlsx")
data2022 <- readxl::read_excel("data/y2022.xlsx")

```

그런 다음 <a href="https://dplyr.tidyverse.org/reference/bind_rows.html" class="orm:hideurl"><code>dplyr::bind_rows()</code></a>를 사용하여 모두 결합합니다.

```

```

특히 파일이 4개가 아니라 수백 개라면 이 작업이 금방 지루해질 것이라고 상상할 수 있습니다. 다음 섹션에서는 이런 종류의 작업을 자동화하는 방법을 보여줍니다. 세 가지 기본 단계가 있습니다. <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a>를 사용하여 디렉터리의 모든 파일을 나열한 다음 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a>을 사용하여 각 파일을 리스트로 읽어 들인 다음 <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>purrr::list_rbind()</code></a>를 사용하여 단일 데이터 프레임으로 결합합니다. 그런 다음 모든 파일에 대해 동일한 작업을 수행할 수 없는 이질성(heterogeneity)이 증가하는 상황을 처리하는 방법에 대해 논의합니다.

## 디렉터리의 파일 나열하기 (Listing Files in a Directory)

이름에서 알 수 있듯이 <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a>는 디렉터리의 파일을 나열합니다. 거의 항상 세 가지 인자를 사용하게 됩니다.

- 첫 번째 인자인 `path`는 살펴볼 디렉터리입니다.

- `pattern`은 파일 이름을 필터링하는 데 사용되는 정규 표현식입니다. 지정된 확장자를 가진 모든 파일을 찾기 위한 가장 일반적인 패턴은 `[.]xlsx$` 또는 `[.]csv$`와 같은 것입니다.

- `full.names`는 디렉터리 이름이 출력에 포함되어야 하는지 여부를 결정합니다. 거의 항상 이것이 `TRUE`가 되기를 원할 것입니다.

우리의 동기 부여 예제를 구체화하기 위해, 이 책에는 gapminder 패키지의 데이터가 포함된 12개의 Excel 스프레드시트가 있는 폴더가 포함되어 있습니다. 각 파일에는 142개국의 1년치 데이터가 포함되어 있습니다. <a href="https://rdrr.io/r/base/list.files.html" class="orm:hideurl"><code>list.files()</code></a>에 대한 적절한 호출을 통해 모두 나열할 수 있습니다.

```

paths <- list.files("data/gapminder", pattern = "[.]xlsx$", full.names = TRUE)
paths
#> [1] "data/gapminder/1952.xlsx" "data/gapminder/1957.xlsx"
#> [3] "data/gapminder/1962.xlsx" "data/gapminder/1967.xlsx"
#> [5] "data/gapminder/1972.xlsx" "data/gapminder/1977.xlsx"
#> [7] "data/gapminder/1982.xlsx" "data/gapminder/1987.xlsx"
#> [9] "data/gapminder/1992.xlsx" "data/gapminder/1997.xlsx"
#> [11] "data/gapminder/2002.xlsx" "data/gapminder/2007.xlsx"

```

## 리스트 (Lists)

이제 이 12개의 경로가 생겼으므로 `read_excel()`을 12번 호출하여 12개의 데이터 프레임을 얻을 수 있습니다.

```

gapminder_1952 <- readxl::read_excel("data/gapminder/1952.xlsx")
gapminder_1957 <- readxl::read_excel("data/gapminder/1957.xlsx")
gapminder_1962 <- readxl::read_excel("data/gapminder/1962.xlsx")
...,
gapminder_2007 <- readxl::read_excel("data/gapminder/2007.xlsx")

```

하지만 각 시트를 자체 변수에 넣으면 몇 단계 후에 작업하기가 어려워질 것입니다. 대신 단일 객체에 넣으면 작업하기가 더 쉬울 것입니다. 리스트는 이 작업에 완벽한 도구입니다.

```

files <- list(
readxl::read_excel("data/gapminder/1952.xlsx"),
readxl::read_excel("data/gapminder/1957.xlsx"),
readxl::read_excel("data/gapminder/1962.xlsx"),
...,
readxl::read_excel("data/gapminder/2007.xlsx")
)

```

이제 이러한 데이터 프레임이 리스트에 있으므로 어떻게 하나를 꺼낼까요? `files[[i]]`를 사용하여 *i*번째 요소를 추출할 수 있습니다.

```

files[[3]]
#> # A tibble: 142 × 5
#> country continent lifeExp pop gdpPercap
#> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 Afghanistan Asia 32.0 10267083 853.
#> 2 Albania Europe 64.8 1728137 2313.
#> 3 Algeria Africa 48.3 11000948 2551.
#> 4 Angola Africa 34 4826015 4269.
#> 5 Argentina Americas 65.1 21283783 7133.
#> 6 Australia Oceania 70.9 10794968 12217.
#> # … with 136 more rows

```

<a href="ch27.html#sec-subset-one" data-type="xref">“$와 [[를 사용한 단일 요소 선택”</a>에서 `[[`에 대해 더 자세히 다시 살펴보겠습니다.

## purrr::map() 및 list_rbind()

이러한 데이터 프레임을 "수작업"으로 리스트에 수집하는 코드는 기본적으로 파일을 하나씩 읽는 코드만큼이나 타이핑하기 지루합니다. 다행히 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>purrr::map()</code></a>을 사용하여 `paths` 벡터를 훨씬 더 잘 활용할 수 있습니다. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>은 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>와 유사하지만 데이터 프레임의 각 열에 무언가를 하는 대신 벡터의 각 요소에 무언가를 합니다. `map(x, f)`는 다음의 줄임말입니다.

```

list(
f(x[[1]]),
f(x[[2]]),
...,
f(x[[n]])
)

```

따라서 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>을 사용하여 12개의 데이터 프레임 리스트를 얻을 수 있습니다.

```

files <- map(paths, readxl::read_excel)
length(files)
#> [1] 12

files[[1]]
#> # A tibble: 142 × 5
#> country continent lifeExp pop gdpPercap
#> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 Afghanistan Asia 28.8 8425333 779.
#> 2 Albania Europe 55.2 1282697 1601.
#> 3 Algeria Africa 43.1 9279525 2449.
#> 4 Angola Africa 30.0 4232095 3521.
#> 5 Argentina Americas 62.5 17876956 5911.
#> 6 Australia Oceania 69.1 8691212 10040.
#> # … with 136 more rows

```

(이것은 <a href="https://rdrr.io/r/utils/str.html" class="orm:hideurl"><code>str()</code></a>을 사용하여 특별히 간결하게 표시되지 않는 또 다른 데이터 구조이므로 RStudio에 로드하고 <a href="https://rdrr.io/r/utils/View.html" class="orm:hideurl"><code>View()</code></a>를 사용하여 검사하는 것이 좋습니다.)

이제 <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>purrr::list_rbind()</code></a>를 사용하여 데이터 프레임의 리스트를 단일 데이터 프레임으로 결합할 수 있습니다.

```

list_rbind(files)
#> # A tibble: 1,704 × 5
#> country continent lifeExp pop gdpPercap
#> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 Afghanistan Asia 28.8 8425333 779.
#> 2 Albania Europe 55.2 1282697 1601.
#> 3 Algeria Africa 43.1 9279525 2449.
#> 4 Angola Africa 30.0 4232095 3521.
#> 5 Argentina Americas 62.5 17876956 5911.
#> 6 Australia Oceania 69.1 8691212 10040.
#> # … with 1,698 more rows

```

또는 파이프라인에서 두 단계를 한 번에 수행할 수 있습니다.

```

paths |>
map(readxl::read_excel) |>
list_rbind()

```

`read_excel()`에 추가 인자를 전달하려면 어떻게 해야 할까요? <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>와 동일한 기술을 사용합니다. 예를 들어 `n_max = 1`을 사용하여 데이터의 처음 몇 행을 살펴보는 것이 종종 유용합니다.

```

paths |>
map(\(path) readxl::read_excel(path, n_max = 1)) |>
list_rbind()
#> # A tibble: 12 × 5
#> country continent lifeExp pop gdpPercap
#> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 Afghanistan Asia 28.8 8425333 779.
#> 2 Afghanistan Asia 30.3 9240934 821.
#> 3 Afghanistan Asia 32.0 10267083 853.
#> 4 Afghanistan Asia 34.0 11537966 836.
#> 5 Afghanistan Asia 36.1 13079460 740.
#> 6 Afghanistan Asia 38.4 14880372 786.
#> # … with 6 more rows

```

이것은 무언가가 빠져 있다는 것을 분명히 합니다. `year` 열이 없습니다. 해당 값이 개별 파일이 아닌 경로에 기록되어 있기 때문입니다. 다음에 그 문제를 다루겠습니다.

## 경로 내의 데이터 (Data in the Path)

때로는 파일 이름 자체가 데이터이기도 합니다. 이 예제에서 파일 이름에는 연도가 포함되어 있는데, 개별 파일에는 기록되어 있지 않습니다. 최종 데이터 프레임으로 해당 열을 가져오려면 두 가지를 해야 합니다.

첫째, 경로의 벡터에 이름을 지정합니다. 이를 수행하는 가장 쉬운 방법은 함수를 취할 수 있는 <a href="https://rlang.r-lib.org/reference/set_names.html" class="orm:hideurl"><code>set_names()</code></a> 함수를 사용하는 것입니다. 여기서는 <a href="https://rdrr.io/r/base/basename.html" class="orm:hideurl"><code>basename()</code></a>을 사용하여 전체 경로에서 파일 이름만 추출합니다.

```

paths |> set_names(basename)
#> 1952.xlsx 1957.xlsx
#> "data/gapminder/1952.xlsx" "data/gapminder/1957.xlsx"
#> 1962.xlsx 1967.xlsx
#> "data/gapminder/1962.xlsx" "data/gapminder/1967.xlsx"
#> 1972.xlsx 1977.xlsx
#> "data/gapminder/1972.xlsx" "data/gapminder/1977.xlsx"
#> 1982.xlsx 1987.xlsx
#> "data/gapminder/1982.xlsx" "data/gapminder/1987.xlsx"
#> 1992.xlsx 1997.xlsx
#> "data/gapminder/1992.xlsx" "data/gapminder/1997.xlsx"
#> 2002.xlsx 2007.xlsx
#> "data/gapminder/2002.xlsx" "data/gapminder/2007.xlsx"

```

이러한 이름은 모든 map 함수에 의해 자동으로 전달되므로 데이터 프레임 리스트에도 동일한 이름이 사용됩니다.

```

files <- paths |>
set_names(basename) |>
map(readxl::read_excel)

```

이것은 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>에 대한 이 호출을 다음의 줄임말로 만듭니다.

```

files <- list(
"1952.xlsx" = readxl::read_excel("data/gapminder/1952.xlsx"),
"1957.xlsx" = readxl::read_excel("data/gapminder/1957.xlsx"),
"1962.xlsx" = readxl::read_excel("data/gapminder/1962.xlsx"),
...,
"2007.xlsx" = readxl::read_excel("data/gapminder/2007.xlsx")
)

```

이름으로 요소를 추출하기 위해 `[[`를 사용할 수도 있습니다.

```

files[["1962.xlsx"]]
#> # A tibble: 142 × 5
#> country continent lifeExp pop gdpPercap
#> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 Afghanistan Asia 32.0 10267083 853.
#> 2 Albania Europe 64.8 1728137 2313.
#> 3 Algeria Africa 48.3 11000948 2551.
#> 4 Angola Africa 34 4826015 4269.
#> 5 Argentina Americas 65.1 21283783 7133.
#> 6 Australia Oceania 70.9 10794968 12217.
#> # … with 136 more rows

```

그런 다음 <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a>에 `names_to` 인자를 사용하여 `year`라는 새 열에 이름을 저장하도록 지시한 다음 <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>readr::parse_number()</code></a>를 사용하여 문자열에서 숫자를 추출합니다.

```

paths |>
set_names(basename) |>
map(readxl::read_excel) |>
list_rbind(names_to = "year") |>
mutate(year = parse_number(year))
#> # A tibble: 1,704 × 6
#> year country continent lifeExp pop gdpPercap
#> <dbl> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 1952 Afghanistan Asia 28.8 8425333 779.
#> 2 1952 Albania Europe 55.2 1282697 1601.
#> 3 1952 Algeria Africa 43.1 9279525 2449.
#> 4 1952 Angola Africa 30.0 4232095 3521.
#> 5 1952 Argentina Americas 62.5 17876956 5911.
#> 6 1952 Australia Oceania 69.1 8691212 10040.
#> # … with 1,698 more rows

```

더 복잡한 경우 디렉터리 이름에 저장된 다른 변수가 있거나 파일 이름에 데이터의 여러 비트가 포함될 수 있습니다. 이 경우 아무 인자 없이 <a href="https://rlang.r-lib.org/reference/set_names.html" class="orm:hideurl"><code>set_names()</code></a>를 사용하여 전체 경로를 기록한 다음 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>tidyr::separate_wider_delim()</code></a> 및 유사한 함수들을 사용하여 유용한 열로 바꿉니다.

```

paths |>
set_names() |>
map(readxl::read_excel) |>
list_rbind(names_to = "year") |>
separate_wider_delim(year, delim = "/", names = c(NA, "dir", "file")) |>
separate_wider_delim(file, delim = ".", names = c("file", "ext"))
#> # A tibble: 1,704 × 8
#> dir file ext country continent lifeExp pop gdpPercap
#> <chr> <chr> <chr> <chr> <chr> <dbl> <dbl> <dbl>
#> 1 gapminder 1952 xlsx Afghanistan Asia 28.8 8425333 779.
#> 2 gapminder 1952 xlsx Albania Europe 55.2 1282697 1601.
#> 3 gapminder 1952 xlsx Algeria Africa 43.1 9279525 2449.
#> 4 gapminder 1952 xlsx Angola Africa 30.0 4232095 3521.
#> 5 gapminder 1952 xlsx Argentina Americas 62.5 17876956 5911.
#> 6 gapminder 1952 xlsx Australia Oceania 69.1 8691212 10040.
#> # … with 1,698 more rows

```

## 작업 저장하기 (Save Your Work)
이제 멋지고 깔끔한 데이터 프레임을 얻기 위해 이 모든 힘든 작업을 완료했으므로 작업을 저장하기에 좋은 시기입니다.

```

gapminder <- paths |>
set_names(basename) |>
map(readxl::read_excel) |>
list_rbind(names_to = "year") |>
mutate(year = parse_number(year))

write_csv(gapminder, "gapminder.csv")

```

이제 나중에 이 문제로 다시 돌아오면 단일 CSV 파일을 읽을 수 있습니다. 크고 풍부한 데이터 세트의 경우 <a href="ch22.html#sec-parquet" data-type="xref">“Parquet 형식”</a>에서 논의한 것처럼 `.csv`보다 parquet을 사용하는 것이 더 나은 선택일 수 있습니다.

프로젝트에서 작업 중인 경우, 이런 종류의 데이터 준비 작업을 수행하는 파일을 `0-cleanup.R`과 같이 호출하는 것이 좋습니다. 파일 이름에 있는 `0`은 이것이 다른 어떤 것보다 먼저 실행되어야 함을 나타냅니다.

입력 데이터 파일이 시간이 지남에 따라 변경되는 경우, 입력 파일 중 하나가 수정될 때마다 데이터 정리 코드가 자동으로 다시 실행되도록 설정하기 위해 [targets](https://oreil.ly/oJsOo)와 같은 도구를 배우는 것을 고려해 볼 수 있습니다.

## 많은 간단한 반복 (Many Simple Iterations)

여기서는 디스크에서 직접 데이터를 로드했고 운 좋게도 깔끔한 데이터 세트를 얻었습니다. 대부분의 경우 몇 가지 추가 정리를 해야 하며 두 가지 기본 옵션이 있습니다. 복잡한 함수로 반복을 한 번 하거나 간단한 함수로 반복을 여러 번 할 수 있습니다. 우리의 경험에 따르면 대부분의 사람들은 처음에는 하나의 복잡한 반복을 사용하려고 하지만, 종종 여러 개의 간단한 반복을 수행하는 것이 더 낫습니다.

예를 들어 많은 파일을 읽어 들이고, 결측값을 필터링하고, 피벗하고, 결합하고 싶다고 가정해 보겠습니다. 이 문제에 접근하는 한 가지 방법은 파일을 가져와서 이러한 모든 단계를 수행하는 함수를 작성한 다음 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>을 한 번 호출하는 것입니다.

```

process_file <- function(path) {
df <- read_csv(path)

df |>
filter(!is.na(id)) |>
mutate(id = tolower(id)) |>
pivot_longer(jan:dec, names_to = "month")
}

paths |>
map(process_file) |>
list_rbind()

```

또는 모든 파일에 대해 `process_file()`의 각 단계를 수행할 수 있습니다.

```

paths |>
map(read_csv) |>
map(\(df) df |> filter(!is.na(id))) |>
map(\(df) df |> mutate(id = tolower(id))) |>
map(\(df) df |> pivot_longer(jan:dec, names_to = "month")) |>
list_rbind()

```

이 접근 방식을 권장하는 이유는 첫 번째 파일을 올바르게 처리하는 데 집착하여 나머지 파일로 넘어가지 못하는 것을 방지하기 때문입니다. 정리하고 정리할 때 모든 데이터를 고려함으로써 전체적으로 생각할 가능성이 높아지고 궁극적으로 더 높은 품질의 결과를 얻을 수 있습니다.

이 특정 예제에서 데이터를 모두 일찍 결합하여 또 다른 최적화를 수행할 수 있습니다. 그런 다음 일반적인 dplyr 동작을 사용할 수 있습니다.

```

paths |>
map(read_csv) |>
list_rbind() |>
filter(!is.na(id)) |>
mutate(id = tolower(id)) |>
pivot_longer(jan:dec, names_to = "month")

```

## 이질적인 데이터 (Heterogeneous Data)

불행히도 데이터 프레임이 너무 이질적이어서 <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a>가 실패하거나 유용하지 않은 데이터 프레임을 생성하기 때문에 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>에서 <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a>로 바로 이동하는 것이 불가능한 경우가 있습니다. 이 경우 여전히 모든 파일을 로드하는 것으로 시작하는 것이 유용합니다.

```

files <- paths |>
map(readxl::read_excel)

```

그런 다음 유용한 전략은 데이터 과학 기술을 사용하여 탐색할 수 있도록 데이터 프레임의 구조를 파악하는 것입니다. 이를 수행하는 한 가지 방법은 각 열에 대해 하나의 행이 있는 티블을 반환하는 편리한 `df_types` 함수<sup><a href="ch26.html#idm44771264920592" id="idm44771264920592-marker" data-type="noteref">6</a></sup>를 사용하는 것입니다.

```

df_types <- function(df) {
tibble(
col_name = names(df),
col_type = map_chr(df, vctrs::vec_ptype_full),
n_miss = map_int(df, \(x) sum(is.na(x)))
)
}

df_types(gapminder)
#> # A tibble: 6 × 3
#> col_name col_type n_miss
#> <chr> <chr> <int>
#> 1 year double 0
#> 2 country character 0
#> 3 continent character 0
#> 4 lifeExp double 0
#> 5 pop double 0
#> 6 gdpPercap double 0

```

그런 다음 이 함수를 모든 파일에 적용하고 약간의 피벗팅을 수행하여 차이점이 있는 위치를 더 쉽게 확인할 수 있습니다. 예를 들어, 우리가 작업해 온 gapminder 스프레드시트가 모두 상당히 동질적이라는 것을 쉽게 확인할 수 있습니다.

```

files |>
map(df_types) |>
list_rbind(names_to = "file_name") |>
select(-n_miss) |>
pivot_wider(names_from = col_name, values_from = col_type)
#> # A tibble: 12 × 6
#> file_name country continent lifeExp pop gdpPercap
#> <chr> <chr> <chr> <chr> <chr> <chr>  
#> 1 1952.xlsx character character double double double  
#> 2 1957.xlsx character character double double double  
#> 3 1962.xlsx character character double double double  
#> 4 1967.xlsx character character double double double  
#> 5 1972.xlsx character character double double double  
#> 6 1977.xlsx character character double double double  
#> # … with 6 more rows

```

파일의 형식이 이질적인 경우 성공적으로 병합하기 전에 추가 처리를 해야 할 수도 있습니다. 불행히도 이제는 여러분이 스스로 알아내도록 맡길 것이지만 <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_if()</code></a> 및 <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_at()</code></a>에 대해 읽어보고 싶을 수 있습니다. <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_if()</code></a>를 사용하면 값에 따라 리스트의 요소를 선택적으로 수정할 수 있고, <a href="https://purrr.tidyverse.org/reference/map_if.html" class="orm:hideurl"><code>map_at()</code></a>을 사용하면 이름에 따라 요소를 선택적으로 수정할 수 있습니다.

## 실패 처리 (Handling Failures)

때로는 데이터의 구조가 단일 명령으로 모든 파일을 읽을 수 없을 만큼 충분히 거칠 수 있습니다. 그런 다음 `map()`의 단점 중 하나를 만나게 됩니다. 전체적으로 성공하거나 실패합니다. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>은 디렉터리의 모든 파일을 성공적으로 읽거나 오류와 함께 실패하여 0개의 파일을 읽습니다. 이것은 짜증나는 일입니다. 하나의 실패로 인해 다른 모든 성공한 데이터에 액세스하지 못하게 되는 이유는 무엇일까요?

다행히 purrr에는 이 문제를 해결하기 위한 헬퍼인 <a href="https://purrr.tidyverse.org/reference/possibly.html" class="orm:hideurl"><code>possibly()</code></a>가 함께 제공됩니다. <a href="https://purrr.tidyverse.org/reference/possibly.html" class="orm:hideurl"><code>possibly()</code></a>는 *함수 연산자(function operator)*로 알려진 것입니다. 함수를 취하고 동작이 수정된 함수를 반환합니다. 특히 <a href="https://purrr.tidyverse.org/reference/possibly.html" class="orm:hideurl"><code>possibly()</code></a>는 함수가 오류를 발생시키는 것을 방지하고 지정한 값을 반환하도록 변경합니다.

```

files <- paths |>
map(possibly(\(path) readxl::read_excel(path), NULL))

data <- files |> list_rbind()

```

많은 tidyverse 함수와 마찬가지로 <a href="https://purrr.tidyverse.org/reference/list_c.html" class="orm:hideurl"><code>list_rbind()</code></a>는 `NULL`을 자동으로 무시하기 때문에 여기서는 특히 잘 작동합니다.

이제 쉽게 읽을 수 있는 모든 데이터를 갖추었으므로 일부 파일을 로드하지 못한 이유와 대처 방법을 파악하는 어려운 부분을 해결할 때입니다. 실패한 경로를 가져오는 것으로 시작합니다.

```

failed <- map_vec(files, is.null)
paths[failed]
#> character(0)

```

그런 다음 각 실패에 대해 가져오기 함수를 다시 호출하고 무엇이 잘못되었는지 파악합니다.

# 다중 출력 저장하기 (Saving Multiple Outputs)

이전 섹션에서는 여러 파일을 단일 객체로 읽는 데 유용한 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>에 대해 배웠습니다. 이 섹션에서는 정반대의 문제를 살펴봅니다. 하나 이상의 R 객체를 하나 이상의 파일에 어떻게 저장할 수 있을까요? 세 가지 예제를 사용하여 이 과제를 탐구하겠습니다.

- 여러 데이터 프레임을 하나의 데이터베이스에 저장
- 여러 데이터 프레임을 여러 `.csv` 파일에 저장
- 여러 플롯을 여러 `.png` 파일에 저장

## 데이터베이스에 쓰기 (Writing to a Database)

때로는 여러 파일을 한 번에 작업할 때 모든 데이터를 메모리에 한 번에 맞출 수 없으며 `map(files, read_csv)`를 수행할 수 없습니다. 이 문제를 다루는 한 가지 접근 방식은 데이터를 데이터베이스에 로드하여 dbplyr로 필요한 부분만 액세스할 수 있도록 하는 것입니다.

운이 좋다면, 사용 중인 데이터베이스 패키지가 경로 벡터를 취하여 데이터베이스에 모두 로드하는 편리한 함수를 제공할 것입니다. duckdb의 `duckdb_read_csv()`의 경우가 그렇습니다.

```

con <- DBI::dbConnect(duckdb::duckdb())
duckdb::duckdb_read_csv(con, "gapminder", paths)

```

이것은 여기에서 잘 작동하겠지만, 우리는 CSV 파일이 없고 대신 Excel 스프레드시트가 있습니다. 그래서 우리는 "수작업"으로 해야 할 것입니다. 수작업으로 하는 방법을 배우면 여러 개의 CSV 파일이 있고 작업 중인 데이터베이스에 모두 로드하는 하나의 함수가 없을 때에도 도움이 될 것입니다.

데이터로 채울 테이블을 만드는 것부터 시작해야 합니다. 이를 수행하는 가장 쉬운 방법은 원하는 모든 열을 포함하되 데이터의 샘플링만 포함하는 더미 데이터 프레임인 템플릿(template)을 만드는 것입니다. gapminder 데이터의 경우 단일 파일을 읽고 연도를 추가하여 해당 템플릿을 만들 수 있습니다.

```

template <- readxl::read_excel(paths[[1]])
template$year <- 1952
template
#> # A tibble: 142 × 6
#> country continent lifeExp pop gdpPercap year
#> <chr> <chr> <dbl> <dbl> <dbl> <dbl>
#> 1 Afghanistan Asia 28.8 8425333 779. 1952
#> 2 Albania Europe 55.2 1282697 1601. 1952
#> 3 Algeria Africa 43.1 9279525 2449. 1952
#> 4 Angola Africa 30.0 4232095 3521. 1952
#> 5 Argentina Americas 62.5 17876956 5911. 1952
#> 6 Australia Oceania 69.1 8691212 10040. 1952
#> # … with 136 more rows

```

이제 데이터베이스에 연결하고 <a href="https://dbi.r-dbi.org/reference/dbCreateTable.html" class="orm:hideurl"><code>DBI::dbCreateTable()</code></a>을 사용하여 템플릿을 데이터베이스 테이블로 전환할 수 있습니다.

```

con <- DBI::dbConnect(duckdb::duckdb())
DBI::dbCreateTable(con, "gapminder", template)

```

`dbCreateTable()`은 `template`의 데이터는 사용하지 않고 변수 이름과 타입만 사용합니다. 그래서 지금 `gapminder` 테이블을 검사해보면 비어 있지만 기대하는 타입으로 필요한 변수들을 가지고 있는 것을 알 수 있습니다.

```

con |> tbl("gapminder")
#> # Source: table<gapminder> [0 x 6]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#> # … with 6 variables: country <chr>, continent <chr>, lifeExp <dbl>,
#> # pop <dbl>, gdpPercap <dbl>, year <dbl>

```

다음으로 단일 파일 경로를 받아 R로 읽고 결과를 `gapminder` 테이블에 추가하는 함수가 필요합니다. `read_excel()`과 <a href="https://dbi.r-dbi.org/reference/dbAppendTable.html" class="orm:hideurl"><code>DBI::dbAppendTable()</code></a>을 결합하여 이를 수행할 수 있습니다.

```

append_file <- function(path) {
df <- readxl::read_excel(path)
df$year <- parse_number(basename(path))

DBI::dbAppendTable(con, "gapminder", df)
}

```

이제 `paths`의 각 요소에 대해 `append_file()`을 한 번씩 호출해야 합니다. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>을 사용하면 확실히 가능합니다.

```

paths |> map(append_file)

```

하지만 우리는 `append_file()`의 출력에는 관심이 없으므로 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a> 대신 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>walk()</code></a>를 사용하는 것이 조금 더 낫습니다. <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>walk()</code></a>는 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>과 정확히 동일한 작업을 수행하지만 출력을 버립니다.

```

paths |> walk(append_file)

```

이제 테이블에 모든 데이터가 있는지 확인할 수 있습니다.

```

con |>
tbl("gapminder") |>
count(year)
#> # Source: SQL [?? x 2]
#> # Database: DuckDB 0.6.1 [root@Darwin 22.3.0:R 4.2.1/:memory:]
#> year n
#> <dbl> <dbl>
#> 1 1952 142
#> 2 1957 142
#> 3 1962 142
#> 4 1967 142
#> 5 1972 142
#> 6 1977 142
#> # … with more rows

```

## CSV 파일 쓰기 (Writing CSV Files)

각 그룹에 대해 하나씩 여러 CSV 파일을 작성하려는 경우에도 동일한 기본 원칙이 적용됩니다. <a href="https://ggplot2.tidyverse.org/reference/diamonds.html" class="orm:hideurl"><code>ggplot2::diamonds</code></a> 데이터를 가져와서 각 `clarity`에 대해 하나의 CSV 파일을 저장하고 싶다고 가정해 보겠습니다. 먼저 이러한 개별 데이터 세트를 만들어야 합니다. 이 작업을 수행할 수 있는 방법은 여러 가지가 있지만 우리가 특히 좋아하는 한 가지 방법인 <a href="https://dplyr.tidyverse.org/reference/group_nest.html" class="orm:hideurl"><code>group_nest()</code></a>가 있습니다.

```

by_clarity <- diamonds |>
group_nest(clarity)

by_clarity
#> # A tibble: 8 × 2

```
#>   clarity               data
#>   <ord>   <list<tibble[,9]>>
#> 1 I1               [741 × 9]
#> 2 SI2            [9,194 × 9]
#> 3 SI1           [13,065 × 9]
#> 4 VS2           [12,258 × 9]
#> 5 VS1            [8,171 × 9]
#> 6 VVS2           [5,066 × 9]
#> # … with 2 more rows
```

이렇게 하면 8개의 행과 2개의 열이 있는 새로운 티블이 만들어집니다. `clarity`는 그룹화 변수이고, `data`는 `clarity`의 각 고유값에 대한 하나의 티블을 포함하는 리스트 열입니다.

```
by_clarity$data[[1]]
#> # A tibble: 741 × 9
#>   carat cut       color depth table price     x     y     z
#>   <dbl> <ord>     <ord> <dbl> <dbl> <int> <dbl> <dbl> <dbl>
#> 1  0.32 Premium   E      60.9    58   345  4.38  4.42  2.68
#> 2  1.17 Very Good J      60.2    61  2774  6.83  6.9   4.13
#> 3  1.01 Premium   F      61.8    60  2781  6.39  6.36  3.94
#> 4  1.01 Fair      E      64.5    58  2788  6.29  6.21  4.03
#> 5  0.96 Ideal     F      60.7    55  2801  6.37  6.41  3.88
#> 6  1.04 Premium   G      62.2    58  2801  6.46  6.41  4
#> # … with 735 more rows
```

이왕 이렇게 된 김에 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>를 사용하여 출력 파일의 이름을 제공하는 열을 만들어 보겠습니다.

```
by_clarity <- by_clarity |>
  mutate(path = str_glue("diamonds-{clarity}.csv"))

by_clarity
#> # A tibble: 8 × 3
#>   clarity               data path
#>   <ord>   <list<tibble[,9]>> <glue>
#> 1 I1               [741 × 9] diamonds-I1.csv
#> 2 SI2            [9,194 × 9] diamonds-SI2.csv
#> 3 SI1           [13,065 × 9] diamonds-SI1.csv
#> 4 VS2           [12,258 × 9] diamonds-VS2.csv
#> 5 VS1            [8,171 × 9] diamonds-VS1.csv
#> 6 VVS2           [5,066 × 9] diamonds-VVS2.csv
#> # … with 2 more rows
```

따라서 이러한 데이터 프레임을 수작업으로 저장하려고 한다면 다음과 같이 작성할 수 있습니다.

```
write_csv(by_clarity$data[[1]], by_clarity$path[[1]])
write_csv(by_clarity$data[[2]], by_clarity$path[[2]])
write_csv(by_clarity$data[[3]], by_clarity$path[[3]])
...
write_csv(by_clarity$by_clarity[[8]], by_clarity$path[[8]])
```

이것은 단지 하나가 아니라 변경되는 두 개의 인자가 있기 때문에 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>의 이전 사용법과는 조금 다릅니다. 이는 첫 번째와 두 번째 인자를 모두 변경하는 <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>map2()</code></a>라는 새로운 함수가 필요하다는 것을 의미합니다. 그리고 우리는 다시 한 번 출력에 관심이 없기 때문에 <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>map2()</code></a>가 아닌 <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>walk2()</code></a>를 원합니다. 그러면 다음을 얻을 수 있습니다.

```
walk2(by_clarity$data, by_clarity$path, write_csv)
```

## 플롯 저장하기 (Saving Plots)

많은 플롯을 생성하기 위해 동일한 기본 접근 방식을 취할 수 있습니다. 먼저 원하는 플롯을 그리는 함수를 만들어 보겠습니다.

```
carat_histogram <- function(df) {
  ggplot(df, aes(x = carat)) + geom_histogram(binwidth = 0.1)
}

carat_histogram(by_clarity$data[[1]])
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_26in01.png" alt="Histogram of carats of diamonds from the by_clarity dataset, ranging from 0 to 5 carats. The distribution is unimodal and right skewed with a peak around 1 carat." />
</figure>

이제 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>을 사용하여 많은 플롯<sup><a href="ch26.html#idm44771263913536" id="idm44771263913536-marker" data-type="noteref">7</a></sup>의 리스트와 그에 따른 최종 파일 경로를 만들 수 있습니다.

```
by_clarity <- by_clarity |>
  mutate(
    plot = map(data, carat_histogram),
    path = str_glue("clarity-{clarity}.png")
  )
```

그런 다음 <a href="https://ggplot2.tidyverse.org/reference/ggsave.html" class="orm:hideurl"><code>ggsave()</code></a>와 함께 <a href="https://purrr.tidyverse.org/reference/map2.html" class="orm:hideurl"><code>walk2()</code></a>를 사용하여 각 플롯을 저장합니다.

```
walk2(
  by_clarity$path,
  by_clarity$plot,
  \(path, plot) ggsave(path, plot, width = 6, height = 6)
)
```

이것은 다음의 줄임말입니다.

```
ggsave(by_clarity$path[[1]], by_clarity$plot[[1]], width = 6, height = 6)
ggsave(by_clarity$path[[2]], by_clarity$plot[[2]], width = 6, height = 6)
ggsave(by_clarity$path[[3]], by_clarity$plot[[3]], width = 6, height = 6)
...
ggsave(by_clarity$path[[8]], by_clarity$plot[[8]], width = 6, height = 6)
```

# 요약 (Summary)

이 장에서는 데이터 과학을 수행할 때 자주 발생하는 세 가지 문제(다중 열 조작, 다중 파일 읽기, 다중 출력 저장)를 해결하기 위해 명시적 반복을 사용하는 방법을 살펴보았습니다. 그러나 일반적으로 반복은 초능력입니다. 올바른 반복 기술을 알고 있다면 한 가지 문제를 해결하는 것에서 모든 문제를 해결하는 것으로 쉽게 이동할 수 있습니다. 이 장의 기술을 마스터하고 나면 *Advanced R*의 [“Functionals” 장](https://oreil.ly/VmXg4)을 읽고 [purrr 웹사이트](https://oreil.ly/f0HWP)를 참조하여 더 자세히 알아보는 것을 강력히 권장합니다.

다른 언어의 반복에 대해 잘 알고 있다면 `for` 루프에 대해 논의하지 않은 것에 놀랄 수도 있습니다. R의 데이터 분석 지향성이 반복 방식을 바꾸기 때문입니다. 대부분의 경우 기존의 관용구를 사용하여 각 열이나 각 그룹에 무언가를 할 수 있습니다. 그리고 불가능한 경우 리스트의 각 요소에 무언가를 수행하는 <a href="https://purrr.tidyverse.org/reference/map.html" class="orm:hideurl"><code>map()</code></a>과 같은 함수형 프로그래밍 도구를 종종 사용할 수 있습니다. 그러나 야생에서 가져온 코드에서는 `for` 루프를 볼 수 있으므로 다음 장에서 이에 대해 배울 것이며 여기서 몇 가지 중요한 기본 R 도구에 대해 논의할 것입니다.

<sup>[1](ch26.html#idm44771267612512-marker)</sup> `<-`를 사용하여 명시적으로 이름을 지정한 적이 없기 때문에 익명(Anonymous)입니다. 프로그래머들이 이에 사용하는 또 다른 용어는 *람다 함수(lambda function)*입니다.

<sup>[2](ch26.html#idm44771267610256-marker)</sup> 이전 코드에서는 `~ .x + 1`과 같은 구문을 볼 수 있습니다. 이것은 익명 함수를 작성하는 또 다른 방법이지만 tidyverse 함수 내에서만 작동하며 항상 변수 이름 `.x`를 사용합니다. 이제는 기본 구문인 `\(x) x + 1`을 권장합니다.

<sup>[3](ch26.html#idm44771267377264-marker)</sup> 현재는 열의 순서를 변경할 수 없지만 <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a> 또는 유사한 함수를 사용하여 사후에 재정렬할 수 있습니다.

<sup>[4](ch26.html#idm44771266547392-marker)</sup> 언젠가는 가능해질지도 모르지만 현재로서는 방법을 알 수 없습니다.

<sup>[5](ch26.html#idm44771266252816-marker)</sup> 형식이 동일한 CSV 파일 디렉터리가 있는 경우 <a href="ch07.html#sec-readr-directory" data-type="xref">“다중 파일에서 데이터 읽기”</a>의 기술을 사용할 수 있습니다.

<sup>[6](ch26.html#idm44771264920592-marker)</sup> 우리는 그것이 어떻게 작동하는지 설명하지 않을 것이지만 사용된 함수의 문서를 살펴보면 알아낼 수 있을 것입니다.

<sup>[7](ch26.html#idm44771263913536-marker)</sup> 조잡한 애니메이션을 얻기 위해 `by_clarity$plot`을 출력할 수 있습니다. `plots`의 각 요소에 대해 하나의 플롯을 얻을 수 있습니다.
