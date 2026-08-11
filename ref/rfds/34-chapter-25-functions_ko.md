# 25장. 함수 (Functions)

# 소개

데이터 과학자로서 역량을 향상시키는 가장 좋은 방법 중 하나는 함수를 작성하는 것입니다. 함수를 사용하면 복사 및 붙여넣기보다 더 강력하고 일반적인 방식으로 일반적인 작업을 자동화할 수 있습니다. 함수 작성은 복사 및 붙여넣기를 사용하는 것에 비해 세 가지 큰 장점이 있습니다.

- 코드를 이해하기 쉽게 만드는 직관적인(evocative) 이름을 함수에 부여할 수 있습니다.

- 요구 사항이 변경될 때, 여러 곳이 아닌 한 곳에서만 코드를 업데이트하면 됩니다.

- 복사하여 붙여넣을 때 부수적으로 발생하는 실수의 가능성을 제거합니다(한 곳에서 변수 이름을 업데이트하고 다른 곳에서는 업데이트하지 않는 경우).

- 프로젝트 간에 작업물을 재사용하기 쉬워져 시간이 지남에 따라 생산성이 향상됩니다.

좋은 경험 법칙은 코드 블록을 두 번 이상 복사하여 붙여넣은 경우(즉, 동일한 코드의 복사본이 세 개가 된 경우) 함수 작성을 고려하는 것입니다. 이 장에서는 세 가지 유용한 함수 유형에 대해 배웁니다.

- 벡터 함수(Vector functions)는 하나 이상의 벡터를 입력으로 받아 벡터를 출력으로 반환합니다.
- 데이터 프레임 함수(Data frame functions)는 데이터 프레임을 입력으로 받아 데이터 프레임을 출력으로 반환합니다.
- 플롯 함수(Plot functions)는 데이터 프레임을 입력으로 받아 플롯을 출력으로 반환합니다.

이러한 각 섹션에는 본 패턴을 일반화하는 데 도움이 되는 많은 예제가 포함되어 있습니다. 이러한 예제는 Twitter 사용자들의 도움 없이는 불가능했으며, 댓글에 있는 링크를 따라가서 원본 영감을 확인해 보시기를 권장합니다. 더 많은 함수를 보려면 [일반 함수](https://oreil.ly/Ymcmk) 및 [플로팅 함수](https://oreil.ly/mXy2q)에 대한 영감을 준 원본 트윗을 읽어보는 것도 좋습니다.

## 사전 준비

tidyverse의 다양한 함수들을 포괄적으로 다룰 것입니다. 또한 함수와 함께 사용할 친숙한 데이터 소스로 nycflights13을 사용할 것입니다.

```
library(tidyverse)
library(nycflights13)
```

# 벡터 함수 (Vector Functions)

우리는 하나 이상의 벡터를 취하여 벡터 결과를 반환하는 함수인 벡터 함수로 시작할 것입니다. 예를 들어, 이 코드를 살펴보겠습니다. 이것은 무엇을 합니까?

```
df <- tibble(
  a = rnorm(5),
  b = rnorm(5),
  c = rnorm(5),
  d = rnorm(5),
)

df |> mutate(
  a = (a - min(a, na.rm = TRUE)) /
    (max(a, na.rm = TRUE) - min(a, na.rm = TRUE)),
  b = (b - min(b, na.rm = TRUE)) /
    (max(b, na.rm = TRUE) - min(a, na.rm = TRUE)),
  c = (c - min(c, na.rm = TRUE)) /
    (max(c, na.rm = TRUE) - min(c, na.rm = TRUE)),
  d = (d - min(d, na.rm = TRUE)) /
    (max(d, na.rm = TRUE) - min(d, na.rm = TRUE)),
)
#> # A tibble: 5 × 4
#>       a     b     c     d
#>   <dbl> <dbl> <dbl> <dbl>
#> 1 0.339  2.59 0.291 0
#> 2 0.880  0    0.611 0.557
#> 3 0      1.37 1     0.752
#> 4 0.795  1.37 0     1
#> 5 1      1.34 0.580 0.394
```

이 코드가 각 열을 0에서 1 사이의 범위를 갖도록 크기를 조정(rescale)한다는 것을 파악할 수 있을 것입니다. 하지만 실수를 발견했나요? Hadley가 이 코드를 작성할 때, 그는 복사하고 붙여넣을 때 `a`를 `b`로 변경하는 것을 잊어버리는 오류를 범했습니다. 이러한 유형의 실수를 방지하는 것이 함수 작성법을 배워야 하는 한 가지 좋은 이유입니다.

## 함수 작성하기 (Writing a Function)

함수를 작성하려면 먼저 반복되는 코드를 분석하여 어느 부분이 일정하고 어느 부분이 변하는지 파악해야 합니다. 앞의 코드를 가져와서 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 밖으로 빼내면, 각 반복이 이제 한 줄이 되기 때문에 패턴을 보기가 조금 더 쉽습니다.

```p
(a - min(a, na.rm = TRUE)) / (max(a, na.rm = TRUE) - min(a, na.rm = TRUE))
(b - min(b, na.rm = TRUE)) / (max(b, na.rm = TRUE) - min(b, na.rm = TRUE))
(c - min(c, na.rm = TRUE)) / (max(c, na.rm = TRUE) - min(c, na.rm = TRUE))
(d - min(d, na.rm = TRUE)) / (max(d, na.rm = TRUE) - min(d, na.rm = TRUE))
```

이것을 좀 더 명확하게 하기 위해, 변하는 부분을 `█`로 바꿀 수 있습니다.

```p
(█ - min(█, na.rm = TRUE)) / (max(█, na.rm = TRUE) - min(█, na.rm = TRUE))
```

이것을 함수로 바꾸려면 세 가지가 필요합니다.

- _이름(name)_. 이 함수는 벡터를 0과 1 사이로 재조정하므로 여기서는 `rescale01`을 사용할 것입니다.

- _인자(arguments)_. 인자는 호출마다 변하는 것들이며, 우리의 분석에 따르면 단 하나뿐입니다. 숫자형 벡터의 관례적인 이름이므로 이를 `x`라고 부를 것입니다.

- _본문(body)_. 본문은 모든 호출에서 반복되는 코드입니다.

그런 다음 템플릿에 따라 함수를 생성합니다.

```
name <- function(arguments) {
  body
}
```

이 경우 다음과 같이 됩니다.

```
rescale01 <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
```

이 시점에서 논리를 올바르게 캡처했는지 확인하기 위해 몇 가지 간단한 입력으로 테스트할 수 있습니다.

```
rescale01(c(-10, 0, 10))
#> [1] 0.0 0.5 1.0
rescale01(c(1, 2, 3, NA, 5))
#> [1] 0.00 0.25 0.50   NA 1.00
```

그런 다음 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에 대한 호출을 다음과 같이 다시 작성할 수 있습니다.

```
df |> mutate(
  a = rescale01(a),
  b = rescale01(b),
  c = rescale01(c),
  d = rescale01(d),
)
#> # A tibble: 5 × 4
#>       a     b     c     d
#>   <dbl> <dbl> <dbl> <dbl>
#> 1 0.339 1     0.291 0
#> 2 0.880 0     0.611 0.557
#> 3 0     0.530 1     0.752
#> 4 0.795 0.531 0     1
#> 5 1     0.518 0.580 0.394
```

(<a href="ch26.html#chp-iteration" data-type="xref">26장</a>에서는 중복을 더욱 줄이기 위해 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>를 사용하여 `df |> mutate(across(a:d, rescale01))`만 필요하게 하는 방법을 배울 것입니다.)

## 함수 개선하기 (Improving Our Function)

`rescale01()` 함수가 불필요한 작업을 수행한다는 것을 눈치챘을 수 있습니다. <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>min()</code></a>을 두 번, <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>max()</code></a>를 한 번 계산하는 대신, <a href="https://rdrr.io/r/base/range.html" class="orm:hideurl"><code>range()</code></a>를 사용하여 한 번에 최솟값과 최댓값을 모두 계산할 수 있습니다.

```
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE)
  (x - rng[1]) / (rng[2] - rng[1])
}
```

또는 무한대 값이 포함된 벡터에서 이 함수를 시도해 볼 수도 있습니다.

```
x <- c(1:10, Inf)
rescale01(x)
#>  [1]   0   0   0   0   0   0   0   0   0   0 NaN
```

그 결과는 특별히 유용하지 않으므로, <a href="https://rdrr.io/r/base/range.html" class="orm:hideurl"><code>range()</code></a>에 무한대 값을 무시하도록 요청할 수 있습니다.

```
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE, finite = TRUE)
  (x - rng[1]) / (rng[2] - rng[1])
}

rescale01(x)
#>  [1] 0.0000000 0.1111111 0.2222222 0.3333333 0.4444444 0.5555556 0.6666667
#>  [8] 0.7777778 0.8888889 1.0000000       Inf
```

이러한 변경 사항은 함수의 중요한 이점을 보여줍니다. 반복되는 코드를 함수로 옮겼기 때문에 한 곳에서만 변경하면 됩니다.

## Mutate 함수 (Mutate Functions)

이제 함수의 기본 개념을 이해했으므로 여러 가지 예제를 살펴보겠습니다. 먼저 입력과 동일한 길이의 출력을 반환하기 때문에 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> 내에서 잘 작동하는 함수인 "mutate" 함수를 살펴보겠습니다.

`rescale01()`의 간단한 변형으로 시작해 보겠습니다. 벡터를 평균이 0이고 표준 편차가 1이 되도록 재조정하는 Z-점수를 계산하고 싶을 수 있습니다.

```
z_score <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}
```

또는 간단한 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a>을 래핑하여 유용한 이름을 지정하고 싶을 수도 있습니다. 예를 들어, 이 `clamp()` 함수는 벡터의 모든 값이 최솟값과 최댓값 사이에 있는지 확인합니다.

```
clamp <- function(x, min, max) {
  case_when(
    x < min ~ min,
    x > max ~ max,
    .default = x
  )
}

clamp(1:10, min = 3, max = 7)
#>  [1] 3 3 3 4 5 6 7 7 7 7
```

물론 함수가 반드시 숫자형 변수에서만 작동해야 하는 것은 아닙니다. 반복적인 문자열 조작을 수행하고 싶을 수 있습니다. 첫 번째 문자를 대문자로 만들어야 할 수도 있습니다.

```
first_upper <- function(x) {
  str_sub(x, 1, 1) <- str_to_upper(str_sub(x, 1, 1))
  x
}

first_upper("hello")
#> [1] "Hello"
```

또는 문자열을 숫자로 변환하기 전에 문자열에서 퍼센트 기호, 쉼표 및 달러 기호를 제거하고 싶을 수도 있습니다.

```
# https://twitter.com/NVlabormarket/status/1571939851922198530
clean_number <- function(x) {
  is_pct <- str_detect(x, "%")
  num <- x |>
    str_remove_all("%") |>
    str_remove_all(",") |>
    str_remove_all(fixed("$")) |>
    as.numeric(x)
  if_else(is_pct, num / 100, num)
}

clean_number("$12,300")
#> [1] 12300
clean_number("45%")
#> [1] 0.45
```

때로는 함수가 하나의 데이터 분석 단계에 고도로 특화되어 있을 수 있습니다. 예를 들어, 결측값을 997, 998 또는 999로 기록하는 변수가 많은 경우 이를 `NA`로 대체하는 함수를 작성하고 싶을 수 있습니다.

```
fix_na <- function(x) {
  if_else(x %in% c(997, 998, 999), NA, x)
}
```

단일 벡터를 취하는 예제에 집중한 이유는 그것이 가장 일반적이라고 생각하기 때문입니다. 하지만 함수가 여러 벡터 입력을 받을 수 없다는 이유는 없습니다.

## 요약 함수 (Summary Functions)

벡터 함수의 또 다른 중요한 제품군은 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>에서 사용하기 위해 단일 값을 반환하는 요약 함수입니다. 때로는 단순히 기본 인자를 한두 개 설정하는 문제일 수 있습니다.

```
commas <- function(x) {
  str_flatten(x, collapse = ", ", last = " and ")
}

commas(c("cat", "dog", "pigeon"))
#> [1] "cat, dog and pigeon"
```

또는 표준 편차를 평균으로 나누는 변동 계수(coefficient of variation)와 같은 간단한 계산을 래핑할 수도 있습니다.

```
cv <- function(x, na.rm = FALSE) {
  sd(x, na.rm = na.rm) / mean(x, na.rm = na.rm)
}

cv(runif(100, min = 0, max = 50))
#> [1] 0.5196276
cv(runif(100, min = 0, max = 500))
#> [1] 0.5652554
```

또는 기억하기 쉬운 이름을 지정하여 일반적인 패턴을 더 쉽게 기억하고 싶을 수도 있습니다.

```
# https://twitter.com/gbganalyst/status/1571619641390252033
n_missing <- function(x) {
  sum(is.na(x))
}
```

여러 벡터 입력을 가진 함수를 작성할 수도 있습니다. 예를 들어 모델 예측을 실제 값과 비교하는 데 도움이 되도록 평균 절대 예측 오차(mean absolute prediction error)를 계산하고 싶을 수 있습니다.

```
# https://twitter.com/neilgcurrie/status/1571607727255834625
mape <- function(actual, predicted) {
  sum(abs((actual - predicted) / actual)) / length(actual)
}
```

# RStudio

함수를 작성하기 시작하면 매우 유용한 두 가지 RStudio 단축키가 있습니다.

- 작성한 함수의 정의를 찾으려면 함수 이름에 커서를 놓고 F2를 누르세요.

- 함수로 빠르게 이동하려면 Ctrl+.를 눌러 퍼지 파일 및 함수 찾기(fuzzy file and function finder)를 열고 함수 이름의 처음 몇 글자를 입력하세요. 파일, Quarto 섹션 등으로 이동할 수도 있어 편리한 내비게이션 도구가 됩니다.

## 연습문제 (Exercises)

1.  다음 코드 스니펫을 함수로 바꾸는 연습을 하세요. 각 함수가 무엇을 하는지 생각해 보세요. 그것을 무엇이라고 부르겠습니까? 인자가 몇 개 필요합니까?

    ```
    mean(is.na(x))
    mean(is.na(y))
    mean(is.na(z))

    x / sum(x, na.rm = TRUE)
    y / sum(y, na.rm = TRUE)
    z / sum(z, na.rm = TRUE)

    round(x / sum(x, na.rm = TRUE) * 100, 1)
    round(y / sum(y, na.rm = TRUE) * 100, 1)
    round(z / sum(z, na.rm = TRUE) * 100, 1)
    ```

2.  `rescale01()`의 두 번째 변형에서는 무한대 값이 변경되지 않은 상태로 유지됩니다. `-Inf`가 0에 매핑되고 `Inf`가 1에 매핑되도록 `rescale01()`을 다시 작성할 수 있습니까?

3.  생년월일 벡터가 주어지면 나이를 연 단위로 계산하는 함수를 작성하세요.

4.  숫자형 벡터의 분산(variance)과 왜도(skewness)를 계산하는 자체 함수를 작성하세요. 위키백과 등에서 정의를 찾아볼 수 있습니다.

5.  길이가 같은 두 벡터를 받아 두 벡터 모두에 `NA`가 있는 위치의 수를 반환하는 요약 함수인 `both_na()`를 작성하세요.

6.  문서를 읽고 다음 함수가 무엇을 하는지 파악하세요. 너무 짧음에도 불구하고 왜 유용할까요?

    ```
    is_directory <- function(x) {
      file.info(x)$isdir
    }
    is_readable <- function(x) {
      file.access(x, 4) == 0
    }
    ```

# 데이터 프레임 함수 (Data Frame Functions)

벡터 함수는 dplyr 동사(verb) 내에서 반복되는 코드를 꺼내는 데 유용합니다. 하지만 특히 대규모 파이프라인 내에서는 동사 자체를 반복하는 경우도 많을 것입니다. 여러 동사를 여러 번 복사하여 붙여넣고 있다는 것을 알게 되면 데이터 프레임 함수 작성을 고려해 볼 수 있습니다. 데이터 프레임 함수는 dplyr 동사와 같이 작동합니다. 첫 번째 인자로 데이터 프레임을 받고 무엇을 할지 알려주는 몇 가지 추가 인자를 받아 데이터 프레임이나 벡터를 반환합니다.

dplyr 동사를 사용하는 함수를 작성할 수 있도록 하려면, 먼저 간접 참조(indirection) 문제와 `{{ }}`를 사용하는 포용(embracing)을 통해 이 문제를 어떻게 극복할 수 있는지 소개하겠습니다. 그런 다음 이를 활용하여 무엇을 할 수 있는지 보여주는 여러 예제를 살펴보겠습니다.

## 간접 참조와 Tidy 평가 (Indirection and Tidy Evaluation)

dplyr 동사를 사용하는 함수를 작성하기 시작하면 간접 참조 문제에 빠르게 직면하게 됩니다. 간단한 함수인 `grouped_mean()`으로 문제를 설명해 보겠습니다. 이 함수의 목표는 `group_var`로 그룹화된 `mean_var`의 평균을 계산하는 것입니다.

```
grouped_mean <- function(df, group_var, mean_var) {
  df |>
    group_by(group_var) |>
    summarize(mean(mean_var))
}
```

사용하려고 하면 오류가 발생합니다.

```
diamonds |> grouped_mean(cut, carat)
#> Error in `group_by()`:
#> ! Must group by variables found in `.data`.
#> ✖ Column `group_var` is not found.
```

문제를 조금 더 명확하게 하기 위해 가상의 데이터 프레임을 사용할 수 있습니다.

```
df <- tibble(
  mean_var = 1,
  group_var = "g",
  group = 1,
  x = 10,
  y = 100
)

df |> grouped_mean(group, x)
#> # A tibble: 1 × 2
#>   group_var `mean(mean_var)`
#>   <chr>                <dbl>
#> 1 g                        1
df |> grouped_mean(group, y)
#> # A tibble: 1 × 2
#>   group_var `mean(mean_var)`
#>   <chr>                <dbl>
#> 1 g                        1
```

`grouped_mean()`을 어떻게 호출하든 항상 `df |> group_by(group) |> summarize(mean(x))` 또는 `df |> group_by(group) |> summarize(mean(y))` 대신 `df |> group_by(group_var) |> summarize(mean(mean_var))`를 실행합니다. 이것은 간접 참조의 문제이며, dplyr이 특별한 처리 없이 데이터 프레임 내 변수 이름을 참조할 수 있도록 *tidy 평가(tidy evaluation)*를 사용하기 때문에 발생합니다.

Tidy 평가는 변수가 어느 데이터 프레임에서 왔는지 명시할 필요가 없으므로 문맥에서 명확하기 때문에 데이터 분석을 매우 간결하게 만들어 주므로 95%의 상황에서는 훌륭합니다. Tidy 평가의 단점은 반복되는 tidyverse 코드를 함수로 감싸고 싶을 때 발생합니다. 여기서 우리는 `group_mean()`과 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>에게 `group_var`와 `mean_var`를 변수의 이름으로 처리하지 않고, 실제로 사용하려는 변수를 찾기 위해 그 안을 들여다보도록 지시할 방법이 필요합니다.

Tidy 평가에는 이 문제에 대한 해결책인 *포용(embracing)*이 포함되어 있습니다. 변수를 포용한다는 것은 변수를 중괄호로 감싸는 것을 의미하므로, 예를 들어 `var`는 `{{ var }}`가 됩니다. 변수를 포용하면 dplyr에게 리터럴 변수 이름으로서의 인자가 아니라 인자 내부에 저장된 값을 사용하도록 지시합니다. 무슨 일이 일어나고 있는지 기억하는 한 가지 방법은 `{{ }}`를 터널을 내려다보는 것으로 생각하는 것입니다. `{{ var }}`는 dplyr 함수가 `var`라는 변수를 찾는 대신 `var` 안을 들여다보게 합니다.

따라서 `grouped_mean()`이 작동하게 하려면 `group_var`와 `mean_var`를 `{{ }}`로 둘러싸야 합니다.

```
grouped_mean <- function(df, group_var, mean_var) {
  df |>
    group_by({{ group_var }}) |>
    summarize(mean({{ mean_var }}))
}

df |> grouped_mean(group, x)
#> # A tibble: 1 × 2
#>   group `mean(x)`
#>   <dbl>     <dbl>
#> 1     1        10
```

성공입니다!

## 언제 포용(Embrace)해야 하는가?

데이터 프레임 함수를 작성할 때 주요 과제는 어떤 인자를 포용해야 하는지 파악하는 것입니다. 다행히 문서에서 찾아볼 수 있기 때문에 쉽습니다. 문서에서 tidy 평가의 두 가지 가장 일반적인 하위 유형에 해당하는 다음 두 가지 용어를 찾을 수 있습니다.

Data masking (데이터 마스킹)  
변수를 사용하여 계산하는 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>, <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 같은 함수에서 사용됩니다.

Tidy selection (Tidy 선택)  
변수를 선택하는 <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename()</code></a>과 같은 함수에 사용됩니다.

어떤 인자가 tidy 평가를 사용하는지에 대한 당신의 직감은 많은 일반적인 함수에서 잘 맞을 것입니다. 계산을 할 수 있는지(`x + 1`) 또는 선택할 수 있는지(`a:x`)를 생각해 보세요.

다음 섹션에서는 포용(embracing)을 이해한 후 작성할 수 있는 종류의 유용한 함수들을 살펴보겠습니다.

## 일반적인 사용 사례 (Common Use Cases)

초기 데이터 탐색을 수행할 때 동일한 요약(summary) 세트를 일반적으로 수행하는 경우 헬퍼 함수로 래핑하는 것을 고려할 수 있습니다.

```
summary6 <- function(data, var) {
  data |> summarize(
    min = min({{ var }}, na.rm = TRUE),
    mean = mean({{ var }}, na.rm = TRUE),
    median = median({{ var }}, na.rm = TRUE),
    max = max({{ var }}, na.rm = TRUE),
    n = n(),
    n_miss = sum(is.na({{ var }})),
    .groups = "drop"
  )
}

diamonds |> summary6(carat)
#> # A tibble: 1 × 6
#>     min  mean median   max     n n_miss
#>   <dbl> <dbl>  <dbl> <dbl> <int>  <int>
#> 1   0.2 0.798    0.7  5.01 53940      0
```

(<a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>를 헬퍼로 래핑할 때는 메시지를 피하고 데이터를 그룹 해제된 상태로 남겨두기 위해 `.groups = "drop"`을 설정하는 것이 좋은 연습이라고 생각합니다.)

이 함수의 장점은 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>를 래핑하기 때문에 그룹화된 데이터에서 사용할 수 있다는 것입니다.

```
diamonds |>
  group_by(cut) |>
  summary6(carat)
#> # A tibble: 5 × 7
#>   cut         min  mean median   max     n n_miss
#>   <ord>     <dbl> <dbl>  <dbl> <dbl> <int>  <int>
#> 1 Fair       0.22 1.05    1     5.01  1610      0
#> 2 Good       0.23 0.849   0.82  3.01  4906      0
#> 3 Very Good  0.2  0.806   0.71  4    12082      0
#> 4 Premium    0.2  0.892   0.86  4.01 13791      0
#> 5 Ideal      0.2  0.703   0.54  3.5  21551      0
```

또한 summarize에 대한 인자가 데이터 마스킹이므로 `summary6()`에 대한 `var` 인자도 데이터 마스킹입니다. 즉, 계산된 변수도 요약할 수 있습니다.

```
diamonds |>
  group_by(cut) |>
  summary6(log10(carat))
#> # A tibble: 5 × 7
#>   cut          min    mean  median   max     n n_miss
#>   <ord>      <dbl>   <dbl>   <dbl> <dbl> <int>  <int>
#> 1 Fair      -0.658 -0.0273  0      0.700  1610      0
#> 2 Good      -0.638 -0.133  -0.0862 0.479  4906      0
#> 3 Very Good -0.699 -0.164  -0.149  0.602 12082      0
#> 4 Premium   -0.699 -0.125  -0.0655 0.603 13791      0
#> 5 Ideal     -0.699 -0.225  -0.268  0.544 21551      0
```

여러 변수를 요약하려면 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>를 사용하는 방법을 배울 수 있는 <a href="ch26.html#sec-across" data-type="xref">“다중 열 수정”</a>을 기다려야 합니다.

또 다른 인기 있는 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a> 헬퍼 함수는 비율도 계산하는 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>의 버전입니다.

```
# https://twitter.com/Diabb6/status/1571635146658402309
count_prop <- function(df, var, sort = FALSE) {
  df |>
    count({{ var }}, sort = sort) |>
    mutate(prop = n / sum(n))
}

diamonds |> count_prop(clarity)
#> # A tibble: 8 × 3
#>   clarity     n   prop
#>   <ord>   <int>  <dbl>
#> 1 I1        741 0.0137
#> 2 SI2      9194 0.170
#> 3 SI1     13065 0.242
#> 4 VS2     12258 0.227
#> 5 VS1      8171 0.151
#> 6 VVS2     5066 0.0939
#> # … with 2 more rows
```

이 함수에는 세 개의 인자 `df`, `var` 및 `sort`가 있습니다. `var`는 모든 변수에 대해 데이터 마스킹을 사용하는 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>에 전달되기 때문에 포용(embrace)되어야 합니다. 사용자가 자신의 값을 제공하지 않으면 `FALSE`로 기본 설정되도록 `sort`에 대해 기본값을 사용한다는 점에 유의하세요.

또는 데이터의 하위 집합에 대해 변수의 정렬된 고유값을 찾고 싶을 수 있습니다. 필터링을 수행하기 위해 변수와 값을 제공하는 대신, 사용자가 조건(condition)을 제공할 수 있도록 허용하겠습니다.

```
unique_where <- function(df, condition, var) {
  df |>
    filter({{ condition }}) |>
    distinct({{ var }}) |>
    arrange({{ var }})
}

# Find all the destinations in December
flights |> unique_where(month == 12, dest)
#> # A tibble: 96 × 1
#>   dest
#>   <chr>
#> 1 ABQ
#> 2 ALB
#> 3 ATL
#> 4 AUS
#> 5 AVL
#> 6 BDL
#> # … with 90 more rows
```

여기서 우리는 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>에 전달되기 때문에 `condition`을 포용하고, <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>에 전달되기 때문에 `var`를 포용합니다.

이러한 모든 예제는 첫 번째 인자로 데이터 프레임을 받도록 만들어졌지만, 동일한 데이터로 반복적으로 작업하는 경우 하드코딩하는 것이 합리적일 수 있습니다. 예를 들어, 다음 함수는 항상 `flights` 데이터 세트와 함께 작동하며 행을 식별할 수 있는 복합 기본 키(compound primary key)를 형성하기 때문에 항상 `time_hour`, `carrier` 및 `flight`를 선택합니다.

```
subset_flights <- function(rows, cols) {
  flights |>
    filter({{ rows }}) |>
    select(time_hour, carrier, flight, {{ cols }})
}
```

## 데이터 마스킹(Data Masking) 대 Tidy 선택(Tidy Selection)

때로는 데이터 마스킹을 사용하는 함수 내부에서 변수를 선택하고 싶을 때가 있습니다. 예를 들어 행의 결측값 관측치 수를 계산하는 `count_missing()` 메서드를 작성한다고 가정해 보겠습니다. 다음과 같이 작성해 볼 수 있습니다.

```
count_missing <- function(df, group_vars, x_var) {
  df |>
    group_by({{ group_vars }}) |>
    summarize(
      n_miss = sum(is.na({{ x_var }})),
      .groups = "drop"
    )
}

flights |>
  count_missing(c(year, month, day), dep_time)
#> Error in `group_by()`:
#> ℹ In argument: `c(year, month, day)`.
#> Caused by error:
#> ! `c(year, month, day)` must be size 336776 or 1, not 1010328.
```

<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>는 tidy 선택이 아닌 데이터 마스킹을 사용하기 때문에 이것은 작동하지 않습니다. 데이터 마스킹 함수 내부에서 tidy 선택을 사용할 수 있게 해주는 편리한 <a href="https://dplyr.tidyverse.org/reference/pick.html" class="orm:hideurl"><code>pick()</code></a> 함수를 사용하여 이 문제를 해결할 수 있습니다.

```
count_missing <- function(df, group_vars, x_var) {
  df |>
    group_by(pick({{ group_vars }})) |>
    summarize(
      n_miss = sum(is.na({{ x_var }})),
      .groups = "drop"
  )
}

flights |>
  count_missing(c(year, month, day), dep_time)
#> # A tibble: 365 × 4
#>    year month   day n_miss
#>   <int> <int> <int>  <int>
#> 1  2013     1     1      4
#> 2  2013     1     2      8
#> 3  2013     1     3     10
#> 4  2013     1     4      6
#> 5  2013     1     5      3
#> 6  2013     1     6      1
#> # … with 359 more rows
```

<a href="https://dplyr.tidyverse.org/reference/pick.html" class="orm:hideurl"><code>pick()</code></a>의 또 다른 편리한 용도는 2D 카운트 테이블을 만드는 것입니다. 여기서 우리는 `rows`와 `columns`의 모든 변수를 사용하여 카운트한 다음 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>를 사용하여 카운트를 그리드로 재배열합니다.

```
# https://twitter.com/pollicipes/status/1571606508944719876
count_wide <- function(data, rows, cols) {
  data |>
    count(pick(c({{ rows }}, {{ cols }}))) |>
    pivot_wider(
      names_from = {{ cols }},
      values_from = n,
      names_sort = TRUE,
      values_fill = 0
    )
}

diamonds |> count_wide(c(clarity, color), cut)
#> # A tibble: 56 × 7
#>   clarity color  Fair  Good `Very Good` Premium Ideal
#>   <ord>   <ord> <int> <int>       <int>   <int> <int>
#> 1 I1      D         4     8           5      12    13
#> 2 I1      E         9    23          22      30    18
#> 3 I1      F        35    19          13      34    42
#> 4 I1      G        53    19          16      46    16
#> 5 I1      H        52    14          12      46    38
#> 6 I1      I        34     9           8      24    17
#> # … with 50 more rows
```

우리의 예제는 주로 dplyr에 초점을 맞추었지만, tidy 평가는 tidyr의 기초가 되기도 하며, <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a> 문서를 보면 `names_from`이 tidy 선택을 사용한다는 것을 알 수 있습니다.

## 연습문제 (Exercises)

1.  nycflights13의 데이터 세트를 사용하여 다음을 수행하는 함수를 작성하세요.
    1.  취소된(즉, `is.na(arr_time)`) 모든 항공편 또는 1시간 이상 지연된 항공편을 찾습니다.

        ```
        flights |> filter_severe()
        ```

    2.  취소된 항공편의 수와 1시간 이상 지연된 항공편의 수를 계산합니다.

        ```
        flights |> group_by(dest) |> summarize_severe()
        ```

    3.  취소되었거나 사용자가 제공한 시간 이상 지연된 모든 항공편을 찾습니다.

        ```
        flights |> filter_severe(hours = 2)
        ```

    4.  사용자가 제공한 변수의 최솟값, 평균 및 최댓값을 계산하기 위해 날씨를 요약합니다.

        ```
        weather |> summarize_weather(temp)
        ```

    5.  시계 시간을 사용하는 사용자가 제공한 변수(`dep_time`, `arr_time` 등)를 십진수 시간(즉, 시간 + \[분 / 60\])으로 변환합니다.

        ```
        weather |> standardize_time(sched_dep_time)
        ```

2.  다음 각 함수에 대해 tidy 평가를 사용하는 모든 인자를 나열하고 데이터 마스킹을 사용하는지 tidy 선택을 사용하는지 설명하세요. <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>, <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>, <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>, <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename_with()</code></a>, <a href="https://dplyr.tidyverse.org/reference/slice.html" class="orm:hideurl"><code>slice_min()</code></a>, <a href="https://dplyr.tidyverse.org/reference/slice.html" class="orm:hideurl"><code>slice_sample()</code></a>.

3.  계산할 변수를 몇 개든 제공할 수 있도록 다음 함수를 일반화하세요.

    ```
    count_prop <- function(df, var, sort = FALSE) {
      df |>
        count({{ var }}, sort = sort) |>
        mutate(prop = n / sum(n))
    }
    ```

# 플롯 함수 (Plot Functions)

데이터 프레임 대신 플롯을 반환하고 싶을 수 있습니다. 다행히 <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a>는 데이터 마스킹 함수이기 때문에 ggplot2에서 동일한 기술을 사용할 수 있습니다. 예를 들어, 많은 히스토그램을 만든다고 가정해 보겠습니다.

```
diamonds |>
  ggplot(aes(x = carat)) +
  geom_histogram(binwidth = 0.1)

diamonds |>
  ggplot(aes(x = carat)) +
  geom_histogram(binwidth = 0.05)
```

이것을 히스토그램 함수로 감쌀 수 있다면 좋지 않을까요? <a href="https://ggplot2.tidyverse.org/reference/aes.html" class="orm:hideurl"><code>aes()</code></a>가 데이터 마스킹 함수이고 포용해야 한다는 것을 알게 되면 이것은 식은 죽 먹기입니다.

```
histogram <- function(df, var, binwidth = NULL) {
  df |>
    ggplot(aes(x = {{ var }})) +
    geom_histogram(binwidth = binwidth)
}

diamonds |> histogram(carat, 0.1)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_25in01.png" alt="A histogram of carats of diamonds, ranging from 0 to 5, showing a unimodal, right-skewed distribution with a peak between 0 to 1 carats." />
</figure>

`histogram()`은 ggplot2 플롯을 반환하므로 원하는 경우 여전히 구성 요소를 추가할 수 있습니다. `|>`에서 `+`로 전환하는 것만 기억하세요.

```
diamonds |>
  histogram(carat, 0.1) +
  labs(x = "Size (in carats)", y = "Number of diamonds")
```

## 더 많은 변수 (More Variables)

혼합물에 더 많은 변수를 추가하는 것은 간단합니다. 예를 들어, 부드러운 선(smooth line)과 직선을 겹쳐 데이터 세트가 선형인지 눈대중으로 쉽게 확인하는 방법을 원할 수 있습니다.

```
# https://twitter.com/tyler_js_smith/status/1574377116988104704
linearity_check <- function(df, x, y) {
  df |>
    ggplot(aes(x = {{ x }}, y = {{ y }})) +
    geom_point() +
    geom_smooth(method = "loess", formula = y ~ x, color = "red", se = FALSE) +
    geom_smooth(method = "lm", formula = y ~ x, color = "blue", se = FALSE)
}

starwars |>
  filter(mass < 1000) |>
  linearity_check(mass, height)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_25in02.png" alt="Scatterplot of height vs. mass of Star Wars characters showing a positive relationship. A smooth curve of the relationship is plotted in red, and the best fit line is plotted in blue." />
</figure>

또는 오버플로팅(overplotting)이 문제가 되는 매우 큰 데이터 세트의 경우 색상이 지정된 산점도에 대한 대안을 원할 수 있습니다.

```
# https://twitter.com/ppaxisa/status/1574398423175921665
hex_plot <- function(df, x, y, z, bins = 20, fun = "mean") {
  df |>
    ggplot(aes(x = {{ x }}, y = {{ y }}, z = {{ z }})) +
    stat_summary_hex(
      aes(color = after_scale(fill)), # make border same color as fill
      bins = bins,
      fun = fun,
    )
}

diamonds |> hex_plot(carat, price, depth)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_25in03.png" alt="Hex plot of price vs. carat of diamonds showing a positive relationship. There are more diamonds that are less than 2 carats than more than 2 carats." />
</figure>

## 다른 Tidyverse 패키지와 결합하기 (Combining with Other Tidyverse Packages)

가장 유용한 헬퍼 중 일부는 데이터 조작의 약간과 ggplot2를 결합합니다. 예를 들어, <a href="https://forcats.tidyverse.org/reference/fct_inorder.html" class="orm:hideurl"><code>fct_infreq()</code></a>를 사용하여 빈도순으로 막대를 자동으로 정렬하는 수직 막대 차트를 그리고 싶을 수 있습니다. 막대 차트가 수직이므로 일반적인 순서를 반대로 하여 맨 위에 가장 높은 값이 오도록 해야 합니다.

```
sorted_bars <- function(df, var) {
  df |>
    mutate({{ var }} := fct_rev(fct_infreq({{ var }})))  |>
    ggplot(aes(y = {{ var }})) +
    geom_bar()
}

diamonds |> sorted_bars(clarity)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_25in04.png" alt="Bar plot of clarify of diamonds, where clarity is on the y-axis and counts are on the x-axis, and the bars are ordered in order of frequency: SI1, VS2, SI2, VS1, VVS2, VVS1, IF, I1." />
</figure>

사용자가 제공한 데이터를 기반으로 변수 이름을 생성하고 있기 때문에 여기서 새로운 연산자인 `:=`을 사용해야 합니다. 변수 이름은 `=`의 왼쪽에 있지만 R의 구문은 단일 리터럴 이름 외에는 `=`의 왼쪽에 아무것도 허용하지 않습니다. 이 문제를 해결하기 위해 특별한 연산자인 `:=`을 사용하며, tidy 평가는 이를 `=`과 동일한 방식으로 처리합니다.

또는 데이터의 하위 집합에 대해서만 막대 플롯을 쉽게 그릴 수 있게 하고 싶을 수 있습니다.

```
conditional_bars <- function(df, condition, var) {
  df |>
    filter({{ condition }}) |>
    ggplot(aes(x = {{ var }})) +
    geom_bar()
}

diamonds |> conditional_bars(cut == "Good", clarity)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_25in05.png" alt="Bar plot of clarity of diamonds. The most common is SI1, then SI2, then VS2, then VS1, then VVS2, then VVS1, then I1, then lastly IF." />
</figure>

창의력을 발휘하여 다른 방식으로 데이터 요약을 표시할 수도 있습니다. [_https://oreil.ly/MV4kQ_](https://oreil.ly/MV4kQ)에서 멋진 응용 프로그램을 찾을 수 있습니다. 축 레이블을 사용하여 가장 높은 값을 표시합니다. ggplot2에 대해 더 많이 배울수록 함수의 위력은 계속해서 증가할 것입니다.

생성한 플롯에 레이블을 지정하는 더 복잡한 사례로 마무리하겠습니다.

## 레이블링 (Labeling)

앞에서 보여드린 히스토그램 함수를 기억하시나요?

```
histogram <- function(df, var, binwidth = NULL) {
  df |>
    ggplot(aes(x = {{ var }})) +
    geom_histogram(binwidth = binwidth)
}
```

사용된 변수와 빈 너비로 출력에 레이블을 지정할 수 있다면 좋지 않을까요? 그렇게 하려면 tidy 평가의 이면을 살펴보고 아직 논의하지 않은 패키지인 rlang의 함수를 사용해야 합니다. rlang은 tidy 평가를 구현(및 기타 여러 유용한 도구)하기 때문에 tidyverse의 거의 모든 다른 패키지에서 사용되는 하위 수준 패키지입니다.

레이블링 문제를 해결하기 위해 <a href="https://rlang.r-lib.org/reference/englue.html" class="orm:hideurl"><code>rlang::englue()</code></a>를 사용할 수 있습니다. 이것은 <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>와 유사하게 작동하므로 <a href="https://rdrr.io/r/base/Paren.html" class="orm:hideurl"><code>{ }</code></a>로 감싼 모든 값이 문자열에 삽입됩니다. 하지만 `{{ }}`도 이해하여 적절한 변수 이름을 자동으로 삽입합니다.

```
histogram <- function(df, var, binwidth) {
  label <- rlang::englue("A histogram of {{var}} with binwidth {binwidth}")

  df |>
    ggplot(aes(x = {{ var }})) +
    geom_histogram(binwidth = binwidth) +
    labs(title = label)
}

diamonds |> histogram(carat, 0.1)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_25in06.png" alt="Histogram of carats of diamonds, ranging from 0 to 5. The distribution is unimodal and right skewed with a peak between 0 to 1 carats." />
</figure>

ggplot2 플롯에서 문자열을 제공하려는 다른 모든 곳에서 동일한 접근 방식을 사용할 수 있습니다.

## 연습문제 (Exercises)

이러한 각 단계를 점진적으로 구현하여 풍부한 플로팅 함수를 구축하세요.

1.  데이터 세트와 `x` 및 `y` 변수가 주어지면 산점도를 그립니다.

2.  최적합 선(즉, 표준 오차가 없는 선형 모델)을 추가합니다.

3.  제목을 추가합니다.

# 스타일 (Style)

R은 함수나 인자의 이름에 신경 쓰지 않지만 이름은 사람에게 큰 차이를 만듭니다. 이상적으로는 함수의 이름이 짧으면서도 함수가 무엇을 하는지 분명하게 연상시켜야 합니다. 그것은 어렵습니다! 하지만 RStudio의 자동 완성 기능을 사용하면 긴 이름을 쉽게 입력할 수 있으므로 짧은 것보다 명확한 것이 더 낫습니다.

일반적으로 함수 이름은 동사여야 하고 인자는 명사여야 합니다. 몇 가지 예외가 있습니다. 함수가 잘 알려진 명사를 계산하거나(즉, `compute_mean()`보다 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>이 더 나음) 객체의 속성에 액세스하는 경우(즉, `get_coefficients()`보다 <a href="https://rdrr.io/r/stats/coef.html" class="orm:hideurl"><code>coef()</code></a>가 더 나음) 명사도 괜찮습니다. 최선의 판단을 내리고 나중에 더 나은 이름을 생각해 냈다면 주저하지 말고 함수 이름을 바꾸세요.

```
# Too short
f()

# Not a verb, or descriptive
my_awesome_function()

# Long, but clear
impute_missing()
collapse_years()
```

R은 함수에서 공백을 어떻게 사용하는지에도 신경 쓰지 않지만 미래의 독자는 신경 쓸 것입니다. <a href="ch04.html#chp-workflow-style" data-type="xref">4장</a>의 규칙을 계속 따르세요. 또한 `function()` 뒤에는 항상 중괄호(<a href="https://rdrr.io/r/base/Paren.html" class="orm:hideurl"><code>{}</code></a>)가 와야 하며 그 내용은 두 칸 더 들여쓰기해야 합니다. 이렇게 하면 왼쪽 여백을 훑어보면서 코드의 계층 구조를 더 쉽게 볼 수 있습니다.

```
# Missing extra two spaces
density <- function(color, facets, binwidth = 0.1) {
diamonds |>
  ggplot(aes(x = carat, y = after_stat(density), color = {{ color }})) +
  geom_freqpoly(binwidth = binwidth) +
  facet_wrap(vars({{ facets }}))
}

# Pipe indented incorrectly
density <- function(color, facets, binwidth = 0.1) {
  diamonds |>
  ggplot(aes(x = carat, y = after_stat(density), color = {{ color }})) +
  geom_freqpoly(binwidth = binwidth) +
  facet_wrap(vars({{ facets }}))
}
```

보시다시피 `{{ }}` 안에 추가 공백을 넣는 것이 좋습니다. 이렇게 하면 특이한 일이 일어나고 있음이 분명해집니다.

## 연습문제 (Exercises)

1.  다음 두 함수의 소스 코드를 각각 읽고 무엇을 하는지 파악한 다음 더 나은 이름을 브레인스토밍하세요.

    ```
    f1 <- function(string, prefix) {
      str_sub(string, 1, str_length(prefix)) == prefix
    }

    f3 <- function(x, y) {
      rep(y, length.out = length(x))
    }
    ```

2.  최근에 작성한 함수를 선택하고 5분 동안 함수와 인자에 대한 더 나은 이름을 브레인스토밍하세요.

3.  `norm_r()`, `norm_d()` 등이 <a href="https://rdrr.io/r/stats/Normal.html" class="orm:hideurl"><code>rnorm()</code></a> 및 <a href="https://rdrr.io/r/stats/Normal.html" class="orm:hideurl"><code>dnorm()</code></a>보다 나은 이유를 주장하세요. 그 반대의 이유를 주장하세요. 이름을 어떻게 하면 더 명확하게 만들 수 있을까요?

# 요약 (Summary)

이 장에서는 벡터 만들기, 데이터 프레임 만들기 또는 플롯 만들기의 세 가지 유용한 시나리오에 대해 함수를 작성하는 방법을 배웠습니다. 그 과정에서 많은 예제를 보았으며, 이상적으로는 창의적인 아이디어가 떠오르기 시작했고, 분석 코드에서 함수가 어디에 도움이 될 수 있는지에 대한 아이디어를 얻었을 것입니다.

우리는 함수를 시작하기 위한 최소한의 것만 보여주었으며 배울 것이 더 많습니다. 자세히 알아볼 몇 곳은 다음과 같습니다.

- Tidy 평가를 사용한 프로그래밍에 대해 자세히 알아보려면 [dplyr을 사용한 프로그래밍](https://oreil.ly/8xygI) 및 [tidyr을 사용한 프로그래밍](https://oreil.ly/QGH9n)의 유용한 레시피를 참조하고 ["데이터 마스킹이란 무엇이며 왜 {{가 필요한가요?"](https://oreil.ly/eecUd)에서 이론에 대해 자세히 알아보세요.
- ggplot2 코드의 중복을 줄이는 방법에 대해 자세히 알아보려면 ggplot2 책의 <a href="https://oreil.ly/Vvt6k" class="uri">"ggplot2를 사용한 프로그래밍" 장</a>을 읽으세요.
- 함수 스타일에 대한 자세한 조언은 <a href="https://oreil.ly/rLKSn" class="uri">tidyverse 스타일 가이드</a>를 참조하세요.

다음 장에서는 코드 중복을 줄이기 위한 추가 도구를 제공하는 반복(iteration)에 대해 알아보겠습니다.
