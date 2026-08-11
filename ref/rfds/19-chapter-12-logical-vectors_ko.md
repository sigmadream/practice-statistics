# 제12장. 논리형 벡터 (Logical Vectors)

# 소개 (Introduction)

이 장에서는 논리형 벡터를 다루는 도구를 배울 것입니다. 논리형 벡터는 각 요소가 `TRUE`, `FALSE`, `NA`라는 세 가지 가능한 값 중 하나만 가질 수 있기 때문에 가장 단순한 유형의 벡터입니다. 원시 데이터(raw data)에서 논리형 벡터를 발견하는 일은 비교적 드물지만, 거의 모든 분석 과정에서 논리형 벡터를 생성하고 조작하게 될 것입니다.

논리형 벡터를 생성하는 가장 일반적인 방법인 숫자 비교부터 논의를 시작하겠습니다. 그런 다음 부울 대수(Boolean algebra)를 사용하여 서로 다른 논리형 벡터를 결합하는 방법과 유용한 요약(summaries)에 대해 배울 것입니다. 마지막으로 논리형 벡터로 구동되는 조건부 변경을 수행하기 위한 두 가지 유용한 함수인 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a>으로 마무리하겠습니다.

## 사전 준비 (Prerequisites)

이 장에서 배울 대부분의 함수는 기본 R(base R)에서 제공되므로 tidyverse가 필요하지는 않지만, 데이터 프레임을 다루기 위해 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>, <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> 및 그와 관련된 함수들을 사용할 수 있도록 계속 로드하겠습니다. 또한 <a href="https://rdrr.io/pkg/nycflights13/man/flights.html" class="orm:hideurl"><code>nycflights13::flights</code></a> 데이터세트에서 계속 예제를 가져올 것입니다.

```
library(tidyverse)
library(nycflights13)
```

그러나 더 많은 도구들을 다루기 시작하면서, 항상 완벽한 실제 예제가 있는 것은 아닙니다. 따라서 <a href="https://rdrr.io/r/base/c.html" class="orm:hideurl"><code>c()</code></a>를 사용하여 더미 데이터(dummy data)를 만들어 시작하겠습니다.

```
x <- c(1, 2, 3, 5, 7, 11, 13)
x * 2
#> [1]  2  4  6 10 14 22 26
```

이렇게 하면 여러분의 데이터 문제에 어떻게 적용될지 파악하기는 어려워지지만, 개별 함수를 설명하기는 더 쉬워집니다. 우리가 독립적인 벡터(free-floating vector)에 대해 수행하는 모든 조작은 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 그 관련 함수들을 사용하여 데이터 프레임 내의 변수에 대해서도 수행할 수 있다는 점만 기억하세요.

```
df <- tibble(x)
df |>
  mutate(y = x *  2)
#> # A tibble: 7 × 2
#>       x     y
#>   <dbl> <dbl>
#> 1     1     2
#> 2     2     4
#> 3     3     6
#> 4     5    10
#> 5     7    14
#> 6    11    22
#> # … with 1 more row
```

# 비교 (Comparisons)

논리형 벡터를 생성하는 일반적인 방법은 `<`, `<=`, `>`, `>=`, `!=`, `==`를 사용한 숫자 비교를 통해서입니다. 지금까지 우리는 주로 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> 내에서 논리형 변수를 일시적으로 만들었습니다—그것들은 계산되고, 사용되고, 그리고 버려졌습니다. 예를 들어, 다음 필터는 거의 정시에 도착하는 모든 주간(daytime) 출발 항공편을 찾습니다.

```
flights |>
  filter(dep_time > 600 & dep_time < 2000 & abs(arr_delay) < 20)
#> # A tibble: 172,286 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      601            600         1      844            850
#> 2  2013     1     1      602            610        -8      812            820
#> 3  2013     1     1      602            605        -3      821            805
#> 4  2013     1     1      606            610        -4      858            910
#> 5  2013     1     1      606            610        -4      837            845
#> 6  2013     1     1      607            607         0      858            915
#> # … with 172,280 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

이것이 단축키(shortcut)이며, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>를 사용하여 근본적인 논리형 변수를 명시적으로 만들 수 있다는 것을 알아두면 유용합니다.

```
flights |>
  mutate(
    daytime = dep_time > 600 & dep_time < 2000,
    approx_ontime = abs(arr_delay) < 20,
    .keep = "used"
  )
#> # A tibble: 336,776 × 4
#>   dep_time arr_delay daytime approx_ontime
#>      <int>     <dbl> <lgl>   <lgl>
#> 1      517        11 FALSE   TRUE
#> 2      533        20 FALSE   FALSE
#> 3      542        33 FALSE   FALSE
#> 4      544       -18 FALSE   TRUE
#> 5      554       -25 FALSE   FALSE
#> 6      554        12 FALSE   TRUE
#> # … with 336,770 more rows
```

이것은 코드를 더 쉽게 읽고 각 단계가 올바르게 계산되었는지 확인하기 쉽게 만들어주기 때문에 더 복잡한 논리의 중간 단계에 이름을 붙일 때 특히 유용합니다.

종합하면, 초기 필터는 다음과 동일합니다.

```
flights |>
  mutate(
    daytime = dep_time > 600 & dep_time < 2000,
    approx_ontime = abs(arr_delay) < 20,
  ) |>
  filter(daytime & approx_ontime)
```

## 부동소수점 비교 (Floating-Point Comparison)

숫자에 `==`를 사용할 때는 주의하세요. 예를 들어, 이 벡터는 숫자 1과 2를 포함하고 있는 것처럼 보입니다.

```
x <- c(1 / 49 * 49, sqrt(2) ^ 2)
x
#> [1] 1 2
```

하지만 이들이 같은지 테스트하면 `FALSE`가 반환됩니다.

```
x == c(1, 2)
#> [1] FALSE FALSE
```

무슨 일이 일어나고 있는 걸까요? 컴퓨터는 소수점 이하 자릿수가 고정된 숫자를 저장하므로 1/49 또는 `sqrt(2)`를 정확하게 표현할 수 있는 방법이 없으며, 후속 계산에서 아주 약간 어긋나게 됩니다. `digits`<sup><a href="ch12.html#idm44771302207136" id="idm44771302207136-marker" data-type="noteref">1</a></sup> 인자와 함께 <a href="https://rdrr.io/r/base/print.html" class="orm:hideurl"><code>print()</code></a>를 호출하면 정확한 값을 볼 수 있습니다.

```
print(x, digits = 16)
#> [1] 0.9999999999999999 2.0000000000000004
```

R이 왜 기본적으로 이 숫자들을 반올림하는지 알 수 있을 것입니다. 그것들은 실제로 여러분이 기대하는 값과 매우 가깝습니다.

이제 `==`가 왜 실패하는지 확인했으니, 이 문제를 어떻게 해결할 수 있을까요? 한 가지 방법은 작은 차이를 무시하는 <a href="https://dplyr.tidyverse.org/reference/near.html" class="orm:hideurl"><code>dplyr::near()</code></a>를 사용하는 것입니다.

```
near(x, c(1, 2))
#> [1] TRUE TRUE
```

## 결측값 (Missing Values)

결측값은 미지의 것을 나타내므로 "전염성(contagious)"이 있습니다. 미지의 값을 포함하는 거의 모든 연산 결과도 미지의 값이 됩니다.

```
NA > 5
#> [1] NA
10 == NA
#> [1] NA
```

가장 혼란스러운 결과는 바로 이것입니다.

```
NA == NA
#> [1] NA
```

우리가 약간의 문맥을 인위적으로 제공하면 이것이 왜 참인지 이해하기 가장 쉽습니다.

```
# 우리는 메리의 나이를 모릅니다.
age_mary <- NA

# 우리는 존의 나이를 모릅니다.
age_john <- NA

# 메리와 존은 동갑인가요?
age_mary == age_john
#> [1] NA
# 우리는 모릅니다!
```

따라서 `dep_time`이 누락된 모든 항공편을 찾으려고 할 때, 다음 코드는 작동하지 않습니다. 왜냐하면 `dep_time == NA`는 모든 단일 행에 대해 `NA`를 반환할 것이고, <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>는 자동으로 결측값을 삭제하기 때문입니다.

```
flights |>
  filter(dep_time == NA)
#> # A tibble: 0 × 19
#> # … with 19 variables: year <int>, month <int>, day <int>, dep_time <int>,
#> #   sched_dep_time <int>, dep_delay <dbl>, arr_time <int>, …
```

대신 우리는 새로운 도구인 <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>가 필요합니다.

## is.na()

`is.na(x)`는 모든 유형의 벡터에서 작동하며 결측값에 대해서는 `TRUE`를, 그 외의 모든 것에 대해서는 `FALSE`를 반환합니다.

```
is.na(c(TRUE, NA, FALSE))
#> [1] FALSE  TRUE FALSE
is.na(c(1, NA, 3))
#> [1] FALSE  TRUE FALSE
is.na(c("a", NA, "b"))
#> [1] FALSE  TRUE FALSE
```

<a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>를 사용하여 `dep_time`이 누락된 모든 행을 찾을 수 있습니다.

```
flights |>
  filter(is.na(dep_time))
#> # A tibble: 8,255 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1       NA           1630        NA       NA           1815
#> 2  2013     1     1       NA           1935        NA       NA           2240
#> 3  2013     1     1       NA           1500        NA       NA           1825
#> 4  2013     1     1       NA            600        NA       NA            901
#> 5  2013     1     2       NA           1540        NA       NA           1747
#> 6  2013     1     2       NA           1620        NA       NA           1746
#> # … with 8,249 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

<a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>는 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>에서도 유용할 수 있습니다. <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>는 일반적으로 모든 결측값을 끝에 배치하지만, 먼저 <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>로 정렬하여 이 기본값을 재정의할 수 있습니다.

```
flights |>
  filter(month == 1, day == 1) |>
  arrange(dep_time)
#> # A tibble: 842 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 836 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …

flights |>
  filter(month == 1, day == 1) |>
  arrange(desc(is.na(dep_time)), dep_time)
#> # A tibble: 842 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1       NA           1630        NA       NA           1815
#> 2  2013     1     1       NA           1935        NA       NA           2240
#> 3  2013     1     1       NA           1500        NA       NA           1825
#> 4  2013     1     1       NA            600        NA       NA            901
#> 5  2013     1     1      517            515         2      830            819
#> 6  2013     1     1      533            529         4      850            830
#> # … with 836 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

결측값에 대해서는 <a href="ch18.html#chp-missing-values" data-type="xref">제18장</a>에서 다시 더 깊이 다루겠습니다.

## 연습문제 (Exercises)

1. <a href="https://dplyr.tidyverse.org/reference/near.html" class="orm:hideurl"><code>dplyr::near()</code></a>는 어떻게 작동하나요? 소스 코드를 보려면 `near`를 입력하세요. `sqrt(2)^2`는 2에 가깝습니까?
2. <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>, <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>, <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>를 함께 사용하여 `dep_time`, `sched_dep_time`, `dep_delay`의 결측값이 서로 어떻게 연결되어 있는지 설명하세요.

# 부울 대수 (Boolean Algebra)

여러 개의 논리형 벡터가 있으면 부울 대수(Boolean algebra)를 사용하여 결합할 수 있습니다. R에서 `&`는 "그리고(and)", `|`는 "또는(or)", `!`는 "아님(not)", <a href="https://rdrr.io/r/base/Logic.html" class="orm:hideurl"><code>xor()</code></a>은 배타적 논리합(exclusive or)입니다.<sup><a href="ch12.html#idm44771301699872" id="idm44771301699872-marker" data-type="noteref">2</a></sup> 예를 들어, `df |> filter(!is.na(x))`는 `x`가 누락되지 않은 모든 행을 찾고, `df |> filter(x < -10 | x > 0)`는 `x`가 -10보다 작거나 0보다 큰 모든 행을 찾습니다. <a href="#fig-bool-ops" data-type="xref">그림 12-1</a>은 전체 부울 연산의 집합과 그 작동 방식을 보여줍니다.
#> # carrier <chr>, flight <int>, tail_num <chr>, origin <chr>, dest <chr>, …

```

일관성 없이 이름 지어진 열이 많고 하나씩 손으로 고치는 것이 고통스럽다면, 유용한 자동화 정리를 제공하는 <a href="https://rdrr.io/pkg/janitor/man/clean_names.html" class="orm:hideurl"><code>janitor::clean_names()</code></a>를 확인해 보세요.

## relocate()

변수들을 이리저리 옮기려면 <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a>를 사용하세요. 관련 변수들을 함께 모으거나 중요한 변수를 앞으로 옮기고 싶을 수 있습니다. 기본적으로 <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a>는 변수를 맨 앞으로 옮깁니다.

```

flights |>
relocate(time_hour, air_time)
#> # A tibble: 336,776 × 19
#> time_hour air_time year month day dep_time sched_dep_time
#> <dttm> <dbl> <int> <int> <int> <int> <int>
#> 1 2013-01-01 05:00:00 227 2013 1 1 517 515
#> 2 2013-01-01 05:00:00 227 2013 1 1 533 529
#> 3 2013-01-01 05:00:00 160 2013 1 1 542 540
#> 4 2013-01-01 05:00:00 183 2013 1 1 544 545
#> 5 2013-01-01 06:00:00 116 2013 1 1 554 600
#> 6 2013-01-01 05:00:00 150 2013 1 1 554 558
#> # … with 336,770 more rows, and 12 more variables: dep_delay <dbl>,
#> # arr_time <int>, sched_arr_time <int>, arr_delay <dbl>, carrier <chr>, …

```

<a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에서와 마찬가지로 `.before`와 `.after` 인자를 사용하여 놓을 위치를 지정할 수도 있습니다.

```

flights |>
relocate(year:dep_time, .after = time_hour)
flights |>
relocate(starts_with("arr"), .before = dep_time)

````

## 연습 문제

1.  `dep_time`, `sched_dep_time`, `dep_delay`를 비교해 보세요. 이 세 숫자가 어떻게 관련되어 있을 것으로 예상하시나요?

2.  `flights`에서 `dep_time`, `dep_delay`, `arr_time`, `arr_delay`를 선택하는 가능한 모든 방법을 브레인스토밍해 보세요.

3.  <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a> 호출에서 동일한 변수의 이름을 여러 번 지정하면 어떻게 되나요?

4.  <a href="https://tidyselect.r-lib.org/reference/all_of.html" class="orm:hideurl"><code>any_of()</code></a> 함수는 어떤 역할을 하나요? 다음 벡터와 함께 사용할 때 왜 유용할 수 있을까요?

    ```
    variables <- c("year", "month", "day", "dep_delay", "arr_delay")
    ```

5.  다음 코드를 실행한 결과가 놀라운가요? select 헬퍼 함수들은 기본적으로 대소문자를 어떻게 처리하나요? 이 기본값을 어떻게 변경할 수 있나요?

    ```
    flights |> select(contains("TIME"))
    ```

6.  측정 단위를 나타내기 위해 `air_time`의 이름을 `air_time_min`으로 바꾸고 데이터 프레임의 맨 앞으로 옮기세요.

7.  다음은 왜 작동하지 않으며, 에러의 의미는 무엇인가요?

    ```
    flights |>
      select(tailnum) |>
      arrange(arr_delay)
    #> Error in `arrange()`:
    #> ℹ In argument: `..1 = arr_delay`.
    #> Caused by error:
    #> ! object 'arr_delay' not found
    ```

# 파이프 (The Pipe)

파이프의 간단한 예제를 보여드렸지만, 파이프의 진정한 위력은 여러 동사를 결합하기 시작할 때 나타납니다.

예를 들어, 휴스턴의 IAH 공항으로 가는 빠른 항공편을 찾고 싶다고 가정해 보세요. <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>, <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>를 결합해야 합니다.

````

flights |>
filter(dest == "IAH") |>
mutate(speed = distance / air_time \* 60) |>
select(year:day, dep_time, carrier, flight, speed) |>
arrange(desc(speed))
#> # A tibble: 7,198 × 7
#> year month day dep_time carrier flight speed
#> <int> <int> <int> <int> <chr> <int> <dbl>
#> 1 2013 7 9 707 UA 226 522.
#> 2 2013 8 27 1850 UA 1128 521.
#> 3 2013 8 28 902 UA 1711 519.
#> 4 2013 8 28 2122 UA 1022 519.
#> 5 2013 6 11 1628 UA 1178 515.
#> 6 2013 8 27 1017 UA 333 515.
#> # … with 7,192 more rows

```

이 파이프라인에는 4개의 단계가 있지만 각 줄의 시작 부분에 동사가 오기 때문에 훑어보기가 쉽습니다. `flights` 데이터로 시작해서(start with), 필터링하고(then filter), 변이시키고(then mutate), 선택하고(then select), 정렬합니다(then arrange).

파이프가 없다면 어떻게 될까요? 이전 호출 내부에 각 함수 호출을 중첩할 수 있습니다.

```

arrange(
select(
mutate(
filter(
flights,
dest == "IAH"
),
speed = distance / air_time \* 60
),
year:day, dep_time, carrier, flight, speed
),
desc(speed)
)

```

또는 중간 객체를 많이 사용할 수도 있습니다.

```

flights1 <- filter(flights, dest == "IAH")
flights2 <- mutate(flights1, speed = distance / air_time \* 60)
flights3 <- select(flights2, year:day, dep_time, carrier, flight, speed)
arrange(flights3, desc(speed))

```

두 형태 모두 각자의 쓰임새가 있지만, 파이프는 일반적으로 쓰기도 읽기도 쉬운 데이터 분석 코드를 생성합니다.

코드에 파이프를 추가하려면 내장 키보드 단축키인 Ctrl/Cmd+Shift+M을 사용하는 것을 권장합니다. <a href="#fig-pipe-options" data-type="xref">그림 3-1</a>과 같이 `%>%` 대신 `|>`를 사용하려면 RStudio 옵션을 한 번 변경해야 합니다. `%>%`에 대해서는 곧 자세히 설명하겠습니다.

<figure>
<p><img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0301.png" alt="Screenshot showing the &quot;Use native pipe operator&quot; option which can be found on the &quot;Editing&quot; panel of the &quot;Code&quot; options." /></p>
<h6 id="figure-3-1.-to-insert-make-sure-the-use-native-pipe-operator-option-is-checked.">그림 3-1. <code>|&gt;</code>를 삽입하려면 "Use native pipe operator" 옵션이 체크되어 있는지 확인하세요.</h6>
</figure>

# magrittr

tidyverse를 한동안 사용해 보셨다면 magrittr 패키지에서 제공하는 `%>%` 파이프에 익숙할 것입니다. magrittr 패키지는 핵심 tidyverse에 포함되어 있으므로 tidyverse를 로드할 때마다 `%>%`를 사용할 수 있습니다.

```

library(tidyverse)

mtcars %>%
group_by(cyl) %>%
summarize(n = n())

```

간단한 경우 `|>`와 `%>%`는 동일하게 작동합니다. 그렇다면 왜 기본 파이프(base pipe)를 추천할까요? 첫째, 기본 R의 일부이기 때문에 tidyverse를 사용하지 않을 때에도 항상 사용할 수 있습니다. 둘째, `|>`가 `%>%`보다 훨씬 더 단순합니다. 2014년 `%>%`가 발명되고 2021년 R 4.1.0에 `|>`가 포함되기까지의 시간 동안 우리는 파이프를 더 잘 이해하게 되었습니다. 이 덕분에 기본(base) 구현체는 자주 사용되지 않고 덜 중요한 기능들을 버릴 수 있었습니다.

# 그룹 (Groups)

지금까지 행과 열에 작동하는 함수에 대해 배웠습니다. 그룹화(work with groups) 기능을 추가하면 dplyr은 훨씬 더 강력해집니다. 이 섹션에서는 가장 중요한 함수인 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>, <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>, 그리고 slice 함수군(family)에 중점을 둘 것입니다.

## group_by()

<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>를 사용하여 분석에 의미 있는 그룹으로 데이터셋을 나눕니다.

```

flights |>
group_by(month)
#> # A tibble: 336,776 × 19
#> # Groups: month [12]
#> year month day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#> <int> <int> <int> <int> <int> <dbl> <int> <int>
#> 1 2013 1 1 517 515 2 830 819
#> 2 2013 1 1 533 529 4 850 830
#> 3 2013 1 1 542 540 2 923 850
#> 4 2013 1 1 544 545 -1 1004 1022
#> 5 2013 1 1 554 600 -6 812 837
#> 6 2013 1 1 554 558 -4 740 728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> # carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …

```

<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>는 데이터를 변경하지 않지만 출력을 자세히 살펴보면 월별로 "그룹화되어 있음(grouped by)"을 나타내는 것(`Groups: month [12]`)을 확인할 수 있습니다. 이는 후속 작업들이 이제 "월별로" 작동함을 의미합니다. <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>는 이러한 그룹화된 특성(클래스(*class*)라고 함)을 데이터 프레임에 추가하며, 이로 인해 데이터에 적용되는 후속 동사의 동작이 변경됩니다.

## summarize()

가장 중요한 그룹화 연산은 요약(summary)으로, 단일 요약 통계를 계산하는 데 사용되는 경우 데이터 프레임을 축소하여 각 그룹당 단일 행을 갖도록 만듭니다. dplyr에서 이 연산은 다음 예제와 같이 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a><sup><a href="ch03.html#idm44771326592976" id="idm44771326592976-marker" data-type="noteref">3</a></sup>에 의해 수행되며, 이는 월별 평균 출발 지연을 계산합니다.

```

flights |>
group_by(month) |>
summarize(
avg_delay = mean(dep_delay)
)
#> # A tibble: 12 × 2
#> month avg_delay
#> <int> <dbl>
#> 1 1 NA
#> 2 2 NA
#> 3 3 NA
#> 4 4 NA
#> 5 5 NA
#> 6 6 NA
#> # … with 6 more rows

```

이런! 무언가 잘못되어 결과가 모두 누락된 값(missing value)을 나타내는 R의 기호인 `NA`("엔-에이"로 발음)가 되었습니다. 이는 관측된 비행 중 일부가 delay 열에 누락된 데이터를 가지고 있어서 이 값들을 포함하여 평균을 계산했을 때 `NA` 결과가 나왔기 때문입니다. 누락된 값에 대해서는 <a href="ch18.html#chp-missing-values" data-type="xref">18장</a>에서 자세히 다시 논의하겠지만, 지금은 `na.rm` 인자를 `TRUE`로 설정하여 모든 누락된 값을 무시하도록 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a> 함수에게 지시하겠습니다.

```

flights |>
group_by(month) |>
summarize(
delay = mean(dep_delay, na.rm = TRUE)
)
#> # A tibble: 12 × 2
#> month delay
#> <int> <dbl>
#> 1 1 10.0
#> 2 2 10.8
#> 3 3 13.2
#> 4 4 13.9
#> 5 5 13.0
#> 6 6 20.8
#> # … with 6 more rows

```

<a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>를 한 번 호출할 때 원하는 수만큼 요약값을 생성할 수 있습니다. 다음 장들에서 다양한 유용한 요약값에 대해 배우게 될 테지만, 각 그룹의 행 수를 반환하는 유용한 요약값 중 하나는 <a href="https://dplyr.tidyverse.org/reference/context.html" class="orm:hideurl"><code>n()</code></a>입니다.

```

flights |>
group_by(month) |>
summarize(
delay = mean(dep_delay, na.rm = TRUE),
n = n()
)
#> # A tibble: 12 × 3
#> month delay n
#> <int> <dbl> <int>
#> 1 1 10.0 27004
#> 2 2 10.8 24951
#> 3 3 13.2 28834
#> 4 4 13.9 28330
#> 5 5 13.0 28796
#> 6 6 20.8 28243
#> # … with 6 more rows

```

평균과 개수만으로도 데이터 과학에서 놀라울 정도로 멀리 갈 수 있습니다!

## slice_ 함수들

각 그룹 내에서 특정 행을 추출할 수 있게 해주는 5개의 편리한 함수가 있습니다.

`df |> slice_head(n = 1)`
각 그룹의 첫 번째 행을 가져옵니다.

`df |> slice_tail(n = 1)`
각 그룹의 마지막 행을 가져옵니다.

`df |> slice_min(x, n = 1)`
`x` 열의 값이 가장 작은 행을 가져옵니다.

`df |> slice_max(x, n = 1)`
## if_else()

조건이 `TRUE`일 때 한 값을 사용하고 `FALSE`일 때 다른 값을 사용하고 싶다면 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>dplyr::if_else()</code></a>를 사용할 수 있습니다.<sup><a href="ch12.html#idm44771300775168" id="idm44771300775168-marker" data-type="noteref">4</a></sup> <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>의 처음 세 가지 인자는 항상 사용하게 될 것입니다. 첫 번째 인자 `condition`은 논리형 벡터입니다. 두 번째 `true`는 조건이 참일 때의 출력을 제공하고, 세 번째 `false`는 조건이 거짓일 때의 출력을 제공합니다.

숫자 벡터를 "+ve"(양수) 또는 "-ve"(음수)로 레이블 지정하는 간단한 예제로 시작해 보겠습니다.

```

x <- c(-3:3, NA)
if_else(x > 0, "+ve", "-ve")
#> [1] "-ve" "-ve" "-ve" "-ve" "+ve" "+ve" "+ve" NA

```

입력이 `NA`인 경우 사용될 선택적인 네 번째 인자인 `missing`이 있습니다.

```

if_else(x > 0, "+ve", "-ve", "???")
#> [1] "-ve" "-ve" "-ve" "-ve" "+ve" "+ve" "+ve" "???"

```

`true` 및 `false` 인자에 벡터를 사용할 수도 있습니다. 예를 들어, 이를 통해 <a href="https://rdrr.io/r/base/MathFun.html" class="orm:hideurl"><code>abs()</code></a>의 최소한의 구현(minimal implementation)을 만들 수 있습니다.

```

if_else(x < 0, -x, x)
#> [1] 3 2 1 0 1 2 3 NA

```

지금까지는 모든 인자가 동일한 벡터를 사용했지만, 물론 혼합하고 일치시킬 수 있습니다. 예를 들어, 다음과 같이 <a href="https://dplyr.tidyverse.org/reference/coalesce.html" class="orm:hideurl"><code>coalesce()</code></a>의 간단한 버전을 구현할 수 있습니다.

```

x1 <- c(NA, 1, 2, NA)
y1 <- c(3, NA, 4, 6)
if_else(is.na(x1), y1, x1)
#> [1] 3 1 2 6

```

이전 레이블 지정 예제에서 약간의 부적절함(infelicity)을 알아챘을 수도 있습니다. 0은 양수도 아니고 음수도 아닙니다. 추가적인 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>를 더하여 이 문제를 해결할 수 있습니다.

```

if_else(x == 0, "0", if_else(x < 0, "-ve", "+ve"), "???")
#> [1] "-ve" "-ve" "-ve" "0" "+ve" "+ve" "+ve" "???"

```

이것은 이미 읽기가 조금 어려우며, 조건이 더 많아지면 훨씬 더 어려워질 것임을 상상할 수 있습니다. 대신 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>dplyr::case_when()</code></a>으로 전환할 수 있습니다.

## case_when()

dplyr의 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a>은 SQL의 `CASE` 문에서 영감을 받았으며, 조건에 따라 서로 다른 계산을 수행하는 유연한 방법을 제공합니다. 불행히도 tidyverse에서 사용할 다른 어떤 것과도 닮지 않은 특별한 구문(syntax)을 가지고 있습니다. 이것은 `condition ~ output`과 같은 형태의 쌍(pairs)을 취합니다. `condition`은 반드시 논리형 벡터여야 하며, 그것이 `TRUE`일 때 `output`이 사용됩니다.

이것은 이전의 중첩된(nested) <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>를 다음과 같이 다시 만들 수 있음을 의미합니다.

```

x <- c(-3:3, NA)
case_when(
x == 0 ~ "0",
x < 0 ~ "-ve",
x > 0 ~ "+ve",
is.na(x) ~ "???"
)
#> [1] "-ve" "-ve" "-ve" "0" "+ve" "+ve" "+ve" "???"

```

코드가 더 길어지긴 했지만, 또한 더 명시적(explicit)입니다.

<a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a>이 어떻게 작동하는지 설명하기 위해 몇 가지 더 간단한 사례를 살펴보겠습니다. 어떤 케이스와도 일치하지 않으면 출력은 `NA`가 됩니다.

```

case_when(
x < 0 ~ "-ve",
x > 0 ~ "+ve"
)
#> [1] "-ve" "-ve" "-ve" NA "+ve" "+ve" "+ve" NA

```

"기본(default)"/모두 잡는(catchall) 값을 만들려면 왼쪽에 `TRUE`를 사용하세요.

```

case_when(
x < 0 ~ "-ve",
x > 0 ~ "+ve",
TRUE ~ "???"
)
#> [1] "-ve" "-ve" "-ve" "???" "+ve" "+ve" "+ve" "???"

```

여러 조건이 일치하는 경우 첫 번째 조건만 사용된다는 점에 유의하세요.

```

case_when(
x > 0 ~ "+ve",
x > 2 ~ "big"
)
#> [1] NA NA NA NA "+ve" "+ve" "+ve" NA

```

<a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>에서와 마찬가지로 `~`의 양쪽에 변수를 사용할 수 있으며, 여러분의 문제에 필요한 대로 변수들을 섞고 일치시킬 수 있습니다. 예를 들어, <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a>을 사용하여 도착 지연에 대해 사람이 읽을 수 있는(human-readable) 레이블을 제공할 수 있습니다.

```

flights |>
mutate(
status = case_when(
is.na(arr_delay) ~ "cancelled",
arr_delay < -30 ~ "very early",
arr_delay < -15 ~ "early",
abs(arr_delay) <= 15 ~ "on time",
arr_delay < 60 ~ "late",
arr_delay < Inf ~ "very late",
),
.keep = "used"
)
#> # A tibble: 336,776 × 2
#> arr_delay status
#> <dbl> <chr>  
#> 1 11 on time
#> 2 20 late  
#> 3 33 late  
#> 4 -18 early  
#> 5 -25 early  
#> 6 12 on time
#> # … with 336,770 more rows

```

이런 종류의 복잡한 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a> 문을 작성할 때는 조심하세요. 저의 첫 두 번의 시도는 `<`와 `>`가 섞여 있었고, 저는 우연히 중복되는 조건을 계속 만들었습니다.

## 호환되는 유형 (Compatible Types)

<a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a> 모두 출력에서 *호환되는(compatible)* 유형(types)을 요구한다는 점에 유의하세요. 만약 호환되지 않는다면, 다음과 같은 오류를 보게 될 것입니다.

```

if_else(TRUE, "a", 1)
#> Error in `if_else()`:
#> ! Can't combine `true` <character> and `false` <double>.

case_when(
x < -1 ~ TRUE,  
 x > 0 ~ now()
)
#> Error in `case_when()`:
#> ! Can't combine `..1 (right)` <logical> and `..2 (right)` <datetime<local>>.

```

한 유형의 벡터를 다른 유형으로 자동 변환하는 것은 오류의 흔한 원인이기 때문에, 전반적으로 비교적 적은 수의 유형들만 호환됩니다. 다음은 호환되는 가장 중요한 경우들입니다.

- <a href="#sec-numeric-summaries-of-logicals" data-type="xref">"논리형 벡터의 숫자 요약"</a>에서 논의했듯이, 숫자(numeric)와 논리형(logical) 벡터는 호환됩니다.
- 문자열(strings)과 요인(factors)(<a href="ch16.html#chp-factors" data-type="xref">제16장</a>)은 호환됩니다. 요인은 제한된 값 집합을 가진 문자열이라고 생각할 수 있기 때문입니다.
- <a href="ch17.html#chp-datetimes" data-type="xref">제17장</a>에서 논의할 날짜(dates)와 날짜-시간(date-times)은 호환됩니다. 날짜를 날짜-시간의 특별한 경우로 생각할 수 있기 때문입니다.
- 엄밀히 말해 논리형 벡터인 `NA`는 모든 것과 호환됩니다. 왜냐하면 모든 벡터는 결측값을 표현하는 어떤 방법을 가지고 있기 때문입니다.

이 규칙들을 외우기를 기대하지는 않지만, tidyverse 전체에 걸쳐 일관되게 적용되기 때문에 시간이 지남에 따라 제2의 천성(second nature)이 되어야 합니다.

## 연습문제 (Exercises)

1. 숫자가 2로 나누어 떨어지면 짝수(even)이며, R에서는 `x %% 2 == 0`으로 알아낼 수 있습니다. 이 사실과 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>를 사용하여 0에서 20 사이의 각 숫자가 짝수인지 홀수(odd)인지 결정하세요.
2. `x <- c("Monday", "Saturday", "Wednesday")`와 같은 요일 벡터가 주어졌을 때, <a href="https://rdrr.io/r/base/ifelse.html" class="orm:hideurl"><code>ifelse()</code></a> 문을 사용하여 그것들을 주말(weekends) 또는 평일(weekdays)로 레이블 지정하세요.
3. `x`라는 숫자 벡터의 절댓값(absolute value)을 계산하기 위해 <a href="https://rdrr.io/r/base/ifelse.html" class="orm:hideurl"><code>ifelse()</code></a>를 사용하세요.
4. `flights`의 `month`와 `day` 열을 사용하여 미국의 중요한 휴일(새해 첫날, 7월 4일 독립기념일, 추수감사절, 크리스마스)에 레이블을 지정하는 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a> 문을 작성하세요. 먼저 `TRUE` 또는 `FALSE`가 되는 논리형 열을 만들고, 그런 다음 휴일 이름을 제공하거나 `NA`가 되는 문자형 열을 만드세요.

# 요약 (Summary)

논리형 벡터의 정의는 간단합니다. 각 값은 반드시 `TRUE`, `FALSE`, 또는 `NA`여야 하기 때문입니다. 그러나 논리형 벡터는 엄청난 힘을 제공합니다. 이 장에서 여러분은 `>`, `<`, `<=`, `>=`, `==`, `!=`, <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>를 사용하여 논리형 벡터를 만드는 방법; `!`, `&`, `|`를 사용하여 결합하는 방법; 그리고 <a href="https://rdrr.io/r/base/any.html" class="orm:hideurl"><code>any()</code></a>, <a href="https://rdrr.io/r/base/all.html" class="orm:hideurl"><code>all()</code></a>, <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>, <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>으로 요약하는 방법을 배웠습니다. 또한 논리형 벡터의 값에 따라 값을 반환할 수 있게 해주는 강력한 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/case_when.html" class="orm:hideurl"><code>case_when()</code></a> 함수도 배웠습니다.

다음 장들에서 우리는 논리형 벡터를 계속해서 볼 것입니다. 예를 들어, <a href="ch14.html#chp-strings" data-type="xref">제14장</a>에서는 `pattern`과 일치하는 `x`의 요소들에 대해 `TRUE`인 논리형 벡터를 반환하는 `str_detect(x, pattern)`에 대해 배울 것이고, <a href="ch17.html#chp-datetimes" data-type="xref">제17장</a>에서는 날짜와 시간의 비교로부터 논리형 벡터를 만들 것입니다. 하지만 일단 지금은 다음으로 가장 중요한 벡터 유형인 숫자 벡터(numeric vectors)로 넘어가겠습니다.

<sup>[1](ch12.html#idm44771302207136-marker)</sup> R은 보통 당신을 위해 print를 호출하지만(즉, `x`는 `print(x)`의 단축키입니다), 다른 인자를 제공하고 싶다면 명시적으로 호출하는 것이 유용합니다.

<sup>[2](ch12.html#idm44771301699872-marker)</sup> 즉, `xor(x, y)`는 `x`가 참이거나 `y`가 참일 때 참이지만, 둘 다 참인 경우는 제외합니다. 이것이 우리가 영어에서 보통 "or"를 사용하는 방식입니다. "Both(둘 다)"는 일반적으로 "아이스크림을 드시겠습니까 아니면 케이크를 드시겠습니까?(Would you like ice cream or cake?)"라는 질문에 대한 수용 가능한 대답이 아닙니다.

<sup>[3](ch12.html#idm44771300937360-marker)</sup> 이것은 <a href="ch19.html#chp-joins" data-type="xref">제19장</a>에서 다룰 것입니다.

<sup>[4](ch12.html#idm44771300775168-marker)</sup> dplyr의 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>는 기본 R의 <a href="https://rdrr.io/r/base/ifelse.html" class="orm:hideurl"><code>ifelse()</code></a>와 유사합니다. <a href="https://rdrr.io/r/base/ifelse.html" class="orm:hideurl"><code>ifelse()</code></a>에 비해 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>의 주요 장점은 두 가지입니다. 결측값에 어떤 일이 일어나야 할지 선택할 수 있고, 변수들이 호환되지 않는 유형을 가질 때 <a href="https://dplyr.tidyverse.org/reference/if_else.html" class="orm:hideurl"><code>if_else()</code></a>가 의미 있는 오류를 제공할 가능성이 훨씬 더 높다는 것입니다.
```
