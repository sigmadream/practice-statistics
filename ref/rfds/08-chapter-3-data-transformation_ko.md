# 3장. 데이터 변환

# 서론

시각화는 통찰력을 생성하는 중요한 도구이지만, 원하는 그래프를 만들기에 정확히 알맞은 형태의 데이터를 얻는 경우는 드뭅니다. 데이터에 대한 질문에 답하기 위해 몇 가지 새로운 변수나 요약값을 만들어야 할 때가 자주 있고, 또는 작업하기 조금 더 쉽게 만들기 위해 단순히 변수 이름을 바꾸거나 관측치의 순서를 재정렬하고 싶을 수도 있습니다. 이번 장에서 이 모든 것(그리고 더 많은 것!)을 하는 방법을 배울 것이며, 2013년에 뉴욕시에서 출발한 항공편에 대한 새로운 데이터셋과 dplyr 패키지를 사용한 데이터 변환을 소개할 것입니다.

이 장의 목표는 데이터 프레임을 변환하기 위한 모든 주요 도구의 개요를 제공하는 것입니다. 데이터 프레임의 행(row)과 그 다음 열(column)에 작용하는 함수부터 시작한 다음, 동사들을 결합하는 데 사용하는 중요한 도구인 파이프(pipe)에 대해 더 이야기하기 위해 되돌아올 것입니다. 그런 다음 그룹(group) 작업 기능을 소개할 것입니다. 이 함수들이 실제로 작동하는 것을 보여주는 사례 연구(case study)로 장을 마무리하고, 이후 장에서 특정 유형의 데이터(숫자, 문자열, 날짜)를 파고들기 시작할 때 더 자세한 함수들로 돌아오겠습니다.

## 사전 준비

이 장에서는 tidyverse의 또 다른 핵심 멤버인 dplyr 패키지에 중점을 둘 것입니다. nycflights13 패키지의 데이터를 사용하여 주요 아이디어를 설명하고, 데이터를 이해하는 데 도움을 주기 위해 ggplot2를 사용할 것입니다.

```
library(nycflights13)
library(tidyverse)
#> ── Attaching core tidyverse packages ───────────────────── tidyverse 2.0.0 ──
#> ✔ dplyr     1.1.0.9000     ✔ readr     2.1.4
#> ✔ forcats   1.0.0          ✔ stringr   1.5.0
#> ✔ ggplot2   3.4.1          ✔ tibble    3.1.8
#> ✔ lubridate 1.9.2          ✔ tidyr     1.3.0
#> ✔ purrr     1.0.1
#> ── Conflicts ─────────────────────────────────────── tidyverse_conflicts() ──
#> ✖ dplyr::filter() masks stats::filter()
#> ✖ dplyr::lag()    masks stats::lag()
#> ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all
#>   conflicts to become errors
```

tidyverse를 로드할 때 출력되는 충돌(conflicts) 메시지에 주의 깊게 유의하세요. dplyr이 기본 R의 일부 함수를 덮어쓴다고 알려줍니다. dplyr을 로드한 후 해당 함수들의 기본 버전을 사용하려면 전체 이름(<a href="https://rdrr.io/r/stats/filter.html" class="orm:hideurl"><code>stats::filter()</code></a> 및 <a href="https://rdrr.io/r/stats/lag.html" class="orm:hideurl"><code>stats::lag()</code></a>)을 사용해야 합니다. 지금까지는 대부분의 경우 어떤 패키지에서 왔는지가 중요하지 않았기 때문에 함수가 어떤 패키지에서 왔는지 무시했습니다. 하지만 패키지를 알면 도움말과 관련 함수를 찾는 데 도움이 될 수 있으므로, 패키지가 어떤 함수에서 왔는지 정확히 명시해야 할 때는 R과 동일한 구문인 `packagename::functionname()`을 사용할 것입니다.

## nycflights13

기본적인 dplyr 동사를 탐색하기 위해 <a href="https://rdrr.io/pkg/nycflights13/man/flights.html" class="orm:hideurl"><code>nycflights13::flights</code></a>를 사용할 것입니다. 이 데이터셋에는 2013년 뉴욕시에서 출발한 모든 336,776편의 항공편이 포함되어 있습니다. 이 데이터는 미국 교통 통계국(US Bureau of Transportation Statistics)에서 제공하며 <a href="https://rdrr.io/pkg/nycflights13/man/flights.html" class="orm:hideurl"><code>?flights</code></a>에 문서화되어 있습니다.

```
flights
#> # A tibble: 336,776 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

`flights`는 몇 가지 일반적인 함정(gotchas)을 피하기 위해 tidyverse에서 사용하는 특수한 유형의 데이터 프레임인 티블(tibble)입니다. 티블과 데이터 프레임의 가장 중요한 차이점은 티블이 출력되는 방식입니다. 티블은 대규모 데이터셋을 위해 설계되었으므로 처음 몇 개의 행과 한 화면에 맞는 열(column)만 표시합니다. 모든 것을 볼 수 있는 몇 가지 옵션이 있습니다. RStudio를 사용하는 경우 가장 편리한 것은 대화형의 스크롤 가능하고 필터링 가능한 뷰를 열어주는 `View(flights)`일 것입니다. 그렇지 않으면 `print(flights, width = Inf)`를 사용하여 모든 열을 표시하거나 <a href="https://pillar.r-lib.org/reference/glimpse.html" class="orm:hideurl"><code>glimpse()</code></a>를 사용할 수 있습니다.

```
glimpse(flights)
#> Rows: 336,776
#> Columns: 19
#> $ year           <int> 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013, 2013…
#> $ month          <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
#> $ day            <int> 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1…
#> $ dep_time       <int> 517, 533, 542, 544, 554, 554, 555, 557, 557, 558, 55…
#> $ sched_dep_time <int> 515, 529, 540, 545, 600, 558, 600, 600, 600, 600, 60…
#> $ dep_delay      <dbl> 2, 4, 2, -1, -6, -4, -5, -3, -3, -2, -2, -2, -2, -2,…
#> $ arr_time       <int> 830, 850, 923, 1004, 812, 740, 913, 709, 838, 753, 8…
#> $ sched_arr_time <int> 819, 830, 850, 1022, 837, 728, 854, 723, 846, 745, 8…
#> $ arr_delay      <dbl> 11, 20, 33, -18, -25, 12, 19, -14, -8, 8, -2, -3, 7,…
#> $ carrier        <chr> "UA", "UA", "AA", "B6", "DL", "UA", "B6", "EV", "B6"…
#> $ flight         <int> 1545, 1714, 1141, 725, 461, 1696, 507, 5708, 79, 301…
#> $ tailnum        <chr> "N14228", "N24211", "N619AA", "N804JB", "N668DN", "N…
#> $ origin         <chr> "EWR", "LGA", "JFK", "JFK", "LGA", "EWR", "EWR", "LG…
#> $ dest           <chr> "IAH", "IAH", "MIA", "BQN", "ATL", "ORD", "FLL", "IA…
#> $ air_time       <dbl> 227, 227, 160, 183, 116, 150, 158, 53, 140, 138, 149…
#> $ distance       <dbl> 1400, 1416, 1089, 1576, 762, 719, 1065, 229, 944, 73…
#> $ hour           <dbl> 5, 5, 5, 5, 6, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 6, 6…
#> $ minute         <dbl> 15, 29, 40, 45, 0, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59…
#> $ time_hour      <dttm> 2013-01-01 05:00:00, 2013-01-01 05:00:00, 2013-01-0…
```

두 뷰 모두에서 변수 이름 뒤에는 각 변수의 유형(type)을 알려주는 약어가 있습니다. `<int>`는 정수(integer), `<dbl>`은 double(일명 실수, real numbers), `<chr>`은 문자(character, 일명 문자열, strings), `<dttm>`은 날짜-시간(date-time)을 나타냅니다. 열에서 수행할 수 있는 작업은 열의 "유형"에 매우 의존하기 때문에 이는 중요합니다.

## dplyr 기본 사항

여러분은 이제 데이터 조작(manipulation) 문제의 대부분을 해결할 수 있게 해주는 주요 dplyr 동사(함수)를 배우려고 합니다. 개별적인 차이점을 논의하기 전에 공통점을 명시해 두는 것이 좋습니다.

- 첫 번째 인자는 항상 데이터 프레임입니다.

- 후속 인자들은 일반적으로 따옴표 없이 변수 이름을 사용하여 연산을 수행할 열을 설명합니다.

- 출력은 항상 새로운 데이터 프레임입니다.

각 동사는 한 가지 일을 잘 수행하므로 복잡한 문제를 해결하려면 일반적으로 여러 동사를 결합해야 하며, 파이프 `|>`를 사용하여 이를 수행할 것입니다. 파이프에 대해서는 <a href="#sec-the-pipe" data-type="xref">"파이프"</a>에서 더 논의하겠지만, 간단히 말해서 파이프는 왼쪽에 있는 것을 가져와 오른쪽에 있는 함수로 전달하므로 `x |> f(y)`는 `f(x, y)`와 같고 `x |> f(y) |> g(z)`는 `g(f(x, y), z)`와 같습니다. 파이프를 발음하는 가장 쉬운 방법은 "then(그러고 나서)"입니다. 이를 통해 세부 사항을 아직 배우지 않았더라도 다음 코드를 이해할 수 있습니다.

```
flights |>
  filter(dest == "IAH") |>
  group_by(year, month, day) |>
  summarize(
    arr_delay = mean(arr_delay, na.rm = TRUE)
  )
```

dplyr의 동사들은 작동 대상에 따라 _행(rows)_, _열(columns)_, _그룹(groups)_, _테이블(tables)_ 의 네 가지 그룹으로 구성됩니다. 다음 섹션에서는 행, 열, 그룹에 대한 가장 중요한 동사를 배우게 될 것이며, <a href="ch19.html#chp-joins" data-type="xref">19장</a>에서 테이블에서 작동하는 조인(join) 동사로 돌아오겠습니다. 그럼 뛰어들어 봅시다!

# 행 (Rows)

데이터셋의 행에 작용하는 가장 중요한 동사는 순서를 변경하지 않고 어떤 행을 유지할지 결정하는 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 어떤 행이 존재하는지 변경하지 않고 행의 순서를 변경하는 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>입니다. 두 함수 모두 행에만 영향을 미치고 열은 변경되지 않은 상태로 둡니다. 고유한(unique) 값을 가진 행을 찾는 <a href="https://dplyr.tidyverse.org/reference/distinct.html" class="orm:hideurl"><code>distinct()</code></a>에 대해서도 논의할 것입니다. 하지만 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 달리 선택적으로 열을 수정할 수도 있습니다.

## filter()

<a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>를 사용하면 열의 값을 기준으로 행을 유지할 수 있습니다.<sup><a href="ch03.html#idm44771332966640" id="idm44771332966640-marker" data-type="noteref">1</a></sup> 첫 번째 인자는 데이터 프레임입니다. 두 번째 및 후속 인자는 행을 유지하기 위해 참(true)이어야 하는 조건입니다. 예를 들어 120분(2시간) 이상 늦게 출발한 모든 항공편을 찾을 수 있습니다.

```
flights |>
  filter(dep_delay > 120)
#> # A tibble: 9,723 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      848           1835       853     1001           1950
#> 2  2013     1     1      957            733       144     1056            853
#> 3  2013     1     1     1114            900       134     1447           1222
#> 4  2013     1     1     1540           1338       122     2020           1825
#> 5  2013     1     1     1815           1325       290     2120           1542
#> 6  2013     1     1     1842           1422       260     1958           1535
#> # … with 9,717 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

`>`(크다)뿐만 아니라 `>=`(크거나 같다), `<`(작다), `<=`(작거나 같다), `==`(같다), `!=`(같지 않다)를 사용할 수 있습니다. 또한 `&` 또는 `,`를 사용하여 조건을 결합하여 "그리고(and)"(두 조건을 모두 확인)를 나타내거나 `|`를 사용하여 "또는(or)"(둘 중 하나라도 충족되는지 확인)을 나타낼 수 있습니다.

```
# Flights that departed on January 1
flights |>
  filter(month == 1 & day == 1)
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

# Flights that departed in January or February
flights |>
  filter(month == 1 | month == 2)
#> # A tibble: 51,955 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 51,949 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

`|`와 `==`를 결합할 때 유용한 단축키인 `%in%`이 있습니다. 이 기호는 변수가 오른쪽의 값 중 하나와 같은 행을 유지합니다.

```
# A shorter way to select flights that departed in January or February
flights |>
  filter(month %in% c(1, 2))
#> # A tibble: 51,955 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 51,949 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

이러한 비교 연산자와 논리 연산자에 대해서는 <a href="ch12.html#chp-logicals" data-type="xref">12장</a>에서 자세히 다시 살펴볼 것입니다.

<a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>를 실행하면 dplyr이 필터링 연산을 수행하여 새 데이터 프레임을 만든 다음 이를 출력합니다. dplyr 함수는 입력을 절대 수정하지 않기 때문에 기존 `flights` 데이터셋은 수정하지 않습니다. 결과를 저장하려면 할당 연산자 `<-`를 사용해야 합니다.

```
jan1 <- flights |>
  filter(month == 1 & day == 1)
```

## 흔한 실수

R을 처음 시작할 때 가장 흔히 저지르기 쉬운 실수는 동일성(equality)을 테스트할 때 `==` 대신 `=`를 사용하는 것입니다. <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>는 이런 일이 발생하면 여러분에게 알려줍니다.

```
flights |>
  filter(month = 1)
#> Error in `filter()`:
#> ! We detected a named input.
#> ℹ This usually means that you've used `=` instead of `==`.
#> ℹ Did you mean `month == 1`?
```

또 다른 실수는 영어에서처럼 "or(또는)" 구문을 작성하는 것입니다.

```
flights |>
  filter(month == 1 | 2)
```

이것은 오류를 발생시키지 않는다는 의미에서는 "작동"하지만, `|`가 먼저 조건 `month == 1`을 확인한 다음 조건 `2`를 확인하기 때문에(이는 논리적인 확인 조건이 아닙니다) 여러분이 원하는 작업을 수행하지 않습니다. 여기서 무슨 일이 일어나고 그 이유는 무엇인지에 대해서는 <a href="ch15.html#sec-boolean-operations" data-type="xref">"부울(Boolean) 연산"</a>에서 더 자세히 알아볼 것입니다.

## arrange()

<a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>는 열의 값을 기준으로 행의 순서를 변경합니다. 데이터 프레임과 정렬 기준이 될 열 이름 집합(또는 더 복잡한 표현식)을 사용합니다. 여러 열 이름을 제공하면 각 추가 열이 이전 열 값에서 순위가 같은 경우(ties) 순위를 매기는 데 사용됩니다. 예를 들어, 다음 코드는 출발 시간에 따라 정렬하며 이 정보는 4개의 열에 분산되어 있습니다. 우리는 가장 이른 연도를 먼저 얻고 그 다음 연도 내에서 가장 이른 달을 얻는 식입니다.

```
flights |>
  arrange(year, month, day, dep_time)
#> # A tibble: 336,776 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

<a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a> 안에서 열에 <a href="https://dplyr.tidyverse.org/reference/desc.html" class="orm:hideurl"><code>desc()</code></a>를 사용하면 해당 열을 기준으로 데이터 프레임을 내림차순(큰 것부터 작은 것 순)으로 재정렬할 수 있습니다. 예를 들어, 이 코드는 지연 시간이 가장 긴 것부터 짧은 것 순으로 항공편을 정렬합니다.

```
flights |>
  arrange(desc(dep_delay))
#> # A tibble: 336,776 × 19
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     9      641            900      1301     1242           1530
#> 2  2013     6    15     1432           1935      1137     1607           2120
#> 3  2013     1    10     1121           1635      1126     1239           1810
#> 4  2013     9    20     1139           1845      1014     1457           2210
#> 5  2013     7    22      845           1600      1005     1044           1815
#> 6  2013     4    10     1100           1900       960     1342           2211
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

행의 수는 변경되지 않았음에 주목하세요. 우리는 데이터를 정렬만 하고 있으며 필터링은 하지 않습니다.

## distinct()

# Left

ggplot(mpg, aes(x = drv, color = drv)) +
geom_bar()

# Right

ggplot(mpg, aes(x = drv, fill = drv)) +
geom_bar()

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in20.png" alt="자동차의 구동계 유형을 나타내는 두 개의 막대 차트. 첫 번째 플롯에서 막대는 색상 테두리를 가집니다. 두 번째 플롯에서는 색상으로 채워져 있습니다. 막대의 높이는 각 구동계 범주의 자동차 수에 해당합니다." />
</figure>

`fill` 심미성을 `class`와 같은 다른 변수에 매핑하면 어떻게 되는지 확인해 보세요. 막대가 자동으로 누적(stacked)됩니다. 각 색상 직사각형은 `drv`와 `class`의 조합을 나타냅니다.

```

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar()

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in21.png" alt="자동차의 구동계 유형을 나타내는 분할된(누적) 막대 차트로, 각 막대는 자동차 클래스 색상으로 채워져 있습니다. 막대의 높이는 각 구동 범주의 자동차 수에 해당하고, 색상 세그먼트의 높이는 주어진 구동계 유형 수준 내에서 특정 클래스 수준을 가진 자동차의 수에 비례합니다." />
</figure>

누적은 `position` 인수로 지정된 *위치 조정(position adjustment)*을 사용하여 자동으로 수행됩니다. 누적 막대 차트를 원하지 않으면 `"identity"`, `"dodge"`, `"fill"`의 세 가지 다른 옵션 중 하나를 사용할 수 있습니다.

- `position = "identity"`는 각 객체를 그래프 문맥에 맞는 바로 그 위치에 정확히 배치합니다. 막대 차트의 경우 막대들이 서로 겹치게(overlap) 되므로 별로 유용하지 않습니다. 겹치는 것을 보려면 `alpha`를 작은 값으로 설정하여 막대를 약간 투명하게 만들거나 `fill = NA`로 설정하여 완전히 투명하게 만들어야 합니다.

```

# Left

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar(alpha = 1/5, position = "identity")

# Right

ggplot(mpg, aes(x = drv, color = class)) +
geom_bar(fill = NA, position = "identity")

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in22.png" alt="위치 조정이 identity인 막대 차트로 인해 겹쳐 보이는 플롯." />
</figure>

identity 위치 조정은 2D geom(점)에 더 유용하며 이것이 점의 기본값입니다.

- `position = "fill"`은 누적과 동일하게 작동하지만 누적된 각 막대 세트의 높이를 같게 만듭니다. 이렇게 하면 그룹 간 비율(proportions)을 더 쉽게 비교할 수 있습니다.

- `position = "dodge"`는 겹치는 객체를 서로 *나란히(beside)* 배치합니다. 이렇게 하면 개별 값을 더 쉽게 비교할 수 있습니다.

```

# Left

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar(position = "fill")

# Right

ggplot(mpg, aes(x = drv, fill = class)) +
geom_bar(position = "dodge")

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in23.png" alt="왼쪽 플롯은 비율을 보여주는 fill 위치 조정을, 오른쪽 플롯은 병렬로 배치된 dodge 위치 조정을 보여줍니다." />
</figure>

막대 차트에는 유용하지 않지만 산점도에는 매우 유용할 수 있는 또 다른 유형의 조정이 있습니다. 첫 번째 산점도를 떠올려보세요. 데이터셋에 234개의 관측치가 있음에도 불구하고 플롯에는 126개의 점만 표시된다는 것을 눈치채셨나요?

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in24.png" alt="음의 상관관계를 보여주는 자동차의 고속도로 연비 대 엔진 크기 산점도." />
</figure>

`hwy` 및 `displ`의 기본 값은 점이 그리드에 나타나도록 반올림되어 있으며 많은 점이 서로 겹칩니다. 이 문제를 *오버플로팅(overplotting)*이라고 합니다. 이러한 배치로 인해 데이터의 분포를 파악하기 어렵습니다. 데이터 점이 그래프 전체에 고르게 퍼져 있는지, 아니면 109개의 값을 포함하는 `hwy`와 `displ`의 하나의 특별한 조합이 있는지 알 수 없습니다.

위치 조정을 `"jitter"`로 설정하면 이 그리드 현상(gridding)을 피할 수 있습니다. `position = "jitter"`를 사용하면 각 점에 약간의 무작위 노이즈(random noise)를 추가합니다. 두 점이 동일한 양의 임의 노이즈를 받을 가능성이 없기 때문에 점들이 넓게 퍼지게 됩니다.

```

ggplot(mpg, aes(x = displ, y = hwy)) +
geom_point(position = "jitter")

````

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in25.png" alt="음의 상관관계를 보여주는 자동차의 고속도로 연비 대 엔진 크기의 지터 처리된(Jittered) 산점도." />
</figure>

무작위성을 추가하는 것은 플롯을 개선하는 이상한 방법처럼 보이지만, 작은 축척(small scales)에서는 그래프의 정확성을 떨어뜨리는 대신 큰 축척(large scales)에서는 그래프가 *더 많은 것*을 보여주게 만듭니다. 이것은 매우 유용한 작업이기 때문에 ggplot2는 `geom_point(position = "jitter")`에 대한 단축어인 <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>를 함께 제공합니다.

위치 조정에 대해 더 자세히 알아보려면 각 조정과 연관된 도움말 페이지를 찾아보세요.

- <a href="https://ggplot2.tidyverse.org/reference/position_dodge.html" class="orm:hideurl"><code>?position_dodge</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_stack.html" class="orm:hideurl"><code>?position_fill</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_identity.html" class="orm:hideurl"><code>?position_identity</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_jitter.html" class="orm:hideurl"><code>?position_jitter</code></a>
- <a href="https://ggplot2.tidyverse.org/reference/position_stack.html" class="orm:hideurl"><code>?position_stack</code></a>

## 연습문제

1.  다음 플롯의 문제점은 무엇입니까? 어떻게 개선할 수 있습니까?

    ```
    ggplot(mpg, aes(x = cty, y = hwy)) +
      geom_point()
    ```

2.  두 플롯 간에 차이점이 있다면 무엇입니까? 그 이유는 무엇입니까?

    ```
    ggplot(mpg, aes(x = displ, y = hwy)) +
      geom_point()
    ggplot(mpg, aes(x = displ, y = hwy)) +
      geom_point(position = "identity")
    ```

3.  <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>의 어떤 매개변수가 지터링(jittering)의 양을 제어합니까?

4.  <a href="https://ggplot2.tidyverse.org/reference/geom_jitter.html" class="orm:hideurl"><code>geom_jitter()</code></a>를 <a href="https://ggplot2.tidyverse.org/reference/geom_count.html" class="orm:hideurl"><code>geom_count()</code></a>와 비교 및 대조해 보세요.

5.  <a href="https://ggplot2.tidyverse.org/reference/geom_boxplot.html" class="orm:hideurl"><code>geom_boxplot()</code></a>의 기본 위치 조정은 무엇입니까? 이를 입증하는 `mpg` 데이터셋의 시각화를 만드세요.

# 좌표계 (Coordinate Systems)

좌표계는 아마도 ggplot2에서 가장 복잡한 부분일 것입니다. 기본 좌표계는 데카르트 좌표계(Cartesian coordinate system)로, 여기서 x와 y 위치는 각 점의 위치를 결정하기 위해 독립적으로 작용합니다. 가끔 유용하게 쓰일 수 있는 두 가지 다른 좌표계가 있습니다.

- <a href="https://ggplot2.tidyverse.org/reference/coord_map.html" class="orm:hideurl"><code>coord_quickmap()</code></a>은 지리적 지도의 가로세로 비율(aspect ratio)을 올바르게 설정합니다. 이는 ggplot2로 공간 데이터를 그릴 때 중요합니다. 이 책에서는 지도를 다룰 공간이 부족하지만, *ggplot2: Elegant Graphics for Data Analysis* (Springer)의 [지도 챕터(Maps chapter)](https://oreil.ly/45GHE)에서 자세히 알아볼 수 있습니다.

````

nz <- map_data("nz")

ggplot(nz, aes(x = long, y = lat, group = group)) +
geom_polygon(fill = "white", color = "black")

ggplot(nz, aes(x = long, y = lat, group = group)) +
geom_polygon(fill = "white", color = "black") +
coord_quickmap()

```

![뉴질랜드 경계의 두 지도. 첫 번째 플롯에서는 종횡비가 올바르지 않고 두 번째 플롯에서는 올바릅니다.](D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in26.png)

![뉴질랜드 경계의 두 지도. 첫 번째 플롯에서는 종횡비가 올바르지 않고 두 번째 플롯에서는 올바릅니다.](D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in27.png)

- <a href="https://ggplot2.tidyverse.org/reference/coord_polar.html" class="orm:hideurl"><code>coord_polar()</code></a>는 극좌표(polar coordinates)를 사용합니다. 극좌표는 막대 차트와 콕스콤(Coxcomb) 차트 사이의 흥미로운 연관성을 보여줍니다.

```

bar <- ggplot(data = diamonds) +
geom_bar(
mapping = aes(x = clarity, fill = clarity),
show.legend = FALSE,
width = 1
) +
theme(aspect.ratio = 1)

bar + coord_flip()
bar + coord_polar()

````

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_09in28.png" alt="왼쪽에는 다이아몬드 투명도의 막대 차트가 있고 오른쪽에는 동일한 데이터의 Coxcomb 차트가 있습니다." />
</figure>

## 연습문제

1.  <a href="https://ggplot2.tidyverse.org/reference/coord_polar.html" class="orm:hideurl"><code>coord_polar()</code></a>를 사용하여 누적 막대 차트를 원형 차트(pie chart)로 바꾸세요.

2.  <a href="https://ggplot2.tidyverse.org/reference/coord_map.html" class="orm:hideurl"><code>coord_quickmap()</code></a>과 <a href="https://ggplot2.tidyverse.org/reference/coord_map.html" class="orm:hideurl"><code>coord_map()</code></a>의 차이점은 무엇입니까?

3.  다음 플롯은 도시(city)와 고속도로(highway) 연비(mpg) 간의 관계에 대해 무엇을 알려줍니까? 왜 <a href="https://ggplot2.tidyverse.org/reference/coord_fixed.html" class="orm:hideurl"><code>coord_fixed()</code></a>가 중요합니까? <a href="https://ggplot2.tidyverse.org/reference/geom_abline.html" class="orm:hideurl"><code>geom_abline()</code></a>은 어떤 역할을 합니까?

  ```
  ggplot(data = mpg, mapping = aes(x = cty, y = hwy)) +
    geom_point() +
    geom_abline() +
    coord_fixed()
  ```

# 그래픽의 레이어 문법 (The Layered Grammar of Graphics)

위치 조정, stat, 좌표계 및 패싯 분할을 추가하여 <a href="ch01.html#sec-ggplot2-calls" data-type="xref">"ggplot2 호출"</a>에서 배운 그래프 템플릿을 확장할 수 있습니다.

  ggplot(data = <데이터>) +
    <GEOM_함수>(
       mapping = aes(<매핑>),
       stat = <STAT>,
       position = <위치>
    ) +
    <좌표계_함수> +
    <패싯_함수>

우리의 새로운 템플릿은 템플릿에 나타나는 대괄호로 묶인 단어인 7개의 매개변수를 사용합니다. 실제로 그래프를 만들기 위해 7개의 매개변수를 모두 제공해야 하는 경우는 드문데, 이는 ggplot2가 데이터, 매핑, geom 함수를 제외한 모든 항목에 대해 유용한 기본값을 제공하기 때문입니다.

템플릿에 있는 이 7개의 매개변수는 플롯을 작성하기 위한 공식 시스템(formal system)인 그래픽의 문법(grammar of graphics)을 구성합니다. 그래픽 문법은 *어떠한* 플롯이든 데이터셋, geom, 매핑 세트, stat, 위치 조정, 좌표계, 패싯 분할 체계, 테마의 조합으로 고유하게 설명할 수 있다는 통찰에 기반합니다.

이것이 어떻게 작동하는지 보려면 처음부터 기본 플롯을 작성하는 방법을 생각해 보세요. 먼저 데이터셋으로 시작한 다음 표시하려는 정보로 변환(stat을 사용하여)할 수 있습니다. 다음으로 변환된 데이터에서 각 관측치를 표현할 기하학적 객체를 선택할 수 있습니다. 그런 다음 geom의 심미적 속성을 사용하여 데이터의 변수를 표현할 수 있습니다. 각 변수의 값을 심미성의 수준(levels)에 매핑하게 됩니다. 이러한 단계들은 <a href="#fig-visualization-grammar" data-type="xref">그림 9-3</a>에 예시되어 있습니다. 그런 다음 geom을 배치할 좌표계를 선택하고, 객체의 위치(위치 자체도 심미적 속성임)를 사용하여 x 및 y 변수의 값을 표시합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_0903.png" alt="원시 데이터에서 빈도 표로 이동한 다음 막대의 높이가 빈도를 나타내는 막대 플롯으로 이동하는 단계를 보여주는 그림." />
<h6 id="figure-9-3.-these-are-the-steps-for-going-from-raw-data-to-a-table-of-frequencies-to-a-bar-plot-where-the-heights-of-the-bar-represent-the-frequencies.">그림 9-3. 이것은 원시 데이터에서 빈도 표를 거쳐 막대의 높이가 빈도를 나타내는 막대 플롯을 만들기까지의 단계입니다.</h6>
</figure>

이 시점에는 완전한 그래프가 완성되지만, 좌표계 내에서 geom의 위치를 추가로 조정(위치 조정)하거나 그래프를 하위 플롯(패싯 분할)으로 나눌 수 있습니다. 또한 하나 이상의 추가 레이어를 더하여 플롯을 확장할 수 있으며, 이 경우 각각의 추가 레이어는 자체 데이터셋, geom, 매핑 세트, stat, 위치 조정을 사용하게 됩니다.

이 방법을 사용하면 상상하는 *모든* 플롯을 작성할 수 있습니다. 즉, 이 장에서 배운 코드 템플릿을 사용하여 수십만 개의 고유한 플롯을 만들 수 있습니다.

ggplot2의 이론적 근간에 대해 자세히 알아보고 싶다면, ggplot2의 이론을 자세히 설명하는 과학 논문인 ["A Layered Grammar of Graphics(그래픽의 레이어 문법)"](https://oreil.ly/8fZzE)를 읽어보시는 것을 즐기실 수 있을 것입니다.

# 요약 (Summary)

이 장에서는 간단한 플롯을 만들기 위한 심미성과 기하 구조, 플롯을 부분 집합으로 나누는 패싯, geom이 어떻게 계산되는지 이해하기 위한 통계량(statistics), geom이 겹칠 수 있을 때 위치의 세부 사항을 제어하기 위한 위치 조정, 그리고 `x`와 `y`가 의미하는 바를 근본적으로 변경할 수 있게 해주는 좌표계로 시작하여 그래픽의 레이어 문법(layered grammar of graphics)에 대해 배웠습니다. 우리가 아직 다루지 않은 한 가지 레이어는 테마(theme)인데, 이는 <a href="ch11.html#sec-themes" data-type="xref">"테마(Themes)"</a>에서 소개할 것입니다.

전체 ggplot2 기능에 대한 개요를 얻을 수 있는 매우 유용한 두 가지 리소스는 [ggplot2 치트시트(cheatsheet)](https://oreil.ly/NlKZF)와 [ggplot2 패키지 웹사이트](https://oreil.ly/W6ci8)입니다.

이 장에서 얻어야 할 중요한 교훈은, ggplot2에서 제공하지 않는 geom이 필요하다고 느낄 때 해당 geom을 제공하는 ggplot2 확장 패키지를 만들어 다른 누군가가 당신의 문제를 이미 해결했는지 항상 확인해보는 것이 좋다는 것입니다.
#> #   carrier <chr>, flight <int>, tail_num <chr>, origin <chr>, dest <chr>, …
````

일관성 없이 이름 지어진 열이 많고 하나씩 손으로 고치는 것이 고통스럽다면, 유용한 자동화 정리를 제공하는 <a href="https://rdrr.io/pkg/janitor/man/clean_names.html" class="orm:hideurl"><code>janitor::clean_names()</code></a>를 확인해 보세요.

## relocate()

변수들을 이리저리 옮기려면 <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a>를 사용하세요. 관련 변수들을 함께 모으거나 중요한 변수를 앞으로 옮기고 싶을 수 있습니다. 기본적으로 <a href="https://dplyr.tidyverse.org/reference/relocate.html" class="orm:hideurl"><code>relocate()</code></a>는 변수를 맨 앞으로 옮깁니다.

```
flights |>
  relocate(time_hour, air_time)
#> # A tibble: 336,776 × 19
#>   time_hour           air_time  year month   day dep_time sched_dep_time
#>   <dttm>                 <dbl> <int> <int> <int>    <int>          <int>
#> 1 2013-01-01 05:00:00      227  2013     1     1      517            515
#> 2 2013-01-01 05:00:00      227  2013     1     1      533            529
#> 3 2013-01-01 05:00:00      160  2013     1     1      542            540
#> 4 2013-01-01 05:00:00      183  2013     1     1      544            545
#> 5 2013-01-01 06:00:00      116  2013     1     1      554            600
#> 6 2013-01-01 05:00:00      150  2013     1     1      554            558
#> # … with 336,770 more rows, and 12 more variables: dep_delay <dbl>,
#> #   arr_time <int>, sched_arr_time <int>, arr_delay <dbl>, carrier <chr>, …
```

<a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에서와 마찬가지로 `.before`와 `.after` 인자를 사용하여 놓을 위치를 지정할 수도 있습니다.

```
flights |>
  relocate(year:dep_time, .after = time_hour)
flights |>
  relocate(starts_with("arr"), .before = dep_time)
```

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

```
flights |>
  filter(dest == "IAH") |>
  mutate(speed = distance / air_time * 60) |>
  select(year:day, dep_time, carrier, flight, speed) |>
  arrange(desc(speed))
#> # A tibble: 7,198 × 7
#>    year month   day dep_time carrier flight speed
#>   <int> <int> <int>    <int> <chr>    <int> <dbl>
#> 1  2013     7     9      707 UA         226  522.
#> 2  2013     8    27     1850 UA        1128  521.
#> 3  2013     8    28      902 UA        1711  519.
#> 4  2013     8    28     2122 UA        1022  519.
#> 5  2013     6    11     1628 UA        1178  515.
#> 6  2013     8    27     1017 UA         333  515.
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
      speed = distance / air_time * 60
    ),
    year:day, dep_time, carrier, flight, speed
  ),
  desc(speed)
)
```

또는 중간 객체를 많이 사용할 수도 있습니다.

```
flights1 <- filter(flights, dest == "IAH")
flights2 <- mutate(flights1, speed = distance / air_time * 60)
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
#> # Groups:   month [12]
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group*by()</code></a>는 데이터를 변경하지 않지만 출력을 자세히 살펴보면 월별로 "그룹화되어 있음(grouped by)"을 나타내는 것(`Groups: month [12]`)을 확인할 수 있습니다. 이는 후속 작업들이 이제 "월별로" 작동함을 의미합니다. <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>는 이러한 그룹화된 특성(클래스(\_class*)라고 함)을 데이터 프레임에 추가하며, 이로 인해 데이터에 적용되는 후속 동사의 동작이 변경됩니다.

## summarize()

가장 중요한 그룹화 연산은 요약(summary)으로, 단일 요약 통계를 계산하는 데 사용되는 경우 데이터 프레임을 축소하여 각 그룹당 단일 행을 갖도록 만듭니다. dplyr에서 이 연산은 다음 예제와 같이 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a><sup><a href="ch03.html#idm44771326592976" id="idm44771326592976-marker" data-type="noteref">3</a></sup>에 의해 수행되며, 이는 월별 평균 출발 지연을 계산합니다.

```
flights |>
  group_by(month) |>
  summarize(
    avg_delay = mean(dep_delay)
  )
#> # A tibble: 12 × 2
#>   month avg_delay
#>   <int>     <dbl>
#> 1     1        NA
#> 2     2        NA
#> 3     3        NA
#> 4     4        NA
#> 5     5        NA
#> 6     6        NA
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
#>   month delay
#>   <int> <dbl>
#> 1     1  10.0
#> 2     2  10.8
#> 3     3  13.2
#> 4     4  13.9
#> 5     5  13.0
#> 6     6  20.8
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
#>   month delay     n
#>   <int> <dbl> <int>
#> 1     1  10.0 27004
#> 2     2  10.8 24951
#> 3     3  13.2 28834
#> 4     4  13.9 28330
#> 5     5  13.0 28796
#> 6     6  20.8 28243
#> # … with 6 more rows
```

평균과 개수만으로도 데이터 과학에서 놀라울 정도로 멀리 갈 수 있습니다!

## slice\_ 함수들

각 그룹 내에서 특정 행을 추출할 수 있게 해주는 5개의 편리한 함수가 있습니다.

`df |> slice_head(n = 1)`  
각 그룹의 첫 번째 행을 가져옵니다.

`df |> slice_tail(n = 1)`  
각 그룹의 마지막 행을 가져옵니다.

`df |> slice_min(x, n = 1)`  
`x` 열의 값이 가장 작은 행을 가져옵니다.

`df |> slice_max(x, n = 1)`  
`x` 열의 값이 가장 큰 행을 가져옵니다.

`df |> slice_sample(n = 1)`  
무작위로 하나의 행을 가져옵니다.

`n`을 변경하여 여러 행을 선택하거나 `n =` 대신 `prop = 0.1`을 사용하여 각 그룹에서 행의 10%를 선택할 수 있습니다. 예를 들어 다음 코드는 각 목적지에서 도착 시 가장 많이 지연된 항공편을 찾습니다.

```
flights |>
  group_by(dest) |>
  slice_max(arr_delay, n = 1) |>
  relocate(dest)
#> # A tibble: 108 × 19
#> # Groups:   dest [105]
#>   dest   year month   day dep_time sched_dep_time dep_delay arr_time
#>   <chr> <int> <int> <int>    <int>          <int>     <dbl>    <int>
#> 1 ABQ    2013     7    22     2145           2007        98      132
#> 2 ACK    2013     7    23     1139            800       219     1250
#> 3 ALB    2013     1    25      123           2000       323      229
#> 4 ANC    2013     8    17     1740           1625        75     2042
#> 5 ATL    2013     7    22     2257            759       898      121
#> 6 AUS    2013     7    10     2056           1505       351     2347
#> # … with 102 more rows, and 11 more variables: sched_arr_time <int>,
#> #   arr_delay <dbl>, carrier <chr>, flight <int>, tailnum <chr>, …
```

목적지는 105개지만 여기서는 108개의 행이 반환됩니다. 무슨 일일까요? <a href="https://dplyr.tidyverse.org/reference/slice.html" class="orm:hideurl"><code>slice_min()</code></a>과 <a href="https://dplyr.tidyverse.org/reference/slice.html" class="orm:hideurl"><code>slice_max()</code></a>는 값이 같은 항목(tied values)을 유지하므로 `n = 1`은 가장 높은 값을 가진 모든 행을 제공함을 의미합니다. 각 그룹당 정확히 한 행만 원한다면 `with_ties = FALSE`로 설정할 수 있습니다.

이것은 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>로 최대 지연을 계산하는 것과 유사하지만, 단일 요약 통계 대신 전체 해당 행(또는 동점인 경우 여러 행)을 얻게 됩니다.

## 여러 변수로 그룹화하기

하나 이상의 변수를 사용하여 그룹을 만들 수 있습니다. 예를 들어 각 날짜에 대한 그룹을 만들 수 있습니다.

```
daily <- flights |>
  group_by(year, month, day)
daily
#> # A tibble: 336,776 × 19
#> # Groups:   year, month, day [365]
#>    year month   day dep_time sched_dep_time dep_delay arr_time sched_arr_time
#>   <int> <int> <int>    <int>          <int>     <dbl>    <int>          <int>
#> 1  2013     1     1      517            515         2      830            819
#> 2  2013     1     1      533            529         4      850            830
#> 3  2013     1     1      542            540         2      923            850
#> 4  2013     1     1      544            545        -1     1004           1022
#> 5  2013     1     1      554            600        -6      812            837
#> 6  2013     1     1      554            558        -4      740            728
#> # … with 336,770 more rows, and 11 more variables: arr_delay <dbl>,
#> #   carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …
```

여러 변수로 그룹화된 티블을 요약할 때, 각 요약은 마지막 그룹을 벗겨냅니다(peels off). 돌이켜보면 이 함수를 이렇게 작동하게 만든 것이 좋은 방법은 아니었지만 기존 코드를 망가뜨리지 않고 변경하기는 어렵습니다. 무슨 일이 일어나고 있는지 명확히 알 수 있도록 dplyr은 이 동작을 변경할 수 있는 방법을 알려주는 메시지를 표시합니다.

```
daily_flights <- daily |>
  summarize(n = n())
#> `summarise()` has grouped output by 'year', 'month'. You can override using
#> the `.groups` argument.
```

이 동작에 만족한다면 메시지가 억제되도록 명시적으로 요청할 수 있습니다.

```
daily_flights <- daily |>
  summarize(
    n = n(),
    .groups = "drop_last"
  )
```

```

또는 다른 값을 설정하여 기본 동작을 변경할 수 있습니다. 예를 들어, `"drop"`을 설정하면 모든 그룹화가 해제되고 `"keep"`을 설정하면 동일한 그룹을 유지합니다.

## 그룹화 해제 (Ungrouping)

<a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>를 사용하지 않고 데이터 프레임에서 그룹화를 제거하고 싶을 수도 있습니다. 이는 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>ungroup()</code></a>으로 수행할 수 있습니다.

```

daily |>
ungroup()
#> # A tibble: 336,776 × 19
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

이제 그룹화되지 않은 데이터 프레임을 요약하면 어떻게 되는지 살펴보겠습니다.

```

daily |>
ungroup() |>
summarize(
avg_delay = mean(dep_delay, na.rm = TRUE),
flights = n()
)
#> # A tibble: 1 × 2
#> avg_delay flights
#> <dbl> <int>
#> 1 12.6 336776

```

dplyr은 그룹화되지 않은 데이터 프레임의 모든 행이 하나의 그룹에 속하는 것으로 취급하므로 단일 행이 반환됩니다.

## .by

dplyr 1.1.0에는 연산 단위 그룹화를 위한 새롭고 실험적인 구문인 `.by` 인자가 포함되어 있습니다. <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>ungroup()</code></a>이 사라지는 것은 아니지만, 이제 단일 연산 내에서 그룹화하는 데 `.by` 인자를 사용할 수도 있습니다.

```

flights |>
summarize(
delay = mean(dep_delay, na.rm = TRUE),
n = n(),
.by = month
)

```

여러 변수로 그룹화하고 싶다면:

```

flights |>
summarize(
delay = mean(dep_delay, na.rm = TRUE),
n = n(),
.by = c(origin, dest)
)

````

`.by`는 모든 동사와 함께 작동하며 작업을 마친 후 그룹화 메시지를 억제하기 위해 `.groups` 인자를 사용하거나 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>ungroup()</code></a>을 사용할 필요가 없다는 장점이 있습니다.

이 책을 쓸 당시에는 이 구문이 매우 새로운 것이었기 때문에 이 장에서는 이 구문에 초점을 맞추지 않았습니다. 하지만 많은 가능성을 가지고 있고 인기를 얻을 가능성이 높다고 생각하여 언급하고 싶었습니다. 자세한 내용은 [dplyr 1.1.0 블로그 포스트](https://oreil.ly/ySpmy)에서 알아볼 수 있습니다.

## 연습 문제

1.  평균 지연 시간이 가장 나쁜 항공사는 어디인가요? 도전: 나쁜 공항의 효과와 나쁜 항공사의 효과를 구분할 수 있나요? 왜 그렇게 생각하시나요/아니라고 생각하시나요? (힌트: `flights |> group_by(carrier, dest) |> summarize(n())`에 대해 생각해 보세요.)

2.  각 목적지에서 출발 시 가장 많이 지연된 항공편을 찾으세요.

3.  지연 시간이 하루 종일 어떻게 변하나요? 그래프로 답을 설명해 보세요.

4.  <a href="https://dplyr.tidyverse.org/reference/slice.html" class="orm:hideurl"><code>slice_min()</code></a>과 유사 함수들에 음수의 `n`을 입력하면 어떻게 되나요?

5.  방금 배운 dplyr 동사의 측면에서 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>가 하는 일을 설명해 보세요. <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>의 `sort` 인자는 어떤 역할을 하나요?

6.  아주 작은(tiny) 데이터 프레임이 있다고 가정해 봅시다.

    ```
    df <- tibble(
      x = 1:5,
      y = c("a", "b", "a", "a", "b"),
      z = c("K", "K", "L", "L", "K")
    )
    ```

    1.  출력이 어떻게 보일지 적어보고, 맞았는지 확인한 다음 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>가 하는 일을 설명하세요.

        ```
        df |>
          group_by(y)
        ```

    2.  출력이 어떻게 보일지 적어보고, 맞았는지 확인한 다음 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>가 하는 일을 설명하세요. 또한 (a) 부분의 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>와 어떻게 다른지도 언급하세요.

        ```
        df |>
          arrange(y)
        ```

    3.  출력이 어떻게 보일지 적어보고, 맞았는지 확인한 다음 파이프라인이 하는 일을 설명하세요.

        ```
        df |>
          group_by(y) |>
          summarize(mean_x = mean(x))
        ```

    4.  출력이 어떻게 보일지 적어보고, 맞았는지 확인한 다음 파이프라인이 하는 일을 설명하세요. 그런 다음 메시지가 무엇을 말하는지 언급하세요.

        ```
        df |>
          group_by(y, z) |>
          summarize(mean_x = mean(x))
        ```

    5.  출력이 어떻게 보일지 적어보고, 맞았는지 확인한 다음 파이프라인이 하는 일을 설명하세요. 출력이 (d) 부분의 출력과 어떻게 다른가요?

        ```
        df |>
          group_by(y, z) |>
          summarize(mean_x = mean(x), .groups = "drop")
        ```

    6.  출력이 어떻게 보일지 적어보고, 맞았는지 확인한 다음 각 파이프라인이 하는 일을 설명하세요. 두 파이프라인의 출력은 어떻게 다른가요?

        ```
        df |>
          group_by(y, z) |>
          summarize(mean_x = mean(x))

        df |>
          group_by(y, z) |>
          mutate(mean_x = mean(x))
        ```

# 사례 연구: 집계와 표본 크기 (Case Study: Aggregates and Sample Size)

집계(aggregation)를 수행할 때마다 항상 개수(<a href="https://dplyr.tidyverse.org/reference/context.html" class="orm:hideurl"><code>n()</code></a>)를 포함하는 것이 좋은 생각입니다. 그렇게 하면 아주 적은 양의 데이터에 기반하여 결론을 내리지 않도록 보장할 수 있습니다. Lahman 패키지의 야구 데이터를 사용하여 이를 시연해 보겠습니다. 구체적으로 선수가 안타를 친 비율(`H`)과 공을 쳐서 인플레이시키려 시도한 횟수(`AB`)를 비교해 볼 것입니다.

````

batters <- Lahman::Batting |>
group_by(playerID) |>
summarize(
performance = sum(H, na.rm = TRUE) / sum(AB, na.rm = TRUE),
n = sum(AB, na.rm = TRUE)
)
batters
#> # A tibble: 20,166 × 3
#> playerID performance n
#> <chr> <dbl> <int>
#> 1 aardsda01 0 4
#> 2 aaronha01 0.305 12364
#> 3 aaronto01 0.229 944
#> 4 aasedo01 0 5
#> 5 abadan01 0.0952 21
#> 6 abadfe01 0.111 9
#> # … with 20,160 more rows

```

타자의 기술(타율인 `performance`로 측정)을 공을 칠 기회의 수(타석 수인 `n`으로 측정)에 대해 그래프를 그려보면 두 가지 패턴을 볼 수 있습니다.

- `performance`의 변동성은 타석 수가 적은 선수들 사이에서 더 큽니다. 이 그래프의 형태는 매우 특징적입니다. 평균(또는 다른 요약 통계)을 그룹 크기에 대해 그릴 때마다 표본 크기가 커짐에 따라 변동성이 감소하는 것을 볼 수 있습니다.<sup><a href="ch03.html#idm44771326013808" id="idm44771326013808-marker" data-type="noteref">4</a></sup>

- 기술(`performance`)과 공을 칠 기회(`n`) 사이에는 양의 상관관계(positive correlation)가 있습니다. 팀은 최고의 타자들에게 공을 칠 수 있는 기회를 가장 많이 주고 싶어하기 때문입니다.

```

batters |>
filter(n > 100) |>
ggplot(aes(x = n, y = performance)) +
geom_point(alpha = 1 / 10) +
geom_smooth(se = FALSE)

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_03in01.png" alt="A scatterplot of number of batting performance vs. batting opportunities overlaid with a smoothed line. Average performance increases sharply from 0.2 at when n is 1 to 0.25 when n is ~1000. Average performance continues to increase linearly at a much shallower slope reaching ~0.3 when n is ~15,000." />
</figure>

ggplot2와 dplyr을 결합하는 유용한 패턴에 주목하세요. 데이터셋 처리를 위한 `|>`에서 그래프에 레이어를 추가하기 위한 `+`로 전환해야 한다는 점만 기억하면 됩니다.

이는 순위 지정(ranking)에도 중요한 의미를 갖습니다. 순진하게(naively) `desc(performance)`로 정렬하면, 최고의 타율을 기록한 사람들은 공을 인플레이시키려 매우 적게 시도했다가 우연히 안타를 친 사람들임이 분명합니다. 그들이 반드시 가장 기술이 뛰어난 선수들인 것은 아닙니다.

```

batters |>
arrange(desc(performance))
#> # A tibble: 20,166 × 3
#> playerID performance n
#> <chr> <dbl> <int>
#> 1 abramge01 1 1
#> 2 alberan01 1 1
#> 3 banisje01 1 1
#> 4 bartocl01 1 1
#> 5 bassdo01 1 1
#> 6 birasst01 1 2
#> # … with 20,160 more rows

```

[David Robinson](https://oreil.ly/OjOwY)과 [Evan Miller](https://oreil.ly/wgS7U)의 블로그 게시물에서 이 문제에 대한 훌륭한 설명과 해결 방법을 찾을 수 있습니다.

# 요약

이 장에서는 dplyr이 데이터 프레임 작업을 위해 제공하는 도구에 대해 배웠습니다. 이러한 도구들은 대략 3가지 범주로 나뉩니다. 행을 조작하는 도구(<a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a> 등), 열을 조작하는 도구(<a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a> 등), 그룹을 조작하는 도구(<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a> 등). 이 장에서는 이러한 "전체 데이터 프레임" 도구에 중점을 두었지만, 아직 개별 변수로 할 수 있는 일에 대해서는 많이 배우지 못했습니다. <a href="part03.html#part-transform" data-type="xref">파트 III</a>에서 이 내용으로 다시 돌아올 것이며, 해당 파트의 각 장에서는 특정 유형의 변수에 대한 도구를 제공할 것입니다.

다음 장에서는 여러분 자신이나 다른 사람들이 코드를 읽고 이해하기 쉽도록 코드를 잘 구성하는 코드 스타일의 중요성을 논의하기 위해 워크플로에 다시 초점을 맞출 것입니다.

<sup>[1](ch03.html#idm44771332966640-marker)</sup> 나중에 위치를 기반으로 행을 선택할 수 있는 `slice_*()` 함수군에 대해 배우게 될 것입니다.

<sup>[2](ch03.html#idm44771330064672-marker)</sup> RStudio에서 열이 많은 데이터셋을 보는 가장 쉬운 방법은 <a href="https://rdrr.io/r/utils/View.html" class="orm:hideurl"><code>View()</code></a>라는 점을 기억하세요.

<sup>[3](ch03.html#idm44771326592976-marker)</sup> 영국식 영어를 선호한다면 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarise()</code></a>를 사용할 수도 있습니다.

<sup>[4](ch03.html#idm44771326013808-marker)</sup> \*기침\* 대수의 법칙(the law of large numbers) \*기침\*
```
