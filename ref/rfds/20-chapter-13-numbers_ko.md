# 제13장. 숫자 (Numbers)

# 소개 (Introduction)

숫자 벡터(Numeric vectors)는 데이터 과학의 중추이며, 여러분은 이미 이 책의 앞부분에서 여러 번 사용해 보았습니다. 이제 R에서 숫자 벡터로 할 수 있는 일들을 체계적으로 조사하여, 향후 숫자 벡터와 관련된 어떤 문제에 직면하더라도 잘 해결할 수 있는 기반을 다질 때입니다.

문자열을 가지고 있을 때 숫자를 만드는 몇 가지 도구를 제공하는 것으로 시작한 다음, <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>에 대해 조금 더 자세히 살펴보겠습니다. 그런 다음 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 잘 어울리는 다양한 숫자 변환(numeric transformations)에 대해 깊이 알아볼 것입니다. 여기에는 다른 유형의 벡터에도 적용될 수 있지만 숫자 벡터와 자주 사용되는 더 일반적인 변환들도 포함됩니다. 마지막으로 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 잘 어울리는 요약 함수(summary functions)들을 다루고, 이들이 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 함께 어떻게 쓰일 수 있는지도 보여드리며 마무리하겠습니다.

## 사전 준비 (Prerequisites)

이 장에서는 패키지를 불러오지 않아도 사용할 수 있는 기본 R(base R)의 함수들을 주로 사용합니다. 하지만 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>나 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 같은 tidyverse 함수 안에서 이러한 기본 R 함수들을 사용할 것이기 때문에 tidyverse는 여전히 필요합니다. 이전 장에서와 마찬가지로 nycflights13의 실제 예제와 더불어 <a href="https://rdrr.io/r/base/c.html" class="orm:hideurl"><code>c()</code></a> 및 <a href="https://tibble.tidyverse.org/reference/tribble.html" class="orm:hideurl"><code>tribble()</code></a>을 사용하여 만든 장난감 예제(toy examples)를 사용할 것입니다.

`library``(``tidyverse``)` `library``(``nycflights13``)`

# 숫자 만들기 (Making Numbers)

대부분의 경우, R의 숫자 데이터 유형인 정수(integer) 또는 배정도 실수(double) 중 하나로 이미 기록된 숫자를 얻게 될 것입니다. 그러나 때로는 문자열로 되어 있는 경우를 접하게 될 텐데, 열 헤더에서 피벗팅(pivoting)하여 만들었거나 데이터 가져오기 과정에서 뭔가 잘못되었기 때문일 수 있습니다.

readr는 문자열을 숫자로 파싱하는 데 유용한 두 가지 함수인 <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>parse_double()</code></a>과 <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>parse_number()</code></a>를 제공합니다. 숫자가 문자열로 작성되어 있을 때는 <a href="https://readr.tidyverse.org/reference/parse_atomic.html" class="orm:hideurl"><code>parse_double()</code></a>을 사용하세요.

`x` `<-` `c``(``"1.2"``,` `"5.6"``,` `"1e3"``)` `parse_double``(``x``)` `#> [1] 1.2 5.6 1000.0`

무시하고 싶은 숫자가 아닌 텍스트가 문자열에 포함되어 있을 때는 <a href="https://readr.tidyverse.org/reference/parse_number.html" class="orm:hideurl"><code>parse_number()</code></a>를 사용하세요. 이것은 특히 통화 데이터와 백분율에 유용합니다.

`x` `<-` `c``(``"$1,234"``,` `"USD 3,513"``,` `"59%"``)` `parse_number``(``x``)` `#> [1] 1234 3513 59`

# 개수 세기 (Counts)

단순한 개수(counts)와 약간의 기초 산술만으로 얼마나 많은 데이터 과학을 할 수 있는지 놀라울 정도입니다. 그래서 dplyr은 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>를 사용하여 개수 세기를 최대한 쉽게 만들고자 노력합니다. 이 함수는 분석 중에 빠른 탐색과 확인을 위해 매우 유용합니다.

`flights` `|>` `count``(``dest``)` `#> # A tibble: 105 × 2` `#> dest n` `#> <chr> <int>` `#> 1 ABQ 254` `#> 2 ACK 265` `#> 3 ALB 439` `#> 4 ANC 8` `#> 5 ATL 17215` `#> 6 AUS 2439` `#> # … with 99 more rows`

(<a href="ch04.html#chp-workflow-style" data-type="xref">제4장</a>의 조언에도 불구하고, 계산이 예상대로 작동하는지 빠르게 확인하기 위해 콘솔에서 주로 사용되기 때문에 우리는 보통 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>를 한 줄에 작성합니다.)

빈번한 값을 확인하려면 `sort = TRUE`를 추가하세요.

`flights` `|>` `count``(``dest``,` `sort` `=` `TRUE``)` `#> # A tibble: 105 × 2` `#> dest n` `#> <chr> <int>` `#> 1 ORD 17283` `#> 2 ATL 17215` `#> 3 LAX 16174` `#> 4 BOS 15508` `#> 5 MCO 14082` `#> 6 CLT 14064` `#> # … with 99 more rows`

그리고 모든 값을 보고 싶다면 `|> View()` 또는 `|> print(n = Inf)`를 사용할 수 있다는 점을 기억하세요.

<a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>, <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>, <a href="https://dplyr.tidyverse.org/reference/context.html" class="orm:hideurl"><code>n()</code></a>을 사용하여 "수동으로(by hand)" 동일한 계산을 수행할 수 있습니다. 이것은 동시에 다른 요약들도 계산할 수 있게 해주기 때문에 유용합니다.

`flights` `|>` `group_by``(``dest``)` `|>` `summarize``(` `n` `=` `n``(),` `delay` `=` `mean``(``arr_delay``,` `na.rm` `=` `TRUE``)` `)` `#> # A tibble: 105 × 3` `#> dest n delay` `#> <chr> <int> <dbl>` `#> 1 ABQ 254 4.38` `#> 2 ACK 265 4.85` `#> 3 ALB 439 14.4 ` `#> 4 ANC 8 -2.5 ` `#> 5 ATL 17215 11.3 ` `#> 6 AUS 2439 6.02` `#> # … with 99 more rows`

<a href="https://dplyr.tidyverse.org/reference/context.html" class="orm:hideurl"><code>n()</code></a>은 인자를 취하지 않고 대신 "현재(current)" 그룹에 대한 정보에 접근하는 특별한 요약 함수입니다. 이것은 dplyr 동사(verbs) 내부에서만 작동한다는 것을 의미합니다.

`n``()` ``#> Error in `n()`:`` ``#> ! Must only be used inside data-masking verbs like `mutate()`,`` ``#> `filter()`, and `group_by()`.``

유용하게 사용할 수 있는 <a href="https://dplyr.tidyverse.org/reference/context.html" class="orm:hideurl"><code>n()</code></a>과 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>의 몇 가지 변형이 있습니다.

- `n_distinct(x)`는 하나 이상의 변수의 고유한(distinct/unique) 값의 개수를 셉니다. 예를 들어, 어떤 목적지에 많은 항공사가 운항하는지 알아낼 수 있습니다.

  `flights` `|>` `group_by``(``dest``)` `|>` `summarize``(``carriers` `=` `n_distinct``(``carrier``))` `|>` `arrange``(``desc``(``carriers``))` `#> # A tibble: 105 × 2` `#> dest carriers` `#> <chr> <int>` `#> 1 ATL 7` `#> 2 BOS 7` `#> 3 CLT 7` `#> 4 ORD 7` `#> 5 TPA 7` `#> 6 AUS 6` `#> # … with 99 more rows`

- 가중된 개수(weighted count)는 합계(sum)입니다. 예를 들어, 각 비행기가 날아간 마일 수를 "셀 수(count)" 있습니다.

  `flights` `|>` `group_by``(``tailnum``)` `|>` `summarize``(``miles` `=` `sum``(``distance``))` `#> # A tibble: 4,044 × 2` `#> tailnum miles` `#> <chr> <dbl>` `#> 1 D942DN 3418` `#> 2 N0EGMQ 250866` `#> 3 N10156 115966` `#> 4 N102UW 25722` `#> 5 N103US 24619` `#> 6 N104UW 25157` `#> # … with 4,038 more rows`
  가중된 개수 세기는 흔한 문제이므로, <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>에는 같은 작업을 수행하는 `wt` 인자가 있습니다.

  `flights` `|>` `count``(``tailnum``,` `wt` `=` `distance``)`

- <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>과 <a href="https://rdrr.io/r/base/NA.html" class="orm:hideurl"><code>is.na()</code></a>를 조합하여 결측값의 개수를 셀 수 있습니다. `flights` 데이터세트에서 이것은 취소된 항공편을 나타냅니다.

  `flights` `|>` `group_by``(``dest``)` `|>` `summarize``(``n_cancelled` `=` `sum``(``is.na``(``dep_time``)))` `#> # A tibble: 105 × 2` `#> dest n_cancelled` `#> <chr> <int>` `#> 1 ABQ 0` `#> 2 ACK 0` `#> 3 ALB 20` `#> 4 ANC 0` `#> 5 ATL 317` `#> 6 AUS 21` `#> # … with 99 more rows`

## 연습문제 (Exercises)

1. 주어진 변수의 결측값이 있는 행의 수를 세기 위해 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>를 어떻게 사용할 수 있습니까?
2. 다음 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a> 호출을 대신 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>, <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>, <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>arrange()</code></a>를 사용하도록 확장하세요.
   1. `flights |> count(dest, sort = TRUE)`
   2. `flights |> count(tailnum, wt = distance)`

# 숫자 변환 (Numeric Transformations)

변환 함수(Transformation functions)는 그 출력이 입력과 동일한 길이이기 때문에 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 잘 작동합니다. 대다수의 변환 함수는 기본 R에 이미 내장되어 있습니다. 그 모든 것을 나열하는 것은 비실용적이므로, 이 섹션에서는 유용한 것들을 보여줄 것입니다. 한 예로, R은 여러분이 상상할 수 있는 모든 삼각 함수(trigonometric functions)를 제공하지만, 데이터 과학에서 거의 필요하지 않기 때문에 여기에 나열하지 않았습니다.

## 산술 및 리사이클링 규칙 (Arithmetic and Recycling Rules)

우리는 <a href="ch02.html#chp-workflow-basics" data-type="xref">제2장</a>에서 산술(`+`, `-`, `*`, `/`, `^`)의 기초를 소개했고 그 이후로 많이 사용해 왔습니다. 이 함수들은 초등학교에서 배운 대로 작동하기 때문에 많은 설명이 필요하지 않습니다. 하지만 우리는 왼쪽과 오른쪽의 길이가 다를 때 어떤 일이 일어나는지 결정하는 *리사이클링 규칙(recycling rules)*에 대해 간략하게 이야기해야 합니다. 이것은 `flights |> mutate(air_time = air_time / 60)`과 같은 연산에서 중요한데, `/`의 왼쪽에는 336,776개의 숫자가 있지만 오른쪽에는 단 하나만 있기 때문입니다.

R은 짝이 맞지 않는 길이를 처리할 때 짧은 벡터를 *재활용(recycling)*하거나 반복합니다. 데이터 프레임 외부에 몇 개의 벡터를 만들면 이러한 작동을 더 쉽게 볼 수 있습니다.

`x` `<-` `c``(``1``,` `2``,` `10``,` `20``)` `x` `/` `5` `#> [1] 0.2 0.4 2.0 4.0` `# is shorthand for` `x` `/` `c``(``5``,` `5``,` `5``,` `5``)` `#> [1] 0.2 0.4 2.0 4.0`

일반적으로, 단일 숫자(즉, 길이가 1인 벡터)만 재활용되기를 원하겠지만, R은 더 짧은 길이의 어떤 벡터든 재활용합니다. R은 긴 벡터가 짧은 벡터의 배수가 아닌 경우 보통(항상 그런 것은 아니지만) 경고를 표시합니다.

`x` `*` `c``(``1``,` `2``)` `#> [1] 1 4 10 40` `x` `*` `c``(``1``,` `2``,` `3``)` `#> Warning in x * c(1, 2, 3): longer object length is not a multiple of shorter` `#> object length` `#> [1] 1 4 30 20`

이러한 리사이클링 규칙은 논리적 비교(`==`, `<`, `<=`, `>`, `>=`, `!=`)에도 적용되며, `%in%` 대신 실수로 `==`를 사용하고 데이터 프레임의 행 수가 운 나쁘게 맞아떨어지면 놀라운 결과를 낳을 수 있습니다. 예를 들어, 1월과 2월의 모든 항공편을 찾으려고 시도하는 이 코드를 살펴보세요.

`flights` `|>` `filter``(``month` `==` `c``(``1``,` `2``))` `#> # A tibble: 25,977 × 19` `#> year month day dep_time sched_dep_time dep_delay arr_time sched_arr_time` `#> <int> <int> <int> <int> <int> <dbl> <int> <int>` `#> 1 2013 1 1 517 515 2 830 819` `#> 2 2013 1 1 542 540 2 923 850` `#> 3 2013 1 1 554 600 -6 812 837` `#> 4 2013 1 1 555 600 -5 913 854` `#> 5 2013 1 1 557 600 -3 838 846` `#> 6 2013 1 1 558 600 -2 849 851` `#> # … with 25,971 more rows, and 11 more variables: arr_delay <dbl>,` `#> # carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …`

이 코드는 오류 없이 실행되지만, 여러분이 원하는 것을 반환하지 않습니다. 리사이클링 규칙 때문에 1월에 출발한 항공편은 홀수 행에서 찾고, 2월에 출발한 항공편은 짝수 행에서 찾습니다. 불행하게도 `flights`의 행 수가 짝수이기 때문에 경고 메시지도 나타나지 않습니다.

이러한 유형의 조용한 실패(silent failure)로부터 여러분을 보호하기 위해, 대부분의 tidyverse 함수들은 단일 값만을 재활용하는 더 엄격한 형태의 리사이클링을 사용합니다. 하지만 안타깝게도 이 경우나 다른 많은 경우에는 도움이 되지 않는데, 핵심 연산이 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>가 아닌 기본 R 함수인 `==`에 의해 수행되기 때문입니다.

## 최솟값과 최댓값 (Minimum and Maximum)

산술 함수들은 변수 쌍으로 작업합니다. 밀접하게 관련된 두 가지 함수로 <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>pmin()</code></a>과 <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>pmax()</code></a>가 있는데, 두 개 이상의 변수가 주어지면 각 행에서 작거나 큰 값을 반환합니다.

`df` `<-` `tribble``(` `~``x``,` `~``y``,` `1``,` `3``,` `5``,` `2``,` `7``,` `NA``,` `)` `df` `|>` `mutate``(` `min` `=` `pmin``(``x``,` `y``,` `na.rm` `=` `TRUE``),` `max` `=` `pmax``(``x``,` `y``,` `na.rm` `=` `TRUE``)` `)` `#> # A tibble: 3 × 4` `#> x y min max` `#> <dbl> <dbl> <dbl> <dbl>` `#> 1 1 3 1 3` `#> 2 5 2 2 5` `#> 3 7 NA 7 7`

여러 관측치를 가져와 하나의 값만 반환하는 요약 함수인 <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>min()</code></a> 및 <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>max()</code></a>와는 다릅니다. 모든 최솟값과 모든 최댓값이 같게 나오면 잘못된 형태의 함수를 사용했음을 알 수 있습니다.

`df` `|>` `mutate``(` `min` `=` `min``(``x``,` `y``,` `na.rm` `=` `TRUE``),` `max` `=` `max``(``x``,` `y``,` `na.rm` `=` `TRUE``)` `)` `#> # A tibble: 3 × 4` `#> x y min max` `#> <dbl> <dbl> <dbl> <dbl>` `#> 1 1 3 1 7` `#> 2 5 2 1 7` `#> 3 7 NA 1 7`

## 모듈러 산술 (Modular Arithmetic)

모듈러 산술(Modular arithmetic)은 여러분이 소수점을 배우기 전에 하던 수학, 즉 정수 몫과 나머지를 구하는 나눗셈을 부르는 기술적인 이름입니다. R에서 `%/%`는 정수 나눗셈(integer division)을 수행하고, `%%`는 나머지(remainder)를 계산합니다.

`1``:``10` `%/%` `3` `#> [1] 0 0 1 1 1 2 2 2 3 3` `1``:``10` `%%` `3` `#> [1] 1 2 0 1 2 0 1 2 0 1`

모듈러 산술은 `flights` 데이터세트에서 유용한데, `sched_dep_time` 변수를 분해하여 `hour`와 `minute`로 나눌 때 이를 사용할 수 있기 때문입니다.

`flights` `|>` `mutate``(` `hour` `=` `sched_dep_time` `%/%` `100``,` `minute` `=` `sched_dep_time` `%%` `100``,` `.keep` `=` `"used"` `)` `#> # A tibble: 336,776 × 3` `#> sched_dep_time hour minute` `#> <int> <dbl> <dbl>` `#> 1 515 5 15` `#> 2 529 5 29` `#> 3 540 5 40` `#> 4 545 5 45` `#> 5 600 6 0` `#> 6 558 5 58` `#> # … with 336,770 more rows`

우리는 이것을 <a href="ch12.html#sec-logical-summaries" data-type="xref">"요약"</a>에서 다룬 `mean(is.na(x))` 트릭과 결합하여, 하루 동안 취소된 항공편의 비율이 어떻게 달라지는지 볼 수 있습니다. 결과는 <a href="#fig-prop-cancelled" data-type="xref">그림 13-1</a>에 나와 있습니다.

`flights` `|>` `group_by``(``hour` `=` `sched_dep_time` `%/%` `100``)` `|>` `summarize``(``prop_cancelled` `=` `mean``(``is.na``(``dep_time``)),` `n` `=` `n``())` `|>` `filter``(``hour` `>` `1``)` `|>` `ggplot``(``aes``(``x` `=` `hour``,` `y` `=` `prop_cancelled``))` `+` `geom_line``(``color` `=` `"grey50"``)` `+` `geom_point``(``aes``(``size` `=` `n``))`

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1301.png" alt="A line plot showing how proportion of cancelled flights changes over the course of the day. The proportion starts low at around 0.5% at 6 a.m., then steadily increases over the course of the day until peaking at 4% at 7 p.m. The proportion of cancelled flights then drops rapidly getting down to around 1% by midnight." />
<h6 id="figure-13-1.-a-line-plot-with-scheduled-departure-hour-on-the-x-axis-and-proportion-of-cancelled-flights-on-the-y-axis.-cancellations-seem-to-accumulate-over-the-course-of-the-day-until-8-p.m.-and-very-late-flights-are-much-less-likely-to-be-cancelled.">그림 13-1. x축에는 예정 출발 시간을, y축에는 결항된 항공편의 비율을 나타낸 꺾은선형 차트. 오후 8시까지 하루 동안 결항이 누적되는 것으로 보이며, 아주 늦은 시간대의 항공편은 결항될 가능성이 훨씬 낮습니다.</h6>
</figure>

## 로그 (Logarithms)

로그(Logarithms)는 여러 자릿수(orders of magnitude)에 걸쳐 있는 데이터를 다루고 지수적인 성장을 선형적인 성장으로 변환하는 데 엄청나게 유용한 변환입니다. R에서는 <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log()</code></a>(자연로그, 밑이 e), <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log2()</code></a>(밑이 2), <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log10()</code></a>(밑이 10)의 세 가지 로그 중에서 선택할 수 있습니다. <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log2()</code></a> 또는 <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log10()</code></a>을 사용하는 것을 권장합니다. 로그 척도에서의 1의 차이는 원래 척도에서의 두 배(doubling)에 해당하고, -1의 차이는 절반(halving)에 해당하기 때문에 <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log2()</code></a>는 해석하기 쉽습니다. 반면 <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log10()</code></a>은, 예를 들어 3이 10^3 = 1000이 되므로 역변환하기 쉽습니다. <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log()</code></a>의 역함수는 <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>exp()</code></a>입니다; <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log2()</code></a> 또는 <a href="https://rdrr.io/r/base/Log.html" class="orm:hideurl"><code>log10()</code></a>의 역함수를 계산하려면 `2^` 또는 `10^`을 사용해야 합니다.

## 반올림 (Rounding)

숫자를 가까운 정수로 반올림하려면 `round(x)`를 사용하세요.

`round``(``123.456``)` `#> [1] 123`

두 번째 인자인 `digits`를 통해 반올림의 정밀도를 제어할 수 있습니다. `round(x, digits)`는 가까운 `10^-n` 단위로 반올림하므로, `digits = 2`는 가까운 0.01 단위로 반올림합니다. 이 정의는 `round(x, -3)`이 가까운 천 단위로 반올림한다는 것을 의미하기 때문에 유용하며, 실제로도 그렇게 작동합니다.

`round``(``123.456``,` `2``)` `# two digits` `#> [1] 123.46` `round``(``123.456``,` `1``)` `# one digit` `#> [1] 123.5` `round``(``123.456``,` `-1``)` `# round to nearest ten` `#> [1] 120` `round``(``123.456``,` `-2``)` `# round to nearest hundred` `#> [1] 100`

얼핏 보기에 놀랍게 여겨질 수 있는 <a href="https://rdrr.io/r/base/Round.html" class="orm:hideurl"><code>round()</code></a>의 한 가지 기이한 점이 있습니다.

`round``(``c``(``1.5``,` `2.5``))` `#> [1] 2 2`

<a href="https://rdrr.io/r/base/Round.html" class="orm:hideurl"><code>round()</code></a>는 "가까운 짝수로 반올림(round half to even)" 또는 은행가 반올림(Banker's rounding)으로 알려진 방식을 사용합니다. 숫자가 두 정수 사이의 정확히 절반에 있을 때, 그것은 _짝수(even)_ 정수 쪽으로 반올림됩니다. 이 방식은 반올림의 편향을 없애주기 때문에 좋은 전략입니다. 0.5로 끝나는 수의 절반은 올림되고, 절반은 내림됩니다.

<a href="https://rdrr.io/r/base/Round.html" class="orm:hideurl"><code>round()</code></a>는 항상 내림하는 <a href="https://rdrr.io/r/base/Round.html" class="orm:hideurl"><code>floor()</code></a>, 그리고 항상 올림하는 <a href="https://rdrr.io/r/base/Round.html" class="orm:hideurl"><code>ceiling()</code></a>과 짝을 이룹니다.

`x` `<-` `123.456` `floor``(``x``)` `#> [1] 123` `ceiling``(``x``)` `#> [1] 124`

이 함수들은 `digits` 인자가 없기 때문에, 대신 축소(scale down)하고, 반올림한 다음, 다시 확대(scale back up)할 수 있습니다.

`# Round down to nearest two digits` `floor``(``x` `/` `0.01``)` `*` `0.01` `#> [1] 123.45` `# Round up to nearest two digits` `ceiling``(``x` `/` `0.01``)` `*` `0.01` `#> [1] 123.46`

다른 숫자의 배수로 [`round()`](https://oreil.ly/YcbwN)를 하고 싶다면 동일한 기술을 사용할 수 있습니다.

`# Round to nearest multiple of 4` `round``(``x` `/` `4``)` `*` `4` `#> [1] 124` `# Round to nearest 0.25` `round``(``x` `/` `0.25``)` `*` `0.25` `#> [1] 123.5`

## 숫자를 범위로 나누기 (Cutting Numbers into Ranges)

숫자 벡터를 분리된 구간(버킷 또는 *bin*이라고도 함)으로 나누려면 <a href="https://rdrr.io/r/base/cut.html" class="orm:hideurl"><code>cut()</code></a><sup><a href="ch13.html#idm44771298681152" id="idm44771298681152-marker" data-type="noteref">1</a></sup>을 사용하세요.

`x` `<-` `c``(``1``,` `2``,` `5``,` `10``,` `15``,` `20``)` `cut``(``x``,` `breaks` `=` `c``(``0``,` `5``,` `10``,` `15``,` `20``))` `#> [1] (0,5] (0,5] (0,5] (5,10] (10,15] (15,20]` `#> Levels: (0,5] (5,10] (10,15] (15,20]`

눈금(`breaks`)의 간격이 균일할 필요는 없습니다.

`cut``(``x``,` `breaks` `=` `c``(``0``,` `5``,` `10``,` `100``))` `#> [1] (0,5] (0,5] (0,5] (5,10] (10,100] (10,100]` `#> Levels: (0,5] (5,10] (10,100]`

선택적으로 고유한 `labels`를 제공할 수 있습니다. `labels`의 개수는 `breaks`의 개수보다 하나 적어야 한다는 점에 유의하세요.

`cut``(``x``,` `breaks` `=` `c``(``0``,` `5``,` `10``,` `15``,` `20``),` `labels` `=` `c``(``"sm"``,` `"md"``,` `"lg"``,` `"xl"``)` `)` `#> [1] sm sm sm md lg xl` `#> Levels: sm md lg xl`

눈금 범위 밖의 값은 `NA`가 됩니다.

`y` `<-` `c``(``NA``,` `-10``,` `5``,` `10``,` `30``)` `cut``(``y``,` `breaks` `=` `c``(``0``,` `5``,` `10``,` `15``,` `20``))` `#> [1] <NA> <NA> (0,5] (5,10] <NA> ` `#> Levels: (0,5] (5,10] (10,15] (15,20]`

구간이 `[a, b)` 또는 `(a, b]`가 될지, 낮은 구간이 `[a, b]`가 될지 여부를 제어하는 `right` 및 `include.lowest`와 같은 다른 유용한 인자들에 대해서는 문서를 참조하세요.

## 누적 및 이동 집계 (Cumulative and Rolling Aggregates)

기본 R은 진행형(running) 또는 누적 합계, 곱, 최솟값, 최댓값을 구하기 위해 <a href="https://rdrr.io/r/base/cumsum.html" class="orm:hideurl"><code>cumsum()</code></a>, <a href="https://rdrr.io/r/base/cumsum.html" class="orm:hideurl"><code>cumprod()</code></a>, <a href="https://rdrr.io/r/base/cumsum.html" class="orm:hideurl"><code>cummin()</code></a>, <a href="https://rdrr.io/r/base/cumsum.html" class="orm:hideurl"><code>cummax()</code></a>를 제공합니다. dplyr은 누적 평균을 구하기 위해 <a href="https://dplyr.tidyverse.org/reference/cumall.html" class="orm:hideurl"><code>cummean()</code></a>을 제공합니다. 누적 합계(Cumulative sums)가 실제로 많이 사용되는 경향이 있습니다.

`x` `<-` `1``:``10` `cumsum``(``x``)` `#> [1] 1 3 6 10 15 21 28 36 45 55`

더 복잡한 롤링(rolling) 또는 슬라이딩(sliding) 집계가 필요하다면 [slider 패키지](https://oreil.ly/XPnjF)를 시도해 보세요.

## 연습문제 (Exercises)

1. <a href="#fig-prop-cancelled" data-type="xref">그림 13-1</a>을 생성하는 데 사용된 코드의 각 줄이 무엇을 하는지 말로 설명해 보세요.
2. R은 어떤 삼각 함수(trigonometric functions)를 제공합니까? 이름을 추측해 보고 문서를 찾아보세요. 각도(degrees)를 사용합니까 아니면 라디안(radians)을 사용합니까?
3. 현재 `dep_time`과 `sched_dep_time`은 보기에는 편하지만 실제로는 연속적인 숫자가 아니기 때문에 계산에 사용하기는 어렵습니다. 다음 코드를 실행해 보면 기본적인 문제를 확인할 수 있습니다. 매 시간 사이에 공백(gap)이 존재합니다.

   `flights` `|>` `filter``(``month` `==` `1``,` `day` `==` `1``)` `|>` `ggplot``(``aes``(``x` `=` `sched_dep_time``,` `y` `=` `dep_delay``))` `+` `geom_point``()`
   이것들을 시간을 더 잘 반영하는 표현(분수 시간(fractional hours)이나 자정 이후의 분(minutes since midnight))으로 변환하세요.

4. `dep_time`과 `arr_time`을 가까운 5분 단위로 반올림하세요.

# 일반적인 변환 (General Transformations)

다음 섹션들에서는 숫자 벡터와 자주 사용되지만 다른 모든 열 유형에도 적용할 수 있는 몇 가지 일반적인 변환에 대해 설명합니다.

## 순위 (Ranks)

dplyr은 SQL에서 영감을 받은 여러 순위 지정(ranking) 함수를 제공하지만, 항상 <a href="https://dplyr.tidyverse.org/reference/row_number.html" class="orm:hideurl"><code>dplyr::min_rank()</code></a>로 시작해야 합니다. 이 함수는 동점(ties)을 처리할 때 전형적인 방법(1등, 2등, 2등, 4등)을 사용합니다.

`x` `<-` `c``(``1``,` `2``,` `2``,` `3``,` `4``,` `NA``)` `min_rank``(``x``)` `#> [1] 1 2 2 4 5 NA`

작은 값이 낮은 순위(1위)를 갖는다는 점에 유의하세요. 큰 값에 낮은 순위를 부여하려면 `desc(x)`를 사용합니다.

`min_rank``(``desc``(``x``))` `#> [1] 5 3 3 2 1 NA`

<a href="https://dplyr.tidyverse.org/reference/row_number.html" class="orm:hideurl"><code>min_rank()</code></a>가 여러분이 필요로 하는 기능을 하지 못한다면, 그 변형들인 <a href="https://dplyr.tidyverse.org/reference/row_number.html" class="orm:hideurl"><code>dplyr::row_number()</code></a>, <a href="https://dplyr.tidyverse.org/reference/row_number.html" class="orm:hideurl"><code>dplyr::dense_rank()</code></a>, <a href="https://dplyr.tidyverse.org/reference/percent_rank.html" class="orm:hideurl"><code>dplyr::percent_rank()</code></a>, <a href="https://dplyr.tidyverse.org/reference/percent_rank.html" class="orm:hideurl"><code>dplyr::cume_dist()</code></a>를 살펴보세요. 자세한 내용은 문서를 참조하세요.

`df` `<-` `tibble``(``x` `=` `x``)` `df` `|>` `mutate``(` `row_number` `=` `row_number``(``x``),` `dense_rank` `=` `dense_rank``(``x``),` `percent_rank` `=` `percent_rank``(``x``),` `cume_dist` `=` `cume_dist``(``x``)` `)` `#> # A tibble: 6 × 5` `#> x row_number dense_rank percent_rank cume_dist` `#> <dbl> <int> <int> <dbl> <dbl>` `#> 1 1 1 1 0 0.2` `#> 2 2 2 2 0.25 0.6` `#> 3 2 3 2 0.25 0.6` `#> 4 3 4 3 0.75 0.8` `#> 5 4 5 4 1 1 ` `#> 6 NA NA NA NA NA`

기본 R의 <a href="https://rdrr.io/r/base/rank.html" class="orm:hideurl"><code>rank()</code></a> 함수에서 적절한 `ties.method` 인자를 선택하여 동일한 결과의 상당수를 얻을 수 있습니다. 또한 `NA`를 `NA`로 유지하기 위해 `na.last = "keep"`을 설정하고 싶을 것입니다.

<a href="https://dplyr.tidyverse.org/reference/row_number.html" class="orm:hideurl"><code>row_number()</code></a>는 dplyr 동사(verb) 내부에서 사용할 때 아무 인자 없이도 사용할 수 있습니다. 이 경우 "현재" 행의 번호를 알려줍니다. `%%` 또는 `%/%`와 결합하면, 데이터를 비슷한 크기의 그룹으로 나누는 데 유용한 도구가 될 수 있습니다.

`df` `<-` `tibble``(``id` `=` `1``:``10``)` `df` `|>` `mutate``(` `row0` `=` `row_number``()` `-` `1``,` `three_groups` `=` `row0` `%%` `3``,` `three_in_each_group` `=` `row0` `%/%` `3` `)` `#> # A tibble: 10 × 4` `#> id row0 three_groups three_in_each_group` `#> <int> <dbl> <dbl> <dbl>` `#> 1 1 0 0 0` `#> 2 2 1 1 0` `#> 3 3 2 2 0` `#> 4 4 3 0 1` `#> 5 5 4 1 1` `#> 6 6 5 2 1` `#> # … with 4 more rows`

## 오프셋 (Offsets)

<a href="https://dplyr.tidyverse.org/reference/lead-lag.html" class="orm:hideurl"><code>dplyr::lead()</code></a>와 <a href="https://dplyr.tidyverse.org/reference/lead-lag.html" class="orm:hideurl"><code>dplyr::lag()</code></a>를 사용하면 "현재(current)" 값의 직전이나 직후 값을 참조할 수 있습니다. 이들은 시작이나 끝에 `NA`를 덧붙인 입력과 동일한 길이의 벡터를 반환합니다.

`x` `<-` `c``(``2``,` `5``,` `11``,` `11``,` `19``,` `35``)` `lag``(``x``)` `#> [1] NA 2 5 11 11 19` `lead``(``x``)` `#> [1] 5 11 11 19 35 NA`

- `x - lag(x)`는 현재 값과 이전 값의 차이를 알려줍니다.

  `x` `-` `lag``(``x``)` `#> [1] NA 3 6 0 8 16`

- `x == lag(x)`는 현재 값이 언제 변경되는지 알려줍니다.

  `x` `==` `lag``(``x``)` `#> [1] NA FALSE FALSE TRUE FALSE FALSE`

두 번째 인자인 `n`을 사용하여 한 위치 이상 떨어지게 리드(lead)하거나 지연(lag)시킬 수 있습니다.

## 연속된 식별자 (Consecutive Identifiers)

때로는 어떤 이벤트가 발생할 때마다 새로운 그룹을 시작하고 싶을 때가 있습니다. 예를 들어, 웹사이트 데이터를 살펴볼 때 이벤트를 세션(sessions)으로 나누는 것은 흔한 일이며, 이 경우 마지막 활동 이후 `x`분 이상 차이가 나면 새 세션을 시작하게 됩니다. 예를 들어 누군가가 웹사이트를 방문한 시간이 있다고 가정해 봅시다.

`events` `<-` `tibble``(` `time` `=` `c``(``0``,` `1``,` `2``,` `3``,` `5``,` `10``,` `12``,` `15``,` `17``,` `19``,` `20``,` `27``,` `28``,` `30``)` `)`

각 이벤트 간의 시간을 계산하고 그것을 세션으로 간주할 만큼 충분히 큰 공백(gap)이 있는지 파악했습니다.

`events` `<-` `events` `|>` `mutate``(` `diff` `=` `time` `-` `lag``(``time``,` `default` `=` `first``(``time``)),` `has_gap` `=` `diff` `>=` `5` `)` `events` `#> # A tibble: 14 × 3` `#> time diff has_gap` `#> <dbl> <dbl> <lgl> ` `#> 1 0 0 FALSE ` `#> 2 1 1 FALSE ` `#> 3 2 1 FALSE ` `#> 4 3 1 FALSE ` `#> 5 5 2 FALSE ` `#> 6 10 5 TRUE ` `#> # … with 8 more rows`

하지만 그 논리형 벡터(logical vector)에서 어떻게 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a> 할 수 있는 무언가로 바꿀 수 있을까요? 공백이 있을 때, 즉 `has_gap`이 `TRUE`일 때 `group`을 1씩 증가시키기 위해(<a href="ch12.html#sec-numeric-summaries-of-logicals" data-type="xref">"논리형 벡터의 숫자 요약"</a>) <a href="#sec-cumulative-and-rolling-aggregates" data-type="xref">"누적 및 이동 집계"</a>의 <a href="https://rdrr.io/r/base/cumsum.html" class="orm:hideurl"><code>cumsum()</code></a>이 도움을 줍니다.

```
events |>
  mutate(
    group = cumsum(has_gap)
  )
#> # A tibble: 14 × 4
#>    time  diff has_gap group
#>   <dbl> <dbl> <lgl>   <int>
#> 1     0     0 FALSE       0
#> 2     1     1 FALSE       0
#> 3     2     1 FALSE       0
#> 4     3     1 FALSE       0
#> 5     5     2 FALSE       0
#> 6    10     5 TRUE        1
#> # … with 8 more rows
```

그룹화 변수를 만드는 또 다른 방법은 인자 중 하나가 변경될 때마다 새 그룹을 시작하는 <a href="https://dplyr.tidyverse.org/reference/consecutive_id.html" class="orm:hideurl"><code>consecutive_id()</code></a>입니다. 예를 들어, [이 StackOverflow 질문](https://oreil.ly/swerV)에서 영감을 받아 반복되는 값이 많은 데이터 프레임이 있다고 상상해 보세요.

`df` `<-` `tibble``(` `x` `=` `c``(``"a"``,` `"a"``,` `"a"``,` `"b"``,` `"c"``,` `"c"``,` `"d"``,` `"e"``,` `"a"``,` `"a"``,` `"b"``,` `"b"``),` `y` `=` `c``(``1``,` `2``,` `3``,` `2``,` `4``,` `1``,` `3``,` `9``,` `4``,` `8``,` `10``,` `199``)` `)`

반복되는 각 `x`에서 첫 번째 행만 유지하고 싶다면 <a href="https://dplyr.tidyverse.org/reference/group_by.html" class="orm:hideurl"><code>group_by()</code></a>, <a href="https://dplyr.tidyverse.org/reference/consecutive_id.html" class="orm:hideurl"><code>consecutive_id()</code></a>, <a href="https://dplyr.tidyverse.org/reference/slice.html" class="orm:hideurl"><code>slice_head()</code></a>를 사용할 수 있습니다.

`df` `|>` `group_by``(``id` `=` `consecutive_id``(``x``))` `|>` `slice_head``(``n` `=` `1``)` `#> # A tibble: 7 × 3` `#> # Groups: id [7]` `#> x y id` `#> <chr> <dbl> <int>` `#> 1 a 1 1` `#> 2 b 2 2` `#> 3 c 4 3` `#> 4 d 3 4` `#> 5 e 9 5` `#> 6 a 4 6` `#> # … with 1 more row`

## 연습문제 (Exercises)

1. 순위 지정(ranking) 함수를 사용하여 많이 지연된 항공편 10편을 찾으세요. 동점(ties)은 어떻게 처리하시겠습니까? <a href="https://dplyr.tidyverse.org/reference/row_number.html" class="orm:hideurl"><code>min_rank()</code></a> 문서를 주의 깊게 읽어보세요.
2. 어떤 비행기(`tailnum`)의 정시 도착 기록이 나쁩니까?
3. 지연을 최대한 피하고 싶다면 하루 중 언제 비행해야 합니까?
4. `flights |> group_by(dest) |> filter(row_number() < 4)`는 무엇을 합니까? `flights |> group_by(dest) |> filter(row_number(dep_delay) < 4)`는 무엇을 합니까?
5. 각 목적지별로 총 지연 시간(분)을 계산하세요. 각 항공편에 대해 목적지의 전체 지연 시간 중 해당 항공편이 차지하는 비율을 계산하세요.
6. 지연은 일반적으로 시간적으로 상관관계가 있습니다. 초기 지연을 유발한 문제가 해결되더라도 이전 항공편이 출발할 수 있도록 후속 항공편이 지연됩니다. <a href="https://dplyr.tidyverse.org/reference/lead-lag.html" class="orm:hideurl"><code>lag()</code></a>를 사용하여 특정 시간대의 평균 항공편 지연이 이전 시간대의 평균 지연과 어떻게 관련되어 있는지 살펴보세요.

   `flights` `|>` `mutate``(``hour` `=` `dep_time` `%/%` `100``)` `|>` `group_by``(``year``,` `month``,` `day``,` `hour``)` `|>` `summarize``(` `dep_delay` `=` `mean``(``dep_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``(),` `.groups` `=` `"drop"` `)` `|>` `filter``(``n` `>` `5``)`

7. 각 목적지를 살펴보세요. 의심스러울 정도로 빠른 항공편(즉, 데이터 입력 오류일 가능성이 있는 항공편)을 찾을 수 있습니까? 해당 목적지로 가는 짧은 항공편과 비교하여 각 항공편의 비행 시간을 계산하세요. 비행 중 많이 지연된 항공편은 무엇입니까?
8. 최소 두 개 이상의 항공사가 운항하는 모든 목적지를 찾으세요. 해당 목적지들을 사용하여 동일한 목적지에 대한 실적을 바탕으로 항공사들의 상대적인 순위를 매겨보세요.

# 숫자 요약 (Numeric Summaries)

우리가 이미 소개한 개수(counts), 평균(means), 합계(sums)만 사용해도 많은 것을 할 수 있지만, R은 유용한 요약 함수들을 더 많이 제공합니다. 여기에 여러분이 유용하다고 여길 만한 몇 가지를 선택해 두었습니다.

## 중심 (Center)

지금까지 우리는 주로 값들의 벡터의 중심(center)을 요약하기 위해 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>을 사용해 왔습니다. <a href="ch03.html#sec-sample-size" data-type="xref">"사례 연구: 집계 및 표본 크기"</a>에서 보았듯이, 평균은 합계를 개수로 나눈 것이기 때문에 몇 개의 비정상적으로 높거나 낮은 값에도 민감하게 반응합니다. 대안은 벡터의 "중간(middle)"에 위치하는 값, 즉 50%의 값이 그보다 크고 50%의 값이 그보다 작은 값을 찾는 중앙값 <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>을 사용하는 것입니다. 관심 있는 변수 분포의 모양에 따라 평균 또는 중앙값이 중심을 측정하는 더 나은 척도가 될 수 있습니다. 예를 들어, 대칭적인 분포에 대해서는 일반적으로 평균을 보고하는 반면 비대칭적인(skewed) 분포에 대해서는 일반적으로 중앙값을 보고합니다.

<a href="#fig-mean-vs-median" data-type="xref">그림 13-2</a>는 각 목적지에 대한 출발 지연(분 단위)의 평균과 중앙값을 비교합니다. 항공편이 몇 시간씩 지연되어 늦게 출발하는 경우는 있어도 몇 시간씩 일찍 출발하는 경우는 없기 때문에 중앙값 지연은 항상 평균 지연보다 작습니다.

`flights` `|>` `group_by``(``year``,` `month``,` `day``)` `|>` `summarize``(` `mean` `=` `mean``(``dep_delay``,` `na.rm` `=` `TRUE``),` `median` `=` `median``(``dep_delay``,` `na.rm` `=` `TRUE``),` `n` `=` `n``(),` `.groups` `=` `"drop"` `)` `|>` `ggplot``(``aes``(``x` `=` `mean``,` `y` `=` `median``))` `+` `geom_abline``(``slope` `=` `1``,` `intercept` `=` `0``,` `color` `=` `"white"``,` `linewidth` `=` `2``)` `+` `geom_point``()`

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1302.png" alt="All points fall below a 45° line, meaning that the median delay is always less than the mean delay. Most points are clustered in a dense region of mean [0, 20] and median [0, 5]. As the mean delay increases, the spread of the median also increases. There are two outlying points with mean ~60, median ~50, and mean ~85, median ~55." />
<h6 id="figure-13-2.-a-scatterplot-showing-the-differences-of-summarizing-hourly-departure-delay-with-median-instead-of-mean.">그림 13-2. 시간당 출발 지연 시간을 평균 대신 중앙값으로 요약할 때의 차이를 보여주는 산점도.</h6>
</figure>

최빈값(mode), 즉 흔한 값에 대해서도 궁금할 수 있습니다. 이것은 매우 단순한 경우에만 잘 작동하는 요약이어서(고등학교 때 배운 이유이기도 합니다), 많은 실제 데이터세트에서는 잘 작동하지 않습니다. 데이터가 이산형(discrete)인 경우 흔한 값이 여러 개일 수 있고, 데이터가 연속형(continuous)인 경우 모든 값이 아주 미세하게 다르기 때문에 흔한 값이 없을 수도 있습니다. 이러한 이유로 최빈값은 통계학자들에 의해 잘 사용되지 않는 경향이 있으며, 기본 R에는 최빈값 함수가 포함되어 있지 않습니다.<sup><a href="ch13.html#idm44771297333504" id="idm44771297333504-marker" data-type="noteref">2</a></sup>

## 최솟값, 최댓값 및 분위수 (Minimum, Maximum, and Quantiles)

중심 이외의 위치에 관심이 있다면 어떨까요? <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>min()</code></a>과 <a href="https://rdrr.io/r/base/Extremes.html" class="orm:hideurl"><code>max()</code></a>는 큰 값과 작은 값을 알려줍니다. 또 다른 강력한 도구는 중앙값의 일반화(generalization)인 <a href="https://rdrr.io/r/stats/quantile.html" class="orm:hideurl"><code>quantile()</code></a>입니다. `quantile(x, 0.25)`는 하위 25%의 값보다 큰 `x`의 값을 찾을 것이고, `quantile(x, 0.5)`는 중앙값과 동일하며, `quantile(x, 0.95)`는 하위 95%의 값보다 큰 값을 찾을 것입니다.

`flights` 데이터의 경우, 극단적일 수 있는 상위 5%의 지연된 항공편을 무시하기 위해 최댓값 대신 지연의 95% 분위수를 살펴보는 것이 좋을 수 있습니다.

`flights` `|>` `group_by``(``year``,` `month``,` `day``)` `|>` `summarize``(` `max` `=` `max``(``dep_delay``,` `na.rm` `=` `TRUE``),` `q95` `=` `quantile``(``dep_delay``,` `0.95``,` `na.rm` `=` `TRUE``),` `.groups` `=` `"drop"` `)` `#> # A tibble: 365 × 5` `#> year month day max q95` `#> <int> <int> <int> <dbl> <dbl>` `#> 1 2013 1 1 853 70.1` `#> 2 2013 1 2 379 85 ` `#> 3 2013 1 3 291 68 ` `#> 4 2013 1 4 288 60 ` `#> 5 2013 1 5 327 41 ` `#> 6 2013 1 6 202 51 ` `#> # … with 359 more rows`

## 퍼짐 (Spread)

때로는 데이터의 대부분이 어디에 위치하는지보다 데이터가 어떻게 퍼져 있는지에 더 관심이 있을 수 있습니다. 일반적으로 사용되는 두 가지 요약은 표준편차(standard deviation) `sd(x)`와 사분위수 범위(inter-quartile range) <a href="https://rdrr.io/r/stats/IQR.html" class="orm:hideurl"><code>IQR()</code></a>입니다. 여러분이 이미 익숙하실 것이므로 여기서 <a href="https://rdrr.io/r/stats/sd.html" class="orm:hideurl"><code>sd()</code></a>에 대해서는 설명하지 않겠지만 <a href="https://rdrr.io/r/stats/IQR.html" class="orm:hideurl"><code>IQR()</code></a>은 새로울 수 있습니다. 이것은 `quantile(x, 0.75) - quantile(x, 0.25)`로 계산되며 데이터의 중간 50%가 포함된 범위를 알려줍니다.

우리는 이것을 사용하여 `flights` 데이터의 작은 이상함을 밝혀낼 수 있습니다. 공항은 항상 같은 위치에 있기 때문에 출발지와 목적지 간 거리의 퍼짐(spread)이 0이 될 것이라고 예상할 수 있습니다. 하지만 다음 코드는 공항 [EGE](https://oreil.ly/Zse1Q)에 대한 데이터의 이상함을 드러냅니다.

`flights` `|>` `group_by``(``origin``,` `dest``)` `|>` `summarize``(` `distance_sd` `=` `IQR``(``distance``),` `n` `=` `n``(),` `.groups` `=` `"drop"` `)` `|>` `filter``(``distance_sd` `>` `0``)` `#> # A tibble: 2 × 4` `#> origin dest distance_sd n` `#> <chr> <chr> <dbl> <int>` `#> 1 EWR EGE 1 110` `#> 2 JFK EGE 1 103`

## 분포 (Distributions)

이전에 설명한 모든 요약 통계량은 분포를 단일 숫자로 줄이는(reducing) 방법이라는 점을 기억할 가치가 있습니다. 이는 그것들이 근본적으로 축소적(reductive)이라는 것을 의미하며, 잘못된 요약을 선택하면 그룹 간의 중요한 차이를 쉽게 놓칠 수 있습니다. 그렇기 때문에 요약 통계량을 확정하기 전에 분포를 시각화하는 것이 항상 좋은 생각입니다.

<a href="#fig-flights-dist" data-type="xref">그림 13-3</a>은 출발 지연 시간의 전반적인 분포를 보여줍니다. 분포가 너무 치우쳐 있어서 데이터의 대부분을 보려면 확대해야 합니다. 이는 평균이 좋은 요약이 될 가능성이 낮으며 대신 중앙값을 선호할 수 있음을 시사합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1303.png" alt="Two histograms of `dep_delay`. On the left, it&#39;s very hard to see any pattern except that there&#39;s a very large spike around zero, the bars rapidly decay in height, and for most of the plot, you can&#39;t see any bars because they are too short to see. On the right, where we&#39;ve discarded delays of greater than two hours, we can see that the spike occurs slightly below zero (i.e. most flights leave a couple of minutes early), but there&#39;s still a very steep decay after that. " />
<h6 id="figure-13-3.-left-the-histogram-of-the-full-data-is-extremely-skewed-making-it-hard-to-get-any-details.-right-zooming-into-delays-of-less-than-two-hours-makes-it-possible-to-see-whats-happening-with-the-bulk-of-the-observations.">그림 13-3. (왼쪽) 전체 데이터의 히스토그램은 극도로 치우쳐 있어 세부 사항을 파악하기 어렵습니다. (오른쪽) 지연 시간이 2시간 미만인 경우를 확대하면 관측치의 대부분에서 무슨 일이 일어나고 있는지 볼 수 있습니다.</h6>
</figure>

하위 그룹의 분포가 전체 분포와 닮았는지 확인하는 것도 좋은 생각입니다. 다음 플롯에서는 `dep_delay`에 대한 365개의 도수 다각형(frequency polygons)(매일 하나씩)이 겹쳐져 있습니다. 분포들이 공통적인 패턴을 따르는 것으로 보이며, 이는 매일 동일한 요약을 사용하는 것이 괜찮음을 시사합니다.

`flights` `|>` `filter``(``dep_delay` `<` `120``)` `|>` `ggplot``(``aes``(``x` `=` `dep_delay``,` `group` `=` `interaction``(``day``,` `month``)))` `+` `geom_freqpoly``(``binwidth` `=` `5``,` `alpha` `=` `1``/``5``)`

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_13in01.png" alt="The distribution of `dep_delay` is highly right skewed with a strong peak slightly less than 0. The 365 frequency polygons are mostly overlapping forming a thick black bland." />
</figure>

여러분이 다루고 있는 데이터에 특별히 맞춤화된 자신만의 요약을 탐색하는 것을 두려워하지 마세요. 이 경우, 조기 출발한 항공편과 지연 출발한 항공편을 별도로 요약하거나, 값이 크게 치우쳐 있다는 점을 고려하여 로그 변환(log transformation)을 시도해 볼 수 있습니다. 마지막으로, <a href="ch03.html#sec-sample-size" data-type="xref">"사례 연구: 집계 및 표본 크기"</a>에서 배운 내용을 잊지 마세요. 수치 요약을 생성할 때는 항상 각 그룹의 관측치 수를 포함하는 것이 좋습니다.

## 위치 (Positions)

숫자 벡터에 유용하지만 다른 모든 유형의 값에도 작동하는 마지막 요약 유형이 하나 있습니다. 특정 위치에 있는 값을 추출하는 `first(x)`, `last(x)`, `nth(x, n)`입니다.

예를 들어, 각 날짜의 첫 번째와 마지막 출발 시간을 찾을 수 있습니다.

`flights` `|>` `group_by``(``year``,` `month``,` `day``)` `|>` `summarize``(` `first_dep` `=` `first``(``dep_time``,` `na_rm` `=` `TRUE``),` `fifth_dep` `=` `nth``(``dep_time``,` `5``,` `na_rm` `=` `TRUE``),` `last_dep` `=` `last``(``dep_time``,` `na_rm` `=` `TRUE``)` `)` ``#> `summarise()` has grouped output by 'year', 'month'. You can override using`` ``#> the `.groups` argument.`` `#> # A tibble: 365 × 6` `#> # Groups: year, month [12]` `#> year month day first_dep fifth_dep last_dep` `#> <int> <int> <int> <int> <int> <int>` `#> 1 2013 1 1 517 554 2356` `#> 2 2013 1 2 42 535 2354` `#> 3 2013 1 3 32 520 2349` `#> 4 2013 1 4 25 531 2358` `#> 5 2013 1 5 14 534 2357` `#> 6 2013 1 6 16 555 2355` `#> # … with 359 more rows`

(dplyr 함수는 함수와 인자 이름의 구성 요소를 분리하기 위해 `_`를 사용하므로 이 함수들은 `na.rm` 대신 `na_rm`을 사용한다는 점에 유의하세요.)

<a href="ch27.html#sec-subset-many" data-type="xref">"[로 여러 요소 선택하기"</a>에서 다시 다루게 될 `[`에 익숙하다면, 이런 함수들이 정말 필요한지 의문이 들 수 있습니다. 세 가지 이유가 있습니다. `default` 인자를 통해 지정된 위치가 존재하지 않을 경우 기본값을 제공할 수 있고, `order_by` 인자를 통해 로컬에서 행의 순서를 재정의(override)할 수 있으며, `na_rm` 인자를 통해 결측값을 제거할 수 있습니다.

위치에서 값을 추출하는 것은 순위로 필터링하는 것을 보완합니다. 필터링은 각 관측치가 별도의 행에 있는 모든 변수를 제공합니다.

`flights` `|>` `group_by``(``year``,` `month``,` `day``)` `|>` `mutate``(``r` `=` `min_rank``(``sched_dep_time``))` `|>` `filter``(``r` `%in%` `c``(``1``,` `max``(``r``)))` `#> # A tibble: 1,195 × 20` `#> # Groups: year, month, day [365]` `#> year month day dep_time sched_dep_time dep_delay arr_time sched_arr_time` `#> <int> <int> <int> <int> <int> <dbl> <int> <int>` `#> 1 2013 1 1 517 515 2 830 819` `#> 2 2013 1 1 2353 2359 -6 425 445` `#> 3 2013 1 1 2353 2359 -6 418 442` `#> 4 2013 1 1 2356 2359 -3 425 437` `#> 5 2013 1 2 42 2359 43 518 442` `#> 6 2013 1 2 458 500 -2 703 650` `#> # … with 1,189 more rows, and 12 more variables: arr_delay <dbl>,` `#> # carrier <chr>, flight <int>, tailnum <chr>, origin <chr>, dest <chr>, …`

## mutate()와 함께 사용하기 (With mutate())

이름에서 알 수 있듯이, 요약 함수는 전형적으로 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 짝을 이룹니다. 그러나 <a href="#sec-recycling" data-type="xref">"산술 및 리사이클링 규칙"</a>에서 논의한 리사이클링 규칙 때문에, 이 함수들은 어떤 종류의 그룹 표준화(group standardization)를 수행하고자 할 때 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 유용하게 짝을 이룰 수도 있습니다. 예를 들면:

`x / sum(x)`  
전체에 대한 비율을 계산합니다.

`(x - mean(x)) / sd(x)`  
Z-점수(평균 0, 표준편차 1로 표준화됨)를 계산합니다.

`(x - min(x)) / (max(x) - min(x))`  
범위 [0, 1]로 표준화합니다.

`x / first(x)`  
첫 번째 관측치를 기준으로 지수(index)를 계산합니다.

## 연습문제 (Exercises)

1. 특정 항공편 그룹의 전형적인 지연 특성을 평가할 수 있는 방법을 최소 5가지 이상 브레인스토밍하세요. <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>은 언제 유용합니까? <a href="https://rdrr.io/r/stats/median.html" class="orm:hideurl"><code>median()</code></a>은 언제 유용합니까? 어떤 경우에 다른 함수를 사용하고 싶을 수 있습니까? 도착 지연(arrival delay)을 사용해야 합니까, 아니면 출발 지연(departure delay)을 사용해야 합니까? `planes` 데이터를 사용해야 하는 이유는 무엇일까요?
2. 어느 목적지가 비행 속도의 변동성이 큰가요?
3. 공항 EGE의 미스터리를 더 탐구하기 위한 플롯을 생성하세요. 공항이 위치를 이전했다는 증거를 찾을 수 있습니까? 그 차이를 설명할 수 있는 다른 변수를 찾을 수 있습니까?

# 요약 (Summary)

여러분은 이미 숫자를 다루는 많은 도구에 익숙해져 있으며, 이 장을 읽고 나면 R에서 이러한 도구들을 어떻게 사용하는지 알게 되었을 것입니다. 또한 순위 및 오프셋과 같이 주로 숫자 벡터에 적용되지만 다른 유형에도 적용할 수 있는 몇 가지 유용한 일반 변환(general transformations)에 대해서도 배웠습니다. 마지막으로, 다양한 숫자 요약을 살펴보고 고려해야 할 몇 가지 통계적 문제에 대해서 논의했습니다.

다음 두 장에서는 stringr 패키지를 사용하여 문자열을 다루는 방법에 대해 깊이 파고들 것입니다. 문자열은 큰 주제이므로, 문자열의 기초에 대한 한 장과 정규 표현식에 대한 또 다른 한 장, 총 두 장이 할당됩니다.

<sup>[1](ch13.html#idm44771298681152-marker)</sup> ggplot2는 <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_interval()</code></a>, <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_number()</code></a>, <a href="https://ggplot2.tidyverse.org/reference/cut_interval.html" class="orm:hideurl"><code>cut_width()</code></a>와 같이 흔한 사례를 위한 몇 가지 도우미를 제공합니다. 이러한 함수들이 ggplot2에 존재하는 것은 다소 이상하게 느껴질 수 있지만, 이 함수들은 히스토그램 계산의 일부로서 유용하며 tidyverse의 다른 어떤 부분들이 존재하기도 전에 작성되었습니다.

<sup>[2](ch13.html#idm44771297333504-marker)</sup> <a href="https://rdrr.io/r/base/mode.html" class="orm:hideurl"><code>mode()</code></a> 함수는 이와 완전히 다른 작업을 수행합니다!
