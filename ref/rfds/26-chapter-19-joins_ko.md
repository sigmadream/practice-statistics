# 19장. 조인(Joins)

# 소개

데이터 분석에 단 하나의 데이터 프레임만 포함되는 경우는 드뭅니다. 일반적으로 여러분은 많은 데이터 프레임을 가지고 있으며, 관심 있는 질문에 답하기 위해 이것들을 함께 *조인(join)*해야 합니다. 이 장에서는 두 가지 중요한 조인 유형을 소개합니다.

- 변형 조인(Mutating joins): 일치하는 관측치를 바탕으로 다른 데이터 프레임에서 가져온 새 변수를 하나의 데이터 프레임에 추가합니다.
- 필터링 조인(Filtering joins): 다른 데이터 프레임의 관측치와 일치하는지 여부에 따라 하나의 데이터 프레임에서 관측치를 필터링합니다.

조인에서 한 쌍의 데이터 프레임을 연결하는 데 사용되는 변수인 키(keys)에 대해 논의하는 것으로 시작하겠습니다. nycflights13 패키지의 데이터 세트들에 있는 키들을 살펴보면서 이론을 확고히 다진 다음, 그 지식을 사용하여 데이터 프레임을 함께 조인하기 시작할 것입니다. 다음으로, 행(rows)에 대한 조인의 동작에 초점을 맞추어 조인이 어떻게 작동하는지 논의할 것입니다. 기본 동등성(equality) 관계보다 더 유연한 키 일치 방법을 제공하는 조인 제품군인 비동등 조인(non-equi joins)에 대한 논의로 마무리하겠습니다.

## 사전 준비

이 장에서는 dplyr의 조인 함수를 사용하여 nycflights13의 5가지 관련 데이터 세트를 탐색할 것입니다.

```
library(tidyverse)
library(nycflights13)
```

# 키(Keys)

조인을 이해하려면 먼저 두 테이블이 각 테이블 내의 한 쌍의 키를 통해 어떻게 연결될 수 있는지 이해해야 합니다. 이 섹션에서는 두 가지 유형의 키에 대해 배우고 nycflights13 패키지의 데이터 세트에서 두 유형의 예시를 살펴볼 것입니다. 또한 키가 유효한지 확인하는 방법과 테이블에 키가 부족한 경우 어떻게 해야 하는지 배울 것입니다.

## 기본 키(Primary Keys)와 외래 키(Foreign Keys)

모든 조인에는 한 쌍의 키가 포함됩니다. 기본 키와 외래 키. *기본 키(primary key)*는 각 관측치를 고유하게 식별하는 변수 또는 변수의 집합입니다. 두 개 이상의 변수가 필요한 경우 해당 키를 *복합 키(compound key)*라고 합니다. 예를 들어 nycflights13에서:

- `airlines`는 각 항공사에 대한 두 가지 데이터, 즉 항공사 코드와 전체 이름을 기록합니다. 두 글자로 된 항공사 코드로 항공사를 식별할 수 있으므로 `carrier`가 기본 키가 됩니다.

  ```
  airlines
  #> # A tibble: 16 × 2
  #>   carrier name
  #>   <chr>   <chr>
  #> 1 9E      Endeavor Air Inc.
  #> 2 AA      American Airlines Inc.
  #> 3 AS      Alaska Airlines Inc.
  #> 4 B6      JetBlue Airways
  #> 5 DL      Delta Air Lines Inc.
  #> 6 EV      ExpressJet Airlines Inc.
  #> # … with 10 more rows
  ```

- `airports`는 각 공항에 대한 데이터를 기록합니다. 세 글자로 된 공항 코드로 각 공항을 식별할 수 있으므로 `faa`가 기본 키가 됩니다.

  ```
  airports
  #> # A tibble: 1,458 × 8
  #>   faa   name                            lat   lon   alt    tz dst
  #>   <chr> <chr>                         <dbl> <dbl> <dbl> <dbl> <chr>
  #> 1 04G   Lansdowne Airport              41.1 -80.6  1044    -5 A
  #> 2 06A   Moton Field Municipal Airport  32.5 -85.7   264    -6 A
  #> 3 06C   Schaumburg Regional            42.0 -88.1   801    -6 A
  #> 4 06N   Randall Airport                41.4 -74.4   523    -5 A
  #> 5 09J   Jekyll Island Airport          31.1 -81.4    11    -5 A
  #> 6 0A9   Elizabethton Municipal Airpo…  36.4 -82.2  1593    -5 A
  #> # … with 1,452 more rows, and 1 more variable: tzone <chr>
  ```

- `planes`는 각 비행기에 대한 데이터를 기록합니다. 꼬리 번호(tail number)로 비행기를 식별할 수 있으므로 `tailnum`이 기본 키가 됩니다.

  ```
  planes
  #> # A tibble: 3,322 × 9
  #>   tailnum  year type              manufacturer    model     engines
  #>   <chr>   <int> <chr>             <chr>           <chr>       <int>
  #> 1 N10156   2004 Fixed wing multi… EMBRAER         EMB-145XR       2
  #> 2 N102UW   1998 Fixed wing multi… AIRBUS INDUSTR… A320-214        2
  #> 3 N103US   1999 Fixed wing multi… AIRBUS INDUSTR… A320-214        2
  #> 4 N104UW   1999 Fixed wing multi… AIRBUS INDUSTR… A320-214        2
  #> 5 N10575   2002 Fixed wing multi… EMBRAER         EMB-145LR       2
  #> 6 N105UW   1999 Fixed wing multi… AIRBUS INDUSTR… A320-214        2
  #> # … with 3,316 more rows, and 3 more variables: seats <int>,
  #> #   speed <int>, engine <chr>
  ```

- `weather`는 출발 공항의 날씨에 대한 데이터를 기록합니다. 위치와 시간의 조합으로 각 관측치를 식별할 수 있으므로 `origin`과 `time_hour`가 복합 기본 키가 됩니다.

  ```
  weather
  #> # A tibble: 26,115 × 15
  #>   origin  year month   day  hour  temp  dewp humid wind_dir
  #>   <chr>  <int> <int> <int> <int> <dbl> <dbl> <dbl>    <dbl>
  #> 1 EWR     2013     1     1     1  39.0  26.1  59.4      270
  #> 2 EWR     2013     1     1     2  39.0  27.0  61.6      250
  #> 3 EWR     2013     1     1     3  39.0  28.0  64.4      240
  #> 4 EWR     2013     1     1     4  39.9  28.0  62.2      250
  #> 5 EWR     2013     1     1     5  39.0  28.0  64.4      260
  #> 6 EWR     2013     1     1     6  37.9  28.0  67.2      240
  #> # … with 26,109 more rows, and 6 more variables: wind_speed <dbl>,
  #> #   wind_gust <dbl>, precip <dbl>, pressure <dbl>, visib <dbl>, …
  ```

*외래 키(foreign key)*는 다른 테이블의 기본 키에 해당하는 변수(또는 변수의 집합)입니다. 예를 들어:

- `flights$tailnum`은 기본 키 `planes$tailnum`에 해당하는 외래 키입니다.
- `flights$carrier`는 기본 키 `airlines$carrier`에 해당하는 외래 키입니다.
- `flights$origin`은 기본 키 `airports$faa`에 해당하는 외래 키입니다.
- `flights$dest`는 기본 키 `airports$faa`에 해당하는 외래 키입니다.
- `flights$origin`-`flights$time_hour`는 복합 기본 키 `weather$origin`-`weather$time_hour`에 해당하는 복합 외래 키입니다.

이러한 관계는 <a href="#fig-flights-relationships" data-type="xref">그림 19-1</a>에 시각적으로 요약되어 있습니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1901.png" alt="nycflights13 패키지의 airports, planes, flights, weather 및 airlines 데이터 세트 간의 관계. airports$faa는 flights$origin 및 flights$dest에 연결됨. planes$tailnum은 flights$tailnum에 연결됨. weather$time_hour와 weather$origin은 결합하여 flights$time_hour 및 flights$origin에 연결됨. airlines$carrier는 flights$carrier에 연결됨. airports, planes, airlines 및 weather 데이터 프레임 간에는 직접적인 연결이 없음." />
<h6 id="figure-19-1.-connections-between-all-five-data-frames-in-the-nycflights13-package.-variables-making-up-a-primary-key-are-gray-and-are-connected-to-their-corresponding-foreign-keys-with-arrows.">그림 19-1. nycflights13 패키지에 있는 5개 데이터 프레임 모두 간의 연결. 기본 키를 구성하는 변수는 회색이며 해당 외래 키에 화살표로 연결되어 있습니다.</h6>
</figure>

이러한 키 설계의 멋진 특징을 알아차릴 수 있을 것입니다. 기본 키와 외래 키의 이름이 거의 항상 동일하다는 점인데, 곧 보게 되겠지만 이 점이 여러분의 조인 작업을 훨씬 더 쉽게 만들어 줄 것입니다. 반대 관계도 주목할 가치가 있습니다. 여러 테이블에서 사용되는 거의 모든 변수 이름은 각 장소에서 동일한 의미를 갖습니다. 단 하나의 예외가 있습니다. `year`는 `flights`에서는 출발 연도를 의미하고 `planes`에서는 제조 연도를 의미합니다. 이것은 우리가 실제로 테이블을 함께 조인하기 시작할 때 중요해질 것입니다.

## 기본 키 확인하기

각 테이블의 기본 키를 식별했으므로, 이 키들이 실제로 각 관측치를 고유하게 식별하는지 확인하는 것이 좋은 관행입니다. 한 가지 방법은 기본 키를 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>하여 `n`이 1보다 큰 항목을 찾는 것입니다. 이렇게 해보면 `planes`와 `weather` 모두 좋아 보인다는 것을 알 수 있습니다.

```
planes |>
  count(tailnum) |>
  filter(n > 1)
#> # A tibble: 0 × 2
#> # … with 2 variables: tailnum <chr>, n <int>

weather |>
  count(time_hour, origin) |>
  filter(n > 1)
#> # A tibble: 0 × 3
#> # … with 3 variables: time_hour <dttm>, origin <chr>, n <int>
```

또한 기본 키에 결측값이 있는지도 확인해야 합니다. 값이 누락된 경우 해당 값은 관측치를 식별할 수 없기 때문입니다!

```
planes |>
  filter(is.na(tailnum))
#> # A tibble: 0 × 9
#> # … with 9 variables: tailnum <chr>, year <int>, type <chr>,
#> #   manufacturer <chr>, model <chr>, engines <int>, seats <int>, …

weather |>
  filter(is.na(time_hour) | is.na(origin))
#> # A tibble: 0 × 15
#> # … with 15 variables: origin <chr>, year <int>, month <int>, day <int>,
#> #   hour <int>, temp <dbl>, dewp <dbl>, humid <dbl>, wind_dir <dbl>, …
```

## 대리 키(Surrogate Keys)

지금까지 우리는 `flights`의 기본 키에 대해 이야기하지 않았습니다. 그것을 외래 키로 사용하는 데이터 프레임이 없기 때문에 여기서는 그다지 중요하지 않지만, 다른 사람들에게 관측치를 설명할 어떤 방법이 있다면 관측치로 작업하기가 더 쉽기 때문에 여전히 고려해 보는 것이 유용합니다.

약간의 생각과 실험 끝에, 우리는 함께 각 비행을 고유하게 식별하는 3개의 변수가 있다는 것을 알아냈습니다.

```
flights |>
  count(time_hour, carrier, flight) |>
  filter(n > 1)
#> # A tibble: 0 × 4
#> # … with 4 variables: time_hour <dttm>, carrier <chr>, flight <int>, n <int>
```

중복이 없다고 해서 자동으로 `time_hour`-`carrier`-`flight`가 기본 키가 될까요? 좋은 출발점인 것은 확실하지만 보장하지는 않습니다. 예를 들어 고도와 위도는 `airports`에 대한 좋은 기본 키일까요?

```
airports |>
  count(alt, lat) |>
  filter(n > 1)
#> # A tibble: 1 × 3
#>     alt   lat     n
#>   <dbl> <dbl> <int>
#> 1    13  40.6     2
```

고도와 위도로 공항을 식별하는 것은 명백히 나쁜 생각이며, 일반적으로 변수 조합이 좋은 기본 키를 만드는지 여부를 데이터만 보고 아는 것은 불가능합니다. 하지만 항공편의 경우, 같은 비행 번호를 가진 여러 항공편이 동시에 하늘에 떠 있다면 항공사와 고객 모두에게 정말 혼란스러울 것이기 때문에 `time_hour`, `carrier` 및 `flight`의 조합은 합리적으로 보입니다.

그렇기는 하지만, 행 번호를 사용하여 간단한 숫자형 대리 키(surrogate key)를 도입하는 것이 더 나을 수도 있습니다.

```
flights2 <- flights |>
  mutate(id = row_number(), .before = 1)
flights2
#> # A tibble: 336,776 × 20
#>      id  year month   day dep_time sched_dep_time dep_delay arr_time
#>   <int> <int> <int> <int>    <int>          <int>     <dbl>    <int>
#> 1     1  2013     1     1      517            515         2      830
#> 2     2  2013     1     1      533            529         4      850
#> 3     3  2013     1     1      542            540         2      923
#> 4     4  2013     1     1      544            545        -1     1004
#> 5     5  2013     1     1      554            600        -6      812
#> 6     6  2013     1     1      554            558        -4      740
#> # … with 336,770 more rows, and 12 more variables: sched_arr_time <int>,
#> #   arr_delay <dbl>, carrier <chr>, flight <int>, tailnum <chr>, …
```

대리 키는 다른 사람들과 의사소통할 때 특히 유용할 수 있습니다. 누군가에게 2013년 1월 3일 오전 9시에 출발한 UA430을 보라고 하는 것보다 항공편 2001을 보라고 말하는 것이 훨씬 쉽습니다.

## 연습문제

1. <a href="#fig-flights-relationships" data-type="xref">그림 19-1</a>에서 `weather`와 `airports` 간의 관계를 그리는 것을 잊었습니다. 어떤 관계이며 다이어그램에 어떻게 나타나야 할까요?

2. `weather`에는 뉴욕의 3개 출발 공항에 대한 정보만 포함되어 있습니다. 미국의 모든 공항에 대한 날씨 기록이 포함되어 있다면 `flights`와 어떤 추가 연결이 이루어질까요?

3. `year`, `month`, `day`, `hour` 및 `origin` 변수는 `weather`에 대한 거의 완벽한 복합 키를 형성하지만, 중복된 관측치가 있는 시간이 하나 있습니다. 그 시간에 대해 무엇이 특별한지 알아낼 수 있나요?

4. 일 년 중 어떤 날들은 특별해서 평소보다 비행기를 타는 사람이 적다는 것을 우리는 알고 있습니다(크리스마스 이브와 크리스마스 당일). 해당 데이터를 어떻게 데이터 프레임으로 표현할 수 있을까요? 기본 키는 무엇이 될까요? 기존 데이터 프레임과는 어떻게 연결될까요?

5. Lahman 패키지에 있는 `Batting`, `People`, `Salaries` 데이터 프레임 간의 연결을 보여주는 다이어그램을 그리세요. `People`, `Managers`, `AwardsManagers` 간의 관계를 보여주는 또 다른 다이어그램을 그리세요. `Batting`, `Pitching`, `Fielding` 데이터 프레임 간의 관계를 어떻게 특징지을 수 있을까요?

# 기본 조인(Basic Joins)

이제 키를 통해 데이터 프레임이 어떻게 연결되는지 이해했으므로 조인을 사용하여 `flights` 데이터 세트를 더 잘 이해해 봅시다. dplyr은 6개의 조인 함수를 제공합니다.

- <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>
- <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>inner_join()</code></a>
- <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>right_join()</code></a>
- <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>full_join()</code></a>
- <a href="https://dplyr.tidyverse.org/reference/filter-joins.html" class="orm:hideurl"><code>semi_join()</code></a>
- <a href="https://dplyr.tidyverse.org/reference/filter-joins.html" class="orm:hideurl"><code>anti_join()</code></a>

이들은 모두 동일한 인터페이스를 갖습니다. 한 쌍의 데이터 프레임(`x`와 `y`)을 취하고 데이터 프레임을 반환합니다. 출력에서 행과 열의 순서는 주로 `x`에 의해 결정됩니다.

이 섹션에서는 하나의 변형 조인인 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>과 두 개의 필터링 조인인 <a href="https://dplyr.tidyverse.org/reference/filter-joins.html" class="orm:hideurl"><code>semi_join()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/filter-joins.html" class="orm:hideurl"><code>anti_join()</code></a>을 사용하는 방법에 대해 배웁니다. 다음 섹션에서는 이러한 함수들이 정확히 어떻게 작동하는지, 그리고 나머지 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>inner_join()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>right_join()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>full_join()</code></a>에 대해 배울 것입니다.

## 변형 조인(Mutating Joins)

*변형 조인(mutating join)*은 두 데이터 프레임의 변수를 결합할 수 있게 해줍니다. 먼저 키를 기준으로 관측치를 일치시킨 다음 한 데이터 프레임에서 다른 데이터 프레임으로 변수를 복사합니다. <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 마찬가지로 조인 함수는 오른쪽으로 변수를 추가하므로, 데이터 세트에 변수가 많으면 새로운 변수가 보이지 않을 것입니다. 이러한 예제들을 위해 단 6개의 변수만 있는 더 좁은(narrower) 데이터 세트를 만들어 무슨 일이 일어나고 있는지 더 쉽게 볼 수 있도록 하겠습니다.<sup><a href="ch19.html#idm44771284613936" id="idm44771284613936-marker" data-type="noteref">1</a></sup>

```
flights2 <- flights |>
  select(year, time_hour, origin, dest, tailnum, carrier)
flights2
#> # A tibble: 336,776 × 6
#>    year time_hour           origin dest  tailnum carrier
#>   <int> <dttm>              <chr>  <chr> <chr>   <chr>
#> 1  2013 2013-01-01 05:00:00 EWR    IAH   N14228  UA
#> 2  2013 2013-01-01 05:00:00 LGA    IAH   N24211  UA
#> 3  2013 2013-01-01 05:00:00 JFK    MIA   N619AA  AA
#> 4  2013 2013-01-01 05:00:00 JFK    BQN   N804JB  B6
#> 5  2013 2013-01-01 06:00:00 LGA    ATL   N668DN  DL
#> 6  2013 2013-01-01 05:00:00 EWR    ORD   N39463  UA
#> # … with 336,770 more rows
```

변형 조인에는 4가지 유형이 있지만 여러분이 거의 항상 사용하게 될 유형이 하나 있습니다. <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>. 이것이 특별한 이유는 출력 결과가 항상 `x`와 동일한 행을 갖기 때문입니다.<sup><a href="ch19.html#idm44771284529504" id="idm44771284529504-marker" data-type="noteref">2</a></sup> <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>의 주된 용도는 추가적인 메타데이터를 덧붙이는 것입니다. 예를 들어, <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>을 사용하여 `flights2` 데이터에 항공사의 전체 이름을 추가할 수 있습니다.

```
flights2 |>
  left_join(airlines)
#> Joining with `by = join_by(carrier)`
#> # A tibble: 336,776 × 7
#>    year time_hour           origin dest  tailnum carrier name
#>   <int> <dttm>              <chr>  <chr> <chr>   <chr>   <chr>
```

#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA United Air Lines In…
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA United Air Lines In…
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA American Airlines I…
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 JetBlue Airways  
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL Delta Air Lines Inc.
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA United Air Lines In…
#> # … with 336,770 more rows

```

또는 각 비행기가 출발할 때의 온도와 풍속을 알아낼 수도 있습니다.

```

flights2 |>
left_join(weather |> select(origin, time_hour, temp, wind_speed))
#> Joining with `by = join_by(time_hour, origin)`
#> # A tibble: 336,776 × 8
#> year time_hour origin dest tailnum carrier temp wind_speed
#> <int> <dttm> <chr> <chr> <chr> <chr> <dbl> <dbl>
#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA 39.0 12.7
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA 39.9 15.0
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA 39.0 15.0
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 39.0 15.0
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL 39.9 16.1
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA 39.0 12.7
#> # … with 336,770 more rows

```

또는 어떤 크기의 비행기가 비행 중이었는지 알아볼 수 있습니다.

```

flights2 |>
left_join(planes |> select(tailnum, type, engines, seats))
#> Joining with `by = join_by(tailnum)`
#> # A tibble: 336,776 × 9
#> year time_hour origin dest tailnum carrier type  
#> <int> <dttm> <chr> <chr> <chr> <chr> <chr>  
#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA Fixed wing multi en…
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA Fixed wing multi en…
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA Fixed wing multi en…
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 Fixed wing multi en…
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL Fixed wing multi en…
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA Fixed wing multi en…
#> # … with 336,770 more rows, and 2 more variables: engines <int>, seats <int>

```

<a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>이 `x`의 행에 대한 일치 항목을 찾지 못하면 새로운 변수를 결측값으로 채웁니다. 예를 들어, 꼬리 번호가 `N3ALAA`인 비행기에 대한 정보가 없으므로 `type`, `engines`, `seats`는 누락(missing)됩니다.

```

flights2 |>
filter(tailnum == "N3ALAA") |>
left_join(planes |> select(tailnum, type, engines, seats))
#> Joining with `by = join_by(tailnum)`
#> # A tibble: 63 × 9
#> year time_hour origin dest tailnum carrier type engines seats
#> <int> <dttm> <chr> <chr> <chr> <chr> <chr> <int> <int>
#> 1 2013 2013-01-01 06:00:00 LGA ORD N3ALAA AA <NA> NA NA
#> 2 2013 2013-01-02 18:00:00 LGA ORD N3ALAA AA <NA> NA NA
#> 3 2013 2013-01-03 06:00:00 LGA ORD N3ALAA AA <NA> NA NA
#> 4 2013 2013-01-07 19:00:00 LGA ORD N3ALAA AA <NA> NA NA
#> 5 2013 2013-01-08 17:00:00 JFK ORD N3ALAA AA <NA> NA NA
#> 6 2013 2013-01-16 06:00:00 LGA ORD N3ALAA AA <NA> NA NA
#> # … with 57 more rows

```

이 장의 나머지 부분에서 이 문제로 몇 번 더 돌아올 것입니다.

## 조인 키 지정하기(Specifying Join Keys)

기본적으로 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>은 두 데이터 프레임에 모두 나타나는 모든 변수를 조인 키로 사용하며, 이를 이른바 *자연(natural)* 조인이라고 합니다. 이것은 유용한 발견적 방법(heuristic)이지만 항상 작동하는 것은 아닙니다. 예를 들어, `flights2`를 완전한 `planes` 데이터 세트와 조인하려고 하면 어떻게 될까요?

```

flights2 |>
left_join(planes)
#> Joining with `by = join_by(year, tailnum)`
#> # A tibble: 336,776 × 13
#> year time_hour origin dest tailnum carrier type manufacturer
#> <int> <dttm> <chr> <chr> <chr> <chr> <chr> <chr>  
#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA <NA> <NA>  
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA <NA> <NA>  
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA <NA> <NA>  
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 <NA> <NA>  
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL <NA> <NA>  
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA <NA> <NA>  
#> # … with 336,770 more rows, and 5 more variables: model <chr>,
#> # engines <int>, seats <int>, speed <int>, engine <chr>

```

조인이 `tailnum`과 `year`를 복합 키로 사용하려고 하기 때문에 많은 결측 일치(missing matches)가 발생합니다. `flights`와 `planes` 모두 `year` 열을 가지고 있지만 의미하는 바가 다릅니다. `flights$year`는 비행이 발생한 연도이고, `planes$year`는 비행기가 제작된 연도입니다. 우리는 `tailnum`으로만 조인하고 싶으므로 <a href="https://dplyr.tidyverse.org/reference/join_by.html" class="orm:hideurl"><code>join_by()</code></a>를 사용하여 명시적으로 지정해야 합니다.

```

flights2 |>
left_join(planes, join_by(tailnum))
#> # A tibble: 336,776 × 14
#> year.x time_hour origin dest tailnum carrier year.y
#> <int> <dttm> <chr> <chr> <chr> <chr> <int>
#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA 1999
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA 1998
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA 1990
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 2012
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL 1991
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA 2012
#> # … with 336,770 more rows, and 7 more variables: type <chr>,
#> # manufacturer <chr>, model <chr>, engines <int>, seats <int>, …

```

`year` 변수들은 출력에서 접미사(`year.x`와 `year.y`)로 구별되어, 변수가 `x` 인수에서 왔는지 `y` 인수에서 왔는지 알려준다는 점에 유의하세요. `suffix` 인수를 사용하여 기본 접미사를 재정의할 수 있습니다.

`join_by(tailnum)`은 `join_by(tailnum == tailnum)`의 줄임말입니다. 이 더 완전한 형태에 대해 아는 것은 두 가지 이유로 중요합니다. 첫째, 그것은 두 테이블 간의 관계를 설명합니다. 키가 같아야 합니다. 이것이 이런 유형의 조인을 흔히 *동등 조인(equi join)*이라고 부르는 이유입니다. <a href="#sec-non-equi-joins" data-type="xref">"필터링 조인(Filtering Joins)"</a>에서 비동등 조인(non-equi joins)에 대해 배울 것입니다.

둘째, 그것은 각 테이블에서 다른 조인 키를 지정하는 방법입니다. 예를 들어, `flight2`와 `airports` 테이블을 조인하는 방법에는 `dest`로 조인하는 것과 `origin`으로 조인하는 두 가지 방법이 있습니다.

```

flights2 |>
left_join(airports, join_by(dest == faa))
#> # A tibble: 336,776 × 13
#> year time_hour origin dest tailnum carrier name  
#> <int> <dttm> <chr> <chr> <chr> <chr> <chr>  
#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA George Bush Interco…
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA George Bush Interco…
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA Miami Intl  
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 <NA>  
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL Hartsfield Jackson …
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA Chicago Ohare Intl  
#> # … with 336,770 more rows, and 6 more variables: lat <dbl>, lon <dbl>,
#> # alt <dbl>, tz <dbl>, dst <chr>, tzone <chr>

flights2 |>
left_join(airports, join_by(origin == faa))
#> # A tibble: 336,776 × 13
#> year time_hour origin dest tailnum carrier name  
#> <int> <dttm> <chr> <chr> <chr> <chr> <chr>  
#> 1 2013 2013-01-01 05:00:00 EWR IAH N14228 UA Newark Liberty Intl
#> 2 2013 2013-01-01 05:00:00 LGA IAH N24211 UA La Guardia  
#> 3 2013 2013-01-01 05:00:00 JFK MIA N619AA AA John F Kennedy Intl
#> 4 2013 2013-01-01 05:00:00 JFK BQN N804JB B6 John F Kennedy Intl
#> 5 2013 2013-01-01 06:00:00 LGA ATL N668DN DL La Guardia  
#> 6 2013 2013-01-01 05:00:00 EWR ORD N39463 UA Newark Liberty Intl
#> # … with 336,770 more rows, and 6 more variables: lat <dbl>, lon <dbl>,
#> # alt <dbl>, tz <dbl>, dst <chr>, tzone <chr>

```

예전 코드에서는 문자 벡터를 사용하여 조인 키를 지정하는 다른 방법을 볼 수 있습니다.

- `by = "x"`는 `join_by(x)`에 해당합니다.
- `by = c("a" = "x")`는 `join_by(a == x)`에 해당합니다.

이제 <a href="https://dplyr.tidyverse.org/reference/join_by.html" class="orm:hideurl"><code>join_by()</code></a>가 존재하므로, 더 명확하고 유연한 지정을 제공하는 이 함수를 선호합니다.

<a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>inner_join()</code></a>, <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>right_join()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>full_join()</code></a>은 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>left_join()</code></a>과 동일한 인터페이스를 가집니다. 차이점은 유지하는 행이 다르다는 것입니다. 왼쪽 조인(left join)은 `x`의 모든 행을 유지하고, 오른쪽 조인(right join)은 `y`의 모든 행을 유지하며, 전체 조인(full join)은 `x` 또는 `y` 중 하나의 모든 행을 유지하고, 내부 조인(inner join)은 `x`와 `y` 모두에 나타나는 행만 유지합니다. 나중에 이에 대해 자세히 다시 설명하겠습니다.

## 필터링 조인(Filtering Joins)

짐작할 수 있듯이, *필터링 조인(filtering join)*의 주요 동작은 행을 필터링하는 것입니다. 세미 조인(semi-joins)과 안티 조인(anti-joins)이라는 두 가지 유형이 있습니다. *세미 조인*은 `y`에 일치 항목이 있는 `x`의 모든 행을 유지합니다. 예를 들어 세미 조인을 사용하여 `airports` 데이터 세트를 필터링하여 출발 공항만 표시할 수 있습니다.

```

airports |>
semi_join(flights2, join_by(faa == origin))
#> # A tibble: 3 × 8
#> faa name lat lon alt tz dst tzone  
#> <chr> <chr> <dbl> <dbl> <dbl> <dbl> <chr> <chr>  
#> 1 EWR Newark Liberty Intl 40.7 -74.2 18 -5 A America/New_York
#> 2 JFK John F Kennedy Intl 40.6 -73.8 13 -5 A America/New_York
#> 3 LGA La Guardia 40.8 -73.9 22 -5 A America/New_York

```

또는 도착 공항만 표시할 수 있습니다.

```

airports |>
semi_join(flights2, join_by(faa == dest))
#> # A tibble: 101 × 8
#> faa name lat lon alt tz dst tzone  
#> <chr> <chr> <dbl> <dbl> <dbl> <dbl> <chr> <chr>  
#> 1 ABQ Albuquerque Internati… 35.0 -107. 5355 -7 A America/Denver
#> 2 ACK Nantucket Mem 41.3 -70.1 48 -5 A America/New_Yo…
#> 3 ALB Albany Intl 42.7 -73.8 285 -5 A America/New_Yo…
#> 4 ANC Ted Stevens Anchorage… 61.2 -150. 152 -9 A America/Anchor…
#> 5 ATL Hartsfield Jackson At… 33.6 -84.4 1026 -5 A America/New_Yo…
#> 6 AUS Austin Bergstrom Intl 30.2 -97.7 542 -6 A America/Chicago
#> # … with 95 more rows

```

*안티 조인*은 반대입니다. `y`와 일치하지 않는 `x`의 모든 행을 반환합니다. 이는 데이터에 *암시적(implicit)*으로 누락된 값, 즉 <a href="ch18.html#sec-missing-implicit" data-type="xref">"암시적 결측값"</a>이라는 주제에서 다룬 값을 찾는 데 유용합니다. 암시적 결측값은 `NA`로 표시되지 않고 부재(absence)로만 존재합니다. 예를 들어, 일치하는 목적지 공항이 없는 항공편을 찾아봄으로써 `airports`에서 누락된 행을 찾을 수 있습니다.

```

flights2 |>
anti_join(airports, join_by(dest == faa)) |>
distinct(dest)
#> # A tibble: 4 × 1
#> dest
#> <chr>
#> 1 BQN  
#> 2 SJU  
#> 3 STT  
#> 4 PSE

```

또는 `planes`에서 어떤 꼬리 번호(`tailnum`)들이 누락되었는지 알아낼 수 있습니다.

```

flights2 |>
anti_join(planes, join_by(tailnum)) |>
distinct(tailnum)
#> # A tibble: 722 × 1
#> tailnum
#> <chr>  
#> 1 N3ALAA
#> 2 N3DUAA
#> 3 N542MQ
#> 4 N730MQ
#> 5 N9EAMQ
#> 6 N532UA
#> # … with 716 more rows

````

## 연습문제

1. (일 년 전체에 걸쳐) 지연이 가장 심한 48시간을 찾으세요. 이를 `weather` 데이터와 교차 참조(Cross-reference)해 보세요. 어떤 패턴이 보이나요?

2. 다음 코드를 사용하여 가장 인기 있는 상위 10개의 목적지를 찾았다고 상상해 보세요.

    ```
    top_dest <- flights2 |>
      count(dest, sort = TRUE) |>
      head(10)
    ```

    이러한 목적지로 향하는 모든 항공편을 어떻게 찾을 수 있나요?

3. 출발하는 모든 항공편에 해당 시간대의 날씨 데이터가 대응되어 있나요?

4. `planes`에 일치하는 레코드가 없는 꼬리 번호들의 공통점은 무엇인가요? (힌트: 한 변수가 문제의 약 90%를 설명합니다.)

5. 해당 비행기를 조종한 모든 `carrier`를 나열하는 열을 `planes`에 추가하세요. 각 비행기는 단일 항공사에서 비행하기 때문에 비행기와 항공사 사이에 암시적 관계가 있을 것으로 예상할 수 있습니다. 이전 장들에서 배운 도구를 사용하여 이 가설을 확인하거나 기각하세요.

6. 출발 공항*과* 도착 공항의 위도와 경도를 `flights`에 추가하세요. 조인 전과 조인 후 중 언제가 열 이름을 바꾸기 더 쉽나요?

7. 목적지별 평균 지연 시간을 계산한 다음 `airports` 데이터 프레임에 조인하여 지연의 공간적 분포를 표시할 수 있도록 하세요. 미국 지도를 그리는 쉬운 방법은 다음과 같습니다.

    ```
    airports |>
      semi_join(flights, join_by(faa == dest)) |>
      ggplot(aes(x = lon, y = lat)) +
        borders("state") +
        geom_point() +
        coord_quickmap()
    ```

    점의 `size`나 `color`를 사용하여 각 공항의 평균 지연을 표시하고 싶을 수 있습니다.
8. 2013년 6월 13일에 무슨 일이 있었나요? 지연 지도를 그리고 Google을 사용하여 날씨와 교차 참조하세요.

# 조인은 어떻게 작동하는가?

이제 조인을 몇 번 사용해 보았으므로, `x`의 각 행이 `y`의 행과 어떻게 일치하는지에 중점을 두고 그것들이 어떻게 작동하는지 더 자세히 알아볼 때입니다. 다음에 정의되고 <a href="#fig-join-setup" data-type="xref">그림 19-2</a>에 표시된 간단한 티블(tibbles)을 사용하여 조인의 시각적 표현을 소개하는 것으로 시작하겠습니다. 이러한 예제에서는 `key`라는 단일 키와 단일 값 열(`val_x` 및 `val_y`)을 사용하지만, 이러한 아이디어는 모두 다중 키와 다중 값으로 일반화됩니다.

````

x <- tribble(
~key, ~val_x,
1, "x1",
2, "x2",
3, "x3"
)
y <- tribble(
~key, ~val_y,
1, "y1",
2, "y2",
4, "y3"
)

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1902.png" alt="x와 y는 본문에 설명된 내용을 포함하는 2개의 열과 3개의 행을 가진 두 개의 데이터 프레임입니다. 키 값에 색상이 지정되어 있습니다. 1은 녹색, 2는 보라색, 3은 주황색, 4는 노란색입니다." />
<h6 id="figure-19-2.-graphical-representation-of-two-simple-tables.-the-colored-key-columns-map-background-color-to-key-value.-the-gray-columns-represent-the-value-columns-that-are-carried-along-for-the-ride.">그림 19-2. 두 개의 간단한 테이블에 대한 시각적 표현. 색상이 지정된 <code>key</code> 열은 배경색을 키 값에 매핑합니다. 회색 열은 함께 따라오는(carried along for the ride) "값(value)" 열을 나타냅니다.</h6>
</figure>

<a href="#fig-join-setup2" data-type="xref">그림 19-3</a>은 시각적 표현의 기초를 소개합니다. `x`의 각 행과 `y`의 각 행에서 그려진 선들의 교차점으로 `x`와 `y` 사이의 모든 잠재적 일치 항목을 보여줍니다. 출력의 행과 열은 주로 `x`에 의해 결정되므로 `x` 테이블은 가로로 놓여 출력과 정렬됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1903.png" alt="x와 y는 x에서 뻗어나가는 가로선과 y에서 뻗어나가는 세로선과 함께 직각으로 배치됩니다. x에 3개의 행이 있고 y에 3개의 행이 있어, 9개의 잠재적인 일치를 나타내는 9개의 교차점이 생깁니다." />
<h6 id="figure-19-3.-to-understand-how-joins-work-its-useful-to-think-of-every-possible-match.-here-we-show-that-with-a-grid-of-connecting-lines.">그림 19-3. 조인이 어떻게 작동하는지 이해하려면 가능한 모든 일치 항목을 생각해보는 것이 유용합니다. 여기서는 연결 선들의 격자(grid)로 이를 보여줍니다.</h6>
</figure>

특정 유형의 조인을 설명하기 위해 점으로 일치 항목을 나타냅니다. 일치 항목은 키, x 값 및 y 값을 포함하는 새로운 데이터 프레임인 출력의 행을 결정합니다. 예를 들어, <a href="#fig-join-inner" data-type="xref">그림 19-4</a>는 내부 조인(inner join)을 보여주며, 키가 같을 때만 행이 유지됩니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1904.png" alt="x와 y는 잠재적 일치의 격자를 형성하는 선들과 함께 직각으로 배치됩니다. 키 1과 2는 x와 y 모두에 나타나므로 점으로 표시된 일치 항목을 얻습니다. 각 점은 출력의 한 행에 해당하므로 결과 조인 데이터 프레임에는 두 개의 행이 있습니다." />
<h6 id="figure-19-4.-an-inner-join-matches-each-row-in-x-to-the-row-in-y-that-has-the-same-value-of-key.-each-match-becomes-a-row-in-the-output.">그림 19-4. 내부 조인은 <code>x</code>의 각 행을 <code>key</code> 값이 같은 <code>y</code>의 행과 일치시킵니다. 각 일치 항목은 출력에서 하나의 행이 됩니다.</h6>
</figure>

적어도 하나의 데이터 프레임에 나타나는 관측치를 유지하는 *외부 조인(outer joins)*을 설명하는 데 동일한 원리를 적용할 수 있습니다. 이러한 조인은 각 데이터 프레임에 추가적인 "가상(virtual)" 관측치를 추가하여 작동합니다. 이 관측치에는 다른 어떤 키와도 일치하지 않는 경우에 일치하는 키가 있으며, 값은 `NA`로 채워져 있습니다. 외부 조인에는 3가지 유형이 있습니다.

- *왼쪽 조인(left join)*은 <a href="#fig-join-left" data-type="xref">그림 19-5</a>에 표시된 것처럼 `x`의 모든 관측치를 유지합니다. `x`의 모든 행은 `y`에 있는 `NA` 행과의 일치로 넘어갈 수 있기 때문에(fall back) 출력에 보존됩니다.

  <figure>
  <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1905.png" alt="내부 조인을 보여주는 이전 다이어그램과 비교하여, y 테이블에는 다르게 일치하지 않은 x의 어떤 행과도 일치할 NA가 포함된 새로운 가상 행이 생깁니다. 이는 이제 출력에 세 개의 행이 있음을 의미합니다. 이 가상 행과 일치하는 키 = 3의 경우 val_y는 NA 값을 갖습니다." />
  <h6 id="figure-19-5.-a-visual-representation-of-the-left-join-where-every-row-in-x-appears-in-the-output.">그림 19-5. <code>x</code>의 모든 행이 출력에 나타나는 왼쪽 조인의 시각적 표현.</h6>
  </figure>

- *오른쪽 조인(right join)*은 <a href="#fig-join-right" data-type="xref">그림 19-6</a>에 표시된 것처럼 `y`의 모든 관측치를 유지합니다. `y`의 모든 행은 `x`에 있는 `NA` 행과의 일치로 넘어갈 수 있기 때문에 출력에 보존됩니다. 출력은 여전히 가능한 한 많이 `x`와 일치합니다. `y`에서 추가된 모든 행은 끝에 추가됩니다.

  <figure>
  <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1906.png" alt="왼쪽 조인을 보여주는 이전 다이어그램과 비교하여, 이제 x 테이블이 가상 행을 얻어 y의 모든 행이 x에서 일치 항목을 얻습니다. val_x는 x와 일치하지 않는 y의 행에 대해 NA를 포함합니다." />
  <h6 id="figure-19-6.-a-visual-representation-of-the-right-join-where-every-row-of-y-appears-in-the-output.">그림 19-6. <code>y</code>의 모든 행이 출력에 나타나는 오른쪽 조인의 시각적 표현.</h6>
  </figure>

- *전체 조인(full join)*은 <a href="#fig-join-full" data-type="xref">그림 19-7</a>에 표시된 것처럼 `x` 또는 `y`에 나타나는 모든 관측치를 유지합니다. `x`와 `y` 모두 `NA`라는 예비(fallback) 행을 가지고 있기 때문에 `x`와 `y`의 모든 행이 출력에 포함됩니다. 여기서도 출력은 `x`의 모든 행으로 시작하고 일치하지 않는 나머지 `y` 행이 그 뒤를 따릅니다.

  <figure>
  <img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1907.png" alt="이제 x와 y 모두 항상 일치하는 가상 행을 갖습니다. 결과는 4개의 행이 있습니다. val_x와 val_y의 모든 값을 갖는 키 1, 2, 3, 4. 그러나 키 2의 val_y와 키 4의 val_x는 다른 데이터 프레임에 일치 항목이 없기 때문에 NA입니다." />
  <h6 id="figure-19-7.-a-visual-representation-of-the-full-join-where-every-row-in-x-and-y-appears-in-the-output.">그림 19-7. <code>x</code>와 <code>y</code>의 모든 행이 출력에 나타나는 전체 조인의 시각적 표현.</h6>
  </figure>

외부 조인 유형이 어떻게 다른지 보여주는 또 다른 방법은 <a href="#fig-join-venn" data-type="xref">그림 19-8</a>과 같은 벤 다이어그램(Venn diagram)을 이용하는 것입니다. 하지만 이것은 어떤 행이 보존되는지 기억을 되살려줄 수는 있지만, 열(columns)에 어떤 일이 일어나는지 설명하지는 못하기 때문에 훌륭한 표현은 아닙니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1908.png" alt="내부, 전체, 왼쪽, 오른쪽 조인에 대한 벤 다이어그램. 각 조인은 데이터 프레임 x와 y를 나타내는 두 개의 교차하는 원으로 표시되며 x는 오른쪽에, y는 왼쪽에 있습니다. 음영은 조인의 결과를 나타냅니다." />
<h6 id="figure-19-8.-venn-diagrams-showing-the-difference-between-inner-left-right-and-full-joins.">그림 19-8. 내부 조인, 왼쪽 조인, 오른쪽 조인, 전체 조인의 차이를 보여주는 벤 다이어그램.</h6>
</figure>

여기에 표시된 조인은 이른바 *동등 조인(equi joins)*으로, 키가 같으면 행이 일치합니다. 동등 조인은 가장 흔한 조인 유형이므로, 우리는 일반적으로 equi라는 접두사를 생략하고 "동등 내부 조인" 대신 그냥 "내부 조인"이라고 말할 것입니다. <a href="#sec-non-equi-joins" data-type="xref">"필터링 조인(Filtering Joins)"</a>에서 비동등 조인으로 돌아오겠습니다.

## 행 일치(Row Matching)

지금까지 우리는 `x`의 행이 `y`의 0개 또는 1개의 행과 일치하는 경우 어떤 일이 일어나는지 탐색했습니다. 두 개 이상의 행과 일치하면 어떻게 될까요? 무슨 일이 일어나고 있는지 이해하기 위해 먼저 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>inner_join()</code></a>으로 초점을 좁힌 다음, <a href="#fig-join-match-types" data-type="xref">그림 19-9</a>와 같이 그림을 그려봅시다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1909.png" alt="x가 키 값 1, 2, 3을 가지고 y가 키 값 1, 2, 2를 가지는 조인 다이어그램. 키 1은 하나의 행과 일치하고, 키 2는 두 개의 행과 일치하며, 키 3은 0개의 행과 일치하므로 출력에는 세 개의 행이 있습니다." />
<h6 id="figure-19-9.-the-three-ways-a-row-in-x-can-match.-x1-matches-one-row-in-y-x2-matches-two-rows-in-y-and-x3-matches-zero-rows-in-y.-note-that-while-there-are-three-rows-in-x-and-three-rows-in-the-output-there-isnt-a-direct-correspondence-between-the-rows.">그림 19-9. <code>x</code>의 행이 일치할 수 있는 세 가지 방법. <code>x1</code>은 <code>y</code>의 한 행과 일치하고, <code>x2</code>는 <code>y</code>의 두 행과 일치하며, <code>x3</code>은 <code>y</code>의 어떤 행과도 일치하지 않습니다(0 rows). <code>x</code>에 세 개의 행이 있고 출력에 세 개의 행이 있지만, 그 행들 사이에 직접적인 대응 관계는 없다는 점에 유의하세요.</h6>
</figure>

`x`의 행에 대해 가능한 결과는 세 가지입니다.

- 아무것도 일치하지 않으면 삭제됩니다.
- `y`의 한 행과 일치하면 보존됩니다.
- `y`의 두 개 이상의 행과 일치하면, 각 일치 항목마다 한 번씩 복제됩니다.

원칙적으로 이는 출력의 행과 `x`의 행 사이에 보장된 대응 관계가 없음을 의미하지만, 실제로는 이것이 문제를 일으키는 경우가 거의 없습니다. 그러나 행의 조합 폭발(combinatorial explosion)을 일으킬 수 있는 특히 위험한 경우가 하나 있습니다. 다음 두 테이블을 조인한다고 상상해 보세요.

```

df1 <- tibble(key = c(1, 2, 2), val_x = c("x1", "x2", "x3"))
df2 <- tibble(key = c(1, 2, 2), val_y = c("y1", "y2", "y3"))

```

`df1`의 첫 번째 행은 `df2`의 한 행과만 일치하지만 두 번째와 세 번째 행은 모두 두 개의 행과 일치합니다. 이것을 종종 *다대다(many-to-many)* 조인이라고 부르며 dplyr이 경고를 출력하게 합니다.

```

df1 |>
inner_join(df2, join_by(key))
#> Warning in inner_join(df1, df2, join_by(key)):
#> Detected an unexpected many-to-many relationship between `x` and `y`.
#> ℹ Row 2 of `x` matches multiple rows in `y`.
#> ℹ Row 2 of `y` matches multiple rows in `x`.
#> ℹ If a many-to-many relationship is expected, set `relationship =
#>   "many-to-many"` to silence this warning.
#> # A tibble: 5 × 3
#> key val_x val_y
#> <dbl> <chr> <chr>
#> 1 1 x1 y1  
#> 2 2 x2 y2  
#> 3 2 x2 y3  
#> 4 2 x3 y2  
#> 5 2 x3 y3

```

이것을 의도적으로 수행하는 경우 경고에서 제안하는 대로 `relationship = "many-to-many"`로 설정할 수 있습니다.

## 필터링 조인

일치 횟수는 필터링 조인의 동작도 결정합니다. 세미 조인은 <a href="#fig-join-semi" data-type="xref">그림 19-10</a>에서와 같이 `y`에 하나 이상의 일치 항목이 있는 `x`의 행을 유지합니다. 안티 조인은 <a href="#fig-join-anti" data-type="xref">그림 19-11</a>에서와 같이 `y`에 일치하는 행이 없는 `x`의 행을 유지합니다. 두 경우 모두 일치의 존재 여부만이 중요합니다. 얼마나 여러 번 일치하는지는 중요하지 않습니다. 이는 필터링 조인이 변형 조인처럼 행을 절대로 복제하지 않는다는 것을 의미합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1910.png" alt="오래된 친구 x와 y가 있는 조인 다이어그램. 세미 조인에서는 일치하는 존재만 중요하므로 출력에는 x와 동일한 열이 포함됩니다." />
<h6 id="figure-19-10.-in-a-semi-join-it-only-matters-that-there-is-a-match-otherwise-values-in-y-dont-affect-the-output.">그림 19-10. 세미 조인에서는 일치 항목이 있다는 사실만 중요합니다. 그렇지 않으면 <code>y</code>의 값이 출력에 영향을 미치지 않습니다.</h6>
</figure>

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1911.png" alt="안티 조인은 세미 조인의 반대이므로 일치 항목이 빨간색 선으로 그려져 출력에서 삭제될 것임을 나타냅니다." />
<h6 id="figure-19-11.-an-anti-join-is-the-inverse-of-a-semi-join-dropping-rows-from-x-that-have-a-match-in-y.">그림 19-11. 안티 조인은 세미 조인의 역으로, <code>y</code>에 일치 항목이 있는 <code>x</code>의 행을 삭제합니다.</h6>
</figure>

# 비동등 조인(Non-Equi Joins)

지금까지 여러분은 동등 조인, 즉 `x` 키가 `y` 키와 같을 때 행이 일치하는 조인만 보았습니다. 이제 우리는 그 제한을 완화하고 한 쌍의 행이 일치하는지 확인하는 다른 방법에 대해 논의할 것입니다.

하지만 그러기 전에 이전에 만들었던 단순화를 다시 살펴볼 필요가 있습니다. 동등 조인에서는 `x` 키와 `y` 키가 항상 같기 때문에 출력에는 하나만 표시하면 됩니다. `keep = TRUE`를 사용하여 두 키를 모두 유지하도록 dplyr에 요청할 수 있으며, 이는 다음 코드와 <a href="#fig-inner-both" data-type="xref">그림 19-12</a>에 다시 그려진 <a href="https://dplyr.tidyverse.org/reference/mutate-joins.html" class="orm:hideurl"><code>inner_join()</code></a>으로 이어집니다.

```

x |> left_join(y, by = "key", keep = TRUE)
#> # A tibble: 3 × 4
#> key.x val_x key.y val_y
#> <dbl> <chr> <dbl> <chr>
#> 1 1 x1 1 y1  
#> 2 2 x2 2 y2  
#> 3 3 x3 NA <NA>

```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1912.png" alt="x와 y 사이의 내부 조인을 보여주는 조인 다이어그램. 이제 결과에는 key.x, val_x, key.y 및 val_y의 4개 열이 포함됩니다. key.x와 key.y의 값은 동일하며, 이것이 우리가 보통 하나만 표시하는 이유입니다." />
<h6 id="figure-19-12.-an-inner-join-showing-both-x-and-y-keys-in-the-output.">그림 19-12. 출력에 <code>x</code>와 <code>y</code> 키를 모두 보여주는 내부 조인.</h6>
</figure>

동등 조인에서 벗어나면 키 값들이 종종 다를 것이기 때문에 항상 키를 표시하게 될 것입니다. 예를 들어 `x$key`와 `y$key`가 같을 때만 일치시키는 대신 `x$key`가 `y$key`보다 크거나 같을 때마다 일치시킬 수 있으며, 이는 <a href="#fig-join-gte" data-type="xref">그림 19-13</a>으로 이어집니다. dplyr의 조인 함수들은 동등 조인과 비동등 조인 사이의 이러한 차이를 이해하므로 비동등 조인을 수행할 때 항상 양쪽 키를 모두 표시합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1913.png" alt="join_by(key &gt;= key)를 설명하는 조인 다이어그램. x의 첫 번째 행은 y의 한 행과 일치하고 두 번째와 세 번째 행은 각각 두 개의 행과 일치합니다. 이는 출력에 다음 (key.x, key.y) 쌍을 각각 포함하는 5개의 행이 있음을 의미합니다. (1, 1), (2, 1), (2, 2), (3, 1), (3, 2)." />
<h6 id="figure-19-13.-a-non-equi-join-where-the-x-key-must-be-greater-than-or-equal-to-the-y-key.-many-rows-generate-multiple-matches.">그림 19-13. <code>x</code> 키가 <code>y</code> 키보다 크거나 같아야 하는 비동등 조인. 많은 행들이 다중 일치를 생성합니다.</h6>
</figure>

비동등 조인(non-equi join)이라는 용어는 그것이 무엇인지가 아니라 조인이 아닌 것만 알려주기 때문에 그다지 유용한 용어는 아닙니다. dplyr은 특히 유용한 4가지 유형의 비동등 조인을 식별하여 도움을 줍니다.

교차 조인(Cross joins)
모든 행의 쌍을 일치시킵니다.

부등 조인(Inequality joins)
`==` 대신 `<`, `<=`, `>`, `>=`를 사용합니다.

롤링 조인(Rolling joins)
부등 조인과 유사하지만 가장 가까운 일치 항목만 찾습니다.

중첩 조인(Overlap joins)
범위(ranges)와 함께 작동하도록 설계된 특별한 유형의 부등 조인입니다.

이들 각각은 다음 섹션에서 더 자세히 설명됩니다.

## 교차 조인(Cross Joins)

교차 조인은 <a href="#fig-join-cross" data-type="xref">그림 19-14</a>에서처럼 모든 것을 일치시켜 행의 데카르트 곱(Cartesian product)을 생성합니다. 이는 출력에 `nrow(x) * nrow(y)`개의 행이 있다는 것을 의미합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1914.png" alt="x와 y의 모든 조합에 대한 점을 보여주는 조인 다이어그램." />
<h6 id="figure-19-14.-a-cross-join-matches-each-row-in-x-with-every-row-in-y.">그림 19-14. 교차 조인은 <code>x</code>의 각 행을 <code>y</code>의 모든 행과 일치시킵니다.</h6>
</figure>

교차 조인은 순열을 생성할 때 유용합니다. 예를 들어, 다음 코드는 가능한 모든 이름 쌍을 생성합니다. 우리는 `df`를 자기 자신에게 조인하고 있으므로 이것을 *셀프 조인(self-join)*이라고 부르기도 합니다. 교차 조인은 모든 행을 일치시킬 때 내부/왼쪽/오른쪽/전체 사이에 구분이 없기 때문에 다른 조인 함수를 사용합니다.

```

df <- tibble(name = c("John", "Simon", "Tracy", "Max"))
df |> cross_join(df)
#> # A tibble: 16 × 2
#> name.x name.y
#> <chr> <chr>
#> 1 John John  
#> 2 John Simon
#> 3 John Tracy
#> 4 John Max  
#> 5 Simon John  
#> 6 Simon Simon
#> # … with 10 more rows

```

## 부등 조인(Inequality Joins)

부등 조인은 <a href="#fig-join-gte" data-type="xref">그림 19-13</a> 및 <a href="#fig-join-lt" data-type="xref">그림 19-15</a>와 같이 `<`, `<=`, `>=`, 또는 `>`를 사용하여 가능한 일치 집합을 제한합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1915.png" alt="x의 키가 y의 키보다 작은 행에서 데이터 프레임 x가 데이터 프레임 y에 의해 조인되어 왼쪽 상단에 삼각형 모양을 만드는 부등 조인을 묘사한 다이어그램." />
<h6 id="figure-19-15.-an-inequality-join-where-x-is-joined-to-y-on-rows-where-the-key-of-x-is-less-than-the-key-of-y.-this-makes-a-triangular-shape-in-the-top-left-corner.">그림 19-15. <code>x</code>의 키가 <code>y</code>의 키보다 작은 행에 대해 <code>x</code>가 <code>y</code>에 조인되는 부등 조인. 이는 왼쪽 상단에 삼각형 모양을 만듭니다.</h6>
</figure>

부등 조인은 매우 일반적이어서, 너무 일반적이라 의미 있는 특정 사용 사례를 생각해내기 어렵습니다. 한 가지 작지만 유용한 기법은 교차 조인을 제한하는 데 사용하여 모든 순열을 생성하는 대신 모든 조합을 생성하는 것입니다.

```

df <- tibble(id = 1:4, name = c("John", "Simon", "Tracy", "Max"))

df |> left_join(df, join_by(id < id))
#> # A tibble: 7 × 4
#> id.x name.x id.y name.y
#> <int> <chr> <int> <chr>
#> 1 1 John 2 Simon
#> 2 1 John 3 Tracy
#> 3 1 John 4 Max  
#> 4 2 Simon 3 Tracy
#> 5 2 Simon 4 Max  
#> 6 3 Tracy 4 Max  
#> # … with 1 more row

```

## 롤링 조인(Rolling Joins)

롤링 조인은 부등식을 만족하는 *모든* 행을 얻는 대신, <a href="#fig-join-closest" data-type="xref">그림 19-16</a>과 같이 가장 가까운 행 하나만 얻는 특별한 유형의 부등 조인입니다. `closest()`를 추가하여 모든 부등 조인을 롤링 조인으로 바꿀 수 있습니다. 예를 들어, `join_by(closest(x <= y))`는 x보다 크거나 같은 것 중 가장 작은 `y`와 일치시키고, `join_by(closest(x > y))`는 `x`보다 작은 것 중 가장 큰 `y`와 일치시킵니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R 소스를 위한 데이터 사이언스/assets/rds2_1916.png" alt="롤링 조인은 부등 조인의 하위 집합이므로 일부 일치 항목은 &quot;가장 가까운&quot; 것이 아니기 때문에 사용되지 않음을 나타내기 위해 회색으로 표시됩니다." />
<h6 id="figure-19-16.-a-rolling-join-is-similar-to-a-greater-than-or-equal-inequality-join-but-matches-only-the-first-value.">그림 19-16. 롤링 조인은 크거나 같은(greater-than-or-equal) 부등 조인과 유사하지만 첫 번째 값만 일치시킵니다.</h6>
</figure>

롤링 조인은 완벽하게 줄지어 있지 않은 두 개의 날짜 테이블이 있고, 예를 들어 테이블 2의 어떤 날짜 이전(또는 이후)에 오는 테이블 1에서 가장 가까운 날짜를 찾고자 할 때 특히 유용합니다.

예를 들어, 여러분이 사무실의 파티 계획 위원회를 맡고 있다고 상상해 보세요. 회사 예산이 좀 쪼들려서 개별 파티를 여는 대신 매 분기마다 한 번씩만 파티를 엽니다. 파티가 언제 열릴지 결정하는 규칙은 약간 복잡합니다. 파티는 항상 월요일에 열리고, 많은 사람들이 휴가 중이므로 1월의 첫째 주는 건너뛰며, 2022년 3분기의 첫 번째 월요일은 7월 4일이므로 일주일 뒤로 미뤄야 합니다. 그러면 다음 파티 날짜들이 나옵니다.

```

parties <- tibble(
q = 1:4,
party = ymd(c("2022-01-10", "2022-04-04", "2022-07-11", "2022-10-03"))
)

```

이제 다음과 같은 직원 생일 테이블이 있다고 상상해 보세요.
```

employees <- tibble(
name = sample(babynames::babynames$name, 100),
birthday = ymd("2022-01-01") + (sample(365, 100, replace = TRUE) - 1)
)
employees
#> # A tibble: 100 × 2
#> name birthday  
#> <chr> <date>  
#> 1 Case 2022-09-13
#> 2 Shonnie 2022-03-30
#> 3 Burnard 2022-01-10
#> 4 Omer 2022-11-25
#> 5 Hillel 2022-07-30
#> 6 Curlie 2022-12-11
#> # … with 94 more rows

```

그리고 각 직원에 대해 생일 이후(또는 당일)에 열리는 첫 번째 파티 날짜를 찾고자 합니다. 롤링 조인으로 이를 표현할 수 있습니다.

```

employees |>
left_join(parties, join_by(closest(birthday >= party)))
#> # A tibble: 100 × 4
#> name birthday q party  
#> <chr> <date> <int> <date>  
#> 1 Case 2022-09-13 3 2022-07-11
#> 2 Shonnie 2022-03-30 1 2022-01-10
#> 3 Burnard 2022-01-10 1 2022-01-10
#> 4 Omer 2022-11-25 4 2022-10-03
#> 5 Hillel 2022-07-30 3 2022-07-11
#> 6 Curlie 2022-12-11 4 2022-10-03
#> # … with 94 more rows

```

그러나 이 접근법에는 한 가지 문제가 있습니다. 생일이 1월 10일 이전인 사람들은 파티를 열지 못합니다.

```

employees |>
anti_join(parties, join_by(closest(birthday >= party)))
#> # A tibble: 0 × 2
#> # … with 2 variables: name <chr>, birthday <date>

```

이 문제를 해결하기 위해 중첩 조인(overlap joins)이라는 다른 방법으로 문제에 접근해야 합니다.

## 중첩 조인(Overlap Joins)

중첩 조인은 구간(intervals)을 더 쉽게 사용할 수 있도록 부등 조인을 사용하는 세 가지 헬퍼(helpers)를 제공합니다.

- `between(x, y_lower, y_upper)`는 `x >= y_lower, x <= y_upper`의 약어입니다.
- `within(x_lower, x_upper, y_lower, y_upper)`는 `x_lower >= y_lower, x_upper <= y_upper`의 약어입니다.
- `overlaps(x_lower, x_upper, y_lower, y_upper)`는 `x_lower <= y_upper, x_upper >= y_lower`의 약어입니다.

이것들을 어떻게 사용할 수 있는지 알아보기 위해 생일 예제를 계속해 보겠습니다. 이전에 사용했던 전략에는 한 가지 문제가 있습니다. 1월 1일부터 9일까지의 생일 이전에는 파티가 없습니다. 따라서 각 파티가 걸쳐 있는 날짜 범위를 명시적으로 지정하고 이러한 이른 생일에 대한 특별한 경우를 만드는 것이 더 나을 수 있습니다.

```

parties <- tibble(
q = 1:4,
party = ymd(c("2022-01-10", "2022-04-04", "2022-07-11", "2022-10-03")),
start = ymd(c("2022-01-01", "2022-04-04", "2022-07-11", "2022-10-03")),
end = ymd(c("2022-04-03", "2022-07-11", "2022-10-02", "2022-12-31"))
)
parties
#> # A tibble: 4 × 4
#> q party start end  
#> <int> <date> <date> <date>  
#> 1 1 2022-01-10 2022-01-01 2022-04-03
#> 2 2 2022-04-04 2022-04-04 2022-07-11
#> 3 3 2022-07-11 2022-07-11 2022-10-02
#> 4 4 2022-10-03 2022-10-03 2022-12-31

```

헤들리(Hadley)는 데이터 입력을 절망적일 정도로 잘 못하므로, 파티 기간이 겹치지 않는지도 확인하고 싶었습니다. 이를 수행하는 한 가지 방법은 셀프 조인(self-join)을 사용하여 시작-종료 구간이 다른 구간과 겹치는지 여부를 확인하는 것입니다.

```

parties |>
inner_join(parties, join_by(overlaps(start, end, start, end), q < q)) |>
select(start.x, end.x, start.y, end.y)
#> # A tibble: 1 × 4
#> start.x end.x start.y end.y  
#> <date> <date> <date> <date>  
#> 1 2022-04-04 2022-07-11 2022-07-11 2022-10-02

```

이런, 중복되는 부분이 있네요. 해당 문제를 수정하고 계속합시다.

```

parties <- tibble(
q = 1:4,
party = ymd(c("2022-01-10", "2022-04-04", "2022-07-11", "2022-10-03")),
start = ymd(c("2022-01-01", "2022-04-04", "2022-07-11", "2022-10-03")),
end = ymd(c("2022-04-03", "2022-07-10", "2022-10-02", "2022-12-31"))
)

```

이제 각 직원을 해당 파티와 일치시킬 수 있습니다. 직원이 파티에 할당되지 않은 경우가 있는지 빨리 알아보고 싶기 때문에 `unmatched = "error"`를 사용하기 좋은 지점입니다.

```

employees |>
inner_join(parties, join_by(between(birthday, start, end)), unmatched = "error")
#> # A tibble: 100 × 6
#> name birthday q party start end  
#> <chr> <date> <int> <date> <date> <date>  
#> 1 Case 2022-09-13 3 2022-07-11 2022-07-11 2022-10-02
#> 2 Shonnie 2022-03-30 1 2022-01-10 2022-01-01 2022-04-03
#> 3 Burnard 2022-01-10 1 2022-01-10 2022-01-01 2022-04-03
#> 4 Omer 2022-11-25 4 2022-10-03 2022-10-03 2022-12-31
#> 5 Hillel 2022-07-30 3 2022-07-11 2022-07-11 2022-10-02
#> 6 Curlie 2022-12-11 4 2022-10-03 2022-10-03 2022-12-31
#> # … with 94 more rows

````

## 연습문제

1. 이 동등 조인(equi join)에서 키에 무슨 일이 일어나고 있는지 설명할 수 있나요? 왜 서로 다를까요?

    ```
    x |> full_join(y, by = "key")
    #> # A tibble: 4 × 3
    #>     key val_x val_y
    #>   <dbl> <chr> <chr>
    #> 1     1 x1    y1
    #> 2     2 x2    y2
    #> 3     3 x3    <NA>
    #> 4     4 <NA>  y3

    x |> full_join(y, by = "key", keep = TRUE)
    #> # A tibble: 4 × 4
    #>   key.x val_x key.y val_y
    #>   <dbl> <chr> <dbl> <chr>
    #> 1     1 x1        1 y1
    #> 2     2 x2        2 y2
    #> 3     3 x3       NA <NA>
    #> 4    NA <NA>      4 y3
    ```

2. 파티 기간이 다른 파티 기간과 겹치는지 확인할 때 <a href="https://dplyr.tidyverse.org/reference/join_by.html" class="orm:hideurl"><code>join_by()</code></a>에 `q < q`를 사용했습니다. 그 이유는 무엇일까요? 이 부등식을 제거하면 어떻게 될까요?

# 요약

이 장에서는 변형 조인과 필터링 조인을 사용하여 한 쌍의 데이터 프레임에서 데이터를 결합하는 방법을 배웠습니다. 그 과정에서 키를 식별하는 방법을 배웠고 기본 키와 외래 키의 차이점을 배웠습니다. 또한 조인이 작동하는 방식과 출력에 행이 몇 개나 될지 알아내는 방법을 이해했습니다. 마지막으로 비동등 조인의 강력함을 살짝 엿보고 흥미로운 사용 사례 몇 가지를 살펴보았습니다.

이 장을 끝으로 개별 열(columns) 및 티블(tibbles)과 함께 사용할 수 있는 도구에 중점을 둔 책의 "변환(Transform)" 부분이 마무리됩니다. 여러분은 논리형 벡터, 숫자, 전체 테이블로 작업하기 위한 dplyr 및 기본(base) 함수, 문자열 작업을 위한 stringr 함수, 날짜-시간 작업을 위한 lubridate 함수, 요인(factors) 작업을 위한 forcats 함수에 대해 배웠습니다.

이 책의 다음 부분에서는 다양한 유형의 데이터를 타이디(tidy)한 형태로 R로 가져오는 방법에 대해 자세히 알아봅니다.

<sup>[1](ch19.html#idm44771284613936-marker)</sup> RStudio에서는 이 문제를 피하기 위해 <a href="https://rdrr.io/r/utils/View.html" class="orm:hideurl"><code>View()</code></a>도 사용할 수 있다는 점을 기억하세요.

<sup>[2](ch19.html#idm44771284529504-marker)</sup> 100% 사실은 아니지만 그렇지 않을 때마다 경고를 받게 될 것입니다.
````
