# 17장. 날짜와 시간(Dates and Times)

# 소개

이 장에서는 R에서 날짜와 시간을 다루는 방법을 보여줄 것입니다. 처음 보기에 날짜와 시간은 단순해 보입니다. 일상생활에서 항상 사용하고 있으며 큰 혼란을 일으키지 않는 것 같습니다. 하지만 날짜와 시간에 대해 더 많이 배울수록 그것들은 더 복잡해 보입니다!

워밍업으로, 일 년에 며칠이 있고 하루에 몇 시간이 있는지 생각해 보세요. 대부분의 해에는 365일이 있지만 윤년에는 366일이 있다는 것을 기억하셨을 것입니다. 어느 해가 윤년인지 결정하는 전체 규칙을 알고 계신가요?<sup><a href="ch17.html#idm44771289225952" id="idm44771289225952-marker" data-type="noteref">1</a></sup> 하루의 시간 수는 조금 덜 명확합니다. 대부분의 날은 24시간이지만, 일광 절약 시간제(DST, 서머타임)를 사용하는 곳에서는 매년 하루는 23시간이고 다른 하루는 25시간입니다.

날짜와 시간이 어려운 이유는 두 가지 물리적 현상(지구의 자전과 태양 주위의 공전)을 월(month), 시간대(time zone) 및 DST를 포함한 수많은 지정학적 현상과 조화시켜야 하기 때문입니다. 이 장에서 날짜와 시간에 대한 모든 세부 사항을 다 가르쳐 주지는 않겠지만, 일반적인 데이터 분석 문제를 해결하는 데 도움이 될 실용적인 기술의 탄탄한 기반을 제공할 것입니다.

다양한 입력으로부터 날짜-시간(date-times)을 생성하는 방법을 보여주는 것으로 시작하여, 날짜-시간을 얻은 후에는 연, 월, 일과 같은 구성 요소를 추출하는 방법을 배울 것입니다. 그런 다음, 여러분이 하려는 작업에 따라 다양한 형태(flavors)로 제공되는 시간 범위(time spans)를 다루는 까다로운 주제로 깊이 들어갈 것입니다. 시간대가 제기하는 추가적인 문제에 대한 간략한 논의로 마무리하겠습니다.

## 사전 준비

이 장에서는 R에서 날짜와 시간을 더 쉽게 다룰 수 있게 해주는 lubridate 패키지에 초점을 맞출 것입니다. 최신 tidyverse 릴리스 기준으로 lubridate는 핵심 tidyverse의 일부입니다. 실습 데이터를 위해 nycflights13도 필요합니다.

```
library(tidyverse)
library(nycflights13)
```

# 날짜/시간 생성하기

시간상의 한 순간을 가리키는 날짜/시간 데이터에는 세 가지 유형이 있습니다.

- _날짜(date)_. 티블은 이를 `<date>`로 인쇄합니다.
- 하루 중의 _시간(time)_. 티블은 이를 `<time>`으로 인쇄합니다.
- *날짜-시간(date-time)*은 날짜에 시간을 더한 것입니다. 이는 시간상의 특정 순간(일반적으로 가까운 초 단위까지)을 고유하게 식별합니다. 티블은 이를 `<dttm>`으로 인쇄합니다. 기본 R에서는 이를 POSIXct라고 부르지만, 발음하기가 쉽지 않습니다.

R에는 시간을 저장하기 위한 기본 클래스(native class)가 없기 때문에, 이 장에서는 날짜와 날짜-시간에 집중할 것입니다. 만약 시간이 필요하다면 hms 패키지를 사용할 수 있습니다.

여러분은 항상 목적에 맞는 가능한 단순한 데이터 유형을 사용해야 합니다. 즉, 날짜-시간 대신 날짜를 사용할 수 있다면 그렇게 해야 합니다. 날짜-시간은 시간대를 처리해야 할 필요성 때문에 상당히 더 복잡해지며, 이에 대해서는 이 장의 끝부분에서 다시 다루겠습니다.

현재 날짜나 날짜-시간을 얻으려면 <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>today()</code></a>나 <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>now()</code></a>를 사용할 수 있습니다.

```
today()
#> [1] "2023-03-12"
now()
#> [1] "2023-03-12 13:07:31 CDT"
```

그렇지 않은 경우, 다음 섹션들에서는 날짜/시간을 생성할 가능성이 높은 네 가지 방법을 설명합니다.

- readr로 파일을 읽는 동안
- 문자열에서
- 개별 날짜-시간 구성 요소에서
- 기존 날짜/시간 객체에서

## 가져오는 동안 (During Import)

CSV에 ISO8601 형식의 날짜나 날짜-시간이 포함되어 있다면 아무것도 할 필요가 없습니다; readr이 자동으로 그것을 인식할 것입니다.

```
csv <- "
  date,datetime
  2022-01-02,2022-01-02 05:12
"
read_csv(csv)
#> # A tibble: 1 × 2
#>   date       datetime
#>   <date>     <dttm>
#> 1 2022-01-02 2022-01-02 05:12:00
```

이전에 *ISO8601*에 대해 들어본 적이 없다면, 이것은 날짜의 구성 요소를 큰 것에서 작은 것 순으로 `-`로 구분하여 작성하는 [국제 표준](https://oreil.ly/19K7t)입니다. 예를 들어 ISO8601에서 2022년 5월 3일은 `2022-05-03`입니다. ISO8601 날짜에는 시, 분, 초가 `:`로 구분되고 날짜와 시간 구성 요소가 `T` 또는 공백으로 구분되는 시간도 포함될 수 있습니다. 예를 들어 2022년 5월 3일 오후 4시 26분은 `2022-05-03 16:26` 또는 `2022-05-03T16:26`으로 작성할 수 있습니다.

다른 날짜-시간 형식의 경우, `col_types`와 더불어 <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>col_date()</code></a> 또는 <a href="https://readr.tidyverse.org/reference/parse_datetime.html" class="orm:hideurl"><code>col_datetime()</code></a>을 날짜-시간 형식과 함께 사용해야 합니다. readr에서 사용하는 날짜-시간 형식은 여러 프로그래밍 언어 전반에서 사용되는 표준으로, `%` 뒤에 단일 문자가 오는 형태로 날짜 구성 요소를 설명합니다. 예를 들어, `%Y-%m-%d`는 연도, `-`, (숫자로 된) 월, `-`, 일로 이루어진 날짜를 지정합니다. <a href="#tbl-date-formats" data-type="xref">표 17-1</a>은 모든 옵션을 나열합니다.

| Type  | Code  | Meaning                        | Example         |
| ----- | ----- | ------------------------------ | --------------- |
| Year  | `%Y`  | 4-digit year                   | 2021            |
|       | `%y`  | 2-digit year                   | 21              |
| Month | `%m`  | Number                         | 2               |
|       | `%b`  | Abbreviated name               | Feb             |
|       | `%B`  | Full name                      | February        |
| Day   | `%d`  | Two digits                     | 02              |
|       | `%e`  | One or two digits              | 2               |
| Time  | `%H`  | 24-hour hour                   | 13              |
|       | `%I`  | 12-hour hour                   | 1               |
|       | `%p`  | a.m./p.m.                      | pm              |
|       | `%M`  | Minutes                        | 35              |
|       | `%S`  | Seconds                        | 45              |
|       | `%OS` | Seconds with decimal component | 45.35           |
|       | `%Z`  | Time zone name                 | America/Chicago |
|       | `%z`  | Offset from UTC                | +0800           |
| Other | `%.`  | Skip one nondigit              | :               |
|       | `%*`  | Skip any number of nondigits   |                 |

표 17-1. readr이 이해하는 모든 날짜 형식 {#tbl-date-formats .table}

이 코드는 매우 모호한 날짜에 적용된 몇 가지 옵션을 보여줍니다.

```
csv <- "
  date
  01/02/15
"

read_csv(csv, col_types = cols(date = col_date("%m/%d/%y")))
#> # A tibble: 1 × 1
#>   date
#>   <date>
#> 1 2015-01-02

read_csv(csv, col_types = cols(date = col_date("%d/%m/%y")))
#> # A tibble: 1 × 1
#>   date
#>   <date>
#> 1 2015-02-01

read_csv(csv, col_types = cols(date = col_date("%y/%m/%d")))
#> # A tibble: 1 × 1
#>   date
#>   <date>
#> 1 2001-02-15
```

날짜 형식을 어떻게 지정하든 일단 R로 가져오면 항상 같은 방식으로 표시된다는 점에 유의하세요.

`%b`나 `%B`를 사용하고 영어가 아닌 날짜로 작업하는 경우, <a href="https://readr.tidyverse.org/reference/locale.html" class="orm:hideurl"><code>locale()</code></a>도 제공해야 합니다. <a href="https://readr.tidyverse.org/reference/date_names.html" class="orm:hideurl"><code>date_names_langs()</code></a>에서 내장된 언어 목록을 참조하거나, <a href="https://readr.tidyverse.org/reference/date_names.html" class="orm:hideurl"><code>date_names()</code></a>로 자신만의 것을 생성하세요.

## 문자열에서

날짜-시간 지정 언어는 강력하지만 날짜 형식에 대한 신중한 분석이 필요합니다. 대안적인 접근 방식은 구성 요소의 순서를 지정하기만 하면 형식을 자동으로 결정하려고 시도하는 lubridate의 도우미들을 사용하는 것입니다. 이를 사용하려면 날짜에 연, 월, 일이 나타나는 순서를 식별한 다음 "y", "m", "d"를 동일한 순서로 배열하세요. 그것이 날짜를 파싱할 lubridate 함수의 이름이 됩니다. 예를 들어:

```
ymd("2017-01-31")
#> [1] "2017-01-31"
mdy("January 31st, 2017")
#> [1] "2017-01-31"
dmy("31-Jan-2017")
#> [1] "2017-01-31"
```

<a href="https://lubridate.tidyverse.org/reference/ymd.html" class="orm:hideurl"><code>ymd()</code></a> 및 유사 함수들은 날짜를 생성합니다. 날짜-시간을 생성하려면 파싱 함수의 이름에 밑줄과 "h", "m", "s" 중 하나 이상을 추가하세요.

```
ymd_hms("2017-01-31 20:11:59")
#> [1] "2017-01-31 20:11:59 UTC"
mdy_hm("01/31/2017 08:01")
#> [1] "2017-01-31 08:01:00 UTC"
```

시간대를 제공하여 날짜로부터 날짜-시간 생성을 강제할 수도 있습니다.

```
ymd("2017-01-31", tz = "UTC")
#> [1] "2017-01-31 UTC"
```

여기서 저는 경도 0°의 시간인 GMT 또는 그리니치 표준시로도 알려져 있을 UTC<sup><a href="ch17.html#idm44771288853952" id="idm44771288853952-marker" data-type="noteref">2</a></sup> 시간대를 사용합니다.<sup><a href="ch17.html#idm44771288853216" id="idm44771288853216-marker" data-type="noteref">3</a></sup> 일광 절약 시간제를 사용하지 않아서 계산하기가 조금 더 쉽습니다.

## 개별 구성 요소에서

단일 문자열 대신, 때로는 여러 열에 걸쳐 날짜-시간의 개별 구성 요소가 분산되어 있을 수 있습니다. `flights` 데이터에 있는 것이 바로 이런 형태입니다.

```
flights |>
  select(year, month, day, hour, minute)
#> # A tibble: 336,776 × 5
#>    year month   day  hour minute
#>   <int> <int> <int> <dbl>  <dbl>
#> 1  2013     1     1     5     15
#> 2  2013     1     1     5     29
#> 3  2013     1     1     5     40
#> 4  2013     1     1     5     45
#> 5  2013     1     1     6      0
#> 6  2013     1     1     5     58
#> # … with 336,770 more rows
```

이런 종류의 입력으로부터 날짜/시간을 생성하려면, 날짜에 대해서는 <a href="https://lubridate.tidyverse.org/reference/make_datetime.html" class="orm:hideurl"><code>make_date()</code></a>를 사용하거나 날짜-시간에 대해서는 <a href="https://lubridate.tidyverse.org/reference/make_datetime.html" class="orm:hideurl"><code>make_datetime()</code></a>을 사용하세요.

```
flights |>
  select(year, month, day, hour, minute) |>
  mutate(departure = make_datetime(year, month, day, hour, minute))
#> # A tibble: 336,776 × 6
#>    year month   day  hour minute departure
#>   <int> <int> <int> <dbl>  <dbl> <dttm>
#> 1  2013     1     1     5     15 2013-01-01 05:15:00
#> 2  2013     1     1     5     29 2013-01-01 05:29:00
#> 3  2013     1     1     5     40 2013-01-01 05:40:00
#> 4  2013     1     1     5     45 2013-01-01 05:45:00
#> 5  2013     1     1     6      0 2013-01-01 06:00:00
#> 6  2013     1     1     5     58 2013-01-01 05:58:00
#> # … with 336,770 more rows
```

`flights`의 네 개 시간 열 각각에 대해 동일한 작업을 수행해 봅시다. 시간이 다소 이상한 형식으로 표시되어 있으므로 모듈로 연산(modulus arithmetic)을 사용하여 시간과 분 구성 요소를 추출합니다. 날짜-시간 변수를 만들고 나면, 이 장의 나머지 부분에서 탐색할 변수들에 집중합니다.

```
make_datetime_100 <- function(year, month, day, time) {
  make_datetime(year, month, day, time %/% 100, time %% 100)
}

flights_dt <- flights |>
  filter(!is.na(dep_time), !is.na(arr_time)) |>
  mutate(
    dep_time = make_datetime_100(year, month, day, dep_time),
    arr_time = make_datetime_100(year, month, day, arr_time),
    sched_dep_time = make_datetime_100(year, month, day, sched_dep_time),
    sched_arr_time = make_datetime_100(year, month, day, sched_arr_time)
  ) |>
  select(origin, dest, ends_with("delay"), ends_with("time"))

flights_dt
#> # A tibble: 328,063 × 9
#>   origin dest  dep_delay arr_delay dep_time            sched_dep_time
#>   <chr>  <chr>     <dbl>     <dbl> <dttm>              <dttm>
#> 1 EWR    IAH           2        11 2013-01-01 05:17:00 2013-01-01 05:15:00
#> 2 LGA    IAH           4        20 2013-01-01 05:33:00 2013-01-01 05:29:00
#> 3 JFK    MIA           2        33 2013-01-01 05:42:00 2013-01-01 05:40:00
#> 4 JFK    BQN          -1       -18 2013-01-01 05:44:00 2013-01-01 05:45:00
#> 5 LGA    ATL          -6       -25 2013-01-01 05:54:00 2013-01-01 06:00:00
#> 6 EWR    ORD          -4        12 2013-01-01 05:54:00 2013-01-01 05:58:00
#> # … with 328,057 more rows, and 3 more variables: arr_time <dttm>,
#> #   sched_arr_time <dttm>, air_time <dbl>
```

이 데이터를 사용하면 1년 동안 출발 시간의 분포를 시각화할 수 있습니다.

```
flights_dt |>
  ggplot(aes(x = dep_time)) +
  geom_freqpoly(binwidth = 86400) # 86400 seconds = 1 day
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in01.png" alt="x축에 출발 시간(2013년 1월~12월)이 있고 y축에 항공편 수(0~1000)가 있는 빈도 다각형. 빈도 다각형은 일별로 구간(bin)이 나뉘어져 있어 일별 항공편 시계열을 볼 수 있습니다. 주간 패턴이 지배적이며 주말에는 항공편이 적습니다. 2월 초, 7월 초, 11월 말, 12월 말에 놀라울 정도로 항공편이 적은 며칠이 두드러집니다." />
</figure>

또는 하루 안에서의 분포:

```
flights_dt |>
  filter(dep_time < ymd(20130102)) |>
  ggplot(aes(x = dep_time)) +
  geom_freqpoly(binwidth = 600) # 600 s = 10 minutes
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in02.png" alt="x축에 출발 시간(1월 1일 오전 6시~자정)이 있고 y축에 항공편 수(0~17)가 있는 10분 단위 구간의 빈도 다각형. 변동성이 커서 패턴을 파악하기 어렵지만, 대부분의 구간에 8~12편의 항공편이 있으며 오전 6시 이전과 오후 8시 이후에는 눈에 띄게 항공편이 적습니다." />
</figure>

숫자 컨텍스트(히스토그램 등)에서 날짜-시간을 사용할 때 1은 1초를 의미하므로, binwidth가 86400이면 하루를 의미한다는 점에 유의하세요. 날짜의 경우 1은 하루를 의미합니다.

## 다른 타입에서

날짜-시간과 날짜 사이를 전환하고 싶을 수 있습니다. 그것이 <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_datetime()</code></a>과 <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_date()</code></a>의 역할입니다.

```
as_datetime(today())
#> [1] "2023-03-12 UTC"
as_date(now())
#> [1] "2023-03-12"
```

때로는 "유닉스 에포크(Unix epoch)"인 1970-01-01로부터의 숫자 오프셋으로 날짜/시간을 얻을 수 있습니다. 오프셋이 초 단위이면 <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_datetime()</code></a>을 사용하고, 일 단위이면 <a href="https://lubridate.tidyverse.org/reference/as_date.html" class="orm:hideurl"><code>as_date()</code></a>를 사용하세요.

```
as_datetime(60 * 60 * 10)
#> [1] "1970-01-01 10:00:00 UTC"
as_date(365 * 10 + 2)
#> [1] "1980-01-01"
```

## 연습문제

1. 유효하지 않은 날짜가 포함된 문자열을 파싱하면 어떻게 되나요?

   ```
   ymd(c("2010-10-10", "bananas"))
   ```

2. <a href="https://lubridate.tidyverse.org/reference/now.html" class="orm:hideurl"><code>today()</code></a>의 `tzone` 인수는 어떤 역할을 하나요? 왜 중요할까요?

3. 다음 각 날짜-시간에 대해, readr 열 지정(column specification)과 lubridate 함수를 사용하여 어떻게 파싱할지 보여주세요.

   ```
   d1 <- "January 1, 2010"
   d2 <- "2015-Mar-07"
   d3 <- "06-Jun-2017"
   d4 <- c("August 19 (2015)", "July 1 (2015)")
   d5 <- "12/30/14" # Dec 30, 2014
   t1 <- "1705"
   t2 <- "11:15:10.12 PM"
   ```

# 날짜-시간 구성 요소

이제 날짜-시간 데이터를 R의 날짜-시간 데이터 구조로 가져오는 방법을 알았으니, 이를 가지고 무엇을 할 수 있는지 탐색해 봅시다. 이 섹션에서는 개별 구성 요소를 가져오고 설정할 수 있게 해주는 접근자(accessor) 함수에 초점을 맞출 것입니다. 다음 섹션에서는 날짜-시간과 함께 산술 연산이 어떻게 작동하는지 살펴볼 것입니다.

## 구성 요소 가져오기

접근자 함수 <a href="https://lubridate.tidyverse.org/reference/year.html" class="orm:hideurl"><code>year()</code></a>, <a href="https://lubridate.tidyverse.org/reference/month.html" class="orm:hideurl"><code>month()</code></a>, <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>mday()</code></a> (월의 일), <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>yday()</code></a> (연의 일), <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>wday()</code></a> (요일), <a href="https://lubridate.tidyverse.org/reference/hour.html" class="orm:hideurl"><code>hour()</code></a>, <a href="https://lubridate.tidyverse.org/reference/minute.html" class="orm:hideurl"><code>minute()</code></a> 및 <a href="https://lubridate.tidyverse.org/reference/second.html" class="orm:hideurl"><code>second()</code></a>를 사용하여 날짜의 개별 부분을 추출할 수 있습니다. 이들은 사실상 <a href="https://lubridate.tidyverse.org/reference/make_datetime.html" class="orm:hideurl"><code>make_datetime()</code></a>의 반대 역할을 합니다.

```
datetime <- ymd_hms("2026-07-08 12:34:56")

year(datetime)
#> [1] 2026
month(datetime)
#> [1] 7
mday(datetime)
#> [1] 8

yday(datetime)
#> [1] 189
wday(datetime)
#> [1] 4
```

<a href="https://lubridate.tidyverse.org/reference/month.html" class="orm:hideurl"><code>month()</code></a> 및 <a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>wday()</code></a>의 경우 `label = TRUE`로 설정하여 월이나 요일의 약식 이름을 반환할 수 있습니다. 전체 이름을 반환하려면 `abbr = FALSE`로 설정하세요.

```
month(datetime, label = TRUE)
#> [1] Jul
#> 12 Levels: Jan < Feb < Mar < Apr < May < Jun < Jul < Aug < Sep < ... < Dec
wday(datetime, label = TRUE, abbr = FALSE)
#> [1] Wednesday
#> 7 Levels: Sunday < Monday < Tuesday < Wednesday < Thursday < ... < Saturday
```

<a href="https://lubridate.tidyverse.org/reference/day.html" class="orm:hideurl"><code>wday()</code></a>를 사용하면 주말보다 주중에 더 많은 항공편이 출발한다는 것을 확인할 수 있습니다.

```
flights_dt |>
  mutate(wday = wday(dep_time, label = TRUE)) |>
  ggplot(aes(x = wday)) +
  geom_bar()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in03.png" alt="x축에 요일이 있고 y축에 항공편 수가 있는 막대 차트. 월요일부터 금요일까지는 약 48,000편으로 항공편 수가 비슷하며 주 후반으로 갈수록 약간 감소합니다. 일요일은 조금 더 낮고(~45,000), 토요일은 훨씬 더 낮습니다(~38,000)." />
</figure>

또한 시간 내의 분(minute) 단위로 평균 출발 지연 시간을 살펴볼 수 있습니다. 흥미로운 패턴이 하나 있습니다. 20~30분 및 50~60분에 출발하는 항공편은 시간 내의 다른 때보다 지연이 훨씬 적습니다!

```
flights_dt |>
  mutate(minute = minute(dep_time)) |>
  group_by(minute) |>
  summarize(
    avg_delay = mean(dep_delay, na.rm = TRUE),
    n = n()
  ) |>
  ggplot(aes(x = minute, y = avg_delay)) +
  geom_line()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for 단원 분석/assets/rds2_17in04.png" alt="x축에 실제 출발 분(0-60)이 있고 y축에 평균 지연(4-20)이 있는 선형 차트. 평균 지연은 (0, 12)에서 시작하여 (18, 20)까지 꾸준히 증가하다가 급격히 떨어져 매 시 23분 경에 최소인 9분 지연에 도달합니다. 그 후 다시 증가하여 (17, 35)에 이르고, 급격히 감소하여 (55, 4)에 도달합니다. 마지막으로 (60, 9)로 증가하며 끝납니다." />
</figure>

흥미롭게도, _예정된(scheduled)_ 출발 시간을 보면 그런 강력한 패턴이 나타나지 않습니다.

```
sched_dep <- flights_dt |>
  mutate(minute = minute(sched_dep_time)) |>
  group_by(minute) |>
  summarize(
    avg_delay = mean(arr_delay, na.rm = TRUE),
    n = n()
  )

ggplot(sched_dep, aes(x = minute, y = avg_delay)) +
  geom_line()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in05.png" alt="x축에 예정된 출발 분(0-60)이 있고 평균 지연(4-16)이 있는 선형 차트. 패턴이 비교적 적으며, 시간이 지남에 따라 평균 지연이 대략 10분에서 8분으로 감소한다는 약간의 암시만 있습니다." />
</figure>

그렇다면 왜 실제 출발 시간에서 그런 패턴이 나타나는 걸까요? 글쎄요, 인간이 수집한 많은 데이터와 마찬가지로 <a href="#fig-human-rounding" data-type="xref">그림 17-1</a>이 보여주듯이, "적당한(nice)" 출발 시간에 출발하는 항공편 쪽으로 강한 편향이 있습니다. 인간의 판단이 개입된 데이터로 작업할 때는 항상 이런 종류의 패턴을 경계하세요!

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_1701.png" alt="x축에 출발 분(0-60)이 있고 y축에 항공편 수(0-60000)가 있는 선형 플롯. 대부분의 항공편은 정각(~60,000)이나 30분(~35,000)에 출발하도록 예정되어 있습니다. 그 외 거의 모든 항공편은 5의 배수에 출발하도록 예정되어 있으며, 15, 45, 55분에 약간 더 많습니다." />
<h6 id="figure-17-1.-a-frequency-polygon-showing-the-number-of-flights-scheduled-to-depart-each-hour.-you-can-see-a-strong-preference-for-round-numbers-like-0-and-30-and-generally-for-numbers-that-are-a-multiple-of-five.">그림 17-1. 매 시 출발 예정인 항공편 수를 보여주는 빈도 다각형. 0과 30 같은 딱 떨어지는 숫자와 일반적으로 5의 배수인 숫자에 대한 강한 선호도를 볼 수 있습니다.</h6>
</figure>

## 반올림(Rounding)

개별 구성 요소를 플로팅하는 대안적인 접근 방식은 <a href="https://lubridate.tidyverse.org/reference/round_date.html" class="orm:hideurl"><code>floor_date()</code></a>, <a href="https://lubridate.tidyverse.org/reference/round_date.html" class="orm:hideurl"><code>round_date()</code></a> 및 <a href="https://lubridate.tidyverse.org/reference/round_date.html" class="orm:hideurl"><code>ceiling_date()</code></a>를 사용하여 날짜를 가까운 시간 단위로 반올림하는 것입니다. 각 함수는 조정할 날짜 벡터와 내림(floor), 올림(ceiling), 또는 반올림할 단위의 이름을 취합니다. 예를 들어 이를 통해 주당 항공편 수를 플로팅할 수 있습니다.

```
flights_dt |>
  count(week = floor_date(dep_time, "week")) |>
  ggplot(aes(x = week, y = n)) +
  geom_line() +
  geom_point()
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in06.png" alt="x축에 주(2013년 1월-12월)가 있고 y축에 항공편 수(2,000-7,000)가 있는 선형 플롯. 패턴은 2월부터 11월까지 주당 약 7,000편의 항공편으로 꽤 평탄합니다. 연초(약 4,500편)와 연말(약 2,500편)에는 항공편이 훨씬 적습니다." />
</figure>

`dep_time`과 해당 일의 이른 순간 사이의 차이를 계산하여 하루 동안의 항공편 분포를 보여주기 위해 반올림을 사용할 수 있습니다.

```
flights_dt |>
  mutate(dep_hour = dep_time - floor_date(dep_time, "day")) |>
  ggplot(aes(x = dep_hour)) +
  geom_freqpoly(binwidth = 60 * 30)
#> Don't know how to automatically pick scale for object of type <difftime>.
#> Defaulting to continuous.
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in07.png" alt="x축에 출발 시간이 있는 선형 플롯. 자정 이후 초 단위이므로 해석하기 어렵습니다." />
</figure>

한 쌍의 날짜-시간 간의 차이를 계산하면 difftime이 산출됩니다(자세한 내용은 <a href="#sec-intervals" data-type="xref">"구간(Intervals)"</a>에서 다룹니다). 이를 `hms` 객체로 변환하여 더 유용한 x축을 얻을 수 있습니다.

```
flights_dt |>
  mutate(dep_hour = hms::as_hms(dep_time - floor_date(dep_time, "day"))) |>
  ggplot(aes(x = dep_hour)) +
  geom_freqpoly(binwidth = 60 * 30)
```

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_17in08.png" alt="x축에 출발 시간(자정부터 자정)이 있고 y축에 항공편 수(0 ~ 15,000)가 있는 선형 플롯. 오전 5시 이전에는 항공편이 매우 적습니다(<100). 그 후 시간당 12,000편으로 급격히 증가하여 오전 9시에 15,000편으로 최고치에 달한 뒤 오전 10시에서 오후 2시 사이에는 시간당 약 8,000편으로 떨어집니다. 그 후 오후 8시까지 시간당 약 12,000편으로 증가하다가 다시 급격히 감소합니다." />
</figure>

## 구성 요소 수정하기

각 접근자 함수를 사용하여 날짜/시간의 구성 요소를 수정할 수도 있습니다. 데이터 분석에서 많이 발생하지는 않지만 명백히 잘못된 날짜가 있는 데이터를 정제(cleaning)할 때 유용할 수 있습니다.

```p
(datetime <- ymd_hms("2026-07-08 12:34:56"))
#> [1] "2026-07-08 12:34:56 UTC"

year(datetime) <- 2030
datetime
#> [1] "2030-07-08 12:34:56 UTC"
month(datetime) <- 01
datetime
#> [1] "2030-01-08 12:34:56 UTC"
hour(datetime) <- hour(datetime) + 1
datetime
#> [1] "2030-01-08 13:34:56 UTC"
```

대안으로, 기존 변수를 수정하는 대신 <a href="https://rdrr.io/r/stats/update.html" class="orm:hideurl"><code>update()</code></a>를 사용하여 새로운 날짜-시간을 생성할 수 있습니다. 이를 통해 한 번에 여러 값을 설정할 수도 있습니다.

```
update(datetime, year = 2030, month = 2, mday = 2, hour = 2)
#> [1] "2030-02-02 02:34:56 UTC"
```

값이 너무 크면 다음 단위로 넘어갑니다(roll over):

```
update(ymd("2023-02-01"), mday = 30)
#> [1] "2023-03-02"
update(ymd("2023-02-01"), hour = 400)
#> [1] "2023-02-17 16:00:00 UTC"
```

## 연습문제

1. 한 해 동안 하루 내의 비행 시간 분포가 어떻게 변하나요?

2. `dep_time`, `sched_dep_time` 및 `dep_delay`를 비교하세요. 이들은 일관성이 있나요? 결과를 설명하세요.

3. `air_time`과 출발 및 도착 사이의 지속 시간을 비교하세요. 결과를 설명하세요. (힌트: 공항의 위치를 고려하세요.)

4. 하루 동안 평균 지연 시간은 어떻게 변하나요? `dep_time`을 사용해야 할까요, 아니면 `sched_dep_time`을 사용해야 할까요? 왜 그런가요?

5. 지연될 가능성을 최소화하고 싶다면 일주일 중 무슨 요일에 출발해야 할까요?

6. `diamonds$carat`과 `flights$sched_dep_time`의 분포를 비슷하게 만드는 것은 무엇인가요?

7. 20-30분과 50-60분에 출발하는 항공편의 이른 출발이 일찍 출발하는 예정된 항공편들 때문이라는 우리의 가설을 확인하세요. 힌트: 항공편이 지연되었는지 여부를 알려주는 이진 변수를 만드세요.

# 시간 범위(Time Spans)

다음으로는 뺄셈, 덧셈 및 나눗셈을 포함하여 날짜 산술 연산이 어떻게 작동하는지 배울 것입니다. 그 과정에서 시간 범위를 나타내는 세 가지 중요한 클래스에 대해 배우게 됩니다.

Durations(지속 시간)  
정확한 초 단위의 수를 나타냅니다.

Periods(기간)  
주(weeks)나 월(months)과 같은 사람의 단위를 나타냅니다.

Intervals(구간)  
시작점과 끝점을 나타냅니다.

지속 시간(duration), 기간(periods) 및 구간(intervals) 중 어떻게 선택할까요? 항상 그렇듯이, 문제를 해결하는 단순한 데이터 구조를 선택하세요. 물리적인 시간만 신경 쓴다면 지속 시간을 사용하고, 사람의 시간을 더해야 한다면 기간을 사용하며, 어떤 범위가 사람 단위로 얼마나 긴지 알아내야 한다면 구간을 사용하세요.

## Durations(지속 시간)

R에서 두 날짜를 빼면 `difftime` 객체를 얻습니다.

```
# Hadley는 몇 살인가요?
h_age <- today() - ymd("1979-10-14")
h_age
#> Time difference of 15855 days
```

`difftime` 클래스 객체는 초, 분, 시, 일 또는 주 단위의 시간 범위를 기록합니다. 이러한 모호성은 difftime 작업을 약간 고통스럽게 만들 수 있으므로, lubridate는 항상 초를 사용하는 대안인 *지속 시간(duration)*을 제공합니다.

```
as.duration(h_age)
#> [1] "1369872000s (~43.41 years)"
```

지속 시간은 다음과 같은 여러 편리한 생성자와 함께 제공됩니다.

```
dseconds(15)
#> [1] "15s"
dminutes(10)
#> [1] "600s (~10 minutes)"
dhours(c(12, 24))
#> [1] "43200s (~12 hours)" "86400s (~1 days)"
ddays(0:5)
#> [1] "0s"                "86400s (~1 days)"  "172800s (~2 days)"
#> [4] "259200s (~3 days)" "345600s (~4 days)" "432000s (~5 days)"
dweeks(3)
#> [1] "1814400s (~3 weeks)"
dyears(1)
#> [1] "31557600s (~1 years)"
```

지속 시간은 항상 초 단위로 시간 범위를 기록합니다. 더 큰 단위는 분, 시, 일, 주 및 년을 초로 변환하여 생성됩니다. 1분은 60초, 1시간은 60분, 하루는 24시간, 1주는 7일입니다. 더 큰 시간 단위는 좀 더 문제가 됩니다. 1년은 1년의 "평균" 일수인 365.25일을 사용합니다. 한 달을 지속 시간으로 변환하는 방법은 없는데, 변동이 너무 많기 때문입니다.

지속 시간을 더하거나 곱할 수 있습니다.

```m
2 * dyears(1)
#> [1] "63115200s (~2 years)"
dyears(1) + dweeks(12) + dhours(15)
#> [1] "38869200s (~1.23 years)"
```

날짜에 지속 시간을 더하거나 뺄 수 있습니다.

```
tomorrow <- today() + ddays(1)
last_year <- today() - dyears(1)
```

하지만 지속 시간은 정확한 초 수를 나타내므로 때로는 예상치 못한 결과를 얻을 수 있습니다.

```
one_am <- ymd_hms("2026-03-08 01:00:00", tz = "America/New_York")

one_am
#> [1] "2026-03-08 01:00:00 EST"
one_am + ddays(1)
#> [1] "2026-03-09 02:00:00 EDT"
```

3월 8일 오전 1시에서 하루가 지났는데 왜 3월 9일 오전 2시로 반환될까요? 날짜를 주의 깊게 보면 시간대가 변경된 것도 알 수 있습니다. 3월 8일은 DST(일광 절약 시간제)가 시작되는 날이므로 23시간밖에 되지 않습니다. 따라서 하루치에 해당하는 초를 더하면 다른 시간이 나오게 됩니다.

## Periods(기간)

이 문제를 해결하기 위해 lubridate는 *기간(periods)*을 제공합니다. 기간은 시간 범위이지만 초 단위의 고정된 길이를 갖지 않습니다. 대신 일(days)과 월(months) 같은 "사람"의 시간으로 작동합니다. 이를 통해 더 직관적인 방식으로 작동할 수 있습니다.

```
one_am
#> [1] "2026-03-08 01:00:00 EST"
one_am + days(1)
#> [1] "2026-03-09 01:00:00 EDT"
```

지속 시간과 마찬가지로, 기간도 다음과 같은 여러 친숙한 생성자 함수로 생성할 수 있습니다.

```
hours(c(12, 24))
#> [1] "12H 0M 0S" "24H 0M 0S"
days(7)
#> [1] "7d 0H 0M 0S"
months(1:6)
#> [1] "1m 0d 0H 0M 0S" "2m 0d 0H 0M 0S" "3m 0d 0H 0M 0S" "4m 0d 0H 0M 0S"
#> [5] "5m 0d 0H 0M 0S" "6m 0d 0H 0M 0S"
```

기간을 더하거나 곱할 수 있습니다.

```m
10 * (months(6) + days(1))
#> [1] "60m 10d 0H 0M 0S"
days(50) + hours(25) + minutes(2)
#> [1] "50d 25H 2M 0S"
```

그리고 물론, 이를 날짜에 더할 수 있습니다. 지속 시간과 비교할 때, 기간은 여러분이 예상하는 대로 작동할 가능성이 더 높습니다.

```
# 윤년
ymd("2024-01-01") + dyears(1)
#> [1] "2024-12-31 06:00:00 UTC"
ymd("2024-01-01") + years(1)
#> [1] "2025-01-01"

# 일광 절약 시간제
one_am + ddays(1)
#> [1] "2026-03-09 02:00:00 EDT"
one_am + days(1)
#> [1] "2026-03-09 01:00:00 EDT"
```

기간을 사용하여 항공편 날짜와 관련된 이상한 점을 고쳐봅시다. 일부 비행기는 뉴욕시에서 출발하기 *전*에 목적지에 도착한 것처럼 보입니다.

```
flights_dt |>
  filter(arr_time < dep_time)
#> # A tibble: 10,633 × 9
#>   origin dest  dep_delay arr_delay dep_time            sched_dep_time
#>   <chr>  <chr>     <dbl>     <dbl> <dttm>              <dttm>
#> 1 EWR    BQN           9        -4 2013-01-01 19:29:00 2013-01-01 19:20:00
#> 2 JFK    DFW          59        NA 2013-01-01 19:39:00 2013-01-01 18:40:00
#> 3 EWR    TPA          -2         9 2013-01-01 20:58:00 2013-01-01 21:00:00
#> 4 EWR    SJU          -6       -12 2013-01-01 21:02:00 2013-01-01 21:08:00
#> 5 EWR    SFO          11       -14 2013-01-01 21:08:00 2013-01-01 20:57:00
#> 6 LGA    FLL         -10        -2 2013-01-01 21:20:00 2013-01-01 21:30:00
#> # … with 10,627 more rows, and 3 more variables: arr_time <dttm>,
#> #   sched_arr_time <dttm>, air_time <dbl>
```

이들은 심야(overnight) 항공편입니다. 우리는 출발 시간과 도착 시간에 동일한 날짜 정보를 사용했지만, 이 항공편들은 다음 날 도착했습니다. 각 심야 항공편의 도착 시간에 `days(1)`을 더하여 이 문제를 해결할 수 있습니다.

```
flights_dt <- flights_dt |>
  mutate(
    overnight = arr_time < dep_time,
    arr_time = arr_time + days(overnight),
    sched_arr_time = sched_arr_time + days(overnight)
  )
```

이제 우리의 모든 항공편이 물리 법칙을 따르게 되었습니다.

```
flights_dt |>
  filter(arr_time < dep_time)
#> # A tibble: 0 × 10
#> # … with 10 variables: origin <chr>, dest <chr>, dep_delay <dbl>,
#> #   arr_delay <dbl>, dep_time <dttm>, sched_dep_time <dttm>, …
#> # ℹ Use `colnames()` to see all variable names
#> # … with 10,627 more rows, and 4 more variables:
```

## Intervals(구간)

`dyears(1) / ddays(365)`는 무엇을 반환할까요? 정확히 1은 아닙니다. 왜냐하면 `dyears()`는 평균 1년당 초 수로 정의되어 있으며, 이는 365.25일이기 때문입니다.

`years(1) / days(1)`는 무엇을 반환할까요? 글쎄요, 만약 연도가 2015년이면 365를 반환해야 하지만 2016년이면 366을 반환해야 합니다! lubridate가 단일한 명확한 답을 주기에는 정보가 충분하지 않습니다. 대신 추정치를 제공합니다.

```
years(1) / days(1)
#> [1] 365.25
```

더 정확한 측정을 원한다면 *구간(interval)*을 사용해야 합니다. 구간은 시작과 끝 날짜-시간의 쌍이거나, 시작점이 있는 지속 시간으로 생각할 수 있습니다.

`start %--% end`라고 작성하여 구간을 생성할 수 있습니다.

```
y2023 <- ymd("2023-01-01") %--% ymd("2024-01-01")
y2024 <- ymd("2024-01-01") %--% ymd("2025-01-01")

y2023
#> [1] 2023-01-01 UTC--2024-01-01 UTC
y2024
#> [1] 2024-01-01 UTC--2025-01-01 UTC
```

그런 다음 이것을 <a href="https://lubridate.tidyverse.org/reference/period.html" class="orm:hideurl"><code>days()</code></a>로 나누어 해당 연도에 며칠이 들어맞는지 알아낼 수 있습니다.

```
y2023 / days(1)
#> [1] 365
y2024 / days(1)
#> [1] 366
```

## 연습문제

1. R을 막 배우기 시작한 사람에게 `days(!overnight)`와 `days(overnight)`를 설명해 주세요. 알아야 할 핵심적인 사실은 무엇인가요?

2. 2015년의 매월 첫째 날을 나타내는 날짜 벡터를 생성하세요. *올해*의 매월 첫째 날을 나타내는 날짜 벡터를 생성하세요.

3. 생일(날짜로)이 주어졌을 때 여러분의 만 나이를 반환하는 함수를 작성하세요.

4. `(today() %--% (today() + years(1))) / months(1)`가 작동할 수 없는 이유는 무엇인가요?

# 시간대(Time Zones)

시간대는 지정학적 실체와의 상호 작용 때문에 엄청나게 복잡한 주제입니다. 다행히 데이터 분석에서 이 모든 세부 사항이 중요한 것은 아니므로 자세히 파고들 필요는 없지만, 정면으로 해결해야 할 몇 가지 과제가 있습니다.

첫 번째 문제는 일상적인 시간대 이름이 모호한 경향이 있다는 것입니다. 예를 들어, 미국인이라면 동부 표준시(Eastern Standard Time, EST)에 익숙할 것입니다. 하지만 호주와 캐나다에도 EST가 있습니다! 혼란을 피하기 위해 R은 국제 표준인 IANA 시간대를 사용합니다. 이는 일반적으로 `{continent}/{city}` 또는 `{ocean}/{city}` 형태의 일관된 명명 체계인 `{area}/{location}`을 사용합니다. 예로는 "America/New_York", "Europe/Paris", "Pacific/Auckland" 등이 있습니다.

보통 시간대가 국가 또는 국가 내의 한 지역과 관련이 있다고 생각할 텐데, 왜 시간대에 도시를 사용하느지 궁금할 수 있습니다. 이는 IANA 데이터베이스가 수십 년 분량의 시간대 규칙을 기록해야 하기 때문입니다. 수십 년 동안 국가들은 꽤 빈번하게 이름을 바꾸거나 분열하지만, 도시 이름은 그대로 유지되는 경향이 있습니다. 또 다른 문제는 이름이 현재의 동작뿐만 아니라 전체 역사도 반영해야 한다는 것입니다. 예를 들어 "America/New_York"과 "America/Detroit"에 대한 시간대가 모두 있습니다. 이 도시들은 현재 모두 동부 표준시를 사용하지만, 1969~1972년 미시간주(디트로이트가 위치한 주)는 DST를 따르지 않았기 때문에 다른 이름이 필요합니다. 이런 이야기들 중 몇 가지를 읽어보기 위해서라도 [원시 시간대 데이터베이스(raw time zone database)](https://oreil.ly/NwvsT)를 읽어볼 만한 가치가 있습니다!

<a href="https://rdrr.io/r/base/timezones.html" class="orm:hideurl"><code>Sys.timezone()</code></a>을 사용하여 R이 생각하는 여러분의 현재 시간대가 무엇인지 알 수 있습니다.

```
Sys.timezone()
#> [1] "America/Chicago"
```

(R이 모른다면 `NA`를 얻게 될 것입니다.)

그리고 <a href="https://rdrr.io/r/base/timezones.html" class="orm:hideurl"><code>OlsonNames()</code></a>를 통해 모든 시간대 이름의 전체 목록을 확인할 수 있습니다.

```
length(OlsonNames())
#> [1] 597
head(OlsonNames())
#> [1] "Africa/Abidjan"     "Africa/Accra"       "Africa/Addis_Ababa"
#> [4] "Africa/Algiers"     "Africa/Asmara"      "Africa/Asmera"
```

R에서 시간대는 출력만 제어하는 날짜-시간의 속성입니다. 예를 들어, 이 세 객체는 시간상의 동일한 순간을 나타냅니다.

```
x1 <- ymd_hms("2024-06-01 12:00:00", tz = "America/New_York")
x1
#> [1] "2024-06-01 12:00:00 EDT"

x2 <- ymd_hms("2024-06-01 18:00:00", tz = "Europe/Copenhagen")
x2
#> [1] "2024-06-01 18:00:00 CEST"

x3 <- ymd_hms("2024-06-02 04:00:00", tz = "Pacific/Auckland")
x3
#> [1] "2024-06-02 04:00:00 NZST"
```

뺄셈을 사용하여 동일한 시간인지 확인할 수 있습니다.

```
x1 - x2
#> Time difference of 0 secs
x1 - x3
#> Time difference of 0 secs
```

다르게 지정하지 않는 한 lubridate는 항상 UTC를 사용합니다. UTC는 과학계에서 사용하는 표준 시간대이며 대략 GMT와 같습니다. DST(일광 절약 시간제)가 없어서 계산에 편리한 표현을 만들어 줍니다. <a href="https://rdrr.io/r/base/c.html" class="orm:hideurl"><code>c()</code></a>와 같이 날짜-시간을 결합하는 연산은 종종 시간대를 삭제합니다. 이 경우 날짜-시간은 첫 번째 요소의 시간대로 표시됩니다.

```
x4 <- c(x1, x2, x3)
```

x4
#> [1] "2024-06-01 12:00:00 EDT" "2024-06-01 12:00:00 EDT"
#> [3] "2024-06-01 12:00:00 EDT"

```

시간대를 두 가지 방법으로 변경할 수 있습니다.

- 시간상의 순간은 동일하게 유지하면서 표시되는 방식만 변경합니다. 순간은 맞지만 더 자연스러운 표시를 원할 때 이것을 사용하세요.

```

x4a <- with_tz(x4, tzone = "Australia/Lord_Howe")
x4a
#> [1] "2024-06-02 02:30:00 +1030" "2024-06-02 02:30:00 +1030"
#> [3] "2024-06-02 02:30:00 +1030"
x4a - x4
#> Time differences in secs
#> [1] 0 0 0

```

(이것은 시간대의 또 다른 과제를 보여줍니다. 시간대가 모두 정수 단위의 시간 오프셋은 아니라는 것입니다!)

- 시간상의 기본 순간 자체를 변경합니다. 잘못된 시간대로 라벨이 지정된 순간이 있어서 이를 고쳐야 할 때 이것을 사용하세요.

```

x4b <- force_tz(x4, tzone = "Australia/Lord_Howe")
x4b
#> [1] "2024-06-01 12:00:00 +1030" "2024-06-01 12:00:00 +1030"
#> [3] "2024-06-01 12:00:00 +1030"
x4b - x4
#> Time differences in hours
#> [1] -14.5 -14.5 -14.5

```

# 요약

이 장에서는 날짜-시간 데이터로 작업하는 데 도움이 되도록 lubridate가 제공하는 도구들을 소개했습니다. 날짜와 시간으로 작업하는 것은 필요 이상으로 어려워 보일 수 있지만, 이 장을 통해 왜 그런지 알게 되셨기를 바랍니다. 날짜-시간은 처음 보이는 것보다 더 복잡하며 발생 가능한 모든 상황을 처리하면 복잡성이 더해집니다. 여러분의 데이터가 절대 DST 경계를 넘지 않거나 윤년을 포함하지 않더라도, 함수들은 이를 처리할 수 있어야 합니다.

다음 장에서는 결측값(missing values)에 대해 요약하여 설명합니다. 여러분은 이미 몇 군데에서 이것들을 보았고, 본인의 분석에서도 의심할 여지 없이 접해 보셨을 것입니다. 이제 이것들을 다루기 위한 유용한 기법들의 모음집(grab bag)을 제공할 때입니다.

<sup>[1](ch17.html#idm44771289225952-marker)</sup> 연도가 4로 나누어 떨어지면 윤년이지만, 100으로 나누어 떨어지면 윤년이 아니며, 단 400으로 나누어 떨어지는 경우는 예외로 윤년입니다. 다시 말해, 400년의 주기마다 97번의 윤년이 있습니다.

<sup>[2](ch17.html#idm44771288853952-marker)</sup> UTC가 무엇의 약자인지 궁금할 수 있습니다. 이것은 영어의 "Coordinated Universal Time"과 프랑스어의 "Temps Universel Coordonné" 사이의 타협안입니다.

<sup>[3](ch17.html#idm44771288853216-marker)</sup> 경도 시스템을 어떤 나라가 고안해냈는지 맞히더라도 상은 없습니다(역주: 영국의 그리니치를 가리킴).
```
