---
title: "날짜와 시간"
---

```{r}
#| echo: false
source("_common.R")

# https://github.com/tidyverse/lubridate/issues/1058
options(warnPartialMatchArgs = FALSE)
```

이 장에서는 R로 날짜와 시간을 다루는 방법을 알아봅니다. 언뜻 보기에(At first glance) 날짜와 시간은 단순합니다. 일상에서 늘 사용해도 크게 혼란스럽지 않기 때문입니다. 하지만 파고들수록 점점 더 복잡해집니다(complicated)!

워밍업(warm up)으로 1년은 며칠이고 하루는 몇 시간인지 생각해 보세요. 대부분의 해는 365일이고 윤년(leap years)은 366일입니다. 어느 해가 윤년인지 정하는 규칙을 모두 알고 계십니까? 하루의 시간 수는 조금 더 모호합니다. 대부분 24시간이지만 일광 절약 시간제(DST)를 사용하는 곳에서는 해마다 하루가 23시간이고 다른 하루는 25시간입니다.

날짜와 시간이 어려운 까닭은 두 가지 물리적 현상, 곧 지구의 자전과 태양 주위의 공전을 월, 시간대(time zones), 일광 절약 시간제(DST) 같은 온갖 지정학적(geopolitical) 현상과 조화(reconcile)시켜야 하기 때문입니다.

이 장에서 날짜와 시간의 모든 세부 사항을 다루지는 않습니다. 대신 일반적인 데이터 분석 과제를 해결하는 실용적인 기술의 탄탄한 기초(solid grounding)를 다집니다.

최신 tidyverse 릴리스부터 lubridate는 핵심 tidyverse에 포함됩니다. 실습 데이터에는 nycflights13도 필요합니다.

```{r}
#| message: false
library(tidyverse)
library(nycflights13)
```

## 날짜/시간 생성하기

시간상의 한순간(instant)을 나타내는 날짜/시간 데이터는 세 유형으로 나뉩니다.

- 날짜(date): 티블(Tibbles)은 이를 `<date>`로 출력합니다.

- 하루 중의 시간(time): 티블은 이를 `<time>`으로 출력합니다.

- 날짜-시간(date-time)은 날짜에 시간을 더한 값입니다. 보통 가까운 초(second)까지 시간상의 한순간을 고유하게 식별합니다. 티블은 `<dttm>`으로 출력합니다. 기본 R에서 부르는 이름은 POSIXct입니다. 입에 착 달라붙는(trip off the tongue) 이름은 아닙니다.

R에는 시간을 저장하는 기본 클래스가 없으므로 이 장에서는 날짜와 날짜-시간에 집중합니다. 필요하다면 hms 패키지를 쓰면 됩니다.

요구 사항에 맞는(works for your needs) 가장 단순한 데이터 유형을 사용하세요. 날짜-시간 대신 날짜로 충분하다면 날짜를 선택합니다.

날짜-시간은 시간대를 처리해야 해서 훨씬(substantially) 복잡합니다. 시간대는 장의 끝에서 다시 설명합니다.

현재 날짜나 날짜-시간은 `today()` 또는 `now()`로 구합니다.

```{r}
today()
now()
```

그 밖의 경우(Otherwise)에는 다음 네 가지 방법으로 날짜/시간을 생성합니다.

- readr을 사용하여 파일을 읽는 동안
- 문자열에서 생성
- 개별 날짜-시간 구성 요소에서 생성
- 기존 날짜/시간 객체에서 생성

### 가져오기 중

CSV에 ISO8601 날짜나 날짜-시간이 들어 있다면 따로 할 일이 없습니다. readr이 자동으로 인식합니다.

```{r}
#| message: false
csv <- "date,datetime\n2022-01-02,2022-01-02 05:12"
read_csv(csv)
```

ISO8601을 처음 접한다면 날짜 구성 요소를 큰 단위부터 작은 단위 순으로 놓고 `-`로 구분하는 국제 표준 날짜 표기법이라고 이해하면 됩니다. 예를 들어 2022년 5월 3일은 `2022-05-03`으로 씁니다. 시간도 함께 표기합니다. 시, 분, 초는 `:`로 나누고 날짜와 시간은 `T` 또는 공백으로 구분합니다.

가령 2022년 5월 3일 오후 4시 26분은 `2022-05-03 16:26` 또는 `2022-05-03T16:26`으로 씁니다.

다른 날짜-시간 형식에는 형식을 지정한 `col_types`와 `col_date()` 또는 `col_datetime()`을 사용합니다. readr의 날짜-시간 형식은 여러 프로그래밍 언어가 공유하는 표준입니다. `%` 뒤의 문자 하나로 날짜 구성 요소를 나타냅니다. 예를 들어 `%Y-%m-%d`는 연도, `-`, 월(숫자), `-`, 일로 구성된 날짜입니다.

아래 표에는 모든 옵션이 나열되어 있습니다.

| 유형(Type) | 코드(Code) | 의미(Meaning)                  | 예시(Example)   |
|------------|------------|--------------------------------|-----------------|
| 연(Year)   | `%Y`       | 4자리 연도                     | 2021            |
|            | `%y`       | 2자리 연도                     | 21              |
| 월(Month)  | `%m`       | 숫자                           | 2               |
|            | `%b`       | 약어 이름                      | Feb             |
|            | `%B`       | 전체 이름                      | February        |
| 일(Day)    | `%d`       | 1자리 또는 2자리 숫자          | 2               |
|            | `%e`       | 2자리 숫자                     | 02              |
| 시간(Time) | `%H`       | 24시간 형식 시간               | 13              |
|            | `%I`       | 12시간 형식 시간               | 1               |
|            | `%p`       | AM/PM                          | pm              |
|            | `%M`       | 분                             | 35              |
|            | `%S`       | 초                             | 45              |
|            | `%OS`      | 소수(decimal) 구성 요소가 있는 초 | 45.35           |
|            | `%Z`       | 시간대 이름                    | America/Chicago |
|            | `%z`       | UTC와의 오프셋(Offset)         | +0800           |
| 기타(Other)| `%.`       | 숫자가 아닌 문자 한 개 건너뛰기| :               |
|            | `%*`       | 숫자가 아닌 임의 개의 문자 건너뛰기 |                 |

다음 코드는 매우 모호한(ambiguous) 날짜에 몇 가지 옵션을 적용합니다.

```{r}
#| messages: false
csv <- "
  date
  01/02/15
"

read_csv(csv, col_types = cols(date = col_date("%m/%d/%y")))

read_csv(csv, col_types = cols(date = col_date("%d/%m/%y")))

read_csv(csv, col_types = cols(date = col_date("%y/%m/%d")))
```

날짜 형식을 어떻게 지정했든 R로 가져온 뒤에는 항상 같은 방식으로 표시(displayed)됩니다.

`%b` 또는 `%B`를 사용해 영어가 아닌 날짜를 다룰 때는 `locale()`도 지정해야 합니다. `date_names_langs()`에서 내장 언어 목록을 확인하거나 `date_names()`로 직접 언어를 만드세요.

### 문자열에서 생성

날짜-시간 지정 언어는 강력하지만 날짜 형식을 꼼꼼히 분석해야 합니다. 더 간단하게는 구성 요소의 순서만 보고 형식을 자동으로 판단(determine)하는 lubridate 도우미 함수를 씁니다. 날짜에서 연, 월, 일이 나타나는 순서를 확인하고 같은 순서로 "y", "m", "d"를 배열(arrange)하세요. 이 배열이 날짜를 구문 분석(parse)할 lubridate 함수의 이름입니다.

```{r}
ymd("2017-01-31")
mdy("January 31st, 2017")
dmy("31-Jan-2017")
```

`ymd()`와 비슷한 함수들(friends)은 날짜를 만듭니다. 날짜-시간을 만들 때는 구문 분석 함수 이름에 밑줄(`_`)과 "h", "m", "s" 중 하나 이상을 붙입니다.

```{r}
ymd_hms("2017-01-31 20:11:59")
mdy_hm("01/31/2017 08:01")
```

시간대를 지정해 날짜를 날짜-시간으로 강제 변환(force)하기도 합니다.

```{r}
ymd("2017-01-31", tz = "UTC")
```

여기서는 GMT 또는 그리니치 표준시(Greenwich Mean Time)로도 알려진 경도(longitude) 0°의 시간대, UTC를 사용합니다. 일광 절약 시간제를 적용하지 않아 계산이 조금 더 쉽습니다. UTC라는 이름은 영어 "Coordinated Universal Time"과 프랑스어 "Temps Universel Coordonné" 사이에서 타협(compromise)한 결과입니다.

### 개별 구성 요소에서 생성

날짜-시간의 개별 구성 요소가 단일 문자열이 아니라 여러 열(columns)에 나뉘어(spread) 있는 경우도 있습니다. `flights` 데이터가 그렇습니다.

```{r}
flights |>
  select(year, month, day, hour, minute)
```

이런 입력으로 날짜를 만들 때는 `make_date()`, 날짜-시간을 만들 때는 `make_datetime()`을 사용합니다.

```{r}
flights |>
  select(year, month, day, hour, minute) |>
  mutate(departure = make_datetime(year, month, day, hour, minute))
```

`flights`의 시간 열 네 개에도 같은 작업을 해봅시다. 시간이 약간 이상한(odd) 형식으로 표시(represented)되므로 나머지 연산(modulus arithmetic)으로 시(hour)와 분(minute)을 추출합니다. 날짜-시간 변수를 만든 뒤에는 이 장에서 살펴볼 변수만 남깁니다.

```{r}
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
```

이 데이터로 한 해 동안의 출발 시간 분포를 시각화합니다.

```{r}
#| fig.alt: >
#|   x축에 출발 시간(2013년 1월-12월), y축에 항공편 수(0-1000)를 나타내는
#|   빈도 다각형(frequency polyon). 이 빈도 다각형은 일별로 구간이
#|   나누어져(binned) 있어 날짜별 항공편 시계열을 볼 수 있습니다. 이 패턴은
#|   주간 패턴(weekly pattern)에 의해 지배됩니다(dominated); 주말에는
#|   항공편이 적습니다. 2월 초, 7월 초, 11월 말, 12월 말에 항공편 수가
#|   놀랍도록(surprisingly) 적어 눈에 띄는 며칠이 있습니다.
flights_dt |>
  ggplot(aes(x = dep_time)) +
  geom_freqpoly(binwidth = 86400) # 86400초 = 1일
```

하루(single day) 안의 분포만 따로 시각화해 봅시다.

```{r}
#| fig.alt: >
#|   x축에 출발 시간(1월 1일 오전 6시 - 자정), y축에 항공편 수(0-17)를 나타내며,
#|   10분 단위(increments)로 구간이 나누어진 빈도 다각형. 변동성(variability)이
#|   높아 패턴을 많이 보기는 어렵지만(hard to see much pattern), 대부분의 구간에는
#|   8-12편의 항공편이 있으며 오전 6시 이전과 오후 8시 이후에는 항공편이 현저히(markedly)
#|   적습니다.
flights_dt |>
  filter(dep_time < ymd(20130102)) |>
  ggplot(aes(x = dep_time)) +
  geom_freqpoly(binwidth = 600) # 600초 = 10분
```

날짜-시간을 히스토그램 같은 수치적 문맥(numeric context)에서 사용하면 1은 1초를 뜻합니다. 따라서 `binwidth` 86400은 하루입니다. 날짜에서는 1이 1일을 뜻합니다.

### 다른 유형에서 생성

날짜-시간과 날짜 사이를 전환(switch)할 때는 `as_datetime()`과 `as_date()`를 사용합니다(job).

```{r}
as_datetime(today())
as_date(now())
```

날짜/시간이 1970-01-01인 "유닉스 에포크(Unix Epoch)"부터의 수치적 오프셋(numeric offsets)으로 주어지기도 합니다. 오프셋이 초 단위라면 `as_datetime()`, 일(days) 단위라면 `as_date()`를 사용합니다.

```{r}
as_datetime(60 * 60 * 10)
as_date(365 * 10 + 2)
```

### 연습 문제

1. 유효하지 않은 날짜가 포함된 문자열을 구문 분석(parse)하면 어떻게 됩니까?

```{r}
#| eval: false
ymd(c("2010-10-10", "bananas"))
```

2. `today()`의 `tzone` 인수는 무슨 일을 합니까? 이 인수가 중요한 이유는 무엇입니까?

3. 다음 날짜-시간을 readr 열 사양(column specification)과 lubridate 함수로 구문 분석하는 방법을 각각 보여주세요.

```{r}
d1 <- "January 1, 2010"
d2 <- "2015-Mar-07"
d3 <- "06-Jun-2017"
d4 <- c("August 19 (2015)", "July 1 (2015)")
d5 <- "12/30/14" # 2014년 12월 30일
t1 <- "1705"
t2 <- "11:15:10.12 PM"
```

## 날짜-시간 구성 요소

날짜-시간 데이터를 R의 데이터 구조로 가져왔으니 이제 활용 방법을 살펴봅시다(explore).

### 구성 요소 가져오기

접근자 함수 `year()`, `month()`, `mday()`(월의 일(day of the month)), `yday()`(연의 일(day of the year)), `wday()`(요일(day of the week)), `hour()`, `minute()`, `second()`로 날짜의 각 부분을 추출합니다(pull out).

이 함수들은 사실상(effectively) `make_datetime()`의 반대 역할을 합니다.

```{r}
datetime <- ymd_hms("2026-07-08 12:34:56")
year(datetime)
month(datetime)
mday(datetime)
yday(datetime)
wday(datetime)
```

`month()`와 `wday()`에 `label = TRUE`를 설정하면 월이나 요일의 약어 이름을 반환합니다. 전체 이름이 필요하면 `abbr = FALSE`도 설정하세요.

```{r}
month(datetime, label = TRUE)
wday(datetime, label = TRUE, abbr = FALSE)
```

`wday()`로 주말보다 주중(during the week)에 더 많은 항공편이 출발한다는 사실을 확인합니다.

```{r}
#| fig-alt: |
#|   x축에 요일(days of the week), y축에 항공편 수를 나타내는
#|   막대 차트. 월요일-금요일은 항공편 수가 대략 48,000편으로 비슷하며,
#|   일주일이 지나면서(over the course of the week) 약간씩 감소합니다.
#|   일요일은 약간 더 낮고(~45,000), 토요일은 훨씬 더 낮습니다(~38,000).
flights_dt |>
  mutate(wday = wday(dep_time, label = TRUE)) |>
  ggplot(aes(x = wday)) +
  geom_bar()
```

한 시간 안에서 분별 평균 출발 지연 시간도 살펴봅시다. 흥미롭게도 20-30분과 50-60분에 떠나는 항공편은 그 시간의 나머지 구간보다 지연이 훨씬 적습니다!

```{r}
#| fig-alt: |
#|   x축에 실제 출발 분(0-60), y축에 평균 지연 시간(4-20)을
#|   나타내는 선형 차트(line chart). 평균 지연 시간은 (0, 12)에서 시작하여
#|   (18, 20)까지 꾸준히(steadily) 증가하다가 급격히(sharply) 떨어져,
#|   매시 약 23분(23 minute past the hour)에 최소인 9분의 지연 시간을 기록합니다.
#|   그 후 다시 (17, 35)로 증가하고 (55, 4)로 급격히 감소합니다.
#|   마지막으로 (60, 9)로 증가하며 마무리됩니다.
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

반면 예정된(scheduled) 출발 시간에서는 이 패턴이 그만큼 뚜렷하지 않습니다(Interestingly).

```{r}
#| fig-alt: |
#|   x축에 예정된 출발 분(0-60), y축에 평균 지연 시간(4-16)을 나타내는
#|   선형 차트. 패턴이 상대적으로 적으며, 한 시간 동안 평균 지연 시간이
#|   아마 10분에서 8분으로 감소한다는 약간의 암시(suggestion)만 있습니다.
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

실제 출발 시간에서 이런 패턴이 나타나는 이유는 무엇일까요?

사람이 수집한 많은 데이터가 그렇듯 항공편에도 "깔끔한(nice)" 시간에 출발하는 강한 편향(bias)이 있습니다.

사람의 판단(judgement)이 개입한 데이터를 다룰 때는 이런 패턴을 늘 경계(alert)하세요!

```{r}
#| label: fig-human-rounding
#| fig-cap: |
#|   매시간(each hour) 출발하도록 예정된 항공편 수를 보여주는
#|   빈도 다각형. 0과 30 같은 딱 떨어지는 숫자(round numbers)와
#|   일반적으로 5의 배수인 숫자에 대한 강한 선호(preference)를 볼 수 있습니다.
#| fig-alt: |
#|   x축에 출발 분(0-60), y축에 항공편 수(0-60000)를 나타내는
#|   선 그래프. 대부분의 항공편은 정각(~60,000)이나
#|   30분(~35,000)에 출발하도록 예정되어 있습니다. 그 외에는
#|   거의 모든 항공편이 5의 배수에 출발하도록 예정되어 있으며,
#|   15분, 45분, 55분에 몇 편 더(a few extra) 있습니다.
#| echo: false
ggplot(sched_dep, aes(x = minute, y = n)) +
  geom_line()
```

### 반올림 (Rounding)

개별 구성 요소를 시각화(plotting)하는 다른(alternative) 방법도 있습니다. `floor_date()`, `round_date()`, `ceiling_date()`로 날짜를 가까운(nearby) 시간 단위에 맞춰 반올림합니다.

각 함수에는 조정(adjust)할 날짜 벡터와 내림(floor), 올림(ceiling), 반올림(round to)에 사용할 단위 이름을 넣습니다. 이를 이용해 주당 항공편 수를 시각화해 봅시다.

```{r}
#| fig-alt: |
#|   x축에 주(2013년 1월-12월), y축에 항공편 수(2,000-7,000)를
#|   나타내는 선 그래프. 2월부터 11월까지는 주당 약 7,000편의
#|   항공편으로 꽤 평탄한(flat) 패턴을 보입니다. 연초의
#|   첫 주(약 4,500편)와 마지막 주(약 2,500편)에는 항공편 수가
#|   훨씬 적습니다.
flights_dt |>
  count(week = floor_date(dep_time, "week")) |>
  ggplot(aes(x = week, y = n)) +
  geom_line() +
  geom_point()
```

`dep_time`과 그날의 가장 이른 순간(earliest instant)의 차이를 계산해 하루 동안의(across the course of a day) 항공편 분포를 나타냅니다.

```{r}
#| fig-alt: |
#|   x축에 출발 시간을 나타내는 선 그래프. 자정 이후의
#|   초 단위(units of seconds)이므로 해석하기 어렵습니다.
flights_dt |>
  mutate(dep_hour = dep_time - floor_date(dep_time, "day")) |>
  ggplot(aes(x = dep_hour)) +
  geom_freqpoly(binwidth = 60 * 30)
```

두(pair of) 날짜-시간의 차이를 계산하면 `difftime`이 나옵니다. 이를 `hms` 객체로 바꾸면 x축을 더 알아보기 쉽습니다.

```{r}
#| fig-alt: |
#|   x축에 출발 시간(자정에서 자정), y축에 항공편 수(0에서 15,000)를
#|   나타내는 선 그래프. 오전 5시 이전에는 항공편이 매우 적습니다(<100).
#|   그런 다음 항공편 수는 시간당 12,000편으로 빠르게 증가하여
#|   오전 9시에 15,000편으로 최고치(peaking)에 달한 후, 오전 10시부터
#|   오후 2시까지 시간당 약 8,000편으로 떨어집니다(falling). 그런 다음 항공편 수는
#|   오후 8시까지 시간당 약 12,000편으로 다시 증가하다가 다시 급격히(rapidly) 감소합니다.
flights_dt |>
  mutate(dep_hour = hms::as_hms(dep_time - floor_date(dep_time, "day"))) |>
  ggplot(aes(x = dep_hour)) +
  geom_freqpoly(binwidth = 60 * 30)
```

### 구성 요소 수정하기

접근자 함수로 날짜/시간의 구성 요소를 수정하기도 합니다. 데이터 분석에서 자주 생기는(come up) 작업은 아니지만 날짜가 명백히 잘못된 데이터를 정리(cleaning)할 때 유용합니다.

```{r}
(datetime <- ymd_hms("2026-07-08 12:34:56"))

year(datetime) <- 2030
datetime

month(datetime) <- 01
datetime

hour(datetime) <- hour(datetime) + 1
datetime
```

기존 변수를 직접 수정하지 않고 `update()`로 새 날짜-시간을 만들 수도 있습니다. 한 번에(in one step) 여러 값도 설정합니다.

```{r}
update(datetime, year = 2030, month = 2, mday = 2, hour = 2)
```

값이 범위를 넘으면 이월(roll-over)됩니다.

```{r}
update(ymd("2023-02-01"), mday = 30)
update(ymd("2023-02-01"), hour = 400)
```

### 연습 문제

1. 하루 중 항공편 시간 분포는 한 해 동안(over the course of the year) 어떻게 변합니까?

2. `dep_time`, `sched_dep_time`, `dep_delay`를 비교하세요. 세 변수는 일관성이 있습니까(consistent)? 발견한 내용을 설명하세요.

3. `air_time`을 출발(departure)부터 도착(arrival)까지의 지속 시간(duration)과 비교하세요. 발견한 내용을 설명하세요. (힌트: 공항의 위치를 고려하세요.)

4. 평균 지연 시간은 하루 동안 어떻게 변합니까? `dep_time`을 사용해야 합니까, 아니면 `sched_dep_time`을 사용해야 합니까?     이유는 무엇입니까?

5. 지연 가능성(chance)을 최소화(minimise)하려면 어느 요일에 출발해야 합니까?

6. `diamonds$carat`의 분포와 `flights$sched_dep_time`의 분포를 비슷하게 만드는 것은 무엇입니까?

7. 20-30분과 50-60분에 항공편이 일찍 출발(early departures)하는 이유가 일찍 떠나도록 예정된 항공편 때문이라는 가설(hypothesis)을 확인하세요. (힌트: 항공편의 지연 여부를 나타내는 이진 변수(binary variable)를 만드세요.)

## 시간 범위

이제 뺄셈(subtraction), 덧셈(addition), 나눗셈(division) 같은 날짜 산술 연산을 알아봅니다. 그 과정에서(Along the way) 시간 범위를 나타내는 중요한 클래스 세 가지를 다룹니다.

1. 지속 시간(Durations)은 정확한 초 수를 나타냅니다.
2. 기간(Periods)은 주(weeks)나 월(months)과 같은 인간의 단위를 나타냅니다.
3. 구간(Intervals)은 시작점과 끝점을 나타냅니다.

지속 시간, 기간, 구간 가운데 무엇을 선택해야 할까요? 늘 그렇듯이(As always) 문제를 해결하는 가장 단순한 데이터 구조를 고르세요. 물리적 시간만 중요하다면(care about) 지속 시간을, 인간의 시간을 더해야 한다면 기간을 사용합니다. 범위의 길이를 인간의 단위로 파악해야 한다면 구간이 알맞습니다.

### 지속 시간 (Durations)

R에서 두 날짜를 빼면 `difftime` 객체를 얻습니다.

```{r}
# Hadley의 나이는 몇 살입니까?
h_age <- today() - ymd("1979-10-14")
h_age
```

`difftime` 클래스 객체는 시간 범위를 초, 분, 시간, 일 또는 주 단위로 기록합니다. 단위가 모호해서(ambiguity) `difftime`을 다루기 번거롭습니다. lubridate의 대안은 항상 초를 쓰는 지속 시간(duration)입니다.

```{r}
as.duration(h_age)
```

지속 시간에는 편리한 생성자(convenient constructors)가 여러 개 있습니다.

```{r}
dseconds(15)
dminutes(10)
dhours(c(12, 24))
ddays(0:5)
dweeks(3)
dyears(1)
```

지속 시간은 시간 범위를 언제나 초 단위로 기록합니다. 분, 시간, 일, 주, 연도 초로 환산합니다. 1분은 60초, 1시간은 60분, 하루는 24시간, 일주일은 7일입니다. 더 큰 시간 단위는 문제가 많습니다(problematic). 1년은 "평균" 일수인 365.25일을 사용합니다. 변동(variation)이 너무 커서 월은 지속 시간으로 바꿀 수 없습니다.

지속 시간은 더하거나 곱합니다.

```{r}
2 * dyears(1)
dyears(1) + dweeks(12) + dhours(15)
```

날짜(days)에도 지속 시간을 더하거나 뺍니다.

```{r}
tomorrow <- today() + ddays(1)
last_year <- today() - dyears(1)
```

다만 지속 시간은 정확한 초 수를 나타내므로 때로는 예상치 못한(unexpected) 결과가 나옵니다.

```{r}
one_am <- ymd_hms("2026-03-08 01:00:00", tz = "America/New_York")
one_am
one_am + ddays(1)
```

왜 3월 8일 오전 1시에서 하루 뒤가 3월 9일 오전 2시일까요? 자세히 보면 시간대도 바뀌었습니다. 3월 8일에는 일광 절약 시간제(DST)가 시작돼 하루가 23시간뿐입니다. 여기에 하루 분량의(full days worth of) 초를 더하면 시각이 달라집니다.

### 기간 (Periods)

이 문제를 해결하는 lubridate의 자료형이 기간(periods)입니다. 기간도 시간 범위지만 초 단위의 고정된 길이는 없습니다. 대신 일(days)과 월(months) 같은 "인간의" 시간을 사용해 좀 더 직관적으로(intuitive way) 작동합니다.

```{r}
one_am
one_am + days(1)
```

기간도 지속 시간처럼 익숙한 여러 생성자 함수(constructor functions)로 만듭니다.

```{r}
hours(c(12, 24))
days(7)
months(1:6)
```

기간도 더하거나 곱합니다.

```{r}
10 * (months(6) + days(1))
days(50) + hours(25) + minutes(2)
```

물론 날짜에도 더합니다. 기간은 지속 시간보다 예상대로 작동하는 경우가 많습니다.

```{r}
# 윤년
ymd("2024-01-01") + dyears(1)
ymd("2024-01-01") + years(1)

# 일광 절약 시간제
one_am + ddays(1)
one_am + days(1)
```

기간을 사용해 항공편 날짜의 이상한 점(oddity)을 고쳐봅시다. 일부 비행기는 뉴욕시에서 출발하기도 전에 목적지에 도착한 것처럼 보입니다.

```{r}
flights_dt |>
  filter(arr_time < dep_time)
```

이 항공편들은 야간 비행(overnight flights)입니다. 출발 시간과 도착 시간에 같은 날짜를 넣었지만 실제 도착일은 다음 날입니다. 야간 항공편의 도착 시간에 `days(1)`을 더해 수정합니다.

```{r}
flights_dt <- flights_dt |>
  mutate(
    overnight = arr_time < dep_time,
    arr_time = arr_time + days(overnight),
    sched_arr_time = sched_arr_time + days(overnight)
  )
```

이제 모든 항공편이 물리학의 법칙(laws of physics)을 따릅니다(obey).

```{r}
flights_dt |>
  filter(arr_time < dep_time)
```

### 구간

`dyears(1) / ddays(365)`는 무엇을 반환할까요? 답은 1이 아닙니다. `dyears()`가 365.25일에 해당하는 평균 1년의 초 수로 정의되기 때문입니다.

`years(1) / days(1)`은 어떨까요? 연도가 2015년이면 365, 2016년이면 366을 반환해야 합니다!

lubridate가 하나의 명확한 답을 내리기에는 정보가 부족합니다. 그래서 추정치(estimate)를 반환합니다.

```{r}
years(1) / days(1)
```

더 정확히 측정(measurement)하려면 구간(interval)을 사용합니다. 구간은 시작 날짜-시간과 종료 날짜-시간의 쌍(pair), 또는 시작점이 있는 지속 시간으로 보면 됩니다.

구간은 `start %--% end`로 만듭니다.

```{r}
y2023 <- ymd("2023-01-01") %--% ymd("2024-01-01")
y2024 <- ymd("2024-01-01") %--% ymd("2025-01-01")

y2023
y2024
```

그런 다음 `days()`로 나눠 해당 연도에 며칠이 들어가는지(fit) 확인합니다.

```{r}
y2023 / days(1)
y2024 / days(1)
```

### 연습 문제

1. R을 막 배우기 시작한 사람에게 `days(!overnight)`와 `days(overnight)`를 설명하세요. 알아야 할 핵심 사실은 무엇입니까?

2. 2015년 각 달의 첫째 날을 담은(giving) 날짜 벡터를 만드세요. 현재 연도의 각 달도 같은 방식으로 만드세요.

3. 생일이 날짜로 주어지면 나이(in years)를 반환하는 함수를 작성하세요.

## 시간대

시간대는 지정학적 엔티티와 얽혀 있어 대단히 복잡합니다. 모든 세부 사항이 데이터 분석에 중요한 것은 아니므로 전부 파헤칠(dig into) 필요는 없습니다. 다만 정면으로 다뤄야 할(tackle head on) 몇 가지 과제는 있습니다.

첫 번째 과제는 일상에서 쓰는 시간대 이름이 대체로 모호하다(ambiguous)는 점입니다. 미국인이라면 EST, 즉 동부 표준시(Eastern Standard Time)가 익숙할 것입니다.

하지만 호주와 캐나다에도 EST가 있습니다! 이런 혼란을 피하려고 R은 국제 표준 IANA 시간대를 사용합니다. 이름은 일관된 `{area}/{location}` 체계(naming scheme)를 따르며 보통 `{continent}/{city}` 또는 `{ocean}/{city}` 형태입니다. 예로 "America/New_York", "Europe/Paris", "Pacific/Auckland"가 있습니다.

시간대는 국가나 그 안의 지역과 관련 있다고 생각하기 쉬운데, 이름에는 왜 도시를 쓸까요? IANA 데이터베이스가 수십 년에 걸친 시간대 규칙을 기록해야 하기 때문입니다. 국가 이름은 수십 년 동안 꽤 자주 바뀌거나 분할되지만(break apart) 도시 이름은 대체로 그대로입니다. 이름에는 현재의 동작(behavior)뿐 아니라 전체 역사도 반영해야(reflect) 합니다. 그래서 "America/New_York"과 "America/Detroit" 시간대가 따로 있습니다. 두 도시 모두 지금은 동부 표준시를 사용하지만 1969-1972년에는 미시간주, 즉 디트로이트가 있는 주에서 일광 절약 시간제(DST)를 따르지 않았습니다.

이런 이야기를 몇 가지 읽어보는 것만으로도 원시 시간대 데이터베이스([https://www.iana.org/time-zones](https://www.iana.org/time-zones))를 살펴볼 가치는 충분합니다!

R이 현재 시간대를 무엇으로 인식하는지는 `Sys.timezone()`으로 확인합니다.

```{r}
Sys.timezone()
```

(R이 모르면 `NA`가 나옵니다.)

모든 시간대 이름의 전체 목록은 `OlsonNames()`로 확인하세요.

```{r}
length(OlsonNames())
head(OlsonNames())
```

R에서 시간대는 날짜-시간의 출력(printing)만 제어하는 속성(attribute)입니다. 다음 세 객체도 시간상의 동일한 순간(instant in time)을 나타냅니다.

```{r}
x1 <- ymd_hms("2024-06-01 12:00:00", tz = "America/New_York")
x1

x2 <- ymd_hms("2024-06-01 18:00:00", tz = "Europe/Copenhagen")
x2

x3 <- ymd_hms("2024-06-02 04:00:00", tz = "Pacific/Auckland")
x3
```

뺄셈으로 같은 시간인지 확인합니다(verify).

```{r}
x1 - x2
x1 - x3
```

별도로 지정하지 않으면(Unless otherwise specified) lubridate는 항상 UTC를 사용합니다. UTC(협정 세계시 - Coordinated Universal Time)는 과학계(scientific community)의 표준 시간대이며 대략(roughly) GMT(그리니치 표준시 - Greenwich Mean Time)와 같습니다. 일광 절약 시간제(DST)가 없어 계산하기 편리합니다(representation). `c()`처럼 날짜-시간을 결합하는 연산은 시간대를 없애기도 합니다(drop). 이때 날짜-시간은 첫 번째 요소의 시간대로 표시됩니다.

```{r}
x4 <- c(x1, x2, x3)
x4
```

시간대를 바꾸는 방법은 두 가지입니다.

1. 시간상의 순간은 그대로 두고(Keep the instant in time the same) 표시 방식만 바꿉니다. 순간은 정확하지만 더 자연스럽게 표시하고 싶을 때 사용합니다.

```{r}
x4a <- with_tz(x4, tzone = "Australia/Lord_Howe")
x4a
x4a - x4
```

(여기서 시간대의 또 다른 문제도 드러납니다. 모든 시간대가 정수(integer) 시간 오프셋을 쓰는 것은 아닙니다!)

2. 시간상의 근본적인(underlying) 순간을 바꿉니다. 순간에 잘못된 시간대 레이블이 붙어 있어 이를 고칠 때 사용합니다.

```{r}
x4b <- force_tz(x4, tzone = "Australia/Lord_Howe")
x4b
x4b - x4
```

<!-- HUMANIZE-SUMMARY
원본 글자수: 20,427자
윤문본 글자수: 19,286자
변경률: 10.0% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 7 → 0
- A-7 가지다 직역: 0 → 0
- A-10 가능 표현 남발: 20 → 0
- A-11 목적절 남발: 4 → 0
- A-15 본문 추상 주어·만능 동사: 5 → 0
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
- "날짜와 시간으로 작업하는 방법을 보여줄 것입니다" → "날짜와 시간을 다루는 방법을 알아봅니다"
- "그것들은 큰 혼란을 일으키는 것 같지 않습니다" → "일상에서 늘 사용해도 크게 혼란스럽지 않기 때문입니다"
- "그것이 하는 일은 추정치를 제공하는 것입니다" → "그래서 추정치를 반환합니다"
- "지정학적 엔티티와의 상호 작용 때문에" → "지정학적 엔티티와 얽혀 있어"
- "시간대는 인쇄만 제어하는 날짜-시간의 속성" → "시간대는 날짜-시간의 출력만 제어하는 속성"
-->
