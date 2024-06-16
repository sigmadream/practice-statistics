---
title: "Joins"
---

```{r}
#| echo: false
source("_common.R")
```

데이터 분석에서 데이터 프레임을 하나만 사용하는 경우는 드뭅니다(rare). 보통 여러 데이터 프레임을 함께 조인(join)해 관심 있는 질문에 답합니다. 이 장에서는 중요한 조인 유형 두 가지를 소개합니다.

- 다른 데이터 프레임에서 일치하는 관측값을 가져와(from matching observations) 한 데이터 프레임에 새 변수를 추가하는 변형 조인(Mutating joins).
- 다른 데이터 프레임의 관측값과 일치하는지 여부를 기준으로 한 데이터 프레임의 관측값을 필터링하는 필터링 조인(Filtering joins).

```{r}
#| label: setup
#| message: false
library(tidyverse)
library(nycflights13)
```

## 키 (Keys)

조인을 이해하려면 먼저 각 테이블의 키 한 쌍이 두 테이블을 어떻게 연결하는지(connected) 알아야 합니다. 이 절에서는 키의 두 유형을 설명하고 nycflights13 데이터셋에서 각각의 예를 살펴봅니다. 키가 유효한지(valid) 확인하는 방법과 테이블에 키가 없을(lacks) 때의 처리 방법도 다룹니다.

### 기본 키와 외래 키

모든 조인에는 기본 키(primary key)와 외래 키(foreign key)가 한 쌍으로 들어갑니다. 기본 키는 각 관측값을 고유하게 식별하는 변수나 변수 집합입니다. 변수가 두 개 이상 필요하면 복합 키(compound key)라고 합니다.

- `airlines`는 각 항공사의 코드(carrier code)와 전체 이름을 기록합니다. 두 글자 코드로 항공사를 식별하므로 `carrier`가 기본 키입니다.

```{r}
airlines
```

- `airports`는 각 공항의 데이터를 기록합니다. 세 글자 공항 코드로 공항을 식별하므로 `faa`가 기본 키입니다.

```{r}
#| R.options:
#|   width: 67
airports
```

- `planes`는 각 비행기의 데이터를 기록합니다. 꼬리 번호(tail number)로 비행기를 식별하므로 `tailnum`이 기본 키입니다.

```{r}
#| R.options:
#|   width: 67
planes
```

- `weather`는 출발지 공항의 날씨를 기록합니다. 위치(location)와 시간의 조합으로 각 관측값을 식별하므로 `origin`과 `time_hour`가 복합 기본 키입니다.

```{r}
#| R.options:
#|   width: 67
weather
```

외래 키(foreign key)는 다른 테이블의 기본 키에 해당하는(corresponds to) 변수(또는 변수의 집합)입니다.

1. `flights$tailnum`은 기본 키 `planes$tailnum`에 해당하는 외래 키입니다.
2. `flights$carrier`는 기본 키 `airlines$carrier`에 해당하는 외래 키입니다.
3. `flights$origin`은 기본 키 `airports$faa`에 해당하는 외래 키입니다.
4. `flights$dest`는 기본 키 `airports$faa`에 해당하는 외래 키입니다.
5. `flights$origin`-`flights$time_hour`는 복합 기본 키 `weather$origin`-`weather$time_hour`에 해당하는 복합 외래 키입니다.

이 키 설계에는 편리한 특징(nice feature)이 있습니다. 기본 키와 외래 키의 이름이 거의 항상 같아서 조인이 훨씬 쉽습니다. 반대 관계(opposite relationship)도 눈여겨볼(noting) 만합니다. 여러 테이블에 쓰인 변수 이름은 거의 모두 같은 의미입니다. 예외는 `year` 하나뿐입니다. `flights`에서는 출발 연도, `planes`에서는 제조 연도를 뜻합니다. 실제로 테이블을 조인할 때 이 차이가 중요합니다.

### 기본 키 확인

각 테이블의 기본 키를 찾았다면 실제로 관측값을 고유하게 식별하는지 확인하는(verify) 습관을 들이세요. 기본 키를 `count()`한 뒤 `n`이 1보다 큰 항목(entries)을 찾으면 됩니다. 검사 결과 `planes`와 `weather`는 모두 문제가 없습니다.

```{r}
planes |> 
  count(tailnum) |> 
  filter(n > 1)

weather |> 
  count(time_hour, origin) |> 
  filter(n > 1)
```

기본 키에 결측값이 있는지도 확인해야 합니다. 결측값이 있으면 관측값을 식별하지 못합니다!

```{r}
planes |> 
  filter(is.na(tailnum))

weather |> 
  filter(is.na(time_hour) | is.na(origin))
```

### 대리 키

지금까지 `flights`의 기본 키는 다루지 않았습니다. 이 데이터 프레임을 외래 키로 사용하는 데이터 프레임이 없어 여기서는 그다지 중요하지 않습니다. 그래도 관측값을 다른 사람에게 설명할 방법이 있으면 작업이 쉬워지므로 고려할(still) 가치는 있습니다.

몇 가지 생각과 실험을 거쳐 각 항공편을 고유하게 식별하는 변수 세 개를 정했습니다(determined).

```{r}
flights |> 
  count(time_hour, carrier, flight) |> 
  filter(n > 1)
```

중복(duplicates)이 없으면 `time_hour`-`carrier`-`flight`가 자동으로 기본 키가 될까요? 좋은 출발점(good start)이지만 보장(guarantee)되지는 않습니다. 예를 들어 고도(altitude)와 위도(latitude)는 `airports`의 좋은 기본 키일까요?

```{r}
airports |>
  count(alt, lat) |> 
  filter(n > 1)
```

고도와 위도로 공항을 식별하는 것은 분명 나쁜 생각입니다(bad idea). 일반적으로 데이터만 보고 어떤 변수 조합이 좋은 기본 키인지 판단하기는 어렵습니다. 하지만 항공편에서는 `time_hour`, `carrier`, `flight`의 조합이 합리적으로(reasonable) 보입니다. 같은 편명(flight number)의 항공편 여러 대가 동시에 떠 있다면 항공사와 승객이 큰 혼란을 겪기 때문입니다.

그렇더라도(That said) 행 번호를 이용한 단순한 숫자형 대리 키(surrogate key)가 더 나을 수 있습니다.

```{r}
flights2 <- flights |> 
  mutate(id = row_number(), .before = 1)
flights2
```

대리 키는 다른 사람과 소통할 때 특히 유용합니다. 2013-01-03 오전 9시에 출발한 UA430을 보라고 말하는 것보다 2001번 항공편을 보라고 하는 편이 훨씬 쉽습니다.

### 연습 문제

1. `weather`에는 NYC에 있는 세 개의 출발지 공항에 대한 정보만 포함되어 있습니다. 미국 내 모든 공항의 날씨 기록이 포함되어 있다면 `flights`와 어떤 추가 연결(additional connection)을 갖게 될까요?

2. `year`, `month`, `day`, `hour` 및 `origin` 변수는 거의 `weather`의 복합 키를 형성(form)하지만 중복된 관측값이 있는 1시간이 있습니다. 그 시간의 어떤 점이 특별한지 알아낼 수 있습니까?

3. 우리는 일 년 중 어떤 날은 특별해서 평소보다 비행기를 타는 사람이 적다는 것을 알고 있습니다(크리스마스이브와 크리스마스). 해당 데이터를 데이터 프레임으로 어떻게 나타낼 수 있습니까? 기본 키는 무엇입니까? 기존 데이터 프레임과 어떻게 연결(connect)됩니까?

5. Lahman 패키지에 있는 `Batting`, `People`, `Salaries` 데이터 프레임 간의 연결을 보여주는 다이어그램을 그리세요. `People`, `Managers`, `AwardsManagers` 간의 관계를 보여주는 또 다른 다이어그램을 그리세요. `Batting`, `Pitching`, `Fielding` 데이터 프레임 간의 관계를 어떻게 특징지을(characterize) 수 있습니까?

## 기본 조인

데이터 프레임이 키로 연결되는 방식을 알았으니 조인으로 `flights` 데이터셋을 더 자세히 살펴봅시다. dplyr에는 `left_join()`, `inner_join()`, `right_join()`, `full_join()`, `semi_join()`, `anti_join()`이라는 조인 함수 여섯 개가 있습니다. 인터페이스는 모두 같습니다. 데이터 프레임 한 쌍(`x`와 `y`)을 받아 데이터 프레임을 반환하며 출력의 행과 열 순서는 주로 `x`가 결정합니다.

### 변형 조인

변형 조인(mutating join)은 두 데이터 프레임의 변수를 결합합니다(combine). 먼저 키를 기준으로 관측값을 일치시킨(matches) 뒤 한 데이터 프레임의 변수를 다른 데이터 프레임으로 복사합니다. `mutate()`처럼 조인 함수도 변수를 오른쪽에 추가하므로 기존 변수가 많으면 새 변수가 보이지 않습니다. 예제에서는 변수 6개만 남긴 더 좁은(narrower) 데이터셋을 만들어 진행 과정을 쉽게 확인하겠습니다.

```{r}
flights2 <- flights |> 
  select(year, time_hour, origin, dest, tailnum, carrier)
flights2
```

변형 조인은 네 유형이지만 실제로 가장 자주 쓰는 것은 `left_join()`입니다. 출력에 조인 대상(joining to) 데이터 프레임 `x`와 같은 행이 항상 남는다는 특징이 있습니다. `left_join()`은 주로 메타데이터를 덧붙일 때 사용합니다. 예를 들어 `flights2` 데이터에 전체 항공사 이름을 추가해 봅시다.

```{r}
flights2 |>
  left_join(airlines)
```

각 비행기가 출발할 때의 온도와 풍속도 알아봅시다.

```{r}
flights2 |> 
  left_join(weather |> select(origin, time_hour, temp, wind_speed))
```

비행 중인 비행기의 크기도 확인합니다.

```{r}
flights2 |> 
  left_join(planes |> select(tailnum, type, engines, seats))
```

`left_join()`이 `x`의 행과 일치하는 항목을 찾지 못하면(fails to find a match) 새 변수를 결측값으로 채웁니다. 예를 들어 꼬리 번호가 `N3ALAA`인 비행기의 정보가 없어서 `type`, `engines`, `seats`가 누락(missing)됩니다.

```{r}
flights2 |> 
  filter(tailnum == "N3ALAA") |> 
  left_join(planes |> select(tailnum, type, engines, seats))
```

이 장의 나머지 부분에서 이 문제로 몇 번 돌아오겠습니다.

### 조인 키 지정하기

기본적으로 `left_join()`은 두 데이터 프레임에 공통으로 나타나는 변수를 모두 조인 키로 사용합니다. 이를 자연(natural) 조인이라고 합니다. 유용한 휴리스틱(heuristic)이지만 언제나 제대로 작동하지는 않습니다. `flights2`를 완전한 `planes` 데이터셋과 조인하면 어떻게 될까요?

```{r}
flights2 |> 
  left_join(planes)
```

조인이 `tailnum`과 `year`를 복합 키로 사용하려 해서 결측 일치 항목이 많이 생깁니다. `flights`와 `planes`에 모두 `year` 열이 있지만 의미는 다릅니다. `flights$year`는 비행 연도, `planes$year`는 비행기 제작 연도입니다. `tailnum`만으로 조인하려면 `join_by()`에 명시적으로(explicit specification) 지정해야 합니다.

```{r}
flights2 |> 
  left_join(planes, join_by(tailnum))
```

출력의 `year` 변수에는 출처가 `x`인지 `y`인지 알려주는 접미사(suffix), 곧 `year.x`와 `year.y`가 붙어 구분됩니다(disambiguated). 기본 접미사는 `suffix` 인수로 재정의합니다(override).

`join_by(tailnum)`은 `join_by(tailnum == tailnum)`의 줄임말입니다. 이 완전한(fuller) 형태를 알아야 하는 이유는 두 가지입니다. 첫째, 두 테이블의 관계(relationship)를 나타냅니다. 키가 같아야(equal) 하므로 이런 조인을 흔히 동등 조인(equi join)이라고 합니다.

둘째, 테이블마다 이름이 다른 조인 키를 지정할 때 이 형태를 씁니다. `flight2`와 `airports` 테이블은 `dest` 또는 `origin`을 기준으로 조인합니다.

```{r}
flights2 |> 
  left_join(airports, join_by(dest == faa))

flights2 |> 
  left_join(airports, join_by(origin == faa))
```

이전 코드에서는 문자형 벡터로 조인 키를 지정하는 방식도 볼 수 있습니다.

1. `by = "x"`는 `join_by(x)`에 해당합니다.
2. `by = c("a" = "x")`는 `join_by(a == x)`에 해당합니다.

이제는 더 명확하고 유연한 사양(specification)을 제공하는 `join_by()`를 권합니다(prefer).

`inner_join()`, `right_join()`, `full_join()`의 인터페이스는 `left_join()`과 같습니다. 차이는 유지하는 행입니다. 왼쪽 조인은 `x`의 모든 행, 오른쪽 조인은 `y`의 모든 행, 전체 조인은 `x` 또는 `y`의 모든 행을 유지합니다. 내부 조인은 `x`와 `y`에 모두 나타나는(occur) 행만 남깁니다.

### 필터링 조인

필터링 조인은 이름 그대로 행을 필터링합니다. 세미 조인(semi-joins)과 안티 조인(anti-joins), 두 유형이 있습니다. 세미 조인은 `y`에 일치 항목이 있는 `x`의 행을 모두 유지합니다. 예를 들어 `airports` 데이터셋에서 출발지 공항만 골라낼 수 있습니다.

```{r}
airports |> 
  semi_join(flights2, join_by(faa == origin))
```

또는 도착지만 표시할 수도 있습니다.

```{r}
airports |> 
  semi_join(flights2, join_by(faa == dest))
```

안티 조인은 반대로 `y`에 일치 항목이 없는 `x`의 행을 모두 반환합니다. 데이터의 암묵적(implicit) 결측값을 찾는 데 유용합니다. 이런 결측값은 `NA`가 아니라 부재(absence)로만 드러납니다. 예를 들어 일치하는 도착지 공항이 없는 항공편을 찾아 `airports`에서 빠진 행을 확인합니다.

```{r}
flights2 |> 
  anti_join(airports, join_by(dest == faa)) |> 
  distinct(dest)
```

`planes`에서 누락된 `tailnum`도 확인합니다.

```{r}
flights2 |>
  anti_join(planes, join_by(tailnum)) |> 
  distinct(tailnum)
```

### 연습 문제

1. (일 년 내내) 심한 지연이 있었던 48시간을 찾으세요. 이를 `weather` 데이터와 교차 참조(Cross-reference)하세요. 어떤 패턴이 보입니까?

2. 이 코드를 사용하여 인기 있는 상위 10개 도착지를 찾았다고 상상해 보세요. 해당 도착지로 가는 모든 항공편을 어떻게 찾을 수 있습니까?

```{r}
top_dest <- flights2 |>
  count(dest, sort = TRUE) |>
  head(10)
```

3. 모든 출발 항공편에 해당 시간의 해당하는 날씨 데이터가 있습니까? 

4. `planes`에 일치하는 레코드가 없는 꼬리 번호들의 공통점은 무엇입니까? (힌트: 하나의 변수가 문제의 약 90%를 설명합니다.)

5.  해당 비행기를 비행한 모든 항공사(`carrier`)를 나열하는 열을 `planes`에 추가하세요. 각 비행기는 단일 항공사에서 비행하므로 비행기와 항공사 사이에 암묵적인 관계가 있다고 예상합니다. 이전 장에서 배운 도구를 사용하여 이 가설(hypothesis)을 확인(Confirm)하거나 기각(reject)하세요.

6. 출발지 *및* 도착지 공항의 위도와 경도를 `flights`에 추가하세요. 열 이름을 바꾸는 것이 조인 이전이 더 쉽습니까, 아니면 조인 이후가 더 쉽습니까? 

7. 목적지별 평균 지연 시간을 계산한 다음 `airports` 데이터 프레임과 조인하여 지연의 공간적 분포(spatial distribution)를 보여줄 수 있도록 하세요. 미국 지도를 그리는 쉬운 방법은 다음과 같습니다.

```{r}
#| eval: false
airports |>
  semi_join(flights, join_by(faa == dest)) |>
  ggplot(aes(x = lon, y = lat)) +
    borders("state") +
    geom_point() +
    coord_quickmap()
```

포인트의 크기(`size`)나 색상(`color`)으로 각 공항의 평균 지연 시간을 표시하세요.

8. 2013년 6월 13일에 무슨 일이 일어났습니까? 지연 지도를 그린 다음 Google을 사용하여 날씨와 교차 참조하세요.

```{r}
#| eval: false
#| include: false
worst <- filter(flights, !is.na(dep_time), month == 6, day == 13)
worst |>
  group_by(dest) |>
  summarize(delay = mean(arr_delay), n = n()) |>
  filter(n > 5) |>
  inner_join(airports, join_by(dest == faa)) |>
  ggplot(aes(x = lon, y = lat)) +
    borders("state") +
    geom_point(aes(size = n, color = delay)) +
    coord_quickmap()
```

## 조인은 어떻게 작동합니까?

조인을 몇 번 사용했으니 이제 `x`의 각 행이 `y`의 행과 일치하는 방식에 초점을 맞춰 작동 원리를 살펴봅시다. 예제에서는 `key`라는 단일 키와 값 열 하나(`val_x`와 `val_y`)를 사용하지만 같은 원리는 여러 키와 값에도 적용됩니다.

```{r}
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

1. 출력의 행과 열은 주로 `x`가 결정합니다. 따라서 `x` 테이블은 수평(horizontal)이며 출력과 나란히 정렬됩니다(lines up).

2. 왼쪽 조인(left join)은 `x`의 모든 관측값을 유지합니다. `x`의 각 행은 `y`의 `NA` 행을 폴백(fall back)으로 삼으므로 출력에 보존됩니다(preserved).

3. 오른쪽 조인(right join)은 `y`의 모든 관측값을 유지합니다. `y`의 각 행은 `x`의 `NA` 행을 폴백으로 삼아 출력에 남습니다. 출력은 가능한 한 `x`와 맞추고 `y`의 추가(extra) 행은 끝에 붙입니다.

4. 전체 조인(full join)은 `x` 또는 `y`에 나타나는 모든 관측값을 유지합니다. 두 테이블 모두 `NA` 폴백 행이 있어 모든 행이 출력에 포함됩니다. 출력은 `x`의 모든 행으로 시작하고 일치하지 않은 나머지(remaining) `y` 행이 뒤따릅니다.

5. 동등 조인(equi joins)에서는 키가 같을 때 행이 일치합니다. 가장 일반적인 조인이라 보통 동등이라는 접두사를 빼고 "동등 내부 조인"이 아닌 "내부 조인"이라고 합니다.

### 행 일치

지금까지는 `x`의 행이 `y`의 행 0개 또는 1개와 일치하는 경우를 살펴봤습니다. 둘 이상의 행과 일치하면 어떻게 될까요? `x`의 행에는 세 가지 결과(outcomes)가 생깁니다.

1. 아무것과도 일치하지 않으면 삭제됩니다.
2. `y`의 1개 행과 일치하면 보존됩니다.
3. `y`의 1개 이상의 행과 일치하면 일치할 때마다 한 번씩 복제(duplicated)됩니다.

원칙적으로(In principle) 출력 행과 `x`의 행 사이에는 보장된 대응(guaranteed correspondence)이 없습니다. 실제로 문제가 생기는 경우는 드물지만 조합적 폭발(combinatorial explosion)을 일으키는 특히 위험한(dangerous) 사례가 하나 있습니다. 다음 두 테이블을 조인한다고 해봅시다.

```{r}
df1 <- tibble(key = c(1, 2, 2), val_x = c("x1", "x2", "x3"))
df2 <- tibble(key = c(1, 2, 2), val_y = c("y1", "y2", "y3"))
```

`df1`의 첫 번째 행은 `df2`의 한 행과만 일치하지만 두 번째와 세 번째 행은 모두 두 행과 일치합니다. 이를 `다대다(many-to-many)` 조인이라고도 하며 dplyr은 경고를 표시합니다.

```{r}
df1 |> 
  inner_join(df2, join_by(key))
```

의도한(deliberately) 작업이라면 경고의 제안대로 `relationship = "many-to-many"`를 설정합니다.

### 필터링 조인

일치 항목의 수도 필터링 조인의 동작을 결정합니다.

1. 세미 조인은 `y`에 하나 이상의 일치 항목이 있는 `x`의 행을 유지합니다.
2. 안티 조인은 `y`에 일치하는 행이 0개인 `x`의 행을 유지합니다.

두 경우 모두 일치 항목의 존재 여부만 중요하고 횟수는 중요하지 않습니다. 따라서 필터링 조인은 변형 조인과 달리 행을 복제하지 않습니다.

## 비동등 조인

지금까지는 `x` 키와 `y` 키가 같을 때 행을 일치시키는 동등 조인만 살펴봤습니다. 이제 그 제한(restriction)을 완화하고(relax) 행 한 쌍의 일치 여부를 정하는 다른 방법을 알아봅시다.

그 전에 앞에서 적용한 단순화(simplification)를 다시 살펴봐야(revisit) 합니다. 동등 조인에서는 `x` 키와 `y` 키가 언제나 같으므로 출력에 하나만 표시합니다. dplyr이 두 키를 모두 유지하게 하려면 `keep = TRUE`를 사용하세요.

```{r}
x |> inner_join(y, join_by(key == key), keep = TRUE)
```

동등 조인에서 벗어나면(move away from) 키 값이 서로 다른 경우가 많아서 키를 모두 표시합니다. 가령 `x$key`와 `y$key`가 같을 때만 일치시키지 않고 `x$key`가 `y$key`보다 크거나 같을 때마다 일치시킬 수 있습니다. dplyr 조인 함수는 동등 조인과 비동등 조인을 구분하므로 비동등 조인에서는 두 키가 항상 나타납니다.

비동등 조인이라는 말은 조인이 무엇인지보다 무엇이 아닌지만 알려줘 그다지 유용하지 않습니다. dplyr은 이 가운데 특히 유용한 네 유형을 구분합니다.

1. 크로스 조인(Cross joins)은 모든 행 쌍을 일치시킵니다.
2. 부등 조인(Inequality joins)은 `==` 대신 `<`, `<=`, `>`, `>=`를 사용합니다.
3. 롤링 조인(Rolling joins)은 부등 조인과 유사하지만 가까운 일치 항목만 찾습니다.
4. 오버랩 조인(Overlap joins)은 범위(ranges)와 함께 작동하도록 설계된 특별한 유형의 부등 조인입니다.

각 유형은 다음 절에서 자세히 설명합니다.

### 크로스 조인

크로스 조인은 모든 행을 서로 일치시켜 데카르트 곱(Cartesian product)을 만듭니다. 출력 행 수는 `nrow(x) * nrow(y)`입니다.

```{r}
df1 <- tibble(key = c(1, 2, 2), val_x = c("x1", "x2", "x3"))
df2 <- tibble(key = c(1, 2, 2), val_y = c("y1", "y2", "y3"))
```

`df1`의 첫 번째 행은 `df2`의 한 행과만 일치하지만 두 번째와 세 번째 행은 모두 두 행과 일치합니다. 이를 `다대다(many-to-many)` 조인이라고도 하며 dplyr은 경고를 표시합니다.

```{r}
df1 |> 
  inner_join(df2, join_by(key))
```

의도한(deliberately) 작업이라면 경고의 제안대로 `relationship = "many-to-many"`를 설정합니다.

### 부등 조인

부등 조인은 `<`, `<=`, `>=`, `>`로 가능한 일치 항목의 범위를 제한합니다. 매우 일반적(general)이라 의미 있는(meaningful) 특정 사용 사례(use cases)를 들기(come up with)는 어렵습니다. 작지만 유용한 기술(technique)로 크로스 조인을 제한하면 모든 순열 대신 모든 조합(combinations)을 만들 수 있습니다.

```{r}
df <- tibble(id = 1:4, name = c("John", "Simon", "Tracy", "Max"))

df |> inner_join(df, join_by(id < id))
```

### 롤링 조인

롤링 조인은 부등식을 만족하는(satisfies) 모든 행이 아니라 가장 가까운 행만 가져오는 특수한 부등 조인입니다. 부등 조인에 `closest()`를 추가하면 롤링 조인이 됩니다. 예를 들어 `join_by(closest(x <= y))`는 x보다 크거나 같은 가장 작은 `y`와 일치하고 `join_by(closest(x > y))`는 `x`보다 작은 가장 큰 `y`와 일치합니다.

롤링 조인은 날짜가 정확히 일치하지 않는 두 테이블에서 특히 유용합니다. 테이블 2의 특정 날짜 전후에 있는 테이블 1의 가장 가까운 날짜를 찾을 수 있기 때문입니다. 회사의 파티 기획 위원회(party planning commission)를 맡고(in charge of) 있다고 해봅시다. 회사는 꽤 짠돌이(cheap)라 개인 파티 대신 분기마다 한 번만 파티를 엽니다. 개최 규칙은 조금 복잡합니다. 파티는 언제나 월요일에 열고 휴가를 가는 사람이 많은 1월 첫째 주는 건너뜁니다(skip). 2022년 3분기의 첫 번째 월요일은 7월 4일이라 일주일 뒤로 미룹니다(pushed back). 이에 따른 파티 날짜는 다음과 같습니다.

```{r}
parties <- tibble(
  q = 1:4,
  party = ymd(c("2022-01-10", "2022-04-04", "2022-07-11", "2022-10-03"))
)
```

이제 직원 생일 테이블이 있다고 해봅시다.

```{r}
set.seed(123)
employees <- tibble(
  name = sample(babynames::babynames$name, 100),
  birthday = ymd("2022-01-01") + (sample(365, 100, replace = TRUE) - 1)
)
employees
```

각 직원의 생일 이전 또는 당일에 열린 마지막 파티 날짜를 찾아야 합니다. 이를 롤링 조인으로 표현합니다.

```{r}
employees |> 
  left_join(parties, join_by(closest(birthday >= party)))
```

하지만 이 접근 방식(approach)에는 문제가 하나 있습니다. 1월 10일 이전에 태어난 사람(folks)은 파티를 열지 못합니다.

```{r}
employees |> 
  anti_join(parties, join_by(closest(birthday >= party)))
```

이 문제에는 오버랩 조인(overlap joins)이라는 다른 방법으로 접근해야 합니다(tackle).

### 오버랩 조인

오버랩 조인에는 구간(intervals)을 쉽게 다루도록 부등 조인을 이용한 도우미(helpers) 세 개가 있습니다.

-   `between(x, y_lower, y_upper)`는 `x >= y_lower, x <= y_upper`의 줄임말입니다.
-   `within(x_lower, x_upper, y_lower, y_upper)`는 `x_lower >= y_lower, x_upper <= y_upper`의 줄임말입니다.
-   `overlaps(x_lower, x_upper, y_lower, y_upper)`는 `x_lower <= y_upper, x_upper >= y_lower`의 줄임말입니다.

생일 예제로 사용법을 알아보겠습니다. 앞의 전략(strategy)에는 1월 1일부터 9일 사이의 생일을 위한 파티가 없다는 문제가 있습니다. 각 파티가 포괄하는(spans) 날짜 범위를 명시적으로(explicit) 정하고 초기 생일(early birthdays)을 특별히 처리하는 편이 낫습니다.

```{r}
parties <- tibble(
  q = 1:4,
  party = ymd(c("2022-01-10", "2022-04-04", "2022-07-11", "2022-10-03")),
  start = ymd(c("2022-01-01", "2022-04-04", "2022-07-11", "2022-10-03")),
  end = ymd(c("2022-04-03", "2022-07-11", "2022-10-02", "2022-12-31"))
)
parties
```

해들리(Hadley)는 데이터 입력을 절망적(hopelessly)으로 못해서 파티 기간이 겹치지(overlap) 않는지 확인하려고 합니다. 셀프 조인으로 각 시작-종료(start-end) 구간이 다른 구간과 겹치는지 검사합니다.

```{r}
parties |> 
  inner_join(parties, join_by(overlaps(start, end, start, end), q < q)) |> 
  select(start.x, end.x, start.y, end.y)
```

이런(Ooops), 겹치는 부분이 있습니다. 문제를 고치고 계속 진행합시다.

```{r}
parties <- tibble(
  q = 1:4,
  party = ymd(c("2022-01-10", "2022-04-04", "2022-07-11", "2022-10-03")),
  start = ymd(c("2022-01-01", "2022-04-04", "2022-07-11", "2022-10-03")),
  end = ymd(c("2022-04-03", "2022-07-10", "2022-10-02", "2022-12-31"))
)
```

이제 각 직원을 자신의 파티와 일치시킵니다. 파티가 할당되지 않은 직원을 빨리 확인하려면(we want to quickly find out) `unmatched = "error"`를 사용하는 편이 좋습니다.

```{r}
employees |> 
  inner_join(parties, join_by(between(birthday, start, end)), unmatched = "error")
```

### 연습 문제

1. 이 동등 조인에서 키에 어떤 일이 일어나는지 설명해 보세요. 왜 서로 다릅니까?

```{r}
x |> full_join(y, join_by(key == key))

x |> full_join(y, join_by(key == key), keep = TRUE)
```

2.  어떤 파티 기간이 다른 파티 기간과 겹치는지 알아낼 때 `join_by()`에서 `q < q`를 사용했습니다? 왜 그랬습니까? 이 부등식을 제거하면(remove) 어떻게 됩니까?

<!-- HUMANIZE-SUMMARY
원본 글자수: 17,243자
윤문본 글자수: 15,948자
변경률: 10.8% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 9 → 0
- A-7 가지다 직역: 2 → 0
- A-10 가능 표현 남발: 21 → 0
- A-11 목적절 남발: 0 → 0
- A-15 본문 추상 주어·만능 동사: 2 → 0
- C-11 연결어미 뒤 쉼표: 13 → 0

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
- "단 하나의 데이터 프레임만 포함되는 경우" → "데이터 프레임을 하나만 사용하는 경우"
- "좋은 특징이 있다는 것을 눈치채실 것입니다" → "편리한 특징이 있습니다"
- "여러분이 거의 항상 사용하게 될 조인" → "실제로 가장 자주 쓰는 조인"
- "조인이 무엇인지가 아니라 무엇이 아닌지만" → "무엇인지보다 무엇이 아닌지만"
- "출력에 `nrow(x) * nrow(y)` 행이 있음을 의미" → "출력 행 수는 `nrow(x) * nrow(y)`"
-->
