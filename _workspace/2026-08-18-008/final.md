---
title: "벡터: 논리형"
---

```{r}
#| echo: false
source("_common.R")
```

> 논리형 벡터는 각 요소가 `TRUE`, `FALSE`, `NA` 가운데 하나이므로 단순한 유형의 벡터입니다. 원시 데이터(raw data)에서는 비교적 드물지만 거의 모든 분석 과정에서 논리형 벡터를 만들고 조작(manipulate)합니다.

이 장에서 배울 함수는 대부분 기본(base) R에서 제공하므로 tidyverse가 필요하지 않습니다. 다만 데이터 프레임을 다룰 때 `mutate()`, `filter()`와 관련 함수(friends)를 쓰려고 계속 로드(load)해 두겠습니다. 예제도 계속 `nycflights13::flights` 데이터셋을 씁니다.

```{r}
#| label: setup
#| message: false
library(tidyverse)
library(nycflights13)
```

도구를 더 많이 다루다 보면 항상 딱 맞는 실제 사례가 있는 것은 아닙니다.
그래서 `c()`로 더미 데이터(dummy data)를 몇 가지 만들어 보겠습니다.

```{r}
x <- c(1, 2, 3, 5, 7, 11, 13)
x * 2
```

이렇게 하면 개별 함수를 설명하기는 쉬워지지만 여러분의 데이터 문제에 적용하는 방법은 파악하기 어려워집니다. 자유롭게 떠다니는(free-floating) 벡터에 하는 모든 조작은 `mutate()`와 관련 함수를 써서 데이터 프레임 안의 변수에도 적용됩니다.

```{r}
df <- tibble(x)
df |> 
  mutate(y = x * 2)
```

## 비교

논리형 벡터는 흔히 `<`, `<=`, `>`, `>=`, `!=`, `==`로 숫자를 비교해 만듭니다.
지금까지는 대부분 `filter()` 안에서 논리형 변수를 일시적으로(transiently) 만들었습니다 --- 계산하고(computed) 사용한 뒤 버리는 변수입니다. 예를 들어 다음 필터는 대략 제시간에(roughly on time) 도착하는 모든 주간(daytime) 출발을 찾습니다.

```{r}
flights |> 
  filter(dep_time > 600 & dep_time < 2000 & abs(arr_delay) < 20)
```

이 방식은 지름길(shortcut)입니다. 기반이 되는(underlying) 논리형 변수를 명시하려면 `mutate()`로 직접 만듭니다.

```{r}
flights |> 
  mutate(
    daytime = dep_time > 600 & dep_time < 2000,
    approx_ontime = abs(arr_delay) < 20,
    .keep = "used"
  )
```

논리가 복잡할수록 이 방식이 유용합니다. 중간 단계에 이름을 붙이면 코드를 읽고 각 단계가 올바르게 계산됐는지 확인하기가 쉬워집니다.

모두 합쳐서 초기 필터는 다음과 동등합니다.

```{r}
#| results: false
flights |> 
  mutate(
    daytime = dep_time > 600 & dep_time < 2000,
    approx_ontime = abs(arr_delay) < 20,
  ) |> 
  filter(daytime & approx_ontime)
```

### 부동 소수점 비교

숫자에 `==`를 사용할 때는 주의하세요. 예를 들어 이 벡터는 숫자 1과 2를 포함하는 것처럼 보입니다.

```{r}
x <- c(1 / 49 * 49, sqrt(2) ^ 2)
x
```

하지만 두 값이 같은지(`==`) 검사하면 `FALSE`가 나옵니다.

```{r}
x == c(1, 2)
```

무슨 일이 일어난 걸까요? 컴퓨터는 숫자를 고정된 수의 소수 자릿수(decimal places)로 저장합니다. 따라서 1/49나 `sqrt(2)`를 정확하게 표현할 방법이 없고 후속 계산도 아주 약간(very slightly) 어긋납니다. `digits`[^logicals-1] 인수를 지정해 `print()`를 호출하면 정확한 값이 드러납니다.

[^logicals-1]: R은 일반적으로 print를 대신 호출하지만(즉, `x`는 `print(x)`의 단축키입니다) 다른 인수를 지정할 때는 명시적으로 호출하는 편이 좋습니다.

```{r}
print(x, digits = 16)
```

R이 이 숫자를 기본적으로 반올림하는 이유는 실제 값이 여러분의 예상과 매우 가깝기 때문입니다. `==`가 실패하는 이유를 알았으니 어떻게 해야 할까요? 작은 차이를 무시하는 `dplyr::near()`가 한 가지 방법입니다.

```{r}
near(x, c(1, 2))
```

### 결측값

결측값(missing values)은 알 수 없는(unknown) 것을 나타내므로 "전염성(contagious)"이 있습니다. 알 수 없는 값이 들어간 연산의 결과도 대부분 알 수 없는 값(unknown)이 됩니다.

```{r}
NA > 5
10 == NA
```

혼란스러운(confusing) 결과는 이것입니다.

```{r}
NA == NA
```

문맥을 조금 보태면 왜 이런 결과가 나오는지 이해하기 쉽습니다.

```{r}
# 메리의 나이를 모릅니다
age_mary <- NA

# 존의 나이를 모릅니다
age_john <- NA

# 메리와 존의 나이가 같은가요?
age_mary == age_john
# 우리는 모릅니다!
```

`dep_time`이 결측된 비행편을 모두 찾을 때 다음 코드는 작동하지 않습니다. `dep_time == NA`가 모든 행에 `NA`를 만들고 `filter()`는 결측값을 자동으로 버리기(drops) 때문입니다.

```{r}
flights |> 
  filter(dep_time == NA)
```

대신 새로운 도구가 필요합니다.

### `is.na()`

`is.na(x)`는 모든 유형의 벡터에서 작동하며 결측값이면 `TRUE`를, 그 외에는 `FALSE`를 반환합니다.

```{r}
is.na(c(TRUE, NA, FALSE))
is.na(c(1, NA, 3))
is.na(c("a", NA, "b"))
```

`is.na()`를 사용하면 `dep_time`이 결측된 모든 행을 찾습니다.

```{r}
flights |> 
  filter(is.na(dep_time))
```

`is.na()`는 `arrange()`에서도 유용합니다. `arrange()`는 보통 모든 결측값을 맨 끝에 두지만 `is.na()`를 먼저 정렬하면 이 기본값을 재정의(override)합니다.

```{r}
flights |> 
  filter(month == 1, day == 1) |> 
  arrange(dep_time)

flights |> 
  filter(month == 1, day == 1) |> 
  arrange(desc(is.na(dep_time)), dep_time)
```

### 연습 문제

1. `dplyr::near()`는 어떻게 작동하나요? 소스 코드를 보려면 `near`를 입력하세요. `sqrt(2)^2`는 2에 가까운가요(near)?

2. `mutate()`, `is.na()`, `count()`를 함께 사용하여 `dep_time`, `sched_dep_time`, `dep_delay`의 결측값이 어떻게 연결되어 있는지 설명하세요.

## 부울 대수

여러 논리형 벡터는 부울 대수(boolean algebra)로 결합합니다.
R에서 `&`는 "and(그리고)", `|`는 "or(또는)", `!`는 "not(아니다)", `xor()`는 배타적 논리합(exclusive or)[^logicals-2]입니다. 예를 들어 `df |> filter(!is.na(x))`는 `x`가 누락되지 않은 모든 행을 찾고 `df |> filter(x < -10 | x > 0)`는 `x`가 -10보다 작거나 0보다 큰 모든 행을 찾습니다.

[^logicals-2]: 즉, `xor(x, y)`는 x가 true이거나 y가 true이지만 둘 다는 아닌 경우에 true입니다. 이것이 우리가 영어(또는 한국어 일상어)에서 일반적으로 "or(또는)"를 사용하는 방식입니다. "아이스크림 또는 케이크 중 무엇을 드시겠습니까?"라는 질문에 "둘 다요"는 일반적으로 허용되는 대답이 아닙니다.

R에는 `&`와 `|` 외에 `&&`와 `||`도 있습니다. dplyr 함수 안에서는 사용하지 마세요! 단락(short-circuiting) 연산자라고 하며 단일(single) `TRUE` 또는 `FALSE`만 반환합니다. 데이터 과학보다 프로그래밍에서 중요합니다.

### 결측값

부울 대수에서 결측값에 적용되는 규칙은 언뜻(at first glance) 일관성이 없어(inconsistent) 보여 설명하기가 조금 까다롭습니다.

```{r}
df <- tibble(x = c(TRUE, FALSE, NA))

df |> 
  mutate(
    and = x & NA,
    or = x | NA
  )
```

무슨 일이 일어나는지 이해하려면 `NA | TRUE`(`NA` 또는 `TRUE`)를 생각해 보세요. 논리형 벡터의 결측값은 `TRUE`일 수도, `FALSE`일 수도 있습니다. `TRUE | TRUE`와 `FALSE | TRUE`는 적어도 하나가 `TRUE`이므로 둘 다 `TRUE`입니다. `NA | TRUE` 역시 `NA`의 실제 값과 관계없이 반드시 `TRUE`입니다. 반면 `NA | FALSE`는 `NA`가 `TRUE`인지 `FALSE`인지 모르므로 `NA`입니다. 두 조건을 모두 충족(fulfilled)해야 하는 `&`에도 비슷한 추론이 적용됩니다. `NA & TRUE`는 `NA`가 `TRUE`이거나 `FALSE`일 수 있어 `NA`이고 `NA & FALSE`는 조건 중 적어도 하나가 `FALSE`이므로 `FALSE`입니다.

### 연산 순서

연산 순서는 영어(또는 한국어 일상어)와 같은 방식으로 작동하지 않습니다. 11월이나 12월에 출발한 모든 비행편을 찾는 코드를 살펴보세요.

```{r}
#| eval: false
flights |> 
   filter(month == 11 | month == 12)
```

"11월이나 12월에 출발한 모든 비행편을 찾아라(Find all flights that departed in November or December)"라는 말처럼 코드를 쓰고 싶은 유혹(tempted)이 생깁니다.

```{r}
flights |> 
   filter(month == 11 | 12)
```

이 코드는 오류를 내지는 않지만 제대로 작동하지도 않습니다. 무슨 일이 일어난 걸까요? R은 먼저 `month == 11`을 평가해 논리형 벡터를 만들고 이를 `nov`라고 부릅니다. 그런 다음 `nov | 12`를 계산합니다. 논리 연산자에 숫자를 쓰면 0을 제외한 모든 숫자가 `TRUE`로 변환됩니다. 따라서 `nov | 12`는 항상 `TRUE`인 `nov | TRUE`와 동일(equivalent)하며 결과적으로 모든 행이 선택됩니다.

```{r}
flights |> 
  mutate(
    nov = month == 11,
    final = nov | 12,
    .keep = "used"
  )
```

### `%in%`

`==`와 `|`의 순서를 맞추는 문제는 `%in%`으로 쉽게 피합니다. `x %in% y`는 `x`의 값이 `y` 어디에든 있으면 `TRUE`가 되는, `x`와 길이가 같은 논리형 벡터를 반환합니다.

```{r}
1:12 %in% c(1, 5, 11)
letters[1:10] %in% c("a", "e", "i", "o", "u")
```

11월과 12월의 모든 비행편은 다음과 같이 찾습니다.

```{r}
#| eval: false
flights |> 
  filter(month %in% c(11, 12))
```

`NA %in% NA`는 `TRUE`이므로 `%in%`은 `NA`를 다룰 때 `==`와 다른 규칙을 따른다는 점에 유의하세요.

```{r}
c(1, 2, NA) == NA
c(1, 2, NA) %in% NA
```

유용한 지름길입니다.

```{r}
flights |> 
  filter(dep_time %in% c(NA, 0800))
```

### 연습 문제

1. `arr_delay`는 누락되었지만 `dep_delay`는 누락되지 않은 모든 비행편을 찾으세요. `arr_time`도 `sched_arr_time`도 누락되지 않았지만 `arr_delay`는 누락된 모든 비행편을 찾으세요.

2. 누락된 `dep_time`이 있는 비행편은 몇 개인가요? 이 행에서 다른 변수는 무엇이 누락되었습니까? 이 행들은 무엇을 나타낼 수 있나요?

3. 누락된 `dep_time`이 비행편이 취소되었음을 암시(implies)한다고 가정하고 하루 취소된 비행편 수를 살펴보세요. 패턴이 있습니까? 취소된 비행편의 비율(proportion)과 취소되지 않은 비행편의 평균 지연 시간 사이에 연관성이 있습니까?

## 논리형 요약

주요 논리형 요약 함수는 `any()`와 `all()`입니다. `any(x)`는 `|`와 같아서 `x`에 `TRUE`가 하나라도 있으면 `TRUE`를 반환합니다. `all(x)`는 `&`와 같아서 `x`의 모든 값이 `TRUE`일 때만 `TRUE`를 반환합니다. 다른 요약 함수와 마찬가지로 `na.rm = TRUE`를 사용하면 결측값을 제거합니다.

예를 들어 `all()`과 `any()`로 모든 비행편이 출발할 때 최대 1시간만 지연됐는지, 도착할 때 5시간 이상 지연된 비행편이 있는지 확인합니다. `group_by()`를 더하면 이를 일(day) 단위로 계산합니다.

```{r}
flights |> 
  group_by(year, month, day) |> 
  summarize(
    all_delayed = all(dep_delay <= 60, na.rm = TRUE),
    any_long_delay = any(arr_delay >= 300, na.rm = TRUE),
    .groups = "drop"
  )
```

하지만 대부분의 경우 `any()`와 `all()`은 너무 투박(crude)합니다. `TRUE`나 `FALSE`가 몇 개인지 더 자세히 알아보려면 숫자 요약이 필요합니다.

### 논리형 벡터의 숫자 요약

논리형 벡터를 숫자 문맥에서 사용하면 `TRUE`는 1, `FALSE`는 0이 됩니다. 따라서 `sum(x)`는 `TRUE`의 수를, `mean(x)`는 `TRUE`의 비율을 구합니다. `mean()`은 단순히 `sum()`을 `length()`로 나눈 값이므로 논리형 벡터를 요약할 때 두 함수가 매우 유용합니다.

예를 들어 출발할 때 최대 1시간만 지연된 비행편의 비율과 도착할 때 5시간 이상 지연된 비행편의 수를 확인합니다.

```{r}
flights |> 
  group_by(year, month, day) |> 
  summarize(
    proportion_delayed = mean(dep_delay <= 60, na.rm = TRUE),
    count_long_delay = sum(arr_delay >= 300, na.rm = TRUE),
    .groups = "drop"
  )
```

### 논리형 부분집합

요약에서 논리형 벡터는 한 가지 용도로 더 쓰입니다. 단일 변수를 관심 있는 부분 집합으로 필터링(filter)하는 일입니다. 이때 기본(base) `[` 연산자(부분집합으로 발음됨(pronounced subset))를 활용합니다.

실제로 지연된 비행편의 평균 지연 시간만 살펴본다고 해봅시다. 먼저 비행편을 필터링한 다음 평균 지연을 계산하는 방법이 있습니다.

```{r}
flights |> 
  filter(arr_delay > 0) |> 
  group_by(year, month, day) |> 
  summarize(
    behind = mean(arr_delay),
    n = n(),
    .groups = "drop"
  )
```

이 방법은 작동하지만 일찍 도착한 비행편의 평균 지연 시간도 계산하려면 어떻게 해야 할까요?
필터 단계를 따로 수행한 뒤 두 데이터 프레임을 결합해야 합니다. 대신 `[`로 인라인(inline) 필터링을 합니다. `arr_delay[arr_delay > 0]`는 양수인 도착 지연만 산출합니다.

이것은 다음으로 이어집니다.

```{r}
flights |> 
  group_by(year, month, day) |> 
  summarize(
    behind = mean(arr_delay[arr_delay > 0], na.rm = TRUE),
    ahead = mean(arr_delay[arr_delay < 0], na.rm = TRUE),
    n = n(),
    .groups = "drop"
  )
```

그룹 크기의 차이에도 유의하세요. 첫 번째 청크의 `n()`은 일별(per day) 지연 비행편 수를, 두 번째 청크의 `n()`은 전체 비행편 수를 구합니다.

### 연습 문제

1. `sum(is.na(x))`는 무엇을 말해 주나요? `mean(is.na(x))`는 어떨까요?

2. 논리형 벡터에 적용될 때 `prod()`는 무엇을 반환합니까? 어떤 논리형 요약 함수와 동일합니까? 논리형 벡터에 적용될 때 `min()`은 무엇을 반환합니까? 어떤 논리형 요약 함수와 동일합니까? 문서를 읽고 몇 가지 실험을 수행해 보세요.

## 조건부 변환

논리형 벡터의 강력한 특징 중 하나는 조건부 변환(conditional transformations)에 쓰인다는 점입니다. 조건 x에서는 한 작업을 하고 조건 y에서는 다른 작업을 합니다. 중요한 도구는 `if_else()`와 `case_when()` 두 가지입니다.

### `if_else()`

조건이 `TRUE`일 때와 `FALSE`일 때 서로 다른 값을 사용하려면 `dplyr::if_else()`[^logicals-4]를 씁니다. `if_else()`의 처음 세 인수는 항상 사용합니다. 첫 번째 인수 `condition`은 논리형 벡터입니다. 두 번째 `true`는 조건이 true일 때의 출력을, 세 번째 `false`는 조건이 false일 때의 출력을 정합니다.

[^logicals-4]: dplyr의 `if_else()`는 기본 R의 `ifelse()`와 매우 비슷하지만 두 가지 주요 장점이 있습니다. 결측값을 처리하는 방식을 선택하고 변수들의 유형이 호환되지 않으면(incompatible) 훨씬 더 의미 있는(meaningful) 오류를 낼 가능성이 높습니다.

숫자 벡터에 "+ve" (양수) 또는 "-ve" (음수) 라벨을 지정하는 간단한 예부터 시작해 보겠습니다.

```{r}
x <- c(-3:3, NA)
if_else(x > 0, "+ve", "-ve")
```

선택적인 네 번째 인수 `missing`이 있는데, 이는 입력이 `NA`일 경우 사용됩니다.

```{r}
if_else(x > 0, "+ve", "-ve", "???")
```

`true`와 `false` 인수에도 벡터를 넣습니다. 예를 들어 `abs()`의 최소 구현(minimal implementation)을 만들 수 있습니다.

```{r}
if_else(x < 0, -x, x)
```

지금까지는 모든 인수에 같은 벡터를 썼지만 물론 서로 혼합하고 일치시켜(mix and match)도 됩니다. 예를 들어 다음처럼 간단한 버전의 `coalesce()`를 구현합니다.

```{r}
x1 <- c(NA, 1, 2, NA)
y1 <- c(3, NA, 4, 6)
if_else(is.na(x1), y1, x1)
```

위의 라벨 지정 예시에는 작은 부적절함(infelicity)이 있습니다. 0은 양수도 음수도 아닙니다. `if_else()`를 하나 더 추가하면 이 문제를 해결합니다.

```{r}
if_else(x == 0, "0", if_else(x < 0, "-ve", "+ve"), "???")
```

벌써 읽기가 조금 어렵고 조건이 늘어나면 더 복잡해집니다. 이때는 `dplyr::case_when()`으로 전환(switch)합니다.

### `case_when()`

dplyr의 `case_when()`은 SQL의 `CASE` 구문에서 영감을 받아(inspired by) 조건마다 다른 계산을 유연하게(flexible way) 수행합니다. 다만 tidyverse의 다른 구문과 전혀 다르게 생긴 특별한 문법(syntax)을 사용합니다. `condition ~ output` 형태의 쌍을 받으며 `condition`은 논리형 벡터여야 합니다. 조건이 `TRUE`이면 `output`을 사용합니다.

앞에서 중첩한(nested) `if_else()`는 다음과 같이 다시 작성합니다.

```{r}
x <- c(-3:3, NA)
case_when(
  x == 0   ~ "0",
  x < 0    ~ "-ve", 
  x > 0    ~ "+ve",
  is.na(x) ~ "???"
)
```

코드는 더 길지만 그만큼 명시적(explicit)입니다.

`case_when()`의 작동 방식을 이해하려고 더 간단한 사례를 몇 가지 살펴보겠습니다. 일치(match)하는 조건이 없으면 출력(output)은 `NA`가 됩니다.

```{r}
case_when(
  x < 0 ~ "-ve",
  x > 0 ~ "+ve"
)
```

"기본값(default)"/모두 잡기(catch all) 값을 생성하려면 `.default`를 사용하세요.

```{r}
case_when(
  x < 0 ~ "-ve",
  x > 0 ~ "+ve",
  .default = "???"
)
```

여러 조건이 일치하면 첫 번째 조건만 사용된다는 점에 유의하세요.

```{r}
case_when(
  x > 0 ~ "+ve",
  x > 2 ~ "big"
)
```

`if_else()`와 마찬가지로 `~` 양쪽에 변수를 쓰며 문제에 맞춰 서로 혼합하고 일치시킵니다. 예를 들어 `case_when()`으로 도착 지연에 사람이 읽기 쉬운(human readable) 라벨을 몇 가지 붙입니다.

```{r}
flights |> 
  mutate(
    status = case_when(
      is.na(arr_delay)      ~ "cancelled",
      arr_delay < -30       ~ "very early",
      arr_delay < -15       ~ "early",
      abs(arr_delay) <= 15  ~ "on time",
      arr_delay < 60        ~ "late",
      arr_delay < Inf       ~ "very late",
    ),
    .keep = "used"
  )
```

이처럼 복잡한 `case_when()` 구문을 작성할 때는 주의하세요. 저도 처음 두 번은 `<`와 `>`를 섞어 쓰다가 겹치는 조건을 계속 만들었습니다.

### 호환 가능한 유형

`if_else()`와 `case_when()`은 모두 출력 유형이 서로 호환되어야(compatible) 합니다. 호환되지 않으면 아래 오류가 나타납니다.

```{r}
#| error: true
if_else(TRUE, "a", 1)

case_when(
  x < -1 ~ TRUE,  
  x > 0  ~ now()
)
```

한 유형의 벡터를 다른 유형으로 자동 변환하면 오류가 자주 발생합니다. 그래서 전반적으로(overall) 호환되는 유형은 비교적 적습니다. 중요한 경우는 다음과 같습니다.

1. 숫자형(numeric)과 논리형(logical) 벡터는 호환됩니다.

2. 문자열(strings)과 팩터(factors)는 호환됩니다. 팩터를 제한된(restricted) 값의 집합을 지닌 문자열로 볼 수 있기 때문입니다.

3. 날짜(dates)와 날짜시간(date-times)은 호환됩니다. 날짜를 날짜시간의 특수한 경우로 볼 수 있기 때문입니다.

4. 기술적으로 논리형 벡터인 `NA`는 모든 유형과 호환됩니다. 모든 벡터에 결측값을 표현하는 방법이 있기 때문입니다.

이 규칙을 외울 필요는 없습니다(we don't expect you to). tidyverse 전반에 일관되게(consistently) 적용되므로 쓰다 보면 제2의 천성(second nature)이 될 것입니다.

### 연습 문제

1. 숫자가 2로 나누어떨어지면 짝수이며 R에서는 `x %% 2 == 0`으로 확인합니다. 이 사실과 `if_else()`를 사용하여 0에서 20 사이의 각 숫자가 짝수인지 홀수인지 확인하세요.

2. `x <- c("Monday", "Saturday", "Wednesday")`와 같은 요일 벡터가 주어지면 `if_else()` 구문을 사용하여 주말(weekends) 또는 평일(weekdays)로 라벨을 지정하세요.

3. `if_else()`를 사용하여 `x`라는 숫자 벡터의 절댓값을 계산하세요.

4. `flights`의 `month`와 `day` 열(columns)을 사용하여 미국의 중요한 선택된 휴일(holidays)(New Years Day, 4th of July, Thanksgiving, Christmas)에 라벨을 지정하는 `case_when()` 구문을 작성하세요. 먼저 `TRUE` 또는 `FALSE`인 논리형 열을 생성한 다음, 휴일의 이름을 제공하거나 `NA`인 문자형 열을 생성하세요.

<!-- HUMANIZE-SUMMARY
원본 글자수: 13,632자
윤문본 글자수: 12,590자
변경률: 10.7% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 14 → 0
- A-7 가지다 직역: 2 → 0
- A-10 가능 표현 남발: 27 → 0
- A-11 목적절 남발: 4 → 0
- A-15 본문 추상 주어·만능 동사: 3 → 0
- C-11 연결어미 뒤 쉼표: 7 → 0

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
- "세 가지 가능한 값 중 하나만 될 수 있기 때문에" → "세 값 가운데 하나이므로"
- "비교를 통하는 것입니다" → "비교해 만듭니다"
- "숫자를 고정된 수의 소수 자릿수로 저장하므로" → 문장을 나눠 부동 소수점 오차를 직접 설명
- "서로 다른 계산을 수행할 수 있는 방법을 제공합니다" → "조건마다 다른 계산을 수행합니다"
-->
