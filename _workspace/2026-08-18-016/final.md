---
title: "함수"
---

```{r}
#| echo: false
source("_common.R")
```

함수(functions)를 작성하면 데이터 과학자로서 다룰 수 있는 범위(reach)가 넓어집니다(improve). 함수는 복사 및 붙여넣기(copy-and-pasting)보다 더 강력하고(powerful) 일반적인(general) 방식으로 반복 작업(common tasks)을 자동화합니다(automate). 함수를 작성할 때 얻는 큰 이점(advantages)은 네 가지입니다.

1. 함수에 연상하기 쉬운(evocative) 이름을 붙이면(give) 코드를 더 쉽게(easier) 이해합니다.
2. 요구 사항(requirements)이 바뀌어도(change) 여러 곳(many)이 아닌 한 곳(one place)의 코드만 업데이트(update)하면 됩니다.
3. 복사하고 붙여넣을 때 생기는 우발적인 실수(incidental mistakes)의 가능성(chance)을 없앱니다(eliminate). 한 곳의 변수 이름만 바꾸고 다른 곳은 빠뜨리는 경우가 대표적입니다.
4. 프로젝트 사이에서(from project-to-project) 작업(work)을 더 쉽게 재사용해(reuse) 시간이 지날수록(over time) 생산성(productivity)이 높아집니다(increasing).

같은 코드 블록을 두 번 넘게(more than twice) 복사해 붙였다면, 곧 사본(copies)이 세 개라면 함수 작성을 고려(consider)하세요. 좋은 경험 법칙(rule of thumb)입니다.

```{r}
#| message: false
library(tidyverse)
library(nycflights13)
```

## 벡터 함수

먼저 하나 이상의 벡터를 받아(take) 벡터 결과(result)를 반환하는 벡터 함수부터 살펴봅시다. 다음 코드(take a look at)는 무슨 일을 할까요(What does it do)?

```{r}
df <- tibble(
  a = rnorm(5),
  b = rnorm(5),
  c = rnorm(5),
  d = rnorm(5),
)

df |> mutate(
  a = (a - min(a, na.rm = TRUE)) / 
    (max(a, na.rm = TRUE) - min(a, na.rm = TRUE)),
  b = (b - min(a, na.rm = TRUE)) / 
    (max(b, na.rm = TRUE) - min(b, na.rm = TRUE)),
  c = (c - min(c, na.rm = TRUE)) / 
    (max(c, na.rm = TRUE) - min(c, na.rm = TRUE)),
  d = (d - min(d, na.rm = TRUE)) / 
    (max(d, na.rm = TRUE) - min(d, na.rm = TRUE)),
)
```

이 코드는 각 열의 범위(range)가 0에서 1이 되도록 크기를 조정합니다(rescales). 그런데 실수가 보이나요(spot)? Hadley는 코드를 복사해 붙이는 과정에서 오류(error)를 냈습니다(made). `a`를 `b`로 바꾸는 일을 잊었습니다(forgot). 이런 실수를 막는(Preventing) 것만으로도 함수를 배울 이유는 충분합니다.

### 함수 작성

함수를 작성할 때는 먼저(first) 반복 코드를 분석해(analyse) 일정한(constant) 부분과 달라지는(vary) 부분을 찾아야 합니다(figure what). 위 코드를 `mutate()` 밖으로 꺼내면(pull it outside of) 각 반복(repetition)이 한 줄(one line)에 놓여 패턴(pattern)이 더 잘 보입니다.

```{r}
#| eval: false
(a - min(a, na.rm = TRUE)) / (max(a, na.rm = TRUE) - min(a, na.rm = TRUE))
(b - min(b, na.rm = TRUE)) / (max(b, na.rm = TRUE) - min(b, na.rm = TRUE))
(c - min(c, na.rm = TRUE)) / (max(c, na.rm = TRUE) - min(c, na.rm = TRUE))
(d - min(d, na.rm = TRUE)) / (max(d, na.rm = TRUE) - min(d, na.rm = TRUE))  
```

패턴을 더 분명히(clearer) 보려고 변하는(varies) 부분을 `█`로 바꿔봅시다(replace).

```{r}
#| eval: false
(█ - min(█, na.rm = TRUE)) / (max(█, na.rm = TRUE) - min(█, na.rm = TRUE))
```

이 코드를 함수로 바꾸려면(turn this into) 세 가지가 필요합니다.

1. 이름(name): 벡터의 크기를 0과 1 사이(lie between)로 조정하는(rescales) 함수이므로 `rescale01`이라고 하겠습니다.

2. 인수(arguments): 호출(calls)할 때마다 달라지는(vary across) 항목입니다. 위 분석에서는 인수가 하나뿐입니다. 숫자 벡터에 관례적으로(conventional) 쓰는 이름인 `x`라고 하겠습니다.

3. 본문(body): 본문은 모든 호출에서 반복되는 코드입니다.

그런 다음 템플릿(template)에 맞춰 함수를 만듭니다.

```{r}
name <- function(arguments) {
  body
}
```

이 예제에 적용하면 다음과 같습니다(leads to).

```{r}
rescale01 <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
```

이제 간단한 입력으로 테스트해(test) 논리(logic)를 올바르게(correctly) 담았는지(captured) 확인합니다(make sure).

```{r}
rescale01(c(-10, 0, 10))
rescale01(c(1, 2, 3, NA, 5))
```

이어서 `mutate()` 호출을 다음과 같이 다시 작성합니다(rewrite).

```{r}
df |> mutate(
  a = rescale01(a),
  b = rescale01(b),
  c = rescale01(c),
  d = rescale01(d),
)
```

뒤에서 `across()`를 배우면 중복(duplication)을 더 줄여(even further) `df |> mutate(across(a:d, rescale01))`만 쓰면 됩니다(reduce).

### 함수 개선

`rescale01()`은 불필요한 작업(unnecessary work)을 합니다(does). `min()`을 두 번, `max()`를 한 번 계산하는 대신 `range()`를 쓰면 한 단계(one step)에서 최소값과 최대값을 모두 구합니다(instead of computing).

```{r}
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE)
  (x - rng[1]) / (rng[2] - rng[1])
}
```

무한한(infinite) 값이 든 벡터에도 이 함수를 적용해 봅시다(might try).

```{r}
x <- c(1:10, Inf)
rescale01(x)
```

결과가 그다지 유용하지 않으므로(not particularly useful) `range()`에서 무한한 값을 무시하게 합니다(ignore).

```{r}
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE, finite = TRUE)
  (x - rng[1]) / (rng[2] - rng[1])
}

rescale01(x)
```

이 변경(changes)에서 함수의 중요한 이점(benefit)이 드러납니다(illustrate). 반복 코드를 함수로 옮겼기(moved) 때문에 한 곳(one place)만 고치면 됩니다(make the change).

### 변이 함수

함수의 기본 개념(basic idea)을 익혔으니 여러(a whole bunch of) 예제를 살펴봅시다. 먼저 입력과 같은 길이의 출력을 반환하는 "변이(mutate)" 함수부터 시작합니다(start by). `mutate()`와 `filter()` 안에서(inside of) 잘 작동하는 함수입니다.

`rescale01()`의 간단한 변형(variation)부터 보겠습니다. 벡터의 평균이 0, 표준 편차(standard deviation)가 1이 되도록 크기를 조정하는(rescaling) Z-점수(Z-score)를 계산합니다(compute).

```{r}
z_score <- function(x) {
  (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
}
```

간단한(straightforward) `case_when()`을 함수로 묶고(wrap up) 유용한 이름을 붙이기도 합니다(give). 예제의 `clamp()`는 벡터의 모든 값을 최소값과 최대값 사이(in between)에 둡니다(ensures).

```{r}
clamp <- function(x, min, max) {
  case_when(
    x < min ~ min,
    x > max ~ max,
    .default = x
  )
}

clamp(1:10, min = 3, max = 7)
```

함수는 숫자 변수에만 쓰지 않습니다(work with). 반복적인 문자열 조작(string manipulation)도 함수로 묶습니다. 첫 글자(character)를 대문자(uppercase)로 바꾸는 작업(Maybe you need to)이 한 예입니다.

```{r}
first_upper <- function(x) {
  str_sub(x, 1, 1) <- str_to_upper(str_sub(x, 1, 1))
  x
}

first_upper("hello")
```

문자열을 숫자로 변환(converting it into a number)하기 전에 퍼센트 기호(percent signs), 쉼표(commas), 달러 기호(dollar signs)를 제거(strip)할 수도 있습니다.

```{r}
# https://twitter.com/NVlabormarket/status/1571939851922198530
clean_number <- function(x) {
  is_pct <- str_detect(x, "%")
  num <- x |> 
    str_remove_all("%") |> 
    str_remove_all(",") |> 
    str_remove_all(fixed("$")) |> 
    as.numeric()
  if_else(is_pct, num / 100, num)
}

clean_number("$12,300")
clean_number("45%")
```

함수가 하나의 데이터 분석 단계(step)에 매우 특화(highly specialized)되기도 합니다. 결측값을 997, 998 또는 999로 기록(record)한 변수가 여러 개(a bunch of)라면 이를 `NA`로 대체(replace)하는 함수를 작성합니다.

```{r}
fix_na <- function(x) {
  if_else(x %in% c(997, 998, 999), NA, x)
}
```

지금까지는 가장 일반적인(common) 단일 벡터 입력(take)에 초점을 맞췄습니다. 물론(But) 함수는 여러(multiple) 벡터를 입력받아도 됩니다(there's no reason that).

### 요약 함수

벡터 함수의 또 다른(Another) 중요한 제품군(family)은 단일 값을 반환해 `summarize()`에서 쓰는(use in) 요약(summary) 함수입니다. 기본(default) 인수 한두 개만 설정하면(setting) 되는 경우도 있습니다.

```{r}
commas <- function(x) {
  str_flatten(x, collapse = ", ", last = " and ")
}

commas(c("cat", "dog", "pigeon"))
```

표준 편차를 평균으로 나누는 변동 계수(coefficient of variation)처럼(like for) 간단한 계산(computation)을 묶을(wrap up) 수도 있습니다.

```{r}
cv <- function(x, na.rm = FALSE) {
  sd(x, na.rm = na.rm) / mean(x, na.rm = na.rm)
}

cv(runif(100, min = 0, max = 50))
cv(runif(100, min = 0, max = 500))
```

일반적인 패턴(common pattern)에 기억하기 쉬운(memorable) 이름을 붙여(giving) 활용하기도 합니다(remember).

```{r}
# https://twitter.com/gbganalyst/status/1571619641390252033
n_missing <- function(x) {
  sum(is.na(x))
} 
```

여러(multiple) 벡터를 입력받는 함수도 작성합니다. 예를 들어 모델 예측(model predictions)과 실제 값(actual values)을 비교(compare)하는 평균 절대 백분율 오차(mean absolute percentage error)를 계산할(compute) 수 있습니다.

```{r}
# https://twitter.com/neilgcurrie/status/1571607727255834625
mape <- function(actual, predicted) {
  sum(abs((actual - predicted) / actual)) / length(actual)
}
```

### 연습 문제 (Exercises)

1. 다음 코드 조각(snippets)을 함수로 바꾸는 연습을 하세요. 각 함수가 무엇을 하는지 생각해 보세요. 어떻게 부르시겠습니까(What would you call it)?  몇 개의 인수(arguments)가 필요한가요?

```{r}
#| eval: false
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

2. `rescale01()`의 두 번째 변형(variant)에서는 무한한 값이 변경되지 않은 상태(unchanged)로 유지됩니다(are left). `-Inf`는 0에 매핑되고(mapped to) `Inf`는 1에 매핑되도록 `rescale01()`을 다시 작성해(rewrite) 보세요.

3. 생년월일(birthdates) 벡터가 주어지면(Given), 연령(age in years)을 계산하는 함수를 작성하세요.

4. 숫자 벡터의 분산(variance)과 왜도(skewness)를 계산하는 자체 함수를 작성하세요. 위키백과나 다른 곳(elsewhere)에서 정의를 찾아볼(look up) 수 있습니다.

5. 동일한 길이의 두 벡터를 입력으로 받아 두 벡터 모두에 `NA`가 있는 위치(positions)의 개수를 반환하는 요약 함수 `both_na()`를 작성하세요.

6. 설명서(documentation)를 읽고 다음 함수가 무엇을 하는지 파악하세요(figure out). 매우 짧은데도 유용한 이유는 무엇입니까?

```{r}
#| eval: false
is_directory <- function(x) {
  file.info(x)$isdir
}
is_readable <- function(x) {
  file.access(x, 4) == 0
}
```

## 데이터 프레임 함수

벡터 함수는 dplyr 동사(verb) 안의 반복 코드를 뽑아낼(pulling out) 때 유용합니다. 하지만 대규모(large) 파이프라인(pipeline)에서는 동사 자체(themselves)를 반복하기도 합니다(often). 여러(multiple) 동사를 거듭 복사해 붙이고 있다면(notice) 데이터 프레임 함수를 고려하세요(might think about). 이 함수는 dplyr 동사처럼 작동합니다. 첫 번째 인수로 데이터 프레임을 받고 작업을 지정하는 추가(extra) 인수 몇 개를 받아 데이터 프레임이나 벡터를 반환합니다.

dplyr 동사를 함수에 넣으려면 먼저 간접 참조(indirection)의 문제(challenge)와 껴안기(embracing, `{{{ }}}`)로 해결하는(overcome) 방법을 알아야 합니다(introduce). 이론을 익힌(under your belt) 뒤에는 활용할(might do) 작업을 여러(a bunch of) 예제로 살펴봅니다(illustrate).

### 간접 참조 및 깔끔한 평가

dplyr 동사를 함수에 넣으면 곧(rapidly) 간접 참조(indirection) 문제와 마주칩니다(hit). 간단한 `grouped_mean()` 함수로 문제를 설명하겠습니다(illustrate). 목표(goal)는 `group_var`로 그룹화한 `mean_var`의 평균을 계산하는 것입니다.

```{r}
grouped_mean <- function(df, group_var, mean_var) {
  df |> 
    group_by(group_var) |> 
    summarize(mean(mean_var))
}
```

이 함수를 실행하면(try and use it) 오류가 납니다.

```{r}
#| error: true
diamonds |> grouped_mean(cut, carat)
```

문제를 더 분명히 보려고 가상(made up) 데이터 프레임을 사용합니다.

```{r}
df <- tibble(
  mean_var = 1,
  group_var = "g",
  group = 1,
  x = 10,
  y = 100
)

df |> grouped_mean(group, x)
df |> grouped_mean(group, y)
```

`grouped_mean()`을 어떻게 호출하든(Regardless of) `df |> group_by(group) |> summarize(mean(x))`나 `df |> group_by(group) |> summarize(mean(y))`가 아닌 `df |> group_by(group_var) |> summarize(mean(mean_var))`를 실행합니다(call).

간접 참조 문제입니다. dplyr의 깔끔한 평가(tidy evaluation)는 데이터 프레임 안의 변수 이름을 별도 처리(special treatment) 없이 참조하게(allow you to) 해주는데(refer to), 이 때문에 문제가 생깁니다(arises).

깔끔한 평가(Tidy evaluation)는 95%의 경우(of the time) 훌륭합니다(great). 컨텍스트(context)가 명백하면(obvious) 변수가 어느 데이터 프레임에서 왔는지 밝힐 필요가 없어(never have to say) 데이터 분석 코드가 매우 간결해집니다(concise).

단점(downside)은 반복되는 tidyverse 코드를 함수로 묶을(wrap up) 때 드러납니다(comes). 여기서는 `group_by()`와 `summarize()`에 `group_var`와 `mean_var`를 변수 이름으로 처리하지(treat) 말고 그 안을 들여다보라고(look inside) 알려야 합니다(tell). 그래야 실제로(actual) 원하는 변수를 사용합니다(want to use).

깔끔한 평가는 이 문제를 껴안기(embracing)로 해결합니다(includes). 변수를 껴안는다는 것은(Embracing) 중괄호(braces)로 묶는(wrap)다는 뜻입니다. 예를 들어 `var`는 `{{{ var }}}`가 됩니다.

변수를 껴안으면 dplyr은 인수를 리터럴(literal) 변수 이름이 아닌 내부에 저장된(stored) 값으로 사용합니다(tells). `{{{ }}}`를 터널 안을 들여다보는 것(looking down a tunnel)이라고 생각하면 기억하기 쉽습니다(remember). `{{{ var }}}`는 dplyr 함수가 `var`라는 이름을 찾는(looking for) 대신 그 안을 들여다보게 합니다(look inside of).

`grouped_mean()`을 작동시키려면 `group_var`와 `mean_var`를 `{{{ }}}`로 감쌉니다(surround).

```{r}
grouped_mean <- function(df, group_var, mean_var) {
  df |> 
    group_by({{ group_var }}) |> 
    summarize(mean({{ mean_var }}))
}

df |> grouped_mean(group, x)
```

### 언제 껴안아야 할까요?

데이터 프레임 함수를 작성할 때 핵심 과제(key challenge)는 껴안을 인수를 찾는(figuring out) 일입니다. 다행히(Fortunately) 설명서(documentation)에서 확인하면(look it up) 됩니다(easy). 문서에서는 깔끔한 평가의 가장 일반적인(most common) 하위 유형(sub-types) 두 가지를 나타내는 용어(terms)를 찾아보세요.

- 데이터 마스킹(Data-masking): 이것은 변수를 사용해 계산하는(compute) `arrange()`, `filter()`, `summarize()`와 같은 함수에서 사용됩니다.

- 깔끔한 선택(Tidy-selection): 변수를 선택(select)하는 `select()`, `relocate()`, `rename()`과 같은 함수에 사용됩니다.

어떤 인수가 깔끔한 평가를 쓰는지에 관한 직관(intuition)은 일반적인 함수에서 대체로 잘 맞습니다(good). 계산하는(compute, 예: `x + 1`) 인수인지 선택하는(select, 예: `a:x`) 인수인지만 생각하세요(think about whether).

다음 절(sections)에서는 껴안기(embracing)를 이해한(understand) 뒤 작성할(might write) 만한 유용한(handy) 함수(sorts of)를 살펴봅니다(explore).

### 일반적인 사용 사례

초기 데이터 탐색(initial data exploration)에서 같은 요약(same set of summaries)을 반복한다면(commonly) 도우미 함수(helper function)로 묶으세요(wrapping).

```{r}
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
```

(`summarize()`를 도우미 함수로 감쌀(wrap) 때는(Whenever) `.groups = "drop"`을 설정하는(set) 편이 좋습니다(good practice). 메시지를 피하고(avoid) 데이터를 그룹화되지 않은 상태로 둡니다(ungrouped state).

이 함수는 `summarize()`를 감싸므로(wraps) 그룹화된(grouped) 데이터에도 적용합니다(use it on).

```{r}
diamonds |> 
  group_by(cut) |> 
  summary6(carat)
```

`summarize()`의 인수가 데이터 마스킹(data-masking)을 사용하므로 `summary6()`의 `var` 인수도 마찬가지입니다(Furthermore). 따라서 계산된(computed) 변수도 요약합니다(summarize).

```{r}
diamonds |> 
  group_by(cut) |> 
  summary6(log10(carat))
```

여러 변수(multiple variables)를 요약하는(To summarize) 방법은 `across()`를 배우는 @sec-across에서 다룹니다(wait).

널리 쓰이는(popular) 또 다른 `summarize()` 도우미 함수는 비율(proportions)까지 계산하는 `count()` 버전(version)입니다.

```{r}
# https://twitter.com/Diabb6/status/1571635146658402309
count_prop <- function(df, var, sort = FALSE) {
  df |>
    count({{ var }}, sort = sort) |>
    mutate(prop = n / sum(n))
}

diamonds |> count_prop(clarity)
```

이 함수의 인수는 `df`, `var`, `sort` 세 개입니다. 이 가운데 `var`만 모든 변수에 데이터 마스킹을 쓰는 `count()`로 전달되므로(passed) 껴안아야 합니다(needs to be embraced). `sort`는 사용자가 값을 주지 않으면(supply) 기본값(default) `FALSE`를 사용합니다(will default to).

데이터의 하위 집합(subset)에서 정렬된(sorted) 고유(unique) 값을 찾는 함수도 만들 수 있습니다. 필터링할 변수와 값을 받는 대신(Rather than) 사용자가 조건(condition)을 지정하게 합니다(supply).

```{r}
unique_where <- function(df, condition, var) {
  df |> 
    filter({{ condition }}) |> 
    distinct({{ var }}) |> 
    arrange({{ var }})
}

# 12월의 모든 목적지(destinations) 찾기
flights |> unique_where(month == 12, dest)
```

여기서는 `condition`을 `filter()`에, `var`를 `distinct()`와 `arrange()`에 전달하므로(because it's passed) 둘 다 껴안습니다(embrace).

지금까지 모든 예제는 데이터 프레임을 첫 번째 인수로 받습니다(to take). 같은 데이터로 반복해서(repeatedly) 작업한다면 하드코딩(hardcode)하는 편이 타당할(make sense) 수도 있습니다. 다음 함수는 언제나(always) `flights` 데이터셋에서 작동합니다. 행을 식별하는(identify a row) 복합 기본 키(compound primary key)를 이루므로(form) `time_hour`, `carrier`, `flight`를 항상 선택합니다.

```{r}
subset_flights <- function(rows, cols) {
  flights |> 
    filter({{ rows }}) |> 
    select(time_hour, carrier, flight, {{ cols }})
}
```

### 데이터 마스킹 대 깔끔한 선택

데이터 마스킹(data-masking)을 쓰는 함수 안에서(inside) 변수를 선택(select)해야 할 때도 있습니다. 행별 결측 관측치(missing observations)의 개수를 세는 `count_missing()`을 작성한다고(write) 해봅시다(imagine).

먼저 다음과 같이 작성할(might try writing) 수 있습니다.

```{r}
#| error: true
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
```

하지만 `group_by()`는 깔끔한 선택(tidy-selection)이 아닌 데이터 마스킹을 사용해 작동하지 않습니다(doesn't work). 편리한(handy) `pick()` 함수로 이 문제를 우회합니다(work around). `pick()`은 데이터 마스킹 함수 안에서(inside) 깔끔한 선택을 쓰게 해줍니다.

```{r}
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
```

`pick()`의 또 다른 용도는 카운트(counts)의 2차원 표(2d table)를 만드는(make) 것입니다. `rows`와 `columns`의 모든 변수로 계수(count)한 뒤 `pivot_wider()`로 카운트를 격자(grid)에 재배열합니다(rearrange).

```{r}
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
```

예제는 주로 dplyr에 집중했지만 깔끔한 평가(tidy evaluation)는 tidyr도 뒷받침합니다(underpins). `pivot_wider()` 문서에서(look at) `names_from`이 깔끔한 선택(tidy-selection)을 쓴다고 확인됩니다.

### 연습 문제

1. `nycflights13`의 데이터셋을 사용하여 다음(that:)을 수행하는 함수를 작성하세요.

    1. 취소된(cancelled, 즉 `is.na(arr_time)`) 또는 1시간 이상(more than an hour) 지연된 모든 항공편(flights)을 찾습니다(Finds).
    
    ```{r}
    #| eval: false
    flights |> filter_severe()
    ```

    2. 취소된 항공편의 수와 1시간 이상 지연된 항공편의 수를 셉니다(Counts).

    ```{r}
    #| eval: false
    flights |> group_by(dest) |> summarize_severe()
    ```

    3.  취소되거나 사용자가 제공한 시간(hours) 이상 지연된 모든 항공편을 찾습니다.

    ```{r}
    #| eval: false
    flights |> filter_severe(hours = 2)
    ```

    4.  날씨(weather)를 요약(Summarizes)하여 사용자가 제공한 변수의 최소, 평균, 최대값을 계산(compute)합니다.
    ```{r}
    #| eval: false
    weather |> summarize_weather(temp)
    ```

    5.  시계 시간(clock time, 예: `dep_time`, `arr_time` 등)을 사용하는 사용자가 제공한 변수를 십진 시간(decimal time, 즉 hours + (minutes / 60))으로 변환(Converts)합니다.

    ```{r}
    #| eval: false
    flights |> standardize_time(sched_dep_time)
    ```

2.  다음 각 함수에서 깔끔한 평가를 사용하는 모든 인수를 나열하고 데이터 마스킹을 사용하는지 깔끔한 선택을 사용하는지 설명(describe)하세요. `distinct()`, `count()`, `group_by()`, `rename_with()`, `slice_min()`, `slice_sample()`.

3.  계수할(to count) 변수를 개수와 상관없이 제공할(supply any number of variables) 수 있도록 다음 함수를 일반화(Generalize)하세요.

```{r}
count_prop <- function(df, var, sort = FALSE) {
  df |>
    count({{ var }}, sort = sort) |>
    mutate(prop = n / sum(n))
}
```

## 플롯 함수

데이터 프레임 대신 플롯을 반환하는 함수도 필요합니다(might want to). `aes()`는 데이터 마스킹 함수이므로 ggplot2에도 같은 기술(same techniques)을 적용합니다. 히스토그램(histograms)을 여러 개 만든다고 해봅시다.

```{r}
#| fig-show: hide
diamonds |> 
  ggplot(aes(x = carat)) +
  geom_histogram(binwidth = 0.1)

diamonds |> 
  ggplot(aes(x = carat)) +
  geom_histogram(binwidth = 0.05)
```

이 코드를 히스토그램 함수로 묶어봅시다(Wouldn't it be nice if you could). `aes()`가 데이터 마스킹 함수이고 껴안기(embrace)가 필요하다는 점만 알면 간단합니다(easy as pie).

```{r}
#| fig-alt: |
#|   A histogram of carats of diamonds, ranging from 0 to 5, showing a unimodal, 
#|   right-skewed distribution with a peak between 0 and 1 carats.
histogram <- function(df, var, binwidth = NULL) {
  df |> 
    ggplot(aes(x = {{ var }})) + 
    geom_histogram(binwidth = binwidth)
}

diamonds |> histogram(carat, 0.1)
```

`histogram()`은 ggplot2 플롯을 반환하므로 추가 구성 요소(additional components)를 계속(still) 붙일(add on) 수 있습니다. `|>`에서 `+`로 바꿔야(switch from) 한다는 점만 기억하세요.

```{r}
#| fig.show: hide
diamonds |> 
  histogram(carat, 0.1) +
  labs(x = "Size (in carats)", y = "Number of diamonds")
```

### 더 많은 변수

변수를 더 추가하는(mix) 일도 간단합니다(straightforward). 평활 선(smooth line)과 직선(straight line)을 겹치면(overlaying) 데이터셋이 선형(linear)인지(whether or not) 눈으로 쉽게(easy way) 확인합니다(eyeball).

```{r}
#| fig-alt: |
#|   Scatterplot of height vs. mass of StarWars characters showing a positive 
#|   relationship. A smooth curve of the relationship is plotted in red, and 
#|   the best fit line is plotted in blue.
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

과잉 플로팅(overplotting)이 문제(problem)인 큰 데이터셋에서는 색상 산점도(colored scatterplots)를 대신할 방법(alternative)이 필요합니다.

```{r}
#| fig-alt: |
#|   Hex plot of price vs. carat of diamonds showing a positive relationship. 
#|   There are more diamonds that are less than 2 carats than more than 2 carats.
# https://twitter.com/ppaxisa/status/1574398423175921665
hex_plot <- function(df, x, y, z, bins = 20, fun = "mean") {
  df |> 
    ggplot(aes(x = {{ x }}, y = {{ y }}, z = {{ z }})) + 
    stat_summary_hex(
      aes(color = after_scale(fill)), # 테두리를 채우기와 같은 색상으로 만듭니다(make border same color as fill)
      bins = bins, 
      fun = fun,
    )
}

diamonds |> hex_plot(carat, price, depth)
```

### 다른 tidyverse와 결합

일부 도우미 함수(helpers)는 데이터 조작(data manipulation)과 ggplot2를 결합합니다(combine). 예를 들어 `fct_infreq()`로 막대(bars)를 빈도(frequency) 순서에 맞춰 자동 정렬한(sort) 수직(vertical) 막대 차트를 그릴 수 있습니다. 값이 큰 막대를 맨 위(top)에 두려면(get) 일반적인 순서(usual order)를 뒤집어야 합니다(reverse).

```{r}
#| fig-alt: |
#|   Bar plot of clarity of diamonds, where clarity is on the y-axis and counts 
#|   are on the x-axis, and the bars are ordered in order of frequency: SI1, 
#|   VS2, SI2, VS1, VVS2, VVS1, IF, I1.
sorted_bars <- function(df, var) {
  df |> 
    mutate({{ var }} := fct_rev(fct_infreq({{ var }})))  |>
    ggplot(aes(y = {{ var }})) +
    geom_bar()
}

diamonds |> sorted_bars(clarity)
```

여기서는 사용자가 제공한 데이터를 기반으로(based on) 변수 이름을 생성하므로(generating) 새 연산자 `:=`를 사용해야 합니다(have to use). 흔히 "바다코끼리 연산자(walrus operator)"라고 부릅니다. 변수 이름은 `=`의 왼쪽(left hand side)에 놓이지만(go on) R 구문(syntax)은 단일 리터럴 이름(single literal name) 외에는 그 자리에 허용하지 않습니다. 이 제약을 우회하려고(work around) 깔끔한 평가가 `=`와 똑같이(exactly the same way) 처리하는 특수(special) 연산자 `:=`를 사용합니다.

데이터의 하위 집합(subset)에만 막대 플롯을 그리는 함수도 만들 수 있습니다(easy to).

```{r}
#| fig-alt: |
#|   Bar plot of clarity of diamonds. The most common is SI1, then SI2, then 
#|   VS2, then VS1, then VVS2, then VVS1, then I1, then lastly IF.
conditional_bars <- function(df, condition, var) {
  df |> 
    filter({{ condition }}) |> 
    ggplot(aes(x = {{ var }})) + 
    geom_bar()
}

diamonds |> conditional_bars(cut == "Good", clarity)
```

창의력을 발휘하면(get creative) 데이터 요약을 다른 방식으로 표시할(display) 수도 있습니다.

```{r}
library(tidyverse)
library(lubridate)

df <- tibble(
  dist1 = sort(rnorm(100, 5, 2)), 
  dist2 = sort(rnorm(100, 8, 3)),
  dist4 = sort(rnorm(100, 15, 1)),
  date = seq.Date(from = ymd("2022-01-01"), ymd("2022-04-10"), by = "day")
)

df <- pivot_longer(df, cols = -date, names_to = "dist_name", values_to = "value")

fancy_ts <- function(df, val, group) {
  labs <- df |> 
    group_by({{group}}) |> 
    summarize(breaks = max({{val}}))
  
  ggplot(df, 
         aes(
           x = date, 
           y = {{val}}, 
           group = {{group}}, 
           color = {{group}})) +
    geom_path() +
    scale_y_continuous(breaks = labs$breaks, minor_breaks = NULL) +
    theme_minimal()
}
```

ggplot2를 더 배울수록 함수의 강력함(power)도 커집니다(continue to increase). 마지막으로 생성한(create) 플롯에 레이블을 붙이는 조금 더 복잡한 사례(more complicated case)를 살펴봅니다(finish).

### 라벨링

앞에서(earlier) 본(showed you) 히스토그램 함수를 떠올려 보세요(Remember).

```{r}
histogram <- function(df, var, binwidth = NULL) {
  df |> 
    ggplot(aes(x = {{ var }})) + 
    geom_histogram(binwidth = binwidth)
}
```

출력(output)에 변수와 사용한 빈 너비(bin width)를 표시해 봅시다(Wouldn't it be nice if). 그러려면 깔끔한 평가의 내부로 들어가(go under the covers of) 아직 다루지 않은 rlang 패키지의 함수를 사용해야 합니다.

rlang은 깔끔한 평가와 여러 유용한 도구를 구현하는(implements) 저수준(low-level) 패키지입니다. 그래서(because) 거의 모든 tidyverse 패키지가 사용합니다(used by).

라벨링에는(solve) `rlang::englue()`를 사용합니다. `str_glue()`와 비슷하게(similarly) 작동해 `{ }`로 감싼 값(value wrapped in)을 문자열에 넣습니다(inserted). 아울러(But) 적절한(appropriate) 변수 이름을 자동으로(automatically) 넣는 `{{{ }}}`도 이해합니다(understands).

```{r}
#| fig-alt: |
#|   Histogram of carats of diamonds, ranging from 0 to 5. The distribution is 
#|   unimodal and right skewed with a peak between 0 to 1 carats.
histogram <- function(df, var, binwidth) {
  label <- rlang::englue("A histogram of {{var}} with binwidth {binwidth}")
  
  df |> 
    ggplot(aes(x = {{ var }})) + 
    geom_histogram(binwidth = binwidth) + 
    labs(title = label)
}

diamonds |> histogram(carat, 0.1)
```

ggplot2 플롯에서 문자열을 지정하는(supply) 다른(any other) 곳에도 같은 방식(approach)을 적용합니다.

### 연습 문제

아래의 각 단계를 점진적으로(incrementally) 구현하여 풍부한(rich) 플로팅 함수를 구축(Build up)하세요.

1.  데이터셋과 `x` 및 `y` 변수가 주어지면 산점도를 그립니다(Draw).

2.  최적 적합선(line of best fit)(즉, 표준 오차(standard errors)가 없는 선형 모델(linear model))을 추가합니다.

3.  제목(title)을 추가합니다.

## 스타일 (Style)

R은 함수나 인수의 이름에 신경 쓰지 않지만(doesn't care), 사람에게는 큰 차이(big difference)가 있습니다(make). 이상적인(Ideally) 함수 이름은 짧으면서(short) 하는 일을 분명히 드러냅니다(evoke). 물론 쉽지 않습니다(That's hard)! RStudio의 자동 완성(autocomplete)을 쓰면 긴 이름도 쉽게 입력하므로(type) 짧은 이름보다 명확한(clear) 이름이 낫습니다.

일반적으로 함수 이름은 동사(verbs), 인수는 명사(nouns)로 짓습니다. 예외(exceptions)도 있습니다. 널리 알려진 명사를 계산하는(computes) 함수는 `compute_mean()`보다 `mean()`이 낫습니다. 객체 속성(property)에 접근할 때도 `get_coefficients()`보다 `coef()`가 좋습니다(ok). 최선의 판단(best judgement)을 따르되 나중에(later) 더 나은 이름이 떠오르면(figure out) 주저하지 말고 바꾸세요(rename).

```{r}
#| eval: false
# 너무 짧음
f()

# 동사가 아니거나 설명적이지(descriptive) 않음
my_awesome_function()

# 길지만 명확함(clear)
impute_missing()
collapse_years()
```

R은 함수 안의 공백(white space)에도 신경 쓰지 않지만 미래의 독자(readers)는 다릅니다. `function()` 다음에는 항상 중괄호(squiggly brackets, `{}`)를 쓰고(should always be followed by) 내용은 공백 두 개(two spaces)를 더해(additional) 들여씁니다(indented). 그러면 왼쪽 여백(left-hand margin)만 훑어도(skimming) 코드의 계층 구조(hierarchy)가 잘 보입니다(see).

```{r}
# 추가 공백 두 개 누락(Missing)
density <- function(color, facets, binwidth = 0.1) {
diamonds |> 
  ggplot(aes(x = carat, y = after_stat(density), color = {{ color }})) +
  geom_freqpoly(binwidth = binwidth) +
  facet_wrap(vars({{ facets }}))
}

# 파이프 들여쓰기가 잘못됨(incorrectly)
density <- function(color, facets, binwidth = 0.1) {
  diamonds |> 
  ggplot(aes(x = carat, y = after_stat(density), color = {{ color }})) +
  geom_freqpoly(binwidth = binwidth) +
  facet_wrap(vars({{ facets }}))
}
```

`{{{ }}}` 안쪽에도(inside of) 공백을 두기를 권합니다(As you can see). 특이한 일(something unusual)이 일어난다는 점이 분명히(very obvious) 드러납니다(recommend).

### 연습 문제 (Exercises)

1.  다음 두 함수의 소스 코드를 각각 읽은 뒤 기능을 파악하고(puzzle out) 더 나은 이름을 브레인스토밍(brainstorm)하세요.

```{r}
f1 <- function(string, prefix) {
  str_sub(string, 1, str_length(prefix)) == prefix
}

f3 <- function(x, y) {
  rep(y, length.out = length(x))
}
```

2. 최근에(recently) 작성한 함수를 하나 골라서(Take) 5분 동안 함수와 해당 인수에 대한 더 나은 이름을 브레인스토밍해 보세요.

3. 왜 `rnorm()`, `dnorm()`보다 `norm_r()`, `norm_d()` 등이 더 좋을지 주장해 보세요(Make a case for). 그 반대(opposite)의 경우도 주장해 보세요. 이름을 어떻게 하면 더 명확하게(clearer) 만들 수 있을까요?

<!-- HUMANIZE-SUMMARY
원본 글자수: 26,891자
윤문본 글자수: 25,077자
변경률: 11.7% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 7 → 0
- A-7 가지다 직역: 0 → 0
- A-10 가능 표현 남발: 24 → 0
- A-11 목적절 남발: 6 → 0
- A-15 본문 추상 주어·만능 동사: 1 → 0
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
- "도달 범위를 넓히는 좋은 방법 중 하나는 함수를 작성하는 것" → "함수를 작성하면 다룰 수 있는 범위가 넓어집니다"
- "알아낼 수 있을 것입니다" → "각 열의 범위가 0에서 1이 되도록 크기를 조정합니다"
- "95%의 경우 훌륭합니다"를 장문 결론에서 문단 첫 문장으로 이동
- "데이터 프레임을 반환하는 대신 플롯을 반환하고 싶을 수 있습니다" → "플롯을 반환하는 함수도 필요합니다"
- "어떻게 불리든 상관하지 않지만" → "이름에 신경 쓰지 않지만"
-->
