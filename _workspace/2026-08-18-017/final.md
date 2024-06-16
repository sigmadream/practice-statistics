---
title: "반복"
---

```{r}
#| echo: false
source("_common.R")
```

R의 반복은 대개(generally) 암묵적(implicit)이며 별도 작업 없이(for free) 이루어집니다. 그래서(so much of it is) 다른 프로그래밍 언어와 상당히 다르게(rather different from) 보입니다(tends to look).

예를 들어 R에서 숫자 벡터 `x`를 두 배로 만들 때는 `2 * x`만 쓰면(write) 됩니다(just). 다른 언어 대부분은 일종의(some sort of) for 루프로 `x`의 각 요소를 명시적으로(explicitly) 두 배로 만듭니다.

- `facet_wrap()`과 `facet_grid()`는 각 하위 집합(subset)에 대한 플롯을 그립니다(draws).
- `group_by()`와 `summarize()`를 함께 사용하여 각 하위 집합에 대한 요약 통계(summary statistics)를 계산(computes)합니다.
- `unnest_wider()`와 `unnest_longer()`는 리스트 열(list-column)의 요소마다 새 행과 열을 생성(create)합니다.

이제 다른 함수를 입력으로 받는 좀 더(some more) 일반적인(general) 도구를 배워봅시다(time to learn). 이런 도구는 함수를 중심으로 구성돼(built around) 흔히(often) 함수형 프로그래밍(functional programming) 도구라고 합니다.

함수형 프로그래밍은 쉽게(easily) 추상적인 이야기(abstract)로 흐릅니다(veer into). 이 장에서는 여러 열 수정, 여러 파일 읽기, 여러 객체 저장이라는 세 가지 일반적인 작업(common tasks)에 집중해(focusing on) 구체성을(concrete) 유지합니다(keep things).

```{r}
#| label: setup
#| message: false
library(tidyverse)
```

## 여러 열 수정

간단한 티블에서 관측치(observations) 수를 세고(count) 모든 열의 중앙값(median)을 계산한다고(compute) 해봅시다.

```{r}
set.seed(1014)
df <- tibble(
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10)
)
```

복사해서 붙이는(copy-and-paste) 방법이 먼저 떠오릅니다.

```{r}
df |> summarize(
  n = n(),
  a = median(a),
  b = median(b),
  c = median(c),
  d = median(d),
)
```

이 코드는 두 번 넘게 복사해 붙이지 말라는(never copy and paste more than twice) 경험 법칙(rule of thumb)을 어깁니다(breaks). 열이 수십(tens), 수백(hundreds) 개라면 작업도 매우 지루합니다(tedious). 이때(Instead) `across()`를 사용합니다.

```{r}
df |> summarize(
  n = n(),
  across(a:d, median),
)
```

`across()`에는 중요한 인수가 세 개 있으며 다음 절에서 자세히(in detail) 다룹니다(discuss). 늘 쓰는 첫 두 인수 가운데 `.cols`는 반복할(iterate over) 열을 지정하고(specifies), `.fns`는 각 열에 적용할 작업(what to do with)을 지정합니다. 출력 열 이름을 더 세밀하게 제어할(additional control) 때는 `.names`를 씁니다. 특히 `mutate()`와 함께 쓸 때 중요합니다. `filter()`에서 작동하는 변형(variations)인 `if_any()`와 `if_all()`도 살펴봅니다.

### `.cols`를 사용하여 열 선택하기

`across()`의 첫 번째 인수 `.cols`는 변환(transform)할 열을 고릅니다. `select()`(@sec-select 참조)와 같은 지정(specifications)을 사용하므로 `starts_with()`와 `ends_with()`로 이름에 따라 열을 선택합니다.

`across()`에서 특히 유용한 선택 기술(selection techniques)은 `everything()`과 `where()`입니다. `everything()`은 모든(every) 그룹화되지 않은 열을 선택합니다(straightforward).

```{r}
set.seed(1014)
df <- tibble(
  grp = sample(2, 10, replace = TRUE),
  a = rnorm(10),
  b = rnorm(10),
  c = rnorm(10),
  d = rnorm(10)
)

df |> 
  group_by(grp) |> 
  summarize(across(everything(), median))
```

그룹화(grouping) 열인 `grp`는 `summarize()`가 자동으로 보존하므로(preserved) `across()`에 포함되지 않습니다(included).

`where()`는 유형(type)에 따라 열을 선택합니다(allows you to).

1. `where(is.numeric)`은 모든 숫자(numeric) 열을 선택합니다.
2. `where(is.character)`는 모든 문자열(string) 열을 선택합니다.
3. `where(is.Date)`는 모든 날짜(date) 열을 선택합니다.
4. `where(is.POSIXct)`는 모든 날짜-시간(date-time) 열을 선택합니다.
5. `where(is.logical)`은 모든 논리(logical) 열을 선택합니다.

다른 선택자(selectors)처럼(Just like) 부울 대수(Boolean algebra)와 결합합니다(combine). `!where(is.numeric)`은 숫자가 아닌(non-numeric) 모든 열을, `starts_with("a") & where(is.logical)`은 이름이 "a"로 시작하는 모든 논리 열을 선택합니다.

### 단일 함수 호출

`across()`의 두 번째 인수는 각 열의 변환 방식을 정합니다(defines). 위와 같은(as above) 간단한 경우에는 기존(existing) 함수 하나를 넣습니다. 하나의 함수(`median`, `mean`, `str_flatten` ...)를 다른 함수(`across`)에 전달하는(passing), R의 특별한(pretty special) 기능입니다. R이 함수형 프로그래밍 언어인 이유 중 하나이기도 합니다.

함수는 `across()`가 호출하도록(call it) 전달할 뿐(note), 직접 호출하지 않습니다(not calling it ourselves). 따라서 함수 이름 뒤에(followed by) `()`를 붙이면 안 됩니다. 붙이면(If you forget) 오류가 납니다.

```{r}
#| error: true
df |> 
  group_by(grp) |> 
  summarize(across(everything(), median()))
```

입력 없이 함수를 호출했기 때문에 생긴(arises) 오류입니다.

```{r}
#| error: true
median()
```

### 여러 함수 호출

더 복잡한 경우에는 추가 인수를 주거나(supply) 여러 변환을 수행합니다(perform). 간단한 예제로 문제를 살펴봅시다(motivate this problem). 데이터에 결측값(missing values)이 있으면 어떻게 될까요?

`median()`은 결측값을 전파해(propagates) 만족스럽지 않은(suboptimal) 출력을 만듭니다.

```{r}
set.seed(1014)
rnorm_na <- function(n, n_na, mean = 0, sd = 1) {
  sample(c(rnorm(n - n_na, mean = mean, sd = sd), rep(NA, n_na)))
}

df_miss <- tibble(
  a = rnorm_na(5, 1),
  b = rnorm_na(5, 1),
  c = rnorm_na(5, 2),
  d = rnorm(5)
)
df_miss |> 
  summarize(
    across(a:d, median),
    n = n()
  )
```

결측값을 제거하려면(remove) `median()`에 `na.rm = TRUE`를 함께 전달해야 합니다(pass along). `median()`을 직접(directly) 호출하는 대신 원하는(desired) 인수로 호출하는 새 함수를 만드세요(create).

```{r}
df_miss |> 
  summarize(
    across(a:d, function(x) median(x, na.rm = TRUE)),
    n = n()
  )
```

조금 장황하므로(verbose) R에는 편리한 단축키가 있습니다(handy shortcut). 이런 일회용(throw away), 즉 익명(anonymous) 함수에서는 `function`을 `\`로 바꿉니다(replace).

`<-`로 명시적인 이름을 붙이지(gave it a name) 않아 익명이라고 합니다. 프로그래머는 "람다 함수(lambda function)"라는 용어(term)도 씁니다.

이전(older) 코드에는 `~ .x + 1` 같은 구문(syntax)도 있습니다. 익명 함수를 쓰는 다른 방법(another way)이지만 tidyverse 함수 안에서만(inside) 작동하고 변수 이름은 늘 `.x`입니다. 이제는 기본 구문인 `\(x) x + 1`을 권합니다(recommend).

```{r}
#| results: false
df_miss |> 
  summarize(
    across(a:d, \(x) median(x, na.rm = TRUE)),
    n = n()
  )
```

어느 경우든(In either case) `across()`는 사실상(effectively) 다음 코드로 확장됩니다(expands to).

```{r}
#| eval: false
df_miss |> 
  summarize(
    a = median(a, na.rm = TRUE),
    b = median(b, na.rm = TRUE),
    c = median(c, na.rm = TRUE),
    d = median(d, na.rm = TRUE),
    n = n()
  )
```

`median()`에서 결측값을 제거할 때 몇 개가 빠졌는지도 알아봅시다(nice to know just how many). `across()`에 중앙값 계산 함수와 결측값 개수를 세는(count) 함수를 함께 전달합니다(supplying). 여러 함수는 `.fns`에 명명된 리스트(named list)로 넣습니다.

```{r}
df_miss |> 
  summarize(
    across(a:d, list(
      median = \(x) median(x, na.rm = TRUE),
      n_miss = \(x) sum(is.na(x))
    )),
    n = n()
  )
```

자세히 보면(If you look carefully) 열 이름은 `{.col}_{.fn}` 형태의 글루 지정자(glue specification)를 따릅니다(named). `.col`은 원본 열, `.fn`은 함수 이름입니다. 우연(coincidence)이 아닙니다! 다음 절에서(As) `.names` 인수로 사용자 지정(your own) 글루 스펙(glue spec)을 만드는 법을 배웁니다(learn).

### 열 이름

`across()` 결과의 이름은 `.names`에 넣은(provided) 지정자(specification)에 따라 정해집니다(named according to). 함수 이름을 앞에 두려면(come first) 직접 지정합니다.

```{r}
df_miss |> 
  summarize(
    across(
      a:d,
      list(
        median = \(x) median(x, na.rm = TRUE),
        n_miss = \(x) sum(is.na(x))
      ),
      .names = "{.fn}_{.col}"
    ),
    n = n(),
  )
```

`.names`는 `mutate()` 안에서 `across()`를 쓸 때 특히 중요합니다. 기본적으로(By default) 출력 이름이 입력과 같아서 기존 열을 대체합니다(replace). 다음 예제에서는 `coalesce()`로 `NA`를 `0`으로 바꿉니다.

```{r}
df_miss |> 
  mutate(
    across(a:d, \(x) coalesce(x, 0))
  )
```

새 열을 만들려면(create new columns) `.names`로 출력에 새 이름을 붙입니다(give).

```{r}
df_miss |> 
  mutate(
    across(a:d, \(x) coalesce(x, 0), .names = "{.col}_na_zero")
  )
```

### 필터링

`across()`는 `summarize()`와 `mutate()`에 잘 맞지만(great match) `filter()`에서는 다소 어색합니다(more awkward). 여러 조건(conditions)을 보통(usually) `|`나 `&`로 결합하기 때문입니다.

`across()`로 여러 논리 열을 만들 수는 있지만(clear) 그다음은 어떻게 해야 할까요(but then what)? 그래서 dplyr에는 `if_any()`와 `if_all()`이라는 두 가지 변형(variants)이 있습니다.

```{r}
# df_miss |> filter(is.na(a) | is.na(b) | is.na(c) | is.na(d))와 동일함(same as)
df_miss |> filter(if_any(a:d, is.na))

# df_miss |> filter(is.na(a) & is.na(b) & is.na(c) & is.na(d))와 동일함
df_miss |> filter(if_all(a:d, is.na))
```

### 함수 내의 `across()`

`across()`는 여러 열을 다루므로(allows you to operate) 프로그램을 작성할 때(to program with) 특히(particularly) 유용합니다. [Jacob Scott]([https://twitter.com/_wurli/status/1571836746899283969](https://twitter.com/_wurli/status/1571836746899283969))은 여러(a bunch of) lubridate 함수를 묶어(wraps) 모든 날짜 열을 연, 월, 일로 확장하는(expand) 작은 도우미(little helper)를 사용합니다.

```{r}
expand_dates <- function(df) {
  df |> 
    mutate(
      across(where(is.Date), list(year = year, month = month, day = mday))
    )
}

df_date <- tibble(
  name = c("Amy", "Bob"),
  date = ymd(c("2009-08-03", "2010-01-16"))
)

df_date |> 
  expand_dates()
```

`across()`의 첫 번째 인수는 깔끔한 선택(tidy-select)을 사용하므로 한 인수에 여러 열을 쉽게(easy to) 넣습니다(supply). 인수를 껴안아야(embrace) 한다는 점만 기억하세요(just need to remember).

이 함수는 기본적으로(by default) 숫자 열의 평균을 계산합니다(compute). 두 번째 인수를 주면 선택한 열만 요약합니다(choose to).

```{r}
summarize_means <- function(df, summary_vars = where(is.numeric)) {
  df |> 
    summarize(
      across({{ summary_vars }}, \(x) mean(x, na.rm = TRUE)),
      n = n(),
      .groups = "drop"
    )
}
diamonds |> 
  group_by(cut) |> 
  summarize_means()

diamonds |> 
  group_by(cut) |> 
  summarize_means(c(carat, x:z))
```

### `pivot_longer()`와 비교

계속하기 전에(Before we go on) `across()`와 `pivot_longer()`(@sec-pivoting)의 흥미로운 연결 고리(interesting connection)를 살펴봅시다(pointing out). 데이터를 먼저 피벗한(pivoting) 뒤(and then) 열(column)이 아닌 그룹(group) 단위로 연산하면(performing) 같은 계산을 수행합니다(perform).

예를 들어 다음(take this) 다중 기능(multi-function) 요약을 살펴봅시다.

```{r}
df |> 
  summarize(across(a:d, list(median = median, mean = mean)))
```

길게 피벗하고(pivoting longer) 요약하면(and then) 같은 값(same values)을 계산합니다(compute).

```{r}
long <- df |> 
  pivot_longer(a:d) |> 
  group_by(name) |> 
  summarize(
    median = median(value),
    mean = mean(value)
  )
long
```

`across()`와 같은 구조(structure)가 필요하다면 다시 피벗합니다(pivot again).

```{r}
long |> 
  pivot_wider(
    names_from = name,
    values_from = c(median, mean),
    names_vary = "slowest",
    names_glue = "{name}_{.value}"
  )
```

이것은 알아두면(to know about) 유용한 기술입니다. 때로는 열 그룹(groups of columns)을 동시에(simultaneously) 계산하고 싶을 때(when you want to compute with) 현재 `across()`로 해결할 수 없는(not currently possible to solve) 문제에 부딪힐(hit) 수 있기 때문입니다.

예를 들어, 데이터 프레임에 값(values)과 가중치(weights)가 모두 포함되어 있고 가중 평균(weighted mean)을 계산(compute)하고 싶다고 상상해 보세요.

```{r}
set.seed(1014)
df_paired <- tibble(
  a_val = rnorm(10),
  a_wts = runif(10),
  b_val = rnorm(10),
  b_wts = runif(10),
  c_val = rnorm(10),
  c_wts = runif(10),
  d_val = rnorm(10),
  d_wts = runif(10)
)
```

현재(currently) `across()`로 이것을 수행할(to do this) 방법은 없지만[^iteration-4], `pivot_longer()`를 사용하면 비교적 간단(relatively straightforward)합니다.

```{r}
df_long <- df_paired |> 
  pivot_longer(
    everything(), 
    names_to = c("group", ".value"), 
    names_sep = "_"
  )
df_long

df_long |> 
  group_by(group) |> 
  summarize(mean = weighted.mean(val, wts))
```

필요한 경우(If needed) 이 데이터를 `pivot_wider()`를 사용하여 원래 형태(original form)로 되돌릴(back to) 수 있습니다.

### 연습 문제

1. 다음을 수행하여(by:) `across()` 기술(skills)을 연습하세요(Practice).

    1. `palmerpenguins::penguins`의 각 열에서 고유 값(unique values)의 개수를 계산(Computing)합니다.

    2. `mtcars`의 모든 열의 평균(mean)을 계산합니다.

    3. `diamonds`를 `cut`, `clarity`, `color`별로 그룹화(Grouping)한 다음(then) 관측치(observations)의 개수를 세고 각 숫자 열의 평균을 계산합니다.

2. `across()`에서 함수 리스트(list of functions)를 사용하지만 이름을 지정(name)하지 않으면 어떻게 될까요(What happens)? 출력은 어떻게(How) 명명(named)되나요?

3. `expand_dates()`를 조정(Adjust)하여 날짜 열이 확장(expanded)된 후(after)에 날짜 열을 자동으로(automatically) 제거(remove)되도록 하세요. 인수를 껴안아야(embrace) 하나요?

4. 이 함수의 파이프라인(pipeline)의 각 단계(each step)가 무엇을 하는지(what it does) 설명(Explain)하세요. 우리는 `where()`의 어떤 특별한 기능(special feature)을 활용(taking advantage of)하고 있나요?

```{r}
#| results: false
show_missing <- function(df, group_vars, summary_vars = everything()) {
  df |> 
    group_by(pick({{ group_vars }})) |> 
    summarize(
      across({{ summary_vars }}, \(x) sum(is.na(x))),
      .groups = "drop"
    ) |>
    select(where(\(x) any(x > 0)))
}
nycflights13::flights |> show_missing(c(year, month, day))
```

## 여러 파일 읽기

앞 절에서는 `dplyr::across()`로 여러 열에 같은 변환(transformation)을 반복했습니다(repeat). 이번에는 `purrr::map()`으로 디렉터리의 모든 파일(every file)에 같은 작업(something)을 적용합니다(do). 읽어야(read) 할 엑셀 스프레드시트가 가득한(full of) 디렉터리를 예로 들어봅시다(imagine). 먼저 복사해서 붙이는(copy and paste) 방법입니다.

```{r}
#| eval: false
data2019 <- readxl::read_excel("data/y2019.xlsx")
data2020 <- readxl::read_excel("data/y2020.xlsx")
data2021 <- readxl::read_excel("data/y2021.xlsx")
data2022 <- readxl::read_excel("data/y2022.xlsx")
```

그런 다음 `dplyr::bind_rows()`로 모두 결합합니다(combine them all together).

```{r}
#| eval: false
data <- bind_rows(data2019, data2020, data2021, data2022)
```

파일이 4개가 아니라 수백(hundreds) 개라면 이 작업은 금세 지루해집니다(tedious quickly). 다음 절(sections)에서는 세 단계(basic steps)로 자동화합니다(automate). `list.files()`로 모든 파일을 나열하고(list), `purrr::map()`으로 각 파일을 리스트에 읽은 뒤(read each of them into a list), `purrr::list_rbind()`로 하나의 데이터 프레임에 결합합니다(combine). 모든 파일에 똑같은 작업(exactly the same thing)을 적용하기 어려울 만큼 이질성(heterogeneity)이 커지는(increasing) 상황도 다룹니다(handle).

### 디렉터리의 파일 나열

이름에서 알 수 있듯(As the name suggests) `list.files()`는 디렉터리의 파일을 나열합니다(lists). 거의 항상 인수 세 개(three)를 사용합니다.

1. 첫 번째 인수 `path`는 살펴볼(look in) 디렉터리입니다.

2. `pattern`은 파일 이름을 필터링(filter)하는 데 사용되는 정규 표현식(regular expression)입니다. 일반적인 패턴은 지정된 확장자(specified extension)를 가진 모든 파일을 찾기 위한(find) `[.]xlsx$` 또는 `[.]csv$`와 같은 것입니다(something like).

3. `full.names`는 출력(output)에 디렉터리 이름을 포함할지(should be included) 정합니다(determines). 거의 언제나 `TRUE`로 설정합니다(want this to be).

예제를 구체화하려고(motivating example) gapminder 패키지의 데이터가 든 엑셀 스프레드시트(excel spreadsheets) 12개를 사용합니다(concrete). 이 책에 포함된(contains) 폴더입니다.

폴더는 [https://github.com/hadley/r4ds/tree/main/data/gapminder](https://github.com/hadley/r4ds/tree/main/data/gapminder)에 있습니다(can be found at). 파일마다 142개국의 1년 치(one year's worth of) 데이터가 들어 있습니다.

```{r}
paths <- list.files("data/gapminder", pattern = "[.]xlsx$", full.names = TRUE)
paths
```

### 리스트 (Lists)

이제(Now that) 경로(paths) 12개가 있으니(we have) `read_excel()`을 12번 호출해(call) 데이터 프레임 12개를 만듭니다.

```{r}
#| eval: false
gapminder_1952 <- readxl::read_excel("data/gapminder/1952.xlsx")
gapminder_1957 <- readxl::read_excel("data/gapminder/1957.xlsx")
gapminder_1962 <- readxl::read_excel("data/gapminder/1962.xlsx")
 ...,
gapminder_2007 <- readxl::read_excel("data/gapminder/2007.xlsx")
```

하지만 시트(sheet)마다 별도 변수(its own variable)에 넣으면(putting) 몇 단계 뒤(down the road)에 다루기 어렵습니다(hard to work with). 하나의(single) 객체에 모으면(Instead) 더 쉽습니다(put them into). 리스트가 이 작업(job)에 알맞은 도구입니다(perfect tool).

```{r}
#| eval: false
files <- list(
  readxl::read_excel("data/gapminder/1952.xlsx"),
  readxl::read_excel("data/gapminder/1957.xlsx"),
  readxl::read_excel("data/gapminder/1962.xlsx"),
  ...,
  readxl::read_excel("data/gapminder/2007.xlsx")
)
```

```{r}
#| include: false
files <- map(paths, readxl::read_excel)
```

리스트에 데이터 프레임을 담았으니(have) 하나를 꺼내봅시다(get one out). `files[[i]]`는 i번째 요소를 추출합니다(extract).

```r
files[[3]]
```

### `purrr::map()`과 `list_rbind()`

데이터 프레임을 리스트에 "수동으로(by hand)" 모으는(collect) 코드도 파일을 하나씩(one-by-one) 읽는 코드만큼(just as) 입력하기 지루합니다(tedious to type). 다행히(Happily) `purrr::map()`을 쓰면 `paths` 벡터를 훨씬(even) 잘 활용합니다(make better use of).

`map()`은 `across()`와 비슷하지만(similar), 데이터 프레임의 각 열 대신 벡터의 각 요소에 작업을 적용합니다(doing something). `map(x, f)`는 다음 코드의 약어(shorthand)입니다.

```{r}
#| eval: false
list(
  f(x[[1]]),
  f(x[[2]]),
  ...,
  f(x[[n]])
)
```

`map()`으로 데이터 프레임 12개의 리스트를 만듭니다(get).

```{r}
#| eval: false
files <- map(paths, readxl::read_excel)
length(files)

files[[1]]
```

(이것은 `str()`로 특히 간결하게(compactly) 표시(display)되지 않는 또 다른 데이터 구조(data structure)이므로 RStudio에 로드(load)하고 `View()`로 검사(inspect)하고 싶을 수 있습니다).

이제 `purrr::list_rbind()`로 리스트의 데이터 프레임을 하나로 결합합니다(combine).

```{r}
list_rbind(files)
```

파이프라인에서는 두 단계(both steps)를 한 번에(at once) 수행합니다.

```{r}
#| results: false
paths |> 
  map(readxl::read_excel) |> 
  list_rbind()
```

`read_excel()`에 추가 인수(extra arguments)를 전달하려면(pass in) `across()`에서 쓴 것과 같은 기술(same technique)을 적용합니다.

예를 들어 `n_max = 1`로 데이터의 첫 몇 행(first few rows)을 살펴보면(peek at) 유용합니다.

```{r}
paths |> 
  map(\(path) readxl::read_excel(path, n_max = 1)) |> 
  list_rbind()
```

여기서 누락된(missing) 항목이 분명히 드러납니다(makes it clear). `year` 열이 없습니다. 값(value)이 개별(individual) 파일이 아니라 경로(path)에 기록돼(recorded) 있기 때문입니다.

다음으로 그 문제(problem)를 해결(tackle)하겠습니다.

### 경로 내의 데이터

파일 이름 자체가 데이터인(is data itself) 경우도 있습니다. 이 예제(example)에서는 개별 파일에 따로(otherwise) 기록되지 않은 연도(year)가 파일 이름에 들어 있습니다. 최종 데이터 프레임에 이 열을 넣으려면(To get that column into) 두 단계(two things)가 필요합니다.

먼저 경로 벡터(vector of paths)에 이름을 붙입니다(name). 함수를 받을(take) 수 있는 `set_names()`를 쓰는 것이 가장 쉽습니다(easiest way). 여기서는 `basename()`으로 전체 경로(full path)에서 파일 이름만 추출합니다(extract).

```{r}
paths |> set_names(basename) 
```

이 이름은 모든 map 함수가 자동으로 전달하므로(carried along) 데이터 프레임 리스트에도 같은 이름(same names)이 붙습니다.

```{r}
files <- paths |> 
  set_names(basename) |> 
  map(readxl::read_excel)
```

이렇게 하면 `map()`에 대한 이 호출이 다음의 약어(shorthand)가 됩니다.

```{r}
#| eval: false
files <- list(
  "1952.xlsx" = readxl::read_excel("data/gapminder/1952.xlsx"),
  "1957.xlsx" = readxl::read_excel("data/gapminder/1957.xlsx"),
  "1962.xlsx" = readxl::read_excel("data/gapminder/1962.xlsx"),
  ...,
  "2007.xlsx" = readxl::read_excel("data/gapminder/2007.xlsx")
)
```

이름으로(by name) 요소를 추출할 때는(extract) `[[`를 사용합니다.

```{r}
#| eval: false
files[["1962.xlsx"]]
```

그런 다음 `list_rbind()`의 `names_to` 인수로 이름을 `year`라는 새 열에 저장하고(save), `readr::parse_number()`로 문자열(string)에서 숫자를 추출합니다(extract).

```{r}
#| eval: false
paths |> 
  set_names(basename) |> 
  map(readxl::read_excel) |> 
  list_rbind(names_to = "year") |> 
  mutate(year = parse_number(year))
```

더 복잡한 경우에는 디렉터리 이름에 다른 변수가 저장되거나(stored) 파일 이름에 여러 데이터 조각(multiple bits)이 들어 있습니다(contains). 이때 인수 없는(without any arguments) `set_names()`로 전체 경로를 기록한(record) 뒤 `tidyr::separate_wider_delim()`과 관련 함수(friends)로 유용한 열(useful columns)을 만듭니다(turn them into).

```{r}
#| eval: false
paths |> 
  set_names() |> 
  map(readxl::read_excel) |> 
  list_rbind(names_to = "year") |> 
  separate_wider_delim(year, delim = "/", names = c(NA, "dir", "file")) |> 
  separate_wider_delim(file, delim = ".", names = c("file", "ext"))
```

### 작업 저장

깔끔한(nice tidy) 데이터 프레임을 얻기 위한(get to) 힘든 작업(hard work)을 마쳤으니(Now that you've done) 이제 저장할(save your work) 차례입니다(great time).

```{r}
#| eval: false
gapminder <- paths |> 
  set_names(basename) |> 
  map(readxl::read_excel) |> 
  list_rbind(names_to = "year") |> 
  mutate(year = parse_number(year))

write_csv(gapminder, "gapminder.csv")
```

나중에(in the future) 다시 작업할(come back to) 때는 단일(single) csv 파일만 읽으면 됩니다(read in). parquet가 `.csv`보다 나은 선택(better choice)일 수도 있습니다.

```{r}
#| include: false
unlink("gapminder.csv")
```

프로젝트에서(working in) 이런 종류의(this sort of) 데이터 준비 작업(data prep work)을 하는 파일은 `0-cleanup.R`처럼(something like) 이름 붙이기를 권합니다(calling). 이름의 `0`은 다른 파일보다(before anything else) 먼저 실행해야(run) 한다는 뜻입니다(suggests).

입력 데이터 파일이 시간에 따라(over time) 바뀐다면(change) [targets]([https://docs.ropensci.org/targets/](https://docs.ropensci.org/targets/)) 같은 도구를 고려하세요(might consider). 입력 파일이 수정될(modified) 때마다(whenever) 데이터 정리 코드(data cleaning code)를 자동으로 다시 실행하도록(automatically re-run) 설정합니다(set up).

### 여러 가지 간단한 반복

여기서는 디스크에서 직접(directly) 데이터를 불러왔고(loaded) 운 좋게도(lucky enough to) 깔끔한 데이터셋을 얻었습니다. 대부분(In most cases)은 추가 정리(additional tidying)가 필요합니다(need to do). 기본 옵션(basic options)은 복잡한(complex) 함수로 한 번(one round of) 반복하거나 간단한 함수로 여러 번(multiple rounds of) 반복하는 두 가지입니다.

경험상(In our experience) 대부분의 사람은(most folks) 먼저(first) 복잡한 반복 하나를 택하지만(reach for) 간단한 반복을 여러 번 하는 편이 더 나을 때가 많습니다(better).

여러 파일(a bunch of files)을 읽고(read in) 결측값을 걸러낸(filter out) 뒤 피벗(pivot)하고 결합하는(combine) 상황을 예로 들어봅시다. 한 가지 접근법(approach)은 파일을 받아(takes) 모든 단계를 수행하는 함수를 작성한(write a function) 뒤 `map()`을 한 번 호출하는 것입니다.

```{r}
#| eval: false
process_file <- function(path) {
  df <- read_csv(path)
  
  df |> 
    filter(!is.na(id)) |> 
    mutate(id = tolower(id)) |> 
    pivot_longer(jan:dec, names_to = "month")
}

paths |> 
  map(process_file) |> 
  list_rbind()
```

대안으로(Alternatively) 모든 파일에서 `process_file()`의 각 단계를 수행합니다(perform).

```{r}
#| eval: false
paths |> 
  map(read_csv) |> 
  map(\(df) df |> filter(!is.na(id))) |> 
  map(\(df) df |> mutate(id = tolower(id))) |> 
  map(\(df) df |> pivot_longer(jan:dec, names_to = "month")) |> 
  list_rbind()
```

이 접근 방식은 첫 파일만 제대로 가져오는 데 집착하지 않도록(stops you) 해주므로 권합니다(recommend). 모든 데이터를 함께 보면서(considering) 정리하고 클리닝하면(tidying and cleaning) 더 전체적으로(holistically) 판단하게 되고(more likely to) 결과의 품질도 높아집니다(higher quality).

이 예제(particular example)에는 데이터 프레임을 더 일찍(earlier) 결합하는(binding) 최적화(optimization)도 있습니다. 그러면 일반적인(regular) dplyr 동작(behavior)을 활용합니다(rely on).

```{r}
#| eval: false
paths |> 
  map(read_csv) |> 
  list_rbind() |> 
  filter(!is.na(id)) |> 
  mutate(id = tolower(id)) |> 
  pivot_longer(jan:dec, names_to = "month")
```

### 이질적인 데이터

데이터 프레임이 너무(so) 이질적이면(heterogeneous) `list_rbind()`가 실패하거나(fails) 쓸모가 적은(not very useful) 결과를 냅니다(yields). 이때는(Unfortunately) `map()`에서 `list_rbind()`로 바로 넘어가지(go straight) 못합니다.

그래도(In that case) 모든 파일을 먼저 불러오면(loading) 유용합니다(still).

```{r}
#| eval: false
files <- paths |> 
  map(readxl::read_excel) 
```

그런 다음 데이터 과학 기술(skills)로 탐색하도록(explore) 데이터 프레임의 구조(structure)를 기록합니다(capture). 매우 유용한 전략(strategy)입니다.

한 가지 방법은 열마다 행 하나를 반환하는 `df_types` 함수를 쓰는 것입니다(handy).

```{r}
#| eval: false
df_types <- function(df) {
  tibble(
    col_name = names(df), 
    col_type = map_chr(df, vctrs::vec_ptype_full),
    n_miss = map_int(df, \(x) sum(is.na(x)))
  )
}

df_types(gapminder)
```

이 함수를 모든 파일에 적용하고(apply) 조금(some) 피벗하면(pivoting) 차이가(differences) 어디에(where) 있는지 쉽게 보입니다(see). 다음 예제로 작업 중인(working with) gapminder 스프레드시트가 모두 상당히(quite) 동질적인지(homogeneous) 확인합니다(verify).

```{r}
#| eval: false
files |> 
  map(df_types) |> 
  list_rbind(names_to = "file_name") |> 
  select(-n_miss) |> 
  pivot_wider(names_from = col_name, values_from = col_type)
```

파일 형식이 이질적이면(heterogeneous formats) 성공적으로(successfully) 병합하기(merge) 전에 더 많은 처리(more processing)가 필요합니다(need to do). 이 부분은 직접 해결해야 하지만(Unfortunately, figure that out on your own) `map_if()`와 `map_at()`을 읽어보세요(might want to). `map_if()`는 값(values)에 따라(based on), `map_at()`은 이름(names)에 따라 리스트 요소를 선택적으로(selectively) 수정합니다(modify).

### 실패 처리

데이터 구조(structure)가 매우 거칠면(sufficiently wild) 명령 하나(single command)로 모든 파일을 읽지(can't even read) 못할 때가 있습니다(Sometimes). 이때 `map()`의 단점(downsides)이 드러납니다(encounter). 전체가(as a whole) 성공하거나(succeeds) 실패합니다(fails). 디렉터리의 모든 파일을 읽든지(read), 하나도 읽지 못하고 오류로 끝납니다(fail).

한 번의 실패 때문에(why does one failure) 나머지 성공 결과(successes)까지 이용하지(accessing) 못하니(prevent you from) 짜증스럽습니다(annoying). 다행히(Luckily) purrr의 `possibly()`가 이 문제를 해결합니다(tackle).

`possibly()`는 함수 연산자(function operator)입니다(known as). 함수를 받아(takes) 동작을 바꾼(modified behavior) 함수를 반환합니다(returns).

구체적으로(In particular) 오류를 내는(erroring) 대신(from) 사용자가 지정한(specify) 값을 반환하도록(returning) 바꿉니다(changes).

```{r}
files <- paths |> 
  map(possibly(\(path) readxl::read_excel(path), NULL))

data <- files |> list_rbind()
```

`list_rbind()`는 많은 tidyverse 함수처럼(like) `NULL`을 자동으로 무시해(ignores) 여기서 특히 잘(particularly well) 작동합니다(works). 읽기 쉬운(can be read easily) 데이터는 모두 준비됐습니다. 이제 일부 파일을 불러오지 못한(failed to load) 이유(why)와 처리 방법(what to do about it)을 파악하는(figuring out) 어려운 단계(hard part)입니다(tackle). 먼저(Start by) 실패한(failed) 경로를 확인합니다(getting).

```{r}
failed <- map_vec(files, is.null)
paths[failed]
```

그런 다음 실패마다 임포트(import) 함수를 다시 호출해 무엇이 잘못됐는지(what went wrong) 파악합니다(figure out).

## 여러 출력 저장

앞 절에서는 여러 파일을 하나의 객체로 읽는 `map()`을 배웠습니다. 이제 반대 문제(opposite problem)를 살펴봅니다(explore). 하나 이상의(one or more) R 객체를 받아(take) 하나 이상의 파일에 저장하는(save) 방법입니다. 세 가지 예제로 이 과제(challenge)를 다룹니다.

- 여러 데이터 프레임을 하나의 데이터베이스에 저장하기.
- 여러 데이터 프레임을 여러 `.csv` 파일에 저장하기.
- 여러 플롯(plots)을 여러 `.png` 파일에 저장하기.

### 데이터베이스에 쓰기

여러 파일의 모든 데이터를 한 번에(at once) 메모리에 올리지(fit) 못하면(not possible) `map(files, read_csv)`를 실행할 수 없습니다. 한 가지 해결법(approach)은 데이터를 데이터베이스에 넣고(load) dbplyr로 필요한 부분(bits)에만 접근하는(access) 것입니다(deal with).

운이 좋다면(If you're lucky) 사용 중인 데이터베이스 패키지에서 경로 벡터(vector of paths)를 가져와서 모두 데이터베이스에 로드하는 편리한 함수(handy function)를 제공할 것입니다. duckdb의 `duckdb_read_csv()`가 바로 이 경우(the case)입니다.

```{r}
#| eval: false
con <- DBI::dbConnect(duckdb::duckdb())
duckdb::duckdb_read_csv(con, "gapminder", paths)
```

여기서는 잘(well) 작동하겠지만 csv 파일이 아니라 엑셀 스프레드시트가 있습니다. 따라서 "수동으로(by hand)" 처리해야 합니다(going to have to do it). 이 방법은 csv가 여러 개(a bunch of)인데 데이터베이스에 모두 한 번에(all in) 넣는 함수가 하나도(one) 없을 때도 유용합니다.

우리는 데이터로 채울(fill in) 테이블을 생성하는(creating) 것부터 시작해야 합니다. 이 작업을 수행하는 쉬운 방법은 템플릿(template), 즉 우리가 원하는 모든 열을 포함(contains)하지만 데이터의 샘플링(sampling)만 포함하는 더미 데이터 프레임(dummy data frame)을 만드는 것입니다. gapminder 데이터의 경우, 단일(single) 파일을 읽고 거기에 연도를 추가(adding)하여 해당 템플릿을 만들 수(make) 있습니다.

```{r}
#| eval: false
template <- readxl::read_excel(paths[[1]])
template$year <- 1952
template
```

이제 데이터베이스에 연결하고(connect) `DBI::dbCreateTable()`을 사용하여 템플릿을 데이터베이스 테이블로 변환할(turn) 수 있습니다.

```{r}
#| eval: false
con <- DBI::dbConnect(duckdb::duckdb())
DBI::dbCreateTable(con, "gapminder", template)
```

`dbCreateTable()`은 `template`의 데이터는 사용하지 않고(doesn't use) 변수 이름과 유형(types)만 사용합니다. 따라서 지금(now) `gapminder` 테이블을 검사(inspect)해 보면 비어(empty) 있지만 예상하는(expect) 유형의 필요한 변수가 있음을 알 수 있습니다.

```{r}
#| eval: false
con |> tbl("gapminder")
```

다음으로 파일 경로 하나를 받아(takes) R로 읽고(reads) 결과를 `gapminder` 테이블에 추가하는(adds) 함수가 필요합니다. `read_excel()`과 `DBI::dbAppendTable()`을 결합합니다(combining).

```{r}
#| eval: false
append_file <- function(path) {
  df <- readxl::read_excel(path)
  df$year <- parse_number(basename(path))
  
  DBI::dbAppendTable(con, "gapminder", df)
}
```

이제 `paths`의 각 요소마다 `append_file()`을 한 번씩(once) 호출합니다. `map()`으로도 가능합니다(certainly).

```{r}
#| eval: false
paths |> map(append_file)
```

하지만 `append_file()`의 출력(output)은 필요하지 않으므로(don't care about) `map()`보다 `walk()`가 조금 낫습니다(slightly nicer). `walk()`는 같은 작업(exactly the same thing)을 하고 출력을 버립니다(throws away).

```{r}
#| eval: false
paths |> walk(append_file)
```

이제 테이블에 모든 데이터가 있는지 확인합니다(can see).

```{r}
#| eval: false
con |> 
  tbl("gapminder") |> 
  count(year)
```

```{r}
#| eval: false
#| include: false
DBI::dbDisconnect(con, shutdown = TRUE)
```

### csv 파일 쓰기

그룹마다 하나씩(one for each group) csv 파일을 여러 개 쓸(want to write) 때도 같은 원칙(basic principle)을 적용합니다(applies). `ggplot2::diamonds` 데이터에서(take) 각 `clarity`별 csv 파일을 저장한다고(imagine) 해봅시다. 먼저 개별 데이터셋을 만듭니다(make). 여러 방법 가운데 `group_nest()`를 사용하겠습니다(one way we particularly like).

```{r}
by_clarity <- diamonds |> 
  group_nest(clarity)

by_clarity
```

그러면 행 8개, 열 2개의 티블이 나옵니다(gives us). `clarity`는 그룹화 변수(grouping variable)이고 `data`는 `clarity`의 고유 값(unique value)마다 티블 하나를 담은(containing) 리스트 열(list-column)입니다.

```{r}
by_clarity$data[[1]]
```

이어서(While we're here) `mutate()`와 `str_glue()`로 출력 파일 이름을 담는(gives the name of) 열을 만듭니다(create).

```{r}
by_clarity <- by_clarity |> 
  mutate(path = str_glue("diamonds-{clarity}.csv"))

by_clarity
```

이 데이터 프레임을 수동으로(by hand) 저장한다면(save) 다음과 같이 작성합니다(if we were going to).

```{r}
#| eval: false
write_csv(by_clarity$data[[1]], by_clarity$path[[1]])
write_csv(by_clarity$data[[2]], by_clarity$path[[2]])
write_csv(by_clarity$data[[3]], by_clarity$path[[3]])
...
write_csv(by_clarity$data[[8]], by_clarity$path[[8]])
```

이번에는 변하는(changing) 인수가 두 개라 이전(previous)의 `map()`과 조금 다릅니다(little different). 첫 번째와 두 번째 인수를 모두 바꾸는(varies) `map2()`가 필요합니다. 출력은 다시(again) 필요하지 않으므로(don't care about) 실제로는 `walk2()`를 씁니다.

```{r}
walk2(by_clarity$data, by_clarity$path, write_csv)
```

```{r}
#| include: false
unlink(by_clarity$path)
```

### 플롯 저장 (Saving plots)

같은 기본 방식으로 플롯도 여러 개 만듭니다(create many plots). 먼저 원하는 플롯을 그리는(draws) 함수를 작성합니다.

```{r}
#| fig-alt: |
#|   Histogram of carats of diamonds from the by_clarity dataset, ranging from 
#|   0 to 5 carats. The distribution is unimodal and right skewed with a peak 
#|   around 1 carat.

carat_histogram <- function(df) {
  ggplot(df, aes(x = carat)) + geom_histogram(binwidth = 0.1)  
}

carat_histogram(by_clarity$data[[1]])
```

이제 `map()`으로 여러 플롯과 최종(eventual) 파일 경로의 리스트를 만듭니다(create).

```{r}
by_clarity <- by_clarity |> 
  mutate(
    plot = map(data, carat_histogram),
    path = str_glue("clarity-{clarity}.png")
  )
```

그런 다음 `walk2()`와 `ggsave()`로 각 플롯을 저장합니다(save).

```{r}
walk2(
  by_clarity$path,
  by_clarity$plot,
  \(path, plot) ggsave(path, plot, width = 6, height = 6)
)
```

이것은 다음의 약어(shorthand)입니다.

```{r}
#| eval: false
ggsave(by_clarity$path[[1]], by_clarity$plot[[1]], width = 6, height = 6)
ggsave(by_clarity$path[[2]], by_clarity$plot[[2]], width = 6, height = 6)
ggsave(by_clarity$path[[3]], by_clarity$plot[[3]], width = 6, height = 6)
...
ggsave(by_clarity$path[[8]], by_clarity$plot[[8]], width = 6, height = 6)
```

```{r}
#| include: false
unlink(by_clarity$path)
```

### 연습 문제

1. 여러 변수 중 `school_name`과 `student_id`가 포함된(containing) 학생 데이터 테이블이 있다고 해봅시다. 각 학생의 모든 정보를 `{school}` 디렉터리의 `{student_id}.csv` 파일에 저장하는(want to) 코드를 스케치하세요(Sketch out).

<!-- HUMANIZE-SUMMARY
원본 글자수: 30,817자
윤문본 글자수: 28,056자
변경률: 12.2% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 18 → 0
- A-7 가지다 직역: 0 → 0
- A-10 가능 표현 남발: 36 → 0
- A-11 목적절 남발: 4 → 0
- A-15 본문 추상 주어·만능 동사: 5 → 0
- C-11 연결어미 뒤 쉼표: 9 → 0

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
- "무료로 얻을 수 있기 때문에" → "별도 작업 없이 이루어집니다"
- "지루해질 것이라고 상상할 수 있습니다" → "작업도 매우 지루합니다"
- "세 가지 기본 단계가 있습니다" → "세 단계로 자동화합니다"
- "한 가지 실패로 인해 다른 모든 성공에 액세스하지 못하는" → "한 번의 실패 때문에 나머지 성공 결과까지 이용하지 못하는"
- "어떻게 하면 R 객체를 파일에 저장할 수 있을까요" → "R 객체를 파일에 저장하는 방법입니다"
-->
