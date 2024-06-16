---
title: "벡터: 문자열"
---

```{r}
#| echo: false
source("_common.R")
```

> 문자열이 작동하는 원리를 배우고 강력한 문자열 조작 도구를 익힐 차례입니다. 먼저 문자열과 문자형 벡터(character vectors)를 만드는 세부 사항을 살펴봅니다. 이어서 데이터에서 문자열을 만들거나 추출(extracting)하는 방법을 자세히 알아보고 마지막에는 개별 글자(letters)를 다루는 도구를 설명합니다.

```{r}
#| label: setup
#| message: false
library(tidyverse)
library(babynames)
```

모든 stringr 함수는 `str_`로 시작하므로 사용 여부를 바로 알 수 있습니다. RStudio에서는 특히 유용합니다. `str_`을 입력하면 자동 완성(autocomplete)이 실행되어(trigger) 사용 가능한 기능을 떠올리는(jog your memory) 데 도움이 됩니다.

## 문자열 만들기

이 책의 앞부분에서도 문자열을 만들었지만 세부 사항은 다루지 않았습니다. 문자열은 단일 따옴표(`'`)나 이중 따옴표(`"`)로 만듭니다. 둘의 동작은 같으므로 [tidyverse 스타일 가이드](https://style.tidyverse.org/syntax.html#character-vectors)는 문자열에 `"`가 여러 개 들어 있지 않다면 `"`를 일관되게 쓰도록 권장합니다.

```{r}
string1 <- "This is a string"
string2 <- 'If I want to include a "quote" inside a string, I use single quotes'
```

따옴표를 닫는 것을 잊어버리면 `+` 프롬프트가 나타납니다.

```         
> "This is a string without a closing quote
+ 
+ 
+ HELP I'M STUCK IN A STRING
```

이런 일이 생겼는데 어떤 따옴표를 닫아야 할지 모르겠다면(can't figure out) Escape 키를 눌러 취소하고 다시 시도하세요.

### 이스케이프

문자열에 리터럴 단일 또는 이중 따옴표(literal single or double quote)를 넣으려면 `\`로 "이스케이프(escape)"합니다.

```{r}
double_quote <- "\"" # or '"'
single_quote <- '\'' # or "'"
```

문자열에 리터럴 백슬래시(literal backslash)를 넣을 때는 `"\\"`로 이스케이프해야 합니다.

```{r}
backslash <- "\\"
```

문자열의 출력 표현(printed representation)은 이스케이프를 표시하므로 문자열 자체와 같지 않습니다. 다시 말해 문자열을 인쇄한 출력을 복사해 붙여넣으면 같은 문자열을 다시 만들 수 있습니다. 원래 내용(raw contents)은 `str_view()`[^strings-1]로 확인하세요.

[^strings-1]: 또는 기본 R 함수인 `writeLines()`를 사용하세요.

```{r}
x <- c(single_quote, double_quote, backslash)
x
str_view(x)
```

### 원시 문자열

따옴표나 백슬래시가 여러 개 들어간 문자열은 금세 복잡해집니다. 문제를 설명하려고(illustrate) `double_quote`와 `single_quote` 변수를 정의한 코드 블록 전체를 문자열로 만들어 보겠습니다.

```{r}
tricky <- "double_quote <- \"\\\"\" # or '\"'
single_quote <- '\\'' # or \"'\""
str_view(tricky)
```

백슬래시가 참 많습니다! 때로는 이를 [기울어진 이쑤시개 증후군](https://en.wikipedia.org/wiki/Leaning_toothpick_syndrome)이라고 합니다. 이스케이프를 없애려면(eliminate) 원시 문자열(raw string)[^strings-2]을 사용합니다.

[^strings-2]: R 4.0.0 이상에서 사용 가능합니다.

```{r}
tricky <- r"(double_quote <- "\"" # or '"'
single_quote <- '\'' # or "'")"
str_view(tricky)
```

원시 문자열은 보통 `r"(`로 시작해 `)"`로 끝납니다. 문자열에 `)"`가 들어 있다면 `r"[]"`나 `r"{}"`를 대신 씁니다. 그래도 부족하면 대시(dashes)를 여러 개 넣어 여닫는 쌍을 고유하게 만듭니다.

### 기타 특수 문자

`"`, `'`, `\` 외에도 유용한 특수 문자가 몇 가지 있습니다. 흔히 쓰는 것은 줄 바꿈(new line) `\n`과 탭(tab) `\t`입니다. 문자열에 `\u`나 `\U`로 시작하는 유니코드 이스케이프가 들어 있을 때도 있습니다. 모든 시스템에서 작동하는 비영어 문자를 작성하는 방법입니다. 다른 특수 문자의 전체 목록은 `?Quotes`에서 확인하세요.

```{r}
x <- c("one\ntwo", "one\ttwo", "\u00b5", "\U0001f604")
x
str_view(x)
```

`str_view()`는 탭이 눈에 잘 띄도록(easier to spot)[^strings-3] 중괄호(curly braces)를 사용합니다.
텍스트에는 공백(white space)이 여러 방식으로 들어갈 수 있어서 다루기가 까다롭습니다. 이 배경(background)을 알면 이상한 부분을 알아보기(recognize) 쉽습니다.

[^strings-3]: `str_view()`는 탭, 공백, 일치 항목 등을 강조(attention)하려고 색상도 사용합니다. 이 책에는 색상이 표시되지 않지만 대화형(interactively)으로 코드를 실행하면 확인됩니다.

### 연습 문제

1.  다음 값을 포함하는 문자열을 만드세요.

  a.  `He said "That's amazing!"`
  b.  `\a\b\c\d`
  c.  `\\\\\\`

2.  R 세션에서 문자열을 만들고 인쇄해 보세요. 특수 문자 "\\u00a0"에는 어떤 일이 일어납니까? `str_view()`는 이것을 어떻게 표시합니까? 구글에서 이 특수 문자가 무엇인지 찾아보세요.

```{r}
x <- "This\u00a0is\u00a0tricky"
```

## 데이터에서 여러 문자열 만들기

"수동(by hand)"으로 문자열 한두 개를 만드는 법을 배웠으니 이제 다른 문자열로 새 문자열을 만드는 방법을 살펴보겠습니다.
작성한 텍스트와 데이터 프레임의 문자열을 결합하는(combine) 일반적인 문제를 해결하는 데 유용합니다.
예를 들어 "Hello"와 `name` 변수를 결합해 인사말(greeting)을 만듭니다.

### `str_c()`

`str_c()`는 여러 개의 벡터를 인수로 받고 문자형 벡터를 반환합니다.

```{r}
str_c("x", "y")
str_c("x", "y", "z")
str_c("Hello ", c("John", "Susan"))
```

`str_c()`는 기본 `paste0()`과 매우 비슷하지만 결측값의 재활용(recycling)과 전파(propagating)에 tidyverse의 일반 규칙을 따르도록 설계되어 `mutate()`와 잘 맞습니다.

```{r}
df <- tibble(name = c("Flora", "David", "Terra", NA))
df |> mutate(greeting = str_c("Hi ", name, "!"))
```

결측값을 다르게 표시하려면 `coalesce()`로 대체하세요.
원하는 방식에 따라 `str_c()` 안이나 밖에서 사용합니다.

```{r}
df |>
    mutate(
        greeting1 = str_c("Hi ", coalesce(name, "you"), "!"),
        greeting2 = coalesce(str_c("Hi ", name, "!"), "Hi!")
    )
```

### `str_glue()`

`str_c()`로 고정 문자열과 변수 문자열을 많이 섞으면(mixing) `"`가 늘어나 코드의 전체적인 목적을 파악하기 어렵습니다. 대안은 glue 패키지([https://glue.tidyverse.org](https://glue.tidyverse.org))의 `str_glue()`[^strings-4]입니다. 이 함수에는 특별한 기능이 있는 문자열 하나를 넣습니다. `{}` 안의 내용은 따옴표 밖에 있는 것처럼 평가(evaluated)됩니다.

[^strings-4]: stringr을 사용하지 않는 경우 `glue::glue()`로 직접 액세스할 수도 있습니다.

```{r}
df |> mutate(greeting = str_glue("Hi {name}!"))
```

`str_glue()`는 현재 결측값을 `"NA"` 문자열로 바꿔 `str_c()`와 일관되지 않습니다.

문자열 안에 일반적인 `{`나 `}`를 넣으려면 어떻게 해야 할까요? 이스케이프가 필요하다고 짐작했다면 올바른 방향(on the right track)입니다. 다만 glue는 조금 다른 기법을 씁니다. \\ 같은 특수 문자를 접두사(prefixing)로 붙이는 대신 해당 문자를 두 번 씁니다(double up).

```{r}
df |> mutate(greeting = str_glue("{{Hi {name}!}}"))
```

### `str_flatten()`

`str_c()`와 `str_glue()`는 출력과 입력의 길이가 같아서 `mutate()`와 잘 작동합니다. `summarize()`에는 언제나 문자열 하나를 반환하는 함수가 필요합니다. 이 역할(job)을 하는 `str_flatten()`[^strings-5]은 문자형 벡터를 받아 모든 요소를 문자열 하나로 결합합니다.

[^strings-5]: 기본 R에 해당하는 것은 `collapse` 인수가 사용된 `paste()`입니다.

```{r}
str_flatten(c("x", "y", "z"))
str_flatten(c("x", "y", "z"), ", ")
str_flatten(c("x", "y", "z"), ", ", last = ", and ")
```

이렇게 하면 `summarize()`와 잘 작동합니다.

```{r}
df <- tribble(
    ~name     , ~fruit       ,
    "Carmen"  , "banana"     ,
    "Carmen"  , "apple"      ,
    "Marvin"  , "nectarine"  ,
    "Terence" , "cantaloupe" ,
    "Terence" , "papaya"     ,
    "Terence" , "mandarin"
)
df |>
    group_by(name) |>
    summarize(fruits = str_flatten(fruit, ", "))
```

### 연습 문제

1.  다음 입력에서 `paste0()`과 `str_c()`의 결과를 비교하고 대조(contrast)해 보세요.

```{r}
#| eval: false
str_c("hi ", NA)
str_c(letters[1:2], letters[1:3])
```

2. `paste()`와 `paste0()`의 차이점은 무엇입니까? `str_c()`를 사용하여 `paste()`와 동일한(equivalent) 결과를 어떻게 다시 만들 수 있습니까?

3. 다음 표현식을 `str_c()`에서 `str_glue()`로 또는 그 반대로 변환하세요.

  a. `str_c("The price of ", food, " is ", price)`
  b. `str_glue("I'm {age} years old and live in {country}")`
  c. `str_c("\\section{", title, "}")`

## 데이터에서 문자열 추출하기

여러 변수가 문자열 하나에 함께 들어 있는(crammed together) 경우는 매우 흔합니다.

- `df |> separate_longer_delim(col, delim)`
- `df |> separate_longer_position(col, width)`
- `df |> separate_wider_delim(col, delim, names)`
- `df |> separate_wider_position(col, widths)`

자세히 보면 공통 패턴이 있습니다. `separate_` 다음에 `longer`나 `wider`, `_`, `delim`이나 `position`이 차례로 옵니다. 네 함수가 더 단순한 원시 요소(primitives) 두 가지로 구성(composed)되어 있기 때문입니다.

-`pivot_longer()`와 `pivot_wider()`처럼 `_longer` 함수는 새 행을 만들어 입력 데이터 프레임을 더 길게(longer) 하고 `_wider` 함수는 새 열을 만들어 더 넓게(wider) 합니다.
- `delim`은 `", "`나 `" "` 같은 구분 기호(delimiter)로 문자열을 나눕니다. `position`은 `c(3, 5, 2)`처럼 지정한 너비로 나눕니다.

`wider` 함수 중에는 더 유연한 함수도 있지만 사용하려면 정규 표현식을 알아야 합니다.

### 행으로 분리하기

구성 요소(components)의 수가 행마다 다를 때(varies)는 문자열을 행으로 분리합니다. 흔히 `separate_longer_delim()`으로 구분 기호를 기준 삼아 나눕니다.

```{r}
df1 <- tibble(x = c("a,b,c", "d,e", "f"))
df1 |>
    separate_longer_delim(x, delim = ",")
```

실전에서(in the wild) `separate_longer_position()`을 볼 일은 더 드뭅니다(rarer). 일부 오래된 데이터셋은 문자 하나가 값 하나를 기록하는 매우 조밀한 형식(compact format)을 사용합니다.

```{r}
df2 <- tibble(x = c("1211", "131", "21"))
df2 |>
    separate_longer_position(x, width = 1)
```

### 열로 분리하기

문자열마다 구성 요소의 수가 일정하고 이를 열로 펼칠(spread) 때는 문자열을 열로 분리합니다. 열 이름을 지정해야 해서 대응하는 `longer` 함수보다 조금 복잡합니다.

다음 데이터셋의 `x`는 코드, 에디션 번호, 연도로 구성되며 `"."`으로 구분됩니다. `separate_wider_delim()`에는 구분 기호와 이름(names)을 인수로 지정합니다.

```{r}
df3 <- tibble(x = c("a10.1.2022", "b10.2.2011", "e15.1.2015"))
df3 |>
    separate_wider_delim(
        x,
        delim = ".",
        names = c("code", "edition", "year")
    )
```

필요 없는 조각(piece)은 이름에 `NA`를 사용해 결과에서 뺍니다.

```{r}
df3 |>
    separate_wider_delim(
        x,
        delim = ".",
        names = c("code", NA, "year")
    )
```

`separate_wider_position()`은 각 열의 너비를 지정해야 해서 조금 다르게 작동합니다. 이름(name)은 새 열 이름이고 값(value)은 해당 열이 차지하는(occupies) 문자 수인 명명된(named) 정수형 벡터를 지정합니다.

이름을 지정하지 않은 값은 출력에서 생략됩니다.

```{r}
df4 <- tibble(x = c("202215TX", "202122LA", "202325CA"))
df4 |>
    separate_wider_position(
        x,
        widths = c(year = 4, age = 2, state = 2)
    )
```

### widening 문제 진단하기

`separate_wider_delim()`[^strings-6]에는 고정되고 알려진(known) 열 집합이 필요합니다. 일부 행의 조각 수가 예상과 다르면 어떻게 될까요?

조각이 너무 적거나(too few) 많은(too many) 두 가지 문제가 생깁니다. `separate_wider_delim()`은 `too_few`와 `too_many` 인수로 이를 처리합니다. 먼저 다음 샘플 데이터셋으로 `too_few`의 경우를 살펴보겠습니다.

[^strings-6]: `separate_wider_position()` 및 `separate_wider_regex()`에도 동일한 원리가 적용됩니다.

```{r}
#| error: true
df <- tibble(a = c("1-1-1", "1-1-2", "1-3", "1-3-2", "1"))

df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z")
    )
```

오류가 발생하지만 어떻게 진행(proceed)할지 몇 가지 제안(suggestions)도 나옵니다.
먼저 문제를 디버깅(debugging)하겠습니다.

```{r}
debug <- df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z"),
        too_few = "debug"
    )
debug
```

디버그 모드를 사용하면 출력에 `a_ok`, `a_pieces`, `a_remainder`라는 열 세 개가 추가됩니다. 변수를 다른 이름으로 분리하면 접두사도 달라집니다. `a_ok`를 사용하면 실패한 입력을 빠르게 찾습니다.

```{r}
debug |> filter(!a_ok)
```

`a_pieces`는 발견된 조각 수를 예상값인 3(`names`의 길이)과 비교합니다. `a_remainder`는 조각이 너무 적을 때는 유용하지 않지만 곧 다시 살펴보겠습니다.

이 디버깅 정보에서 구분 기호 전략의 문제가 드러나거나(reveal) 분리 전에 전처리(preprocessing)가 더 필요하다는 사실을 알게 되기도 합니다. 이때는 상류(upstream)에서 문제를 고치고 `too_few = "debug"`를 제거해 새로운 문제를 오류로 표시하세요.

누락된 부분을 `NA`로 채우고 계속 진행(move on)해야 할 때도 있습니다. `too_few = "align_start"`와 `too_few = "align_end"`가 이 역할(job)을 하며 `NA`의 위치를 제어합니다.

```{r}
df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z"),
        too_few = "align_start"
    )
```

너무 많은 조각이 있는 경우에도 동일한 원리가 적용됩니다.

```{r}
#| error: true
df <- tibble(a = c("1-1-1", "1-1-2", "1-3-5-6", "1-3-2", "1-3-5-7-9"))

df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z")
    )
```

이제 결과를 디버깅하면 `a_remainder`의 용도가 드러납니다.

```{r}
debug <- df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z"),
        too_many = "debug"
    )
debug |> filter(!a_ok)
```

조각이 너무 많을 때는 다른 옵션을 사용합니다. 추가 조각을 조용히 "삭제(drop)"하거나 모두 마지막 열로 "병합(merge)"합니다.

```{r}
df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z"),
        too_many = "drop"
    )


df |>
    separate_wider_delim(
        a,
        delim = "-",
        names = c("x", "y", "z"),
        too_many = "merge"
    )
```

## 글자

이 절에서는 문자열 안의 개별 글자를 다루는 함수를 소개합니다.
문자열 길이를 구하고 부분 문자열(substrings)을 추출하며 플롯과 표에서 긴 문자열을 처리하는(handle) 방법을 배웁니다.

### 길이

`str_length()`는 문자열의 글자 수를 반환합니다.

```{r}
str_length(c("a", "R for data science", NA))
```

`count()`와 함께 사용하면 미국 아기 이름 길이의 분포를 구합니다. 이어서 `filter()`로 긴 이름을 살펴봅니다. 공교롭게도(happen to) 모두 15글자입니다[^strings-7].

[^strings-7]: 이 항목들(entries)을 보면 babynames 데이터가 공백이나 하이픈을 버리고 15글자 이후를 잘라낸다고(truncates) 추측할 만합니다.

```{r}
babynames |>
    count(length = str_length(name), wt = n)

babynames |>
    filter(str_length(name) == 15) |>
    count(name, wt = n, sort = TRUE)
```

### 부분집합

`str_sub(string, start, end)`는 문자열의 일부를 추출합니다. `start`와 `end`는 부분 문자열의 시작과 끝 위치입니다. 두 위치를 모두 포함(inclusive)하므로 반환되는 문자열의 길이는 `end - start + 1`입니다.

```{r}
x <- c("Apple", "Banana", "Pear")
str_sub(x, 1, 3)
```

음수 값을 사용하면 문자열 끝에서부터 거꾸로(back) 셉니다. -1은 마지막 문자, -2는 끝에서 두 번째 문자입니다.

```{r}
str_sub(x, -3, -1)
```

문자열이 너무 짧아도 `str_sub()`는 실패하지 않고 가능한 만큼 반환합니다.

```{r}
str_sub("a", 1, 5)
```

`str_sub()`를 `mutate()`와 함께 사용하면 각 이름의 첫 글자와 마지막 글자를 찾습니다.

```{r}
babynames |>
    mutate(
        first = str_sub(name, 1, 1),
        last = str_sub(name, -1, -1)
    )
```

### 연습 문제

1. 아기 이름 길이의 분포를 계산할 때 왜 `wt = n`을 사용했습니까?

2. `str_length()`와 `str_sub()`를 사용하여 각 아기 이름에서 중간 글자를 추출하세요. 문자열에 짝수 개의 문자가 있으면 어떻게 할 것입니까?

3. 시간이 지남에 따라 아기 이름의 길이에 주요 추세(major trends)가 있습니까? 첫 글자와 마지막 글자의 인기는 어떻습니까?

## 영어가 아닌 텍스트

지금까지 영어 텍스트에 초점을 맞춘 이유는 다루기가 특히 쉽기 때문입니다. 첫째, 영어 알파벳은 26글자뿐이라 비교적 단순합니다. 둘째, 아마 더 중요한 이유로 오늘날의 컴퓨팅 인프라는 주로(predominantly) 영어 사용자가 설계했습니다. 비영어권 언어를 온전히(full treatment) 다룰 공간은 없지만 인코딩(encoding), 글자 변형(letter variations), 로케일 종속 함수(locale-dependent functions)처럼 자주 마주칠(encounter) 과제는 짚고 넘어가겠습니다(draw your attention).

### 인코딩

비영어권 텍스트를 다룰 때 첫 번째 과제는 흔히 인코딩(encoding)입니다. 문제를 이해하려면 컴퓨터가 문자열을 표현하는 방식을 살펴봐야(dive into) 합니다. R에서는 `charToRaw()`로 문자열의 기본 표현(underlying representation)을 확인합니다.

```{r}
charToRaw("Hadley")
```

이 여섯 개의 16진수(hexadecimal numbers)는 각각 글자 하나를 나타냅니다. `48`은 H, `61`은 a입니다. 16진수와 문자의 매핑(mapping)을 인코딩이라고 하며 이 경우에는 ASCII입니다. ASCII는 정보 교환을 위한 미국 표준 코드(American Standard Code for Information Interchange)이므로 영어 문자를 잘 표현합니다.

비영어권 언어는 상황이 더 복잡합니다. 컴퓨팅 초창기에는 비영어권 문자를 인코딩하는 경쟁 표준(competing standards)이 여러 개였습니다. 유럽에도 두 인코딩이 있었습니다. 서유럽 언어에는 Latin1(일명 ISO-8859-1), 중부 유럽 언어에는 Latin2(일명 ISO-8859-2)를 사용했습니다. Latin1에서 바이트 `b1`은 "±"이지만 Latin2에서는 "ą"입니다!
다행히 오늘날에는 거의 모든 곳에서 지원하는 단일 표준인 UTF-8이 있습니다. 오늘날 사람이 사용하는 거의 모든 문자와 이모티콘 같은 여러 추가 기호를 인코딩합니다.

readr은 모든 곳에서 UTF-8을 사용합니다. 좋은 기본값이지만 UTF-8을 쓰지 않는 구형(older) 시스템에서 만든 데이터는 제대로 읽지 못합니다. 그러면 문자열을 인쇄할 때 이상하게 보입니다. 한두 문자만 엉망이 되기도(messed up) 하고 완전히 뜻 모를 글(gibberish)이 나오기도 합니다. 다음은 비정상적인 인코딩을 지닌 인라인 CSV 두 개입니다[^strings-8]:

[^strings-8]: 여기서는 특수한 `\x`를 사용해 이진 데이터(binary data)를 문자열로 직접 인코딩합니다.

```{r}
#| eval: false

x1 <- "text\nEl Ni\xf1o was particularly bad this year"
read_csv(x1)$text
#> [1] "El Ni\xf1o was particularly bad this year"

x2 <- "text\n\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd"
read_csv(x2)$text
#> [1] "\x82\xb1\x82\xf1\x82ɂ\xbf\x82\xcd"
```

올바르게 읽으려면 `locale` 인수로 인코딩을 지정합니다.

```{r}
#| eval: false
read_csv(x1, locale = locale(encoding = "Latin1"))$text
#> [1] "El Niño was particularly bad this year"

read_csv(x2, locale = locale(encoding = "Shift-JIS"))$text
#> [1] "こんにちは"
```

올바른 인코딩은 어떻게 찾을까요? 운이 좋다면 데이터 설명서 어딘가에 적혀 있습니다. 하지만 그런 경우는 드물어서(rarely the case) readr에는 `guess_encoding()`이 있습니다. 완벽하지는 않고(not foolproof) 여기보다 텍스트가 많을 때 더 잘 작동하지만 출발점으로는 합리적입니다. 올바른 인코딩을 찾을 때까지 몇 가지를 시도할 각오를 하세요.

인코딩은 내용이 풍부하고(rich) 복잡한 주제라 여기서는 표면만 살짝 다뤘습니다(scratched the surface). 더 자세한 설명은 [http://kunststube.net/encoding/](http://kunststube.net/encoding/)에서 읽어보세요.

### 글자 변형 (Letter variations)

강세(accents)가 있는 언어에서는 문자 위치를 판단할 때(`str_length()`와 `str_sub()` 사용 시) 큰(significant) 문제가 생깁니다(poses). 악센트 문자를 단일 문자(ü)로 인코딩하기도 하고 강세가 없는 문자(u)와 발음 구별 기호(diacritic mark)(¨)를 결합한 두 문자로 인코딩하기도 하기 때문입니다. 다음 코드는 똑같이(identical) 보이는 ü를 표현하는 두 방법입니다.

```{r}
u <- c("\u00fc", "u\u0308")
str_view(u)
```

하지만 두 문자열은 길이가 다르고 첫 번째 문자도 다릅니다.

```{r}
str_length(u)
str_sub(u, 1, 1)
```

`==`로 비교하면 두 문자열을 다르게 해석하지만 stringr의 `str_equal()` 함수는 외형(appearance)이 같다는 사실을 인식합니다.

```{r}
u[[1]] == u[[2]]

str_equal(u[[1]], u[[2]])
```

### 로케일 종속 함수

마지막으로 동작이 로케일(locale)에 따라 달라지는 stringr 함수가 몇 가지 있습니다. 로케일은 언어와 비슷하지만 언어 안의 지역 차이를 처리하는 선택적(optional) 지역 지정자(region specifier)도 포함합니다. 소문자 언어 약어(abbreviation)로 지정하고 필요하면 `_`와 대문자 지역 식별자(region identifier)를 붙입니다. "en"은 영어, "en_GB"는 영국 영어, "en_US"는 미국 영어입니다.

언어 코드를 모른다면 위키백과([https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes))의 목록을 참고하세요. stringr에서 지원하는 언어는 `stringi::stri_locale_list()`로 확인합니다.

기본 R 문자열 함수는 운영 체제에 설정된 로케일을 자동으로 사용합니다. 따라서 여러분의 언어에서는 예상대로 작동하지만 다른 국가의 사람과 코드를 공유하면 결과가 달라질 수 있습니다. 이 문제를 피하려고 stringr은 기본값인 "en" 로케일로 영어 규칙을 적용하며 이를 바꾸려면 `locale` 인수를 지정해야 합니다(override). 다행히 로케일이 정말 중요한 함수는 대소문자 변경(changing case)과 정렬(sorting), 두 종류뿐입니다.

대소문자 변경 규칙은 언어마다 다릅니다. 예를 들어 터키어에는 점이 있는 i와 점이 없는 i가 있습니다. 서로 다른 글자이므로 대문자로 표기하는 방법도 다릅니다.

```{r}
str_to_upper(c("i", "ı"))
str_to_upper(c("i", "ı"), locale = "tr")
```

문자열 정렬은 알파벳 순서에 따라 달라지며 그 순서는 언어마다 다릅니다[^strings-9]. 예를 들어 체코어에서 "ch"는 알파벳 `h` 다음에 오는 복합 문자(compound letter)입니다.

[^strings-9]: 중국어처럼 알파벳이 없는 언어의 정렬은 이보다 더 복잡합니다.

```{r}
str_sort(c("a", "c", "ch", "h", "z"))
str_sort(c("a", "c", "ch", "h", "z"), locale = "cs")
```

`dplyr::arrange()`로 문자열을 정렬할 때도 같은 문제가 생깁니다. 그래서 이 함수에도 `locale` 인수가 있습니다.

<!-- HUMANIZE-SUMMARY
원본 글자수: 17,185자
윤문본 글자수: 15,720자
변경률: 12.5% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 7 → 0
- A-7 가지다 직역: 1 → 0
- A-10 가능 표현 남발: 18 → 0
- A-11 목적절 남발: 6 → 0
- A-15 본문 추상 주어·만능 동사: 7 → 0
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
- "마음대로 사용할 수 있는 강력한 도구를 숙달할 때" → "강력한 도구를 익힐 차례"
- "여러 변수가 단일 문자열에 함께 쑤셔 넣어지는 경우" → "여러 변수가 문자열 하나에 함께 들어 있는 경우"
- "str_flatten()의 임무입니다" → "str_flatten()은 모든 요소를 문자열 하나로 결합합니다"
- "영어 이외의 언어에 대해서는 상황이 그렇게 쉽지 않습니다" → "비영어권 언어는 상황이 더 복잡합니다"
-->
