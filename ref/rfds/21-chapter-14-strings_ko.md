# 제14장. 문자열 (Strings)

# 소개 (Introduction)

지금까지 세부 사항에 대해 많이 배우지 않은 채로 많은 문자열을 사용해 왔습니다. 이제 문자열에 대해 깊이 파고들어, 문자열이 어떻게 작동하는지 알아보고, 자유자재로 다룰 수 있는 강력한 문자열 조작 도구들을 마스터할 때입니다.

문자열과 문자형 벡터(character vectors)를 만드는 구체적인 방법부터 시작하겠습니다. 그런 다음 데이터에서 문자열을 생성하는 방법과 그 반대로 데이터에서 문자열을 추출하는 방법을 자세히 알아볼 것입니다. 그다음에는 개별 글자(letters)를 다루는 도구들에 대해 논의할 것입니다. 이 장은 다른 언어로 작업할 때 영어로 인한 예상이 여러분을 어떻게 잘못된 방향으로 이끌 수 있는지에 대한 간략한 논의와 함께 마무리됩니다.

다음 장에서도 계속해서 문자열을 다룰 것이며, 그곳에서 정규 표현식(regular expressions)의 위력에 대해 더 많이 배우게 될 것입니다.

## 사전 준비 (Prerequisites)

이 장에서는 핵심 tidyverse의 일부인 stringr 패키지의 함수들을 사용할 것입니다. 또한 조작하기에 재미있는 몇 가지 문자열을 제공하는 babynames 데이터도 사용할 것입니다.

```
library(tidyverse)
library(babynames)
```

모든 stringr 함수는 `str_`로 시작하기 때문에 stringr 함수를 언제 사용하고 있는지 빠르게 알 수 있습니다. RStudio를 사용하는 경우 `str_`를 입력하면 자동 완성 기능이 실행되어 사용 가능한 함수들을 떠올릴 수 있으므로 특히 유용합니다.

<figure>
<img src="D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_14in01.png" alt="str_c typed into the RStudio console with the autocomplete tooltip shown on top, which lists functions beginning with str_c. The function signature and beginning of the main page for the highlighted function from the autocomplete list are shown in a panel to its right." />
</figure>

# 문자열 만들기 (Creating a String)

이 책의 앞부분에서 지나가듯 문자열을 만들었지만 세부 사항은 논의하지 않았습니다. 첫째, 작은따옴표(`'`)나 큰따옴표(`"`)를 사용하여 문자열을 만들 수 있습니다. 둘 사이의 동작에는 차이가 없으므로 일관성을 위해 [tidyverse 스타일 가이드](https://oreil.ly/_zF3d)에서는 문자열 안에 여러 개의 `"`가 포함되어 있지 않은 한 `"`를 사용하는 것을 권장합니다.

```
string1 <- "This is a string"
string2 <- 'If I want to include a "quote" inside a string, I use single quotes'
```

따옴표를 닫는 것을 잊어버리면 연속 프롬프트(continuation prompt)인 `+`가 표시됩니다.

    > "This is a string without a closing quote
    +
    +
    + HELP I'M STUCK IN A STRING

이런 일이 발생하고 어떤 따옴표를 닫아야 할지 알 수 없는 경우, Escape를 눌러 취소하고 다시 시도하세요.

## 이스케이프 (Escapes)

문자열 안에 문자 그대로의 작은따옴표나 큰따옴표를 포함하려면 `\`를 사용하여 "이스케이프(escape)" 할 수 있습니다.

```
double_quote <- "\"" # or '"'
single_quote <- '\'' # or "'"
```

따라서 문자열 안에 문자 그대로의 백슬래시를 포함하려면 이스케이프해야 합니다. `"\\"`:

```
backslash <- "\\"
```

문자열의 인쇄된 표현이 문자열 자체와 동일하지 않다는 점에 주의하세요. 인쇄된 표현은 이스케이프를 보여주기 때문입니다(다시 말해, 문자열을 인쇄할 때 출력을 복사하여 붙여넣으면 해당 문자열을 다시 만들 수 있습니다). 문자열의 원시 내용(raw contents)을 보려면 <a href="https://stringr.tidyverse.org/reference/str_view.html" class="orm:hideurl"><code>str_view()</code></a>를 사용하세요.<sup><a href="ch14.html#idm44771296672656" id="idm44771296672656-marker" data-type="noteref">1</a></sup>

```
x <- c(single_quote, double_quote, backslash)
x
#> [1] "'"  "\"" "\\"

str_view(x)
#> [1] │ '
#> [2] │ "
#> [3] │ \
```

## 원시 문자열 (Raw Strings)

여러 개의 따옴표나 백슬래시를 사용하여 문자열을 만들면 금방 혼란스러워집니다. 문제를 설명하기 위해 `double_quote`와 `single_quote` 변수를 정의한 코드 블록의 내용을 포함하는 문자열을 만들어 보겠습니다.

```
tricky <- "double_quote <- \"\\\"\" # or '\"'
single_quote <- '\\'' # or \"'\""
str_view(tricky)
#> [1] │ double_quote <- "\"" # or '"'
#>     │ single_quote <- '\'' # or "'"
```

백슬래시가 엄청나게 많습니다! (이것을 때때로 [기울어진 이쑤시개 증후군(leaning toothpick syndrome)](https://oreil.ly/Fs-YL)이라고 부릅니다.) 이스케이프를 없애려면 대신 *원시 문자열(raw string)*을 사용할 수 있습니다.<sup><a href="ch14.html#idm44771296567072" id="idm44771296567072-marker" data-type="noteref">2</a></sup>

```
tricky <- r"(double_quote <- "\"" # or '"'
single_quote <- '\'' # or "'")"
str_view(tricky)
#> [1] │ double_quote <- "\"" # or '"'
#>     │ single_quote <- '\'' # or "'"
```

원시 문자열은 일반적으로 `r"(`로 시작하고 `)"`로 끝납니다. 그러나 문자열에 `)"`가 포함된 경우 대신 `r"[]"` 또는 `r"{}"`를 사용할 수 있으며, 여전히 충분하지 않다면 여러 개의 대시(dashes)를 삽입하여 열기 및 닫기 쌍을 고유하게 만들 수 있습니다(`` `r"--()--" ``, `` `r"---()---" `` 등). 원시 문자열은 어떤 텍스트든 처리할 수 있을 만큼 유연합니다.

## 기타 특수 문자 (Other Special Characters)

`\"`, `\'`, `\\` 외에도 유용하게 사용할 수 있는 다른 특수 문자들이 몇 가지 있습니다. 가장 흔한 것은 줄바꿈인 `\n`과 탭인 `\t`입니다. 또한 `\u` 또는 `\U`로 시작하는 유니코드 이스케이프가 포함된 문자열을 가끔 볼 수도 있습니다. 이것은 모든 시스템에서 작동하는 비영어권 문자를 작성하는 방법입니다. <a href="https://rdrr.io/r/base/Quotes.html" class="orm:hideurl"><code>?Quotes</code></a>에서 다른 특수 문자의 전체 목록을 확인할 수 있습니다.

```
x <- c("one\ntwo", "one\ttwo", "\u00b5", "\U0001f604")
x
#> [1] "one\ntwo" "one\ttwo" "µ"        "ߘ䢊str_view(x)
#> [1] │ one
#>     │ two
#> [2] │ one{\t}two
#> [3] │ µ
#> [4] │ ߘ伯
```

<a href="https://stringr.tidyverse.org/reference/str_view.html" class="orm:hideurl"><code>str_view()</code></a>는 탭을 더 쉽게 발견할 수 있도록 파란색 배경을 사용한다는 점에 유의하세요. 텍스트로 작업할 때 겪는 어려움 중 하나는 텍스트에 공백(whitespace)이 들어가는 방식이 다양하다는 것인데, 이 배경색은 무언가 이상한 일이 일어나고 있음을 인식하는 데 도움을 줍니다.

## 연습문제 (Exercises)

1. 다음 값을 포함하는 문자열을 만드세요.
   1. `He said "That's amazing!"`
   2. `\a\b\c\d`
   3. `\\\\\\`
2. R 세션에서 다음 문자열을 만들고 인쇄하세요. 특수한 "\u00a0"에는 무슨 일이 일어납니까? <a href="https://stringr.tidyverse.org/reference/str_view.html" class="orm:hideurl"><code>str_view()</code></a>는 그것을 어떻게 표시합니까? 이 특수 문자가 무엇인지 구글링을 좀 해볼 수 있습니까?

   ```
   x <- "This\u00a0is\u00a0tricky"
   ```

# 데이터로부터 여러 문자열 만들기 (Creating Many Strings from Data)

이제 한두 개의 문자열을 "수동으로(by hand)" 만드는 기초를 배웠으니, 다른 문자열들로부터 문자열을 만드는 세부적인 내용으로 들어가 보겠습니다. 이것은 여러분이 작성한 텍스트를 데이터 프레임의 문자열과 결합하고 싶을 때 흔히 발생하는 문제를 해결하는 데 도움이 될 것입니다. 예를 들어, 인사말을 만들기 위해 "Hello"를 `name` 변수와 결합할 수 있습니다. <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>를 사용하여 이것을 수행하는 방법과 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>에서 그것들을 어떻게 사용할 수 있는지 보여드리겠습니다. 이는 자연스럽게 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 함께 어떤 stringr 함수를 사용할 수 있는지에 대한 의문으로 이어지며, 문자열 요약 함수인 <a href="https://stringr.tidyverse.org/reference/str_flatten.html" class="orm:hideurl"><code>str_flatten()</code></a>에 대한 논의로 이 섹션을 마무리할 것입니다.

## str_c()

<a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>는 여러 개의 벡터를 인자로 취하여 문자형 벡터를 반환합니다.

```
str_c("x", "y")
#> [1] "xy"
str_c("x", "y", "z")
#> [1] "xyz"
str_c("Hello ", c("John", "Susan"))
#> [1] "Hello John"  "Hello Susan"
```

<a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>는 기본 <a href="https://rdrr.io/r/base/paste.html" class="orm:hideurl"><code>paste0()</code></a>와 유사하지만 리사이클링 및 결측값 전파(propagating)에 대한 일반적인 tidyverse 규칙을 따름으로써 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 함께 사용되도록 설계되었습니다.

```
df <- tibble(name = c("Flora", "David", "Terra", NA))
df |> mutate(greeting = str_c("Hi ", name, "!"))
#> # A tibble: 4 × 2
#>   name  greeting
#>   <chr> <chr>
#> 1 Flora Hi Flora!
#> 2 David Hi David!
#> 3 Terra Hi Terra!
#> 4 <NA>  <NA>
```

결측값이 다른 방식으로 표시되기를 원한다면 <a href="https://dplyr.tidyverse.org/reference/coalesce.html" class="orm:hideurl"><code>coalesce()</code></a>를 사용하여 교체하세요. 원하는 바에 따라 <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a> 내부 또는 외부에서 사용할 수 있습니다.

```
df |>
  mutate(
    greeting1 = str_c("Hi ", coalesce(name, "you"), "!"),
    greeting2 = coalesce(str_c("Hi ", name, "!"), "Hi!")
  )
#> # A tibble: 4 × 3
#>   name  greeting1 greeting2
#>   <chr> <chr>     <chr>
#> 1 Flora Hi Flora! Hi Flora!
#> 2 David Hi David! Hi David!
#> 3 Terra Hi Terra! Hi Terra!
#> 4 <NA>  Hi you!   Hi!
```

## str_glue()

<a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>를 사용하여 많은 수의 고정 문자열과 가변 문자열을 혼합하다 보면, `"`를 아주 많이 입력하게 되어 코드의 전체적인 목표를 파악하기 어렵다는 것을 알게 될 것입니다. 이에 대한 대안적인 접근법은 [glue 패키지](https://oreil.ly/NHBNe)가 <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>를 통해 제공합니다.<sup><a href="ch14.html#idm44771296183072" id="idm44771296183072-marker" data-type="noteref">3</a></sup> 여기에 특별한 기능을 가진 단일 문자열을 제공합니다. <a href="https://rdrr.io/r/base/Paren.html" class="orm:hideurl"><code>{}</code></a> 안에 있는 모든 것은 따옴표 밖에 있는 것처럼 평가됩니다.

```
df |> mutate(greeting = str_glue("Hi {name}!"))
#> # A tibble: 4 × 2
#>   name  greeting
#>   <chr> <glue>
#> 1 Flora Hi Flora!
#> 2 David Hi David!
#> 3 Terra Hi Terra!
#> 4 <NA>  Hi NA!
```

보시다시피, <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>는 현재 결측값을 `"NA"` 문자열로 변환하므로 안타깝게도 <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>와 일관성이 없습니다.

문자열에 일반적인 `{` 또는 `}`를 포함해야 하는 경우 어떻게 되는지 궁금할 수도 있습니다. 어떻게든 그것을 이스케이프해야 한다고 추측했다면 제대로 가고 있는 것입니다. 요령은 glue가 약간 다른 이스케이프 기술을 사용한다는 것입니다. `\`와 같은 특수 문자를 접두사로 붙이는 대신 특수 문자를 두 번 씁니다.

```
df |> mutate(greeting = str_glue("{{Hi {name}!}}"))
#> # A tibble: 4 × 2
#>   name  greeting
#>   <chr> <glue>
#> 1 Flora {Hi Flora!}
#> 2 David {Hi David!}
#> 3 Terra {Hi Terra!}
#> 4 <NA>  {Hi NA!}
```

## str_flatten()

<a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a> 및 <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>는 출력이 입력과 동일한 길이이므로 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 잘 작동합니다. 만약 항상 단일 문자열을 반환하는 함수처럼 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 잘 작동하는 함수를 원한다면 어떨까요? 그것이 <a href="https://stringr.tidyverse.org/reference/str_flatten.html" class="orm:hideurl"><code>str_flatten()</code></a>의 임무입니다.<sup><a href="ch14.html#idm44771296091984" id="idm44771296091984-marker" data-type="noteref">4</a></sup> 이 함수는 문자형 벡터를 취하여 벡터의 각 요소를 단일 문자열로 결합합니다.

```
str_flatten(c("x", "y", "z"))
#> [1] "xyz"
str_flatten(c("x", "y", "z"), ", ")
#> [1] "x, y, z"
str_flatten(c("x", "y", "z"), ", ", last = ", and ")
#> [1] "x, y, and z"
```

이것은 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 함께 잘 작동하도록 만듭니다.

```
df <- tribble(
  ~ name, ~ fruit,
  "Carmen", "banana",
  "Carmen", "apple",
  "Marvin", "nectarine",
  "Terence", "cantaloupe",
  "Terence", "papaya",
  "Terence", "mandarin"
)
df |>
  group_by(name) |>
  summarize(fruits = str_flatten(fruit, ", "))
#> # A tibble: 3 × 2
#>   name    fruits
#>   <chr>   <chr>
#> 1 Carmen  banana, apple
#> 2 Marvin  nectarine
#> 3 Terence cantaloupe, papaya, mandarin
```

## 연습문제 (Exercises)

1. 다음 입력에 대해 <a href="https://rdrr.io/r/base/paste.html" class="orm:hideurl"><code>paste0()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>의 결과를 비교하고 대조하세요.

   ```
   str_c("hi ", NA)
   str_c(letters[1:2], letters[1:3])
   ```

2. <a href="https://rdrr.io/r/base/paste.html" class="orm:hideurl"><code>paste()</code></a>와 <a href="https://rdrr.io/r/base/paste.html" class="orm:hideurl"><code>paste0()</code></a>의 차이점은 무엇입니까? <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>를 사용하여 <a href="https://rdrr.io/r/base/paste.html" class="orm:hideurl"><code>paste()</code></a>와 동등한 기능을 어떻게 다시 만들 수 있습니까?
3. 다음 표현식을 <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>에서 <a href="https://stringr.tidyverse.org/reference/str_glue.html" class="orm:hideurl"><code>str_glue()</code></a>로 변환하거나 그 반대로 변환하세요.
   1. `str_c("The price of ", food, " is ", price)`
   2. `str_glue("I'm {age} years old and live in {country}")`
   3. `str_c("\\section{", title, "}")`

# 문자열에서 데이터 추출하기 (Extracting Data from Strings)

여러 변수가 단일 문자열에 함께 압축되어 있는 경우가 종종 있습니다. 이 섹션에서는 다음과 같은 네 가지 tidyr 함수를 사용하여 변수를 추출하는 방법을 배웁니다.

- `df |> separate_longer_delim(col, delim)`
- `df |> separate_longer_position(col, width)`
- `df |> separate_wider_delim(col, delim, names)`
- `df |> separate_wider_position(col, widths)`

자세히 보면 여기에 공통적인 패턴이 있음을 알 수 있습니다. `separate_`, 그 다음에 `longer` 또는 `wider`, 그 다음에 `_`, 그 다음에 `delim` 또는 `position`입니다. 이는 이 네 가지 함수가 두 가지의 더 간단한 기본 요소(primitives)로 구성되어 있기 때문입니다.

- <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>와 마찬가지로 `_longer` 함수는 새로운 행을 만들어 입력 데이터 프레임을 더 길게(longer) 만들고, `_wider` 함수는 새로운 열을 생성하여 입력 데이터 프레임을 더 넓게(wider) 만듭니다.
- `delim`은 `", "`나 `" "` 같은 구분 기호(delimiter)를 기준으로 문자열을 나눕니다. `position`은 `c(3, 5, 2)`와 같이 지정된 너비(width)에서 나눕니다.

이 제품군의 마지막 구성원인 `separate_wider_regex()`로 <a href="ch15.html#chp-regexps" data-type="xref">제15장</a>에서 다시 돌아올 것입니다. 이것은 `wider` 함수들 중에서 가장 유연하지만 이것을 사용하려면 정규 표현식에 대해 알아야 합니다.

다음 두 섹션에서는 이러한 분리 함수(separate functions) 이면에 있는 기본 아이디어를 설명할 것입니다. 먼저 행으로 분리하는 방법(이것이 약간 더 간단합니다)을 알아보고, 그 다음 열로 분리하는 방법을 알아볼 것입니다. 마지막으로 `wider` 함수가 문제를 진단하기 위해 제공하는 도구들을 논의하며 마무리하겠습니다.

## 행으로 분리하기 (Separating into Rows)

문자열을 행으로 분리하는 것은 행마다 구성 요소의 개수가 다를 때 가장 유용합니다. 가장 일반적인 경우는 구분 기호를 기준으로 나누기 위해 <a href="https://tidyr.tidyverse.org/reference/separate_longer_delim.html" class="orm:hideurl"><code>separate_longer_delim()</code></a>이 필요한 경우입니다.

```
df1 <- tibble(x = c("a,b,c", "d,e", "f"))
df1 |>
  separate_longer_delim(x, delim = ",")
#> # A tibble: 6 × 1
#>   x
#>   <chr>
#> 1 a
#> 2 b
#> 3 c
#> 4 d
#> 5 e
#> 6 f
```

실전에서 <a href="https://tidyr.tidyverse.org/reference/separate_longer_delim.html" class="orm:hideurl"><code>separate_longer_position()</code></a>을 보는 것은 좀 더 드물지만, 일부 오래된 데이터세트는 각 문자가 값을 기록하는 데 사용되는 압축된(compact) 형식을 사용합니다.

```
df2 <- tibble(x = c("1211", "131", "21"))
df2 |>
  separate_longer_position(x, width = 1)
#> # A tibble: 9 × 1
#>   x
#>   <chr>
#> 1 1
#> 2 2
#> 3 1
#> 4 1
#> 5 1
#> 6 3
#> # … with 3 more rows
```

## 열로 분리하기 (Separating into Columns)

문자열을 열로 분리하는 것은 각 문자열에 고정된 수의 구성 요소가 있고 그것들을 열로 펼치고자(spread) 할 때 가장 유용합니다. 열의 이름을 지정해야 하기 때문에 대응하는 `longer` 함수들보다 약간 더 복잡합니다. 예를 들어, 다음 데이터세트에서 `x`는 코드, 에디션 번호, 연도로 구성되며 `"."`으로 구분되어 있습니다. <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_delim()</code></a>을 사용하려면 두 개의 인자에 구분 기호와 이름들을 제공해야 합니다.

```
df3 <- tibble(x = c("a10.1.2022", "b10.2.2011", "e15.1.2015"))
df3 |>
  separate_wider_delim(
    x,
    delim = ".",
    names = c("code", "edition", "year")
  )
#> # A tibble: 3 × 3
#>   code  edition year
#>   <chr> <chr>   <chr>
#> 1 a10   1       2022
#> 2 b10   2       2011
#> 3 e15   1       2015
```

특정 조각이 유용하지 않은 경우 이름에 `NA`를 사용하여 결과에서 생략할 수 있습니다.

```
df3 |>
  separate_wider_delim(
    x,
    delim = ".",
    names = c("code", NA, "year")
  )
#> # A tibble: 3 × 2
#>   code  year
#>   <chr> <chr>
#> 1 a10   2022
#> 2 b10   2011
#> 3 e15   2015
```

<a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_position()</code></a>은 일반적으로 각 열의 너비를 지정해야 하므로 약간 다르게 작동합니다. 따라서 이름이 새 열의 이름을 나타내고, 값이 차지하는 문자 수를 나타내는 이름이 지정된 정수 벡터(named integer vector)를 제공합니다. 이름을 지정하지 않으면 출력에서 값을 생략할 수 있습니다.

```
df4 <- tibble(x = c("202215TX", "202122LA", "202325CA"))
df4 |>
  separate_wider_position(
    x,
    widths = c(year = 4, age = 2, state = 2)
  )
#> # A tibble: 3 × 3
#>   year  age   state
#>   <chr> <chr> <chr>
#> 1 2022  15    TX
#> 2 2021  22    LA
#> 3 2023  25    CA
```

## 넓히기 문제 진단하기 (Diagnosing Widening Problems)

<a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_delim()</code></a><sup><a href="ch14.html#idm44771295456640" id="idm44771295456640-marker" data-type="noteref">5</a></sup>는 알려진 고정된 열 집합이 필요합니다. 일부 행에 예상되는 개수의 조각이 없다면 어떻게 될까요? 조각이 너무 적거나 조각이 너무 많은 두 가지 가능한 문제가 있으므로, <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_delim()</code></a>은 문제를 돕기 위해 `too_few`와 `too_many`라는 두 가지 인자를 제공합니다. 다음 샘플 데이터세트를 사용하여 먼저 `too_few` 경우를 살펴보겠습니다.

```
df <- tibble(x = c("1-1-1", "1-1-2", "1-3", "1-3-2", "1"))

df |>
  separate_wider_delim(
    x,
    delim = "-",
    names = c("x", "y", "z")
  )
#> Error in `separate_wider_delim()`:
#> ! Expected 3 pieces in each element of `x`.
#> ! 2 values were too short.
#> ℹ Use `too_few = "debug"` to diagnose the problem.
#> ℹ Use `too_few = "align_start"/"align_end"` to silence this message.
```

오류가 발생했지만, 그 오류가 앞으로 어떻게 진행해야 할지 몇 가지 제안을 해주는 것을 알 수 있습니다. 문제를 디버깅하는 것부터 시작하겠습니다.

```
debug <- df |>
  separate_wider_delim(
    x,
    delim = "-",
    names = c("x", "y", "z"),
    too_few = "debug"
  )
#> Warning: Debug mode activated: adding variables `x_ok`, `x_pieces`, and
#> `x_remainder`.
debug
#> # A tibble: 5 × 6
#>   x     y     z     x_ok  x_pieces x_remainder
#>   <chr> <chr> <chr> <lgl>    <int> <chr>
#> 1 1-1-1 1     1     TRUE         3 ""
#> 2 1-1-2 1     2     TRUE         3 ""
#> 3 1-3   3     <NA>  FALSE        2 ""
#> 4 1-3-2 3     2     TRUE         3 ""
#> 5 1     <NA>  <NA>  FALSE        1 ""
```

디버그(debug) 모드를 사용하면 출력에 `x_ok`, `x_pieces`, `x_remainder`라는 세 개의 추가 열이 추가됩니다(다른 이름의 변수를 분리하면 다른 접두사를 얻게 됩니다). 여기서 `x_ok`를 통해 실패한 입력을 빠르게 찾을 수 있습니다.

```
debug |> filter(!x_ok)
#> # A tibble: 2 × 6
#>   x     y     z     x_ok  x_pieces x_remainder
#>   <chr> <chr> <chr> <lgl>    <int> <chr>
#> 1 1-3   3     <NA>  FALSE        2 ""
#> 2 1     <NA>  <NA>  FALSE        1 ""
```

`x_pieces`는 예상된 세 개(`names`의 길이)와 비교하여 발견된 조각의 개수를 알려줍니다. 조각이 너무 적은 경우 `x_remainder`는 유용하지 않지만, 곧 다시 보게 될 것입니다.

때로는 이러한 디버깅 정보를 보는 것만으로도 구분 기호 전략의 문제점을 드러내거나, 분리하기 전에 더 시처리를 수행해야 함을 시사할 수 있습니다. 그런 경우 업스트림(upstream)에서 문제를 수정하고 `too_few = "debug"`를 제거하여 새로운 문제가 발생할 경우 오류로 나타나도록 확실히 해두세요.

다른 경우에는 누락된 조각을 `NA`로 채우고 계속 진행하고 싶을 수도 있습니다. 그것이 바로 `NA`가 들어가야 할 위치를 제어할 수 있게 해주는 `too_few = "align_start"`와 `too_few = "align_end"`의 임무입니다.

```
df |>
  separate_wider_delim(
    x,
    delim = "-",
    names = c("x", "y", "z"),
    too_few = "align_start"
  )
#> # A tibble: 5 × 3
#>   x     y     z
#>   <chr> <chr> <chr>
#> 1 1     1     1
#> 2 1     1     2
#> 3 1     3     <NA>
#> 4 1     3     2
#> 5 1     <NA>  <NA>
```

조각이 너무 많은 경우에도 동일한 원리가 적용됩니다.

```
df <- tibble(x = c("1-1-1", "1-1-2", "1-3-5-6", "1-3-2", "1-3-5-7-9"))

df |>
  separate_wider_delim(
    x,
    delim = "-",
    names = c("x", "y", "z")
  )
#> Error in `separate_wider_delim()`:
#> ! Expected 3 pieces in each element of `x`.
#> ! 2 values were too long.
#> ℹ Use `too_many = "debug"` to diagnose the problem.
#> ℹ Use `too_many = "drop"/"merge"` to silence this message.
```

하지만 이제 결과를 디버그해 보면 `x_remainder`의 목적을 알 수 있습니다.

```
debug <- df |>
  separate_wider_delim(
    x,
    delim = "-",
    names = c("x", "y", "z"),
    too_many = "debug"
  )
#> Warning: Debug mode activated: adding variables `x_ok`, `x_pieces`, and
#> `x_remainder`.
debug |> filter(!x_ok)
#> # A tibble: 2 × 6
#>   x         y     z     x_ok  x_pieces x_remainder
#>   <chr>     <chr> <chr> <lgl>    <int> <chr>
#> 1 1-3-5-6   3     5     FALSE        4 -6
#> 2 1-3-5-7-9 3     5     FALSE        5 -7-9
```

조각이 너무 많을 때 처리하기 위한 옵션들은 약간 다릅니다. 추가 조각들을 조용히 "버리거나(drop)", 아니면 모두 마지막 열에 "병합(merge)"할 수 있습니다.

```
df |>
  separate_wider_delim(
    x,
    delim = "-",
    names = c("x", "y", "z"),
    too_many = "drop"
  )
#> # A tibble: 5 × 3
#>   x     y     z
#>   <chr> <chr> <chr>
#> 1 1     1     1
```

#> 2 1 1 2  
#> 3 1 3 5  
#> 4 1 3 2  
#> 5 1 3 5

df |>
separate_wider_delim(
x,
delim = "-",
names = c("x", "y", "z"),
too_many = "merge"
)
#> # A tibble: 5 × 3
#> x y z  
#> <chr> <chr> <chr>
#> 1 1 1 1  
#> 2 1 1 2  
#> 3 1 3 5-6  
#> 4 1 3 2  
#> 5 1 3 5-7-9

```

# 글자 (Letters)

이 섹션에서는 문자열 내의 개별 글자(letters)를 다룰 수 있게 해주는 함수들을 소개합니다. 문자열의 길이를 알아내고, 부분 문자열(substrings)을 추출하고, 플롯이나 표에서 긴 문자열을 처리하는 방법을 배울 것입니다.

## 길이 (Length)

<a href="https://stringr.tidyverse.org/reference/str_length.html" class="orm:hideurl"><code>str_length()</code></a>는 문자열의 글자 수를 알려줍니다.

```

str_length(c("a", "R for data science", NA))
#> [1] 1 18 NA

```

이것을 <a href="https://dplyr.tidyverse.org/reference/count.html" class="orm:hideurl"><code>count()</code></a>와 함께 사용하여 미국 아기 이름 길이의 분포를 찾고, 그런 다음 <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 함께 사용하여 가장 긴 이름(마침 15글자인)을 살펴볼 수 있습니다.<sup><a href="ch14.html#idm44771294893344" id="idm44771294893344-marker" data-type="noteref">6</a></sup>

```

babynames |>
count(length = str_length(name), wt = n)
#> # A tibble: 14 × 2
#> length n
#> <int> <int>
#> 1 2 338150
#> 2 3 8589596
#> 3 4 48506739
#> 4 5 87011607
#> 5 6 90749404
#> 6 7 72120767
#> # … with 8 more rows

babynames |>
filter(str_length(name) == 15) |>
count(name, wt = n, sort = TRUE)
#> # A tibble: 34 × 2
#> name n
#> <chr> <int>
#> 1 Franciscojavier 123
#> 2 Christopherjohn 118
#> 3 Johnchristopher 118
#> 4 Christopherjame 108
#> 5 Christophermich 52
#> 6 Ryanchristopher 45
#> # … with 28 more rows

```

## 서브세팅 (Subsetting)

`str_sub(string, start, end)`를 사용하여 문자열의 일부분을 추출할 수 있습니다. 여기서 `start`와 `end`는 부분 문자열이 시작하고 끝나는 위치입니다. `start` 및 `end` 인자는 모두 포함(inclusive)되므로, 반환되는 문자열의 길이는 `end - start + 1`이 됩니다.

```

x <- c("Apple", "Banana", "Pear")
str_sub(x, 1, 3)
#> [1] "App" "Ban" "Pea"

```

음수 값을 사용하여 문자열의 끝에서부터 거꾸로 계산할 수 있습니다. -1은 마지막 문자, -2는 끝에서 두 번째 문자 등입니다.

```

str_sub(x, -3, -1)
#> [1] "ple" "ana" "ear"

```

<a href="https://stringr.tidyverse.org/reference/str_sub.html" class="orm:hideurl"><code>str_sub()</code></a>는 문자열이 너무 짧더라도 실패하지 않고 가능한 한 많은 문자열을 반환합니다.

```

str_sub("a", 1, 5)
#> [1] "a"

```

<a href="https://stringr.tidyverse.org/reference/str_sub.html" class="orm:hideurl"><code>str_sub()</code></a>를 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 함께 사용하여 각 이름의 첫 글자와 마지막 글자를 찾을 수 있습니다.

```

babynames |>
mutate(
first = str_sub(name, 1, 1),
last = str_sub(name, -1, -1)
)
#> # A tibble: 1,924,665 × 7
#> year sex name n prop first last
#> <dbl> <chr> <chr> <int> <dbl> <chr> <chr>
#> 1 1880 F Mary 7065 0.0724 M y  
#> 2 1880 F Anna 2604 0.0267 A a  
#> 3 1880 F Emma 2003 0.0205 E a  
#> 4 1880 F Elizabeth 1939 0.0199 E h  
#> 5 1880 F Minnie 1746 0.0179 M e  
#> 6 1880 F Margaret 1578 0.0162 M t  
#> # … with 1,924,659 more rows

```

## 연습문제 (Exercises)

1. 아기 이름의 길이 분포를 계산할 때 왜 `wt = n`을 사용했습니까?
2. <a href="https://stringr.tidyverse.org/reference/str_length.html" class="orm:hideurl"><code>str_length()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_sub.html" class="orm:hideurl"><code>str_sub()</code></a>를 사용하여 각 아기 이름에서 가운데 글자를 추출하세요. 문자열의 글자 수가 짝수이면 어떻게 하겠습니까?
3. 시간이 지남에 따라 아기 이름의 길이에 주요한 추세(trends)가 있습니까? 첫 글자와 마지막 글자의 인기는 어떻습니까?

# 영어가 아닌 텍스트 (Non-English Text)

지금까지는 영어 텍스트에 중점을 두었는데, 두 가지 이유로 다루기가 특히 쉽습니다. 첫째, 영어 알파벳은 비교적 단순합니다. 단 26개의 글자만 있습니다. 둘째(그리고 아마도 더 중요한 것은), 오늘날 우리가 사용하는 컴퓨팅 인프라가 주로 영어를 사용하는 사람들에 의해 설계되었다는 것입니다. 안타깝게도 비영어권 언어를 완벽하게 다룰 만한 지면이 없습니다. 그래도 인코딩, 글자 변형, 로케일 의존(locale-dependent) 함수와 같이 여러분이 직면할 수 있는 가장 큰 과제 중 일부에 주의를 환기시키고 싶었습니다.

## 인코딩 (Encoding)

비영어권 텍스트로 작업할 때 가장 먼저 겪는 어려움은 종종 *인코딩(encoding)*입니다. 무슨 일이 일어나고 있는지 이해하려면 컴퓨터가 문자열을 표현하는 방식에 대해 파고들어야 합니다. R에서는 <a href="https://rdrr.io/r/base/rawConversion.html" class="orm:hideurl"><code>charToRaw()</code></a>를 사용하여 문자열의 기본 표현을 얻을 수 있습니다.

```

charToRaw("Hadley")
#> [1] 48 61 64 6c 65 79

```

이러한 6개의 16진수(hexadecimal numbers) 각각은 하나의 문자를 나타냅니다. `48`은 H, `61`은 a 등입니다. 16진수를 문자로 매핑하는 것이 인코딩이며, 이 경우 이 인코딩을 ASCII라고 합니다. ASCII는 영문자를 훌륭하게 표현하는데, 왜냐하면 그것이 정보 교환을 위한 *미국* 표준 코드(American Standard Code for Information Interchange)이기 때문입니다.

영어 이외의 언어에 대해서는 상황이 그렇게 쉽지 않습니다. 컴퓨팅 초기에는 비영어권 문자를 인코딩하기 위해 경쟁하는 여러 표준이 있었습니다. 예를 들어, 유럽에는 서유럽 언어에 사용되는 Latin1(ISO-8859-1로도 알려짐)과 중부 유럽 언어에 사용되는 Latin2(ISO-8859-2로도 알려짐)라는 두 가지 서로 다른 인코딩이 있었습니다. Latin1에서 `b1` 바이트는 ±이지만, Latin2에서는 ą입니다! 다행히 오늘날에는 거의 모든 곳에서 지원되는 하나의 표준, 즉 UTF-8이 있습니다. UTF-8은 오늘날 인간이 사용하는 거의 모든 문자와 이모티콘 같은 많은 추가 기호를 인코딩할 수 있습니다.

readr은 어디에서나 UTF-8을 사용합니다. 이것은 좋은 기본값이지만, UTF-8을 사용하지 않는 구형 시스템에서 생성된 데이터에 대해서는 실패할 것입니다. 이런 일이 발생하면 인쇄할 때 문자열이 이상하게 보일 것입니다. 때로는 한두 글자만 깨질 수도 있고, 어떨 때는 완전히 알 수 없는 횡설수설(gibberish)로 표시될 수도 있습니다. 예를 들어, 다음은 인코딩이 특이한 두 개의 인라인 CSV입니다.<sup><a href="ch14.html#idm44771294565056" id="idm44771294565056-marker" data-type="noteref">7</a></sup>

```

x1 <- "text\nEl Ni\xf1o was particularly bad this year"
read_csv(x1)
#> # A tibble: 1 × 1
#> text  
#> <chr>  
#> 1 "El Ni\xf1o was particularly bad this year"

x2 <- "text\n\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd"
read_csv(x2)
#> # A tibble: 1 × 1
#> text  
#> <chr>  
#> 1 "\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd"

```

이것들을 올바르게 읽으려면 `locale` 인자를 통해 인코딩을 지정합니다.

```

read_csv(x1, locale = locale(encoding = "Latin1"))
#> # A tibble: 1 × 1
#> text  
#> <chr>  
#> 1 El Niño was particularly bad this year

read_csv(x2, locale = locale(encoding = "Shift-JIS"))
#> # A tibble: 1 × 1
#> text  
#> <chr>  
#> 1 こんにちは

```

올바른 인코딩을 어떻게 찾을까요? 운이 좋다면 데이터 문서 어딘가에 포함되어 있을 것입니다. 안타깝게도 그런 경우는 드물기 때문에 readr은 파악을 돕기 위해 <a href="https://readr.tidyverse.org/reference/encoding.html" class="orm:hideurl"><code>guess_encoding()</code></a>을 제공합니다. 이는 완벽하지 않으며 (여기서와 달리) 텍스트가 많을 때 더 잘 작동하지만 시작하기에는 합리적인 방법입니다. 올바른 것을 찾기 전에 몇 가지 다른 인코딩을 시도해 볼 것을 예상하세요.

인코딩은 방대하고 복잡한 주제입니다. 여기서는 수박 겉핥기만 했을 뿐입니다. 더 자세히 알고 싶다면 [상세한 설명](https://oreil.ly/v8ZQf)을 읽어보실 것을 권장합니다.

## 글자 변형 (Letter Variations)

악센트가 있는 언어에서 작업할 때 글자의 위치를 파악하는 것(<a href="https://stringr.tidyverse.org/reference/str_length.html" class="orm:hideurl"><code>str_length()</code></a> 및 <a href="https://stringr.tidyverse.org/reference/str_sub.html" class="orm:hideurl"><code>str_sub()</code></a> 사용)은 상당한 어려움을 제기합니다. 악센트가 있는 글자는 단일한 개별 문자(ü)로 인코딩될 수도 있고, 악센트가 없는 글자(u)와 발음 구별 기호(¨)를 결합하여 두 문자로 인코딩될 수도 있기 때문입니다. 예를 들어, 다음 코드는 똑같아 보이는 ü를 표현하는 두 가지 방법을 보여줍니다.

```

u <- c("\u00fc", "u\u0308")
str_view(u)
#> [1] │ ü
#> [2] │ ü

```

하지만 두 문자열은 길이가 다르며 첫 번째 문자도 다릅니다.

```

str_length(u)
#> [1] 1 2
str_sub(u, 1, 1)
#> [1] "ü" "u"

```

마지막으로, `==`를 사용하여 이러한 문자열을 비교하면 다른 문자열로 해석되는 반면, stringr의 편리한 <a href="https://stringr.tidyverse.org/reference/str_equal.html" class="orm:hideurl"><code>str_equal()</code></a> 함수는 두 문자열의 외형이 동일함을 인식한다는 점에 유의하세요.

```

u[[1]] == u[[2]]
#> [1] FALSE

str_equal(u[[1]], u[[2]])
#> [1] TRUE

```

## 로케일 의존 함수 (Locale-Dependent Functions)

마지막으로, *로케일(locale)*에 따라 동작이 달라지는 stringr 함수들이 몇 가지 있습니다. 로케일은 언어와 비슷하지만 언어 내 지역적 차이를 처리하기 위한 선택적 지역 지정자(region specifier)를 포함합니다. 로케일은 소문자 언어 약어로 지정되며, 선택적으로 `_`와 대문자 지역 식별자가 올 수 있습니다. 예를 들어 "en"은 영어, "en_GB"는 영국 영어, "en_US"는 미국 영어입니다. 언어 코드를 모르는 경우 [Wikipedia](https://oreil.ly/c1P2g)에 좋은 목록이 있으며, <a href="https://rdrr.io/pkg/stringi/man/stri_locale_list.html" class="orm:hideurl"><code>stringi::stri_locale_list()</code></a>를 보면 stringr에서 지원하는 로케일을 확인할 수 있습니다.

기본 R 문자열 함수는 운영체제에서 설정한 로케일을 자동으로 사용합니다. 이는 기본 R 문자열 함수가 여러분의 언어에서 예상하는 대로 작동하지만, 다른 국가에 사는 사람과 코드를 공유할 경우 다르게 작동할 수 있음을 의미합니다. 이 문제를 피하기 위해 stringr은 "en" 로케일을 사용하여 영어 규칙을 기본값으로 지정하며, 이를 재정의하려면 `locale` 인자를 지정해야 합니다. 다행히 대소문자 변경과 정렬이라는 두 가지 함수 세트에서는 로케일이 정말로 중요합니다.

대소문자를 변경하는 규칙은 언어마다 다릅니다. 예를 들어 튀르키예어(Turkish)에는 두 개의 i(점이 있는 것과 없는 것)가 있습니다. 두 글자는 서로 다른 글자이므로 다르게 대문자로 변환됩니다.

```

str_to_upper(c("i", "ı"))
#> [1] "I" "I"
str_to_upper(c("i", "ı"), locale = "tr")
#> [1] "İ" "I"

```

문자열 정렬은 알파벳 순서에 따라 다르며, 알파벳 순서는 모든 언어에서 동일하지 않습니다!<sup><a href="ch14.html#idm44771294301712" id="idm44771294301712-marker" data-type="noteref">8</a></sup> 예를 들어: 체코어에서 "ch"는 알파벳에서 `h` 뒤에 나타나는 합성 글자(compound letter)입니다.

```

str_sort(c("a", "c", "ch", "h", "z"))
#> [1] "a" "c" "ch" "h" "z"
str_sort(c("a", "c", "ch", "h", "z"), locale = "cs")
#> [1] "a" "c" "h" "ch" "z"

```

이는 <a href="https://dplyr.tidyverse.org/reference/arrange.html" class="orm:hideurl"><code>dplyr::arrange()</code></a>로 문자열을 정렬할 때도 발생하며, 이 때문에 그것 역시 `locale` 인자를 가집니다.

# 요약 (Summary)

이 장에서는 문자열을 만들고, 결합하고, 추출하는 방법과 같은 stringr 패키지의 강력한 기능 일부를 배웠고, 비영어권 문자열을 다룰 때 직면할 수 있는 몇 가지 문제점에 대해서도 배웠습니다. 이제 문자열 작업을 위한 가장 중요하고 강력한 도구 중 하나인 정규 표현식에 대해 알아볼 차례입니다. 정규 표현식은 문자열 내의 패턴을 기술하기 위한 간결하면서도 표현력이 풍부한 언어이며 다음 장의 주제입니다.

<sup>[1](ch14.html#idm44771296672656-marker)</sup> 또는 기본 R 함수인 <a href="https://rdrr.io/r/base/writeLines.html" class="orm:hideurl"><code>writeLines()</code></a>를 사용하세요.

<sup>[2](ch14.html#idm44771296567072-marker)</sup> R 4.0.0 이상부터 사용 가능합니다.

<sup>[3](ch14.html#idm44771296183072-marker)</sup> stringr을 사용하지 않는 경우 <a href="https://glue.tidyverse.org/reference/glue.html" class="orm:hideurl"><code>glue::glue()</code></a>를 사용하여 직접 액세스할 수도 있습니다.

<sup>[4](ch14.html#idm44771296091984-marker)</sup> 상응하는 기본 R은 `collapse` 인자와 함께 사용된 <a href="https://rdrr.io/r/base/paste.html" class="orm:hideurl"><code>paste()</code></a>입니다.

<sup>[5](ch14.html#idm44771295456640-marker)</sup> 동일한 원리가 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_position()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>에도 적용됩니다.

<sup>[6](ch14.html#idm44771294893344-marker)</sup> 이 항목들을 보면, babynames 데이터가 공백이나 하이픈을 버리고 15글자 뒤에서 자른(truncates) 것으로 짐작됩니다.

<sup>[7](ch14.html#idm44771294565056-marker)</sup> 여기서 저는 특수한 `\x`를 사용하여 이진 데이터(binary data)를 문자열에 직접 인코딩하고 있습니다.

<sup>[8](ch14.html#idm44771294301712-marker)</sup> 중국어처럼 알파벳이 없는 언어에서의 정렬은 훨씬 더 복잡합니다.
```
