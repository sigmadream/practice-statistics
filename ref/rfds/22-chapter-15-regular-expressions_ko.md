# 15장. 정규 표현식(Regular Expressions)

# 소개

<a href="ch14.html#chp-strings" data-type="xref">14장</a>에서는 문자열을 다루는 데 유용한 여러 가지 함수에 대해 배웠습니다. 이 장에서는 문자열 내의 패턴을 설명하기 위한 간결하고 강력한 언어인 *정규 표현식(regular expressions)*을 사용하는 함수에 중점을 둡니다. *regular expression*이라는 용어는 다소 길기 때문에 대부분의 사람들은 *regex*<sup><a href="ch15.html#idm44771294160112" id="idm44771294160112-marker" data-type="noteref">1</a></sup> 또는 *regexp*로 축약하여 부릅니다.

이 장은 정규 표현식의 기초와 데이터 분석에 가장 유용한 stringr 함수들로 시작합니다. 그런 다음 패턴에 대한 지식을 확장하여 일곱 가지 중요한 새로운 주제(이스케이핑(escaping), 앵커링(anchoring), 문자 클래스(character classes), 단축 클래스(shorthand classes), 수량자(quantifiers), 우선순위(precedence) 및 그룹화(grouping))를 다룰 것입니다. 다음으로, stringr 함수가 다룰 수 있는 다른 유형의 패턴들과 정규 표현식의 동작을 미세 조정할 수 있는 다양한 "플래그(flags)"에 대해 이야기할 것입니다. 마지막으로, tidyverse 및 기본 R에서 정규 표현식을 사용할 수 있는 다른 곳들을 살펴보며 마무리하겠습니다.

## 사전 준비

이 장에서는 tidyverse의 핵심 멤버인 stringr 및 tidyr의 정규 표현식 함수와, babynames 패키지의 데이터를 사용합니다.

```
library(tidyverse)
library(babynames)
```

이 장 전반에 걸쳐 기본 개념을 파악할 수 있는 간단한 인라인 예제와, 아기 이름(baby names) 데이터, 그리고 stringr의 세 가지 문자 벡터를 혼합하여 사용할 것입니다:

- `fruit`는 80개의 과일 이름을 포함합니다.
- `words`는 980개의 흔한 영어 단어를 포함합니다.
- `sentences`는 720개의 짧은 문장을 포함합니다.

# 패턴 기초

정규 표현식 패턴이 어떻게 작동하는지 배우기 위해 <a href="https://stringr.tidyverse.org/reference/str_view.html" class="orm:hideurl"><code>str_view()</code></a>를 사용할 것입니다. 이전 장에서 우리는 문자열과 그 출력 형태를 더 잘 이해하기 위해 <a href="https://stringr.tidyverse.org/reference/str_view.html" class="orm:hideurl"><code>str_view()</code></a>를 사용했었지만, 이제는 두 번째 인수로 정규 표현식을 전달하여 사용할 것입니다. 이것이 제공되면, <a href="https://stringr.tidyverse.org/reference/str_view.html" class="orm:hideurl"><code>str_view()</code></a>는 일치하는 문자열 벡터의 요소만 표시하고, 각 일치 항목을 `<>`로 둘러싸며, 가능한 경우 일치 항목을 파란색으로 강조 표시합니다.

가장 단순한 패턴은 해당 문자와 정확히 일치하는 문자와 숫자로 구성됩니다:

```
str_view(fruit, "berry")
#>  [6] │ bil<berry>
#>  [7] │ black<berry>
#> [10] │ blue<berry>
#> [11] │ boysen<berry>
#> [19] │ cloud<berry>
#> [21] │ cran<berry>
#> ... and 8 more
```

문자와 숫자는 정확히 일치하며 이를 *리터럴 문자(literal characters)*라고 합니다. `.`, `+`, `*`, `[`, `]`, `?`와 같은 대부분의 구두점 문자는 특별한 의미를 가지며<sup><a href="ch15.html#idm44771294089152" id="idm44771294089152-marker" data-type="noteref">2</a></sup> *메타문자(metacharacters)*라고 부릅니다. 예를 들어, `.`는 임의의 문자와 일치하므로,<sup><a href="ch15.html#idm44771294083376" id="idm44771294083376-marker" data-type="noteref">3</a></sup> `"a."`는 "a" 뒤에 다른 문자가 오는 모든 문자열과 일치합니다:

```
str_view(c("a", "ab", "ae", "bd", "ea", "eab"), "a.")
#> [2] │ <ab>
#> [3] │ <ae>
#> [6] │ e<ab>
```

또는 "a"로 시작해서 세 글자가 오고 그 뒤에 "e"가 오는 모든 과일을 찾을 수도 있습니다:

```
str_view(fruit, "a...e")
#>  [1] │ <apple>
#>  [7] │ bl<ackbe>rry
#> [48] │ mand<arine>
#> [51] │ nect<arine>
#> [62] │ pine<apple>
#> [64] │ pomegr<anate>
#> ... and 2 more
```

*수량자(Quantifiers)*는 패턴이 일치할 수 있는 횟수를 제어합니다:

- `?`는 패턴을 선택 사항으로 만듭니다 (즉, 0번 또는 1번 일치).
- `+`는 패턴이 반복되게 합니다 (즉, 적어도 1번 일치).
- `*`는 패턴을 선택 사항으로 만들거나 반복되게 합니다 (즉, 0번을 포함하여 임의의 횟수로 일치).

```
# ab?는 "a" 뒤에 "b"가 선택적으로 오는 것과 일치합니다.
str_view(c("a", "ab", "abb"), "ab?")
#> [1] │ <a>
#> [2] │ <ab>
#> [3] │ <ab>b

# ab+는 "a" 뒤에 적어도 하나의 "b"가 오는 것과 일치합니다.
str_view(c("a", "ab", "abb"), "ab+")
#> [2] │ <ab>
#> [3] │ <abb>

# ab*는 "a" 뒤에 임의의 개수의 "b"가 오는 것과 일치합니다.
str_view(c("a", "ab", "abb"), "ab*")
#> [1] │ <a>
#> [2] │ <ab>
#> [3] │ <abb>
```

*문자 클래스(Character classes)*는 `[]`로 정의되며 여러 문자 집합 중 하나와 일치하게 합니다; 예: `[abcd]`는 "a", "b", "c", "d" 중 하나와 일치합니다. `^`로 시작하여 일치를 반전시킬 수도 있습니다: `[^abcd]`는 "a", "b", "c", "d"를 *제외한* 모든 것과 일치합니다. 이 아이디어를 사용하여 모음으로 둘러싸인 "x"나 자음으로 둘러싸인 "y"를 포함하는 단어를 찾을 수 있습니다:

```
str_view(words, "[aeiou]x[aeiou]")
#> [284] │ <exa>ct
#> [285] │ <exa>mple
#> [288] │ <exe>rcise
#> [289] │ <exi>st
str_view(words, "[^aeiou]y[^aeiou]")
#> [836] │ <sys>tem
#> [901] │ <typ>e
```

*대체(alternation)* 기호인 `|`를 사용하여 하나 이상의 대안 패턴 중 하나를 선택할 수 있습니다. 예를 들어, 다음 패턴들은 "apple", "melon", "nut"을 포함하거나 모음이 반복되는 과일을 찾습니다:

```
str_view(fruit, "apple|melon|nut")
#>  [1] │ <apple>
#> [13] │ canary <melon>
#> [20] │ coco<nut>
#> [52] │ <nut>
#> [62] │ pine<apple>
#> [72] │ rock <melon>
#> ... and 1 more
str_view(fruit, "aa|ee|ii|oo|uu")
#>  [9] │ bl<oo>d orange
#> [33] │ g<oo>seberry
#> [47] │ lych<ee>
#> [66] │ purple mangost<ee>n
```

정규 표현식은 매우 압축적이고 구두점 문자를 많이 사용하므로 처음에는 압도적이고 읽기 어려워 보일 수 있습니다. 걱정하지 마세요: 연습을 통해 점차 나아질 것이며, 간단한 패턴들은 곧 자연스러워질 것입니다. 유용한 stringr 함수들로 연습하며 그 과정을 시작해 보겠습니다.

# 주요 함수

이제 정규 표현식의 기본을 이해했으므로, 이를 몇 가지 stringr 및 tidyr 함수들과 함께 사용해 보겠습니다. 다음 섹션에서는 일치 항목의 존재 여부를 감지하는 방법, 일치 항목의 개수를 세는 방법, 일치 항목을 고정된 텍스트로 바꾸는 방법, 그리고 패턴을 사용하여 텍스트를 추출하는 방법을 배울 것입니다.

## 일치 항목 감지

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>는 패턴이 문자 벡터의 요소와 일치하면 `TRUE`를, 그렇지 않으면 `FALSE`를 반환하는 논리 벡터를 반환합니다:

```
str_detect(c("a", "b", "c"), "[aeiou]")
#> [1]  TRUE FALSE FALSE
```

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>는 초기 벡터와 동일한 길이의 논리 벡터를 반환하기 때문에, <a href="https://dplyr.tidyverse.org/reference/filter.html" class="orm:hideurl"><code>filter()</code></a>와 잘 맞습니다. 예를 들어, 이 코드는 소문자 "x"를 포함하는 가장 인기 있는 이름들을 모두 찾습니다:

```
babynames |> 
  filter(str_detect(name, "x")) |> 
  count(name, wt = n, sort = TRUE)
#> # A tibble: 974 × 2
#>   name           n
#>   <chr>      <int>
#> 1 Alexander 665492
#> 2 Alexis    399551
#> 3 Alex      278705
#> 4 Alexandra 232223
#> 5 Max       148787
#> 6 Alexa     123032
#> # … with 968 more rows
```

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>를 <a href="https://rdrr.io/r/base/sum.html" class="orm:hideurl"><code>sum()</code></a>이나 <a href="https://rdrr.io/r/base/mean.html" class="orm:hideurl"><code>mean()</code></a>과 짝지어 <a href="https://dplyr.tidyverse.org/reference/summarise.html" class="orm:hideurl"><code>summarize()</code></a>와 함께 사용할 수도 있습니다: `sum(str_detect(x, pattern))`는 일치하는 관측치의 수를 알려주고, `mean(str_detect(x, pattern))`는 일치하는 비율을 알려줍니다. 예를 들어, 다음 코드는 연도별로 분류하여 "x"를 포함하는 아기 이름의 비율<sup><a href="ch15.html#idm44771293675344" id="idm44771293675344-marker" data-type="noteref">4</a></sup>을 계산하고 시각화합니다. 최근에 인기가 급격히 상승한 것 같네요!

```
babynames |> 
  group_by(year) |> 
  summarize(prop_x = mean(str_detect(name, "x"))) |> 
  ggplot(aes(x = year, y = prop_x)) + 
  geom_line()
```

![x라는 문자를 포함하는 아기 이름의 비율을 나타내는 시계열 그래프. 비율은 1880년에 1000명당 8명에서 1980년에 1000명당 4명으로 점차 감소하다가 2019년에 1000명당 16명으로 빠르게 증가합니다.](D:\sd\Practices\any2md\output\[2023] R for Data Science/assets/rds2_15in01.png)

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>와 밀접하게 관련된 두 가지 함수가 있습니다: <a href="https://stringr.tidyverse.org/reference/str_subset.html" class="orm:hideurl"><code>str_subset()</code></a>과 <a href="https://stringr.tidyverse.org/reference/str_which.html" class="orm:hideurl"><code>str_which()</code></a>. <a href="https://stringr.tidyverse.org/reference/str_subset.html" class="orm:hideurl"><code>str_subset()</code></a>은 일치하는 문자열만 포함하는 문자 벡터를 반환합니다. <a href="https://stringr.tidyverse.org/reference/str_which.html" class="orm:hideurl"><code>str_which()</code></a>는 일치하는 문자열의 위치를 나타내는 정수 벡터를 반환합니다.

## 일치 항목 수 세기

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>에서 한 단계 더 복잡해진 함수는 <a href="https://stringr.tidyverse.org/reference/str_count.html" class="orm:hideurl"><code>str_count()</code></a>입니다: 참이나 거짓이 아니라 각 문자열에서 몇 개의 일치 항목이 있는지 알려줍니다.

```
x <- c("apple", "banana", "pear")
str_count(x, "p")
#> [1] 2 0 1
```

각 일치 항목은 이전 일치 항목의 끝에서 시작한다는 점에 유의하세요; 즉, 정규 표현식의 일치 항목은 절대 겹치지 않습니다. 예를 들어, `"abababa"`에서 `"aba"` 패턴은 몇 번 일치할까요? 정규 표현식에 따르면 3번이 아니라 2번입니다:

```
str_count("abababa", "aba")
#> [1] 2
str_view("abababa", "aba")
#> [1] │ <aba>b<aba>
```

<a href="https://stringr.tidyverse.org/reference/str_count.html" class="orm:hideurl"><code>str_count()</code></a>를 <a href="https://dplyr.tidyverse.org/reference/mutate.html" class="orm:hideurl"><code>mutate()</code></a>와 함께 사용하는 것이 자연스럽습니다. 다음 예제는 각 이름의 모음과 자음의 개수를 세기 위해 문자 클래스와 함께 <a href="https://stringr.tidyverse.org/reference/str_count.html" class="orm:hideurl"><code>str_count()</code></a>를 사용합니다:

```
babynames |> 
  count(name) |> 
  mutate(
    vowels = str_count(name, "[aeiou]"),
    consonants = str_count(name, "[^aeiou]")
  )
#> # A tibble: 97,310 × 4
#>   name          n vowels consonants
#>   <chr>     <int>  <int>      <int>
#> 1 Aaban        10      2          3
#> 2 Aabha         5      2          3
#> 3 Aabid         2      2          3
#> 4 Aabir         1      2          3
#> 5 Aabriella     5      4          5
#> 6 Aada          1      2          2
#> # … with 97,304 more rows
```

자세히 살펴보면, 계산에 무언가 문제가 있다는 것을 알 수 있습니다: "Aaban"에는 세 개의 a가 있지만, 요약 결과에는 모음이 두 개만 보고됩니다. 그 이유는 정규 표현식이 대소문자를 구분하기 때문입니다. 이를 해결할 수 있는 세 가지 방법이 있습니다:

- 문자 클래스에 대문자 모음을 추가합니다: `str_count(name, "[aeiouAEIOU]")`.
- 정규 표현식에 대소문자를 무시하라고 알려줍니다: `str_count(name, regex("[aeiou]", ignore_case = TRUE))`. 자세한 내용은 <a href="#sec-flags" data-type="xref">“정규 표현식 플래그”</a>에서 논의하겠습니다.
- 이름을 소문자로 변환하기 위해 <a href="https://stringr.tidyverse.org/reference/case.html" class="orm:hideurl"><code>str_to_lower()</code></a>를 사용합니다: `str_count(str_to_lower(name), "[aeiou]")`.

문자열로 작업할 때 이러한 다양한 접근 방식은 꽤 일반적입니다. 패턴을 더 복잡하게 만들거나 문자열에 전처리를 수행하는 등 목표에 도달하는 여러 가지 방법이 종종 존재합니다. 한 가지 접근 방식을 시도하다 막힌다면, 방식을 바꿔서 다른 관점에서 문제에 접근해 보는 것이 종종 유용할 수 있습니다.

여기서는 이름에 두 가지 함수를 적용하고 있으므로, 먼저 변환하는 것이 더 쉽다고 생각합니다:

```
babynames |> 
  count(name) |> 
  mutate(
    name = str_to_lower(name),
    vowels = str_count(name, "[aeiou]"),
    consonants = str_count(name, "[^aeiou]")
  )
#> # A tibble: 97,310 × 4
#>   name          n vowels consonants
#>   <chr>     <int>  <int>      <int>
#> 1 aaban        10      3          2
#> 2 aabha         5      3          2
#> 3 aabid         2      3          2
#> 4 aabir         1      3          2
#> 5 aabriella     5      5          4
#> 6 aada          1      3          1
#> # … with 97,304 more rows
```

## 값 변경하기

일치 항목을 감지하고 개수를 세는 것 외에도, <a href="https://stringr.tidyverse.org/reference/str_replace.html" class="orm:hideurl"><code>str_replace()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_replace.html" class="orm:hideurl"><code>str_replace_all()</code></a>을 사용하여 일치 항목을 수정할 수도 있습니다. <a href="https://stringr.tidyverse.org/reference/str_replace.html" class="orm:hideurl"><code>str_replace()</code></a>는 첫 번째 일치 항목을 변경하고, 이름에서 유추할 수 있듯이 <a href="https://stringr.tidyverse.org/reference/str_replace.html" class="orm:hideurl"><code>str_replace_all()</code></a>은 모든 일치 항목을 변경합니다:

```
x <- c("apple", "pear", "banana")
str_replace_all(x, "[aeiou]", "-")
#> [1] "-ppl-"  "p--r"   "b-n-n-"
```

<a href="https://stringr.tidyverse.org/reference/str_remove.html" class="orm:hideurl"><code>str_remove()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_remove.html" class="orm:hideurl"><code>str_remove_all()</code></a>은 `str_replace(x, pattern, "")`를 위한 유용한 단축키입니다:

```
x <- c("apple", "pear", "banana")
str_remove_all(x, "[aeiou]")

입력에 출력의 한 셀에 해당하는 여러 행이 있는 경우 어떻게 되는지 궁금할 수도 있습니다. 다음 예에는 `id A` 및 `measurement bp1`에 해당하는 두 개의 행이 있습니다:

`df` `<-` `tribble``(` `~``id``,` `~``measurement``,` `~``value``,` `"A"``,` `"bp1"``,` `100``,` `"A"``,` `"bp1"``,` `102``,` `"A"``,` `"bp2"``,` `120``,` `"B"``,` `"bp1"``,` `140``,` `"B"``,` `"bp2"``,` `115` `)`

이 데이터를 피벗하려고 하면 리스트 열(list-columns)이 포함된 출력을 얻게 됩니다. 이에 대해서는 <a href="ch23.html#chp-rectangling" data-type="xref">23장</a>에서 자세히 알아볼 것입니다:

`df` `|>` `pivot_wider``(` `names_from` `=` `measurement``,` `values_from` `=` `value` `)` `` #> Warning: Values from `value` are not uniquely identified; output will contain `` `#> list-cols.` `` #> • Use `values_fn = list` to suppress this warning. `` `` #> • Use `values_fn = {summary_fun}` to summarise duplicates. `` `#> • Use the following dplyr code to identify duplicates.` `#> {data} %>%` `#> dplyr::group_by(id, measurement) %>%` `#> dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%` `#> dplyr::filter(n > 1L)` `#> # A tibble: 2 × 3` `#> id bp1 bp2 ` `#> <chr> <list> <list> ` `#> 1 A <dbl [2]> <dbl [1]>` `#> 2 B <dbl [1]> <dbl [1]>`

아직 이런 종류의 데이터로 작업하는 방법을 모르기 때문에, 어디에 문제가 있는지 파악하기 위해 경고 메시지의 힌트를 따르고 싶을 것입니다:

`df` `|>` `group_by``(``id``,` `measurement``)` `|>` `summarize``(``n` `=` `n``(),` `.groups` `=` `"drop"``)` `|>` `filter``(``n` `>` `1``)` `#> # A tibble: 1 × 3` `#> id measurement n` `#> <chr> <chr> <int>` `#> 1 A bp1 2`

그런 다음 데이터에 무엇이 잘못되었는지 파악하고 근본적인 손상을 복구하거나 그룹화 및 요약 기술을 사용하여 행과 열 값의 각 조합이 단일 행만 갖도록 보장하는 것은 여러분의 몫입니다.

# 요약 (Summary)

이 장에서는 변수가 열에 있고 관측치가 행에 있는 데이터인 정돈된 데이터(tidy data)에 대해 배웠습니다. 정돈된 데이터는 대부분의 함수에서 이해되는 일관된 구조이기 때문에 tidyverse에서의 작업을 더 쉽게 만듭니다. 주요 과제는 전달받은 어떤 구조의 데이터든 정돈된 형식으로 변환하는 것입니다. 이를 위해 많은 정돈되지 않은 데이터셋을 정돈할 수 있게 해주는 <a href="https://tidyr.tidyverse.org/reference/pivot_longer.html" class="orm:hideurl"><code>pivot_longer()</code></a>와 <a href="https://tidyr.tidyverse.org/reference/pivot_wider.html" class="orm:hideurl"><code>pivot_wider()</code></a>에 대해 배웠습니다. 여기서 제시한 예제는 <a href="https://tidyr.tidyverse.org/articles/pivot.html" class="orm:hideurl"><code>vignette("pivot", package = "tidyr")</code></a>에서 선택한 것이므로, 이 장에서 도움을 받지 못한 문제에 직면한다면 다음에 시도해 볼 좋은 자료가 바로 그 비네트(vignette)입니다.

또 다른 과제는, 주어진 데이터셋에 대해 더 길거나 넓은 버전을 "정돈된(tidy)" 것이라고 명명하는 것이 불가능할 수 있다는 점입니다. 이는 정돈된 데이터가 각 열에 하나의 변수를 가진다고 말했지만 실제로 변수가 무엇인지는 정의하지 않았던(그리고 정의하기가 놀랍도록 어렵습니다) 우리의 정돈된 데이터 정의를 일부 반영하는 것입니다. 분석을 가장 쉽게 만들어주는 것이라면 무엇이든 변수라고 말하는 등 실용적으로 접근해도 전혀 문제가 없습니다. 따라서 어떤 계산을 어떻게 수행할지 알아내는 데 막혔다면 데이터 구성을 전환하는 것을 고려해 보세요. 필요에 따라 정돈을 해제하고, 변환하고, 다시 정돈하는 것을 두려워하지 마세요!

이 장을 즐겁게 읽었고 기본 이론에 대해 더 알고 싶다면, *Journal of Statistical Software*에 게시된 [“Tidy Data” 논문](https://oreil.ly/86uxw)에서 그 역사와 이론적 토대에 대해 자세히 알아볼 수 있습니다.

이제 상당한 양의 R 코드를 작성하고 있으므로 코드를 파일 및 디렉터리로 구성하는 방법에 대해 더 자세히 알아볼 때입니다. 다음 장에서는 스크립트와 프로젝트의 이점과 여러분의 삶을 편하게 만들어 줄 많은 도구에 대해 모두 알아볼 것입니다.

<sup>[1](ch05.html#idm44771326722336-marker)</sup> 2000년 어느 시점에 탑 100에 포함되었고, 차트에 등장한 후 최대 72주까지 추적된 노래라면 포함됩니다.

<sup>[2](ch05.html#idm44771328141216-marker)</sup> 이 아이디어에 대해서는 <a href="ch18.html#chp-missing-values" data-type="xref">18장</a>에서 다시 다루겠습니다.
#> [20] │ <coco>nut
#> [22] │ <cucu>mber
#> [41] │ <juju>be
#> [56] │ <papa>ya
#> [73] │ s<alal> berry
```

이 패턴은 같은 두 글자 쌍으로 시작하고 끝나는 모든 단어를 찾습니다:

```
str_view(words, "^(..).*\\1$")
#> [152] │ <church>
#> [217] │ <decide>
#> [617] │ <photograph>
#> [699] │ <require>
#> [739] │ <sense>
```

<a href="https://stringr.tidyverse.org/reference/str_replace.html" class="orm:hideurl"><code>str_replace()</code></a>에서도 역참조를 사용할 수 있습니다. 예를 들어, 이 코드는 `sentences`의 두 번째 단어와 세 번째 단어의 순서를 바꿉니다:

```
sentences |> 
  str_replace("(\\w+) (\\w+) (\\w+)", "\\1 \\3 \\2") |> 
  str_view()
#> [1] │ The canoe birch slid on the smooth planks.
#> [2] │ Glue sheet the to the dark blue background.
#> [3] │ It's to easy tell the depth of a well.
#> [4] │ These a days chicken leg is a rare dish.
#> [5] │ Rice often is served in round bowls.
#> [6] │ The of juice lemons makes fine punch.
#> ... and 714 more
```

각 그룹에 대한 일치 항목을 추출하려면 <a href="https://stringr.tidyverse.org/reference/str_match.html" class="orm:hideurl"><code>str_match()</code></a>를 사용할 수 있습니다. 하지만 <a href="https://stringr.tidyverse.org/reference/str_match.html" class="orm:hideurl"><code>str_match()</code></a>는 행렬을 반환하므로 다루기가 그리 쉽지 않습니다:<sup><a href="ch15.html#idm44771292399008" id="idm44771292399008-marker" data-type="noteref">8</a></sup>

```
sentences |> 
  str_match("the (\\w+) (\\w+)") |> 
  head()
#>      [,1]                [,2]     [,3]    
#> [1,] "the smooth planks" "smooth" "planks"
#> [2,] "the sheet to"      "sheet"  "to"    
#> [3,] "the depth of"      "depth"  "of"    
#> [4,] NA                  NA       NA      
#> [5,] NA                  NA       NA      
#> [6,] NA                  NA       NA
```

이를 티블로 변환하고 열의 이름을 지정할 수 있습니다:

```
sentences |> 
  str_match("the (\\w+) (\\w+)") |> 
  as_tibble(.name_repair = "minimal") |> 
  set_names("match", "word1", "word2")
#> # A tibble: 720 × 3
#>   match             word1  word2 
#>   <chr>             <chr>  <chr> 
#> 1 the smooth planks smooth planks
#> 2 the sheet to      sheet  to    
#> 3 the depth of      depth  of    
#> 4 <NA>              <NA>   <NA>  
#> 5 <NA>              <NA>   <NA>  
#> 6 <NA>              <NA>   <NA>  
#> # … with 714 more rows
```

하지만 그러면 기본적으로 여러분만의 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a> 버전을 다시 만든 셈이 됩니다. 실제로 이면에서 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>는 패턴 벡터를 이름이 지정된 구성 요소를 캡처하기 위해 그룹화를 사용하는 단일 정규 표현식으로 변환합니다.

때때로 일치 그룹을 생성하지 않고 괄호를 사용하고 싶을 수 있습니다. 그럴 때는 `(?:)`를 사용하여 비캡처(noncapturing) 그룹을 생성할 수 있습니다.

```
x <- c("a gray cat", "a grey dog")
str_match(x, "gr(e|a)y")
#>      [,1]   [,2]
#> [1,] "gray" "a" 
#> [2,] "grey" "e"
str_match(x, "gr(?:e|a)y")
#>      [,1]  
#> [1,] "gray"
#> [2,] "grey"
```

## 연습문제

1. 리터럴 문자열 `"'\`와 어떻게 일치시킬 수 있을까요? `"$^$"`는 어떻게 할까요?

2. 다음 패턴들이 각각 왜 `\`와 일치하지 않는지 설명하세요: `"\"`, `"\\"`, `"\\\"`.

3. <a href="https://stringr.tidyverse.org/reference/stringr-data.html" class="orm:hideurl"><code>stringr::words</code></a>에 있는 흔한 단어들의 말뭉치(corpus)가 주어졌을 때, 다음에 해당하는 모든 단어를 찾는 정규 표현식을 작성하세요:

    1. "y"로 시작하는 단어.
    2. "y"로 시작하지 않는 단어.
    3. "x"로 끝나는 단어.
    4. 정확히 세 글자 길이인 단어. (<a href="https://stringr.tidyverse.org/reference/str_length.html" class="orm:hideurl"><code>str_length()</code></a>를 사용하여 속이지 마세요!)
    5. 7글자 이상인 단어.
    6. 모음-자음 쌍을 포함하는 단어.
    7. 최소 두 개의 모음-자음 쌍이 연속으로 포함된 단어.
    8. 오직 반복되는 모음-자음 쌍으로만 구성된 단어.

4. 다음 각 단어에 대해 영국식 또는 미국식 철자와 일치하는 11개의 정규 표현식을 만드세요: airplane/aeroplane, aluminum/aluminium, analog/analogue, ass/arse, center/centre, defense/defence, donut/doughnut, gray/grey, modeling/modelling, skeptic/sceptic, summarize/summarise. 가능한 가장 짧은 정규 표현식을 만들어 보세요!

5. `words`의 첫 글자와 마지막 글자를 바꾸세요. 변경된 문자열 중 여전히 `words`에 있는 것은 무엇인가요?

6. 다음 정규 표현식들이 무엇과 일치하는지 말로 설명하세요 (각 항목이 정규 표현식인지 정규 표현식을 정의하는 문자열인지 주의 깊게 읽어보세요):

    1. `^.*$`
    2. `"\\{.+\\}"`
    3. `\d{4}-\d{2}-\d{2}`
    4. `"\\\\{4}"`
    5. `\..\..\..`
    6. `(.)\1\1`
    7. `"(..)\\1"`

7. [초보자용 정규 표현식 십자말풀이](https://oreil.ly/Db3NF)를 풀어보세요.

# 패턴 제어

단순히 문자열을 사용하는 대신 패턴 객체를 사용하여 일치 항목의 세부 사항에 대해 추가적인 제어를 할 수 있습니다. 이를 통해 정규 표현식 플래그(regex flags)라는 것을 제어하고, 다음에 설명할 다양한 유형의 고정된 문자열과 일치시킬 수 있습니다.

## 정규 표현식 플래그

정규 표현식의 세부 사항을 제어하는 데 여러 가지 설정을 사용할 수 있습니다. 이러한 설정들은 다른 프로그래밍 언어에서 종종 *플래그(flags)*라고 불립니다. stringr에서는 패턴을 <a href="https://stringr.tidyverse.org/reference/modifiers.html" class="orm:hideurl"><code>regex()</code></a> 호출로 감싸서 이를 사용할 수 있습니다. 문자가 대문자나 소문자 형태 중 하나와 일치할 수 있게 해 주기 때문에 가장 유용한 플래그는 아마도 `ignore_case = TRUE`일 것입니다:

```
bananas <- c("banana", "Banana", "BANANA")
str_view(bananas, "banana")
#> [1] │ <banana>
str_view(bananas, regex("banana", ignore_case = TRUE))
#> [1] │ <banana>
#> [2] │ <Banana>
#> [3] │ <BANANA>
```

여러 줄로 된 문자열(즉, `\n`을 포함하는 문자열) 작업을 많이 한다면, `dotall`과 `multiline`도 유용할 수 있습니다:

- `dotall = TRUE`는 `.`이 `\n`을 포함한 모든 것과 일치하게 합니다:

  ```
  x <- "Line 1\nLine 2\nLine 3"
  str_view(x, ".Line")
  str_view(x, regex(".Line", dotall = TRUE))
  #> [1] │ Line 1<
  #>     │ Line> 2<
  #>     │ Line> 3
  ```

- `multiline = TRUE`는 `^`와 `$`가 전체 문자열의 시작과 끝이 아니라 각 줄의 시작과 끝과 일치하게 합니다:

  ```
  x <- "Line 1\nLine 2\nLine 3"
  str_view(x, "^Line")
  #> [1] │ <Line> 1
  #>     │ Line 2
  #>     │ Line 3
  str_view(x, regex("^Line", multiline = TRUE))
  #> [1] │ <Line> 1
  #>     │ <Line> 2
  #>     │ <Line> 3
  ```

마지막으로, 복잡한 정규 표현식을 작성 중이고 나중에 이해하지 못할까 봐 걱정된다면, `comments = TRUE`를 시도해 볼 수 있습니다. 이것은 패턴 언어를 조정하여 공백과 줄바꿈, 그리고 `#` 뒤의 모든 것을 무시하게 합니다. 이를 통해 주석과 공백을 사용하여 복잡한 정규 표현식을 더 이해하기 쉽게 만들 수 있습니다,<sup><a href="ch15.html#idm44771291954256" id="idm44771291954256-marker" data-type="noteref">9</a></sup> 다음 예제처럼 말입니다:

```
phone <- regex(
  r"(
    \(?     # optional opening parens
    (\d{3}) # area code
    [)\-]?  # optional closing parens or dash
    \ ?     # optional space
    (\d{3}) # another three numbers
    [\ -]?  # optional space or dash
    (\d{4}) # four more numbers
  )", 
  comments = TRUE
)

str_extract(c("514-791-8141", "(123) 456 7890", "123456"), phone)
#> [1] "514-791-8141"   "(123) 456 7890" NA
```

주석을 사용하면서 스페이스, 줄바꿈, 또는 `#`와 일치시키고 싶다면, `\`를 사용하여 이스케이프해야 합니다.

## 고정된 일치 항목

<a href="https://stringr.tidyverse.org/reference/modifiers.html" class="orm:hideurl"><code>fixed()</code></a>를 사용하여 정규 표현식 규칙에서 벗어날 수 있습니다:

```
str_view(c("", "a", "."), fixed("."))
#> [3] │ <.>
```

<a href="https://stringr.tidyverse.org/reference/modifiers.html" class="orm:hideurl"><code>fixed()</code></a>는 또한 대소문자를 무시하는 기능을 제공합니다:

```
str_view("x X", "X")
#> [1] │ x <X>
str_view("x X", fixed("X", ignore_case = TRUE))
#> [1] │ <x> <X>
```

영어가 아닌 텍스트로 작업하는 경우, 여러분이 지정한 `locale`에 사용되는 전체 대소문자 표기 규칙을 구현하는 <a href="https://stringr.tidyverse.org/reference/modifiers.html" class="orm:hideurl"><code>fixed()</code></a> 대신 <a href="https://stringr.tidyverse.org/reference/modifiers.html" class="orm:hideurl"><code>coll()</code></a>을 원할 수 있습니다. 로캘(locale)에 대한 자세한 내용은 <a href="ch14.html#sec-other-languages" data-type="xref">"비영어권 텍스트(Non-English Text)"</a>를 참고하세요.

```
str_view("i İ ı I", fixed("İ", ignore_case = TRUE))
#> [1] │ i <İ> ı I
str_view("i İ ı I", coll("İ", ignore_case = TRUE, locale = "tr"))
#> [1] │ <i> <İ> ı I
```

# 실습

이러한 아이디어들을 실제로 적용해 보기 위해, 다음으로 몇 가지 반쯤 실제와 같은 문제들을 해결해 보겠습니다. 세 가지 일반적인 기법에 대해 논의할 것입니다:

- 간단한 양성 및 음성 대조군(positive and negative controls)을 생성하여 작업 확인하기
- 정규 표현식을 불리언 대수(Boolean algebra)와 결합하기
- 문자열 조작을 사용하여 복잡한 패턴 생성하기

## 작업 확인하기

먼저 "The"로 시작하는 모든 문장을 찾아봅시다. `^` 앵커만 사용하는 것으로는 충분하지 않습니다:

```
str_view(sentences, "^The")
#>  [1] │ <The> birch canoe slid on the smooth planks.
#>  [4] │ <The>se days a chicken leg is a rare dish.
#>  [6] │ <The> juice of lemons makes fine punch.
#>  [7] │ <The> box was thrown beside the parked truck.
#>  [8] │ <The> hogs were fed chopped corn and garbage.
#> [11] │ <The> boy was there when the sun rose.
#> ... and 271 more
```

그 패턴은 `They`나 `These`와 같은 단어로 시작하는 문장과도 일치합니다. 우리는 "e"가 단어의 마지막 글자인지 확인해야 하며, 단어 경계를 추가하여 이를 수행할 수 있습니다:

```
str_view(sentences, "^The\\b")
#>  [1] │ <The> birch canoe slid on the smooth planks.
#>  [6] │ <The> juice of lemons makes fine punch.
#>  [7] │ <The> box was thrown beside the parked truck.
#>  [8] │ <The> hogs were fed chopped corn and garbage.
#> [11] │ <The> boy was there when the sun rose.
#> [13] │ <The> source of the huge river is the clear spring.
#> ... and 250 more
```

대명사로 시작하는 모든 문장을 찾는 것은 어떨까요?

```
str_view(sentences, "^She|He|It|They\\b")
#>  [3] │ <It>'s easy to tell the depth of a well.
#> [15] │ <He>lp the woman get back to her feet.
#> [27] │ <He>r purse was full of useless trash.
#> [29] │ <It> snowed, rained, and hailed the same morning.
#> [63] │ <He> ran half way to the hardware store.
#> [90] │ <He> lay prone and hardly moved a limb.
#> ... and 57 more
```

결과를 빠르게 살펴보면 일부 잘못된 일치 항목이 발생하고 있음을 알 수 있습니다. 그 이유는 우리가 괄호 사용을 잊었기 때문입니다:

```
str_view(sentences, "^(She|He|It|They)\\b")
#>   [3] │ <It>'s easy to tell the depth of a well.
#>  [29] │ <It> snowed, rained, and hailed the same morning.
#>  [63] │ <He> ran half way to the hardware store.
#>  [90] │ <He> lay prone and hardly moved a limb.
#> [116] │ <He> ordered peach pie with ice cream.
#> [127] │ <It> caught its hind paw in a rusty trap.
#> ... and 51 more
```

만약 그런 실수가 처음 몇 개의 일치 항목에서 발생하지 않았다면 어떻게 발견할 수 있었을지 궁금할 것입니다. 좋은 기법 중 하나는 몇 개의 양성 및 음성 일치 항목을 생성하여 이를 통해 여러분의 패턴이 예상대로 작동하는지 테스트하는 것입니다:

```
pos <- c("He is a boy", "She had a good time")
neg <- c("Shells come from the sea", "Hadley said 'It's a great day'")

pattern <- "^(She|He|It|They)\\b"
str_detect(pos, pattern)
#> [1] TRUE TRUE
str_detect(neg, pattern)
#> [1] FALSE FALSE
```

일반적으로 좋은 양성 예시를 생각해내는 것이 음성 예시를 생각해내는 것보다 훨씬 쉽습니다. 그 이유는 정규 표현식에서 여러분의 약점이 어디인지 예측할 수 있을 만큼 충분히 능숙해지기까지 시간이 걸리기 때문입니다. 그럼에도 불구하고 이는 여전히 유용합니다. 문제를 해결해 나가면서 천천히 자신의 실수 모음을 축적할 수 있고, 같은 실수를 두 번 다시 하지 않도록 보장할 수 있습니다.

## 불리언 연산(Boolean Operations)

자음만 포함된 단어를 찾고 싶다고 상상해 봅시다. 한 가지 방법은 모음을 제외한 모든 글자를 포함하는 문자 클래스(`[^aeiou]`)를 만든 다음, 그것이 임의의 수의 글자와 일치하도록 허용(`[^aeiou]+`)하고, 시작과 끝에 앵커를 고정하여 전체 문자열과 일치하도록 강제(`^[^aeiou]+$`)하는 것입니다:

```
str_view(words, "^[^aeiou]+$")
#> [123] │ <by>
#> [249] │ <dry>
#> [328] │ <fly>
#> [538] │ <mrs>
#> [895] │ <try>
#> [952] │ <why>
```

하지만 문제를 거꾸로 뒤집으면 이 문제를 조금 더 쉽게 만들 수 있습니다. 자음만 포함하는 단어를 찾는 대신, 모음이 전혀 포함되지 않은 단어를 찾을 수 있습니다:

```
str_view(words[!str_detect(words, "[aeiou]")])
#> [1] │ by
#> [2] │ dry
#> [3] │ fly
#> [4] │ mrs
#> [5] │ try
#> [6] │ why
```

이것은 특히 "and" 또는 "not"과 관련된 논리적 조합을 다룰 때마다 유용한 기법입니다. 예를 들어 "a"와 "b"를 모두 포함하는 단어를 찾고 싶다고 상상해 봅시다. 정규 표현식에는 "and" 연산자가 내장되어 있지 않으므로, "a" 뒤에 "b"가 오거나 "b" 뒤에 "a"가 오는 모든 단어를 찾는 식으로 해결해야 합니다:

```
str_view(words, "a.*b|b.*a")
#>  [2] │ <ab>le
#>  [3] │ <ab>out
#>  [4] │ <ab>solute
#> [62] │ <availab>le
#> [66] │ <ba>by
#> [67] │ <ba>ck
#> ... and 24 more
```

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>를 두 번 호출한 결과를 결합하는 것이 더 간단합니다:

```
words[str_detect(words, "a") & str_detect(words, "b")]
#>  [1] "able"      "about"     "absolute"  "available" "baby"      "back"     
#>  [7] "bad"       "bag"       "balance"   "ball"      "bank"      "bar"      
#> [13] "base"      "basis"     "bear"      "beat"      "beauty"    "because"  
#> [19] "black"     "board"     "boat"      "break"     "brilliant" "britain"  
#> [25] "debate"    "husband"   "labour"    "maybe"     "probable"  "table"
```

모든 모음을 포함하는 단어가 있는지 확인하고 싶다면 어떻게 해야 할까요? 패턴으로 이 작업을 수행한다면 `5!`(120)개의 서로 다른 패턴을 생성해야 합니다:

```
words[str_detect(words, "a.*e.*i.*o.*u")]
# ...
words[str_detect(words, "u.*o.*i.*e.*a")]
```

<a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a>를 다섯 번 호출하여 결합하는 것이 훨씬 간단합니다:

```
words[
  str_detect(words, "a") &
  str_detect(words, "e") &
  str_detect(words, "i") &
  str_detect(words, "o") &
  str_detect(words, "u")
]
#> character(0)
```

일반적으로 문제를 해결하는 단일 정규 표현식을 만드는 데 어려움을 겪고 있다면, 한 걸음 물러서서 문제를 더 작은 조각으로 나누고 각 과제를 해결한 다음 다음으로 넘어갈 수 있는지 생각해 보세요.

## 코드로 패턴 생성하기

색상을 언급하는 모든 `sentences`를 찾고 싶다면 어떻게 해야 할까요? 기본 아이디어는 간단합니다: 대체(alternation) 기호를 단어 경계와 결합하기만 하면 됩니다:

```
str_view(sentences, "\\b(red|green|blue)\\b")
#>   [2] │ Glue the sheet to the dark <blue> background.
#>  [26] │ Two <blue> fish swam in the tank.
#>  [92] │ A wisp of cloud hung in the <blue> air.
#> [148] │ The spot on the blotter was made by <green> ink.
#> [160] │ The sofa cushion is <red> and of light weight.
#> [174] │ The sky that morning was clear and bright <blue>.
#> ... and 20 more
```

하지만 색상의 수가 늘어나면 이 패턴을 손으로 구성하는 것은 금세 지루해질 것입니다. 색상을 벡터에 저장할 수 있다면 좋지 않을까요?

```
rgb <- c("red", "green", "blue")
```

글쎄요, 가능합니다! <a href="https://stringr.tidyverse.org/reference/str_c.html" class="orm:hideurl"><code>str_c()</code></a>와 <a href="https://stringr.tidyverse.org/reference/str_flatten.html" class="orm:hideurl"><code>str_flatten()</code></a>을 사용하여 벡터로부터 패턴을 생성하기만 하면 됩니다:

```
str_c("\\b(", str_flatten(rgb, "|"), ")\\b")
#> [1] "\\b(red|green|blue)\\b"
```

좋은 색상 목록이 있다면 이 패턴을 더 포괄적으로 만들 수 있습니다. 시작할 수 있는 곳 중 하나는 R이 플롯(plot)에 사용할 수 있는 내장 색상 목록입니다:

```
str_view(colors())
#> [1] │ white
#> [2] │ aliceblue
#> [3] │ antiquewhite
#> [4] │ antiquewhite1
#> [5] │ antiquewhite2
#> [6] │ antiquewhite3
#> ... and 651 more
```

하지만 먼저 번호가 매겨진 변형들을 제거해 봅시다:

```
cols <- colors()
cols <- cols[!str_detect(cols, "\\d")]
str_view(cols)
#> [1] │ white
#> [2] │ aliceblue
#> [3] │ antiquewhite
#> [4] │ aquamarine
#> [5] │ azure
#> [6] │ beige
#> ... and 137 more
```

그런 다음 이것을 하나의 거대한 패턴으로 바꿀 수 있습니다. 패턴이 너무 크기 때문에 여기에 표시하지는 않겠지만, 작동하는 것을 볼 수는 있습니다:

```
pattern <- str_c("\\b(", str_flatten(cols, "|"), ")\\b")
str_view(sentences, pattern)
#>   [2] │ Glue the sheet to the dark <blue> background.
#>  [12] │ A rod is used to catch <pink> <salmon>.
#>  [26] │ Two <blue> fish swam in the tank.
#>  [66] │ Cars and busses stalled in <snow> drifts.
#>  [92] │ A wisp of cloud hung in the <blue> air.
#> [112] │ Leaves turn <brown> and <yellow> in the fall.
#> ... and 57 more
```

이 예제에서 `cols`는 숫자와 문자만 포함하므로 메타문자에 대해 걱정할 필요가 없습니다. 하지만 일반적으로 기존 문자열에서 패턴을 생성할 때는, 리터럴과 일치하도록 <a href="https://stringr.tidyverse.org/reference/str_escape.html" class="orm:hideurl"><code>str_escape()</code></a>를 통해 실행하는 것이 현명합니다.

## 연습문제

1. 다음 각 과제에 대해 단일 정규 표현식을 사용하는 방법과 다중 <a href="https://stringr.tidyverse.org/reference/str_detect.html" class="orm:hideurl"><code>str_detect()</code></a> 호출의 조합을 사용하는 방법 모두를 시도해 보세요:

    1. `x`로 시작하거나 끝나는 모든 `words` 찾기.
    2. 모음으로 시작하고 자음으로 끝나는 모든 `words` 찾기.
    3. 서로 다른 각 모음을 최소한 하나씩 모두 포함하는 `words`가 있나요?

2. "c 뒤를 제외하고는 e 앞에 i(i before e except after c)" 규칙에 대한 찬성과 반대의 증거를 찾기 위한 패턴을 구성하세요.

3. <a href="https://rdrr.io/r/grDevices/colors.html" class="orm:hideurl"><code>colors()</code></a>는 "lightgray" 및 "darkblue"와 같은 많은 수식어를 포함합니다. 이러한 수식어들을 어떻게 자동으로 식별할 수 있을까요? (수식어가 붙은 색상들을 감지한 다음 제거하는 방법에 대해 생각해 보세요.)

4. 어떤 기본 R 데이터셋이든 찾는 정규 표현식을 작성하세요. <a href="https://rdrr.io/r/utils/data.html" class="orm:hideurl"><code>data()</code></a> 함수의 특별한 사용을 통해 이러한 데이터셋 목록을 얻을 수 있습니다: `data(package = "datasets")$results[, "Item"]`. 많은 옛날 데이터셋들이 개별 벡터라는 점에 유의하세요; 이것들은 괄호 안에 그룹화된 "데이터 프레임(data frame)"의 이름을 포함하므로, 이것들을 떼어내야 할 것입니다.

# 다른 곳에서의 정규 표현식

stringr 및 tidyr 함수에서와 마찬가지로 R에서 정규 표현식을 사용할 수 있는 곳은 많이 있습니다. 다음 섹션에서는 더 넓은 tidyverse와 기본 R에 있는 다른 유용한 함수들을 설명합니다.

## Tidyverse

정규 표현식을 사용하고 싶을 수 있는 특별히 유용한 세 군데의 다른 곳이 있습니다:

- `matches(pattern)`는 이름이 제공된 패턴과 일치하는 모든 변수를 선택합니다. 이것은 변수를 선택하는 모든 tidyverse 함수(예: <a href="https://dplyr.tidyverse.org/reference/select.html" class="orm:hideurl"><code>select()</code></a>, <a href="https://dplyr.tidyverse.org/reference/rename.html" class="orm:hideurl"><code>rename_with()</code></a> 및 <a href="https://dplyr.tidyverse.org/reference/across.html" class="orm:hideurl"><code>across()</code></a>)에서 사용할 수 있는 "tidyselect" 함수입니다.

- `pivot_longer()`의 `names_pattern` 인수는 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>separate_wider_regex()</code></a>처럼 정규 표현식 벡터를 받습니다. 복잡한 구조를 가진 변수 이름에서 데이터를 추출할 때 유용합니다.

- <a href="https://tidyr.tidyverse.org/reference/separate_longer_delim.html" class="orm:hideurl"><code>separate_longer_delim()</code></a> 및 <a href="https://tidyr.tidyverse.org/reference/separate_wider_delim.html" class="orm:hideurl"><code>sepa⁠rate_​wider_delim()</code></a>의 `delim` 인수는 보통 고정된 문자열과 일치하지만, <a href="https://stringr.tidyverse.org/reference/modifiers.html" class="orm:hideurl"><code>regex()</code></a>를 사용하여 패턴과 일치시킬 수 있습니다. 이것은 예를 들어, 선택적으로 공백이 뒤따르는 쉼표와 일치시키고 싶을 때, 즉 `regex(", ?")`를 사용할 때 유용합니다.

## Base R

`apropos(pattern)`는 전역 환경(global environment)에서 사용할 수 있는 주어진 패턴과 일치하는 모든 객체를 검색합니다. 이것은 함수 이름이 잘 기억나지 않을 때 유용합니다:

```
apropos("replace")
#> [1] "%+replace%"       "replace"          "replace_na"      
#> [4] "setReplaceMethod" "str_replace"      "str_replace_all" 
#> [7] "str_replace_na"   "theme_replace"
```

`list.files(path, pattern)`는 `path`에 있는 파일 중 정규 표현식 `pattern`과 일치하는 모든 파일을 나열합니다. 예를 들어, 다음과 같이 현재 디렉터리의 모든 R Markdown 파일을 찾을 수 있습니다:

```
head(list.files(pattern = "\\.Rmd$"))
#> character(0)
```

기본 R에서 사용하는 패턴 언어는 stringr에서 사용하는 것과 약간 다르다는 점에 주목할 가치가 있습니다. stringr은 [stringi 패키지](https://oreil.ly/abQNx) 위에 구축되어 있고, 이는 다시 [ICU 엔진](https://oreil.ly/A9Gbl) 위에 구축되어 있기 때문입니다. 반면 기본 R 함수들은 `perl = TRUE`를 설정했는지 여부에 따라 [TRE 엔진](https://oreil.ly/yGQ5U) 또는 [PCRE 엔진](https://oreil.ly/VhVuy)을 사용합니다. 다행히도 정규 표현식의 기초는 아주 잘 확립되어 있어서, 이 책에서 배울 패턴으로 작업할 때 변형을 겪는 일이 거의 없을 것입니다. 복잡한 유니코드 문자 범위나 `(?…)` 구문을 사용하는 특별한 기능 같은 고급 기능에 의존하기 시작할 때만 그 차이를 인지하면 됩니다.

# 요약

잠재적으로 과부하될 정도로 많은 의미를 가진 모든 구두점 문자를 사용하므로, 정규 표현식은 현존하는 가장 압축적인 언어 중 하나입니다. 처음에는 확실히 헷갈리지만, 정규 표현식을 읽도록 눈을 훈련시키고 이를 이해하도록 뇌를 훈련시키면 R 및 다른 많은 곳에서 사용할 수 있는 강력한 기술을 얻게 됩니다.

이 장에서 여러분은 가장 유용한 stringr 함수와 정규 표현식 언어의 가장 중요한 구성 요소를 배움으로써 정규 표현식 마스터가 되기 위한 여정을 시작했습니다. 그리고 더 배울 수 있는 리소스도 많이 있습니다.

시작하기 좋은 곳은 <a href="https://stringr.tidyverse.org/articles/regular-expressions.html" class="orm:hideurl"><code>vignette("regular-expressions", package = "stringr")</code></a>입니다. 이 문서는 stringr에서 지원하는 전체 구문 집합을 문서화합니다. 또 다른 유용한 참고 자료는 [*https://oreil.ly/MVwoC*](https://oreil.ly/MVwoC)입니다. R에 국한된 것은 아니지만, 정규 표현식의 가장 고급 기능들과 그 기능이 내부적으로 어떻게 작동하는지 배우는 데 사용할 수 있습니다.

stringr가 Marek Gagolewski의 stringi 패키지 위에 구현되어 있다는 것을 알아두는 것도 좋습니다. 만약 stringr에서 원하는 작업을 수행하는 함수를 찾는 데 어려움을 겪고 있다면, 주저하지 말고 stringi를 살펴보세요. stringi는 stringr과 동일한 규칙을 많이 따르기 때문에 쉽게 익힐 수 있을 것입니다.

다음 장에서는 문자열과 밀접하게 관련된 데이터 구조인 팩터(factors)에 대해 이야기할 것입니다. 팩터는 R에서 범주형 데이터, 즉 문자열 벡터로 식별되는 고정되고 알려진 가능한 값들의 집합을 가진 데이터를 나타내는 데 사용됩니다.

<sup>[1](ch15.html#idm44771294160112-marker)</sup> 하드 g("reg-x")나 소프트 g("rej-x") 발음 중 하나로 발음할 수 있습니다.

<sup>[2](ch15.html#idm44771294089152-marker)</sup> 이러한 특별한 의미를 이스케이프하는 방법은 <a href="#sec-regexp-escaping" data-type="xref">"이스케이핑(Escaping)"</a>에서 배울 것입니다.

<sup>[3](ch15.html#idm44771294083376-marker)</sup> 글쎄요, `\n`을 제외한 모든 문자입니다.

<sup>[4](ch15.html#idm44771293675344-marker)</sup> 이것은 "x"를 포함하는 *이름*의 비율을 알려줍니다; x를 포함하는 이름을 가진 아기의 비율을 원한다면 가중 평균을 수행해야 할 것입니다.

<sup>[5](ch15.html#idm44771293255808-marker)</sup> 실제 생활에서는 이렇게 이상한 것을 절대 볼 수 없을 거라고 안심시켜 드리고 싶지만, 안타깝게도 커리어를 쌓는 과정에서 훨씬 더 이상한 것을 보게 될 가능성이 높습니다!

<sup>[6](ch15.html#idm44771293083536-marker)</sup> 메타문자의 전체 집합은 `.^$\|*+?{}[]()`입니다.

<sup>[7](ch15.html#idm44771292559200-marker)</sup> 기억하세요, `\d` 또는 `\s`를 포함하는 정규 표현식을 만들려면 문자열에 대해 `\`를 이스케이프해야 하므로 `"\\d"` 또는 `"\\s"`라고 입력해야 합니다.

<sup>[8](ch15.html#idm44771292399008-marker)</sup> 주로 이 책에서는 행렬에 대해 전혀 논의하지 않기 때문입니다!

<sup>[9](ch15.html#idm44771291954256-marker)</sup> 우리가 여기서 사용하는 것처럼 `comments = TRUE`는 원시 문자열(raw string)과 결합할 때 특히 효과적입니다.
