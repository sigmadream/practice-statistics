---
title: "벡터: 정규 표현식"
---

```{r}
#| echo: false
source("_common.R")
```

> 문자열 안의 패턴을 표현하는 간결하고 강력한 언어, 정규 표현식(regular expressions)을 사용하는 함수를 중점적으로 다룹니다. 정규 표현식이라는 말은 길고 복잡해서 대부분 "regex"나 "regexp"로 줄여 부릅니다.

먼저 정규 표현식의 기본과 데이터 분석에 유용한 stringr 함수를 살펴봅니다. 이어서 이스케이핑(escaping), 앵커링(anchoring), 문자 클래스(character classes), 단축 클래스(shorthand classes), 수량자(quantifiers), 우선순위(precedence), 그룹화(grouping)라는 일곱 주제로 패턴 지식을 넓힙니다. stringr 함수가 처리하는 다른 유형의 패턴과 정규 표현식의 동작을 미세 조정하는(tweak) 여러 "플래그(flags)"도 다룹니다. 마지막에는 tidyverse와 기본 R에서 정규 표현식을 사용하는 다른 곳을 살펴봅니다.

```{r}
#| label: setup
#| message: false
library(tidyverse)
library(babynames)
```

## 패턴의 기초

정규 표현식 패턴의 작동 방식은 `str_view()`로 배웁니다. 앞에서는 문자열과 출력 표현을 이해할 때 썼지만 이제 두 번째 인수에 정규 표현식을 넣습니다. `str_view()`는 문자열 벡터에서 일치하는 요소만 표시하고 각 일치 항목을 `<>`로 감싸며 가능하면 파란색으로 강조합니다.

단순한 패턴은 해당 문자와 정확히 일치하는 글자와 숫자로 구성됩니다.

```{r}
str_view(fruit, "berry")
```

글자와 숫자는 그대로 일치하며 리터럴 문자(literal characters)라고 합니다. `.`, `+`, `*`, `[`, `]`, `?` 같은 구두점(punctuation) 문자는 대부분 특별한 의미가 있어 메타문자(metacharacters)라고 부릅니다. 예를 들어 `.`은 아무 문자와 일치하므로 `"a."`는 "a" 뒤에 다른 문자가 오는 모든 문자열과 일치합니다. 단, `\n`은 제외합니다.

```{r}
str_view(c("a", "ab", "ae", "bd", "ea", "eab"), "a.")
```

"a" 뒤에 세 글자와 "e"가 차례로 오는 모든 과일도 찾습니다.

```{r}
str_view(fruit, "a...e")
```

수량자(Quantifiers)는 패턴의 일치 횟수를 제어합니다.

1. `?`는 패턴을 선택 사항(optional)으로 만듭니다 (즉, 0번 또는 1번 일치함).
2. `+`는 패턴을 반복합니다(즉, 한 번 이상 일치함).
3. `*`는 패턴을 선택 사항으로 만들거나 반복합니다(즉, 0번을 포함하여 임의의 횟수만큼 일치함).

```{r}
# ab?는 "a"와 일치하고 선택적으로 "b"가 뒤따릅니다.
str_view(c("a", "ab", "abb"), "ab?")

# ab+는 "a"와 일치하고 최소 한 개의 "b"가 뒤따릅니다.
str_view(c("a", "ab", "abb"), "ab+")

# ab*는 "a"와 일치하고 임의의 개수의 "b"가 뒤따릅니다.
str_view(c("a", "ab", "abb"), "ab*")
```

문자 클래스(Character classes)는 `[]`로 정의하며 일련의(set of) 문자와 일치합니다. `[abcd]`는 "a", "b", "c", "d" 가운데 하나와 일치합니다. `^`로 시작하면 일치를 반전(invert)합니다. `[^abcd]`는 네 문자를 제외한 모든 것과 일치합니다. 이 원리로 모음(vowels)에 둘러싸인 "x"나 자음(consonants)에 둘러싸인 "y"가 포함된 단어를 찾습니다.

```{r}
str_view(words, "[aeiou]x[aeiou]")
str_view(words, "[^aeiou]y[^aeiou]")
```

교체(alternation) 기호 `|`는 여러 대안(alternative) 패턴 가운데 하나를 고릅니다. 다음 패턴은 "apple", "melon", "nut"이 들어 있거나 모음이 반복되는 과일을 찾습니다.

```{r}
str_view(fruit, "apple|melon|nut")
str_view(fruit, "aa|ee|ii|oo|uu")
```

정규 표현식은 매우 간결하고(compact) 구두점 문자를 많이 써서 처음에는 압도적이고(overwhelming) 읽기 어려워 보입니다. 걱정하지 마세요. 연습하면 익숙해지고 간단한 패턴은 곧 제2의 천성(second nature)이 됩니다. 유용한 stringr 함수부터 연습해 보겠습니다.

## 핵심 함수

정규 표현식의 기초를 익혔으니(got the basics under your belt) 몇 가지 stringr·tidyr 함수와 함께 사용해 봅시다. 일치 항목을 감지(detect)하고 개수를 세며(count) 고정된 텍스트로 바꾸고(replace) 패턴에 맞는 텍스트를 추출(extract)하는 방법을 배웁니다.

### 일치 항목 감지

`str_detect()`는 패턴이 문자형 벡터의 요소와 일치하면 `TRUE`를, 그렇지 않으면 `FALSE`를 나타내는 논리형 벡터를 반환합니다.

```{r}
str_detect(c("a", "b", "c"), "[aeiou]")
```

`str_detect()`는 입력 벡터와 길이가 같은 논리형 벡터를 반환하므로 `filter()`와 잘 맞습니다(pairs well). 다음 코드는 소문자 "x"가 들어간 인기 이름을 모두 찾습니다.

```{r}
babynames |>
    filter(str_detect(name, "x")) |>
    count(name, wt = n, sort = TRUE)
```

`str_detect()`는 `sum()`이나 `mean()`과 짝지어 `summarize()`에서도 사용합니다. `sum(str_detect(x, pattern))`은 일치하는 관측치 수를, `mean(str_detect(x, pattern))`은 일치 비율을 구합니다.

다음 스니펫은 "x"가 들어간 아기 이름의 비율을 연도별로 계산하고 시각화합니다. 최근 인기가 급격히 상승했습니다(radically increased)!

```{r}
#| fig-alt: |
#|   문자 x가 포함된 아기 이름의 비율을 보여주는 시계열. 
#|   이 비율은 1880년 1000명당 8명에서 1980년 1000명당 4명으로 
#|   점진적으로 감소(declines)하다가 2019년에는 1000명당 16명으로 빠르게 증가합니다.
babynames |>
    group_by(year) |>
    summarize(prop_x = mean(str_detect(name, "x"))) |>
    ggplot(aes(x = year, y = prop_x)) +
    geom_line()
```

1. `str_detect()`와 밀접하게(closely) 관련된 함수는 `str_subset()`과 `str_which()`입니다.
2. `str_subset()`은 일치하는 문자열만 담은 문자형 벡터를 반환합니다.
3. `str_which()`는 일치하는 문자열의 위치를 담은 정수형 벡터를 반환합니다.

### 일치 항목 개수 세기

`str_count()`는 `str_detect()`보다 한 단계 더 복잡합니다(next step up in complexity). 참과 거짓 대신 각 문자열의 일치 항목 수를 반환합니다.

```{r}
x <- c("apple", "banana", "pear")
str_count(x, "p")
```

각 일치 항목은 이전 일치 항목이 끝난 곳에서 시작합니다. 즉, 정규 표현식 일치 항목은 결코 겹치지 않습니다(overlap). 예를 들어 `"abababa"`에서 `"aba"` 패턴이 몇 번이나 일치할까요? 정규 표현식은 3개가 아니라 2개라고 말합니다.

```{r}
str_count("abababa", "aba")
str_view("abababa", "aba")
```

`str_count()`는 `mutate()`와 자연스럽게 어울립니다.
다음 예제에서는 문자 클래스와 `str_count()`를 함께 써서 각 이름의 모음과 자음 수를 셉니다.

```{r}
babynames |>
    count(name) |>
    mutate(
        vowels = str_count(name, "[aeiou]"),
        consonants = str_count(name, "[^aeiou]")
    )
```

자세히 살펴보면(look closely) 계산이 조금 이상합니다(off). "Aaban"에는 "a"가 3개 있지만 요약에는 모음이 2개만 나옵니다.
정규 표현식은 대소문자를 구분하기(case sensitive) 때문입니다.
이 문제를 해결할(fix) 방법은 세 가지입니다.

-   문자 클래스에 대문자 모음을 추가합니다. `str_count(name, "[aeiouAEIOU]")`.
-   정규 표현식에 대소문자를 무시하도록 지시(Tell)합니다. `str_count(name, regex("[aeiou]", ignore_case = TRUE))`.
-   `str_to_lower()`로 이름을 소문자로 바꿉니다. `str_count(str_to_lower(name), "[aeiou]")`.

문자열 작업에서 이런 다양한 접근 방식은 꽤 전형적입니다(typical) --- 패턴을 더 복잡하게 만들거나 문자열을 전처리하는 등 목표에 도달하는(reach) 방법이 여러 가지입니다.
한 가지 방식에서 막히면(get stuck) 다른 관점(perspective)으로 전환해(switch gears) 문제를 다루는(tackle) 편이 좋습니다.

여기서는 이름에 함수 두 개를 적용하므로 먼저 변환하는 편이 쉽습니다.

```{r}
babynames |>
    count(name) |>
    mutate(
        name = str_to_lower(name),
        vowels = str_count(name, "[aeiou]"),
        consonants = str_count(name, "[^aeiou]")
    )
```

### 값 바꾸기

일치 항목을 감지하고 세는 데 그치지 않고 `str_replace()`와 `str_replace_all()`로 수정(modify)합니다. `str_replace()`는 첫 번째 일치 항목을, 이름에서 알 수 있듯이(as the name suggests) `str_replace_all()`은 모든 일치 항목을 바꿉니다.

```{r}
x <- c("apple", "pear", "banana")
str_replace_all(x, "[aeiou]", "-")
```

`str_remove()` 및 `str_remove_all()`은 `str_replace(x, pattern, "")`의 유용한 단축키(handy shortcuts)입니다.

```{r}
x <- c("apple", "pear", "banana")
str_remove_all(x, "[aeiou]")
```

이 함수들은 데이터를 정리할 때 `mutate()`와 자연스럽게 짝을 이룹니다. 일관성 없는 형식(inconsistent formatting)을 차례로 걷어내려고(peel off layers) 반복해서 적용하기도 합니다.

### 변수 추출

마지막으로 살펴볼 `separate_wider_regex()`는 정규 표현식으로 한 열의 데이터를 새 열 하나 이상으로 추출합니다. `separate_wider_position()`과 `separate_wider_delim()`의 동료(peer)입니다. 개별 벡터가 아니라 데이터 프레임의 열에서 작동하므로 tidyr에 속합니다(live).

작동 방식을 확인하려고 간단한 데이터셋을 만들겠습니다.

다음은 `babynames`에서 가져온 데이터로 여러 사람의 이름, 성별, 나이가 꽤 이상한 형식으로 들어 있습니다.

```{r}
df <- tribble(
    ~str              ,
    "<Sheryl>-F_34"   ,
    "<Kisha>-F_45"    ,
    "<Brandon>-N_33"  ,
    "<Sharon>-F_38"   ,
    "<Penny>-F_58"    ,
    "<Justin>-M_41"   ,
    "<Patricia>-F_84" ,
)
```

`separate_wider_regex()`로 데이터를 추출하려면 각 조각과 일치하는 정규 표현식 시퀀스(sequence)를 구성(construct)합니다. 출력에 나타낼 조각(contents)에는 이름을 지정합니다.

```{r}
df |>
    separate_wider_regex(
        str,
        patterns = c(
            "<",
            name = "[A-Za-z]+",
            ">-",
            gender = ".",
            "_",
            age = "[0-9]+"
        )
    )
```

일치에 실패하면 `separate_wider_delim()`이나 `separate_wider_position()`처럼 `too_few = "debug"`로 잘못된 부분을 파악합니다(figure out).

### 연습 문제

1. 많은 모음을 가진 아기 이름은 무엇입니까? 높은 비율의 모음을 가진 이름은 무엇입니까? (힌트: 분모(denominator)는 무엇입니까?)

2. `"a/b/c/d/e"`의 모든 슬래시(forward slashes)를 백슬래시로 바꾸세요. 모든 백슬래시를 슬래시로 바꾸어 변환을 실행 취소(undo)하려고 시도하면 어떻게 됩니까? (이 문제는 곧 논의합니다.)

3. `str_replace_all()`을 사용하여 `str_to_lower()`의 간단한 버전을 구현하세요.

4. 여러분의 국가에서 일반적으로 작성되는 전화번호와 일치하는 정규 표현식을 만드세요.

## 패턴 세부 정보

패턴 언어의 기초와 일부 stringr·tidyr 함수에서 사용하는 방법을 익혔으니 이제 세부 사항을 더 살펴보겠습니다(dig into).

여기서 쓰는 용어는 각 구성 요소의 기술적인 이름입니다. 목적을 바로 떠올리게 하는(evocative) 이름은 아니지만 나중에 구글에서 자세한 내용을 검색할 때 큰 도움이 됩니다.

### 이스케이핑

리터럴 `.`을 일치시키려면 메타문자를 리터럴하게(literally) 처리하라고 정규 표현식에 지시하는 이스케이프(escape)가 필요합니다. 문자열처럼 정규 표현식도 백슬래시로 이스케이프합니다. 따라서 `.`을 일치시키려면 정규 표현식 `\.`가 필요하지만 여기서 문제가 생깁니다.

정규 표현식은 문자열로 나타내는데 `\`는 문자열에서도 이스케이프 기호입니다. 따라서 정규 표현식 `\.`를 만들려면 다음 예제처럼 문자열 `"\\."`가 필요합니다.

```{r}
# 정규 표현식 \.를 만들려면 \\.를 사용해야 합니다.
dot <- "\\."

# 그러나 표현식 자체에는 \가 하나만 포함되어 있습니다.
str_view(dot)

# 그리고 이것은 R에게 명시적인(explicit) .을 찾도록 지시합니다.
str_view(c("abc", "a.c", "bef"), "a\\.c")
```

이 책에서는 보통 `\.`처럼 정규 표현식을 따옴표 없이 적습니다. 실제 입력할 내용을 강조할(emphasize) 때는 따옴표로 묶고 `"\\."`처럼 이스케이프를 추가합니다.

정규 표현식에서 `\`가 이스케이프 문자라면 리터럴 `\`는 어떻게 일치시킬까요? 다시 이스케이프해 정규 표현식 `\\`를 만들어야 합니다.

그 정규 표현식을 문자열로 만들 때는 `\`를 또 이스케이프해야 합니다. 리터럴 `\` 하나와 일치시키려면 `"\\\\"`라고 써야 하므로 백슬래시 4개가 필요합니다!

```{r}
x <- "a\\b"
str_view(x)
str_view(x, "\\\\")
```

원시 문자열을 쓰면 더 쉽습니다. 이스케이핑 계층(layer) 하나를 피하기 때문입니다.

```{r}
str_view(x, r"{\\}")
```

리터럴 `.`, `$`, `|`, `*`, `+`, `?`, `{`, `}`, `(`, `)`와 일치할 때는 백슬래시 대신 문자 클래스를 쓰는 대안(alternative)도 있습니다.

```{r}
str_view(c("abc", "a.c", "a*c", "a c"), "a[.]c")
str_view(c("abc", "a.c", "a*c", "a c"), ".[*]c")
```

### 앵커

기본적으로 정규 표현식은 문자열의 어느 부분과도 일치합니다. 시작에서 일치할 때는 `^`, 끝에서 일치할 때는 `$`로 정규 표현식을 고정(anchor)합니다.

```{r}
str_view(fruit, "^a")
str_view(fruit, "a$")
```

달러 금액 표기 때문에(how) `$`가 문자열의 시작과 일치한다고 생각하기 쉽지만(tempting) 정규 표현식의 의미는 다릅니다.

정규 표현식을 전체 문자열과 강제로(force) 일치시키려면 `^`와 `$`를 모두 사용해 고정(anchor)하세요.

```{r}
str_view(fruit, "apple")
str_view(fruit, "^apple$")
```

단어의 시작이나 끝인 경계(boundary)는 `\b`로 일치시킵니다. `summarize`, `summary`, `rowsum` 등을 제외하고 `sum()`만 모두 찾으려면 `\bsum\b`를 검색하면 됩니다.

```{r}
x <- c("summary(x)", "summarize(df)", "rowsum(x)", "sum(x)")
str_view(x, "sum")
str_view(x, "\\bsum\\b")
```

단독으로 사용할 때 앵커는 너비가 없는 일치 항목(zero-width match)을 생성합니다.

```{r}
str_view("abc", c("$", "^", "\\b"))
```

독립적인 앵커(standalone anchor)를 바꿀 때(replace) 일어나는 일을 이해하는 데 도움이 됩니다.

```{r}
str_replace_all("abc", c("$", "^", "\\b"), "--")
```

### 문자 클래스

문자 클래스(character class), 또는 문자 집합(set)은 집합 안의 모든 문자와 일치합니다. 앞에서 살펴본 것처럼 `[]`로 고유한 집합을 구성합니다. `[abc]`는 "a", "b", "c" 가운데 하나와 일치하고 `[^abc]`는 세 문자를 제외한 모든 문자와 일치합니다. `^` 외에도 `[]:` 안에서 특별한 의미를 지닌 문자가 두 개 있습니다.

- `-`는 범위를 정의합니다. 예: `[a-z]`는 소문자 글자와 일치하고 `[0-9]`는 모든 숫자와 일치합니다.
- `\`는 특수 문자를 이스케이프하므로 `[\^\-\]]`는 `^`, `-` 또는 `]`와 일치합니다.

다음은 몇 가지 예제입니다.

```{r}
x <- "abcd ABCD 12345 -!@#%."
str_view(x, "[abc]+")
str_view(x, "[a-z]+")
str_view(x, "[^a-z0-9]+")

# 그렇지 않으면 [] 내에서 특별한 의미를 갖는 문자를 일치시키려면
# 이스케이프가 필요합니다.
str_view("a-b-c", "[a-c]")
str_view("a-b-c", "[a\\-c]")
```

일부 문자 클래스는 자주 사용되어 고유한 단축키(shortcut)가 있습니다.
줄 바꿈(newline)을 제외한 모든 문자와 일치하는 `.`는 이미 살펴봤습니다.
특히 유용한 단축키가 세 쌍(pairs) 더 있습니다.

`\d`나 `\s`가 들어간 정규 표현식을 문자열로 만들 때는 `\`를 이스케이프해야 합니다. 따라서 `"\\d"` 또는 `"\\s"`를 입력합니다.

1. `\d`는 임의의 숫자와 일치합니다; `\D`는 숫자가 아닌 모든 것과 일치합니다.

2. `\s`는 임의의 공백(스페이스, 탭, 줄 바꿈)과 일치합니다; `\S`는 공백이 아닌 모든 것과 일치합니다.

3. `\w`는 임의의 "단어(word)" 문자(즉, 글자와 숫자)와 일치합니다; `\W`는 임의의 "단어 아님(non-word)" 문자와 일치합니다.

다음 코드는 글자, 숫자, 구두점 문자(selection)로 여섯 단축키를 시연합니다(demonstrates).

```{r}
x <- "abcd ABCD 12345 -!@#%."
str_view(x, "\\d+")
str_view(x, "\\D+")
str_view(x, "\\s+")
str_view(x, "\\S+")
str_view(x, "\\w+")
str_view(x, "\\W+")
```

### 수량자

수량자(Quantifiers)는 패턴의 일치 횟수를 제어합니다. 앞에서 `?`(0개 또는 1개 일치), `+`(1개 이상 일치), `*`(0개 이상 일치)를 배웠습니다. `colou?r`은 미국식이나 영국식 철자(spelling)와 일치하고 `\d+`는 숫자 하나 이상과 일치하며 `\s?`는 공백 하나와 선택적으로 일치합니다.
`{}`를 사용하여 일치 횟수를 정확하게 지정할 수도 있습니다.

1. `{n}`은 정확히 n번 일치합니다.
2. `{n,}`은 최소 n번 일치합니다.
3. `{n,m}`은 n번에서 m번 사이로 일치합니다.

### 연산자 우선순위와 괄호

1. `ab+`는 무엇과 일치합니까?: "a" 다음에 "b"가 하나 이상 오는 것을 일치시킵니까, 아니면 "ab"가 임의의 횟수만큼 반복되는 것을 일치시킵니까?
2. `^a|b$`는 무엇과 일치합니까?: 전체 문자열 a 또는 전체 문자열 b와 일치합니까, 아니면 a로 시작하는 문자열 또는 b로 끝나는 문자열과 일치합니까?

답은 학창 시절에 배웠을 법한 PEMDAS나 BEDMAS 규칙과 비슷한 연산자 우선순위(operator precedence)에 따라 결정됩니다.

`*`의 우선순위가 `+`보다 높아서 `a + b * c`는 `(a + b) * c`가 아니라 `a + (b * c)`와 같습니다.

이와 유사하게 정규 표현식에는 고유한 우선순위 규칙이 있습니다. 수량자는 우선순위가 높고 교체(alternation)는 우선순위가 낮습니다. 즉, `ab+`는 `a(b+)`와 같고 `^a|b$`는 `(^a)|(b$)`와 같습니다.

대수학(algebra)처럼 괄호로 일반적인 순서를 바꿉니다. 정규 표현식의 우선순위 규칙은 기억하기 어려우므로(unlikely to remember) 괄호를 자유롭게(liberally) 사용하세요.

### 그룹화 및 캡처

괄호에는 연산자 우선순위를 바꾸는 것 외에도 중요한 효과가 있습니다. 일치 항목의 하위 구성 요소(sub-components)를 사용하는 캡처 그룹(capturing groups)을 만듭니다.

캡처 그룹은 먼저 후방 참조(back reference)에 사용합니다. 일치 항목 안에서 그룹을 다시 참조하는(refer back) 방법입니다. `\1`은 첫 번째 괄호의 일치 항목을, `\2`는 두 번째 괄호의 일치 항목을 나타냅니다(refers to).
예를 들어, 다음 패턴은 쌍이 반복되는 문자를 가진 모든 과일을 찾습니다.

```{r}
str_view(fruit, "(..)\\1")
```

그리고 다음 패턴은 동일한 문자 쌍으로 시작하고 끝나는 모든 단어를 찾습니다.

```{r}
str_view(words, "^(..).*\\1$")
```

`str_replace()`에서 후방 참조를 사용할 수도 있습니다. 예를 들어 이 코드는 `sentences`에서 두 번째 단어와 세 번째 단어의 순서를 바꿉니다(switches).

```{r}
sentences |>
    str_replace("(\\w+) (\\w+) (\\w+)", "\\1 \\3 \\2") |>
    str_view()
```

각 그룹의 일치 항목은 `str_match()`로 추출합니다.
다만 행렬(matrix)을 반환해서 다루기가 쉽지는 않습니다.

```{r}
sentences |>
    str_match("the (\\w+) (\\w+)") |>
    head()
```

티블(tibble)로 변환한 뒤 열의 이름을 지정합니다.

```{r}
sentences |>
    str_match("the (\\w+) (\\w+)") |>
    as_tibble(.name_repair = "minimal") |>
    set_names("match", "word1", "word2")
```

하지만 이는 사실상 자신만의 `separate_wider_regex()`를 다시 만드는 일입니다. 실제로 내부에서(behind the scenes) `separate_wider_regex()`는 패턴 벡터를 정규 표현식 하나로 바꿉니다. 이 표현식은 그룹화로 명명된 구성 요소를 캡처합니다.

일치 그룹을 만들지 않고 괄호만 써야 할 때도 있습니다. `(?:)`로 캡처하지 않는 그룹(non-capturing group)을 만듭니다.

```{r}
x <- c("a gray cat", "a grey dog")
str_match(x, "gr(e|a)y")
str_match(x, "gr(?:e|a)y")
```

### 연습 문제 (Exercises)

1. 리터럴 문자열 `"'\`를 어떻게 일치시키시겠습니까? `"$^$"`는 어떻습니까?

2. 이러한 각 패턴이 `\`와 일치하지 않는 이유를 설명하세요. `"\"`, `"\\"`, `"\\\"`.

3. `stringr::words`에 있는 일반적인 단어 말뭉치(corpus)가 주어지면 다음을 수행하는 모든 단어를 찾는 정규 표현식을 만드세요.

    a. "y"로 시작하는 단어
    b. "y"로 시작하지 않는 단어
    c. "x"로 끝나는 단어
    d. 길이가 정확히 세 글자인 단어 (`str_length()`를 사용하여 부정행위를 하지 마세요!)
    e. 일곱 글자 이상인 단어
    f. 모음-자음 쌍(vowel-consonant pair)이 포함된 단어
    g. 모음-자음 쌍이 연속으로(in a row) 최소 두 개 포함된 단어
    h. 반복되는 모음-자음 쌍으로만 구성된 단어

4. 다음 각 단어에 대한 영국식 또는 미국식 철자와 일치하는 11개의 정규 표현식을 만드세요. `airplane/aeroplane`, `aluminum/aluminium`, `analog/analogue`, `ass/arse`, `center/centre`, `defense/defence`, `donut/doughnut`, `gray/grey`, `modeling/modelling`, `skeptic/sceptic`, `summarize/summarise`. 가능한 한 짧은 정규 표현식을 만들어 보세요!

5. `words`의 첫 번째 글자와 마지막 글자를 바꾸세요. 그중 아직 `words`에 있는 문자열은 무엇입니까?

6. 다음 정규 표현식이 일치하는 대상을 말로 설명하세요. (각 항목이 정규 표현식인지 정규 표현식을 정의하는 문자열인지 주의 깊게 읽어보세요.)

    a. `^.*$`
    b. `"\\{.+\\}"`
    c. `\d{4}-\d{2}-\d{2}`
    d. `"\\\\{4}"`
    e. `\..\..\..`
    f. `(.)\1\1`
    g. `"(..)\\1"`

7. [https://regexcrossword.com/challenges/beginner](https://regexcrossword.com/challenges/beginner)에서 초보자용 정규 표현식 십자말풀이(crosswords)를 풀어보세요.

## 패턴 제어

단순한 문자열 대신 패턴 객체를 사용하면 일치 항목의 세부 사항을 더 세밀하게 제어합니다. 이른바 정규 표현식 플래그를 설정하고 아래 설명처럼 여러 유형의 고정 문자열을 일치시킵니다.

### 정규 표현식 플래그 

정규 표현식의 세부 사항을 제어하는 설정이 여러 가지 있습니다. 다른 프로그래밍 언어에서는 흔히 플래그(flags)라고 부릅니다. stringr에서는 패턴을 `regex()` 호출로 감싸서(wrapping) 사용합니다.

가장 유용한 플래그는 아마 `ignore_case = TRUE`일 것입니다. 문자가 대문자와 소문자 형식에 모두 일치합니다.

```{r}
bananas <- c("banana", "Banana", "BANANA")
str_view(bananas, "banana")
str_view(bananas, regex("banana", ignore_case = TRUE))
```

여러 줄(multiline) 문자열, 곧 `\n`이 들어간 문자열을 자주 다룬다면 `dotall`과 `multiline`도 유용합니다.

- `dotall = TRUE`는 `.`이 `\n`을 포함한 모든 것과 일치하게 합니다.

```{r}
x <- "Line 1\nLine 2\nLine 3"
str_view(x, ".Line")
str_view(x, regex(".Line", dotall = TRUE))
```

- `multiline = TRUE`는 `^` 및 `$`가 전체 문자열의 시작과 끝이 아니라 각 줄의 시작과 끝과 일치하게 합니다.

```{r}
x <- "Line 1\nLine 2\nLine 3"
str_view(x, "^Line")
str_view(x, regex("^Line", multiline = TRUE))
```

복잡한 정규 표현식을 나중에 이해하지 못할까 걱정된다면 `comments = TRUE`를 사용해 보세요. 패턴 언어가 공백(spaces), 줄 바꿈(new lines), `#` 뒤의 모든 내용을 무시하도록 조정됩니다. 다음 예제처럼 주석과 공백을 넣으면 복잡한 정규 표현식도 이해하기 쉬워집니다.

```{r}
phone <- regex(
    r"(
    \(?     # 선택적 여는 괄호(optional opening parens)
    (\d{3}) # 지역 번호(area code)
    [)\-]?  # 선택적 닫는 괄호 또는 대시(optional closing parens or dash)
    \ ?     # 선택적 공백(optional space)
    (\d{3}) # 또 다른 숫자 세 개(another three numbers)
    [\ -]?  # 선택적 공백 또는 대시(optional space or dash)
    (\d{4}) # 네 개의 숫자 더(four more numbers)
  )",
    comments = TRUE
)

str_extract(c("514-791-8141", "(123) 456 7890", "123456"), phone)
```

주석을 사용하고 공백, 줄 바꿈 또는 `#`을 일치시키려면 `\`로 이스케이프해야 합니다.

### 고정 일치

`fixed()`를 사용하면 정규 표현식 규칙을 적용하지 않습니다(opt-out).

```{r}
str_view(c("", "a", "."), fixed("."))
```

`fixed()`는 대소문자를 무시하는 기능도 지원합니다.

```{r}
str_view("x X", "X")
str_view("x X", fixed("X", ignore_case = TRUE))
```

비영어권 텍스트를 다룰 때는 `fixed()` 대신 `coll()`이 알맞습니다. 지정한 `locale`의 대문자 표기 규칙 전체를 구현하기(implements) 때문입니다.

```{r}
str_view("i İ ı I", fixed("İ", ignore_case = TRUE))
str_view("i İ ı I", coll("İ", ignore_case = TRUE, locale = "tr"))
```

## 실습

이 아이디어를 실천에 옮기려고(To put into practice) 반-실제적인(semi-authentic) 문제를 몇 가지 풀어보겠습니다. 여기서는 세 가지 일반적인 기법을 다룹니다.

1. 단순한 긍정 및 부정 대조군(positive and negative controls)을 만들어 작업 검사하기
2. 정규 표현식을 부울 대수(Boolean algebra)와 결합하기
3. 문자열 조작을 사용하여 복잡한 패턴 만들기

### 작업 검사하기 (Check your work)

먼저 "The"로 시작하는 모든 문장을 찾아봅시다. `^` 앵커만 사용하는 것으로는 충분하지 않습니다.

```{r}
str_view(sentences, "^The")
```

해당 패턴은 `They`나 `These` 같은 단어로 시작하는 문장과도 일치합니다. 단어 경계(word boundary)를 추가해 "e"가 단어의 마지막 글자인지 확인해야 합니다.

```{r}
str_view(sentences, "^The\\b")
```

대명사(pronoun)로 시작하는 모든 문장을 찾는 것은 어떨까요?

```{r}
str_view(sentences, "^She|He|It|They\\b")
```

결과를 빠르게 검사(inspection)하면 거짓 일치(spurious matches)가 몇 가지 보입니다(shows). 괄호를 빠뜨렸기 때문입니다.

```{r}
str_view(sentences, "^(She|He|It|They)\\b")
```

처음 몇 개의 일치 항목에는 나타나지 않는 실수(mistake)를 어떻게 발견할까요(spot)? 긍정·부정 일치 항목을 몇 가지 만들어 패턴이 예상대로 작동하는지 테스트하면 좋습니다.

```{r}
pos <- c("He is a boy", "She had a good time")
neg <- c("Shells come from the sea", "Hadley said 'It's a great day'")

pattern <- "^(She|He|It|They)\\b"
str_detect(pos, pattern)
str_detect(neg, pattern)
```

보통(typically) 부정적인 예보다 좋은 긍정적인 예를 떠올리기가 훨씬 쉽습니다. 정규 표현식의 약점을 예측할 만큼 능숙해지려면 시간이 꽤(a while) 걸리기 때문입니다. 그래도 두 예 모두 유용합니다. 문제를 풀며 실수 사례를 천천히 축적하면(accumulate) 같은 실수를 두 번 하지 않게 됩니다(ensuring).

### 부울 연산

자음만 들어 있는 단어를 찾는다고 해봅시다. 한 가지 방법은 모음을 제외한 모든 문자(`[^aeiou]`)의 문자 클래스를 만들고 임의의 개수와 일치(`[^aeiou]+`)시킨 뒤 시작과 끝에 고정해(anchoring) 전체 문자열과 강제로 일치(`^[^aeiou]+$`)시키는 것입니다.

```{r}
str_view(words, "^[^aeiou]+$")
```

하지만 문제를 뒤집으면(flipping the problem around) 조금 더 쉬워집니다. 자음만 들어 있는 단어 대신 모음이 전혀 없는 단어를 찾으면 됩니다.

```{r}
str_view(words[!str_detect(words, "[aeiou]")])
```

이 기법은 논리적 조합(logical combinations), 특히 "AND(그리고)"나 "NOT(아님)"이 들어간 조합을 다룰 때 유용합니다. "a"와 "b"가 모두 들어 있는 단어를 찾는 경우를 생각해 보세요.

정규 표현식에는 "and" 연산자가 없습니다. 따라서 "a" 다음에 "b"가 오거나 "b" 다음에 "a"가 오는 단어를 모두 찾는 방식으로 접근해야 합니다(tackle).

```{r}
str_view(words, "a.*b|b.*a")
```

두 번의 `str_detect()` 호출 결과를 결합하는 것이 더 간단합니다.

```{r}
words[str_detect(words, "a") & str_detect(words, "b")]
```

모든 모음이 들어 있는 단어를 찾는다면 어떨까요? 패턴을 사용하면 5!(120)개의 서로 다른 패턴이 필요합니다.

```{r}
#| results: false
words[str_detect(words, "a.*e.*i.*o.*u")]
# ...
words[str_detect(words, "u.*o.*i.*e.*a")]
```

다섯 번의 `str_detect()` 호출을 결합하는 것이 훨씬 더 간단합니다.

```{r}
words[
    str_detect(words, "a") &
        str_detect(words, "e") &
        str_detect(words, "i") &
        str_detect(words, "o") &
        str_detect(words, "u")
]
```

일반적으로(In general) 정규 표현식 하나로 문제를 해결하다 막히면 한 걸음 물러서세요(take a step back). 문제를 더 작은 조각으로 나누고(break the problem down) 각 과제를 해결한 뒤 다음 단계로 넘어갑니다.

### 코드로 패턴 만들기

색상(color)을 언급하는(mention) 모든 `sentences`는 어떻게 찾을까요? 기본 원리는 간단합니다. 교체(alternation)와 단어 경계를 결합하면 됩니다.

```{r}
str_view(sentences, "\\b(red|green|blue)\\b")
```

하지만 색상 수가 늘어나면(grows) 패턴을 손으로(by hand) 구성하는 일이 금세 지루해집니다(tedious). 색상을 벡터에 저장해 보겠습니다.

```{r}
rgb <- c("red", "green", "blue")
```

`str_c()`와 `str_flatten()`을 사용하면 벡터에서 패턴을 만듭니다.

```{r}
str_c("\\b(", str_flatten(rgb, "|"), ")\\b")
```

좋은 색상 목록이 있다면 패턴을 더 포괄적으로(comprehensive) 만듭니다. R의 내장(built-in) 플롯 색상 목록에서 시작해 보겠습니다.

```{r}
str_view(colors())
```

하지만 먼저 숫자가 포함된 변형(variants)을 제거(eliminate)합시다.

```{r}
cols <- colors()
cols <- cols[!str_detect(cols, "\\d")]
str_view(cols)
```

그런 다음 하나의 거대한 패턴으로 바꿉니다. 너무 커서 여기에 표시되지는 않지만 작동 결과는 확인합니다.

```{r}
pattern <- str_c("\\b(", str_flatten(cols, "|"), ")\\b")
str_view(sentences, pattern)
```

이 예제의 `cols`에는 숫자와 글자만 있어서 메타문자를 걱정할 필요가 없습니다. 하지만 기존(existing) 문자열로 패턴을 만들 때는 `str_escape()`에 패턴을 통과시켜 문자 그대로(literally) 일치하는지 확인하는 편이 좋습니다(wise).

### 연습 문제

1. 다음 각 과제를 단일 정규 표현식과 여러 번의 `str_detect()` 호출 조합, 두 방식으로 모두 풀어보세요.

    a. `x`로 시작하거나 끝나는 모든 `words`를 찾으세요.
    b. 모음으로 시작하고 자음으로 끝나는 모든 `words`를 찾으세요.
    c. 서로 다른 모음을 각각 최소 한 개씩 포함하는 `words`가 있습니까?

2. "c 뒤에 오는 경우를 제외하고 e 앞에 i(i before e except after c)"라는 규칙을 뒷받침하는(for) 근거(evidence)와 반대되는(against) 근거를 찾는 패턴을 구성(Construct)하세요.

3. `colors()`에는 "lightgray" 및 "darkblue"와 같은 많은 수식어(modifiers)가 포함되어 있습니다. 이러한 수식어를 어떻게 자동으로 식별합니까? (수식된 색상을 감지(detect)한 다음 제거(remove)하는 방법을 생각해 보세요.)

4. 기본 R 데이터셋을 찾는 정규 표현식을 만드세요. `data(package = "datasets")$results[, "Item"]`이라는 `data()` 함수의 특수한 사용법을 통해 이러한 데이터셋 목록을 얻을 수 있습니다. 여러 오래된 데이터셋은 개별 벡터입니다; 이들은 괄호 안에 그룹화 "데이터 프레임"의 이름을 포함하므로 이를 벗겨내야(strip off) 한다는 점에 유의하세요.

## 다른 곳에서의 정규 표현식

stringr와 tidyr 함수 외에도 R의 여러 곳에서 정규 표현식을 사용합니다.

### tidyverse

정규 표현식이 특히 유용한 곳이 세 군데 더 있습니다.

- `matches(pattern)`는 이름이 주어진 패턴과 일치하는 모든 변수를 선택합니다. 변수를 선택하는 모든 tidyverse 함수(`select()`, `rename_with()`, `across()`)에서 사용하는 "tidyselect" 함수입니다.

- `pivot_longer()`의 `names_pattern` 인수는 `separate_wider_regex()`와 마찬가지로 정규 표현식의 벡터를 취합니다. 복잡한 구조를 가진 변수 이름에서 데이터를 추출할 때 유용합니다.

- `separate_longer_delim()`과 `separate_wider_delim()`의 `delim` 인수는 보통 고정 문자열과 일치하지만 `regex()`를 사용하면 패턴과 일치합니다. 예를 들어 공백이 선택적으로 뒤따르는 쉼표, 곧 `regex(", ?")`를 일치시킬 때 유용합니다.

### 기본 R

`apropos(pattern)`는 전역 환경의 객체 가운데 주어진 패턴과 일치하는 것을 모두 검색합니다. 함수 이름이 잘 떠오르지 않을 때 유용합니다.

```{r}
apropos("replace")
```

`list.files(path, pattern)`는 `path`의 파일 가운데 정규 표현식 `pattern`과 일치하는 것을 모두 나열합니다. 다음과 같이 현재 디렉터리의 R 마크다운 파일을 모두 찾습니다.

```{r}
head(list.files(pattern = "\\.Rmd$"))
```

기본 R과 stringr의 패턴 언어는 조금 다릅니다. stringr은 ICU 엔진([https://unicode-org.github.io/icu/userguide/strings/regexp.html](https://unicode-org.github.io/icu/userguide/strings/regexp.html)) 위에 구축된 stringi 패키지([https://stringi.gagolewski.com](https://stringi.gagolewski.com))를 사용합니다. 반면 기본 R 함수는 `perl = TRUE` 설정 여부에 따라 TRE 엔진([https://github.com/laurikari/tre](https://github.com/laurikari/tre))이나 PCRE 엔진([https://www.pcre.org](https://www.pcre.org))을 사용합니다.

다행히 정규 표현식의 기초는 잘 정립되어 있어서 이 책의 패턴으로 작업할 때는 차이를 거의 만나지 않습니다. 복잡한 유니코드 문자 범위나 `(?…)` 구문을 쓰는 특수 기능처럼 고급 기능에 의존할 때만 차이를 인지하면 됩니다.

<!-- HUMANIZE-SUMMARY
원본 글자수: 23,256자
윤문본 글자수: 21,432자
변경률: 11.2% (마크업 제외, verify_change_rate.py)

카테고리별 탐지 건수(before → after):
- A-1 "~에 대해" 번역투: 7 → 0
- A-7 가지다 직역: 0 → 0
- A-10 가능 표현 남발: 36 → 0
- A-11 목적절 남발: 5 → 0
- A-15 본문 추상 주어·만능 동사: 1 → 0
- C-11 연결어미 뒤 쉼표: 4 → 0

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
- "정규 표현식이라는 용어는 발음하기 약간 길고 복잡" → "정규 표현식이라는 말은 길고 복잡"
- "일관성 없는 형식의 껍질을 벗겨내기" → "일관성 없는 형식을 차례로 걷어내기"
- "장막 뒤에서" → "내부에서"
- "글쎄요, 가능합니다!" → "str_c()와 str_flatten()을 사용하면 벡터에서 패턴을 만듭니다"
-->
